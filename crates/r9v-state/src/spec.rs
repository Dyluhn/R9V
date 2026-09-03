// SPDX-License-Identifier: Apache-2.0
//! Authoritative typed state declarations (Spec 3 §2, §6).
//!
//! A model definition declares, per layer, a list of [`StateSpec`]. Layers
//! with identical specs share a layer-group: one pool and one block table
//! in `BatchMeta` (Spec 3 §6.1). `r9v-ir` owns only the opaque
//! `StateHandle(layer, kind)`; the parameterized spec lives here.

use crate::error::{InvalidItem, StateError, StateResult};

/// Tokens per block (Spec 3 §3.1: one lane per token in a wave32 QK pass).
pub const BLOCK_TOKENS: u32 = 32;

/// Padding sentinel for unused `block_table` entries (Spec 3 §3.3).
pub const BLOCK_SENTINEL: u32 = u32::MAX;

// DECISION(A1.11): hard caps below bound every untrusted size before any
// allocation so oversized requests fail as typed errors instead of
// over-allocating; rejected: uncapped construction (a hostile or buggy caller
// could force gigabyte arenas). Spec 3 §9 leaves absolute maxima open.

/// Hard cap on `max_ctx` (1M tokens; Spec 3 §9 leaves the maximum open).
pub const MAX_CTX_HARD: u32 = 1 << 20;
/// Hard cap on `max_seqs`.
pub const MAX_SEQS_HARD: u32 = 65_536;
/// Hard cap on layers per model.
pub const MAX_LAYERS_HARD: u32 = 1024;
/// Hard cap on layer-groups (Spec 3 §6.1: group count is small).
pub const MAX_GROUPS_HARD: usize = 16;
/// Hard cap on KV heads, head dims, recurrent heads/dims, conv channels.
pub const MAX_DIM_HARD: u32 = 4096;
/// Hard cap on tokens in one batch (`T` across all sequences).
pub const MAX_BATCH_TOKENS_HARD: u64 = 1 << 20;
/// Hard cap on tokens reserved by one `reserve` call.
pub const MAX_RESERVE_HARD: u32 = 1 << 20;

/// KV cache dtype (Spec 3 §2).
///
/// Scale granularity is per token-head for `E4M3` and `I8` (spec default);
/// `F16` carries no scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheDtype {
    /// 8-bit float with per-token-head f16 scales (spec default).
    E4M3,
    /// 8-bit int with per-token-head f16 scales.
    I8,
    /// Half precision, no scales.
    F16,
}

impl CacheDtype {
    /// Cache value bytes per element (Spec 3 §3.2).
    pub const fn bytes(self) -> u64 {
        match self {
            Self::E4M3 | Self::I8 => 1,
            Self::F16 => 2,
        }
    }

    /// Stable lowercase name (CONVENTIONS.md §3.2: never raw discriminants).
    pub const fn name(self) -> &'static str {
        match self {
            Self::E4M3 => "e4m3",
            Self::I8 => "i8",
            Self::F16 => "f16",
        }
    }
}

/// Retention policy (Spec 3 §2, §3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Retain {
    /// Blocks retained until the sequence is freed (Spec 3 §3.5).
    All,
    /// Sliding window of `w` tokens (Spec 3 §3.5).
    Window {
        /// Window size in tokens.
        w: u32,
    },
    /// Pinned sink of `n` tokens plus a sliding window of `w` (Spec 3 §3.5).
    SinkWindow {
        /// Sink size in tokens.
        n: u32,
        /// Window size in tokens.
        w: u32,
    },
}

impl Retain {
    /// Window size in tokens, or `None` for [`Retain::All`].
    pub const fn window(self) -> Option<u32> {
        match self {
            Self::All => None,
            Self::Window { w } => Some(w),
            Self::SinkWindow { w, .. } => Some(w),
        }
    }

    /// Whether this policy ever releases blocks before `free_seq`.
    pub const fn is_windowed(self) -> bool {
        !matches!(self, Self::All)
    }
}

/// Per-layer state declaration (Spec 3 §2).
///
/// Closed enum: the v1 kinds. A new kind lands via RFC (Spec 1 §7); every
/// `match` on this type stays exhaustive with no wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateSpec {
    /// Paged KV cache (Spec 3 §3).
    KvPaged {
        /// Local KV heads.
        hkv: u32,
        /// Head dim (K).
        d: u32,
        /// Head dim (V).
        dv: u32,
        /// Cache dtype.
        cache: CacheDtype,
        /// Retention policy.
        retain: Retain,
    },
    /// MLA latent + rope part (Spec 3 §2, §3.2).
    KvLatent {
        /// Latent dim (cache dtype).
        latent: u32,
        /// Rope dim (always F16).
        rope: u32,
        /// Cache dtype for the latent part.
        cache: CacheDtype,
        /// Retention policy.
        retain: Retain,
    },
    /// Fixed-size per-head recurrent state, f32 `[h, d, dv]` (Spec 3 §4.1).
    Recurrent {
        /// Heads.
        h: u32,
        /// State dim.
        d: u32,
        /// State dim.
        dv: u32,
    },
    /// Causal-conv window, f16 `[w - 1, c]` (Spec 3 §4.1).
    ConvWindow {
        /// Channels.
        c: u32,
        /// Window length.
        w: u32,
    },
}

impl StateSpec {
    /// Whether this spec pages through the block pool (Spec 3 §3).
    pub const fn is_paged(self) -> bool {
        matches!(self, Self::KvPaged { .. } | Self::KvLatent { .. })
    }

    /// Whether this spec uses A/B double-buffered slots (Spec 3 §4.2).
    pub const fn is_recurrent(self) -> bool {
        matches!(self, Self::Recurrent { .. } | Self::ConvWindow { .. })
    }

    /// Retention policy, or `None` for fixed-lifetime recurrent/conv slots.
    pub const fn retain(self) -> Option<Retain> {
        match self {
            Self::KvPaged { retain, .. } | Self::KvLatent { retain, .. } => Some(retain),
            Self::Recurrent { .. } | Self::ConvWindow { .. } => None,
        }
    }

    /// Validates one layer spec, collecting every problem (CONVENTIONS.md §1.4).
    pub(crate) fn validate(self, index: u32, out: &mut Vec<InvalidItem>) {
        match self {
            Self::KvPaged {
                hkv, d, dv, retain, ..
            } => {
                check_dim("hkv", hkv, index, out);
                check_dim("d", d, index, out);
                check_dim("dv", dv, index, out);
                check_retain(retain, index, out);
            }
            Self::KvLatent {
                latent,
                rope,
                retain,
                ..
            } => {
                check_dim("latent", latent, index, out);
                check_dim("rope", rope, index, out);
                check_retain(retain, index, out);
            }
            Self::Recurrent { h, d, dv } => {
                check_dim("h", h, index, out);
                check_dim("d", d, index, out);
                check_dim("dv", dv, index, out);
            }
            Self::ConvWindow { c, w } => {
                check_dim("c", c, index, out);
                if w == 0 || w > MAX_DIM_HARD {
                    out.push(InvalidItem {
                        index,
                        reason: format!("w={w} out of range 1..={}", MAX_DIM_HARD),
                    });
                }
            }
        }
    }

    /// Per-token bytes for one layer of this spec (Spec 3 §6.2).
    ///
    /// `KvPaged`: `hkv * ((d + dv) * cache_bytes + 4)` (+4 for two f16
    /// scales). `KvLatent`: `latent * cache_bytes + 2 + rope * 2`.
    pub fn per_token_bytes(self) -> StateResult<u64> {
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        match self {
            Self::KvPaged {
                hkv, d, dv, cache, ..
            } => {
                let dd = u64::from(d)
                    .checked_add(u64::from(dv))
                    .ok_or_else(|| overflow("kv dims"))?;
                let vals = dd
                    .checked_mul(cache.bytes())
                    .ok_or_else(|| overflow("kv values"))?;
                let per_head = vals.checked_add(4).ok_or_else(|| overflow("kv scales"))?;
                per_head
                    .checked_mul(u64::from(hkv))
                    .ok_or_else(|| overflow("kv heads"))
            }
            Self::KvLatent {
                latent,
                rope,
                cache,
                ..
            } => {
                let vals = u64::from(latent)
                    .checked_mul(cache.bytes())
                    .ok_or_else(|| overflow("latent values"))?;
                let rope_bytes = u64::from(rope)
                    .checked_mul(2)
                    .ok_or_else(|| overflow("rope values"))?;
                vals.checked_add(2)
                    .and_then(|v| v.checked_add(rope_bytes))
                    .ok_or_else(|| overflow("latent total"))
            }
            Self::Recurrent { .. } | Self::ConvWindow { .. } => Ok(0),
        }
    }

    /// Fixed slot bytes per sequence for one layer (Spec 3 §6.2).
    ///
    /// `Recurrent`: `h * d * dv * 4`. `ConvWindow`: `(w - 1) * c * 2`.
    /// Paged specs return 0 (they page per token instead).
    pub fn slot_bytes(self) -> StateResult<u64> {
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        match self {
            Self::Recurrent { h, d, dv } => u64::from(h)
                .checked_mul(u64::from(d))
                .and_then(|v| v.checked_mul(u64::from(dv)))
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| overflow("recurrent slot")),
            Self::ConvWindow { c, w } => u64::from(w.saturating_sub(1))
                .checked_mul(u64::from(c))
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| overflow("conv slot")),
            Self::KvPaged { .. } | Self::KvLatent { .. } => Ok(0),
        }
    }
}

fn check_dim(name: &str, v: u32, index: u32, out: &mut Vec<InvalidItem>) {
    if v == 0 || v > MAX_DIM_HARD {
        out.push(InvalidItem {
            index,
            reason: format!("{name}={v} out of range 1..={}", MAX_DIM_HARD),
        });
    }
}

fn check_retain(retain: Retain, index: u32, out: &mut Vec<InvalidItem>) {
    match retain {
        Retain::All => {}
        Retain::Window { w } => {
            if w == 0 || w > MAX_CTX_HARD {
                out.push(InvalidItem {
                    index,
                    reason: format!("window w={w} out of range 1..={}", MAX_CTX_HARD),
                });
            }
        }
        Retain::SinkWindow { n, w } => {
            if n == 0 || n > MAX_CTX_HARD {
                out.push(InvalidItem {
                    index,
                    reason: format!("sink n={n} out of range 1..={}", MAX_CTX_HARD),
                });
            }
            if w == 0 || w > MAX_CTX_HARD {
                out.push(InvalidItem {
                    index,
                    reason: format!("window w={w} out of range 1..={}", MAX_CTX_HARD),
                });
            }
        }
    }
}

/// Layers sharing one [`StateSpec`]: one pool, one block table (Spec 3 §6.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGroup {
    /// Position of this group in `BatchMeta` (`G` axis).
    pub index: usize,
    /// The shared spec.
    pub spec: StateSpec,
    /// Layer ids in this group, ascending.
    pub layers: Vec<u32>,
}

impl LayerGroup {
    /// Per-token bytes across all layers in the group (Spec 3 §6.2).
    pub fn per_token_bytes(&self) -> StateResult<u64> {
        let per_layer = self.spec.per_token_bytes()?;
        per_layer
            .checked_mul(self.layers.len() as u64)
            .ok_or_else(|| StateError::Overflow {
                what: "group per-token bytes".to_owned(),
            })
    }

    /// Block bytes across all layers in the group.
    pub fn block_bytes(&self) -> StateResult<u64> {
        self.per_token_bytes()?
            .checked_mul(u64::from(BLOCK_TOKENS))
            .ok_or_else(|| StateError::Overflow {
                what: "group block bytes".to_owned(),
            })
    }

    /// Fixed slot bytes per sequence across all layers (double-buffered
    /// recurrent/conv groups count both A and B slots, Spec 3 §6.2).
    pub fn slots_bytes_per_seq(&self) -> StateResult<u64> {
        let per_layer = self.spec.slot_bytes()?;
        let layers = per_layer
            .checked_mul(self.layers.len() as u64)
            .ok_or_else(|| StateError::Overflow {
                what: "group slot bytes".to_owned(),
            })?;
        if self.spec.is_recurrent() {
            layers.checked_mul(2).ok_or_else(|| StateError::Overflow {
                what: "double-buffer slots".to_owned(),
            })
        } else {
            Ok(layers)
        }
    }
}

/// Groups layers by identical [`StateSpec`] in first-occurrence order.
///
/// Deterministic: iteration is over the layer list in order; no hash-map
/// iteration touches the output (engineering standards §2.6).
///
/// Spec 3 §6.1.
pub fn group_layers(specs: &[StateSpec]) -> Vec<LayerGroup> {
    let mut groups: Vec<LayerGroup> = Vec::new();
    for (layer, spec) in specs.iter().enumerate() {
        let layer = layer as u32;
        match groups.iter_mut().find(|g| g.spec == *spec) {
            Some(g) => g.layers.push(layer),
            None => groups.push(LayerGroup {
                index: groups.len(),
                spec: *spec,
                layers: vec![layer],
            }),
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_is_first_occurrence_order() {
        let a = StateSpec::KvPaged {
            hkv: 8,
            d: 128,
            dv: 128,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        };
        let b = StateSpec::Recurrent {
            h: 4,
            d: 64,
            dv: 64,
        };
        let groups = group_layers(&[a, b, a]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].layers, vec![0, 2]);
        assert_eq!(groups[1].layers, vec![1]);
        assert_eq!(groups[0].index, 0);
        assert_eq!(groups[1].index, 1);
    }

    #[test]
    fn per_token_bytes_match_spec_example_shape() {
        // Spec 3 §6.2 shape: L · hkv · ((d + dv) · cache_bytes + 4) per token.
        let spec = StateSpec::KvPaged {
            hkv: 8,
            d: 128,
            dv: 128,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        };
        assert_eq!(spec.per_token_bytes().unwrap(), 8 * (256 + 4));
    }
}
