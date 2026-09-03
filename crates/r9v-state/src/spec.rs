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

// DECISION(A1.15): BLOCK_SENTINEL in r9v-state is a compatibility re-export of r9v_ir::BLOCK_TABLE_SENTINEL (u32::MAX); rejected independent sentinel definitions across crates. Spec 1 §2.5, Spec 3 §3.3, card A1.15.
pub use r9v_ir::BLOCK_TABLE_SENTINEL;
pub use r9v_ir::BLOCK_TABLE_SENTINEL as BLOCK_SENTINEL;

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
/// Hard cap on total layer state declarations (Spec 3 §9).
pub const MAX_DECLS_HARD: usize = 4096;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CacheDtype {
    /// 8-bit float with per-token-head f16 scales (spec default).
    #[default]
    E4M3,
    /// 8-bit int with per-token-head f16 scales.
    I8,
    /// Half precision, no scales.
    F16,
}

impl CacheDtype {
    /// Compatibility alias matching casing in `r9v-models`.
    #[allow(non_upper_case_globals)]
    pub const E4m3: CacheDtype = CacheDtype::E4M3;

    /// Cache value bytes per element (Spec 3 §3.2).
    pub const fn bytes(self) -> u64 {
        match self {
            Self::E4M3 | Self::I8 => 1,
            Self::F16 => 2,
        }
    }

    /// Cache value bytes per element as `usize` (Spec 3 §3.2).
    pub const fn element_bytes(self) -> usize {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Retain {
    /// Blocks retained until the sequence is freed (Spec 3 §3.5).
    #[default]
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

    /// Constructs a sliding window retention policy (Spec 3 §2, §3.5).
    ///
    /// Migration constructor for callers transitioning from earlier `Window(w)` variants.
    pub const fn sliding_window(w: u32) -> Self {
        Self::Window { w }
    }

    /// Constructs a pinned sink plus sliding window retention policy (Spec 3 §2, §3.5).
    ///
    /// Migration constructor for callers transitioning from earlier `SinkAndWindow` forms.
    pub const fn sink_and_window(n: u32, w: u32) -> Self {
        Self::SinkWindow { n, w }
    }

    /// Constructs a retention policy from optional sliding window and sink count (Spec 3 §2).
    pub fn from_window_sinks(window: Option<u32>, sinks: u32) -> Result<Self, StateError> {
        match (window, sinks) {
            (None, 0) => Ok(Self::All),
            (Some(w), 0) => Ok(Self::Window { w }),
            (Some(w), n) => Ok(Self::SinkWindow { n, w }),
            (None, s) => Err(StateError::invalid(vec![InvalidItem {
                index: 0,
                reason: format!(
                    "attention sinks ({s}) require a sliding window (window is None); Spec 3 §2 has no sink-only Retain form"
                ),
            }])),
        }
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
        self.push_checks(index, out);
    }

    /// Rejects invalid dims/policies as a typed error instead of computing a
    /// plausible value from them (e.g. a zero head count yielding zero bytes
    /// that would silently size an empty pool).
    fn check_dims(self) -> StateResult<()> {
        let mut problems = Vec::new();
        self.push_checks(u32::MAX, &mut problems);
        if problems.is_empty() {
            Ok(())
        } else {
            Err(StateError::invalid(problems))
        }
    }

    /// Shared dimension/policy checks behind [`Self::validate`] and
    /// [`Self::check_dims`].
    fn push_checks(self, index: u32, out: &mut Vec<InvalidItem>) {
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
    /// `KvPaged`: `hkv * ((d + dv) * cache_bytes + scale_bytes)` (+4 for two f16
    /// scales under `E4M3`/`I8`, 0 for `F16`). `KvLatent`: `latent * cache_bytes + scale_bytes + rope * 2`
    /// (+2 for one f16 scale under `E4M3`/`I8`, 0 for `F16`).
    /// Recurrent/conv layers page nothing, but their dims are still
    /// validated: an invalid spec is a typed error, never `Ok(0)`.
    pub fn per_token_bytes(self) -> StateResult<u64> {
        self.check_dims()?;
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
                let scale_bytes: u64 = match cache {
                    CacheDtype::E4M3 | CacheDtype::I8 => 4,
                    CacheDtype::F16 => 0,
                };
                let per_head = vals
                    .checked_add(scale_bytes)
                    .ok_or_else(|| overflow("kv scales"))?;
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
                let scale_bytes: u64 = match cache {
                    CacheDtype::E4M3 | CacheDtype::I8 => 2,
                    CacheDtype::F16 => 0,
                };
                let vals_with_scales = vals
                    .checked_add(scale_bytes)
                    .ok_or_else(|| overflow("latent scales"))?;
                let rope_bytes = u64::from(rope)
                    .checked_mul(2)
                    .ok_or_else(|| overflow("rope values"))?;
                vals_with_scales
                    .checked_add(rope_bytes)
                    .ok_or_else(|| overflow("latent total"))
            }
            Self::Recurrent { .. } | Self::ConvWindow { .. } => Ok(0),
        }
    }

    /// Fixed slot bytes per sequence for one layer (Spec 3 §6.2).
    ///
    /// `Recurrent`: `h * d * dv * 4`. `ConvWindow`: `(w - 1) * c * 2`.
    /// Paged specs return 0 (they page per token instead).
    ///
    /// Invalid dims are a typed error, never a plausible zero: `w = 0` must
    /// not silently size an empty conv slot via a saturating subtract.
    pub fn slot_bytes(self) -> StateResult<u64> {
        self.check_dims()?;
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        match self {
            Self::Recurrent { h, d, dv } => u64::from(h)
                .checked_mul(u64::from(d))
                .and_then(|v| v.checked_mul(u64::from(dv)))
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| overflow("recurrent slot")),
            Self::ConvWindow { c, w } => {
                u64::from(w.checked_sub(1).ok_or_else(|| overflow("conv window"))?)
                    .checked_mul(u64::from(c))
                    .and_then(|v| v.checked_mul(2))
                    .ok_or_else(|| overflow("conv slot"))
            }
            Self::KvPaged { .. } | Self::KvLatent { .. } => Ok(0),
        }
    }

    /// Associated state kind in Op IR (Spec 1 §2.6).
    pub const fn kind(self) -> r9v_ir::StateKind {
        match self {
            Self::KvPaged { .. } => r9v_ir::StateKind::KvPaged,
            Self::KvLatent { .. } => r9v_ir::StateKind::KvLatent,
            Self::Recurrent { .. } => r9v_ir::StateKind::Recurrent,
            Self::ConvWindow { .. } => r9v_ir::StateKind::ConvWindow,
        }
    }

    /// Fixed bytes per sequence for one layer across both double-buffered slots (Spec 3 §4.2, §6.2).
    ///
    /// `Recurrent`: `h * d * dv * 4 * 2`. `ConvWindow`: `(w - 1) * c * 2 * 2`.
    /// Paged specs return 0 (they page per token instead).
    // DECISION(A1.15): StateSpec::per_seq_bytes reports exact double-buffered per-sequence allocation bytes for both recurrent (h*d*dv*4*2) and conv ((w-1)*c*2*2), aligning ModelSummary totals with StateManager pool sizing; rejected single-buffer conv accounting in model summaries which caused budget divergence. Spec 3 §4.2, §6.2, Spec 8 §7, card A1.15.
    pub fn per_seq_bytes(self) -> StateResult<u64> {
        let single = self.slot_bytes()?;
        if self.is_recurrent() {
            single.checked_mul(2).ok_or_else(|| StateError::Overflow {
                what: "double-buffer per-sequence bytes".to_owned(),
            })
        } else {
            Ok(0)
        }
    }

    /// Compatibility alias for [`Self::per_token_bytes`] (Spec 8 §7).
    pub fn state_per_token_bytes(self) -> StateResult<u64> {
        self.per_token_bytes()
    }

    /// Compatibility alias for [`Self::per_seq_bytes`] (Spec 8 §7).
    pub fn state_per_seq_bytes(self) -> StateResult<u64> {
        self.per_seq_bytes()
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
        let count = u64::try_from(self.layers.len()).map_err(|_| StateError::Overflow {
            what: "group layer count".to_owned(),
        })?;
        per_layer
            .checked_mul(count)
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
        let per_layer = self.spec.per_seq_bytes()?;
        let count = u64::try_from(self.layers.len()).map_err(|_| StateError::Overflow {
            what: "group layer count".to_owned(),
        })?;
        per_layer
            .checked_mul(count)
            .ok_or_else(|| StateError::Overflow {
                what: "group slot bytes".to_owned(),
            })
    }
}

// DECISION(A1.15): explicit (layer, spec) StateDecl declarations and group_layer_specs preserve hybrid multi-spec true declaring model layer indices; rejected flat enumerate renumbering which corrupts layer mappings when layers declare multiple specs. Spec 3 §2, §6.1, Spec 8 §2, card A1.15.
/// A state specification paired with its true declaring model layer index (Spec 3 §2, Spec 8 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateDecl {
    /// Declaring model layer index.
    pub layer: u32,
    /// Authoritative state specification.
    pub spec: StateSpec,
}

impl StateDecl {
    /// Associates a state specification with its declaring model layer index.
    pub const fn new(layer: u32, spec: StateSpec) -> Self {
        Self { layer, spec }
    }
}

impl From<(u32, StateSpec)> for StateDecl {
    fn from((layer, spec): (u32, StateSpec)) -> Self {
        Self { layer, spec }
    }
}

impl From<&(u32, StateSpec)> for StateDecl {
    fn from(&(layer, spec): &(u32, StateSpec)) -> Self {
        Self { layer, spec }
    }
}

impl From<&StateDecl> for StateDecl {
    fn from(decl: &StateDecl) -> Self {
        *decl
    }
}

impl From<(u32, StateSpec, r9v_ir::state::StateHandle)> for StateDecl {
    fn from((layer, spec, _): (u32, StateSpec, r9v_ir::state::StateHandle)) -> Self {
        Self { layer, spec }
    }
}

impl From<&(u32, StateSpec, r9v_ir::state::StateHandle)> for StateDecl {
    fn from(&(layer, spec, _): &(u32, StateSpec, r9v_ir::state::StateHandle)) -> Self {
        Self { layer, spec }
    }
}

pub(crate) fn group_declarations_infallible(
    declarations: impl IntoIterator<Item = StateDecl>,
) -> Vec<LayerGroup> {
    let mut groups: Vec<LayerGroup> = Vec::new();
    for decl in declarations {
        match groups.iter_mut().find(|g| g.spec == decl.spec) {
            Some(g) => {
                if !g.layers.contains(&decl.layer) {
                    let pos = g.layers.partition_point(|&l| l < decl.layer);
                    g.layers.insert(pos, decl.layer);
                }
            }
            None => groups.push(LayerGroup {
                index: groups.len(),
                spec: decl.spec,
                layers: vec![decl.layer],
            }),
        }
    }
    groups
}

/// Groups layers by identical [`StateSpec`] in first-occurrence order,
/// assigning one sequential layer index `0..specs.len()` per entry (Spec 3 §6.1).
///
/// Returns [`StateResult<Vec<LayerGroup>>`]. Validates `specs.len() <= MAX_LAYERS_HARD`
/// and validates each specification. Propagates typed errors without panics or silent drops.
pub fn group_layers(specs: &[StateSpec]) -> StateResult<Vec<LayerGroup>> {
    if specs.len() > MAX_LAYERS_HARD as usize {
        return Err(StateError::invalid(vec![InvalidItem {
            index: u32::MAX,
            reason: format!("layers={} exceeds cap {}", specs.len(), MAX_LAYERS_HARD),
        }]));
    }
    let mut problems = Vec::new();
    let mut decls = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let layer = match u32::try_from(i) {
            Ok(l) => l,
            Err(_) => {
                problems.push(InvalidItem {
                    index: u32::MAX,
                    reason: format!("layer index {i} exceeds u32 range"),
                });
                continue;
            }
        };
        if layer >= MAX_LAYERS_HARD {
            problems.push(InvalidItem {
                index: layer,
                reason: format!("layer index {layer} exceeds cap {MAX_LAYERS_HARD}"),
            });
        }
        spec.validate(layer, &mut problems);
        decls.push(StateDecl::new(layer, *spec));
    }
    if !problems.is_empty() {
        return Err(StateError::invalid(problems));
    }
    let groups = group_declarations_infallible(decls);
    if groups.len() > MAX_GROUPS_HARD {
        return Err(StateError::invalid(vec![InvalidItem {
            index: u32::MAX,
            reason: format!("groups={} exceeds cap {}", groups.len(), MAX_GROUPS_HARD),
        }]));
    }
    Ok(groups)
}

/// Groups explicit [`StateDecl`] declarations by identical [`StateSpec`] in first-occurrence order,
/// retaining each specification's true declaring model layer index (Spec 3 §6.1).
///
/// Returns [`StateResult<Vec<LayerGroup>>`]. Validates `decls.len() <= MAX_DECLS_HARD`
/// immediately before allocation, bounds every `decl.layer < MAX_LAYERS_HARD`, validates unique
/// model layer count against [`MAX_LAYERS_HARD`], and validates each specification before grouping.
/// Direct callers cannot bypass StateManager bounds.
pub fn group_layer_specs(decls: &[StateDecl]) -> StateResult<Vec<LayerGroup>> {
    if decls.len() > MAX_DECLS_HARD {
        return Err(StateError::invalid(vec![InvalidItem {
            index: u32::MAX,
            reason: format!(
                "declarations={} exceeds cap {}",
                decls.len(),
                MAX_DECLS_HARD
            ),
        }]));
    }
    let mut problems = Vec::new();
    let mut unique_layers = std::collections::BTreeSet::new();
    for decl in decls {
        if decl.layer >= MAX_LAYERS_HARD {
            problems.push(InvalidItem {
                index: decl.layer,
                reason: format!("layer index {} exceeds cap {}", decl.layer, MAX_LAYERS_HARD),
            });
        } else {
            unique_layers.insert(decl.layer);
        }
        decl.spec.validate(decl.layer, &mut problems);
    }
    if unique_layers.len() > MAX_LAYERS_HARD as usize {
        problems.push(InvalidItem {
            index: u32::MAX,
            reason: format!(
                "unique model layers={} exceeds cap {}",
                unique_layers.len(),
                MAX_LAYERS_HARD
            ),
        });
    }
    if !problems.is_empty() {
        return Err(StateError::invalid(problems));
    }
    let groups = group_declarations_infallible(decls.iter().copied());
    if groups.len() > MAX_GROUPS_HARD {
        return Err(StateError::invalid(vec![InvalidItem {
            index: u32::MAX,
            reason: format!("groups={} exceeds cap {}", groups.len(), MAX_GROUPS_HARD),
        }]));
    }
    Ok(groups)
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
        // Legacy group_layers assigns sequential indices 0..3:
        let groups = group_layers(&[a, b, a]).expect("group_layers must succeed");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].layers, vec![0, 2]);
        assert_eq!(groups[1].layers, vec![1]);
        assert_eq!(groups[0].index, 0);
        assert_eq!(groups[1].index, 1);

        // Explicit group_layer_specs retains true declaring model layers, including hybrid layers:
        let c = StateSpec::ConvWindow { c: 64, w: 4 };
        let hybrid_decls = [
            StateDecl::new(0, a),
            StateDecl::new(1, a),
            StateDecl::new(3, c),
            StateDecl::new(3, b),
            StateDecl::new(4, c),
        ];
        let hybrid_groups =
            group_layer_specs(&hybrid_decls).expect("group_layer_specs must succeed");
        assert_eq!(hybrid_groups.len(), 3);
        assert_eq!(hybrid_groups[0].spec, a);
        assert_eq!(hybrid_groups[0].layers, vec![0, 1]);
        assert_eq!(hybrid_groups[1].spec, c);
        assert_eq!(hybrid_groups[1].layers, vec![3, 4]);
        assert_eq!(hybrid_groups[2].spec, b);
        assert_eq!(hybrid_groups[2].layers, vec![3]);
    }

    #[test]
    fn retain_migration_constructors() {
        let w = Retain::sliding_window(128);
        assert_eq!(w, Retain::Window { w: 128 });
        assert_eq!(w.window(), Some(128));
        assert!(w.is_windowed());

        let sw = Retain::sink_and_window(4, 256);
        assert_eq!(sw, Retain::SinkWindow { n: 4, w: 256 });
        assert_eq!(sw.window(), Some(256));
        assert!(sw.is_windowed());
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

    #[test]
    fn cache_dtype_scale_bytes_f16_vs_quantized() {
        let paged_f16 = StateSpec::KvPaged {
            hkv: 4,
            d: 64,
            dv: 64,
            cache: CacheDtype::F16,
            retain: Retain::All,
        };
        // F16: exactly zero scale bytes, (64 + 64) * 2 = 256 bytes per head, * 4 = 1024.
        assert_eq!(paged_f16.per_token_bytes().unwrap(), 4 * 128 * 2);

        let paged_e4m3 = StateSpec::KvPaged {
            hkv: 4,
            d: 64,
            dv: 64,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        };
        // E4M3: exactly 4 scale bytes per head, ((64 + 64) * 1 + 4) * 4 = 132 * 4 = 528.
        assert_eq!(paged_e4m3.per_token_bytes().unwrap(), 4 * (128 + 4));

        let paged_i8 = StateSpec::KvPaged {
            hkv: 4,
            d: 64,
            dv: 64,
            cache: CacheDtype::I8,
            retain: Retain::All,
        };
        assert_eq!(paged_i8.per_token_bytes().unwrap(), 4 * (128 + 4));

        let latent_f16 = StateSpec::KvLatent {
            latent: 128,
            rope: 32,
            cache: CacheDtype::F16,
            retain: Retain::All,
        };
        // F16: zero scale bytes, 128 * 2 + 32 * 2 = 320.
        assert_eq!(latent_f16.per_token_bytes().unwrap(), 128 * 2 + 32 * 2);

        let latent_e4m3 = StateSpec::KvLatent {
            latent: 128,
            rope: 32,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        };
        // E4M3: exactly 2 scale bytes, 128 * 1 + 2 + 32 * 2 = 194.
        assert_eq!(latent_e4m3.per_token_bytes().unwrap(), 128 + 2 + 64);

        let latent_i8 = StateSpec::KvLatent {
            latent: 128,
            rope: 32,
            cache: CacheDtype::I8,
            retain: Retain::All,
        };
        assert_eq!(latent_i8.per_token_bytes().unwrap(), 128 + 2 + 64);
    }

    #[test]
    fn group_layer_specs_bounds_and_validations() {
        let spec = StateSpec::Recurrent { h: 1, d: 8, dv: 8 };

        // 1. Layer 1023 is accepted (valid boundary):
        let decl_valid = StateDecl::new(1023, spec);
        let groups = group_layer_specs(&[decl_valid]).expect("1023 must be accepted");
        assert_eq!(groups[0].layers, vec![1023]);

        // 2. Layer 1024 is rejected (exceeds MAX_LAYERS_HARD):
        let decl_1024 = StateDecl::new(1024, spec);
        let err_1024 = group_layer_specs(&[decl_1024]).unwrap_err();
        match err_1024 {
            StateError::InvalidConfig { problems } => {
                assert!(problems
                    .iter()
                    .any(|p| p.index == 1024 && p.reason.contains("exceeds cap 1024")));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }

        // 3. Layer u32::MAX is rejected:
        let decl_max = StateDecl::new(u32::MAX, spec);
        let err_max = group_layer_specs(&[decl_max]).unwrap_err();
        match err_max {
            StateError::InvalidConfig { problems } => {
                assert!(problems
                    .iter()
                    .any(|p| p.index == u32::MAX && p.reason.contains("exceeds cap 1024")));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }

        // 4. Hybrid multi-spec layers: 2 specs per layer on 512 layers (1024 declarations total, unique layers 512)
        // must NOT be falsely counted as > 1024 model layers:
        let spec2 = StateSpec::ConvWindow { c: 8, w: 4 };
        let mut hybrid_decls = Vec::with_capacity(1024);
        for l in 0..512 {
            hybrid_decls.push(StateDecl::new(l, spec));
            hybrid_decls.push(StateDecl::new(l, spec2));
        }
        let hybrid_groups =
            group_layer_specs(&hybrid_decls).expect("512 hybrid layers must be accepted");
        assert_eq!(hybrid_groups.len(), 2);
        assert_eq!(hybrid_groups[0].layers.len(), 512);
        assert_eq!(hybrid_groups[1].layers.len(), 512);

        // 5. Total declarations cap MAX_DECLS_HARD: 4097 declarations is rejected immediately:
        let huge_decls: Vec<StateDecl> = (0..=4096).map(|_| StateDecl::new(0, spec)).collect();
        let err_decls = group_layer_specs(&huge_decls).unwrap_err();
        match err_decls {
            StateError::InvalidConfig { problems } => {
                assert!(problems
                    .iter()
                    .any(|p| p.reason.contains("declarations=4097 exceeds cap 4096")));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }

        // 6. Total layers cap MAX_LAYERS_HARD in group_layers: 1025 specs rejected immediately:
        let huge_specs: Vec<StateSpec> = (0..=1024).map(|_| spec).collect();
        let err_layers = group_layers(&huge_specs).unwrap_err();
        match err_layers {
            StateError::InvalidConfig { problems } => {
                assert!(problems
                    .iter()
                    .any(|p| p.reason.contains("layers=1025 exceeds cap 1024")));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }
}
