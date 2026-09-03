// SPDX-License-Identifier: Apache-2.0
//! Model architecture and layer specifications (Spec 8 §3, §3.1, §6; card A1.3).
//!
//! A model definition produces a [`ModelSpec`] containing a sequence of [`LayerSpec`]s.
//! The generic layer builder lowers these specifications into an Op IR step graph.

use r9v_ir::op::{
    ActivationKind, HashId, LinearAttnKind, MoeScoring, NgramCombine, NormKind, RopeScaling,
    RopeStyle,
};

use crate::error::ModelsError;

/// Implementation limits for allocation- and loop-bound dimensions (Spec 8 §6;
/// card A1.3).
///
/// Every bound below guards a Rust allocation, a loop trip count, or a tensor
/// extent the loader will later materialize (stacked `[E, ...]` experts,
/// `hot_hint` of length `E`, TP-divisor enumeration over `hkv`, the MTP head
/// loop, the layer loop). Each value carries wide margin above every v1
/// family in Spec 8 §4 and exists so untrusted metadata fails with typed
/// validation *before* allocation, never with a panic or OOM. Raise one only
/// when a family needs it, with the family named at the change.
// DECISION(A1.3): powers-of-two caps with wide margin above v1 families, not
// exact family maxima; rejected exact maxima (a new checkpoint would turn
// into a spurious rejection) and no caps (u32::MAX metadata could drive
// allocations and loops). Spec 8 §6 is silent on numeric caps.
/// Maximum transformer layers in one [`ModelSpec`].
pub const MAX_MODEL_LAYERS: u32 = 1024;
/// Maximum query heads `h` per attention layer.
pub const MAX_ATTENTION_HEADS: u32 = 4096;
/// Maximum key/value heads `hkv` per attention layer.
pub const MAX_KV_HEADS: u32 = 4096;
/// Maximum feature dimension (`dm`, `d`, `dv`, `dff`, `dff_e`, MLA ranks and
/// dims, SSM conv width).
pub const MAX_FEATURE_DIM: u32 = 1 << 20;
/// Maximum vocabulary size `V`.
pub const MAX_VOCAB_SIZE: u32 = 1 << 24;
/// Maximum routed experts `e` (stacked `[E, ...]` residency unit, Spec 8 §5).
pub const MAX_EXPERTS: u32 = 4096;
/// Maximum MTP prediction heads.
pub const MAX_MTP_HEADS: u32 = 64;
/// Maximum layers executed per MTP head.
pub const MAX_MTP_LAYERS_PER_HEAD: u32 = 64;
/// Maximum n-gram hash heads.
pub const MAX_NGRAM_HEADS: u32 = 1024;
/// Maximum entries in one n-gram hash table.
pub const MAX_NGRAM_TABLE_ENTRIES: u32 = 1 << 28;
/// Maximum sliding-window length / attention sinks (retained-token counts).
pub const MAX_WINDOW: u32 = 1 << 28;

/// Pushes an invalid-layer problem when `value` exceeds the implementation
/// `limit` (CONVENTIONS.md §1.4; Spec 8 §6).
fn check_layer_dim(
    problems: &mut Vec<ModelsError>,
    layer: u32,
    name: &'static str,
    value: u32,
    limit: u32,
) {
    if value > limit {
        problems.push(ModelsError::InvalidLayerSpec {
            layer,
            reason: format!(
                "dimension '{name}' value {value} exceeds implementation limit {limit}"
            ),
        });
    }
}

/// Pushes an invalid-model problem when `value` exceeds the implementation
/// `limit` (CONVENTIONS.md §1.4; Spec 8 §6).
fn check_model_dim(problems: &mut Vec<ModelsError>, name: &'static str, value: u32, limit: u32) {
    if value > limit {
        problems.push(ModelsError::InvalidModelSpec {
            reason: format!(
                "dimension '{name}' value {value} exceeds implementation limit {limit}"
            ),
        });
    }
}

/// Placement strategy for normalization layers within a transformer block (Spec 8 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NormPlacement {
    /// Pre-layer normalization: `norm(x)` before each sublayer (standard LLaMA/Mistral).
    #[default]
    Pre,
    /// Sandwich normalization: norm before and after each sublayer before residual (Gemma-style).
    Sandwich,
    /// Parallel formulation: attention and FFN both compute from the same pre-norm input (GPT-J/Falcon).
    Parallel,
    /// Post-normalization: `x = x + post_norm(sublayer(x))` (e.g. OLMo 2). Mixer and FFN receive raw stream before residual addition.
    Post,
}

/// Parameterized normalization specification (Spec 8 §3; Spec 1 §4.B).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormSpec {
    /// Normalization kind: RMSNorm or LayerNorm (Spec 1 §4.B).
    pub kind: NormKind,
    /// Epsilon variance floor (Spec 1 §4.B).
    pub eps: f32,
    /// Weight offset (e.g. 1.0 for Gemma's 1+w parameterization, 0.0 for standard).
    pub weight_offset: f32,
}

impl NormSpec {
    /// Creates a standard RMSNorm specification with offset 0.0.
    pub const fn rms(eps: f32) -> Self {
        Self {
            kind: NormKind::Rms,
            eps,
            weight_offset: 0.0,
        }
    }

    /// Creates a standard LayerNorm specification with offset 0.0.
    pub const fn layer(eps: f32) -> Self {
        Self {
            kind: NormKind::Layer,
            eps,
            weight_offset: 0.0,
        }
    }

    /// Creates a Gemma-style RMSNorm specification with offset 1.0.
    pub const fn gemma(eps: f32) -> Self {
        Self {
            kind: NormKind::Rms,
            eps,
            weight_offset: 1.0,
        }
    }

    /// Validates normalization specification fields (CONVENTIONS.md §1.4).
    pub fn validate(&self, context: &'static str) -> Result<(), ModelsError> {
        let mut problems = Vec::new();
        if !self.eps.is_finite() || self.eps <= 0.0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "{context}: norm eps must be finite and > 0, got {}",
                    self.eps
                ),
            });
        }
        if !self.weight_offset.is_finite() {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "{context}: norm weight_offset must be finite, got {}",
                    self.weight_offset
                ),
            });
        }
        ModelsError::from_problems(problems)
    }
}

/// Rotary Position Embedding (RoPE) specification (Spec 8 §3; Spec 1 §4.B).
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    /// Base theta frequency.
    pub theta: f32,
    /// Dimension to which rotary embedding is applied.
    pub rot_dim: u32,
    /// Interleaved (LLaMA) or NeoX style pairs.
    pub style: RopeStyle,
    /// Context frequency scaling algorithm.
    pub scaling: RopeScaling,
    /// Multimodal RoPE section dimensions [T, H, W], if applicable.
    pub mrope_sections: Option<[u32; 3]>,
}

impl RopeSpec {
    /// Validates RoPE specification fields.
    pub fn validate(&self, context: &'static str) -> Result<(), ModelsError> {
        let mut problems = Vec::new();
        if !self.theta.is_finite() || self.theta <= 0.0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "{context}: rope theta must be finite and > 0, got {}",
                    self.theta
                ),
            });
        }
        if self.rot_dim == 0 || !self.rot_dim.is_multiple_of(2) {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "{context}: rope rot_dim must be positive and even, got {}",
                    self.rot_dim
                ),
            });
        }
        ModelsError::from_problems(problems)
    }
}

/// Multi-Head Latent Attention (MLA) low-rank compression specification (Spec 8 §3; Spec 1 §4.D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MlaSpec {
    /// Query low-rank compression rank `d_c`.
    pub q_lora_rank: u32,
    /// Key/Value low-rank compression rank `d_c'`.
    pub kv_lora_rank: u32,
    /// Decoupled Query/Key non-rotary feature dimension.
    pub qk_nope_dim: u32,
    /// Decoupled Query/Key rotary feature dimension.
    pub qk_rope_dim: u32,
    /// Value feature dimension per head.
    pub v_dim: u32,
}

impl MlaSpec {
    /// Validates MLA specification dimensions.
    pub fn validate(&self, context: &'static str) -> Result<(), ModelsError> {
        let mut problems = Vec::new();
        if self.q_lora_rank == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!("{context}: mla q_lora_rank must be > 0"),
            });
        }
        if self.kv_lora_rank == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!("{context}: mla kv_lora_rank must be > 0"),
            });
        }
        if self.qk_nope_dim == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!("{context}: mla qk_nope_dim must be > 0"),
            });
        }
        if self.qk_rope_dim == 0 || !self.qk_rope_dim.is_multiple_of(2) {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "{context}: mla qk_rope_dim must be positive and even, got {}",
                    self.qk_rope_dim
                ),
            });
        }
        if self.v_dim == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!("{context}: mla v_dim must be > 0"),
            });
        }
        for (name, value) in [
            ("mla q_lora_rank", self.q_lora_rank),
            ("mla kv_lora_rank", self.kv_lora_rank),
            ("mla qk_nope_dim", self.qk_nope_dim),
            ("mla qk_rope_dim", self.qk_rope_dim),
            ("mla v_dim", self.v_dim),
        ] {
            if value > MAX_FEATURE_DIM {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!(
                        "{context}: dimension '{name}' value {value} exceeds implementation limit {MAX_FEATURE_DIM}"
                    ),
                });
            }
        }
        ModelsError::from_problems(problems)
    }
}

// DECISION(A1.15): r9v-state is sole owner of StateSpec, CacheDtype, Retain, and checked state byte accounting, with r9v-models re-exporting them; rejected maintaining duplicate parallel definitions in r9v-models. Spec 3 §2, Spec 8 §2, §7, card A1.15.
pub use r9v_state::{
    group_layer_specs, group_layers, CacheDtype, LayerGroup, Retain, StateDecl, StateSpec,
};

/// Token mixing sublayer specification (Spec 8 §3).
#[derive(Debug, Clone, PartialEq)]
pub enum Mixer {
    /// Multi-Head Attention, Grouped-Query Attention, or Multi-Head Latent Attention (Spec 8 §3).
    Attention {
        /// Number of query heads `H`.
        h: u32,
        /// Number of key/value heads `Hkv`.
        hkv: u32,
        /// Head query/key dimension `D`.
        d: u32,
        /// Head value dimension `Dv`.
        dv: u32,
        /// Whether QKV projection includes bias vectors.
        qkv_bias: bool,
        /// Whether output projection includes a bias vector.
        o_bias: bool,
        /// Optional per-head Q/K normalization after projection (e.g. Qwen2, Gemma2).
        qk_norm: Option<NormSpec>,
        /// Rotary position embedding parameters.
        rope: RopeSpec,
        /// Optional sliding window context length.
        window: Option<u32>,
        /// Number of initial attention sinks.
        sinks: u32,
        /// Optional attention logit soft-capping threshold (e.g. 50.0 for Gemma2).
        logit_softcap: Option<f32>,
        /// Output gating: `o = attn ⊙ σ(W_g x)` (Qwen3-Next style).
        output_gate: bool,
        /// Multi-Head Latent Attention compression parameters, if enabled.
        mla: Option<MlaSpec>,
        /// Cache storage data type.
        cache: CacheDtype,
        /// Whether QKV projections are pre-fused in checkpoint weights (e.g. Phi-3 `blk.N.attn_qkv.weight`).
        pre_fused: bool,
    },
    /// Linear Attention or Structured State-Space Model scan (Spec 8 §3).
    LinearAttention {
        /// Scan recurrence kind (GatedDeltaNet, GLA, or Mamba2).
        kind: LinearAttnKind,
        /// Number of scan heads.
        h: u32,
        /// Input head dimension `D`.
        d: u32,
        /// Output head dimension `Dv`.
        dv: u32,
        /// Optional causal 1D convolution kernel width before the scan.
        conv: Option<u32>,
        /// Gating activation function (e.g. SiLU).
        gate_act: ActivationKind,
        /// Optional normalization applied to scan output.
        output_norm: Option<NormSpec>,
        /// Output gating: `o = scan ⊙ act(W_g x)`.
        output_gate: bool,
    },
    /// Sublayer omitted (e.g. pure feed-forward layer).
    None,
}

/// Grouped routing parameters for MoE architectures (Spec 8 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeGroupSpec {
    /// Number of expert groups `n_group`.
    pub n_group: u32,
    /// Number of experts selected per group `topk_group`.
    pub topk_group: u32,
}

/// Shared expert parameters executed unconditionally (Spec 8 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeSharedSpec {
    /// Number of shared experts.
    pub n: u32,
    /// Intermediate hidden dimension of shared expert FFN.
    pub dff: u32,
}

/// Feed-forward network sublayer specification (Spec 8 §3).
#[derive(Debug, Clone, PartialEq)]
pub enum Ffn {
    /// Standard dense feed-forward network (Spec 8 §3).
    Dense {
        /// Intermediate hidden dimension `Dff`.
        dff: u32,
        /// Non-linear activation function.
        act: ActivationKind,
        /// Gated formulation: `act(W_gate x) ⊙ W_up x`.
        gated: bool,
        /// Whether projections include bias vectors.
        bias: bool,
        /// Whether Gate and Up projections are pre-fused in checkpoint weights (e.g. Phi-3 `blk.N.ffn_up.weight`).
        pre_fused: bool,
    },
    /// Mixture of Experts feed-forward network (Spec 8 §3).
    Moe {
        /// Total number of routed experts `E`.
        e: u32,
        /// Number of active experts selected per token `K`.
        k: u32,
        /// Intermediate hidden dimension per expert `Dff_e`.
        dff_e: u32,
        /// Expert MLP activation function.
        act: ActivationKind,
        /// Router scoring function (Softmax or Sigmoid).
        scoring: MoeScoring,
        /// Whether router weights are renormalized to sum to 1.
        renormalize: bool,
        /// Grouped expert routing constraints, if applicable.
        group: Option<MoeGroupSpec>,
        /// Whether router projection includes a bias vector.
        route_bias: bool,
        /// Router scaling factor.
        route_scale: f32,
        /// Shared expert parameters, if present.
        shared: Option<MoeSharedSpec>,
        /// Gated shared expert activation: `σ(W_sg x) ⊙ shared_ffn(x)`.
        shared_gate: bool,
    },
    /// Sublayer omitted (e.g. pure attention block).
    None,
}

/// Transformer layer block specification (Spec 8 §3).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    /// Normalization layer placement strategy.
    pub norm: NormPlacement,
    /// Normalization parameters.
    pub norm_kind: NormSpec,
    /// Token mixing sublayer.
    pub mixer: Mixer,
    /// Feed-forward network sublayer.
    pub ffn: Ffn,
    /// Residual stream scaling factor (typically 1.0).
    pub residual_scale: f32,
}

impl LayerSpec {
    /// Validates layer specification parameters (Spec 8 §6; CONVENTIONS.md §1.4).
    pub fn validate(&self, layer_idx: u32) -> Result<(), ModelsError> {
        let mut problems = Vec::new();

        if !self.residual_scale.is_finite() || self.residual_scale == 0.0 {
            problems.push(ModelsError::InvalidLayerSpec {
                layer: layer_idx,
                reason: format!(
                    "residual_scale must be finite and non-zero, got {}",
                    self.residual_scale
                ),
            });
        }

        if let Err(e) = self.norm_kind.validate("layer norm") {
            problems.push(e);
        }

        match &self.mixer {
            Mixer::Attention {
                h,
                hkv,
                d,
                dv,
                qk_norm,
                rope,
                window,
                sinks,
                logit_softcap,
                mla,
                pre_fused,
                ..
            } => {
                if *h == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "attention head count h must be > 0".to_string(),
                    });
                }
                if *hkv == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "attention KV head count hkv must be > 0".to_string(),
                    });
                }
                if *hkv > *h {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: format!("hkv ({hkv}) cannot exceed h ({h})"),
                    });
                }
                if !h.is_multiple_of(*hkv) {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: format!("h ({h}) must be divisible by hkv ({hkv})"),
                    });
                }
                if *d == 0 || !d.is_multiple_of(16) {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: format!("head dimension d ({d}) must be > 0 and divisible by 16"),
                    });
                }
                if *dv == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "head value dimension dv must be > 0".to_string(),
                    });
                }
                if let Some(norm) = qk_norm {
                    if let Err(e) = norm.validate("qk_norm") {
                        problems.push(e);
                    }
                }
                if let Err(e) = rope.validate("attention rope") {
                    problems.push(e);
                }
                if let Some(cap) = logit_softcap {
                    if !cap.is_finite() || *cap <= 0.0 {
                        problems.push(ModelsError::InvalidLayerSpec {
                            layer: layer_idx,
                            reason: format!("logit_softcap must be finite and > 0, got {cap}"),
                        });
                    }
                }
                if let Some(mla_spec) = mla {
                    if let Err(e) = mla_spec.validate("attention mla") {
                        problems.push(e);
                    }
                    // Card A1.14 lowers qk_norm on the MLA form instead of
                    // rejecting it (per-side norms in the generic builder);
                    // nothing to refuse here beyond the norm's own validation
                    // above. SI-29.
                }
                if *pre_fused && mla.is_some() {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "pre-fused QKV topology is incompatible with MLA compression"
                            .to_string(),
                    });
                }
                check_layer_dim(&mut problems, layer_idx, "h", *h, MAX_ATTENTION_HEADS);
                check_layer_dim(&mut problems, layer_idx, "hkv", *hkv, MAX_KV_HEADS);
                check_layer_dim(&mut problems, layer_idx, "d", *d, MAX_FEATURE_DIM);
                check_layer_dim(&mut problems, layer_idx, "dv", *dv, MAX_FEATURE_DIM);
                // Spec 3 §2 admits no sink-only form: sinks are only meaningful
                // with a window, so the combination is rejected, never ignored.
                if *sinks > 0 && window.is_none() {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: format!(
                            "attention sinks ({sinks}) require a sliding window (window is None); Spec 3 §2 has no sink-only Retain form"
                        ),
                    });
                }
                if let Some(w) = window {
                    check_layer_dim(&mut problems, layer_idx, "window", *w, MAX_WINDOW);
                }
                check_layer_dim(&mut problems, layer_idx, "sinks", *sinks, MAX_WINDOW);
            }
            Mixer::LinearAttention {
                h,
                d,
                dv,
                conv,
                output_norm,
                ..
            } => {
                if *h == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "linear attention head count h must be > 0".to_string(),
                    });
                }
                if *d == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "linear attention d must be > 0".to_string(),
                    });
                }
                if *dv == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "linear attention dv must be > 0".to_string(),
                    });
                }
                if let Some(w) = conv {
                    if *w == 0 {
                        problems.push(ModelsError::InvalidLayerSpec {
                            layer: layer_idx,
                            reason: "linear attention conv kernel must be > 0".to_string(),
                        });
                    }
                    check_layer_dim(&mut problems, layer_idx, "conv", *w, MAX_FEATURE_DIM);
                }
                if let Some(norm) = output_norm {
                    if let Err(e) = norm.validate("linear attention output_norm") {
                        problems.push(e);
                    }
                }
                check_layer_dim(&mut problems, layer_idx, "h", *h, MAX_ATTENTION_HEADS);
                check_layer_dim(&mut problems, layer_idx, "d", *d, MAX_FEATURE_DIM);
                check_layer_dim(&mut problems, layer_idx, "dv", *dv, MAX_FEATURE_DIM);
            }
            Mixer::None => {}
        }

        match &self.ffn {
            Ffn::Dense {
                dff,
                gated,
                pre_fused,
                ..
            } => {
                if *dff == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "dense FFN intermediate dimension dff must be > 0".to_string(),
                    });
                }
                if *pre_fused && !*gated {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "pre-fused FFN topology requires gated formulation".to_string(),
                    });
                }
                check_layer_dim(&mut problems, layer_idx, "dff", *dff, MAX_FEATURE_DIM);
            }
            Ffn::Moe {
                e,
                k,
                dff_e,
                route_scale,
                group,
                shared,
                shared_gate,
                ..
            } => {
                if *e == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "MoE expert count e must be > 0".to_string(),
                    });
                }
                if *k == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "MoE active experts k must be > 0".to_string(),
                    });
                }
                if *k > *e {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: format!("MoE top_k ({k}) cannot exceed total experts e ({e})"),
                    });
                }
                if *dff_e == 0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "MoE expert intermediate dimension dff_e must be > 0".to_string(),
                    });
                }
                check_layer_dim(&mut problems, layer_idx, "e", *e, MAX_EXPERTS);
                check_layer_dim(&mut problems, layer_idx, "dff_e", *dff_e, MAX_FEATURE_DIM);
                if !route_scale.is_finite() || *route_scale <= 0.0 {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: format!(
                            "MoE route_scale must be finite and > 0, got {route_scale}"
                        ),
                    });
                }
                if let Some(g) = group {
                    if g.n_group == 0 || g.topk_group == 0 || g.topk_group > *k {
                        problems.push(ModelsError::InvalidLayerSpec {
                            layer: layer_idx,
                            reason: format!("invalid MoE group spec: {g:?}"),
                        });
                    }
                    check_layer_dim(&mut problems, layer_idx, "n_group", g.n_group, MAX_EXPERTS);
                }
                if let Some(s) = shared {
                    if s.n == 0 || s.dff == 0 {
                        problems.push(ModelsError::InvalidLayerSpec {
                            layer: layer_idx,
                            reason: format!("invalid MoE shared expert spec: {s:?}"),
                        });
                    }
                    check_layer_dim(&mut problems, layer_idx, "shared n", s.n, MAX_EXPERTS);
                    check_layer_dim(
                        &mut problems,
                        layer_idx,
                        "shared dff",
                        s.dff,
                        MAX_FEATURE_DIM,
                    );
                }
                // A gated shared path with no shared experts has nothing to
                // gate; rejecting beats silently ignoring the flag.
                if *shared_gate && shared.is_none() {
                    problems.push(ModelsError::InvalidLayerSpec {
                        layer: layer_idx,
                        reason: "MoE shared_gate is true but shared is None; the gate would be silently ignored".to_string(),
                    });
                }
            }
            Ffn::None => {}
        }

        ModelsError::from_problems(problems)
    }
}

/// Token position encoding scheme (Spec 8 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PositionEncoding {
    /// Scalar 1D token position `[T]`.
    Scalar,
    /// Multimodal 3D rotary position sections `[T, 3]` (Spec 8 §3: MRope([u32; 3])).
    MRope([u32; 3]),
}

/// Speculative decoding N-gram specification (Spec 8 §3; Spec 1 §4.A; Spec 7 §6).
// DECISION(A1.3): explicit `dim` (Dn) instead of the hardcoded 32. Spec 1 §4.A
// requires `gather_staging [T, Np, Dn]` and `x [T, Np·Dn]`, but Spec 8 §3 omits
// Dn from NgramSpec; the table/projection shapes need it. Rejected keeping the
// unexplained `ngram_dim = 32` (unverifiable against either spec) and rejected
// inferring Dn from weights (shapes are declared before the loader binds).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NgramSpec {
    /// N-gram orders evaluated by hash heads.
    pub orders: Vec<u32>,
    /// Number of parallel hash heads `Np`.
    pub heads: u32,
    /// Row dimension `Dn` of each n-gram hash-table row (Spec 1 §4.A).
    pub dim: u32,
    /// Table sizes per head.
    pub table_sizes: Vec<u32>,
    /// Hash algorithm identifier.
    pub hash: HashId,
    /// Head combination method.
    pub combine: NgramCombine,
    /// Zero-indexed layer index at which n-gram embedding is injected.
    pub inject_at: u32,
}

impl NgramSpec {
    /// Validates n-gram specification fields.
    pub fn validate(&self, total_layers: usize) -> Result<(), ModelsError> {
        let mut problems = Vec::new();
        if self.heads == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "ngram heads count must be > 0".to_string(),
            });
        }
        check_model_dim(&mut problems, "ngram heads", self.heads, MAX_NGRAM_HEADS);
        if self.dim == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "ngram dim (Dn) must be > 0".to_string(),
            });
        }
        check_model_dim(&mut problems, "ngram dim", self.dim, MAX_FEATURE_DIM);
        if self.orders.is_empty() || self.orders.contains(&0) {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "ngram orders must be non-empty and all > 0".to_string(),
            });
        }
        if self.orders.len() != self.heads as usize {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "ngram orders length ({}) must equal heads ({})",
                    self.orders.len(),
                    self.heads
                ),
            });
        }
        if self.table_sizes.is_empty() || self.table_sizes.contains(&0) {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "ngram table_sizes must be non-empty and all > 0".to_string(),
            });
        }
        if self.table_sizes.len() != self.heads as usize {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "ngram table_sizes length ({}) must equal heads ({})",
                    self.table_sizes.len(),
                    self.heads
                ),
            });
        }
        for entries in &self.table_sizes {
            check_model_dim(
                &mut problems,
                "ngram table entries",
                *entries,
                MAX_NGRAM_TABLE_ENTRIES,
            );
        }
        if (self.inject_at as usize) >= total_layers {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "ngram inject_at ({}) out of bounds for total layers ({total_layers})",
                    self.inject_at
                ),
            });
        }
        ModelsError::from_problems(problems)
    }
}

/// Source of hidden states for multi-token prediction heads (Spec 8 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MtpSource {
    /// Consumes the post-final-norm hidden state from the last base model layer.
    Last,
    /// Consumes the hidden state output after base model layer `n`.
    Layer(u32),
}

/// Multi-Token Prediction (MTP) specification (Spec 8 §3, §5; Spec 7 §6).
#[derive(Debug, Clone, PartialEq)]
pub struct MtpSpec {
    /// Number of prediction heads.
    pub heads: u32,
    /// Layer specifications executed for each MTP head.
    pub layers_per_head: Vec<LayerSpec>,
    /// Where the initial hidden state for the MTP heads originates.
    pub takes_hidden_from: MtpSource,
}

impl MtpSpec {
    /// Validates MTP specification fields.
    pub fn validate(&self, total_layers: usize) -> Result<(), ModelsError> {
        let mut problems = Vec::new();
        if self.heads == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "mtp heads count must be > 0".to_string(),
            });
        }
        check_model_dim(&mut problems, "mtp heads", self.heads, MAX_MTP_HEADS);
        if self.layers_per_head.is_empty() {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "mtp layers_per_head must be non-empty".to_string(),
            });
        }
        if self.layers_per_head.len() > MAX_MTP_LAYERS_PER_HEAD as usize {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "mtp layers_per_head length {} exceeds implementation limit {MAX_MTP_LAYERS_PER_HEAD}",
                    self.layers_per_head.len()
                ),
            });
        }
        if let MtpSource::Layer(n) = self.takes_hidden_from {
            if (n as usize) >= total_layers {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!(
                        "mtp takes_hidden_from layer {n} exceeds total layers ({total_layers})"
                    ),
                });
            }
        }
        for (i, layer) in self.layers_per_head.iter().enumerate() {
            match u32::try_from(i) {
                Ok(idx) => {
                    if let Err(e) = layer.validate(idx) {
                        problems.push(e);
                    }
                }
                Err(_) => problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("mtp layer ordinal {i} exceeds u32 range"),
                }),
            }
        }
        ModelsError::from_problems(problems)
    }
}

/// Top-level model architecture specification (Spec 8 §3).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Model hidden dimension `Dm`.
    pub dm: u32,
    /// Layer specifications for each transformer block.
    pub layers: Vec<LayerSpec>,
    /// Vocabulary size `V`.
    pub vocab: u32,
    /// Embedding scaling factor (1.0 for standard, sqrt(Dm) for Gemma).
    pub embed_scale: f32,
    /// Whether the output LM head projection weight is tied to the embedding weight.
    pub tied_embeddings: bool,
    /// Final normalization layer specification.
    pub final_norm: NormSpec,
    /// Optional final output logit soft-capping threshold.
    pub final_logit_softcap: Option<f32>,
    /// Position encoding kind.
    pub positions: PositionEncoding,
    /// Optional n-gram speculative feature injection.
    pub ngram: Option<NgramSpec>,
    /// Optional multi-token prediction head specification.
    pub mtp: Option<MtpSpec>,
    /// Whether to export pre-lm_head final hidden states for external proposers (Eagle, block drafters).
    pub export_hidden: bool,
    /// End-of-sequence token IDs.
    pub eos_ids: Vec<u32>,
    /// Beginning-of-sequence token ID, if defined.
    pub bos_id: Option<u32>,
}

impl ModelSpec {
    /// Validates model specification parameters against Spec 8 §6 constraints.
    pub fn validate(&self) -> Result<(), ModelsError> {
        let mut problems = Vec::new();

        if self.dm == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "model dimension dm must be > 0".to_string(),
            });
        }
        check_model_dim(&mut problems, "dm", self.dm, MAX_FEATURE_DIM);
        if self.vocab == 0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "model vocab size must be > 0".to_string(),
            });
        }
        check_model_dim(&mut problems, "vocab", self.vocab, MAX_VOCAB_SIZE);
        if self.layers.is_empty() {
            problems.push(ModelsError::InvalidModelSpec {
                reason: "model must contain at least one layer".to_string(),
            });
        }
        if self.layers.len() > MAX_MODEL_LAYERS as usize {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "model layer count {} exceeds implementation limit {MAX_MODEL_LAYERS}",
                    self.layers.len()
                ),
            });
        }
        if !self.embed_scale.is_finite() || self.embed_scale <= 0.0 {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "embed_scale must be finite and > 0, got {}",
                    self.embed_scale
                ),
            });
        }
        if let Err(e) = self.final_norm.validate("final_norm") {
            problems.push(e);
        }
        if let Some(cap) = self.final_logit_softcap {
            if !cap.is_finite() || cap <= 0.0 {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("final_logit_softcap must be finite and > 0, got {cap}"),
                });
            }
        }
        if let PositionEncoding::MRope(sections) = self.positions {
            if sections.contains(&0) || sections.iter().any(|&s| !s.is_multiple_of(2)) {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("invalid mrope sections {sections:?}"),
                });
            }
        }
        if let Some(ngram) = &self.ngram {
            if let Err(e) = ngram.validate(self.layers.len()) {
                problems.push(e);
            }
        }
        if let Some(mtp) = &self.mtp {
            if let Err(e) = mtp.validate(self.layers.len()) {
                problems.push(e);
            }
        }

        for (i, layer) in self.layers.iter().enumerate() {
            match u32::try_from(i) {
                Ok(idx) => {
                    if let Err(e) = layer.validate(idx) {
                        problems.push(e);
                    }
                }
                Err(_) => problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("model layer ordinal {i} exceeds u32 range"),
                }),
            }
        }

        ModelsError::from_problems(problems)
    }
}
