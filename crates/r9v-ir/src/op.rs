// SPDX-License-Identifier: Apache-2.0
//! Closed Op catalog and op-level validation (Spec 1 §4; card A1.2).
//!
//! Every operation supported by the R9V inference engine belongs to the closed
//! set defined in Spec 1 §4. Ops are immutable descriptors with typed attributes.
//! Validation collects every failure before returning (CONVENTIONS.md §1.4).

use crate::sharding::{self, ShardingRule};
use crate::{
    matmul_numerics, moe_ffn_gemm_numerics, Class, DType, Dim, IrError, LayoutId, Numerics,
    Placement, QuantScheme, ReductionOrder, ShardLayoutPattern, StateHandle, StateKind,
};

// -----------------------------------------------------------------------------
// Attribute types and closed sets (Spec 1 §4)
// -----------------------------------------------------------------------------

/// Opaque n-gram hash function identifier (Spec 1 §4.A, CONVENTIONS.md §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashId(u64);

impl HashId {
    /// Creates a new hash identifier.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying raw identifier value.
    pub const fn as_u64(&self) -> u64 {
        self.0
    }
}

/// N-gram combination mode across hash heads (Spec 1 §4.A).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NgramCombine {
    /// Concatenate head embeddings along the feature axis.
    Concat,
    /// Elementwise sum across head embeddings.
    Sum,
}

// DECISION(A1.2): NgramSource specifies Staged versus Device per Spec 1 §4.A and SI-8; rejected unconstrained signatures because staged and device modes have distinct tensor signatures and placement rules.
/// N-gram source mode (Spec 1 §4.A, SI-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NgramSource {
    /// Staged mode: host gathers rows into gather_staging [T, Np, Dn] and row_scales.
    Staged,
    /// Device-table mode: hashes on device and gathers directly from table [TotalEntries, Dn].
    Device,
}

/// Weight smoothing representation for activation quantization (Spec 1 §4.A).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Smoothing {
    /// No smoothing transform applied.
    None,
    /// Folded smoothing pre-applied into weight matrices.
    Folded,
}

/// Normalization kind (Spec 1 §4.B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormKind {
    /// Root Mean Square normalization.
    Rms,
    /// Layer normalization (mean and variance centering).
    Layer,
}

/// Normalization axis (Spec 1 §4.B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormAxis {
    /// Normalize over the last feature dimension `N`.
    Last,
    /// Normalize over head dimension `D` for per-head QK-norm.
    Head(u32),
}

/// Non-linear activation function kind (Spec 1 §4.B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    /// SiLU (swish) activation.
    Silu,
    /// Standard Gaussian Error Linear Unit.
    Gelu,
    /// GELU with tanh approximation.
    GeluTanh,
    /// Squared ReLU: max(0, x)^2.
    Relu2,
    /// Identity pass-through.
    Identity,
}

/// Rotary Position Embedding style (Spec 1 §4.B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeStyle {
    /// GPT-NeoX style (split halves).
    Neox,
    /// LLaMA interleaved pairs style.
    Interleaved,
}

/// Rotary Position Embedding frequency scaling configuration (Spec 1 §4.B).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// No context window scaling.
    None,
    /// Linear position interpolation by a constant factor.
    Linear(f32),
    /// YaRN context extension scaling.
    Yarn {
        /// Interpolation factor.
        factor: f32,
        /// Fast beta frequency cutoff.
        beta_fast: f32,
        /// Slow beta frequency cutoff.
        beta_slow: f32,
        /// Original model training context limit.
        orig_ctx: u32,
        /// Target scale factor.
        mscale: f32,
    },
    /// Dynamic runtime NTK-aware frequency scaling.
    Dynamic,
}

/// Matmul epilogue fusion specification (Spec 1 §4.C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Epilogue {
    /// No epilogue.
    None,
    /// Add bias vector.
    Bias,
    /// Add residual activation tensor.
    Residual,
    /// Compute fused activation function.
    Act(ActivationKind),
}

/// MoE router scoring method (Spec 1 §4.C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeScoring {
    /// Softmax over router logits.
    Softmax,
    /// Independent sigmoid gating per expert.
    Sigmoid,
}

/// Grouped MoE configuration (Spec 1 §4.C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeGroup {
    /// Total number of expert groups.
    pub n_group: u32,
    /// Experts selected per group.
    pub topk_group: u32,
}

/// Cache scale quantization granularity (Spec 1 §4.D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheScaleGranularity {
    /// One scale per token per attention head.
    PerTokenHead,
    /// One scale per 32 or 64-element KV block.
    PerBlock,
}

/// MLA latent projection specification (Spec 1 §4.D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MlaLatent {
    /// Low-rank KV compression rank.
    pub kv_lora_rank: u32,
    /// Rotary embedding dimension for decoupled key.
    pub rope_dim: u32,
}

/// Attention causal and verification mask kind (Spec 1 §4.D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionMask {
    /// Standard autoregressive lower-triangular causal mask.
    Causal,
    /// Sliding window causal mask with window length `w`.
    CausalWindow(u32),
    /// Tree-verification mask driven by speculative parent pointers.
    Tree,
}

/// Multi-Head Latent Attention detailed configuration (Spec 1 §4.D, Spec 8 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MlaAttentionSpec {
    /// Query low-rank compression rank, if present.
    pub q_lora_rank: Option<u32>,
    /// KV low-rank compression rank.
    pub kv_lora_rank: u32,
    /// Non-rotary head dimension for Q/K.
    pub qk_nope_dim: u32,
    /// Rotary head dimension for decoupled Q/K.
    pub qk_rope_dim: u32,
    /// Value head dimension.
    pub v_dim: u32,
}

/// 1D convolution activation function (Spec 1 §4.E).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvActivation {
    /// SiLU activation.
    Silu,
    /// Identity (linear) activation.
    Identity,
}

/// Linear attention / SSM scan architecture kind (Spec 1 §4.E).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearAttnKind {
    /// Gated Delta Net recurrence.
    GatedDeltaNet,
    /// Gated Linear Attention scan.
    GLA,
    /// Mamba2 structured state-space duality scan.
    Mamba2,
}

/// Counter-based pseudo-random number generator algorithm (Spec 1 §4.F).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RngAlgorithm {
    /// Philox 4x32 10-round counter-based PRNG.
    Philox4x32,
}

/// Speculative decoding acceptance verification method (Spec 1 §4.F, Spec 7 §4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerifyMethod {
    /// Standard speculative rejection sampling with probability ratio.
    Rejection,
    /// Greedy argmax match verification.
    Greedy,
    /// Typical acceptance threshold verification with epsilon and delta parameters (Spec 7 §4).
    TypicalAcceptance {
        /// Acceptance probability floor epsilon.
        eps: f32,
        /// Entropy scaling factor delta.
        delta: f32,
    },
}

// DECISION(A1.2): GroupId wraps a private u64 field per CONVENTIONS.md §3.1 with as_u64() and as_u32() accessors; rejected bare primitive integers in public collective op signatures.
/// Communication collective group identifier (Spec 1 §4.G, CONVENTIONS.md §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(u64);

impl GroupId {
    /// Creates a new communication group identifier.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying group identifier integer as u64.
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// Compatibility accessor returning the group identifier as u32.
    pub const fn as_u32(&self) -> u32 {
        self.0 as u32
    }
}

impl std::fmt::Display for GroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Reduction operator for collective operations (Spec 1 §4.G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Elementwise summation across ranks.
    Sum,
}

// DECISION(A1.2): CopyKind defines Contiguize, DeviceToDevice, HostToDevice, and DeviceToHost per Spec 1 §4.A (device↔device, device↔host staging, or contiguization); rejected unconstrained copy because placement-aware validation requires checking transfer boundaries.
/// Copy boundary kind (Spec 1 §4.A).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CopyKind {
    /// Contiguization within the same placement.
    #[default]
    Contiguize,
    /// Transfer between distinct devices.
    DeviceToDevice,
    /// Host-to-device staging transfer.
    HostToDevice,
    /// Device-to-host staging transfer.
    DeviceToHost,
}

// DECISION(A1.2): SamplingParams is a typed non-Tensor external structure per Spec 1 §4.F and SI-12; rejected modeling as Tensor because DType is closed and cannot represent sparse bias tuples or heterogeneous float/int hyperparameters.
/// Sampling parameters per sequence (Spec 1 §4.F, SI-12).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Sampling temperature (>= 0.0, 0.0 means greedy).
    pub temperature: f32,
    /// Top-k candidate filtering (0 means disabled).
    pub top_k: u32,
    /// Top-p nucleus filtering threshold in (0.0, 1.0].
    pub top_p: f32,
    /// Min-p relative probability threshold in [0.0, 1.0].
    pub min_p: f32,
    /// Repetition penalty factor (> 0.0, 1.0 means disabled).
    pub repetition_penalty: f32,
    /// Additive presence penalty.
    pub presence_penalty: f32,
    /// Additive frequency penalty.
    pub frequency_penalty: f32,
    /// Sparse logit bias tuples `[(token_id, bias)]`.
    pub logit_bias: Vec<(u32, f32)>,
}

impl SamplingParams {
    /// Validates sampling parameters against Spec 1 §4.F constraints.
    pub fn validate(&self) -> Result<(), IrError> {
        let mut problems = Vec::new();
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "sampling_params",
                attribute: "temperature",
                reason: format!("must be finite and >= 0.0, got {}", self.temperature),
            });
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "sampling_params",
                attribute: "top_p",
                reason: format!("must be finite and in (0.0, 1.0], got {}", self.top_p),
            });
        }
        if !self.min_p.is_finite() || self.min_p < 0.0 || self.min_p > 1.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "sampling_params",
                attribute: "min_p",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.min_p),
            });
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "sampling_params",
                attribute: "repetition_penalty",
                reason: format!("must be finite and > 0.0, got {}", self.repetition_penalty),
            });
        }
        if !self.presence_penalty.is_finite() {
            problems.push(IrError::OpAttributeInvalid {
                op: "sampling_params",
                attribute: "presence_penalty",
                reason: format!("must be finite, got {}", self.presence_penalty),
            });
        }
        if !self.frequency_penalty.is_finite() {
            problems.push(IrError::OpAttributeInvalid {
                op: "sampling_params",
                attribute: "frequency_penalty",
                reason: format!("must be finite, got {}", self.frequency_penalty),
            });
        }
        for &(token, bias) in &self.logit_bias {
            if !bias.is_finite() {
                problems.push(IrError::OpAttributeInvalid {
                    op: "sampling_params",
                    attribute: "logit_bias",
                    reason: format!("token {} has non-finite bias {}", token, bias),
                });
            }
        }
        IrError::from_problems(problems)
    }
}

// -----------------------------------------------------------------------------
// Helper validation routines
// -----------------------------------------------------------------------------

fn check_rank(
    op: &'static str,
    tensor_name: &'static str,
    tensor: &crate::Tensor,
    expected: usize,
    problems: &mut Vec<IrError>,
) {
    if tensor.rank() != expected {
        problems.push(IrError::OpRankMismatch {
            op,
            tensor: tensor_name,
            expected,
            got: tensor.rank(),
        });
    }
}

fn check_dtype_in(
    op: &'static str,
    tensor_name: &'static str,
    tensor: &crate::Tensor,
    allowed: &[DType],
    problems: &mut Vec<IrError>,
) {
    if !allowed.contains(&tensor.dtype()) {
        problems.push(IrError::OpDTypeMismatch {
            op,
            tensor: tensor_name,
            expected: allowed.to_vec().into_boxed_slice(),
            got: tensor.dtype(),
        });
    }
}

fn check_dim_match(
    op: &'static str,
    t1_name: &'static str,
    d1: Dim,
    t2_name: &'static str,
    d2: Dim,
    axis_name: &'static str,
    problems: &mut Vec<IrError>,
) {
    match (d1, d2) {
        (Dim::Concrete(a), Dim::Concrete(b)) if a != b => {
            problems.push(IrError::OpShapeMismatch {
                op,
                tensor: t1_name,
                detail: format!(
                    "axis `{axis_name}` extent {a} does not match `{t2_name}` extent {b}"
                ),
            });
        }
        (Dim::Symbolic(s1), Dim::Symbolic(s2)) if s1 != s2 => {
            problems.push(IrError::OpShapeMismatch {
                op,
                tensor: t1_name,
                detail: format!(
                    "axis `{axis_name}` symbol {s1:?} does not match `{t2_name}` symbol {s2:?}"
                ),
            });
        }
        (Dim::Concrete(_), Dim::Concrete(_))
        | (Dim::Concrete(_), Dim::Symbolic(_))
        | (Dim::Symbolic(_), Dim::Concrete(_))
        | (Dim::Symbolic(_), Dim::Symbolic(_)) => {}
    }
}

fn check_gemm_activation_operand(
    op: &'static str,
    name: &'static str,
    tensor: &crate::Tensor,
    problems: &mut Vec<IrError>,
) {
    check_dtype_in(
        op,
        name,
        tensor,
        &[DType::F16, DType::Bf16, DType::I8, DType::E4m3],
        problems,
    );
    let quant_valid = match tensor.dtype() {
        DType::F16 | DType::Bf16 => Some(tensor.quant() == QuantScheme::None),
        DType::I8 => Some(matches!(
            tensor.quant(),
            QuantScheme::PerToken | QuantScheme::PerBlock32
        )),
        DType::E4m3 => Some(tensor.quant() == QuantScheme::PerToken),
        DType::F32 | DType::E5m2 | DType::I4 | DType::I32 | DType::U32 | DType::Bool => None,
    };
    if quant_valid == Some(false) {
        problems.push(IrError::OpQuantMismatch {
            op,
            tensor: name,
            quant: tensor.quant(),
        });
    }
}

fn check_gemm_weight_operand(
    op: &'static str,
    name: &'static str,
    tensor: &crate::Tensor,
    problems: &mut Vec<IrError>,
) {
    check_dtype_in(
        op,
        name,
        tensor,
        &[DType::I4, DType::I8, DType::E4m3, DType::F16],
        problems,
    );
    let quant_valid = match tensor.dtype() {
        DType::I4 | DType::I8 | DType::E4m3 => Some(matches!(
            tensor.quant(),
            QuantScheme::PerRow | QuantScheme::Scheme(_)
        )),
        DType::F16 => Some(tensor.quant() == QuantScheme::None),
        DType::F32 | DType::Bf16 | DType::E5m2 | DType::I32 | DType::U32 | DType::Bool => None,
    };
    if quant_valid == Some(false) {
        problems.push(IrError::OpQuantMismatch {
            op,
            tensor: name,
            quant: tensor.quant(),
        });
    }
}

// -----------------------------------------------------------------------------
// §4.A Data movement and lookup
// -----------------------------------------------------------------------------

/// Token embedding lookup op (Spec 1 §4.A).
///
/// `token_ids [T] u32, table [V, Dm] (i4|i8|f16, Block|PerRow, Device|Tiered) -> x [T, Dm] act_dtype`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbedGatherOp {
    /// Scaling applied to gathered embeddings (e.g. sqrt(Dm) for Gemma).
    pub scale: f32,
    /// Destination activation dtype.
    pub out_dtype: DType,
}

impl EmbedGatherOp {
    /// Validates inputs and outputs against Spec 1 §4.A constraints.
    // DECISION(A1.2): Op validate() records attribute errors and both input and output count mismatches simultaneously into problems before guarding tensor indexing per CONVENTIONS.md §1.4; rejected early return on first count mismatch so that all structural defects are surfaced at once.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !self.scale.is_finite() || self.scale <= 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "embed_gather",
                attribute: "scale",
                reason: format!("must be finite and > 0, got {}", self.scale),
            });
        }
        if !matches!(self.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "embed_gather",
                attribute: "out_dtype",
                reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
            });
        }

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "embed_gather",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "embed_gather",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let token_ids = &inputs[0];
            let table = &inputs[1];

            check_rank("embed_gather", "token_ids", token_ids, 1, &mut problems);
            check_dtype_in(
                "embed_gather",
                "token_ids",
                token_ids,
                &[DType::U32],
                &mut problems,
            );
            if token_ids.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "embed_gather",
                    tensor: "token_ids",
                    expected: Class::Activation,
                    got: token_ids.class(),
                });
            }
            if !matches!(token_ids.placement(), Placement::Device { .. }) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "embed_gather",
                    tensor: "token_ids",
                    placement: token_ids.placement(),
                });
            }

            check_rank("embed_gather", "table", table, 2, &mut problems);
            check_dtype_in(
                "embed_gather",
                "table",
                table,
                &[DType::I4, DType::I8, DType::F16],
                &mut problems,
            );
            if table.class() != Class::Weight {
                problems.push(IrError::OpClassMismatch {
                    op: "embed_gather",
                    tensor: "table",
                    expected: Class::Weight,
                    got: table.class(),
                });
            }
            if !matches!(
                table.placement(),
                Placement::Device { .. } | Placement::Tiered
            ) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "embed_gather",
                    tensor: "table",
                    placement: table.placement(),
                });
            }

            match table.dtype() {
                DType::F16 => {
                    if table.quant() != QuantScheme::None {
                        problems.push(IrError::OpQuantMismatch {
                            op: "embed_gather",
                            tensor: "table",
                            quant: table.quant(),
                        });
                    }
                }
                DType::I4 => {
                    if !matches!(table.quant(), QuantScheme::PerRow | QuantScheme::Scheme(_)) {
                        problems.push(IrError::OpQuantMismatch {
                            op: "embed_gather",
                            tensor: "table",
                            quant: table.quant(),
                        });
                    }
                }
                DType::I8 => {
                    if !matches!(table.quant(), QuantScheme::PerRow | QuantScheme::Scheme(_)) {
                        problems.push(IrError::OpQuantMismatch {
                            op: "embed_gather",
                            tensor: "table",
                            quant: table.quant(),
                        });
                    }
                }
                DType::F32
                | DType::Bf16
                | DType::E4m3
                | DType::E5m2
                | DType::I32
                | DType::U32
                | DType::Bool => {}
            }
        }

        if output_count_valid {
            let x = &outputs[0];
            check_rank("embed_gather", "x", x, 2, &mut problems);
            if x.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "embed_gather",
                    tensor: "x",
                    expected: vec![self.out_dtype].into_boxed_slice(),
                    got: x.dtype(),
                });
            }
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "embed_gather",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
            if !matches!(x.placement(), Placement::Device { .. }) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "embed_gather",
                    tensor: "x",
                    placement: x.placement(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let token_ids = &inputs[0];
            let table = &inputs[1];
            let x = &outputs[0];

            if token_ids.rank() == 1 && x.rank() == 2 {
                check_dim_match(
                    "embed_gather",
                    "x",
                    x.shape()[0],
                    "token_ids",
                    token_ids.shape()[0],
                    "T",
                    &mut problems,
                );
            }
            if table.rank() == 2 && x.rank() == 2 {
                check_dim_match(
                    "embed_gather",
                    "x",
                    x.shape()[1],
                    "table",
                    table.shape()[1],
                    "Dm",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.A, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::EMBED_GATHER_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::EMBED_GATHER_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.A, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::None)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "embed_gather"
    }
}

/// N-gram speculative prefix gather op (Spec 1 §4.A, SI-8).
///
/// Staged: `gather_staging [T, Np, Dn] (i4|i8, Block), row_scales -> x [T, Np*Dn] act_dtype`
/// Device: `token_ids [T] u32, table [TotalEntries, Dn] -> x [T, Np*Dn] act_dtype`
#[derive(Debug, Clone, PartialEq)]
pub struct NgramGatherOp {
    /// N-gram source mode: Staged buffer vs Device table.
    pub source: NgramSource,
    /// N-gram orders evaluated by the hash heads.
    pub orders: Box<[u32]>,
    /// Number of parallel n-gram hash heads `Np`.
    pub heads: u32,
    /// Hash function identifier.
    pub hash: HashId,
    /// Size of the n-gram table per head.
    pub table_sizes: Box<[u32]>,
    /// How gathered head embeddings are combined.
    pub combine: NgramCombine,
    /// Destination activation dtype.
    pub out_dtype: DType,
}

impl NgramGatherOp {
    /// Validates inputs and outputs against Spec 1 §4.A constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if self.heads == 0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "ngram_gather",
                attribute: "heads",
                reason: "heads must be > 0".to_string(),
            });
        }
        if self.orders.is_empty() || self.orders.contains(&0) {
            problems.push(IrError::OpAttributeInvalid {
                op: "ngram_gather",
                attribute: "orders",
                reason: "orders must be non-empty with elements > 0".to_string(),
            });
        }
        if self.table_sizes.is_empty() || self.table_sizes.contains(&0) {
            problems.push(IrError::OpAttributeInvalid {
                op: "ngram_gather",
                attribute: "table_sizes",
                reason: "table_sizes must be non-empty with elements > 0".to_string(),
            });
        }
        if self.orders.len() != self.heads as usize {
            problems.push(IrError::OpAttributeInvalid {
                op: "ngram_gather",
                attribute: "orders",
                reason: format!(
                    "length {} must equal heads {}",
                    self.orders.len(),
                    self.heads
                ),
            });
        }
        if self.table_sizes.len() != self.heads as usize {
            problems.push(IrError::OpAttributeInvalid {
                op: "ngram_gather",
                attribute: "table_sizes",
                reason: format!(
                    "length {} must equal heads {}",
                    self.table_sizes.len(),
                    self.heads
                ),
            });
        }
        if !matches!(self.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "ngram_gather",
                attribute: "out_dtype",
                reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
            });
        }

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "ngram_gather",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "ngram_gather",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            match self.source {
                NgramSource::Staged => {
                    let staging = &inputs[0];
                    let scales = &inputs[1];

                    check_rank("ngram_gather", "gather_staging", staging, 3, &mut problems);
                    check_dtype_in(
                        "ngram_gather",
                        "gather_staging",
                        staging,
                        &[DType::I4, DType::I8],
                        &mut problems,
                    );
                    if !matches!(staging.quant(), QuantScheme::Scheme(_)) {
                        problems.push(IrError::OpQuantMismatch {
                            op: "ngram_gather",
                            tensor: "gather_staging",
                            quant: staging.quant(),
                        });
                    }
                    if !matches!(staging.class(), Class::Staging | Class::Weight) {
                        problems.push(IrError::OpClassMismatch {
                            op: "ngram_gather",
                            tensor: "gather_staging",
                            expected: Class::Staging,
                            got: staging.class(),
                        });
                    }
                    if !matches!(staging.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "ngram_gather",
                            tensor: "gather_staging",
                            placement: staging.placement(),
                        });
                    }

                    if scales.rank() != 1 && scales.rank() != 2 {
                        problems.push(IrError::OpRankMismatch {
                            op: "ngram_gather",
                            tensor: "row_scales",
                            expected: 2,
                            got: scales.rank(),
                        });
                    }
                    check_dtype_in(
                        "ngram_gather",
                        "row_scales",
                        scales,
                        &[DType::F32, DType::F16],
                        &mut problems,
                    );
                    if !matches!(scales.class(), Class::Activation | Class::Staging) {
                        problems.push(IrError::OpClassMismatch {
                            op: "ngram_gather",
                            tensor: "row_scales",
                            expected: Class::Activation,
                            got: scales.class(),
                        });
                    }
                    if !matches!(scales.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "ngram_gather",
                            tensor: "row_scales",
                            placement: scales.placement(),
                        });
                    }

                    if staging.rank() == 3 {
                        if let Dim::Concrete(heads) = staging.shape()[1] {
                            if heads != self.heads {
                                problems.push(IrError::OpShapeMismatch {
                                    op: "ngram_gather",
                                    tensor: "gather_staging",
                                    detail: format!(
                                        "dim 1 heads {heads} != attr heads {}",
                                        self.heads
                                    ),
                                });
                            }
                        }
                    }

                    // Independent input-to-input checks:
                    if staging.rank() == 3 && (scales.rank() == 1 || scales.rank() == 2) {
                        check_dim_match(
                            "ngram_gather",
                            "row_scales",
                            scales.shape()[0],
                            "gather_staging",
                            staging.shape()[0],
                            "T",
                            &mut problems,
                        );
                        if scales.rank() == 2 {
                            if let (Dim::Concrete(sc_h), Dim::Concrete(st_h)) =
                                (scales.shape()[1], staging.shape()[1])
                            {
                                if sc_h != 1 && sc_h != st_h {
                                    problems.push(IrError::OpShapeMismatch {
                                        op: "ngram_gather",
                                        tensor: "row_scales",
                                        detail: format!(
                                            "axis 1 heads {sc_h} != gather_staging heads {st_h}"
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
                NgramSource::Device => {
                    let token_ids = &inputs[0];
                    let table = &inputs[1];

                    check_rank("ngram_gather", "token_ids", token_ids, 1, &mut problems);
                    check_dtype_in(
                        "ngram_gather",
                        "token_ids",
                        token_ids,
                        &[DType::U32],
                        &mut problems,
                    );
                    if token_ids.class() != Class::Activation {
                        problems.push(IrError::OpClassMismatch {
                            op: "ngram_gather",
                            tensor: "token_ids",
                            expected: Class::Activation,
                            got: token_ids.class(),
                        });
                    }
                    if !matches!(token_ids.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "ngram_gather",
                            tensor: "token_ids",
                            placement: token_ids.placement(),
                        });
                    }

                    check_rank("ngram_gather", "table", table, 2, &mut problems);
                    check_dtype_in(
                        "ngram_gather",
                        "table",
                        table,
                        &[DType::I4, DType::I8, DType::F16, DType::Bf16, DType::F32],
                        &mut problems,
                    );
                    if table.class() != Class::Weight {
                        problems.push(IrError::OpClassMismatch {
                            op: "ngram_gather",
                            tensor: "table",
                            expected: Class::Weight,
                            got: table.class(),
                        });
                    }
                    if !matches!(table.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "ngram_gather",
                            tensor: "table",
                            placement: table.placement(),
                        });
                    }
                    if table.rank() == 2 {
                        let expected_entries = self
                            .table_sizes
                            .iter()
                            .try_fold(0u32, |sum, size| sum.checked_add(*size));
                        match (table.shape()[0], expected_entries) {
                            (Dim::Concrete(e), Some(expected)) if e != expected => {
                                problems.push(IrError::OpShapeMismatch {
                                    op: "ngram_gather",
                                    tensor: "table",
                                    detail: format!(
                                        "table dim 0 extent {e} != sum of table_sizes ({expected})"
                                    ),
                                });
                            }
                            (_, None) => problems.push(IrError::OpAttributeInvalid {
                                op: "ngram_gather",
                                attribute: "table_sizes",
                                reason: "sum of table_sizes exceeds u32::MAX".to_string(),
                            }),
                            (Dim::Concrete(_), Some(_)) | (Dim::Symbolic(_), Some(_)) => {}
                        }
                    }
                }
            }
        }

        if output_count_valid {
            let x = &outputs[0];
            check_rank("ngram_gather", "x", x, 2, &mut problems);
            if x.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "ngram_gather",
                    tensor: "x",
                    expected: vec![self.out_dtype].into_boxed_slice(),
                    got: x.dtype(),
                });
            }
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "ngram_gather",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
            if !matches!(x.placement(), Placement::Device { .. }) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "ngram_gather",
                    tensor: "x",
                    placement: x.placement(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &outputs[0];
            match self.source {
                NgramSource::Staged => {
                    let staging = &inputs[0];
                    if staging.rank() == 3 && x.rank() == 2 {
                        check_dim_match(
                            "ngram_gather",
                            "x",
                            x.shape()[0],
                            "gather_staging",
                            staging.shape()[0],
                            "T",
                            &mut problems,
                        );
                        if let (Dim::Concrete(x_d), Dim::Concrete(st_d)) =
                            (x.shape()[1], staging.shape()[2])
                        {
                            match self.combine {
                                NgramCombine::Concat => {
                                    if self.heads.checked_mul(st_d) != Some(x_d) {
                                        problems.push(IrError::OpShapeMismatch {
                                            op: "ngram_gather",
                                            tensor: "x",
                                            detail: format!(
                                                "concat dim 1 {x_d} != heads({}) * Dn({})",
                                                self.heads, st_d
                                            ),
                                        });
                                    }
                                }
                                NgramCombine::Sum => {
                                    if x_d != st_d {
                                        problems.push(IrError::OpShapeMismatch {
                                            op: "ngram_gather",
                                            tensor: "x",
                                            detail: format!("sum dim 1 {x_d} != Dn({st_d})"),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                NgramSource::Device => {
                    let token_ids = &inputs[0];
                    let table = &inputs[1];
                    if token_ids.rank() == 1 && x.rank() == 2 {
                        check_dim_match(
                            "ngram_gather",
                            "x",
                            x.shape()[0],
                            "token_ids",
                            token_ids.shape()[0],
                            "T",
                            &mut problems,
                        );
                    }
                    if table.rank() == 2 && x.rank() == 2 {
                        if let (Dim::Concrete(x_d), Dim::Concrete(tb_d)) =
                            (x.shape()[1], table.shape()[1])
                        {
                            match self.combine {
                                NgramCombine::Concat => {
                                    if self.heads.checked_mul(tb_d) != Some(x_d) {
                                        problems.push(IrError::OpShapeMismatch {
                                            op: "ngram_gather",
                                            tensor: "x",
                                            detail: format!(
                                                "concat dim 1 {x_d} != heads({}) * Dn({})",
                                                self.heads, tb_d
                                            ),
                                        });
                                    }
                                }
                                NgramCombine::Sum => {
                                    if x_d != tb_d {
                                        problems.push(IrError::OpShapeMismatch {
                                            op: "ngram_gather",
                                            tensor: "x",
                                            detail: format!("sum dim 1 {x_d} != Dn({tb_d})"),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.A, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::NGRAM_GATHER_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::NGRAM_GATHER_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.A, §6.1).
    pub fn numerics(&self) -> Numerics {
        match self.combine {
            NgramCombine::Concat => Numerics::f32(ReductionOrder::None),
            NgramCombine::Sum => Numerics::f32(ReductionOrder::AscendingIndex),
        }
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "ngram_gather"
    }
}

// DECISION(A1.2): QuantActOp carries scheme: QuantScheme (PerToken or PerBlock32) and target: DType per Spec 1 §4.A, Spec 2 §3.4, and SI-7; rejected restricting to PerToken because GGUF MMQ models require i8 PerBlock32 with scale [T, N/32] for llama.cpp parity.
/// Activation quantization op (Spec 1 §4.A, Spec 2 §3.4, SI-7).
///
/// `x [T, N] (f16|bf16|f32) -> xq [T, N] (i8|e4m3), scale [T] or [T, N/32] f32`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantActOp {
    /// Activation quantization scheme: PerToken or PerBlock32.
    pub scheme: QuantScheme,
    /// Target quantized element dtype (`i8` or `e4m3`).
    pub target: DType,
    /// Weight smoothing mode.
    pub smoothing: Smoothing,
}

impl QuantActOp {
    /// Validates inputs and outputs against Spec 1 §4.A constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !matches!(self.scheme, QuantScheme::PerToken | QuantScheme::PerBlock32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "quant_act",
                attribute: "scheme",
                reason: format!("must be PerToken or PerBlock32, got {:?}", self.scheme),
            });
        }
        if self.target != DType::I8 && self.target != DType::E4m3 {
            problems.push(IrError::OpAttributeInvalid {
                op: "quant_act",
                attribute: "target",
                reason: format!("must be i8 or e4m3, got {:?}", self.target),
            });
        }
        if self.scheme == QuantScheme::PerBlock32 && self.target != DType::I8 {
            problems.push(IrError::OpAttributeInvalid {
                op: "quant_act",
                attribute: "target",
                reason: format!(
                    "PerBlock32 activation quantization is only valid with target i8, got {:?}",
                    self.target
                ),
            });
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "quant_act",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 2;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "quant_act",
                expected: 2,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            check_rank("quant_act", "x", x, 2, &mut problems);
            check_dtype_in(
                "quant_act",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "quant_act",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
        }

        if output_count_valid {
            let xq = &outputs[0];
            let scale = &outputs[1];

            check_rank("quant_act", "xq", xq, 2, &mut problems);
            if xq.dtype() != self.target {
                problems.push(IrError::OpDTypeMismatch {
                    op: "quant_act",
                    tensor: "xq",
                    expected: vec![self.target].into_boxed_slice(),
                    got: xq.dtype(),
                });
            }
            if xq.quant() != self.scheme {
                problems.push(IrError::OpQuantMismatch {
                    op: "quant_act",
                    tensor: "xq",
                    quant: xq.quant(),
                });
            }
            if xq.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "quant_act",
                    tensor: "xq",
                    expected: Class::Activation,
                    got: xq.class(),
                });
            }
            if !matches!(xq.placement(), Placement::Device { .. }) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "quant_act",
                    tensor: "xq",
                    placement: xq.placement(),
                });
            }

            match self.scheme {
                QuantScheme::PerToken => {
                    check_rank("quant_act", "scale", scale, 1, &mut problems);
                }
                QuantScheme::PerBlock32 => {
                    check_rank("quant_act", "scale", scale, 2, &mut problems);
                }
                QuantScheme::None | QuantScheme::PerRow | QuantScheme::Scheme(_) => {
                    check_rank("quant_act", "scale", scale, 1, &mut problems);
                }
            }
            check_dtype_in("quant_act", "scale", scale, &[DType::F32], &mut problems);
            if scale.quant() != QuantScheme::None {
                problems.push(IrError::OpQuantMismatch {
                    op: "quant_act",
                    tensor: "scale",
                    quant: scale.quant(),
                });
            }
            if scale.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "quant_act",
                    tensor: "scale",
                    expected: Class::Activation,
                    got: scale.class(),
                });
            }
            if !matches!(scale.placement(), Placement::Device { .. }) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "quant_act",
                    tensor: "scale",
                    placement: scale.placement(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let xq = &outputs[0];
            let scale = &outputs[1];

            if x.rank() == 2 && xq.rank() == 2 {
                check_dim_match(
                    "quant_act",
                    "xq",
                    xq.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "quant_act",
                    "xq",
                    xq.shape()[1],
                    "x",
                    x.shape()[1],
                    "N",
                    &mut problems,
                );
            }
            if x.rank() == 2 {
                check_dim_match(
                    "quant_act",
                    "scale",
                    scale.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );

                if self.scheme == QuantScheme::PerBlock32 {
                    if let Dim::Concrete(n) = x.shape()[1] {
                        if n % 32 != 0 {
                            problems.push(IrError::OpShapeMismatch {
                                op: "quant_act",
                                tensor: "x",
                                detail: format!(
                                    "axis 1 extent {n} must be divisible by 32 for PerBlock32"
                                ),
                            });
                        } else if scale.rank() == 2 {
                            if let Dim::Concrete(scale_blocks) = scale.shape()[1] {
                                if scale_blocks != n / 32 {
                                    problems.push(IrError::OpShapeMismatch {
                                        op: "quant_act",
                                        tensor: "scale",
                                        detail: format!(
                                            "scale axis 1 extent {scale_blocks} != N/32 ({})",
                                            n / 32
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.A, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::QUANT_ACT_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::QUANT_ACT_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.A, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingAxis)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "quant_act"
    }
}

/// Elementwise precision cast op (Spec 1 §4.A).
///
/// `x -> y` with `attrs: dtype`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CastOp {
    /// Destination element dtype.
    pub dtype: DType,
}

impl CastOp {
    /// Validates inputs and outputs against Spec 1 §4.A constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "cast",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "cast",
                expected: 1,
                got: outputs.len(),
            });
        }

        if output_count_valid {
            let y = &outputs[0];
            if y.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "cast",
                    tensor: "y",
                    expected: vec![self.dtype].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let y = &outputs[0];
            if x.rank() != y.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "cast",
                    tensor: "y",
                    expected: x.rank(),
                    got: y.rank(),
                });
            } else {
                for i in 0..x.rank() {
                    check_dim_match(
                        "cast",
                        "y",
                        y.shape()[i],
                        "x",
                        x.shape()[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
            if y.layout() != x.layout() {
                problems.push(IrError::OpLayoutMismatch {
                    op: "cast",
                    tensor: "y",
                    expected: x.layout(),
                    got: y.layout(),
                });
            }
            if y.quant() != x.quant() {
                problems.push(IrError::OpQuantMismatch {
                    op: "cast",
                    tensor: "y",
                    quant: y.quant(),
                });
            }
            if y.placement() != x.placement() {
                problems.push(IrError::OpPlacementMismatch {
                    op: "cast",
                    tensor: "y",
                    placement: y.placement(),
                });
            }
            if y.class() != x.class() {
                problems.push(IrError::OpClassMismatch {
                    op: "cast",
                    tensor: "y",
                    expected: x.class(),
                    got: y.class(),
                });
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.A, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::PASSTHROUGH_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::PASSTHROUGH_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.A, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "cast"
    }
}

/// Tensor memory copy and contiguization op (Spec 1 §4.A).
///
/// `x -> y`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CopyOp {
    /// Copy kind and transfer boundary.
    pub kind: CopyKind,
}

impl CopyOp {
    /// Validates inputs and outputs against Spec 1 §4.A constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "copy",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "copy",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let y = &outputs[0];

            if y.dtype() != x.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "copy",
                    tensor: "y",
                    expected: vec![x.dtype()].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if x.rank() != y.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "copy",
                    tensor: "y",
                    expected: x.rank(),
                    got: y.rank(),
                });
            } else {
                for i in 0..x.rank() {
                    check_dim_match(
                        "copy",
                        "y",
                        y.shape()[i],
                        "x",
                        x.shape()[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
            if y.quant() != x.quant() {
                problems.push(IrError::OpQuantMismatch {
                    op: "copy",
                    tensor: "y",
                    quant: y.quant(),
                });
            }
            if y.class() != x.class() {
                problems.push(IrError::OpClassMismatch {
                    op: "copy",
                    tensor: "y",
                    expected: x.class(),
                    got: y.class(),
                });
            }

            match self.kind {
                CopyKind::Contiguize => {
                    if x.placement() != y.placement() {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "y",
                            placement: y.placement(),
                        });
                    }
                    if y.layout() != LayoutId::CONTIGUOUS {
                        problems.push(IrError::OpLayoutMismatch {
                            op: "copy",
                            tensor: "y",
                            expected: LayoutId::CONTIGUOUS,
                            got: y.layout(),
                        });
                    }
                }
                CopyKind::DeviceToDevice => {
                    if !matches!(x.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "x",
                            placement: x.placement(),
                        });
                    }
                    if !matches!(y.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "y",
                            placement: y.placement(),
                        });
                    }
                    if matches!(x.placement(), Placement::Device { .. })
                        && matches!(y.placement(), Placement::Device { .. })
                        && x.placement() == y.placement()
                    {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "y",
                            placement: y.placement(),
                        });
                    }
                    if y.layout() != x.layout() {
                        problems.push(IrError::OpLayoutMismatch {
                            op: "copy",
                            tensor: "y",
                            expected: x.layout(),
                            got: y.layout(),
                        });
                    }
                }
                CopyKind::HostToDevice => {
                    if !matches!(x.placement(), Placement::Host | Placement::Tiered) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "x",
                            placement: x.placement(),
                        });
                    }
                    if !matches!(y.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "y",
                            placement: y.placement(),
                        });
                    }
                    if y.layout() != x.layout() {
                        problems.push(IrError::OpLayoutMismatch {
                            op: "copy",
                            tensor: "y",
                            expected: x.layout(),
                            got: y.layout(),
                        });
                    }
                }
                CopyKind::DeviceToHost => {
                    if !matches!(x.placement(), Placement::Device { .. }) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "x",
                            placement: x.placement(),
                        });
                    }
                    if !matches!(y.placement(), Placement::Host | Placement::Tiered) {
                        problems.push(IrError::OpPlacementMismatch {
                            op: "copy",
                            tensor: "y",
                            placement: y.placement(),
                        });
                    }
                    if y.layout() != x.layout() {
                        problems.push(IrError::OpLayoutMismatch {
                            op: "copy",
                            tensor: "y",
                            expected: x.layout(),
                            got: y.layout(),
                        });
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.A, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::PASSTHROUGH_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::PASSTHROUGH_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.A, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "copy"
    }
}

/// Row gather op by integer indices (Spec 1 §4.A, SI-10).
///
/// `x [N, D], indices [M] u32 -> y [M, D]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GatherRowsOp;

impl GatherRowsOp {
    /// Validates inputs and outputs against Spec 1 §4.A constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "gather_rows",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "gather_rows",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            let indices = &inputs[1];

            check_rank("gather_rows", "x", x, 2, &mut problems);
            check_dtype_in(
                "gather_rows",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "gather_rows",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }

            check_rank("gather_rows", "indices", indices, 1, &mut problems);
            check_dtype_in(
                "gather_rows",
                "indices",
                indices,
                &[DType::U32],
                &mut problems,
            );
            if indices.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "gather_rows",
                    tensor: "indices",
                    expected: Class::Activation,
                    got: indices.class(),
                });
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("gather_rows", "y", y, 2, &mut problems);
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "gather_rows",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let indices = &inputs[1];
            let y = &outputs[0];

            if y.dtype() != x.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "gather_rows",
                    tensor: "y",
                    expected: vec![x.dtype()].into_boxed_slice(),
                    got: y.dtype(),
                });
            }

            if indices.rank() == 1 && y.rank() == 2 {
                check_dim_match(
                    "gather_rows",
                    "y",
                    y.shape()[0],
                    "indices",
                    indices.shape()[0],
                    "M",
                    &mut problems,
                );
            }
            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "gather_rows",
                    "y",
                    y.shape()[1],
                    "x",
                    x.shape()[1],
                    "D",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.A, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::GATHER_ROWS_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::GATHER_ROWS_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.A, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "gather_rows"
    }
}

/// Deterministic scatter-add rows op (Spec 1 §4.A, SI-10).
///
/// `x [M, D], indices [M] u32, dest? [N, D] -> y [N, D]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ScatterAddRowsOp;

impl ScatterAddRowsOp {
    /// Validates inputs and outputs against Spec 1 §4.A constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 2 || inputs.len() == 3;
        if !input_count_valid {
            problems.push(IrError::OpInputCountCandidatesMismatch {
                op: "scatter_add_rows",
                expected: vec![2, 3].into_boxed_slice(),
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "scatter_add_rows",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            let indices = &inputs[1];

            check_rank("scatter_add_rows", "x", x, 2, &mut problems);
            check_dtype_in(
                "scatter_add_rows",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "scatter_add_rows",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }

            check_rank("scatter_add_rows", "indices", indices, 1, &mut problems);
            check_dtype_in(
                "scatter_add_rows",
                "indices",
                indices,
                &[DType::U32],
                &mut problems,
            );
            if indices.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "scatter_add_rows",
                    tensor: "indices",
                    expected: Class::Activation,
                    got: indices.class(),
                });
            }

            if inputs.len() == 3 {
                let dest = &inputs[2];
                check_rank("scatter_add_rows", "dest", dest, 2, &mut problems);
                if dest.dtype() != x.dtype() {
                    problems.push(IrError::OpDTypeMismatch {
                        op: "scatter_add_rows",
                        tensor: "dest",
                        expected: vec![x.dtype()].into_boxed_slice(),
                        got: dest.dtype(),
                    });
                }
                if dest.class() != Class::Activation {
                    problems.push(IrError::OpClassMismatch {
                        op: "scatter_add_rows",
                        tensor: "dest",
                        expected: Class::Activation,
                        got: dest.class(),
                    });
                }
            }

            // Independent input-to-input checks:
            if x.rank() == 2 && indices.rank() == 1 {
                check_dim_match(
                    "scatter_add_rows",
                    "x",
                    x.shape()[0],
                    "indices",
                    indices.shape()[0],
                    "M",
                    &mut problems,
                );
            }
            if inputs.len() == 3 {
                let dest = &inputs[2];
                if dest.rank() == 2 && x.rank() == 2 {
                    check_dim_match(
                        "scatter_add_rows",
                        "dest",
                        dest.shape()[1],
                        "x",
                        x.shape()[1],
                        "D",
                        &mut problems,
                    );
                }
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("scatter_add_rows", "y", y, 2, &mut problems);
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "scatter_add_rows",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let y = &outputs[0];

            if y.dtype() != x.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "scatter_add_rows",
                    tensor: "y",
                    expected: vec![x.dtype()].into_boxed_slice(),
                    got: y.dtype(),
                });
            }

            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "scatter_add_rows",
                    "y",
                    y.shape()[1],
                    "x",
                    x.shape()[1],
                    "D",
                    &mut problems,
                );
            }

            if inputs.len() == 3 {
                let dest = &inputs[2];
                if dest.rank() == 2 && y.rank() == 2 {
                    check_dim_match(
                        "scatter_add_rows",
                        "y",
                        y.shape()[0],
                        "dest",
                        dest.shape()[0],
                        "N",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.A, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::SCATTER_ADD_ROWS_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::SCATTER_ADD_ROWS_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.A, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "scatter_add_rows"
    }
}

/// Last-axis channel split op (card A1.14, SI-29).
///
/// `x [T, H, D] -> (a [T, H, first], b [T, H, D - first])`
///
/// Splits the MLA compressed-latent / decoupled-rotary channel ranges into
/// explicit edges so `rope` consumes only the rotary part (Spec 1 §4.B,
/// Spec 8 §3.1). Pure data movement: values are copied unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SplitOp {
    /// Width of the first output along the last axis; must satisfy
    /// `0 < first < D` so both outputs are non-empty.
    pub first: u32,
}

impl SplitOp {
    /// Validates inputs and outputs against the split contract.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if self.first == 0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "split",
                attribute: "first",
                reason: "split width must be > 0 so both outputs are non-empty".to_string(),
            });
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "split",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 2;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "split",
                expected: 2,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            check_rank("split", "x", x, 3, &mut problems);
            check_dtype_in(
                "split",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "split",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
            if x.rank() == 3 {
                if let Dim::Concrete(d) = x.shape()[2] {
                    if self.first >= d {
                        problems.push(IrError::OpAttributeInvalid {
                            op: "split",
                            attribute: "first",
                            reason: format!(
                                "split width {} must be < last-axis dim {d} so both outputs are non-empty",
                                self.first
                            ),
                        });
                    }
                }
            }
        }

        if output_count_valid {
            for (name, o) in [("a", &outputs[0]), ("b", &outputs[1])] {
                check_rank("split", name, o, 3, &mut problems);
                check_dtype_in(
                    "split",
                    name,
                    o,
                    &[DType::F16, DType::Bf16, DType::F32],
                    &mut problems,
                );
                if o.class() != Class::Activation {
                    problems.push(IrError::OpClassMismatch {
                        op: "split",
                        tensor: name,
                        expected: Class::Activation,
                        got: o.class(),
                    });
                }
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let a = &outputs[0];
            let b = &outputs[1];
            if a.dtype() != x.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "split",
                    tensor: "a",
                    expected: vec![x.dtype()].into_boxed_slice(),
                    got: a.dtype(),
                });
            }
            if b.dtype() != x.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "split",
                    tensor: "b",
                    expected: vec![x.dtype()].into_boxed_slice(),
                    got: b.dtype(),
                });
            }
            if x.rank() == 3 && a.rank() == 3 && b.rank() == 3 {
                for (axis, axis_name) in [(0, "T"), (1, "H")] {
                    check_dim_match(
                        "split",
                        "a",
                        a.shape()[axis],
                        "x",
                        x.shape()[axis],
                        axis_name,
                        &mut problems,
                    );
                    check_dim_match(
                        "split",
                        "b",
                        b.shape()[axis],
                        "x",
                        x.shape()[axis],
                        axis_name,
                        &mut problems,
                    );
                }
                if let (Dim::Concrete(d), Dim::Concrete(da), Dim::Concrete(db)) =
                    (x.shape()[2], a.shape()[2], b.shape()[2])
                {
                    match da.checked_add(db) {
                        Some(sum) if sum == d => {}
                        Some(sum) => problems.push(IrError::OpShapeMismatch {
                            op: "split",
                            tensor: "a/b",
                            detail: format!(
                                "output widths {da} + {db} = {sum} do not reconstruct input dim {d}"
                            ),
                        }),
                        None => problems.push(IrError::OpShapeMismatch {
                            op: "split",
                            tensor: "a/b",
                            detail: format!("output widths {da} + {db} overflow u32"),
                        }),
                    }
                    if da != self.first {
                        problems.push(IrError::OpShapeMismatch {
                            op: "split",
                            tensor: "a",
                            detail: format!(
                                "first output width {da} does not match split attr first={}",
                                self.first
                            ),
                        });
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (card A1.14, SI-29; Spec 1 §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::SPLIT_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::SPLIT_RULES.iter().map(|r| r.as_tuple()).collect()
    }

    /// Returns op numerics contract: pure data movement, no arithmetic.
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "split"
    }
}

/// Last-axis channel concatenation op (card A1.14, SI-29).
///
/// `(a [T, H, Da], b [T, H, Db]) -> y [T, H, Da + Db]`
///
/// Reconstructs the MLA per-head query from its explicit non-rotary and
/// rotated-rotary parts (Spec 1 §4.B, Spec 8 §3.1). Pure data movement:
/// values are copied unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConcatOp;

impl ConcatOp {
    /// Validates inputs and outputs against the concat contract.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "concat",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "concat",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            for (name, t) in [("a", &inputs[0]), ("b", &inputs[1])] {
                check_rank("concat", name, t, 3, &mut problems);
                check_dtype_in(
                    "concat",
                    name,
                    t,
                    &[DType::F16, DType::Bf16, DType::F32],
                    &mut problems,
                );
                if t.class() != Class::Activation {
                    problems.push(IrError::OpClassMismatch {
                        op: "concat",
                        tensor: name,
                        expected: Class::Activation,
                        got: t.class(),
                    });
                }
            }
            let (a, b) = (&inputs[0], &inputs[1]);
            if b.dtype() != a.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "concat",
                    tensor: "b",
                    expected: vec![a.dtype()].into_boxed_slice(),
                    got: b.dtype(),
                });
            }
            if a.rank() == 3 && b.rank() == 3 {
                for (axis, axis_name) in [(0, "T"), (1, "H")] {
                    check_dim_match(
                        "concat",
                        "b",
                        b.shape()[axis],
                        "a",
                        a.shape()[axis],
                        axis_name,
                        &mut problems,
                    );
                }
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("concat", "y", y, 3, &mut problems);
            check_dtype_in(
                "concat",
                "y",
                y,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "concat",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let (a, b, y) = (&inputs[0], &inputs[1], &outputs[0]);
            if y.dtype() != a.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "concat",
                    tensor: "y",
                    expected: vec![a.dtype()].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if a.rank() == 3 && b.rank() == 3 && y.rank() == 3 {
                for (axis, axis_name) in [(0, "T"), (1, "H")] {
                    check_dim_match(
                        "concat",
                        "y",
                        y.shape()[axis],
                        "a",
                        a.shape()[axis],
                        axis_name,
                        &mut problems,
                    );
                }
                if let (Dim::Concrete(da), Dim::Concrete(db), Dim::Concrete(dy)) =
                    (a.shape()[2], b.shape()[2], y.shape()[2])
                {
                    match da.checked_add(db) {
                        Some(sum) if sum == dy => {}
                        Some(sum) => problems.push(IrError::OpShapeMismatch {
                            op: "concat",
                            tensor: "y",
                            detail: format!(
                                "input widths {da} + {db} = {sum} do not match output dim {dy}"
                            ),
                        }),
                        None => problems.push(IrError::OpShapeMismatch {
                            op: "concat",
                            tensor: "y",
                            detail: format!("input widths {da} + {db} overflow u32"),
                        }),
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (card A1.14, SI-29; Spec 1 §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::CONCAT_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::CONCAT_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract: pure data movement, no arithmetic.
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "concat"
    }
}

// -----------------------------------------------------------------------------
// §4.B Normalization and elementwise
// -----------------------------------------------------------------------------

/// Root Mean Square or Layer normalization op (Spec 1 §4.B).
///
/// `x [T, N] act_dtype, weight [N] f32, bias? [N] f32 -> y [T, N] out_dtype`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormOp {
    /// RMS or Layer norm variant.
    pub kind: NormKind,
    /// Epsilon variance floor.
    pub eps: f32,
    /// Reduction axis (last dim or per-head).
    pub axis: NormAxis,
    /// Weight offset (e.g. 1.0 for Gemma's 1+w parameterization).
    pub weight_offset: f32,
    /// Output activation dtype.
    pub out_dtype: DType,
}

impl NormOp {
    /// Validates inputs and outputs against Spec 1 §4.B constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !self.eps.is_finite() || self.eps <= 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "norm",
                attribute: "eps",
                reason: format!("must be finite and > 0, got {}", self.eps),
            });
        }
        if !matches!(self.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "norm",
                attribute: "out_dtype",
                reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
            });
        }
        if !self.weight_offset.is_finite() {
            problems.push(IrError::OpAttributeInvalid {
                op: "norm",
                attribute: "weight_offset",
                reason: format!("must be finite, got {}", self.weight_offset),
            });
        }
        if let NormAxis::Head(d) = self.axis {
            if d == 0 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "norm",
                    attribute: "axis",
                    reason: "head dimension must be > 0".to_string(),
                });
            }
        }

        let input_count_valid = inputs.len() == 2 || inputs.len() == 3;
        if !input_count_valid {
            problems.push(IrError::OpInputCountCandidatesMismatch {
                op: "norm",
                expected: vec![2, 3].into_boxed_slice(),
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "norm",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            let weight = &inputs[1];

            check_rank("norm", "x", x, 2, &mut problems);
            check_dtype_in(
                "norm",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "norm",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }

            check_rank("norm", "weight", weight, 1, &mut problems);
            check_dtype_in("norm", "weight", weight, &[DType::F32], &mut problems);
            if !matches!(weight.class(), Class::Weight | Class::Param) {
                problems.push(IrError::OpClassMismatch {
                    op: "norm",
                    tensor: "weight",
                    expected: Class::Weight,
                    got: weight.class(),
                });
            }

            if inputs.len() == 3 {
                let bias = &inputs[2];
                check_rank("norm", "bias", bias, 1, &mut problems);
                check_dtype_in("norm", "bias", bias, &[DType::F32], &mut problems);
                if !matches!(bias.class(), Class::Weight | Class::Param) {
                    problems.push(IrError::OpClassMismatch {
                        op: "norm",
                        tensor: "bias",
                        expected: Class::Weight,
                        got: bias.class(),
                    });
                }
            }

            // Independent input-to-input checks:
            if x.rank() == 2 && weight.rank() == 1 {
                let last_dim = x.shape()[1];
                check_dim_match(
                    "norm",
                    "weight",
                    weight.shape()[0],
                    "x",
                    last_dim,
                    "N",
                    &mut problems,
                );
                if let (NormAxis::Head(d), Dim::Concrete(n)) = (self.axis, last_dim) {
                    if d > 0 && n % d != 0 {
                        problems.push(IrError::OpShapeMismatch {
                            op: "norm",
                            tensor: "x",
                            detail: format!("feature dim N={n} not divisible by head dim {d}"),
                        });
                    }
                }
            }

            if inputs.len() == 3 {
                let bias = &inputs[2];
                if weight.rank() == 1 && bias.rank() == 1 {
                    check_dim_match(
                        "norm",
                        "bias",
                        bias.shape()[0],
                        "weight",
                        weight.shape()[0],
                        "N",
                        &mut problems,
                    );
                }
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("norm", "y", y, 2, &mut problems);
            if y.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "norm",
                    tensor: "y",
                    expected: vec![self.out_dtype].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "norm",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let y = &outputs[0];

            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "norm",
                    "y",
                    y.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "norm",
                    "y",
                    y.shape()[1],
                    "x",
                    x.shape()[1],
                    "N",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.B, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::NORM_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::NORM_RULES.iter().map(|r| r.as_tuple()).collect()
    }

    /// Returns op numerics contract (Spec 1 §4.B, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingAxis)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "norm"
    }
}

/// Elementwise residual addition op (Spec 1 §4.B).
///
/// `a + scale * b in f32 -> y`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualAddOp {
    /// Output activation dtype.
    pub out_dtype: DType,
    /// Residual branch scale from `LayerSpec.residual_scale` (Spec 8 §3;
    /// card A1.14, SI-27). `1.0` reproduces the A1.3 `a + b` form exactly.
    pub scale: f32,
}

impl ResidualAddOp {
    /// Validates inputs and outputs against Spec 1 §4.B constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !matches!(self.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "residual_add",
                attribute: "out_dtype",
                reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
            });
        }

        if !self.scale.is_finite() || self.scale == 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "residual_add",
                attribute: "scale",
                reason: format!("must be finite and non-zero, got {}", self.scale),
            });
        }

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "residual_add",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "residual_add",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let a = &inputs[0];
            let b = &inputs[1];

            check_dtype_in(
                "residual_add",
                "a",
                a,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if a.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "residual_add",
                    tensor: "a",
                    expected: Class::Activation,
                    got: a.class(),
                });
            }
            check_dtype_in(
                "residual_add",
                "b",
                b,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if b.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "residual_add",
                    tensor: "b",
                    expected: Class::Activation,
                    got: b.class(),
                });
            }

            // Independent input-to-input checks:
            if a.rank() != b.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "residual_add",
                    tensor: "b",
                    expected: a.rank(),
                    got: b.rank(),
                });
            } else {
                for i in 0..a.rank() {
                    check_dim_match(
                        "residual_add",
                        "b",
                        b.shape()[i],
                        "a",
                        a.shape()[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            if y.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "residual_add",
                    tensor: "y",
                    expected: vec![self.out_dtype].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "residual_add",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let a = &inputs[0];
            let y = &outputs[0];

            if y.rank() != a.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "residual_add",
                    tensor: "y",
                    expected: a.rank(),
                    got: y.rank(),
                });
            } else {
                for i in 0..a.rank() {
                    check_dim_match(
                        "residual_add",
                        "y",
                        y.shape()[i],
                        "a",
                        a.shape()[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.B, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::RESIDUAL_ADD_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::RESIDUAL_ADD_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.B, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::None)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "residual_add"
    }
}

/// Gated activation product op (Spec 1 §4.B).
///
/// `gate [T, Dff], up [T, Dff] -> y [T, Dff]`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActMulOp {
    /// Activation function applied to the gate tensor.
    pub act: ActivationKind,
    /// Optional upper clamp limit.
    pub clamp: Option<f32>,
}

impl ActMulOp {
    /// Validates inputs and outputs against Spec 1 §4.B constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if let Some(c) = self.clamp {
            if !c.is_finite() || c <= 0.0 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "act_mul",
                    attribute: "clamp",
                    reason: format!("must be finite and > 0, got {c}"),
                });
            }
        }

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "act_mul",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "act_mul",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let gate = &inputs[0];
            let up = &inputs[1];

            check_rank("act_mul", "gate", gate, 2, &mut problems);
            check_rank("act_mul", "up", up, 2, &mut problems);
            check_dtype_in(
                "act_mul",
                "gate",
                gate,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            check_dtype_in(
                "act_mul",
                "up",
                up,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if up.dtype() != gate.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "act_mul",
                    tensor: "up",
                    expected: vec![gate.dtype()].into_boxed_slice(),
                    got: up.dtype(),
                });
            }
            if gate.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "act_mul",
                    tensor: "gate",
                    expected: Class::Activation,
                    got: gate.class(),
                });
            }
            if up.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "act_mul",
                    tensor: "up",
                    expected: Class::Activation,
                    got: up.class(),
                });
            }

            // Independent input-to-input checks:
            if gate.rank() == 2 && up.rank() == 2 {
                check_dim_match(
                    "act_mul",
                    "up",
                    up.shape()[0],
                    "gate",
                    gate.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "act_mul",
                    "up",
                    up.shape()[1],
                    "gate",
                    gate.shape()[1],
                    "Dff",
                    &mut problems,
                );
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("act_mul", "y", y, 2, &mut problems);
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "act_mul",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let gate = &inputs[0];
            let y = &outputs[0];

            if y.dtype() != gate.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "act_mul",
                    tensor: "y",
                    expected: vec![gate.dtype()].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if gate.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "act_mul",
                    "y",
                    y.shape()[0],
                    "gate",
                    gate.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "act_mul",
                    "y",
                    y.shape()[1],
                    "gate",
                    gate.shape()[1],
                    "Dff",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.B, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::ACT_MUL_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::ACT_MUL_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.B, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::None)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "act_mul"
    }
}

/// Standalone non-gated activation function op (Spec 1 §4.B).
///
/// `x [T, Dff] -> y [T, Dff]`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationOp {
    /// Activation function kind.
    pub act: ActivationKind,
    /// Optional upper clamp limit.
    pub clamp: Option<f32>,
}

impl ActivationOp {
    /// Validates inputs and outputs against Spec 1 §4.B constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if let Some(c) = self.clamp {
            if !c.is_finite() || c <= 0.0 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "activation",
                    attribute: "clamp",
                    reason: format!("must be finite and > 0, got {c}"),
                });
            }
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "activation",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "activation",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            check_rank("activation", "x", x, 2, &mut problems);
            check_dtype_in(
                "activation",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "activation",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("activation", "y", y, 2, &mut problems);
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "activation",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let y = &outputs[0];

            if y.dtype() != x.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "activation",
                    tensor: "y",
                    expected: vec![x.dtype()].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "activation",
                    "y",
                    y.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "activation",
                    "y",
                    y.shape()[1],
                    "x",
                    x.shape()[1],
                    "Dff",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.B, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::ACTIVATION_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::ACTIVATION_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.B, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::None)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "activation"
    }
}

/// Final-logit soft-capping op (card A1.14, SI-28).
///
/// `x [T, V] f32 -> y [T, V] f32` with `y = cap * tanh(x / cap)` computed in
/// f32, applied once to the `lm_head` output when
/// `ModelSpec.final_logit_softcap` is set (Spec 8 §3). `None` lowers to no op,
/// which reproduces the A1.3 graph exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitSoftcapOp {
    /// Soft-cap threshold; must be finite and positive.
    pub cap: f32,
}

impl LogitSoftcapOp {
    /// Validates inputs and outputs against the softcap contract.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !self.cap.is_finite() || self.cap <= 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "logit_softcap",
                attribute: "cap",
                reason: format!("must be finite and > 0, got {}", self.cap),
            });
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "logit_softcap",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "logit_softcap",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            check_rank("logit_softcap", "x", x, 2, &mut problems);
            check_dtype_in("logit_softcap", "x", x, &[DType::F32], &mut problems);
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "logit_softcap",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("logit_softcap", "y", y, 2, &mut problems);
            check_dtype_in("logit_softcap", "y", y, &[DType::F32], &mut problems);
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "logit_softcap",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let (x, y) = (&inputs[0], &outputs[0]);
            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "logit_softcap",
                    "y",
                    y.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "logit_softcap",
                    "y",
                    y.shape()[1],
                    "x",
                    x.shape()[1],
                    "V",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (card A1.14, SI-28; Spec 1 §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::LOGIT_SOFTCAP_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::LOGIT_SOFTCAP_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §6.1, §6.4: f32 elementwise).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::None)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "logit_softcap"
    }
}

/// Rotary Position Embedding application op (Spec 1 §4.B).
///
/// `x [T, H, D], positions [T] | [T, 3] -> x' [T, H, D]`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeOp {
    /// Dimension to which rotary embedding is applied.
    pub rot_dim: u32,
    /// Base theta frequency.
    pub theta: f32,
    /// Interleaved or NeoX style.
    pub style: RopeStyle,
    /// Frequency scaling configuration.
    pub scaling: RopeScaling,
    /// Multimodal RoPE section dimensions [T, H, W], if applicable.
    pub mrope_sections: Option<[u32; 3]>,
    /// Destination activation dtype.
    pub out_dtype: DType,
}

impl RopeOp {
    /// Validates inputs and outputs against Spec 1 §4.B constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if self.rot_dim == 0 || !self.rot_dim.is_multiple_of(2) {
            problems.push(IrError::OpAttributeInvalid {
                op: "rope",
                attribute: "rot_dim",
                reason: format!("rot_dim must be positive and even, got {}", self.rot_dim),
            });
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "rope",
                attribute: "theta",
                reason: format!("theta must be finite and > 0, got {}", self.theta),
            });
        }
        if let Some(sections) = self.mrope_sections {
            if sections.contains(&0) {
                problems.push(IrError::OpAttributeInvalid {
                    op: "rope",
                    attribute: "mrope_sections",
                    reason: "mrope sections must be positive".to_string(),
                });
            }
            if sections.iter().any(|&s| s % 2 != 0) {
                problems.push(IrError::OpAttributeInvalid {
                    op: "rope",
                    attribute: "mrope_sections",
                    reason: "mrope section dimensions must be even".to_string(),
                });
            }
            match sections
                .iter()
                .try_fold(0u32, |sum, section| sum.checked_add(*section))
            {
                Some(total) if total > self.rot_dim => {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "mrope_sections",
                        reason: format!(
                            "sum of mrope sections {total} exceeds rot_dim {}",
                            self.rot_dim
                        ),
                    });
                }
                None => problems.push(IrError::OpAttributeInvalid {
                    op: "rope",
                    attribute: "mrope_sections",
                    reason: "sum of mrope sections exceeds u32::MAX".to_string(),
                }),
                Some(_) => {}
            }
        }
        if !matches!(self.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "rope",
                attribute: "out_dtype",
                reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
            });
        }
        match self.scaling {
            RopeScaling::None | RopeScaling::Dynamic => {}
            RopeScaling::Linear(factor) => {
                if !factor.is_finite() || factor <= 0.0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "scaling.factor",
                        reason: format!("must be finite and > 0, got {factor}"),
                    });
                }
            }
            RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                orig_ctx,
                mscale,
            } => {
                if !factor.is_finite() || factor <= 0.0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "scaling.factor",
                        reason: format!("must be finite and > 0, got {factor}"),
                    });
                }
                if !beta_fast.is_finite() || beta_fast <= 0.0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "scaling.beta_fast",
                        reason: format!("must be finite and > 0, got {beta_fast}"),
                    });
                }
                if !beta_slow.is_finite() || beta_slow <= 0.0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "scaling.beta_slow",
                        reason: format!("must be finite and > 0, got {beta_slow}"),
                    });
                }
                if beta_slow > beta_fast {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "scaling.beta_slow",
                        reason: format!(
                            "beta_slow ({beta_slow}) cannot exceed beta_fast ({beta_fast})"
                        ),
                    });
                }
                if orig_ctx == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "scaling.orig_ctx",
                        reason: "must be > 0".to_string(),
                    });
                }
                if !mscale.is_finite() || mscale <= 0.0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "rope",
                        attribute: "scaling.mscale",
                        reason: format!("must be finite and > 0, got {mscale}"),
                    });
                }
            }
        }

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "rope",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "rope",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            let positions = &inputs[1];

            check_rank("rope", "x", x, 3, &mut problems);
            check_dtype_in(
                "rope",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "rope",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }

            if self.mrope_sections.is_some() {
                check_rank("rope", "positions", positions, 2, &mut problems);
                if positions.rank() == 2 {
                    if let Dim::Concrete(width) = positions.shape()[1] {
                        if width != 3 {
                            problems.push(IrError::OpShapeMismatch {
                                op: "rope",
                                tensor: "positions",
                                detail: format!("mrope position width must be 3, got {width}"),
                            });
                        }
                    }
                }
            } else {
                check_rank("rope", "positions", positions, 1, &mut problems);
            }
            check_dtype_in("rope", "positions", positions, &[DType::U32], &mut problems);
            if positions.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "rope",
                    tensor: "positions",
                    expected: Class::Activation,
                    got: positions.class(),
                });
            }

            // Independent input-to-input checks:
            if x.rank() == 3 && (positions.rank() == 1 || positions.rank() == 2) {
                check_dim_match(
                    "rope",
                    "positions",
                    positions.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
            }
            if x.rank() == 3 {
                if let Dim::Concrete(d) = x.shape()[2] {
                    if self.rot_dim > d {
                        problems.push(IrError::OpAttributeInvalid {
                            op: "rope",
                            attribute: "rot_dim",
                            reason: format!("rot_dim {} exceeds head dim {}", self.rot_dim, d),
                        });
                    }
                }
            }
        }

        if output_count_valid {
            let x_prime = &outputs[0];
            check_rank("rope", "x_prime", x_prime, 3, &mut problems);
            if x_prime.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "rope",
                    tensor: "x_prime",
                    expected: vec![self.out_dtype].into_boxed_slice(),
                    got: x_prime.dtype(),
                });
            }
            if x_prime.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "rope",
                    tensor: "x_prime",
                    expected: Class::Activation,
                    got: x_prime.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let x_prime = &outputs[0];

            if x.rank() == 3 && x_prime.rank() == 3 {
                for i in 0..3 {
                    check_dim_match(
                        "rope",
                        "x_prime",
                        x_prime.shape()[i],
                        "x",
                        x.shape()[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.B, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::ROPE_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::ROPE_RULES.iter().map(|r| r.as_tuple()).collect()
    }

    /// Returns op numerics contract (Spec 1 §4.B, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::None)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "rope"
    }
}

// -----------------------------------------------------------------------------
// §4.C Matmul family
// -----------------------------------------------------------------------------

/// Matrix multiplication with optional fused epilogue (Spec 1 §4.C).
///
/// `x [M, K], w [N, K], bias? [N] f32 -> y [M, N] out_dtype`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulOp {
    /// Destination activation dtype.
    pub out_dtype: DType,
    /// Fused post-GEMM operation.
    pub epilogue: Epilogue,
    /// Whether weight matrix `w` is stored transposed `[K, N]` instead of standard `[N, K]`.
    pub transpose_w: bool,
}

impl MatmulOp {
    /// Validates inputs and outputs against Spec 1 §4.C constraints.
    // DECISION(A1.2): MatmulOp validates inputs conditionally: None/Act require 2 inputs (x, w), Bias requires 3 (x, w, bias [N] f32), Residual requires 3 (x, w, residual [M, N]) per Spec 1 §4.C and SI-9; rejected fixed 2-input signature because fused residual partial sum accumulation requires the residual tensor in the step graph.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !matches!(self.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "matmul",
                attribute: "out_dtype",
                reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
            });
        }

        let expected_inputs = match self.epilogue {
            Epilogue::Bias | Epilogue::Residual => 3,
            Epilogue::None | Epilogue::Act(_) => 2,
        };

        let input_count_valid = inputs.len() == expected_inputs;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "matmul",
                expected: expected_inputs,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "matmul",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            let w = &inputs[1];

            check_rank("matmul", "x", x, 2, &mut problems);
            check_gemm_activation_operand("matmul", "x", x, &mut problems);
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "matmul",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }

            check_rank("matmul", "w", w, 2, &mut problems);
            check_gemm_weight_operand("matmul", "w", w, &mut problems);
            if w.class() != Class::Weight {
                problems.push(IrError::OpClassMismatch {
                    op: "matmul",
                    tensor: "w",
                    expected: Class::Weight,
                    got: w.class(),
                });
            }
            if !matches!(w.placement(), Placement::Device { .. }) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "matmul",
                    tensor: "w",
                    placement: w.placement(),
                });
            }

            match self.epilogue {
                Epilogue::Bias => {
                    let bias = &inputs[2];
                    check_rank("matmul", "bias", bias, 1, &mut problems);
                    check_dtype_in("matmul", "bias", bias, &[DType::F32], &mut problems);
                    if bias.class() != Class::Param {
                        problems.push(IrError::OpClassMismatch {
                            op: "matmul",
                            tensor: "bias",
                            expected: Class::Param,
                            got: bias.class(),
                        });
                    }
                }
                Epilogue::Residual => {
                    let residual = &inputs[2];
                    check_rank("matmul", "residual", residual, 2, &mut problems);
                    check_dtype_in(
                        "matmul",
                        "residual",
                        residual,
                        &[DType::F16, DType::Bf16, DType::F32],
                        &mut problems,
                    );
                    if residual.class() != Class::Activation {
                        problems.push(IrError::OpClassMismatch {
                            op: "matmul",
                            tensor: "residual",
                            expected: Class::Activation,
                            got: residual.class(),
                        });
                    }
                }
                Epilogue::None | Epilogue::Act(_) => {}
            }

            // Independent input-to-input checks:
            if x.rank() == 2 && w.rank() == 2 {
                let (k_w, n_w) = if self.transpose_w {
                    (w.shape()[0], w.shape()[1])
                } else {
                    (w.shape()[1], w.shape()[0])
                };
                check_dim_match("matmul", "w", k_w, "x", x.shape()[1], "K", &mut problems);

                match self.epilogue {
                    Epilogue::Bias => {
                        let bias = &inputs[2];
                        if bias.rank() == 1 {
                            check_dim_match(
                                "matmul",
                                "bias",
                                bias.shape()[0],
                                "w",
                                n_w,
                                "N",
                                &mut problems,
                            );
                        }
                    }
                    Epilogue::Residual => {
                        let residual = &inputs[2];
                        if residual.rank() == 2 {
                            check_dim_match(
                                "matmul",
                                "residual",
                                residual.shape()[0],
                                "x",
                                x.shape()[0],
                                "M",
                                &mut problems,
                            );
                            check_dim_match(
                                "matmul",
                                "residual",
                                residual.shape()[1],
                                "w",
                                n_w,
                                "N",
                                &mut problems,
                            );
                        }
                    }
                    Epilogue::None | Epilogue::Act(_) => {}
                }
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("matmul", "y", y, 2, &mut problems);
            if y.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "matmul",
                    tensor: "y",
                    expected: vec![self.out_dtype].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "matmul",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let w = &inputs[1];
            let y = &outputs[0];

            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "matmul",
                    "y",
                    y.shape()[0],
                    "x",
                    x.shape()[0],
                    "M",
                    &mut problems,
                );
            }
            if w.rank() == 2 && y.rank() == 2 {
                let n_w = if self.transpose_w {
                    w.shape()[1]
                } else {
                    w.shape()[0]
                };
                check_dim_match("matmul", "y", y.shape()[1], "w", n_w, "N", &mut problems);
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.C, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::MATMUL_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::MATMUL_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    // DECISION(A1.2): numerics_for inspects operand dtype and quant scheme per Spec 1 §6.1 & §6.2 (AscendingBlock for PerBlock32, AscendingK for PerToken / i8 / float); rejected static-only numerics descriptor on MatmulOp because execution contract depends on input activation quantization.
    /// Returns dynamic input-dependent numerics contract for given activation and weight tensors.
    pub fn numerics_for(&self, x: &crate::Tensor, w: &crate::Tensor) -> Result<Numerics, IrError> {
        matmul_numerics(x.dtype(), w.dtype(), x.quant(), w.quant())
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "matmul"
    }
}

/// Mixture of Experts routing and top-k gating op (Spec 1 §4.C).
///
/// `logits [T, E] f32 -> expert_ids [T, K] u32, weights [T, K] f32`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeRouteOp {
    /// Number of experts chosen per token `K`.
    pub top_k: u32,
    /// Router scoring method.
    pub scoring: MoeScoring,
    /// Whether router weights are renormalized to sum to 1.
    pub renormalize: bool,
    /// Grouped expert routing constraints, if applicable.
    pub group: Option<MoeGroup>,
    /// Router scale factor.
    pub scale: f32,
}

impl MoeRouteOp {
    /// Validates inputs and outputs against Spec 1 §4.C constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if self.top_k == 0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "moe_route",
                attribute: "top_k",
                reason: "top_k must be > 0".to_string(),
            });
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "moe_route",
                attribute: "scale",
                reason: format!("scale must be finite and > 0, got {}", self.scale),
            });
        }
        if let Some(g) = self.group {
            if g.n_group == 0 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "moe_route",
                    attribute: "group.n_group",
                    reason: "n_group must be > 0".to_string(),
                });
            }
            if g.topk_group == 0 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "moe_route",
                    attribute: "group.topk_group",
                    reason: "topk_group must be > 0".to_string(),
                });
            }
            if g.topk_group > self.top_k {
                problems.push(IrError::OpAttributeInvalid {
                    op: "moe_route",
                    attribute: "group.topk_group",
                    reason: format!(
                        "topk_group {} cannot exceed top_k {}",
                        g.topk_group, self.top_k
                    ),
                });
            }
        }

        let input_count_valid = inputs.len() == 1 || inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountCandidatesMismatch {
                op: "moe_route",
                expected: vec![1, 2].into_boxed_slice(),
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 2;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "moe_route",
                expected: 2,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let logits = &inputs[0];
            check_rank("moe_route", "logits", logits, 2, &mut problems);
            check_dtype_in("moe_route", "logits", logits, &[DType::F32], &mut problems);
            if logits.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_route",
                    tensor: "logits",
                    expected: Class::Activation,
                    got: logits.class(),
                });
            }

            if inputs.len() == 2 {
                let bias = &inputs[1];
                check_rank("moe_route", "bias", bias, 1, &mut problems);
                check_dtype_in("moe_route", "bias", bias, &[DType::F32], &mut problems);
                if bias.class() != Class::Param {
                    problems.push(IrError::OpClassMismatch {
                        op: "moe_route",
                        tensor: "bias",
                        expected: Class::Param,
                        got: bias.class(),
                    });
                }
            }

            // Independent input-to-input checks:
            if logits.rank() == 2 {
                if let Dim::Concrete(e) = logits.shape()[1] {
                    if self.top_k > e {
                        problems.push(IrError::OpAttributeInvalid {
                            op: "moe_route",
                            attribute: "top_k",
                            reason: format!(
                                "top_k {} cannot exceed number of experts E={e}",
                                self.top_k
                            ),
                        });
                    }
                    if inputs.len() == 2 && inputs[1].rank() == 1 {
                        check_dim_match(
                            "moe_route",
                            "bias",
                            inputs[1].shape()[0],
                            "logits",
                            logits.shape()[1],
                            "E",
                            &mut problems,
                        );
                    }
                    if let Some(g) = self.group {
                        if g.n_group > 0 {
                            if e % g.n_group != 0 {
                                problems.push(IrError::OpShapeMismatch {
                                    op: "moe_route",
                                    tensor: "logits",
                                    detail: format!(
                                        "experts E={e} must be divisible by n_group={}",
                                        g.n_group
                                    ),
                                });
                            } else if g.topk_group > e / g.n_group {
                                problems.push(IrError::OpAttributeInvalid {
                                    op: "moe_route",
                                    attribute: "topk_group",
                                    reason: format!(
                                        "topk_group {} exceeds experts per group {}",
                                        g.topk_group,
                                        e / g.n_group
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }

        if output_count_valid {
            let expert_ids = &outputs[0];
            let weights = &outputs[1];

            check_rank("moe_route", "expert_ids", expert_ids, 2, &mut problems);
            check_dtype_in(
                "moe_route",
                "expert_ids",
                expert_ids,
                &[DType::U32],
                &mut problems,
            );
            if expert_ids.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_route",
                    tensor: "expert_ids",
                    expected: Class::Activation,
                    got: expert_ids.class(),
                });
            }

            check_rank("moe_route", "weights", weights, 2, &mut problems);
            check_dtype_in(
                "moe_route",
                "weights",
                weights,
                &[DType::F32],
                &mut problems,
            );
            if weights.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_route",
                    tensor: "weights",
                    expected: Class::Activation,
                    got: weights.class(),
                });
            }

            if expert_ids.rank() == 2 && weights.rank() == 2 {
                check_dim_match(
                    "moe_route",
                    "weights",
                    weights.shape()[0],
                    "expert_ids",
                    expert_ids.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "moe_route",
                    "weights",
                    weights.shape()[1],
                    "expert_ids",
                    expert_ids.shape()[1],
                    "K",
                    &mut problems,
                );
            }

            if let Dim::Concrete(k) = expert_ids.shape()[1] {
                if k != self.top_k {
                    problems.push(IrError::OpShapeMismatch {
                        op: "moe_route",
                        tensor: "expert_ids",
                        detail: format!("axis 1 extent {k} != top_k {}", self.top_k),
                    });
                }
            }
        }

        if input_count_valid && output_count_valid {
            let logits = &inputs[0];
            let expert_ids = &outputs[0];

            if logits.rank() == 2 && expert_ids.rank() == 2 {
                check_dim_match(
                    "moe_route",
                    "expert_ids",
                    expert_ids.shape()[0],
                    "logits",
                    logits.shape()[0],
                    "T",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.C, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::MOE_ROUTE_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::MOE_ROUTE_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.C, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "moe_route"
    }
}

/// Mixture of Experts feed-forward execution op (Spec 1 §4.C).
///
/// `x [T, Dm], expert_ids [T, K], weights [T, K], w_gate_up [E, 2*Dff, Dm], w_down [E, Dm, Dff] -> y [T, Dm]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeFfnOp {
    /// Expert MLP activation function.
    pub act: ActivationKind,
    /// Destination activation dtype.
    pub out_dtype: DType,
    /// Number of shared experts executed concurrently.
    pub shared_experts: u32,
}

impl MoeFfnOp {
    /// Validates inputs and outputs against Spec 1 §4.C constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !matches!(self.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(IrError::OpAttributeInvalid {
                op: "moe_ffn",
                attribute: "out_dtype",
                reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
            });
        }

        let input_count_valid = inputs.len() == 5;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "moe_ffn",
                expected: 5,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "moe_ffn",
                expected: 1,
                got: outputs.len(),
            });
        }

        if input_count_valid {
            let x = &inputs[0];
            let expert_ids = &inputs[1];
            let weights = &inputs[2];
            let w_gate_up = &inputs[3];
            let w_down = &inputs[4];

            check_rank("moe_ffn", "x", x, 2, &mut problems);
            check_gemm_activation_operand("moe_ffn", "x", x, &mut problems);
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_ffn",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }

            check_rank("moe_ffn", "expert_ids", expert_ids, 2, &mut problems);
            check_dtype_in(
                "moe_ffn",
                "expert_ids",
                expert_ids,
                &[DType::U32],
                &mut problems,
            );
            if expert_ids.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_ffn",
                    tensor: "expert_ids",
                    expected: Class::Activation,
                    got: expert_ids.class(),
                });
            }

            check_rank("moe_ffn", "weights", weights, 2, &mut problems);
            check_dtype_in("moe_ffn", "weights", weights, &[DType::F32], &mut problems);
            if weights.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_ffn",
                    tensor: "weights",
                    expected: Class::Activation,
                    got: weights.class(),
                });
            }

            check_rank("moe_ffn", "w_gate_up", w_gate_up, 3, &mut problems);
            check_gemm_weight_operand("moe_ffn", "w_gate_up", w_gate_up, &mut problems);
            if w_gate_up.class() != Class::Weight {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_ffn",
                    tensor: "w_gate_up",
                    expected: Class::Weight,
                    got: w_gate_up.class(),
                });
            }
            if !matches!(
                w_gate_up.placement(),
                Placement::Device { .. } | Placement::Tiered
            ) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "moe_ffn",
                    tensor: "w_gate_up",
                    placement: w_gate_up.placement(),
                });
            }

            check_rank("moe_ffn", "w_down", w_down, 3, &mut problems);
            check_gemm_weight_operand("moe_ffn", "w_down", w_down, &mut problems);
            if w_down.class() != Class::Weight {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_ffn",
                    tensor: "w_down",
                    expected: Class::Weight,
                    got: w_down.class(),
                });
            }
            if !matches!(
                w_down.placement(),
                Placement::Device { .. } | Placement::Tiered
            ) {
                problems.push(IrError::OpPlacementMismatch {
                    op: "moe_ffn",
                    tensor: "w_down",
                    placement: w_down.placement(),
                });
            }

            // Independent input-to-input checks:
            if expert_ids.rank() == 2 && x.rank() == 2 {
                check_dim_match(
                    "moe_ffn",
                    "expert_ids",
                    expert_ids.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
            }
            if weights.rank() == 2 && x.rank() == 2 {
                check_dim_match(
                    "moe_ffn",
                    "weights",
                    weights.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
            }
            if expert_ids.rank() == 2 && weights.rank() == 2 {
                check_dim_match(
                    "moe_ffn",
                    "expert_ids",
                    expert_ids.shape()[1],
                    "weights",
                    weights.shape()[1],
                    "K",
                    &mut problems,
                );
            }
            if w_gate_up.rank() == 3 && x.rank() == 2 {
                check_dim_match(
                    "moe_ffn",
                    "w_gate_up",
                    w_gate_up.shape()[2],
                    "x",
                    x.shape()[1],
                    "Dm",
                    &mut problems,
                );
            }
            if w_down.rank() == 3 && x.rank() == 2 {
                check_dim_match(
                    "moe_ffn",
                    "w_down",
                    w_down.shape()[1],
                    "x",
                    x.shape()[1],
                    "Dm",
                    &mut problems,
                );
            }
            if w_gate_up.rank() == 3 && w_down.rank() == 3 {
                check_dim_match(
                    "moe_ffn",
                    "w_gate_up",
                    w_gate_up.shape()[0],
                    "w_down",
                    w_down.shape()[0],
                    "E",
                    &mut problems,
                );
                if let (Dim::Concrete(gu), Dim::Concrete(dff)) =
                    (w_gate_up.shape()[1], w_down.shape()[2])
                {
                    let expected_gate_up = dff.checked_mul(2);
                    if expected_gate_up != Some(gu) {
                        let expected = expected_gate_up.map_or_else(
                            || "overflow (> u32::MAX)".to_string(),
                            |value| value.to_string(),
                        );
                        problems.push(IrError::OpShapeMismatch {
                            op: "moe_ffn",
                            tensor: "w_gate_up",
                            detail: format!(
                                "w_gate_up dim 1 extent {gu} != 2 * w_down dim 2 ({expected})"
                            ),
                        });
                    }
                }
            }
        }

        if output_count_valid {
            let y = &outputs[0];
            check_rank("moe_ffn", "y", y, 2, &mut problems);
            if y.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "moe_ffn",
                    tensor: "y",
                    expected: vec![self.out_dtype].into_boxed_slice(),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "moe_ffn",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if input_count_valid && output_count_valid {
            let x = &inputs[0];
            let y = &outputs[0];

            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "moe_ffn",
                    "y",
                    y.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "moe_ffn",
                    "y",
                    y.shape()[1],
                    "x",
                    x.shape()[1],
                    "Dm",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.C, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::MOE_FFN_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::MOE_FFN_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns dynamic input-dependent numerics contract for given activation and weight tensors.
    pub fn numerics_for(&self, x: &crate::Tensor, w: &crate::Tensor) -> Result<Numerics, IrError> {
        moe_ffn_gemm_numerics(x.dtype(), w.dtype(), x.quant(), w.quant())
    }

    /// Returns op numerics contract (Spec 1 §4.C, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "moe_ffn"
    }
}

// -----------------------------------------------------------------------------
// §4.D Attention
// -----------------------------------------------------------------------------

/// Attention key/value state cache write op (Spec 1 §4.D).
///
/// Tensor operands are `k [T, Hkv, D], v [T, Hkv, Dv] -> ()`.
/// `slot_map` arrives through the structured `BatchMeta` input (SI-12).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateWriteKvOp {
    /// Cache storage element dtype (`f16`, `i8`, or `e4m3`).
    pub cache_dtype: DType,
    /// Cache scale quantization granularity.
    pub scale_granularity: CacheScaleGranularity,
    /// MLA compressed latent metadata, if applicable.
    pub latent: Option<MlaLatent>,
    /// State handle identifying the KV cache buffer.
    pub handle: StateHandle,
}

impl StateWriteKvOp {
    /// Validates inputs and outputs against Spec 1 §4.D constraints.
    // DECISION(A1.2): StateWriteKvOp validates exactly the tensor operands (k, v); slot_map is carried by the required structured BatchMeta graph input. Rejected a third fake slot_map Tensor because SI-12 keeps BatchMeta outside tensor slices. Spec 1 §4.D, SI-12.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        match self.cache_dtype {
            DType::F16 | DType::I8 | DType::E4m3 => {}
            DType::F32
            | DType::Bf16
            | DType::E5m2
            | DType::I4
            | DType::I32
            | DType::U32
            | DType::Bool => {
                problems.push(IrError::OpAttributeInvalid {
                    op: "state_write_kv",
                    attribute: "cache_dtype",
                    reason: format!("must be f16, i8, or e4m3; got {:?}", self.cache_dtype),
                });
            }
        }

        match self.scale_granularity {
            CacheScaleGranularity::PerTokenHead => {}
            CacheScaleGranularity::PerBlock => {}
        }

        match self.latent {
            Some(ref l) => {
                if l.kv_lora_rank == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "state_write_kv",
                        attribute: "latent.kv_lora_rank",
                        reason: "kv_lora_rank must be > 0".to_string(),
                    });
                }
                if l.rope_dim == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "state_write_kv",
                        attribute: "latent.rope_dim",
                        reason: "rope_dim must be > 0".to_string(),
                    });
                }
                match self.handle.kind() {
                    StateKind::KvLatent => {}
                    StateKind::KvPaged | StateKind::ConvWindow | StateKind::Recurrent => {
                        problems.push(IrError::StateHandleKindMismatch {
                            op: "state_write_kv",
                            expected: StateKind::KvLatent,
                            got: self.handle.kind(),
                        });
                    }
                }
            }
            None => match self.handle.kind() {
                StateKind::KvPaged => {}
                StateKind::KvLatent | StateKind::ConvWindow | StateKind::Recurrent => {
                    problems.push(IrError::StateHandleKindMismatch {
                        op: "state_write_kv",
                        expected: StateKind::KvPaged,
                        got: self.handle.kind(),
                    });
                }
            },
        }

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "state_write_kv",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.is_empty();
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "state_write_kv",
                expected: 0,
                got: outputs.len(),
            });
        }

        if let Some(k) = inputs.first() {
            check_rank("state_write_kv", "k", k, 3, &mut problems);
            check_dtype_in(
                "state_write_kv",
                "k",
                k,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if k.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "state_write_kv",
                    tensor: "k",
                    expected: Class::Activation,
                    got: k.class(),
                });
            }
        }

        if let Some(v) = inputs.get(1) {
            check_rank("state_write_kv", "v", v, 3, &mut problems);
            check_dtype_in(
                "state_write_kv",
                "v",
                v,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if v.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "state_write_kv",
                    tensor: "v",
                    expected: Class::Activation,
                    got: v.class(),
                });
            }
        }

        if let (Some(k), Some(v)) = (inputs.first(), inputs.get(1)) {
            if v.dtype() != k.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "state_write_kv",
                    tensor: "v",
                    expected: Box::new([k.dtype()]),
                    got: v.dtype(),
                });
            }
            if k.rank() == 3 && v.rank() == 3 {
                check_dim_match(
                    "state_write_kv",
                    "v",
                    v.shape()[0],
                    "k",
                    k.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "state_write_kv",
                    "v",
                    v.shape()[1],
                    "k",
                    k.shape()[1],
                    "Hkv",
                    &mut problems,
                );
            }
            // DECISION(A1.14): with `latent`, canonical exact-split form writes
            // operand 0 as compressed latent c_kv ([T, H, kv_lora_rank]) and
            // operand 1 as rotated k_rope ([T, H, rope_dim]), consistent with
            // Spec 1 §4.D compressed-latent-plus-rope wording, Spec 3 §3.2
            // physical regions, and preexisting A1.3 call order. The combined form
            // (operand 0 holding kv_lora_rank + rope_dim) remains accepted for
            // A1.2-era compatibility; the inverted order (rope first, latent second)
            // is rejected. Spec 1 §4.D, Spec 3 §3.2, SI-29.
            if let Some(ref l) = self.latent {
                if k.rank() == 3 && v.rank() == 3 {
                    if let (Dim::Concrete(d0), Dim::Concrete(d1)) = (k.shape()[2], v.shape()[2]) {
                        let combined = l.kv_lora_rank.checked_add(l.rope_dim);
                        let is_combined = combined == Some(d0);
                        let is_split = d0 == l.kv_lora_rank && d1 == l.rope_dim;
                        if !is_combined && !is_split {
                            problems.push(IrError::OpShapeMismatch {
                                op: "state_write_kv",
                                tensor: "k/v",
                                detail: format!(
                                    "MLA latent dim {d0} / rotary dim {d1} must be the combined latent (rank {} + rope {}) or the exact split pair (latent {} / rotary {})",
                                    l.kv_lora_rank,
                                    l.rope_dim,
                                    l.kv_lora_rank,
                                    l.rope_dim
                                ),
                            });
                        }
                    }
                } else if k.rank() == 3 {
                    if let Dim::Concrete(d) = k.shape()[2] {
                        let latent_dim = l.kv_lora_rank.checked_add(l.rope_dim);
                        if latent_dim != Some(d) && d != l.kv_lora_rank {
                            problems.push(IrError::OpShapeMismatch {
                                op: "state_write_kv",
                                tensor: "k",
                                detail: format!(
                                    "latent dim {d} does not match MLA latent specs (rank {} + rope {} or rank {})",
                                    l.kv_lora_rank, l.rope_dim, l.kv_lora_rank
                                ),
                            });
                        }
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.D, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::STATE_WRITE_KV_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::STATE_WRITE_KV_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.D, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "state_write_kv"
    }
}

/// Paged attention read and compute op (Spec 1 §4.D).
///
/// `q [T, H, D], StateHandle, BatchMeta -> o [T, H, D]`
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionOp {
    /// Softmax scale factor (typically 1 / sqrt(D)).
    pub softmax_scale: f32,
    /// Causal, sliding window, or tree verification mask.
    pub mask: AttentionMask,
    /// Attention sink token count.
    pub sinks: u32,
    /// Logit soft-capping threshold (e.g. 50.0 for Gemma2).
    pub logit_softcap: Option<f32>,
    /// MLA configuration, if applicable.
    pub mla: Option<MlaAttentionSpec>,
    /// Output activation dtype.
    pub out_dtype: DType,
    /// State handle for KV cache reading.
    pub handle: StateHandle,
}

impl AttentionOp {
    /// Validates inputs and outputs against Spec 1 §4.D constraints.
    // DECISION(A1.2): AttentionOp validates tensor operands q -> o; StateHandle and BatchMeta (with optional TreeMask) are structured external metadata per Spec 1 §4.D and SI-12.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "attention",
                attribute: "softmax_scale",
                reason: format!("must be finite and > 0, got {}", self.softmax_scale),
            });
        }
        match self.out_dtype {
            DType::F16 | DType::Bf16 | DType::F32 => {}
            DType::E4m3
            | DType::E5m2
            | DType::I4
            | DType::I8
            | DType::I32
            | DType::U32
            | DType::Bool => {
                problems.push(IrError::OpAttributeInvalid {
                    op: "attention",
                    attribute: "out_dtype",
                    reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
                });
            }
        }
        match self.mask {
            AttentionMask::Causal => {}
            AttentionMask::CausalWindow(w) => {
                if w == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "attention",
                        attribute: "mask",
                        reason: "causal window must be > 0".to_string(),
                    });
                }
            }
            AttentionMask::Tree => {}
        }
        if let Some(c) = self.logit_softcap.filter(|c| !c.is_finite() || *c <= 0.0) {
            problems.push(IrError::OpAttributeInvalid {
                op: "attention",
                attribute: "logit_softcap",
                reason: format!("must be finite and > 0, got {c}"),
            });
        }

        match self.mla {
            Some(ref m) => {
                if m.kv_lora_rank == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "attention",
                        attribute: "mla.kv_lora_rank",
                        reason: "kv_lora_rank must be > 0".to_string(),
                    });
                }
                if m.qk_nope_dim == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "attention",
                        attribute: "mla.qk_nope_dim",
                        reason: "qk_nope_dim must be > 0".to_string(),
                    });
                }
                if m.qk_rope_dim == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "attention",
                        attribute: "mla.qk_rope_dim",
                        reason: "qk_rope_dim must be > 0".to_string(),
                    });
                }
                if m.v_dim == 0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "attention",
                        attribute: "mla.v_dim",
                        reason: "v_dim must be > 0".to_string(),
                    });
                }
                if m.q_lora_rank == Some(0) {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "attention",
                        attribute: "mla.q_lora_rank",
                        reason: "q_lora_rank must be > 0".to_string(),
                    });
                }
                match self.handle.kind() {
                    StateKind::KvLatent => {}
                    StateKind::KvPaged | StateKind::ConvWindow | StateKind::Recurrent => {
                        problems.push(IrError::StateHandleKindMismatch {
                            op: "attention",
                            expected: StateKind::KvLatent,
                            got: self.handle.kind(),
                        });
                    }
                }
            }
            None => match self.handle.kind() {
                StateKind::KvPaged => {}
                StateKind::KvLatent | StateKind::ConvWindow | StateKind::Recurrent => {
                    problems.push(IrError::StateHandleKindMismatch {
                        op: "attention",
                        expected: StateKind::KvPaged,
                        got: self.handle.kind(),
                    });
                }
            },
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "attention",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "attention",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(q) = inputs.first() {
            check_rank("attention", "q", q, 3, &mut problems);
            check_dtype_in(
                "attention",
                "q",
                q,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if q.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "attention",
                    tensor: "q",
                    expected: Class::Activation,
                    got: q.class(),
                });
            }
            if let Some(ref m) = self.mla {
                if q.rank() == 3 {
                    if let Dim::Concrete(qd) = q.shape()[2] {
                        if m.qk_nope_dim.checked_add(m.qk_rope_dim) != Some(qd) {
                            problems.push(IrError::OpShapeMismatch {
                                op: "attention",
                                tensor: "q",
                                detail: format!(
                                    "q head dim {qd} != qk_nope_dim({}) + qk_rope_dim({})",
                                    m.qk_nope_dim, m.qk_rope_dim
                                ),
                            });
                        }
                    }
                }
            }
        }

        if let Some(o) = outputs.first() {
            check_rank("attention", "o", o, 3, &mut problems);
            if o.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "attention",
                    tensor: "o",
                    expected: Box::new([self.out_dtype]),
                    got: o.dtype(),
                });
            }
            if o.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "attention",
                    tensor: "o",
                    expected: Class::Activation,
                    got: o.class(),
                });
            }
            if let Some(ref m) = self.mla {
                if o.rank() == 3 {
                    if let Dim::Concrete(od) = o.shape()[2] {
                        if od != m.v_dim {
                            problems.push(IrError::OpShapeMismatch {
                                op: "attention",
                                tensor: "o",
                                detail: format!("o head dim {od} != v_dim {}", m.v_dim),
                            });
                        }
                    }
                }
            }
        }

        if let (Some(q), Some(o)) = (inputs.first(), outputs.first()) {
            if q.rank() == 3 && o.rank() == 3 {
                check_dim_match(
                    "attention",
                    "o",
                    o.shape()[0],
                    "q",
                    q.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "attention",
                    "o",
                    o.shape()[1],
                    "q",
                    q.shape()[1],
                    "H",
                    &mut problems,
                );

                if self.mla.is_none() {
                    check_dim_match(
                        "attention",
                        "o",
                        o.shape()[2],
                        "q",
                        q.shape()[2],
                        "D",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.D, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::ATTENTION_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::ATTENTION_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.D, §6.1, §6.3).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingBlock)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "attention"
    }
}

// -----------------------------------------------------------------------------
// §4.E Sequence-state ops beyond attention
// -----------------------------------------------------------------------------

/// 1D Causal Convolution state op (Spec 1 §4.E).
///
/// `x [T, C], w [C, W_k], bias? [C], StateHandle(ConvWindow) -> y [T, C]`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CausalConv1dOp {
    /// Convolution kernel length `W_k`.
    pub kernel: u32,
    /// Post-convolution activation function.
    pub act: ConvActivation,
    /// State handle managing the convolution window buffer.
    pub handle: StateHandle,
}

impl CausalConv1dOp {
    /// Validates inputs and outputs against Spec 1 §4.E constraints.
    // DECISION(A1.2): CausalConv1dOp accepts 2 inputs (x, w) or 3 inputs (x, w, bias [C]) per Spec 1 §4.E; candidate count error reported when input count is invalid.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if self.kernel == 0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "causal_conv1d",
                attribute: "kernel",
                reason: "kernel must be > 0".to_string(),
            });
        }
        match self.act {
            ConvActivation::Silu => {}
            ConvActivation::Identity => {}
        }
        match self.handle.kind() {
            StateKind::ConvWindow => {}
            StateKind::KvPaged | StateKind::KvLatent | StateKind::Recurrent => {
                problems.push(IrError::StateHandleKindMismatch {
                    op: "causal_conv1d",
                    expected: StateKind::ConvWindow,
                    got: self.handle.kind(),
                });
            }
        }

        let input_count_valid = inputs.len() == 2 || inputs.len() == 3;
        if !input_count_valid {
            problems.push(IrError::OpInputCountCandidatesMismatch {
                op: "causal_conv1d",
                expected: Box::new([2, 3]),
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "causal_conv1d",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(x) = inputs.first() {
            check_rank("causal_conv1d", "x", x, 2, &mut problems);
            check_dtype_in(
                "causal_conv1d",
                "x",
                x,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "causal_conv1d",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
        }

        if let Some(w) = inputs.get(1) {
            check_rank("causal_conv1d", "w", w, 2, &mut problems);
            check_dtype_in(
                "causal_conv1d",
                "w",
                w,
                &[DType::F16, DType::Bf16, DType::F32, DType::I8, DType::I4],
                &mut problems,
            );
            if w.class() != Class::Weight {
                problems.push(IrError::OpClassMismatch {
                    op: "causal_conv1d",
                    tensor: "w",
                    expected: Class::Weight,
                    got: w.class(),
                });
            }
            if w.rank() == 2 {
                if let Dim::Concrete(k) = w.shape()[1] {
                    if k != self.kernel {
                        problems.push(IrError::OpShapeMismatch {
                            op: "causal_conv1d",
                            tensor: "w",
                            detail: format!("kernel width {k} != attr kernel {}", self.kernel),
                        });
                    }
                }
            }
        }

        if let (Some(x), Some(w)) = (inputs.first(), inputs.get(1)) {
            if x.rank() == 2 && w.rank() == 2 {
                check_dim_match(
                    "causal_conv1d",
                    "w",
                    w.shape()[0],
                    "x",
                    x.shape()[1],
                    "C",
                    &mut problems,
                );
            }
        }

        if let Some(bias) = inputs.get(2) {
            check_rank("causal_conv1d", "bias", bias, 1, &mut problems);
            check_dtype_in(
                "causal_conv1d",
                "bias",
                bias,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if bias.class() != Class::Param {
                problems.push(IrError::OpClassMismatch {
                    op: "causal_conv1d",
                    tensor: "bias",
                    expected: Class::Param,
                    got: bias.class(),
                });
            }
            if let Some(x) = inputs.first() {
                if x.rank() == 2 && bias.rank() == 1 {
                    check_dim_match(
                        "causal_conv1d",
                        "bias",
                        bias.shape()[0],
                        "x",
                        x.shape()[1],
                        "C",
                        &mut problems,
                    );
                }
            }
        }

        if let Some(y) = outputs.first() {
            check_rank("causal_conv1d", "y", y, 2, &mut problems);
            check_dtype_in(
                "causal_conv1d",
                "y",
                y,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "causal_conv1d",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if let (Some(x), Some(y)) = (inputs.first(), outputs.first()) {
            if y.dtype() != x.dtype() {
                problems.push(IrError::OpDTypeMismatch {
                    op: "causal_conv1d",
                    tensor: "y",
                    expected: Box::new([x.dtype()]),
                    got: y.dtype(),
                });
            }
            if x.rank() == 2 && y.rank() == 2 {
                check_dim_match(
                    "causal_conv1d",
                    "y",
                    y.shape()[0],
                    "x",
                    x.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "causal_conv1d",
                    "y",
                    y.shape()[1],
                    "x",
                    x.shape()[1],
                    "C",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.E, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::CAUSAL_CONV1D_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::CAUSAL_CONV1D_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.E, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "causal_conv1d"
    }
}

/// Linear attention and structured state-space scan op (Spec 1 §4.E).
///
/// `q [T, H, D], k [T, H, D], v [T, H, Dv], alpha [T, H] f32, beta [T, H] f32, StateHandle -> o [T, H, Dv]`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LinearAttnScanOp {
    /// Recurrence / scan architecture kind.
    pub kind: LinearAttnKind,
    /// Chunk size for chunked parallel scan (default 64).
    pub chunk: u32,
    /// Destination activation dtype.
    pub out_dtype: DType,
    /// State handle managing the recurrent state matrices.
    pub handle: StateHandle,
}

impl LinearAttnScanOp {
    /// Validates inputs and outputs against Spec 1 §4.E constraints.
    // DECISION(A1.2): LinearAttnScanOp validates exact tensor operands (q, k, v, alpha, beta) -> o; recurrent state is managed via StateHandle per Spec 1 §4.E.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        if self.chunk == 0 {
            problems.push(IrError::OpAttributeInvalid {
                op: "linear_attn_scan",
                attribute: "chunk",
                reason: "chunk must be > 0".to_string(),
            });
        }
        match self.out_dtype {
            DType::F16 | DType::Bf16 | DType::F32 => {}
            DType::E4m3
            | DType::E5m2
            | DType::I4
            | DType::I8
            | DType::I32
            | DType::U32
            | DType::Bool => {
                problems.push(IrError::OpAttributeInvalid {
                    op: "linear_attn_scan",
                    attribute: "out_dtype",
                    reason: format!("must be f16, bf16, or f32, got {:?}", self.out_dtype),
                });
            }
        }
        match self.kind {
            LinearAttnKind::GatedDeltaNet => {}
            LinearAttnKind::GLA => {}
            LinearAttnKind::Mamba2 => {}
        }
        match self.handle.kind() {
            StateKind::Recurrent => {}
            StateKind::KvPaged | StateKind::KvLatent | StateKind::ConvWindow => {
                problems.push(IrError::StateHandleKindMismatch {
                    op: "linear_attn_scan",
                    expected: StateKind::Recurrent,
                    got: self.handle.kind(),
                });
            }
        }

        let input_count_valid = inputs.len() == 5;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "linear_attn_scan",
                expected: 5,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "linear_attn_scan",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(q) = inputs.first() {
            check_rank("linear_attn_scan", "q", q, 3, &mut problems);
            check_dtype_in(
                "linear_attn_scan",
                "q",
                q,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if q.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "linear_attn_scan",
                    tensor: "q",
                    expected: Class::Activation,
                    got: q.class(),
                });
            }
        }

        if let Some(k) = inputs.get(1) {
            check_rank("linear_attn_scan", "k", k, 3, &mut problems);
            check_dtype_in(
                "linear_attn_scan",
                "k",
                k,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if k.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "linear_attn_scan",
                    tensor: "k",
                    expected: Class::Activation,
                    got: k.class(),
                });
            }
            if let Some(q) = inputs.first() {
                if k.dtype() != q.dtype() {
                    problems.push(IrError::OpDTypeMismatch {
                        op: "linear_attn_scan",
                        tensor: "k",
                        expected: Box::new([q.dtype()]),
                        got: k.dtype(),
                    });
                }
            }
        }

        if let Some(v) = inputs.get(2) {
            check_rank("linear_attn_scan", "v", v, 3, &mut problems);
            check_dtype_in(
                "linear_attn_scan",
                "v",
                v,
                &[DType::F16, DType::Bf16, DType::F32],
                &mut problems,
            );
            if v.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "linear_attn_scan",
                    tensor: "v",
                    expected: Class::Activation,
                    got: v.class(),
                });
            }
            if let Some(q) = inputs.first() {
                if v.dtype() != q.dtype() {
                    problems.push(IrError::OpDTypeMismatch {
                        op: "linear_attn_scan",
                        tensor: "v",
                        expected: Box::new([q.dtype()]),
                        got: v.dtype(),
                    });
                }
            }
        }

        if let Some(alpha) = inputs.get(3) {
            check_rank("linear_attn_scan", "alpha", alpha, 2, &mut problems);
            check_dtype_in(
                "linear_attn_scan",
                "alpha",
                alpha,
                &[DType::F32],
                &mut problems,
            );
            if alpha.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "linear_attn_scan",
                    tensor: "alpha",
                    expected: Class::Activation,
                    got: alpha.class(),
                });
            }
        }

        if let Some(beta) = inputs.get(4) {
            check_rank("linear_attn_scan", "beta", beta, 2, &mut problems);
            check_dtype_in(
                "linear_attn_scan",
                "beta",
                beta,
                &[DType::F32],
                &mut problems,
            );
            if beta.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "linear_attn_scan",
                    tensor: "beta",
                    expected: Class::Activation,
                    got: beta.class(),
                });
            }
        }

        if let (Some(q), Some(k)) = (inputs.first(), inputs.get(1)) {
            if q.rank() == 3 && k.rank() == 3 {
                check_dim_match(
                    "linear_attn_scan",
                    "k",
                    k.shape()[0],
                    "q",
                    q.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "linear_attn_scan",
                    "k",
                    k.shape()[1],
                    "q",
                    q.shape()[1],
                    "H",
                    &mut problems,
                );
                check_dim_match(
                    "linear_attn_scan",
                    "k",
                    k.shape()[2],
                    "q",
                    q.shape()[2],
                    "D",
                    &mut problems,
                );
            }
        }

        if let (Some(q), Some(v)) = (inputs.first(), inputs.get(2)) {
            if q.rank() == 3 && v.rank() == 3 {
                check_dim_match(
                    "linear_attn_scan",
                    "v",
                    v.shape()[0],
                    "q",
                    q.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "linear_attn_scan",
                    "v",
                    v.shape()[1],
                    "q",
                    q.shape()[1],
                    "H",
                    &mut problems,
                );
            }
        }

        if let (Some(q), Some(alpha)) = (inputs.first(), inputs.get(3)) {
            if q.rank() == 3 && alpha.rank() == 2 {
                check_dim_match(
                    "linear_attn_scan",
                    "alpha",
                    alpha.shape()[0],
                    "q",
                    q.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "linear_attn_scan",
                    "alpha",
                    alpha.shape()[1],
                    "q",
                    q.shape()[1],
                    "H",
                    &mut problems,
                );
            }
        }

        if let (Some(q), Some(beta)) = (inputs.first(), inputs.get(4)) {
            if q.rank() == 3 && beta.rank() == 2 {
                check_dim_match(
                    "linear_attn_scan",
                    "beta",
                    beta.shape()[0],
                    "q",
                    q.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "linear_attn_scan",
                    "beta",
                    beta.shape()[1],
                    "q",
                    q.shape()[1],
                    "H",
                    &mut problems,
                );
            }
        }

        if let Some(o) = outputs.first() {
            check_rank("linear_attn_scan", "o", o, 3, &mut problems);
            if o.dtype() != self.out_dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "linear_attn_scan",
                    tensor: "o",
                    expected: Box::new([self.out_dtype]),
                    got: o.dtype(),
                });
            }
            if o.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "linear_attn_scan",
                    tensor: "o",
                    expected: Class::Activation,
                    got: o.class(),
                });
            }
        }

        if let (Some(q), Some(o)) = (inputs.first(), outputs.first()) {
            if q.rank() == 3 && o.rank() == 3 {
                check_dim_match(
                    "linear_attn_scan",
                    "o",
                    o.shape()[0],
                    "q",
                    q.shape()[0],
                    "T",
                    &mut problems,
                );
                check_dim_match(
                    "linear_attn_scan",
                    "o",
                    o.shape()[1],
                    "q",
                    q.shape()[1],
                    "H",
                    &mut problems,
                );
            }
        }

        if let (Some(v), Some(o)) = (inputs.get(2), outputs.first()) {
            if v.rank() == 3 && o.rank() == 3 {
                check_dim_match(
                    "linear_attn_scan",
                    "o",
                    o.shape()[2],
                    "v",
                    v.shape()[2],
                    "Dv",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.E, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::LINEAR_ATTN_SCAN_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::LINEAR_ATTN_SCAN_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.E, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "linear_attn_scan"
    }
}

// -----------------------------------------------------------------------------
// §4.F Sampling and verification
// -----------------------------------------------------------------------------

/// Logit postprocessing, temperature, penalties, and grammar masking op (Spec 1 §4.F).
///
/// `logits [S, q, V] f32, params, history_counts?, grammar_mask? -> probs [S, q, V] f32`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LogitsPostprocessOp;

impl LogitsPostprocessOp {
    /// Validates inputs and outputs against Spec 1 §4.F constraints.
    // DECISION(A1.2): LogitsPostprocessOp validates exact rank-3 logits [S, q, V] -> probs [S, q, V], distinguishing optional history_counts [S, V] (U32, rank 2) from grammar_mask [S, q, V] (Bool, rank 3) by dtype and rank; SamplingParams is validated as structured external metadata per Spec 1 §4.F and SI-12; rejected rank-2 logits/probs in op validation because multi-sequence verify steps require explicit query dimension q.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = !inputs.is_empty() && inputs.len() <= 3;
        if !input_count_valid {
            problems.push(IrError::OpInputCountCandidatesMismatch {
                op: "logits_postprocess",
                expected: Box::new([1, 2, 3]),
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "logits_postprocess",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(logits) = inputs.first() {
            check_rank("logits_postprocess", "logits", logits, 3, &mut problems);
            check_dtype_in(
                "logits_postprocess",
                "logits",
                logits,
                &[DType::F32],
                &mut problems,
            );
            if logits.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "logits_postprocess",
                    tensor: "logits",
                    expected: Class::Activation,
                    got: logits.class(),
                });
            }
        }

        if inputs.len() == 2 {
            let opt = &inputs[1];
            if opt.dtype() == DType::Bool || opt.rank() == 3 {
                check_rank("logits_postprocess", "grammar_mask", opt, 3, &mut problems);
                check_dtype_in(
                    "logits_postprocess",
                    "grammar_mask",
                    opt,
                    &[DType::Bool],
                    &mut problems,
                );
                if opt.class() != Class::Activation {
                    problems.push(IrError::OpClassMismatch {
                        op: "logits_postprocess",
                        tensor: "grammar_mask",
                        expected: Class::Activation,
                        got: opt.class(),
                    });
                }
                if let Some(logits) = inputs.first() {
                    if logits.rank() == 3 && opt.rank() == 3 {
                        check_dim_match(
                            "logits_postprocess",
                            "grammar_mask",
                            opt.shape()[0],
                            "logits",
                            logits.shape()[0],
                            "S",
                            &mut problems,
                        );
                        check_dim_match(
                            "logits_postprocess",
                            "grammar_mask",
                            opt.shape()[1],
                            "logits",
                            logits.shape()[1],
                            "q",
                            &mut problems,
                        );
                        check_dim_match(
                            "logits_postprocess",
                            "grammar_mask",
                            opt.shape()[2],
                            "logits",
                            logits.shape()[2],
                            "V",
                            &mut problems,
                        );
                    }
                }
            } else {
                check_rank(
                    "logits_postprocess",
                    "history_counts",
                    opt,
                    2,
                    &mut problems,
                );
                check_dtype_in(
                    "logits_postprocess",
                    "history_counts",
                    opt,
                    &[DType::U32],
                    &mut problems,
                );
                if opt.class() != Class::Activation {
                    problems.push(IrError::OpClassMismatch {
                        op: "logits_postprocess",
                        tensor: "history_counts",
                        expected: Class::Activation,
                        got: opt.class(),
                    });
                }
                if let Some(logits) = inputs.first() {
                    if logits.rank() == 3 && opt.rank() == 2 {
                        check_dim_match(
                            "logits_postprocess",
                            "history_counts",
                            opt.shape()[0],
                            "logits",
                            logits.shape()[0],
                            "S",
                            &mut problems,
                        );
                        check_dim_match(
                            "logits_postprocess",
                            "history_counts",
                            opt.shape()[1],
                            "logits",
                            logits.shape()[2],
                            "V",
                            &mut problems,
                        );
                    }
                }
            }
        } else if inputs.len() == 3 {
            let history = &inputs[1];
            let mask = &inputs[2];

            check_rank(
                "logits_postprocess",
                "history_counts",
                history,
                2,
                &mut problems,
            );
            check_dtype_in(
                "logits_postprocess",
                "history_counts",
                history,
                &[DType::U32],
                &mut problems,
            );
            if history.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "logits_postprocess",
                    tensor: "history_counts",
                    expected: Class::Activation,
                    got: history.class(),
                });
            }
            if let Some(logits) = inputs.first() {
                if logits.rank() == 3 && history.rank() == 2 {
                    check_dim_match(
                        "logits_postprocess",
                        "history_counts",
                        history.shape()[0],
                        "logits",
                        logits.shape()[0],
                        "S",
                        &mut problems,
                    );
                    check_dim_match(
                        "logits_postprocess",
                        "history_counts",
                        history.shape()[1],
                        "logits",
                        logits.shape()[2],
                        "V",
                        &mut problems,
                    );
                }
            }

            check_rank("logits_postprocess", "grammar_mask", mask, 3, &mut problems);
            check_dtype_in(
                "logits_postprocess",
                "grammar_mask",
                mask,
                &[DType::Bool],
                &mut problems,
            );
            if mask.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "logits_postprocess",
                    tensor: "grammar_mask",
                    expected: Class::Activation,
                    got: mask.class(),
                });
            }
            if let Some(logits) = inputs.first() {
                if logits.rank() == 3 && mask.rank() == 3 {
                    check_dim_match(
                        "logits_postprocess",
                        "grammar_mask",
                        mask.shape()[0],
                        "logits",
                        logits.shape()[0],
                        "S",
                        &mut problems,
                    );
                    check_dim_match(
                        "logits_postprocess",
                        "grammar_mask",
                        mask.shape()[1],
                        "logits",
                        logits.shape()[1],
                        "q",
                        &mut problems,
                    );
                    check_dim_match(
                        "logits_postprocess",
                        "grammar_mask",
                        mask.shape()[2],
                        "logits",
                        logits.shape()[2],
                        "V",
                        &mut problems,
                    );
                }
            }
        }

        if let Some(probs) = outputs.first() {
            check_rank("logits_postprocess", "probs", probs, 3, &mut problems);
            check_dtype_in(
                "logits_postprocess",
                "probs",
                probs,
                &[DType::F32],
                &mut problems,
            );
            if probs.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "logits_postprocess",
                    tensor: "probs",
                    expected: Class::Activation,
                    got: probs.class(),
                });
            }
        }

        if let (Some(logits), Some(probs)) = (inputs.first(), outputs.first()) {
            if logits.rank() == 3 && probs.rank() == 3 {
                check_dim_match(
                    "logits_postprocess",
                    "probs",
                    probs.shape()[0],
                    "logits",
                    logits.shape()[0],
                    "S",
                    &mut problems,
                );
                check_dim_match(
                    "logits_postprocess",
                    "probs",
                    probs.shape()[1],
                    "logits",
                    logits.shape()[1],
                    "q",
                    &mut problems,
                );
                check_dim_match(
                    "logits_postprocess",
                    "probs",
                    probs.shape()[2],
                    "logits",
                    logits.shape()[2],
                    "V",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.F, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::LOGITS_POSTPROCESS_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::LOGITS_POSTPROCESS_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.F, §6.1, §6.5).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "logits_postprocess"
    }
}

/// Stochastic or greedy token sampling op (Spec 1 §4.F).
///
/// `probs [S, V] f32, rng_state [S] -> token [S] u32, rng_state' [S]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleOp {
    /// PRNG algorithm.
    pub rng: RngAlgorithm,
}

impl SampleOp {
    /// Validates inputs and outputs against Spec 1 §4.F constraints.
    // DECISION(A1.2): SampleOp validates exactly one probs [S, V] tensor to one token [S] tensor; rng_state is typed external metadata per Spec 1 §4.F and SI-12.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        match self.rng {
            RngAlgorithm::Philox4x32 => {}
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "sample",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "sample",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(probs) = inputs.first() {
            check_rank("sample", "probs", probs, 2, &mut problems);
            check_dtype_in("sample", "probs", probs, &[DType::F32], &mut problems);
            if probs.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "sample",
                    tensor: "probs",
                    expected: Class::Activation,
                    got: probs.class(),
                });
            }
        }

        if let Some(token) = outputs.first() {
            check_rank("sample", "token", token, 1, &mut problems);
            check_dtype_in("sample", "token", token, &[DType::U32], &mut problems);
            if token.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "sample",
                    tensor: "token",
                    expected: Class::Activation,
                    got: token.class(),
                });
            }
        }

        if let (Some(probs), Some(token)) = (inputs.first(), outputs.first()) {
            if probs.rank() == 2 && token.rank() == 1 {
                check_dim_match(
                    "sample",
                    "token",
                    token.shape()[0],
                    "probs",
                    probs.shape()[0],
                    "S",
                    &mut problems,
                );
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.F, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::SAMPLE_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::SAMPLE_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.F, §6.1, §6.5).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "sample"
    }
}

/// Speculative decoding verification op (Spec 1 §4.F).
///
/// `draft_tokens [S, k] u32, target_probs [S, k+1, V] f32, ... -> accepted [S, k+1] u32, accept_len [S] u32`
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyOp {
    /// Verification and acceptance policy.
    pub method: VerifyMethod,
}

impl VerifyOp {
    /// Validates inputs and outputs against Spec 1 §4.F constraints.
    // DECISION(A1.2): VerifyOp validates exact tensor operands draft_tokens [S, k], target_probs [S, k+1, V], and optional draft_probs [S, k, V] to accepted [S, k+1] and accept_len [S]; tree mask and rng_state are typed external metadata per Spec 1 §4.F, Spec 7 §4, SI-12.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        match self.method {
            VerifyMethod::Rejection => {}
            VerifyMethod::Greedy => {}
            VerifyMethod::TypicalAcceptance { eps, delta } => {
                if !eps.is_finite() || eps <= 0.0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "verify",
                        attribute: "method.eps",
                        reason: format!("eps must be finite and > 0, got {eps}"),
                    });
                }
                if !delta.is_finite() || delta <= 0.0 {
                    problems.push(IrError::OpAttributeInvalid {
                        op: "verify",
                        attribute: "method.delta",
                        reason: format!("delta must be finite and > 0, got {delta}"),
                    });
                }
            }
        }

        let input_count_valid = inputs.len() == 2 || inputs.len() == 3;
        if !input_count_valid {
            problems.push(IrError::OpInputCountCandidatesMismatch {
                op: "verify",
                expected: Box::new([2, 3]),
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 2;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "verify",
                expected: 2,
                got: outputs.len(),
            });
        }

        let draft_probs_opt = (inputs.len() == 3).then(|| &inputs[1]);
        let target_probs_opt = match inputs.len() {
            2 => inputs.get(1),
            3 => inputs.get(2),
            0 | 1 | 4.. => None,
        };

        if let Some(draft_tokens) = inputs.first() {
            check_rank("verify", "draft_tokens", draft_tokens, 2, &mut problems);
            check_dtype_in(
                "verify",
                "draft_tokens",
                draft_tokens,
                &[DType::U32],
                &mut problems,
            );
            if draft_tokens.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "verify",
                    tensor: "draft_tokens",
                    expected: Class::Activation,
                    got: draft_tokens.class(),
                });
            }
        }

        if let Some(target_probs) = target_probs_opt {
            check_rank("verify", "target_probs", target_probs, 3, &mut problems);
            check_dtype_in(
                "verify",
                "target_probs",
                target_probs,
                &[DType::F32],
                &mut problems,
            );
            if target_probs.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "verify",
                    tensor: "target_probs",
                    expected: Class::Activation,
                    got: target_probs.class(),
                });
            }
            if let Some(draft_tokens) = inputs.first() {
                if draft_tokens.rank() == 2 && target_probs.rank() == 3 {
                    check_dim_match(
                        "verify",
                        "target_probs",
                        target_probs.shape()[0],
                        "draft_tokens",
                        draft_tokens.shape()[0],
                        "S",
                        &mut problems,
                    );
                    if let (Dim::Concrete(k), Dim::Concrete(k1)) =
                        (draft_tokens.shape()[1], target_probs.shape()[1])
                    {
                        let expected_k1 = k.checked_add(1);
                        if expected_k1 != Some(k1) {
                            let expected = expected_k1.map_or_else(
                                || "overflow (> u32::MAX)".to_string(),
                                |value| value.to_string(),
                            );
                            problems.push(IrError::OpShapeMismatch {
                                op: "verify",
                                tensor: "target_probs",
                                detail: format!(
                                    "target_probs axis 1 extent {k1} != draft_tokens axis 1 + 1 ({})",
                                    expected
                                ),
                            });
                        }
                    }
                }
            }
        }

        if let Some(draft_probs) = draft_probs_opt {
            check_rank("verify", "draft_probs", draft_probs, 3, &mut problems);
            check_dtype_in(
                "verify",
                "draft_probs",
                draft_probs,
                &[DType::F32],
                &mut problems,
            );
            if draft_probs.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "verify",
                    tensor: "draft_probs",
                    expected: Class::Activation,
                    got: draft_probs.class(),
                });
            }
            if let Some(draft_tokens) = inputs.first() {
                if draft_tokens.rank() == 2 && draft_probs.rank() == 3 {
                    check_dim_match(
                        "verify",
                        "draft_probs",
                        draft_probs.shape()[0],
                        "draft_tokens",
                        draft_tokens.shape()[0],
                        "S",
                        &mut problems,
                    );
                    check_dim_match(
                        "verify",
                        "draft_probs",
                        draft_probs.shape()[1],
                        "draft_tokens",
                        draft_tokens.shape()[1],
                        "k",
                        &mut problems,
                    );
                }
            }
            if let Some(target_probs) = target_probs_opt {
                if draft_probs.rank() == 3 && target_probs.rank() == 3 {
                    check_dim_match(
                        "verify",
                        "draft_probs",
                        draft_probs.shape()[2],
                        "target_probs",
                        target_probs.shape()[2],
                        "V",
                        &mut problems,
                    );
                }
            }
        }

        if let Some(accepted) = outputs.first() {
            check_rank("verify", "accepted", accepted, 2, &mut problems);
            check_dtype_in("verify", "accepted", accepted, &[DType::U32], &mut problems);
            if accepted.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "verify",
                    tensor: "accepted",
                    expected: Class::Activation,
                    got: accepted.class(),
                });
            }
            if let Some(draft_tokens) = inputs.first() {
                if draft_tokens.rank() == 2 && accepted.rank() == 2 {
                    check_dim_match(
                        "verify",
                        "accepted",
                        accepted.shape()[0],
                        "draft_tokens",
                        draft_tokens.shape()[0],
                        "S",
                        &mut problems,
                    );
                    if let (Dim::Concrete(k), Dim::Concrete(ak1)) =
                        (draft_tokens.shape()[1], accepted.shape()[1])
                    {
                        let expected_ak1 = k.checked_add(1);
                        if expected_ak1 != Some(ak1) {
                            let expected = expected_ak1.map_or_else(
                                || "overflow (> u32::MAX)".to_string(),
                                |value| value.to_string(),
                            );
                            problems.push(IrError::OpShapeMismatch {
                                op: "verify",
                                tensor: "accepted",
                                detail: format!(
                                    "accepted axis 1 extent {ak1} != draft_tokens axis 1 + 1 ({})",
                                    expected
                                ),
                            });
                        }
                    }
                }
            }
        }

        if let Some(accept_len) = outputs.get(1) {
            check_rank("verify", "accept_len", accept_len, 1, &mut problems);
            check_dtype_in(
                "verify",
                "accept_len",
                accept_len,
                &[DType::U32],
                &mut problems,
            );
            if accept_len.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "verify",
                    tensor: "accept_len",
                    expected: Class::Activation,
                    got: accept_len.class(),
                });
            }
            if let Some(draft_tokens) = inputs.first() {
                if draft_tokens.rank() == 2 && accept_len.rank() == 1 {
                    check_dim_match(
                        "verify",
                        "accept_len",
                        accept_len.shape()[0],
                        "draft_tokens",
                        draft_tokens.shape()[0],
                        "S",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.F, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::VERIFY_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::VERIFY_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.F, §6.1, §6.5).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingIndex)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "verify"
    }
}

// -----------------------------------------------------------------------------
// §4.G Collectives
// -----------------------------------------------------------------------------

/// All-reduce collective op (Spec 1 §4.G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllReduceOp {
    /// Communication group identifier.
    pub group: GroupId,
    /// Reduction operator (e.g. Sum).
    pub op: ReduceOp,
    /// Element dtype.
    pub dtype: DType,
    /// Internal accumulator dtype (f32 per Spec 1 §4.G).
    pub reduce_in: DType,
}

impl AllReduceOp {
    /// Validates inputs and outputs against Spec 1 §4.G constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        match self.op {
            ReduceOp::Sum => {}
        }

        if self.reduce_in != DType::F32 {
            problems.push(IrError::OpAttributeInvalid {
                op: "all_reduce",
                attribute: "reduce_in",
                reason: format!("must be f32 per Spec 1 §4.G, got {:?}", self.reduce_in),
            });
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "all_reduce",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "all_reduce",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(x) = inputs.first() {
            if x.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "all_reduce",
                    tensor: "x",
                    expected: Box::new([self.dtype]),
                    got: x.dtype(),
                });
            }
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "all_reduce",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
        }

        if let Some(y) = outputs.first() {
            if y.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "all_reduce",
                    tensor: "y",
                    expected: Box::new([self.dtype]),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "all_reduce",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if let (Some(x), Some(y)) = (inputs.first(), outputs.first()) {
            if y.rank() != x.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "all_reduce",
                    tensor: "y",
                    expected: x.rank(),
                    got: y.rank(),
                });
            } else {
                for i in 0..x.rank() {
                    check_dim_match(
                        "all_reduce",
                        "y",
                        y.shape()[i],
                        "x",
                        x.shape()[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.G, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::ALL_REDUCE_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::ALL_REDUCE_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.G, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingRank)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "all_reduce"
    }
}

/// All-gather collective op (Spec 1 §4.G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllGatherOp {
    /// Communication group identifier.
    pub group: GroupId,
    /// Axis along which tensors are concatenated.
    pub axis: u32,
    /// Element dtype.
    pub dtype: DType,
}

impl AllGatherOp {
    /// Validates inputs and outputs against Spec 1 §4.G constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "all_gather",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "all_gather",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(x) = inputs.first() {
            if x.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "all_gather",
                    tensor: "x",
                    expected: Box::new([self.dtype]),
                    got: x.dtype(),
                });
            }
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "all_gather",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
            if self.axis >= x.rank() as u32 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "all_gather",
                    attribute: "axis",
                    reason: format!("axis {} out of bounds for rank {}", self.axis, x.rank()),
                });
            }
        }

        if let Some(y) = outputs.first() {
            if y.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "all_gather",
                    tensor: "y",
                    expected: Box::new([self.dtype]),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "all_gather",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
            if inputs.is_empty() && self.axis >= y.rank() as u32 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "all_gather",
                    attribute: "axis",
                    reason: format!("axis {} out of bounds for rank {}", self.axis, y.rank()),
                });
            }
        }

        if let (Some(x), Some(y)) = (inputs.first(), outputs.first()) {
            if y.rank() != x.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "all_gather",
                    tensor: "y",
                    expected: x.rank(),
                    got: y.rank(),
                });
            } else {
                for i in 0..x.rank() {
                    if i as u32 != self.axis {
                        check_dim_match(
                            "all_gather",
                            "y",
                            y.shape()[i],
                            "x",
                            x.shape()[i],
                            "dim",
                            &mut problems,
                        );
                    } else if let (Dim::Concrete(x_dim), Dim::Concrete(y_dim)) =
                        (x.shape()[i], y.shape()[i])
                    {
                        if x_dim == 0 || y_dim < x_dim || y_dim % x_dim != 0 {
                            problems.push(IrError::OpShapeMismatch {
                                op: "all_gather",
                                tensor: "y",
                                detail: format!(
                                    "gathered axis {} output extent {y_dim} is not a valid non-zero multiple of input extent {x_dim}",
                                    self.axis
                                ),
                            });
                        }
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.G, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::ALL_GATHER_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::ALL_GATHER_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.G, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "all_gather"
    }
}

/// Reduce-scatter collective op (Spec 1 §4.G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceScatterOp {
    /// Communication group identifier.
    pub group: GroupId,
    /// Axis along which reduction partitions are distributed.
    pub axis: u32,
    /// Reduction operator.
    pub op: ReduceOp,
    /// Element dtype.
    pub dtype: DType,
    /// Internal accumulator dtype.
    pub reduce_in: DType,
}

impl ReduceScatterOp {
    /// Validates inputs and outputs against Spec 1 §4.G constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        match self.op {
            ReduceOp::Sum => {}
        }

        if self.reduce_in != DType::F32 {
            problems.push(IrError::OpAttributeInvalid {
                op: "reduce_scatter",
                attribute: "reduce_in",
                reason: format!("must be f32 per Spec 1 §4.G, got {:?}", self.reduce_in),
            });
        }

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "reduce_scatter",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "reduce_scatter",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(x) = inputs.first() {
            if x.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "reduce_scatter",
                    tensor: "x",
                    expected: Box::new([self.dtype]),
                    got: x.dtype(),
                });
            }
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "reduce_scatter",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
            if self.axis >= x.rank() as u32 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "reduce_scatter",
                    attribute: "axis",
                    reason: format!("axis {} out of bounds for rank {}", self.axis, x.rank()),
                });
            }
        }

        if let Some(y) = outputs.first() {
            if y.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "reduce_scatter",
                    tensor: "y",
                    expected: Box::new([self.dtype]),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "reduce_scatter",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
            if inputs.is_empty() && self.axis >= y.rank() as u32 {
                problems.push(IrError::OpAttributeInvalid {
                    op: "reduce_scatter",
                    attribute: "axis",
                    reason: format!("axis {} out of bounds for rank {}", self.axis, y.rank()),
                });
            }
        }

        if let (Some(x), Some(y)) = (inputs.first(), outputs.first()) {
            if y.rank() != x.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "reduce_scatter",
                    tensor: "y",
                    expected: x.rank(),
                    got: y.rank(),
                });
            } else {
                for i in 0..x.rank() {
                    if i as u32 != self.axis {
                        check_dim_match(
                            "reduce_scatter",
                            "y",
                            y.shape()[i],
                            "x",
                            x.shape()[i],
                            "dim",
                            &mut problems,
                        );
                    } else if let (Dim::Concrete(x_dim), Dim::Concrete(y_dim)) =
                        (x.shape()[i], y.shape()[i])
                    {
                        if y_dim == 0 || x_dim < y_dim || x_dim % y_dim != 0 {
                            problems.push(IrError::OpShapeMismatch {
                                op: "reduce_scatter",
                                tensor: "y",
                                detail: format!(
                                    "scattered axis {} output extent {y_dim} is not a valid non-zero divisor of input extent {x_dim}",
                                    self.axis
                                ),
                            });
                        }
                    }
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.G, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::REDUCE_SCATTER_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::REDUCE_SCATTER_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.G, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::f32(ReductionOrder::AscendingRank)
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "reduce_scatter"
    }
}

/// All-to-all collective op for Expert Parallel token dispatch/combine (Spec 1 §4.G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllToAllOp {
    /// Communication group identifier.
    pub group: GroupId,
    /// Element dtype.
    pub dtype: DType,
}

impl AllToAllOp {
    /// Validates inputs and outputs against Spec 1 §4.G constraints.
    // DECISION(A1.2): AllToAllOp requires exactly two inputs (x, counts [P] u32) and one output y per Spec 1 §4.G and SI-11; rejected optional arities or invented recv_counts output.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 2;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "all_to_all",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "all_to_all",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(x) = inputs.first() {
            if x.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "all_to_all",
                    tensor: "x",
                    expected: Box::new([self.dtype]),
                    got: x.dtype(),
                });
            }
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "all_to_all",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
        }

        if let Some(counts) = inputs.get(1) {
            check_rank("all_to_all", "counts", counts, 1, &mut problems);
            check_dtype_in("all_to_all", "counts", counts, &[DType::U32], &mut problems);
            if counts.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "all_to_all",
                    tensor: "counts",
                    expected: Class::Activation,
                    got: counts.class(),
                });
            }
        }

        if let Some(y) = outputs.first() {
            if y.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "all_to_all",
                    tensor: "y",
                    expected: Box::new([self.dtype]),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "all_to_all",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
        }

        if let (Some(x), Some(y)) = (inputs.first(), outputs.first()) {
            if y.rank() != x.rank() {
                problems.push(IrError::OpRankMismatch {
                    op: "all_to_all",
                    tensor: "y",
                    expected: x.rank(),
                    got: y.rank(),
                });
            } else if x.rank() >= 2 {
                for i in 1..x.rank() {
                    check_dim_match(
                        "all_to_all",
                        "y",
                        y.shape()[i],
                        "x",
                        x.shape()[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.G, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::ALL_TO_ALL_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::ALL_TO_ALL_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.G, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "all_to_all"
    }
}

/// Point-to-point send collective op (Spec 1 §4.G).
// DECISION(A1.2): SendOp includes group: GroupId per Spec 1 §4.G and SI-11; rejected peer-only struct because point-to-point operations require communication group context for communicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SendOp {
    /// Communication group identifier.
    pub group: GroupId,
    /// Destination peer device rank.
    pub peer: u32,
    /// Transferred element dtype.
    pub dtype: DType,
}

impl SendOp {
    /// Validates inputs and outputs against Spec 1 §4.G constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.len() == 1;
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "send",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.is_empty();
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "send",
                expected: 0,
                got: outputs.len(),
            });
        }

        if let Some(x) = inputs.first() {
            if x.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "send",
                    tensor: "x",
                    expected: Box::new([self.dtype]),
                    got: x.dtype(),
                });
            }
            if x.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "send",
                    tensor: "x",
                    expected: Class::Activation,
                    got: x.class(),
                });
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.G, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::SEND_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::SEND_RULES.iter().map(|r| r.as_tuple()).collect()
    }

    /// Returns op numerics contract (Spec 1 §4.G, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "send"
    }
}

/// Point-to-point receive collective op (Spec 1 §4.G).
// DECISION(A1.2): RecvOp includes group: GroupId per Spec 1 §4.G and SI-11; rejected peer-only struct because point-to-point operations require communication group context for communicators.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecvOp {
    /// Communication group identifier.
    pub group: GroupId,
    /// Source peer device rank.
    pub peer: u32,
    /// Expected received tensor shape.
    pub shape: Box<[Dim]>,
    /// Received element dtype.
    pub dtype: DType,
}

impl RecvOp {
    /// Validates inputs and outputs against Spec 1 §4.G constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();

        let input_count_valid = inputs.is_empty();
        if !input_count_valid {
            problems.push(IrError::OpInputCountMismatch {
                op: "recv",
                expected: 0,
                got: inputs.len(),
            });
        }

        let output_count_valid = outputs.len() == 1;
        if !output_count_valid {
            problems.push(IrError::OpOutputCountMismatch {
                op: "recv",
                expected: 1,
                got: outputs.len(),
            });
        }

        if let Some(y) = outputs.first() {
            if y.dtype() != self.dtype {
                problems.push(IrError::OpDTypeMismatch {
                    op: "recv",
                    tensor: "y",
                    expected: Box::new([self.dtype]),
                    got: y.dtype(),
                });
            }
            if y.class() != Class::Activation {
                problems.push(IrError::OpClassMismatch {
                    op: "recv",
                    tensor: "y",
                    expected: Class::Activation,
                    got: y.class(),
                });
            }
            if y.rank() != self.shape.len() {
                problems.push(IrError::OpRankMismatch {
                    op: "recv",
                    tensor: "y",
                    expected: self.shape.len(),
                    got: y.rank(),
                });
            } else {
                for i in 0..self.shape.len() {
                    check_dim_match(
                        "recv",
                        "y",
                        y.shape()[i],
                        "attr.shape",
                        self.shape[i],
                        "dim",
                        &mut problems,
                    );
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.G, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::RECV_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::RECV_RULES.iter().map(|r| r.as_tuple()).collect()
    }

    /// Returns op numerics contract (Spec 1 §4.G, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "recv"
    }
}

/// Collective cross-device synchronization barrier op (Spec 1 §4.G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BarrierOp {
    /// Synchronized communication group identifier.
    pub group: GroupId,
}

impl BarrierOp {
    /// Validates inputs and outputs against Spec 1 §4.G constraints.
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();
        if !inputs.is_empty() {
            problems.push(IrError::OpInputCountMismatch {
                op: "barrier",
                expected: 0,
                got: inputs.len(),
            });
        }
        if !outputs.is_empty() {
            problems.push(IrError::OpOutputCountMismatch {
                op: "barrier",
                expected: 0,
                got: outputs.len(),
            });
        }
        IrError::from_problems(problems)
    }

    /// Returns legal sharding rules (Spec 1 §4.G, §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::BARRIER_RULES
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::BARRIER_RULES
            .iter()
            .map(|r| r.as_tuple())
            .collect()
    }

    /// Returns op numerics contract (Spec 1 §4.G, §6.1).
    pub fn numerics(&self) -> Numerics {
        Numerics::none()
    }

    /// Returns op identifier name.
    pub const fn op_name(&self) -> &'static str {
        "barrier"
    }
}

// -----------------------------------------------------------------------------
// Closed Op Enum (Spec 1 §4)
// -----------------------------------------------------------------------------

/// Closed set of operations supported by the Op IR (Spec 1 §4).
///
/// Exhaustive matching required per CONVENTIONS.md §3.2.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Token embedding lookup (Spec 1 §4.A).
    EmbedGather(EmbedGatherOp),
    /// N-gram prefix gather (Spec 1 §4.A).
    NgramGather(NgramGatherOp),
    /// Activation quantization (Spec 1 §4.A).
    QuantAct(QuantActOp),
    /// Precision casting (Spec 1 §4.A).
    Cast(CastOp),
    /// Memory copy and contiguization (Spec 1 §4.A).
    Copy(CopyOp),
    /// Row gather (Spec 1 §4.A).
    GatherRows(GatherRowsOp),
    /// Deterministic scatter-add rows (Spec 1 §4.A).
    ScatterAddRows(ScatterAddRowsOp),
    /// Last-axis channel split (card A1.14, SI-29).
    Split(SplitOp),
    /// Last-axis channel concatenation (card A1.14, SI-29).
    Concat(ConcatOp),
    /// Normalization: RMS or Layer norm (Spec 1 §4.B).
    Norm(NormOp),
    /// Residual addition (Spec 1 §4.B).
    ResidualAdd(ResidualAddOp),
    /// Gated activation multiplication (Spec 1 §4.B).
    ActMul(ActMulOp),
    /// Standalone activation (Spec 1 §4.B).
    Activation(ActivationOp),
    /// Final-logit softcap (card A1.14, SI-28).
    LogitSoftcap(LogitSoftcapOp),
    /// Rotary Position Embedding (Spec 1 §4.B).
    Rope(RopeOp),
    /// Matrix multiplication with epilogue (Spec 1 §4.C).
    Matmul(MatmulOp),
    /// Mixture of Experts routing (Spec 1 §4.C).
    MoeRoute(MoeRouteOp),
    /// Mixture of Experts feed-forward (Spec 1 §4.C).
    MoeFfn(MoeFfnOp),
    /// KV state cache write (Spec 1 §4.D).
    StateWriteKv(StateWriteKvOp),
    /// Paged / latent attention (Spec 1 §4.D).
    Attention(AttentionOp),
    /// 1D Causal Convolution (Spec 1 §4.E).
    CausalConv1d(CausalConv1dOp),
    /// Linear attention scan (Spec 1 §4.E).
    LinearAttnScan(LinearAttnScanOp),
    /// Logits postprocessing (Spec 1 §4.F).
    LogitsPostprocess(LogitsPostprocessOp),
    /// Stochastic or greedy sampling (Spec 1 §4.F).
    Sample(SampleOp),
    /// Speculative verification (Spec 1 §4.F).
    Verify(VerifyOp),
    /// All-reduce collective (Spec 1 §4.G).
    AllReduce(AllReduceOp),
    /// All-gather collective (Spec 1 §4.G).
    AllGather(AllGatherOp),
    /// Reduce-scatter collective (Spec 1 §4.G).
    ReduceScatter(ReduceScatterOp),
    /// All-to-all collective (Spec 1 §4.G).
    AllToAll(AllToAllOp),
    /// Point-to-point send (Spec 1 §4.G).
    Send(SendOp),
    /// Point-to-point receive (Spec 1 §4.G).
    Recv(RecvOp),
    /// Device barrier (Spec 1 §4.G).
    Barrier(BarrierOp),
}

impl Op {
    /// Validates inputs and outputs against the constraints for this op (Spec 1 §4).
    pub fn validate(
        &self,
        inputs: &[crate::Tensor],
        outputs: &[crate::Tensor],
    ) -> Result<(), IrError> {
        match self {
            Op::EmbedGather(op) => op.validate(inputs, outputs),
            Op::NgramGather(op) => op.validate(inputs, outputs),
            Op::QuantAct(op) => op.validate(inputs, outputs),
            Op::Cast(op) => op.validate(inputs, outputs),
            Op::Copy(op) => op.validate(inputs, outputs),
            Op::GatherRows(op) => op.validate(inputs, outputs),
            Op::ScatterAddRows(op) => op.validate(inputs, outputs),
            Op::Split(op) => op.validate(inputs, outputs),
            Op::Concat(op) => op.validate(inputs, outputs),
            Op::Norm(op) => op.validate(inputs, outputs),
            Op::ResidualAdd(op) => op.validate(inputs, outputs),
            Op::ActMul(op) => op.validate(inputs, outputs),
            Op::Activation(op) => op.validate(inputs, outputs),
            Op::LogitSoftcap(op) => op.validate(inputs, outputs),
            Op::Rope(op) => op.validate(inputs, outputs),
            Op::Matmul(op) => op.validate(inputs, outputs),
            Op::MoeRoute(op) => op.validate(inputs, outputs),
            Op::MoeFfn(op) => op.validate(inputs, outputs),
            Op::StateWriteKv(op) => op.validate(inputs, outputs),
            Op::Attention(op) => op.validate(inputs, outputs),
            Op::CausalConv1d(op) => op.validate(inputs, outputs),
            Op::LinearAttnScan(op) => op.validate(inputs, outputs),
            Op::LogitsPostprocess(op) => op.validate(inputs, outputs),
            Op::Sample(op) => op.validate(inputs, outputs),
            Op::Verify(op) => op.validate(inputs, outputs),
            Op::AllReduce(op) => op.validate(inputs, outputs),
            Op::AllGather(op) => op.validate(inputs, outputs),
            Op::ReduceScatter(op) => op.validate(inputs, outputs),
            Op::AllToAll(op) => op.validate(inputs, outputs),
            Op::Send(op) => op.validate(inputs, outputs),
            Op::Recv(op) => op.validate(inputs, outputs),
            Op::Barrier(op) => op.validate(inputs, outputs),
        }
    }

    /// Returns legal sharding rules (Spec 1 §5.2).
    pub fn legal_layouts(&self) -> &'static [ShardingRule] {
        sharding::legal_layouts(self)
    }

    /// Returns legal sharding rules as tuples.
    pub fn legal_layout_tuples(
        &self,
    ) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
        sharding::legal_layout_tuples(self)
    }

    /// Returns the op numerics contract for the supplied tensor inputs (Spec 1 §6.1).
    pub fn numerics(&self, inputs: &[crate::Tensor]) -> Result<Numerics, IrError> {
        match self {
            Op::EmbedGather(op) => Ok(op.numerics()),
            Op::NgramGather(op) => Ok(op.numerics()),
            Op::QuantAct(op) => Ok(op.numerics()),
            Op::Cast(op) => Ok(op.numerics()),
            Op::Copy(op) => Ok(op.numerics()),
            Op::GatherRows(op) => Ok(op.numerics()),
            Op::ScatterAddRows(op) => Ok(op.numerics()),
            Op::Split(op) => Ok(op.numerics()),
            Op::Concat(op) => Ok(op.numerics()),
            Op::Norm(op) => Ok(op.numerics()),
            Op::ResidualAdd(op) => Ok(op.numerics()),
            Op::ActMul(op) => Ok(op.numerics()),
            Op::Activation(op) => Ok(op.numerics()),
            Op::LogitSoftcap(op) => Ok(op.numerics()),
            Op::Rope(op) => Ok(op.numerics()),
            Op::Matmul(op) => {
                if inputs.len() < 2 {
                    Err(IrError::OpInputCountMismatch {
                        op: "matmul_numerics",
                        expected: 2,
                        got: inputs.len(),
                    })
                } else {
                    op.numerics_for(&inputs[0], &inputs[1])
                }
            }
            Op::MoeRoute(op) => Ok(op.numerics()),
            Op::MoeFfn(op) => {
                if inputs.len() < 5 {
                    Err(IrError::OpInputCountMismatch {
                        op: "moe_ffn_numerics",
                        expected: 5,
                        got: inputs.len(),
                    })
                } else {
                    let n = op.numerics_for(&inputs[0], &inputs[3])?;
                    let _ = op.numerics_for(&inputs[0], &inputs[4])?;
                    Ok(n)
                }
            }
            Op::StateWriteKv(op) => Ok(op.numerics()),
            Op::Attention(op) => Ok(op.numerics()),
            Op::CausalConv1d(op) => Ok(op.numerics()),
            Op::LinearAttnScan(op) => Ok(op.numerics()),
            Op::LogitsPostprocess(op) => Ok(op.numerics()),
            Op::Sample(op) => Ok(op.numerics()),
            Op::Verify(op) => Ok(op.numerics()),
            Op::AllReduce(op) => Ok(op.numerics()),
            Op::AllGather(op) => Ok(op.numerics()),
            Op::ReduceScatter(op) => Ok(op.numerics()),
            Op::AllToAll(op) => Ok(op.numerics()),
            Op::Send(op) => Ok(op.numerics()),
            Op::Recv(op) => Ok(op.numerics()),
            Op::Barrier(op) => Ok(op.numerics()),
        }
    }

    /// Returns op identifier name.
    pub fn op_name(&self) -> &'static str {
        match self {
            Op::EmbedGather(op) => op.op_name(),
            Op::NgramGather(op) => op.op_name(),
            Op::QuantAct(op) => op.op_name(),
            Op::Cast(op) => op.op_name(),
            Op::Copy(op) => op.op_name(),
            Op::GatherRows(op) => op.op_name(),
            Op::ScatterAddRows(op) => op.op_name(),
            Op::Split(op) => op.op_name(),
            Op::Concat(op) => op.op_name(),
            Op::Norm(op) => op.op_name(),
            Op::ResidualAdd(op) => op.op_name(),
            Op::ActMul(op) => op.op_name(),
            Op::Activation(op) => op.op_name(),
            Op::LogitSoftcap(op) => op.op_name(),
            Op::Rope(op) => op.op_name(),
            Op::Matmul(op) => op.op_name(),
            Op::MoeRoute(op) => op.op_name(),
            Op::MoeFfn(op) => op.op_name(),
            Op::StateWriteKv(op) => op.op_name(),
            Op::Attention(op) => op.op_name(),
            Op::CausalConv1d(op) => op.op_name(),
            Op::LinearAttnScan(op) => op.op_name(),
            Op::LogitsPostprocess(op) => op.op_name(),
            Op::Sample(op) => op.op_name(),
            Op::Verify(op) => op.op_name(),
            Op::AllReduce(op) => op.op_name(),
            Op::AllGather(op) => op.op_name(),
            Op::ReduceScatter(op) => op.op_name(),
            Op::AllToAll(op) => op.op_name(),
            Op::Send(op) => op.op_name(),
            Op::Recv(op) => op.op_name(),
            Op::Barrier(op) => op.op_name(),
        }
    }
}
