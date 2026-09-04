// SPDX-License-Identifier: Apache-2.0
//! Core registry types, closed-set enums, and op-static parameter descriptors (Spec 4 §2, §3, §7).

use std::fmt;
use std::num::ParseIntError;
use std::ops::Deref;
use std::path::PathBuf;

use r9v_ir::{
    ActivationKind, AttentionMask, CacheScaleGranularity, ConvActivation, CopyKind, DType, Dim,
    Epilogue, HashId, LayoutId, LinearAttnKind, MlaAttentionSpec, MlaLatent, MoeGroup, MoeScoring,
    NgramCombine, NgramSource, NormAxis, NormKind, P2pTransport, Placement, QuantScheme, ReduceOp,
    RngAlgorithm, RopeScaling, RopeStyle, Smoothing, VerifyMethod,
};
use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------------
// Execution Tier (Spec 4 §2)
// -----------------------------------------------------------------------------

/// Execution tier of an op or kernel implementation (Spec 4 §2).
///
/// Closed four-tier set per Spec 4 §2 and spec-map.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Scalar CPU reference implementation (oracle, Spec 4 §2).
    T0,
    /// Vectorized SIMD CPU reference implementation (Spec 4 §2).
    T0v,
    /// Portable reference HIP implementation (Spec 4 §2).
    T1,
    /// Generated architecture-specialized HIP fast path (Spec 4 §2).
    T2,
}

impl Tier {
    /// Returns the canonical string identifier for this tier (Spec 4 §2).
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::T0 => "t0",
            Self::T0v => "t0v",
            Self::T1 => "t1",
            Self::T2 => "t2",
        }
    }

    /// Parses a tier from a case-insensitive string.
    pub fn parse_tier(s: &str) -> Option<Self> {
        use std::str::FromStr;
        Self::from_str(s).ok()
    }
}

impl std::str::FromStr for Tier {
    type Err = crate::error::RegistryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("t0") {
            Ok(Self::T0)
        } else if s.eq_ignore_ascii_case("t0v") {
            Ok(Self::T0v)
        } else if s.eq_ignore_ascii_case("t1") {
            Ok(Self::T1)
        } else if s.eq_ignore_ascii_case("t2") {
            Ok(Self::T2)
        } else {
            Err(crate::error::RegistryError::ValidationFailed {
                problems: vec![format!(
                    "unknown tier '{s}', expected one of: t0, t0v, t1, t2"
                )],
            })
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// -----------------------------------------------------------------------------
// Operation Identifiers (Spec 1 §4, Spec 4 §3)
// -----------------------------------------------------------------------------

/// Closed set of operations supported by R9V (Spec 1 §4, Spec 4 §3).
///
/// Exhaustive matching required per CONVENTIONS.md §3.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpId {
    /// Token embedding lookup (Spec 1 §4.A).
    EmbedGather,
    /// N-gram prefix gather (Spec 1 §4.A).
    NgramGather,
    /// Activation quantization (Spec 1 §4.A).
    QuantAct,
    /// Precision casting (Spec 1 §4.A).
    Cast,
    /// Memory copy and contiguization (Spec 1 §4.A).
    Copy,
    /// Row gather (Spec 1 §4.A).
    GatherRows,
    /// Deterministic scatter-add rows (Spec 1 §4.A).
    ScatterAddRows,
    /// Last-axis channel split (card A1.14, SI-29).
    Split,
    /// Last-axis channel concatenation (card A1.14, SI-29).
    Concat,
    /// Normalization: RMS or Layer norm (Spec 1 §4.B).
    Norm,
    /// Residual addition (Spec 1 §4.B).
    ResidualAdd,
    /// Gated activation multiplication (Spec 1 §4.B).
    ActMul,
    /// Standalone activation (Spec 1 §4.B).
    Activation,
    /// Final-logit softcap (card A1.14, SI-28).
    LogitSoftcap,
    /// Rotary Position Embedding (Spec 1 §4.B).
    Rope,
    /// Matrix multiplication with epilogue (Spec 1 §4.C).
    Matmul,
    /// Mixture of Experts routing (Spec 1 §4.C).
    MoeRoute,
    /// Mixture of Experts feed-forward (Spec 1 §4.C).
    MoeFfn,
    /// KV state cache write (Spec 1 §4.D).
    StateWriteKv,
    /// Paged / latent attention (Spec 1 §4.D).
    Attention,
    /// 1D Causal Convolution (Spec 1 §4.E).
    CausalConv1d,
    /// Linear attention scan (Spec 1 §4.E).
    LinearAttnScan,
    /// Logits postprocessing (Spec 1 §4.F).
    LogitsPostprocess,
    /// Stochastic or greedy sampling (Spec 1 §4.F).
    Sample,
    /// Speculative verification (Spec 1 §4.F).
    Verify,
    /// All-reduce collective (Spec 1 §4.G).
    AllReduce,
    /// All-gather collective (Spec 1 §4.G).
    AllGather,
    /// Reduce-scatter collective (Spec 1 §4.G).
    ReduceScatter,
    /// All-to-all collective (Spec 1 §4.G).
    AllToAll,
    /// Point-to-point send (Spec 1 §4.G).
    Send,
    /// Point-to-point receive (Spec 1 §4.G).
    Recv,
    /// Device barrier (Spec 1 §4.G).
    Barrier,
}

impl OpId {
    /// Returns the canonical op name string (Spec 1 §4, CONVENTIONS.md §3.2).
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::EmbedGather => "embed_gather",
            Self::NgramGather => "ngram_gather",
            Self::QuantAct => "quant_act",
            Self::Cast => "cast",
            Self::Copy => "copy",
            Self::GatherRows => "gather_rows",
            Self::ScatterAddRows => "scatter_add_rows",
            Self::Split => "split",
            Self::Concat => "concat",
            Self::Norm => "norm",
            Self::ResidualAdd => "residual_add",
            Self::ActMul => "act_mul",
            Self::Activation => "activation",
            Self::LogitSoftcap => "logit_softcap",
            Self::Rope => "rope",
            Self::Matmul => "matmul",
            Self::MoeRoute => "moe_route",
            Self::MoeFfn => "moe_ffn",
            Self::StateWriteKv => "state_write_kv",
            Self::Attention => "attention",
            Self::CausalConv1d => "causal_conv1d",
            Self::LinearAttnScan => "linear_attn_scan",
            Self::LogitsPostprocess => "logits_postprocess",
            Self::Sample => "sample",
            Self::Verify => "verify",
            Self::AllReduce => "all_reduce",
            Self::AllGather => "all_gather",
            Self::ReduceScatter => "reduce_scatter",
            Self::AllToAll => "all_to_all",
            Self::Send => "send",
            Self::Recv => "recv",
            Self::Barrier => "barrier",
        }
    }

    /// Parses an op identifier from its canonical string name.
    pub fn parse_op(s: &str) -> Option<Self> {
        use std::str::FromStr;
        Self::from_str(s).ok()
    }
}

impl std::str::FromStr for OpId {
    type Err = crate::error::RegistryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "embed_gather" => Ok(Self::EmbedGather),
            "ngram_gather" => Ok(Self::NgramGather),
            "quant_act" => Ok(Self::QuantAct),
            "cast" => Ok(Self::Cast),
            "copy" => Ok(Self::Copy),
            "gather_rows" => Ok(Self::GatherRows),
            "scatter_add_rows" => Ok(Self::ScatterAddRows),
            "split" => Ok(Self::Split),
            "concat" => Ok(Self::Concat),
            "norm" => Ok(Self::Norm),
            "residual_add" => Ok(Self::ResidualAdd),
            "act_mul" => Ok(Self::ActMul),
            "activation" => Ok(Self::Activation),
            "logit_softcap" => Ok(Self::LogitSoftcap),
            "rope" => Ok(Self::Rope),
            "matmul" => Ok(Self::Matmul),
            "moe_route" => Ok(Self::MoeRoute),
            "moe_ffn" => Ok(Self::MoeFfn),
            "state_write_kv" => Ok(Self::StateWriteKv),
            "attention" => Ok(Self::Attention),
            "causal_conv1d" => Ok(Self::CausalConv1d),
            "linear_attn_scan" => Ok(Self::LinearAttnScan),
            "logits_postprocess" => Ok(Self::LogitsPostprocess),
            "sample" => Ok(Self::Sample),
            "verify" => Ok(Self::Verify),
            "all_reduce" => Ok(Self::AllReduce),
            "all_gather" => Ok(Self::AllGather),
            "reduce_scatter" => Ok(Self::ReduceScatter),
            "all_to_all" => Ok(Self::AllToAll),
            "send" => Ok(Self::Send),
            "recv" => Ok(Self::Recv),
            "barrier" => Ok(Self::Barrier),
            other => Err(crate::error::RegistryError::ValidationFailed {
                problems: vec![format!("unknown op '{other}'")],
            }),
        }
    }
}

impl OpId {
    /// Derives the OpId from an `r9v_ir::Op` reference (Spec 1 §4).
    pub fn from_op(op: &r9v_ir::Op) -> Self {
        match op {
            r9v_ir::Op::EmbedGather(_) => Self::EmbedGather,
            r9v_ir::Op::NgramGather(_) => Self::NgramGather,
            r9v_ir::Op::QuantAct(_) => Self::QuantAct,
            r9v_ir::Op::Cast(_) => Self::Cast,
            r9v_ir::Op::Copy(_) => Self::Copy,
            r9v_ir::Op::GatherRows(_) => Self::GatherRows,
            r9v_ir::Op::ScatterAddRows(_) => Self::ScatterAddRows,
            r9v_ir::Op::Split(_) => Self::Split,
            r9v_ir::Op::Concat(_) => Self::Concat,
            r9v_ir::Op::Norm(_) => Self::Norm,
            r9v_ir::Op::ResidualAdd(_) => Self::ResidualAdd,
            r9v_ir::Op::ActMul(_) => Self::ActMul,
            r9v_ir::Op::Activation(_) => Self::Activation,
            r9v_ir::Op::LogitSoftcap(_) => Self::LogitSoftcap,
            r9v_ir::Op::Rope(_) => Self::Rope,
            r9v_ir::Op::Matmul(_) => Self::Matmul,
            r9v_ir::Op::MoeRoute(_) => Self::MoeRoute,
            r9v_ir::Op::MoeFfn(_) => Self::MoeFfn,
            r9v_ir::Op::StateWriteKv(_) => Self::StateWriteKv,
            r9v_ir::Op::Attention(_) => Self::Attention,
            r9v_ir::Op::CausalConv1d(_) => Self::CausalConv1d,
            r9v_ir::Op::LinearAttnScan(_) => Self::LinearAttnScan,
            r9v_ir::Op::LogitsPostprocess(_) => Self::LogitsPostprocess,
            r9v_ir::Op::Sample(_) => Self::Sample,
            r9v_ir::Op::Verify(_) => Self::Verify,
            r9v_ir::Op::AllReduce(_) => Self::AllReduce,
            r9v_ir::Op::AllGather(_) => Self::AllGather,
            r9v_ir::Op::ReduceScatter(_) => Self::ReduceScatter,
            r9v_ir::Op::AllToAll(_) => Self::AllToAll,
            r9v_ir::Op::Send(_) => Self::Send,
            r9v_ir::Op::Recv(_) => Self::Recv,
            r9v_ir::Op::Barrier(_) => Self::Barrier,
        }
    }
}

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// -----------------------------------------------------------------------------
// Architecture Name (Spec 4 §3)
// -----------------------------------------------------------------------------

/// Target GPU or host architecture name identifier (Spec 4 §3, CONVENTIONS.md §3.1).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArchName(String);

impl ArchName {
    /// Creates a new architecture name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the architecture name string view.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Deref for ArchName {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for ArchName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ArchName {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for ArchName {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::str::FromStr for ArchName {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(s.to_owned()))
    }
}

// -----------------------------------------------------------------------------
// Variant Hash (Spec 4 §3)
// -----------------------------------------------------------------------------

/// Opaque handle identifying a specific compiled kernel variant (Spec 4 §3, CONVENTIONS.md §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VariantHash(u64);

impl VariantHash {
    /// Creates a new variant hash from a raw 64-bit integer.
    pub const fn new(hash: u64) -> Self {
        Self(hash)
    }

    /// Returns the underlying raw 64-bit hash.
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// Formats the variant hash as a 16-character lowercase hex string.
    pub fn to_hex(&self) -> String {
        format!("{:016x}", self.0)
    }

    /// Parses a variant hash from a hex string.
    pub fn from_hex(s: &str) -> Result<Self, ParseIntError> {
        let cleaned = s.strip_prefix("0x").unwrap_or(s);
        u64::from_str_radix(cleaned, 16).map(Self)
    }
}

impl std::str::FromStr for VariantHash {
    type Err = ParseIntError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_hex(s)
    }
}

impl fmt::Display for VariantHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

// -----------------------------------------------------------------------------
// Launch Geometry (Spec 4 §7)
// -----------------------------------------------------------------------------

/// Grid and thread block launch dimensions for a kernel launch (Spec 4 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LaunchGeometry {
    /// Grid dimensions (blocks).
    pub grid: [u32; 3],
    /// Block dimensions (threads per block).
    pub block: [u32; 3],
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
}

impl LaunchGeometry {
    /// Constructs a new launch geometry specification (Spec 4 §7).
    pub const fn new(grid: [u32; 3], block: [u32; 3], shared_mem_bytes: u32) -> Self {
        Self {
            grid,
            block,
            shared_mem_bytes,
        }
    }
}

// -----------------------------------------------------------------------------
// Tile Configuration (Spec 4 §3, §4.1)
// -----------------------------------------------------------------------------

/// Autotuned tile and wave configuration for generated kernels (Spec 4 §3, §4.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileConfig {
    /// M-dimension tile size.
    pub block_m: u32,
    /// N-dimension tile size.
    pub block_n: u32,
    /// K-dimension tile size.
    pub block_k: u32,
    /// Wave count along M.
    pub waves_m: u32,
    /// Wave count along N.
    pub waves_n: u32,
    /// Wave count along K.
    pub waves_k: u32,
    /// Split-K partial count.
    pub k_splits: u32,
    /// Local Data Share (LDS) bytes consumed.
    pub lds_bytes: u32,
    /// Vector General-Purpose Registers (VGPRs) per lane.
    pub vgprs: u32,
}

impl TileConfig {
    /// Constructs a basic tile configuration with common dimensions.
    pub fn new(block_m: u32, block_n: u32, block_k: u32) -> Self {
        Self {
            block_m,
            block_n,
            block_k,
            waves_m: 1,
            waves_n: 1,
            waves_k: 1,
            k_splits: 1,
            lds_bytes: 0,
            vgprs: 0,
        }
    }

    // DECISION(A3.1): TileConfig deserialization rejects unknown fields via serde deny_unknown_fields and open extra map is removed; rejected open parameter maps because unvalidated parameters cause silent tuning drift and future extension fields must be added explicitly in A3.8. Spec 4 §3, §6.1.
    /// Validates tile dimensions and wave counts (Spec 4 §3, CONVENTIONS §3.2).
    pub fn validate(&self, problems: &mut Vec<String>, context: &str) {
        if self.block_m == 0 || self.block_n == 0 || self.block_k == 0 {
            problems.push(format!(
                "{context}: tile dimensions must be non-zero, got ({}, {}, {})",
                self.block_m, self.block_n, self.block_k
            ));
        }
        if self.waves_m == 0 || self.waves_n == 0 || self.waves_k == 0 {
            problems.push(format!(
                "{context}: wave counts must be non-zero, got ({}, {}, {})",
                self.waves_m, self.waves_n, self.waves_k
            ));
        }
        if self.k_splits == 0 {
            problems.push(format!("{context}: k_splits must be non-zero"));
        }
    }
}

// -----------------------------------------------------------------------------
// Linear Attention Scan Mode (Spec 4 §3)
// -----------------------------------------------------------------------------

/// Execution mode for linear attention scan kernels (Spec 4 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanMode {
    /// Chunked parallel execution mode (Spec 4 §3).
    Chunked,
    /// Sequential recurrent execution mode (Spec 4 §3).
    Recurrent,
}

// -----------------------------------------------------------------------------
// PlacementKind (Spec 1 §2.3, Spec 4 §3, CONVENTIONS.md §3.2)
// -----------------------------------------------------------------------------

/// Rank-free placement classification for kernel execution (Spec 1 §2.3, Spec 4 §3, Spec 5 §3.4, CONVENTIONS.md §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementKind {
    /// Accelerator device execution (Spec 1 §2.3).
    Device,
    /// Pinned host memory execution (Spec 1 §2.3).
    Host,
    /// Slab-backed hierarchical tiered memory (Spec 1 §2.3, Spec 9 §6).
    Tiered,
}

impl PlacementKind {
    /// Returns the canonical snake_case string for this placement kind.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Host => "host",
            Self::Tiered => "tiered",
        }
    }
}

impl fmt::Display for PlacementKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for PlacementKind {
    type Err = crate::error::RegistryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "device" => Ok(Self::Device),
            "host" => Ok(Self::Host),
            "tiered" => Ok(Self::Tiered),
            other => Err(crate::error::RegistryError::ValidationFailed {
                problems: vec![format!(
                    "unknown placement kind '{other}', expected one of: device, host, tiered"
                )],
            }),
        }
    }
}

impl From<Placement> for PlacementKind {
    fn from(p: Placement) -> Self {
        match p {
            Placement::Device { .. } => Self::Device,
            Placement::Host => Self::Host,
            Placement::Tiered => Self::Tiered,
        }
    }
}

// -----------------------------------------------------------------------------
// SamplingMethod (Spec 1 §4.F, Spec 4 §3, Spec 7 §4, CONVENTIONS.md §3.2)
// -----------------------------------------------------------------------------

/// Origin of a resolved kernel code object artifact (Spec 4 §9.2, §11).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactOrigin {
    /// Shipped in the release bundle directory (Spec 4 §11).
    Shipped,
    /// Local autotune or JIT artifact located relative to a local base directory; paths are deliberately relative-only (Spec 4 §6.2, §9.2).
    Local {
        /// Base directory of the local artifact if loaded from a tune file or directory.
        base_dir: Option<PathBuf>,
    },
}

// -----------------------------------------------------------------------------
// SamplingMethod (Spec 1 §4.F, Spec 4 §3, Spec 7 §4, CONVENTIONS.md §3.2)
// -----------------------------------------------------------------------------

/// Closed verification policy for `verify` kernels with bit-exact floats (Spec 1 §4.F, Spec 7 §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifyMethodStatic {
    /// Standard speculative rejection sampling with probability ratio (Spec 1 §4.F, Spec 7 §4).
    Rejection,
    /// Greedy argmax match verification (Spec 1 §4.F, Spec 7 §4).
    Greedy,
    /// Speculative typical acceptance threshold verification with IEEE-754 bit-preserved epsilon and delta (Spec 7 §4).
    TypicalAcceptance {
        /// Acceptance probability floor epsilon IEEE-754 bits.
        eps_bits: u32,
        /// Entropy scaling factor delta IEEE-754 bits.
        delta_bits: u32,
    },
}

impl VerifyMethodStatic {
    /// Constructs a [`VerifyMethodStatic::TypicalAcceptance`] with bit-exact IEEE-754 floats.
    pub const fn typical(eps: f32, delta: f32) -> Self {
        Self::TypicalAcceptance {
            eps_bits: eps.to_bits(),
            delta_bits: delta.to_bits(),
        }
    }

    /// Returns the epsilon float for typical acceptance.
    pub fn eps(&self) -> Option<f32> {
        match self {
            Self::TypicalAcceptance { eps_bits, .. } => Some(f32::from_bits(*eps_bits)),
            _ => None,
        }
    }

    /// Returns the delta float for typical acceptance.
    pub fn delta(&self) -> Option<f32> {
        match self {
            Self::TypicalAcceptance { delta_bits, .. } => Some(f32::from_bits(*delta_bits)),
            _ => None,
        }
    }

    /// Converts from an `r9v_ir::VerifyMethod`, preserving float bits exactly.
    pub fn from_ir(method: &VerifyMethod) -> Self {
        match method {
            VerifyMethod::Rejection => Self::Rejection,
            VerifyMethod::Greedy => Self::Greedy,
            VerifyMethod::TypicalAcceptance { eps, delta } => Self::typical(*eps, *delta),
        }
    }

    /// Converts to an `r9v_ir::VerifyMethod`.
    pub fn to_ir(&self) -> VerifyMethod {
        match self {
            Self::Rejection => VerifyMethod::Rejection,
            Self::Greedy => VerifyMethod::Greedy,
            Self::TypicalAcceptance {
                eps_bits,
                delta_bits,
            } => VerifyMethod::TypicalAcceptance {
                eps: f32::from_bits(*eps_bits),
                delta: f32::from_bits(*delta_bits),
            },
        }
    }
}

/// Static parameters for `logits_postprocess` kernels (Spec 1 §4.F, Spec 4 §3).
///
/// `LogitsPostprocess` carries no method: sampling hyperparameters travel as
/// per-step `SamplingParams` ABI input, not compile-time statics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LogitsPostprocessStatic {
    /// Sequence count bucket (Spec 1 §3.5, Spec 4 §3).
    pub s_bucket: u32,
    /// Vocabulary size V (exact, Spec 4 §3).
    pub v: u32,
    /// Query tokens bucket (Spec 1 §3.5, Spec 4 §3).
    pub q_bucket: u32,
    /// Whether the optional `history_counts [S, V] u32` input is present:
    /// repetition/presence/frequency penalties emit distinct code when live
    /// (Spec 1 §4.F, Spec 4 §3).
    pub has_history_counts: bool,
    /// Whether the optional `grammar_mask [S, q, V] bool` input is present:
    /// mask application emits distinct code when live (Spec 1 §4.F, Spec 4 §3).
    pub has_grammar_mask: bool,
}

/// Static parameters for `sample` kernels (Spec 1 §4.F, Spec 4 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleStatic {
    /// Sequence count bucket (Spec 1 §3.5, Spec 4 §3).
    pub s_bucket: u32,
    /// Vocabulary size V (exact, Spec 4 §3).
    pub v: u32,
    /// Counter-based PRNG algorithm (Spec 1 §4.F).
    #[serde(with = "crate::serde_helpers::serde_rng_algorithm")]
    pub rng: RngAlgorithm,
}

/// Static parameters for `verify` kernels (Spec 1 §4.F, Spec 7 §4, Spec 4 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerifyStatic {
    /// Sequence count bucket (Spec 1 §3.5, Spec 4 §3).
    pub s_bucket: u32,
    /// Vocabulary size V (exact, Spec 4 §3).
    pub v: u32,
    /// Query tokens bucket (Spec 1 §3.5, Spec 4 §3).
    pub q_bucket: u32,
    /// Verification and acceptance policy (Spec 1 §4.F, Spec 7 §4).
    pub method: VerifyMethodStatic,
    /// Whether tree-based speculative verification is enabled (adds TreeParents/TreeAncestors to Verify ABI, Spec 7 §4).
    pub tree: bool,
    /// Whether the optional `draft_probs [S, k, V] f32` input is present:
    /// the weighted acceptance path emits distinct code from the
    /// deterministic one-hot path (Spec 1 §4.F, Spec 7 §4, Spec 4 §3).
    pub has_draft_probs: bool,
}

impl VerifyStatic {
    /// Constructs typical-acceptance verify statics with bit-exact floats.
    pub const fn typical(
        s_bucket: u32,
        v: u32,
        q_bucket: u32,
        eps: f32,
        delta: f32,
        tree: bool,
        has_draft_probs: bool,
    ) -> Self {
        Self {
            s_bucket,
            v,
            q_bucket,
            method: VerifyMethodStatic::typical(eps, delta),
            tree,
            has_draft_probs,
        }
    }
}

// -----------------------------------------------------------------------------
// OpStatic per family (Spec 4 §3)
// -----------------------------------------------------------------------------

/// Static parameters for GEMM / GEMV `matmul` variants (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatmulStatic {
    /// M-dimension bucket (Spec 1 §3.5, Spec 4 §3).
    pub m_bucket: u32,
    /// Output column count (exact, Spec 4 §3).
    pub n: u32,
    /// Inner reduction dimension (exact, Spec 4 §3).
    pub k: u32,
    /// Weight element data type: QuantScheme alone is ambiguous across i4/i8/e4m3 (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub w_dtype: DType,
    /// Weight quantization scheme (Spec 2 §3, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub w_scheme: QuantScheme,
    /// Weight tensor layout (Spec 2 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub w_layout: LayoutId,
    /// Activation input element data type: QuantScheme alone is ambiguous across f16/bf16 (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Activation quantization scheme (Spec 1 §2.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub act_scheme: QuantScheme,
    /// Output data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Fused epilogue operation (Spec 1 §4.C, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_epilogue")]
    pub epilogue: Epilogue,
    /// Residual epilogue input `[M, N]` element data type: `Some` with the
    /// residual tensor dtype (f16/bf16/f32, independent of `out_dtype`,
    /// selecting distinct residual loads) exactly when the epilogue is
    /// `Residual`, `None` otherwise. Presence is already the epilogue, so a
    /// bare presence flag would duplicate it; `validate` and `from_op` reject
    /// a `Some` without a `Residual` epilogue and vice versa
    /// (Spec 1 §4.C, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_opt_dtype")]
    pub residual_dtype: Option<DType>,
    /// Whether weight matrix w is stored transposed [K, N] (Spec 1 §4.C, Spec 4 §3).
    pub transpose_w: bool,
    /// Interleaving mode enabled (Spec 4 §3).
    pub interleave: bool,
    /// SWMMAC 2:4 structured sparsity enabled (Spec 4 §3).
    pub sparse: bool,
}

/// Static parameters for Mixture of Experts routing variants (Spec 4 §3, Spec 1 §4.C).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoeRouteStatic {
    /// Token step bucket T (Spec 1 §3.5, Spec 4 §3).
    pub t_bucket: u32,
    /// Total expert count E (Spec 1 §4.C, Spec 4 §3).
    pub e_total: u32,
    /// Number of experts selected per token K (Spec 1 §4.C, Spec 4 §3).
    pub top_k: u32,
    /// Router scoring method (Spec 1 §4.C, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_moe_scoring")]
    pub scoring: MoeScoring,
    /// Whether router weights are renormalized to sum to 1 (Spec 1 §4.C).
    pub renormalize: bool,
    /// Grouped expert routing configuration, if applicable (Spec 1 §4.C).
    #[serde(with = "crate::serde_helpers::serde_opt_moe_group")]
    pub group: Option<MoeGroup>,
    /// Router scale factor encoded as IEEE-754 bit-pattern for determinism (Spec 1 §4.C, Spec 4 §3).
    pub scale_bits: u32,
    /// Whether the optional router bias tensor `[E] f32` is present (Spec 1 §4.C).
    pub has_bias: bool,
}

impl MoeRouteStatic {
    /// Returns the floating-point scale factor.
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }

    /// Sets the scale factor from a floating point scalar.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale_bits = scale.to_bits();
    }
}

/// Closed per-projection expert weight descriptor for `moe_ffn` (Spec 1 §4.C, Spec 4 §3).
///
/// Each projection carries its own element dtype, quantization scheme, and
/// physical layout as one fixed struct: gate/up and down weights are
/// independent tensors and independent kernel semantics. A variable-length
/// list would admit lengths other than two and alias distinct variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoeFfnProjStatic {
    /// Expert weight element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Expert weight quantization scheme (Spec 2 §3, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub scheme: QuantScheme,
    /// Expert weight tensor layout (Spec 2 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub layout: LayoutId,
}

/// Static parameters for Mixture of Experts feed-forward variants (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoeFfnStatic {
    /// Total token step bucket T = T_dec + T_pre (Spec 1 §3.5, Spec 4 §3).
    pub t_bucket: u32,
    /// Local expert count on this rank (Spec 4 §3).
    pub e_local: u32,
    /// Top-K selected experts per token (Spec 1 §4.C, Spec 4 §3).
    pub k_topk: u32,
    /// Hidden dimension Dm (exact, Spec 4 §3).
    pub dm: u32,
    /// Intermediate projection dimension Dff (exact, Spec 4 §3).
    pub dff: u32,
    /// Gate/up projection `[E, 2·Dff, Dm]` weight descriptor (Spec 1 §4.C, Spec 4 §3).
    pub gate_up: MoeFfnProjStatic,
    /// Down projection `[E, Dm, Dff]` weight descriptor (Spec 1 §4.C, Spec 4 §3).
    pub down: MoeFfnProjStatic,
    /// Input activation element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Activation quantization scheme (Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub act_scheme: QuantScheme,
    /// Expert MLP activation function (Spec 1 §4.C, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_activation_kind")]
    pub act: ActivationKind,
    /// Destination activation dtype (Spec 1 §4.C, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Number of shared experts executed concurrently (Spec 1 §4.C, Spec 4 §3).
    pub shared_experts: u32,
    /// Execution placement class (Spec 1 §2.3, Spec 4 §3, Spec 5 §3.4).
    pub placement_kind: PlacementKind,
}

/// Closed full MLA descriptor preserving every `MlaAttentionSpec` field (Spec 1 §4.D, Spec 8 §3, Spec 4 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlaAttentionStatic {
    /// Query low-rank compression rank, if present (Spec 1 §4.D).
    pub q_lora_rank: Option<u32>,
    /// KV low-rank compression rank (Spec 1 §4.D).
    pub kv_lora_rank: u32,
    /// Non-rotary head dimension for Q/K (Spec 1 §4.D).
    pub qk_nope_dim: u32,
    /// Rotary head dimension for decoupled Q/K (Spec 1 §4.D).
    pub qk_rope_dim: u32,
    /// Value head dimension (Spec 1 §4.D).
    pub v_dim: u32,
}

impl MlaAttentionStatic {
    /// Converts from an `r9v_ir::MlaAttentionSpec`, copying every field.
    pub const fn from_ir(spec: &MlaAttentionSpec) -> Self {
        Self {
            q_lora_rank: spec.q_lora_rank,
            kv_lora_rank: spec.kv_lora_rank,
            qk_nope_dim: spec.qk_nope_dim,
            qk_rope_dim: spec.qk_rope_dim,
            v_dim: spec.v_dim,
        }
    }

    /// Converts to an `r9v_ir::MlaAttentionSpec`.
    pub const fn to_ir(&self) -> MlaAttentionSpec {
        MlaAttentionSpec {
            q_lora_rank: self.q_lora_rank,
            kv_lora_rank: self.kv_lora_rank,
            qk_nope_dim: self.qk_nope_dim,
            qk_rope_dim: self.qk_rope_dim,
            v_dim: self.v_dim,
        }
    }
}

/// Closed MLA latent descriptor preserving every `MlaLatent` field (Spec 1 §4.D, Spec 4 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlaLatentStatic {
    /// Low-rank KV compression rank (Spec 1 §4.D).
    pub kv_lora_rank: u32,
    /// Rotary embedding dimension for decoupled key (Spec 1 §4.D).
    pub rope_dim: u32,
}

impl MlaLatentStatic {
    /// Converts from an `r9v_ir::MlaLatent`, copying every field.
    pub const fn from_ir(latent: &MlaLatent) -> Self {
        Self {
            kv_lora_rank: latent.kv_lora_rank,
            rope_dim: latent.rope_dim,
        }
    }

    /// Converts to an `r9v_ir::MlaLatent`.
    pub const fn to_ir(&self) -> MlaLatent {
        MlaLatent {
            kv_lora_rank: self.kv_lora_rank,
            rope_dim: self.rope_dim,
        }
    }
}

/// Static parameters for attention variants (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttentionStatic {
    /// Query token bucket (Spec 1 §3.5, Spec 4 §3).
    pub q_bucket: u32,
    /// Local query heads on this rank (Spec 4 §3).
    pub h_local: u32,
    /// Local key-value heads on this rank (Spec 4 §3).
    pub hkv_local: u32,
    /// Query head dimension (exact, Spec 4 §3).
    pub d: u32,
    /// Value head dimension (exact, Spec 4 §3).
    pub dv: u32,
    /// Query input element data type: f16/bf16/f32 select distinct Q fragment
    /// loads in decode and prefill kernels (Spec 1 §4.D, Spec 4 §3, §5.3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub q_dtype: DType,
    /// KV cache data type (Spec 3 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub cache_dtype: DType,
    /// Memory layout of the KV cache (Spec 3 §3.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub attention_layout: LayoutId,
    /// Causal or windowed attention mask (Spec 1 §4.D, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_attention_mask")]
    pub mask_kind: AttentionMask,
    /// Softmax scale factor encoded as IEEE-754 bit-pattern for determinism (Spec 1 §4.D, Spec 4 §3).
    pub softmax_scale_bits: u32,
    /// Output activation dtype (Spec 1 §4.D, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Full MLA descriptor when latent attention applies; `None` for paged attention (Spec 1 §4.D, Spec 8 §3, Spec 4 §3).
    pub mla: Option<MlaAttentionStatic>,
    /// Softcapping constant encoded as IEEE-754 bit-pattern for determinism (Spec 4 §3).
    pub softcap_bits: Option<u32>,
    /// Retention sink token count, exactly `0` when no sinks are used (Spec 3 §2, Spec 4 §3).
    pub sinks: u32,
}

impl AttentionStatic {
    /// Returns the floating-point softcap value if configured.
    pub fn softcap_f32(&self) -> Option<f32> {
        self.softcap_bits.map(f32::from_bits)
    }

    /// Sets the softcap value from a floating point scalar.
    pub fn set_softcap(&mut self, softcap: Option<f32>) {
        self.softcap_bits = softcap.map(|v| v.to_bits());
    }

    /// Returns the floating-point softmax scale factor.
    pub fn softmax_scale(&self) -> f32 {
        f32::from_bits(self.softmax_scale_bits)
    }

    /// Sets the softmax scale factor from a floating point scalar.
    pub fn set_softmax_scale(&mut self, scale: f32) {
        self.softmax_scale_bits = scale.to_bits();
    }
}

/// Static parameters for KV cache write variants (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateWriteKvStatic {
    /// Local key-value heads on this rank (Spec 4 §3).
    pub hkv_local: u32,
    /// Key head dimension (exact, Spec 4 §3).
    pub d: u32,
    /// Value head dimension (exact, Spec 4 §3).
    pub dv: u32,
    /// Input key/value projection element data type (Spec 1 §4.D, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Cache storage data type (Spec 3 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub cache_dtype: DType,
    /// Cache scale quantization granularity (Spec 1 §4.D, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_cache_scale_granularity")]
    pub scale_granularity: CacheScaleGranularity,
    /// Target attention cache layout (Spec 3 §3.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub attention_layout: LayoutId,
    /// Full MLA latent descriptor when compressed latent caching applies; `None` for paged caching (Spec 1 §4.D, Spec 4 §3).
    pub latent: Option<MlaLatentStatic>,
}

/// Static parameters for 1D causal convolution variants (Spec 4 §3, Spec 1 §4.E).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CausalConv1dStatic {
    /// Full-step token bucket T = T_dec + T_pre (Spec 1 §3.5, Spec 4 §3).
    pub t_bucket: u32,
    /// Channel count C (exact, Spec 1 §4.E, Spec 4 §3).
    pub channels: u32,
    /// Convolution kernel length W_k (exact, Spec 1 §4.E, Spec 4 §3).
    pub kernel: u32,
    /// Post-convolution activation function (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_conv_activation")]
    pub act: ConvActivation,
    /// Input sequence element data type (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub x_dtype: DType,
    /// Convolution weight element data type (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub w_dtype: DType,
    /// Convolution weight quantization scheme: distinct schemes select
    /// distinct dequantization in the kernel (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub w_scheme: QuantScheme,
    /// Convolution weight physical layout: distinct layouts select distinct
    /// fragment loads in the kernel (Spec 2 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub w_layout: LayoutId,
    /// Output sequence element data type (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Optional bias `[C]` element data type: `None` when the bias input is
    /// absent, `Some` with the bias tensor dtype otherwise. The bias dtype is
    /// independent of the input dtype (Spec 1 §4.E) and selects distinct bias
    /// loads, so presence and dtype are one closed field, never a bare flag
    /// plus an unconstrained dtype.
    #[serde(with = "crate::serde_helpers::serde_opt_dtype")]
    pub bias_dtype: Option<DType>,
}

/// Static parameters for linear attention scan variants (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinearAttnScanStatic {
    /// Architecture-specific linear attention algorithm kind (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_linear_attn_kind")]
    pub kind: LinearAttnKind,
    /// Local heads on this rank (Spec 4 §3).
    pub h_local: u32,
    /// Key dimension (exact, Spec 4 §3).
    pub d: u32,
    /// Value dimension (exact, Spec 4 §3).
    pub dv: u32,
    /// Chunk size for parallel scan (Spec 4 §3).
    pub chunk: u32,
    /// Chunked vs recurrent scan mode (Spec 4 §3).
    pub mode: ScanMode,
    /// Input query/key/value activation element data type (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Output activation element data type (Spec 1 §4.E, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
}

/// Closed deterministic enum of RoPE frequency scaling modes with bit-exact float fields (Spec 1 §4.B, Spec 4 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RopeScalingStatic {
    /// No context window scaling.
    None,
    /// Linear position interpolation by a constant factor.
    Linear {
        /// Linear scale factor IEEE-754 bits.
        factor_bits: u32,
    },
    /// YaRN context extension scaling.
    Yarn {
        /// YaRN interpolation factor IEEE-754 bits.
        factor_bits: u32,
        /// Fast beta frequency cutoff IEEE-754 bits.
        beta_fast_bits: u32,
        /// Slow beta frequency cutoff IEEE-754 bits.
        beta_slow_bits: u32,
        /// Original model training context limit.
        orig_ctx: u32,
        /// Target scale factor IEEE-754 bits.
        mscale_bits: u32,
    },
    /// Dynamic runtime NTK-aware frequency scaling.
    Dynamic,
}

impl RopeScalingStatic {
    /// Constructs a `Linear` scaling with bit-exact IEEE-754 float.
    pub const fn linear(factor: f32) -> Self {
        Self::Linear {
            factor_bits: factor.to_bits(),
        }
    }

    /// Constructs a `Yarn` scaling with bit-exact IEEE-754 floats.
    pub const fn yarn(
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        orig_ctx: u32,
        mscale: f32,
    ) -> Self {
        Self::Yarn {
            factor_bits: factor.to_bits(),
            beta_fast_bits: beta_fast.to_bits(),
            beta_slow_bits: beta_slow.to_bits(),
            orig_ctx,
            mscale_bits: mscale.to_bits(),
        }
    }

    /// Converts from an `r9v_ir::RopeScaling`.
    pub fn from_ir(scaling: &RopeScaling) -> Self {
        match scaling {
            RopeScaling::None => Self::None,
            RopeScaling::Linear(f) => Self::linear(*f),
            RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                orig_ctx,
                mscale,
            } => Self::yarn(*factor, *beta_fast, *beta_slow, *orig_ctx, *mscale),
            RopeScaling::Dynamic => Self::Dynamic,
        }
    }

    /// Converts to an `r9v_ir::RopeScaling`.
    pub fn to_ir(&self) -> RopeScaling {
        match self {
            Self::None => RopeScaling::None,
            Self::Linear { factor_bits } => RopeScaling::Linear(f32::from_bits(*factor_bits)),
            Self::Yarn {
                factor_bits,
                beta_fast_bits,
                beta_slow_bits,
                orig_ctx,
                mscale_bits,
            } => RopeScaling::Yarn {
                factor: f32::from_bits(*factor_bits),
                beta_fast: f32::from_bits(*beta_fast_bits),
                beta_slow: f32::from_bits(*beta_slow_bits),
                orig_ctx: *orig_ctx,
                mscale: f32::from_bits(*mscale_bits),
            },
            Self::Dynamic => RopeScaling::Dynamic,
        }
    }
}

/// Static parameters for token embedding lookup (Spec 1 §4.A, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbedGatherStatic {
    /// Scaling applied to gathered embeddings IEEE-754 bits (e.g. sqrt(Dm)).
    pub scale_bits: u32,
    /// Placement class of the embedding table (Spec 1 §2.3, Spec 4 §3).
    pub table_placement: PlacementKind,
    /// Embedding table element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub table_dtype: DType,
    /// Embedding table quantization scheme (Spec 1 §2.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub table_scheme: QuantScheme,
    /// Embedding table layout (Spec 2 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub table_layout: LayoutId,
    /// Destination activation dtype.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Vocabulary size V (exact).
    pub vocab_size: u32,
    /// Embedding dimension Dm (exact).
    pub dim: u32,
}

impl EmbedGatherStatic {
    /// Returns the floating-point scale factor.
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }

    /// Sets the scale factor from a floating point scalar.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale_bits = scale.to_bits();
    }
}

/// Static parameters for n-gram prefix gather (Spec 1 §4.A, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NgramGatherStatic {
    /// N-gram source mode: Staged buffer vs Device table.
    #[serde(with = "crate::serde_helpers::serde_ngram_source")]
    pub source: NgramSource,
    /// Hash function identifier.
    #[serde(with = "crate::serde_helpers::serde_hash_id")]
    pub hash: HashId,
    /// N-gram orders evaluated by the hash heads; length must equal `heads`.
    pub orders: Vec<u32>,
    /// Number of parallel n-gram hash heads Np.
    pub heads: u32,
    /// Size of the n-gram table per head; length must equal `heads`.
    pub table_sizes: Vec<u32>,
    /// Embedding dimension per head Dn (exact, Spec 4 §3).
    pub dn: u32,
    /// Device table element data type, used in Device-table mode (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub table_dtype: DType,
    /// Device table quantization scheme, used in Device-table mode (Spec 1 §2.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub table_scheme: QuantScheme,
    /// Device table layout, used in Device-table mode (Spec 2 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub table_layout: LayoutId,
    /// Staging buffer element data type, used in Staged mode (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub staging_dtype: DType,
    /// Staging buffer quantization scheme, used in Staged mode: the IR admits
    /// any spec 2 §3 scheme, and distinct schemes select distinct dequantization
    /// (Spec 1 §4.A, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub staging_scheme: QuantScheme,
    /// Staging buffer physical layout, used in Staged mode: distinct layouts
    /// select distinct fragment loads (Spec 2 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub staging_layout: LayoutId,
    /// Row-scales element data type, used in Staged mode only: `Some` with the
    /// `row_scales` tensor dtype (f32 or f16, selecting distinct scale loads)
    /// in Staged mode, `None` in Device-table mode where no scales tensor
    /// exists (Spec 1 §4.A, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_opt_dtype")]
    pub scales_dtype: Option<DType>,
    /// How gathered head embeddings are combined.
    #[serde(with = "crate::serde_helpers::serde_ngram_combine")]
    pub combine: NgramCombine,
    /// Destination activation dtype.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
}

/// Static parameters for activation quantization (Spec 1 §4.A, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuantActStatic {
    /// Activation quantization scheme.
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub scheme: QuantScheme,
    /// Input activation element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Target quantized element dtype (i8 or e4m3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub target: DType,
    /// Weight smoothing mode.
    #[serde(with = "crate::serde_helpers::serde_smoothing")]
    pub smoothing: Smoothing,
    /// Feature width N (exact, Spec 4 §3).
    pub n: u32,
}

/// Static parameters for precision casting (Spec 1 §4.A, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CastStatic {
    /// Input operand data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Destination output data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Row width N (exact, Spec 4 §3).
    pub n: u32,
}

/// Static parameters for memory copy and contiguization (Spec 1 §4.A, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CopyStatic {
    /// Transfer boundary kind.
    #[serde(with = "crate::serde_helpers::serde_copy_kind")]
    pub kind: CopyKind,
    /// Element data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Row width N (exact, Spec 4 §3).
    pub n: u32,
}

/// Static parameters for row gather (Spec 1 §4.A, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GatherRowsStatic {
    /// Table element data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Index operand data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub index_dtype: DType,
    /// Row width D (exact, Spec 4 §3).
    pub width: u32,
}

/// Static parameters for scatter-add rows (Spec 1 §4.A, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScatterAddRowsStatic {
    /// Accumulator element data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Index operand data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub index_dtype: DType,
    /// Row width D (exact, Spec 4 §3).
    pub width: u32,
    /// Whether the optional `dest [N, D]` base tensor input is present: the
    /// out-of-place form reads a distinct base tensor while the two-input form
    /// accumulates into the output, so presence selects distinct pointer
    /// interpretation (Spec 1 §4.A, SI-10, Spec 4 §3).
    pub has_dest: bool,
}

/// Static parameters for channel split (Spec 1 §4.A, card A1.14, SI-29).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SplitStatic {
    /// Width of the first output along the last axis, copied exactly from IR `SplitOp.first` (Spec 1 §4.A).
    pub first: u32,
    /// Total input channel width C along the last axis, with `0 < first < C` (exact resolved fact, Spec 4 §3).
    pub total: u32,
    /// Element data type preserved across the split (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
}

/// Static parameters for channel concatenation (Spec 1 §4.A, card A1.14, SI-29).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConcatStatic {
    /// First input channel width C0 along the last axis (exact, Spec 4 §3).
    pub c0: u32,
    /// Second input channel width C1 along the last axis (exact, Spec 4 §3).
    pub c1: u32,
    /// First input element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub a_dtype: DType,
    /// Second input element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub b_dtype: DType,
    /// Destination activation data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
}

/// Static parameters for normalization (Spec 1 §4.B, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NormStatic {
    /// RMS or Layer norm variant.
    #[serde(with = "crate::serde_helpers::serde_norm_kind")]
    pub kind: NormKind,
    /// Epsilon variance floor IEEE-754 bits.
    pub eps_bits: u32,
    /// Reduction axis.
    #[serde(with = "crate::serde_helpers::serde_norm_axis")]
    pub axis: NormAxis,
    /// Weight offset IEEE-754 bits.
    pub weight_offset_bits: u32,
    /// Input activation element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Output activation data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Feature width N (exact, Spec 4 §3).
    pub n: u32,
    /// Whether the optional bias `[N] f32` input is present: a biased norm
    /// kernel emits a distinct add, so presence is a variant semantic
    /// (Spec 1 §4.B, Spec 4 §3).
    pub has_bias: bool,
}

impl NormStatic {
    /// Returns the epsilon floor float value.
    pub fn eps(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }

    /// Sets the epsilon floor float value.
    pub fn set_eps(&mut self, eps: f32) {
        self.eps_bits = eps.to_bits();
    }

    /// Returns the weight offset float value.
    pub fn weight_offset(&self) -> f32 {
        f32::from_bits(self.weight_offset_bits)
    }

    /// Sets the weight offset float value.
    pub fn set_weight_offset(&mut self, offset: f32) {
        self.weight_offset_bits = offset.to_bits();
    }
}

/// Static parameters for residual addition (Spec 1 §4.B, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResidualAddStatic {
    /// First addend element data type: the IR admits f16/bf16/f32 per input
    /// independently, and distinct input dtypes select distinct loads
    /// (Spec 1 §4.B, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub a_dtype: DType,
    /// Second addend element data type, independent of `a_dtype` (Spec 1 §4.B, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub b_dtype: DType,
    /// Output activation data type (Spec 1 §4.B, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Residual branch scale encoded as IEEE-754 bit-pattern for determinism (Spec 1 §4.B, Spec 4 §3).
    pub scale_bits: u32,
    /// Feature width N (exact, Spec 4 §3).
    pub n: u32,
}

impl ResidualAddStatic {
    /// Returns the residual branch scale float value.
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }

    /// Sets the residual branch scale float value.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale_bits = scale.to_bits();
    }
}

/// Static parameters for gated activation multiplication (Spec 1 §4.B, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActMulStatic {
    /// Activation function applied to the gate tensor.
    #[serde(with = "crate::serde_helpers::serde_activation_kind")]
    pub act: ActivationKind,
    /// Optional upper clamp limit IEEE-754 bits.
    pub clamp_bits: Option<u32>,
    /// Gate/up activation element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Feature width Dff (exact, Spec 4 §3).
    pub width: u32,
}

impl ActMulStatic {
    /// Returns the clamp limit float if configured.
    pub fn clamp(&self) -> Option<f32> {
        self.clamp_bits.map(f32::from_bits)
    }

    /// Sets the clamp limit float.
    pub fn set_clamp(&mut self, clamp: Option<f32>) {
        self.clamp_bits = clamp.map(|c| c.to_bits());
    }
}

/// Static parameters for standalone activation (Spec 1 §4.B, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActivationStatic {
    /// Activation function kind.
    #[serde(with = "crate::serde_helpers::serde_activation_kind")]
    pub act: ActivationKind,
    /// Optional upper clamp limit IEEE-754 bits.
    pub clamp_bits: Option<u32>,
    /// Input activation element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Feature width Dff (exact, Spec 4 §3).
    pub width: u32,
}

impl ActivationStatic {
    /// Returns the clamp limit float if configured.
    pub fn clamp(&self) -> Option<f32> {
        self.clamp_bits.map(f32::from_bits)
    }

    /// Sets the clamp limit float.
    pub fn set_clamp(&mut self, clamp: Option<f32>) {
        self.clamp_bits = clamp.map(|c| c.to_bits());
    }
}

/// Static parameters for final-logit softcap (card A1.14, SI-28).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LogitSoftcapStatic {
    /// Softcap limit constant IEEE-754 bits.
    pub cap_bits: u32,
    /// Vocabulary width V (exact, Spec 4 §3).
    pub v: u32,
}

impl LogitSoftcapStatic {
    /// Returns the softcap limit float.
    pub fn cap(&self) -> f32 {
        f32::from_bits(self.cap_bits)
    }

    /// Sets the softcap limit float.
    pub fn set_cap(&mut self, cap: f32) {
        self.cap_bits = cap.to_bits();
    }
}

/// Static parameters for Rotary Position Embedding (Spec 1 §4.B, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RopeStatic {
    /// Dimension to which rotary embedding is applied.
    pub rot_dim: u32,
    /// Base theta frequency IEEE-754 bits.
    pub theta_bits: u32,
    /// Interleaved or NeoX style.
    #[serde(with = "crate::serde_helpers::serde_rope_style")]
    pub style: RopeStyle,
    /// Frequency scaling configuration.
    pub scaling: RopeScalingStatic,
    /// Multimodal RoPE section dimensions [T, H, W], if applicable.
    pub mrope_sections: Option<[u32; 3]>,
    /// Input activation element data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub in_dtype: DType,
    /// Destination activation data type.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Head count H (exact, Spec 4 §3).
    pub h: u32,
    /// Head dimension D (exact, Spec 4 §3).
    pub d: u32,
}

impl RopeStatic {
    /// Returns the base theta float.
    pub fn theta(&self) -> f32 {
        f32::from_bits(self.theta_bits)
    }

    /// Sets the base theta float.
    pub fn set_theta(&mut self, theta: f32) {
        self.theta_bits = theta.to_bits();
    }
}

/// Typed, closed parameter descriptor for each elementwise operation variant (Spec 1 §4, Spec 4 §3).
// DECISION(A3.API): the internal tag is "op" rather than "kind" because NormStatic and
// CopyStatic already carry a "kind" field and an internally-tagged enum rejects duplicates;
// rejected tag "kind" (duplicate-field deserialization failure). Spec 4 §3.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ElementwiseParams {
    /// Token embedding lookup.
    EmbedGather(EmbedGatherStatic),
    /// N-gram prefix gather.
    NgramGather(NgramGatherStatic),
    /// Activation quantization.
    QuantAct(QuantActStatic),
    /// Precision casting.
    Cast(CastStatic),
    /// Memory copy and contiguization.
    Copy(CopyStatic),
    /// Row gather.
    GatherRows(GatherRowsStatic),
    /// Scatter-add rows.
    ScatterAddRows(ScatterAddRowsStatic),
    /// Channel split.
    Split(SplitStatic),
    /// Channel concatenation.
    Concat(ConcatStatic),
    /// Normalization.
    Norm(NormStatic),
    /// Residual addition.
    ResidualAdd(ResidualAddStatic),
    /// Gated activation multiplication.
    ActMul(ActMulStatic),
    /// Standalone activation.
    Activation(ActivationStatic),
    /// Final-logit softcap.
    LogitSoftcap(LogitSoftcapStatic),
    /// Rotary Position Embedding.
    Rope(RopeStatic),
}

impl ElementwiseParams {
    /// Returns the corresponding `OpId` for this elementwise parameter descriptor.
    pub const fn op_id(&self) -> OpId {
        match self {
            Self::EmbedGather(_) => OpId::EmbedGather,
            Self::NgramGather(_) => OpId::NgramGather,
            Self::QuantAct(_) => OpId::QuantAct,
            Self::Cast(_) => OpId::Cast,
            Self::Copy(_) => OpId::Copy,
            Self::GatherRows(_) => OpId::GatherRows,
            Self::ScatterAddRows(_) => OpId::ScatterAddRows,
            Self::Split(_) => OpId::Split,
            Self::Concat(_) => OpId::Concat,
            Self::Norm(_) => OpId::Norm,
            Self::ResidualAdd(_) => OpId::ResidualAdd,
            Self::ActMul(_) => OpId::ActMul,
            Self::Activation(_) => OpId::Activation,
            Self::LogitSoftcap(_) => OpId::LogitSoftcap,
            Self::Rope(_) => OpId::Rope,
        }
    }
}

/// Static parameters for memory-bound elementwise variants (Spec 4 §3).
///
/// Covers: `norm`, `rope`, `act_mul`, `quant_act`, `residual_add`, `embed_gather`, `ngram_gather`,
/// `cast`, `copy`, `gather_rows`, `scatter_add_rows`, `split`, `concat`, `activation`, `logit_softcap`.
///
/// Shape and dtype facts live as exact fields on the nested [`ElementwiseParams`]
/// descriptor; there is no generic dims/dtypes bag that could silently omit behavior.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElementwiseStatic {
    /// Full step token bucket T = T_dec + T_pre (Spec 1 §3.5, Spec 4 §3).
    pub t_bucket: u32,
    /// Optional fused successor op identifier (Spec 1 §3.4, Spec 4 §3, CONVENTIONS.md §3.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_with: Option<OpId>,
    /// Closed typed parameters specific to the elementwise op (Spec 4 §3).
    pub op_params: ElementwiseParams,
}

/// Closed per-op static descriptor for sampling-family operations (Spec 1 §4.F, Spec 7 §4, Spec 4 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SamplingStatic {
    /// Logits postprocessing; carries no method (Spec 1 §4.F).
    LogitsPostprocess(LogitsPostprocessStatic),
    /// Stochastic or greedy sampling with an explicit PRNG algorithm (Spec 1 §4.F).
    Sample(SampleStatic),
    /// Speculative verification with an explicit method plus tree flag (Spec 1 §4.F, Spec 7 §4).
    Verify(VerifyStatic),
}

impl SamplingStatic {
    /// Returns the exact `OpId` for this sampling descriptor.
    pub const fn op_id(&self) -> OpId {
        match self {
            Self::LogitsPostprocess(_) => OpId::LogitsPostprocess,
            Self::Sample(_) => OpId::Sample,
            Self::Verify(_) => OpId::Verify,
        }
    }
}

/// Static parameters for `all_reduce` collectives (Spec 1 §4.G, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllReduceStatic {
    /// Communication group identifier (Spec 1 §4.G, Spec 4 §3).
    pub group: u64,
    /// Local communicator rank (Spec 1 §4.G, Spec 4 §3).
    pub rank: u32,
    /// Communicator world size (Spec 1 §4.G, Spec 4 §3).
    pub world: u32,
    /// Communication element data type (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Internal accumulator dtype, always f32 per Spec 1 §4.G.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub reduce_in: DType,
    /// Reduction operator (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_reduce_op")]
    pub reduction_op: ReduceOp,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
    /// Transferred payload size bucket in bytes (Spec 4 §3).
    pub bytes_bucket: u64,
}

/// Static parameters for `all_gather` collectives (Spec 1 §4.G, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllGatherStatic {
    /// Communication group identifier (Spec 1 §4.G, Spec 4 §3).
    pub group: u64,
    /// Local communicator rank (Spec 1 §4.G, Spec 4 §3).
    pub rank: u32,
    /// Communicator world size (Spec 1 §4.G, Spec 4 §3).
    pub world: u32,
    /// Communication element data type (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Axis along which tensors are concatenated (Spec 1 §4.G, Spec 4 §3).
    pub axis: u32,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
    /// Transferred payload size bucket in bytes (Spec 4 §3).
    pub bytes_bucket: u64,
}

/// Static parameters for `reduce_scatter` collectives (Spec 1 §4.G, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReduceScatterStatic {
    /// Communication group identifier (Spec 1 §4.G, Spec 4 §3).
    pub group: u64,
    /// Local communicator rank (Spec 1 §4.G, Spec 4 §3).
    pub rank: u32,
    /// Communicator world size (Spec 1 §4.G, Spec 4 §3).
    pub world: u32,
    /// Communication element data type (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Internal accumulator dtype, always f32 per Spec 1 §4.G.
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub reduce_in: DType,
    /// Reduction operator (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_reduce_op")]
    pub reduction_op: ReduceOp,
    /// Axis along which reduction partitions are distributed (Spec 1 §4.G, Spec 4 §3).
    pub axis: u32,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
    /// Transferred payload size bucket in bytes (Spec 4 §3).
    pub bytes_bucket: u64,
}

/// Static parameters for `all_to_all` collectives (Spec 1 §4.G, Spec 4 §3).
///
/// Per-rank counts travel as the dynamic `counts [P] u32` ABI pointer/scalar,
/// never as compile-time statics (Spec 1 §4.G, SI-11).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllToAllStatic {
    /// Communication group identifier (Spec 1 §4.G, Spec 4 §3).
    pub group: u64,
    /// Local communicator rank (Spec 1 §4.G, Spec 4 §3).
    pub rank: u32,
    /// Communicator world size (Spec 1 §4.G, Spec 4 §3).
    pub world: u32,
    /// Communication element data type (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
    /// Transferred payload size bucket in bytes (Spec 4 §3).
    pub bytes_bucket: u64,
}

/// Static parameters for `send` point-to-point operations (Spec 1 §4.G, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SendStatic {
    /// Communication group identifier (Spec 1 §4.G, Spec 4 §3).
    pub group: u64,
    /// Local communicator rank (Spec 1 §4.G, Spec 4 §3).
    pub rank: u32,
    /// Communicator world size (Spec 1 §4.G, Spec 4 §3).
    pub world: u32,
    /// Destination peer device rank (Spec 1 §4.G, Spec 4 §3).
    pub peer: u32,
    /// Transferred element dtype (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
    /// Transferred payload size bucket in bytes (Spec 4 §3).
    pub bytes_bucket: u64,
}

/// Static parameters for `recv` point-to-point operations (Spec 1 §4.G, Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecvStatic {
    /// Communication group identifier (Spec 1 §4.G, Spec 4 §3).
    pub group: u64,
    /// Local communicator rank (Spec 1 §4.G, Spec 4 §3).
    pub rank: u32,
    /// Communicator world size (Spec 1 §4.G, Spec 4 §3).
    pub world: u32,
    /// Source peer device rank (Spec 1 §4.G, Spec 4 §3).
    pub peer: u32,
    /// Expected received tensor shape in deterministic order; non-empty with all extents > 0 (Spec 1 §4.G, Spec 4 §3).
    pub shape: Vec<u32>,
    /// Received element dtype (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
    /// Transferred payload size bucket in bytes (Spec 4 §3).
    pub bytes_bucket: u64,
}

/// Static parameters for `barrier` operations (Spec 1 §4.G, Spec 4 §3).
///
/// Barriers carry no dtype, payload bucket, reduction, axis, peer, or shape:
/// those fields are meaningless for synchronization and are unrepresentable here.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BarrierStatic {
    /// Synchronized communication group identifier (Spec 1 §4.G, Spec 4 §3).
    pub group: u64,
    /// Local communicator rank (Spec 1 §4.G, Spec 4 §3).
    pub rank: u32,
    /// Communicator world size (Spec 1 §4.G, Spec 4 §3).
    pub world: u32,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
}

/// Closed per-op static descriptor for collective-family operations (Spec 1 §4.G, Spec 4 §3).
///
/// Each variant carries exactly the fields its op admits: no meaningless
/// peer/reduction fields, no optional-field soup.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum CollectivesStatic {
    /// All-reduce collective.
    AllReduce(AllReduceStatic),
    /// All-gather collective.
    AllGather(AllGatherStatic),
    /// Reduce-scatter collective.
    ReduceScatter(ReduceScatterStatic),
    /// All-to-all collective.
    AllToAll(AllToAllStatic),
    /// Point-to-point send.
    Send(SendStatic),
    /// Point-to-point receive with exact deterministic shape.
    Recv(RecvStatic),
    /// Device barrier.
    Barrier(BarrierStatic),
}

impl CollectivesStatic {
    /// Returns the exact `OpId` for this collective descriptor.
    pub const fn op_id(&self) -> OpId {
        match self {
            Self::AllReduce(_) => OpId::AllReduce,
            Self::AllGather(_) => OpId::AllGather,
            Self::ReduceScatter(_) => OpId::ReduceScatter,
            Self::AllToAll(_) => OpId::AllToAll,
            Self::Send(_) => OpId::Send,
            Self::Recv(_) => OpId::Recv,
            Self::Barrier(_) => OpId::Barrier,
        }
    }
}

/// Closed enum of static parameter descriptors per op family (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum OpStatic {
    /// Matrix multiplication variants (Spec 4 §3, §5.1, §5.2).
    Matmul(MatmulStatic),
    /// Mixture of Experts routing variants (Spec 4 §3, §5.6).
    MoeRoute(MoeRouteStatic),
    /// Mixture of Experts feed-forward variants (Spec 4 §3, §5.6).
    MoeFfn(MoeFfnStatic),
    /// Paged and latent attention variants (Spec 4 §3, §5.3).
    Attention(AttentionStatic),
    /// KV state cache write variants (Spec 4 §3, §5.4).
    StateWriteKv(StateWriteKvStatic),
    /// 1D causal convolution state variants (Spec 4 §3, §5.5).
    CausalConv1d(CausalConv1dStatic),
    /// Linear attention scan variants (Spec 4 §3, §5.5).
    LinearAttnScan(LinearAttnScanStatic),
    /// Memory-bound elementwise operations (Spec 4 §3, §5.7).
    Elementwise(ElementwiseStatic),
    /// Sampling and speculative verification operations (Spec 4 §3, §5.8).
    Sampling(SamplingStatic),
    /// Inter-device collective communication operations (Spec 4 §3, §5.9).
    Collectives(CollectivesStatic),
}

impl OpStatic {
    /// Returns the canonical snake_case family name for this static descriptor.
    pub const fn family_name(&self) -> &'static str {
        match self {
            Self::Matmul(_) => "matmul",
            Self::MoeRoute(_) => "moe_route",
            Self::MoeFfn(_) => "moe_ffn",
            Self::Attention(_) => "attention",
            Self::StateWriteKv(_) => "state_write_kv",
            Self::CausalConv1d(_) => "causal_conv1d",
            Self::LinearAttnScan(_) => "linear_attn_scan",
            Self::Elementwise(_) => "elementwise",
            Self::Sampling(_) => "sampling",
            Self::Collectives(_) => "collectives",
        }
    }

    /// Returns the exact `OpId` this static descriptor was built for (Spec 4 §3).
    ///
    /// Total: shared families resolve through their closed nested descriptor,
    /// so two different compile-time kernel semantics never share an identity.
    pub const fn op_id(&self) -> OpId {
        match self {
            Self::Matmul(_) => OpId::Matmul,
            Self::MoeRoute(_) => OpId::MoeRoute,
            Self::MoeFfn(_) => OpId::MoeFfn,
            Self::Attention(_) => OpId::Attention,
            Self::StateWriteKv(_) => OpId::StateWriteKv,
            Self::CausalConv1d(_) => OpId::CausalConv1d,
            Self::LinearAttnScan(_) => OpId::LinearAttnScan,
            Self::Elementwise(s) => s.op_params.op_id(),
            Self::Sampling(s) => s.op_id(),
            Self::Collectives(s) => s.op_id(),
        }
    }

    /// Enforces exact `OpId`-to-nested-descriptor agreement (Spec 4 §3).
    ///
    /// Cross-family and within-family mismatches are typed errors, never panics.
    pub fn check_pair(&self, op: OpId) -> Result<(), crate::error::RegistryError> {
        let mine = self.op_id();
        if mine == op {
            Ok(())
        } else {
            Err(crate::error::RegistryError::StaticOpMismatch {
                op,
                static_op: mine,
            })
        }
    }

    /// Validates closed-shape invariants of this static descriptor (Spec 4 §3).
    ///
    /// Collects every problem before returning (CONVENTIONS.md §1.4).
    pub fn validate(&self) -> Result<(), crate::error::RegistryError> {
        let mut problems = Vec::new();
        match self {
            Self::Matmul(s) => {
                if s.m_bucket == 0 {
                    problems.push("matmul.m_bucket must be > 0".to_owned());
                }
                if s.n == 0 {
                    problems.push("matmul.n must be > 0".to_owned());
                }
                if s.k == 0 {
                    problems.push("matmul.k must be > 0".to_owned());
                }
                let wants_residual = matches!(s.epilogue, Epilogue::Residual);
                if s.residual_dtype.is_some() != wants_residual {
                    problems.push(
                        "matmul.residual_dtype must be Some exactly when the epilogue is Residual"
                            .to_owned(),
                    );
                }
                if let Some(d) = s.residual_dtype {
                    if !matches!(d, DType::F16 | DType::Bf16 | DType::F32) {
                        problems.push(format!(
                            "matmul.residual_dtype must be f16, bf16, or f32, got {d:?}"
                        ));
                    }
                }
            }
            Self::MoeRoute(s) => {
                if s.t_bucket == 0 {
                    problems.push("moe_route.t_bucket must be > 0".to_owned());
                }
                if s.e_total == 0 {
                    problems.push("moe_route.e_total must be > 0".to_owned());
                }
                if s.top_k == 0 {
                    problems.push("moe_route.top_k must be > 0".to_owned());
                }
                if s.top_k > s.e_total {
                    problems.push(format!(
                        "moe_route.top_k ({}) must not exceed e_total ({})",
                        s.top_k, s.e_total
                    ));
                }
                if let Some(g) = s.group {
                    if g.n_group == 0 {
                        problems.push("moe_route.group.n_group must be > 0".to_owned());
                    }
                    if g.topk_group == 0 {
                        problems.push("moe_route.group.topk_group must be > 0".to_owned());
                    }
                    if g.topk_group > s.top_k {
                        problems.push(format!(
                            "moe_route.group.topk_group ({}) must not exceed top_k ({})",
                            g.topk_group, s.top_k
                        ));
                    }
                }
            }
            Self::MoeFfn(s) => {
                if s.t_bucket == 0 {
                    problems.push("moe_ffn.t_bucket must be > 0".to_owned());
                }
                if s.e_local == 0 {
                    problems.push("moe_ffn.e_local must be > 0".to_owned());
                }
                if s.k_topk == 0 {
                    problems.push("moe_ffn.k_topk must be > 0".to_owned());
                }
                if s.dm == 0 {
                    problems.push("moe_ffn.dm must be > 0".to_owned());
                }
                if s.dff == 0 {
                    problems.push("moe_ffn.dff must be > 0".to_owned());
                }
                check_moe_proj_problems("gate_up", &s.gate_up, &mut problems);
                check_moe_proj_problems("down", &s.down, &mut problems);
            }
            Self::Attention(s) => {
                if s.q_bucket == 0 {
                    problems.push("attention.q_bucket must be > 0".to_owned());
                }
                if !matches!(s.q_dtype, DType::F16 | DType::Bf16 | DType::F32) {
                    problems.push(format!(
                        "attention.q_dtype must be f16, bf16, or f32, got {:?}",
                        s.q_dtype
                    ));
                }
                if s.h_local == 0 {
                    problems.push("attention.h_local must be > 0".to_owned());
                }
                if s.hkv_local == 0 {
                    problems.push("attention.hkv_local must be > 0".to_owned());
                }
                if s.d == 0 {
                    problems.push("attention.d must be > 0".to_owned());
                }
                if s.dv == 0 {
                    problems.push("attention.dv must be > 0".to_owned());
                }
                if let Some(mla) = s.mla {
                    if mla.kv_lora_rank == 0 {
                        problems.push("attention.mla.kv_lora_rank must be > 0".to_owned());
                    }
                    if mla.qk_nope_dim == 0 {
                        problems.push("attention.mla.qk_nope_dim must be > 0".to_owned());
                    }
                    if mla.qk_rope_dim == 0 {
                        problems.push("attention.mla.qk_rope_dim must be > 0".to_owned());
                    }
                    if mla.v_dim == 0 {
                        problems.push("attention.mla.v_dim must be > 0".to_owned());
                    }
                    if let Some(q) = mla.q_lora_rank {
                        if q == 0 {
                            problems.push(
                                "attention.mla.q_lora_rank must be > 0 when present".to_owned(),
                            );
                        }
                    }
                }
            }
            Self::StateWriteKv(s) => {
                if s.hkv_local == 0 {
                    problems.push("state_write_kv.hkv_local must be > 0".to_owned());
                }
                if s.d == 0 {
                    problems.push("state_write_kv.d must be > 0".to_owned());
                }
                if s.dv == 0 {
                    problems.push("state_write_kv.dv must be > 0".to_owned());
                }
                if let Some(l) = s.latent {
                    if l.kv_lora_rank == 0 {
                        problems.push("state_write_kv.latent.kv_lora_rank must be > 0".to_owned());
                    }
                    if l.rope_dim == 0 {
                        problems.push("state_write_kv.latent.rope_dim must be > 0".to_owned());
                    }
                }
            }
            Self::CausalConv1d(s) => {
                if s.t_bucket == 0 {
                    problems.push("causal_conv1d.t_bucket must be > 0".to_owned());
                }
                if s.channels == 0 {
                    problems.push("causal_conv1d.channels must be > 0".to_owned());
                }
                if s.kernel == 0 {
                    problems.push("causal_conv1d.kernel must be > 0".to_owned());
                }
                // DECISION(A3.API): Spec 1 §4.E is silent on conv weight
                // quantization, so the rule mirrors the IR (see SI-55 for the
                // sanctioned T0 weight-scale/state rule): float weights are
                // unquantized, i8/i4 weights require PerRow or spec 2 block
                // scales (the GEMM weight rule extended to the IR-legal
                // bf16/f32 conv set). Rejected leaving w_scheme unconstrained
                // because distinct schemes select distinct dequantization.
                match s.w_dtype {
                    DType::F16 | DType::Bf16 | DType::F32 => {
                        if s.w_scheme != QuantScheme::None {
                            problems.push(format!(
                                "causal_conv1d.w_scheme must be None for {:?} weights, got {:?}",
                                s.w_dtype, s.w_scheme
                            ));
                        }
                    }
                    DType::I8 | DType::I4 => {
                        if !matches!(s.w_scheme, QuantScheme::PerRow | QuantScheme::Scheme(_)) {
                            problems.push(format!(
                                "causal_conv1d.w_scheme must be PerRow or a spec 2 block scheme for {:?} weights, got {:?}",
                                s.w_dtype, s.w_scheme
                            ));
                        }
                    }
                    other => {
                        problems.push(format!(
                            "causal_conv1d.w_dtype must be f16, bf16, f32, i8, or i4, got {other:?}"
                        ));
                    }
                }
                if let Some(d) = s.bias_dtype {
                    if !matches!(d, DType::F16 | DType::Bf16 | DType::F32) {
                        problems.push(format!(
                            "causal_conv1d.bias_dtype must be f16, bf16, or f32, got {d:?}"
                        ));
                    }
                }
            }
            Self::LinearAttnScan(s) => {
                if s.h_local == 0 {
                    problems.push("linear_attn_scan.h_local must be > 0".to_owned());
                }
                if s.d == 0 {
                    problems.push("linear_attn_scan.d must be > 0".to_owned());
                }
                if s.dv == 0 {
                    problems.push("linear_attn_scan.dv must be > 0".to_owned());
                }
                if s.chunk == 0 {
                    problems.push("linear_attn_scan.chunk must be > 0".to_owned());
                }
            }
            Self::Elementwise(s) => {
                if s.t_bucket == 0 {
                    problems.push("elementwise.t_bucket must be > 0".to_owned());
                }
                match &s.op_params {
                    ElementwiseParams::NgramGather(p) => {
                        // DECISION(A3.API): Spec 1 §4.A names the staging
                        // quant only as "Block" and is silent on its physical
                        // layout, so the rule mirrors the IR (see SI-8 for the
                        // Staged/Device signatures and SI-53 for the sanctioned
                        // carrier/hash rules): Staged mode requires a spec 2
                        // block scheme and f32/f16 scales, Device mode carries
                        // no scales. Rejected leaving the Staged dequant inputs
                        // unconstrained because distinct schemes select distinct
                        // dequantization.
                        if p.source == NgramSource::Staged {
                            if !matches!(p.staging_scheme, QuantScheme::Scheme(_)) {
                                problems.push(format!(
                                    "ngram_gather.staging_scheme must be a spec 2 block scheme in Staged mode, got {:?}",
                                    p.staging_scheme
                                ));
                            }
                            match p.scales_dtype {
                                Some(DType::F32 | DType::F16) => {}
                                other => {
                                    problems.push(format!(
                                        "ngram_gather.scales_dtype must be Some(f32|f16) in Staged mode, got {other:?}"
                                    ));
                                }
                            }
                        } else if p.scales_dtype.is_some() {
                            problems.push(
                                "ngram_gather.scales_dtype must be None in Device-table mode"
                                    .to_owned(),
                            );
                        }
                        if p.heads == 0 {
                            problems.push("ngram_gather.heads must be > 0".to_owned());
                        }
                        if p.orders.len() != p.heads as usize {
                            problems.push(format!(
                                "ngram_gather.orders len ({}) must equal heads ({})",
                                p.orders.len(),
                                p.heads
                            ));
                        }
                        if p.table_sizes.len() != p.heads as usize {
                            problems.push(format!(
                                "ngram_gather.table_sizes len ({}) must equal heads ({})",
                                p.table_sizes.len(),
                                p.heads
                            ));
                        }
                        if p.dn == 0 {
                            problems.push("ngram_gather.dn must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::Split(p) => {
                        if p.first == 0 {
                            problems.push("split.first must be > 0".to_owned());
                        }
                        if p.total == 0 {
                            problems.push("split.total must be > 0".to_owned());
                        }
                        if p.first >= p.total {
                            problems.push(format!(
                                "split requires 0 < first ({}) < total ({})",
                                p.first, p.total
                            ));
                        }
                    }
                    ElementwiseParams::Concat(p) => {
                        if p.c0 == 0 {
                            problems.push("concat.c0 must be > 0".to_owned());
                        }
                        if p.c1 == 0 {
                            problems.push("concat.c1 must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::QuantAct(p) => {
                        if p.n == 0 {
                            problems.push("quant_act.n must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::Cast(p) => {
                        if p.n == 0 {
                            problems.push("cast.n must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::Copy(p) => {
                        if p.n == 0 {
                            problems.push("copy.n must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::GatherRows(p) => {
                        if p.width == 0 {
                            problems.push("gather_rows.width must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::ScatterAddRows(p) => {
                        if p.width == 0 {
                            problems.push("scatter_add_rows.width must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::Norm(p) => {
                        if p.n == 0 {
                            problems.push("norm.n must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::ResidualAdd(p) => {
                        if p.n == 0 {
                            problems.push("residual_add.n must be > 0".to_owned());
                        }
                        for (name, d) in [("a_dtype", p.a_dtype), ("b_dtype", p.b_dtype)] {
                            if !matches!(d, DType::F16 | DType::Bf16 | DType::F32) {
                                problems.push(format!(
                                    "residual_add.{name} must be f16, bf16, or f32, got {d:?}"
                                ));
                            }
                        }
                    }
                    ElementwiseParams::ActMul(p) => {
                        if p.width == 0 {
                            problems.push("act_mul.width must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::Activation(p) => {
                        if p.width == 0 {
                            problems.push("activation.width must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::LogitSoftcap(p) => {
                        if p.v == 0 {
                            problems.push("logit_softcap.v must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::Rope(p) => {
                        if p.h == 0 {
                            problems.push("rope.h must be > 0".to_owned());
                        }
                        if p.d == 0 {
                            problems.push("rope.d must be > 0".to_owned());
                        }
                        if p.rot_dim == 0 {
                            problems.push("rope.rot_dim must be > 0".to_owned());
                        }
                    }
                    ElementwiseParams::EmbedGather(p) => {
                        if p.vocab_size == 0 {
                            problems.push("embed_gather.vocab_size must be > 0".to_owned());
                        }
                        if p.dim == 0 {
                            problems.push("embed_gather.dim must be > 0".to_owned());
                        }
                    }
                }
            }
            Self::Sampling(s) => match s {
                SamplingStatic::LogitsPostprocess(p) => {
                    if p.s_bucket == 0 {
                        problems.push("logits_postprocess.s_bucket must be > 0".to_owned());
                    }
                    if p.v == 0 {
                        problems.push("logits_postprocess.v must be > 0".to_owned());
                    }
                    if p.q_bucket == 0 {
                        problems.push("logits_postprocess.q_bucket must be > 0".to_owned());
                    }
                }
                SamplingStatic::Sample(p) => {
                    if p.s_bucket == 0 {
                        problems.push("sample.s_bucket must be > 0".to_owned());
                    }
                    if p.v == 0 {
                        problems.push("sample.v must be > 0".to_owned());
                    }
                }
                SamplingStatic::Verify(p) => {
                    if p.s_bucket == 0 {
                        problems.push("verify.s_bucket must be > 0".to_owned());
                    }
                    if p.v == 0 {
                        problems.push("verify.v must be > 0".to_owned());
                    }
                    if p.q_bucket == 0 {
                        problems.push("verify.q_bucket must be > 0".to_owned());
                    }
                }
            },
            Self::Collectives(s) => match s {
                CollectivesStatic::AllReduce(p) => {
                    if p.world == 0 {
                        problems.push("all_reduce.world must be > 0".to_owned());
                    }
                    if p.rank >= p.world {
                        problems.push(format!(
                            "all_reduce.rank ({}) must be < world ({})",
                            p.rank, p.world
                        ));
                    }
                    if p.reduce_in != DType::F32 {
                        problems.push(format!(
                            "all_reduce.reduce_in must be f32 per Spec 1 §4.G, got {:?}",
                            p.reduce_in
                        ));
                    }
                    if p.bytes_bucket == 0 {
                        problems.push("all_reduce.bytes_bucket must be > 0".to_owned());
                    }
                }
                CollectivesStatic::AllGather(p) => {
                    if p.world == 0 {
                        problems.push("all_gather.world must be > 0".to_owned());
                    }
                    if p.rank >= p.world {
                        problems.push(format!(
                            "all_gather.rank ({}) must be < world ({})",
                            p.rank, p.world
                        ));
                    }
                    if p.bytes_bucket == 0 {
                        problems.push("all_gather.bytes_bucket must be > 0".to_owned());
                    }
                }
                CollectivesStatic::ReduceScatter(p) => {
                    if p.world == 0 {
                        problems.push("reduce_scatter.world must be > 0".to_owned());
                    }
                    if p.rank >= p.world {
                        problems.push(format!(
                            "reduce_scatter.rank ({}) must be < world ({})",
                            p.rank, p.world
                        ));
                    }
                    if p.reduce_in != DType::F32 {
                        problems.push(format!(
                            "reduce_scatter.reduce_in must be f32 per Spec 1 §4.G, got {:?}",
                            p.reduce_in
                        ));
                    }
                    if p.bytes_bucket == 0 {
                        problems.push("reduce_scatter.bytes_bucket must be > 0".to_owned());
                    }
                }
                CollectivesStatic::AllToAll(p) => {
                    if p.world == 0 {
                        problems.push("all_to_all.world must be > 0".to_owned());
                    }
                    if p.rank >= p.world {
                        problems.push(format!(
                            "all_to_all.rank ({}) must be < world ({})",
                            p.rank, p.world
                        ));
                    }
                    if p.bytes_bucket == 0 {
                        problems.push("all_to_all.bytes_bucket must be > 0".to_owned());
                    }
                }
                CollectivesStatic::Send(p) => {
                    if p.world == 0 {
                        problems.push("send.world must be > 0".to_owned());
                    }
                    if p.rank >= p.world {
                        problems.push(format!(
                            "send.rank ({}) must be < world ({})",
                            p.rank, p.world
                        ));
                    }
                    if p.peer >= p.world {
                        problems.push(format!(
                            "send.peer ({}) must be < world ({})",
                            p.peer, p.world
                        ));
                    }
                    if p.bytes_bucket == 0 {
                        problems.push("send.bytes_bucket must be > 0".to_owned());
                    }
                }
                CollectivesStatic::Recv(p) => {
                    if p.world == 0 {
                        problems.push("recv.world must be > 0".to_owned());
                    }
                    if p.rank >= p.world {
                        problems.push(format!(
                            "recv.rank ({}) must be < world ({})",
                            p.rank, p.world
                        ));
                    }
                    if p.peer >= p.world {
                        problems.push(format!(
                            "recv.peer ({}) must be < world ({})",
                            p.peer, p.world
                        ));
                    }
                    if p.shape.is_empty() {
                        problems.push("recv.shape must be non-empty".to_owned());
                    }
                    if p.shape.contains(&0) {
                        problems.push("recv.shape extents must all be > 0".to_owned());
                    }
                    if p.bytes_bucket == 0 {
                        problems.push("recv.bytes_bucket must be > 0".to_owned());
                    }
                }
                CollectivesStatic::Barrier(p) => {
                    if p.world == 0 {
                        problems.push("barrier.world must be > 0".to_owned());
                    }
                    if p.rank >= p.world {
                        problems.push(format!(
                            "barrier.rank ({}) must be < world ({})",
                            p.rank, p.world
                        ));
                    }
                }
            },
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(crate::error::RegistryError::ValidationFailed { problems })
        }
    }
}

/// Closed resolved kernel facts for `matmul` lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulFacts {
    /// M-dimension bucket (Spec 1 §3.5).
    pub m_bucket: u32,
    /// Output column count N (exact).
    pub n: u32,
    /// Inner reduction dimension K (exact).
    pub k: u32,
    /// Weight element data type from the weight tensor.
    pub w_dtype: DType,
    /// Weight quantization scheme from the weight tensor.
    pub w_scheme: QuantScheme,
    /// Weight tensor layout from the weight tensor.
    pub w_layout: LayoutId,
    /// Activation input element data type from the activation tensor.
    pub in_dtype: DType,
    /// Activation quantization scheme from the activation tensor.
    pub act_scheme: QuantScheme,
    /// Residual epilogue input dtype from the residual tensor, `Some` exactly
    /// when the IR epilogue is `Residual` (Spec 1 §4.C).
    pub residual_dtype: Option<DType>,
    /// Interleaved weight layout mode (resolved kernel fact).
    pub interleave: bool,
    /// SWMMAC 2:4 structured sparsity mode (resolved kernel fact).
    pub sparse: bool,
}

/// Closed resolved kernel facts for `moe_route` lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeRouteFacts {
    /// Token step bucket T.
    pub t_bucket: u32,
    /// Total expert count E from the logits tensor trailing dim.
    pub e_total: u32,
    /// Whether the optional router bias tensor `[E] f32` is present.
    pub has_bias: bool,
}

/// Closed resolved kernel facts for `moe_ffn` lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeFfnFacts {
    /// Total token step bucket T.
    pub t_bucket: u32,
    /// Local expert count on this rank.
    pub e_local: u32,
    /// Top-K selected experts per token from the routing tensors.
    pub k_topk: u32,
    /// Hidden dimension Dm (exact).
    pub dm: u32,
    /// Intermediate projection dimension Dff (exact).
    pub dff: u32,
    /// Gate/up projection weight descriptor from the `w_gate_up` tensor.
    pub gate_up: MoeFfnProjStatic,
    /// Down projection weight descriptor from the `w_down` tensor.
    pub down: MoeFfnProjStatic,
    /// Input activation element data type.
    pub in_dtype: DType,
    /// Activation quantization scheme.
    pub act_scheme: QuantScheme,
    /// Execution placement class.
    pub placement_kind: PlacementKind,
}

/// Closed resolved kernel facts for `attention` lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionFacts {
    /// Query token bucket.
    pub q_bucket: u32,
    /// Local query heads on this rank.
    pub h_local: u32,
    /// Local key-value heads on this rank.
    pub hkv_local: u32,
    /// Query head dimension (exact).
    pub d: u32,
    /// Value head dimension (exact).
    pub dv: u32,
    /// Query input element data type from the `q` tensor.
    pub q_dtype: DType,
    /// KV cache data type.
    pub cache_dtype: DType,
    /// Memory layout of the KV cache.
    pub attention_layout: LayoutId,
}

/// Closed resolved kernel facts for `state_write_kv` lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateWriteKvFacts {
    /// Local key-value heads on this rank.
    pub hkv_local: u32,
    /// Key head dimension (exact).
    pub d: u32,
    /// Value head dimension (exact).
    pub dv: u32,
    /// Input key/value projection element data type.
    pub in_dtype: DType,
    /// Target attention cache layout.
    pub attention_layout: LayoutId,
}

/// Closed resolved kernel facts for `causal_conv1d` lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalConv1dFacts {
    /// Full-step token bucket T.
    pub t_bucket: u32,
    /// Channel count C (exact).
    pub channels: u32,
    /// Input sequence element data type.
    pub x_dtype: DType,
    /// Convolution weight element data type.
    pub w_dtype: DType,
    /// Convolution weight quantization scheme from the weight tensor.
    pub w_scheme: QuantScheme,
    /// Convolution weight physical layout from the weight tensor.
    pub w_layout: LayoutId,
    /// Output sequence element data type.
    pub out_dtype: DType,
    /// Optional bias `[C]` element data type from the bias tensor, `None`
    /// when the bias input is absent.
    pub bias_dtype: Option<DType>,
}

/// Closed resolved kernel facts for `linear_attn_scan` lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearAttnScanFacts {
    /// Local heads on this rank.
    pub h_local: u32,
    /// Key dimension (exact).
    pub d: u32,
    /// Value dimension (exact).
    pub dv: u32,
    /// Input query/key/value activation element data type.
    pub in_dtype: DType,
    /// Chunked vs recurrent scan mode (resolved execution choice).
    pub mode: ScanMode,
}

/// Closed per-op resolved facts for elementwise lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementwiseFacts {
    /// `embed_gather` table facts; scale/out_dtype come from IR.
    EmbedGather {
        /// Placement class of the embedding table.
        table_placement: PlacementKind,
        /// Embedding table element data type.
        table_dtype: DType,
        /// Embedding table quantization scheme.
        table_scheme: QuantScheme,
        /// Embedding table layout.
        table_layout: LayoutId,
        /// Vocabulary size V (exact).
        vocab_size: u32,
        /// Embedding dimension Dm (exact).
        dim: u32,
    },
    /// `ngram_gather` table facts; source/orders/heads/hash/table_sizes/combine/out_dtype come from IR.
    NgramGather {
        /// Embedding dimension per head Dn (exact).
        dn: u32,
        /// Device table element data type.
        table_dtype: DType,
        /// Device table quantization scheme.
        table_scheme: QuantScheme,
        /// Device table layout.
        table_layout: LayoutId,
        /// Staging buffer element data type.
        staging_dtype: DType,
        /// Staging buffer quantization scheme.
        staging_scheme: QuantScheme,
        /// Staging buffer physical layout.
        staging_layout: LayoutId,
        /// Row-scales element data type (`Some` in Staged mode, `None` in
        /// Device-table mode).
        scales_dtype: Option<DType>,
    },
    /// `quant_act` facts; scheme/target/smoothing come from IR.
    QuantAct {
        /// Input activation element data type.
        in_dtype: DType,
        /// Feature width N (exact).
        n: u32,
    },
    /// `cast` facts; destination dtype comes from IR.
    Cast {
        /// Input operand data type.
        in_dtype: DType,
        /// Row width N (exact).
        n: u32,
    },
    /// `copy` facts; kind comes from IR.
    Copy {
        /// Element data type.
        dtype: DType,
        /// Row width N (exact).
        n: u32,
    },
    /// `gather_rows` facts (IR carries no attributes).
    GatherRows {
        /// Table element data type.
        dtype: DType,
        /// Index operand data type.
        index_dtype: DType,
        /// Row width D (exact).
        width: u32,
    },
    /// `scatter_add_rows` facts (IR carries no attributes).
    ScatterAddRows {
        /// Accumulator element data type.
        dtype: DType,
        /// Index operand data type.
        index_dtype: DType,
        /// Row width D (exact).
        width: u32,
        /// Whether the optional `dest` base tensor input is present.
        has_dest: bool,
    },
    /// `split` facts; `first` comes from IR.
    Split {
        /// Total input channel width C (exact).
        total: u32,
        /// Element data type preserved across the split.
        dtype: DType,
    },
    /// `concat` facts (IR carries no attributes).
    Concat {
        /// First input channel width C0 (exact).
        c0: u32,
        /// Second input channel width C1 (exact).
        c1: u32,
        /// First input element data type.
        a_dtype: DType,
        /// Second input element data type.
        b_dtype: DType,
        /// Destination activation data type.
        out_dtype: DType,
    },
    /// `norm` facts; kind/eps/axis/weight_offset/out_dtype come from IR.
    Norm {
        /// Input activation element data type.
        in_dtype: DType,
        /// Feature width N (exact).
        n: u32,
        /// Whether the optional bias `[N]` input is present.
        has_bias: bool,
    },
    /// `residual_add` facts; out_dtype/scale come from IR.
    ResidualAdd {
        /// First addend element data type from the `a` tensor.
        a_dtype: DType,
        /// Second addend element data type from the `b` tensor.
        b_dtype: DType,
        /// Feature width N (exact).
        n: u32,
    },
    /// `act_mul` facts; act/clamp come from IR.
    ActMul {
        /// Gate/up activation element data type.
        dtype: DType,
        /// Feature width Dff (exact).
        width: u32,
    },
    /// `activation` facts; act/clamp come from IR.
    Activation {
        /// Input activation element data type.
        dtype: DType,
        /// Feature width Dff (exact).
        width: u32,
    },
    /// `logit_softcap` facts; cap comes from IR.
    LogitSoftcap {
        /// Vocabulary width V (exact).
        v: u32,
    },
    /// `rope` facts; rot_dim/theta/style/scaling/mrope_sections/out_dtype come from IR.
    Rope {
        /// Input activation element data type.
        in_dtype: DType,
        /// Head count H (exact).
        h: u32,
        /// Head dimension D (exact).
        d: u32,
    },
}

impl ElementwiseFacts {
    /// Returns the exact `OpId` these facts were built for.
    pub const fn op_id(&self) -> OpId {
        match self {
            Self::EmbedGather { .. } => OpId::EmbedGather,
            Self::NgramGather { .. } => OpId::NgramGather,
            Self::QuantAct { .. } => OpId::QuantAct,
            Self::Cast { .. } => OpId::Cast,
            Self::Copy { .. } => OpId::Copy,
            Self::GatherRows { .. } => OpId::GatherRows,
            Self::ScatterAddRows { .. } => OpId::ScatterAddRows,
            Self::Split { .. } => OpId::Split,
            Self::Concat { .. } => OpId::Concat,
            Self::Norm { .. } => OpId::Norm,
            Self::ResidualAdd { .. } => OpId::ResidualAdd,
            Self::ActMul { .. } => OpId::ActMul,
            Self::Activation { .. } => OpId::Activation,
            Self::LogitSoftcap { .. } => OpId::LogitSoftcap,
            Self::Rope { .. } => OpId::Rope,
        }
    }
}

/// Closed per-op resolved facts for sampling lowering (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingFacts {
    /// `logits_postprocess` facts (IR carries no attributes).
    LogitsPostprocess {
        /// Sequence count bucket.
        s_bucket: u32,
        /// Vocabulary size V (exact).
        v: u32,
        /// Query tokens bucket.
        q_bucket: u32,
        /// Whether the optional `history_counts` input is present.
        has_history_counts: bool,
        /// Whether the optional `grammar_mask` input is present.
        has_grammar_mask: bool,
    },
    /// `sample` facts; rng comes from IR.
    Sample {
        /// Sequence count bucket.
        s_bucket: u32,
        /// Vocabulary size V (exact).
        v: u32,
    },
    /// `verify` facts; method comes from IR.
    Verify {
        /// Sequence count bucket.
        s_bucket: u32,
        /// Vocabulary size V (exact).
        v: u32,
        /// Query tokens bucket.
        q_bucket: u32,
        /// Whether tree-based speculative verification applies.
        tree: bool,
        /// Whether the optional `draft_probs` input is present.
        has_draft_probs: bool,
    },
}

impl SamplingFacts {
    /// Returns the exact `OpId` these facts were built for.
    pub const fn op_id(&self) -> OpId {
        match self {
            Self::LogitsPostprocess { .. } => OpId::LogitsPostprocess,
            Self::Sample { .. } => OpId::Sample,
            Self::Verify { .. } => OpId::Verify,
        }
    }
}

/// Closed per-op resolved facts for collective lowering (Spec 4 §3).
///
/// Group, dtype, reduction, axis, peer, and recv rank-shape come from IR;
/// rank/world membership, transport, and the bytes bucket are resolved facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectiveFacts {
    /// `all_reduce` facts.
    AllReduce {
        /// Local communicator rank.
        rank: u32,
        /// Communicator world size.
        world: u32,
        /// Underlying transport mechanism.
        transport: P2pTransport,
        /// Transferred payload size bucket in bytes.
        bytes_bucket: u64,
    },
    /// `all_gather` facts.
    AllGather {
        /// Local communicator rank.
        rank: u32,
        /// Communicator world size.
        world: u32,
        /// Underlying transport mechanism.
        transport: P2pTransport,
        /// Transferred payload size bucket in bytes.
        bytes_bucket: u64,
    },
    /// `reduce_scatter` facts.
    ReduceScatter {
        /// Local communicator rank.
        rank: u32,
        /// Communicator world size.
        world: u32,
        /// Underlying transport mechanism.
        transport: P2pTransport,
        /// Transferred payload size bucket in bytes.
        bytes_bucket: u64,
    },
    /// `all_to_all` facts.
    AllToAll {
        /// Local communicator rank.
        rank: u32,
        /// Communicator world size.
        world: u32,
        /// Underlying transport mechanism.
        transport: P2pTransport,
        /// Transferred payload size bucket in bytes.
        bytes_bucket: u64,
    },
    /// `send` facts.
    Send {
        /// Local communicator rank.
        rank: u32,
        /// Communicator world size.
        world: u32,
        /// Underlying transport mechanism.
        transport: P2pTransport,
        /// Transferred payload size bucket in bytes.
        bytes_bucket: u64,
    },
    /// `recv` facts.
    Recv {
        /// Local communicator rank.
        rank: u32,
        /// Communicator world size.
        world: u32,
        /// Underlying transport mechanism.
        transport: P2pTransport,
        /// Transferred payload size bucket in bytes.
        bytes_bucket: u64,
        /// Expected received tensor shape, resolved to concrete extents in deterministic order.
        shape: Vec<u32>,
    },
    /// `barrier` facts.
    Barrier {
        /// Local communicator rank.
        rank: u32,
        /// Communicator world size.
        world: u32,
        /// Underlying transport mechanism.
        transport: P2pTransport,
    },
}

impl CollectiveFacts {
    /// Returns the exact `OpId` these facts were built for.
    pub const fn op_id(&self) -> OpId {
        match self {
            Self::AllReduce { .. } => OpId::AllReduce,
            Self::AllGather { .. } => OpId::AllGather,
            Self::ReduceScatter { .. } => OpId::ReduceScatter,
            Self::AllToAll { .. } => OpId::AllToAll,
            Self::Send { .. } => OpId::Send,
            Self::Recv { .. } => OpId::Recv,
            Self::Barrier { .. } => OpId::Barrier,
        }
    }
}

/// Closed resolved kernel facts for lowering an `r9v_ir::Op` into an [`OpStatic`] (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticFacts {
    /// `matmul` resolved facts.
    Matmul(MatmulFacts),
    /// `moe_route` resolved facts.
    MoeRoute(MoeRouteFacts),
    /// `moe_ffn` resolved facts.
    MoeFfn(MoeFfnFacts),
    /// `attention` resolved facts.
    Attention(AttentionFacts),
    /// `state_write_kv` resolved facts.
    StateWriteKv(StateWriteKvFacts),
    /// `causal_conv1d` resolved facts.
    CausalConv1d(CausalConv1dFacts),
    /// `linear_attn_scan` resolved facts.
    LinearAttnScan(LinearAttnScanFacts),
    /// Elementwise resolved facts plus step bucket and fusion.
    Elementwise {
        /// Full step token bucket T = T_dec + T_pre.
        t_bucket: u32,
        /// Optional fused successor op identifier.
        fused_with: Option<OpId>,
        /// Per-op resolved facts.
        params: ElementwiseFacts,
    },
    /// Sampling resolved facts.
    Sampling(SamplingFacts),
    /// Collective resolved facts.
    Collectives(CollectiveFacts),
}

/// Collects `moe_ffn` per-projection (dtype, scheme) legality problems for one
/// named projection, mirroring the IR GEMM weight rule (`check_gemm_weight_operand`):
/// i4/i8/e4m3 weights require `PerRow` or spec 2 block scales, f16 weights are
/// unquantized, and every other dtype is illegal here (Spec 1 §4.C, Spec 4 §3).
fn check_moe_proj_problems(name: &str, proj: &MoeFfnProjStatic, problems: &mut Vec<String>) {
    match proj.dtype {
        DType::I4 | DType::I8 | DType::E4m3 => {
            if !matches!(proj.scheme, QuantScheme::PerRow | QuantScheme::Scheme(_)) {
                problems.push(format!(
                    "moe_ffn.{name}.scheme must be PerRow or a spec 2 block scheme for {:?} weights, got {:?}",
                    proj.dtype, proj.scheme
                ));
            }
        }
        DType::F16 => {
            if proj.scheme != QuantScheme::None {
                problems.push(format!(
                    "moe_ffn.{name}.scheme must be None for f16 weights, got {:?}",
                    proj.scheme
                ));
            }
        }
        other => {
            problems.push(format!(
                "moe_ffn.{name}.dtype must be i4, i8, e4m3, or f16, got {other:?}"
            ));
        }
    }
}

fn facts_mismatch(op: OpId, what: &str) -> crate::error::RegistryError {
    crate::error::RegistryError::FactsOpMismatch {
        op,
        detail: what.to_owned(),
    }
}

impl OpStatic {
    /// Total construction seam from an `r9v_ir::Op` plus closed resolved kernel facts (Spec 1 §4, Spec 4 §3).
    ///
    /// Copies every behavior attribute from the IR op; resolved
    /// dimensions, buckets, dtypes, layouts, and placements come from `facts`.
    /// Mismatched facts/op pairs are typed errors. No family-name guesses, no open strings.
    pub fn from_op(
        op: &r9v_ir::Op,
        facts: &StaticFacts,
    ) -> Result<Self, crate::error::RegistryError> {
        match (op, facts) {
            (r9v_ir::Op::Matmul(o), StaticFacts::Matmul(f)) => {
                let wants_residual = matches!(o.epilogue, Epilogue::Residual);
                if f.residual_dtype.is_some() != wants_residual {
                    return Err(facts_mismatch(
                        OpId::Matmul,
                        "matmul facts residual_dtype presence must match a Residual epilogue",
                    ));
                }
                Ok(Self::Matmul(MatmulStatic {
                    m_bucket: f.m_bucket,
                    n: f.n,
                    k: f.k,
                    w_dtype: f.w_dtype,
                    w_scheme: f.w_scheme,
                    w_layout: f.w_layout,
                    in_dtype: f.in_dtype,
                    act_scheme: f.act_scheme,
                    out_dtype: o.out_dtype,
                    epilogue: o.epilogue,
                    residual_dtype: f.residual_dtype,
                    transpose_w: o.transpose_w,
                    interleave: f.interleave,
                    sparse: f.sparse,
                }))
            }
            (r9v_ir::Op::MoeRoute(o), StaticFacts::MoeRoute(f)) => {
                Ok(Self::MoeRoute(MoeRouteStatic {
                    t_bucket: f.t_bucket,
                    e_total: f.e_total,
                    top_k: o.top_k,
                    scoring: o.scoring,
                    renormalize: o.renormalize,
                    group: o.group,
                    scale_bits: o.scale.to_bits(),
                    has_bias: f.has_bias,
                }))
            }
            (r9v_ir::Op::MoeFfn(o), StaticFacts::MoeFfn(f)) => Ok(Self::MoeFfn(MoeFfnStatic {
                t_bucket: f.t_bucket,
                e_local: f.e_local,
                k_topk: f.k_topk,
                dm: f.dm,
                dff: f.dff,
                gate_up: f.gate_up,
                down: f.down,
                in_dtype: f.in_dtype,
                act_scheme: f.act_scheme,
                act: o.act,
                out_dtype: o.out_dtype,
                shared_experts: o.shared_experts,
                placement_kind: f.placement_kind,
            })),
            (r9v_ir::Op::Attention(o), StaticFacts::Attention(f)) => {
                Ok(Self::Attention(AttentionStatic {
                    q_bucket: f.q_bucket,
                    h_local: f.h_local,
                    hkv_local: f.hkv_local,
                    d: f.d,
                    dv: f.dv,
                    q_dtype: f.q_dtype,
                    cache_dtype: f.cache_dtype,
                    attention_layout: f.attention_layout,
                    mask_kind: o.mask,
                    softmax_scale_bits: o.softmax_scale.to_bits(),
                    out_dtype: o.out_dtype,
                    mla: o.mla.as_ref().map(MlaAttentionStatic::from_ir),
                    softcap_bits: o.logit_softcap.map(|v| v.to_bits()),
                    sinks: o.sinks,
                }))
            }
            (r9v_ir::Op::StateWriteKv(o), StaticFacts::StateWriteKv(f)) => {
                Ok(Self::StateWriteKv(StateWriteKvStatic {
                    hkv_local: f.hkv_local,
                    d: f.d,
                    dv: f.dv,
                    in_dtype: f.in_dtype,
                    cache_dtype: o.cache_dtype,
                    scale_granularity: o.scale_granularity,
                    attention_layout: f.attention_layout,
                    latent: o.latent.as_ref().map(MlaLatentStatic::from_ir),
                }))
            }
            (r9v_ir::Op::CausalConv1d(o), StaticFacts::CausalConv1d(f)) => {
                Ok(Self::CausalConv1d(CausalConv1dStatic {
                    t_bucket: f.t_bucket,
                    channels: f.channels,
                    kernel: o.kernel,
                    act: o.act,
                    x_dtype: f.x_dtype,
                    w_dtype: f.w_dtype,
                    w_scheme: f.w_scheme,
                    w_layout: f.w_layout,
                    out_dtype: f.out_dtype,
                    bias_dtype: f.bias_dtype,
                }))
            }
            (r9v_ir::Op::LinearAttnScan(o), StaticFacts::LinearAttnScan(f)) => {
                Ok(Self::LinearAttnScan(LinearAttnScanStatic {
                    kind: o.kind,
                    h_local: f.h_local,
                    d: f.d,
                    dv: f.dv,
                    chunk: o.chunk,
                    mode: f.mode,
                    in_dtype: f.in_dtype,
                    out_dtype: o.out_dtype,
                }))
            }
            (
                r9v_ir::Op::EmbedGather(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params:
                        ElementwiseFacts::EmbedGather {
                            table_placement,
                            table_dtype,
                            table_scheme,
                            table_layout,
                            vocab_size,
                            dim,
                        },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::EmbedGather(EmbedGatherStatic {
                    scale_bits: o.scale.to_bits(),
                    table_placement: *table_placement,
                    table_dtype: *table_dtype,
                    table_scheme: *table_scheme,
                    table_layout: *table_layout,
                    out_dtype: o.out_dtype,
                    vocab_size: *vocab_size,
                    dim: *dim,
                }),
            })),
            (
                r9v_ir::Op::NgramGather(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params:
                        ElementwiseFacts::NgramGather {
                            dn,
                            table_dtype,
                            table_scheme,
                            table_layout,
                            staging_dtype,
                            staging_scheme,
                            staging_layout,
                            scales_dtype,
                        },
                },
            ) => {
                let staged = o.source == NgramSource::Staged;
                if scales_dtype.is_some() != staged {
                    return Err(facts_mismatch(
                        OpId::NgramGather,
                        "ngram_gather facts scales_dtype presence must match Staged source",
                    ));
                }
                Ok(Self::Elementwise(ElementwiseStatic {
                    t_bucket: *t_bucket,
                    fused_with: *fused_with,
                    op_params: ElementwiseParams::NgramGather(NgramGatherStatic {
                        source: o.source,
                        hash: o.hash,
                        orders: o.orders.to_vec(),
                        heads: o.heads,
                        table_sizes: o.table_sizes.to_vec(),
                        dn: *dn,
                        table_dtype: *table_dtype,
                        table_scheme: *table_scheme,
                        table_layout: *table_layout,
                        staging_dtype: *staging_dtype,
                        staging_scheme: *staging_scheme,
                        staging_layout: *staging_layout,
                        scales_dtype: *scales_dtype,
                        combine: o.combine,
                        out_dtype: o.out_dtype,
                    }),
                }))
            }
            (
                r9v_ir::Op::QuantAct(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::QuantAct { in_dtype, n },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::QuantAct(QuantActStatic {
                    scheme: o.scheme,
                    in_dtype: *in_dtype,
                    target: o.target,
                    smoothing: o.smoothing,
                    n: *n,
                }),
            })),
            (
                r9v_ir::Op::Cast(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::Cast { in_dtype, n },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::Cast(CastStatic {
                    in_dtype: *in_dtype,
                    out_dtype: o.dtype,
                    n: *n,
                }),
            })),
            (
                r9v_ir::Op::Copy(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::Copy { dtype, n },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::Copy(CopyStatic {
                    kind: o.kind,
                    dtype: *dtype,
                    n: *n,
                }),
            })),
            (
                r9v_ir::Op::GatherRows(_),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params:
                        ElementwiseFacts::GatherRows {
                            dtype,
                            index_dtype,
                            width,
                        },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::GatherRows(GatherRowsStatic {
                    dtype: *dtype,
                    index_dtype: *index_dtype,
                    width: *width,
                }),
            })),
            (
                r9v_ir::Op::ScatterAddRows(_),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params:
                        ElementwiseFacts::ScatterAddRows {
                            dtype,
                            index_dtype,
                            width,
                            has_dest,
                        },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::ScatterAddRows(ScatterAddRowsStatic {
                    dtype: *dtype,
                    index_dtype: *index_dtype,
                    width: *width,
                    has_dest: *has_dest,
                }),
            })),
            (
                r9v_ir::Op::Split(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::Split { total, dtype },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::Split(SplitStatic {
                    first: o.first,
                    total: *total,
                    dtype: *dtype,
                }),
            })),
            (
                r9v_ir::Op::Concat(_),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params:
                        ElementwiseFacts::Concat {
                            c0,
                            c1,
                            a_dtype,
                            b_dtype,
                            out_dtype,
                        },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::Concat(ConcatStatic {
                    c0: *c0,
                    c1: *c1,
                    a_dtype: *a_dtype,
                    b_dtype: *b_dtype,
                    out_dtype: *out_dtype,
                }),
            })),
            (
                r9v_ir::Op::Norm(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params:
                        ElementwiseFacts::Norm {
                            in_dtype,
                            n,
                            has_bias,
                        },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::Norm(NormStatic {
                    kind: o.kind,
                    eps_bits: o.eps.to_bits(),
                    axis: o.axis,
                    weight_offset_bits: o.weight_offset.to_bits(),
                    in_dtype: *in_dtype,
                    out_dtype: o.out_dtype,
                    n: *n,
                    has_bias: *has_bias,
                }),
            })),
            (
                r9v_ir::Op::ResidualAdd(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params:
                        ElementwiseFacts::ResidualAdd {
                            a_dtype,
                            b_dtype,
                            n,
                        },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::ResidualAdd(ResidualAddStatic {
                    a_dtype: *a_dtype,
                    b_dtype: *b_dtype,
                    out_dtype: o.out_dtype,
                    scale_bits: o.scale.to_bits(),
                    n: *n,
                }),
            })),
            (
                r9v_ir::Op::ActMul(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::ActMul { dtype, width },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::ActMul(ActMulStatic {
                    act: o.act,
                    clamp_bits: o.clamp.map(|v| v.to_bits()),
                    dtype: *dtype,
                    width: *width,
                }),
            })),
            (
                r9v_ir::Op::Activation(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::Activation { dtype, width },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::Activation(ActivationStatic {
                    act: o.act,
                    clamp_bits: o.clamp.map(|v| v.to_bits()),
                    dtype: *dtype,
                    width: *width,
                }),
            })),
            (
                r9v_ir::Op::LogitSoftcap(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::LogitSoftcap { v },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::LogitSoftcap(LogitSoftcapStatic {
                    cap_bits: o.cap.to_bits(),
                    v: *v,
                }),
            })),
            (
                r9v_ir::Op::Rope(o),
                StaticFacts::Elementwise {
                    t_bucket,
                    fused_with,
                    params: ElementwiseFacts::Rope { in_dtype, h, d },
                },
            ) => Ok(Self::Elementwise(ElementwiseStatic {
                t_bucket: *t_bucket,
                fused_with: *fused_with,
                op_params: ElementwiseParams::Rope(RopeStatic {
                    rot_dim: o.rot_dim,
                    theta_bits: o.theta.to_bits(),
                    style: o.style,
                    scaling: RopeScalingStatic::from_ir(&o.scaling),
                    mrope_sections: o.mrope_sections,
                    in_dtype: *in_dtype,
                    out_dtype: o.out_dtype,
                    h: *h,
                    d: *d,
                }),
            })),
            (
                r9v_ir::Op::LogitsPostprocess(_),
                StaticFacts::Sampling(SamplingFacts::LogitsPostprocess {
                    s_bucket,
                    v,
                    q_bucket,
                    has_history_counts,
                    has_grammar_mask,
                }),
            ) => Ok(Self::Sampling(SamplingStatic::LogitsPostprocess(
                LogitsPostprocessStatic {
                    s_bucket: *s_bucket,
                    v: *v,
                    q_bucket: *q_bucket,
                    has_history_counts: *has_history_counts,
                    has_grammar_mask: *has_grammar_mask,
                },
            ))),
            (
                r9v_ir::Op::Sample(o),
                StaticFacts::Sampling(SamplingFacts::Sample { s_bucket, v }),
            ) => Ok(Self::Sampling(SamplingStatic::Sample(SampleStatic {
                s_bucket: *s_bucket,
                v: *v,
                rng: o.rng,
            }))),
            (
                r9v_ir::Op::Verify(o),
                StaticFacts::Sampling(SamplingFacts::Verify {
                    s_bucket,
                    v,
                    q_bucket,
                    tree,
                    has_draft_probs,
                }),
            ) => Ok(Self::Sampling(SamplingStatic::Verify(VerifyStatic {
                s_bucket: *s_bucket,
                v: *v,
                q_bucket: *q_bucket,
                method: VerifyMethodStatic::from_ir(&o.method),
                tree: *tree,
                has_draft_probs: *has_draft_probs,
            }))),
            (
                r9v_ir::Op::AllReduce(o),
                StaticFacts::Collectives(CollectiveFacts::AllReduce {
                    rank,
                    world,
                    transport,
                    bytes_bucket,
                }),
            ) => Ok(Self::Collectives(CollectivesStatic::AllReduce(
                AllReduceStatic {
                    group: o.group.as_u64(),
                    rank: *rank,
                    world: *world,
                    dtype: o.dtype,
                    reduce_in: o.reduce_in,
                    reduction_op: o.op,
                    transport: *transport,
                    bytes_bucket: *bytes_bucket,
                },
            ))),
            (
                r9v_ir::Op::AllGather(o),
                StaticFacts::Collectives(CollectiveFacts::AllGather {
                    rank,
                    world,
                    transport,
                    bytes_bucket,
                }),
            ) => Ok(Self::Collectives(CollectivesStatic::AllGather(
                AllGatherStatic {
                    group: o.group.as_u64(),
                    rank: *rank,
                    world: *world,
                    dtype: o.dtype,
                    axis: o.axis,
                    transport: *transport,
                    bytes_bucket: *bytes_bucket,
                },
            ))),
            (
                r9v_ir::Op::ReduceScatter(o),
                StaticFacts::Collectives(CollectiveFacts::ReduceScatter {
                    rank,
                    world,
                    transport,
                    bytes_bucket,
                }),
            ) => Ok(Self::Collectives(CollectivesStatic::ReduceScatter(
                ReduceScatterStatic {
                    group: o.group.as_u64(),
                    rank: *rank,
                    world: *world,
                    dtype: o.dtype,
                    reduce_in: o.reduce_in,
                    reduction_op: o.op,
                    axis: o.axis,
                    transport: *transport,
                    bytes_bucket: *bytes_bucket,
                },
            ))),
            (
                r9v_ir::Op::AllToAll(o),
                StaticFacts::Collectives(CollectiveFacts::AllToAll {
                    rank,
                    world,
                    transport,
                    bytes_bucket,
                }),
            ) => Ok(Self::Collectives(CollectivesStatic::AllToAll(
                AllToAllStatic {
                    group: o.group.as_u64(),
                    rank: *rank,
                    world: *world,
                    dtype: o.dtype,
                    transport: *transport,
                    bytes_bucket: *bytes_bucket,
                },
            ))),
            (
                r9v_ir::Op::Send(o),
                StaticFacts::Collectives(CollectiveFacts::Send {
                    rank,
                    world,
                    transport,
                    bytes_bucket,
                }),
            ) => Ok(Self::Collectives(CollectivesStatic::Send(SendStatic {
                group: o.group.as_u64(),
                rank: *rank,
                world: *world,
                peer: o.peer,
                dtype: o.dtype,
                transport: *transport,
                bytes_bucket: *bytes_bucket,
            }))),
            (
                r9v_ir::Op::Recv(o),
                StaticFacts::Collectives(CollectiveFacts::Recv {
                    rank,
                    world,
                    transport,
                    bytes_bucket,
                    shape,
                }),
            ) => {
                if shape.len() != o.shape.len() {
                    return Err(facts_mismatch(
                        OpId::Recv,
                        "recv facts shape rank must match IR recv shape rank",
                    ));
                }
                // Exact per-extent agreement: a concrete IR extent must equal
                // the resolved facts extent bit-for-bit; a symbolic extent
                // resolves to whatever the facts carry. Zero extents and an
                // overflowing element count fail typed here, not downstream.
                let mut elements: u64 = 1;
                for (axis, extent) in shape.iter().enumerate() {
                    if *extent == 0 {
                        return Err(facts_mismatch(
                            OpId::Recv,
                            "recv facts shape extents must all be > 0",
                        ));
                    }
                    if let Dim::Concrete(concrete) = o.shape[axis] {
                        if *extent != concrete {
                            return Err(facts_mismatch(
                                OpId::Recv,
                                "recv facts shape extent must equal the concrete IR extent",
                            ));
                        }
                    }
                    elements = elements.checked_mul(u64::from(*extent)).ok_or_else(|| {
                        facts_mismatch(OpId::Recv, "recv facts shape product overflows u64")
                    })?;
                }
                Ok(Self::Collectives(CollectivesStatic::Recv(RecvStatic {
                    group: o.group.as_u64(),
                    rank: *rank,
                    world: *world,
                    peer: o.peer,
                    shape: shape.clone(),
                    dtype: o.dtype,
                    transport: *transport,
                    bytes_bucket: *bytes_bucket,
                })))
            }
            (
                r9v_ir::Op::Barrier(o),
                StaticFacts::Collectives(CollectiveFacts::Barrier {
                    rank,
                    world,
                    transport,
                }),
            ) => Ok(Self::Collectives(CollectivesStatic::Barrier(
                BarrierStatic {
                    group: o.group.as_u64(),
                    rank: *rank,
                    world: *world,
                    transport: *transport,
                },
            ))),
            (other, _) => Err(facts_mismatch(
                OpId::from_op(other),
                "facts variant does not match op",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_all_32_op_ids_roundtrip() {
        let all_ops = [
            (OpId::EmbedGather, "embed_gather"),
            (OpId::NgramGather, "ngram_gather"),
            (OpId::QuantAct, "quant_act"),
            (OpId::Cast, "cast"),
            (OpId::Copy, "copy"),
            (OpId::GatherRows, "gather_rows"),
            (OpId::ScatterAddRows, "scatter_add_rows"),
            (OpId::Split, "split"),
            (OpId::Concat, "concat"),
            (OpId::Norm, "norm"),
            (OpId::ResidualAdd, "residual_add"),
            (OpId::ActMul, "act_mul"),
            (OpId::Activation, "activation"),
            (OpId::LogitSoftcap, "logit_softcap"),
            (OpId::Rope, "rope"),
            (OpId::Matmul, "matmul"),
            (OpId::MoeRoute, "moe_route"),
            (OpId::MoeFfn, "moe_ffn"),
            (OpId::StateWriteKv, "state_write_kv"),
            (OpId::Attention, "attention"),
            (OpId::CausalConv1d, "causal_conv1d"),
            (OpId::LinearAttnScan, "linear_attn_scan"),
            (OpId::LogitsPostprocess, "logits_postprocess"),
            (OpId::Sample, "sample"),
            (OpId::Verify, "verify"),
            (OpId::AllReduce, "all_reduce"),
            (OpId::AllGather, "all_gather"),
            (OpId::ReduceScatter, "reduce_scatter"),
            (OpId::AllToAll, "all_to_all"),
            (OpId::Send, "send"),
            (OpId::Recv, "recv"),
            (OpId::Barrier, "barrier"),
        ];

        assert_eq!(all_ops.len(), 32, "exactly 32 ops in the closed op set");

        for (op, name) in all_ops {
            assert_eq!(op.as_str(), name);
            assert_eq!(OpId::from_str(name).unwrap(), op);
            assert_eq!(OpId::parse_op(name), Some(op));
            assert_eq!(op.to_string(), name);

            // Serde roundtrip
            let json = serde_json::to_string(&op).unwrap();
            let de: OpId = serde_json::from_str(&json).unwrap();
            assert_eq!(op, de);
        }

        assert!(OpId::from_str("unknown_op").is_err());
    }

    #[test]
    fn test_tier_parsing_and_display() {
        for (tier, name) in [
            (Tier::T0, "t0"),
            (Tier::T0v, "t0v"),
            (Tier::T1, "t1"),
            (Tier::T2, "t2"),
        ] {
            assert_eq!(tier.as_str(), name);
            assert_eq!(Tier::from_str(name).unwrap(), tier);
            assert_eq!(tier.to_string(), name);
        }
        assert!(Tier::from_str("t3").is_err());
    }

    #[test]
    fn test_arch_name() {
        let arch1 = ArchName::from("gfx942");
        let arch2 = ArchName::new("gfx942");
        assert_eq!(arch1, arch2);
        assert_eq!(arch1.as_str(), "gfx942");
        assert_eq!(arch1.to_string(), "gfx942");
        assert_eq!(&*arch1, "gfx942");
        assert_eq!(ArchName::from_str("gfx1100").unwrap().as_str(), "gfx1100");
    }

    #[test]
    fn test_variant_hash_hex() {
        let vh = VariantHash::new(0xdeadbeef12345678);
        assert_eq!(vh.as_u64(), 0xdeadbeef12345678);
        assert_eq!(vh.to_hex(), "deadbeef12345678");
        assert_eq!(vh.to_string(), "deadbeef12345678");

        let parsed = VariantHash::from_hex("deadbeef12345678").unwrap();
        assert_eq!(parsed, vh);

        let parsed_prefix = VariantHash::from_hex("0xdeadbeef12345678").unwrap();
        assert_eq!(parsed_prefix, vh);

        let parsed_trait = VariantHash::from_str("deadbeef12345678").unwrap();
        assert_eq!(parsed_trait, vh);
    }
}
