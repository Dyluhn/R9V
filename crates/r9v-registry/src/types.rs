// SPDX-License-Identifier: Apache-2.0
//! Core registry types, closed-set enums, and op-static parameter descriptors (Spec 4 §2, §3, §7).

use std::fmt;
use std::num::ParseIntError;
use std::ops::Deref;
use std::path::PathBuf;

use r9v_ir::{
    AttentionMask, DType, Epilogue, LayoutId, LinearAttnKind, P2pTransport, Placement, QuantScheme,
    VerifyMethod,
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

/// Closed set of token sampling and verification kernel methods (Spec 1 §4.F, Spec 4 §3, Spec 7 §4, CONVENTIONS.md §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMethod {
    /// Logits postprocessing kernel: temperature scaling, presence/frequency penalties, logit bias, top-k/top-p/min-p filtering (Spec 1 §4.F).
    LogitsPostprocess,
    /// Categorical inverse-CDF sampling kernel from postprocessed distribution via counter-based PRNG (Spec 1 §4.F, Spec 4 §5.8).
    InverseCdfSample,
    /// Speculative rejection verification kernel with acceptance probability ratio (Spec 1 §4.F, Spec 7 §4).
    VerifyRejection,
    /// Greedy argmax verification kernel (Spec 1 §4.F, Spec 7 §4).
    VerifyGreedy,
    /// Speculative typical acceptance threshold verification kernel with IEEE-754 bit-preserved epsilon and delta (Spec 7 §4).
    VerifyTypical {
        /// Acceptance probability floor epsilon IEEE-754 bits.
        eps_bits: u32,
        /// Entropy scaling factor delta IEEE-754 bits.
        delta_bits: u32,
    },
}

impl SamplingMethod {
    /// Constructs a [`SamplingMethod::VerifyTypical`] with bit-exact IEEE-754 floats.
    pub const fn typical(eps: f32, delta: f32) -> Self {
        Self::VerifyTypical {
            eps_bits: eps.to_bits(),
            delta_bits: delta.to_bits(),
        }
    }

    /// Returns epsilon float if this is a `VerifyTypical` method.
    pub fn eps(&self) -> Option<f32> {
        match self {
            Self::VerifyTypical { eps_bits, .. } => Some(f32::from_bits(*eps_bits)),
            _ => None,
        }
    }

    /// Returns delta float if this is a `VerifyTypical` method.
    pub fn delta(&self) -> Option<f32> {
        match self {
            Self::VerifyTypical { delta_bits, .. } => Some(f32::from_bits(*delta_bits)),
            _ => None,
        }
    }

    /// Returns the canonical snake_case string for this sampling kernel method.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LogitsPostprocess => "logits_postprocess",
            Self::InverseCdfSample => "inverse_cdf_sample",
            Self::VerifyRejection => "verify_rejection",
            Self::VerifyGreedy => "verify_greedy",
            Self::VerifyTypical { .. } => "verify_typical",
        }
    }
}

impl fmt::Display for SamplingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerifyTypical {
                eps_bits,
                delta_bits,
            } => {
                write!(
                    f,
                    "verify_typical(eps_bits={:#010x},delta_bits={:#010x})",
                    eps_bits, delta_bits
                )
            }
            other => write!(f, "{}", other.as_str()),
        }
    }
}

impl From<&VerifyMethod> for SamplingMethod {
    fn from(vm: &VerifyMethod) -> Self {
        match vm {
            VerifyMethod::Rejection => Self::VerifyRejection,
            VerifyMethod::Greedy => Self::VerifyGreedy,
            VerifyMethod::TypicalAcceptance { eps, delta } => Self::typical(*eps, *delta),
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
    /// Weight quantization scheme (Spec 2 §3, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub w_scheme: QuantScheme,
    /// Weight tensor layout (Spec 2 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub w_layout: LayoutId,
    /// Activation quantization scheme (Spec 1 §2.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub act_scheme: QuantScheme,
    /// Output data type (Spec 1 §2.1, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub out_dtype: DType,
    /// Fused epilogue operation (Spec 1 §4.C, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_epilogue")]
    pub epilogue: Epilogue,
    /// Interleaving mode enabled (Spec 4 §3).
    pub interleave: bool,
    /// SWMMAC 2:4 structured sparsity enabled (Spec 4 §3).
    pub sparse: bool,
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
    /// Quantization schemes per projection (Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme_vec")]
    pub schemes: Vec<QuantScheme>,
    /// Activation quantization scheme (Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_quant_scheme")]
    pub act_scheme: QuantScheme,
    /// Execution placement class (Spec 1 §2.3, Spec 4 §3, Spec 5 §3.4).
    pub placement_kind: PlacementKind,
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
    /// KV cache data type (Spec 3 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub cache_dtype: DType,
    /// Memory layout of the KV cache (Spec 3 §3.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub attention_layout: LayoutId,
    /// Causal or windowed attention mask (Spec 1 §4.D, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_attention_mask")]
    pub mask_kind: AttentionMask,
    /// Low-rank latent MLA mode flag (Spec 4 §3).
    pub latent: Option<bool>,
    /// Softcapping constant encoded as IEEE-754 bit-pattern for determinism (Spec 4 §3).
    pub softcap_bits: Option<u32>,
    /// Retention sinks count or presence (Spec 3 §2, Spec 4 §3).
    pub sinks: Option<u32>,
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
    /// Cache storage data type (Spec 3 §2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub cache_dtype: DType,
    /// Target attention cache layout (Spec 3 §3.2, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_layout_id")]
    pub attention_layout: LayoutId,
    /// Latent MLA cache mode flag (Spec 4 §3).
    pub latent: Option<bool>,
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
}

/// Static parameters for memory-bound elementwise variants (Spec 4 §3).
///
/// Covers: `norm`, `rope`, `act_mul`, `quant_act`, `residual_add`, `embed_gather`, `ngram_gather`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElementwiseStatic {
    /// Full step token bucket T = T_dec + T_pre (Spec 1 §3.5, Spec 4 §3).
    pub t_bucket: u32,
    /// Static tensor dimensions (Spec 4 §3).
    pub dims: Vec<u32>,
    /// Operand and result data types (Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype_vec")]
    pub dtypes: Vec<DType>,
    /// Optional fused successor op identifier (Spec 1 §3.4, Spec 4 §3, CONVENTIONS.md §3.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_with: Option<OpId>,
}

/// Static parameters for sampling and verify variants (Spec 4 §3).
///
/// Covers: `logits_postprocess`, `sample`, `verify`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SamplingStatic {
    /// Sequence count bucket (Spec 1 §3.5, Spec 4 §3).
    pub s_bucket: u32,
    /// Vocabulary size V (exact, Spec 4 §3).
    pub v: u32,
    /// Query tokens bucket (Spec 1 §3.5, Spec 4 §3).
    pub q_bucket: u32,
    /// Sampling or verification algorithm method (Spec 1 §4.F, Spec 4 §3, Spec 7 §4, CONVENTIONS.md §3.2).
    pub method: SamplingMethod,
}

/// Static parameters for multi-rank collective operations (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectivesStatic {
    /// Transferred payload size bucket in bytes (Spec 4 §3).
    pub bytes_bucket: u64,
    /// Communication element data type (Spec 1 §4.G, Spec 4 §3).
    #[serde(with = "crate::serde_helpers::serde_dtype")]
    pub dtype: DType,
    /// Underlying transport mechanism identifier (Spec 1 §4.G, Spec 4 §3, Spec 5, CONVENTIONS.md §3.2).
    #[serde(with = "crate::serde_helpers::serde_p2p_transport")]
    pub transport: P2pTransport,
}

/// Closed enum of static parameter descriptors per op family (Spec 4 §3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum OpStatic {
    /// Matrix multiplication variants (Spec 4 §3, §5.1, §5.2).
    Matmul(MatmulStatic),
    /// Mixture of Experts feed-forward variants (Spec 4 §3, §5.6).
    MoeFfn(MoeFfnStatic),
    /// Paged and latent attention variants (Spec 4 §3, §5.3).
    Attention(AttentionStatic),
    /// KV state cache write variants (Spec 4 §3, §5.4).
    StateWriteKv(StateWriteKvStatic),
    /// Linear attention scan variants (Spec 4 §3, §5.5).
    LinearAttnScan(LinearAttnScanStatic),
    /// Memory-bound elementwise operations (Spec 4 §3, §5.7).
    Elementwise(ElementwiseStatic),
    /// Sampling and speculative verification operations (Spec 4 §3, §5.8).
    Sampling(SamplingStatic),
    /// Inter-device collective communication operations (Spec 4 §3, §5.9).
    Collectives(CollectivesStatic),
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
