// SPDX-License-Identifier: Apache-2.0
//! Step graph DAG, typed edges, stride tracking, and copy insertion (Spec 1 §3; card A1.2).
//!
//! A graph is a DAG of op instances over tensors. There is one graph kind,
//! the **step graph**, captured per `(plan, rank, S_bucket, T_dec_bucket, T_pre_bucket, segment)`
//! (Spec 1 §3.1; Spec 6 §5.1).
//!
//! The compiler tracks strides. A `copy` op is inserted only when a kernel's
//! declared input requirement cannot be met by a view. Every inserted copy is
//! reported in the graph summary (Spec 1 §3.3).

use std::collections::{HashMap, VecDeque};

use crate::op::{CopyKind, CopyOp};
use crate::{
    Class, DType, Dim, IrError, LayoutId, Op, Placement, QuantScheme, ShapeSymbol, ShardLayout,
    Tensor,
};

// -----------------------------------------------------------------------------
// StepGraphKey and Bucket Functions (Spec 1 §3.1, §3.5)
// -----------------------------------------------------------------------------

/// Opaque plan identifier (Spec 1 §3.1, CONVENTIONS.md §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanId(u64);

impl PlanId {
    /// Creates a new plan identifier.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying raw integer value.
    pub const fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Allowed discrete shape bucket sizes for S, T_dec, and T_pre (Spec 1 §3.5).
pub const BUCKET_SIZES: [u32; 13] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

/// Resolves a sequence count `S` to its discrete bucket size (Spec 1 §3.5).
// DECISION(A1.2): bucket functions reject zero batch dims and values > 4096 with explicit IrError variants carrying axis and value; rejected wrapping or silent clamping to 4096 (Spec 1 §3.5).
pub fn bucket_s(s: u32) -> Result<u32, IrError> {
    if s == 0 {
        return Err(IrError::InvalidBucket {
            axis: "S",
            value: s,
        });
    }
    for &b in &BUCKET_SIZES {
        if s <= b {
            return Ok(b);
        }
    }
    Err(IrError::BucketExceeded {
        axis: "S",
        value: s,
        max: 4096,
    })
}

/// Resolves a decode token count `T_dec` to its discrete bucket size (Spec 1 §3.5).
pub fn bucket_t_dec(t_dec: u32) -> Result<u32, IrError> {
    if t_dec == 0 {
        return Err(IrError::InvalidBucket {
            axis: "T_dec",
            value: t_dec,
        });
    }
    for &b in &BUCKET_SIZES {
        if t_dec <= b {
            return Ok(b);
        }
    }
    Err(IrError::BucketExceeded {
        axis: "T_dec",
        value: t_dec,
        max: 4096,
    })
}

/// Resolves a prefill token count `T_pre` to its discrete bucket size (Spec 1 §3.5).
///
/// `T_pre = 0` is legal and represents a pure decode step with no prefill chunk.
pub fn bucket_t_pre(t_pre: u32) -> Result<u32, IrError> {
    if t_pre == 0 {
        return Ok(0);
    }
    for &b in &BUCKET_SIZES {
        if t_pre <= b {
            return Ok(b);
        }
    }
    Err(IrError::BucketExceeded {
        axis: "T_pre",
        value: t_pre,
        max: 4096,
    })
}

/// Resolves `(S, T_dec, T_pre)` into their discrete bucket tuple (Spec 1 §3.5).
pub fn bucket_step(s: u32, t_dec: u32, t_pre: u32) -> Result<(u32, u32, u32), IrError> {
    let mut problems = Vec::new();
    let b_s = match bucket_s(s) {
        Ok(v) => v,
        Err(e) => {
            problems.push(e);
            0
        }
    };
    let b_dec = match bucket_t_dec(t_dec) {
        Ok(v) => v,
        Err(e) => {
            problems.push(e);
            0
        }
    };
    let b_pre = match bucket_t_pre(t_pre) {
        Ok(v) => v,
        Err(e) => {
            problems.push(e);
            0
        }
    };
    IrError::from_problems(problems)?;
    Ok((b_s, b_dec, b_pre))
}

/// Capture key for step graphs (Spec 1 §3.1; Spec 6 §5.1; card A1.2).
///
/// Graphs are captured per `(plan, rank, S_bucket, T_dec_bucket, T_pre_bucket, segment)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StepGraphKey {
    /// Execution plan identifier.
    pub plan_id: PlanId,
    /// Device rank executing this graph instance.
    pub rank: u32,
    /// Bucketed sequence count `S`.
    pub s: u32,
    /// Bucketed decode token count `T_dec`.
    pub t_dec: u32,
    /// Bucketed prefill token count `T_pre` (0 if decode-only).
    pub t_pre: u32,
    /// Model segment index (for host-computed expert pipelining, Spec 6 §5.1).
    pub segment: u32,
}

impl StepGraphKey {
    /// Constructs a step-graph key, validating that dimensions match bucket boundaries.
    pub fn new(
        plan_id: PlanId,
        rank: u32,
        s: u32,
        t_dec: u32,
        t_pre: u32,
        segment: u32,
    ) -> Result<Self, IrError> {
        let mut problems = Vec::new();
        if s > 4096 {
            problems.push(IrError::BucketExceeded {
                axis: "S",
                value: s,
                max: 4096,
            });
        } else if !BUCKET_SIZES.contains(&s) {
            problems.push(IrError::InvalidBucket {
                axis: "S",
                value: s,
            });
        }
        if t_dec > 4096 {
            problems.push(IrError::BucketExceeded {
                axis: "T_dec",
                value: t_dec,
                max: 4096,
            });
        } else if !BUCKET_SIZES.contains(&t_dec) {
            problems.push(IrError::InvalidBucket {
                axis: "T_dec",
                value: t_dec,
            });
        }
        if t_pre > 4096 {
            problems.push(IrError::BucketExceeded {
                axis: "T_pre",
                value: t_pre,
                max: 4096,
            });
        } else if t_pre != 0 && !BUCKET_SIZES.contains(&t_pre) {
            problems.push(IrError::InvalidBucket {
                axis: "T_pre",
                value: t_pre,
            });
        }
        IrError::from_problems(problems)?;
        Ok(Self {
            plan_id,
            rank,
            s,
            t_dec,
            t_pre,
            segment,
        })
    }

    /// Constructs a step-graph key by bucketing raw counts.
    pub fn from_unbucketed(
        plan_id: PlanId,
        rank: u32,
        raw_s: u32,
        raw_t_dec: u32,
        raw_t_pre: u32,
        segment: u32,
    ) -> Result<Self, IrError> {
        let (s, t_dec, t_pre) = bucket_step(raw_s, raw_t_dec, raw_t_pre)?;
        Ok(Self {
            plan_id,
            rank,
            s,
            t_dec,
            t_pre,
            segment,
        })
    }
}

// -----------------------------------------------------------------------------
// External Inputs and Outputs (Spec 1 §3.2)
// -----------------------------------------------------------------------------

/// Closed set of external inputs provided to a step graph (Spec 1 §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExternalInputKind {
    /// Token IDs batch `[T] u32`.
    TokenIds,
    /// Batch metadata struct.
    BatchMeta,
    /// PRNG states per sequence `[S]`.
    RngState,
    /// N-gram staging rows gathered by host `[T, Np, Dn]`.
    GatherStaging,
    /// Grammar acceptance mask per sequence and token position `[S, q, V] bool`.
    GrammarMask,
    /// Per-sequence sampling parameters.
    SamplingParams,
    /// Multimodal embedding override tensor `[T, Dm]`.
    EmbedOverride,
    /// Multimodal embedding replacement mask `[T] bool`.
    EmbedMask,
}

impl ExternalInputKind {
    /// Returns true if this external input kind represents a tensor-backed edge (Spec 1 §3.2, SI-12).
    pub const fn is_tensor_backed(&self) -> bool {
        match self {
            Self::TokenIds
            | Self::GatherStaging
            | Self::GrammarMask
            | Self::EmbedOverride
            | Self::EmbedMask => true,
            Self::BatchMeta | Self::SamplingParams | Self::RngState => false,
        }
    }
}

/// Closed set of external outputs emitted by a step graph (Spec 1 §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExternalOutputKind {
    /// Sampled token IDs `[S, k+1] u32`.
    Sampled,
    /// Number of accepted tokens per sequence `[S] u32`.
    AcceptLen,
    /// Unnormalized output logits (optional, for logprobs).
    Logits,
    /// Pre-lm_head final hidden state `[T, Dm]` (optional, for proposers).
    Hidden,
    /// Updated PRNG states per sequence `[S]`.
    UpdatedRngState,
}

impl ExternalOutputKind {
    /// Returns true if this external output kind represents a tensor-backed edge (Spec 1 §3.2, SI-12).
    pub const fn is_tensor_backed(&self) -> bool {
        match self {
            Self::Sampled | Self::AcceptLen | Self::Logits | Self::Hidden => true,
            Self::UpdatedRngState => false,
        }
    }
}

/// External input binding registered with a step graph (Spec 1 §3.2, SI-12).
// DECISION(A1.2): structured BatchMeta, SamplingParams, and RngState are registered without fake Tensor descriptors, while tensor-backed inputs require valid Tensors and create edges; rejected synthesizing dummy Tensor descriptors for metadata (Spec 1 §3.1, §3.2; SI-12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalInput {
    /// Tensor-backed external input bound to a specific DAG edge.
    Tensor {
        /// Semantic external input kind.
        kind: ExternalInputKind,
        /// Edge identifier carrying the tensor.
        edge: EdgeId,
    },
    /// Non-tensor batch execution metadata (Spec 1 §2.5, §3.2).
    BatchMeta,
    /// Non-tensor per-sequence sampling parameters (Spec 1 §3.2).
    SamplingParams,
    /// Non-tensor PRNG state per sequence (Spec 1 §3.2).
    RngState,
}

impl ExternalInput {
    /// Returns the semantic kind of this external input.
    pub const fn kind(&self) -> ExternalInputKind {
        match self {
            Self::Tensor { kind, .. } => *kind,
            Self::BatchMeta => ExternalInputKind::BatchMeta,
            Self::SamplingParams => ExternalInputKind::SamplingParams,
            Self::RngState => ExternalInputKind::RngState,
        }
    }

    /// Returns the associated edge ID if this is a tensor-backed input.
    pub const fn edge_id(&self) -> Option<EdgeId> {
        match self {
            Self::Tensor { edge, .. } => Some(*edge),
            Self::BatchMeta | Self::SamplingParams | Self::RngState => None,
        }
    }
}

/// External output binding registered with a step graph (Spec 1 §3.2, SI-12).
// DECISION(A1.2): updated RngState is registered as an explicit non-tensor external output binding without a fake Tensor descriptor or EdgeId, while tensor-backed outputs bind to producing DAG edges; rejected synthesizing dummy Tensor descriptors for state (Spec 1 §3.2; SI-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExternalOutput {
    /// Tensor-backed external output bound to a specific DAG edge.
    Tensor {
        /// Semantic external output kind.
        kind: ExternalOutputKind,
        /// Edge identifier producing the output tensor.
        edge: EdgeId,
    },
    /// Non-tensor updated PRNG state per sequence (Spec 1 §3.2).
    UpdatedRngState,
}

impl ExternalOutput {
    /// Returns the semantic kind of this external output.
    pub const fn kind(&self) -> ExternalOutputKind {
        match self {
            Self::Tensor { kind, .. } => *kind,
            Self::UpdatedRngState => ExternalOutputKind::UpdatedRngState,
        }
    }

    /// Returns the associated edge ID if this is a tensor-backed output.
    pub const fn edge_id(&self) -> Option<EdgeId> {
        match self {
            Self::Tensor { edge, .. } => Some(*edge),
            Self::UpdatedRngState => None,
        }
    }
}

fn bucket_dim_matches(dim: Dim, concrete: u32, symbolic: ShapeSymbol) -> bool {
    matches!(dim, Dim::Concrete(value) if value == concrete)
        || matches!(dim, Dim::Symbolic(value) if value == symbolic)
}

fn validate_external_tensor_common(
    role: &'static str,
    tensor: &Tensor,
    expected_rank: usize,
    expected_dtypes: &[DType],
    expected_class: Class,
    device_rank: u32,
) -> Vec<IrError> {
    let mut problems = Vec::new();
    if tensor.shape().len() != expected_rank {
        problems.push(IrError::OpRankMismatch {
            op: "step_graph_external",
            tensor: role,
            expected: expected_rank,
            got: tensor.shape().len(),
        });
    }
    if !expected_dtypes.contains(&tensor.dtype()) {
        problems.push(IrError::OpDTypeMismatch {
            op: "step_graph_external",
            tensor: role,
            expected: expected_dtypes.to_vec().into_boxed_slice(),
            got: tensor.dtype(),
        });
    }
    if tensor.class() != expected_class {
        problems.push(IrError::OpClassMismatch {
            op: "step_graph_external",
            tensor: role,
            expected: expected_class,
            got: tensor.class(),
        });
    }
    if tensor.layout() != LayoutId::CONTIGUOUS {
        problems.push(IrError::OpLayoutMismatch {
            op: "step_graph_external",
            tensor: role,
            expected: LayoutId::CONTIGUOUS,
            got: tensor.layout(),
        });
    }
    if tensor.placement() != (Placement::Device { rank: device_rank }) {
        problems.push(IrError::OpPlacementMismatch {
            op: "step_graph_external",
            tensor: role,
            placement: tensor.placement(),
        });
    }
    problems
}

type ExternalTensorContract = (
    &'static str,
    usize,
    &'static [DType],
    Class,
    Option<(u32, ShapeSymbol)>,
);

fn validate_external_input_tensor(
    key: StepGraphKey,
    kind: ExternalInputKind,
    tensor: &Tensor,
) -> Result<(), IrError> {
    let total_tokens = key.t_dec.saturating_add(key.t_pre);
    let (role, rank, dtypes, class, leading): ExternalTensorContract = match kind {
        ExternalInputKind::TokenIds => (
            "token_ids",
            1,
            &[DType::U32],
            Class::Activation,
            Some((total_tokens, ShapeSymbol::T)),
        ),
        ExternalInputKind::GatherStaging => (
            "gather_staging",
            3,
            &[DType::I4, DType::I8],
            Class::Staging,
            Some((total_tokens, ShapeSymbol::T)),
        ),
        ExternalInputKind::GrammarMask => (
            "grammar_mask",
            3,
            &[DType::Bool],
            Class::Activation,
            Some((key.s, ShapeSymbol::S)),
        ),
        ExternalInputKind::EmbedOverride => (
            "embed_override",
            2,
            &[DType::F16, DType::Bf16, DType::F32],
            Class::Activation,
            Some((total_tokens, ShapeSymbol::T)),
        ),
        ExternalInputKind::EmbedMask => (
            "embed_mask",
            1,
            &[DType::Bool],
            Class::Activation,
            Some((total_tokens, ShapeSymbol::T)),
        ),
        ExternalInputKind::BatchMeta
        | ExternalInputKind::RngState
        | ExternalInputKind::SamplingParams => {
            return Err(IrError::OpAttributeInvalid {
                op: "add_external_input",
                attribute: "kind",
                reason: format!(
                    "{kind:?} is non-tensor structured data and cannot bind a Tensor edge"
                ),
            });
        }
    };

    let mut problems = validate_external_tensor_common(role, tensor, rank, dtypes, class, key.rank);
    let quant_is_valid = match kind {
        ExternalInputKind::GatherStaging => {
            matches!(tensor.quant(), QuantScheme::Scheme(_))
        }
        ExternalInputKind::TokenIds
        | ExternalInputKind::GrammarMask
        | ExternalInputKind::EmbedOverride
        | ExternalInputKind::EmbedMask => tensor.quant() == QuantScheme::None,
        ExternalInputKind::BatchMeta
        | ExternalInputKind::RngState
        | ExternalInputKind::SamplingParams => false,
    };
    if !quant_is_valid {
        problems.push(IrError::OpQuantMismatch {
            op: "step_graph_external",
            tensor: role,
            quant: tensor.quant(),
        });
    }
    if let Some((concrete, symbolic)) = leading {
        if let Some(&dim) = tensor.shape().first() {
            if !bucket_dim_matches(dim, concrete, symbolic) {
                problems.push(IrError::OpShapeMismatch {
                    op: "step_graph_external",
                    tensor: role,
                    detail: format!(
                        "leading dimension must be {concrete} for this bucket or symbolic {symbolic:?}, got {dim:?}"
                    ),
                });
            }
        }
    }
    IrError::from_problems(problems)
}

fn validate_external_output_tensor(
    key: StepGraphKey,
    kind: ExternalOutputKind,
    tensor: &Tensor,
) -> Result<(), IrError> {
    let total_tokens = key.t_dec.saturating_add(key.t_pre);
    let (role, rank, dtypes, leading): (&'static str, usize, &[DType], Option<(u32, ShapeSymbol)>) =
        match kind {
            ExternalOutputKind::Sampled => {
                ("sampled", 2, &[DType::U32], Some((key.s, ShapeSymbol::S)))
            }
            ExternalOutputKind::AcceptLen => (
                "accept_len",
                1,
                &[DType::U32],
                Some((key.s, ShapeSymbol::S)),
            ),
            ExternalOutputKind::Logits => {
                ("logits", 3, &[DType::F32], Some((key.s, ShapeSymbol::S)))
            }
            ExternalOutputKind::Hidden => (
                "hidden",
                2,
                &[DType::F16, DType::Bf16, DType::F32],
                Some((total_tokens, ShapeSymbol::T)),
            ),
            ExternalOutputKind::UpdatedRngState => {
                return Err(IrError::OpAttributeInvalid {
                op: "add_external_output",
                attribute: "kind",
                reason: "UpdatedRngState is non-tensor structured data and cannot bind a Tensor edge"
                    .to_string(),
            });
            }
        };

    let mut problems =
        validate_external_tensor_common(role, tensor, rank, dtypes, Class::Activation, key.rank);
    if tensor.quant() != QuantScheme::None {
        problems.push(IrError::OpQuantMismatch {
            op: "step_graph_external",
            tensor: role,
            quant: tensor.quant(),
        });
    }
    if let Some((concrete, symbolic)) = leading {
        if let Some(&dim) = tensor.shape().first() {
            if !bucket_dim_matches(dim, concrete, symbolic) {
                problems.push(IrError::OpShapeMismatch {
                    op: "step_graph_external",
                    tensor: role,
                    detail: format!(
                        "leading dimension must be {concrete} for this bucket or symbolic {symbolic:?}, got {dim:?}"
                    ),
                });
            }
        }
    }
    IrError::from_problems(problems)
}

// -----------------------------------------------------------------------------
// DAG Types, Stride Requirements, and Stride Tracking (Spec 1 §3.1, §3.3)
// -----------------------------------------------------------------------------

/// Unique node index in a `Graph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Unique edge index in a `Graph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub usize);

/// Closed set of input stride requirements enforced by kernel implementations (Spec 1 §2.3, §3.3; card A1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StrideRequirement {
    /// The kernel accepts any stride configuration (e.g. elementwise or stride-aware kernels).
    #[default]
    Any,
    /// The kernel requires standard row-major contiguous strides.
    Contiguous,
}

impl StrideRequirement {
    /// Returns true if the provided strides satisfy this requirement for the given shape.
    pub fn is_satisfied(&self, strides: &[i64], shape: &[Dim]) -> bool {
        match self {
            Self::Any => true,
            Self::Contiguous => strides == compute_contiguous_strides(shape),
        }
    }
}

/// Computes standard row-major contiguous strides for a shape (Spec 1 §2.3, §3.3).
// DECISION(A1.2): symbolic dimensions use a synthetic non-unit extent for stride
// comparison. A unit extent would collapse every symbolic stride to one and make
// transposed symbolic views appear contiguous.
pub fn compute_contiguous_strides(shape: &[Dim]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut stride = 1i64;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        let extent = match shape[i] {
            Dim::Concrete(c) => c as i64,
            Dim::Symbolic(_) => 2i64,
        };
        stride = stride.saturating_mul(extent.max(1));
    }
    strides
}

fn concrete_element_count(shape: &[Dim]) -> Option<u128> {
    shape.iter().try_fold(1u128, |count, dim| match dim {
        Dim::Concrete(extent) => Some(count * u128::from(*extent)),
        Dim::Symbolic(_) => None,
    })
}

/// A typed tensor edge in the graph DAG (Spec 1 §3.1, §3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdge {
    /// Unique edge identifier.
    pub id: EdgeId,
    /// Tensor metadata (shape, dtype, quant, layout, placement, sharding, class).
    pub tensor: Tensor,
    /// Memory strides tracked across views and transposes.
    pub strides: Vec<i64>,
    /// Parent edge for a metadata-only view, or `None` for source and op-produced edges.
    pub source_edge: Option<EdgeId>,
}

impl GraphEdge {
    /// Returns true if this edge has standard contiguous row-major strides.
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_contiguous_strides(self.tensor.shape());
        self.strides == expected
    }
}

/// A node in the step graph DAG (Spec 1 §3.1).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Operation descriptor.
    pub op: Op,
    /// Incoming input edge IDs in signature order.
    pub inputs: Vec<EdgeId>,
    /// Per-input stride requirements in signature order (Spec 1 §3.3).
    pub input_requirements: Vec<StrideRequirement>,
    /// Outgoing output edge IDs in signature order.
    pub outputs: Vec<EdgeId>,
}

/// Record of an explicit copy inserted due to a stride mismatch (Spec 1 §3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertedCopy {
    /// NodeId of the inserted `Copy` op.
    pub copy_node: NodeId,
    /// EdgeId carrying the non-contiguous source tensor.
    pub source_edge: EdgeId,
    /// EdgeId carrying the newly contiguized destination tensor.
    pub dest_edge: EdgeId,
    /// Consumer nodes whose inputs were rewired to the shared copy output.
    pub consumer_nodes: Vec<NodeId>,
    /// Actual memory strides before contiguization.
    pub actual_strides: Vec<i64>,
    /// Expected contiguous strides required by the consumer kernel.
    pub expected_strides: Vec<i64>,
    /// Diagnostic explanation.
    pub reason: &'static str,
}

/// High-level report of a captured step graph (Spec 1 §3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphSummary {
    /// Step-graph key.
    pub key: StepGraphKey,
    /// Total number of operation nodes in the DAG.
    pub op_count: usize,
    /// Total number of tensor edges in the DAG.
    pub edge_count: usize,
    /// Registered external inputs.
    pub external_inputs: Vec<ExternalInputKind>,
    /// Registered external outputs.
    pub external_outputs: Vec<ExternalOutputKind>,
    /// All copies inserted by the compiler on stride mismatches.
    pub inserted_copies: Vec<InsertedCopy>,
}

impl GraphSummary {
    /// Returns the number of copies inserted to resolve stride mismatches.
    pub fn inserted_copy_count(&self) -> usize {
        self.inserted_copies.len()
    }
}

// -----------------------------------------------------------------------------
// Graph DAG Implementation
// -----------------------------------------------------------------------------

/// Step graph DAG (Spec 1 §3.1, §3.3; card A1.2).
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    key: StepGraphKey,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    external_inputs: Vec<ExternalInput>,
    external_outputs: Vec<ExternalOutput>,
    inserted_copies: Vec<InsertedCopy>,
}

impl Graph {
    /// Creates an empty step graph with the given capture key.
    pub fn new(key: StepGraphKey) -> Self {
        Self {
            key,
            nodes: Vec::new(),
            edges: Vec::new(),
            external_inputs: Vec::new(),
            external_outputs: Vec::new(),
            inserted_copies: Vec::new(),
        }
    }

    /// Step graph capture key.
    pub fn key(&self) -> StepGraphKey {
        self.key
    }

    /// Returns a slice of all op nodes in insertion order.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Returns a slice of all tensor edges.
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Returns a slice of registered external inputs.
    pub fn external_inputs(&self) -> &[ExternalInput] {
        &self.external_inputs
    }

    /// Returns a slice of registered external outputs.
    pub fn external_outputs(&self) -> &[ExternalOutput] {
        &self.external_outputs
    }

    /// Returns a slice of inserted copy records.
    pub fn inserted_copies(&self) -> &[InsertedCopy] {
        &self.inserted_copies
    }

    fn push_tensor_edge(&mut self, tensor: Tensor) -> EdgeId {
        let edge_id = EdgeId(self.edges.len());
        let strides = compute_contiguous_strides(tensor.shape());
        self.edges.push(GraphEdge {
            id: edge_id,
            tensor,
            strides,
            source_edge: None,
        });
        edge_id
    }

    /// Adds a graph-owned weight, parameter, or state source and returns its edge.
    ///
    /// Request-time activation and staging sources must use the typed external
    /// input API; they cannot bypass the Spec 1 §3.2 signature.
    pub fn add_tensor(&mut self, tensor: Tensor) -> Result<EdgeId, IrError> {
        let mut problems = Vec::new();
        if !matches!(tensor.class(), Class::Weight | Class::Param | Class::State) {
            problems.push(IrError::GraphSourceClassInvalid {
                class: tensor.class(),
            });
        }
        if let Placement::Device { rank } = tensor.placement() {
            if rank != self.key.rank {
                problems.push(IrError::GraphTensorRankMismatch {
                    edge: self.edges.len(),
                    tensor_rank: rank,
                    graph_rank: self.key.rank,
                });
            }
        }
        IrError::from_problems(problems)?;
        Ok(self.push_tensor_edge(tensor))
    }

    /// Registers an external input tensor, creating and returning an input edge (Spec 1 §3.2).
    ///
    /// Non-tensor inputs (`BatchMeta`, `SamplingParams`, `RngState`) must be registered via
    /// [`add_external_non_tensor`](Self::add_external_non_tensor) or dedicated convenience methods
    /// without fake tensor descriptors.
    pub fn add_external_input(
        &mut self,
        kind: ExternalInputKind,
        tensor: Tensor,
    ) -> Result<EdgeId, IrError> {
        if !kind.is_tensor_backed() {
            return Err(IrError::OpAttributeInvalid {
                op: "add_external_input",
                attribute: "kind",
                reason: format!(
                    "{kind:?} is a non-tensor input; register without a fake Tensor descriptor via add_external_non_tensor or convenience method"
                ),
            });
        }

        if self
            .external_inputs
            .iter()
            .any(|input| input.kind() == kind)
        {
            return Err(IrError::OpAttributeInvalid {
                op: "add_external_input",
                attribute: "kind",
                reason: format!("duplicate {kind:?} external input"),
            });
        }
        validate_external_input_tensor(self.key, kind, &tensor)?;

        let edge_id = self.push_tensor_edge(tensor);
        self.external_inputs.push(ExternalInput::Tensor {
            kind,
            edge: edge_id,
        });
        Ok(edge_id)
    }

    /// Registers a non-tensor external input (`BatchMeta`, `SamplingParams`, or `RngState`) without a fake tensor descriptor (Spec 1 §3.2, SI-12).
    pub fn add_external_non_tensor(&mut self, kind: ExternalInputKind) -> Result<(), IrError> {
        match kind {
            ExternalInputKind::BatchMeta => {
                if self
                    .external_inputs
                    .iter()
                    .any(|i| matches!(i, ExternalInput::BatchMeta))
                {
                    return Err(IrError::OpAttributeInvalid {
                        op: "add_external_non_tensor",
                        attribute: "kind",
                        reason: "duplicate BatchMeta external input; step graph admits exactly one BatchMeta".to_string(),
                    });
                }
                self.external_inputs.push(ExternalInput::BatchMeta);
                Ok(())
            }
            ExternalInputKind::SamplingParams => {
                if self
                    .external_inputs
                    .iter()
                    .any(|i| matches!(i, ExternalInput::SamplingParams))
                {
                    return Err(IrError::OpAttributeInvalid {
                        op: "add_external_non_tensor",
                        attribute: "kind",
                        reason: "duplicate SamplingParams external input; step graph admits exactly one SamplingParams".to_string(),
                    });
                }
                self.external_inputs.push(ExternalInput::SamplingParams);
                Ok(())
            }
            ExternalInputKind::RngState => {
                if self
                    .external_inputs
                    .iter()
                    .any(|i| matches!(i, ExternalInput::RngState))
                {
                    return Err(IrError::OpAttributeInvalid {
                        op: "add_external_non_tensor",
                        attribute: "kind",
                        reason: "duplicate RngState external input; step graph admits exactly one RngState".to_string(),
                    });
                }
                self.external_inputs.push(ExternalInput::RngState);
                Ok(())
            }
            ExternalInputKind::TokenIds
            | ExternalInputKind::GatherStaging
            | ExternalInputKind::GrammarMask
            | ExternalInputKind::EmbedOverride
            | ExternalInputKind::EmbedMask => Err(IrError::OpAttributeInvalid {
                op: "add_external_non_tensor",
                attribute: "kind",
                reason: format!(
                    "{kind:?} is a tensor-backed input; register via add_external_input with a valid Tensor descriptor"
                ),
            }),
        }
    }

    /// Registers the non-tensor `BatchMeta` external input signature (Spec 1 §2.5, §3.2).
    pub fn add_batch_meta_input(&mut self) -> Result<(), IrError> {
        self.add_external_non_tensor(ExternalInputKind::BatchMeta)
    }

    /// Registers the non-tensor `SamplingParams` external input signature (Spec 1 §3.2).
    pub fn add_sampling_params_input(&mut self) -> Result<(), IrError> {
        self.add_external_non_tensor(ExternalInputKind::SamplingParams)
    }

    /// Registers the non-tensor `RngState` external input signature (Spec 1 §3.2).
    pub fn add_rng_state_input(&mut self) -> Result<(), IrError> {
        self.add_external_non_tensor(ExternalInputKind::RngState)
    }

    /// Registers an external output, binding it to an existing edge (Spec 1 §3.2).
    ///
    /// Non-tensor outputs (`UpdatedRngState`) must be registered via
    /// [`add_external_non_tensor_output`](Self::add_external_non_tensor_output) or
    /// [`add_updated_rng_state_output`](Self::add_updated_rng_state_output).
    pub fn add_external_output(
        &mut self,
        kind: ExternalOutputKind,
        edge: EdgeId,
    ) -> Result<(), IrError> {
        if !kind.is_tensor_backed() {
            return Err(IrError::OpAttributeInvalid {
                op: "add_external_output",
                attribute: "kind",
                reason: format!(
                    "{kind:?} is a non-tensor output; register without a fake Tensor descriptor via add_external_non_tensor_output or add_updated_rng_state_output"
                ),
            });
        }
        if edge.0 >= self.edges.len() {
            return Err(IrError::GraphEdgeNotFound { edge: edge.0 });
        }
        if self
            .external_outputs
            .iter()
            .any(|output| output.kind() == kind)
        {
            return Err(IrError::OpAttributeInvalid {
                op: "add_external_output",
                attribute: "kind",
                reason: format!("duplicate {kind:?} external output"),
            });
        }
        validate_external_output_tensor(self.key, kind, &self.edges[edge.0].tensor)?;
        let producers = self.producer_map();
        if self.resolve_producer(edge, &producers).is_none() {
            return Err(IrError::GraphExternalOutputUnproduced { kind, edge: edge.0 });
        }
        self.external_outputs
            .push(ExternalOutput::Tensor { kind, edge });
        Ok(())
    }

    /// Registers a non-tensor external output (`UpdatedRngState`) without a fake tensor descriptor (Spec 1 §3.2, SI-12).
    pub fn add_external_non_tensor_output(
        &mut self,
        kind: ExternalOutputKind,
    ) -> Result<(), IrError> {
        match kind {
            ExternalOutputKind::UpdatedRngState => {
                if self
                    .external_outputs
                    .iter()
                    .any(|o| matches!(o, ExternalOutput::UpdatedRngState))
                {
                    return Err(IrError::OpAttributeInvalid {
                        op: "add_external_non_tensor_output",
                        attribute: "kind",
                        reason: "duplicate UpdatedRngState external output; step graph admits exactly one UpdatedRngState".to_string(),
                    });
                }
                self.external_outputs.push(ExternalOutput::UpdatedRngState);
                Ok(())
            }
            ExternalOutputKind::Sampled
            | ExternalOutputKind::AcceptLen
            | ExternalOutputKind::Logits
            | ExternalOutputKind::Hidden => Err(IrError::OpAttributeInvalid {
                op: "add_external_non_tensor_output",
                attribute: "kind",
                reason: format!(
                    "{kind:?} is a tensor-backed output; register via add_external_output with a valid EdgeId"
                ),
            }),
        }
    }

    /// Registers the non-tensor `UpdatedRngState` external output signature (Spec 1 §3.2).
    pub fn add_updated_rng_state_output(&mut self) -> Result<(), IrError> {
        self.add_external_non_tensor_output(ExternalOutputKind::UpdatedRngState)
    }

    /// Adds an operation node to the graph without imposing a stride requirement.
    ///
    /// Kernels that require contiguous inputs use
    /// [`add_op_with_requirements`](Self::add_op_with_requirements) to declare it.
    pub fn add_op(
        &mut self,
        op: Op,
        input_edges: &[EdgeId],
        output_tensors: &[Tensor],
    ) -> Result<NodeId, IrError> {
        let default_requirements = vec![StrideRequirement::Any; input_edges.len()];
        self.add_op_with_requirements(op, input_edges, &default_requirements, output_tensors)
    }

    /// Adds an operation node with explicit per-input stride requirements (Spec 1 §3.3; card A1.2).
    pub fn add_op_with_requirements(
        &mut self,
        op: Op,
        input_edges: &[EdgeId],
        input_requirements: &[StrideRequirement],
        output_tensors: &[Tensor],
    ) -> Result<NodeId, IrError> {
        let mut problems = Vec::new();

        for &e in input_edges {
            if e.0 >= self.edges.len() {
                problems.push(IrError::GraphEdgeNotFound { edge: e.0 });
            }
        }

        if input_requirements.len() != input_edges.len() {
            problems.push(IrError::OpInputCountMismatch {
                op: "add_op",
                expected: input_edges.len(),
                got: input_requirements.len(),
            });
        }

        IrError::from_problems(problems)?;

        let node_id = NodeId(self.nodes.len());
        let mut output_edge_ids = Vec::with_capacity(output_tensors.len());

        for t in output_tensors {
            let edge_id = EdgeId(self.edges.len());
            let strides = compute_contiguous_strides(t.shape());
            self.edges.push(GraphEdge {
                id: edge_id,
                tensor: t.clone(),
                strides,
                source_edge: None,
            });
            output_edge_ids.push(edge_id);
        }

        self.nodes.push(GraphNode {
            id: node_id,
            op,
            inputs: input_edges.to_vec(),
            input_requirements: input_requirements.to_vec(),
            outputs: output_edge_ids,
        });

        Ok(node_id)
    }

    fn producer_map(&self) -> HashMap<EdgeId, usize> {
        let mut producers = HashMap::new();
        for node in &self.nodes {
            for &output in &node.outputs {
                producers.insert(output, node.id.0);
            }
        }
        producers
    }

    fn resolve_producer(&self, edge: EdgeId, producers: &HashMap<EdgeId, usize>) -> Option<usize> {
        let mut current = edge;
        for _ in 0..=self.edges.len() {
            if let Some(&producer) = producers.get(&current) {
                return Some(producer);
            }
            current = self.edges.get(current.0)?.source_edge?;
        }
        None
    }

    /// Returns node IDs in dependency-safe execution order.
    ///
    /// Node IDs remain stable insertion identifiers, including across compiler
    /// copy insertion. Executors use this order rather than assuming numeric
    /// IDs are topologically sorted.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, IrError> {
        let node_count = self.nodes.len();
        let producers = self.producer_map();
        let mut in_degree = vec![0usize; node_count];
        let mut adjacency = vec![Vec::new(); node_count];
        let mut problems = Vec::new();

        for node in &self.nodes {
            for &input in &node.inputs {
                if input.0 >= self.edges.len() {
                    problems.push(IrError::GraphEdgeNotFound { edge: input.0 });
                } else if let Some(producer) = self.resolve_producer(input, &producers) {
                    adjacency[producer].push(node.id.0);
                    in_degree[node.id.0] += 1;
                }
            }
        }
        IrError::from_problems(problems)?;

        let mut queue = VecDeque::new();
        for (node, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(node);
            }
        }

        let mut order = Vec::with_capacity(node_count);
        while let Some(node) = queue.pop_front() {
            order.push(NodeId(node));
            for &consumer in &adjacency[node] {
                in_degree[consumer] -= 1;
                if in_degree[consumer] == 0 {
                    queue.push_back(consumer);
                }
            }
        }

        if order.len() != node_count {
            let node = in_degree.iter().position(|&degree| degree > 0).unwrap_or(0);
            return Err(IrError::GraphCycle { node });
        }
        Ok(order)
    }

    /// Modifies the memory strides of an edge (e.g. for testing stride tracking).
    pub fn set_edge_strides(&mut self, edge: EdgeId, strides: Vec<i64>) -> Result<(), IrError> {
        if edge.0 >= self.edges.len() {
            return Err(IrError::GraphEdgeNotFound { edge: edge.0 });
        }

        let rank = self.edges[edge.0].tensor.shape().len();
        if strides.len() != rank {
            return Err(IrError::OpRankMismatch {
                op: "set_edge_strides",
                tensor: "edge",
                expected: rank,
                got: strides.len(),
            });
        }

        self.edges[edge.0].strides = strides;
        Ok(())
    }

    /// Modifies the stride requirement for an existing node's input.
    pub fn set_node_input_requirement(
        &mut self,
        node: NodeId,
        input_pos: usize,
        requirement: StrideRequirement,
    ) -> Result<(), IrError> {
        if node.0 >= self.nodes.len() {
            return Err(IrError::GraphNodeNotFound { node: node.0 });
        }
        if input_pos >= self.nodes[node.0].inputs.len() {
            return Err(IrError::OpInputCountMismatch {
                op: "set_node_input_requirement",
                expected: self.nodes[node.0].inputs.len(),
                got: input_pos,
            });
        }

        self.nodes[node.0].input_requirements[input_pos] = requirement;
        Ok(())
    }

    /// Applies a dimension transpose permutation to an edge, updating logical shape
    /// and retaining physical memory strides to model a view (Spec 1 §2.3).
    pub fn transpose_edge(&mut self, edge: EdgeId, perm: &[usize]) -> Result<EdgeId, IrError> {
        if edge.0 >= self.edges.len() {
            return Err(IrError::GraphEdgeNotFound { edge: edge.0 });
        }

        let old_edge = &self.edges[edge.0];
        let old_shape = old_edge.tensor.shape();
        let rank = old_shape.len();

        if perm.len() != rank {
            return Err(IrError::OpRankMismatch {
                op: "transpose_view",
                tensor: "edge",
                expected: rank,
                got: perm.len(),
            });
        }

        let mut problems = Vec::new();
        let mut seen = vec![false; rank];
        let mut out_of_bounds = Vec::new();
        let mut duplicates = Vec::new();

        for &p in perm {
            if p >= rank {
                out_of_bounds.push(p);
            } else if seen[p] {
                duplicates.push(p);
            } else {
                seen[p] = true;
            }
        }

        if !out_of_bounds.is_empty() {
            problems.push(IrError::OpAttributeInvalid {
                op: "transpose_view",
                attribute: "perm",
                reason: format!(
                    "permutation indices {out_of_bounds:?} out of bounds for rank {rank}"
                ),
            });
        }

        if !duplicates.is_empty() {
            problems.push(IrError::OpAttributeInvalid {
                op: "transpose_view",
                attribute: "perm",
                reason: format!("permutation contains duplicate axes: {duplicates:?}"),
            });
        }

        let missing: Vec<usize> = seen
            .iter()
            .enumerate()
            .filter(|(_, &s)| !s)
            .map(|(i, _)| i)
            .collect();
        if !missing.is_empty() && duplicates.is_empty() && out_of_bounds.is_empty() {
            problems.push(IrError::OpAttributeInvalid {
                op: "transpose_view",
                attribute: "perm",
                reason: format!("permutation missing axes: {missing:?}"),
            });
        }

        IrError::from_problems(problems)?;

        let mut new_shape = Vec::with_capacity(rank);
        let mut new_strides = Vec::with_capacity(rank);
        for &p in perm {
            new_shape.push(old_shape[p]);
            new_strides.push(old_edge.strides[p]);
        }

        let remap_axis = |old_axis: u32| -> u32 {
            perm.iter()
                .position(|&source_axis| source_axis == old_axis as usize)
                .map_or(old_axis, |new_axis| new_axis as u32)
        };
        let new_sharding = match old_edge.tensor.sharding() {
            ShardLayout::Replicated => ShardLayout::Replicated,
            ShardLayout::ColShard { axis } => ShardLayout::ColShard {
                axis: remap_axis(axis),
            },
            ShardLayout::RowShard { axis } => ShardLayout::RowShard {
                axis: remap_axis(axis),
            },
            ShardLayout::HeadShard { heads } => ShardLayout::HeadShard { heads },
            ShardLayout::ExpertShard { experts } => ShardLayout::ExpertShard { experts },
            ShardLayout::Partial => ShardLayout::Partial,
        };

        let new_tensor = Tensor::new(
            new_shape,
            old_edge.tensor.dtype(),
            old_edge.tensor.quant(),
            old_edge.tensor.layout(),
            old_edge.tensor.placement(),
            new_sharding,
            old_edge.tensor.class(),
        )?;

        let new_edge_id = EdgeId(self.edges.len());
        self.edges.push(GraphEdge {
            id: new_edge_id,
            tensor: new_tensor,
            strides: new_strides,
            source_edge: Some(edge),
        });

        Ok(new_edge_id)
    }

    /// Creates a metadata-only contiguous reshape view of an edge (Spec 1 §2.3).
    ///
    /// Concrete shapes must preserve element count. Symbolic relations are
    /// checked after capture, when all dimensions are concrete. A non-contiguous
    /// source must first be materialized because its reshape strides cannot be
    /// inferred as a row-major view.
    pub fn reshape_edge(&mut self, edge: EdgeId, new_shape: Vec<Dim>) -> Result<EdgeId, IrError> {
        if edge.0 >= self.edges.len() {
            return Err(IrError::GraphEdgeNotFound { edge: edge.0 });
        }

        let old_edge = &self.edges[edge.0];
        let expected_strides = compute_contiguous_strides(old_edge.tensor.shape());
        if old_edge.strides != expected_strides {
            return Err(IrError::StrideMismatch {
                edge: edge.0,
                actual: old_edge.strides.clone().into_boxed_slice(),
                expected: expected_strides.into_boxed_slice(),
            });
        }
        if old_edge.tensor.layout() != LayoutId::CONTIGUOUS {
            return Err(IrError::OpLayoutMismatch {
                op: "reshape_view",
                tensor: "edge",
                expected: LayoutId::CONTIGUOUS,
                got: old_edge.tensor.layout(),
            });
        }
        if new_shape != old_edge.tensor.shape()
            && !matches!(
                old_edge.tensor.sharding(),
                ShardLayout::Replicated | ShardLayout::Partial
            )
        {
            return Err(IrError::OpAttributeInvalid {
                op: "reshape_view",
                attribute: "sharding",
                reason: format!(
                    "reshape of {:?} requires an explicit sharding remap",
                    old_edge.tensor.sharding()
                ),
            });
        }
        if let (Some(old_count), Some(new_count)) = (
            concrete_element_count(old_edge.tensor.shape()),
            concrete_element_count(&new_shape),
        ) {
            if old_count != new_count {
                return Err(IrError::OpShapeMismatch {
                    op: "reshape_view",
                    tensor: "edge",
                    detail: format!(
                        "reshape must preserve element count, got {old_count} -> {new_count}"
                    ),
                });
            }
        }

        let new_tensor = Tensor::new(
            new_shape,
            old_edge.tensor.dtype(),
            old_edge.tensor.quant(),
            old_edge.tensor.layout(),
            old_edge.tensor.placement(),
            old_edge.tensor.sharding(),
            old_edge.tensor.class(),
        )?;
        let new_edge_id = EdgeId(self.edges.len());
        let strides = compute_contiguous_strides(new_tensor.shape());
        self.edges.push(GraphEdge {
            id: new_edge_id,
            tensor: new_tensor,
            strides,
            source_edge: Some(edge),
        });
        Ok(new_edge_id)
    }

    /// Rewires an input of a node to a new edge (used for graph transformations and tests).
    pub fn rewire_node_input(
        &mut self,
        node: NodeId,
        input_pos: usize,
        new_edge: EdgeId,
    ) -> Result<(), IrError> {
        let mut problems = Vec::new();
        if node.0 >= self.nodes.len() {
            problems.push(IrError::GraphNodeNotFound { node: node.0 });
        }
        if new_edge.0 >= self.edges.len() {
            problems.push(IrError::GraphEdgeNotFound { edge: new_edge.0 });
        }
        if node.0 < self.nodes.len() && input_pos >= self.nodes[node.0].inputs.len() {
            problems.push(IrError::OpInputCountMismatch {
                op: "rewire",
                expected: self.nodes[node.0].inputs.len(),
                got: input_pos,
            });
        }

        IrError::from_problems(problems)?;

        self.nodes[node.0].inputs[input_pos] = new_edge;
        Ok(())
    }

    /// Scans the graph for stride mismatches on inputs to nodes and inserts
    /// an explicit `CopyOp` where non-contiguous strides cannot be satisfied by a view (Spec 1 §3.3).
    ///
    /// Copies are deduplicated per source edge: all consumers requiring contiguous layout from
    /// the same non-contiguous source edge share exactly one inserted copy node.
    // DECISION(A1.2): materialize_copies checks explicit per-input StrideRequirement and inserts Contiguize copies only when a Contiguous requirement is failed by a view, deduplicating per source edge; rejected blanket contiguization of all noncontiguous edges or silent transmutation (Spec 1 §2.3, §3.3).
    pub fn materialize_copies(&mut self) -> Result<usize, IrError> {
        struct CopyPlan {
            src_edge_id: EdgeId,
            actual_strides: Vec<i64>,
            expected_strides: Vec<i64>,
            consumers: Vec<(usize, usize)>,
        }

        let mut plans: Vec<CopyPlan> = Vec::new();
        let mut plan_by_source: HashMap<EdgeId, usize> = HashMap::new();
        let mut existing_rewires: Vec<(EdgeId, usize, usize)> = Vec::new();

        for (node_idx, node) in self.nodes.iter().enumerate() {
            // Copy op itself accepts any stride configuration to contiguize it.
            if matches!(node.op, Op::Copy(_)) {
                continue;
            }

            for (input_pos, &edge_id) in node.inputs.iter().enumerate() {
                let req = node
                    .input_requirements
                    .get(input_pos)
                    .copied()
                    .unwrap_or(StrideRequirement::Any);

                match req {
                    StrideRequirement::Any => {
                        // Kernel accepts arbitrary strides, no copy inserted.
                        continue;
                    }
                    StrideRequirement::Contiguous => {
                        let edge = &self.edges[edge_id.0];
                        let expected = compute_contiguous_strides(edge.tensor.shape());
                        if edge.strides != expected {
                            // Check if this source edge was already contiguized in a previous pass
                            if let Some(existing) = self
                                .inserted_copies
                                .iter()
                                .find(|c| c.source_edge == edge_id)
                            {
                                existing_rewires.push((existing.dest_edge, node_idx, input_pos));
                            } else if let Some(&plan_idx) = plan_by_source.get(&edge_id) {
                                plans[plan_idx].consumers.push((node_idx, input_pos));
                            } else {
                                let plan_idx = plans.len();
                                plan_by_source.insert(edge_id, plan_idx);
                                plans.push(CopyPlan {
                                    src_edge_id: edge_id,
                                    actual_strides: edge.strides.clone(),
                                    expected_strides: expected,
                                    consumers: vec![(node_idx, input_pos)],
                                });
                            }
                        }
                    }
                }
            }
        }

        // Apply rewires for previously inserted copies (incremental materialization)
        for (dest_edge, node_idx, input_pos) in existing_rewires {
            self.nodes[node_idx].inputs[input_pos] = dest_edge;
            if let Some(existing) = self
                .inserted_copies
                .iter_mut()
                .find(|c| c.dest_edge == dest_edge)
            {
                let consumer_node_id = NodeId(node_idx);
                if !existing.consumer_nodes.contains(&consumer_node_id) {
                    existing.consumer_nodes.push(consumer_node_id);
                    existing.consumer_nodes.sort_by_key(|n| n.0);
                }
            }
        }

        let count = plans.len();
        for plan in plans {
            let src_edge = &self.edges[plan.src_edge_id.0];
            let copy_out_tensor = Tensor::new(
                src_edge.tensor.shape().to_vec(),
                src_edge.tensor.dtype(),
                src_edge.tensor.quant(),
                LayoutId::CONTIGUOUS,
                src_edge.tensor.placement(),
                src_edge.tensor.sharding(),
                src_edge.tensor.class(),
            )?;

            let copy_out_edge_id = EdgeId(self.edges.len());
            let copy_out_strides = compute_contiguous_strides(copy_out_tensor.shape());
            self.edges.push(GraphEdge {
                id: copy_out_edge_id,
                tensor: copy_out_tensor,
                strides: copy_out_strides,
                source_edge: None,
            });

            let copy_node_id = NodeId(self.nodes.len());
            self.nodes.push(GraphNode {
                id: copy_node_id,
                op: Op::Copy(CopyOp {
                    kind: CopyKind::Contiguize,
                }),
                inputs: vec![plan.src_edge_id],
                input_requirements: vec![StrideRequirement::Any],
                outputs: vec![copy_out_edge_id],
            });

            // Consumer lists are deterministic and deduplicated per source contiguization.
            let mut consumer_nodes: Vec<NodeId> = plan
                .consumers
                .iter()
                .map(|(node_idx, _)| NodeId(*node_idx))
                .collect();
            consumer_nodes.sort_by_key(|n| n.0);
            consumer_nodes.dedup();

            for &(node_idx, input_pos) in &plan.consumers {
                self.nodes[node_idx].inputs[input_pos] = copy_out_edge_id;
            }

            self.inserted_copies.push(InsertedCopy {
                copy_node: copy_node_id,
                source_edge: plan.src_edge_id,
                dest_edge: copy_out_edge_id,
                consumer_nodes,
                actual_strides: plan.actual_strides,
                expected_strides: plan.expected_strides,
                reason: "stride mismatch against contiguous kernel requirement",
            });
        }

        Ok(count)
    }

    /// Validates graph structure: checks for cycles, validates every node's op constraints,
    /// requires global structured inputs needed by Attention, LogitsPostprocess, Sample, and Verify,
    /// and verifies edge references (Spec 1 §3.1, §3.2, §4).
    // DECISION(A1.2): graph validation requires global structured inputs needed by Attention, LogitsPostprocess, Sample, and Verify (BatchMeta, SamplingParams, RngState); rejected deferring missing structured metadata checks to runtime (Spec 1 §3.2, §4.D, §4.F; SI-12).
    pub fn validate(&self) -> Result<(), IrError> {
        let mut problems = Vec::new();

        for edge in &self.edges {
            if let Placement::Device { rank } = edge.tensor.placement() {
                if rank != self.key.rank {
                    problems.push(IrError::GraphTensorRankMismatch {
                        edge: edge.id.0,
                        tensor_rank: rank,
                        graph_rank: self.key.rank,
                    });
                }
            }
        }

        // 1. Cycle detection includes metadata-only view dependencies.
        if let Err(problem) = self.topological_order() {
            match problem {
                IrError::Multiple { problems: inner } => problems.extend(inner),
                other => problems.push(other),
            }
        }

        // 2. Global structured inputs requirement checks
        let has_batch_meta = self
            .external_inputs
            .iter()
            .any(|i| matches!(i, ExternalInput::BatchMeta));
        let has_sampling_params = self
            .external_inputs
            .iter()
            .any(|i| matches!(i, ExternalInput::SamplingParams));
        let has_rng_state = self
            .external_inputs
            .iter()
            .any(|i| matches!(i, ExternalInput::RngState));
        let has_updated_rng_state = self
            .external_outputs
            .iter()
            .any(|output| matches!(output, ExternalOutput::UpdatedRngState));

        let mut needs_batch_meta_attention = false;
        let mut needs_batch_meta_state_write = false;
        let mut needs_sampling_params = false;
        let mut needs_rng_sample = false;
        let mut needs_rng_verify = false;

        for node in &self.nodes {
            match node.op {
                Op::StateWriteKv(_) => needs_batch_meta_state_write = true,
                Op::Attention(_) => needs_batch_meta_attention = true,
                Op::LogitsPostprocess(_) => needs_sampling_params = true,
                Op::Sample(_) => needs_rng_sample = true,
                Op::Verify(_) => needs_rng_verify = true,
                Op::EmbedGather(_)
                | Op::NgramGather(_)
                | Op::QuantAct(_)
                | Op::Cast(_)
                | Op::Copy(_)
                | Op::GatherRows(_)
                | Op::ScatterAddRows(_)
                | Op::Norm(_)
                | Op::ResidualAdd(_)
                | Op::ActMul(_)
                | Op::Activation(_)
                | Op::Rope(_)
                | Op::Matmul(_)
                | Op::MoeRoute(_)
                | Op::MoeFfn(_)
                | Op::CausalConv1d(_)
                | Op::LinearAttnScan(_)
                | Op::AllReduce(_)
                | Op::AllGather(_)
                | Op::ReduceScatter(_)
                | Op::AllToAll(_)
                | Op::Send(_)
                | Op::Recv(_)
                | Op::Barrier(_) => {}
            }
        }

        if needs_batch_meta_attention && !has_batch_meta {
            problems.push(IrError::GraphExternalInputMissing {
                kind: ExternalInputKind::BatchMeta,
                required_by: "attention",
            });
        }
        if needs_batch_meta_state_write && !has_batch_meta {
            problems.push(IrError::GraphExternalInputMissing {
                kind: ExternalInputKind::BatchMeta,
                required_by: "state_write_kv",
            });
        }
        if needs_sampling_params && !has_sampling_params {
            problems.push(IrError::GraphExternalInputMissing {
                kind: ExternalInputKind::SamplingParams,
                required_by: "logits_postprocess",
            });
        }
        if needs_rng_sample && !has_rng_state {
            problems.push(IrError::GraphExternalInputMissing {
                kind: ExternalInputKind::RngState,
                required_by: "sample",
            });
        }
        if needs_rng_verify && !has_rng_state {
            problems.push(IrError::GraphExternalInputMissing {
                kind: ExternalInputKind::RngState,
                required_by: "verify",
            });
        }
        if needs_rng_sample && !has_updated_rng_state {
            problems.push(IrError::GraphExternalOutputMissing {
                kind: ExternalOutputKind::UpdatedRngState,
                required_by: "sample",
            });
        }
        if needs_rng_verify && !has_updated_rng_state {
            problems.push(IrError::GraphExternalOutputMissing {
                kind: ExternalOutputKind::UpdatedRngState,
                required_by: "verify",
            });
        }

        // 3. Validate every node's op constraints over typed tensor edges
        for node in &self.nodes {
            let mut in_tensors = Vec::with_capacity(node.inputs.len());
            let mut edge_missing = false;
            for &e in &node.inputs {
                if e.0 < self.edges.len() {
                    in_tensors.push(self.edges[e.0].tensor.clone());
                } else {
                    problems.push(IrError::GraphEdgeNotFound { edge: e.0 });
                    edge_missing = true;
                }
            }

            let mut out_tensors = Vec::with_capacity(node.outputs.len());
            for &e in &node.outputs {
                if e.0 < self.edges.len() {
                    out_tensors.push(self.edges[e.0].tensor.clone());
                } else {
                    problems.push(IrError::GraphEdgeNotFound { edge: e.0 });
                    edge_missing = true;
                }
            }

            if !edge_missing {
                if let Err(problem) = node.op.validate(&in_tensors, &out_tensors) {
                    match problem {
                        IrError::Multiple { problems: inner } => problems.extend(inner),
                        other => problems.push(other),
                    }
                }
            }
        }

        // 4. Validate external outputs referencing DAG edges
        for out in &self.external_outputs {
            if let ExternalOutput::Tensor { edge, .. } = out {
                if edge.0 >= self.edges.len() {
                    problems.push(IrError::GraphEdgeNotFound { edge: edge.0 });
                }
            }
        }

        IrError::from_problems(problems)
    }

    /// Generates a graph summary report (Spec 1 §3.3).
    pub fn summary(&self) -> GraphSummary {
        GraphSummary {
            key: self.key,
            op_count: self.nodes.len(),
            edge_count: self.edges.len(),
            external_inputs: self.external_inputs.iter().map(|i| i.kind()).collect(),
            external_outputs: self.external_outputs.iter().map(|o| o.kind()).collect(),
            inserted_copies: self.inserted_copies.clone(),
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests (CONVENTIONS.md §4.1)
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{
        AttentionMask, AttentionOp, CacheScaleGranularity, CastOp, LogitsPostprocessOp,
        RngAlgorithm, SampleOp, StateWriteKvOp, VerifyMethod, VerifyOp,
    };
    use crate::{Class, DType, Placement, QuantScheme, ShardLayout, StateHandle, StateKind};

    fn make_tensor(shape: Vec<Dim>, dtype: DType) -> Tensor {
        Tensor::new(
            shape,
            dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )
        .expect("valid tensor")
    }

    #[test]
    fn structured_non_tensor_contract_and_convenience_methods() {
        let plan = PlanId::new(10);
        let key = StepGraphKey::new(plan, 0, 4, 4, 0, 0).expect("valid key");
        let mut graph = Graph::new(key);

        // Register structured non-tensor inputs via convenience methods
        graph.add_batch_meta_input().expect("batch meta registers");
        graph
            .add_sampling_params_input()
            .expect("sampling params registers");
        graph.add_rng_state_input().expect("rng state registers");

        // Duplicate registration must be rejected
        assert!(graph.add_batch_meta_input().is_err());
        assert!(graph.add_sampling_params_input().is_err());
        assert!(graph.add_rng_state_input().is_err());

        // Registering non-tensor as tensor-backed with fake Tensor descriptor must fail
        let fake_tensor = make_tensor(vec![Dim::Concrete(4)], DType::U32);
        assert!(graph
            .add_external_input(ExternalInputKind::BatchMeta, fake_tensor.clone())
            .is_err());
        assert!(graph
            .add_external_input(ExternalInputKind::SamplingParams, fake_tensor.clone())
            .is_err());
        assert!(graph
            .add_external_input(ExternalInputKind::RngState, fake_tensor)
            .is_err());

        // Registering tensor-backed kind via non-tensor registration must fail
        assert!(graph
            .add_external_non_tensor(ExternalInputKind::TokenIds)
            .is_err());

        // Register non-tensor output via convenience method
        graph
            .add_updated_rng_state_output()
            .expect("updated rng state registers");
        assert!(graph.add_updated_rng_state_output().is_err());

        // Registering non-tensor output with a fake edge must fail
        assert!(graph
            .add_external_output(ExternalOutputKind::UpdatedRngState, EdgeId(0))
            .is_err());

        // Check bindings
        assert_eq!(graph.external_inputs().len(), 3);
        assert_eq!(graph.external_inputs()[0], ExternalInput::BatchMeta);
        assert_eq!(graph.external_inputs()[1], ExternalInput::SamplingParams);
        assert_eq!(graph.external_inputs()[2], ExternalInput::RngState);

        assert_eq!(graph.external_outputs().len(), 1);
        assert_eq!(graph.external_outputs()[0], ExternalOutput::UpdatedRngState);

        // Summary correctly reflects all external inputs and outputs
        let summary = graph.summary();
        assert_eq!(
            summary.external_inputs,
            vec![
                ExternalInputKind::BatchMeta,
                ExternalInputKind::SamplingParams,
                ExternalInputKind::RngState,
            ]
        );
        assert_eq!(
            summary.external_outputs,
            vec![ExternalOutputKind::UpdatedRngState]
        );
    }

    #[test]
    fn graph_validation_requires_global_structured_inputs_and_outputs() {
        let plan = PlanId::new(20);
        let key = StepGraphKey::new(plan, 0, 4, 4, 0, 0).expect("valid key");

        // 1. Attention requires BatchMeta
        let mut g_attn = Graph::new(key);
        let q_tensor = make_tensor(
            vec![Dim::Concrete(4), Dim::Concrete(8), Dim::Concrete(64)],
            DType::F16,
        );
        let flat_q = make_tensor(vec![Dim::Concrete(4), Dim::Concrete(512)], DType::F16);
        let e_flat_q = g_attn
            .add_external_input(ExternalInputKind::EmbedOverride, flat_q)
            .expect("flat q input");
        let e_q = g_attn
            .reshape_edge(
                e_flat_q,
                vec![Dim::Concrete(4), Dim::Concrete(8), Dim::Concrete(64)],
            )
            .expect("q reshape");
        let attn_op = Op::Attention(AttentionOp {
            softmax_scale: 0.125,
            mask: AttentionMask::Causal,
            sinks: 0,
            logit_softcap: None,
            mla: None,
            out_dtype: DType::F16,
            handle: StateHandle::new(0, StateKind::KvPaged),
        });
        g_attn
            .add_op(attn_op, &[e_q], &[q_tensor])
            .expect("add attention");
        // Validation fails because BatchMeta is missing
        assert!(g_attn.validate().is_err());
        // Adding BatchMeta fixes it
        g_attn.add_batch_meta_input().expect("add batch meta");
        assert!(g_attn.validate().is_ok());

        // 2. StateWriteKv consumes slot_map through BatchMeta.
        let mut g_state_write = Graph::new(key);
        let flat_k = make_tensor(vec![Dim::Concrete(4), Dim::Concrete(16)], DType::F16);
        let e_flat_k = g_state_write
            .add_external_input(ExternalInputKind::EmbedOverride, flat_k)
            .expect("flat key input");
        let e_k = g_state_write
            .reshape_edge(
                e_flat_k,
                vec![Dim::Concrete(4), Dim::Concrete(2), Dim::Concrete(8)],
            )
            .expect("key reshape");
        let kv = make_tensor(
            vec![Dim::Concrete(4), Dim::Concrete(2), Dim::Concrete(8)],
            DType::F16,
        );
        let v_node = g_state_write
            .add_op(Op::Cast(CastOp { dtype: DType::F16 }), &[e_k], &[kv])
            .expect("value producer");
        let e_v = g_state_write.nodes()[v_node.0].outputs[0];
        g_state_write
            .add_op(
                Op::StateWriteKv(StateWriteKvOp {
                    cache_dtype: DType::F16,
                    scale_granularity: CacheScaleGranularity::PerTokenHead,
                    latent: None,
                    handle: StateHandle::new(0, StateKind::KvPaged),
                }),
                &[e_k, e_v],
                &[],
            )
            .expect("state write");
        assert!(matches!(
            g_state_write.validate(),
            Err(IrError::GraphExternalInputMissing {
                kind: ExternalInputKind::BatchMeta,
                required_by: "state_write_kv"
            })
        ));
        g_state_write
            .add_batch_meta_input()
            .expect("add batch meta");
        assert!(g_state_write.validate().is_ok());

        // 3. LogitsPostprocess requires SamplingParams
        let mut g_lp = Graph::new(key);
        let logits_tensor = make_tensor(
            vec![Dim::Concrete(4), Dim::Concrete(1), Dim::Concrete(32)],
            DType::F32,
        );
        let probs_tensor = make_tensor(
            vec![Dim::Concrete(4), Dim::Concrete(1), Dim::Concrete(32)],
            DType::F32,
        );
        let flat_logits = make_tensor(vec![Dim::Concrete(4), Dim::Concrete(32)], DType::F32);
        let e_flat_logits = g_lp
            .add_external_input(ExternalInputKind::EmbedOverride, flat_logits)
            .expect("flat logits input");
        let e_logits = g_lp
            .reshape_edge(
                e_flat_logits,
                vec![Dim::Concrete(4), Dim::Concrete(1), Dim::Concrete(32)],
            )
            .expect("logits reshape");
        assert_eq!(g_lp.edges()[e_logits.0].tensor, logits_tensor);
        let lp_op = Op::LogitsPostprocess(LogitsPostprocessOp);
        g_lp.add_op(lp_op, &[e_logits], &[probs_tensor])
            .expect("add lp");
        // Validation fails because SamplingParams is missing
        assert!(g_lp.validate().is_err());
        // Adding SamplingParams fixes it
        g_lp.add_sampling_params_input()
            .expect("add sampling params");
        assert!(g_lp.validate().is_ok());

        // 4. Sample requires RngState
        let mut g_sample = Graph::new(key);
        let s_probs = make_tensor(vec![Dim::Concrete(4), Dim::Concrete(32)], DType::F32);
        let s_token = make_tensor(vec![Dim::Concrete(4)], DType::U32);
        let e_sprobs = g_sample
            .add_external_input(ExternalInputKind::EmbedOverride, s_probs)
            .expect("sample probabilities input");
        let sample_op = Op::Sample(SampleOp {
            rng: RngAlgorithm::Philox4x32,
        });
        g_sample
            .add_op(sample_op, &[e_sprobs], &[s_token])
            .expect("add sample");
        // Validation fails because RngState is missing
        assert!(g_sample.validate().is_err());
        // Both input and updated-state output are part of the structured contract.
        g_sample.add_rng_state_input().expect("add rng state");
        assert!(g_sample.validate().is_err());
        g_sample
            .add_updated_rng_state_output()
            .expect("add updated rng state");
        assert!(g_sample.validate().is_ok());

        // 5. Verify requires RngState
        let mut g_verify = Graph::new(key);
        let draft_tokens = make_tensor(vec![Dim::Concrete(4), Dim::Concrete(1)], DType::U32);
        let target_probs = make_tensor(
            vec![Dim::Concrete(4), Dim::Concrete(2), Dim::Concrete(32)],
            DType::F32,
        );
        let accepted = make_tensor(vec![Dim::Concrete(4), Dim::Concrete(2)], DType::U32);
        let accept_len = make_tensor(vec![Dim::Concrete(4)], DType::U32);
        let token_ids = make_tensor(vec![Dim::Concrete(4)], DType::U32);
        let e_token_ids = g_verify
            .add_external_input(ExternalInputKind::TokenIds, token_ids)
            .expect("token IDs input");
        let e_draft = g_verify
            .reshape_edge(e_token_ids, vec![Dim::Concrete(4), Dim::Concrete(1)])
            .expect("draft token reshape");
        assert_eq!(g_verify.edges()[e_draft.0].tensor, draft_tokens);
        let flat_target = make_tensor(vec![Dim::Concrete(4), Dim::Concrete(64)], DType::F32);
        let e_flat_target = g_verify
            .add_external_input(ExternalInputKind::EmbedOverride, flat_target)
            .expect("flat target probabilities input");
        let e_target = g_verify
            .reshape_edge(
                e_flat_target,
                vec![Dim::Concrete(4), Dim::Concrete(2), Dim::Concrete(32)],
            )
            .expect("target probability reshape");
        assert_eq!(g_verify.edges()[e_target.0].tensor, target_probs);
        let verify_op = Op::Verify(VerifyOp {
            method: VerifyMethod::Greedy,
        });
        g_verify
            .add_op(verify_op, &[e_draft, e_target], &[accepted, accept_len])
            .expect("add verify");
        // Validation fails because RngState is missing
        assert!(g_verify.validate().is_err());
        // Both input and updated-state output are required.
        g_verify.add_rng_state_input().expect("add rng state");
        assert!(g_verify.validate().is_err());
        g_verify
            .add_updated_rng_state_output()
            .expect("add updated rng state");
        assert!(g_verify.validate().is_ok());
    }

    #[test]
    fn materialize_copies_deduplicates_with_deterministic_consumers() {
        let plan = PlanId::new(30);
        let key = StepGraphKey::new(plan, 0, 4, 16, 0, 0).expect("valid key");
        let mut graph = Graph::new(key);

        let t_in = make_tensor(vec![Dim::Concrete(16), Dim::Concrete(64)], DType::F16);
        let e_in = graph
            .add_external_input(ExternalInputKind::EmbedOverride, t_in)
            .expect("activation input");

        // Transpose creates non-contiguous view [64, 16] with strides [1, 64]
        let e_trans = graph.transpose_edge(e_in, &[1, 0]).expect("transpose");
        assert!(!graph.edges()[e_trans.0].is_contiguous());

        let t_out1 = make_tensor(vec![Dim::Concrete(64), Dim::Concrete(16)], DType::F16);
        let t_out2 = make_tensor(vec![Dim::Concrete(64), Dim::Concrete(16)], DType::F16);

        let op1 = Op::Activation(crate::op::ActivationOp {
            act: crate::op::ActivationKind::Silu,
            clamp: None,
        });
        let op2 = Op::Activation(crate::op::ActivationOp {
            act: crate::op::ActivationKind::Gelu,
            clamp: None,
        });

        // Node 0 and Node 1 both consume e_trans with Contiguous requirement
        let n0 = graph
            .add_op_with_requirements(op1, &[e_trans], &[StrideRequirement::Contiguous], &[t_out1])
            .expect("n0");
        let n1 = graph
            .add_op_with_requirements(op2, &[e_trans], &[StrideRequirement::Contiguous], &[t_out2])
            .expect("n1");

        // Exactly one copy must be inserted for e_trans
        let count = graph.materialize_copies().expect("materialize");
        assert_eq!(
            count, 1,
            "Exactly one shared copy must be inserted for e_trans"
        );

        let summary = graph.summary();
        assert_eq!(summary.inserted_copy_count(), 1);
        let copy_info = &summary.inserted_copies[0];
        assert_eq!(copy_info.source_edge, e_trans);
        // Consumers are deterministic and deduplicated
        assert_eq!(copy_info.consumer_nodes, vec![n0, n1]);

        // Both nodes rewired to the single shared copy output
        assert_eq!(graph.nodes()[n0.0].inputs[0], copy_info.dest_edge);
        assert_eq!(graph.nodes()[n1.0].inputs[0], copy_info.dest_edge);

        // Idempotent
        assert_eq!(graph.materialize_copies().expect("second run"), 0);
        assert_eq!(graph.summary().inserted_copy_count(), 1);
    }
}
