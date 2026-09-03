// SPDX-License-Identifier: Apache-2.0
//! Crate error type for the Op IR (Spec 1 §2–§6, App. A; CONVENTIONS.md §1.1).
//!
//! [`IrError`] is the per-crate domain error enum for `r9v-ir`. Every variant
//! carries the numbers a caller needs to fix the problem (CONVENTIONS.md §1.3),
//! and constructors that check several fields return every failure at once
//! (CONVENTIONS.md §1.4) via [`IrError::Multiple`].

use crate::{Class, DType, LayoutId, Placement, QuantScheme};

/// Domain error for Op IR type construction and validation.
///
/// Covers Spec 1 §2 (core types), §3.1 (step-graph shape rules that constrain
/// batch metadata), §4.D.1 (tree masks) and App. A (arch descriptor rates).
/// Wrapping into the top-level `r9v_common::R9vError` is owned by a later card
/// once a downstream crate composes this error down the dependency graph
/// (CONVENTIONS.md §1.1); this crate cannot declare that impl without either
/// editing `r9v-common` (outside card A1.1's crates) or orphan-rule issues.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IrError {
    /// Tensor rank outside `1..=4` (Spec 1 §2.3: `shape: [Dim; ≤4]`, and every
    /// op signature in §4 has rank ≥ 1).
    #[error("invalid tensor rank {got}: rank must be 1..=4 (Spec 1 §2.3)")]
    InvalidRank {
        /// Rank that was supplied.
        got: usize,
    },

    /// A concrete tensor extent is zero (Spec 1 §2.3; every §4 signature
    /// requires at least one element per axis).
    #[error("tensor axis {axis} has zero extent (Spec 1 §2.3)")]
    ZeroExtent {
        /// Axis index holding the zero extent.
        axis: usize,
    },

    /// A `Host` or `Tiered` placement was attached to a non-`Weight` tensor
    /// class (Spec 1 §2.3: both are legal only for `Weight` class tensors).
    #[error("placement {placement} requires Weight class, got {class} (Spec 1 §2.3)")]
    PlacementForClass {
        /// Requested placement.
        placement: Placement,
        /// Tensor class it was attached to.
        class: Class,
    },

    /// A quantization scheme was attached to a tensor class on which Spec 1
    /// §2.2 does not permit it.
    #[error("quantization {quant:?} is not legal for tensor class {class} (Spec 1 §2.2)")]
    QuantForClass {
        /// Requested quantization scheme.
        quant: QuantScheme,
        /// Tensor class it was attached to.
        class: Class,
    },

    /// A logical layout was attached to a tensor class on which it is not
    /// defined (Spec 1 §2.3, Spec 2 §2).
    #[error("layout {layout} is not legal for tensor class {class} (Spec 1 §2.3, Spec 2 §2)")]
    LayoutForClass {
        /// Requested logical layout.
        layout: LayoutId,
        /// Tensor class it was attached to.
        class: Class,
    },

    /// A dtype was attached to a class for which Spec 1 §2.1 forbids it.
    #[error("dtype {dtype} is not legal for tensor class {class} (Spec 1 §2.1)")]
    DTypeForClass {
        /// Requested element dtype.
        dtype: DType,
        /// Tensor class it was attached to.
        class: Class,
    },

    /// A quantization scheme and stored dtype are incompatible (Spec 1 §2.2,
    /// §4.A; Spec 2 §3.4).
    #[error("quantization {quant:?} is not legal with dtype {dtype} (Spec 1 §2.2, §4.A)")]
    QuantDType {
        /// Requested quantization scheme.
        quant: QuantScheme,
        /// Stored element dtype.
        dtype: DType,
    },

    /// One or more batch dims are zero. `G`, `S`, `T` follow the bucket rules
    /// (Spec 1 §3.5: buckets start at 1; `T_pre` may be 0 but `T ≥ 1` always
    /// holds) and `max_blocks = ceil(max_ctx / 32) ≥ 1` (Spec 3 §3.3).
    #[error("batch dims must be nonzero: G={g} S={s} T={t} max_blocks={max_blocks} (Spec 1 §2.5)")]
    ZeroBatchDim {
        /// Layer-group count.
        g: u32,
        /// Sequence count.
        s: u32,
        /// Token count (`T_dec + T_pre`).
        t: u32,
        /// Blocks per sequence per group.
        max_blocks: u32,
    },

    /// A batch field's length disagrees with the `(G, S, T, max_blocks)` shape
    /// (Spec 1 §2.5).
    #[error("batch field `{field}` has length {actual}, expected {expected} (Spec 1 §2.5)")]
    BatchLength {
        /// Field name, e.g. `"slot_map"`.
        field: &'static str,
        /// Required length from the batch dims.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },

    /// A declared batch shape cannot be represented as a host allocation
    /// length (Spec 1 §2.5). The factors are reported in their documented
    /// order, for example `[G, S, max_blocks]`.
    #[error(
        "batch field `{field}` shape product overflows usize: factors={factors:?} (Spec 1 §2.5)"
    )]
    BatchShapeOverflow {
        /// Field whose flattened length overflowed.
        field: &'static str,
        /// Shape factors in spec order.
        factors: Box<[u32]>,
    },

    /// The builder is missing a required field (Spec 1 §2.5).
    #[error("batch field `{field}` is missing (Spec 1 §2.5)")]
    MissingField {
        /// Missing field name.
        field: &'static str,
    },

    /// A sequence declares `query_len == 0`. Decode sequences have
    /// `1..=k+1`, prefill sequences a chunk size ≥ 1 (Spec 1 §3.1).
    #[error("sequence {seq} has query_len 0, must be >= 1 (Spec 1 §3.1)")]
    EmptyQuery {
        /// Sequence index.
        seq: usize,
    },

    /// The per-sequence query lengths do not sum to the declared token count
    /// `T` (Spec 1 §2.4–§2.5).
    #[error("query_len sum is {actual}, expected batch T={expected} (Spec 1 §2.4–§2.5)")]
    QueryTokenCount {
        /// Declared batch token count.
        expected: u32,
        /// Sum of every sequence's query length.
        actual: u64,
    },

    /// A tree mask's token count disagrees with the batch `T` it is attached
    /// to (`parents [T]`, Spec 1 §4.D.1).
    #[error("tree parents length {tree_t} does not match batch T={batch_t} (Spec 1 §4.D.1)")]
    TreeBatchMismatch {
        /// `parents.len()`.
        tree_t: usize,
        /// Batch `T`.
        batch_t: u32,
    },

    /// A tree parent id is neither −1 (root) nor a valid token index
    /// (Spec 1 §4.D.1).
    #[error("tree parent of token {token} is {parent}, must be -1 or < {t} (Spec 1 §4.D.1)")]
    BadParent {
        /// Token index.
        token: usize,
        /// Offending parent id.
        parent: i32,
        /// Token count `T`.
        t: usize,
    },

    /// A token names itself as its own parent; never a valid tree
    /// (Spec 1 §4.D.1).
    #[error("token {token} is its own parent (Spec 1 §4.D.1)")]
    SelfParent {
        /// Token index.
        token: usize,
    },

    /// The ancestor mask length disagrees with `T * t_max` (Spec 1 §4.D.1).
    #[error(
        "tree ancestors length {actual} != T({t}) * t_max({t_max}) = {expected} (Spec 1 §4.D.1)"
    )]
    AncestorLength {
        /// Token count `T` (`parents.len()`).
        t: usize,
        /// Columns per row.
        t_max: u32,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },

    /// The ancestor-mask shape cannot be represented as a host allocation
    /// length (Spec 1 §4.D.1).
    #[error("tree ancestor shape T={t} by t_max={t_max} overflows usize (Spec 1 §4.D.1)")]
    AncestorShapeOverflow {
        /// Token count.
        t: usize,
        /// Ancestor columns per token.
        t_max: u32,
    },

    /// A non-empty tree has no ancestor-mask columns (Spec 1 §4.D.1).
    #[error("tree with T={t} has t_max=0 (Spec 1 §4.D.1)")]
    ZeroTreeMax {
        /// Token count.
        t: usize,
    },

    /// Parent pointers contain a cycle and therefore do not describe a tree
    /// (Spec 1 §4.D.1).
    #[error("tree parent chain from token {token} contains a cycle (Spec 1 §4.D.1)")]
    TreeCycle {
        /// Deterministic lowest start token whose walk found the cycle.
        token: usize,
    },

    /// A parent pointer crosses between two sequences in a flattened batch
    /// (Spec 1 §4.D.1: −1 is the root of its sequence).
    #[error(
        "tree token {token} in sequence {seq} points to parent {parent} outside [{seq_start}, {seq_end}) (Spec 1 §4.D.1)"
    )]
    TreeParentCrossesSequence {
        /// Token index in the flattened batch.
        token: usize,
        /// Parent index supplied by the scheduler.
        parent: i32,
        /// Sequence index.
        seq: usize,
        /// First flattened token belonging to the sequence.
        seq_start: usize,
        /// Exclusive end of the sequence's flattened token range.
        seq_end: usize,
    },

    /// The ancestor-mask column count is too small for the longest sequence
    /// query in the batch (Spec 1 §4.D.1).
    #[error("tree t_max={actual} is smaller than required {required} (Spec 1 §4.D.1)")]
    TreeMaxTooSmall {
        /// Longest query length in the batch.
        required: u32,
        /// Supplied mask width.
        actual: u32,
    },

    /// A matrix-op descriptor uses an accumulator dtype outside the closed
    /// f32/i32 set (Spec 1 §6.1, App. A).
    #[error("matrix accumulator dtype {got} must be f32 or i32 (Spec 1 §6.1, App. A)")]
    InvalidAccumulator {
        /// Rejected accumulator dtype.
        got: DType,
    },

    /// A relative matrix-throughput rate is not finite and positive
    /// (Spec 1 App. A).
    #[error("invalid relative rate {value}: must be finite and > 0 (Spec 1 App. A)")]
    NonPositiveRate {
        /// Rejected value.
        value: f32,
    },

    /// An op received an unexpected number of input tensors (Spec 1 §4).
    #[error("op `{op}` expected {expected} inputs, got {got} (Spec 1 §4)")]
    OpInputCountMismatch {
        /// Op name.
        op: &'static str,
        /// Expected input count.
        expected: usize,
        /// Actual input count.
        got: usize,
    },

    /// An op received an unexpected number of input tensors when multiple counts are accepted (Spec 1 §4).
    #[error("op `{op}` expected one of {expected:?} inputs, got {got} (Spec 1 §4)")]
    OpInputCountCandidatesMismatch {
        /// Op name.
        op: &'static str,
        /// Accepted input counts.
        expected: Box<[usize]>,
        /// Actual input count.
        got: usize,
    },

    /// An op received an unexpected number of output tensors (Spec 1 §4).
    #[error("op `{op}` expected {expected} outputs, got {got} (Spec 1 §4)")]
    OpOutputCountMismatch {
        /// Op name.
        op: &'static str,
        /// Expected output count.
        expected: usize,
        /// Actual output count.
        got: usize,
    },

    /// An op received an unexpected number of output tensors when multiple counts are accepted (Spec 1 §4).
    #[error("op `{op}` expected one of {expected:?} outputs, got {got} (Spec 1 §4)")]
    OpOutputCountCandidatesMismatch {
        /// Op name.
        op: &'static str,
        /// Accepted output counts.
        expected: Box<[usize]>,
        /// Actual output count.
        got: usize,
    },

    /// An op tensor has an unexpected dtype (Spec 1 §4).
    #[error("op `{op}` tensor `{tensor}` expected dtype in {expected:?}, got {got} (Spec 1 §4)")]
    OpDTypeMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor role or parameter name.
        tensor: &'static str,
        /// Allowed dtypes.
        expected: Box<[DType]>,
        /// Actual dtype.
        got: DType,
    },

    /// An op tensor has an unexpected rank (Spec 1 §4).
    #[error("op `{op}` tensor `{tensor}` expected rank {expected}, got {got} (Spec 1 §4)")]
    OpRankMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor role or parameter name.
        tensor: &'static str,
        /// Expected rank.
        expected: usize,
        /// Actual rank.
        got: usize,
    },

    /// An op tensor has a shape mismatch against constraints (Spec 1 §4).
    #[error("op `{op}` tensor `{tensor}` shape mismatch: {detail} (Spec 1 §4)")]
    OpShapeMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor role or parameter name.
        tensor: &'static str,
        /// Detail of the mismatch.
        detail: String,
    },

    /// An op tensor has an unexpected layout (Spec 1 §4).
    #[error("op `{op}` tensor `{tensor}` expected layout {expected}, got {got} (Spec 1 §4)")]
    OpLayoutMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor role or parameter name.
        tensor: &'static str,
        /// Expected layout.
        expected: LayoutId,
        /// Actual layout.
        got: LayoutId,
    },

    /// An op tensor has an unexpected quantization scheme (Spec 1 §4).
    #[error("op `{op}` tensor `{tensor}` unexpected quant scheme {quant:?} (Spec 1 §4)")]
    OpQuantMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor role or parameter name.
        tensor: &'static str,
        /// Actual quant scheme.
        quant: QuantScheme,
    },

    /// An op tensor has an illegal placement (Spec 1 §4).
    #[error("op `{op}` tensor `{tensor}` illegal placement {placement} (Spec 1 §4)")]
    OpPlacementMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor role or parameter name.
        tensor: &'static str,
        /// Actual placement.
        placement: Placement,
    },

    /// An op tensor has an unexpected class (Spec 1 §4).
    #[error("op `{op}` tensor `{tensor}` expected class {expected}, got {got} (Spec 1 §4)")]
    OpClassMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor role or parameter name.
        tensor: &'static str,
        /// Expected class.
        expected: Class,
        /// Actual class.
        got: Class,
    },

    /// An op attribute violates its specification (Spec 1 §4).
    #[error("op `{op}` attribute `{attribute}` invalid: {reason} (Spec 1 §4)")]
    OpAttributeInvalid {
        /// Op name.
        op: &'static str,
        /// Attribute name.
        attribute: &'static str,
        /// Failure reason.
        reason: String,
    },

    /// An op received an incompatible state handle kind (Spec 1 §4.D, §4.E).
    #[error("op `{op}` expected state kind {expected:?}, got {got:?} (Spec 1 §4)")]
    StateHandleKindMismatch {
        /// Op name.
        op: &'static str,
        /// Expected state kind.
        expected: crate::StateKind,
        /// Actual state kind.
        got: crate::StateKind,
    },

    /// A batch axis value exceeded the maximum bucket size of 4096 (Spec 1 §3.5).
    #[error("axis `{axis}` value {value} exceeds max bucket {max} (Spec 1 §3.5)")]
    BucketExceeded {
        /// Axis name.
        axis: &'static str,
        /// Actual value.
        value: u32,
        /// Maximum bucket size (4096).
        max: u32,
    },

    /// A bucket value is not valid for the given axis (Spec 1 §3.5).
    #[error("invalid bucket value {value} for axis `{axis}` (Spec 1 §3.5)")]
    InvalidBucket {
        /// Axis name.
        axis: &'static str,
        /// Value.
        value: u32,
    },

    /// Graph contains a cycle (Spec 1 §3.1).
    #[error("graph contains a cycle involving node {node} (Spec 1 §3.1)")]
    GraphCycle {
        /// Offending node index.
        node: usize,
    },

    /// Referenced graph node does not exist.
    #[error("graph node {node} not found")]
    GraphNodeNotFound {
        /// Missing node id.
        node: usize,
    },

    /// Referenced graph edge does not exist.
    #[error("graph edge {edge} not found")]
    GraphEdgeNotFound {
        /// Missing edge id.
        edge: usize,
    },

    /// A graph op requires a structured external input that was not bound
    /// (Spec 1 §3.2, §4).
    #[error("graph op `{required_by}` requires external input {kind:?} (Spec 1 §3.2, §4)")]
    GraphExternalInputMissing {
        /// Missing structured input kind.
        kind: crate::graph::ExternalInputKind,
        /// Op that requires the input.
        required_by: &'static str,
    },

    /// A graph op mutates structured state but its external output was not
    /// declared (Spec 1 §3.2, §4.F).
    #[error("graph op `{required_by}` requires external output {kind:?} (Spec 1 §3.2, §4.F)")]
    GraphExternalOutputMissing {
        /// Missing structured output kind.
        kind: crate::graph::ExternalOutputKind,
        /// Op that requires the output.
        required_by: &'static str,
    },

    /// An external output was bound to an edge that is not produced by an op
    /// in this graph (Spec 1 §3.1–§3.2).
    #[error("external output {kind:?} cannot bind unproduced edge {edge} (Spec 1 §3.1–§3.2)")]
    GraphExternalOutputUnproduced {
        /// External output kind being bound.
        kind: crate::graph::ExternalOutputKind,
        /// Edge without an operation producer.
        edge: usize,
    },

    /// A graph-owned source tensor uses a class reserved for request-time or
    /// operation-produced data (Spec 1 §2.3, §3.1–§3.2).
    #[error("graph-owned source edge cannot use tensor class {class} (Spec 1 §2.3, §3.1–§3.2)")]
    GraphSourceClassInvalid {
        /// Rejected tensor class.
        class: Class,
    },

    /// A second `BatchMeta.positions` projection was bound when one already
    /// exists (Spec 1 §2.5; card A1.14). A step graph carries exactly one
    /// structured `BatchMeta`, so its positions projection is bound at most
    /// once: the same kind twice is a duplicate, a different kind is a
    /// conflict.
    #[error("positions projection already bound as {existing:?}; cannot bind {requested:?} (Spec 1 §2.5: one BatchMeta per step graph)")]
    PositionsConflict {
        /// Kind of the already-bound projection.
        existing: crate::graph::PositionsKind,
        /// Kind the caller attempted to bind.
        requested: crate::graph::PositionsKind,
    },

    /// A graph op reads `BatchMeta.positions` but no positions projection was
    /// bound (Spec 1 §2.5, §4.B; card A1.14).
    #[error("graph op `{required_by}` requires a bound BatchMeta.positions projection (Spec 1 §2.5, §4.B)")]
    GraphPositionsMissing {
        /// Op that requires the projection.
        required_by: &'static str,
    },

    /// A `rope` node reads its positions from an edge that is not the bound
    /// `BatchMeta.positions` projection (Spec 1 §2.5, §4.B; card A1.14).
    #[error("rope node {node} reads positions from edge {edge}, expected positions projection edge {expected} (Spec 1 §2.5, §4.B)")]
    GraphRopePositionsMismatch {
        /// Rope node index.
        node: usize,
        /// Edge the node actually reads positions from.
        edge: usize,
        /// Bound positions projection edge.
        expected: usize,
    },

    /// A tensor edge is placed on a device other than the graph capture rank
    /// (Spec 1 §3.1).
    #[error("graph edge {edge} is on device rank {tensor_rank}, expected graph rank {graph_rank} (Spec 1 §3.1)")]
    GraphTensorRankMismatch {
        /// Edge carrying the mismatched tensor.
        edge: usize,
        /// Rank in the tensor placement.
        tensor_rank: u32,
        /// Rank in the step-graph key.
        graph_rank: u32,
    },

    /// A stride mismatch occurred between actual and expected layout (Spec 1 §3.3).
    #[error(
        "stride mismatch on edge {edge}: actual {actual:?}, expected {expected:?} (Spec 1 §3.3)"
    )]
    StrideMismatch {
        /// Edge id.
        edge: usize,
        /// Actual strides.
        actual: Box<[i64]>,
        /// Expected contiguous strides.
        expected: Box<[i64]>,
    },

    /// Collect-all wrapper: every problem found before returning
    /// (CONVENTIONS.md §1.4). Constructors return the single problem directly
    /// when only one exists.
    #[error("multiple validation failures: {problems:?}")]
    Multiple {
        /// Every problem found, in deterministic field order.
        problems: Box<[IrError]>,
    },
}

impl IrError {
    /// Collapses a list of accumulated errors into `Ok(())`, single `Err(e)`,
    /// or [`IrError::Multiple`] per CONVENTIONS.md §1.4.
    pub fn from_problems(mut problems: Vec<IrError>) -> Result<(), Self> {
        if problems.is_empty() {
            Ok(())
        } else if problems.len() == 1 {
            Err(problems.pop().expect("problems holds exactly one entry"))
        } else {
            Err(Self::Multiple {
                problems: problems.into_boxed_slice(),
            })
        }
    }
}
