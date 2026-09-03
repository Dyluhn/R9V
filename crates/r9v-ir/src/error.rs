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

    /// Collect-all wrapper: every problem found before returning
    /// (CONVENTIONS.md §1.4). Constructors return the single problem directly
    /// when only one exists.
    #[error("multiple validation failures: {problems:?}")]
    Multiple {
        /// Every problem found, in deterministic field order.
        problems: Box<[IrError]>,
    },
}
