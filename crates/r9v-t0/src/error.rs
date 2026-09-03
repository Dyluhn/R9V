// SPDX-License-Identifier: Apache-2.0
//! Error types for the R9V T0 CPU reference implementation
//! (Spec 1 §4.B, §4.F, Spec 4 §2, CONVENTIONS.md §1, Cards A1.5 and A1.8).

use r9v_ir::DType;

/// Domain error enum for T0 reference op execution (Spec 4 §2, CONVENTIONS.md §1.1).
#[derive(Debug, thiserror::Error)]
pub enum T0Error {
    /// Wrapping an IR-level error from `r9v-ir`.
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),

    /// Wrapping a format-level error from `r9v-format`.
    #[error(transparent)]
    Format(#[from] r9v_format::FormatError),

    /// Tensor quantization scheme mismatch.
    #[error("tensor `{tensor}`: expected quant scheme {expected:?}, got {got:?}")]
    QuantMismatch {
        /// Name of the affected tensor operand.
        tensor: &'static str,
        /// Expected legal quantization schemes.
        expected: Vec<r9v_ir::QuantScheme>,
        /// Observed quantization scheme.
        got: r9v_ir::QuantScheme,
    },

    /// Tensor layout mismatch.
    #[error("tensor `{tensor}`: expected layout {expected:?}, got {got:?}")]
    LayoutMismatch {
        /// Name of the affected tensor operand.
        tensor: &'static str,
        /// Expected legal layout ids.
        expected: Vec<r9v_ir::LayoutId>,
        /// Observed layout id.
        got: r9v_ir::LayoutId,
    },

    /// A row or token index falls outside valid bounds.
    #[error(
        "index {index} at `{tensor}[{position}]` is outside bounds 0..{upper_bound} in op `{op}`"
    )]
    RowIndexOutOfRange {
        /// Op name.
        op: &'static str,
        /// Tensor name.
        tensor: &'static str,
        /// Flat position of the invalid index.
        position: usize,
        /// Observed invalid index.
        index: u32,
        /// Upper bound.
        upper_bound: usize,
    },

    /// Numeric or shape calculation overflow.
    #[error("arithmetic overflow in op `{op}`: {detail}")]
    ArithmeticOverflow {
        /// Op name.
        op: &'static str,
        /// Detail of what overflowed.
        detail: String,
    },

    /// Tensor rank mismatch.
    #[error("tensor `{tensor}`: expected rank {expected}, got {got} with shape {shape:?}")]
    RankMismatch {
        /// Name of the affected tensor operand.
        tensor: &'static str,
        /// Expected tensor rank.
        expected: usize,
        /// Observed tensor rank.
        got: usize,
        /// Observed full shape.
        shape: Vec<usize>,
    },

    /// Tensor element data type mismatch.
    #[error("tensor `{tensor}`: expected dtype {expected:?}, got {got:?}")]
    DTypeMismatch {
        /// Name of the affected tensor operand.
        tensor: &'static str,
        /// Expected legal data types.
        expected: Vec<DType>,
        /// Observed data type.
        got: DType,
    },

    /// Buffer length in bytes or elements does not match expected extent.
    #[error("tensor `{tensor}`: buffer element count {buffer_len} does not match expected element count {expected_len} for shape {shape:?}")]
    BufferLengthMismatch {
        /// Name of the affected tensor operand.
        tensor: &'static str,
        /// Number of elements available in buffer.
        buffer_len: usize,
        /// Number of elements required by the shape.
        expected_len: usize,
        /// Tensor shape.
        shape: Vec<usize>,
    },

    /// Shape dimension mismatch between related tensors.
    #[error("dimension mismatch on `{dim_name}`: expected {expected} (from `{expected_from}`), got {got} in `{tensor}`")]
    DimensionMismatch {
        /// Symbolic or named dimension (e.g. "T", "N", "Dff", "D").
        dim_name: &'static str,
        /// Source operand setting the expected dimension extent.
        expected_from: &'static str,
        /// Expected dimension size.
        expected: usize,
        /// Failing tensor operand name.
        tensor: &'static str,
        /// Observed dimension size.
        got: usize,
    },

    /// Operation attribute invalid for execution.
    #[error("invalid attribute `{attribute}` for op `{op}`: {reason}")]
    InvalidAttribute {
        /// Name of the operation.
        op: &'static str,
        /// Name of the attribute.
        attribute: &'static str,
        /// Detailed reason.
        reason: String,
    },

    /// Missing required optional operand.
    #[error("op `{op}` requires missing operand `{operand}`: {detail}")]
    MissingOperand {
        /// Name of the operation.
        op: &'static str,
        /// Missing operand name.
        operand: &'static str,
        /// Contextual reason.
        detail: String,
    },

    /// Unsupported precision cast.
    #[error("unsupported cast from `{from:?}` to `{to:?}`")]
    UnsupportedCast {
        /// Source data type.
        from: DType,
        /// Destination data type.
        to: DType,
    },

    /// Validated same-dtype operands use incompatible backing representations.
    #[error("op `{op}` has no bit-exact backing conversion for dtype `{dtype:?}`")]
    BackingRepresentationMismatch {
        /// Operation that requires a bit-exact transfer.
        op: &'static str,
        /// Shared logical data type of the operands.
        dtype: DType,
    },

    /// Multiple validation errors collected together (CONVENTIONS.md §1.4).
    #[error("{count} validation error(s) in op `{op}`: {problems:?}")]
    Multiple {
        /// Name of the failing op.
        op: &'static str,
        /// Count of collected problems.
        count: usize,
        /// Vector of formatted problem descriptions.
        problems: Vec<String>,
    },

    /// Flat-slice length mismatch for sampling ops (Spec 1 §4.F).
    ///
    /// Renamed from the A1.8 `DimensionMismatch` so it cannot collide with the
    /// A1.5 symbolic-dimension `DimensionMismatch` above, which carries
    /// `(dim_name, expected_from, ...)` instead of `(op, detail)`.
    #[error("shape length mismatch in {op}: tensor {tensor} expected length {expected}, got {got}; detail: {detail}")]
    ShapeLengthMismatch {
        /// Op name.
        op: &'static str,
        /// Tensor name.
        tensor: &'static str,
        /// Expected size/length.
        expected: usize,
        /// Actual size/length.
        got: usize,
        /// Extra detail.
        detail: String,
    },

    /// Empty tensor or slice where non-empty was required.
    #[error("empty tensor in {op}: {tensor} has length 0")]
    EmptyInput {
        /// Op name.
        op: &'static str,
        /// Tensor name.
        tensor: &'static str,
    },

    /// All tokens masked out by grammar mask.
    #[error("all tokens masked out by grammar mask in sequence {seq}, query {query}; vocabulary size was {vocab_size}")]
    AllTokensMasked {
        /// Sequence index.
        seq: usize,
        /// Query index.
        query: usize,
        /// Vocab size.
        vocab_size: usize,
    },

    /// Invalid probability distribution.
    #[error("invalid probability distribution in {op} for sequence {seq}, position {pos}: sum is {sum}, expected positive finite sum")]
    InvalidDistribution {
        /// Op name.
        op: &'static str,
        /// Sequence index.
        seq: usize,
        /// Position index.
        pos: usize,
        /// Observed sum.
        sum: f32,
    },

    /// A token id falls outside the vocabulary addressed by an operation.
    #[error(
        "token id {token} at {tensor}[{position}] is outside vocabulary 0..{vocab_size} in {op}"
    )]
    TokenOutOfRange {
        /// Op name.
        op: &'static str,
        /// Tensor or parameter name.
        tensor: &'static str,
        /// Flat position of the invalid token id.
        position: usize,
        /// Invalid token id.
        token: u32,
        /// Vocabulary size.
        vocab_size: usize,
    },

    /// A probability is negative or non-finite.
    #[error(
        "invalid probability {value} at token {token} in {op}, sequence {seq}, position {pos}"
    )]
    InvalidProbability {
        /// Op name.
        op: &'static str,
        /// Sequence index.
        seq: usize,
        /// Position index.
        pos: usize,
        /// Token index.
        token: usize,
        /// Invalid probability.
        value: f32,
    },

    /// Invalid tree structure.
    #[error("invalid tree draft structure for sequence {seq}: {detail}")]
    InvalidTree {
        /// Sequence index.
        seq: usize,
        /// Reason for failure.
        detail: String,
    },

    /// Multiple coexisting typed validation problems (Spec 1 §4.F).
    ///
    /// Renamed from the A1.8 `Multiple` so it cannot collide with the A1.5
    /// `Multiple` above, which carries `(op, count, problems: Vec<String>)`.
    #[error("multiple T0 validation problems ({} failures): {problems:?}", problems.len())]
    MultipleErrors {
        /// Accumulated problems.
        problems: Box<[T0Error]>,
    },
}

impl T0Error {
    /// Helper to produce a `Multiple` error if problems were collected (Spec 4 §2, CONVENTIONS.md §1.4).
    pub fn from_problems(op: &'static str, problems: Vec<String>) -> Result<(), Self> {
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Self::Multiple {
                op,
                count: problems.len(),
                problems,
            })
        }
    }

    /// Aggregates typed problems: `Ok(())` if empty, the single error if one,
    /// or `MultipleErrors` otherwise (Spec 1 §4.F).
    ///
    /// Renamed from the A1.8 `from_problems` so it cannot collide with the
    /// A1.5 `from_problems(op, problems: Vec<String>)` above.
    pub fn from_typed_problems(mut problems: Vec<T0Error>) -> Result<(), Self> {
        if problems.is_empty() {
            Ok(())
        } else if problems.len() == 1 {
            // Invariant: length was just checked to be exactly 1, so pop cannot fail.
            Err(problems.pop().expect("exactly one problem"))
        } else {
            Err(Self::MultipleErrors {
                problems: problems.into_boxed_slice(),
            })
        }
    }
}

/// Helper to convert a `u64` to `usize`, returning `T0Error::ArithmeticOverflow` if it overflows.
#[inline(always)]
pub fn u64_to_usize(val: u64, what: &'static str) -> Result<usize, T0Error> {
    usize::try_from(val).map_err(|_| T0Error::ArithmeticOverflow {
        op: "u64_to_usize",
        detail: format!("{what} value {val} overflows usize"),
    })
}
