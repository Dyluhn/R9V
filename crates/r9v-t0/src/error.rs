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

    /// Flat-slice length mismatch for sampling ops (Spec 1 §4.F).
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

    /// All tokens masked out by grammar mask or invalid logits.
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

    /// A logit value is NaN or +Inf (Spec 1 §4.F).
    ///
    /// `-Inf` is never reported through this variant: it is the intentional
    /// encoding of masked or impossible tokens (grammar mask, `logit_bias`
    /// to `-Inf`). NaN and `+Inf` can never be intentional and are rejected
    /// with their exact `(seq, query, token)` location, both on the raw input
    /// and after bias/penalty/temperature transforms (overflow).
    #[error("invalid logit {value} at sequence {seq}, query {query}, token {token} in op `{op}`")]
    InvalidLogit {
        /// Op name.
        op: &'static str,
        /// Sequence index.
        seq: usize,
        /// Query index.
        query: usize,
        /// Token index within the vocabulary.
        token: usize,
        /// Invalid logit value.
        value: f32,
    },

    /// An RNG sequence id exceeds the canonical u32 Philox counter word (Spec 1 §4.F).
    ///
    /// The 128-bit Philox counter carries the sequence id in one 32-bit word,
    /// so ids above `u32::MAX` cannot be represented without truncation
    /// collision. RNG construction rejects it before a usable state exists.
    #[error("rng seq_id {seq_id} exceeds the canonical u32 Philox counter word (max {max}) in op `{op}`")]
    SeqIdOutOfRange {
        /// Op name.
        op: &'static str,
        /// Offending 64-bit sequence id.
        seq_id: u64,
        /// Maximum representable sequence id.
        max: u64,
    },

    /// Advancing an RNG state would wrap its per-step draw counter (Spec 1 §4.F).
    #[error(
        "rng draw index {draw_index} cannot advance by {advance} without overflow in op `{op}`"
    )]
    DrawIndexOverflow {
        /// Operation that requested the advance.
        op: &'static str,
        /// Current draw index.
        draw_index: u32,
        /// Requested advance.
        advance: u32,
    },

    /// Invalid tree structure.
    #[error("invalid tree draft structure for sequence {seq}: {detail}")]
    InvalidTree {
        /// Sequence index.
        seq: usize,
        /// Reason for failure.
        detail: String,
    },

    /// Multiple coexisting typed validation problems (Spec 1 §4.F, CONVENTIONS.md §1.4).
    #[error("multiple validation error(s) ({} failures): {problems:?}", problems.len())]
    Multiple {
        /// Accumulated typed problems.
        problems: Box<[T0Error]>,
    },
}

impl T0Error {
    /// Aggregates typed problems: `Ok(())` if empty, the single error if one,
    /// or `Multiple` otherwise (CONVENTIONS.md §1.4).
    pub fn from_problems(mut problems: Vec<T0Error>) -> Result<(), Self> {
        match problems.len() {
            0 => Ok(()),
            1 => {
                if let Some(problem) = problems.pop() {
                    Err(problem)
                } else {
                    Ok(())
                }
            }
            _ => Err(Self::Multiple {
                problems: problems.into_boxed_slice(),
            }),
        }
    }
}

/// Positional dimension names for shape-agreement checks on ops whose dimensions
/// carry no symbolic name (Spec 4 §2).
const POS_DIM_NAMES: [&str; 8] = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"];

/// Pushes typed shape-agreement problems between two shapes (Spec 4 §2, CONVENTIONS.md §1.4).
///
/// Pushes `RankMismatch` when the ranks differ, otherwise one
/// `DimensionMismatch` per disagreeing dimension. Used by elementwise ops to
/// replace stringly formatted shape problems with typed errors.
pub fn push_shape_agreement(
    problems: &mut Vec<T0Error>,
    tensor: &'static str,
    expected_from: &'static str,
    actual_shape: &[usize],
    expected_shape: &[usize],
) {
    if actual_shape.len() != expected_shape.len() {
        problems.push(T0Error::RankMismatch {
            tensor,
            expected: expected_shape.len(),
            got: actual_shape.len(),
            shape: actual_shape.to_vec(),
        });
        return;
    }
    for (i, (&got, &expected)) in actual_shape.iter().zip(expected_shape.iter()).enumerate() {
        if got != expected {
            problems.push(T0Error::DimensionMismatch {
                dim_name: POS_DIM_NAMES.get(i).copied().unwrap_or("dim"),
                expected_from,
                expected,
                tensor,
                got,
            });
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
