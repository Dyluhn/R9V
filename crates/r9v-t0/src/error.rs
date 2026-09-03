// SPDX-License-Identifier: Apache-2.0
//! Error types for the R9V T0 CPU reference implementation (Spec 4 §2, CONVENTIONS.md §1).

use r9v_ir::DType;

/// Domain error enum for T0 reference op execution (Spec 4 §2, CONVENTIONS.md §1.1).
#[derive(Debug, thiserror::Error)]
pub enum T0Error {
    /// Wrapping an IR-level error from `r9v-ir`.
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),

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
}
