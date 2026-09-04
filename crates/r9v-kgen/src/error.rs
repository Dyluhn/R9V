// SPDX-License-Identifier: Apache-2.0
//! Domain-specific error types for the kernel generator and ABI generator (Spec 4 §7, CONVENTIONS §1.1).

use r9v_registry::OpId;

/// Domain error type for kernel generation and kernel ABI generation (Spec 4 §7, CONVENTIONS §1.1).
#[derive(Debug, thiserror::Error)]
pub enum KgenError {
    /// Collect-all validation failure (CONVENTIONS §1.4).
    #[error("validation failed with {} problem(s): {problems:?}", problems.len())]
    ValidationFailed {
        /// All validation problems accumulated.
        problems: Vec<String>,
    },

    /// Arithmetic overflow in layout or offset calculation (Spec 4 §7).
    #[error("arithmetic overflow in {context}: {lhs} {op} {rhs}")]
    ArithmeticOverflow {
        /// Context description of where the overflow occurred.
        context: &'static str,
        /// Left-hand side operand.
        lhs: usize,
        /// Operation symbol.
        op: &'static str,
        /// Right-hand side operand.
        rhs: usize,
    },

    /// An operation does not match the provided static parameter family (Spec 4 §7).
    #[error("operation '{op}' is incompatible with static family '{family}'")]
    MismatchedOpFamily {
        /// Operation identifier.
        op: OpId,
        /// ABI family name of the provided static descriptor.
        family: &'static str,
    },

    /// A shared static parameter family cannot determine OpId without explicit OpId (Spec 4 §4.1).
    #[error("ambiguous op family '{family}': cannot determine OpId without explicit OpId; valid ops are: {valid_ops:?}")]
    AmbiguousOpFamily {
        /// ABI family name.
        family: &'static str,
        /// Valid operation identifiers belonging to this family.
        valid_ops: Vec<OpId>,
    },

    /// Inconsistent variant collision where two inputs share the same variant name but differ (Spec 4 §7).
    #[error("inconsistent variant collision for '{name}': {details}")]
    InconsistentVariantCollision {
        /// Variant struct name.
        name: String,
        /// Inconsistency details.
        details: String,
    },

    /// An operation is unsupported for the requested ABI family.
    #[error("unsupported op '{op:?}' for ABI family '{family}'")]
    UnsupportedOp {
        /// Operation identifier.
        op: OpId,
        /// ABI family name.
        family: &'static str,
    },

    /// Alignment validation failure.
    #[error("alignment error for {context}: required multiple of {required}, got {actual}")]
    AlignmentError {
        /// Context description of alignment error.
        context: &'static str,
        /// Expected multiple.
        required: usize,
        /// Actual alignment or offset.
        actual: usize,
    },

    /// Empty argument struct is invalid (every op variant requires arguments).
    #[error("empty ABI struct for op '{op:?}'")]
    EmptyStruct {
        /// Operation identifier.
        op: OpId,
    },

    /// HIP compilation error during layout verification.
    #[error("HIP compilation error: {details}")]
    CompileError {
        /// Failure details and compiler output.
        details: String,
    },

    /// Layout discrepancy between Rust struct and HIP struct or canonical description.
    #[error("layout mismatch for struct '{name}': {reason}")]
    LayoutMismatch {
        /// Name of the struct with mismatched layout.
        name: String,
        /// Description of the layout mismatch.
        reason: String,
    },

    /// Wrapped Op IR error (CONVENTIONS §1.1).
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),

    /// Wrapped Registry error (CONVENTIONS §1.1).
    #[error(transparent)]
    Registry(#[from] r9v_registry::RegistryError),

    /// Wrapped I/O error (CONVENTIONS §1.1).
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
