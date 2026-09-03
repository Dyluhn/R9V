// SPDX-License-Identifier: Apache-2.0
//! Typed errors for HIP operations (Spec 14 §2, §3, CONVENTIONS.md §1).

// DECISION(A0.2): HipError is crate-local and composes upward at the engine integration boundary; rejected adding HIP-specific error variants to r9v-common because Spec 14 §2 strictly forbids upward dependencies from foundational crates.

/// Result alias using crate-local [`HipError`] (Spec 14 §3).
pub type Result<T> = r9v_common::Result<T, HipError>;

/// Errors arising from dynamic HIP runtime interactions (Spec 14 §3).
#[derive(Debug, Clone, thiserror::Error)]
pub enum HipError {
    /// The HIP dynamic library (`libamdhip64`) could not be located or loaded.
    #[error("HIP runtime library not found; searched: {searched:?}")]
    LibraryNotFound {
        /// Candidate search paths attempted during dlopen with reason for failure.
        searched: Vec<String>,
    },

    /// A required HIP symbol could not be resolved from the dynamic library.
    #[error("HIP symbol '{symbol}' could not be resolved: {details}")]
    SymbolNotFound {
        /// Name of the missing symbol.
        symbol: &'static str,
        /// Description of the lookup failure.
        details: String,
    },

    /// A HIP runtime API call returned a non-zero error code.
    #[error("HIP API '{op}' failed with status {code}: {message}")]
    ApiError {
        /// Name of the HIP operation (e.g. `hipMalloc`).
        op: &'static str,
        /// Numeric status code returned by HIP runtime (`hipError_t`).
        code: i32,
        /// Human-readable description of the error code.
        message: String,
    },

    /// A null pointer was unexpectedly returned or passed.
    #[error("unexpected null pointer in HIP operation: {0}")]
    NullPointer(&'static str),

    /// A string argument contained an interior NUL byte.
    #[error("interior NUL byte in string for {context} at byte index {nul_position}")]
    InvalidNulByte {
        /// Description of the string argument context.
        context: &'static str,
        /// Position within string where interior NUL byte occurred.
        nul_position: usize,
    },

    /// A destination buffer or slice is too small for the requested operation.
    #[error(
        "buffer too small for {operation}: required {required} bytes, available {available} bytes"
    )]
    BufferTooSmall {
        /// The operation that requested buffer capacity.
        operation: &'static str,
        /// Minimum byte capacity required.
        required: usize,
        /// Byte capacity available in the destination.
        available: usize,
    },

    /// The driver or runtime reported a negative or invalid device count.
    #[error("driver reported invalid negative device count: {count}")]
    InvalidDeviceCount {
        /// The invalid device count value returned by the driver.
        count: i32,
    },
}

impl HipError {
    pub(crate) fn api_error(op: &'static str, code: i32, desc: Option<&str>) -> Self {
        let message = desc
            .map(|s| s.to_owned())
            .unwrap_or_else(|| format!("hipError_{code}"));
        Self::ApiError { op, code, message }
    }
}
