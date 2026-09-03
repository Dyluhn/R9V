// SPDX-License-Identifier: Apache-2.0
//! Top-level error types for R9V (Spec 14 §2, Spec 15 §3, CONVENTIONS.md §1).

use crate::bytes::ByteSizeError;

/// Result type alias with [`R9vError`] as the default error type (Spec 14 §2).
pub type Result<T, E = R9vError> = std::result::Result<T, E>;

/// Top-level error type representing failures across the R9V engine (Spec 14 §2, CONVENTIONS.md §1.2).
///
/// At this card stage, [`R9vError`] strictly wraps common errors shared across crates.
#[derive(Debug, thiserror::Error)]
pub enum R9vError {
    /// Byte-size parsing error.
    #[error("byte size parse error: {0}")]
    ByteSize(#[from] ByteSizeError),

    /// Standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
