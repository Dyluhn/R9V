// SPDX-License-Identifier: Apache-2.0
//! Error types for model definitions and architecture building (Spec 8; card A1.3).
//!
//! Adheres to `CONVENTIONS.md` §1 (per-crate `ModelsError`, collect-all validation,
//! no panics on untrusted input).

use thiserror::Error;

/// Domain error type for `r9v-models` (Spec 8; CONVENTIONS.md §1.1).
#[derive(Debug, Error)]
pub enum ModelsError {
    /// Underlying IR error from graph construction, tensor validation, or op checking.
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),

    /// Common infrastructure error (hashing, byte sizes, identifiers).
    #[error(transparent)]
    Common(#[from] r9v_common::R9vError),

    /// Underlying sequence-state error (CONVENTIONS.md §1.1).
    #[error(transparent)]
    State(#[from] r9v_state::StateError),

    /// Missing required metadata key.
    #[error("missing metadata key '{key}'; expected type {expected_type}")]
    MissingMetaKey {
        /// Metadata key path.
        key: String,
        /// Expected data type description.
        expected_type: &'static str,
    },

    /// Metadata value type mismatch.
    #[error("metadata key '{key}' type mismatch: expected {expected}, found {found}")]
    MetaTypeMismatch {
        /// Metadata key path.
        key: String,
        /// Expected type name.
        expected: &'static str,
        /// Description of found value.
        found: String,
    },

    /// Missing weight tensor during validation or building.
    #[error("missing required weight tensor '{name}'")]
    MissingTensor {
        /// Tensor name.
        name: String,
    },

    /// Tensor shape mismatch.
    #[error("tensor '{name}' shape mismatch: expected {expected:?}, got {actual:?}")]
    TensorShapeMismatch {
        /// Tensor name.
        name: String,
        /// Expected dimension extents.
        expected: Vec<u64>,
        /// Actual dimension extents.
        actual: Vec<u64>,
    },

    /// Invalid model specification.
    #[error("invalid model spec: {reason}")]
    InvalidModelSpec {
        /// Diagnostic reason.
        reason: String,
    },

    /// Invalid layer specification.
    #[error("invalid layer spec at layer {layer}: {reason}")]
    InvalidLayerSpec {
        /// Zero-based layer index.
        layer: u32,
        /// Diagnostic reason.
        reason: String,
    },

    /// Invalid dimension constraint.
    #[error("dimension '{name}' value {value} violates constraint: {requirement}")]
    InvalidDimension {
        /// Dimension name.
        name: &'static str,
        /// Value provided.
        value: u32,
        /// Requirement violated.
        requirement: &'static str,
    },

    /// Unknown architecture string in metadata.
    #[error("unknown architecture '{arch}'; nearest family is '{nearest}'")]
    UnknownArchitecture {
        /// Unknown architecture string.
        arch: String,
        /// Nearest known family name.
        nearest: &'static str,
    },

    /// Subgraph not found or already exists.
    #[error("subgraph error for '{name}': {reason}")]
    SubgraphError {
        /// Subgraph name.
        name: String,
        /// Diagnostic reason.
        reason: String,
    },

    /// Tensor shape access failure while lowering (rank too small for the
    /// requested dimension index, or an expected output missing).
    #[error("tensor shape access failed in '{context}': {reason}")]
    ShapeAccess {
        /// Builder helper or op being lowered.
        context: String,
        /// What was requested and the rank found, with every value reported.
        reason: String,
    },

    /// Untrusted dimension arithmetic overflowed instead of wrapping.
    #[error("dimension arithmetic overflow in '{context}': {operation}")]
    ArithmeticOverflow {
        /// Builder helper or op being lowered.
        context: String,
        /// Operation with operands, e.g. `8 * 4294967295`.
        operation: String,
    },

    /// Multiple collected errors (CONVENTIONS.md §1.4).
    #[error("{} model building problem(s): {problems:?}", problems.len())]
    Multiple {
        /// All collected problems.
        problems: Vec<ModelsError>,
    },
}

impl ModelsError {
    /// Collects a list of problems into a single `Result` (CONVENTIONS.md §1.4).
    pub fn from_problems(problems: Vec<ModelsError>) -> Result<(), Self> {
        if problems.is_empty() {
            Ok(())
        } else {
            let mut iter = problems.into_iter();
            let first = iter.next().unwrap_or(Self::InvalidModelSpec {
                reason: "from_problems called with an empty list after the empty check".to_string(),
            });
            let rest: Vec<ModelsError> = iter.collect();
            if rest.is_empty() {
                Err(first)
            } else {
                let mut all = Vec::with_capacity(rest.len() + 1);
                all.push(first);
                all.extend(rest);
                Err(Self::Multiple { problems: all })
            }
        }
    }
}
