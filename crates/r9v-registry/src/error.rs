// SPDX-License-Identifier: Apache-2.0
//! Error types for the R9V kernel registry and bundle manager (Spec 4, CONVENTIONS.md §1).

use crate::types::OpId;

/// Errors produced by the R9V kernel registry, bundle loader, and resolution engine (Spec 4).
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Target architecture is unlisted and cannot execute without a runtime JIT compiler (Spec 4 §9.2).
    #[error("architecture '{arch}' is unlisted; supported architectures: {supported:?}")]
    UnlistedArchRefused {
        /// The requested unlisted architecture name.
        arch: String,
        /// The list of supported architectures known to the registry.
        supported: Vec<String>,
    },

    /// Bundle manifest generator version does not match expected version (Spec 4 §11).
    #[error("generator version mismatch: expected {expected}, got {got}{}", path.as_deref().map(|p| format!(" in {p}")).unwrap_or_default())]
    GenVersionMismatch {
        /// Expected generator version.
        expected: u32,
        /// Actual generator version in manifest or tune file.
        got: u32,
        /// Optional path to the offending file.
        path: Option<String>,
    },

    /// No kernel variant found for the specified op, arch, and static parameters (Spec 4 §9.2).
    #[error("variant not found for op '{op}', arch '{arch}', static hash 0x{static_hash:016x}")]
    VariantNotFound {
        /// Operation identifier.
        op: OpId,
        /// Architecture name.
        arch: String,
        /// 64-bit static parameters hash.
        static_hash: u64,
    },

    /// Variant exists but has not been validated and cannot be selected (Spec 4 §9.3).
    #[error("variant 0x{hash:016x} for op '{op}' on arch '{arch}' is not validated and cannot be selected (Spec 4 §9.3)")]
    VariantNotValidated {
        /// Variant hash.
        hash: u64,
        /// Operation identifier.
        op: OpId,
        /// Architecture name.
        arch: String,
    },

    /// T1 fallback lookup failed on a supported architecture (Spec 4 §9.2).
    #[error("T1 fallback failed for op '{op}' on arch '{arch}': {detail}")]
    T1FallbackFailed {
        /// Operation identifier.
        op: OpId,
        /// Architecture name.
        arch: String,
        /// Diagnostic detail.
        detail: String,
    },

    /// Failed to parse a bundle manifest file (Spec 4 §11).
    #[error("failed to parse manifest at {path}: {detail}")]
    ManifestParseError {
        /// Manifest path.
        path: String,
        /// Parser error description.
        detail: String,
    },

    /// Failed to parse an autotune file (Spec 4 §6.2).
    #[error("failed to parse tune file at {path}: {detail}")]
    TuneParseError {
        /// Tune file path.
        path: String,
        /// Parser error description.
        detail: String,
    },

    /// Failed to lazily load a code object module via `hipModuleLoadData` (Spec 4 §11).
    #[error("failed to load kernel module 0x{hash:016x} ('{symbol}'): {detail}")]
    ModuleLoadError {
        /// Variant hash.
        hash: u64,
        /// Entry symbol name.
        symbol: String,
        /// Runtime error detail.
        detail: String,
    },

    /// Symbol resolution failed within a loaded module (Spec 4 §9.1).
    #[error("kernel function '{symbol}' not found in module: {detail}")]
    SymbolNotFound {
        /// Function symbol name.
        symbol: String,
        /// Detail.
        detail: String,
    },

    /// Kernel launch dispatch failed (Spec 4 §12).
    #[error("kernel launch of '{symbol}' failed: {detail}")]
    LaunchError {
        /// Kernel symbol name.
        symbol: String,
        /// Failure message.
        detail: String,
    },

    /// JIT compilation or autotuning failed (Spec 4 §6.1, §9.2).
    #[error("JIT compilation failed for variant 0x{hash:016x}: {detail}")]
    JitFailed {
        /// Variant hash.
        hash: u64,
        /// Failure message.
        detail: String,
    },

    /// Validation failed across multiple items (CONVENTIONS.md §1.4).
    #[error("{} validation problem(s): {problems:?}", problems.len())]
    ValidationFailed {
        /// Collected validation problems.
        problems: Vec<String>,
    },

    /// An `OpId` was paired with a static descriptor built for a different op (Spec 4 §3).
    #[error("operation '{op}' paired with static descriptor built for '{static_op}': OpId-to-nested descriptor agreement violated")]
    StaticOpMismatch {
        /// Requested operation identifier.
        op: OpId,
        /// Operation identifier the static descriptor was built for.
        static_op: OpId,
    },

    /// Closed resolved facts do not match the `r9v_ir::Op` they were supplied with (Spec 4 §3).
    #[error("resolved facts do not match op '{op}': {detail}")]
    FactsOpMismatch {
        /// Operation identifier of the IR op.
        op: OpId,
        /// What mismatched.
        detail: String,
    },

    /// Concurrency lock poisoned (CONVENTIONS.md §1.5).
    #[error("concurrency lock poisoned: {resource}")]
    LockPoisoned {
        /// Name of the poisoned lock resource.
        resource: String,
    },

    /// Multiple underlying registry errors occurred (CONVENTIONS.md §1.4).
    #[error("{} registry error(s) occurred: {problems:?}", problems.len())]
    Multiple {
        /// Collected errors.
        problems: Vec<RegistryError>,
    },

    /// Standard I/O error (Spec 14 §2).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// HIP runtime error (Spec 14 §3).
    #[error(transparent)]
    Hip(#[from] r9v_hip::HipError),

    /// IR domain error (Spec 1 §2).
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),
}

/// Convenience result alias for registry operations (Spec 4).
pub type Result<T, E = RegistryError> = std::result::Result<T, E>;
