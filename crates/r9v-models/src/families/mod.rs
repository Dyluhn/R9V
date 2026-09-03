// SPDX-License-Identifier: Apache-2.0
//! Model architecture families and registry (Spec 8 §4; card A1.4).
//!
//! Each model family is registered under one or more `general.architecture` strings.
//! An unknown architecture fails at load with [`ModelsError::UnknownArchitecture`]
//! naming the architecture and the nearest family per Spec 8 §4.

pub mod llama;

use crate::error::ModelsError;
use crate::meta::GgufMeta;
use crate::spec::ModelSpec;

/// Supported architecture strings registered under v1 model families (Spec 8 §4).
pub const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "llama", "mistral", "qwen2", "qwen3", "gemma2", "gemma3", "phi3", "olmo2",
];

/// Returns true if the given `arch` string is registered in the family registry (Spec 8 §4).
pub fn is_supported_architecture(arch: &str) -> bool {
    SUPPORTED_ARCHITECTURES.contains(&arch)
}

/// Returns the complete list of supported architecture strings (Spec 8 §4).
pub fn supported_architectures() -> &'static [&'static str] {
    SUPPORTED_ARCHITECTURES
}

/// Returns the family name for a supported architecture, or `None` if unknown (Spec 8 §4).
pub fn find_family(arch: &str) -> Option<&'static str> {
    if is_supported_architecture(arch) {
        Some("llama")
    } else {
        None
    }
}

/// Returns the nearest known family name for diagnostic reporting on unknown architectures (Spec 8 §4).
pub fn nearest_family(_arch: &str) -> &'static str {
    "llama"
}

/// Builds a [`ModelSpec`] from typed GGUF metadata by resolving `general.architecture`
/// against the family registry (Spec 8 §4; card A1.4).
///
/// Returns [`ModelsError::MissingMetaKey`] if `general.architecture` is missing,
/// [`ModelsError::MetaTypeMismatch`] if `general.architecture` is not a string,
/// or [`ModelsError::UnknownArchitecture`] if the architecture is not registered.
pub fn build(meta: &(impl GgufMeta + ?Sized)) -> Result<ModelSpec, ModelsError> {
    if !meta.has("general.architecture") {
        return Err(ModelsError::MissingMetaKey {
            key: "general.architecture".to_string(),
            expected_type: "string",
        });
    }

    let arch = meta.str("general.architecture")?;

    if !is_supported_architecture(arch) {
        return Err(ModelsError::UnknownArchitecture {
            arch: arch.to_string(),
            nearest: nearest_family(arch),
        });
    }

    llama::build_for_arch(arch, meta)
}
