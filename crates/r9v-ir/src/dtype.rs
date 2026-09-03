// SPDX-License-Identifier: Apache-2.0
//! Element data types (Spec 1 §2.1).
//!
//! Closed set: a new dtype lands via the RFC process (Spec 1 §7), never as a
//! one-off. Codebook GGUF types have no dtype here: spec 2 §3.3 maps them to
//! the `i8` matrix path, so the IR never sees a codebook (Spec 1 §2.1).

use std::fmt;

/// Element data type (Spec 1 §2.1).
///
/// Closed enum: every `match` on this type is exhaustive with no wildcard arm
/// so an RFC-added variant fails to compile at each site that must care.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE single; accumulators, norm stats, logits (Spec 1 §2.1).
    F32,
    /// IEEE half; activations (Spec 1 §2.1).
    F16,
    /// bfloat16; activations (Spec 1 §2.1).
    Bf16,
    /// fp8 E4M3; activations, KV cache, fp8 weights (Spec 1 §2.1).
    E4m3,
    /// fp8 E5M2; second WMMA operand only (Spec 1 §2.1).
    E5m2,
    /// Signed int8; weights, quantized activations (Spec 1 §2.1).
    I8,
    /// Signed/unsigned int4, packed 2 per byte; weights only (Spec 1 §2.1).
    I4,
    /// int32; accumulators, ids, counts (Spec 1 §2.1).
    I32,
    /// uint32; token ids, row ids, block ids (Spec 1 §2.1).
    U32,
    /// Mask; attention masks, grammar masks (Spec 1 §2.1).
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Stable lowercase snake_case names, never discriminants
        // (CONVENTIONS.md §3.2).
        let name = match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::E4m3 => "e4m3",
            DType::E5m2 => "e5m2",
            DType::I8 => "i8",
            DType::I4 => "i4",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::Bool => "bool",
        };
        write!(f, "{name}")
    }
}
