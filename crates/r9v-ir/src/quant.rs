// SPDX-License-Identifier: Apache-2.0
//! Quantization schemes attached to tensors (Spec 1 §2.2).
//!
//! The scheme is attached to a tensor, not a dtype (Spec 1 §2.2). Block
//! structure, scale records and dequant formulas live in spec 2 §3 and are
//! owned by `r9v-format` (card A2.2); this crate only tags tensors.

use std::fmt;

use crate::DType;

/// Opaque weight-scheme handle (Spec 1 §2.2 `Scheme { id }`, Spec 2 §3).
///
/// The code space is owned by spec 2 §3 / `r9v-format` (card A2.2), which maps
/// its `SchemeId` enum to these codes. `r9v-ir` only transports the handle so
/// the two crates cannot form a dependency cycle (card A1.1). Accepts any
/// `u64`: future schemes (e.g. a new spec 2 id) must not fail to decode here.
// DECISION(A1.1): opaque u64 newtype per CONVENTIONS.md §3.1, with codes
// assigned by r9v-format (A2.2);
// rejected an r9v-ir enum mirroring spec 2 §3.2–§3.3 ids because that would
// give two crates ownership of one closed set and force a dependency cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemeId(u64);

impl SchemeId {
    /// Wraps a spec 2 scheme code (Spec 2 §3; code assignment owned by A2.2).
    pub const fn new(code: u64) -> Self {
        Self(code)
    }

    /// Returns the underlying spec 2 scheme code.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SchemeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "scheme({})", self.0)
    }
}

/// Quantization scheme attached to a tensor (Spec 1 §2.2).
///
/// Closed enum: adding a scheme is an RFC-level change (Spec 1 §7), and every
/// `match` stays exhaustive with no wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    /// Unquantized (Spec 1 §2.2).
    None,
    /// One scale per output row, weights (Spec 1 §2.2).
    PerRow,
    /// A spec 2 §3 scheme (`I8_B128`, `I4_K`, `I8_B32F`, ...); block structure
    /// and scale records are defined there (Spec 1 §2.2).
    Scheme(SchemeId),
    /// Activations only; one scale per row of x, native path with smoothing
    /// folded (Spec 1 §2.2).
    PerToken,
    /// Activations only; one scale per 32 along K, GGUF parity path
    /// (Spec 1 §2.2, Spec 2 §3.4).
    PerBlock32,
}

impl QuantScheme {
    /// Scale-record dtype implied by the scheme, from the `{ scale }`
    /// annotations in Spec 1 §2.2 (`PerRow` → f16, activation schemes → f32).
    /// Returns `None` for [`QuantScheme::None`] (no scales) and
    /// [`QuantScheme::Scheme`] (record format owned by Spec 2 §3 / A2.2).
    ///
    /// Scales themselves are data, not enum payloads: weight scale records
    /// live in the spec 2 §3.1 region and activation scales travel as separate
    /// tensors (e.g. `quant_act` emits `scale [T] f32`, Spec 1 §4.A).
    // DECISION(A1.1): marker variants plus this accessor; rejected embedding
    // live f16/f32 payloads because that would duplicate scale storage and
    // need a half-float dependency for one annotation. Spec 1 §2.2 is silent
    // on where a parsed scale value would live; see SI-4.
    pub const fn scale_dtype(self) -> Option<DType> {
        match self {
            QuantScheme::None | QuantScheme::Scheme(_) => None,
            QuantScheme::PerRow => Some(DType::F16),
            QuantScheme::PerToken | QuantScheme::PerBlock32 => Some(DType::F32),
        }
    }
}
