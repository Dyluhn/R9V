// SPDX-License-Identifier: Apache-2.0
//! Logical layout handles (Spec 1 §2.3, Spec 2 §2).
//!
//! Layout values (`L0`, `L1`, `L1S`, future `L2`) are owned by spec 2 §2 and
//! defined by `r9v-format` (card A2.1). This crate carries them opaquely so
//! tensors and the arch descriptor can name layouts without depending on
//! `r9v-format` (card A1.1; same cycle-avoidance as [`SchemeId`](crate::SchemeId)).

use std::fmt;

/// Opaque logical-layout handle (Spec 1 §2.3, Spec 2 §2).
///
/// Accepts any `u64`: a future layout version (spec 2 §2.4: "a new fragment
/// order is `L2`") must decode here without an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayoutId(u64);

impl LayoutId {
    /// Contiguous activations layout (Spec 1 §2.3: "activations use
    /// Contiguous").
    pub const CONTIGUOUS: Self = Self(0);
    /// Row-major layout for lookup tables and vectors (Spec 2 §2.1).
    pub const L0: Self = Self(1);
    /// Tiled layout for matmul weights; also the gfx12 native B-fragment
    /// order, which is what makes zero-copy load possible (Spec 2 §2.2).
    pub const L1: Self = Self(2);
    /// Tiled 2:4 structured-sparse layout over compressed K plus an index
    /// region (Spec 2 §2.3).
    pub const L1S: Self = Self(3);
    /// Provisional id for the gfx1201 intra-block K/V order described in
    /// Spec 4 §5.3 (`[d/16][32 tokens][16]` B-fragment lane order for K,
    /// `[32 tokens][dv]` A-fragment order for V). No spec names a `LayoutId`
    /// for it; see SI-3.
    pub const ATTENTION_GFX1201: Self = Self(4);

    /// Wraps a layout code. Code assignment for spec 2 ids is owned by
    /// `r9v-format` (card A2.2); the `0..=4` values above are provisional
    /// until A2.1 adopts or remaps them.
    // DECISION(A1.1): provisional codes mirroring spec order (Contiguous from
    // spec 1 §2.3, then spec 2 §2.1–§2.3); rejected leaving gfx1201() without
    // layouts because card A1.1 requires both constructors with these fields.
    pub const fn new(code: u64) -> Self {
        Self(code)
    }

    /// Returns the underlying layout code.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for LayoutId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::CONTIGUOUS => write!(f, "contiguous"),
            Self::L0 => write!(f, "l0"),
            Self::L1 => write!(f, "l1"),
            Self::L1S => write!(f, "l1s"),
            Self::ATTENTION_GFX1201 => write!(f, "attention_gfx1201"),
            other => write!(f, "layout({})", other.0),
        }
    }
}
