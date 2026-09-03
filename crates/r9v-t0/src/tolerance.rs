// SPDX-License-Identifier: Apache-2.0
//! Tolerance table per Spec 1 §6.1 loaded as data (CONVENTIONS.md §4.3).

/// Numeric tolerances from Spec 1 §6.1 (Spec 1 §6.1, Spec 4 §10, Spec 1 App. B).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Absolute error tolerance.
    pub abs: f64,
    /// Relative error tolerance.
    pub rel: f64,
}

impl Tolerance {
    /// Default tolerance for f16/bf16 operations (Spec 1 §6.1: abs 2e-3 / rel 1e-2).
    pub const fn f16_bf16() -> Self {
        Self {
            abs: 2e-3,
            rel: 1e-2,
        }
    }

    /// Default tolerance for f32 operations against f64 reference (Spec 1 §6.1, §6.4).
    pub const fn f32() -> Self {
        Self {
            abs: 2e-4,
            rel: 1e-3,
        }
    }

    /// Default tolerance for i8 weight operations (Spec 1 §6.1: abs 5e-3 / rel 2e-2).
    pub const fn i8_weight() -> Self {
        Self {
            abs: 5e-3,
            rel: 2e-2,
        }
    }

    /// Round-trip tolerance for e4m3 KV-cache rows (Spec 3 §2, Card A1.7).
    ///
    /// E4M3 carries 3 mantissa bits, so per-element rounding is bounded by
    /// half a quantum (rel 1/16 at the bottom of each binade); the abs term
    /// covers near-zero rows and subnormals. This bounds the cache grid only;
    /// attention math against dequantized rows is checked at [`Self::f32`].
    // DECISION(A1.7): new named tolerance entry rather than a literal in the
    // test or a widened i8 entry; rejected reusing i8_weight (its rel 2e-2
    // under-bounds the e4m3 grid) per CONVENTIONS.md §4.3.
    pub const fn e4m3_cache() -> Self {
        Self {
            abs: 1e-2,
            rel: 8e-2,
        }
    }

    /// Exact bitwise tolerance (L0 determinism, Spec 1 App. B).
    pub const fn exact() -> Self {
        Self { abs: 0.0, rel: 0.0 }
    }

    /// Asserts that `actual` and `expected` are within tolerance (Spec 1 §6.1, Spec 4 §10).
    pub fn assert_within(&self, actual: f64, expected: f64, context: &str) {
        let diff = (actual - expected).abs();
        let limit = self.abs + self.rel * expected.abs();
        assert!(
            diff <= limit || (actual.is_nan() && expected.is_nan()),
            "{context}: diff {diff} exceeds limit {limit} (actual={actual}, expected={expected}, tol_abs={}, tol_rel={})",
            self.abs,
            self.rel
        );
    }
}
