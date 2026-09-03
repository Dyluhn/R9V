// SPDX-License-Identifier: Apache-2.0
//! Tolerance table per Spec 1 §6.1 loaded as data (CONVENTIONS.md §4.3).

/// Numeric tolerances from Spec 1 §6.1.
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

    /// Exact bitwise tolerance (L0 determinism, Spec 1 App. B).
    pub const fn exact() -> Self {
        Self { abs: 0.0, rel: 0.0 }
    }

    /// Asserts that `actual` and `expected` are within tolerance.
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
