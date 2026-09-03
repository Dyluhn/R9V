// SPDX-License-Identifier: Apache-2.0
//! IR version (Spec 1 §7).

use std::fmt;

/// IR version pinned by model definitions.
///
/// Removing or changing an op's signature bumps the minor version, and model
/// definitions pin the IR version they were written against (Spec 1 §7); a
/// `ModelSpec` whose pin mismatches is an error naming both versions
/// (Spec 8 §4, owned by card A1.3, which also owns the compatibility rule —
/// this type provides identity and ordering only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IrVersion {
    /// Major version.
    pub major: u16,
    /// Minor version; bumped on op signature change (Spec 1 §7).
    pub minor: u16,
    /// Patch version.
    pub patch: u16,
}

impl IrVersion {
    /// Current IR version.
    // DECISION(A1.14): 0.2.0, bumped from 0.1.0 because ResidualAddOp gained
    // a scale attribute; Spec 1 §7 explicitly requires a minor version bump
    // on op signature change. SI-27.
    pub const CURRENT: Self = Self {
        major: 0,
        minor: 2,
        patch: 0,
    };

    /// Builds a version triple.
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl fmt::Display for IrVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}
