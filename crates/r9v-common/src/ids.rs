// SPDX-License-Identifier: Apache-2.0
//! Distinct opaque identifier newtypes (Spec 3 §2, Spec 6 §9, Spec 10 §4, Spec 11 §11, Spec 14 §2).

use std::fmt;

// DECISION(A0.4): SeqId, ReqId, and StepId wrap u64; rejected u32 to prevent rollover in long-running serving deployments and high-step benchmark runs while retaining cheap Copy semantics.

/// Opaque sequence identifier newtype (Spec 3 §2, Spec 7 §2, Spec 14 §2, CONVENTIONS.md §3.1).
///
/// Identifies an active or completed sequence managed by `r9v-state` and scheduled
/// by `r9v-sched`. The inner representation is private to prevent accidental swapping
/// with other integer IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(u64);

impl SeqId {
    /// Creates a new [`SeqId`] from a 64-bit integer (Spec 14 §2).
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Returns the underlying 64-bit ID (Spec 14 §2).
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque request identifier newtype (Spec 10 §4, Spec 11 §11, Spec 14 §2, CONVENTIONS.md §2.2, §3.1).
///
/// Identifies an in-flight serving request in `r9v-serve`, tracing logs, and error envelopes.
/// Every request-scoped log line must carry `req_id = %req.id()` (Spec 11 §11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReqId(u64);

impl ReqId {
    /// Creates a new [`ReqId`] from a 64-bit integer (Spec 14 §2).
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Returns the underlying 64-bit ID (Spec 14 §2).
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ReqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque step identifier newtype (Spec 6 §9, Spec 11 §11, Spec 14 §2, CONVENTIONS.md §2.2, §3.1).
///
/// Identifies an execution step emitted by `r9v-sched`. Every step-scoped log line
/// and schedule log entry must carry `step_id = %step.id()` (Spec 11 §11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StepId(u64);

impl StepId {
    /// Creates a new [`StepId`] from a 64-bit integer (Spec 14 §2).
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Returns the underlying 64-bit ID (Spec 14 §2).
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
