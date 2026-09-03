// SPDX-License-Identifier: Apache-2.0
//! Per-sequence state handles (Spec 1 §2.6).
//!
//! Ops that read or write per-sequence state take a
//! `StateHandle(layer, kind)` argument. The state manager (Spec 3, card A1.11)
//! owns allocation, eviction, checkpoint and rollback; the IR only names the
//! handle and declares read/write (Spec 1 §2.6). The parameterized
//! `StateSpec` (dims, cache dtype, retention) is declared per layer by model
//! definitions (Spec 3 §2, Spec 8) and owned by `r9v-state`, not here.

/// State kind (Spec 1 §2.6).
///
/// Closed enum: the v1 kinds. A new kind lands via the RFC process
/// (Spec 1 §7); every `match` stays exhaustive with no wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateKind {
    /// Paged KV cache (Spec 1 §2.6, Spec 3 §3).
    KvPaged,
    /// MLA compressed latent + rope part (Spec 1 §2.6, Spec 3 §3).
    KvLatent,
    /// Fixed-size per-head recurrent state (Spec 1 §2.6, Spec 3 §4).
    Recurrent,
    /// Convolution window state (Spec 1 §2.6, Spec 3 §4).
    ConvWindow,
}

/// Opaque handle naming one layer's state (Spec 1 §2.6).
///
/// Handles are opaque: only the owning crate (`r9v-state`, Spec 3) interprets
/// them; graph code names them (r9v-card-work §6). Fields stay private so a
/// handle cannot be forged or destructured outside this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateHandle {
    layer: u32,
    kind: StateKind,
}

impl StateHandle {
    /// Names the state of `layer` of the given kind (Spec 1 §2.6:
    /// `StateHandle(layer, kind)`).
    pub const fn new(layer: u32, kind: StateKind) -> Self {
        Self { layer, kind }
    }

    /// Layer index this handle names.
    pub const fn layer(self) -> u32 {
        self.layer
    }

    /// State kind this handle names.
    pub const fn kind(self) -> StateKind {
        self.kind
    }
}
