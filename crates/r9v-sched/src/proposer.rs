// SPDX-License-Identifier: Apache-2.0
//! Internal speculative decoding proposer adapter with k=0 (Spec 7 §2, Card A3.9).
//!
//! Note: The public `Proposer` trait and multi-token speculative verification belong
//! to Card A6.1. Card A3.9 maintains an internal adapter with k=0.

use r9v_common::SeqId;

/// Internal no-op speculative decoding adapter for Card A3.9 minimal scheduler (Spec 7 §2).
#[derive(Debug, Clone, Default)]
pub(crate) struct NoOpProposer;

impl NoOpProposer {
    /// Constructs a new no-op proposer.
    pub(crate) const fn new() -> Self {
        Self
    }

    /// Internal prefill hook (no-op in A3.9).
    pub(crate) fn on_prefill(&mut self, _seq: SeqId, _tokens: &[u32]) {}

    /// Internal draft generation hook (k=0 in A3.9).
    pub(crate) fn draft(&mut self, _seq: SeqId, _k: u32) {}

    /// Internal verification outcome observation hook (no-op in A3.9).
    pub(crate) fn observe(&mut self, _seq: SeqId, _accepted: &[u32]) {}

    /// Internal sequence state reset hook (no-op in A3.9).
    pub(crate) fn reset(&mut self, _seq: SeqId) {}
}
