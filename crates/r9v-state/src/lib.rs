// SPDX-License-Identifier: Apache-2.0
//! R9V KV cache sequence state, memory arenas, and block allocation
//! (Spec 3, Spec 14 §2).
//!
//! The [`StateManager`] owns paged-KV block pools as
//! offset arithmetic over an abstract arena, deterministic free-list block
//! allocation, retention with `window_start`, `BatchMeta` construction, and
//! recurrent A/B double-buffer bookkeeping. The authoritative typed layer
//! declarations live in [`spec`]; fallible operations report
//! [`error::StateError`].
//!
//! Prefix/session reuse is deferred to roadmap B1: [`StateManager`] never
//! shares blocks between sequences and retains nothing on `free_seq`.

pub mod error;
pub mod manager;
pub mod spec;

pub use error::{InvalidItem, StateError, StateResult};
pub use manager::{
    block_offset, required_pool_bytes, BatchMeta, Budget, CompactOp, GroupBudget, SlotRange,
    StateConfig, StateManager, Stats, MAX_SLOT_BLOCKS, SLOT_NONE,
};
pub use spec::{
    group_layers, CacheDtype, LayerGroup, Retain, StateSpec, BLOCK_SENTINEL, BLOCK_TOKENS,
    MAX_BATCH_TOKENS_HARD, MAX_CTX_HARD, MAX_GROUPS_HARD, MAX_LAYERS_HARD, MAX_RESERVE_HARD,
    MAX_SEQS_HARD,
};
