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

// DECISION(A1.15): r9v-state depends downward on r9v-ir, preserving Spec 14 §2 crate layering; rejected cyclic dependencies or embedding state semantics in r9v-ir. Spec 14 §2, card A1.15.
pub use error::{InvalidItem, StateError, StateResult};
pub use manager::{
    block_offset, required_pool_bytes, BatchWorkspace, Budget, CompactOp, GroupBudget, SlotRange,
    StateConfig, StateManager, Stats, TreeInput, TreeView, MAX_COMPACT_TOKENS, MAX_SLOT_BLOCKS,
    SLOT_NONE,
};
pub use r9v_ir::{BatchMeta, BatchMetaBuilder, Positions, TreeMask, BLOCK_TABLE_SENTINEL};
pub use spec::{
    group_layer_specs, group_layers, CacheDtype, LayerGroup, Retain, StateDecl, StateSpec,
    BLOCK_SENTINEL, BLOCK_TOKENS, MAX_BATCH_TOKENS_HARD, MAX_CTX_HARD, MAX_DECLS_HARD,
    MAX_GROUPS_HARD, MAX_LAYERS_HARD, MAX_RESERVE_HARD, MAX_SEQS_HARD,
};
