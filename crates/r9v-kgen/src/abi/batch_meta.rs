// SPDX-License-Identifier: Apache-2.0
//! BatchMeta field selection per kernel operation (Spec 1 §2.5, Spec 4 §7).

use serde::{Deserialize, Serialize};

use crate::abi::types::{AbiType, FieldRole, PointeeType};

// DECISION(A3.2): BatchMeta fields needed by an op are represented as individual typed 256-byte aligned device pointers in the argument struct rather than passing the composite BatchMeta host object; rejected passing composite BatchMeta because device kernels need direct pointers to GPU-accessible buffers (e.g. block_table, slot_map, positions). Spec 1 §2.5, Spec 4 §7.

/// Closed enum of selectable BatchMeta fields required by device kernels (Spec 1 §2.5, Spec 4 §7).
///
/// Exhaustive matching required per CONVENTIONS §3.2. No open string maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchMetaField {
    /// Sequence IDs: `[S] u32` (Spec 1 §2.5). Used for Philox PRNG keying.
    SeqIds,
    /// Query token lengths: `[S] u32` (Spec 1 §2.5). Used for attention and scan sequence boundaries.
    QueryLen,
    /// Context token lengths: `[S] u32` (Spec 1 §2.5). Used for KV cache addressing.
    CtxLen,
    /// Token positions: `[T] u32` or `[T, 3] u32` (Spec 1 §2.5). Used for RoPE.
    Positions,
    /// KV slot map: `[G, T] u32` (Spec 1 §2.5, Spec 3 §3.3). Used for KV cache writes.
    SlotMap,
    /// KV block table: `[G, S, max_blocks] u32` (Spec 1 §2.5, Spec 3 §3.3). Used for paged attention.
    BlockTable,
    /// Window start: `[G, S] u32` (Spec 1 §2.5, Spec 3 §3.5). Used for sliding window attention.
    WindowStart,
    /// Tree speculative parent pointers: `[T] i32` (Spec 1 §4.D.1). Used for tree attention / verify.
    TreeParents,
    /// Tree speculative ancestor bitmask: `[T, T_max]` (Spec 1 §4.D.1).
    TreeAncestors,
}

impl BatchMetaField {
    /// Canonical snake_case name of the BatchMeta field.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SeqIds => "seq_ids",
            Self::QueryLen => "query_lens",
            Self::CtxLen => "ctx_lens",
            Self::Positions => "positions",
            Self::SlotMap => "slot_map",
            Self::BlockTable => "block_table",
            Self::WindowStart => "window_start",
            Self::TreeParents => "tree_parents",
            Self::TreeAncestors => "tree_ancestors",
        }
    }

    /// Pointee data type for this BatchMeta buffer.
    pub const fn pointee_type(&self) -> PointeeType {
        match self {
            Self::SeqIds => PointeeType::U32,
            Self::QueryLen => PointeeType::U32,
            Self::CtxLen => PointeeType::U32,
            Self::Positions => PointeeType::U32,
            Self::SlotMap => PointeeType::U32,
            Self::BlockTable => PointeeType::U32,
            Self::WindowStart => PointeeType::U32,
            Self::TreeParents => PointeeType::I32,
            Self::TreeAncestors => PointeeType::U8,
        }
    }

    /// ABI argument type representing this BatchMeta buffer pointer.
    pub const fn abi_type(&self) -> AbiType {
        AbiType::const_ptr(self.pointee_type())
    }

    /// Semantic field role for this BatchMeta field.
    pub const fn role(&self) -> FieldRole {
        FieldRole::BatchMeta(*self)
    }

    /// Documentation comment describing this BatchMeta field.
    pub const fn doc(&self) -> &'static str {
        match self {
            Self::SeqIds => "BatchMeta sequence IDs [S] u32 for Philox PRNG keying (Spec 1 §2.5)",
            Self::QueryLen => "BatchMeta query token lengths [S] u32 (Spec 1 §2.5)",
            Self::CtxLen => "BatchMeta context token lengths [S] u32 (Spec 1 §2.5)",
            Self::Positions => "BatchMeta token positions [T] u32 for RoPE (Spec 1 §2.5)",
            Self::SlotMap => "BatchMeta KV cache slot map [G, T] u32 (Spec 1 §2.5, Spec 3 §3.3)",
            Self::BlockTable => {
                "BatchMeta paged block table [G, S, max_blocks] u32 (Spec 1 §2.5, Spec 3 §3.3)"
            }
            Self::WindowStart => {
                "BatchMeta sliding window start [G, S] u32 (Spec 1 §2.5, Spec 3 §3.5)"
            }
            Self::TreeParents => {
                "BatchMeta tree speculative parent indices [T] i32 (Spec 1 §4.D.1)"
            }
            Self::TreeAncestors => {
                "BatchMeta tree speculative ancestor mask [T, T_max] (Spec 1 §4.D.1)"
            }
        }
    }
}
