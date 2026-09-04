// SPDX-License-Identifier: Apache-2.0
//! Host-side sequence-state manager (Spec 3 §5, §6).
//!
//! Single instance per engine, called only by the scheduler. All calls are
//! synchronous bookkeeping; device work is issued by the scheduler as ops.
//! Block ids and slot ids are a deterministic function of the request history
//! alone: two runs with the same requests produce identical `BatchMeta`
//! (Spec 3 §5, §8).
//!
//! Pools are offset arithmetic over an abstract arena: block `b` of layer
//! group `g` lives at `base[g] + b * block_bytes[g]`; no per-block pointers.
//! Fixed recurrent/conv pools are offset arithmetic the same way: buffer
//! `2 * slot + parity` of group `g` lives at `base[g] + buffer * buf_bytes`.
//! The in-memory mirror stores only token ids per reserved position so the
//! Spec 3 §8 commit/window laws can be tested without a device.

use std::collections::{BTreeMap, BTreeSet};

use r9v_common::SeqId;
use r9v_ir::{validate_tree_slices, BatchMeta, IrError, Positions, TreeMask, BLOCK_TABLE_SENTINEL};

use crate::error::{InvalidItem, StateError, StateResult};
use crate::spec::{
    group_layer_specs, group_layers, LayerGroup, StateDecl, StateSpec, BLOCK_TOKENS,
    MAX_BATCH_TOKENS_HARD, MAX_CTX_HARD, MAX_GROUPS_HARD, MAX_RESERVE_HARD, MAX_SEQS_HARD,
};

/// Slot value for layer-groups with no per-token slots (recurrent/conv).
///
/// Spec 3 §3.3 defines `slot_map` as per-token KV slots; recurrent state is
/// addressed per sequence (A/B slots, §4.2), so there is no per-token slot
// DECISION(A1.11): recurrent/conv `slot_map` rows carry `SLOT_NONE` and their
// `block_table` rows carry `BLOCK_SENTINEL`; rejected: omitting the rows
// (would break the fixed `[G, ...]` shape Spec 1 §2.5 requires).
pub const SLOT_NONE: u32 = u32::MAX;

/// Maximum blocks per paged pool: the largest count whose flattened slots
/// (`block_id * 32 + lane`) strictly precede `SLOT_NONE` (`u32::MAX`, Spec 1 §2.5).
/// `(134_217_727 - 1) * 32 + 31 == 4_294_967_263 < u32::MAX`. Pools that would need more
/// blocks are refused at construction, before any mutation.
///
/// Spec 1 §2.5, Spec 3 §3.3.
pub const MAX_SLOT_BLOCKS: u32 = 134_217_727;

/// Engine state configuration (Spec 3 §9 `[state]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateConfig {
    /// Tokens per sequence; multiple of 32 (Spec 3 §9).
    pub max_ctx: u32,
    /// Maximum live sequences; sizes recurrent/conv pools (Spec 3 §6.3).
    pub max_seqs: u32,
}

impl StateConfig {
    /// Validates the config, collecting every problem (CONVENTIONS.md §1.4).
    pub fn validate(self) -> StateResult<()> {
        let mut problems = Vec::new();
        if self.max_ctx == 0 || self.max_ctx > MAX_CTX_HARD {
            problems.push(InvalidItem {
                index: u32::MAX,
                reason: format!("max_ctx={} out of range 1..={}", self.max_ctx, MAX_CTX_HARD),
            });
        }
        if !self.max_ctx.is_multiple_of(BLOCK_TOKENS) {
            problems.push(InvalidItem {
                index: u32::MAX,
                reason: format!(
                    "max_ctx={} is not a multiple of {}",
                    self.max_ctx, BLOCK_TOKENS
                ),
            });
        }
        if self.max_seqs == 0 || self.max_seqs > MAX_SEQS_HARD {
            problems.push(InvalidItem {
                index: u32::MAX,
                reason: format!(
                    "max_seqs={} out of range 1..={}",
                    self.max_seqs, MAX_SEQS_HARD
                ),
            });
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(StateError::invalid(problems))
        }
    }

    /// Blocks per sequence at full context (Spec 3 §3.3).
    pub const fn max_blocks(self) -> u32 {
        self.max_ctx / BLOCK_TOKENS
    }
}

/// Minimum pool bytes: one aggregate `max_ctx` of blocks per paged group plus
/// `max_seqs` double-buffered slots per recurrent/conv group (Spec 3 §6.3).
///
/// Paged groups each hold one aggregate pool of at least `max_ctx` tokens of
/// blocks in total across all sequences — not `max_seqs` full sequences.
/// Only recurrent/conv pools multiply by `max_seqs` (fixed-size state per
/// sequence, not per token). The loader refuses with the numbers when the
/// device pool is smaller.
pub fn required_pool_bytes(config: StateConfig, groups: &[LayerGroup]) -> StateResult<u64> {
    let overflow = |what: &str| StateError::Overflow {
        what: what.to_owned(),
    };
    let mut total: u64 = 0;
    for g in groups {
        if g.spec.is_paged() {
            let full = g
                .block_bytes()?
                .checked_mul(u64::from(config.max_blocks()))
                .ok_or_else(|| overflow("full-context paged pool"))?;
            total = total
                .checked_add(full)
                .ok_or_else(|| overflow("pool total"))?;
        } else {
            let fixed = g
                .slots_bytes_per_seq()?
                .checked_mul(u64::from(config.max_seqs))
                .ok_or_else(|| overflow("recurrent pool"))?;
            total = total
                .checked_add(fixed)
                .ok_or_else(|| overflow("pool total"))?;
        }
    }
    Ok(total)
}

/// Byte offset of a block in the abstract arena (Spec 3 §3.1).
///
/// `base + block_id * block_bytes`, fully checked: overflow is a typed
/// error, never a panic or wrap.
pub fn block_offset(base: u64, block_id: u32, block_bytes: u64) -> StateResult<u64> {
    u64::from(block_id)
        .checked_mul(block_bytes)
        .and_then(|v| v.checked_add(base))
        .ok_or_else(|| StateError::Overflow {
            what: "block offset".to_owned(),
        })
}

// DECISION(A1.16): `reserve` returns a compact descriptor, never materialized
// slot rows. Slot values are a deterministic function of the block tables,
// so every group/token slot is resolved on demand through
// [`StateManager::slot`] / [`StateManager::fill_slots`] with checked,
// typed errors. Rejected: allocating `Vec<Vec<u32>>` rows per call (a heap
// allocation on every decode step, and the shape that forced a row-cap
// constant); a bounded thread-local row recycle pool (ties reservations to
// thread identity, caps live ranges and widths by magic constants, and pairs
// a public movable field with a `Drop` impl so `let s = r.slots;` fails to
// compile); borrowed rows tied to the manager borrow (would forbid holding
// two sequences' reservations across a `batch_meta` call); and a
// lock-guarded global pool (spec 6 §3.3 keeps the scheduler single-threaded
// with no locks on the hot path). Outputs stay a deterministic function of
// the request history alone (Spec 3 §5, §8).
// DECISION(A1.16): a `SlotRange` is scoped to its open reservation,
// not "valid until freed". `slot`/`fill_slots` validate `start`/`len`
// against the live `ctx_len`/`tail_len` and reject stale (post-commit, freed,
// or superseded by a later reserve) and foreign descriptors typed; an
// identical re-reservation after `accepted == 0` compares equal and remains
// usable, which is documented, not a scope hole. Rejected: trusting
// `start`/`len` without the live tail check (lets a stale descriptor read a
// later step's overwritten tail). Spec 3 §3.6, §5.
/// Slots reserved for one step (Spec 3 §5 `reserve`).
///
/// Compact descriptor: the owning sequence, the first reserved position (the
/// sequence's `ctx_len` at reserve time), and the reserved token count.
/// `Copy` with private fields and no `Drop`, so ranges move across threads
/// and any number of ranges stay live at once. Resolve values with
/// [`StateManager::slot`] or [`SlotRange::slot`], or fill a caller-owned row
/// with [`StateManager::fill_slots`].
///
/// Scope: valid only while its reservation is open (`ctx_len == start` and
/// `tail_len == len` on the owning sequence). `commit`/`free_seq`/a later
/// `reserve` invalidate it; use after that is a typed error, never a read of
/// another step's tail. An identical re-reservation after `accepted == 0`
/// (`ctx` unchanged, same `n`) compares equal and resolves identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotRange {
    seq: SeqId,
    start: u32,
    len: u32,
}

impl SlotRange {
    /// Owning sequence (Spec 3 §5).
    pub const fn seq(self) -> SeqId {
        self.seq
    }

    /// First reserved position (the sequence's `ctx_len` at reserve time).
    pub const fn start(self) -> u32 {
        self.start
    }

    /// Reserved token count.
    pub const fn len(self) -> u32 {
        self.len
    }

    /// One past the last reserved position (`start + len`, Spec 3 §3.6).
    ///
    /// Cannot overflow: `reserve` admits a range only when `start + len` fits
    /// in `max_ctx`.
    pub const fn end(self) -> u32 {
        self.start + self.len
    }

    /// Whether the reservation holds no tokens (never produced by `reserve`,
    /// which refuses `n == 0`; provided for caller-side emptiness checks).
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Flattened slot for token `k` of this reservation in group `group`
    /// (Spec 1 §2.5, Spec 3 §3.3): `SLOT_NONE` for recurrent/conv groups.
    ///
    /// Checked: out-of-range group or token index, an unknown/freed
    /// sequence, a stale/foreign reservation (`start`/`len` must match the
    /// live `ctx_len`/`tail_len`), and an unmapped position are typed
    /// errors, never a clamp.
    pub fn slot(self, manager: &StateManager, group: usize, k: u32) -> StateResult<u32> {
        manager.slot(&self, group, k)
    }
}

// DECISION(A1.16): tree/query paths are bounded by spec and config
// (spec 1 §4: `query_len <= 16` for spec verify; spec 7 §5: tree size engine
// cap 16; spec 12 §3: `k_max <= 15` so `k + 1 <= 16`, `tree_max <= 16`), so
// `CompactOp` is an exact fixed-capacity descriptor: 16 inline source slots,
// no heap, `Copy`, with slice access. Duplicate detection is an O(n^2) scan
// over the stack array and overlap staging is per-group stack copies —
// bounded by 16, never a `BTreeSet`/`Vec`/`Vec<Vec>>`. Lengths above 16 are
// refused typed before any mutation. Rejected: heap `src`/`dsts`/`staged`
// vecs (allocate on the cold first verify step); warmup/TLS/pool sizing to
// hide them (ties zero-alloc to history, thread, or magic constants). Spec 3
// §3.6; spec 1 §4.D.1; spec 7 §5; spec 12 §3.
/// Maximum accepted tokens in one tree-verify compaction (Spec 1 §4,
/// Spec 7 §5, Spec 12 §3: `query_len <= 16`, tree engine cap 16,
/// `k_max <= 15`, `tree_max <= 16`).
pub const MAX_COMPACT_TOKENS: usize = 16;

/// Tree-verify compaction descriptor (Spec 3 §3.6).
///
/// The scheduler enqueues this as a tiny kernel copying the accepted tokens'
/// K/V (and scales) into `dst_start .. dst_start + len` within the same
/// blocks, then commits. The manager applies the same copy to its in-memory
/// mirror eagerly so `commit` observes compacted positions.
///
/// Exact fixed-capacity: at most [`MAX_COMPACT_TOKENS`] sources in
/// accepted-path order, stored inline with no heap. `Copy` with private
/// fields and no `Drop`. Read sources with [`Self::src_positions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactOp {
    seq: SeqId,
    src: [u32; MAX_COMPACT_TOKENS],
    dst_start: u32,
    len: u32,
}

impl CompactOp {
    /// Sequence being compacted (Spec 3 §5).
    pub const fn seq(self) -> SeqId {
        self.seq
    }

    /// Destination start (the sequence's `ctx_len` at compact time).
    pub const fn dst_start(self) -> u32 {
        self.dst_start
    }

    /// Accepted token count (`src_positions().len()`).
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Whether no token was accepted (only produced by compacting an empty
    /// path; the scheduler always compacts a nonempty accepted path).
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Absolute source positions, in accepted-path order (Spec 3 §3.6).
    ///
    /// Exact bytes in exact order: `len` entries, no padding, no reorder.
    pub const fn src_positions(&self) -> &[u32] {
        self.src.as_slice().split_at(self.len as usize).0
    }
}

/// Free/total pool state per group (Spec 3 §5 `budget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupBudget {
    /// Layer-group index.
    pub index: usize,
    /// Total blocks in the group pool (paged groups; 0 for fixed groups).
    pub total_blocks: u32,
    /// Free blocks in the group pool.
    pub free_blocks: u32,
    /// Bytes per block across the group's layers.
    pub block_bytes: u64,
    /// Arena base offset of this group's pool.
    pub base_offset: u64,
    /// Total sequence slots in the group pool (recurrent/conv groups sized
    /// by `max_seqs`; 0 for paged groups).
    pub total_slots: u32,
    /// Free sequence slots in the group pool.
    pub free_slots: u32,
    /// Double-buffered (A+B) bytes per sequence slot (fixed groups; 0 for
    /// paged groups).
    pub slot_bytes_per_seq: u64,
}

/// Pool budget snapshot (Spec 3 §5 `budget`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    /// Per-group budgets, in group order.
    pub groups: Vec<GroupBudget>,
    /// Assigned arena bytes across paged pools.
    pub pool_bytes_total: u64,
    /// Free arena bytes across paged pools.
    pub pool_bytes_free: u64,
    /// Assigned arena bytes across recurrent/conv fixed pools.
    pub fixed_bytes_total: u64,
    /// Free arena bytes across recurrent/conv fixed pools.
    pub fixed_bytes_free: u64,
    /// Free host-side block bytes (Spec 3 §3.7). Host swap is deferred to
    /// roadmap B1 alongside the prefix cache, so this is always 0: an
    /// explicit zero, not an omission.
    pub host_free: u64,
    /// Supplied pool bytes not assigned to any pool. The paged split only
    /// deals whole blocks, so a sub-block remainder stays unusable and is
    /// reported here rather than silently absorbed.
    pub unusable_bytes: u64,
}

/// Manager statistics (Spec 3 §5 `stats`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Prefix-cache hit rate; always 0.0 until the roadmap B1 prefix cache
    /// lands (no hidden hits: `new_seq` never shares blocks).
    pub prefix_hit_rate: f64,
    /// Prefix-cache evictions; always 0 until B1.
    pub evictions: u64,
    /// Recurrent A/B swaps performed.
    pub swaps: u64,
    /// Commits performed.
    pub commits: u64,
    /// Fraction of paged blocks allocated, `0.0..=1.0`, from exact
    /// allocated/total block counts (not an estimate).
    pub utilization: f32,
}

/// One pool of fixed-size blocks with a deterministic free list (Spec 3 §3.1).
///
/// Allocation always takes the smallest free id, so block ids are a function
/// of the request history alone (Spec 3 §5, §8).
#[derive(Debug)]
struct BlockPool {
    total: u32,
    // DECISION(A1.16): free ids live in a `Vec` sorted descending so `alloc`
    // pops the smallest id in O(1) and `release` inserts at its sorted
    // position, both without heap allocation after construction (capacity is
    // fixed at `total`, and occupancy never exceeds it: an id is released
    // only after being drained from exactly one sequence table). Rejected: a
    // `BTreeSet` free list (node splits allocate on the commit release path
    // of windowed groups, breaking steady-state zero-allocation). Spec 3
    // §3.1, §5; spec 6 §3.3 (no locks on the hot path).
    free: Vec<u32>,
    base_offset: u64,
    block_bytes: u64,
}

impl BlockPool {
    fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }

    // DECISION(A1.16): `release` validates in release builds and reports
    // corruption as a typed error the caller propagates transactionally.
    // Rejected: `debug_assert`-only double-release detection (correctness
    // that vanishes in release) and a panicking release (destroys the
    // atomicity the reserve/commit/free paths guarantee). Spec 3 §3.1, §5.
    fn release(&mut self, id: u32) -> StateResult<()> {
        if id >= self.total {
            return Err(StateError::InvalidBatch {
                detail: format!(
                    "release block {id} out of range: pool holds {total}",
                    total = self.total
                ),
            });
        }
        if self.contains(id) {
            return Err(StateError::InvalidBatch {
                detail: format!("double release of block {id}"),
            });
        }
        let at = self.free.partition_point(|&x| x > id);
        self.free.insert(at, id);
        Ok(())
    }

    fn contains(&self, id: u32) -> bool {
        self.free
            .binary_search_by(|&x| x.cmp(&id).reverse())
            .is_ok()
    }
}

/// One fixed pool of per-sequence recurrent/conv slots (Spec 3 §4.1, §6.3).
///
/// Each live sequence owns exactly one sequence slot per fixed group; the
/// slot holds both the verified (A) and working (B) buffers contiguously.
/// Allocation takes the smallest free slot id, so slot ids are a
/// deterministic function of the request history alone (Spec 3 §5, §8).
#[derive(Debug)]
struct FixedPool {
    total_slots: u32,
    free: BTreeSet<u32>,
    slot_bytes_per_seq: u64,
    buffer_bytes: u64,
    base_offset: u64,
}

impl FixedPool {
    /// Byte offset of buffer `buffer` (`2 * slot + parity`) in the abstract
    /// arena (Spec 3 §4.1). Fully checked: overflow is a typed error.
    fn buffer_offset(&self, buffer: u32) -> StateResult<u64> {
        u64::from(buffer)
            .checked_mul(self.buffer_bytes)
            .and_then(|v| v.checked_add(self.base_offset))
            .ok_or_else(|| StateError::Overflow {
                what: "fixed buffer offset".to_owned(),
            })
    }
}

// DECISION(A1.16): the scheduler hot path fills caller-owned batch buffers
// instead of building an owned `BatchMeta` per step. `BatchWorkspace` holds
// reusable flat buffers sized once (cold) for the largest batch the
// scheduler admits; `fill_batch_meta` validates and fills them with no heap
// allocation on success and fails closed with a typed error when a buffer is
// undersized — never silently growing on the hot path. Rejected: keeping the
// owned `batch_meta` builder as the only metadata path (it allocates seven
// `Vec`s plus validation sets per call), and auto-growing workspace buffers
// (hides a hot allocation behind a sizing mistake). Spec 1 §2.5, Spec 3 §5.
// DECISION(A1.16): the hot workspace carries exact explicit positions
// for both scalar `[T]` and MRoPE `[T,3]` (Spec 1 §2.5) in caller-owned
// buffers with no hot allocation. `position_width()` (0 = empty, 1 = scalar,
// 3 = MRoPE) disambiguates `positions()` from `positions_mrope()`; there is
// no overloaded accessor that means different shapes on different fills.
// `try_with_capacity` is the checked constructor: dimension overflow and
// capacity refusal are typed errors. `with_capacity` is kept only as the
// truthful cold panicking wrapper (panics on overflow/OOM, never silent
// saturation). Rejected: a scalar-only workspace (forces MRoPE through the
// owned builder on the hot path) and `saturating_mul` sizing (silently
// undersizes huge dims). Spec 1 §2.5.
// DECISION(A1.16): tree-verify metadata lives in the same caller-owned
// `BatchWorkspace` as the batch tensors, not in a second preallocated
// object. The scheduler already threads one workspace through every step;
// a parallel tree workspace would double the cold-sizing surface and let
// the two drift apart. The hot tree buffers (`tree_parents`,
// `tree_ancestors`) plus the cycle-check scratch (`tree_cycle_state`,
// `tree_cycle_path`) are sized once with `try_reserve_tree` and never grow
// on the hot path; `TreeView` borrows them for kernels without fabricating
// an owned `TreeMask`. Rejected: a standalone tree workspace type and
// keeping the owned `TreeMask` builder as the only verify path (it owns two
// `Vec`s per step, forcing a heap allocation per verify). Spec 1 §4.D.1,
// Spec 3 §5.
/// Borrowed tree inputs for one verify step (Spec 1 §4.D.1).
///
/// `Copy` bundle of the scheduler's tree slices: `parents [T]`, columns per
/// row `t_max`, and the flattened `[T, t_max]` ancestor mask. Storage stays
/// caller-owned until a fill copies it into the workspace's preallocated
/// buffers; pass to
/// [`StateManager::fill_batch_meta_with_tree_input`] or
/// [`StateManager::fill_batch_meta_with_options_and_tree_input`], or store
/// alone with [`BatchWorkspace::fill_tree`].
#[derive(Debug, Clone, Copy)]
pub struct TreeInput<'a> {
    parents: &'a [i32],
    t_max: u32,
    ancestors: &'a [bool],
}

impl<'a> TreeInput<'a> {
    /// Bundles the scheduler's tree slices (Spec 1 §4.D.1).
    pub const fn new(parents: &'a [i32], t_max: u32, ancestors: &'a [bool]) -> Self {
        Self {
            parents,
            t_max,
            ancestors,
        }
    }

    /// Parent ids, −1 for roots (Spec 1 §4.D.1).
    pub const fn parents(&self) -> &[i32] {
        self.parents
    }

    /// Columns per row (`T_max`, Spec 1 §4.D.1).
    pub const fn t_max(&self) -> u32 {
        self.t_max
    }

    /// Flattened `[T, T_max]` row-major ancestor mask (Spec 1 §4.D.1).
    pub const fn ancestors(&self) -> &[bool] {
        self.ancestors
    }

    /// Token count `T` (`parents.len()`).
    pub const fn t(&self) -> usize {
        self.parents.len()
    }
}

/// Stable borrowed tree view for one verify step (Spec 1 §4.D.1).
///
/// Returned by [`BatchWorkspace::tree_view`]: the tree stored by the last
/// fill, whether it arrived through the allocation-free slices path or the
/// owned compat path. This borrows workspace (or owned-compat) storage the
/// scheduler and kernels read directly; no owned `TreeMask` is fabricated.
#[derive(Debug, Clone, Copy)]
pub struct TreeView<'a> {
    parents: &'a [i32],
    t_max: u32,
    ancestors: &'a [bool],
}

impl<'a> TreeView<'a> {
    /// Token count `T` (`parents.len()`).
    pub const fn t(&self) -> usize {
        self.parents.len()
    }

    /// Columns per row (`T_max`).
    pub const fn t_max(&self) -> u32 {
        self.t_max
    }

    /// Parent ids, −1 for roots (Spec 1 §4.D.1).
    pub const fn parents(&self) -> &[i32] {
        self.parents
    }

    /// Flattened `[T, T_max]` row-major ancestor mask (Spec 1 §4.D.1).
    pub const fn ancestors(&self) -> &[bool] {
        self.ancestors
    }

    /// Ancestor bit for token `tok` at column `pos`.
    ///
    /// Panics on out-of-bounds indices like [`TreeMask::is_ancestor`]:
    /// dims are fixed at fill time, so a bad index is a caller bug, not
    /// input data.
    pub fn is_ancestor(&self, tok: usize, pos: usize) -> bool {
        assert!(
            tok < self.parents.len(),
            "TreeView::is_ancestor token {tok} out of bounds for T={}",
            self.parents.len(),
        );
        assert!(
            (pos as u32) < self.t_max,
            "TreeView::is_ancestor pos {pos} out of bounds for t_max={}",
            self.t_max,
        );
        self.ancestors[tok * self.t_max as usize + pos]
    }
}

#[derive(Debug, Default)]
pub struct BatchWorkspace {
    last_g: u32,
    last_s: u32,
    last_t: u32,
    last_max_blocks: u32,
    last_position_width: u32,
    seq_ids: Vec<u32>,
    query_lens: Vec<u32>,
    ctx_lens: Vec<u32>,
    positions: Vec<u32>,
    positions_mrope: Vec<[u32; 3]>,
    slot_map: Vec<u32>,
    block_table: Vec<u32>,
    window_start: Vec<u32>,
    id_scratch: Vec<u64>,
    tree: Option<TreeMask>,
    tree_parents: Vec<i32>,
    tree_ancestors: Vec<bool>,
    tree_t_max: u32,
    tree_len: usize,
    tree_anc_len: usize,
    tree_present: bool,
    tree_cycle_state: Vec<u8>,
    tree_cycle_path: Vec<u32>,
}

impl BatchWorkspace {
    /// Empty workspace: every buffer grows on first use (cold sizing).
    pub fn new() -> Self {
        Self::default()
    }

    /// Checked workspace sizing (Spec 1 §2.5 shapes).
    ///
    /// Cold sizing for the largest batch admitted: `groups` layer groups,
    /// `max_seqs` sequences, `max_tokens` total tokens, `max_blocks` blocks
    /// per sequence per group. Sizes scalar `[T]`, MRoPE `[T,3]`, `[G,T]`,
    /// `[G,S,max_blocks]`, and `[G,S]` buffers. Overflow or a refused
    /// capacity reservation is a typed error; nothing is partially sized.
    pub fn try_with_capacity(
        groups: usize,
        max_seqs: usize,
        max_tokens: usize,
        max_blocks: u32,
    ) -> StateResult<Self> {
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        let max_b = usize::try_from(max_blocks).map_err(|_| overflow("workspace max blocks"))?;
        let need_slot = groups
            .checked_mul(max_tokens)
            .ok_or_else(|| overflow("workspace slot map size"))?;
        let need_block = groups
            .checked_mul(max_seqs)
            .and_then(|v| v.checked_mul(max_b))
            .ok_or_else(|| overflow("workspace block table size"))?;
        let need_window = groups
            .checked_mul(max_seqs)
            .ok_or_else(|| overflow("workspace window start size"))?;
        let mut ws = Self {
            last_g: 0,
            last_s: 0,
            last_t: 0,
            last_max_blocks: 0,
            last_position_width: 0,
            seq_ids: Vec::new(),
            query_lens: Vec::new(),
            ctx_lens: Vec::new(),
            positions: Vec::new(),
            positions_mrope: Vec::new(),
            slot_map: Vec::new(),
            block_table: Vec::new(),
            window_start: Vec::new(),
            id_scratch: Vec::new(),
            tree: None,
            tree_parents: Vec::new(),
            tree_ancestors: Vec::new(),
            tree_t_max: 0,
            tree_len: 0,
            tree_anc_len: 0,
            tree_present: false,
            tree_cycle_state: Vec::new(),
            tree_cycle_path: Vec::new(),
        };
        let reserve = |v: &mut Vec<u32>, cap: usize| {
            v.try_reserve_exact(cap)
                .map_err(|_| overflow("workspace capacity"))?;
            Ok::<(), StateError>(())
        };
        reserve(&mut ws.seq_ids, max_seqs)?;
        reserve(&mut ws.query_lens, max_seqs)?;
        reserve(&mut ws.ctx_lens, max_seqs)?;
        reserve(&mut ws.positions, max_tokens)?;
        ws.positions_mrope
            .try_reserve_exact(max_tokens)
            .map_err(|_| overflow("workspace capacity"))?;
        reserve(&mut ws.slot_map, need_slot)?;
        reserve(&mut ws.block_table, need_block)?;
        reserve(&mut ws.window_start, need_window)?;
        ws.id_scratch
            .try_reserve_exact(max_seqs)
            .map_err(|_| overflow("workspace capacity"))?;
        Ok(ws)
    }

    /// Workspace with every buffer pre-sized (Spec 1 §2.5 shapes).
    ///
    /// Cold sizing for the largest batch admitted: `groups` layer groups,
    /// `max_seqs` sequences, `max_tokens` total tokens, `max_blocks` blocks
    /// per sequence per group. A later [`StateManager::fill_batch_meta`]
    /// whose dims fit these caps allocates nothing; one that exceeds them
    /// fails closed with a typed error instead of growing.
    ///
    /// Panics on dimension overflow or refused capacity: use
    /// [`Self::try_with_capacity`] when dims come from untrusted input.
    pub fn with_capacity(
        groups: usize,
        max_seqs: usize,
        max_tokens: usize,
        max_blocks: u32,
    ) -> Self {
        Self::try_with_capacity(groups, max_seqs, max_tokens, max_blocks)
            .expect("workspace dims overflowed or capacity refused: use try_with_capacity")
    }

    /// Checked workspace sizing with tree-verify storage (Spec 1 §2.5, §4.D.1).
    ///
    /// Sizes the batch buffers like [`Self::try_with_capacity`] and reserves
    /// the hot tree buffers for at most `max_tree_tokens` tokens with at
    /// most `max_tree_t_max` ancestor columns per row, plus the cycle-check
    /// scratch for that width. A later tree fill whose `T` and `T * t_max`
    /// fit these caps allocates nothing; one that exceeds them fails closed
    /// typed instead of growing. Overflow or a refused capacity reservation
    /// is a typed error; nothing is partially sized.
    pub fn try_with_capacity_and_tree(
        groups: usize,
        max_seqs: usize,
        max_tokens: usize,
        max_blocks: u32,
        max_tree_tokens: usize,
        max_tree_t_max: u32,
    ) -> StateResult<Self> {
        let mut ws = Self::try_with_capacity(groups, max_seqs, max_tokens, max_blocks)?;
        ws.try_reserve_tree(max_tree_tokens, max_tree_t_max)?;
        Ok(ws)
    }

    /// Checked tree-verify storage sizing (Spec 1 §4.D.1).
    ///
    /// Cold: reserves room for at most `max_tokens` parent ids, a
    /// `max_tokens * max_tree_t_max` ancestor mask, and the cycle-check
    /// scratch for that width. Only grows — calling again with smaller caps
    /// keeps the existing capacity. Overflow or a refused reservation is a
    /// typed error; partial growth is still reported as an error and the
    /// caller must re-size cold (caps never shrink, so retrying with fitting
    /// caps is exact).
    pub fn try_reserve_tree(&mut self, max_tokens: usize, max_tree_t_max: u32) -> StateResult<()> {
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        let max_t = usize::try_from(max_tree_t_max).map_err(|_| overflow("tree t_max"))?;
        let need_anc = max_tokens
            .checked_mul(max_t)
            .ok_or_else(|| overflow("tree ancestor mask size"))?;
        reserve_at_least(&mut self.tree_parents, max_tokens, "tree parents capacity")?;
        reserve_at_least(
            &mut self.tree_ancestors,
            need_anc,
            "tree ancestors capacity",
        )?;
        reserve_at_least(
            &mut self.tree_cycle_state,
            max_tokens,
            "tree cycle state capacity",
        )?;
        reserve_at_least(
            &mut self.tree_cycle_path,
            max_tokens,
            "tree cycle path capacity",
        )?;
        Ok(())
    }

    /// Layer-group count `G` of the last fill (Spec 1 §2.5).
    pub const fn groups(&self) -> u32 {
        self.last_g
    }

    /// Sequence count `S` of the last fill (Spec 1 §2.5).
    pub const fn seqs(&self) -> u32 {
        self.last_s
    }

    /// Token count `T` of the last fill (Spec 1 §2.5).
    pub const fn tokens(&self) -> u32 {
        self.last_t
    }

    /// Blocks per sequence per group of the last fill (Spec 3 §3.3).
    pub const fn max_blocks(&self) -> u32 {
        self.last_max_blocks
    }

    /// Device sequence ids `[S]` (Spec 1 §2.5, §4.F).
    pub fn seq_ids(&self) -> &[u32] {
        &self.seq_ids
    }

    /// Query lengths `[S]` (Spec 1 §2.5).
    pub fn query_lens(&self) -> &[u32] {
        &self.query_lens
    }

    /// Context lengths `[S]` (Spec 1 §2.5).
    pub fn ctx_lens(&self) -> &[u32] {
        &self.ctx_lens
    }

    /// Position width of the last fill: 0 (empty), 1 (scalar `[T]`), or 3
    /// (MRoPE `[T,3]`) (Spec 1 §2.5).
    ///
    /// Check this before reading positions: `positions()` is meaningful only
    /// when it is 1, `positions_mrope()` only when it is 3.
    pub const fn position_width(&self) -> u32 {
        self.last_position_width
    }

    /// Scalar token positions `[T]` (`ctx + k` per token, or the exact
    /// explicit values; Spec 1 §2.5).
    ///
    /// Valid only when [`Self::position_width`] is 1 (or the workspace is
    /// empty); empty otherwise. MRoPE fills leave this empty — read
    /// [`Self::positions_mrope`] when the width is 3.
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// MRoPE token positions `[T,3]` (Spec 1 §2.5).
    ///
    /// Valid only when [`Self::position_width`] is 3; empty otherwise.
    /// Scalar fills leave this empty — read [`Self::positions`] when the
    /// width is 1.
    pub fn positions_mrope(&self) -> &[[u32; 3]] {
        &self.positions_mrope
    }

    /// Flat MRoPE length row-major (`3 * T` u32s, Spec 1 §2.5 `[T,3]`).
    ///
    /// Unambiguous flat size of [`Self::positions_mrope`]: triplet `i`
    /// occupies flat indices `3*i .. 3*i + 3`. Zero unless the width is 3.
    /// Read values with [`Self::positions_mrope_flat_value`]; the triplet
    /// slice itself stays typed so no raw-pointer reinterpretation is
    /// needed (raw-pointer blocks live only in the HIP/T0 SIMD modules).
    pub fn positions_mrope_flat_len(&self) -> usize {
        self.positions_mrope.len() * 3
    }

    /// One flat MRoPE value row-major (Spec 1 §2.5 `[T,3]` flattened).
    ///
    /// `None` when out of bounds (including whenever the width is not 3).
    /// Triplet `i` is `[flat(3*i), flat(3*i+1), flat(3*i+2)]`.
    pub fn positions_mrope_flat_value(&self, flat_idx: usize) -> Option<u32> {
        let t = flat_idx / 3;
        let lane = flat_idx % 3;
        self.positions_mrope.get(t).map(|trip| trip[lane])
    }

    /// Flattened slots `[G, T]` row-major (Spec 1 §2.5, Spec 3 §3.3).
    pub fn slot_map(&self) -> &[u32] {
        &self.slot_map
    }

    /// Block tables `[G, S, max_blocks]` row-major with sentinel holes
    /// (Spec 1 §2.5, Spec 3 §3.3).
    pub fn block_table(&self) -> &[u32] {
        &self.block_table
    }

    /// First retained positions `[G, S]` row-major (Spec 1 §2.5, Spec 3 §3.5).
    pub fn window_start(&self) -> &[u32] {
        &self.window_start
    }

    /// Speculative tree mask of the last fill, if the step verifies drafts
    /// (Spec 1 §4.D.1).
    ///
    /// Cold compat only: `Some` exactly when the last fill arrived through
    /// an owned-`TreeMask` entry point (`batch_meta*`, `fill_batch_meta*`).
    /// Slices-path fills never fabricate an owned mask, so this is `None`
    /// after them — read [`Self::tree_view`] on the hot path instead.
    pub fn tree(&self) -> Option<&TreeMask> {
        self.tree.as_ref()
    }

    /// Stable borrowed tree view of the last fill, if the step verifies
    /// drafts (Spec 1 §4.D.1).
    ///
    /// The scheduler/kernel view: `Some` whenever a tree was stored, whether
    /// it arrived through the allocation-free slices path or the owned
    /// compat path. A slices fill clears the owned mask and an owned fill
    /// clears the slices staging, so exactly one source is live; when both
    /// are somehow present the slices win. Borrows workspace storage;
    /// nothing is fabricated or allocated.
    pub fn tree_view(&self) -> Option<TreeView<'_>> {
        if self.tree_present {
            Some(TreeView {
                parents: &self.tree_parents[..self.tree_len],
                t_max: self.tree_t_max,
                ancestors: &self.tree_ancestors[..self.tree_anc_len],
            })
        } else {
            self.tree.as_ref().map(|t| TreeView {
                parents: t.parents(),
                t_max: t.t_max(),
                ancestors: t.ancestors(),
            })
        }
    }

    /// Stores and validates a tree without touching batch tensors
    /// (Spec 1 §4.D.1).
    ///
    /// Copies `tree` into the preallocated buffers and runs the complete
    /// intrinsic validation — shape, parent bounds/self-parent, cycles — via
    /// [`validate_tree_slices`](r9v_ir::validate_tree_slices) with the
    /// workspace's cold-sized cycle scratch. Success allocates nothing;
    /// undersized buffers are a typed error naming the capacity and the
    /// requirement (buffers never grow here — size them cold with
    /// [`Self::try_reserve_tree`]); invalid trees are the same typed
    /// [`IrError`](r9v_ir::IrError) variants the owned [`TreeMask`] builder
    /// reports. On failure nothing is stored (`tree_view()` is `None` for
    /// the hot path). Stores clear the owned compat mask so the hot slices
    /// are the single live source.
    pub fn fill_tree(&mut self, tree: TreeInput<'_>) -> StateResult<()> {
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        let t = tree.t();
        let t_max = tree.t_max();
        let need_anc = t
            .checked_mul(usize::try_from(t_max).map_err(|_| overflow("tree t_max"))?)
            .ok_or_else(|| overflow("tree ancestor mask size"))?;
        if self.tree_parents.capacity() < t {
            return Err(StateError::InvalidBatch {
                detail: format!(
                    "batch workspace tree parents capacity {} < required {t}",
                    self.tree_parents.capacity()
                ),
            });
        }
        if self.tree_ancestors.capacity() < need_anc {
            return Err(StateError::InvalidBatch {
                detail: format!(
                    "batch workspace tree ancestors capacity {} < required {need_anc}",
                    self.tree_ancestors.capacity()
                ),
            });
        }
        if self.tree_cycle_state.capacity() < t || self.tree_cycle_path.capacity() < t {
            return Err(StateError::InvalidBatch {
                detail: format!("batch workspace tree cycle scratch capacity < required {t}"),
            });
        }
        // Capacities fit, so nothing below allocates: clears keep capacity,
        // extends and the scratch `resize` land within it.
        self.tree_parents.clear();
        self.tree_parents.extend(tree.parents().iter().copied());
        self.tree_ancestors.clear();
        self.tree_ancestors.extend(tree.ancestors().iter().copied());
        self.tree_cycle_state.clear();
        self.tree_cycle_state.resize(t, 0);
        self.tree_cycle_path.clear();
        let validated = validate_tree_slices(
            &self.tree_parents,
            t_max,
            &self.tree_ancestors,
            &mut self.tree_cycle_state,
            &mut self.tree_cycle_path,
        );
        match validated {
            Ok(()) => {
                self.tree = None;
                self.tree_t_max = t_max;
                self.tree_len = t;
                self.tree_anc_len = need_anc;
                self.tree_present = true;
                Ok(())
            }
            Err(e) => {
                self.tree_parents.clear();
                self.tree_ancestors.clear();
                self.tree_t_max = 0;
                self.tree_len = 0;
                self.tree_anc_len = 0;
                self.tree_present = false;
                Err(StateError::Ir(e))
            }
        }
    }

    /// Drops the stored hot slices-path tree, if any (Spec 1 §4.D.1).
    ///
    /// Lengths go to zero; capacities stay for the next fill. The owned
    /// compat mask is untouched.
    pub fn clear_tree(&mut self) {
        self.tree_parents.clear();
        self.tree_ancestors.clear();
        self.tree_t_max = 0;
        self.tree_len = 0;
        self.tree_anc_len = 0;
        self.tree_present = false;
    }

    /// Slot for new token `tok` in group `group` (row-major `[G, T]`).
    ///
    /// Bounds-checked: out-of-range indices are `None` (the scheduler sizes
    /// uploads from [`Self::groups`]/[`Self::tokens`], so this is a guard,
    /// not input validation).
    pub fn slot(&self, group: usize, tok: usize) -> Option<u32> {
        let t = self.last_t as usize;
        self.slot_map
            .get(group.checked_mul(t)?.checked_add(tok)?)
            .copied()
    }

    /// Block id for sequence `seq`, block entry `b`, in group `group`
    /// (row-major `[G, S, max_blocks]`). Bounds-checked like [`Self::slot`].
    pub fn block(&self, group: usize, seq: usize, b: usize) -> Option<u32> {
        let s = self.last_s as usize;
        let m = self.last_max_blocks as usize;
        self.block_table
            .get(
                group
                    .checked_mul(s)?
                    .checked_add(seq)?
                    .checked_mul(m)?
                    .checked_add(b)?,
            )
            .copied()
    }
}

/// Validated batch dims shared by the owned and fill metadata paths
/// (Spec 1 §2.5, Spec 3 §5).
#[derive(Debug, Clone, Copy)]
struct BatchPlan {
    total_tokens: usize,
    total_u64: u64,
}

/// Per-sequence state (Spec 3 §3.3).
#[derive(Debug)]
struct SeqState {
    ctx_len: u32,
    tail_len: u32,
    /// Block ids per group in ascending block-index order (paged groups).
    tables: Vec<Vec<u32>>,
    /// Block indices per group, aligned with `tables`.
    indices: Vec<Vec<u32>>,
    /// Sparse mirror of written token ids: `(group, pos) -> token`.
    mirror: BTreeMap<(usize, u32), u32>,
    /// Compacted length since the last reserve, if `compact` ran.
    compacted: Option<usize>,
    /// Owned sequence slot per group for recurrent/conv groups (`None` for
    /// paged groups). Taken from the group's [`FixedPool`] at `new_seq`,
    /// released back at `free_seq`, reused smallest-first.
    fixed_slots: Vec<Option<u32>>,
    /// Active A/B buffer per group (`0 = A`, `1 = B`); meaningful only for
    /// recurrent/conv groups, where it selects the buffer inside the owned
    /// sequence slot (Spec 3 §4.2).
    parity: Vec<u8>,
}

/// Host-side sequence-state manager (Spec 3 §5).
///
/// Owns allocation, retention, checkpoint/rollback bookkeeping, and
/// `BatchMeta` construction. Deterministic given the call sequence.
#[derive(Debug)]
pub struct StateManager {
    config: StateConfig,
    groups: Vec<LayerGroup>,
    pools: Vec<Option<BlockPool>>,
    fixed: Vec<Option<FixedPool>>,
    max_blocks: u32,
    pool_bytes_total: u64,
    fixed_bytes_total: u64,
    unusable_bytes: u64,
    next_seq: u64,
    seqs: BTreeMap<u64, SeqState>,
    live_count: u32,
    swaps: u64,
    commits: u64,
    // DECISION(A1.16): reusable owned scratch for `reserve`'s missing-block
    // records (`(group, block index)` in ascending group/index order),
    // capacity fixed at construction to the mathematical maximum
    // (`paged groups × max_blocks`: every missing index lies in
    // `[0, max_ctx / 32)`), cleared per call. Pushes never reallocate, so
    // the first `reserve` after admission allocates exactly as little as
    // steady state: nothing. Rejected: a per-call `Vec` plus `BTreeSet`
    // membership probe (a heap allocation on every step), and
    // warmup-grown capacity (makes the first steps allocate while claiming
    // the hot path is allocation-free). Spec 3 §3.6, §5.
    reserve_scratch: Vec<(u32, u32)>,
}

impl StateManager {
    /// Roadmap item owning the content-addressed prefix cache (§3.4) and the
    /// session cache (§4.3). Until then `new_seq` returns `matched_len = 0`,
    /// `free_seq` retains nothing, and stats report zero hits — explicitly,
    /// with no hidden sharing.
    pub const PREFIX_CACHE_DEFERRED_TO: &'static str = "B1";

    /// Builds the manager, sizing pools from `pool_bytes` (Spec 3 §6.3).
    ///
    /// Fixed recurrent/conv pools take `slots_bytes_per_seq * max_seqs`
    /// first; the remaining paged bytes are split across paged groups in
    /// proportion to their block costs so every paged group holds the same
    /// aggregate block count of at least `max_ctx / 32`. Refuses with the
    /// numbers when `pool_bytes` cannot cover that minimum. All validation
    /// is collected before any allocation.
    pub fn new(
        config: StateConfig,
        layer_specs: Vec<StateSpec>,
        pool_bytes: u64,
    ) -> StateResult<Self> {
        let config_res = config.validate();
        let groups_res = group_layers(&layer_specs);

        let groups = match (config_res, groups_res) {
            (
                Err(StateError::InvalidConfig { problems: mut cp }),
                Err(StateError::InvalidConfig { problems: mut gp }),
            ) => {
                cp.append(&mut gp);
                return Err(StateError::InvalidConfig { problems: cp });
            }
            (Err(e), _) => return Err(e),
            (_, Err(e)) => return Err(e),
            (Ok(()), Ok(groups)) => groups,
        };

        Self::new_from_validated_groups(config, groups, pool_bytes)
    }

    /// Builds the manager from explicit state declarations retaining declaring model layer indices (Spec 3 §2, §6.3).
    ///
    /// Supports hybrid layers declaring multiple state specifications (e.g. ConvWindow and Recurrent)
    /// with identical layer indices.
    pub fn new_with_declarations(
        config: StateConfig,
        declarations: Vec<StateDecl>,
        pool_bytes: u64,
    ) -> StateResult<Self> {
        let config_res = config.validate();
        let groups_res = group_layer_specs(&declarations);

        let groups = match (config_res, groups_res) {
            (
                Err(StateError::InvalidConfig { problems: mut cp }),
                Err(StateError::InvalidConfig { problems: mut gp }),
            ) => {
                cp.append(&mut gp);
                return Err(StateError::InvalidConfig { problems: cp });
            }
            (Err(e), _) => return Err(e),
            (_, Err(e)) => return Err(e),
            (Ok(()), Ok(groups)) => groups,
        };

        Self::new_from_validated_groups(config, groups, pool_bytes)
    }

    fn new_from_validated_groups(
        config: StateConfig,
        groups: Vec<LayerGroup>,
        pool_bytes: u64,
    ) -> StateResult<Self> {
        let max_blocks = config.max_blocks();
        let max_seqs = config.max_seqs;
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };

        // Fixed pools first: per-sequence double-buffered bytes times max_seqs
        // (Spec 3 §6.3: only recurrent/conv pools multiply by max_seqs).
        let mut fixed_total: u64 = 0;
        for g in &groups {
            if g.spec.is_paged() {
                continue;
            }
            let per_seq = g.slots_bytes_per_seq()?;
            let grp = per_seq
                .checked_mul(u64::from(max_seqs))
                .ok_or_else(|| overflow("fixed group pool"))?;
            fixed_total = fixed_total
                .checked_add(grp)
                .ok_or_else(|| overflow("fixed pool total"))?;
        }
        if pool_bytes < fixed_total {
            return Err(StateError::invalid(vec![InvalidItem {
                index: u32::MAX,
                reason: format!(
                    "pool_bytes={pool_bytes} below fixed_pool={fixed_total}, shortfall={}",
                    fixed_total - pool_bytes,
                ),
            }]));
        }
        // Safe: `pool_bytes >= fixed_total` was just established.
        let usable = pool_bytes - fixed_total;

        // DECISION(A1.11): after fixed pools, split the usable paged bytes in
        // proportion to group block costs so every paged group gets the same
        // aggregate block capacity, at least max_ctx/32 per group (Spec 3
        // §6.3: the minimum pool size must guarantee full context for one
        // sequence across all groups).
        let paged_groups: Vec<&LayerGroup> = groups.iter().filter(|g| g.spec.is_paged()).collect();
        let mut block_cost_sum: u64 = 0;
        for g in &paged_groups {
            let bb = g.block_bytes()?;
            block_cost_sum = block_cost_sum
                .checked_add(bb)
                .ok_or_else(|| overflow("block cost sum"))?;
        }

        let (blocks_per_group, assigned_paged) = if paged_groups.is_empty() {
            // Purely recurrent / conv model: no paged pools.
            (0u32, 0u64)
        } else {
            let blocks_u64 = usable.checked_div(block_cost_sum).ok_or_else(|| {
                StateError::invalid(vec![InvalidItem {
                    index: u32::MAX,
                    reason: "paged block cost is zero; cannot proportion the pool".to_owned(),
                }])
            })?;
            let blocks_per_group =
                u32::try_from(blocks_u64).map_err(|_| StateError::InvalidConfig {
                    problems: vec![InvalidItem {
                        index: u32::MAX,
                        reason: format!(
                            "assigned blocks_per_group={blocks_u64} exceeds u32 slot_map range {}",
                            MAX_SLOT_BLOCKS
                        ),
                    }],
                })?;
            if blocks_per_group > MAX_SLOT_BLOCKS {
                return Err(StateError::InvalidConfig {
                    problems: vec![InvalidItem {
                        index: u32::MAX,
                        reason: format!(
                            "assigned blocks_per_group={blocks_per_group} exceeds u32 slot_map range {}",
                            MAX_SLOT_BLOCKS
                        ),
                    }],
                });
            }
            if blocks_per_group < max_blocks {
                let need_paged = u64::from(max_blocks)
                    .checked_mul(block_cost_sum)
                    .ok_or_else(|| overflow("minimum paged pool"))?;
                let required = fixed_total
                    .checked_add(need_paged)
                    .ok_or_else(|| overflow("minimum pool"))?;
                let shortfall = required - pool_bytes;
                return Err(StateError::invalid(vec![InvalidItem {
                    index: u32::MAX,
                    reason: format!(
                        "pool_bytes={pool_bytes} below required={required}, shortfall={shortfall}",
                    ),
                }]));
            }
            let assigned = u64::from(blocks_per_group)
                .checked_mul(block_cost_sum)
                .ok_or_else(|| overflow("assigned paged bytes"))?;
            (blocks_per_group, assigned)
        };

        // Safe: `assigned_paged <= usable` by construction of the division.
        let unusable = usable - assigned_paged;

        let mut pools: Vec<Option<BlockPool>> = Vec::with_capacity(groups.len());
        let mut fixed: Vec<Option<FixedPool>> = Vec::with_capacity(groups.len());
        let mut base: u64 = 0;
        for g in &groups {
            if g.spec.is_paged() {
                let block_bytes = g.block_bytes()?;
                let bytes = u64::from(blocks_per_group)
                    .checked_mul(block_bytes)
                    .ok_or_else(|| overflow("group pool bytes"))?;
                let group_base = base;
                base = base
                    .checked_add(bytes)
                    .ok_or_else(|| overflow("arena base"))?;
                pools.push(Some(BlockPool {
                    total: blocks_per_group,
                    // Descending so `alloc` pops the smallest id; capacity is
                    // exactly `total`, so later releases never reallocate.
                    free: (0..blocks_per_group).rev().collect(),
                    base_offset: group_base,
                    block_bytes,
                }));
                fixed.push(None);
            } else {
                let per_seq = g.slots_bytes_per_seq()?;
                let grp_bytes = per_seq
                    .checked_mul(u64::from(max_seqs))
                    .ok_or_else(|| overflow("fixed group bytes"))?;
                // `per_seq` already counts both A and B buffers, so each
                // buffer is exactly half; zero-size slots (e.g. `w = 1`
                // conv) still allocate slot ids for ownership bookkeeping.
                let buffer_bytes = per_seq / 2;
                let group_base = base;
                base = base
                    .checked_add(grp_bytes)
                    .ok_or_else(|| overflow("arena base"))?;
                fixed.push(Some(FixedPool {
                    total_slots: max_seqs,
                    free: (0..max_seqs).collect(),
                    slot_bytes_per_seq: per_seq,
                    buffer_bytes,
                    base_offset: group_base,
                }));
                pools.push(None);
            }
        }

        // `reserve` scratch fits every legal reservation: each group's
        // missing block indices lie in `[0, max_blocks)`, so `paged groups ×
        // max_blocks` entries cover the worst case (`n = max_ctx` on every
        // paged group at once). Sized once here, on the cold path.
        let paged_count = groups.iter().filter(|g| g.spec.is_paged()).count();
        let scratch_cap = paged_count
            .checked_mul(max_blocks as usize)
            .ok_or_else(|| overflow("reserve scratch capacity"))?;
        let mut reserve_scratch = Vec::new();
        reserve_scratch
            .try_reserve_exact(scratch_cap)
            .map_err(|_| StateError::Overflow {
                what: "reserve scratch capacity".to_owned(),
            })?;

        Ok(Self {
            config,
            groups,
            pools,
            fixed,
            max_blocks,
            pool_bytes_total: assigned_paged,
            fixed_bytes_total: fixed_total,
            unusable_bytes: unusable,
            next_seq: 0,
            seqs: BTreeMap::new(),
            live_count: 0,
            swaps: 0,
            commits: 0,
            reserve_scratch,
        })
    }

    /// Layer-groups in `BatchMeta` order (Spec 3 §6.1).
    pub fn groups(&self) -> &[LayerGroup] {
        &self.groups
    }

    /// Engine config.
    pub const fn config(&self) -> StateConfig {
        self.config
    }

    /// Starts a sequence (Spec 3 §5 `new_seq`).
    ///
    /// No prefix cache yet: `matched_len` is always 0 and no blocks are
    /// allocated or shared; the prompt is covered by the first `reserve`
    /// (roadmap B1 owns prefix/session reuse).
    ///
    /// The token slice is only the future prefix-cache key: length validation
    /// is sufficient here and contents are never interpreted. Allocation is
    /// atomic across groups: every fixed group must have a free slot before
    /// any is taken, so a refusal mutates nothing.
    pub fn new_seq(&mut self, tokens: &[u32]) -> StateResult<(SeqId, u32)> {
        let len_u32 = u32::try_from(tokens.len()).map_err(|_| StateError::Overflow {
            what: "prompt length".to_owned(),
        })?;
        if tokens.len() > self.config.max_ctx as usize {
            return Err(StateError::ReserveTooLarge {
                end: len_u32,
                max_ctx: self.config.max_ctx,
                n: len_u32,
            });
        }
        // Fallible checks first: the live-count and id counters must have room
        // before checking limits or taking slots.
        let next_live = self
            .live_count
            .checked_add(1)
            .ok_or_else(|| StateError::Overflow {
                what: "live sequence count".to_owned(),
            })?;
        // DECISION(A1.15): sequence allocation checks that next_seq <= u32::MAX before mutating state; rejected lossy as-casts, rollover, batch-local indexing that breaks Philox batch invariance, and unbounded host-to-device ID maps. Spec 1 §2.5, §4.F, Spec 3 §5, SI-40.
        if self.next_seq > u64::from(u32::MAX) {
            return Err(StateError::SeqIdOverflow {
                seq: self.next_seq,
                max: u32::MAX,
            });
        }
        let next_id = self
            .next_seq
            .checked_add(1)
            .ok_or_else(|| StateError::Overflow {
                what: "sequence id".to_owned(),
            })?;
        let live = self.live_count;
        if live >= self.config.max_seqs {
            return Err(StateError::SeqLimit {
                live,
                cap: self.config.max_seqs,
            });
        }
        for (gi, g) in self.groups.iter().enumerate() {
            if !g.spec.is_recurrent() {
                continue;
            }
            let pool = self.fixed.get(gi).and_then(|p| p.as_ref()).ok_or_else(|| {
                StateError::InvalidBatch {
                    detail: format!("group {gi} is not a fixed group"),
                }
            })?;
            if pool.free.is_empty() {
                return Err(StateError::SeqLimit {
                    live,
                    cap: self.config.max_seqs,
                });
            }
        }
        // DECISION(A1.11): deterministic smallest-free sequence-slot
        // allocation per fixed group, two buffers per sequence inside the
        // slot; rejected: implicit parity-only bookkeeping with no pool
        // ownership (loses release/reuse accounting and exact budgeting).
        // All groups were pre-checked, but any later failure rolls back
        // already taken slots so no resources leak.
        let mut fixed_slots: Vec<Option<u32>> = vec![None; self.groups.len()];
        for (gi, g) in self.groups.iter().enumerate() {
            if !g.spec.is_recurrent() {
                continue;
            }
            let pool = match self.fixed.get_mut(gi).and_then(|p| p.as_mut()) {
                Some(p) => p,
                None => {
                    for (rgi, rslot) in fixed_slots.into_iter().enumerate() {
                        if let Some(sid) = rslot {
                            if let Some(Some(rpool)) = self.fixed.get_mut(rgi) {
                                rpool.free.insert(sid);
                            }
                        }
                    }
                    return Err(StateError::InvalidBatch {
                        detail: format!("group {gi} is not a fixed group"),
                    });
                }
            };
            let id = match pool.free.iter().next().copied() {
                Some(id) => id,
                None => {
                    for (rgi, rslot) in fixed_slots.into_iter().enumerate() {
                        if let Some(sid) = rslot {
                            if let Some(Some(rpool)) = self.fixed.get_mut(rgi) {
                                rpool.free.insert(sid);
                            }
                        }
                    }
                    return Err(StateError::SeqLimit {
                        live,
                        cap: self.config.max_seqs,
                    });
                }
            };
            pool.free.remove(&id);
            fixed_slots[gi] = Some(id);
        }
        let id = self.next_seq;
        // DECISION(A1.16): per-sequence paged tables are presized to
        // `max_blocks` at admission so steady-state `reserve` pushes never
        // reallocate, including new-block boundary transitions; rejected:
        // growing them geometrically per call (a heap allocation mid-run).
        // Admission-time sizing only; `max_ctx`/`max_seqs` semantics are
        // unchanged. Spec 3 §3.3, §6.3.
        let max_blocks = self.max_blocks as usize;
        let mut tables: Vec<Vec<u32>> = Vec::with_capacity(self.groups.len());
        let mut indices: Vec<Vec<u32>> = Vec::with_capacity(self.groups.len());
        for g in &self.groups {
            if g.spec.is_paged() {
                tables.push(Vec::with_capacity(max_blocks));
                indices.push(Vec::with_capacity(max_blocks));
            } else {
                tables.push(Vec::new());
                indices.push(Vec::new());
            }
        }
        self.seqs.insert(
            id,
            SeqState {
                ctx_len: 0,
                tail_len: 0,
                tables,
                indices,
                mirror: BTreeMap::new(),
                compacted: None,
                fixed_slots,
                parity: vec![0; self.groups.len()],
            },
        );
        self.next_seq = next_id;
        self.live_count = next_live;
        Ok((SeqId::new(id), 0))
    }

    fn seq_mut(&mut self, seq: SeqId) -> StateResult<&mut SeqState> {
        self.seqs
            .get_mut(&seq.as_u64())
            .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })
    }

    fn seq(&self, seq: SeqId) -> StateResult<&SeqState> {
        self.seqs
            .get(&seq.as_u64())
            .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })
    }

    /// Verified length (Spec 3 §3.3 `ctx_len`).
    pub fn ctx_len(&self, seq: SeqId) -> StateResult<u32> {
        Ok(self.seq(seq)?.ctx_len)
    }

    /// Outstanding (reserved, uncommitted) tokens (Spec 3 §3.6 `tail_len`).
    pub fn tail_len(&self, seq: SeqId) -> StateResult<u32> {
        Ok(self.seq(seq)?.tail_len)
    }

    /// First retained position for a group (Spec 3 §3.5 `window_start`).
    pub fn window_start(&self, seq: SeqId, group: usize) -> StateResult<u32> {
        let s = self.seq(seq)?;
        let spec = self
            .groups
            .get(group)
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} out of range {}", self.groups.len()),
            })?;
        Ok(match spec.spec.retain() {
            None | Some(crate::spec::Retain::All) => 0,
            // DECISION(A1.11) per SI-17: Sink+Window reports the window
            // start; the sink range is implicitly [0, ceil(n/32)*32) pinned
            // from position 0. Rejected: reporting 0 (would hide the window
            // from the kernel). Spec 3 §3.5 gives the kernel "both ranges"
            // but BatchMeta carries a single window_start per group.
            Some(crate::spec::Retain::Window { w })
            | Some(crate::spec::Retain::SinkWindow { w, .. }) => window_start_of(s.ctx_len, w),
        })
    }

    /// Active double-buffer slot for a group: `0 = A`, `1 = B` (Spec 3 §4.2).
    pub fn recurrent_active(&self, seq: SeqId, group: usize) -> StateResult<u8> {
        let s = self.seq(seq)?;
        s.parity
            .get(group)
            .copied()
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} out of range {}", self.groups.len()),
            })
    }

    /// Owned sequence slot id for a recurrent/conv group (Spec 3 §4.1).
    ///
    /// Deterministic smallest-free allocation at `new_seq`, released at
    /// `free_seq`. Paged groups hold no sequence slots and are refused.
    pub fn fixed_slot(&self, seq: SeqId, group: usize) -> StateResult<u32> {
        let s = self.seq(seq)?;
        let spec = self
            .groups
            .get(group)
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} out of range {}", self.groups.len()),
            })?;
        if !spec.spec.is_recurrent() {
            return Err(StateError::InvalidBatch {
                detail: format!("group {group} is not a recurrent/conv group"),
            });
        }
        s.fixed_slots
            .get(group)
            .and_then(|slot| *slot)
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("sequence {} holds no slot in group {group}", seq.as_u64()),
            })
    }

    /// Active and working buffer ids `(active, working)` for a
    /// recurrent/conv group: buffer `2 * slot + parity` (Spec 3 §4.2).
    ///
    /// The scheduler reads verified state from the active buffer and writes
    /// the step into the working buffer; `commit` with any `accepted > 0`
    /// swaps them, so the next step's active buffer holds the verified state.
    pub fn recurrent_buffers(&self, seq: SeqId, group: usize) -> StateResult<(u32, u32)> {
        let slot = self.fixed_slot(seq, group)?;
        let active_bit = u32::from(self.recurrent_active(seq, group)?);
        let base = slot.checked_mul(2).ok_or_else(|| StateError::Overflow {
            what: "fixed buffer id".to_owned(),
        })?;
        let active = base
            .checked_add(active_bit)
            .ok_or_else(|| StateError::Overflow {
                what: "fixed buffer id".to_owned(),
            })?;
        // The working buffer is the other half of the slot: parity only ever
        // holds 0 or 1 (flipped by `commit`); anything else fails closed.
        let working_bit = match active_bit {
            0 => 1,
            1 => 0,
            _ => {
                return Err(StateError::Overflow {
                    what: "fixed buffer parity".to_owned(),
                });
            }
        };
        let working = base
            .checked_add(working_bit)
            .ok_or_else(|| StateError::Overflow {
                what: "fixed buffer id".to_owned(),
            })?;
        Ok((active, working))
    }

    /// Byte offset of a fixed-pool buffer in the abstract arena (Spec 3 §4.1).
    ///
    /// Buffers outside the group's `2 * total_slots` range are refused
    /// before any arithmetic.
    pub fn fixed_buffer_offset(&self, group: usize, buffer: u32) -> StateResult<u64> {
        let pool = self
            .fixed
            .get(group)
            .and_then(|p| p.as_ref())
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} is not a recurrent/conv group"),
            })?;
        let total_buffers =
            pool.total_slots
                .checked_mul(2)
                .ok_or_else(|| StateError::Overflow {
                    what: "fixed buffer range".to_owned(),
                })?;
        if buffer >= total_buffers {
            return Err(StateError::OutOfRange {
                start: buffer,
                len: 1,
                end: total_buffers,
            });
        }
        pool.buffer_offset(buffer)
    }

    /// Free blocks in a paged group pool.
    pub fn free_blocks(&self, group: usize) -> StateResult<u32> {
        let pool = self
            .pools
            .get(group)
            .and_then(|p| p.as_ref())
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} is not a paged group"),
            })?;
        u32::try_from(pool.free.len()).map_err(|_| StateError::Overflow {
            what: "free block count".to_owned(),
        })
    }

    /// Free sequence slots in a recurrent/conv group pool.
    pub fn free_slots(&self, group: usize) -> StateResult<u32> {
        let pool = self
            .fixed
            .get(group)
            .and_then(|p| p.as_ref())
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} is not a recurrent/conv group"),
            })?;
        u32::try_from(pool.free.len()).map_err(|_| StateError::Overflow {
            what: "free slot count".to_owned(),
        })
    }

    /// Ensures blocks exist for `ctx_len .. ctx_len + n` (Spec 3 §3.6).
    ///
    /// Windowed groups allocate every block touched by the full new range,
    /// even when `n` exceeds the window: retention releases only on `commit`,
    /// so the reservation must cover the whole `ctx_len .. end` span.
    /// Atomic: every group is checked before any block is allocated, and a
    /// failed slot build rolls its inserts back, so a refusal leaves the
    /// sequence and the pools untouched.
    pub fn reserve(&mut self, seq: SeqId, n: u32) -> StateResult<SlotRange> {
        if n == 0 || n > MAX_RESERVE_HARD {
            let tail = self.seq(seq)?.tail_len;
            return Err(StateError::InvalidReserve { n, tail });
        }
        let (ctx, tail) = {
            let s = self.seq(seq)?;
            (s.ctx_len, s.tail_len)
        };
        if tail != 0 {
            return Err(StateError::InvalidReserve { n, tail });
        }
        let end = ctx.checked_add(n).ok_or_else(|| StateError::Overflow {
            what: "reserve end".to_owned(),
        })?;
        if end > self.config.max_ctx {
            return Err(StateError::ReserveTooLarge {
                end,
                max_ctx: self.config.max_ctx,
                n,
            });
        }

        // Full-range block span for the new range. Retention has not run for
        // these positions yet, so nothing is window-clipped here: a reserve
        // of 64 tokens with a 32-token window still takes both blocks, and
        // `commit` releases the aged one afterwards.
        let first_block = ctx / BLOCK_TOKENS;
        let last_block = end.div_ceil(BLOCK_TOKENS);
        // Pass 1 (shared borrows only): check every group before allocating
        // anything, recording missing `(group, index)` pairs into reusable
        // owned scratch. Membership is an allocation-free binary search over
        // the sorted per-group tables (Spec 3 §3.3: ascending block order).
        {
            let Self {
                groups,
                pools,
                seqs,
                reserve_scratch,
                ..
            } = &mut *self;
            reserve_scratch.clear();
            let s = seqs
                .get(&seq.as_u64())
                .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })?;
            for (gi, g) in groups.iter().enumerate() {
                if !g.spec.is_paged() {
                    continue;
                }
                let group = u32::try_from(gi).map_err(|_| StateError::Overflow {
                    what: "group index".to_owned(),
                })?;
                let held = &s.indices[gi];
                let before = reserve_scratch.len();
                for idx in first_block..last_block {
                    let at = held.partition_point(|&i| i < idx);
                    if held.get(at) != Some(&idx) {
                        reserve_scratch.push((group, idx));
                    }
                }
                let want = u64::try_from(reserve_scratch.len() - before).map_err(|_| {
                    StateError::Overflow {
                        what: "missing block count".to_owned(),
                    }
                })?;
                let free = u64::try_from(pools[gi].as_ref().map_or(0, |p| p.free.len())).map_err(
                    |_| StateError::Overflow {
                        what: "free block count".to_owned(),
                    },
                )?;
                if want > free {
                    let available = u32::try_from(free).map_err(|_| StateError::Overflow {
                        what: "free block count".to_owned(),
                    })?;
                    let required = u32::try_from(want).map_err(|_| StateError::Overflow {
                        what: "missing block count".to_owned(),
                    })?;
                    return Err(StateError::PoolExhausted {
                        group: gi,
                        required,
                        available,
                        shortfall: required.checked_sub(available).ok_or_else(|| {
                            StateError::Overflow {
                                what: "pool shortfall".to_owned(),
                            }
                        })?,
                        end,
                        max_ctx: self.config.max_ctx,
                    });
                }
            }
        }

        // Pass 2: allocate. Counts were verified above on this same call with
        // no interleaving (`&mut self`), so every step below succeeds on a
        // consistent manager; every remainder is still a typed error (never
        // `expect`/panic), and a mid-pass failure rolls its own prefix back,
        // so a refusal leaves the sequence and the pools untouched. Table
        // pushes land within the admission-sized capacity: no heap
        // allocation, including on block boundary transitions and on the
        // first call after admission.
        self.place_reserve_blocks(seq, end)?;
        let s = self.seq_mut(seq)?;
        s.tail_len = n;
        s.compacted = None;
        Ok(SlotRange {
            seq,
            start: ctx,
            len: n,
        })
    }

    /// Allocates every block recorded in [`Self::reserve_scratch`], in
    /// ascending group/index order (deterministic smallest-free ids, Spec 3
    /// §5, §8). A typed failure rolls the placed prefix back before
    /// returning, so the sequence and the pools are untouched.
    fn place_reserve_blocks(&mut self, seq: SeqId, end: u32) -> StateResult<()> {
        let Self {
            pools,
            seqs,
            reserve_scratch,
            config,
            ..
        } = &mut *self;
        let max_ctx = config.max_ctx;
        let mut placed = 0usize;
        let mut failure: Option<StateError> = None;
        for &(group, idx) in reserve_scratch.iter() {
            let gi = group as usize;
            // Unreachable on a consistent manager: pass 1 counted a free
            // block for every recorded index on this same call. Typed (never
            // `expect`) so even a corrupted pool fails closed with the
            // numbers, after the prefix rolls back below.
            let step = place_one_block(pools, seqs, seq, gi, idx, end, max_ctx);
            match step {
                Ok(()) => placed += 1,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = failure {
            // Roll back exactly the placed prefix: remove those entries and
            // return their ids. Each removal re-searches by binary search, so
            // order is irrelevant and no allocation is needed. The `release`
            // calls cannot fail here (the ids were just allocated), but any
            // failure still propagates typed, never panics.
            for &(rgroup, ridx) in reserve_scratch.iter().take(placed) {
                let rgi = rgroup as usize;
                let mut released: Option<u32> = None;
                if let Some(s) = seqs.get_mut(&seq.as_u64()) {
                    let at = s
                        .indices
                        .get(rgi)
                        .map(|v| v.partition_point(|&i| i < ridx))
                        .unwrap_or(usize::MAX);
                    let aligned = s.indices.get(rgi).and_then(|v| v.get(at)).copied() == Some(ridx)
                        && s.tables.get(rgi).is_some_and(|t| at < t.len());
                    if aligned {
                        s.indices[rgi].remove(at);
                        released = Some(s.tables[rgi].remove(at));
                    }
                }
                if let Some(id) = released {
                    if let Some(pool) = pools.get_mut(rgi).and_then(|p| p.as_mut()) {
                        pool.release(id)?;
                    }
                }
            }
            return Err(e);
        }
        Ok(())
    }

    /// Flattened slot for token `k` of `range` in group `group`
    /// (`slots[g][k]` covers `start + k`; Spec 1 §2.5, Spec 3 §3.3).
    ///
    /// Hot path: shared borrows only, no heap allocation. Recurrent/conv
    /// groups report [`SLOT_NONE`]; paged groups resolve through the live
    /// block tables. The range must be the sequence's current open
    /// reservation (`start == ctx_len`, `len == tail_len`); a stale
    /// (post-commit/freed/superseded) or foreign descriptor is a typed
    /// error, never a read of another step's tail. A missing mapping is a
    /// typed error, never a clamp to a neighboring block.
    pub fn slot(&self, range: &SlotRange, group: usize, k: u32) -> StateResult<u32> {
        let g = self
            .groups
            .get(group)
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} out of range {}", self.groups.len()),
            })?;
        if k >= range.len() {
            return Err(StateError::OutOfRange {
                start: k,
                len: 1,
                end: range.len(),
            });
        }
        let s = self
            .seqs
            .get(&range.seq().as_u64())
            .ok_or(StateError::UnknownSeq {
                seq: range.seq().as_u64(),
            })?;
        check_range_live(s, range)?;
        if !g.spec.is_paged() {
            return Ok(SLOT_NONE);
        }
        let pos = range
            .start()
            .checked_add(k)
            .ok_or_else(|| StateError::Overflow {
                what: "slot position".to_owned(),
            })?;
        flatten_slot(s, group, pos, range.end())
    }

    /// Fills a caller-owned row with every slot of `range` in group `group`
    /// (`out[k]` covers `start + k`; Spec 1 §2.5, Spec 3 §3.3).
    ///
    /// Hot path: the caller owns the buffer (scheduler upload staging), so
    /// filling never allocates regardless of width. A short buffer is a
    /// typed error before anything is written; a mid-row mapping failure is
    /// a typed error naming the position (the buffer is caller scratch, not
    /// manager state, so no rollback applies).
    pub fn fill_slots(&self, range: &SlotRange, group: usize, out: &mut [u32]) -> StateResult<()> {
        let need = range.len() as usize;
        if out.len() < need {
            return Err(StateError::OutOfRange {
                start: 0,
                len: out.len(),
                end: range.len(),
            });
        }
        let g = self
            .groups
            .get(group)
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} out of range {}", self.groups.len()),
            })?;
        let s = self
            .seqs
            .get(&range.seq().as_u64())
            .ok_or(StateError::UnknownSeq {
                seq: range.seq().as_u64(),
            })?;
        check_range_live(s, range)?;
        if !g.spec.is_paged() {
            out[..need].fill(SLOT_NONE);
            return Ok(());
        }
        let end = range.end();
        for (k, cell) in out[..need].iter_mut().enumerate() {
            let pos = range
                .start()
                .checked_add(k as u32)
                .ok_or_else(|| StateError::Overflow {
                    what: "slot position".to_owned(),
                })?;
            *cell = flatten_slot(s, group, pos, end)?;
        }
        Ok(())
    }

    /// Simulates `state_write_kv` into reserved slots (Spec 3 §8 test support).
    ///
    /// Records token ids for `start .. start + tokens.len()`; the range must
    /// lie inside the reserved region. The device path is owned by the
    /// scheduler as ops; this mirrors it for host-side law tests.
    pub fn write_tokens(&mut self, seq: SeqId, start: u32, tokens: &[u32]) -> StateResult<()> {
        let s = self.seq(seq)?;
        let len = u32::try_from(tokens.len()).map_err(|_| StateError::Overflow {
            what: "write length".to_owned(),
        })?;
        let write_end = start.checked_add(len).ok_or_else(|| StateError::Overflow {
            what: "write end".to_owned(),
        })?;
        let reserved_end =
            s.ctx_len
                .checked_add(s.tail_len)
                .ok_or_else(|| StateError::Overflow {
                    what: "reserved end".to_owned(),
                })?;
        // Writes must land inside the open reservation: before `ctx_len` is
        // already-verified history, past the reserved end is unmapped.
        if start < s.ctx_len {
            return Err(StateError::OutOfRange {
                start,
                len: tokens.len(),
                end: reserved_end,
            });
        }
        if write_end > reserved_end {
            return Err(StateError::OutOfRange {
                start,
                len: tokens.len(),
                end: reserved_end,
            });
        }
        // DECISION(A1.16): no collected `is_paged` vector — the group
        // loop borrows `groups` immutably to test paging, then takes the
        // sequence borrow per paged group. A collected `Vec<bool>` would heap
        // allocate on every mirror write, including the counted tree flow.
        // Mirror inserts of new keys still allocate BTreeMap nodes by
        // construction (test support only; the device path owns scheduler
        // writes as ops): replacing existing keys allocates nothing, and
        // removing missing keys allocates nothing, which the counted
        // device-only and replace flows prove. Spec 3 §8.
        let paged_count = self.groups.len();
        for gi in 0..paged_count {
            let is_paged = self
                .groups
                .get(gi)
                .map(|g| g.spec.is_paged())
                .unwrap_or(false);
            if !is_paged {
                continue;
            }
            let s = self.seq_mut(seq)?;
            for (k, tok) in tokens.iter().enumerate() {
                let pos = start
                    .checked_add(k as u32)
                    .ok_or_else(|| StateError::Overflow {
                        what: "write position".to_owned(),
                    })?;
                s.mirror.insert((gi, pos), *tok);
            }
        }
        Ok(())
    }

    /// Reads a mirrored token id (Spec 3 §8 test support).
    pub fn read_token(&self, seq: SeqId, group: usize, pos: u32) -> StateResult<Option<u32>> {
        let s = self.seq(seq)?;
        if group >= self.groups.len() {
            return Err(StateError::InvalidBatch {
                detail: format!("group {group} out of range {}", self.groups.len()),
            });
        }
        let end = s
            .ctx_len
            .checked_add(s.tail_len)
            .ok_or_else(|| StateError::Overflow {
                what: "reserved end".to_owned(),
            })?;
        if pos >= end {
            return Err(StateError::OutOfRange {
                start: pos,
                len: 1,
                end,
            });
        }
        Ok(s.mirror.get(&(group, pos)).copied())
    }

    /// Tree-verify compaction: copies accepted tokens' K/V into
    /// `ctx_len .. ctx_len + a` within the same blocks, then the caller
    /// commits (Spec 3 §3.6).
    ///
    /// Applies the copy to the in-memory mirror eagerly and returns the
    /// descriptor the scheduler enqueues on device. Atomic: validation
    /// precedes any mutation.
    ///
    /// Hot path: success allocates nothing from the cold first verify step.
    /// At most [`MAX_COMPACT_TOKENS`] accepted positions (spec 1 §4, spec 7
    /// §5, spec 12 §3); duplicates are an O(n^2) stack scan and overlap
    /// staging is per-group stack copies. Mirror updates replace existing
    /// keys (allocation-free) or remove missing keys (allocation-free); the
    /// device-only flow with no mirror entries therefore allocates nothing.
    /// Inserting brand-new mirror keys (partial-mirror test writes) still
    /// allocates BTreeMap nodes by construction — test support only.
    pub fn compact(&mut self, seq: SeqId, accepted_positions: &[u32]) -> StateResult<CompactOp> {
        let (ctx, tail) = {
            let s = self.seq(seq)?;
            (s.ctx_len, s.tail_len)
        };
        let detail = |msg: String| StateError::InvalidCompact {
            len: accepted_positions.len(),
            tail,
            detail: msg,
        };
        if tail == 0 {
            return Err(detail("no outstanding tail".to_owned()));
        }
        if accepted_positions.len() > MAX_COMPACT_TOKENS {
            return Err(detail(format!(
                "accepted len {} exceeds cap {MAX_COMPACT_TOKENS}",
                accepted_positions.len()
            )));
        }
        // Bounded duplicate + range scan on the stack: no `BTreeSet`, no heap.
        for (i, p) in accepted_positions.iter().enumerate() {
            if *p >= tail {
                return Err(detail(format!("position {p} out of range tail {tail}")));
            }
            for q in &accepted_positions[..i] {
                if q == p {
                    return Err(detail(format!("duplicate position {p}")));
                }
            }
        }
        let a = accepted_positions.len();
        // Fixed-capacity sources in accepted-path order; destinations are
        // `ctx + i` computed on the fly below, so no `dsts` vec is needed.
        // Checked before any mutation, so the write phase cannot fail partway.
        let mut src = [0u32; MAX_COMPACT_TOKENS];
        for (i, p) in accepted_positions.iter().enumerate() {
            src[i] = ctx.checked_add(*p).ok_or_else(|| StateError::Overflow {
                what: "compact source".to_owned(),
            })?;
        }
        for i in 0..a {
            ctx.checked_add(i as u32)
                .ok_or_else(|| StateError::Overflow {
                    what: "compact destination".to_owned(),
                })?;
        }
        // Read-then-write per paged group with a stack staging copy, so
        // overlapping src/dst sets compact correctly. Staging holds at most
        // 16 tokens per group; groups are visited in index order
        // (deterministic, Spec 1 §6.1).
        let group_count = self.groups.len();
        for gi in 0..group_count {
            let is_paged = self
                .groups
                .get(gi)
                .map(|g| g.spec.is_paged())
                .unwrap_or(false);
            if !is_paged {
                continue;
            }
            let mut staged: [Option<u32>; MAX_COMPACT_TOKENS] = [None; MAX_COMPACT_TOKENS];
            {
                let s = self.seq(seq)?;
                for (slot, abs) in staged.iter_mut().zip(src.iter()).take(a) {
                    *slot = s.mirror.get(&(gi, *abs)).copied();
                }
            }
            let s = self.seq_mut(seq)?;
            for (i, tok) in staged.iter().enumerate().take(a) {
                let dst = ctx
                    .checked_add(i as u32)
                    .ok_or_else(|| StateError::Overflow {
                        what: "compact destination".to_owned(),
                    })?;
                match tok {
                    Some(t) => {
                        s.mirror.insert((gi, dst), *t);
                    }
                    None => {
                        s.mirror.remove(&(gi, dst));
                    }
                }
            }
        }
        let s = self.seq_mut(seq)?;
        s.compacted = Some(a);
        let len = u32::try_from(a).map_err(|_| StateError::Overflow {
            what: "compact length".to_owned(),
        })?;
        Ok(CompactOp {
            seq,
            src,
            dst_start: ctx,
            len,
        })
    }

    /// Commits `accepted` tokens (Spec 3 §3.6).
    ///
    /// `ctx_len += accepted`; `tail_len = 0`. Over-reserved positions stay
    /// allocated and are overwritten by the next reserve — rejection is a
    /// smaller `accepted`, with no data movement. Windowed groups release
    /// blocks older than the window. Full accepts swap recurrent A/B slots;
    /// Any `accepted > 0` swaps the recurrent/conv A/B buffers: the
    /// scheduler re-ran the accepted prefix from verified A into working B
    /// before this call (Spec 3 §4.2), so the working buffer already holds
    /// the verified state and the swap publishes it. Full rejection
    /// (`accepted == 0`) swaps nothing and keeps the checkpoint. Atomic:
    /// validation precedes any mutation.
    ///
    // DECISION(A1.11): the manager swaps on every `accepted > 0` and keeps
    // no pending re-run descriptor; rejected: deferring the swap behind a
    // `recompute_pending` flag cleared by the next reserve (that models the
    // re-run as reserving the accepted tokens again at later positions,
    // which double-counts them — the scheduler already knows accepted versus
    // tail and re-runs the prefix in place before committing).
    pub fn commit(&mut self, seq: SeqId, accepted: u32) -> StateResult<()> {
        let (ctx, tail, compacted) = {
            let s = self.seq(seq)?;
            (s.ctx_len, s.tail_len, s.compacted)
        };
        if tail == 0 {
            return Err(StateError::NoReservation { seq: seq.as_u64() });
        }
        if accepted > tail {
            return Err(StateError::CommitTooLarge { accepted, tail });
        }
        if let Some(c) = compacted {
            if (accepted as usize) != c {
                return Err(StateError::InvalidCompact {
                    len: c,
                    tail,
                    detail: format!("commit accepted={accepted} != compacted={c}"),
                });
            }
        }
        let new_ctx = ctx
            .checked_add(accepted)
            .ok_or_else(|| StateError::Overflow {
                what: "commit ctx".to_owned(),
            })?;
        // Pre-check the stats counters so the mutation phase cannot fail.
        let next_commits = self
            .commits
            .checked_add(1)
            .ok_or_else(|| StateError::Overflow {
                what: "commit count".to_owned(),
            })?;
        let swap = accepted > 0 && self.groups.iter().any(|g| g.spec.is_recurrent());
        let next_swaps = if swap {
            Some(
                self.swaps
                    .checked_add(1)
                    .ok_or_else(|| StateError::Overflow {
                        what: "swap count".to_owned(),
                    })?,
            )
        } else {
            None
        };
        // DECISION(A1.16): window releases are computed as per-group
        // evictable block-index ranges on the stack — no per-call `Vec`s.
        // Held indices are ascending, so the held entries inside
        // `[sink_blocks, ws / 32)` form one contiguous run drained in reverse.
        // `(idx + 1) * 32 <= ws` is exactly `idx < ws / 32` (floor division:
        // `idx + 1 <= ws / 32` over integers), matching the previous
        // per-entry `block_end <= ws` test bit for bit. Rejected: collecting
        // `(position, index)` pairs plus id lists per group (a heap
        // allocation on every windowed commit). Spec 3 §3.5.
        //
        // `groups.len() <= MAX_GROUPS_HARD` is established at construction
        // (`group_layers`/`group_layer_specs` refuse more), so indexing this
        // array by `gi` is always in bounds.
        debug_assert!(self.groups.len() <= MAX_GROUPS_HARD);
        let mut evict: [(u32, u32); MAX_GROUPS_HARD] = [(u32::MAX, 0); MAX_GROUPS_HARD];
        for (gi, g) in self.groups.iter().enumerate() {
            let retain = match g.spec.retain() {
                Some(r) if r.is_windowed() => r,
                _ => continue,
            };
            let w = retain.window().ok_or_else(|| StateError::Overflow {
                what: "window size".to_owned(),
            })?;
            let ws = window_start_of(new_ctx, w);
            let sink_blocks = match retain {
                crate::spec::Retain::SinkWindow { n, .. } => n.div_ceil(BLOCK_TOKENS),
                _ => 0,
            };
            let hi = ws / BLOCK_TOKENS;
            if hi > sink_blocks {
                evict[gi] = (sink_blocks, hi);
            }
        }

        // Pre-validate every evicted block id before mutating anything: an
        // out-of-range or already-free id is a typed error with the sequence
        // and pools untouched (same atomicity as `free_seq`). The scan is
        // allocation-free (binary searches over the sorted tables); it runs
        // only for groups with a nonempty evict range. The releases below
        // then cannot fail on a validated manager, but still propagate typed
        // (never `expect`/panic).
        {
            let s = self.seq(seq)?;
            for (gi, g) in self.groups.iter().enumerate() {
                let (lo, hi) = evict[gi];
                if hi <= lo {
                    continue;
                }
                if !g.spec.is_paged() {
                    return Err(StateError::InvalidBatch {
                        detail: format!("group {gi} is not a paged group"),
                    });
                }
                let pool = self.pools.get(gi).and_then(|p| p.as_ref()).ok_or_else(|| {
                    StateError::InvalidBatch {
                        detail: format!("group {gi} is not a paged group"),
                    }
                })?;
                let indices = s.indices.get(gi).ok_or_else(|| StateError::InvalidBatch {
                    detail: format!("sequence {} has no table", seq.as_u64()),
                })?;
                let tables = s.tables.get(gi).ok_or_else(|| StateError::InvalidBatch {
                    detail: format!("sequence {} has no table", seq.as_u64()),
                })?;
                let start_at = indices.partition_point(|&i| i < lo);
                let end_at = indices.partition_point(|&i| i < hi);
                for at in start_at..end_at {
                    let id = tables
                        .get(at)
                        .copied()
                        .ok_or_else(|| StateError::InvalidBatch {
                            detail: format!("sequence {} has no table", seq.as_u64()),
                        })?;
                    if id >= pool.total {
                        return Err(StateError::InvalidBatch {
                            detail: format!(
                                "release block {id} out of range: pool holds {}",
                                pool.total
                            ),
                        });
                    }
                    if pool.contains(id) {
                        return Err(StateError::InvalidBatch {
                            detail: format!("double release of block {id}"),
                        });
                    }
                }
            }
        }

        // Mutate (disjoint field borrows: seqs, pools, groups, swaps,
        // commits). Every fallible step propagates typed errors; the
        // evict-id pre-check above keeps window releases transactional.
        {
            let Self {
                seqs,
                pools,
                groups,
                swaps,
                commits,
                ..
            } = self;
            let s = seqs
                .get_mut(&seq.as_u64())
                .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })?;
            s.ctx_len = new_ctx;
            s.tail_len = 0;
            s.compacted = None;
            for (gi, g) in groups.iter().enumerate() {
                let (lo, hi) = evict[gi];
                if hi <= lo {
                    continue;
                }
                if !g.spec.is_paged() {
                    return Err(StateError::InvalidBatch {
                        detail: format!("group {gi} is not a paged group"),
                    });
                }
                let held = &s.indices[gi];
                let start_at = held.partition_point(|&i| i < lo);
                let end_at = held.partition_point(|&i| i < hi);
                let pool = pools.get_mut(gi).and_then(|p| p.as_mut()).ok_or_else(|| {
                    StateError::InvalidBatch {
                        detail: format!("group {gi} is not a paged group"),
                    }
                })?;
                for at in (start_at..end_at).rev() {
                    s.indices[gi].remove(at);
                    let id = s.tables[gi].remove(at);
                    pool.release(id)?;
                }
                // Released blocks are gone: drop their mirrored tokens so a
                // re-read of an evicted position reports absence (Spec 3 §3.5).
                // Mirror entries always name mapped blocks, so testing the
                // evicted index range is exactly testing the dropped set.
                s.mirror.retain(|(g, p), _| {
                    *g != gi || {
                        let b = *p / crate::spec::BLOCK_TOKENS;
                        b < lo || b >= hi
                    }
                });
            }
            if swap {
                for g in groups.iter() {
                    if g.spec.is_recurrent() {
                        s.parity[g.index] ^= 1;
                    }
                }
                if let Some(next) = next_swaps {
                    *swaps = next;
                }
            }
            *commits = next_commits;
        }
        Ok(())
    }

    /// Releases all references; may retain session state (Spec 3 §5).
    ///
    /// Removes the sequence, its block references, its fixed slots, and its
    /// mirrors outright: no dead map entries remain, so repeated create/free
    /// cycles cannot grow unbounded tombstones. Session retention is deferred
    /// to roadmap B1: everything is released and nothing is retained.
    pub fn free_seq(&mut self, seq: SeqId) -> StateResult<()> {
        // Validation phase: verify all preconditions, referenced pools,
        // slots/blocks, and live-count without mutating any state.
        let s = self
            .seqs
            .get(&seq.as_u64())
            .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })?;

        let next_live = self
            .live_count
            .checked_sub(1)
            .ok_or_else(|| StateError::Overflow {
                what: "live sequence count".to_owned(),
            })?;

        for (gi, ids) in s.tables.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let pool = self.pools.get(gi).and_then(|p| p.as_ref()).ok_or_else(|| {
                StateError::InvalidBatch {
                    detail: format!("group {gi} is not a paged group"),
                }
            })?;
            let mut seen = BTreeSet::new();
            for &id in ids {
                if id >= pool.total {
                    return Err(StateError::InvalidBatch {
                        detail: format!(
                            "block {id} in group {gi} exceeds pool capacity {}",
                            pool.total
                        ),
                    });
                }
                if pool.contains(id) {
                    return Err(StateError::InvalidBatch {
                        detail: format!("block {id} in group {gi} is already free"),
                    });
                }
                if !seen.insert(id) {
                    return Err(StateError::InvalidBatch {
                        detail: format!("block {id} in group {gi} is duplicated in sequence table"),
                    });
                }
            }
        }

        for (gi, slot) in s.fixed_slots.iter().enumerate() {
            let Some(&id) = slot.as_ref() else { continue };
            let pool = self.fixed.get(gi).and_then(|p| p.as_ref()).ok_or_else(|| {
                StateError::InvalidBatch {
                    detail: format!("group {gi} is not a fixed group"),
                }
            })?;
            if id >= pool.total_slots {
                return Err(StateError::InvalidBatch {
                    detail: format!(
                        "fixed slot {id} in group {gi} exceeds pool capacity {}",
                        pool.total_slots
                    ),
                });
            }
            if pool.free.contains(&id) {
                return Err(StateError::InvalidBatch {
                    detail: format!("fixed slot {id} in group {gi} is already free"),
                });
            }
        }

        // Commit phase: validation passed, so every step below succeeds on a
        // consistent manager; every remainder still propagates typed (never
        // `expect`/panic).
        let s = self
            .seqs
            .remove(&seq.as_u64())
            .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })?;

        for (gi, ids) in s.tables.into_iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let pool = self
                .pools
                .get_mut(gi)
                .and_then(|p| p.as_mut())
                .ok_or_else(|| StateError::InvalidBatch {
                    detail: format!("group {gi} is not a paged group"),
                })?;
            for id in ids {
                pool.release(id)?;
            }
        }
        for (gi, slot) in s.fixed_slots.into_iter().enumerate() {
            let Some(id) = slot else { continue };
            let pool = self
                .fixed
                .get_mut(gi)
                .and_then(|p| p.as_mut())
                .ok_or_else(|| StateError::InvalidBatch {
                    detail: format!("group {gi} is not a fixed group"),
                })?;
            pool.free.insert(id);
        }
        self.live_count = next_live;
        Ok(())
    }

    /// Builds `BatchMeta` for one step (Spec 1 §2.5, Spec 3 §5).
    ///
    /// Order follows the input slices. Each `query_len` must be covered by
    /// that sequence's outstanding reservation. Problems across sequences are
    /// collected into one typed error.
    // DECISION(A1.15): StateManager produces canonical r9v_ir::BatchMeta directly, preserving row-major flattened [G,T], [G,S,max_blocks], [G,S] tensors, exact block-table sentinel holes, and optional TreeMask; rejected maintaining a duplicate parallel BatchMeta struct in r9v-state. Spec 1 §2.5, Spec 3 §3.3, §5, card A1.15.
    // DECISION(A1.16): this owned builder is cold convenience only — it
    // allocates every tensor per call. The scheduler hot path uses
    // `fill_batch_meta` into a pre-sized `BatchWorkspace` and never calls
    // this. Spec 1 §2.5, Spec 3 §5.
    pub fn batch_meta(&self, seqs: &[SeqId], query_lens: &[u32]) -> StateResult<BatchMeta> {
        self.batch_meta_with_options(seqs, query_lens, None, None)
    }

    /// Builds canonical [`r9v_ir::BatchMeta`] with optional speculative [`TreeMask`] (Spec 1 §4.D.1, Spec 3 §5).
    ///
    /// Cold convenience like [`Self::batch_meta`]; the hot path uses
    /// [`Self::fill_batch_meta`].
    pub fn batch_meta_with_tree(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        tree: Option<TreeMask>,
    ) -> StateResult<BatchMeta> {
        self.batch_meta_with_options(seqs, query_lens, None, tree)
    }

    /// Builds canonical [`r9v_ir::BatchMeta`] with explicit positions (scalar or MRoPE) and optional [`TreeMask`].
    ///
    /// Cold convenience like [`Self::batch_meta`]; the hot path uses
    /// [`Self::fill_batch_meta`].
    // DECISION(A1.15): device seq_ids are checked global u32 identifiers validated with u32::try_from; sequence allocation checks that next_seq <= u32::MAX before mutating state; rejected lossy as-casts, rollover, batch-local indexing that breaks Philox batch invariance, and unbounded host-to-device ID maps. Spec 1 §2.5, §4.F, Spec 3 §5, SI-40.
    pub fn batch_meta_with_options(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        positions: Option<Positions>,
        tree: Option<TreeMask>,
    ) -> StateResult<BatchMeta> {
        // Cold: fresh scratch and fresh tensors may allocate per call. Values
        // are bit-identical to the hot fill path (same plan, same emit).
        let mut seen_ids = Vec::new();
        let plan = self.plan_batch(
            seqs,
            query_lens,
            positions.as_ref().map(|p| p.len()),
            tree.as_ref().map(|t| t.t()),
            &mut seen_ids,
        )?;
        let total_tokens = plan.total_tokens;

        let mut seq_ids = Vec::with_capacity(seqs.len());
        let mut ctx_len = Vec::with_capacity(seqs.len());
        let mut default_positions: Vec<u32> = Vec::with_capacity(total_tokens);
        let total_slots =
            self.groups
                .len()
                .checked_mul(total_tokens)
                .ok_or_else(|| StateError::Overflow {
                    what: "flat slot map size".to_owned(),
                })?;
        let mut flat_slot_map = Vec::with_capacity(total_slots);
        let max_b = usize::try_from(self.max_blocks).map_err(|_| StateError::Overflow {
            what: "max blocks conversion".to_owned(),
        })?;
        let total_blocks = self
            .groups
            .len()
            .checked_mul(seqs.len())
            .and_then(|v| v.checked_mul(max_b))
            .ok_or_else(|| StateError::Overflow {
                what: "flat block table size".to_owned(),
            })?;
        let mut flat_block_table = vec![BLOCK_TABLE_SENTINEL; total_blocks];
        let total_windows =
            self.groups
                .len()
                .checked_mul(seqs.len())
                .ok_or_else(|| StateError::Overflow {
                    what: "flat window start size".to_owned(),
                })?;
        let mut flat_window_start = Vec::with_capacity(total_windows);
        self.emit_batch_into(
            seqs,
            query_lens,
            &plan,
            &mut seq_ids,
            &mut ctx_len,
            Some(&mut default_positions),
            &mut flat_slot_map,
            &mut flat_block_table,
            &mut flat_window_start,
        )?;
        let final_positions = positions.unwrap_or(Positions::PerToken(default_positions));

        let num_groups = u32::try_from(self.groups.len()).map_err(|_| StateError::Overflow {
            what: "batch meta group count".to_owned(),
        })?;
        let num_seqs = u32::try_from(seqs.len()).map_err(|_| StateError::Overflow {
            what: "batch meta seq count".to_owned(),
        })?;
        let total_tokens_u32 = u32::try_from(plan.total_u64).map_err(|_| StateError::Overflow {
            what: "batch meta total tokens".to_owned(),
        })?;

        BatchMeta::builder(num_groups, num_seqs, total_tokens_u32, self.max_blocks)
            .seq_ids(seq_ids)
            .query_len(query_lens.to_vec())
            .ctx_len(ctx_len)
            .positions(final_positions)
            .slot_map(flat_slot_map)
            .block_table(flat_block_table)
            .window_start(flat_window_start)
            .tree(tree)
            .build()
            .map_err(StateError::Ir)
    }

    /// Fills caller-owned batch buffers for one step (Spec 1 §2.5, Spec 3 §5).
    ///
    /// The scheduler hot path: validates and emits the same tensors the
    /// owned [`Self::batch_meta`] builds — device ids, query/context
    /// lengths, default positions, `[G, T]` slot map, `[G, S, max_blocks]`
    /// block table, `[G, S]` window starts — into `workspace`, moving `tree`
    /// in when the step verifies drafts. Order follows the input slices;
    /// each `query_len` must be covered by that sequence's outstanding
    /// reservation; problems across sequences are collected into one typed
    /// error, exactly like the owned builder.
    ///
    /// Hot path: success allocates nothing. Every required length is
    /// computed with checked arithmetic up front, and an undersized buffer
    /// is a typed error naming the capacity and the requirement — buffers
    /// never grow here. Size the workspace once with
    /// [`BatchWorkspace::with_capacity`] (cold) and reuse it.
    ///
    /// Default scalar positions (`ctx + k`); for exact explicit positions
    /// including MRoPE `[T,3]` use
    /// [`Self::fill_batch_meta_with_options`], which is bit-identical to
    /// [`Self::batch_meta_with_options`].
    pub fn fill_batch_meta(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        tree: Option<TreeMask>,
        workspace: &mut BatchWorkspace,
    ) -> StateResult<()> {
        self.fill_batch_meta_with_options(seqs, query_lens, None, tree, workspace)
    }

    /// Fills caller-owned batch buffers with exact explicit positions
    /// (Spec 1 §2.5, Spec 3 §5).
    ///
    /// Hot equivalent of [`Self::batch_meta_with_options`]: `positions`
    /// carries exact scalar `[T]` or MRoPE `[T,3]` values (`None` builds the
    /// default `ctx + k` scalar positions). Values are copied into the
    /// workspace's caller-owned buffers with no heap allocation on success;
    /// an undersized positions/MRoPE buffer is a typed error, never a grow.
    /// `position_width()` reports 1 (scalar) or 3 (MRoPE) afterwards, so
    /// `positions()` vs `positions_mrope()` is unambiguous. Bit-identical
    /// to the owned builder for scalar, MRoPE, and tree.
    pub fn fill_batch_meta_with_options(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        positions: Option<&Positions>,
        tree: Option<TreeMask>,
        workspace: &mut BatchWorkspace,
    ) -> StateResult<()> {
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        let max_b =
            usize::try_from(self.max_blocks).map_err(|_| overflow("max blocks conversion"))?;
        // Required lengths, all checked: fail before touching the workspace.
        let s_len = seqs.len();
        let g_len = self.groups.len();
        // Validation needs the token total before the buffers can be sized;
        // run the plan against workspace scratch first (no allocation when
        // the scratch fits, which the capacity check below guarantees — the
        // check itself only reads capacities).
        if workspace.id_scratch.capacity() < s_len {
            return Err(StateError::InvalidBatch {
                detail: format!(
                    "batch workspace id scratch capacity {} < required {s_len}",
                    workspace.id_scratch.capacity()
                ),
            });
        }
        let positions_len = positions.as_ref().map(|p| p.len());
        let plan = self.plan_batch(
            seqs,
            query_lens,
            positions_len,
            tree.as_ref().map(|t| t.t()),
            &mut workspace.id_scratch,
        )?;
        let t_len = plan.total_tokens;
        let need_slot = g_len
            .checked_mul(t_len)
            .ok_or_else(|| overflow("flat slot map size"))?;
        let need_block = g_len
            .checked_mul(s_len)
            .and_then(|v| v.checked_mul(max_b))
            .ok_or_else(|| overflow("flat block table size"))?;
        let need_window = g_len
            .checked_mul(s_len)
            .ok_or_else(|| overflow("flat window start size"))?;
        let check = |what: &str, cap: usize, need: usize| {
            if cap < need {
                Err(StateError::InvalidBatch {
                    detail: format!("batch workspace {what} capacity {cap} < required {need}"),
                })
            } else {
                Ok(())
            }
        };
        check("seq_ids", workspace.seq_ids.capacity(), s_len)?;
        check("query_lens", workspace.query_lens.capacity(), s_len)?;
        check("ctx_lens", workspace.ctx_lens.capacity(), s_len)?;
        // Positions buffers are exclusive by width: scalar fills need
        // `positions`, MRoPE fills need `positions_mrope`. Check only the
        // active one so a scalar-sized workspace is not forced to also fit
        // triplets and vice versa.
        let want_mrope = matches!(positions, Some(Positions::Mrope(_)));
        if want_mrope {
            check(
                "positions_mrope",
                workspace.positions_mrope.capacity(),
                t_len,
            )?;
        } else {
            check("positions", workspace.positions.capacity(), t_len)?;
        }
        check("slot_map", workspace.slot_map.capacity(), need_slot)?;
        check("block_table", workspace.block_table.capacity(), need_block)?;
        check(
            "window_start",
            workspace.window_start.capacity(),
            need_window,
        )?;
        // Batch-relative tree rules the owned builder enforces (Spec 1
        // §4.D.1): `t_max` covers the longest query and no parent crosses
        // its sequence. The fill path used to skip these (the `TreeMask`
        // was built valid, but never against this batch); closing that hole
        // keeps validation complete on every entry point. Runs before any
        // workspace mutation, like every other check above.
        // DECISION(A1.16): the owned fill enforces the builder's
        // batch-relative tree rules instead of trusting a prebuilt mask;
        // rejected: trusting `TreeMask::new` alone (it cannot see
        // `query_lens`). Spec 1 §4.D.1.
        if let Some(t) = tree.as_ref() {
            check_tree_batch_rules(t.parents(), t.t_max(), query_lens).map_err(StateError::Ir)?;
        }
        // Capacities fit, so nothing below allocates: clears keep capacity,
        // pushes/extends land within it, and `resize` only writes sentinels.
        workspace.seq_ids.clear();
        workspace.query_lens.clear();
        workspace.ctx_lens.clear();
        workspace.positions.clear();
        workspace.positions_mrope.clear();
        workspace.slot_map.clear();
        workspace.block_table.clear();
        workspace.window_start.clear();
        workspace.query_lens.extend(query_lens.iter().copied());
        workspace
            .block_table
            .resize(need_block, BLOCK_TABLE_SENTINEL);
        // Explicit positions are copied verbatim (no alloc within capacity);
        // `None` emits the default `ctx + k` scalar positions. Either way
        // the width is recorded so the two accessors stay unambiguous.
        match positions {
            None => {
                self.emit_batch_into(
                    seqs,
                    query_lens,
                    &plan,
                    &mut workspace.seq_ids,
                    &mut workspace.ctx_lens,
                    Some(&mut workspace.positions),
                    &mut workspace.slot_map,
                    &mut workspace.block_table,
                    &mut workspace.window_start,
                )?;
                workspace.last_position_width = 1;
            }
            Some(Positions::PerToken(v)) => {
                self.emit_batch_into(
                    seqs,
                    query_lens,
                    &plan,
                    &mut workspace.seq_ids,
                    &mut workspace.ctx_lens,
                    None,
                    &mut workspace.slot_map,
                    &mut workspace.block_table,
                    &mut workspace.window_start,
                )?;
                // `plan_batch` already proved `v.len() == T`; the capacity
                // check above proves the extend below cannot reallocate.
                workspace.positions.extend(v.iter().copied());
                workspace.last_position_width = 1;
            }
            Some(Positions::Mrope(v)) => {
                self.emit_batch_into(
                    seqs,
                    query_lens,
                    &plan,
                    &mut workspace.seq_ids,
                    &mut workspace.ctx_lens,
                    None,
                    &mut workspace.slot_map,
                    &mut workspace.block_table,
                    &mut workspace.window_start,
                )?;
                workspace.positions_mrope.extend(v.iter().copied());
                workspace.last_position_width = 3;
            }
        }
        workspace.last_g = u32::try_from(g_len).map_err(|_| overflow("batch meta group count"))?;
        workspace.last_s = u32::try_from(s_len).map_err(|_| overflow("batch meta seq count"))?;
        workspace.last_t =
            u32::try_from(plan.total_u64).map_err(|_| overflow("batch meta total tokens"))?;
        workspace.last_max_blocks = self.max_blocks;
        // Single live source: an owned fill clears the hot slices staging.
        workspace.clear_tree();
        workspace.tree = tree;
        Ok(())
    }

    /// Fills caller-owned batch buffers for a verify step from borrowed tree
    /// slices (Spec 1 §2.5, §4.D.1, Spec 3 §5).
    ///
    /// The scheduler hot path: `tree` stays in the scheduler's buffers until
    /// this call copies it into the workspace's preallocated storage and
    /// runs the complete validation — shape, parent bounds/self-parent,
    /// cycles, plus the batch rules (`T` match, `t_max` covering the longest
    /// query, no parent crossing its sequence) — with the same typed
    /// [`IrError`](r9v_ir::IrError) variants the owned builder reports.
    /// Default scalar positions (`ctx + k`); for exact explicit positions
    /// use [`Self::fill_batch_meta_with_options_and_tree_input`].
    ///
    /// Hot path: success allocates nothing and fabricates no owned
    /// `TreeMask` — read the stored tree with
    /// [`BatchWorkspace::tree_view`]. Undersized buffers are a typed error
    /// naming the capacity and the requirement, never a grow. Size the
    /// workspace once cold with
    /// [`BatchWorkspace::try_with_capacity_and_tree`] and reuse it.
    pub fn fill_batch_meta_with_tree_input(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        tree: TreeInput<'_>,
        workspace: &mut BatchWorkspace,
    ) -> StateResult<()> {
        self.fill_batch_meta_with_options_and_tree_input(
            seqs,
            query_lens,
            None,
            Some(tree),
            workspace,
        )
    }

    /// Fills caller-owned batch buffers with exact explicit positions and an
    /// optional borrowed tree (Spec 1 §2.5, §4.D.1, Spec 3 §5).
    ///
    /// Hot equivalent of [`Self::batch_meta_with_options`] without the owned
    /// `TreeMask`: `positions` carries exact scalar `[T]` or MRoPE `[T,3]`
    /// values (`None` builds the default `ctx + k` scalar positions) and
    /// `tree` carries the verify tree (`None` stores no tree). Bit-identical
    /// to the owned builder for scalar, MRoPE, and tree. Same allocation
    /// contract as [`Self::fill_batch_meta_with_tree_input`]: success
    /// allocates nothing, undersized buffers fail closed typed.
    pub fn fill_batch_meta_with_options_and_tree_input(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        positions: Option<&Positions>,
        tree: Option<TreeInput<'_>>,
        workspace: &mut BatchWorkspace,
    ) -> StateResult<()> {
        let overflow = |what: &str| StateError::Overflow {
            what: what.to_owned(),
        };
        let max_b =
            usize::try_from(self.max_blocks).map_err(|_| overflow("max blocks conversion"))?;
        // Required lengths, all checked: fail before touching the workspace.
        let s_len = seqs.len();
        let g_len = self.groups.len();
        // Validation needs the token total before the buffers can be sized;
        // run the plan against workspace scratch first (no allocation when
        // the scratch fits, which the capacity check below guarantees — the
        // check itself only reads capacities).
        if workspace.id_scratch.capacity() < s_len {
            return Err(StateError::InvalidBatch {
                detail: format!(
                    "batch workspace id scratch capacity {} < required {s_len}",
                    workspace.id_scratch.capacity()
                ),
            });
        }
        let positions_len = positions.as_ref().map(|p| p.len());
        // The tree/batch size match is reported as the builder's
        // `TreeBatchMismatch` below, so the plan skips its own count check.
        let plan = self.plan_batch(
            seqs,
            query_lens,
            positions_len,
            None,
            &mut workspace.id_scratch,
        )?;
        let t_len = plan.total_tokens;
        if let Some(t) = tree.as_ref() {
            if t.t() != t_len {
                return Err(StateError::Ir(IrError::TreeBatchMismatch {
                    tree_t: t.t(),
                    batch_t: u32::try_from(plan.total_u64)
                        .map_err(|_| overflow("batch meta total tokens"))?,
                }));
            }
        }
        let need_slot = g_len
            .checked_mul(t_len)
            .ok_or_else(|| overflow("flat slot map size"))?;
        let need_block = g_len
            .checked_mul(s_len)
            .and_then(|v| v.checked_mul(max_b))
            .ok_or_else(|| overflow("flat block table size"))?;
        let need_window = g_len
            .checked_mul(s_len)
            .ok_or_else(|| overflow("flat window start size"))?;
        let check = |what: &str, cap: usize, need: usize| {
            if cap < need {
                Err(StateError::InvalidBatch {
                    detail: format!("batch workspace {what} capacity {cap} < required {need}"),
                })
            } else {
                Ok(())
            }
        };
        check("seq_ids", workspace.seq_ids.capacity(), s_len)?;
        check("query_lens", workspace.query_lens.capacity(), s_len)?;
        check("ctx_lens", workspace.ctx_lens.capacity(), s_len)?;
        // Positions buffers are exclusive by width: scalar fills need
        // `positions`, MRoPE fills need `positions_mrope`. Check only the
        // active one so a scalar-sized workspace is not forced to also fit
        // triplets and vice versa.
        let want_mrope = matches!(positions, Some(Positions::Mrope(_)));
        if want_mrope {
            check(
                "positions_mrope",
                workspace.positions_mrope.capacity(),
                t_len,
            )?;
        } else {
            check("positions", workspace.positions.capacity(), t_len)?;
        }
        check("slot_map", workspace.slot_map.capacity(), need_slot)?;
        check("block_table", workspace.block_table.capacity(), need_block)?;
        check(
            "window_start",
            workspace.window_start.capacity(),
            need_window,
        )?;
        // Tree before any buffer is touched. The batch-relative rules
        // (`t_max` cover, no cross-sequence parent) read only the input
        // slices, so they run first and mutate nothing; `fill_tree` then
        // copies into the preallocated buffers with the intrinsic validation
        // (shape, bounds/self-parent, cycles). Every failure below leaves
        // the batch buffers exactly as they were.
        if let Some(t) = tree.as_ref() {
            check_tree_batch_rules(t.parents(), t.t_max(), query_lens).map_err(StateError::Ir)?;
            workspace.fill_tree(*t)?;
        } else {
            workspace.clear_tree();
        }
        // Capacities fit, so nothing below allocates: clears keep capacity,
        // pushes/extends land within it, and `resize` only writes sentinels.
        workspace.seq_ids.clear();
        workspace.query_lens.clear();
        workspace.ctx_lens.clear();
        workspace.positions.clear();
        workspace.positions_mrope.clear();
        workspace.slot_map.clear();
        workspace.block_table.clear();
        workspace.window_start.clear();
        workspace.query_lens.extend(query_lens.iter().copied());
        workspace
            .block_table
            .resize(need_block, BLOCK_TABLE_SENTINEL);
        // Explicit positions are copied verbatim (no alloc within capacity);
        // `None` emits the default `ctx + k` scalar positions. Either way
        // the width is recorded so the two accessors stay unambiguous.
        match positions {
            None => {
                self.emit_batch_into(
                    seqs,
                    query_lens,
                    &plan,
                    &mut workspace.seq_ids,
                    &mut workspace.ctx_lens,
                    Some(&mut workspace.positions),
                    &mut workspace.slot_map,
                    &mut workspace.block_table,
                    &mut workspace.window_start,
                )?;
                workspace.last_position_width = 1;
            }
            Some(Positions::PerToken(v)) => {
                self.emit_batch_into(
                    seqs,
                    query_lens,
                    &plan,
                    &mut workspace.seq_ids,
                    &mut workspace.ctx_lens,
                    None,
                    &mut workspace.slot_map,
                    &mut workspace.block_table,
                    &mut workspace.window_start,
                )?;
                // `plan_batch` already proved `v.len() == T`; the capacity
                // check above proves the extend below cannot reallocate.
                workspace.positions.extend(v.iter().copied());
                workspace.last_position_width = 1;
            }
            Some(Positions::Mrope(v)) => {
                self.emit_batch_into(
                    seqs,
                    query_lens,
                    &plan,
                    &mut workspace.seq_ids,
                    &mut workspace.ctx_lens,
                    None,
                    &mut workspace.slot_map,
                    &mut workspace.block_table,
                    &mut workspace.window_start,
                )?;
                workspace.positions_mrope.extend(v.iter().copied());
                workspace.last_position_width = 3;
            }
        }
        workspace.last_g = u32::try_from(g_len).map_err(|_| overflow("batch meta group count"))?;
        workspace.last_s = u32::try_from(s_len).map_err(|_| overflow("batch meta seq count"))?;
        workspace.last_t =
            u32::try_from(plan.total_u64).map_err(|_| overflow("batch meta total tokens"))?;
        workspace.last_max_blocks = self.max_blocks;
        // The slices path never fabricates an owned mask: `fill_tree` (or
        // `clear_tree`) already settled the single live source.
        workspace.tree = None;
        Ok(())
    }

    /// Validates a batch into a [`BatchPlan`]: no allocation on success
    /// (`seen_ids` is caller scratch; details formatted only on failure).
    fn plan_batch(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        positions_len: Option<usize>,
        tree_t: Option<usize>,
        seen_ids: &mut Vec<u64>,
    ) -> StateResult<BatchPlan> {
        if seqs.len() != query_lens.len() {
            return Err(StateError::InvalidBatch {
                detail: format!("seqs={} query_lens={}", seqs.len(), query_lens.len()),
            });
        }
        if seqs.is_empty() {
            return Err(StateError::InvalidBatch {
                detail: "empty batch".to_owned(),
            });
        }
        let mut problems: Vec<String> = Vec::new();
        seen_ids.clear();
        for (i, seq) in seqs.iter().enumerate() {
            if seq.as_u64() > u64::from(u32::MAX) {
                return Err(StateError::SeqIdOverflow {
                    seq: seq.as_u64(),
                    max: u32::MAX,
                });
            }
            if seen_ids.contains(&seq.as_u64()) {
                problems.push(format!("batch[{i}]: duplicate sequence {}", seq.as_u64()));
            } else {
                seen_ids.push(seq.as_u64());
            }
        }
        let mut total: u64 = 0;
        for (i, (seq, q)) in seqs.iter().zip(query_lens.iter()).enumerate() {
            match self.seq(*seq) {
                Err(_) => problems.push(format!("batch[{i}]: unknown sequence {}", seq.as_u64())),
                Ok(s) => {
                    if *q == 0 || *q > s.tail_len {
                        problems.push(format!(
                            "batch[{i}]: query_len={q} not in 1..={} (tail)",
                            s.tail_len
                        ));
                    }
                    match s.ctx_len.checked_add(*q) {
                        Some(end) if end <= self.config.max_ctx => {}
                        _ => problems.push(format!(
                            "batch[{i}]: ctx_len={} + query_len={q} exceeds max_ctx={}",
                            s.ctx_len, self.config.max_ctx
                        )),
                    }
                    match total.checked_add(u64::from(*q)) {
                        Some(next) => total = next,
                        None => problems.push(format!("batch[{i}]: token total overflows")),
                    }
                }
            }
        }
        if total > MAX_BATCH_TOKENS_HARD {
            problems.push(format!(
                "batch tokens={total} exceed cap {MAX_BATCH_TOKENS_HARD}"
            ));
        }
        let total_tokens = usize::try_from(total).map_err(|_| StateError::Overflow {
            what: "total tokens conversion".to_owned(),
        })?;
        if let Some(len) = positions_len {
            if len != total_tokens {
                problems.push(format!(
                    "positions len {len} != total tokens {total_tokens}"
                ));
            }
        }
        if let Some(t) = tree_t {
            if t != total_tokens {
                problems.push(format!(
                    "tree token count {t} != total tokens {total_tokens}"
                ));
            }
        }
        if !problems.is_empty() {
            return Err(StateError::InvalidBatch {
                detail: problems.join("; "),
            });
        }
        Ok(BatchPlan {
            total_tokens,
            total_u64: total,
        })
    }

    /// Emits validated batch tensors into caller-provided buffers, shared by
    /// the owned and fill metadata paths (Spec 1 §2.5, Spec 3 §3.3, §3.5).
    ///
    /// Buffers arrive empty with sufficient capacity, except `block_table`,
    /// which arrives pre-filled with [`BLOCK_TABLE_SENTINEL`] to its full
    /// `[G, S, max_blocks]` length for scattered writes. `positions` is
    /// `Some` when the caller wants the default `ctx + k` scalar positions
    /// emitted and `None` when it copies exact explicit positions itself
    /// afterwards. Success pushes within capacity only and never allocates.
    //
    // Nine arguments is over the default lint count; they are the tensors of
    // one batch (Spec 1 §2.5), so bundling them would only rename the
    // argument.
    #[allow(clippy::too_many_arguments)]
    fn emit_batch_into(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        plan: &BatchPlan,
        seq_ids: &mut Vec<u32>,
        ctx_lens: &mut Vec<u32>,
        mut positions: Option<&mut Vec<u32>>,
        slot_map: &mut Vec<u32>,
        block_table: &mut [u32],
        window_start: &mut Vec<u32>,
    ) -> StateResult<()> {
        let total_tokens = plan.total_tokens;
        for seq in seqs {
            let dev_id = u32::try_from(seq.as_u64()).map_err(|_| StateError::SeqIdOverflow {
                seq: seq.as_u64(),
                max: u32::MAX,
            })?;
            seq_ids.push(dev_id);
        }

        for (seq, q) in seqs.iter().zip(query_lens.iter()) {
            let s = self.seq(*seq).map_err(|_| StateError::InvalidBatch {
                detail: "sequence vanished during batch build".to_owned(),
            })?;
            ctx_lens.push(s.ctx_len);
            if let Some(positions) = positions.as_deref_mut() {
                for k in 0..*q {
                    positions.push(s.ctx_len.checked_add(k).ok_or_else(|| {
                        StateError::Overflow {
                            what: "batch position".to_owned(),
                        }
                    })?);
                }
            }
        }

        let max_b = usize::try_from(self.max_blocks).map_err(|_| StateError::Overflow {
            what: "max blocks conversion".to_owned(),
        })?;

        for (gi, g) in self.groups.iter().enumerate() {
            // slot_map [G, T] row-major
            if !g.spec.is_paged() {
                slot_map.extend(std::iter::repeat_n(SLOT_NONE, total_tokens));
            } else {
                for (si, (seq, q)) in seqs.iter().zip(query_lens.iter()).enumerate() {
                    let s = self.seq(*seq).map_err(|_| StateError::InvalidBatch {
                        detail: "sequence vanished during batch build".to_owned(),
                    })?;
                    let end = ctx_lens[si]
                        .checked_add(*q)
                        .ok_or_else(|| StateError::Overflow {
                            what: "batch reserved end".to_owned(),
                        })?;
                    for k in 0..*q {
                        let pos =
                            ctx_lens[si]
                                .checked_add(k)
                                .ok_or_else(|| StateError::Overflow {
                                    what: "batch position".to_owned(),
                                })?;
                        slot_map.push(flatten_slot(s, gi, pos, end)?);
                    }
                }
            }

            // block_table [G, S, max_blocks] row-major & window_start [G, S] row-major
            //
            // DECISION(A1.11) per SI-17: the §3.5 ring sentence describes the
            // eviction policy, not the storage shape. Every row stays width
            // `max_blocks` with each block id at its absolute logical block
            // index and sentinel holes where window eviction released blocks;
            // rejected: compacting live ids to the front (destroys the
            // position-to-index mapping the slot formula relies on).
            for (si, seq) in seqs.iter().enumerate() {
                let s = self.seq(*seq).map_err(|_| StateError::InvalidBatch {
                    detail: "sequence vanished during batch build".to_owned(),
                })?;
                if g.spec.is_paged() {
                    let indices = s.indices.get(gi).ok_or_else(|| StateError::InvalidBatch {
                        detail: format!("sequence {} has no table", seq.as_u64()),
                    })?;
                    let tables = s.tables.get(gi).ok_or_else(|| StateError::InvalidBatch {
                        detail: format!("sequence {} has no table", seq.as_u64()),
                    })?;
                    for (at, idx) in indices.iter().enumerate() {
                        let id =
                            tables
                                .get(at)
                                .copied()
                                .ok_or_else(|| StateError::InvalidBatch {
                                    detail: format!("sequence {} has no table", seq.as_u64()),
                                })?;
                        let cell: usize =
                            usize::try_from(*idx).map_err(|_| StateError::Overflow {
                                what: "block table index".to_owned(),
                            })?;
                        if cell >= max_b {
                            return Err(StateError::Overflow {
                                what: "block table index".to_owned(),
                            });
                        }
                        let offset = gi
                            .checked_mul(seqs.len())
                            .and_then(|v| v.checked_add(si))
                            .and_then(|v| v.checked_mul(max_b))
                            .and_then(|v| v.checked_add(cell))
                            .ok_or_else(|| StateError::Overflow {
                                what: "block table offset".to_owned(),
                            })?;
                        let cell_ref =
                            block_table
                                .get_mut(offset)
                                .ok_or_else(|| StateError::Overflow {
                                    what: "block table index".to_owned(),
                                })?;
                        *cell_ref = id;
                    }
                }
                let ws = match g.spec.retain() {
                    None | Some(crate::spec::Retain::All) => 0,
                    Some(crate::spec::Retain::Window { w })
                    | Some(crate::spec::Retain::SinkWindow { w, .. }) => {
                        window_start_of(s.ctx_len, w)
                    }
                };
                window_start.push(ws);
            }
        }
        Ok(())
    }

    /// Pool budget snapshot (Spec 3 §5 `budget`).
    pub fn budget(&self) -> Budget {
        // DECISION(A1.11): the sub-block remainder of the proportional split
        // is reported as `unusable_bytes` and `host_free` is an explicit zero
        // while host swap is deferred to B1; rejected: silently absorbing the
        // remainder into a total or omitting the host line (both hide bytes
        // the scheduler cannot spend).
        //
        // All products below are bounded by construction (per-group assigned
        // bytes fit in u64 and the free sums cannot exceed them), so the
        // widening `as u128` casts are lossless and the single narrowing
        // conversions name the invariant they rely on.
        let mut free_paged: u128 = 0;
        let mut free_fixed: u128 = 0;
        let mut groups = Vec::with_capacity(self.groups.len());
        for gi in 0..self.groups.len() {
            let mut entry = GroupBudget {
                index: gi,
                total_blocks: 0,
                free_blocks: 0,
                block_bytes: 0,
                base_offset: 0,
                total_slots: 0,
                free_slots: 0,
                slot_bytes_per_seq: 0,
            };
            if let Some(p) = &self.pools[gi] {
                free_paged += p.free.len() as u128 * p.block_bytes as u128;
                entry.total_blocks = p.total;
                entry.free_blocks = p
                    .free
                    .len()
                    .try_into()
                    .expect("free blocks fit u32: bounded by the pool total");
                entry.block_bytes = p.block_bytes;
                entry.base_offset = p.base_offset;
            }
            if let Some(f) = &self.fixed[gi] {
                free_fixed += f.free.len() as u128 * f.slot_bytes_per_seq as u128;
                entry.total_slots = f.total_slots;
                entry.free_slots = f
                    .free
                    .len()
                    .try_into()
                    .expect("free slots fit u32: bounded by max_seqs");
                entry.slot_bytes_per_seq = f.slot_bytes_per_seq;
                entry.base_offset = f.base_offset;
            }
            groups.push(entry);
        }
        Budget {
            groups,
            pool_bytes_total: self.pool_bytes_total,
            pool_bytes_free: u64::try_from(free_paged)
                .expect("free paged bytes fit: bounded by assigned pool bytes"),
            fixed_bytes_total: self.fixed_bytes_total,
            fixed_bytes_free: u64::try_from(free_fixed)
                .expect("free fixed bytes fit: bounded by assigned fixed bytes"),
            host_free: 0,
            unusable_bytes: self.unusable_bytes,
        }
    }

    /// Manager statistics (Spec 3 §5 `stats`).
    pub fn stats(&self) -> Stats {
        let mut alloc: u128 = 0;
        let mut total: u128 = 0;
        for pool in self.pools.iter().flatten() {
            let pool_total = pool.total as u128;
            let pool_free = pool.free.len() as u128;
            alloc += pool_total
                .checked_sub(pool_free)
                .expect("free blocks cannot exceed the pool total");
            total += pool_total;
        }
        Stats {
            prefix_hit_rate: 0.0,
            evictions: 0,
            swaps: self.swaps,
            commits: self.commits,
            utilization: if total == 0 {
                0.0
            } else {
                alloc as f32 / total as f32
            },
        }
    }
}

/// Batch-relative tree rules shared by the owned and slices fill paths,
/// mirroring [`BatchMetaBuilder`](r9v_ir::BatchMetaBuilder) exactly
/// (Spec 1 §4.D.1).
///
/// `t_max` covers the longest query and no parent points outside its own
/// sequence's flattened token range (out-of-range ids are the intrinsic
/// check's job and are skipped here, as in the builder). Pure: collects
/// every failure, allocates only while reporting.
fn check_tree_batch_rules(parents: &[i32], t_max: u32, query_lens: &[u32]) -> Result<(), IrError> {
    let mut problems = Vec::new();
    let required = query_lens.iter().copied().max().unwrap_or(0);
    if t_max < required {
        problems.push(IrError::TreeMaxTooSmall {
            required,
            actual: t_max,
        });
    }
    let total: usize = query_lens.iter().map(|&q| q as usize).sum();
    if parents.len() == total {
        let mut seq_start = 0usize;
        for (seq, &seq_len) in query_lens.iter().enumerate() {
            let seq_end = seq_start + seq_len as usize;
            for (token, &parent) in parents
                .iter()
                .enumerate()
                .skip(seq_start)
                .take(seq_end - seq_start)
            {
                if parent >= 0 && ((parent as usize) < seq_start || (parent as usize) >= seq_end) {
                    problems.push(IrError::TreeParentCrossesSequence {
                        token,
                        parent,
                        seq,
                        seq_start,
                        seq_end,
                    });
                }
            }
            seq_start = seq_end;
        }
    }
    IrError::from_problems(problems)
}

/// Grows `v` toward `need` once, cold (Spec 3 §5).
///
/// Already-fitting buffers are untouched; a refused reservation is a typed
/// error, never a partial size the hot path could trip over. The request is
/// measured from `len`, not `capacity`: `try_reserve_exact` counts additional
/// space beyond `len`, so `need - capacity` silently under-serves reused
/// (nonempty) vectors with spare capacity while reporting success.
fn reserve_at_least<T>(v: &mut Vec<T>, need: usize, what: &'static str) -> StateResult<()> {
    if v.capacity() < need {
        // `capacity < need` with the `Vec` invariant `len <= capacity` gives
        // `len < need`, so the checked subtraction below cannot fail on a
        // live vector; it stays checked so a violated invariant is a typed
        // refusal rather than an underflow wrap.
        let additional = need
            .checked_sub(v.len())
            .ok_or_else(|| StateError::Overflow {
                what: what.to_owned(),
            })?;
        v.try_reserve_exact(additional)
            .map_err(|_| StateError::Overflow {
                what: what.to_owned(),
            })?;
    }
    Ok(())
}

/// Mathematical `max(0, ctx - w)` via checked branching (Spec 3 §3.5).
///
/// A short unverified sequence is normal, not malformed state, so this is an
/// explicit branch — never a saturating subtract that could hide a genuine
/// underflow elsewhere.
//
// The `implicit_saturating_sub` lint suggests `saturating_sub` here; that
// spelling is deliberately refused because silent saturation is exactly what
// the hostile audit for this card bans on state paths.
#[allow(clippy::implicit_saturating_sub)]
fn window_start_of(ctx: u32, w: u32) -> u32 {
    if ctx >= w {
        ctx - w
    } else {
        0
    }
}

/// Allocates one recorded `(group, index)` block into its sequence table
/// (Spec 3 §3.6, §5). Takes the smallest free id, keeping block ids a
/// deterministic function of the request history alone.
fn place_one_block(
    pools: &mut [Option<BlockPool>],
    seqs: &mut BTreeMap<u64, SeqState>,
    seq: SeqId,
    gi: usize,
    idx: u32,
    end: u32,
    max_ctx: u32,
) -> StateResult<()> {
    let pool =
        pools
            .get_mut(gi)
            .and_then(|p| p.as_mut())
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {gi} is not a paged group"),
            })?;
    let id = pool.alloc().ok_or(StateError::PoolExhausted {
        group: gi,
        required: 1,
        available: 0,
        shortfall: 1,
        end,
        max_ctx,
    })?;
    let s = seqs
        .get_mut(&seq.as_u64())
        .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })?;
    let indices = s
        .indices
        .get_mut(gi)
        .ok_or_else(|| StateError::InvalidBatch {
            detail: format!("sequence {} has no table", seq.as_u64()),
        })?;
    let at = indices.partition_point(|&i| i < idx);
    indices.insert(at, idx);
    let tables = s
        .tables
        .get_mut(gi)
        .ok_or_else(|| StateError::InvalidBatch {
            detail: format!("sequence {} has no table", seq.as_u64()),
        })?;
    tables.insert(at, id);
    Ok(())
}

/// Flattened slot for one retained position: `block_id * 32 + lane`
/// (Spec 1 §2.5, Spec 3 §3.3).
///
// DECISION(A1.11) per SI-17: the flattened value carries the pool-global
// block id, not the within-table position, so the device derives
// `base[group] + block_id * block_bytes` directly; ids are per-group-pool,
// never arena-global across groups. A position with no mapped block is a
// typed error, never a clamp to a neighboring block.
/// Scope check for a `SlotRange`: it must name the sequence's current open
/// reservation (Spec 3 §3.6, §5).
///
/// `UnknownSeq` is reported by the caller before this runs; here a mismatch
/// of `start`/`len` against the live `ctx_len`/`tail_len` (post-commit,
/// freed, superseded by a later reserve, or foreign) is a typed
/// `InvalidBatch` naming both sides, never a read of another step's tail.
fn check_range_live(s: &SeqState, range: &SlotRange) -> StateResult<()> {
    if s.ctx_len != range.start() || s.tail_len != range.len() {
        return Err(StateError::InvalidBatch {
            detail: format!(
                "stale SlotRange: start={} len={} but live ctx_len={} tail_len={}",
                range.start(),
                range.len(),
                s.ctx_len,
                s.tail_len,
            ),
        });
    }
    Ok(())
}

fn flatten_slot(s: &SeqState, group: usize, pos: u32, end: u32) -> StateResult<u32> {
    let missing = || StateError::UnmappedPosition { group, pos, end };
    let indices = s.indices.get(group).ok_or_else(missing)?;
    let tables = s.tables.get(group).ok_or_else(missing)?;
    let block = pos / BLOCK_TOKENS;
    let lane = pos % BLOCK_TOKENS;
    let at = indices.partition_point(|&i| i < block);
    if indices.get(at) != Some(&block) {
        return Err(missing());
    }
    let id = tables.get(at).copied().ok_or_else(missing)?;
    u64::from(id)
        .checked_mul(u64::from(BLOCK_TOKENS))
        .and_then(|v| v.checked_add(u64::from(lane)))
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| StateError::Overflow {
            what: "slot id".to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{CacheDtype, Retain, StateSpec};

    fn test_kv_spec() -> StateSpec {
        StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        }
    }

    fn test_rec_spec() -> StateSpec {
        StateSpec::Recurrent { h: 2, d: 8, dv: 8 }
    }

    fn test_conv_spec() -> StateSpec {
        StateSpec::ConvWindow { c: 4, w: 4 }
    }

    fn test_hybrid_manager() -> StateManager {
        let cfg = StateConfig {
            max_ctx: 128,
            max_seqs: 4,
        };
        let specs = vec![test_kv_spec(), test_rec_spec(), test_conv_spec()];
        let groups = group_layers(&specs).unwrap();
        let req = required_pool_bytes(cfg, &groups).expect("valid required bytes");
        StateManager::new(cfg, specs, req * 2).expect("valid manager")
    }

    #[test]
    fn max_slot_blocks_boundary_and_slot_none_collision() {
        assert_eq!(MAX_SLOT_BLOCKS, 134_217_727);
        let max_admitted_block = MAX_SLOT_BLOCKS - 1;
        assert_eq!(max_admitted_block, 134_217_726);

        // Prove no admitted real slot equals SLOT_NONE
        for lane in 0..BLOCK_TOKENS {
            let slot = u64::from(max_admitted_block) * u64::from(BLOCK_TOKENS) + u64::from(lane);
            assert!(slot < u64::from(SLOT_NONE));
            assert_ne!(slot as u32, SLOT_NONE);
        }

        let max_admitted_slot = max_admitted_block * BLOCK_TOKENS + (BLOCK_TOKENS - 1);
        assert_eq!(max_admitted_slot, 4_294_967_263);
        assert_eq!(SLOT_NONE, u32::MAX);
        assert_eq!(u64::from(max_admitted_slot) + 32, u64::from(SLOT_NONE));

        // Exact proof: block 134_217_727 (admitted by 134_217_728) would collide at lane 31
        let collided_block = MAX_SLOT_BLOCKS;
        let collided_slot =
            u64::from(collided_block) * u64::from(BLOCK_TOKENS) + u64::from(BLOCK_TOKENS - 1);
        assert_eq!(collided_slot, u64::from(SLOT_NONE));

        // Verify with flatten_slot
        let seq_state = SeqState {
            ctx_len: 32,
            tail_len: 0,
            tables: vec![vec![max_admitted_block]],
            indices: vec![vec![0]],
            mirror: BTreeMap::new(),
            compacted: None,
            fixed_slots: vec![None],
            parity: vec![0],
        };
        let flattened = flatten_slot(&seq_state, 0, 31, 32).expect("flattened slot ok");
        assert_eq!(flattened, max_admitted_slot);
        assert_ne!(flattened, SLOT_NONE);
    }

    #[test]
    fn max_slot_blocks_plus_one_rejected_without_impractical_allocation() {
        let cfg = StateConfig {
            max_ctx: 32,
            max_seqs: 1,
        };
        let spec = test_kv_spec();
        let groups = group_layers(&[spec]).unwrap();
        let block_bytes = groups[0].block_bytes().expect("valid block bytes");
        let pool_bytes = (u64::from(MAX_SLOT_BLOCKS) + 1) * block_bytes;
        let err = StateManager::new(cfg, vec![spec], pool_bytes).unwrap_err();
        match err {
            StateError::InvalidConfig { problems } => {
                assert!(
                    problems
                        .iter()
                        .any(|p| p.reason.contains("exceeds u32 slot_map range 134217727")),
                    "unexpected error: {problems:?}"
                );
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn block_pool_release_validates_in_release_builds() {
        // Direct pool proof: valid release/alloc round-trips, while
        // out-of-range and double releases fail typed (never `debug_assert`
        // only, never panic) — the guarantee every reserve/commit/free path
        // relies on transactionally.
        let mut pool = BlockPool {
            total: 4,
            free: (0..4).rev().collect(),
            base_offset: 0,
            block_bytes: 64,
        };
        assert_eq!(pool.alloc(), Some(0));
        assert_eq!(pool.alloc(), Some(1));
        pool.release(1).expect("releasing a held block succeeds");
        // Double release fails typed, in every build profile.
        let err = pool.release(1).unwrap_err();
        assert!(
            matches!(err, StateError::InvalidBatch { .. }),
            "got {err:?}"
        );
        // Out-of-range ids fail typed, including the `u32::MAX` free_seq
        // corruption probe.
        for bad in [4, 5, u32::MAX - 10, u32::MAX] {
            let err = pool.release(bad).unwrap_err();
            assert!(
                matches!(err, StateError::InvalidBatch { .. }),
                "got {err:?} for {bad}"
            );
        }
        // The failed releases mutated nothing: smallest-free order is intact.
        assert_eq!(pool.alloc(), Some(1));
        assert_eq!(pool.alloc(), Some(2));
        assert_eq!(pool.alloc(), Some(3));
        assert_eq!(pool.alloc(), None);
    }

    #[test]
    fn free_seq_atomic_rejects_already_free_block_and_mutates_nothing() {
        let mut m = test_hybrid_manager();
        let (a, _) = m.new_seq(&[]).unwrap();
        m.reserve(a, 64).unwrap();
        m.write_tokens(a, 0, &vec![1; 64]).unwrap();
        m.commit(a, 64).unwrap();

        let s = m.seq(a).unwrap();
        let block_to_corrupt = s.tables[0][0];

        // Corrupt internal state: insert block back into pool.free
        // (sorted insert keeps the descending free-list order intact).
        m.pools[0]
            .as_mut()
            .unwrap()
            .release(block_to_corrupt)
            .expect("releasing a held block is a valid pool op (the corruption is that the sequence still references it)");

        let before_stats = m.stats();
        let before_budget = m.budget();
        let before_seqs_len = m.seqs.len();
        let before_live = m.live_count;
        let before_next_seq = m.next_seq;
        let before_free_blocks = m.pools[0].as_ref().unwrap().free.clone();
        let before_free_slots_1 = m.fixed[1].as_ref().unwrap().free.clone();
        let before_free_slots_2 = m.fixed[2].as_ref().unwrap().free.clone();

        let err = m.free_seq(a).unwrap_err();
        assert!(
            matches!(err, StateError::InvalidBatch { .. }),
            "got {err:?}"
        );

        // Invariant: failure is atomic, zero mutations
        assert_eq!(m.seqs.len(), before_seqs_len);
        assert!(m.seqs.contains_key(&a.as_u64()));
        assert_eq!(m.live_count, before_live);
        assert_eq!(m.next_seq, before_next_seq);
        assert_eq!(m.stats(), before_stats);
        assert_eq!(m.budget(), before_budget);
        assert_eq!(m.pools[0].as_ref().unwrap().free, before_free_blocks);
        assert_eq!(m.fixed[1].as_ref().unwrap().free, before_free_slots_1);
        assert_eq!(m.fixed[2].as_ref().unwrap().free, before_free_slots_2);
    }

    #[test]
    fn free_seq_atomic_rejects_out_of_bounds_block_and_mutates_nothing() {
        let mut m = test_hybrid_manager();
        let (a, _) = m.new_seq(&[]).unwrap();
        m.reserve(a, 32).unwrap();
        m.commit(a, 32).unwrap();

        // Corrupt internal state: inject out-of-bounds block id
        m.seqs.get_mut(&a.as_u64()).unwrap().tables[0].push(u32::MAX - 10);

        let before_stats = m.stats();
        let before_budget = m.budget();
        let before_seqs_len = m.seqs.len();
        let before_live = m.live_count;
        let before_free_blocks = m.pools[0].as_ref().unwrap().free.clone();

        let err = m.free_seq(a).unwrap_err();
        assert!(
            matches!(err, StateError::InvalidBatch { .. }),
            "got {err:?}"
        );

        assert_eq!(m.seqs.len(), before_seqs_len);
        assert!(m.seqs.contains_key(&a.as_u64()));
        assert_eq!(m.live_count, before_live);
        assert_eq!(m.stats(), before_stats);
        assert_eq!(m.budget(), before_budget);
        assert_eq!(m.pools[0].as_ref().unwrap().free, before_free_blocks);
    }

    #[test]
    fn free_seq_atomic_rejects_already_free_fixed_slot_and_mutates_nothing() {
        let mut m = test_hybrid_manager();
        let (a, _) = m.new_seq(&[]).unwrap();

        let slot_id = m.fixed_slot(a, 1).unwrap();
        // Corrupt fixed pool: put slot back into free list
        m.fixed[1].as_mut().unwrap().free.insert(slot_id);

        let before_stats = m.stats();
        let before_budget = m.budget();
        let before_seqs_len = m.seqs.len();
        let before_live = m.live_count;
        let before_free_slots_1 = m.fixed[1].as_ref().unwrap().free.clone();
        let before_free_slots_2 = m.fixed[2].as_ref().unwrap().free.clone();

        let err = m.free_seq(a).unwrap_err();
        assert!(
            matches!(err, StateError::InvalidBatch { .. }),
            "got {err:?}"
        );

        assert_eq!(m.seqs.len(), before_seqs_len);
        assert!(m.seqs.contains_key(&a.as_u64()));
        assert_eq!(m.live_count, before_live);
        assert_eq!(m.stats(), before_stats);
        assert_eq!(m.budget(), before_budget);
        assert_eq!(m.fixed[1].as_ref().unwrap().free, before_free_slots_1);
        assert_eq!(m.fixed[2].as_ref().unwrap().free, before_free_slots_2);
    }

    #[test]
    fn free_seq_atomic_rejects_zero_live_count_and_mutates_nothing() {
        let mut m = test_hybrid_manager();
        let (a, _) = m.new_seq(&[]).unwrap();

        // Corrupt live_count to 0
        m.live_count = 0;

        let before_stats = m.stats();
        let before_budget = m.budget();
        let before_free_blocks = m.pools[0].as_ref().unwrap().free.clone();
        let before_free_slots_1 = m.fixed[1].as_ref().unwrap().free.clone();

        let err = m.free_seq(a).unwrap_err();
        assert!(matches!(err, StateError::Overflow { .. }), "got {err:?}");

        assert_eq!(m.live_count, 0);
        assert!(m.seqs.contains_key(&a.as_u64()));
        assert_eq!(m.stats(), before_stats);
        assert_eq!(m.budget(), before_budget);
        assert_eq!(m.pools[0].as_ref().unwrap().free, before_free_blocks);
        assert_eq!(m.fixed[1].as_ref().unwrap().free, before_free_slots_1);
    }

    #[test]
    fn free_seq_atomic_multi_group_failure_releases_zero_blocks() {
        let cfg = StateConfig {
            max_ctx: 128,
            max_seqs: 4,
        };
        let spec0 = test_kv_spec();
        let spec1 = StateSpec::KvPaged {
            hkv: 4,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        };
        let specs = vec![spec0, spec1];
        let groups = group_layers(&specs).unwrap();
        assert_eq!(groups.len(), 2);
        let req = required_pool_bytes(cfg, &groups).expect("valid required bytes");
        let mut m = StateManager::new(cfg, specs, req * 2).expect("valid manager");

        let (a, _) = m.new_seq(&[]).unwrap();
        m.reserve(a, 64).unwrap();
        m.write_tokens(a, 0, &vec![1; 64]).unwrap();
        m.commit(a, 64).unwrap();

        let free_0 = m.pools[0].as_ref().unwrap().free.len();
        let free_1 = m.pools[1].as_ref().unwrap().free.len();

        // Corrupt group 1 table: inject an invalid block id
        m.seqs.get_mut(&a.as_u64()).unwrap().tables[1].push(u32::MAX - 5);

        let err = m.free_seq(a).unwrap_err();
        assert!(
            matches!(err, StateError::InvalidBatch { .. }),
            "got {err:?}"
        );

        // Group 0 must not have released its blocks!
        assert_eq!(m.pools[0].as_ref().unwrap().free.len(), free_0);
        assert_eq!(m.pools[1].as_ref().unwrap().free.len(), free_1);
        assert!(m.seqs.contains_key(&a.as_u64()));
    }

    #[test]
    fn new_seq_atomic_multi_group_exhaustion_leaks_nothing() {
        let mut m = test_hybrid_manager();
        let _ = m.new_seq(&[]).unwrap();
        let _ = m.new_seq(&[]).unwrap();

        // Group 1 has 2 slots free, Group 2 has 2 slots free.
        // Exhaust group 2 manually:
        m.fixed[2].as_mut().unwrap().free.clear();

        let before_g1_free = m.fixed[1].as_ref().unwrap().free.clone();
        let before_next_seq = m.next_seq;
        let before_live = m.live_count;
        let before_seqs_len = m.seqs.len();

        let err = m.new_seq(&[]).unwrap_err();
        assert!(matches!(err, StateError::SeqLimit { .. }), "got {err:?}");

        // Group 1 must NOT have leaked any slots!
        assert_eq!(m.fixed[1].as_ref().unwrap().free, before_g1_free);
        assert_eq!(m.next_seq, before_next_seq);
        assert_eq!(m.live_count, before_live);
        assert_eq!(m.seqs.len(), before_seqs_len);

        // Restore a slot to group 2 and verify deterministic allocation succeeds
        m.fixed[2].as_mut().unwrap().free.insert(3);
        let (c, _) = m.new_seq(&[]).unwrap();
        assert_eq!(c.as_u64(), before_next_seq);
        assert_eq!(m.fixed_slot(c, 1).unwrap(), 2); // smallest free in group 1
        assert_eq!(m.fixed_slot(c, 2).unwrap(), 3);
    }

    #[test]
    fn new_seq_atomic_next_seq_overflow_without_leak_or_mutation() {
        let mut m = test_hybrid_manager();
        m.next_seq = u64::MAX;

        let before_g1_free = m.fixed[1].as_ref().unwrap().free.clone();
        let before_g2_free = m.fixed[2].as_ref().unwrap().free.clone();
        let before_live = m.live_count;
        let before_seqs_len = m.seqs.len();

        let err = m.new_seq(&[]).unwrap_err();
        assert!(
            matches!(
                err,
                StateError::Overflow { .. } | StateError::SeqIdOverflow { .. }
            ),
            "got {err:?}"
        );

        assert_eq!(m.next_seq, u64::MAX);
        assert_eq!(m.live_count, before_live);
        assert_eq!(m.seqs.len(), before_seqs_len);
        assert_eq!(m.fixed[1].as_ref().unwrap().free, before_g1_free);
        assert_eq!(m.fixed[2].as_ref().unwrap().free, before_g2_free);
    }

    #[test]
    fn new_seq_atomic_live_count_overflow_without_leak_or_mutation() {
        let mut m = test_hybrid_manager();
        m.config.max_seqs = u32::MAX;
        m.live_count = u32::MAX;

        let before_g1_free = m.fixed[1].as_ref().unwrap().free.clone();
        let before_g2_free = m.fixed[2].as_ref().unwrap().free.clone();
        let before_next_seq = m.next_seq;
        let before_seqs_len = m.seqs.len();

        let err = m.new_seq(&[]).unwrap_err();
        assert!(matches!(err, StateError::Overflow { .. }), "got {err:?}");

        assert_eq!(m.next_seq, before_next_seq);
        assert_eq!(m.live_count, u32::MAX);
        assert_eq!(m.seqs.len(), before_seqs_len);
        assert_eq!(m.fixed[1].as_ref().unwrap().free, before_g1_free);
        assert_eq!(m.fixed[2].as_ref().unwrap().free, before_g2_free);
    }

    #[test]
    fn new_seq_rollback_on_injected_allocation_failure() {
        let mut m = test_hybrid_manager();
        // Remove fixed pool 2 to force failure in group 2 during allocation
        m.fixed[2] = None;

        let before_g1_free = m.fixed[1].as_ref().unwrap().free.clone();
        let before_next_seq = m.next_seq;
        let before_live = m.live_count;
        let before_seqs_len = m.seqs.len();

        let err = m.new_seq(&[]).unwrap_err();
        assert!(
            matches!(err, StateError::InvalidBatch { .. }),
            "got {err:?}"
        );

        // Group 1 slot was taken and must have rolled back!
        assert_eq!(m.fixed[1].as_ref().unwrap().free, before_g1_free);
        assert_eq!(m.next_seq, before_next_seq);
        assert_eq!(m.live_count, before_live);
        assert_eq!(m.seqs.len(), before_seqs_len);
    }

    #[test]
    fn new_seq_atomic_next_seq_u32_boundary_and_overflow_fails_without_mutation() {
        let mut m = test_hybrid_manager();
        m.next_seq = u64::from(u32::MAX);

        let (seq_max, _) = m.new_seq(&[]).expect("u32::MAX sequence ID must succeed");
        assert_eq!(seq_max.as_u64(), u64::from(u32::MAX));

        let before_next_seq = m.next_seq;
        let before_live = m.live_count;
        let before_seqs_len = m.seqs.len();

        let err = m.new_seq(&[]).unwrap_err();
        assert!(
            matches!(
                err,
                StateError::SeqIdOverflow { seq, max } if seq == u64::from(u32::MAX) + 1 && max == u32::MAX
            ),
            "got {err:?}"
        );

        assert_eq!(m.next_seq, before_next_seq);
        assert_eq!(m.live_count, before_live);
        assert_eq!(m.seqs.len(), before_seqs_len);
    }

    #[test]
    fn batch_meta_out_of_range_block_table_cell_fails_before_mutation() {
        let mut m = test_hybrid_manager();
        let (seq, _) = m.new_seq(&[]).unwrap();
        m.reserve(seq, 32).unwrap();

        // Inject an additional block entry whose logical cell exceeds max_blocks
        if let Some(s) = m.seqs.get_mut(&seq.as_u64()) {
            s.indices[0].push(m.max_blocks + 10);
            s.tables[0].push(0);
        }

        let err = m.batch_meta(&[seq], &[32]).unwrap_err();
        assert!(
            matches!(err, StateError::Overflow { ref what } if what == "block table index"),
            "got {err:?}"
        );
    }

    #[test]
    fn batch_meta_seq_id_overflow_returns_exact_error_variant() {
        let m = test_hybrid_manager();
        let huge_seq = r9v_common::SeqId::new(u64::from(u32::MAX) + 42);
        let err = m.batch_meta(&[huge_seq], &[1]).unwrap_err();
        assert_eq!(
            err,
            StateError::SeqIdOverflow {
                seq: u64::from(u32::MAX) + 42,
                max: u32::MAX,
            }
        );
    }

    #[test]
    fn new_with_declarations_boundary_and_hybrid_layers_test() {
        let cfg = StateConfig {
            max_ctx: 32,
            max_seqs: 1,
        };
        let spec = StateSpec::Recurrent { h: 1, d: 8, dv: 8 };
        let decl_valid = StateDecl::new(1023, spec);
        let groups_valid = group_layer_specs(&[decl_valid]).unwrap();
        let pool_valid = required_pool_bytes(cfg, &groups_valid).unwrap();

        // 1. Layer 1023 accepted at boundary:
        let mgr = StateManager::new_with_declarations(cfg, vec![decl_valid], pool_valid);
        assert!(mgr.is_ok(), "layer 1023 must be accepted");

        // 2. Layer 1024 rejected before allocation:
        let decl_1024 = StateDecl::new(1024, spec);
        let err_1024 =
            StateManager::new_with_declarations(cfg, vec![decl_1024], pool_valid).unwrap_err();
        match err_1024 {
            StateError::InvalidConfig { problems } => {
                assert!(problems
                    .iter()
                    .any(|p| p.index == 1024 && p.reason.contains("exceeds cap 1024")));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }

        // 3. Layer u32::MAX rejected before allocation:
        let decl_max = StateDecl::new(u32::MAX, spec);
        let err_max =
            StateManager::new_with_declarations(cfg, vec![decl_max], pool_valid).unwrap_err();
        match err_max {
            StateError::InvalidConfig { problems } => {
                assert!(problems
                    .iter()
                    .any(|p| p.index == u32::MAX && p.reason.contains("exceeds cap 1024")));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }

        // 4. Hybrid multi-spec layers: 2 specs per layer on 512 layers (1024 declarations total, unique layers 512)
        // must NOT be falsely counted as > 1024 model layers:
        let spec2 = StateSpec::ConvWindow { c: 8, w: 4 };
        let mut hybrid_decls = Vec::with_capacity(1024);
        for l in 0..512 {
            hybrid_decls.push(StateDecl::new(l, spec));
            hybrid_decls.push(StateDecl::new(l, spec2));
        }
        let hybrid_groups = group_layer_specs(&hybrid_decls).unwrap();
        let hybrid_pool = required_pool_bytes(cfg, &hybrid_groups).unwrap();
        let hybrid_mgr = StateManager::new_with_declarations(cfg, hybrid_decls, hybrid_pool);
        assert!(hybrid_mgr.is_ok(), "512 hybrid layers must be accepted");
    }
}
