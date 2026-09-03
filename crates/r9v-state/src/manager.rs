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
use r9v_ir::{BatchMeta, Positions, TreeMask, BLOCK_TABLE_SENTINEL};

use crate::error::{InvalidItem, StateError, StateResult};
use crate::spec::{
    group_layer_specs, group_layers, LayerGroup, StateDecl, StateSpec, BLOCK_TOKENS,
    MAX_BATCH_TOKENS_HARD, MAX_CTX_HARD, MAX_RESERVE_HARD, MAX_SEQS_HARD,
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

/// Slots reserved for one step (Spec 3 §5 `reserve`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotRange {
    /// First reserved position (the sequence's `ctx_len` at reserve time).
    pub start: u32,
    /// Reserved token count.
    pub len: u32,
    /// Flattened slots per group (`slots[g][i]`), `SLOT_NONE` for
    /// recurrent/conv groups.
    pub slots: Vec<Vec<u32>>,
}

/// Tree-verify compaction descriptor (Spec 3 §3.6).
///
/// The scheduler enqueues this as a tiny kernel copying the accepted tokens'
/// K/V (and scales) into `dst_start .. dst_start + len` within the same
/// blocks, then commits. The manager applies the same copy to its in-memory
/// mirror eagerly so `commit` observes compacted positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactOp {
    /// Sequence being compacted.
    pub seq: SeqId,
    /// Absolute source positions, in accepted-path order.
    pub src_positions: Vec<u32>,
    /// Destination start (the sequence's `ctx_len`).
    pub dst_start: u32,
    /// Accepted token count.
    pub len: u32,
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
    free: BTreeSet<u32>,
    base_offset: u64,
    block_bytes: u64,
}

impl BlockPool {
    fn alloc(&mut self) -> Option<u32> {
        let id = *self.free.iter().next()?;
        self.free.remove(&id);
        Some(id)
    }

    fn release(&mut self, id: u32) {
        self.free.insert(id);
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
                    free: (0..blocks_per_group).collect(),
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
        let g = self.groups.len();
        self.seqs.insert(
            id,
            SeqState {
                ctx_len: 0,
                tail_len: 0,
                tables: vec![Vec::new(); g],
                indices: vec![Vec::new(); g],
                mirror: BTreeMap::new(),
                compacted: None,
                fixed_slots,
                parity: vec![0; g],
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
        // Check every group first (atomicity): needed new block indices.
        // Each entry carries its checked block count so later error reports
        // need no further conversion.
        let mut need: Vec<(Vec<u32>, u32)> = Vec::with_capacity(self.groups.len());
        for (gi, g) in self.groups.iter().enumerate() {
            if !g.spec.is_paged() {
                need.push((Vec::new(), 0));
                continue;
            }
            let held: BTreeSet<u32> = self.seq(seq)?.indices[gi].iter().copied().collect();
            let mut missing: Vec<u32> = Vec::new();
            for idx in first_block..last_block {
                if !held.contains(&idx) {
                    missing.push(idx);
                }
            }
            let free = u64::try_from(self.pools[gi].as_ref().map_or(0, |p| p.free.len())).map_err(
                |_| StateError::Overflow {
                    what: "free block count".to_owned(),
                },
            )?;
            let want = u64::try_from(missing.len()).map_err(|_| StateError::Overflow {
                what: "missing block count".to_owned(),
            })?;
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
            let count = u32::try_from(missing.len()).map_err(|_| StateError::Overflow {
                what: "missing block count".to_owned(),
            })?;
            need.push((missing, count));
        }

        // All checks passed: allocate (disjoint field borrows: pools, seqs).
        // Every insert is recorded; any failure below rolls the recorded
        // inserts back instead of leaving half-built tables.
        let mut inserted: Vec<(usize, u32, u32)> = Vec::new();
        let mut alloc_err: Option<StateError> = None;
        {
            let pools = &mut self.pools;
            let seqs = &mut self.seqs;
            'alloc: for (gi, (missing, count)) in need.iter().enumerate() {
                if missing.is_empty() {
                    continue;
                }
                let pool = match pools[gi].as_mut() {
                    Some(pool) => pool,
                    None => {
                        alloc_err = Some(StateError::InvalidBatch {
                            detail: format!("group {gi} is not a paged group"),
                        });
                        break 'alloc;
                    }
                };
                let s = match seqs.get_mut(&seq.as_u64()) {
                    Some(s) => s,
                    None => {
                        alloc_err = Some(StateError::UnknownSeq { seq: seq.as_u64() });
                        break 'alloc;
                    }
                };
                for idx in missing {
                    let Some(id) = pool.alloc() else {
                        // Unreachable after the pre-check above (single
                        // thread, counts already verified); handled like any
                        // other failure: roll back, then report.
                        alloc_err = Some(StateError::PoolExhausted {
                            group: gi,
                            required: *count,
                            available: 0,
                            shortfall: *count,
                            end,
                            max_ctx: self.config.max_ctx,
                        });
                        break 'alloc;
                    };
                    let pos = s.indices[gi].partition_point(|&i| i < *idx);
                    s.indices[gi].insert(pos, *idx);
                    s.tables[gi].insert(pos, id);
                    inserted.push((gi, *idx, id));
                }
            }
        }
        if let Some(e) = alloc_err {
            self.rollback_reserve(seq, &inserted);
            return Err(e);
        }
        // Flattened slots per group for the reserved range. A missing mapping
        // is a typed error, never a clamp to a neighboring block; the inserts
        // above roll back on failure so the sequence keeps no tail.
        let slots = match self.slot_rows_for(seq, ctx, n, end) {
            Ok(rows) => rows,
            Err(e) => {
                self.rollback_reserve(seq, &inserted);
                return Err(e);
            }
        };
        let s = self.seq_mut(seq)?;
        s.tail_len = n;
        s.compacted = None;
        Ok(SlotRange {
            start: ctx,
            len: n,
            slots,
        })
    }

    /// Undoes a half-built reservation's block inserts: removes the recorded
    /// `(group, index)` entries in reverse and returns their ids to the
    /// pools. The tail is still 0 on this path, so the sequence lands exactly
    /// where it was before `reserve`.
    fn rollback_reserve(&mut self, seq: SeqId, inserted: &[(usize, u32, u32)]) {
        for &(gi, idx, id) in inserted.iter().rev() {
            if let Some(s) = self.seqs.get_mut(&seq.as_u64()) {
                let at = s
                    .indices
                    .get(gi)
                    .map(|v| v.partition_point(|&i| i < idx))
                    .unwrap_or(usize::MAX);
                let aligned = s.indices.get(gi).and_then(|v| v.get(at)).copied() == Some(idx)
                    && s.tables.get(gi).is_some_and(|t| at < t.len());
                if aligned {
                    s.indices[gi].remove(at);
                    s.tables[gi].remove(at);
                }
            }
            if let Some(pool) = self.pools.get_mut(gi).and_then(|p| p.as_mut()) {
                pool.release(id);
            }
        }
    }

    /// Flattened slot rows for a reservation: `slots[g][k]` covers
    /// `ctx + k` (Spec 1 §2.5, Spec 3 §3.3).
    fn slot_rows_for(&self, seq: SeqId, ctx: u32, n: u32, end: u32) -> StateResult<Vec<Vec<u32>>> {
        let mut slots: Vec<Vec<u32>> = Vec::with_capacity(self.groups.len());
        for (gi, g) in self.groups.iter().enumerate() {
            if !g.spec.is_paged() {
                slots.push(vec![SLOT_NONE; n as usize]);
                continue;
            }
            let s = self.seq(seq)?;
            let mut row = Vec::with_capacity(n as usize);
            for k in 0..n {
                let pos = ctx.checked_add(k).ok_or_else(|| StateError::Overflow {
                    what: "slot position".to_owned(),
                })?;
                row.push(flatten_slot(s, gi, pos, end)?);
            }
            slots.push(row);
        }
        Ok(slots)
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
        let paged: Vec<bool> = self.groups.iter().map(|g| g.spec.is_paged()).collect();
        let s = self.seq_mut(seq)?;
        for (gi, is_paged) in paged.iter().enumerate() {
            if !is_paged {
                continue;
            }
            for (k, tok) in tokens.iter().enumerate() {
                s.mirror.insert((gi, start + k as u32), *tok);
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
        let mut seen = BTreeSet::new();
        for p in accepted_positions {
            if *p >= tail {
                return Err(detail(format!("position {p} out of range tail {tail}")));
            }
            if !seen.insert(*p) {
                return Err(detail(format!("duplicate position {p}")));
            }
        }
        let a = accepted_positions.len();
        let mut src: Vec<u32> = Vec::with_capacity(accepted_positions.len());
        for p in accepted_positions {
            src.push(ctx.checked_add(*p).ok_or_else(|| StateError::Overflow {
                what: "compact source".to_owned(),
            })?);
        }
        // Destination positions are computed before any mutation so the write
        // phase below cannot fail partway.
        let mut dsts: Vec<u32> = Vec::with_capacity(accepted_positions.len());
        for i in 0..accepted_positions.len() {
            dsts.push(
                ctx.checked_add(i as u32)
                    .ok_or_else(|| StateError::Overflow {
                        what: "compact destination".to_owned(),
                    })?,
            );
        }
        // Read phase (immutable): stage every group's copies before mutating.
        let staged: Vec<Vec<(u32, Option<u32>)>> = {
            let s = self.seq(seq)?;
            self.groups
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    if !g.spec.is_paged() {
                        Vec::new()
                    } else {
                        src.iter()
                            .map(|abs| (*abs, s.mirror.get(&(gi, *abs)).copied()))
                            .collect()
                    }
                })
                .collect()
        };
        // Write phase: apply staged copies, then record the compacted length.
        let s = self.seq_mut(seq)?;
        for (gi, copies) in staged.iter().enumerate() {
            for (i, (_, tok)) in copies.iter().enumerate() {
                let dst = dsts[i];
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
        s.compacted = Some(a);
        Ok(CompactOp {
            seq,
            src_positions: src,
            dst_start: ctx,
            len: a as u32,
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
        let swap_groups: Vec<bool> = self.groups.iter().map(|g| g.spec.is_recurrent()).collect();

        // Compute window releases before mutating: (table position, block index).
        let mut releases: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.groups.len()];
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
            let indices = &self.seq(seq)?.indices[gi];
            for (at, idx) in indices.iter().enumerate() {
                if *idx < sink_blocks {
                    continue;
                }
                let block_end = idx
                    .checked_add(1)
                    .and_then(|v| v.checked_mul(BLOCK_TOKENS))
                    .ok_or_else(|| StateError::Overflow {
                        what: "block end".to_owned(),
                    })?;
                if block_end <= ws {
                    releases[gi].push((at as u32, *idx));
                }
            }
        }

        // Mutate (disjoint field borrows: seqs, pools, swaps, commits).
        {
            let Self {
                seqs,
                pools,
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
            for (gi, ato) in releases.iter().enumerate() {
                let mut ids = Vec::with_capacity(ato.len());
                let mut dropped: Vec<u32> = Vec::with_capacity(ato.len());
                for &(at, idx) in ato.iter().rev() {
                    s.indices[gi].remove(at as usize);
                    ids.push(s.tables[gi].remove(at as usize));
                    dropped.push(idx);
                }
                // Released blocks are gone: drop their mirrored tokens so a
                // re-read of an evicted position reports absence (Spec 3 §3.5).
                s.mirror.retain(|(g, p), _| {
                    *g != gi || !dropped.contains(&(*p / crate::spec::BLOCK_TOKENS))
                });
                for id in ids {
                    pools[gi]
                        .as_mut()
                        .ok_or_else(|| StateError::InvalidBatch {
                            detail: format!("group {gi} is not a paged group"),
                        })?
                        .release(id);
                }
            }
            if swap {
                for (gi, is_rec) in swap_groups.iter().enumerate() {
                    if *is_rec {
                        s.parity[gi] ^= 1;
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
                if pool.free.contains(&id) {
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

        // Infallible commit phase: all validations passed, so mutations cannot fail.
        let s = self
            .seqs
            .remove(&seq.as_u64())
            .expect("validated sequence exists in seqs");

        for (gi, ids) in s.tables.into_iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let pool = self
                .pools
                .get_mut(gi)
                .and_then(|p| p.as_mut())
                .expect("validated paged pool exists");
            for id in ids {
                pool.release(id);
            }
        }
        for (gi, slot) in s.fixed_slots.into_iter().enumerate() {
            let Some(id) = slot else { continue };
            let pool = self
                .fixed
                .get_mut(gi)
                .and_then(|p| p.as_mut())
                .expect("validated fixed pool exists");
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
    pub fn batch_meta(&self, seqs: &[SeqId], query_lens: &[u32]) -> StateResult<BatchMeta> {
        self.batch_meta_with_options(seqs, query_lens, None, None)
    }

    /// Builds canonical [`r9v_ir::BatchMeta`] with optional speculative [`TreeMask`] (Spec 1 §4.D.1, Spec 3 §5).
    pub fn batch_meta_with_tree(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        tree: Option<TreeMask>,
    ) -> StateResult<BatchMeta> {
        self.batch_meta_with_options(seqs, query_lens, None, tree)
    }

    /// Builds canonical [`r9v_ir::BatchMeta`] with explicit positions (scalar or MRoPE) and optional [`TreeMask`].
    // DECISION(A1.15): device seq_ids are checked global u32 identifiers validated with u32::try_from; sequence allocation checks that next_seq <= u32::MAX before mutating state; rejected lossy as-casts, rollover, batch-local indexing that breaks Philox batch invariance, and unbounded host-to-device ID maps. Spec 1 §2.5, §4.F, Spec 3 §5, SI-40.
    pub fn batch_meta_with_options(
        &self,
        seqs: &[SeqId],
        query_lens: &[u32],
        positions: Option<Positions>,
        tree: Option<TreeMask>,
    ) -> StateResult<BatchMeta> {
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
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        for (i, seq) in seqs.iter().enumerate() {
            if seq.as_u64() > u64::from(u32::MAX) {
                return Err(StateError::SeqIdOverflow {
                    seq: seq.as_u64(),
                    max: u32::MAX,
                });
            }
            if !seen.insert(seq.as_u64()) {
                problems.push(format!("batch[{i}]: duplicate sequence {}", seq.as_u64()));
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
        if let Some(ref pos) = positions {
            if pos.len() != total_tokens {
                problems.push(format!(
                    "positions len {} != total tokens {total_tokens}",
                    pos.len()
                ));
            }
        }
        if let Some(ref tr) = tree {
            if tr.t() != total_tokens {
                problems.push(format!(
                    "tree token count {} != total tokens {total_tokens}",
                    tr.t()
                ));
            }
        }
        if !problems.is_empty() {
            return Err(StateError::InvalidBatch {
                detail: problems.join("; "),
            });
        }

        let mut seq_ids = Vec::with_capacity(seqs.len());
        for seq in seqs {
            let dev_id = u32::try_from(seq.as_u64()).map_err(|_| StateError::SeqIdOverflow {
                seq: seq.as_u64(),
                max: u32::MAX,
            })?;
            seq_ids.push(dev_id);
        }

        let mut ctx_len = Vec::with_capacity(seqs.len());
        let mut default_positions: Vec<u32> = Vec::with_capacity(total_tokens);
        for (seq, q) in seqs.iter().zip(query_lens.iter()) {
            let s = self.seq(*seq).map_err(|_| StateError::InvalidBatch {
                detail: "sequence vanished during batch build".to_owned(),
            })?;
            ctx_len.push(s.ctx_len);
            for k in 0..*q {
                default_positions.push(s.ctx_len.checked_add(k).ok_or_else(|| {
                    StateError::Overflow {
                        what: "batch position".to_owned(),
                    }
                })?);
            }
        }
        let final_positions = positions.unwrap_or(Positions::PerToken(default_positions));

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

        for (gi, g) in self.groups.iter().enumerate() {
            // slot_map [G, T] row-major
            if !g.spec.is_paged() {
                flat_slot_map.extend(std::iter::repeat_n(SLOT_NONE, total_tokens));
            } else {
                for (si, (seq, q)) in seqs.iter().zip(query_lens.iter()).enumerate() {
                    let s = self.seq(*seq).map_err(|_| StateError::InvalidBatch {
                        detail: "sequence vanished during batch build".to_owned(),
                    })?;
                    let end = ctx_len[si]
                        .checked_add(*q)
                        .ok_or_else(|| StateError::Overflow {
                            what: "batch reserved end".to_owned(),
                        })?;
                    for k in 0..*q {
                        let pos =
                            ctx_len[si]
                                .checked_add(k)
                                .ok_or_else(|| StateError::Overflow {
                                    what: "batch position".to_owned(),
                                })?;
                        flat_slot_map.push(flatten_slot(s, gi, pos, end)?);
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
                        let cell_ref = flat_block_table.get_mut(offset).ok_or_else(|| {
                            StateError::Overflow {
                                what: "block table index".to_owned(),
                            }
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
                flat_window_start.push(ws);
            }
        }

        let num_groups = u32::try_from(self.groups.len()).map_err(|_| StateError::Overflow {
            what: "batch meta group count".to_owned(),
        })?;
        let num_seqs = u32::try_from(seqs.len()).map_err(|_| StateError::Overflow {
            what: "batch meta seq count".to_owned(),
        })?;
        let total_tokens_u32 = u32::try_from(total).map_err(|_| StateError::Overflow {
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

/// Flattened slot for one retained position: `block_id * 32 + lane`
/// (Spec 1 §2.5, Spec 3 §3.3).
///
// DECISION(A1.11) per SI-17: the flattened value carries the pool-global
// block id, not the within-table position, so the device derives
// `base[group] + block_id * block_bytes` directly; ids are per-group-pool,
// never arena-global across groups. A position with no mapped block is a
// typed error, never a clamp to a neighboring block.
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
    fn free_seq_atomic_rejects_already_free_block_and_mutates_nothing() {
        let mut m = test_hybrid_manager();
        let (a, _) = m.new_seq(&[]).unwrap();
        m.reserve(a, 64).unwrap();
        m.write_tokens(a, 0, &vec![1; 64]).unwrap();
        m.commit(a, 64).unwrap();

        let s = m.seq(a).unwrap();
        let block_to_corrupt = s.tables[0][0];

        // Corrupt internal state: insert block back into pool.free
        m.pools[0].as_mut().unwrap().free.insert(block_to_corrupt);

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
