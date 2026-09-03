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
//! The in-memory mirror stores only token ids per reserved position so the
//! Spec 3 §8 commit/window laws can be tested without a device.

use std::collections::{BTreeMap, BTreeSet};

use r9v_common::SeqId;

use crate::error::{InvalidItem, StateError, StateResult};
use crate::spec::{
    group_layers, LayerGroup, StateSpec, BLOCK_SENTINEL, BLOCK_TOKENS, MAX_BATCH_TOKENS_HARD,
    MAX_CTX_HARD, MAX_GROUPS_HARD, MAX_LAYERS_HARD, MAX_RESERVE_HARD, MAX_SEQS_HARD,
};

/// Slot value for layer-groups with no per-token slots (recurrent/conv).
///
/// Spec 3 §3.3 defines `slot_map` as per-token KV slots; recurrent state is
/// addressed per sequence (A/B slots, §4.2), so there is no per-token slot
// DECISION(A1.11): recurrent/conv `slot_map` rows carry `SLOT_NONE` and their
// `block_table` rows carry `BLOCK_SENTINEL`; rejected: omitting the rows
// (would break the fixed `[G, ...]` shape Spec 1 §2.5 requires).
pub const SLOT_NONE: u32 = u32::MAX;

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

/// Total pool bytes required to hold every sequence at full context plus the
/// double-buffered recurrent/conv slots (Spec 3 §6.3).
///
/// The loader refuses with the numbers when the device pool is smaller.
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

/// Batch metadata shared by ops (Spec 1 §2.5, Spec 3 §3.3, §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchMeta {
    /// Sequence ids in call order, `[S]`.
    pub seq_ids: Vec<SeqId>,
    /// Query lengths, `[S]`.
    pub query_len: Vec<u32>,
    /// Verified lengths before this step, `[S]`.
    pub ctx_len: Vec<u32>,
    /// Absolute positions of the step's tokens, `[T]`.
    pub positions: Vec<u32>,
    /// Per-token slots per group, `[G, T]` (flattened; `SLOT_NONE` for
    /// recurrent/conv groups).
    pub slot_map: Vec<Vec<u32>>,
    /// Block ids per group per sequence, `[G, S, max_blocks]`, padded with
    /// [`BLOCK_SENTINEL`].
    pub block_table: Vec<Vec<Vec<u32>>>,
    /// First retained position per group per sequence, `[G, S]` (Spec 3 §3.5).
    pub window_start: Vec<Vec<u32>>,
}

impl BatchMeta {
    /// Number of layer-groups (`G`).
    pub fn num_groups(&self) -> usize {
        self.slot_map.len()
    }

    /// Number of sequences (`S`).
    pub fn num_seqs(&self) -> usize {
        self.seq_ids.len()
    }

    /// Total step tokens (`T`).
    pub fn total_tokens(&self) -> usize {
        self.positions.len()
    }

    /// Table width (`max_blocks = ceil(max_ctx / 32)`).
    pub fn max_blocks(&self) -> usize {
        self.block_table
            .first()
            .and_then(|g| g.first())
            .map_or(0, Vec::len)
    }
}

/// Free/total pool state per group (Spec 3 §5 `budget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupBudget {
    /// Layer-group index.
    pub index: usize,
    /// Total blocks in the group pool.
    pub total_blocks: u32,
    /// Free blocks in the group pool.
    pub free_blocks: u32,
    /// Bytes per block across the group's layers.
    pub block_bytes: u64,
    /// Arena base offset of this group's pool.
    pub base_offset: u64,
}

/// Pool budget snapshot (Spec 3 §5 `budget`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    /// Per-group budgets, in group order.
    pub groups: Vec<GroupBudget>,
    /// Total arena bytes across paged pools.
    pub pool_bytes_total: u64,
    /// Free arena bytes across paged pools.
    pub pool_bytes_free: u64,
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
    /// Fraction of paged blocks allocated, `0.0..=1.0`.
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
    /// Accepted tokens awaiting a recurrent re-run after a partial commit
    /// (Spec 3 §4.2); cleared by the next `reserve`.
    recompute_pending: u32,
    /// Active A/B slot per group (`0 = A`, `1 = B`); meaningful only for
    /// recurrent/conv groups.
    parity: Vec<u8>,
    live: bool,
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
    max_blocks: u32,
    pool_bytes_total: u64,
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
    /// Refuses with the numbers when `pool_bytes` cannot hold every sequence
    /// at full context plus the double-buffered recurrent slots. All
    /// validation is collected before any allocation.
    pub fn new(
        config: StateConfig,
        layer_specs: Vec<StateSpec>,
        pool_bytes: u64,
    ) -> StateResult<Self> {
        config.validate()?;
        let mut problems = Vec::new();
        if layer_specs.len() > MAX_LAYERS_HARD as usize {
            problems.push(InvalidItem {
                index: u32::MAX,
                reason: format!(
                    "layers={} exceeds cap {}",
                    layer_specs.len(),
                    MAX_LAYERS_HARD
                ),
            });
        }
        for (i, spec) in layer_specs.iter().enumerate() {
            spec.validate(i as u32, &mut problems);
        }
        let groups = group_layers(&layer_specs);
        if groups.len() > MAX_GROUPS_HARD {
            problems.push(InvalidItem {
                index: u32::MAX,
                reason: format!("groups={} exceeds cap {}", groups.len(), MAX_GROUPS_HARD),
            });
        }
        if !problems.is_empty() {
            return Err(StateError::invalid(problems));
        }

        let required = required_pool_bytes(config, &groups)?;
        if pool_bytes < required {
            return Err(StateError::invalid(vec![InvalidItem {
                index: u32::MAX,
                reason: format!(
                    "pool_bytes={pool_bytes} below required={required}, shortfall={}",
                    required - pool_bytes,
                ),
            }]));
        }

        // DECISION(A1.11): each paged group holds exactly max_ctx/32 blocks
        // (full context per sequence) with contiguous arena bases; rejected: a
        // proportionally-shrunk pool (would make admission depend on group
        // mix instead of failing loudly at construction). Spec 3 §6.3 is
        // silent on sub-full pools.
        let max_blocks = config.max_blocks();
        let mut pools: Vec<Option<BlockPool>> = Vec::with_capacity(groups.len());
        let mut base: u64 = 0;
        let mut pool_bytes_total: u64 = 0;
        for g in &groups {
            if g.spec.is_paged() {
                let block_bytes = g.block_bytes()?;
                let bytes = block_bytes
                    .checked_mul(u64::from(max_blocks))
                    .ok_or_else(|| StateError::Overflow {
                        what: "group pool bytes".to_owned(),
                    })?;
                base = base
                    .checked_add(bytes)
                    .ok_or_else(|| StateError::Overflow {
                        what: "arena base".to_owned(),
                    })?;
                pool_bytes_total =
                    pool_bytes_total
                        .checked_add(bytes)
                        .ok_or_else(|| StateError::Overflow {
                            what: "pool total".to_owned(),
                        })?;
                pools.push(Some(BlockPool {
                    total: max_blocks,
                    free: (0..max_blocks).collect(),
                    base_offset: base - bytes,
                    block_bytes,
                }));
            } else {
                pools.push(None);
            }
        }

        Ok(Self {
            config,
            groups,
            pools,
            max_blocks,
            pool_bytes_total,
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
    pub fn new_seq(&mut self, tokens: &[u32]) -> StateResult<(SeqId, u32)> {
        if tokens.len() > self.config.max_ctx as usize {
            return Err(StateError::ReserveTooLarge {
                end: tokens.len().min(u32::MAX as usize) as u32,
                max_ctx: self.config.max_ctx,
                n: tokens.len().min(u32::MAX as usize) as u32,
            });
        }
        if self.live_count >= self.config.max_seqs {
            return Err(StateError::SeqLimit {
                live: self.live_count,
                cap: self.config.max_seqs,
            });
        }
        let id = self.next_seq;
        self.next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or_else(|| StateError::Overflow {
                what: "sequence id".to_owned(),
            })?;
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
                recompute_pending: 0,
                parity: vec![0; g],
                live: true,
            },
        );
        self.live_count += 1;
        Ok((SeqId::new(id), 0))
    }

    fn seq_mut(&mut self, seq: SeqId) -> StateResult<&mut SeqState> {
        self.seqs
            .get_mut(&seq.as_u64())
            .filter(|s| s.live)
            .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })
    }

    fn seq(&self, seq: SeqId) -> StateResult<&SeqState> {
        self.seqs
            .get(&seq.as_u64())
            .filter(|s| s.live)
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
            // DECISION(A1.11): Sink+Window reports the window start; the sink
            // range is implicitly [0, ceil(n/32)*32) pinned from position 0.
            // Rejected: reporting 0 (would hide the window from the kernel).
            // Spec 3 §3.5 gives the kernel "both ranges" but BatchMeta carries
            // a single window_start per group.
            Some(crate::spec::Retain::Window { w })
            | Some(crate::spec::Retain::SinkWindow { w, .. }) => s.ctx_len.saturating_sub(w),
        })
    }

    /// Tokens awaiting a recurrent re-run after a partial commit (Spec 3 §4.2).
    pub fn recompute_pending(&self, seq: SeqId) -> StateResult<u32> {
        Ok(self.seq(seq)?.recompute_pending)
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

    /// Free blocks in a paged group pool.
    pub fn free_blocks(&self, group: usize) -> StateResult<u32> {
        self.pools
            .get(group)
            .and_then(|p| p.as_ref())
            .map(|p| p.free.len() as u32)
            .ok_or_else(|| StateError::InvalidBatch {
                detail: format!("group {group} is not a paged group"),
            })
    }

    /// Ensures blocks exist for `ctx_len .. ctx_len + n` (Spec 3 §3.6).
    ///
    /// Atomic: every group is checked before any block is allocated, so a
    /// refusal leaves the sequence and the pools untouched.
    pub fn reserve(&mut self, seq: SeqId, n: u32) -> StateResult<SlotRange> {
        if n == 0 || n > MAX_RESERVE_HARD {
            let tail = self.seq(seq).map(|s| s.tail_len).unwrap_or(0);
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

        // Check every group first (atomicity): needed new block indices.
        let mut need: Vec<Vec<u32>> = Vec::with_capacity(self.groups.len());
        for (gi, g) in self.groups.iter().enumerate() {
            if !g.spec.is_paged() {
                need.push(Vec::new());
                continue;
            }
            let want = retained_indices(g.spec, end, self.config.max_ctx);
            let held: BTreeSet<u32> = self.seq(seq)?.indices[gi].iter().copied().collect();
            let missing: Vec<u32> = want.into_iter().filter(|i| !held.contains(i)).collect();
            let free = self.pools[gi].as_ref().map_or(0, |p| p.free.len() as u64);
            if (missing.len() as u64) > free {
                let pool = self.pools[gi].as_ref().map_or(0, |p| p.free.len() as u32);
                let required = missing.len() as u32;
                return Err(StateError::PoolExhausted {
                    group: gi,
                    required,
                    available: pool,
                    shortfall: required - pool,
                    end,
                    max_ctx: self.config.max_ctx,
                });
            }
            need.push(missing);
        }

        // All checks passed: mutate (disjoint field borrows: pools, seqs).
        {
            let pools = &mut self.pools;
            let seqs = &mut self.seqs;
            for (gi, missing) in need.iter().enumerate() {
                if missing.is_empty() {
                    continue;
                }
                let pool = pools[gi].as_mut().ok_or_else(|| StateError::InvalidBatch {
                    detail: format!("group {gi} is not a paged group"),
                })?;
                let s = seqs
                    .get_mut(&seq.as_u64())
                    .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })?;
                for idx in missing {
                    let id = pool.alloc().ok_or(StateError::PoolExhausted {
                        group: gi,
                        required: missing.len() as u32,
                        available: 0,
                        shortfall: missing.len() as u32,
                        end,
                        max_ctx: self.config.max_ctx,
                    })?;
                    let pos = s.indices[gi].partition_point(|&i| i < *idx);
                    s.indices[gi].insert(pos, *idx);
                    s.tables[gi].insert(pos, id);
                }
            }
        }
        let s = self.seq_mut(seq)?;
        s.tail_len = n;
        s.compacted = None;
        s.recompute_pending = 0;

        // Flattened slots per group for the reserved range.
        let mut slots: Vec<Vec<u32>> = Vec::with_capacity(self.groups.len());
        for (gi, g) in self.groups.iter().enumerate() {
            if !g.spec.is_paged() {
                slots.push(vec![SLOT_NONE; n as usize]);
                continue;
            }
            let table = &self.seq(seq)?.tables[gi];
            let indices = &self.seq(seq)?.indices[gi];
            let mut row = Vec::with_capacity(n as usize);
            for k in 0..n {
                let pos = ctx + k;
                let bi = pos / BLOCK_TOKENS;
                let lane = pos % BLOCK_TOKENS;
                let at = indices
                    .partition_point(|&i| i < bi)
                    .min(indices.len().saturating_sub(1));
                debug_assert!(indices.get(at) == Some(&bi));
                let id = table[at];
                let slot = u64::from(id)
                    .checked_mul(u64::from(BLOCK_TOKENS))
                    .and_then(|v| v.checked_add(u64::from(lane)))
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| StateError::Overflow {
                        what: "slot id".to_owned(),
                    })?;
                row.push(slot);
            }
            slots.push(row);
        }
        Ok(SlotRange {
            start: ctx,
            len: n,
            slots,
        })
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
        let src: Vec<u32> = accepted_positions.iter().map(|p| ctx + p).collect();
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
                let dst = ctx + i as u32;
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
    /// partial accepts defer the swap and record [`Self::recompute_pending`].
    /// Atomic: validation precedes any mutation.
    pub fn commit(&mut self, seq: SeqId, accepted: u32) -> StateResult<()> {
        let (ctx, tail, compacted) = {
            let s = self.seq(seq)?;
            (s.ctx_len, s.tail_len, s.compacted)
        };
        if accepted > tail {
            return Err(StateError::CommitTooLarge { accepted, tail });
        }
        if let Some(c) = compacted {
            if accepted != c as u32 {
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

        // Compute window releases before mutating: (table position, block index).
        let mut releases: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.groups.len()];
        for (gi, g) in self.groups.iter().enumerate() {
            let retain = match g.spec.retain() {
                Some(r) if r.is_windowed() => r,
                _ => continue,
            };
            let ws = new_ctx.saturating_sub(retain.window().unwrap_or(0));
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

        // Mutate (disjoint field borrows: seqs, pools, groups, swaps).
        let has_recurrent = self.groups.iter().any(|g| g.spec.is_recurrent());
        let recurrent: Vec<bool> = self.groups.iter().map(|g| g.spec.is_recurrent()).collect();
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
            if has_recurrent {
                if accepted == tail && accepted > 0 {
                    for (gi, is_rec) in recurrent.iter().enumerate() {
                        if *is_rec {
                            s.parity[gi] ^= 1;
                        }
                    }
                    *swaps += 1;
                } else if accepted < tail {
                    s.recompute_pending = accepted;
                }
            }
            *commits += 1;
        }
        Ok(())
    }

    /// Releases all references; may retain session state (Spec 3 §5).
    ///
    /// Session retention is deferred to roadmap B1: everything is released
    /// and nothing is retained.
    pub fn free_seq(&mut self, seq: SeqId) -> StateResult<()> {
        let s = self
            .seqs
            .get_mut(&seq.as_u64())
            .filter(|s| s.live)
            .ok_or(StateError::UnknownSeq { seq: seq.as_u64() })?;
        s.live = false;
        let tables = std::mem::take(&mut s.tables);
        for (gi, ids) in tables.into_iter().enumerate() {
            if let Some(pool) = self.pools[gi].as_mut() {
                for id in ids {
                    pool.release(id);
                }
            }
        }
        self.live_count = self.live_count.saturating_sub(1);
        Ok(())
    }

    /// Builds `BatchMeta` for one step (Spec 1 §2.5, Spec 3 §5).
    ///
    /// Order follows the input slices. Each `query_len` must be covered by
    /// that sequence's outstanding reservation. Problems across sequences are
    /// collected into one typed error.
    pub fn batch_meta(&self, seqs: &[SeqId], query_lens: &[u32]) -> StateResult<BatchMeta> {
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
                    total = total.saturating_add(u64::from(*q));
                }
            }
        }
        if total > MAX_BATCH_TOKENS_HARD {
            problems.push(format!(
                "batch tokens={total} exceed cap {MAX_BATCH_TOKENS_HARD}"
            ));
        }
        if !problems.is_empty() {
            return Err(StateError::InvalidBatch {
                detail: problems.join("; "),
            });
        }

        let mut ctx_len = Vec::with_capacity(seqs.len());
        let mut positions: Vec<u32> = Vec::new();
        for (seq, q) in seqs.iter().zip(query_lens.iter()) {
            let s = self.seq(*seq).map_err(|_| StateError::InvalidBatch {
                detail: "sequence vanished during batch build".to_owned(),
            })?;
            ctx_len.push(s.ctx_len);
            for k in 0..*q {
                positions.push(s.ctx_len + k);
            }
        }

        let mut slot_map: Vec<Vec<u32>> = Vec::with_capacity(self.groups.len());
        let mut block_table: Vec<Vec<Vec<u32>>> = Vec::with_capacity(self.groups.len());
        let mut window_start: Vec<Vec<u32>> = Vec::with_capacity(self.groups.len());
        for (gi, g) in self.groups.iter().enumerate() {
            // slot_map row
            if !g.spec.is_paged() {
                slot_map.push(vec![SLOT_NONE; positions.len()]);
            } else {
                let mut row = Vec::with_capacity(positions.len());
                for (si, (seq, q)) in seqs.iter().zip(query_lens.iter()).enumerate() {
                    let s = &self.seqs[&seq.as_u64()];
                    for k in 0..*q {
                        let pos = ctx_len[si] + k;
                        let bi = pos / BLOCK_TOKENS;
                        let lane = pos % BLOCK_TOKENS;
                        let at = s.indices[gi].partition_point(|&i| i < bi);
                        let id = s.tables[gi][at];
                        row.push(id * BLOCK_TOKENS + lane);
                    }
                }
                slot_map.push(row);
            }
            // block_table + window_start rows
            let mut table_rows = Vec::with_capacity(seqs.len());
            let mut ws_rows = Vec::with_capacity(seqs.len());
            for seq in seqs {
                let s = &self.seqs[&seq.as_u64()];
                let mut row = vec![BLOCK_SENTINEL; self.max_blocks as usize];
                if g.spec.is_paged() {
                    for (at, id) in s.tables[gi].iter().enumerate() {
                        row[at] = *id;
                    }
                }
                table_rows.push(row);
                let ws = match g.spec.retain() {
                    None | Some(crate::spec::Retain::All) => 0,
                    Some(crate::spec::Retain::Window { w })
                    | Some(crate::spec::Retain::SinkWindow { w, .. }) => {
                        s.ctx_len.saturating_sub(w)
                    }
                };
                ws_rows.push(ws);
            }
            block_table.push(table_rows);
            window_start.push(ws_rows);
        }

        Ok(BatchMeta {
            seq_ids: seqs.to_vec(),
            query_len: query_lens.to_vec(),
            ctx_len,
            positions,
            slot_map,
            block_table,
            window_start,
        })
    }

    /// Pool budget snapshot (Spec 3 §5 `budget`).
    pub fn budget(&self) -> Budget {
        let mut groups = Vec::with_capacity(self.groups.len());
        let mut free_bytes: u64 = 0;
        for gi in 0..self.groups.len() {
            match &self.pools[gi] {
                Some(p) => {
                    free_bytes += (p.free.len() as u64) * p.block_bytes;
                    groups.push(GroupBudget {
                        index: gi,
                        total_blocks: p.total,
                        free_blocks: p.free.len() as u32,
                        block_bytes: p.block_bytes,
                        base_offset: p.base_offset,
                    });
                }
                None => groups.push(GroupBudget {
                    index: gi,
                    total_blocks: 0,
                    free_blocks: 0,
                    block_bytes: 0,
                    base_offset: 0,
                }),
            }
        }
        Budget {
            groups,
            pool_bytes_total: self.pool_bytes_total,
            pool_bytes_free: free_bytes,
        }
    }

    /// Manager statistics (Spec 3 §5 `stats`).
    pub fn stats(&self) -> Stats {
        let mut alloc: u64 = 0;
        let mut total: u64 = 0;
        for pool in self.pools.iter().flatten() {
            alloc += (pool.total - pool.free.len() as u32) as u64;
            total += pool.total as u64;
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

/// Block indices retained at `end` for a paged spec (Spec 3 §3.5).
fn retained_indices(spec: StateSpec, end: u32, max_ctx: u32) -> Vec<u32> {
    let _ = max_ctx;
    let first_block = |pos: u32| pos / BLOCK_TOKENS;
    let last_block = |end: u32| end.div_ceil(BLOCK_TOKENS);
    match spec.retain() {
        None | Some(crate::spec::Retain::All) => (0..last_block(end)).collect(),
        Some(crate::spec::Retain::Window { w }) => {
            let ws = end.saturating_sub(w);
            (first_block(ws)..last_block(end)).collect()
        }
        Some(crate::spec::Retain::SinkWindow { n, w }) => {
            let sink_end = n.min(end);
            let ws = end.saturating_sub(w);
            let mut out: Vec<u32> = (0..last_block(sink_end)).collect();
            for b in first_block(ws)..last_block(end) {
                if !out.contains(&b) {
                    out.push(b);
                }
            }
            out
        }
    }
}
