# Spec 3 — Sequence State

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 1, 2. Constrains: specs 4, 5, 6, 7, 9.

## 0. Purpose and scope

Everything a sequence carries between steps that isn't weights: attention KV, MLA latents, linear-attention recurrent state, causal-conv windows. This spec defines the state kinds, how each is stored and addressed, the manager that owns allocation, prefix reuse, checkpoint and rollback, and the VRAM budget it lives in.

Out of scope: the attention and scan kernels themselves (spec 4), who decides which sequences run this step (spec 6), what a proposer does with rollback (spec 7).

## 1. Principles

1. **Per-layer, opaque to the graph.** The graph sees `StateHandle(layer, kind)`. What's behind it is this spec's business.
2. **Retention is a policy, sharing is by content.** Whether a layer keeps all tokens or a window is a per-layer policy. Whether a block can be shared across sequences depends only on its content hash. The two never interact.
3. **Rollback is cheap by construction.** For paged KV, rejecting draft tokens is a counter decrement. For recurrent state, it is a buffer swap. Neither moves data on the common path.
4. **Write and read are both ours.** Nothing outside the engine ever reads state bytes, so the intra-block element order is whatever the arch's attention kernel wants (a `LayoutId` from the registry), not a portable format. Host swap copies bytes verbatim.
5. **One manager, replicated views.** The manager runs once on the host. Every device gets the same block tables in `BatchMeta`; under head-sharded TP each device holds its head slice of every block, under PP each device holds its layer range.
6. **Latency first.** Running decodes are never preempted or swapped. Admission waits.

## 2. State kinds

A model definition (spec 8) declares, per layer, a list of `StateSpec`:

```
KvPaged   { hkv: u32, d: u32, dv: u32, cache: CacheDtype, retain: Retain }
KvLatent  { latent: u32, rope: u32, cache: CacheDtype, retain: Retain }   # MLA
Recurrent { h: u32, d: u32, dv: u32 }                                     # f32 [h, d, dv]
ConvWindow{ c: u32, w: u32 }                                              # f16 [w-1, c]

CacheDtype = E4M3 { scale: PerTokenHead } | I8 { scale: PerTokenHead } | F16
Retain     = All | Window(w) | Sink(n) + Window(w)
```

A hybrid layer (Qwen3.8-Next style) declares `[ConvWindow, Recurrent]`. A dense layer declares `[KvPaged]`. Layers with no state (pure MLP) declare `[]`.

Defaults: `cache = E4M3 { PerTokenHead }` for both `KvPaged` and the latent part of `KvLatent`; the rope part of `KvLatent` is always `F16` (it is small and precision-sensitive). Config may force `F16` or `I8` per model.

## 3. Paged KV (`KvPaged`, `KvLatent`)

### 3.1 Blocks

- **Block size: 32 tokens.** One lane per token in a wave32 QK pass, two 16-wide WMMA K-tiles, and a prefix-cache granularity that still catches most shared prompts.
- A **block** is the unit of allocation, sharing and eviction. It holds, for one layer, all local KV heads for 32 consecutive positions of one sequence.
- **Pool**: one contiguous region per layer-group (§6) inside the device arena (spec 9 §4.1), divided into fixed-size blocks. Block `b` of layer `l` lives at a computable offset; no per-block pointers on device.

### 3.2 Block contents (`KvPaged`)

Per block, per local KV head:

| region | shape | dtype | bytes (E4M3, d=128) |
|---|---|---|---|
| K values | `[32, d]` | cache dtype | 4096 |
| V values | `[32, dv]` | cache dtype | 4096 |
| K scales | `[32]` | f16 | 64 |
| V scales | `[32]` | f16 | 64 |

Intra-region element order is `arch.attention_layout` (spec 4), chosen so the attention kernel's K and V fragments are contiguous loads. `state_write_kv` writes in that order; nothing else touches it.

`KvLatent`: one region `[32, latent]` in cache dtype with `[32]` f16 scales, plus `[32, rope]` f16. No head dimension.

### 3.3 Slots and block tables

- A sequence's state is a list of block ids per layer-group, plus `ctx_len` (verified tokens) and `tail_len` (tokens written this step but not yet committed).
- `slot(s, p) = (block_table[g][s][p / 32], p % 32)` for layer group `g`. `BatchMeta.slot_map` is `[G, T]` (one flattened slot per new token per group) and `BatchMeta.block_table` is `[G, S, max_blocks]` where `max_blocks = ceil(state.max_ctx / 32)`, padded with a sentinel. `BatchMeta.window_start` is `[G, S]` (§3.5).
- Block tables are identical across layers that share a `StateSpec`, so the manager emits one table per **layer-group** (§6.1) and the graph binds each layer to its group's table. `G` is fixed per model, which keeps `BatchMeta` fixed-shape.

### 3.4 Prefix cache

- Every full block is content-addressed: `hash(b) = xxh3(hash(prev) ‖ token_ids[32] ‖ layer_group ‖ cache_dtype)`. The tail (partial) block is private and unhashed until full.
- A hash map from `hash → block_id` with a refcount per block. A new sequence's prompt is matched block by block from the start; matched blocks are shared by incrementing refcount, the rest are allocated fresh and the scheduler is told the matched length so prefill starts there.
- Blocks are immutable once full, so no copy-on-write is needed. Refcount 0 blocks go to an LRU free list and are reclaimed only when the free pool is empty; until then they are cache hits waiting to happen.
- **Retention does not affect sharing.** A `Window(w)` layer group hashes and shares blocks the same way; it just releases its reference to blocks older than the window. Two sequences with the same prefix share window-layer blocks for positions both still retain.
- Prefix matching is by token id, so it is tokenizer-exact and chat-template-exact. The cache is keyed additionally by `model_fp` (spec 9 §3) so two models never collide.

### 3.5 Retention policies

- `All`: blocks retained until the sequence is freed.
- `Window(w)`: after commit, blocks whose newest position is `< ctx_len − w` have their reference released. The block table for the group is a ring of `ceil(w/32) + 1` entries; the attention kernel is told `window_start` and skips earlier slots.
- `Sink(n) + Window(w)`: the first `ceil(n/32)` blocks are pinned in addition to the window. The kernel receives both ranges.

Positions are absolute; RoPE was applied at write time, so retained blocks are valid regardless of what was released around them.

### 3.6 Append, commit, rollback

Per step, per sequence:

1. **`reserve(s, n)`** — ensure blocks exist for positions `ctx_len .. ctx_len + n − 1`, allocating from the free pool as needed; return `n` slots. Fails if the pool cannot cover it (the scheduler then doesn't admit that work this step).
2. The graph runs. `state_write_kv` writes all `n` tokens (draft candidates included) into their slots. `tail_len = n`.
3. **`commit(s, a)`** with `a ≤ n` — `ctx_len += a`; `tail_len = 0`. Positions `ctx_len .. ctx_len + n − a − 1` simply remain allocated and will be overwritten by the next step's reserve. Nothing is copied. Blocks that became full during commit are hashed and inserted into the prefix cache.

That is the entire rollback mechanism for paged KV: rejection is `commit` with a smaller `a`.

**Tree verify**: the scheduler assigns the `T` candidate tokens linear scratch positions `ctx_len .. ctx_len + T − 1` for writing; the attention kernel uses `BatchMeta.tree` for masking, not positions. After verify, the accepted path (`a` tokens) is generally not the first `a` scratch positions, so the manager issues `compact(s, accepted_positions[])`, a tiny kernel that copies the accepted tokens' K/V (and scales) into positions `ctx_len .. ctx_len + a − 1` within the same blocks, then commits. Cost is at most `a` token copies per layer; it runs only on tree verify.

### 3.7 Host swap (prefix cache only)

When a device's free pool is exhausted and refcount-0 blocks exist, the manager may evict them to a host-side block store (verbatim bytes, tagged with layout id and arch) instead of discarding, up to a configured host budget. A future prefix hit against a host block is a H2D copy before prefill starts and counts as a hit for scheduling. Running sequences are never swapped.

## 4. Recurrent and conv state

### 4.1 Storage

- `Recurrent`: per (sequence, layer, head) a `[d, dv]` f32 matrix. Allocated as a contiguous slot per sequence per layer-group: `[h, d, dv]` f32, 64 KB per head at `d = dv = 128`.
- `ConvWindow`: per (sequence, layer) `[w − 1, c]` f16.
- Slots come from a separate fixed pool sized by `max_seqs`. There is no paging: the state is fixed-size per sequence and lives for the sequence's lifetime.

### 4.2 Double buffering (checkpoint and rollback)

Each sequence has two slots per recurrent/conv layer-group: **A** (verified) and **B** (working).

- Plain decode (`query_len = 1`): read A, write B, `commit` swaps A↔B. No copy.
- Spec verify (`query_len = k + 1`): read A, write B (the scan writes state for all `k + 1` tokens). `commit(a)`:
  - `a = k + 1`: swap A↔B.
  - `a < k + 1`: the scheduler re-runs the accepted prefix (`a` tokens) through the recurrent form reading A and writing B, then swaps. Bounded by `k ≤ 16` tokens per layer; budgeted as part of the step in spec 6.
- Tree verify: same as above with the accepted path as the prefix.

Because A is never written during a step, there is no separate "checkpoint" call: A **is** the checkpoint.

### 4.3 Prefix reuse for recurrent state

Recurrent state is not content-addressed per block; snapshots are large and the state at an arbitrary block boundary is rarely reused. The manager keeps a **session cache**: on `free_seq`, the final A slot may be retained (verbatim) under the hash of the full token sequence, within a small budget (default 2 sessions per GB of pool). A new sequence whose prompt exactly equals a cached session's tokens starts from that state with `ctx_len` set accordingly; hybrid models get multi-turn continuation for free even though they don't get mid-prompt sharing. Attention layers in the same model still use the paged prefix cache normally, so a session hit on a hybrid model is a hit for every layer.

## 5. Manager API

Host-side, single instance per engine, called only by the scheduler (spec 6). All calls are synchronous bookkeeping; device work is issued by the scheduler as ops.

```
new_seq(tokens: &[u32]) -> (SeqId, matched_len)      # prefix / session lookup included
reserve(seq, n) -> Result<SlotRange>                  # may fail: budget
batch_meta(seqs, query_lens, tree?) -> BatchMeta      # fills slot_map, block_table per group, positions
commit(seq, accepted: u32)
compact(seq, accepted_positions: &[u32]) -> CompactOp # tree verify only; returns the op to enqueue
free_seq(seq)                                          # releases refs; may retain session state
budget() -> { free_blocks, free_slots, host_free }
stats() -> { prefix_hit_rate, evictions, swaps, utilization }
```

Deterministic given the call sequence: block ids and slot ids are a function of the request history alone, so two runs with the same requests produce identical `BatchMeta`, which the doctor bundle can diff.

## 6. Layer groups and budgeting

### 6.1 Layer groups

Layers are grouped by identical `StateSpec` (kind, dims, cache dtype, retention). A typical dense model has one group; a local/global model has two; a hybrid has two or three. Each group has its own pool and its own block table in `BatchMeta`. Group count is small and fixed per model, so `BatchMeta` stays fixed-shape.

### 6.2 Per-token cost

For a `KvPaged` group with `L` layers, per token:

```
bytes = L · hkv_local · ( (d + dv) · cache_bytes + 4 )      # +4 for two f16 scales
```

Example, 30B dense, `L = 48`, `hkv = 8`, `d = dv = 128`, E4M3: `48 · 8 · (256 + 4) = 99,840 B ≈ 97.5 KB/token`, so 128K of context is ≈ 12.2 GB. F16 doubles it.

For a `Recurrent` group: `L · h · d · dv · 4 · 2` (double buffered) per sequence, not per token. Qwen3.8-Next-like, 12 linear layers × 16 heads × 128 × 128 × 4 × 2 ≈ 25 MB per sequence.

### 6.3 Budget

```
pool = vram − weights_resident − activation_workspace(max bucket) − reserve
```

`reserve` (`state.reserve_bytes`) defaults to 512 MB. The pool is split across groups in proportion to their per-token cost times `state.max_ctx`, with recurrent/conv pools sized by `state.max_seqs`. Both are config (§9); if the pool cannot satisfy both, the loader refuses with the numbers rather than silently lowering either.

Admission rule the scheduler follows: a chunk of prefill or a decode step is admitted only if `reserve` succeeds for every sequence in it. Because decode reserves at most `k + 1` tokens per sequence per step and blocks are 32 tokens, a running decode fails to reserve only when the pool is genuinely full, at which point new prompts have already stopped being admitted. Running decodes therefore never stall on state in practice.

## 7. Multi-GPU

- **TP (head-sharded)**: each rank's pool holds `hkv / ranks` heads per block. Block ids and tables are identical on all ranks; only the per-block bytes differ. Recurrent slots are head-sharded the same way.
- **PP**: each rank holds pools only for its layer range. Block tables are still emitted for all groups; unused ones are ignored on a rank.
- **EP**: no effect on state.
- Prefix cache hashing is independent of the parallel layout, so a cache built under TP2 is not reusable under TP4 (block bytes differ). The cache is keyed by `(model_fp, plan)`.

## 8. Determinism and validation

- Block allocation, eviction and slot assignment are deterministic functions of the request history (§5). No timing-dependent decisions.
- `commit` semantics are tested by writing `k + 1` tokens, committing `a`, re-reading positions `< ctx_len`, and requiring bit-equality with a sequence that never speculated.
- Double-buffer correctness: after `commit(a < k+1)` and recompute, the A slot must bit-equal a sequence that processed the same `a` tokens without speculation.
- Prefix cache correctness: a sequence that hits the cache must produce logits bit-identical to one that prefills from scratch (same arch, same kernel hash).
- The doctor bundle records pool sizes, group layout, block size, cache dtypes, and the per-step `BatchMeta` for any reported issue.

## 9. Config

```
[state]
max_ctx            = 32768        # tokens per sequence; multiple of 32
max_seqs           = 8
cache_dtype        = "e4m3" | "i8" | "f16"
reserve_bytes      = "512MB"
host_block_budget  = 0            # §3.7; 0 disables host swap of prefix-cache blocks
session_cache      = 2            # §4.3; retained sessions per GB of recurrent pool
```

`max_ctx`, `max_seqs`, `cache_dtype`, `reserve_bytes` and `host_block_budget` size arenas and require a reload; `session_cache` is runtime-mutable (spec 12 §3).
