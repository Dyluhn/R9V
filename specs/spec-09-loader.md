# Spec 9 — Loader

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 2–8. Constrains: specs 11, 12, 14.

## 0. Purpose and scope

Everything between "here is a path" and "the engine is ready to schedule": parsing and fingerprinting, binding to a model definition, planning and budgeting, allocating every byte the engine will ever use, moving weights from disk to their placement (with repack or zero-copy), the caches that back host and SSD tiers, secondary loads (draft, eagle), warmup, and the load report. Also unload.

Out of scope: the repack rules themselves (spec 2 §7), the plan (spec 5), what warmup captures (spec 6 §5.1).

## 1. Principles

1. **Allocate once.** Every device and pinned buffer the engine uses is allocated at load. There is no `hipMalloc` on the step path, and the load fails with numbers rather than succeeding into an OOM later.
2. **Second load is the fast load.** A standard GGUF repacks on first load and writes a native cache beside it; every later load is the zero-copy path. Autotune and capture results persist the same way.
3. **Bytes move once.** Disk → pinned → device, with repack (if any) applied on the way through pinned memory. Nothing is loaded to device and rearranged there.
4. **Weights don't move after load.** Placement, hot sets and arenas are fixed for the model's lifetime (spec 5 §5.4). The only runtime traffic is the tiered row cache and the host-expert hidden states.
5. **Everything about the load is reportable.** The load report is a complete account of what ended up where, at which tier, from which source, and it is the first thing in every doctor bundle.

## 2. Pipeline

```
1. open       parse GGUF header(s) (split shards supported), read metadata, fingerprint
2. bind       resolve family (spec 8 §4), build ModelSpec, validate (spec 8 §6)
3. plan       ModelSummary → planner (spec 5 §5) → Plan; secondary models get their own plan
4. budget     per device and host: weights + state pools + workspace + comms + reserve; refuse with shortfall
5. allocate   device arenas, state pools, workspace, comms buffers, pinned staging, pinned host tensors, tiered slab
6. materialize for each tensor in graph order: read → (repack | passthrough) → destination
7. tokenizer  build tokenizer and chat template from metadata
8. secondary  load draft / eagle models through steps 1–7 on their rank
9. warmup     registry resolution, autotune, validation, capture, replay-mode choice, budget resolution
10. ready     emit load report; hand the scheduler its graphs, state manager and plan
```

Steps 1–4 touch no weight data and finish in well under a second, so an infeasible load is rejected before any I/O. Progress is reported per step and per tensor for step 6 (spec 10 exposes it).

## 3. Fingerprint

```
file_fp   = xxh3(header bytes ‖ tensor-info table ‖ metadata KV bytes ‖ file size ‖ shard count)
model_fp  = xxh3(file_fp ‖ every r9v.tensor.*.xxh3 or, for standard GGUF, per-tensor xxh3 computed during repack)
```

`file_fp` is cheap (metadata only) and keys the repack cache lookup. `model_fp` is complete and keys prefix caches, session caches, tune baselines and benchmark receipts. For a standard GGUF the per-tensor hashes are computed once during the first repack and stored in the cache manifest, so `model_fp` never requires re-reading the source file.

## 4. Budget and arenas

### 4.1 Per device

```
arena = [ weights (graph consumption order) ]
        [ state pools, one per layer group (spec 3 §6) ]
        [ workspace (spec 6 §5.3) ]
        [ comms buffers (spec 5 §6.1) ]
        [ reserve (spec 3 §6.3, default 512 MB) ]
```

One `hipMalloc` per device. Tensors are placed at 256-byte-aligned offsets; regions the direct-IO path targets are 4 KiB aligned. The weight region is laid out in the order the step graph consumes tensors, which is also the native file's tensor order, so a native load is a sequential sweep of the file into a sequential sweep of the arena.

### 4.2 Host

```
pinned = [ staging ring for I/O (io.chunk_mb × io.queue_depth) ]
         [ host-resident tensors (cold experts, host n-gram tables, host embeddings) in their spec 2 layout ]
         [ tiered slab (§6) ]
         [ per-step buffers: gather_staging, comms bounce, readback ]
```

`host.pinned_budget` defaults to `min(free_ram − 4 GB, need)`; if `need` exceeds it the load refuses with the shortfall and the config line that would fix it. Pinned memory is registered once (`hipHostMalloc`) and never reallocated.

### 4.3 Refusal

A budget failure reports, per device and for host: required, available, shortfall, and the largest contributors (top five tensors or pools). It suggests the smallest single change that would fit (`state.max_ctx`, `state.max_seqs`, `experts.hot_set_vram`, or a smaller quant) with the resulting numbers. It never silently lowers a setting.

Under a spoof-constrained plan (spec 1 App. A), per-device budgets run against the `EffectiveDeviceView` VRAM bound, not physical VRAM: a plan that fits the physical card but exceeds the spoof target refuses here, before I/O, with the spoof shortfall. Before the first device allocation, the loader also creates exactly one shared `r9v_hip::AllocationBudget` per constrained physical identity with that same byte limit; every engine-owned allocation for that identity uses `BudgetedDeviceBuffer`. Its atomic reservation happens before `hipMalloc`, rolls back on HIP failure, and releases only after `hipFree`, so concurrent allocations cannot exceed the spoof cap. Direct unbudgeted `DeviceBuffer` allocation is forbidden on a constrained execution path. The CU mask independently narrows CU visibility and enforces nothing about allocation.

## 5. Materialization

### 5.1 I/O

- **Direct I/O** (`O_DIRECT`, Linux) is the default: 16 MB chunks, queue depth 8, into the pinned staging ring. Regions are 4 KiB aligned by the spec 2 container rules, and split shards are treated as one logical file. Target: NVMe line rate (≥ 5 GB/s on Gen4).
- **mmap** is the fallback when `O_DIRECT` is refused (network filesystems, unusual block sizes) and when `io.mode = mmap`. It uses `madvise(SEQUENTIAL)` and the same chunked H2D.
- H2D copies run on the copy stream, overlapped with the next chunk's read. For native zero-copy tensors the CPU never touches the bytes.

### 5.2 Repack path (standard GGUF)

For each tensor whose `(source type, target scheme, layout, fusion)` is not in the repack cache:

1. Read the source tensor through the staging ring.
2. A thread pool (default `cores − 2`) applies the spec 2 §7 permutation into a second pinned buffer, tile by tile; workers are independent per row-block so the pool scales linearly.
3. The repacked bytes go H2D (or to their host/tiered destination) **and** are appended to the repack cache file.
4. Per-tensor `xxh3` of the result is computed on the way through and stored in the cache manifest.

Throughput target: ≥ 2 GB/s per core, so an 8-thread repack is I/O-bound on any NVMe. A 25 GB Q4_K/Q8_0 file repacks in 10–20 s on first load.

### 5.3 Repack cache

```
<model>.r9v-cache/
  manifest.json      # source file_fp, gen-independent; per tensor: scheme, layout, fusion, xxh3, offsets
  weights.gguf       # native-format GGUF (spec 2 §6) containing every repacked tensor
```

- Keyed by `(file_fp, target layout, fusion declarations, scheme mapping version)`. A different arch with a different `fragment_layout` gets a separate cache entry; the common case (same machine) hits every time.
- `cache_dir` config redirects it when the model directory is read-only. Size is roughly the source size; the loader logs it and never deletes it.
- On hit, the source GGUF is opened only for metadata; all weight bytes come from `weights.gguf` via the zero-copy path.

### 5.4 Destinations

| placement | destination | layout |
|---|---|---|
| `Device(r)` | device `r` arena | spec 2 |
| `Host` | pinned host-tensor region | spec 2 (same bytes as a device copy would have) |
| `Tiered` | not loaded; registered with the slab (§6) by residency unit and file offset | spec 2 |

Hot-set experts (spec 5 §3.4) are `Device`; cold experts are `Host` (for `host_compute`) or `Tiered` (for `host_fetch`). N-gram tables default to `Host` when they fit the pinned budget and `Tiered` otherwise.

## 6. Tiered slab (row cache)

A fixed-size pinned slab holding residency units (rows for `L0` tables, experts for stacked expert tensors) read on demand from the file by direct I/O.

```
ensure(units: &[UnitId]) -> Future<()>      # issue reads for absent units; returns when resident
pin(units) / unpin(units)                    # protect a step's working set from eviction
addr(unit) -> pinned pointer                 # valid while pinned
stats() -> { hit_rate, evictions, bytes_read }
```

- **Eviction**: clock (second-chance) over slots; pinned slots are skipped. No LRU list maintenance on the hot path.
- **Prefetch**: the scheduler's pre-step phase calls `ensure` for every n-gram row the step will touch (exact, from token ids) and, in `host_fetch` mode, for experts as routing becomes known per segment. Reads are issued at queue depth 8; a step whose units aren't resident by graph start waits, and the wait is logged as a slab miss.
- **Slab size**: `tiered.slab_bytes`, default `min(25% of pinned_budget, total tiered bytes)`. The load report shows expected hit rate from calibration data (spec 2 metadata) when present.
- Units are read as whole spec 2 regions (row + scales, or a full expert), so a resident unit is directly usable in its layout without copying.

## 7. Tokenizer and chat template

Built from `tokenizer.ggml.*` metadata (model type, tokens, scores, merges, special tokens, add-BOS flag) and `tokenizer.chat_template`. The tokenizer is the same implementation the helper and the quant tool use, so prefix-cache hashes (spec 3 §3.4) and calibration tokenization agree byte-for-byte. `eos_ids` and `bos_id` are handed to the scheduler from `ModelSpec` (spec 8).

## 8. Secondary models

Draft and eagle models (spec 7 §7) run steps 1–7 on the rank the plan assigns (rank 0 or the last PP stage), with their own arena, state pools and repack cache. Their VRAM is subtracted before the target's state pools are sized, so `max_ctx` in the load report is the real number. The helper model (spec 12) is loaded through the same pipeline onto the CPU device only when no target model is loaded, and unloaded before a target load begins (spec 6's ordering rule: release before load, never after).

## 9. Warmup

Order, each timed and reported:

1. **Resolve** every op instance in every warm bucket against the registry (spec 4 §9). Missing T2 variants with JIT available → autotune now, in parallel across variants; missing without JIT → T1 with a warning.
2. **Validate** any newly tuned variants (spec 4 §9.3).
3. **Capture** graphs for `warm_buckets` (spec 6 §5.1) as launch lists, then as `hipGraph` where supported.
4. **Choose replay mode** by timing both on the `(1, 1, 0)` bucket (spec 6 §5.2).
5. **Measure** `C(S, T_dec, T_pre)` for the warm buckets on the real plan, and `C_draft(k)` for `k ≤ k_max` when the proposer has a device pass; write the step cost table the scheduler uses and resolve `step_budget_ms = auto`.
6. **Initialize** the state manager with the pools and hand everything to the scheduler.

Target: under 30 s on a warm cache for a 30B model; first load on a new machine is dominated by autotune (minutes, reported as such). `warmup.enabled = false` skips 3–5 for development, in which case every bucket is captured lazily and the budget uses the tune-file estimate.

## 10. Load report

Written to the log and stored for the doctor bundle:

- source: path(s), `file_fp`, `model_fp`, format (native / standard GGUF via cache / standard GGUF first load), shard count
- model: family, `general.architecture`, layer classes, vocab, tied, mtp present, export_hidden
- plan: strategy, stages, TP degree, expert map summary, transport per link, `expected` costs, provenance (`Physical` or the qualified `MODEL (SPOOF)` target plus the physical identity)
- per device: arena size and map (weights / state pools per group / workspace / comms / reserve), `max_ctx` and `max_seqs` as actually provisioned
- host: pinned usage by category, slab size, expected slab hit rate
- per tensor (collapsed by layer in the summary, full in the bundle): source type → scheme, layout, zero-copy or permuted, placement, resolved tier (T0v / T1 / T2), bytes
- tune coverage: shipped / local / partial / T1 counts per op
- warmup timings, replay mode chosen, measured `C(S, T_dec, T_pre)` and `C_draft(k)` tables, resolved `step_budget_ms`
- warnings: unused tensors, replicated KV heads, forced T1 fallbacks, missing ROCm compiler

## 11. Unload and reload

`unload()` frees arenas, pools, pinned regions and graphs, drops secondary models, and leaves the repack cache, tune files and capture metadata in place. Reload of the same `model_fp` on the same plan skips autotune entirely and captures from the recorded launch lists; the fault path in spec 6 §8 relies on this being fast. Loading a different model while one is loaded is unload-then-load; concurrent target models in one process are not in v1.

## 12. Failures

| failure | behavior |
|---|---|
| unknown `general.architecture` | error naming it and the nearest family |
| missing / mis-shaped tensors | error listing all of them (spec 8 §6) |
| checksum mismatch | error naming the tensor; suggests deleting the repack cache if the source verifies |
| budget shortfall | refusal with numbers and a suggested change (§4.3) |
| `O_DIRECT` refused | fallback to mmap, logged |
| unshipped variant, no compiler | T1 with warning; error if `require_fast_path = true` |
| short read / I/O error | error with tensor name and file offset; no partial engine state survives |

## 13. Config

```
[load]
model            = path
draft_model      = path          # spec 7
eagle_head       = path
cache_dir        = auto          # beside the model by default
require_fast_path = false
[io]
mode             = "auto" | "direct" | "mmap"
chunk_mb         = 16
queue_depth      = 8
repack_threads   = auto
[host]
pinned_budget    = auto
[tiered]
slab_bytes       = auto
[warmup]
enabled          = true
buckets          = (spec 6 warm_buckets)
```
