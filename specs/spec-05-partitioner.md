# Spec 5 — Sharding and Partitioner

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 1–4. Constrains: specs 6, 9, 11, 12.

## 0. Purpose and scope

How a single-device graph becomes a multi-device execution: the topology model, the parallel strategies, the partitioner that rewrites the graph using each op's sharding table, the placement planner that picks a strategy at load time, and the comms transport under the collective ops. Also what "correct" means across device counts, and the config surface.

Out of scope: which sequences run in a step (spec 6), the row cache that backs tiered tensors (spec 9), kernel-level reductions (spec 4 §5.9).

## 1. Principles

1. **The partitioner only applies tables.** Every sharding decision is a lookup in an op's spec 1 table plus a collective inserted where layouts mismatch. It contains no model knowledge.
2. **Placement is decided at load, not per step.** Changing TP degree or PP boundaries moves weights; weights don't move at runtime. The planner chooses one placement for the configured workload profile, and per-step variation is limited to things that don't move bytes.
3. **Latency profile favors one hop per token.** On 2–4 consumer cards over PCIe, pipeline parallel costs one transfer per token per step; tensor parallel costs two collectives per layer. The default plan for `profile = latency` is PP unless VRAM or an explicit config says otherwise.
4. **Host experts are computed on the host.** At low batch, moving an expert's weights to the GPU costs 100× more than moving the hidden state to the CPU. A `tiered` placement hint on experts resolves to CPU compute for the cold set (`Host`, spec 2 §5); fetching weights to device (`Tiered`, host_fetch) is a throughput-mode option only.
5. **P2P is measured, never assumed.** Every link is `Direct` or `HostStaged` per the doctor's measurement; the collective ops are identical either way.
6. **Replicated tensors are bit-identical across ranks.** Every reduction is computed in the same fixed order on every rank. Divergence between ranks is a bug class this spec is designed to make impossible.

## 2. Topology

```
Topology {
  devices: [ { rank, arch: ArchDescriptor, vram_free: u64 } ]
  links:   [ (a, b, transport: Direct | HostStaged, gbps: f32, latency_us: f32) ]   # measured
  host:    { cores, simd: AVX2 | AVX512 | AVX512_VNNI | AMX, mem_gbps: f32, pinned_budget: u64 }
}
```

Populated by the doctor's measurement pass (spec 11) and cached by hardware fingerprint. On the current rig: two gfx1201, one on x16 and one on x4; the x4 link sets the PCIe cost for every plan, and P2P is `HostStaged` until measured otherwise.

## 3. Strategies

| strategy | what is sharded | comms per layer per step | when |
|---|---|---|---|
| **PP** | layers by range | one `send/recv` of `[T, Dm]` at each stage boundary (per step, not per layer) | default for `latency`; any model that doesn't fit one device |
| **TP** | attention heads; MLP columns/rows | 2 × `all_reduce` of `[T, Dm]` | `throughput`, or a single layer too large for one device |
| **EP** | experts by device | `all_to_all` dispatch + combine of routed tokens | `throughput` on MoE; never at `latency` |
| **PP + TP** | PP stages, TP inside a stage | both | 4+ devices, large dense models |
| **Host experts** | cold experts on CPU | hidden state `[T_routed, Dm]` D2H and H2D per MoE layer | MoE that doesn't fit VRAM, any profile |
| **Replicas** | nothing; independent engines | none | small models, `throughput`; handled by spec 10, not here |

### 3.1 PP

- Stage boundaries chosen to balance **estimated step time**, not layer count: per-layer decode cost is proportional to weight bytes resident on device (bandwidth-bound) plus, for MoE, the host-compute share. Embedding and `lm_head` are charged to their stages.
- Each stage owns state pools only for its layers (spec 3 §7).
- Micro-batching (`pp_microbatches ∈ {1, 2}`) splits a batch so both stages work concurrently; `latency` uses 1, `throughput` uses 2 when `S ≥ 2`.
- Sampling runs on the last stage; the sampled tokens are sent back to stage 0 as part of the next step's `token_ids` (one small transfer).

### 3.2 TP

Canonical per-layer pattern, from the spec 1 tables:

```
qkv:   x Replicated × w ColShard(heads)   → q,k,v HeadShard
attn:  HeadShard → HeadShard                (state head-sharded, spec 3 §7)
o:     HeadShard × w RowShard              → Partial
residual + all_reduce                       → Replicated        (§4.2 residual trick)
norm:  Replicated
gate/up: Replicated × ColShard             → ColShard
act_mul: ColShard
down:  ColShard × RowShard                 → Partial
residual + all_reduce                       → Replicated
```

Two `all_reduce` per layer. `hkv % ranks == 0` is required; models where it isn't can replicate KV heads (`hkv_local = hkv`) at a small VRAM cost, which the partitioner does automatically and logs. `lm_head` is `ColShard(V)` followed by `all_gather` of logits (spec 1 §4.F).

### 3.3 EP

- Expert placement map `expert → rank`. Routing runs replicated; the `all_to_all` dispatch carries each token's hidden state to the ranks holding its `K` experts; `moe_ffn` runs on local experts; a second `all_to_all` returns partial outputs, combined in f32 in expert-index order.
- Buffers are fixed-size per bucket at worst case (`T · K` rows per peer); valid counts travel in a small header exchanged inside the collective, so the captured graph never changes shape.
- Shared experts, if any, run replicated as a plain `matmul`.

### 3.4 Host experts

- The `tiered` placement hint on expert tensors (spec 2 §4) resolves at load into a **hot set** (`Device`, chosen by calibration frequency from the quant tool or by a warm-up pass) and a **cold set** (`Host`: `L1` layout in pinned memory, computed by T0v; or `Tiered`: slab-backed, fetched to device in `host_fetch` mode). The hot set is fixed for the model's lifetime (spec 9 §5.4).
- Per MoE layer, the graph is captured as a segment boundary: router runs on device → expert ids and the routed hidden rows for cold experts are copied D2H → the CPU tier (spec 4 §2, T0v) computes those experts' outputs → results H2D → combine. Hot experts run on device concurrently. The scheduler (spec 6) owns the segment sequencing; this spec only fixes that the segment exists.
- Cost per layer at batch 1: ≈ 2 transfers of `Dm × 2 B` (≈ 16 KB) plus CPU GEMV over `K_cold` experts. With AVX-512 VNNI, a 4096 × 1536 int4 expert is ~1 ms on 8 cores; the design assumes ≤ 2 cold experts per token per layer on average for the latency profile to hold. The planner reports the expected cold rate from calibration data before load.
- `HostFetch` (pull cold expert weights to device instead of computing on host) is available for `throughput` when `T ≥ 64`, where the fetch amortizes across tokens. It uses the spec 9 row cache with `residency_unit = expert`.

## 4. Partitioner

### 4.1 Algorithm

Input: the single-device graph, a `Plan` (§5), and the spec 1 sharding tables.

1. **Assign weights.** Every weight tensor gets a `ShardLayout` and a rank set from the plan (PP: rank by layer; TP: `ColShard`/`RowShard`/`HeadShard` by role; EP: `ExpertShard`).
2. **Propagate.** Walk the graph in topological order; for each op pick the table row whose input layouts match; if none matches, insert the cheapest collective that makes one match (`all_reduce` for `Partial → Replicated`, `all_gather` for `ColShard → Replicated`, `reduce_scatter` for `Partial → RowShard`).
3. **Cut.** At PP boundaries insert `send`/`recv` of every live tensor crossing the cut (normally just the residual stream) and split the graph into per-rank subgraphs.
4. **Coalesce.** Apply §4.2 and merge adjacent collectives on the same tensor.
5. **Emit** one graph per rank per bucket, with the collectives' peer sets and transports filled in from the topology.

The partitioner is a pure function of `(graph, plan, topology)` and its output is diffable; the doctor bundle stores the emitted per-rank graph summaries.

### 4.2 Residual trick

`residual_add(Partial, Replicated)` is rewritten so the residual is added on rank 0 only (other ranks add zero), producing a `Partial` whose single `all_reduce` yields the correct replicated sum. This folds the residual into the existing reduce and removes a separate replicate step. It is numerically identical to reduce-then-add because addition order inside the reduce is fixed.

### 4.3 What the partitioner refuses

- A model whose sharding tables can't be satisfied with the plan's device count (e.g. `hkv = 1` under TP without replication allowed).
- A plan that doesn't fit: weights + state pools + workspace per rank exceed `vram_free`. The error lists the per-rank shortfall.

## 5. Planner

### 5.1 Inputs and output

```
plan(model: ModelSummary, topology, config) -> Plan
Plan {
  strategy:        PP | TP | EP | PP+TP
  stages:          [ { rank_set, layer_range } ]
  tp_degree:       u32
  expert_map:      { expert → (rank, Device | HostCompute | HostFetch) }
  transport:       { link → Direct | HostStaged }
  pp_microbatches: u32
  expected:        { step_us_by_bucket: {bucket → f32}, cold_expert_rate: f32 }
}
```

`ModelSummary` comes from the model definition (spec 8): per-layer weight bytes per scheme, state cost per token, expert count and sizes, `hkv`, `V`, `Dm`.

### 5.2 Selection

1. Enumerate candidate plans for the device set: PP with each balanced boundary set; TP at each degree dividing `hkv`; EP for MoE; PP+TP for 4+ devices; host-expert variants when experts exceed VRAM.
2. Drop infeasible plans (§4.3).
3. Score each with the cost model for the profile's target buckets: `latency` scores `T ∈ {1, 2, 4}` with weights `{0.6, 0.3, 0.1}`; `throughput` scores `T ∈ {32, 128, 512}` equally.
4. Pick the minimum. Ties go to fewer collectives.

Cost model per step for a bucket: Σ over ranks on the critical path of `weight_bytes / mem_bw` + `activation_flops / rate` + Σ collectives `(latency_us + bytes / gbps)` × (2 if `HostStaged`) + host expert compute. Coefficients come from the tune files and the topology measurements, so the model is calibrated per machine.

### 5.3 Overrides

Config may pin any field of the plan. A pinned plan that is infeasible fails at load with the shortfall; a feasible but slower pinned plan loads and the log states the estimated cost versus the auto choice.

### 5.4 Stability

The plan is computed once per `(model, topology fingerprint, config)` and cached. It never changes while a model is loaded. If a device disappears, the engine stops rather than replanning under load.

## 6. Comms transport

### 6.1 Interface (used only by collective ops)

```
send(buf, peer, stream)         recv(buf, peer, stream)
all_reduce(buf, group, stream)  all_gather(bufs, group, stream)
reduce_scatter(...)              all_to_all(bufs, counts_hdr, group, stream)
barrier(group)
```

All buffers are fixed-size per bucket and pre-registered (pinned for `HostStaged`, peer-mapped for `Direct`). No allocation on the step path.

### 6.2 Implementation

v1 is an in-process implementation over HIP streams and events for 2–4 devices in one process. Not RCCL: it adds a large dependency with inconsistent behavior on consumer RDNA, and at the message sizes this engine moves (`[T, Dm]` at small `T` is tens of KB) a direct exchange beats ring algorithms. An RCCL backend can be added behind the same interface for larger nodes.

- `Direct`: `hipMemcpyPeerAsync` or peer-mapped loads, whichever the doctor measured faster for the size class.
- `HostStaged`: D2H into a pinned bounce buffer, H2D to the peer; two copies, both on the sender's stream, with an event the receiver waits on.
- `all_reduce` for `ranks ≤ 4`: every rank sends its buffer to every other rank; each rank adds all buffers in ascending rank order (spec 4 §5.9). Same bits on every rank. For `ranks > 4` a reduce-scatter/all-gather pair with the same fixed order.
- `all_to_all`: a 64-byte count header per peer travels first; payload copies are fixed-size; the receiver reads only `counts` rows.

### 6.3 Streams

Each rank has a compute stream and a comms stream. Collectives are enqueued on the comms stream with events on both sides. With `pp_microbatches = 2`, stage `i`'s send of micro-batch 0 overlaps its compute of micro-batch 1. At `latency` there is nothing to overlap and the collective sits on the critical path; the cost model charges it fully.

## 7. Correctness across device counts

- **PP vs single device: bit-identical (L0).** No arithmetic changes; only where it runs.
- **TP or EP vs single device: within tolerance (L1).** Splitting K or experts across ranks changes reduction order. Run-to-run on the same plan is L0.
- **Across ranks under TP: bit-identical replicated tensors (L0).** Enforced by §6.2 ordering and tested by hashing every replicated tensor on every rank after each layer in debug mode.
- Batch invariance (spec 1 §6.1) holds under every plan because collectives operate on fixed-size buffers with masked padding and never reorder rows.

## 8. Config surface

```
[parallel]
profile        = "latency" | "throughput"           # default latency
devices        = [0, 1]
mode           = "auto" | "pp" | "tp" | "ep" | "pp+tp"
tp_degree      = auto | n
pp_stages      = auto | [[0..23], [24..47]]
pp_microbatches = auto | 1 | 2
[experts]
placement      = "auto" | "device" | "host_compute" | "host_fetch"
hot_set_vram   = auto | bytes
[comm]
transport      = "auto" | "p2p" | "host"
```

Every `auto` resolves to a concrete value in the plan file, which the doctor bundle includes, so a reported result always states exactly how the model was placed.
