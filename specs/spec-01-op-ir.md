# Spec 1 — Op IR

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: nothing. Constrains: specs 2–15.

## 0. Purpose and scope

The Op IR is the closed set of operations that kernels implement and that model definitions compose. It is the contract between four things that must be able to evolve independently:

- model definitions (spec 8) — build graphs over ops, never touch kernels
- kernel generator and registry (spec 4) — implement ops per arch, never see models
- partitioner (spec 5) — rewrites graphs using each op's sharding rules
- scheduler (spec 6) — executes captured graphs, owns nothing inside an op

Out of scope here: on-disk weight layout (spec 2), state allocation and rollback (spec 3), how kernel variants are chosen (spec 4).

## 1. Principles

1. **Closed set.** A model that needs something not expressible here triggers the RFC process in §7, not a one-off kernel. Parametrize before adding.
2. **Reference kernel mandatory.** Every op has a portable HIP reference and a CPU f32 reference. An arch is "supported" the day both run; fast paths are promotions, not prerequisites.
3. **Pure except declared state.** An op's outputs are a function of its inputs and attributes. The only mutable things are sequence-state handles (§2.6) and RNG state (§4.F), and ops that touch them say so in their signature.
4. **Sharding is declared, not discovered.** Every op carries a legal-layouts table (§5). The partitioner only ever applies that table.
5. **Numerics are part of the signature.** Accumulation type, reduction order, and batch invariance are specified per op (§6) and tested, not left to the kernel author.
6. **Static graphs.** Decode and prefill-chunk graphs are captured per shape bucket (§3.5). Nothing data-dependent changes graph structure; data-dependent work happens inside ops or in the scheduler's pre-step phase.
7. **Fusion is from a table.** The compiler may fuse only pairs/chains listed in §3.4, and a fused kernel must satisfy the union of the fused ops' numerics contracts.

## 2. Core types

### 2.1 DType

| id | meaning | notes |
|---|---|---|
| `f32` | IEEE single | accumulators, norms stats, logits |
| `f16` | IEEE half | activations |
| `bf16` | bfloat16 | activations |
| `e4m3` | fp8 E4M3 | activations, KV cache, fp8 weights |
| `e5m2` | fp8 E5M2 | second WMMA operand only |
| `i8` | signed int8 | weights, quantized activations |
| `i4` | signed/unsigned int4, packed 2 per byte | weights only |
| `i32` | int32 | accumulators, ids, counts |
| `u32` | uint32 | token ids, row ids, block ids |
| `bool` | mask | attention masks, grammar masks |

The enum is extensible by RFC (§7). Codebook GGUF types (IQ-style) have no dtype of their own: spec 2 §3.3 maps them to the `i8` matrix path by in-register LUT expansion, so the IR never sees a codebook.

### 2.2 QuantScheme (attached to a tensor, not a dtype)

```
None
PerRow   { scale: f16 }                      # one scale per output row (weights)
Scheme   { id: spec2::SchemeId }             # weights: a spec 2 §3 scheme (`I8_B128`, `I4_K`, `I8_B32F`, ...); block structure and scale records are defined there
PerToken   { scale: f32 }                    # activations only; one scale per row of x (native, smoothing folded)
PerBlock32 { scale: f32 }                    # activations only; one scale per 32 along K (GGUF parity path, spec 2 §3.4)
```

Every scheme's block (or superblock) size is a power of two ≥ 16, and the K dimension of any consuming op must be a multiple of it.

### 2.3 Tensor

```
Tensor {
  shape:     [Dim; ≤4]           # symbolic dims resolved at capture
  dtype:     DType
  quant:     QuantScheme
  layout:    LayoutId            # spec 2 logical layout version; activations use Contiguous
  placement: Placement
  sharding:  ShardLayout         # §5.1
  class:     Weight | Activation | State | Staging | Param
}
Placement = Device(rank) | Host | Tiered      # Host: pinned host memory (host-computed or host-gathered); Tiered: slab-backed, fetched by unit (spec 9 §6)
```

`Host` and `Tiered` are legal only for `Weight` class tensors of the classes spec 2 §5 lists (expert weights, n-gram tables, embeddings). A quant tool's `tiered` *hint* (spec 2 §4) is resolved by the planner into per-unit placements at load (spec 5 §3.4). Reshapes and transposes are metadata on the edge, not ops; if a consuming kernel cannot accept the resulting strides the compiler inserts an explicit `copy` and flags it (§3.3).

### 2.4 Shape symbols

`T` tokens in the batch (sum of query lengths), `S` sequences, `Dm` model dim, `Dff` FFN dim, `H` query heads, `Hkv` KV heads, `D` head dim, `E` experts, `K` top-k, `V` vocab, `Np` n-gram hash heads, `L` layers. Attributes may reference these; kernels see concrete integers per bucket.

### 2.5 Batch metadata (one external input, shared by ops)

```
BatchMeta {
  seq_ids:      [S] u32
  query_len:    [S] u32          # 1 for plain decode, k+1 for spec verify, chunk size for prefill
  ctx_len:      [S] u32          # tokens already in state before this step
  positions:    [T] u32 | [T,3] u32 (mrope)
  slot_map:     [G, T] u32       # where each new token's KV goes (block, offset) flattened, per layer group
  block_table:  [G, S, max_blocks] u32   # one table per layer group (spec 3 §6.1); G is fixed per model
  window_start: [G, S] u32       # first retained position for Window groups (spec 3 §3.5); 0 otherwise
  tree:         Option<TreeMask> # §4.D
}
```

### 2.6 State handles

Ops that read or write per-sequence state take a `StateHandle(layer, kind)` argument. Kinds in v1: `KvPaged`, `KvLatent` (MLA), `Recurrent` (fixed-size per head), `ConvWindow`. The state manager (spec 3) owns allocation, eviction, checkpoint and rollback; the IR only names the handle and declares read/write.

## 3. Graph model

### 3.1 Structure

A graph is a DAG of op instances over tensors. There is one graph kind, the **step graph**: `S` sequences, each with a `query_len` that is either `1..k+1` (decode, including spec-decode verify) or a prefill chunk. A step's tokens are `T = T_dec + T_pre`, and the graph is captured per `(plan, rank, S_bucket, T_dec_bucket, T_pre_bucket, segment)` (spec 6 §5.1).

Ops over the token axis (`matmul`, `norm`, `rope`, ...) run over all `T`. `attention` and `linear_attn_scan` are emitted as one launch for the decode-class sequences (`query_len ≤ 16`) and, when `T_pre > 0`, a second launch for the prefill-class sequences, because the two classes resolve to different kernel variants (spec 4 §5.3). Spec-decode verify is a decode-class sequence with `query_len > 1`, not a third kind. Padding rows and prefill positions that are not the last token of a completed prompt are masked out of sampling.

### 3.2 External inputs and outputs

Inputs: `token_ids [T] u32`, `BatchMeta`, `rng_state [S]`, `gather_staging` (n-gram rows, §4.A), `grammar_mask [S, q, V] bool` (optional; one mask per verified position, spec 10 §4), per-seq sampling params, `embed_override [T, Dm] act_dtype` with `embed_mask [T] bool` (optional; rows where the mask is set replace `embed_gather`'s output, which is how an external vision or audio encoder feeds embeddings without the graph knowing about it).
Outputs: `sampled [S, k+1] u32`, `accept_len [S] u32`, `logits` (optional, for logprobs), `hidden [T, Dm]` (optional; the pre-`lm_head` hidden state, exported when a proposer needs it, spec 7 §3), updated `rng_state`.

All external tensors have fixed shape within a bucket. Variable-count work (which rows to gather, which tokens per expert) is resolved either in the scheduler pre-step phase or inside the op.

### 3.3 Materialization

The compiler tracks strides. A `copy` op is inserted only when a kernel's declared input requirement cannot be met by a view. Every inserted `copy` is reported in the graph summary; a model definition that causes one is considered to have a layout bug until proven otherwise.

### 3.4 Fusion table

Allowed fusions (compiler may apply; all preserve §6 contracts):

| pattern | result |
|---|---|
| `residual_add → norm` | fused add-norm, f32 stats |
| `norm → quant_act` | norm emits quantized activation + per-token scale directly |
| `matmul → bias / residual_add / activation` | epilogue |
| `matmul(gate) ∥ matmul(up) → act_mul` | interleaved gate/up GEMM with gated epilogue |
| `rope → state_write_kv` | rope applied on the write path |
| `rope → attention` (prefill) | rope applied in the Q load |
| `state_write_kv → attention` (decode, `query_len ≤ 16`) | single launch: write the new K/V, then attend; prefill keeps them separate |
| `logits_postprocess → sample` | single sampling kernel |
| `quant_act → all_to_all` (EP) | dispatch already-quantized tokens |

Anything else is an RFC. Fusion never changes an op's declared sharding rule.

### 3.5 Shape buckets

`S`, `T_dec` and `T_pre` buckets: `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}`, plus `0` for `T_pre` (no prefill this step). A batch is padded to the next bucket per axis; padding tokens are masked, never sampled, and do not write state. Weight dims `N, K` are fixed per tensor and are part of the kernel key, not a bucket. Sequence lengths are not bucketed; paged attention iterates block tables.

## 4. Op catalog

Notation: `→` outputs. Inputs listed with dtype constraints. `attrs` are compile-time. `[R]` = reads state, `[W]` = writes state. Fast-path notes are gfx1201; other archs use the reference tier until promoted.

### A. Data movement and lookup

**`embed_gather`**
`token_ids [T] u32, table [V, Dm] (i4|i8|f16, Block|PerRow, Device|Tiered)` → `x [T, Dm] act_dtype`
attrs: `scale: f32` (e.g. √Dm for Gemma), `out_dtype`
Sharding: table `Replicated` or `RowShard(V)` (vocab-sharded, each rank gathers its rows then `all_reduce`).
Numerics: dequant to f32, scale, cast.

**`ngram_gather`**
`gather_staging [T, Np, Dn] (i4|i8, Block)`, `row_scales` → `x [T, Np·Dn] act_dtype`
attrs: `orders: [u32]`, `heads: Np`, `hash: HashId`, `table_sizes: [u32]`, `combine: Concat|Sum`
Two placement cases, one op. `Tiered` or `Host` table: row ids are computed by the scheduler pre-step phase from `token_ids` and context (spec 6), the host gathers rows into a pinned buffer and issues one fixed-size H2D copy into `gather_staging`, and the op only dequantizes and lays out. `Device` table (small models): the op hashes on device and gathers directly from the table; `gather_staging` is unused. The kernel registry resolves which by the table's placement.
Sharding: `Replicated` staging (each rank stages the same rows) or `RowShard(Np)` by hash head.

**`quant_act`**
`x [T, N] (f16|bf16|f32)` → `xq [T, N] (i8|e4m3) PerToken`, `scale [T] f32`
attrs: `target: i8|e4m3`, `smoothing: None|Folded` (folded smoothing lives in weights; op is a plain per-token absmax)
Numerics: absmax in f32; i8 uses symmetric round-to-nearest-even; e4m3 uses saturating cast.

**`cast`** — `x → y` with `attrs: dtype`. No sharding change.

**`copy`** — device↔device, device↔host staging, or contiguization. Fixed shape. Reported when compiler-inserted.

**`gather_rows`** / **`scatter_add_rows`**
Used inside `moe_ffn` and for tree-verify bookkeeping. Exposed as ops so the reference tier can compose them; the fast path fuses them. `scatter_add_rows` must use a deterministic order (sorted indices, sequential accumulate per destination).

### B. Normalization and elementwise

**`norm`**
`x [T, N] act_dtype, weight [N] f32, bias? [N] f32` → `y [T, N] out_dtype`
attrs: `kind: Rms|Layer`, `eps: f32`, `axis: Last|Head(D)` (per-head QK-norm uses `Head`), `weight_offset: f32` (Gemma's 1+w), `out_dtype`
Numerics: mean/variance in f32 over the full axis in a fixed reduction order.
Sharding: input must be `Replicated` on the normalized axis. Partitioner inserts `all_gather` or `all_reduce` before it as needed.

**`residual_add`** — `a + b` in f32, cast to `out_dtype`. Fusion target.

**`act_mul`**
`gate [T, Dff], up [T, Dff]` → `y [T, Dff]`
attrs: `act: Silu|Gelu|GeluTanh|Relu2|Identity`, `clamp: Option<f32>`
Numerics: activation computed in f32.

**`activation`** — non-gated form of `act_mul`.

**`rope`**
`x [T, H, D], positions [T] | [T,3]` → `x' [T, H, D]`
attrs: `rot_dim: u32` (partial rotary), `theta: f32`, `style: Neox|Interleaved`, `scaling: None|Linear(f)|Yarn{factor, beta_fast, beta_slow, orig_ctx, mscale}|Dynamic`, `mrope_sections: Option<[u32;3]>`
Numerics: cos/sin in f32 from a per-position table computed on device in f32 (no host table); result cast to `out_dtype`. Fusion into KV write and prefill attention permitted.

### C. Matmul family

**`matmul`**
`x [M, K] (f16|bf16|i8 PerToken|i8 PerBlock32|e4m3 PerToken), w [N, K] (i4|i8|e4m3|f16, Block|PerRow, LayoutId), bias? [N] f32` → `y [M, N] out_dtype`
attrs: `out_dtype: f16|bf16|f32`, `epilogue: None|Bias|Residual|Act(act)`, `transpose_w: bool` (should be false in practice; spec 2 stores `[N, K]`)
Numerics (§6.2): `i8×i8 PerToken → i32` accumulate over full K, then `(w_scale · x_scale)` in f32; `i8×i8 PerBlock32 → i32` per 32-block, scaled and summed in f32 in ascending block order; `e4m3×e4m3 → f32`; `f16×f16 → f32`. Reduction order fixed per kernel variant. Split-K only via deterministic partials + tree reduce.
Sharding table:

| x | w | y |
|---|---|---|
| `Replicated` | `ColShard(N)` | `ColShard(N)` |
| `ColShard(K)` | `RowShard(K)` | `Partial` (needs `all_reduce`) |
| `Replicated` | `Replicated` | `Replicated` |

gfx1201 fast paths: `i8×i8`, M ≤ 8: `v_dot4_i32_i8` GEMV; M > 8: `v_wmma_i32_16x16x16_iu8`. `i4` weights, M ≤ 8: unpack + dot4; M > 8: `iu4` WMMA. `e4m3×e4m3`: `wmma_f32_16x16x16_fp8_fp8`. `f16`: `wmma_f32_16x16x16_f16`. 5/6-bit GGUF repacks: unpack to i8 in registers → `iu8` path (parity with llama.cpp, not above it).

**`moe_route`**
`logits [T, E] f32` → `expert_ids [T, K] u32, weights [T, K] f32`
attrs: `top_k: K`, `scoring: Softmax|Sigmoid`, `renormalize: bool`, `group: Option<{n_group, topk_group}>`, `bias: Option<Param [E]>` (aux-loss-free routing correction), `scale: f32`
Numerics: f32 throughout; ties broken by lowest expert index (deterministic).
Sharding: `Replicated`.

**`moe_ffn`**
`x [T, Dm], expert_ids [T, K], weights [T, K], w_gate_up [E, 2·Dff, Dm] (Tiered|Device), w_down [E, Dm, Dff]` → `y [T, Dm]`
attrs: `act`, `out_dtype`, `shared_experts: u32` (shared-expert path is a plain `matmul` in the graph; this attr only sizes internal buffers)
Semantics: sort tokens by expert; per-expert grouped GEMM gate/up → `act_mul` → down; weighted `scatter_add_rows` in sorted order.
Sharding: `ExpertShard(E)` (EP; partitioner inserts `all_to_all` dispatch and combine) or `ColShard`/`RowShard` on each expert (TP inside experts) or both. Expert placement is resolved at load by the planner (spec 5 §3.4); the op receives a resident-set table and computes only device-resident experts. Tokens routed to host-computed experts are handled by the segment mechanism (spec 6 §3.2) outside this op.
Numerics: same as `matmul` per expert; combine in f32 in sorted order. Batch invariant by construction (per-token result independent of other tokens' routing).

### D. Attention

**`state_write_kv`** `[W]`
`k [T, Hkv, D], v [T, Hkv, Dv], slot_map, StateHandle(KvPaged|KvLatent)` → ()
attrs: `cache_dtype: f16|i8|e4m3`, `scale_granularity: PerTokenHead|PerBlock`, `latent: Option<{kv_lora_rank, rope_dim}>` (MLA: writes the compressed latent + rope part instead of K/V)
The only op that writes attention state. Padded tokens are skipped.

**`attention`** `[R]`
`q [T, H, D], StateHandle(KvPaged|KvLatent), BatchMeta` → `o [T, H, D]`
attrs: `softmax_scale: f32`, `mask: Causal|CausalWindow(w)|Tree`, `sinks: u32`, `logit_softcap: Option<f32>`, `mla: Option<{...}>`, `out_dtype`
Semantics: for each sequence s, query rows `query_len[s]` attend to `ctx_len[s] + query_len[s]` positions through `block_table[s]`. `Tree` uses `BatchMeta.tree` (§4.D.1). A single op covers decode (`query_len = 1`), spec verify (`1 < query_len ≤ 16`) and prefill chunks; the registry resolves variants by `query_len` bucket.
Numerics (§6.3): online softmax with f32 running max and sum; QKᵀ and PV accumulate in f32; cache dequant to f32 or fed as fp8 into WMMA with the per-token-head scale applied to P. Block iteration order fixed (ascending block index).
Sharding: `HeadShard(H)` with `Hkv % ranks == 0`; state sharded identically. No collectives inside.

**D.1 TreeMask** — `parents [T] i32` (−1 = root of its sequence) plus a derived `[T, T_max] bool` ancestor mask per sequence, built by the scheduler. Kernels may consume either.

### E. Sequence-state ops beyond attention

**`causal_conv1d`** `[R][W]`
`x [T, C], w [C, W_k], bias? [C], StateHandle(ConvWindow)` → `y [T, C]`
attrs: `kernel: W_k`, `act: Silu|Identity`
Reads the last `W_k − 1` inputs from state, writes the new tail.

**`linear_attn_scan`** `[R][W]`
`q [T, H, D], k [T, H, D], v [T, H, Dv], alpha [T, H] f32 (decay), beta [T, H] f32, StateHandle(Recurrent)` → `o [T, H, Dv]`
attrs: `kind: GatedDeltaNet|GLA|Mamba2`, `chunk: u32` (64 default), `out_dtype`
Semantics: chunked parallel form for `query_len ≥ 32`, sequential recurrent form otherwise (spec 4 §5.5); both produce identical state and output within tolerance. State per (seq, head) is `[D, Dv]` f32.
Spec-decode interaction: double buffer and recompute. The state manager (spec 3 §4.2) holds the verified state in slot A and the op reads A and writes slot B for all `query_len` tokens; commit swaps them. On partial acceptance the scheduler re-runs the accepted prefix from A into B through the recurrent form, then swaps. Cost is bounded by `k ≤ 16` tokens per layer and is budgeted in the scheduler's step accounting (spec 6 §4.2). The op itself has no notion of acceptance. Tree verify uses the same mechanism on the accepted path only.
Sharding: `HeadShard(H)`.

### F. Sampling and verification

**`logits_postprocess`**
`logits [S, q, V] f32, params [S] SamplingParams, history_counts? [S, V] u32, grammar_mask? [S, q, V] bool` → `probs [S, q, V] f32`
`SamplingParams { temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty, frequency_penalty, logit_bias: sparse [(token, f32)] }` per sequence. `logit_bias` is added before temperature; the mask is applied after all penalties and before the softmax.
Numerics: f32; sort for top-k/top-p is a fixed stable sort by (−logit, index).
Sharding: `Replicated`. Under TP the `lm_head` matmul produces `ColShard(V)` and the partitioner inserts an `all_gather` so this op always sees full rows. Sharded softmax and distributed top-k are not in the IR; the logits gather costs `S·V·4` bytes per step, which is small against the weight stream at any batch size this engine targets.

**`sample`**
`probs [S, V] f32, rng_state [S]` → `token [S] u32, rng_state' [S]`
attrs: `rng: Philox4x32`
Numerics: counter-based RNG keyed by `(seq_id, step, draw_index)`; inverse-CDF over the fixed-order probs. Reproducible for a given seed regardless of batch composition.

**`verify`**
`draft_tokens [S, k] u32, draft_probs? [S, k, V] f32, target_probs [S, k+1, V] f32, rng_state [S], tree?` → `accepted [S, k+1] u32, accept_len [S] u32, rng_state'`
attrs: `method: Rejection|Greedy|TypicalAcceptance{...}`
Semantics: standard speculative rejection sampling (accept with `min(1, p/q)`, resample from `norm(max(0, p−q))` on first rejection, bonus token from position `accept_len`). `draft_probs = None` means the proposer was deterministic (n-gram, MTP argmax) and `q` is a one-hot. Tree form walks `parents` and accepts the longest verified path.
Extensibility: new acceptance rules are new `method` values; proposers never touch this op (spec 7).

### G. Collectives (inserted by the partitioner only)

`all_reduce(op=Sum)`, `all_gather`, `reduce_scatter`, `all_to_all(counts)`, `send(peer)`, `recv(peer)`, `barrier`.
attrs: `group: GroupId`, `dtype`, `reduce_in: f32`
Numerics: reduction in f32 in ascending rank order on every rank (spec 5 §6.2); result bit-identical across ranks. `all_to_all` for EP carries variable per-peer counts resolved in the pre-step phase; buffers are fixed-size per bucket.
Transport: P2P where the measured topology says the directed pair supports it, host-staged otherwise. The ISA descriptor never carries peer facts. The op is the same either way.

## 5. Sharding

### 5.1 ShardLayout

```
Replicated
ColShard(axis)     # split along output features (Megatron column-parallel)
RowShard(axis)     # split along input features (row-parallel); consumer output is Partial
HeadShard(H)       # attention heads; implies KV heads and state split identically
ExpertShard(E)     # experts distributed across ranks
Partial            # sum across ranks pending; must be resolved by all_reduce before any op that requires Replicated
```

### 5.2 Rules

- Each op's table (in §4) lists legal `(input layouts) → output layout` tuples. The partitioner picks a tuple per op and inserts collectives to make inputs legal.
- `Partial` may flow through `residual_add` and `matmul` epilogues (partial sums add) but must be resolved before `norm`, `rope`, `attention`, `moe_route`, sampling, or any state write.
- Pipeline parallel is layer-range assignment plus `send`/`recv` at the boundary; no op rules change.
- The planner (spec 5) chooses between PP, TP, EP and combinations once at load, for the configured workload profile, using measured link costs from the topology. Placement never changes while a model is loaded. The IR is agnostic to the choice.

## 6. Numerics contract

### 6.1 Global

- **Batch invariance.** For any token, the outputs of every op are bit-identical regardless of which other tokens share the batch, which bucket the batch landed in, or the token's row index. Consequences: no atomics into any tensor that reaches logits; split-K uses fixed-shape partials and a fixed-order tree reduce; MoE combine is sorted; row-level reductions never depend on `T`.
- **Run-to-run determinism.** Same arch, same kernel cache hash, same seed → bit-identical outputs. Across archs or kernel versions: within tolerance.
- **Accumulation is f32 or i32.** No f16/bf16 accumulation anywhere.
- **Golden tests.** Every op has a CPU f32 reference. Tolerance defaults (initial, to be tightened empirically per op in the test suite): f16/bf16 paths abs 2e-3 / rel 1e-2 per element; i8 weight paths compared against reference dequant-then-f32 at abs 5e-3 / rel 2e-2; logits compared at top-1 agreement ≥ 99.9% and KL ≤ 1e-3 over the calibration set. (These bound a fast path against the reference on the same weights; quantization loss is bounded separately by spec 13 §6.2.)

### 6.2 matmul / moe_ffn

`i8` weights × `i8` activations: full-K i32 accumulate (K ≤ 65536 is safe against overflow at 127·127·K), single scale multiply in f32. Block-quantized weights with `Block.size < K` still accumulate the full K in i32 when scales are folded to per-row; when true per-block scales are used the kernel accumulates per block in i32 and sums blocks in f32 in ascending block order. `i4` weights with zero points: `s·(x·q − z·Σx)`, `Σx` computed once per token per block in i32.

### 6.3 attention

Online softmax, f32 max/sum, ascending block order. `e4m3` cache: dequant scale applied to P (not to K/V) so WMMA runs on raw fp8. `logit_softcap` applied in f32 before the max. Sinks are extra learned logits prepended to the softmax denominator.

### 6.4 norm, rope, activations

All in f32; cast once on output. RoPE uses on-device f32 cos/sin computed from `theta` and position; no precomputed half-precision tables.

### 6.5 sampling

f32 end-to-end; fixed stable sort order; counter-based RNG (§4.F). Rejection sampling accepts based on f32 probabilities computed identically on draft and target sides.

## 7. Adding or changing an op

An RFC must include:

1. Why existing ops plus attributes cannot express it (attempted parametrization shown).
2. Full signature in the §4 format, including sharding table and numerics contract.
3. CPU f32 reference and portable HIP reference implementations.
4. Golden test and batch-invariance test.
5. At least one model definition that uses it.
6. Fusion table changes, if any.

An op lands with the reference tier only. Fast paths are separate PRs gated on the golden tests. Removing or changing an op's signature bumps the IR minor version; model definitions pin the IR version they were written against.

---

## Appendix A — Arch descriptor

Consumed by the kernel generator (spec 4), the partitioner (spec 5), the loader (spec 9) and the scheduler (spec 6). ISA capabilities and physical-device facts are separate types. An ISA descriptor may be checked in; a physical-device descriptor may only be constructed from runtime discovery and doctor measurement. The planner also gets a measured link matrix.

```
ArchDescriptor {
  name:              str            # "gfx1201"
  family:            RDNA4 | RDNA3 | CDNA3 | Reference | CPU     # CPU is the T0/T0v device (spec 4 §2)
  wave_size:         u32            # 32
  lds_bytes_per_wg:  u32
  vgprs_per_lane:    u32
  matrix_ops: [ { shape: (16,16,16), a: f16, b: f16, acc: f32, rate: RelRate }, ... ]
                                     # complete list of WMMA/MFMA forms with relative throughput
  valu_dot:          [ dot4_i32_i8, dot2_f32_f16, dot2_f32_bf16, ... ]
  fp8_convert:       bool           # hardware cvt to/from e4m3/e5m2
  sparse_matrix:     bool           # SWMMAC / 2:4 support
  fragment_layout:   LayoutId       # native B-fragment order the zero-copy loader checks against (spec 2 §2.4)
  attention_layout:  LayoutId       # intra-block K/V element order the attention kernels use (spec 3 §3.2, spec 4 §5.3)
  max_wg_size:       u32
}

DeviceDescriptor {
  arch:              ArchDescriptor
  facts: {
    identity:        CPU | GPU { uuid: Option<[u8; 16]>, pci_bdf: str }
    cu_count:        u32
    vram_bytes:      u64
    l2_bytes:        Option<u64>
    l3_bytes:        Option<u64>
    nominal_mem_bw_gbps: Option<f32> # optional matched-board information; never a planning input
    clock_mhz:       Option<f32>
    graph_capture:   Supported | Unstable | None
  }
  measured: {                       # filled by the doctor's measurement pass (spec 11 §7); empty until then
    mem_bw_gbps, dispatch_overhead_us, matrix_rates: [RelRate], h2d_gbps, d2h_gbps
  }
  p2p:               [ (peer_rank, Direct | HostStaged, measured_gbps) ]   # mirrored in Topology.links (spec 5 §2)
}
```

gfx1201 ISA values: wave 32; 64 KB LDS/WG; matrix ops f16/bf16 (1×), e4m3/e5m2 (2×), iu8 (2×), iu4 (2× nominal, verify); `dot4_i32_i8` present; fp8 convert present; SWMMAC present. CU count, VRAM, caches, clocks, board bandwidth, graph-capture reliability, PCIe topology and P2P are not gfx1201 properties and must never be supplied by this constructor.

HIP device ordinals are ephemeral handles valid only inside the process that enumerated them. They never appear in a persistent device identity, plan-cache key or receipt fingerprint. GPU identity uses the runtime UUID when available plus canonical PCI BDF; rank is assigned only after discovery. CPU is always a valid device and does not depend on HIP being installed.

## Appendix B — Determinism and tolerance policy

- **Levels.** L0: bit-exact (same arch, same kernel hash, same seed). L1: within §6.1 tolerances (across kernel versions or archs). L2: statistical (across quant formats; measured by KL and top-1 on the calibration set).
- **What CI enforces.** L0 on the self-hosted RDNA4 runner for every op and for end-to-end logits on a fixed model set; L1 between the reference tier and every fast path; L2 for any format change. GitHub-hosted CI runs the CPU reference tier only and says so in its status name.
- **Batch-invariance test.** For each op: run the same token alone, in a padded bucket, and embedded among random other tokens; require L0 equality.
- **Escape hatch.** A kernel may be registered as `NonDeterministic` (e.g. an experimental atomic split-K) but the registry refuses to select it unless the config sets `allow_nondeterministic = true`, and the doctor bundle records that flag.
