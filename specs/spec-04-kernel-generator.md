# Spec 4 — Kernel Generator and Registry

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 1, 2, 3. Constrains: specs 6, 9, 11, 14.

## 0. Purpose and scope

How every op in spec 1 becomes runnable code on a given arch: the three implementation tiers, the generator that produces the fast tier from the arch descriptor, the variant key that names a kernel, autotuning, how compiled code is shipped and cached, and how the registry resolves a variant at graph capture. Also the kernel ABI, the inline-asm policy, the test gates a kernel must pass, and the checklist for bringing up a new arch.

Out of scope: the numerics each kernel must honor (spec 1 §6), the transport under collectives (spec 5), profiling output format (spec 11).

## 1. Principles

1. **Generated, from the descriptor.** Fast-path kernels are emitted by a generator that reads `ArchDescriptor` and a tile config. No hand-written per-arch kernel files. Arch-specific instructions live only in leaf wrappers (§8).
2. **Three tiers, always present.** CPU f32 reference (oracle), portable HIP reference (any gfx9+), generated fast path. A new arch runs on tier 1 the day its descriptor exists.
3. **Shapes are baked, not passed.** Every static dimension is part of the variant; the kernel receives only pointers and per-step dynamic data. This is what makes graph capture and batch invariance cheap.
4. **Autotune chooses among correct configs only.** Every config in a search space satisfies the numerics contract; tuning changes speed, never results.
5. **Ship code objects, JIT as fallback.** A release contains compiled variants for every supported arch and every entry in the tune table. Runtime compilation covers unshipped shapes and never blocks the reference tier.
6. **Generated source is checked in.** The generator is the source of truth; the emitted `.hip` files for the shipped tune table are committed so kernel diffs remain auditable, and CI fails if regeneration differs.

## 2. Tiers

| tier | language | scope | selected when |
|---|---|---|---|
| **T0** | Rust, scalar f32 | every op; the oracle | tests; any op without a T0v on the CPU device |
| **T0v** | Rust + SIMD (AVX2 / AVX-512 VNNI / AMX) | `matmul`, `moe_ffn` (GEMV/GEMM over `L0`/`L1` int schemes), `attention`, `state_write_kv`, `linear_attn_scan`, `norm`, `rope`, `act_mul`, `embed_gather`, sampling | CPU device: host-computed experts (spec 5 §3.4), the helper model (spec 12 §7), `r9v eval` reference runs (spec 13 §11–12) |
| **T1** | HIP C++, portable | every op; wave intrinsics only, no WMMA/dot4/fp8 cvt | arch has no validated T2 for the variant, or config forces it |
| **T2** | generated HIP C++ + leaf intrinsics | ops with a fast-path generator for this arch | validated for `(arch, generator version)` and present in bundle or JIT-able |

Rules:
- T0 is written once per op and never optimized; correctness and readability only.
- T0v honors the same numerics contract as T2 (i32 accumulate, fixed reduction order) and is golden-tested against T0. It exists because host-computed experts are on the latency path for large MoE models, and because the helper runs on it. Ops without a T0v fall back to T0 on the CPU device.
- T1 must pass golden tests against T0 on every arch in the support matrix and is the fallback for every variant. It is allowed to be 2–5× slower than T2.
- T2 exists per op family (§5). An op without a T2 generator on a given arch simply resolves to T1 and the doctor bundle says so.

## 3. Variant key

A kernel variant is identified by:

```
VariantKey {
  op:          OpId
  arch:        ArchName
  gen_version: u32                 # generator version; bumps invalidate tunes and bundles
  static:      OpStatic            # per-op static parameters, below
  config:      TileConfig          # chosen by autotune from the op's search space
}
variant_hash = xxh3(VariantKey serialized)
```

`OpStatic` per family (dynamic values come through the ABI, §7):

- `matmul`: `(M_bucket, N, K, w_scheme, w_layout, act_scheme, out_dtype, epilogue, interleave, sparse)`
- `moe_ffn`: `(T_bucket, E_local, K_topk, Dm, Dff, schemes, act_scheme, placement_kind)`
- `attention`: `(q_bucket, h_local, hkv_local, d, dv, cache_dtype, attention_layout, mask_kind, latent?, softcap?, sinks?)`
- `state_write_kv`: `(hkv_local, d, dv, cache_dtype, attention_layout, latent?)`
- `linear_attn_scan`: `(kind, h_local, d, dv, chunk, mode: Chunked|Recurrent)`
- `norm`, `rope`, `act_mul`, `quant_act`, `residual_add`, `embed_gather`, `ngram_gather`: `(T_bucket, dims, dtypes, fused_with)`
- `logits_postprocess`, `sample`, `verify`: `(S_bucket, V, q_bucket, method)`
- collectives: `(bytes_bucket, dtype, transport)`

`M_bucket`/`T_bucket`/`S_bucket`/`q_bucket` are the spec 1 §3.5 buckets; `M_bucket` and `T_bucket` are the bucket of the full step `T = T_dec + T_pre`, since those ops run over every token, while `q_bucket` is per sequence class (decode ≤ 16, prefill above). `N`, `K`, `d`, `V` are exact.

## 4. Generator

### 4.1 Structure

A Rust crate `r9v-kgen`:

```
kgen::emit(op_static, arch: &ArchDescriptor, config: &TileConfig) -> HipSource
kgen::search_space(op_static, arch) -> Vec<TileConfig>
kgen::abi(op_static) -> AbiStruct
kgen::cost_model(op_static, arch, config) -> Estimate    # bytes, flops, LDS, regs, occupancy
```

Emission is string assembly from typed building blocks (fragment loaders for each `LayoutId` and dtype, epilogue writers, reduction trees, softmax stages), not text templates. Each block knows its LDS and register cost so `cost_model` is derived, not estimated by hand. The cost model prunes the search space before autotune runs anything.

### 4.2 What the descriptor drives

- fragment loader selection: `arch.fragment_layout` and `arch.matrix_ops` decide whether B fragments come straight from global memory (the `L1` = native case) or through a permute
- instruction selection: `matrix_ops` for GEMM inner loops, `valu_dot` for GEMV, `fp8_convert` for cache dequant, `sparse_matrix` for `L1S`
- tile bounds: `lds_bytes_per_wg`, `vgprs_per_lane`, `wave_size`, `max_wg_size`
- grid shaping: `cu_count` for split-KV and split-K factors
- pipelining depth: derived from `mem_bw_gbps · latency` against the tile's bytes per iteration

A descriptor field the generator does not know is an error at emit time, not a silent default.

### 4.3 Compilation

`clang++ -x hip --offload-arch=<arch> -O3 -ffast-math=off -fno-gpu-approx-transcendentals` from the ROCm LLVM, invoked directly (not via the `hipcc` wrapper) with flags pinned in `kgen`. Output is a code object per variant. `-ffast-math` stays off because it licenses reassociation, which breaks determinism; per-op fast intrinsics (`__expf`) are opted into explicitly in the emitted source where the numerics contract allows.

## 5. Kernel families (T2 designs for gfx1201)

Each family lists the config space; the generator emits the cross product minus what the cost model rejects.

### 5.1 GEMM (`matmul`, M > 8; `moe_ffn` per-expert)

- B (weights) fragments loaded directly from global in `L1` order into WMMA operands; no LDS staging for weights.
- A (activations, int8 or fp8 per-token) staged in LDS once per WG and re-read per N-tile.
- Inner loop: `wmma_i32_16x16x16_iu8` / `iu4` / `fp8_fp8` / `f16` per `w_scheme × act_scheme`; scale application per spec 1 §6.2.
- Epilogue fused per variant: bias, residual, activation, gated (`gate_up` interleave), `quant_act` for the next matmul when the fusion table allows.
- Config: `BM ∈ {16, 32, 64, 128}`, `BN ∈ {64, 128, 256}`, `BK ∈ {64, 128, 256}`, `waves ∈ {4, 8}`, `pipeline ∈ {1, 2, 3}`, `split_k ∈ {1, 2, 4}` (deterministic partials into a fixed-shape workspace + tree reduce kernel).
- `L1S`: same with `swmmac_*` in the inner loop and the index region streamed alongside.

### 5.2 GEMV (`matmul`, M ≤ 8)

- One WG per row-block group; each wave owns 16 output rows and streams K in `L1` order: each lane holds 8 K-consecutive weights of one row, issues two `v_dot4_i32_i8` against activation bytes broadcast from LDS, accumulates i32 (`PerToken`) or i32-then-f32 per 32-block (`PerBlock32`).
- The two `kgroup` lanes of each row are reduced with a fixed-direction `ds_swizzle`/`v_permlane`; K split across waves in a WG is reduced through LDS in ascending wave order.
- `i4` weights: nibble unpack to i8 before `dot4`. `e4m3` weights: `v_cvt` to f16 then `dot2` — slower per byte, allowed, reported.
- M-batched: the same weights feed up to 8 activation rows resident in LDS, so weight bytes are read once per step.
- Config: `rows_per_wave = 16` fixed, `waves_per_wg ∈ {4, 8, 16}`, `k_unroll ∈ {2, 4, 8}`, `k_split ∈ {1, 2, 4}`.

### 5.3 Attention

**Decode (`q_bucket ≤ 16`)**: split-KV. A sequence's blocks are partitioned across `P` WGs (`P` from `cu_count` and `ctx_len`); each computes a partial `(m, l, acc)` over its block range in ascending block order; a merge kernel combines partials in ascending partition order. Q (up to 16 rows × `h_local`) held in registers; K tiles loaded from the cache in `attention_layout` straight into `iu8`/`fp8` WMMA fragments with the per-token K scale applied to the logits column; P quantized per row to e4m3 (or i8), V scale applied to P's column before the PV WMMA. Tree masks are applied from the ancestor bitmask.

**Prefill (`q_bucket > 16`)**: flash-attention-2 structure. Q tiles of `BQ` rows per WG, K/V streamed by block, online softmax in f32, same fragment path as decode. Causal and window masks skip whole blocks where possible.

- Config: `BQ ∈ {32, 64, 128}` (prefill), `kv_per_wave ∈ {32, 64}`, `partitions ∈ {1..cu_count}` (decode), `waves ∈ {4, 8}`.
- `attention_layout` for gfx1201: K stored per block as `[d/16][32 tokens][16]` in B-fragment lane order so a 16-token × 16-d fragment is one 64-bit load per lane; V stored as `[32 tokens][dv]` in A-fragment order. Chosen by the generator and recorded in the doctor bundle.

### 5.4 `state_write_kv`

Vectorized scatter: each token's `hkv_local × (d + dv)` values quantized to the cache dtype with an f32 absmax per (token, head), written into `attention_layout` positions computed from `slot_map`. Fused with decode attention per spec 1 §3.4 when `q_bucket ≤ 16`.

### 5.5 `linear_attn_scan`

- **Chunked** (`query_len ≥ 32`): per (sequence, head), chunks of 64 tokens; intra-chunk via WMMA (`f16` operands, f32 accumulate), inter-chunk state carried in f32. State read from slot A, written to slot B.
- **Recurrent** (`query_len < 32`): one wave per (sequence, head), `[d, dv]` state in registers/LDS, sequential update per token. Same outputs as chunked within spec 1 tolerances; tested against T0.
- Config: `chunk ∈ {32, 64}`, `heads_per_wg ∈ {1, 2, 4}`.

### 5.6 `moe_ffn`

- Pre-pass: stable sort `(expert, token)` pairs, prefix-sum per expert → per-expert token ranges. Deterministic.
- Per expert with ≥ 1 token: GEMM family kernel over that token range (M = tokens for this expert, bucketed), gate/up interleaved, gated epilogue, then down GEMM. Experts with 0 tokens are skipped by a device-side predicate, not by graph structure, so the graph stays static.
- Combine: `scatter_add_rows` in sorted order with the routing weight applied in f32.
- Expert placement: the kernel receives a resident-set table (expert → device address, or `Host`). Tokens whose expert is `Host` are excluded from the device pass and handled by the segment (spec 5 §3.4, spec 6 §3.2). In `host_fetch` mode the scheduler guarantees residency before launch and an absent expert is a hard error at this level.
- Config: inherits GEMM configs per `M_bucket`; `experts_per_wg ∈ {1, 2}`.

### 5.7 Memory-bound elementwise (`norm`, `rope`, `act_mul`, `residual_add`, `quant_act`, `cast`, `embed_gather`, `ngram_gather`)

128-bit loads and stores, one row per wave or per WG depending on width, f32 math, single pass. Fusions from spec 1 §3.4 are emitted as one kernel with the same structure. Config: `rows_per_wg ∈ {1, 2, 4}`, `vec_width ∈ {8, 16}`.

### 5.8 Sampling (`logits_postprocess`, `sample`, `verify`)

Per sequence one WG. Penalties and temperature in f32; top-k/top-p via a wave-level bitonic sort of `(−logit, index)` over `V` in LDS chunks (stable by construction); inverse-CDF sampling with Philox4x32 keyed by `(seq_id, step, draw)`. `verify` walks candidates in order and draws one uniform per position. Config: `V_chunk ∈ {1024, 2048, 4096}`.

### 5.9 Reductions for collectives

Element-wise f32 add over fixed-size buffers in fixed rank order; the transport (spec 5) delivers peer buffers, the kernel here adds them. `all_to_all` packing/unpacking kernels use the sorted expert ranges from §5.6.

## 6. Autotune

### 6.1 Procedure

For each `(op, static)` requested:
1. `search_space` → `cost_model` prune (drop configs that exceed LDS/VGPR budgets or fall below 50% of the best estimate).
2. Compile survivors (parallel, cached by source hash).
3. Benchmark each: 5 warmups, 20 timed launches via `hipEvent`, take the median. Reject any config whose golden test against T0 fails (should never happen; it is checked anyway).
4. Record the winner and the full table in the tune file.

Time budget online: 2 s per variant by default; if exceeded, the best-so-far wins and the entry is flagged `partial` so the doctor can show it and a later offline pass can finish it.

### 6.2 Tune files

```
tune/<arch>/<gen_version>.toml         # shipped, produced on the reference machine
~/.cache/r9v/tune/<arch>/<gen_version>/<driver_hash>.toml   # local additions
```

Entries keyed by `(op, static_hash)` → `config`, `median_us`, `bytes`, `flops`, `measured_on: { driver, rocm, clock }`. A local entry overrides a shipped one only if measured on the same `gen_version`; a bump discards both.

### 6.3 Determinism of tuning

Only configs from the search space are eligible, and every config is deterministic (spec 1 §6.1). Tuning therefore changes timing, never bits. Two machines with different winners produce identical outputs.

## 7. Kernel ABI

Every variant has a single argument struct passed by value:

```
struct <op>_<static_hash>_args {
  // device pointers (values, scales, indices, activations, outputs, workspaces)
  // per-step dynamic scalars and tables: BatchMeta fields the op uses
  // nothing static: N, K, d, dtypes, layouts are baked into the variant
}
```

- Pointers are 256-byte aligned; the generator emits `__builtin_assume_aligned`.
- Workspaces (split-K partials, split-KV partials, MoE sort buffers) are fixed-size per bucket and owned by the scheduler's per-graph arena (spec 6), never allocated by a kernel.
- Launch geometry is a property of the variant, recorded in the tune entry, so graph capture replays it verbatim.

## 8. Inline asm and intrinsics policy

- Arch-specific instructions appear only in `kgen/src/leaf/<arch>.rs`, which emits small `__device__` wrapper functions (`wmma_iu8`, `dot4_i8`, `cvt_e4m3_f16`, `swmmac_iu8`, `permlane_reduce`). Everything else the generator emits is plain HIP C++.
- Each wrapper is a compiler builtin (`__builtin_amdgcn_*`) where one exists. Inline asm is permitted only when the builtin is missing or miscompiles on a pinned ROCm version, and the wrapper then carries both forms with a compile-time switch and a unit test asserting they agree.
- No inline asm outside leaf wrappers. CI and `check_card.py` reject a change that adds asm elsewhere.

## 9. Registry

### 9.1 Contents

The registry is an in-memory table built at engine start from: the shipped bundle manifest, the local JIT cache, and the tune files. Entry: `variant_hash → { tier, code_object, entry_symbol, launch_geometry, workspace_bytes, validated: bool }`.

### 9.2 Resolution (at graph capture)

For each op instance:
1. Build `OpStatic` from resolved shapes and bucket.
2. Look up T2 for `(arch, gen_version, static)`: shipped entry → use. Local tune entry with code object → use. Otherwise, if JIT is available and `allow_jit`, run autotune (§6.1) now, then use. Otherwise → T1.
3. T1 code objects for every op are always in the bundle for every arch in the support matrix; resolution cannot fail on a supported arch. On an unlisted arch, T1 is JIT-compiled from its portable source if a compiler is present; if not, the engine refuses to start and names the arch.
4. The resolved tier per op instance is written to the graph summary and the doctor bundle.

### 9.3 Validation flag

A T2 variant is `validated` only if its golden, batch-invariance and determinism tests (§10) have passed on this `(arch, gen_version, driver_hash)`. Shipped variants carry validation from the reference machine; a JIT-tuned variant validates itself immediately after tuning (the tests run in-process, a few hundred ms). An unvalidated variant is never selected.

## 10. Test gates

Every T1 and T2 variant, on the self-hosted RDNA4 runner (spec 14):

1. **Golden**: output vs T0 within spec 1 §6.1 tolerances on 32 random inputs per shape, including edge shapes (padding rows, single token, max bucket).
2. **Batch invariance**: the same row alone, padded, and embedded among random rows → bit-equal.
3. **Determinism**: two runs → bit-equal.
4. **Perf regression**: median time vs the stored baseline for the variant; fail on > 3% slower. Baselines update only through an explicit `tune/` PR.
5. **Achieved bandwidth/rate**: recorded and compared against spec 11 §9.5 for the variant's family: GEMV against the measured-bandwidth floor (0.93), GEMM and attention against their diagnostic thresholds. Variants under their line stay `validated` (they are correct) but are listed as `below-floor` in the tune coverage report; the phase A fast-path exit requires no below-floor GEMV variants in the dense graph and the two step-level floors met.

T1 additionally gets **shape fuzzing**: random `(N, K, T)` within constraints, compared to T0.

GitHub-hosted CI runs T0 tests only, compiles T1 and T2 for all archs to catch emission and compile errors, and diffs regenerated source against the checked-in files.

## 11. Bundle

```
bundle/
  manifest.json           # gen_version, arch list, per-variant hash → file, tier, validated_on
  gfx1201/*.co            # code objects, one per variant
  gfx1100/*.co            # T1 for every op + any validated T2
  reference/*.hip         # T1 source for JIT on unlisted archs
```

Loaded lazily by `hipModuleLoadData` on first resolution. The manifest hash is part of the doctor fingerprint. A bundle built for `gen_version = n` refuses to load under generator `n + 1`.

## 12. Profiling hooks

Every launch goes through one dispatch function that, when profiling is enabled (spec 11), records `hipEvent` start/end, the variant hash, the workspace used, and the static `bytes`/`flops` from the tune entry, so achieved bandwidth and matrix rate are computed per launch without instrumenting kernels. Overhead when disabled is one branch.

## 13. Bringing up a new arch

1. Write `ArchDescriptor` (spec 1 App. A) from the ISA documentation. Do not put SKU, board, clock, memory or topology facts in it; those enter through runtime `DeviceDescriptor` discovery.
2. Run the doctor's measurement pass to fill bandwidth, dispatch overhead, P2P matrix, and confirm `matrix_ops` rates empirically.
3. T1 for every op compiles and passes golden. The arch is now **supported (reference tier)**.
4. Add the arch's leaf wrappers (§8) for whichever of `matrix_ops`, `valu_dot`, `fp8_convert`, `sparse_matrix` it has. Set `fragment_layout` and `attention_layout` to its native orders (new `LayoutId`s if they differ from `L1`).
5. Run autotune for the shipped static set; run §10 gates; commit the tune file and generated source.
6. Promote ops to T2 in the support matrix as they validate. The arch is **supported (fast path)** for those ops.

Expected effort for an RDNA-family successor: steps 1–3 in a day, step 4 in a week if the fragment layouts changed, step 5 mostly machine time.

## 14. Config

```
[kernels]
allow_jit               = true     # §9.2: autotune and compile unshipped variants at runtime when a compiler is present
allow_nondeterministic  = false    # spec 1 App. B escape hatch; recorded in the doctor bundle when true
tune_budget_ms          = 2000     # §6.1 online autotune budget per variant
```
