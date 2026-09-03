# Roadmap

Status: draft 0.1 (2026-09-02). Companion to specs 1–15; spec 15 §11 defines the phases, this document sequences them.

## 0. What this is

The order of work, what each milestone must prove before the next starts, and which parts of the codebase are open to contributors at each point. Dates are deliberately absent; exit criteria are not. A milestone is done when its criteria hold on the runner, not when the code exists.

Two rules carry through every phase:

- **Implement the interface, exercise the subset.** Phase A runs one GPU, dense models, one sequence. It still builds `BatchMeta` with layer groups, declares sharding tables on every op, enumerates every state kind, and captures step graphs bucketed on `(S, T_dec, T_pre)`. Nothing is stubbed in a way that a later phase has to rip out. The single-device path is the general path with `ranks = 1`.
- **Every phase ends with a receipt, not a number.** The bench and doctor machinery arrives early (A4) precisely so that A5's kernels are judged by spec 11 receipts on the reference model.

## 1. What "worth building off of" means

Phase A is complete when all of these hold:

1. **Frozen interfaces at 1.0**: the spec 1 op set and `Tensor`/`BatchMeta` types, the spec 8 builder API and `LayerSpec`, the spec 2 scheme enum and `L0`/`L1` layouts, the spec 4 registry interface and tier contract, the spec 7 `Proposer` trait. Changes after this point go through the RFC process.
2. **One real receipt**: `decode` and `prefill` suites on the large dense reference model, valid, same-file compared against llama.cpp, decode at or above the current R9V numbers and prefill above llama.cpp. Published in the repo.
3. **Both file paths gated**: the same model loads from standard GGUF (Q4_K_M and Q8_0, parity activation path) and from a native file (per-token path), and both are in CI.
4. **Contributor lanes open with no core access**: a new model family, a T1 port to another arch, an n-gram-class proposer, and a quant-tool improvement can each land as a PR that touches only their lane's crate plus tests.
5. **A stranger can reproduce the receipt** from a README, a config file and `r9v bench`, and file a bundle-backed issue if they can't.

Anything short of all five is not the end of phase A.

## 2. Phase A — Foundation

### A0. Skeleton, toolchain, spikes

Deliverables

- Workspace with every crate from spec 14 §2 present (most empty), `rust-toolchain.toml`, `toolchain.toml`, `ci/Dockerfile`, `cargo deny` allowlist, DCO check.
- `r9v-hip`: dlopen binding for the dozen HIP entry points; `r9v --version` printing the five version numbers (spec 14 §8), working on a machine with no ROCm.
- `r9v-config`: the schema macro, `r9v config gen`, settings-index consistency check, unknown-key errors. Only the settings phase A uses are declared, but the machinery is complete.
- Hosted CI (`ci/cpu-only`) running fmt, clippy, deny, tests, docs build.
- `specs/` committed with `DECISIONS.md` seeded (spec 15 §9).
- **Spikes**, each a throwaway program run and judged by the lead implementing agent, with a recorded result in `spikes/<name>/RESULT.md`:
  - `wmma-l1`: an `iu8` and `iu4` GEMM whose B fragments load straight from global memory in the spec 2 `L1` lane order. Confirms the layout claim the whole format rests on, and measures the real `iu4` rate against `iu8` (spec 1 App. A says "verify").
  - `dot4-gemv`: a `v_dot4_i32_i8` GEMV over `L1` at M ∈ {1, 4, 8}; achieved GB/s against the streaming-read ceiling.
  - `fp8-wmma`: the fp8 builtins compile and run correctly on the pinned ROCm; confirms whether a leaf wrapper needs asm.
  - `hipgraph`: capture and replay of a 400-launch list on gfx1201; stability and dispatch overhead versus a plain launch list.
  - `direct-io`: `O_DIRECT` → pinned → H2D at queue depth 8; GB/s on the reference machine's NVMe.
  - `p2p`: whether the two R9700s can peer-map at all on this board; direct vs host-staged latency at 16 KB.

Exit: CI green; all six spike results recorded and judged pass/fail by the lead implementing agent. A contradictory result fails only the dependent line, produces a `SPEC-ISSUES.md` entry with the evidence and proposed correction, and does not stop independent lanes.

Size: small. This phase exists to make the next five cheap.

### A1. IR and the CPU reference

Deliverables

- `r9v-ir`: every spec 1 type, the full op catalog as typed op structs with attribute validation, sharding tables as data, the fusion table, `BatchMeta` with layer groups, the arch descriptor struct with the gfx1201 and CPU instances.
- `r9v-models`: the spec 8 builder API (sealed), `LayerSpec` in full, the generic layer builder emitting spec 8 §3.1 for every `norm` placement and mixer/FFN combination the spec lists, the `llama` family, weight binding, fusion and tied declarations, `ModelSummary`.
- `r9v-t0`: scalar T0 for every op in the catalog, including the ones phase A won't run on GPU (`moe_ffn`, `linear_attn_scan`, `ngram_gather`, collectives as local identity). The golden-test harness (spec 4 §10 items 1–3) and shape fuzzing.
- `r9v-state` in memory: block allocation, `reserve`/`commit`, layer groups, `BatchMeta` construction; tested against an in-memory pool.
- `r9v eval --logits` running a model on the CPU device end to end.

Exit: a 30M-parameter synthetic Llama-family checkpoint runs on T0 end to end; its logits match a torch implementation of the same architecture within spec 1 §6.1 tolerances; every op passes golden and batch-invariance tests against itself under padding and reordering; the partitioner-facing tables exist for every op even though no partitioner exists yet.

Contributor lane opened: **model families** (anyone can add a family against the builder and test it on T0).

### A2. Format and loader

Deliverables

- `r9v-format`: `L0`, `L1`, the native schemes (`I8_R`, `I8_B128`, `I4_K`, `E4M3_B128`), the repack-only schemes for `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `F16`/`BF16`, the container reader/writer with R9V type IDs and `r9v.*` metadata, the repack rules, the spec 2 §10 round-trip test on every scheme.
- `r9v-loader`: pipeline steps 1–7 for a single device; fingerprints; budget and refusal with numbers; the device arena; direct I/O with mmap fallback; the repack pipeline and cache; tokenizer and chat template; the load report.
- IQ4_NL/XS and IQ2/IQ3/IQ1 repack rules exist but are exercised only in the format tests (their kernels arrive with T1 in A3).

Exit: a real 27–30B Q4_K_M and Q8_0 GGUF and a hand-built native file all load into a device arena with a complete load report; second load hits the cache and is zero-copy; repack round-trip is bit-exact for every scheme; an infeasible load refuses before I/O with the shortfall.

Contributor lane opened: **format tooling** (readers, inspectors, converters in other languages against the spec 2 container).

### A3. Portable GPU tier and the minimal runtime

Deliverables

- `r9v-registry`: bundle manifest, tune-file reader, resolution order, validation flag, launch list replay, the profiling dispatch hook (disabled path only).
- `kernels/reference/`: T1 HIP for every op the dense path needs, plus the `moe_ffn`, `linear_attn_scan`, `causal_conv1d` and `ngram_gather` reference kernels so later phases promote rather than create. IQ LUT-expansion paths land here.
- `r9v-kgen` skeleton: the emission framework, `cost_model`, `search_space`, the ABI generator, and the gfx1201 leaf wrappers (`wmma_iu8`, `wmma_iu4`, `wmma_fp8`, `dot4_i8`, `cvt_e4m3`, `permlane_reduce`) with their builtin-vs-asm agreement tests. No T2 kernels yet.
- `r9v-sched` minimal: pre-step / device / post-step for a single sequence, step-graph capture bucketed on `(S=1, T_dec, T_pre)`, the workspace arena, stream and event discipline, `state_write_kv` and attention with paged KV, sampling with Philox, finish handling, the schedule log. `k = 0` (no proposer yet); prefill runs as chunks through the same step graph.
- Runner CI (`gpu/gfx1201`) online with the spec 14 §5.3 isolation, running spec 4 §10 gates for T1 and the spec 3 §8 tests on real pools.

Exit: the reference model decodes on gfx1201 through T1 with logits within L1 of T0 and bit-identical run to run; a 2K prompt prefills in chunks and continues decoding with correct KV; hosted and runner CI both required for merge.

Contributor lane opened: **T1 for other archs** (a descriptor plus the portable kernels compiling and passing golden on their hardware, receipt attached).

### A4. Receipts machinery

Deliverables

- `r9v-obs`: profiling `step` and `kernel` modes, metrics, the schedule-log export, the measurement pass (spec 11 §7) filling the descriptor's measured block and the topology, the doctor bundle with redaction, `r9v bench` with the `decode` and `prefill` suites and `--compare llama.cpp`, the receipt format with the achieved-bandwidth definition, `bench/baselines/`.
- The perf-regression gate wired into the runner against T1 baselines (they will be replaced by T2 baselines in A5, but the gate exists first).

Exit: a valid receipt for the reference model on the T1 path exists in the repo, including the llama.cpp comparison, with `utilization_meas` computed from a measured bandwidth. It will be slow. It is the first honest number.

This milestone is short and it comes before the fast kernels on purpose: A5 is judged by this machinery, and building the ruler after the thing it measures is how numbers get flattered.

### A5. Fast paths on gfx1201

Deliverables, in this order, each promoted only after passing spec 4 §10 and moving the receipt:

1. GEMV (`matmul`, M ≤ 8): `I8_*` and `I4_K`, `PerToken` and `PerBlock32`; then `I8_B32F`, `I4_B32F`, `I5_K`, `I6_K`.
2. Decode attention with split-KV, fp8 cache, the `state_write_kv → attention` fusion, tree mask support (unused until B).
3. Elementwise fusions from the spec 1 table (`residual_add → norm → quant_act`, matmul epilogues, `rope` fusions).
4. GEMM (`matmul`, M > 8): `iu8` and `iu4` WMMA from `L1` fragments, gate/up interleave, split-K deterministic partials.
5. Prefill attention.
6. Sampling kernels.
7. Autotune for the static set; tune files and generated source committed by the implementing agent; `hipGraph` vs launch-list choice at warmup.

Exit: `decode` receipt on the large dense reference model meeting the spec 11 §9.5 TG floor (`utilization_meas ≥ 0.93`, spec off) and at or above the current R9V decode figure at equal file; `prefill` receipt meeting the PP floor (≥ 1.45× the fastest llama.cpp backend on the same file at each of 2K and 8K, spec 11 §9.3/§9.5); both valid, both committed as baselines. Native-file and GGUF paths both have receipts. T2 coverage for the dense op set is complete; the tune coverage report shows no T1 fallbacks and no below-floor GEMV variants in the dense graph.

This is the milestone where the "numbers on my machine" problem is solved: the receipt carries the measurement pass, the byte breakdown, the comparison commit and the plan, and anyone with an R9700 can rerun it.

### A6. Contributor surfaces

Precondition: the A5 exit receipt exists and meets the floors. Speculative decoding (A6.1) is introduced to the optimized engine and measured as a multiplier over that receipt; it is not started, tuned or measured before A5 is closed.

Deliverables

- `r9v-spec`: the `Proposer` trait, `ngram` proposer, the `verify` op wired through the scheduler with the spec 6 §4.2 budget (`C_draft = 0`), the `decode-spec` and `accept` suites.
- `tools/r9v-quant` MVP: `quantize` for dense models with presets and `--fit`, folded smoothing, sensitivity assignment, GPTQ on the `I4_K` and `I8_B128` grids, `verify`, `compare`, `verify-arch`, the `r9v-cal-v1` manifest. Determinism check in hosted CI.
- The support matrix generator and `support/` entries for the reference dense models.
- The A6.7a API cohesion audit: one CPU integration fixture exercises metadata → graph → registry → load plan → scheduler → proposer/verify entirely through the spec-defined public surfaces.
- `CONTRIBUTING.md`, issue and RFC templates, the PR template question, `CREDITS.md`, `NOTICE`.
- Interface freeze: spec 1, 2, 4 (registry and tiers), 7 and 8 marked `accepted 1.0` per spec 15 §10.

Exit: the five criteria in §1 hold. A `compare` output exists showing native `I4_K` against `llama-quantize` `Q4_K_M` on the same source model, whichever way it comes out. Phase A is over.

### A. Explicitly deferred (built against, not built)

Multi-device (spec 5 beyond `ranks = 1`), prefix cache and host swap (spec 3 §3.4, §3.7), recurrent and conv state (spec 3 §4), multi-sequence scheduling and chunked-prefill admission across prompts, `mtp`/`draft`/`eagle` proposers, MoE and hybrid families, host experts and segments, the tiered slab and `ngram_gather` at runtime, the full serving API, the helper, replicas, releases. Every one of these has its interface in place at the end of A, its reference kernel in `kernels/reference/` where it needs one, and its state kind or op enumerated. None has a fast path or a receipt.

## 3. Phase B — Runtime

Milestones, in order:

- **B1. State manager complete**: prefix cache with refcounts, retention policies, host swap, recurrent double-buffer and session cache, `compact`. Gate: spec 3 §8 in full.
- **B2. Scheduler complete**: multi-sequence admission, serial and interleaved prefill, the budget with `max_wait_ms`, `k` selection with `C_draft`, memory-pressure pausing, the `multi` suite. Gate: `multi` receipt at `S ∈ {4, 8}` without forced admissions.
- **B3. Proposers**: `mtp` (with the hybrid family's MTP head), `draft` as a secondary load, `eagle`; tree verify with compaction; `decode-spec` receipts per proposer as multipliers over a floor-meeting spec-off `decode` receipt from the same session (spec 11 §9.5); `accept` suite per class. Any new proposer kind (including DFlash-style block drafters) follows the same rule.
- **B4. Families**: `moe` and `hybrid`, `causal_conv1d` and `linear_attn_scan` promoted to T2, `moe_ffn` T2 with the sorted grouped GEMM, MLA through `KvLatent`; `verify-arch` for each reference checkpoint.
- **B5. Multi-device**: topology and measurement, the partitioner as a pure function with golden per-rank graphs, comms over streams with `HostStaged` and `Direct`, PP2 (bit-identical to single device) and TP2 (cross-rank hash test), the planner with the latency and throughput profiles. Gate: spec 5 §7 in full on the reference rig.
- **B6. Host experts and tiers**: T0v (`matmul`, `moe_ffn` and the rest of its spec 4 scope), segments in the scheduler, the tiered slab, `host_fetch`, `ngram_gather` end to end on a table that doesn't fit VRAM. Gate: a MoE larger than one card's VRAM decodes with cold experts on the CPU at the planner's predicted cold rate.
- **B7. `depth` receipts** at 32K on the dense and hybrid reference models.

Interfaces added in B and frozen at its end: spec 3, 5, 6 (accepted 1.0).

Contributor lanes opened during B: **proposers** (after B3), **second-arch T2 generators** (after B5, when the descriptor-driven paths are proven on two plans), **MoE/hybrid families**.

## 4. Phase C — Surface

- **C1. Serving API**: the OpenAI-compatible routes, the `/r9v` routes, Unix socket, streaming with usage receipts, cancellation, grammar masks per verified position, tool-call and reasoning parsing, replicas router.
- **C2. Config schema complete**: every setting declared with docs, mutability enforced by the API, `r9v config gen` producing the shipped `r9v.toml`.
- **C3. Observability complete**: tracing, the full metrics list, the doctor bundle as the required issue attachment, nightly on the runner.
- **C4. Quant tool complete**: MoE and hybrid families, hot hints, `E4M3_B128` and activation-mode selection, `--sparse-check`, the quant determinism gate on a real small model.
- **C5. Release**: `cargo xtask release`, reproducible bundle, the tarball and wheel, generated `SUPPORT.md`, release notes with embedded receipts.

Exit: the spec 15 §11 phase C test — a stranger installs a release, loads a GGUF, gets a valid receipt and files a bundle-backed issue without asking anything. Specs 9–15 accepted 1.0.

## 5. Phase D — Helper and growth

- **D1. Helper**: T0v serving a small native model on the CPU device, release and live indexes, the tool list, the proposal flow against the config schema, draft issues.
- **D2. Second arch at reference tier** by an external contributor, with a committed receipt, and the descriptor-driven promotion path exercised by someone other than the owner.
- **D3. First external family and first external proposer merged** through the lanes without core changes.

Exit: the lanes have been used by people who didn't build them. That is the only proof that the boundaries are real.

## 6. Contributor boundaries by phase

| lane | crate / path | opens | acceptance gate |
|---|---|---|---|
| model families | `r9v-models` | A1 | spec 8 structural and reference-match tests |
| format tooling | against spec 2 (any language) | A2 | spec 2 round-trip tests |
| T1 for another arch | `kernels/reference`, one descriptor | A3 | spec 4 §10 gates plus contributor receipt |
| docs, FAQ, known issues | `docs/` | A0 | docs build |
| quant tool | `tools/r9v-quant` | A6 | spec 13 verification and determinism gates |
| proposers | `r9v-spec` | B3 | spec 7 acceptance suite and runner gates |
| T2 for another arch | `r9v-kgen` leaf wrappers + tune | B5 | spec 4 §10 gates and receipt |
| MoE / hybrid families | `r9v-models` | B4 | spec 8 structural and reference-match tests |
| core crates (`ir`, `format`, `state`, `registry`, `part`, `sched`) | — | never without an RFC or spec change | affected spec tests and both CI tiers |

A contribution that needs to cross a line in this table is, by definition, a spec conversation first.

## 7. Risks the sequence is designed around

- **The layout claim is wrong.** If A0's `wmma-l1` spike shows gfx12 fragments don't match `L1`, spec 2 changes before any loader code exists. This is why the spikes come first.
- **`iu4` isn't faster than `iu8`.** Then `I4_K` prefill runs at int8 rate and the format decision still holds (bytes are the TG win); only spec 1 App. A's rate table changes.
- **hipGraph is unstable on RDNA4.** The launch list is primary; nothing depends on hipGraph.
- **P2P is unavailable on the reference board.** Phase B's comms are host-staged by default; `Direct` is an upgrade, not a requirement.
- **The fast tier slips.** A4 exists so that a slow-but-honest receipt is published before A5 begins; the project has a real number at every point after A4.
- **Contributors arrive before the lanes are ready.** The table in §6 says when each opens; before that, the answer is "families and docs," which are genuinely useful and touch no core.

## 8. Ordering summary

```
A0 skeleton + spikes
A1 IR + T0 + builder            → families lane opens
A2 format + loader              → format tooling lane opens
A3 T1 + registry + minimal runtime + runner CI   → T1-arch lane opens
A4 receipts machinery           → first honest number
A5 fast paths                   → the receipt that beats current R9V
A6 proposer trait, quant MVP, API cohesion, freeze   → phase A done
B1 state → B2 scheduler → B3 proposers → B4 families → B5 multi-device → B6 host tiers → B7 depth
C1 API → C2 config → C3 obs → C4 quant → C5 release
D1 helper → D2 external arch → D3 external family and proposer
```
