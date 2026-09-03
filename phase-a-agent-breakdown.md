# Phase A — Agent Work Breakdown

Status: draft 0.1 (2026-09-03). Companion to `roadmap.md` and specs 1–15. This document turns phase A into task cards an agent can execute one at a time, with the dependency graph that says which can run in parallel.

## 0. How the cards work

Each card has: **id**, **crate(s)** it may touch, **deps** (cards that must be merged first), **spec** sections that define it, **deliverables** (concrete files, types, commands), **done when** (tests or artifacts that must exist and pass), **GPU** (whether the runner is required), **size** (S/M/L, relative).

Two card kinds:

- **API cards** produce the complete spec-defined public types, traits, function signatures, constructors, validation, doc comments, and API-shape tests needed by their named deliverables. They do not wait for approval; acceptance is the card's tests plus the mechanical rubric. Bodies may remain deferred only where a later card id explicitly owns them.
- **Implementation cards** fill in behavior owned by that card without redesigning the spec-defined public surface. Cross-crate cohesion is verified once all individual API-bearing cards complete by A6.7a.

## 1. Rules for agents

1. **Never edit `specs/`.** Dylan's decisions are the specs. A discrepancy, gap or ambiguity goes into `SPEC-ISSUES.md`: card id, spec section, what's unclear, the option taken, and the proposed correction. The spec controls wherever it is unambiguous; contradictory work stops only on the affected dependency line while independent cards continue.
2. **Stay inside the card's crates.** If the card needs a change elsewhere, that is a `SPEC-ISSUES.md` entry or a request for a new card, not an edit.
3. **No card lands without its "done when" tests.** Tests are part of the deliverable, not a follow-up.
4. **When the spec is silent, take the simplest option that satisfies every stated principle**, mark it in code with a doc comment `// DECISION(<card-id>): <one line>`, and list every such line in the PR description.
5. **Do not invent ops, dtypes, schemes, metadata keys, config settings or state kinds.** Those are closed sets; use `SPEC-ISSUES.md`.
6. **`unsafe` only in `r9v-hip` and the SIMD paths of `r9v-t0`.** Inline asm only in `r9v-kgen/src/leaf/`. `cargo deny` and a CI grep enforce both.
7. **Determinism applies to CPU code.** Fixed reduction order, no `HashMap` iteration in anything that affects output order, seeded RNG everywhere.
8. **Pure where the spec says pure**: the builder, the partitioner, the cost model, the planner. No I/O, no globals, no clocks inside them.
9. **New dependencies are listed in the PR** with one line of justification each; the allowlist in `deny.toml` is the gate.
10. **Read `CONVENTIONS.md` (card A0.4) before writing code.** Error types, logging fields, naming, test layout and fixture locations are defined there once.

## 2. Autonomous execution authority

No implementation card waits for human review or sign-off. The specs are the decisions; the card is the scope; tests, receipts, and CI are the acceptance authority.

The root agent follows `ORCHESTRATION.md`: it owns dispatch, acceptance, gap resolution, and forward motion; it verifies all subagent output rather than delegating acceptance.

- The lead implementing agent runs and judges the A0 spikes on the reference rig and records results in `spikes/<name>/RESULT.md`. A contradiction fails the dependent gate and is recorded in `SPEC-ISSUES.md`; independent work continues.
- API-bearing cards proceed directly when their dependencies and acceptance tests are satisfied. A6.7a performs the cross-crate cohesion audit before freeze.
- The implementing agent sets up or uses the self-hosted runner within spec 14 §5.3, stages reference-model fingerprints, and authorizes only trusted code for GPU execution. The `gpu-approved` label remains a security authorization for untrusted fork code, not an implementation review.
- `SPEC-ISSUES.md` records genuine contract conflicts. Clear spec text wins automatically; no approval is needed to implement it.
- The implementing agent runs `cargo xtask tune`, commits generated outputs, validates receipts, and updates baselines when their specified mechanical gates pass.

## 3. Dependency graph

```
A0.1 skeleton ──┬── A0.2 hip ─────────────────────────────┐
                ├── A0.3 config ─┐                        │
                └── A0.4 conventions/common ──┐           │
                     (lead agent: spikes S1–S6)│           │
                                              ▼           │
A1.1 ir-types(API) → A1.2 ir-ops → A1.3 models-builder(API) → A1.4 llama family
                        │                    │
                        ├── A1.5–A1.9 t0 op groups (parallel) ── A1.10 harness
                        ├── A1.11 state-core                       │
                        └── A1.12 cpu-executor + eval ── A1.13 torch-match test
                                              │
A2.1 layouts → A2.2 native schemes → A2.3 gguf schemes → A2.4 iq schemes
     │              └── A2.5 container ── A2.6 loader-open/bind/budget(API+impl)
     │                                          ├── A2.7 materialize (GPU)
     │                                          ├── A2.8 repack+cache
     │                                          ├── A2.9 tokenizer+template
     │                                          └── A2.10 load report
     ▼
A3.1 registry(API) → A3.2 abi-gen → A3.3–A3.7 T1 groups (parallel) → A3.10 gpu e2e
A3.8 kgen framework + leaf wrappers (GPU)           A3.9 sched-minimal(API+impl)
A3.11 runner CI (agent-owned)
                                              │
A4.1 obs core → A4.2 measure (GPU) → A4.3 doctor → A4.4 bench+receipt (GPU)
                                              │
A5.1 gemv → A5.2 decode attn → A5.3 fusions → A5.4 gemm → A5.5 prefill attn → A5.6 sampling
A5.7 gguf-scheme variants (after A5.1)     A5.8 tune commit (agent)     A5.9 receipts
                                              │
A6.1 proposer+ngram    A6.2–A6.6 quant tool (parallel from A2.5)    A6.7 support matrix → A6.7a API cohesion → A6.8 contributing + freeze
```

Parallel lanes once A1.2 exists: (a) T0 op groups, (b) format and loader, (c) T1 kernels once A3.1/A3.2 exist, (d) obs core, (e) quant tool once A2.5 exists. The serialized chain is A0 spikes → A3.8 → A3.10 → A4.2/A4.4 → A5.x → A5.9 because it needs the GPU runner; it does not wait for human review.

## 4. Cards

### A0

**A0.1 workspace skeleton** — crates: all (empty) — deps: none — spec: 14 §2, §3
Deliverables: `Cargo.toml` workspace with every crate from spec 14 §2 plus `r9v-common`, `rust-toolchain.toml`, `toolchain.toml`, `deny.toml`, `ci/Dockerfile`, `.github/workflows/cpu-only.yml` (fmt, clippy, deny, test, docs stub, asm grep, DCO), `xtask` with `docs` and `gen` stubs, `specs/` copied in, `DECISIONS.md` seeded from spec 15 §9, `SPEC-ISSUES.md` empty, `spikes/` with the six spike program skeletons.
Done when: `cargo build --locked` and `cargo test` pass in the container; CI green on an empty PR; `cargo deny check` passes. GPU: no. Size: S.

**A0.2 r9v-hip** — crates: `r9v-hip` — deps: A0.1 — spec: 14 §3
Deliverables: dlopen of `libamdhip64` with lazy symbol resolution for: device count/props, malloc/free, hostMalloc, memcpy async (H2D/D2H/D2D/peer), streams, events, module load/get function/launch, graph capture/instantiate/launch, peer access query/enable. A `Device` enum with `Cpu` and `Hip(rank)`. Error mapping to `r9v-common` errors.
Done when: unit tests run against a stub `libamdhip64` in hosted CI (symbol resolution, error mapping); a smoke test on the runner allocates, copies and launches an empty kernel; `r9v --version` runs on a machine with no ROCm. GPU: for the smoke test. Size: S.

**A0.3 r9v-config** — crates: `r9v-config` — deps: A0.1 — spec: 12 §2, §4, §5
Deliverables: `#[section]`/`#[setting]` macros; `Auto<T>`; precedence (defaults, file, env, CLI, runtime); source tracking per value; unknown-key errors with nearest key; `r9v config gen` producing `r9v.toml`, `docs/config.md`, JSON schema; the settings-index consistency check against `specs/spec-12-*.md` §3; cross-field rule framework. Declare the settings phase A uses (`load.*`, `io.*`, `host.*`, `warmup.*`, `state.*`, `scheduler.*`, `graph.*`, `kernels.*`, `spec.*`, `profile.*`, `log.*`, `doctor.*`, `bench.*`).
Done when: round-trip tests (file → effective → file); every declared setting has doc, range/enum, mutability; index check passes; a typo produces the expected error text. GPU: no. Size: M.

**A0.4 conventions and common** — crates: `r9v-common`, root — deps: A0.1 — spec: 15 §3 (PR template), 11 §11 (log format)
Deliverables: `CONVENTIONS.md` (error handling with `thiserror` per crate and a top-level `R9vError`, `tracing` fields `req_id`/`step_id`, naming, test layout, fixture directories, how to write a `DECISION` comment, how to file `SPEC-ISSUES.md`); `r9v-common` with the error type, ids (`SeqId`, `ReqId`, `StepId`), xxh3 helper, byte-size parsing, seeded RNG helper. PR template with the "which spec" question and the `DECISION` list section.
Done when: every other crate compiles against `r9v-common`; `CONVENTIONS.md` is linked from the README. GPU: no. Size: S.

**A0.S1–S6 spikes** (lead implementing agent drafts, runs and judges) — `spikes/` — deps: A0.2 — spec: roadmap §A0
`wmma-l1`, `dot4-gemv`, `fp8-wmma`, `hipgraph`, `direct-io`, `p2p`. Each: a single HIP or Rust program, a `RESULT.md` template with the numbers the spec depends on (fragment order match yes/no, `iu4`/`iu8` ratio, GB/s, µs per launch, peer-map yes/no).
Done when: every result is recorded with command, hardware fingerprint, raw measurements, and a pass/fail judgment against the named spec claim; contradictions are filed in `SPEC-ISSUES.md` and fail only the dependent gate. GPU: yes. Size: S each.

### A1

**A1.1 ir types (API)** — crates: `r9v-ir` — deps: A0.4 — spec: 1 §2, App. A
Deliverables: `DType`, `QuantScheme` (with `Scheme(SchemeId)` referencing `r9v-format` ids as an opaque newtype to avoid a cycle), `Tensor`, `Placement`, `ShardLayout`, `Class`, `BatchMeta` (with `G` groups, `slot_map [G,T]`, `block_table [G,S,max_blocks]`, `window_start`, `TreeMask`), `StateHandle`/`StateKind`, `ArchDescriptor` with `measured` block, `gfx1201()` and `cpu()` constructors, `IrVersion`.
Done when: compiles with doc comments on every public item; API-shape tests pass and A6.7a can consume the surface without a private escape hatch. GPU: no. Size: S.

**A1.2 ir ops and graph** — crates: `r9v-ir` — deps: A1.1 — spec: 1 §3, §4, §5, §6
Deliverables: one typed struct per op with attributes and a `validate()` that enforces shape/dtype constraints from §4; sharding tables as data (`legal_layouts(op) -> &[(inputs, output)]`); the fusion table as data; `Graph` DAG with typed edges, external inputs/outputs, `copy` insertion on stride mismatch with a graph-summary report; step-graph key `(plan_id, rank, S, T_dec, T_pre, segment)`; bucket functions; the numerics contract as a `Numerics` struct per op (accumulator dtype, reduction order tag) used by tests.
Done when: every op in spec 1 §4 exists; `validate()` tests for accepted and rejected shapes per op; sharding-table tests that every op has at least one legal tuple; a graph with a stride mismatch reports exactly one inserted `copy`. GPU: no. Size: M.

**A1.3 models builder (API + impl)** — crates: `r9v-models` — deps: A1.2 — spec: 8 §2, §3, §3.1, §5, §7
Deliverables: sealed `GraphBuilder`; `LayerSpec`, `Mixer`, `Ffn`, `ModelSpec`, `NormPlacement`, `RopeSpec`, `MlaSpec`, `MtpSpec`, `NgramSpec`; the generic layer builder emitting spec 8 §3.1 for every `norm` placement × mixer × ffn combination, including `LinearAttention`, `Moe`, `mla`, `output_gate`, `ngram` injection and the MTP subgraph; `declare_fusion`, `declare_tied`, `export`; `ModelSummary` computation; `GgufMeta` trait (key lookup by name, typed) so `r9v-models` never depends on the container crate.
Done when: a structural test builds every combination with synthetic metadata and checks op counts and edge dtypes against expected tables; `ModelSummary` matches hand-computed bytes on a synthetic model; the public surface passes its API-shape tests. GPU: no. Size: L.

**A1.4 llama family** — crates: `r9v-models` — deps: A1.3 — spec: 8 §4, §5, §9
Deliverables: `families/llama.rs` reading the `llama`, `mistral`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, `phi3`, `olmo2` key sets (explicit list of `<arch>.*` keys per value, in a table in the file), weight name binding, fusion declarations, tied handling; family registry keyed by `general.architecture`.
Done when: synthetic `GgufMeta` fixtures for each architecture value build without error and produce the expected `LayerSpec` list (golden JSON per fixture). GPU: no. Size: M.

**A1.5 t0 elementwise group** — crates: `r9v-t0` — deps: A1.2 — spec: 1 §4.B, §6.4, 4 §2
Deliverables: scalar f32 `norm` (Rms/Layer, Last/Head, weight_offset), `residual_add`, `act_mul`, `activation`, `rope` (both styles, all scalings, mrope), `cast`, `copy`, `quant_act` (i8 and e4m3, PerToken and PerBlock32).
Done when: each op has a property test against a straightforward f64 implementation and a batch-invariance test. GPU: no. Size: M.

**A1.6 t0 matmul and lookup group** — crates: `r9v-t0` — deps: A1.2, A2.2 (scheme decoders) — spec: 1 §4.A, §4.C, §6.2
Deliverables: `matmul` for every activation scheme × weight scheme with exact i32/f32 accumulation semantics from §6.2 (full-K i32 for PerToken, per-32 for PerBlock32, zero-point form for `I4_K`), `embed_gather` from `L0` and from tiled `L1` rows, `gather_rows`, `scatter_add_rows` (sorted).
Done when: matmul results match an f64 reference within §6.1 tolerances for every scheme; accumulation-order tests show bit-identical results across `T`. GPU: no. Size: M.

**A1.7 t0 attention group** — crates: `r9v-t0` — deps: A1.2, A1.11 — spec: 1 §4.D, §6.3, 3 §3
Deliverables: `state_write_kv` into paged blocks with per-token-head scales for e4m3/i8/f16 caches and the `KvLatent` form; `attention` over block tables with Causal, CausalWindow(+sinks), Tree masks, softcap, MLA; online softmax in f32 in ascending block order.
Done when: matches a dense f64 attention on random sequences for every mask kind; tree-mask test against explicit ancestor sets; cache-dtype round-trip tolerance tests. GPU: no. Size: M.

**A1.8 t0 sampling group** — crates: `r9v-t0` — deps: A1.2 — spec: 1 §4.F, §6.5, 7 §4
Deliverables: `logits_postprocess` (all params, `logit_bias`, grammar mask, stable sort), `sample` with Philox4x32 keyed `(seq, step, draw)`, `verify` with `Rejection`, `Greedy`, `Typical` and tree walk.
Done when: distribution tests (rejection sampling output frequency matches target sampling within statistical tolerance on synthetic distributions, 1e5 draws); determinism tests; greedy equivalence at temperature 0. GPU: no. Size: M.

**A1.9 t0 deferred-op group** — crates: `r9v-t0` — deps: A1.2, A1.11 — spec: 1 §4.C (moe), §4.E, §4.A (ngram), §4.G
Deliverables: `moe_route`, `moe_ffn` (sorted, deterministic combine), `causal_conv1d`, `linear_attn_scan` (chunked and recurrent forms, GatedDeltaNet/GLA/Mamba2), `ngram_gather` (both placement cases), collectives as local identity/sum for `ranks = 1`.
Done when: chunked and recurrent scan forms agree within tolerance on random inputs; moe combine order test; conv state continuity test across steps. GPU: no. Size: M.

**A1.10 test harness** — crates: `r9v-t0`, `r9v-ir` (test utils) — deps: A1.5–A1.9 — spec: 4 §10, 1 App. B
Deliverables: a generic harness that, given an op instance and an implementation under test, runs golden (vs T0), batch invariance (alone / padded / embedded), determinism (twice), and shape fuzz; seeded fixture generators for every tensor kind and scheme; tolerance table per op as data (spec 1 §6.1 defaults).
Done when: the harness runs T0 against itself for every op and passes; fixtures are deterministic across runs. GPU: no. Size: M.

**A1.11 state core** — crates: `r9v-state` — deps: A1.1 — spec: 3 §2, §3.1–3.3, §3.5–3.6, §5, §6
Deliverables: `StateSpec`, layer grouping, pools as offset arithmetic over an abstract arena, block allocation (deterministic free list), `new_seq` (no prefix cache yet: `matched_len = 0`), `reserve`, `commit`, `free_seq`, retention policies with `window_start`, `batch_meta` builder, `budget`, `stats`; `compact` returning an op descriptor; recurrent double-buffer slot bookkeeping (A/B swap), session-cache and prefix-cache as `todo!()` behind the same API.
Done when: spec 3 §8 commit and window tests against an in-memory arena; deterministic-allocation test (same request history → identical block ids); `BatchMeta` shape tests for multi-group models. GPU: no. Size: M.

**A1.12 cpu executor and eval** — crates: `r9v-t0`, `r9v` (CLI) — deps: A1.2, A1.5–A1.11 — spec: 4 §2 (T0 device), 14 §10
Deliverables: a graph interpreter that runs a `Graph` on the CPU device over T0 ops with `BatchMeta` and state; `r9v eval --logits --model <path> --tokens <file>` writing logits as `.npy`; single-sequence decode loop on CPU (greedy) for tests.
Done when: a synthetic model decodes deterministically on CPU; logits file round-trips. GPU: no. Size: M.

**A1.13 torch-match test** — crates: `tools/r9v-quant/tests` (python), `tests/` — deps: A1.12, A1.4, A2.5 (to write a fixture GGUF) — spec: 8 §8, 13 §12
Deliverables: a python script generating a 30M-parameter Llama-family checkpoint with random weights as an F16 GGUF (using `gguf-py`) plus its torch forward; a test that runs `r9v eval` and torch on 64 fixed token sequences and compares within spec 1 §6.1.
Done when: the comparison passes in hosted CI (CPU, minutes). GPU: no. Size: S.

### A2

**A2.1 layouts** — crates: `r9v-format` — deps: A0.4 — spec: 2 §2
Deliverables: `LayoutId` (`L0`, `L1`, `L1S`), the `L1` lane-order permutation and its inverse for every packing (i4 nibbles, i8/e4m3/e5m2, f16/bf16, bit-plane types), tile/row-block indexing, padding rules, `L1S` index region layout (operand order per A0.S1 result; `SPEC-ISSUES` if the spike changed it).
Done when: permute → inverse is identity for random tensors of every dtype and shape class; documented lane formula matches spec 2 §2.2 by test. GPU: no. Size: M.

**A2.2 native schemes** — crates: `r9v-format` — deps: A2.1 — spec: 2 §3.1, §3.2, §8
Deliverables: `SchemeId` enum (all ids from §3.2 and §3.3, reserved), scale-record structs and SoA placement for `I8_R`, `I8_B128`, `I4_K`, `E4M3_B128`; reference dequant `decode(scheme, values, scales) -> f32` used by T0 and tests; bpw calculator.
Done when: encode → decode of synthetic weights is within the scheme's expected error; `I4_K` record packing is bit-compatible with `Q4_K`'s field layout (test against `gguf-py`'s reference). GPU: no. Size: M.

**A2.3 gguf repack-only schemes** — crates: `r9v-format` — deps: A2.2 — spec: 2 §3.3, §7, §10
Deliverables: `ggml_type → SchemeId` mapping; repack rules (pure permutations plus bit-plane regrouping) for `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q3_K`, `Q2_K`, `F16`, `BF16`; reference dequant for each.
Done when: the spec 2 §10 round-trip test (dequant of source bytes == dequant of repacked bytes, bit-exact) passes on fixtures produced by `gguf-py` quantize functions for every type. GPU: no. Size: M.

**A2.4 iq schemes** — crates: `r9v-format` — deps: A2.3 — spec: 2 §3.3
Deliverables: repack rules preserving codebook indices and scales for `IQ4_NL`, `IQ4_XS`, `IQ3_*`, `IQ2_*`, `IQ1_*`; the LUTs as data; reference dequant.
Done when: same round-trip test on `gguf-py`-produced fixtures. GPU: no. Size: M.

**A2.5 container** — crates: `r9v-format` — deps: A2.2 — spec: 2 §6, §9, §10, 9 §3
Deliverables: GGUF v3 reader (metadata KV types, tensor info, alignment, split shards) and writer; R9V type IDs 1000–1099; `r9v.*` key schema as typed accessors; region offsets; `xxh3` per entry; `file_fp` and `model_fp`; format-version acceptance rule.
Done when: reads a real llama.cpp-produced GGUF's metadata and tensor table; writes a native file that `gguf-py` can parse (header/metadata level); fingerprint stability tests. GPU: no. Size: M.

**A2.6 loader open/bind/plan/budget** — crates: `r9v-loader` — deps: A2.5, A1.3, A1.4, A1.11 — spec: 9 §2 steps 1–4, §4, §12
Deliverables: pipeline steps 1–4 with a single-device `Plan` (a local `plan_single_device()` until `r9v-part` exists, behind the spec 5 §5.1 `Plan` type defined in `r9v-ir`); budget computation per spec 9 §4 and spec 3 §6.3; refusal messages with shortfall and suggested change; validation per spec 8 §6 collecting all failures.
Done when: infeasible synthetic configs refuse with the exact numbers; missing-tensor fixtures list every missing name; steps 1–4 complete without touching tensor data (test with a truncated file). GPU: no. Size: M.

**A2.7 materialization** — crates: `r9v-loader`, `r9v-hip` — deps: A2.6, A0.2 — spec: 9 §4.1, §5.1, §5.4
Deliverables: device arena allocation (one `hipMalloc`), pinned staging ring, `O_DIRECT` reader with mmap fallback, H2D on the copy stream overlapped with reads, `Host` destination path; a `FakeDevice` implementing the same trait for hosted tests.
Done when: hosted tests with `FakeDevice` verify byte placement and alignment; runner test loads a real GGUF into VRAM and reads back a sampled tensor bit-exact; throughput logged. GPU: yes. Size: M.

**A2.8 repack pipeline and cache** — crates: `r9v-loader` — deps: A2.3, A2.4, A2.7 — spec: 9 §5.2, §5.3
Deliverables: thread-pool repack per row-block; interleave on `gate_up`/`qkv` declarations; cache directory with `manifest.json` and `weights.gguf`; cache key; per-tensor `xxh3` recorded; second-load zero-copy path.
Done when: first load repacks and writes the cache, second load reads only metadata from the source (verified by file access instrumentation); repack throughput ≥ 2 GB/s per core on the runner; cache invalidates on `file_fp` change. GPU: for the throughput number. Size: M.

**A2.9 tokenizer and chat template** — crates: `r9v-loader` (default; a separate `r9v-tok` crate requires a `SPEC-ISSUES` entry showing why the specified ownership cannot work) — deps: A2.5 — spec: 9 §7, 10 §3.1
Deliverables: BPE, SentencePiece and WordPiece tokenizers from `tokenizer.ggml.*`; special tokens, add-BOS, pre-tokenizer rules by `tokenizer.ggml.pre`; incremental detokenizer for stop strings; Jinja subset renderer with the llama.cpp-compatible feature set and no filesystem/network.
Done when: tokenization matches `gguf-py`/llama.cpp output on a corpus for each tokenizer type in the reference set; template rendering matches for the reference models' templates; detokenizer handles split UTF-8. GPU: no. Size: L.

**A2.10 load report and progress** — crates: `r9v-loader` — deps: A2.7, A2.8 — spec: 9 §10, 10 §2.2 (progress)
Deliverables: `LoadReport` struct serialized to JSON and rendered to the log; progress events per pipeline step and per tensor on a channel.
Done when: report contains every field in spec 9 §10 for a real load; snapshot test on a synthetic load. GPU: no. Size: S.

### A3

**A3.1 registry (API + impl)** — crates: `r9v-registry` — deps: A1.2, A0.2 — spec: 4 §3, §7, §9, §11, §12
Deliverables: `VariantKey`, `OpStatic` per family, `variant_hash`; bundle manifest reader; tune-file reader/writer; resolution order with `validated` flag; `hipModuleLoadData` on demand; `LaunchList` record and replay; the dispatch hook with the profiling branch (recording to a sink trait; obs implements it in A4.1); `allow_jit` gate (JIT itself lands with A3.8).
Done when: resolution unit tests over synthetic manifests (shipped / local / T1 fallback / unlisted arch refusal); launch-list replay determinism test with a stub device; public API-shape tests pass. GPU: for module load smoke. Size: M.

**A3.2 abi generator** — crates: `r9v-kgen` — deps: A3.1 — spec: 4 §7
Deliverables: per-op args struct generation (Rust side and HIP side from one description), 256-byte alignment assumptions, workspace slots, `BatchMeta` field selection per op.
Done when: generated HIP structs compile; Rust/HIP layout equality test via `offsetof` dump. GPU: compile only. Size: S.

**A3.3–A3.7 T1 kernel groups** — crates: `kernels/reference`, `r9v-registry` (tests) — deps: A3.2, A1.10 — spec: 4 §2, §10; 1 §4
Five cards mirroring A1.5–A1.9: portable HIP (wave intrinsics only) for each op group. `matmul` covers every scheme including IQ LUT expansion to i8; attention covers paged KV, all masks, all cache dtypes; sampling implements Philox on device.
Done when: each group passes the A1.10 harness against T0 on the runner (golden, invariance, determinism, fuzz); no arch-specific builtins (CI grep). GPU: yes. Size: M each (attention L).

**A3.8 kgen framework and gfx1201 leaves** — crates: `r9v-kgen` — deps: A3.2, spikes — spec: 4 §4, §6, §8
Deliverables: emission building blocks (fragment loaders per `LayoutId`×dtype, epilogues, reduction trees, softmax stages) with LDS/VGPR cost accounting; `cost_model`, `search_space` types, `TileConfig`; `clang++` invocation with pinned flags; leaf wrappers for gfx1201 (`wmma_iu8`, `wmma_iu4`, `wmma_fp8`, `dot4_i8`, `cvt_e4m3`, `swmmac_iu8`, `permlane_reduce`) with builtin/asm switch and agreement tests; JIT path (`hiprtc` or subprocess `clang++`, per A0.S3) behind `allow_jit`; autotune loop (§6.1) writing local tune files. No op emitters yet beyond a trivial `copy` used to test the pipeline end to end.
Done when: the `copy` emitter round-trips through emit → compile → tune → validate → registry on the runner; leaf agreement tests pass; cost model unit tests. GPU: yes. Size: L.

**A3.9 scheduler minimal (API + impl)** — crates: `r9v-sched` — deps: A1.11, A3.1, A1.2 — spec: 6 §2, §3, §5, §7, §9 (single sequence; §4 with `S = 1`, `k = 0`)
Deliverables: `Request`/`Sequence`/`Step`; pre-step → device → post-step loop; step-graph capture per `(S=1, T_dec, T_pre)` from the builder against the registry; workspace arena; three streams and event chain; chunked prefill through the step graph; sampling and readback; finish handling with EOS/max_tokens/stop strings; schedule log ring; `step_budget_ms` resolution stub reading a cost table; the `Proposer` call sites present but empty (`k = 0`).
Done when: hosted simulation with a stub device and fake cost table runs 1000 steps deterministically; public API-shape tests pass. GPU: no for the simulation. Size: L.

**A3.10 gpu end-to-end** — crates: `tests/gpu` — deps: A3.3–A3.7, A3.9, A2.7, A2.8 — spec: roadmap §A3
Deliverables: runner tests: load the small dense reference model (GGUF and native), decode 64 tokens greedy through T1, compare logits to `r9v eval` T0 within L1; run twice, compare bit-exact; prefill a 2K prompt in 128-token chunks and continue decoding, compare to a single-chunk run bit-exact.
Done when: all three pass on the runner and are required checks. GPU: yes. Size: M.

**A3.11 runner CI** — the implementing agent owns runner setup and writes `.github/workflows/gpu-gfx1201.yml`, the isolation scripts (spec 14 §5.3), the model staging manifest with fingerprints, and the PR comment reporter. Untrusted fork execution still requires explicit security authorization.
Done when: a PR from the main repo runs the A3.10 suite and posts results; a fork PR does nothing until labeled. GPU: yes. Size: S (agent part).

### A4

**A4.1 obs core** — crates: `r9v-obs` — deps: A3.1, A3.9 — spec: 11 §2–§5, §11
Deliverables: profiling modes implementing the registry's sink trait; Prometheus metrics registry with the spec 11 §4 names phase A can populate; JSON-lines logging setup with `req_id`/`step_id`; Perfetto trace writer for step and kernel modes; schedule-log export.
Done when: metrics text snapshot test; trace file validates against the Perfetto JSON schema; overhead test shows `step` mode adds one event per step. GPU: no. Size: M.

**A4.2 measurement pass** — crates: `r9v-obs`, `r9v-hip` — deps: A0.2, A3.8 (a streaming kernel) — spec: 11 §7, 1 App. A `measured`, 5 §2
Deliverables: `r9v doctor measure` implementing every row of spec 11 §7 (single device plus the p2p matrix for whatever devices exist), the hardware fingerprint, the cache under `~/.cache/r9v/measure/`, invalidation on component change; `rocm-smi` capture.
Done when: on the runner, produces a `measured` block and topology JSON; a second run hits the cache; a forced fingerprint change re-measures. GPU: yes. Size: M.

**A4.3 doctor bundle** — crates: `r9v-obs`, `r9v` — deps: A4.1, A4.2, A2.10 — spec: 11 §8
Deliverables: `r9v doctor` producing the tarball with every file in spec 11 §8 that phase A has (`incident.json` from the fault path, `batchmeta/` from the ring); redaction; `include_tokens` gating.
Done when: bundle contents snapshot test; redaction test finds no absolute paths, hostnames or env values; size under 10 MB on a real run. GPU: no. Size: S.

**A4.4 bench and receipts** — crates: `r9v-obs`, `r9v`, `bench/` — deps: A4.2, A3.10 — spec: 11 §6, §9, §10
Deliverables: `r9v bench --suite decode|prefill` with fixed seeded prompts, warmups and repeats, p50/p95, thermal recording, validity rules; achieved-bandwidth computation per spec 11 §6 from registry statics and `BatchMeta`; `--compare llama.cpp:<dir>,<dir>` running `llama-bench` for every llama.cpp backend built for the machine (HIP and Vulkan at minimum, same commit) with recorded commit/flags/command line per build, and the fastest per workload used as the comparison figure; `receipt.json` + `receipt.md`; `bench/baselines/` writer and the > 3% regression check wired into the runner workflow.
Done when: a valid receipt for the small and large dense reference models on the T1 path is committed under `bench/baselines/` with the llama.cpp comparison; the regression check fails a deliberately slowed variant. GPU: yes. Size: M.

### A5

Each A5 card: emitter in `r9v-kgen`, search space, cost model entries, autotune on the runner, spec 4 §10 gates, tune entry, generated source committed, receipt delta recorded in the PR.

**A5.1 GEMV emitter** — deps: A3.8, A4.4 — spec: 4 §5.2, 2 §2.2, 1 §6.2
`I8_R`/`I8_B128`/`I4_K` × `PerToken`/`PerBlock32`, M ≤ 8, `dot4` path, kgroup permlane reduce, K-split LDS reduce in fixed order, M-batched activations in LDS. Done when: gates pass; `decode` receipt improves and the achieved GB/s is within 5% of A0.S2's spike number. GPU: yes. Size: L.

**A5.2 decode attention emitter** — deps: A5.1 — spec: 4 §5.3 (decode), §5.4, 3 §3.2, 1 §3.4
Split-KV partials and deterministic merge, `attention_layout` for gfx1201 recorded in the descriptor, fp8/i8/f16 caches, tree masks, the `state_write_kv → attention` fusion. Done when: gates pass including tree-mask golden tests; `depth`-style internal test at 8K shows attention share in the trace. GPU: yes. Size: L.

**A5.3 elementwise fusion emitters** — deps: A5.1 — spec: 1 §3.4, 4 §5.7
`residual_add → norm → quant_act`, matmul epilogues (bias/residual/act/gated), `rope` fused into `state_write_kv` and into prefill attention's Q load (the latter lands with A5.5). Done when: fused variants pass gates against the unfused T0 composition; launch count per layer in the trace drops to the spec 8 §3.1 minimum. GPU: yes. Size: M.

**A5.4 GEMM emitter** — deps: A5.3 — spec: 4 §5.1, 2 §2.2
`iu8`/`iu4`/`fp8`/`f16` WMMA with B fragments direct from `L1`, A staged in LDS, gate/up interleave, split-K deterministic partials + tree reduce kernel, `PerBlock32` rescale variant. Done when: gates pass; `prefill` receipt improves; `L1S` path compiles and passes on a synthetic 2:4 tensor. GPU: yes. Size: L.

**A5.5 prefill attention emitter** — deps: A5.4 — spec: 4 §5.3 (prefill)
Flash-attention-2 structure, `BQ` tiles, block skipping for causal/window, fused rope in the Q load. Done when: gates pass; `prefill` receipt at 8K improves. GPU: yes. Size: L.

**A5.6 sampling emitters** — deps: A5.1 — spec: 4 §5.8
`logits_postprocess → sample` fused, `verify`, bitonic sort in LDS chunks, Philox. Done when: gates pass including the distribution tests from A1.8 run on device. GPU: yes. Size: M.

**A5.7 gguf-scheme GEMV/GEMM variants** — deps: A5.1, A5.4 — spec: 2 §3.3, 1 §4.C
`I8_B32F`, `I4_B32F`, `I4_B32FM`, `I5_*`, `I6_K`, `I4_NL/XS` (LUT expand → `iu8`). Done when: gates pass; `decode` receipt on the Q4_K_M and Q8_0 GGUF files is within the parity target (≥ llama.cpp same file). GPU: yes. Size: M.

**A5.8 tune commit** (agent-owned) — `cargo xtask tune --arch gfx1201` for the static set; `hipGraph` vs list measured at warmup; commit `tune/gfx1201/<gen_version>.toml` and `kernels/gen/gfx1201/**`. Done when: hosted CI's regenerate-and-diff passes; tune coverage report shows zero T1 fallbacks in the dense graph. GPU: yes.

**A5.9 receipts** — deps: A5.1–A5.8 — spec: 11 §9, 15 §5
`decode` and `prefill` receipts on the large dense model for native and GGUF paths, same-file compared; baselines updated by PR. Done when: the `decode` receipt (spec off) meets the spec 11 §9.5 TG floor (`utilization_meas ≥ 0.93`) and is ≥ the current R9V figure at equal file; the `prefill` receipt meets the PP floor (≥ 1.45× the fastest llama.cpp backend, HIP and Vulkan at the pinned commit, on the same file, at each of 2K and 8K); no below-floor GEMV variants in the dense graph; both valid; README numbers generated from the receipts by `xtask docs`. GPU: yes. Size: S.

### A6

**A6.1 proposer trait and ngram** — crates: `r9v-spec`, `r9v-sched` — deps: A3.9, A5.6, **A5.9 (floor-meeting receipt must exist first)** — spec: 7 §2, §4, §6 (`ngram`), 6 §4.2, 11 §9.5
Deliverables: `Proposer` trait exactly as spec 7 §2; `ngram` proposer; scheduler `k` selection with `C_draft = 0` and `accept_ema`; `verify` wired into the step graph; `decode-spec` and `accept` suites in bench.
Done when: on the runner, a `decode-spec` receipt with `ngram` on the code half of the `accept` set shows tokens/step > 1, reported as a multiplier over the A5.9 spec-off `decode` receipt from the same session; the output distribution test (A1.8) holds end to end; `k` drops to 0 on a random-token prompt. GPU: yes. Size: M.

**A6.2 quant tool skeleton** — crates: `tools/r9v-quant` — deps: A2.5 — spec: 13 §2, §3, §13, §14
Deliverables: package with `uv` lockfile; `gguf` read/write of standard and native containers; torch `llama` family forward (f16/f32); calibration manifest schema, `cal build`, `r9v-cal-v1` manifest; CLI scaffold.
Done when: torch forward of the 30M fixture matches `r9v eval` (reuses A1.13); `cal build` materializes a manifest deterministically (hash test). GPU: no. Size: M.

**A6.3 statistics and smoothing** — deps: A6.2 — spec: 13 §4 steps 1–3, §5
Deliverables: streamed activation stats (absmax, Hessian diagonal) per matmul input; folding table from spec 13 §5 implemented for every group in the llama family; layer-at-a-time device residency.
Done when: a folded model's f16 logits equal the unfolded model's within f16 noise (the fold is exact in real arithmetic); memory test shows only one layer resident on the device. GPU: optional. Size: M.

**A6.4 sensitivity and assignment** — deps: A6.3 — spec: 13 §6
Deliverables: `ε` proxy per tensor; greedy promotion against a byte budget; presets as KL ceilings; `--fit` using spec 3 §6.2 costs; the per-layer mix table.
Done when: assignment is monotone in budget on the fixture; `--fit` produces a file whose load report shows the requested `max_ctx` fits. GPU: optional. Size: M.

**A6.5 rounding** — deps: A6.4, A2.2 — spec: 13 §7
Deliverables: GPTQ with Cholesky error feedback and act-order; the `I4_K` importance-weighted grid fitter; `I8_*` and `E4M3_B128` paths; torch determinism settings.
Done when: byte-identical output across two runs on the 30M fixture (hosted CI); KL on the fixture lower than round-to-nearest at the same scheme. GPU: optional. Size: L.

**A6.6 emit, verify, compare, verify-arch** — deps: A6.5, A1.12, A5.9 — spec: 13 §8–§12
Deliverables: activation-mode selection; native emission with every `r9v.*` key from spec 2 §6 and interleaving from spec 8 declarations; `verify` calling `r9v eval` (device or CPU) and writing `r9v.quality.*`; `compare`; `inspect`; `verify-arch` committing `support/<family>/<model>.json`.
Done when: a native file from the large dense reference model loads zero-copy (load report says so) and its `compare` against `llama-quantize` Q4_K_M is committed; `verify-arch` entries exist for every reference dense checkpoint. GPU: yes for the real model. Size: M.

**A6.7 support matrix** — crates: `xtask`, `docs/` — deps: A6.6, A5.9 — spec: 15 §6
Deliverables: `xtask support-matrix` generating `SUPPORT.md` from `support/`, `tune/`, the bundle manifest and `bench/baselines/`, using the three defined words.
Done when: the generated file lists the reference dense models as `supported` with receipt links and everything else as `untested`. GPU: no. Size: S.

**A6.7a API cohesion audit** — crates: integration tests and public API surfaces of `r9v-ir`, `r9v-models`, `r9v-loader`, `r9v-registry`, `r9v-sched`, `r9v-spec` — deps: A1.1, A1.3, A2.6, A3.1, A3.9, A6.1 — spec: 1 §2–§5, 6 §2–§5, 7 §2–§3, 8 §2–§3, 9 §2
Deliverables: a compile-time integration fixture that constructs the spec-defined path from metadata → model graph → registry resolution → load plan → scheduler step → proposer/verify without reaching into private fields; a public-surface inventory mapped to owning spec sections; removal of redundant public escape hatches discovered by the fixture, without changing any closed set.
Done when: the fixture builds and runs on `ci/cpu-only`; every public item in the six API-bearing crates maps to a named cross-crate consumer and spec section; no integration requires a downcast, stringly typed closed-set value, duplicate type, or private-field access. GPU: no. Size: M.

**A6.8 contributing and freeze** — root, `docs/` — deps: everything including A6.7a — spec: 15 §2–§4, §10
Deliverables: `CONTRIBUTING.md`, `CREDITS.md`, `NOTICE`, `SECURITY.md`, issue templates (bug with bundle requirement, performance with receipt requirement, model request, RFC per closed set), PR template; the freeze PR that marks specs 1, 2, 4 (registry and tiers), 7 and 8 `accepted 1.0` after CI confirms `SPEC-ISSUES.md` has no unresolved entry affecting those interfaces.
Done when: the five criteria in `roadmap.md` §1 are each linked to the artifact that proves them. GPU: no. Size: S.

## 5. Card count and shape

66 executable cards when the six spikes and five T1 kernel groups are counted individually: 10 in A0, 13 in A1, 10 in A2, 11 in A3, 4 in A4, 9 in A5, and 9 in A6 including A6.7a. Each card declares whether its acceptance needs the runner; all others run in hosted CI. Six cards carry public API work (A1.1, A1.3, A2.6, A3.1, A3.9, plus the `Proposer` trait inside A6.1); none has a human review gate, and A6.7a checks their cohesion before freeze.

The critical path is A0.S1–S6 → A3.8 → A3.10 → A4.2 → A4.4 → A5.1 → A5.2 → A5.4 → A5.5 → A5.8 → A5.9. Everything else runs beside it.
