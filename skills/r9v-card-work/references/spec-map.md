# Spec map

Which spec and which crate own what. Use this to find the section a card refers to and to check whether something you're about to add belongs to a closed set.

## Specs and crates

| spec | title | owns | crate(s) |
|---|---|---|---|
| 1 | Op IR | op set, `Tensor`, `BatchMeta`, state handles, step graph and buckets, fusion table, sharding layouts, numerics contract, arch descriptor (App. A), determinism levels (App. B) | `r9v-ir` |
| 2 | Tensor Layout and Weight Format | `L0`/`L1`/`L1S`, quant schemes (native and repack-only), activation metadata, structural flags, placement classes, GGUF container and `r9v.*` keys, repack rules, versioning | `r9v-format` |
| 3 | Sequence State | state kinds, paged KV blocks, block tables per layer group, prefix cache, retention, reserve/commit/compact, recurrent double buffering, session cache, budgeting, `[state]` config | `r9v-state` |
| 4 | Kernel Generator and Registry | tiers T0/T0v/T1/T2, variant keys, generator structure, kernel families, autotune, ABI, asm policy, registry resolution, test gates, bundle, profiling hook, new-arch bring-up, `[kernels]` config | `r9v-kgen`, `r9v-registry`, `r9v-t0`, `kernels/` |
| 5 | Sharding and Partitioner | topology, strategies (PP/TP/EP/host experts), partitioner algorithm, planner, comms transport, correctness across device counts, `[parallel]`/`[experts]`/`[comm]` config | `r9v-part` |
| 6 | Scheduler | step structure, pre/device/post phases, prefill admission, spec-decode budget, capture and replay, workspace, memory pressure, finishing, faults, schedule log, `[scheduler]`/`[graph]`/`[spec]` config | `r9v-sched` |
| 7 | Speculative Decoding | `Proposer` trait, `Draft`, verifier contract, acceptance rules, tree drafts, proposer kinds, draft-model loading | `r9v-spec` |
| 8 | Model Definition | builder API, `LayerSpec`/`ModelSpec`, generic layer builder, families, weight binding and declarations, validation, `ModelSummary`, family testing | `r9v-models` |
| 9 | Loader | pipeline, fingerprints, budget and arenas, materialization, repack cache, tiered slab, tokenizer, secondary models, warmup, load report, `[load]`/`[io]`/`[host]`/`[tiered]`/`[warmup]` config | `r9v-loader` |
| 10 | Serving API | OpenAI and `/r9v` routes, request mapping, constrained decoding, streaming, lifecycle, runtime-mutable config, replicas, security, `[server]`/`[unix]`/`[sampling]`/`[replicas]` config | `r9v-serve` |
| 11 | Observability and Benchmark Protocol | profiling modes, metrics, tracing, achieved-bandwidth definition, measurement pass, doctor bundle, bench suites and receipts, baselines, logs, `[profile]`/`[log]`/`[doctor]`/`[bench]` config | `r9v-obs` |
| 12 | Config Schema and Helper | schema macros, settings index, precedence, validation, versioning, the helper, `[helper]` config | `r9v-config`, `r9v-helper` |
| 13 | Quant Tool | inputs, calibration, pipeline, folded smoothing, sensitivity, rounding, activation mode, hints, output, verify/compare/verify-arch | `tools/r9v-quant` |
| 14 | Build, Toolchain and CI | repo layout, toolchain pins, bundle build, CI tiers, runner isolation, release, versioning, platform scope | root, `ci/`, `xtask/` |
| 15 | Contributing and Governance | license, requirements by change type, RFC process, claims, support matrix, issues, security, specification authority, spec changes, roadmap | root docs |

Companion documents: `roadmap.md` (phases and exit criteria), `phase-a-agent-breakdown.md` (cards), `DECISIONS.md` (recorded choices), `SPEC-ISSUES.md` (open discrepancies), `CONVENTIONS.md` (code conventions).

## Closed sets

Adding to any of these is an RFC (spec 15 §4), never a card deliverable:

| set | defined in | where it appears in code |
|---|---|---|
| ops | spec 1 §4 | `r9v-ir` op structs and `OpId` |
| dtypes | spec 1 §2.1 | `r9v-ir::DType` |
| quant schemes | spec 2 §3.2, §3.3 | `r9v-format::SchemeId` |
| layout ids | spec 2 §2 | `r9v-format::LayoutId` |
| activation schemes | spec 1 §2.2, spec 2 §3.4 | `r9v-ir::QuantScheme::{PerToken, PerBlock32}` |
| state kinds | spec 3 §2 | `r9v-ir::StateKind`, `r9v-state::StateSpec` |
| retention policies | spec 3 §2 | `r9v-state::Retain` |
| shard layouts | spec 1 §5.1 | `r9v-ir::ShardLayout` |
| fusion patterns | spec 1 §3.4 | `r9v-ir` fusion table |
| `verify` methods | spec 1 §4.F, spec 7 §4 | `r9v-ir` verify op attrs |
| proposer kinds | spec 7 §6 | `r9v-spec::ProposerKind` |
| kernel tiers | spec 4 §2 | `r9v-registry::Tier` |
| collective ops | spec 1 §4.G | `r9v-ir` collectives |
| `r9v.*` metadata keys | spec 2 §6 | `r9v-format` typed accessors |
| config settings | spec 12 §3 | `r9v-config` schema declarations |
| bench suites | spec 11 §9.1 | `r9v-obs` bench |
| `LayerSpec` fields | spec 8 §3 | `r9v-models::LayerSpec` (additions are spec 8 §9 step 2, still a spec change) |
| arch descriptor fields | spec 1 App. A | `r9v-ir::ArchDescriptor` |

## Config sections by owner

`load`, `io`, `host`, `tiered`, `warmup` → spec 9 · `parallel`, `experts`, `comm` → spec 5 · `state` → spec 3 · `scheduler`, `graph` → spec 6 · `spec` → specs 6 and 7 · `kernels` → spec 4 · `server`, `unix`, `sampling`, `replicas` → spec 10 · `profile`, `log`, `doctor`, `bench` → spec 11 · `helper` → spec 12.

A setting must appear in its owning spec's config section **and** the spec 12 §3 index, or the docs build fails.

## Where the numbers come from

If you find yourself typing one of these, stop and read the source instead:

| number | source |
|---|---|
| wave size, CU count, LDS, VGPRs, matrix ops, dot instructions | `ArchDescriptor` (spec 1 App. A) |
| memory bandwidth, dispatch overhead, P2P, H2D/D2H | `ArchDescriptor.measured`, `Topology.links` (spec 11 §7) |
| step cost `C(S, T_dec, T_pre)`, `C_draft(k)` | warmup measurement (spec 9 §9) and tune-entry statics |
| tolerances | spec 1 §6.1 table, as data in the harness |
| block size (32), decode-class limit (16), bucket list | spec 3 §3.1, spec 1 §3.5 — constants in `r9v-ir`, referenced by name |
| bytes per weight | `r9v-format` bpw calculator |
