# Spec 11 — Observability and Benchmark Protocol

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 4, 6, 9, 10. Constrains: specs 12, 14, 15.

## 0. Purpose and scope

What the engine records, how it exposes it, and what a performance claim has to carry to be believed. Covers logs, metrics, per-step and per-kernel tracing, the hardware measurement pass, the doctor bundle, the benchmark protocol and its receipt format, the achieved-bandwidth definition, and perf-regression baselines.

Out of scope: the endpoints that serve these (spec 10), the helper that reads them (spec 12), CI mechanics (spec 14).

## 1. Principles

1. **Measured, then claimed.** Every number in a receipt is produced by the engine on the machine it describes, with the denominator stated. There is no "theoretical" figure without the measured one beside it.
2. **The receipt is the claim.** A performance statement without a receipt is a conversation; with one it is reproducible by anyone with the same hardware. The project's public numbers are receipts.
3. **Free when off, honest when on.** Step-level timing is always on and costs nothing measurable. Kernel-level timing perturbs the thing it measures and is opt-in, labeled as such in any receipt it appears in.
4. **A bug report is a bundle.** The doctor bundle contains everything needed to reproduce a schedule, not a description of it.
5. **Prompts are private.** No output in this spec contains prompt or completion text unless explicitly asked for; counts and hashes only.

## 2. Outputs

| output | form | always on | consumer |
|---|---|---|---|
| **log** | JSON lines, stderr or file | yes | humans, doctor bundle |
| **metrics** | Prometheus text at `/r9v/metrics` | yes | dashboards |
| **schedule log** | ring of step records (spec 6 §9) | yes | doctor bundle, tracing |
| **trace** | Perfetto JSON per request or per bench | opt-in | profiling |
| **doctor bundle** | tarball at `/r9v/doctor` or `r9v doctor` | on demand | bug reports, receipts |
| **receipt** | JSON + markdown table from `r9v bench` | on demand | performance claims |

## 3. Profiling modes

```
[profile] mode = "step" | "kernel" | "off"
```

- `step` (default): pre-step, draft, device and post-step wall time per step from host clocks and one device event per graph; `S`, `T_dec`, `T_pre`, `chunk`, `k`, `accept_len`, bytes streamed (from tune-entry statics summed over the graph). Overhead: one event per step.
- `kernel`: per-launch `hipEvent` pairs through the spec 4 §12 dispatch hook, giving time, bytes, flops, variant hash, tier, stream and rank per launch. Adds roughly 2–5 µs per launch and serializes some overlap; receipts produced in this mode say so and are not comparable to `step` receipts.
- `off`: no timing at all; for the tightest possible latency measurements from outside (spec 10 usage fields still report step counts).

External profilers (`rocprofv3`) are supported by setting `R9V_ROCPROF=1`, which makes every launch carry a kernel name that includes op and variant hash.

## 4. Metrics

Prometheus names, all prefixed `r9v_`. Histograms use fixed buckets so dashboards are stable across versions.

**Engine**: `up`, `model_loaded{model_fp}`, `load_seconds`, `warmup_seconds`, `plan{strategy,tp,stages}`.

**Scheduler**: `step_seconds` (hist), `step_pre_seconds`, `step_post_seconds`, `step_tokens{kind=decode|prefill}` (hist), `bucket{S,T_dec,T_pre}` (counter), `prefill_chunk_tokens` (hist), `forced_admissions_total`, `queue_depth`, `queue_wait_seconds` (hist), `active_sequences`, `paused_sequences`, `captures_total`, `budget_ms`.

**Spec decode**: `spec_k` (hist), `spec_accept_len` (hist), `spec_accept_rate{proposer}`, `spec_tokens_per_step`, `spec_disabled_sequences`, `spec_draft_seconds`.

**State**: `kv_blocks_total{group}`, `kv_blocks_used`, `kv_utilization`, `prefix_hit_tokens_total`, `prefix_lookup_total`, `session_hits_total`, `evictions_total`, `host_swaps_total`, `slab_hit_rate`, `slab_bytes_read_total`.

**Kernels** (from `step` statics, refined by `kernel` mode): `op_seconds_total{op,tier}`, `weight_bytes_total`, `state_bytes_total`, `achieved_gbps` (gauge, §6), `achieved_matrix_tops{dtype}`, `t1_fallback_ops`.

**Comms**: `collective_seconds_total{op,link}`, `collective_bytes_total`, `link_latency_us{a,b}`.

**Host tier**: `cold_expert_rate`, `t0v_seconds_total`, `segment_sync_seconds_total`.

**Requests**: `ttft_seconds` (hist), `tpot_seconds` (hist; time per output token, step-based), `request_tokens_per_second` (hist), `requests_total{status}`, `cancelled_total`, `grammar_mask_seconds_total`.

## 5. Tracing

A Perfetto JSON trace with one track per rank per stream plus host tracks (`scheduler`, `proposer`, `t0v`, `io`). In `step` mode spans are per phase; in `kernel` mode every launch is a span labeled `op/variant/tier` with bytes and flops as args, and collectives show their peer. Produced for a single request with `r9v_trace: true` in the request body, or for a whole bench run. Traces contain no token ids.

## 6. Achieved bandwidth and rate

Defined precisely because it is the headline number for decode:

```
decode_bytes(step) = Σ_launch weight_bytes(variant)            # from tune-entry statics; each weight counted once per step
                   + Σ_seq state_bytes_read(seq, step)          # KV blocks touched × block bytes, from BatchMeta
                   + activation_bytes(bucket)                   # from the graph summary
achieved_gbps      = decode_bytes / t_device(step)
utilization_spec   = achieved_gbps / arch.mem_bw_gbps           # spec-sheet denominator
utilization_meas   = achieved_gbps / measured.mem_bw_gbps       # §7 streaming-read denominator
```

Weights that were not read this step (embedding rows other than those gathered, cold experts computed on host, `L0` tables) are not counted; that is what makes the number a statement about the memory system rather than about file size. A receipt reports `achieved_gbps`, both utilizations, and the byte breakdown, so "93% of bandwidth" is checkable from the receipt alone. Matrix rate for prefill is `Σ flops / t_device` against `arch.matrix_ops` rates at the dtypes actually used.

## 7. Measurement pass

`r9v doctor measure` (also run automatically on first GPU load on a new hardware fingerprint) fills the measured fields of each physical-device descriptor and the topology (spec 1 App. A, spec 5 §2):

| measurement | method |
|---|---|
| `pcie_path` | resolve each GPU's canonical PCI BDF in sysfs; record endpoint and every upstream PCI ancestor through the root port, including current and maximum speed/width; select configured capacity from maximum speed plus negotiated width over the whole path, never the endpoint alone; a partial link-bearing hop is a typed failure, never silently omitted |
| `mem_bw_gbps` | streaming read kernel over 1 GB, median of 10 |
| `dispatch_overhead_us` | 1000 empty launches, launch list and hipGraph |
| `matrix_ops[*].rate` | synthetic 4096³ GEMM per dtype, WMMA path, median of 10 |
| `p2p{a,b}` | for every directed pair `a != b`, direct and host-staged copies at 16 KB / 256 KB / 16 MB, latency and bandwidth each; skipped for zero or one GPU |
| `h2d_gbps`, `d2h_gbps` | pinned copies, 256 MB |
| `host.mem_gbps` | multi-threaded streaming read |
| `nvme_gbps` | direct-IO sequential read, 1 GB, on the model's volume |
| clocks and temperatures | via `rocm-smi` before and after, recorded not gated |

Results are cached under the hardware fingerprint (`GPU UUID+BDF identities ‖ driver ‖ ROCm ‖ kernel ‖ CPU architecture/features ‖ stable PCIe path capacities`) and invalidated when any component changes. HIP ordinals are excluded because they are process-local and may reorder. Transient current PCIe speed is recorded but excluded from the stable fingerprint because an idle link may downshift; every hop's maximum speed, negotiated width, maximum width and measured transfer rates are retained. Optional board-sheet values stay in the device facts beside the measured ones; both are reported.

No-GPU is a valid measurement result. With no HIP runtime, the bundle records `hip_runtime = absent`, the GPU list and link matrix are empty, and CPU measurements continue. A present but broken HIP runtime is a typed diagnostic failure rather than being silently treated as absence.

## 8. Doctor bundle

`r9v doctor` or `GET /r9v/doctor`. A tarball containing:

```
manifest.json         r9v version, gen_version, bundle manifest hash, tune file hashes, timestamp, redaction flags
hardware.json         ISA and physical-device descriptors, measurements, topology, CPU, RAM, NVMe, driver, ROCm, kernel, rocm-smi snapshot
model.json            load report (spec 9 §10), model_fp, family, per-tensor table
plan.json             the plan (spec 5 §5.1) and per-rank graph summaries (spec 5 §4.1)
config.toml           effective config, every auto resolved, with the source of each value (default / file / runtime change)
schedule.jsonl        the schedule-log ring (last 4096 steps)
batchmeta/            BatchMeta for the last 64 steps, for exact schedule reproduction
metrics.txt           snapshot
log.jsonl             last 10,000 log lines
incident.json         present after a fault: error, faulting variant, graph summary, last step record
receipts/             any receipts produced this session
```

Redaction (`?redact=true`, default on for the helper's draft issues): absolute paths → basenames, hostname and usernames removed, environment variables dropped. Prompt text is never included; `batchmeta/` carries token counts and slot maps but not token ids unless `doctor.include_tokens = true`.

Typical size: 2–10 MB. The bundle is the required attachment for bug reports and the `hardware.json` + `model.json` + `plan.json` triple is embedded in every receipt.

## 9. Benchmark protocol

`r9v bench [--suite ...] [--compare llama.cpp:<path>]`. Runs against the loaded model and produces a receipt.

### 9.1 Suites

| suite | workload | reports |
|---|---|---|
| `decode` | 1 sequence, 128-token fixed prompt, 256 generated, greedy, spec off | tok/s, step time p50/p95, achieved GB/s, utilizations |
| `decode-spec` | same with the configured proposer | tok/s, accept rate, tokens/step, `k` distribution |
| `prefill` | prompts of 512 / 2048 / 8192 tokens, batch 1 | prompt tok/s, TTFT, matrix rate |
| `multi` | 4 and 8 concurrent sequences, 128-token prompts, 256 generated | aggregate tok/s, per-sequence tok/s, step time |
| `depth` | 32K-token context (or `max_ctx` if smaller), then 128 generated | decode tok/s at depth, state bytes/step, attention share |
| `accept` | spec decode on a fixed 32-prompt set (16 code, 16 prose) | accept rate per class, per proposer |

Fixed prompts are token-id sequences generated from a seeded random walk over the tokenizer plus the fixed text set for `accept`, so content is identical across machines and never contains anything private.

### 9.2 Procedure

- 2 warmup runs, then `repeats = 5` timed runs per workload; report median, min, max and p95 of step time. Runs are sequential with no other requests admitted.
- `rocm-smi` clocks and temperature recorded before and after each suite; a receipt whose GPU clock varied by more than 5% across repeats is marked `thermal: unstable` but still produced.
- Profile mode is `step` unless `--kernel` is passed, and the receipt states which.
- A receipt is produced only if every workload ran at the resolved plan with no forced admissions, T1 fallbacks in the graph, or captures during timed runs; otherwise the receipt is marked `invalid` with the reason.

### 9.3 Comparisons

`--compare llama.cpp:<build dir>[,<build dir>...]` runs `llama-bench` in the same session with the **same GGUF file**, `-fa 1`, matching context and batch, for **every llama.cpp backend built for this machine** (at minimum the HIP/ROCm build and the Vulkan build, at the same pinned commit), and records each build's commit, flags and command line. The comparison figure per workload is the **fastest backend for that workload**; the receipt shows every backend so the choice is visible. Only same-file comparisons are valid: native-format runs compare against the source GGUF they were quantized from, and the receipt says so. Comparisons against other engines follow the same rule: same file, same session, exact version recorded, command line included.

### 9.4 Receipt

```
receipt.json
  r9v:        version, gen_version, bundle hash, tune coverage for the graph (shipped/local/partial), profile mode
  hardware:   hardware.json (spec + measured)
  model:      model_fp, source file name and file_fp, format, per-scheme byte totals, plan
  config:     effective config
  suites:     per workload: runs[], median/min/max/p95, achieved_gbps, utilization_spec, utilization_meas, byte breakdown,
              accept stats, thermal note, validity
  compare:    tool, commit, build flags, command lines, per-workload results
  sha256:     of everything above
receipt.md    a table a person can paste into a thread
```

The markdown table always includes the utilization denominators and the comparison tool's commit so a reader can't misread it.

### 9.5 Efficiency floors

The engine is optimized first; everything layered on it is measured against the optimized engine. The floors below are the definition of "optimized" for this project, measured on the reference dense models (spec 14 §6) on gfx1201, for both the native file and the GGUF parity path.

| regime | metric | floor | measured by |
|---|---|---|---|
| decode (TG) | `utilization_meas` on the `decode` suite (batch 1, greedy, spec off) | **≥ 0.93** | §6, step mode |
| prefill (PP) | prompt tokens/s on the `prefill` suite at 2K and 8K vs the **fastest llama.cpp backend** on the same file, same session (§9.3) | **≥ 1.45×** at each of 2K and 8K; 512 reported, not gated | `prefill` suite |
| GEMM kernels | achieved rate / `measured.matrix_rates[dtype]` at `M ≥ 256` | reported; 0.80 is the diagnostic threshold for `below-floor` listing | §3 kernel mode, per variant |
| prefill attention kernels | achieved rate / measured matrix peak | reported; 0.60 diagnostic threshold | kernel mode, per variant |

Rules:

- The TG floor (0.93 of measured bandwidth) and the PP floor (1.45× the fastest llama.cpp backend at 2K and 8K) are the gates; the phase A fast path is not done until both hold (roadmap A5.9). Both are revised **upward only**, by a PR that carries the receipt showing the new value is reachable. The per-kernel matrix utilizations are diagnostics: they explain a PP shortfall, they do not define it.
- The PP floor is relative to llama.cpp on purpose: prefill is compute-bound and no whole-step percentage of matrix peak is meaningful across models. The comparison is always the same file, the same session, the same machine, and the fastest of every llama.cpp backend that builds for it, so a Vulkan build that beats the HIP build sets the bar.
- Floors apply to the spec-off engine. **A `decode-spec` or `accept` receipt is valid only when a `decode` (spec-off) receipt from the same machine, plan and session exists and meets the TG floor**, and spec-decode results are reported as a multiplier over that receipt (`tokens/s with proposer ÷ tokens/s without`). A spec-decode number without its spec-off companion is `invalid`. Speculative decoding is introduced to an engine that already meets the floor; it is never tuned or measured on one that doesn't.
- A variant that is correct but below its floor is `validated` (spec 4 §9.3) and listed as `below-floor` in the tune coverage report. Receipts state how many below-floor variants the graph contains; a receipt with any is still valid but not floor-meeting, and the README numbers are generated only from floor-meeting receipts.

## 10. Perf regression baselines

`bench/baselines/<arch>/<model_fp>.json` holds the last accepted receipt per suite for the reference model set. The CI runner (spec 14) runs `decode`, `prefill` and `multi` on every merge to main and fails on a median regression > 3% against the baseline, matching the per-variant gate in spec 4 §10. Baselines change only through a PR that includes the new receipt.

## 11. Logs

JSON lines: `ts, level, target, msg, fields{}`. Every request-scoped line carries `req_id`; every step-scoped line carries `step_id`. `info` never includes token ids or text. Levels: `error`, `warn`, `info`, `debug`, `trace`; `debug` adds per-step records inline; `trace` adds per-launch records and is meant for single-step investigations only.

## 12. Config

```
[profile]
mode            = "step"
[log]
level           = "info"
file            = none          # stderr by default
[doctor]
include_tokens  = false
redact          = true
[bench]
repeats         = 5
warmup          = 2
suites          = ["decode", "decode-spec", "prefill", "multi"]
```
