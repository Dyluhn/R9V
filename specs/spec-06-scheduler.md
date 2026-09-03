# Spec 6 — Scheduler

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 1–5. Constrains: specs 7, 9, 10, 11, 12.

## 0. Purpose and scope

The loop that turns queued requests into steps: what runs each step, how much prefill is admitted alongside decode, how many draft tokens are verified, how captured graphs are replayed across ranks and segments, and what happens when a sequence finishes. One policy, one number: decode step time is the SLO and `step_budget_ms` is the knob.

Out of scope: how draft tokens are produced (spec 7), the API that enqueues requests (spec 10), the state manager's internals (spec 3), placement (spec 5).

## 1. Principles

1. **Decode step time is the SLO.** Every other goal is satisfied only within the budget. Throughput mode is the same scheduler with a larger budget.
2. **Admission waits; running decodes are never preempted.** A new prompt gets prefill capacity only when the step has room. A decoding sequence is never paused to make room for prefill.
3. **Everything data-dependent happens before or after the graph.** Draft tokens, gather rows, block tables and sampling params are prepared in a pre-step phase; acceptance, commits and finishes are handled in a post-step phase. The device replays a fixed graph in between.
4. **Costs are measured, not guessed.** The tune files (spec 4) give the step cost per bucket on this machine. The scheduler plans against those numbers.
5. **Spec decode is always on when a proposer exists**, and `k` is chosen per step from measured cost and observed acceptance, not fixed by config.
6. **The schedule is reproducible.** Decisions depend only on request arrival order, the cost table and the config. The per-step schedule log is part of any bug report.

## 2. Objects

```
Request  { id, tokens, sampling: SamplingParams, max_tokens, stop: [EOS ids ∪ stop strings], stream: bool }
Sequence { req, seq_id (spec 3), phase: Queued | Prefilling { done: u32 } | Decoding | Finished,
           ctx_len, generated: Vec<u32>, accept_ema: f32, proposer_state }
Step     { seqs_decode: [Seq], seqs_prefill: [(Seq, chunk)], k: [u32], bucket: (S, T_dec, T_pre), graphs: [GraphId per rank] }
```

Buckets are spec 1 §3.5. `T_dec = Σ_decode (k_s + 1)`, `T_pre = Σ_prefill chunk_s`, `T = T_dec + T_pre`. Decode and prefill tokens share one step graph (spec 1 §3.1).

## 3. The step

### 3.1 Pre-step (host, single scheduler thread)

1. **Finish and free.** Apply the previous step's post-step results (§3.3) if not already done.
2. **Admit decode.** All `Decoding` sequences, up to `max_seqs`, in arrival order.
3. **Choose k.** For each decoding sequence with a proposer, provisional `k_s = min(k_max, proposer.max_k)`. Then §4.2 shrinks `k` globally until the step fits the budget.
4. **Admit prefill.** §4.1 picks which queued/prefilling sequences get a chunk and how big.
5. **Draft.** Call the proposer for each decoding sequence to produce `k_s` draft tokens (spec 7). Proposers that need a device pass (draft model, EAGLE) run their own small graph here; n-gram and MTP-from-last-step are host-only.
6. **Reserve state.** `manager.reserve(s, k_s + 1)` for decode, `reserve(s, chunk)` for prefill. A prefill reserve failure removes that chunk from the step; a decode reserve failure follows §6.
7. **Build `BatchMeta`** (spec 3 §5) including the tree mask if any proposer emitted a tree.
8. **Gather rows.** For models with `ngram_gather` on host or tiered tables: compute row ids for every token in the step (including draft candidates), host-gather into the pinned staging buffer (spec 9 row cache), enqueue one H2D on the copy stream.
9. **Upload** token ids, positions, sampling params, RNG state, `BatchMeta`, `gather_staging` on the copy stream; record an event.

Host-side pre-step work (steps 1–4, 6–9) is budgeted at ≤ 10% of `step_budget_ms`; if the host gather or `BatchMeta` construction exceeds that on a step, the log flags it. Proposer device passes (step 5) are not part of that 10%; they are charged in §4.2.

### 3.2 Device

For each rank, replay the graph for `(plan, S_bucket, T_dec_bucket, T_pre_bucket)`:

- **Unsegmented models**: one graph per rank. Under PP, stage `i+1`'s graph waits on stage `i`'s boundary `send` event.
- **Segmented models** (host-computed experts, spec 5 §3.4): the graph is a list of segments split at each MoE layer with cold experts. Sequence per segment: replay segment → router output and routed rows D2H → T0v computes cold experts on the thread pool → H2D → next segment. Hot experts run inside the segment. The scheduler pipelines the CPU work of layer `l` with the device's hot-expert work of layer `l` and starts layer `l+1` only when both are done.

The last stage's graph ends with `logits_postprocess → sample` (and `verify` when `k > 0`), then a single D2H of `sampled [S, k+1]` and `accept_len [S]` on the copy stream.

### 3.3 Post-step (host)

1. Wait for the readback event.
2. For each decoding sequence: `manager.commit(s, accept_len)` (with `compact` first for tree verify), append accepted tokens to `generated`, update `accept_ema`, update proposer state.
3. For each prefill chunk: advance `Prefilling.done`; if the prompt is complete, phase → `Decoding` and the first sampled token (from the chunk's last position) is recorded.
4. **Finish check** per sequence: EOS id in accepted tokens, `generated.len() ≥ max_tokens`, stop string match on the incrementally detokenized tail, or client cancel. Finished sequences → `manager.free_seq`, results delivered to spec 10.
5. Stream accepted tokens to clients (spec 10) for `stream = true` requests.
6. Append the step record to the schedule log (§9).

Post-step and the next pre-step overlap with nothing on the device; at `latency` the device idles for their duration, which is why both are kept small and single-threaded (no locks on the hot path).

## 4. Budgeting

### 4.1 Prefill admission

Let `C(S, T_dec, T_pre)` be the step cost for a bucket: measured at warmup for the warm buckets (spec 9 §9), summed from tune-entry statics for others, plus segment host cost from the plan. Let `D = C(S_dec, T_dec, 0)` be the cost of this step's decode work alone.

- Room `R = budget − D − pre_step_estimate`.
- Prompts are served **one at a time in arrival order** under `latency` (serial prefill gives the earliest waiting prompt the best TTFT); `throughput` interleaves chunks across waiting prompts round-robin.
- For the prompt at the head: `chunk = largest bucket size b ≤ remaining_prompt` such that `C(S_dec + 1, T_dec, b) − D ≤ R`, with `b ≥ prefill_min_chunk`.
- If no `b ≥ prefill_min_chunk` fits, the prompt waits. A prompt that has waited longer than `max_wait_ms` is admitted at `prefill_min_chunk` regardless; that one step exceeds the budget and is logged as a forced admission. Default `max_wait_ms = 500`.
- Prefix-cache hits (spec 3 §3.4) reduce `remaining_prompt` before this calculation; a fully cached prompt skips prefill entirely and its first decode step is scheduled immediately.

### 4.2 Speculative decoding budget

Decode is bandwidth-bound at small `T_dec`, so `C(S, T_dec, T_pre)` is nearly flat from `T_dec = S` to `T_dec ≈ 8S`; the cost table says exactly how flat on this machine. Given provisional `k_s`:

- Expected accepted tokens per step for sequence `s`: `E_s(k) = Σ_{i=1..k} accept_ema_s^i + 1` (geometric model, standard for rejection sampling).
- Step cost including the proposer: `C_step(k) = C_draft(k) + C(S, Σ(k_s + 1), T_pre) + Σ_s k_s · c_recur`, where `C_draft(k)` is the proposer's device-pass cost at depth `k` (zero for `ngram` and `mtp`, measured at warmup for `draft` and `eagle`) and `c_recur` is the hybrid-layer recompute cost per token from the tune table (zero for models without recurrent layers).
- Step throughput proxy: `G(k) = Σ_s E_s(k_s) / C_step(k)`.
- Shrink: while `C_step(k) > budget` or `G(k) < G(k − 1)`, decrement the largest `k_s`. Stop when the step fits and the proxy is non-increasing on further decrement. A draft model whose pass costs more than it earns is therefore turned down by the same rule that turns down a poorly accepting n-gram draft.
- `k_s = 0` for a sequence whose `accept_ema < min_accept` (default 0.3) for the last 8 steps; retried every 32 steps with `k = 2` so a change in text regime can turn it back on.

`accept_ema` is a per-sequence exponential moving average of `accept_len / (k + 1)` with `α = 0.2`, seeded from the proposer's global rate.

### 4.3 Budget default

`step_budget_ms = auto` resolves at warmup to `1.25 × C(1, 1, 0)` for `latency` and `8 × C(1, 1, 0)` for `throughput`, measured on the actual plan. Manual values are absolute milliseconds. The resolved number is written to the doctor bundle and the log at load.

## 5. Graphs and replay

### 5.1 Capture

A graph is captured per `(plan, rank, S_bucket, T_dec_bucket, T_pre_bucket, segment)` by running the model definition's builder against the registry (spec 4 §9) with the workspace arena bound. A graph with `T_pre > 0` contains two `attention` (and `linear_attn_scan`) launches, one per sequence class (spec 1 §3.1); every other op runs over the full `T`. Capture happens:

- eagerly at load for `warm_buckets` (default `S ∈ {1, 2, 4}`, `T_dec ∈ {1, 2, 4, 8, 16, 32}`, `T_pre ∈ {0, 128, 512, 2048}`, valid combinations only, i.e. `T_dec ≥ S`), which also runs the autotune and validation path for those variants;
- lazily on first use for anything else, with the step that triggers it logged as a capture step (its timing is excluded from perf baselines).

### 5.2 Replay mechanism

Two mechanisms behind one interface:

- **Launch list** (always available): the recorded sequence of `(code object, args struct, geometry, stream, events)`; replay issues them in order. Deterministic, debuggable, every launch visible to the profiler.
- **`hipGraph`** (when `arch.graph_capture = Supported`): the same list captured into a graph object.

At warmup the scheduler replays both for the `(1, 1, 0)` bucket and picks the faster (`graph.mode = auto`); the choice and the measured dispatch overhead are recorded. `hipGraph` on RDNA4 is treated as an accelerator, not a dependency: any capture error falls back to the launch list for that graph and logs it.

### 5.3 Workspace

One arena per rank sized to the maximum workspace over all captured buckets (split-K partials, split-KV partials, MoE sort buffers, activation ping-pong, comms buffers). Allocated once at load; graphs bind fixed offsets. Growing the arena (a new larger bucket captured lazily) is the only runtime allocation, and it forces recapture of every graph on that rank, so `warm_buckets` should cover the buckets a deployment will actually see.

### 5.4 Streams and events

Per rank: `compute`, `comms` (spec 5 §6.3), `copy`. Order per step: copy stream uploads → event → compute stream replays → (comms as needed) → compute records `done` → copy stream reads back → event → host. Under PP the boundary `send` is on stage `i`'s comms stream and stage `i+1`'s compute waits on it.

## 6. Memory pressure

- Prefill reserve failure: the chunk is not admitted; the prompt waits. Logged.
- Decode reserve failure (should not occur per spec 3 §6.3, handled anyway): the **youngest** decoding sequence is paused — state retained, no tokens generated — until a reserve succeeds. Paused sequences are excluded from bucket sizing. Never killed. Clients see a stall, not an error.
- Prefix-cache blocks with refcount 0 are reclaimed before any pause happens (spec 3 §3.4).

## 7. Finishing

- `EOS` ids come from the model definition's tokenizer metadata; a request may add stop token ids.
- Stop strings are matched on the host against the incrementally detokenized tail (keeping the last `max_stop_len` bytes); the match trims output to the match start.
- On finish: `manager.free_seq` (which may retain session state, spec 3 §4.3), proposer state dropped, result delivered. The finished sequence's slot in the batch is filled by the next admitted sequence at the next step; the bucket shrinks or grows accordingly.

## 8. Faults

A device fault during a step (kernel exception, memory error) fails every in-flight sequence with an engine error, dumps the schedule log and the graph summary to the doctor bundle, and reloads the model. No partial recovery in v1; the reload path is fast because the repack cache and tune files are warm.

## 9. Schedule log

Per step, one record: `step_id, t_pre_us, t_draft_us, t_device_us, t_post_us, S, T_dec, T_pre, chunk, k[], accept_len[], forced_admission, budget_ms, bucket, graph_mode, captured: bool, paused: [seq], segment_sync_us`. Ring buffer of the last 4096 steps in memory; flushed to the doctor bundle on request or fault. Spec 11 reads the same records for metrics.

## 10. Config

```
[scheduler]
step_budget_ms     = "auto" | f32
prefill_min_chunk  = 128
prefill_max_chunk  = 2048
max_wait_ms        = 500
[warmup]
buckets            = { S = [1, 2, 4], T_dec = [1, 2, 4, 8, 16, 32], T_pre = [0, 128, 512, 2048] }
[spec]
k_max              = 8
min_accept         = 0.3
[graph]
mode               = "auto" | "list" | "hipgraph"
```

`state.max_ctx` and `state.max_seqs` (spec 3 §9) bound the batch; they size arenas and are not scheduler settings.

`profile = throughput` (spec 5 §8) changes only the `auto` resolution of `step_budget_ms` and the prefill interleaving rule; every other line is shared.
