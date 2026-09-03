# R9V configuration

Generated from the `r9v-config` schema (Spec 12 §2).

## `load`

### `load.model`

Primary model artifact path.

- Type: `path`
- Default: `(none)`
- Mutability: `Load`
- Since schema: `1`

### `load.draft_model`

Draft model artifact path, if any.

- Type: `path`
- Default: `none`
- Mutability: `Load`
- Since schema: `1`

### `load.eagle_head`

Eagle head artifact path, if any.

- Type: `path`
- Default: `none`
- Mutability: `Load`
- Since schema: `1`

### `load.cache_dir`

Cache directory. auto = beside the model.

- Type: `Auto<path>`
- Default: `auto (beside model)`
- Mutability: `Load`
- Since schema: `1`

### `load.require_fast_path`

Require the fast execution path at load.

- Type: `bool`
- Default: `false`
- Mutability: `Load`
- Since schema: `1`

## `io`

### `io.mode`

IO mode for weight reads. auto = direct I/O when supported, otherwise mmap.

- Type: `Auto<IoMode>`
- Default: `auto`
- Mutability: `Load`
- Range/enum: `direct|mmap|auto`
- Since schema: `1`

### `io.chunk_mb`

Read chunk size.

- Type: `u32`
- Default: `16`
- Mutability: `Load`
- Range/enum: `1..=1024`
- Unit: `MB`
- Since schema: `1`

### `io.queue_depth`

IO queue depth.

- Type: `u32`
- Default: `8`
- Mutability: `Load`
- Range/enum: `1..=128`
- Since schema: `1`

### `io.repack_threads`

Repack worker threads. auto = cores minus 2.

- Type: `Auto<u32>`
- Default: `auto (cores-2)`
- Mutability: `Load`
- Range/enum: `1..=256`
- Since schema: `1`

## `host`

### `host.pinned_budget`

Pinned host-memory budget. auto = min(free minus 4 GB, need).

- Type: `Auto<bytes>`
- Default: `auto (min(free-4GB, need))`
- Mutability: `Load`
- Unit: `bytes`
- Since schema: `1`

## `warmup`

### `warmup.enabled`

Run warmup over the bucket set at load.

- Type: `bool`
- Default: `true`
- Mutability: `Load`
- Since schema: `1`

### `warmup.buckets`

Warmup buckets over S, T_dec and T_pre.

- Type: `buckets`
- Default: `{S:[1,2,4], T_dec:[1,2,4,8,16,32], T_pre:[0,128,512,2048]}`
- Mutability: `Load`
- Since schema: `1`
- Interacts with: `scheduler.prefill_max_chunk`

## `state`

### `state.max_ctx`

Maximum context length; must be a multiple of 32.

- Type: `u32`
- Default: `32768`
- Mutability: `Reload`
- Range/enum: `32..=1048576`
- Since schema: `1`

### `state.max_seqs`

Maximum concurrent sequences.

- Type: `u32`
- Default: `8`
- Mutability: `Reload`
- Range/enum: `1..=1024`
- Since schema: `1`

### `state.cache_dtype`

KV cache dtype.

- Type: `CacheDtype`
- Default: `e4m3`
- Mutability: `Reload`
- Range/enum: `e4m3|i8|f16`
- Since schema: `1`

### `state.reserve_bytes`

Bytes reserved outside the state pool.

- Type: `bytes`
- Default: `512 MB`
- Mutability: `Reload`
- Unit: `bytes`
- Since schema: `1`

### `state.host_block_budget`

Host block spill budget; 0 disables spilling.

- Type: `bytes`
- Default: `0`
- Mutability: `Reload`
- Unit: `bytes`
- Since schema: `1`

### `state.session_cache`

Retained sessions per GB of cache.

- Type: `u32`
- Default: `2`
- Mutability: `Runtime`
- Range/enum: `0..=64`
- Since schema: `1`

## `scheduler`

### `scheduler.step_budget_ms`

Target wall time per step in milliseconds. Prefill chunks and speculative depth are sized so the step fits. auto = 1.25 x measured single-sequence step time (latency) or 8 x (throughput), resolved at warmup on the loaded plan.

- Type: `Auto<f32>`
- Default: `auto`
- Mutability: `Runtime`
- Range/enum: `1.0..=1000.0`
- Unit: `ms`
- Since schema: `1`
- Interacts with: `scheduler.max_wait_ms`, `spec.k_max`, `parallel.profile`

### `scheduler.prefill_min_chunk`

Minimum prefill chunk.

- Type: `u32`
- Default: `128`
- Mutability: `Runtime`
- Range/enum: `1..=16384`
- Unit: `tokens`
- Since schema: `1`
- Interacts with: `scheduler.prefill_max_chunk`

### `scheduler.prefill_max_chunk`

Maximum prefill chunk.

- Type: `u32`
- Default: `2048`
- Mutability: `Runtime`
- Range/enum: `1..=16384`
- Unit: `tokens`
- Since schema: `1`
- Interacts with: `scheduler.prefill_min_chunk`

### `scheduler.max_wait_ms`

Maximum time a request waits before a step starts.

- Type: `u32`
- Default: `500`
- Mutability: `Runtime`
- Range/enum: `0..=60000`
- Unit: `ms`
- Since schema: `1`

## `graph`

### `graph.mode`

Graph capture mode. auto = measured at warmup.

- Type: `Auto<GraphMode>`
- Default: `auto (measured)`
- Mutability: `Reload`
- Range/enum: `auto|list|hipgraph`
- Since schema: `1`

## `spec`

### `spec.proposer`

Speculative proposer. auto = MTP, Eagle, draft, then n-gram according to loaded artifacts.

- Type: `Auto<ProposerKind>`
- Default: `auto`
- Mutability: `Reload`
- Range/enum: `auto|none|ngram|mtp|draft|eagle`
- Since schema: `1`
- Interacts with: `load.draft_model`, `load.eagle_head`, `spec.k_max`

### `spec.k_max`

Maximum speculative draft depth; k + 1 verified positions must fit the decode-class limit.

- Type: `u32`
- Default: `8`
- Mutability: `Runtime`
- Range/enum: `0..=15`
- Since schema: `1`
- Interacts with: `spec.tree_max`, `scheduler.step_budget_ms`

### `spec.tree_max`

Maximum speculative tree size.

- Type: `u32`
- Default: `16`
- Mutability: `Runtime`
- Range/enum: `1..=16`
- Since schema: `1`
- Interacts with: `spec.k_max`

### `spec.min_accept`

Disable speculation temporarily when the recent acceptance EMA is below this value.

- Type: `f32`
- Default: `0.3`
- Mutability: `Runtime`
- Range/enum: `0.0..=1.0`
- Since schema: `1`

### `spec.lossy`

Permit opt-in lossy Typical acceptance.

- Type: `bool`
- Default: `false`
- Mutability: `Runtime`
- Since schema: `1`

## `spec.ngram`

### `spec.ngram.n`

N-gram width.

- Type: `u32`
- Default: `3`
- Mutability: `Runtime`
- Range/enum: `1..=16`
- Since schema: `1`

### `spec.ngram.min_match`

Minimum match length to propose.

- Type: `u32`
- Default: `2`
- Mutability: `Runtime`
- Range/enum: `1..=16`
- Since schema: `1`

## `kernels`

### `kernels.allow_jit`

Allow just-in-time kernel builds.

- Type: `bool`
- Default: `true`
- Mutability: `Load`
- Since schema: `1`

### `kernels.allow_nondeterministic`

Allow kernels that may be nondeterministic.

- Type: `bool`
- Default: `false`
- Mutability: `Load`
- Since schema: `1`

### `kernels.tune_budget_ms`

Autotune time budget.

- Type: `u64`
- Default: `2000`
- Mutability: `Load`
- Range/enum: `0..=600000`
- Unit: `ms`
- Since schema: `1`

## `profile`

### `profile.mode`

Profiling mode.

- Type: `ProfileMode`
- Default: `step`
- Mutability: `Runtime`
- Range/enum: `step|kernel|off`
- Since schema: `1`

## `log`

### `log.level`

Log level.

- Type: `LogLevel`
- Default: `info`
- Mutability: `Runtime`
- Range/enum: `trace|debug|info|warn|error`
- Since schema: `1`

### `log.file`

Log file path; none disables file logging.

- Type: `path`
- Default: `none`
- Mutability: `Runtime`
- Since schema: `1`

## `doctor`

### `doctor.include_tokens`

Include token text in the doctor bundle.

- Type: `bool`
- Default: `false`
- Mutability: `Runtime`
- Since schema: `1`

### `doctor.redact`

Redact secrets in the doctor bundle.

- Type: `bool`
- Default: `true`
- Mutability: `Runtime`
- Since schema: `1`

## `bench`

### `bench.repeats`

Measured repeats per benchmark.

- Type: `u32`
- Default: `5`
- Mutability: `Runtime`
- Range/enum: `1..=100`
- Since schema: `1`

### `bench.warmup`

Warmup runs before measurement.

- Type: `u32`
- Default: `2`
- Mutability: `Runtime`
- Range/enum: `0..=100`
- Since schema: `1`

### `bench.suites`

Benchmark suites to run.

- Type: `[str]`
- Default: `[decode, decode-spec, prefill, multi]`
- Mutability: `Runtime`
- Since schema: `1`
