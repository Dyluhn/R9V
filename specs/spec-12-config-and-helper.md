# Spec 12 — Config Schema and Helper

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 3–11. Constrains: specs 14, 15.

## 0. Purpose and scope

One source of truth for every setting: the schema in code, and everything generated from it (the commented config file, the docs page, `/r9v/schema`, validation messages, the helper's explanations). Also precedence, validation, versioning, the one profile preset, and the helper: a small local model that fronts the doctor bundle and docs, proposes config changes for approval, and drafts issues.

Out of scope: the meaning of each setting (its owning spec), the endpoints (spec 10), the bundle contents (spec 11).

## 1. Principles

1. **The schema is code; everything else is generated.** A setting's doc string, range, default, unit, mutability and interactions are declared once, next to the field. The config file's comments, the docs, the JSON schema and the error messages are derived. There is no hand-maintained list of settings anywhere.
2. **`auto` is a rule, not a mystery.** Every `auto` default names the rule that resolves it, and the effective config shows the resolved value with its source.
3. **Unknown keys are errors.** A typo in a config file fails the load with the nearest valid key, rather than silently using a default.
4. **Mutability is declared.** Each setting is `runtime`, `reload` or `load`, and the API enforces it (spec 10 §7).
5. **The helper proposes; the user applies.** It has one write path, gated by approval, and every explanation it shows for a setting is the schema doc string, not its own paraphrase.

## 2. Schema

Declared in `r9v-config` as a struct tree with attribute macros:

```rust
#[section("scheduler", doc = "Step scheduling. Decode step time is the SLO.")]
struct Scheduler {
    #[setting(doc = "Target wall time per step in milliseconds. Prefill chunks and speculative depth \
                    are sized so the step fits. `auto` = 1.25 × measured single-sequence step time \
                    (latency) or 8 × (throughput), resolved at warmup on the loaded plan.",
              default = "auto", range = "1.0..=1000.0", unit = "ms",
              mutable = Runtime, interacts = ["scheduler.max_wait_ms", "spec.k_max", "parallel.profile"])]
    step_budget_ms: Auto<f32>,
    ...
}
```

Fields per setting: `key`, `type`, `default` (value or `auto` + rule), `range` or `enum`, `unit`, `mutable ∈ {Runtime, Reload, Load}`, `doc`, `interacts` (keys whose meaning or resolution this one affects), `since` (schema version), `renamed_from` (migration).

Generated artifacts, all from `r9v config gen`:

- `r9v.toml` skeleton with every setting present, commented with its doc, range and mutability, defaults uncommented, `auto` shown with its rule
- `docs/config.md`
- JSON schema served at `/r9v/schema`
- the validation error text (§5) and the 400 messages in spec 10 §6
- the helper's per-setting explanation (§7.4)

## 3. Settings index

The authoritative list is the generated `docs/config.md`; this index is the map from setting to owning spec so the specs stay consistent. `M` = mutability.

| section.key | type | default | M | spec |
|---|---|---|---|---|
| `load.model`, `load.draft_model`, `load.eagle_head` | path | — | Load | 9, 7 |
| `load.cache_dir` | path | auto (beside model) | Load | 9 |
| `load.require_fast_path` | bool | false | Load | 9 |
| `io.mode` | direct/mmap/auto | auto | Load | 9 |
| `io.chunk_mb`, `io.queue_depth`, `io.repack_threads` | int | 16, 8, auto (cores−2) | Load | 9 |
| `host.pinned_budget` | bytes | auto (min(free−4 GB, need)) | Load | 9 |
| `tiered.slab_bytes` | bytes | auto (min(25% pinned, tiered total)) | Load | 9 |
| `warmup.enabled`, `warmup.buckets` | bool, buckets | true, {S:[1,2,4], T_dec:[1,2,4,8,16,32], T_pre:[0,128,512,2048]} | Load | 9, 6 |
| `parallel.profile` | latency/throughput | latency | Reload | 5 |
| `parallel.devices` | [int] | all | Reload | 5 |
| `parallel.mode`, `parallel.tp_degree`, `parallel.pp_stages`, `parallel.pp_microbatches` | enum/int/ranges | auto | Reload | 5 |
| `experts.placement`, `experts.hot_set_vram` | enum, bytes | auto | Reload | 5 |
| `comm.transport` | auto/p2p/host | auto (measured) | Reload | 5 |
| `state.max_ctx`, `state.max_seqs` | int | 32768, 8 | Reload | 3, 6 |
| `state.cache_dtype` | e4m3/i8/f16 | e4m3 | Reload | 3 |
| `state.reserve_bytes` | bytes | 512 MB | Reload | 3 |
| `state.host_block_budget` | bytes | 0 | Reload | 3 |
| `state.session_cache` | int per GB | 2 | Runtime | 3 |
| `scheduler.step_budget_ms` | ms | auto | Runtime | 6 |
| `scheduler.prefill_min_chunk`, `scheduler.prefill_max_chunk` | tokens | 128, 2048 | Runtime | 6 |
| `scheduler.max_wait_ms` | ms | 500 | Runtime | 6 |
| `graph.mode` | auto/list/hipgraph | auto (measured) | Reload | 6 |
| `spec.proposer` | enum | auto | Reload | 7 |
| `spec.k_max`, `spec.tree_max`, `spec.min_accept`, `spec.lossy` | int, int, f32, bool | 8, 16, 0.3, false | Runtime | 6, 7 |
| `spec.ngram.n`, `spec.ngram.min_match` | int | 3, 2 | Runtime | 7 |
| `kernels.allow_jit`, `kernels.allow_nondeterministic` | bool | true, false | Load | 4 |
| `kernels.tune_budget_ms` | ms | 2000 | Load | 4 |
| `server.bind`, `server.api_key`, `server.admin_key` | str | 127.0.0.1:8080, none, auto (random) | Load | 10 |
| `server.max_queue`, `server.request_timeout` | int, s | 64, none | Runtime | 10 |
| `server.max_prompt_tokens`, `server.chat_template` | int, path | auto (max_ctx−64), none | Reload | 10 |
| `unix.path` | path | auto | Load | 10 |
| `sampling.defaults.*` | per spec 10 §3.2 | per spec 10 | Runtime | 10 |
| `replicas.devices` | [[int]] | none | Load | 10 |
| `profile.mode` | step/kernel/off | step | Runtime | 11 |
| `log.level`, `log.file` | enum, path | info, none | Runtime | 11 |
| `doctor.include_tokens`, `doctor.redact` | bool | false, true | Runtime | 11 |
| `bench.repeats`, `bench.warmup`, `bench.suites` | int, int, [str] | 5, 2, [...] | Runtime | 11 |
| `helper.enabled`, `helper.model`, `helper.embed_model`, `helper.ram_budget`, `helper.threads` | bool, path, path, bytes, int | true, auto, auto, 4 GB, auto | Load | 12 |

A setting that appears in an owning spec but not here, or here but not in a spec, fails the docs build (spec 14).

## 4. Precedence and sources

```
defaults  <  config file  <  environment  <  CLI flags  <  runtime changes (POST /r9v/config)
```

- Config file: `--config <path>`, else `./r9v.toml`, else `$XDG_CONFIG_HOME/r9v/r9v.toml`. Exactly one file; no includes.
- Environment: `R9V__SECTION__KEY=value` (double underscore), parsed with the same types.
- CLI: `--section.key value`, generated from the schema.
- The effective config (`GET /r9v/config`, `config.toml` in the bundle) records for every value: the source (`default`, `file:line`, `env`, `cli`, `runtime:<requester>:<time>`) and, for `auto`, the resolved value and the rule text.
- `ROC_GLOBAL_CU_MASK` is not config: it is launcher-applied data derived from the spoof plan's pre-queue launch contract (spec 14 §3), validated before HIP queue creation, and never set through any config source above.

## 5. Validation

At parse: type, range, enum membership, path existence for `Load` paths. Cross-field rules, each with a message that quotes the relevant doc strings:

- `scheduler.prefill_min_chunk ≤ scheduler.prefill_max_chunk ≤ max T bucket`
- `spec.k_max ≤ 15` (so `k + 1 ≤ 16` verified positions, the decode-class limit in spec 1 §4.D), `spec.tree_max ≤ 16`, `spec.k_max ≤ spec.tree_max`
- `server.max_prompt_tokens < state.max_ctx`
- `state.max_ctx % 32 == 0`
- `parallel.tp_degree` divides `hkv` or replication is possible (checked at bind, spec 8 §6)
- `server.bind` off-loopback requires `server.api_key`
- budgets vs hardware after the measurement pass (spec 9 §4.3 produces these messages)

Unknown keys are errors with the nearest valid key by edit distance. Keys under `[x-*]` sections are ignored and preserved, for tooling that wants to annotate the file.

## 6. Versioning and the profile preset

- `config_version = 1` at the top of the file. Renamed keys carry `renamed_from`; the old name loads with a warning naming the new one for two minor releases, then errors.
- `parallel.profile` is the only preset. It changes exactly the `auto` resolutions that name it in their rule: `scheduler.step_budget_ms`, the prefill interleaving rule (spec 6 §4.1), planner scoring (spec 5 §5.2), `parallel.pp_microbatches`. Nothing else consults it, and the effective config shows which values it influenced.

## 7. Helper

### 7.1 What it is

`r9v helper` (CLI chat) or the `/r9v/helper` chat route: a small instruct model plus a small embedding model, both in native format, running on the CPU device (spec 4 T0v) while no target model is loaded. It is a client of the local API over the Unix socket, with the tool list in §7.3 and nothing else. It is unloaded before a target load begins (spec 9 §8).

Default models are bundled or downloaded on first use (`helper.model = auto` resolves to the release's pinned small instruct model in `I4_K`, ~2–4B parameters; `helper.embed_model = auto` to a pinned ~50M embedding model). Both are configurable; a target model already loaded in native format can be pointed at instead, in which case the helper stays available while it is loaded and runs on the engine like any client.

### 7.2 Knowledge

Two retrieval indexes, built with the same embedding model:

- **Release index**, built at release time and shipped: the specs, `docs/`, `docs/config.md` (per-setting chunks), the FAQ, the known-issues list with their fixes, and the support matrix.
- **Live index**, built at helper start and refreshed on demand: the current doctor bundle (hardware, load report, plan, effective config with sources, schedule log summary, incident if any), the last 2,000 log lines, and any receipts in the session.

Every answer the helper gives cites its sources (doc section ids or log line ids). An answer with no retrieved source is prefixed as unsourced. The helper's context holds retrieved chunks and tool results only; it never receives whole documents.

### 7.3 Tools

| tool | effect | approval |
|---|---|---|
| `status`, `config.get`, `schema`, `tokenize` | read | no |
| `docs.search`, `logs.search`, `bundle.get` | read (redacted bundle) | no |
| `config.propose(patch)` | validates against the schema; returns diff + doc blocks + mutability; **applies nothing** | — |
| `config.apply(proposal_id)` | `POST /r9v/config` for runtime keys; writes `reload`/`load` keys to the file and says a reload is needed | **yes** |
| `load(model, …)`, `unload` | spec 9 | **yes** |
| `bench(suite)` | spec 11; takes minutes | **yes** |
| `issue.draft` | fills the issue template from the bundle (§7.5) | user submits |

No shell, no filesystem beyond the bundle and the config file, no network. The tool list is fixed in code; the model cannot request a tool that isn't there.

### 7.4 Proposal flow

1. The model emits a typed patch: `[{ key, value, reason }]`.
2. `config.propose` validates each entry (type, range, cross-field rules) and rejects the whole proposal on any failure, returning the validation messages to the model so it can revise.
3. The user sees three labeled sections: the **diff** (current → proposed, with each value's current source), the **schema doc block** for every touched key, verbatim, including range, unit, mutability and interactions, and the **model's reasoning** as a separate section marked as the model's.
4. Approve applies (§7.3); deny discards and the model is told which entries were declined. Runtime keys take effect at the next pre-step and are logged with `source = runtime:helper-approved`. Reload-required keys are written to the file and the helper says so; it does not trigger the reload without a second approval.

A proposal touching `server.bind`, `server.api_key`, `server.admin_key` or `replicas.*` is refused at step 2 regardless of the model's reasoning; those are set by a person at a keyboard.

### 7.5 Draft issues

`issue.draft` produces a markdown body from the issue template: title, environment (from `hardware.json`), model (`model_fp`, format, source file name), plan, effective non-default config, what happened (from the incident record or the user's description), reproduction (the exact command or request shape), and the path of the redacted bundle to attach. The user reviews the text, edits it, and submits: copy to clipboard, or `gh issue create` if the CLI is installed and the user approves the call. The helper never posts anywhere itself.

### 7.6 Resources

CPU only; `helper.ram_budget` (default 4 GB) covers both models, the indexes and the KV for a 4K-context chat. `helper.threads = auto` uses half the cores so the loader's repack threads still have headroom if a load starts. Response latency on a modern desktop CPU at 2–4B parameters in `I4_K` is a few tokens per second, which is adequate for a setup and troubleshooting assistant and is stated in the docs so nobody expects more.

### 7.7 Config

```
[helper]
enabled      = true
model        = auto
embed_model  = auto
ram_budget   = "4GB"
threads      = auto
```
