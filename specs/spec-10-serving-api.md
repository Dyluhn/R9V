# Spec 10 — Serving API

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 6, 7, 9. Constrains: specs 11, 12.

## 0. Purpose and scope

The surface clients talk to: an OpenAI-compatible HTTP API for generation, a native `/r9v` API for control and diagnostics, the same API over a Unix socket for local clients (CLI, helper), request-to-engine mapping (chat templates, sampling parameters, stop conditions, logprobs), constrained decoding, streaming semantics, request lifecycle, multi-replica routing, and security defaults.

Out of scope: the metrics content (spec 11), the config schema and doc strings (spec 12), the helper's own behavior (spec 12).

## 1. Principles

1. **One engine, one process, one API.** The HTTP server, the Unix socket and the CLI all call the same in-process `Engine` interface. No client has a private path.
2. **OpenAI-compatible where it's cheap, native where it matters.** `/v1/chat/completions` and `/v1/completions` behave as clients expect. Engine control, diagnostics and anything spec-decode or placement related lives under `/r9v/` and is not disguised as an OpenAI field.
3. **Every request is reproducible.** A request with a `seed` on the same `model_fp`, plan and kernel bundle produces the same tokens. The response carries what's needed to say so.
4. **Constrained decoding is a mask, not a sampler.** Grammars produce per-position token masks fed to `logits_postprocess`; sampling, spec decode and verification are unchanged.
5. **Loopback by default.** Binding off-loopback requires an API key; control endpoints require an admin key.

## 2. Endpoints

### 2.1 OpenAI-compatible

| method | path | notes |
|---|---|---|
| `POST` | `/v1/chat/completions` | messages → chat template → tokens; streaming via SSE |
| `POST` | `/v1/completions` | raw prompt; supports FIM via `suffix` when the tokenizer declares FIM tokens |
| `GET` | `/v1/models` | the loaded model (and replicas' models) with `model_fp` in `metadata` |

### 2.2 Native

| method | path | auth | notes |
|---|---|---|---|
| `GET` | `/r9v/status` | key | loaded model, plan summary, resolved budget, queue depth, active sequences, uptime |
| `POST` | `/r9v/load` | admin | body: spec 9 `[load]` fields; returns a job id; progress at `/r9v/load/progress` (SSE) |
| `POST` | `/r9v/unload` | admin | |
| `GET` | `/r9v/config` | key | effective config, every `auto` resolved, with the spec 12 doc string per field |
| `POST` | `/r9v/config` | admin | applies a diff of **runtime-mutable** fields (§7); anything else returns which fields need a reload |
| `GET` | `/r9v/schema` | key | the spec 12 config schema as JSON |
| `GET` | `/r9v/doctor` | admin | the doctor bundle (spec 11) as a tarball; `?redact=true` strips paths and hostnames |
| `GET` | `/r9v/metrics` | key | Prometheus text (spec 11) |
| `POST` | `/r9v/bench` | admin | runs the benchmark protocol (spec 11) and returns the receipt |
| `POST` | `/r9v/tokenize` / `/r9v/detokenize` | key | with the loaded tokenizer; `apply_chat_template: true` supported |
| `GET` | `/r9v/load/progress` | key | SSE of spec 9 pipeline steps and per-tensor progress |
| `POST` | `/r9v/helper` | admin | chat with the helper (spec 12 §7); available only when the helper is running |

The Unix socket (`unix.path`, default `$XDG_RUNTIME_DIR/r9v.sock`) serves the same routes without keys; filesystem permissions are the auth.

## 3. Request mapping

### 3.1 Chat template

`tokenizer.chat_template` from GGUF metadata is rendered with a Jinja subset (the same subset llama.cpp supports, so templates that work there work here). `tools`, `tool_choice`, `reasoning` flags and any `chat_template_kwargs` the request passes are exposed to the template. A `chat_template` path in config overrides the file's template for models whose embedded one is broken.

### 3.2 Sampling parameters

| request field | `SamplingParams` (spec 1 §4.F) | default |
|---|---|---|
| `temperature` | `temperature` | 1.0 (0 → `Greedy` verify) |
| `top_p` | `top_p` | 1.0 |
| `top_k` | `top_k` | 0 (off) |
| `min_p` | `min_p` | 0.0 |
| `presence_penalty` | `presence_penalty` | 0.0 |
| `frequency_penalty` | `frequency_penalty` | 0.0 |
| `repetition_penalty` | `repetition_penalty` | 1.0 |
| `logit_bias` | `logit_bias` (sparse) | none |
| `seed` | RNG seed for Philox (spec 1 §4.F) | random, returned in the response |
| `n` | `n` sequences sharing the prompt; prefix cache makes the extra prefill free | 1 |
| `max_tokens` / `max_completion_tokens` | `max_tokens` | model `max_ctx − prompt` |
| `stop` | stop strings (≤ 8) and stop token ids | model `eos_ids` |
| `logprobs`, `top_logprobs` | export `logits` for sampled positions; top-k over the postprocessed probs | off |
| `response_format` | `text` / `json_object` / `json_schema` → grammar (§4) | text |
| `tools`, `tool_choice` | template + tool-call parsing (§3.4); `tool_choice: required` → grammar forcing a call | none |
| `stream`, `stream_options` | SSE; `include_usage` adds the final usage chunk | false |

Model-level defaults for any of these live in config (`[sampling.defaults]`) and appear in `/r9v/config`.

Unsupported OpenAI fields (`best_of`, `suffix` without FIM tokens, `user`) are accepted and ignored, with an `r9v_warnings` array in the response naming them. Unknown fields are ignored silently, matching OpenAI behavior.

### 3.3 Prompt handling

- Prompt tokens > `max_prompt_tokens` (default `max_ctx − 64`) → 400 with the counts.
- Prefix-cache and session-cache hits (spec 3) are automatic; `usage.prompt_tokens_details.cached_tokens` reports them.
- `n > 1` enqueues `n` sequences with a shared prompt; the second onward is a full prefix hit.

### 3.4 Reasoning and tool calls

- Models whose template marks reasoning (`<think>`, family-declared tags) get the reasoning span split into `reasoning_content` and the remainder into `content`, both streamed. `reasoning: { effort }` is passed to the template if it uses it.
- Tool calls are parsed from the model's declared convention (Hermes/Qwen `<tool_call>` JSON, or the family's format) into OpenAI `tool_calls` objects. With `tool_choice: required` or a `json_schema` format, the grammar (§4) forces well-formed output, so the parser never fails on model drift.

## 4. Constrained decoding

- Grammar sources: `json_schema` (compiled to a grammar), `json_object` (generic JSON), a native `grammar` field accepting GBNF for `/v1/completions` and `/r9v` clients, and tool-call forcing.
- Implementation: the grammar is compiled to an automaton over bytes; token masks are computed by walking the tokenizer's byte trie from the automaton state, with masks cached per `(grammar, state)` so repeated states cost a lookup. Budget: ≤ 0.5 ms per uncached state at `V = 150k` on one core, amortized to ~0 over a response.
- Masks are produced for **every verified position**: for a linear draft of `k` tokens the automaton is advanced speculatively along the draft, producing `k + 1` masks; for a tree, along each path. After verify, the automaton is reset to the accepted prefix's state (the states are stored per position, so this is a pointer move). The masks are the `grammar_mask [S, q, V]` external input (spec 1 §3.2).
- A mask that admits no token is a grammar bug; the sequence finishes with `finish_reason: "error"` and the grammar state in the log rather than sampling garbage.

## 5. Streaming

- SSE, `data: {chunk}` per step with all tokens accepted that step in one delta (spec decode accepts several per step; the OpenAI format allows multi-token deltas). `data: [DONE]` last.
- The final chunk (or the non-streamed response) carries `usage`: `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`, and an `r9v` object: `seed`, `model_fp`, `plan_id`, `steps`, `accepted_per_step` (mean), `ttft_ms`, `tokens_per_second`, `proposer`.
- Client disconnect cancels the request at the next post-step; the sequence is freed and its state returned to the pool.

## 6. Lifecycle and errors

- Requests get an id, enter the FIFO queue (spec 6 §4.1) and are admitted as prefill capacity allows. `max_queue` exceeded → 429 with `Retry-After`.
- No model loaded, or loading → 503 with the load job id and progress URL.
- Bad parameters → 400 whose message includes the spec 12 doc string for the offending field, so the error explains the constraint.
- Engine fault (spec 6 §8) → in-flight requests get 500 with the incident id that names the doctor bundle written for it; the API stays up through the reload and returns 503 until ready.
- `request_timeout` (default none) applies from admission, not enqueue.

Error envelope is OpenAI's (`{ "error": { "message", "type", "code", "param" } }`) with `code` carrying the `r9v` error id.

## 7. Runtime-mutable config

Applied via `POST /r9v/config` without reload: exactly the settings marked `Runtime` in the spec 12 §3 index (`scheduler.step_budget_ms`, `scheduler.max_wait_ms`, `scheduler.prefill_*`, `spec.k_max`, `spec.min_accept`, `spec.tree_max`, `spec.lossy`, `spec.ngram.*`, `state.session_cache`, `sampling.defaults.*`, `server.max_queue`, `server.request_timeout`, `profile.mode`, `log.*`, `doctor.*`, `bench.*`). Everything else (placement, model paths, budgets that size arenas, `state.max_ctx`, `state.max_seqs`) returns `requires_reload: [fields]`. A change takes effect at the next pre-step and is logged with the requester.

The helper (spec 12) uses exactly this endpoint after user approval; it has no other write path.

## 8. Replicas

`[replicas] devices = [[0], [1]]` starts one engine process per device group with the same model, and the API process becomes a router. Routing: least active sequences, with prefix affinity — the hash of the first 512 prompt tokens maps to a preferred replica so repeated system prompts hit that replica's prefix cache. `/v1/models` lists one model; `/r9v/status` lists per-replica state. Replicas are the spec 5 §3 "replicas" strategy and never share weights or state.

## 9. Security

- Default bind `127.0.0.1:8080`. Binding to any other address requires `server.api_key`; `server.admin_key` is required for admin routes regardless of bind and defaults to a random value printed at startup if unset.
- Request body limit 32 MB; prompt size limited by tokens (§3.3), not bytes.
- No TLS in-process; the docs recommend a reverse proxy, and the server refuses `0.0.0.0` without a key rather than warning.
- The Unix socket is created `0600`.

## 10. Config

```
[server]
bind             = "127.0.0.1:8080"
api_key          = none
admin_key        = auto
max_queue        = 64
request_timeout  = none
max_prompt_tokens = auto            # max_ctx − 64
chat_template    = none             # override path
[unix]
path             = auto
[sampling.defaults]
temperature = 1.0, top_p = 1.0, top_k = 0, min_p = 0.0, repetition_penalty = 1.0
[replicas]
devices          = none
```
