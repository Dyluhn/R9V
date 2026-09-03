# R9V orchestration contract

The root agent is the orchestrator. It owns forward progress, card sequencing and dispatch, acceptance decisions, and the resolution of gaps or contradictions.

## Authority

- The specs are Dylan's decisions. Apply them directly; do not introduce a human-review or sign-off gate.
- Judge acceptance from the card's stated criteria, tests, CI, receipts, and the `r9v-card-work` rubric.
- Resolve gaps by following the principles of the owning specs and choosing the simplest conforming option. Record a `DECISION` or `SPEC-ISSUES.md` entry when the card workflow requires it, then keep independent work moving.
- The orchestrator judges spikes and independently verifies subagent output. A subagent's claim is not acceptance evidence.
- Do not weaken, reinterpret, or negotiate the performance floors in spec 11: decode is at least 0.93 of measured bandwidth with speculation off; prefill is at least 1.45 times the fastest llama.cpp backend on the same file at 2K and 8K.

## Subagent routing

Use this order without substitution:

1. Dispatch through `agy` to Gemini 3.8 Flash High (`gemini-3.8-flash-high`) until its usable quota is exhausted.
2. Then dispatch through `muse` with model id `muse-spark-1.3-contributor` until its usable quota is exhausted.
3. Do not use GPT or Claude subagents.

If the required `agy` model is not exposed by the installed client, report the exact availability problem. Do not silently substitute another model for production work.

## Gemini quota orchestration

The zero-cost machine-readable status command is `agy -p '/usage' --output-format json`. Gemini availability is constrained by both `gemini-5h` and `gemini-weekly` buckets; `reset_time` values are rolling-window timestamps. After any quota-exhaustion response, the root starts a non-blocking background timer for the latest `reset_time` among depleted Gemini buckets, continues on Muse meanwhile, re-queries status when the timer fires, rearms if necessary, and immediately returns new work to Gemini once both buckets permit it.

## Communication

Between work periods, communicate only completed evidence, decisions, blockers that need action, and the next material dispatch. Do not spend usage repeating that an agent is still running or narrating routine matching and testing.
