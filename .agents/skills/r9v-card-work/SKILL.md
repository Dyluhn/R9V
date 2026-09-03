---
name: r9v-card-work
description: Autonomous workflow, boundaries and acceptance rubric for implementing or auditing work cards in the R9V inference engine repository (spec-driven Rust, HIP and Python tasks from phase-a-agent-breakdown.md and the fifteen specs in specs/). Use this whenever a task mentions a card id like A3.4, a spec number, SPEC-ISSUES.md, DECISIONS.md, CONVENTIONS.md, or any r9v crate, kernels/, tools/r9v-quant, tune/ or bench/; and whenever working inside the r9v repo at all. The specs are the decisions, and cards are accepted by their tests, receipts, CI, and this rubric without a human review gate.
---

# R9V card work

You are implementing one card from `phase-a-agent-breakdown.md` (later phases will have their own breakdowns with the same card format) against the specs in `specs/`. The specs are the contract. The card is the scope. This skill is the rubric. A card is accepted when its "done when" tests pass **and** every item in `references/acceptance-checklist.md` is true. Nothing else counts, including how good the code looks.

The reason the bar is shaped this way: this engine's whole pitch is that its numbers are reproducible and its architecture stays clean as features land. Both of those are properties of process, not talent. A card that is 95% right but edits a spec, adds an op, or lands without its tests costs more than a card that isn't done yet, because it silently moves the contract everyone else is building against.

## 1. Read before you write

In this order, every time, even for a card you think you understand:

1. **`ORCHESTRATION.md`** at the repo root. It defines root-agent authority, permitted subagent routing, and communication policy.
2. **The card.** Its `crates`, `deps`, `spec` sections, `deliverables`, `done when`, `GPU`, `size`. If a dep isn't merged, stop and say so; do not implement against an unmerged branch.
3. **The spec sections the card names**, in full, plus §1 (Principles) of each spec involved. The principles are what you fall back on when the detailed text is silent.
4. **`CONVENTIONS.md`** at the repo root. Error types, logging fields, naming, test layout, fixture locations. Do not improvise any of these.
5. **`SPEC-ISSUES.md` and `DECISIONS.md`**, searching for the card's crate and spec sections. Someone may have already decided the thing you're about to guess.
6. **The existing public API of the crate you're touching**, and the spec sections that define it. Implement that API; do not redesign it. Cross-crate cohesion is checked by card A6.7a.
7. **`references/spec-map.md`** if you need to find which spec owns something.

If reading these takes a while, that's the job. The specs are long because they're meant to replace conversation.

## 2. Boundaries

These are hard lines. Crossing one is an automatic reject regardless of what else the PR does.

**Specs are read-only.** Never edit anything under `specs/`. If a spec is wrong, ambiguous, or silent in a way that a `DECISION` comment can't honestly cover, write a `SPEC-ISSUES.md` entry (§5) and proceed with the option you took, or stop if you can't proceed honestly.

**Stay inside the card's crates.** The card lists the crates you may touch, plus tests and fixtures for those crates. Needing a change in another crate means one of: the card's deps aren't done, the card is mis-scoped, or a spec has a gap. All three are `SPEC-ISSUES.md` entries, not edits.

**Closed sets are closed.** Do not add an op, a dtype, a quant scheme, a layout id, a state kind, a `verify` method, a config setting, a metadata key, or a proposer kind. If the card's deliverable seems to require one, the spec disagrees with the card and that's a `SPEC-ISSUES.md` entry. The lists live in spec 1 §4, spec 2 §3, spec 3 §2, spec 12 §3.

**`unsafe` only in `r9v-hip` and the SIMD paths of `r9v-t0`.** Inline asm only in `r9v-kgen/src/leaf/`. Both are grepped in CI; `scripts/check_card.py` runs the same grep locally.

**Generated and measured files are never hand-edited.** `kernels/gen/**`, `tune/**`, `bench/baselines/**`, `docs/config.md`, `SUPPORT.md`, `support/**` are produced by `xtask` commands or by the runner. A card that owns one of them regenerates it and commits the output; no other card touches them.

**No new dependencies without a line of justification in the PR**, and none that `cargo deny` rejects. Prefer the workspace's existing crates. The engine starts without ROCm and must keep doing so; nothing above `r9v-hip` links HIP.

**No hardware facts outside the arch descriptor.** Wave size, LDS size, instruction availability, bandwidth, P2P: all come from `ArchDescriptor` or the topology. A literal `32` for wave size in kernel-adjacent code is a reject.

**Never claim performance.** A number in a PR is a receipt (spec 11) or it's an observation labelled as such. Do not write "this is faster" in a commit, a doc comment or a PR without the receipt path.

## 3. Quality bar

**Tests are the deliverable.** Every card's "done when" names tests; they exist, they run in the right CI tier (`cpu-only` or `gpu/gfx1201`), and they fail without your change. Use the shared harness from card A1.10 for any op-level work rather than writing a bespoke comparison. Fixtures are seeded and live where `CONVENTIONS.md` says.

**Determinism is not optional, on any tier.** Fixed reduction order, no iteration over `HashMap`/`HashSet` in anything that affects output or order, no wall-clock in logic, seeded RNG only. Spec 1 §6.1's batch invariance applies to your CPU code exactly as it applies to kernels. If you can't make something deterministic, that's a `SPEC-ISSUES.md` entry, not an `#[ignore]`.

**Purity where the spec says pure.** The builder, partitioner, planner, cost model and repack rules are functions of their inputs. No I/O, no globals, no clocks, no logging side effects inside them. Tests for these are golden-output tests and they should be easy to write; if they aren't, something impure leaked in.

**Numerics follow the contract, not the convenient path.** Accumulation types, block-scale application order, softmax in f32, the `I4_K` zero-point form: spec 1 §6 says exactly what to do and the T0 reference is the oracle. Matching T0 within tolerance by a different formula is a bug waiting for a batch-size change.

**Errors carry the numbers.** A refusal (budget, shape, missing tensor) reports what was required, what was available, and every failing item, not the first one. Use the `r9v-common` error type; no `unwrap()` or `expect()` on anything that depends on input.

**Public items have doc comments** that say what the thing is for and which spec section defines it (`/// Spec 3 §3.6`). Private items need comments only where the spec's reasoning isn't obvious from the code.

**No unowned stubs.** `todo!()`, `unimplemented!()` and `// TODO` are rejects unless the line carries the later card id that explicitly owns the missing behavior (`// TODO(A5.2): ...`). API-bearing cards implement everything their own deliverables require; they do not stop at a placeholder surface.

**Match the size.** A card marked S that arrives as 3,000 lines is a signal you've solved a different problem. Stop and check the scope before continuing.

## 4. When the spec is silent

Take the simplest option that satisfies every stated principle of the specs involved, and mark it:

```rust
// DECISION(A2.8): repack cache manifest is written after the last tensor,
// not incrementally; a crash mid-repack leaves no manifest and forces a
// clean re-repack rather than a partial cache. Spec 9 §5.3 doesn't say.
```

List every `DECISION` line in the PR body under its own heading. The acceptance audit reads that list first; it records what the spec did not cover. A `DECISION` is for choices the spec genuinely leaves open. It is not a way to override something the spec does say.

## 5. When the spec is wrong or unclear

Append to `SPEC-ISSUES.md`:

```
## SI-<n> — <card id> — spec <n> §<x>
What: <the sentence or gap, quoted or precisely located>
Why it blocks or misleads: <one paragraph>
Option taken: <what you did, or "stopped">
Proposed resolution: <the spec edit you'd make, in one or two sentences>
```

Then either continue with the option you took (and reference `SI-<n>` in the relevant `DECISION` comment) or stop and hand the card back if the option would be a guess about numerics, layout, or an interface someone else is building against. The person resolving issues turns them into spec edits or `DECISIONS.md` lines; you do not.

## 6. API-bearing cards

An API-bearing card produces the complete public surface and the behavior its deliverables assign. It proceeds directly from the specs and does not wait for approval:

- Every public type, trait and function from the card's spec sections is present with a doc comment naming its spec section.
- Signatures reflect the spec's semantics: things the spec calls immutable are not `&mut`; things the spec calls pure take and return values; state handles are opaque; closed sets are enums, not strings.
- Implement and test every behavior owned by the card. A placeholder is allowed only when a later card id explicitly owns that behavior.
- Include `tests/api_shape.rs` to assert compilation, visibility boundaries, and the `Send`/`Sync` requirements the spec states.
- Do not redesign the public surface during downstream work. If implementation reveals a real contradiction, file `SPEC-ISSUES.md`; clear spec text wins automatically.

After the individual API-bearing cards complete, A6.7a builds a cross-crate integration fixture and audits public-surface ownership before the interface freeze.

## 7. GPU cards

If the card says `GPU: yes`, you probably can't run its tests where you are. That's expected. Deliver:

- the code, the runner test in `tests/gpu/` or the harness invocation, and the workflow hook if the card asks for one;
- a PR body that says plainly which tests you could run (hosted `cpu-only`, a stub device, compile-only) and which are waiting on the runner;
- for kernel cards, the emitted HIP source committed under `kernels/gen/` **only** via `cargo xtask gen`, never pasted.

Do not mark a GPU card done before its required runner checks pass. The runner result flips it automatically; no human sign-off is required.

## 8. Delivering

**Commits**: `<card id>: <imperative summary>` in the first line; DCO `Signed-off-by` on every commit.

**PR body**: use `references/pr-template.md` verbatim. It asks which spec sections you implemented, lists `DECISION` lines, new dependencies, `SPEC-ISSUES` filed, tests run and tests pending on the runner.

**Before opening the PR**, run:

```bash
python scripts/check_card.py --card <id> --base main --pr-body <path-to-pr-body.md>
```

It checks the mechanical rules (no `specs/` edits, `unsafe` and asm placement, generated-file edits, `DECISION` lines enumerated, card id in commits, dependency justification). It cannot check judgment; the checklist does that.

**Then walk `references/acceptance-checklist.md` yourself**, honestly, and put the result in the PR body. CI and the acceptance audit verify the mechanical items. If any required item is "no", the card is not accepted; record the failure and continue independent work.

## 9. Accepting a card autonomously

A card is accepted when all of the following hold. There is no partial credit and no "good enough for now":

1. `scripts/check_card.py` passes on the final commit.
2. Every "done when" test in the card exists, is in the correct CI tier, and is green (or, for GPU cards, green on the runner).
3. Every applicable item in `references/acceptance-checklist.md` is true, verified from the repository state and command outputs rather than asserted from the PR body.
4. For API-bearing cards: API-shape tests pass; A6.7a is the later cross-crate cohesion gate.
5. A `SPEC-ISSUES.md` entry blocks only the dependency line it contradicts. Unaffected deliverables and independent cards continue.

Failures cite the checklist item or the card's "done when" line. Preference is not a failure reason unless it maps to the specs or rubric. If the checklist is missing something material, record it in `SPEC-ISSUES.md`; do not invent a one-off gate.

## 10. Things that get cards rejected, from experience

- Editing a spec "just to fix a typo".
- Adding a helper op or a "temporary" scheme because it made the card easier.
- A hard-coded wave size, LDS size, or `640.0` GB/s.
- Tests that pass because they compare the implementation to itself.
- A `HashMap` iteration that determines output order.
- `// TODO` with no card id.
- A PR body that says "faster" with no receipt.
- Implementing the card's deps yourself because they weren't merged yet.
- Touching `tune/` or `kernels/gen/` by hand.
- Leaving behavior owned by an API-bearing card as an unowned placeholder.
- An `unwrap()` on a value that came from a file.
- A `SPEC-ISSUES.md` entry that proposes no resolution.

## References

- `references/acceptance-checklist.md` — the rubric, walked by the implementing agent and verified from repository evidence.
- `references/spec-map.md` — which spec and which crate own what; the closed sets; the config sections.
- `references/pr-template.md` — the PR body template.
- `scripts/check_card.py` — the mechanical checks; run it before opening a PR.
