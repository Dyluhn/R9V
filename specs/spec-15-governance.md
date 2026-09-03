# Spec 15 — Contributing and Governance

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: all specs.

## 0. Purpose and scope

The rules for changing the project: license and sign-off, what each kind of change must include, the RFC process for the closed sets (ops, dtypes, schemes, acceptance rules), the standard a performance or quality claim has to meet, what "supported" means, how issues and security reports work, and how the specs themselves change.

## 1. Principles

1. **Specs are the contract; code implements them.** A change that alters behavior a spec describes changes the spec in the same PR. Code that contradicts a spec is a bug in one or the other, and the PR says which.
2. **Closed sets stay closed without an RFC.** Ops, dtypes, schemes, `verify` methods and layout ids are added by RFC, never by a PR that "just needs one more."
3. **Claims carry receipts.** A number without a spec 11 receipt or a spec 13 `compare` output is not a project claim; it's a conversation.
4. **The bar is the same for the owner.** Every rule here applies to the owner's own changes, including the runner gate and the receipt rule.
5. **Small surface for contributors, deterministic acceptance.** Most contributions should land in `r9v-models`, `r9v-spec`, `kernels/reference` or `tune/`. A change that touches core crates to support a model or method fails acceptance as a spec gap unless the governing spec already requires it.

## 2. License and sign-off

- **Apache-2.0** for the engine, the quant tool and the specs. Third-party code (T1 reference kernels adapted from MIT sources, the `gguf` Python package) keeps its license and is listed in `NOTICE`.
- **DCO sign-off** (`Signed-off-by`) on every commit; no CLA.
- Generated kernel source in `kernels/gen/` is covered by the same license as the generator.

## 3. Requirements by change type

| change | must include | gate |
|---|---|---|
| **new op / signature change** | spec 1 §7 RFC: attempted parametrization, full signature with sharding table and numerics, T0 + T1 implementations, golden and batch-invariance tests, at least one model using it, fusion-table changes, `ir_version` bump | RFC accepted, `gpu/gfx1201` green |
| **new dtype / scheme / layout id** | spec 2 RFC: scale record, dequant formula, bpw, matrix path, repack rule if from GGUF, round-trip test, `r9v-quant` support or an explicit "repack-only" note | RFC accepted |
| **new `verify` method** | spec 7 §9: proof or reference of distribution exactness, or `lossy` flag; tests against `Rejection` on the same drafts | RFC accepted |
| **new model family / checkpoint** | spec 8 §9: family function or list entry, `verify-arch` output committed under `support/`, golden prompts, support-matrix row | `gpu/gfx1201` green |
| **new proposer** | spec 7 §9: trait impl behind a feature, metrics evidence on the `accept` suite, `auto` order justification if it should be preferred | `gpu/gfx1201` green |
| **T2 fast path (new op or arch)** | spec 4 §10 gates, tune entry, regenerated source committed, achieved-bandwidth numbers in the PR | perf gate: no regression on existing variants |
| **T1 for a new arch** | spec 4 §13 steps 1–3; arch descriptor; golden on that hardware (contributor-run, receipt attached; the runner cannot test it) | `cpu-only` green plus the attached contributor receipt; no unavailable test is claimed |
| **new config setting** | schema declaration with doc, range, mutability, interactions; the owning spec's config section; settings index row | docs build |
| **scheduler / partitioner / state change** | spec update, simulation or pure-function tests in `cpu-only`, the relevant runner tests | both gates |
| **benchmark baseline update** | the new receipt, and the reason the number moved | receipt validity plus regression gates |
| **spec text only** | an owner-authored course correction stating which code it affects (or "none yet"); the affected spec's status line updated | docs/schema consistency gates |

Every PR template asks: *which spec does this implement or change?* "None" is a valid answer only for docs, CI and tooling.

## 4. RFC process

1. Open an issue with the `rfc` label using the template for the closed set being extended (op, dtype/scheme, verify method, layout id). The template mirrors the spec section's requirements.
2. Discussion happens on the issue. Dylan's resulting spec text is the decision: the issue is marked `accepted`, `rejected` or `needs-revision`, and acceptance names the spec sections that changed. There is no separate implementation-review gate after the spec decides it.
3. The implementing PR links the RFC, updates the spec, bumps the relevant version number, and lands with the reference tier first (spec 1 §7). Fast paths follow in separate PRs.
4. Rejected RFCs stay closed with the reason; a revised proposal is a new issue linking the old one.

Bar for an op RFC: the parametrization attempt is shown and fails for a concrete, named model. "It would be cleaner" is not a reason to add an op.

## 5. Claims

- **Performance**: a spec 11 receipt from `r9v bench`, valid, with the comparison tool's commit and command line if a comparison is made. Same-file rule (spec 11 §9.3). Receipts are attached to release notes, README numbers and any thread post that quotes a number. Kernel-mode receipts are labeled. README numbers come only from floor-meeting receipts (spec 11 §9.5).
- **Speculative decoding**: never claimed alone. A spec-decode number is accompanied by the spec-off `decode` receipt from the same session that meets the TG floor, and is stated as a multiplier over it (spec 11 §9.5).
- **Quality**: a spec 13 `verify` output, and for "better than X" a `compare` run on the same source model with both files' metadata. KL, top-1 and perplexity, with the calibration and holdout hashes.
- **Support**: a family or checkpoint is "supported" only with a `support/` entry (spec 8 §8) and a support-matrix row generated from it.
- Numbers in the README are regenerated from receipts in the repo by `xtask docs`; there are no hand-typed performance numbers anywhere in the docs.

## 6. Support matrix

Generated into `SUPPORT.md` from `support/`, `tune/` and the bundle manifest. Per checkpoint: family, format(s) tested, plan(s) tested, tier per op family on gfx1201 (T2 / T1), receipt links, verify-arch numbers. Per arch: reference-tier status (all ops T1 pass) and fast-path coverage (which op families have validated T2). Three words are defined and used consistently:

- **supported**: gated on the runner or by a committed contributor receipt; regressions are bugs
- **reference**: runs on T1/T0, correctness gated, no performance claim
- **untested**: nothing in `support/`; the loader still tries and the log says so

## 7. Issues

- **Bug**: the template requires the doctor bundle (redacted is fine), the request shape or command, and expected versus observed. Issues without a bundle get the `needs-bundle` label and a pointer to `r9v doctor`; the helper (spec 12 §7.5) exists to make this step trivial.
- **Performance**: requires a receipt, and the receipt's `validity` must be `valid`; an `invalid` receipt's reason is usually the answer.
- **Model request**: family, checkpoint, `general.architecture`, and whether a converter already produces standard GGUF for it.
- **Feature / RFC**: §4.

Triage labels: area (`ir`, `format`, `state`, `kgen`, `part`, `sched`, `spec`, `models`, `loader`, `serve`, `obs`, `config`, `quant`, `ci`), plus `needs-bundle`, `needs-receipt`, `gpu-approved`, `rfc`, `good-first-issue`. Good first issues are, by design, mostly in `models`, `kernels/reference`, proposers and docs.

## 8. Security

- Report privately to the address in `SECURITY.md`; acknowledged within 72 hours. Scope: the serving API (auth bypass, request-body handling, grammar engine), the loader (malformed GGUF), the helper's tool boundary, and the runner (spec 14 §5.3).
- The loader treats every file as untrusted: bounds-checked offsets, checksum verification, no code execution from metadata (chat templates render in a sandboxed Jinja subset with no filesystem or network access).
- Fixes for reported issues ship as patch releases with the receipt gate still applied.

## 9. Specification authority and decisions

- Dylan is the specification authority for v1. The checked-in specs are his decisions; implementations are accepted by their specified tests, receipts, and CI without human review. Area stewards may maintain evidence and triage for `support/`, `tune/`, families, or architectures, but they do not add an approval gate. Changes to a closed-set contract require Dylan to course-correct the governing spec first.
- Decisions that change a principle in any spec are recorded in `specs/DECISIONS.md` with the date, the reason and the alternative rejected. The corrections made while writing these specs (placement per load, PP-only bit-identity, host-computed experts, `I4_K` = `Q4_K` fields, one step graph with decode and prefill classes instead of two graph kinds, draft-model cost charged to the spec-decode budget rather than a fixed pre-step cap) are the first entries.
- The code of conduct is the Contributor Covenant, unmodified.

## 10. Changing the specs

- Each spec carries a status line (`draft 0.1` → `accepted 1.0` when its implementation lands and passes its gates → `1.x` on additive edits → `2.0` on a breaking change).
- A spec edit states which of the three it is and which code it affects. Dylan's checked-in edit is authoritative; breaking edits also require the RFC evidence described in §4, with no additional implementation review after the spec changes.
- Specs and code are versioned together; a release's docs include the spec versions it implements.

## 11. Roadmap (what "v1" means)

| phase | delivers | done when |
|---|---|---|
| **A. Foundation** | specs 1–4 implemented; single-GPU dense path with a minimal loader and a single-sequence scheduler (the subsets of specs 9 and 6 needed to run); T0/T0v/T1/T2 for the op set; native + GGUF loading; the spec 11 `decode` and `prefill` suites brought forward so the phase ends with receipts on the large dense reference model | runner gates green; a receipt beats the current R9V numbers at equal file |
| **B. Runtime** | specs 5–9: PP2 and TP2, state manager with prefix cache, scheduler with budget, spec decode (ngram, mtp, draft), hybrid and MoE families, host experts | all reference models supported; `multi`, `depth`, `accept` receipts |
| **C. Surface** | specs 10–15: API, observability, config schema, quant tool with `verify`/`compare`, CI with the runner, releases | a stranger can install a release, load a GGUF, get a valid receipt and file a bundle-backed issue without asking anything |
| **D. Helper and growth** | spec 12 §7; second arch at reference tier; external contributors on families and proposers | first external T1 arch receipt; first external family PR merged |

v1 is the end of phase C. Phase A is where the "numbers on my machine" problem is solved, because it ends with a receipt, not a number.

## 12. Attribution

`CREDITS.md` names the work this design borrows from and where: ggml/llama.cpp (GGUF container, K-quant grids, the closed-op-set idea, reference-kernel structure), vLLM (paged KV and block-table scheduling, the proposer/verifier split), SGLang (radix-style prefix reuse, tree verification), Megatron-LM (column/row-parallel sharding), GPTQ and SmoothQuant (rounding and folded smoothing), and the RDNA4 WMMA documentation from AMD GPUOpen. Borrowed ideas are credited in the spec that uses them; borrowed code is credited in `NOTICE`.
