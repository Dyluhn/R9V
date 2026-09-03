---
name: r9v-engineering-standards
description: The engineering standard for all work in the R9V inference engine repository — Rust crates, HIP kernels and the r9v-quant Python tool. Read this in full before writing any code for any card, spec, bug fix, test, kernel, or tool in r9v, and re-read the relevant section whenever you are about to make a design choice, handle an error, write a test, touch numerics, or optimize anything. It defines what correct, deterministic, reviewable, high-quality work looks like here, why each rule exists, and how to check your own work before handing it in. Use it together with r9v-card-work (process and acceptance); this skill is the craft.
---

# R9V engineering standards

Read all of it. It is long because it replaces the judgment calls you would otherwise make alone, and on this project a wrong judgment call is expensive: many agents are building against the same specs in parallel, and the engine's public promise is that its numbers are reproducible on anyone's machine. Both of those depend on every piece of code holding to the same standard, not on any one piece being brilliant.

The priorities, in order, when they conflict:

1. **Correct** — does what the spec says, for every input the spec admits, and refuses everything else with a reason.
2. **Deterministic** — same inputs, same bits, on every tier, every run.
3. **Clear** — the next agent, holding only the specs, can read it and know why it is the way it is.
4. **Fast** — only after the first three, only where the profiler says, and only with a receipt.

Nothing on this list is traded for anything below it. A fast kernel that is nondeterministic is a bug. A clear function that is wrong is a bug with good comments.

---

## 1. Mindset

**Understand before building.** Before the first line, be able to write, in one sentence each: what this change must do, which invariants the spec names for it, what the oracle is (T0, a torch reference, a golden file, an algebraic law), and how it can fail. If you can't write those four sentences, you're not ready; go back to the spec sections and their §1 principles. The principles are what decide the cases the detailed text doesn't cover.

**The spec is the requirement, the test is the proof, the receipt is the claim.** Code that matches your understanding but not the spec is wrong. Code that matches the spec but has no test is unproven. A performance number without a spec 11 receipt is an anecdote.

**Smallest correct change.** Solve the card. Do not add flexibility, configuration, abstraction or "hooks for later" that no current card needs. Speculative generality is how closed sets get quietly opened. If a later card needs something, it will add it against a spec that names it.

**No "while I'm here."** Unrelated cleanup in a card's PR makes the diff unreviewable and moves code someone else is building against. Note it in `SPEC-ISSUES.md` or a separate issue and leave it.

**Prefer reversible decisions.** When the spec leaves something open, choose the option that is cheapest to change later, mark it with a `DECISION` comment, and move on. When the open question is about an interface others depend on, numerics, or an on-disk layout, it is not reversible; stop and file a `SPEC-ISSUES.md` entry instead of guessing.

**Report exactly.** Say what you ran and what you didn't. "All tests pass" means every test named in the card ran green on the tier it belongs to. If you could not run the GPU tests, say so in those words. Never round up.

---

## 2. Correctness

### 2.1 Make invariants explicit in types

If the spec says a value is one of a fixed set, it is an enum. If it says a dimension is a multiple of 32, the constructor checks it once and the type carries the guarantee. If two ids are different kinds of thing (`SeqId`, `BlockId`, `Rank`), they are different newtypes so the compiler catches a swap. Prefer making illegal states unrepresentable over documenting that they're illegal.

```rust
// Right: the constraint lives in one place and the type proves it afterwards.
pub struct CtxLen(u32);
impl CtxLen {
    pub fn new(n: u32) -> Result<Self, IrError> {
        if n % 32 != 0 { return Err(IrError::NotBlockAligned { value: n, block: 32 }); }
        Ok(Self(n))
    }
}
```

### 2.2 Parse at the boundary, then trust the types

Files, requests, device results and environment are untrusted. Convert them into typed structures exactly once, at the edge (`r9v-format` reader, `r9v-serve` request mapping, `r9v-hip` result decoding), returning every problem found rather than the first. Inside the engine, functions take the typed form and do not re-validate. A function that re-checks what its argument type already guarantees is either wrong about the type or wasting the reader's attention.

### 2.3 Total functions

Every `match` on a closed-set enum is exhaustive with no wildcard arm. That is deliberate: when an RFC adds a variant, every site that must care fails to compile. `unreachable!()` is allowed only with a comment proving why, and never for input-dependent paths.

### 2.4 Errors carry the numbers

An error is a report to someone who has to fix something. "Budget exceeded" is useless; "device 0: required 33.2 GB, available 31.1 GB, shortfall 2.1 GB; largest: state pool group 0 (12.2 GB), blk.* ffn_up (6.4 GB)…" is actionable. Validation collects every failure before returning. Use the crate's `thiserror` enum and `r9v-common::R9vError` at the top; add context with `?` chains, not string concatenation.

`unwrap()`, `expect()` and `panic!` are for programming errors only — a violated internal invariant — and even then they carry a message naming the invariant. Anything that originated outside the process goes through `Result`.

### 2.5 Numerics by contract

Spec 1 §6 says, per op, the accumulation dtype, the order of reductions, where scales are applied and what is computed in f32. Implement exactly that formula, in that order. Do not rearrange for convenience even when the result would be within tolerance today: batch invariance and cross-tier agreement depend on the *order*, and a rearranged formula passes until the day a bucket changes.

- Accumulate in `i32` or `f32`. Never f16/bf16.
- Softmax: running max and sum in f32, ascending block order.
- Block scales: per spec 1 §6.2, the `I4_K` zero-point form as written.
- Tolerances are data from the spec 1 §6.1 table, loaded by the harness. A test that fails is a bug in the code, not a reason to widen the tolerance. Widening a tolerance is a spec change.

### 2.6 Determinism is correctness

The same inputs produce the same bits, on CPU and GPU, on every run, regardless of what else is in the batch. Concretely:

- No iteration over `HashMap`, `HashSet` or any unordered container that affects output values or order. Use `BTreeMap`, `Vec` with sort, or an index. If you need a map for lookup only, that's fine; don't iterate it into output.
- Reductions have a stated, fixed order: ascending index, ascending rank, ascending block. Write the order in a comment. No `par_iter().sum()`, no atomics into anything that reaches an output.
- Randomness is seeded and counter-based where the spec says (`Philox4x32` keyed by `(seq, step, draw)`); no `thread_rng()`, no `random()` outside tests.
- No wall-clock, no environment, no filesystem state in logic that affects outputs. Timing goes in the profiler, not in decisions.
- Floating-point: no `-ffast-math`, no `fma` contraction changes between tiers (`-ffp-contract=off` in kernels unless the numerics contract states the contraction), no reassociation.
- Test it: run twice, compare bytes. Run the same row alone, padded, and among random neighbors; compare bytes. The harness (card A1.10) does both; use it.

### 2.7 Purity where the spec says pure

The builder, partitioner, planner, cost model, repack rules and quant math are functions of their arguments. They do not read files, clocks or globals, do not log, and do not touch devices. This is what makes them testable with golden outputs and reusable from the doctor and the quant tool. If you find yourself adding an `info!` inside one, return the information instead and let the caller log it.

---

## 3. Rust

### 3.1 Ownership and data flow

- Own data at boundaries (the loader owns the arena; the scheduler owns sequences); borrow inside. Pass `&[T]`/`&mut [T]` and views, not `Vec` clones.
- No `Arc<Mutex<_>>` or locks on the step path. The scheduler is one thread by design (spec 6 §3.3); if you think you need a lock there, the design is being violated. Thread pools are for the loader's repack and T0v.
- No allocation on the step path. Arenas and workspaces are allocated at load (spec 9 §1); per-step buffers are reused. A `Vec::new()` inside the step loop is a bug even if it's small.
- Large buffers are never cloned. If two owners are needed, the design is wrong; use indices into one owner.

### 3.2 API shape

- Small public surface. `pub(crate)` by default; `pub` only for what another crate is documented to use.
- Closed-set enums are exhaustive, `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]`, and serialize by stable names, never by discriminant.
- Opaque handles (`StateHandle`, `SeqId`, `VariantHash`) expose no internals; the owning crate is the only one that can create or interpret them.
- Traits are for real polymorphism the spec names (`Proposer`, `Device`, `ProfileSink`, `GgufMeta`), with the smallest method set that spec requires. Do not introduce a trait to make one test easier; use a fake implementation of an existing trait instead.
- `dyn Trait` at boundaries where objects are few and long-lived (the device, the sink); generics in loops where monomorphization pays. Do not put `dyn` in a hot loop.
- Builders for anything with more than four optional fields; no `new(a, b, c, d, e, f)`.
- Public functions document their spec section, their preconditions, and what they return on each failure class.

### 3.3 `unsafe`

Only in `r9v-hip` and `r9v-t0`'s SIMD modules. Every `unsafe` block is preceded by `// SAFETY:` stating the invariant that makes it sound and who guarantees it, and is wrapped in a safe function immediately. An `unsafe` block longer than a screen is a design smell; split it. FFI signatures are transcribed from the header, not from memory.

### 3.4 Concurrency

- Single scheduler thread; device work is asynchronous via streams and events, not threads.
- Channels for progress and results (`std::sync::mpsc` or `crossbeam`); no shared mutable state between the loader's threads and the scheduler.
- Any thread pool has a fixed, configured size and a deterministic work split (by row-block, by tensor); reductions across workers happen in worker order.

### 3.5 Performance in Rust code

- Measure with the profiler and the trace before changing anything; the schedule log tells you where step time goes.
- Hot paths: no bounds-check-heavy indexing in inner loops where slices and iterators would do; no `format!` or logging; no `Result` allocation on the happy path.
- Cold paths: clarity wins; do not micro-optimize the loader's metadata parsing.
- Every optimization comes with the before/after from the same machine, fingerprint and command in the PR, labelled as an observation unless it is a receipt.

### 3.6 Hygiene

- `cargo fmt`, `cargo clippy -D warnings`, `cargo deny` clean. `#[allow(...)]` only with a comment saying why and scoped to the item.
- Module layout, naming, logging fields and test placement follow `CONVENTIONS.md`. Consistency with the codebase beats personal preference every time.
- `tracing` with structured fields (`req_id`, `step_id`, `card` where relevant); no `println!`; no logging of token ids or text at `info`.
- Doc comments on every `pub` item: what it is, which spec section defines it, what it returns on failure. Private items are commented where the *why* isn't visible in the code.
- No commented-out code, no `TODO` without a card id, no dead feature flags.

---

## 4. Kernels (HIP, T1 and generated T2)

### 4.1 Portable reference (T1)

T1 is the arch-independent proof that an op can run on a GPU. It is read far more than it is run:

- Index math in one clearly named place; bounds explicit; padding handled by masks, never by clamping that silently reads the wrong element.
- Wave intrinsics (`__shfl`, `__ballot`, LDS) are fine; `__builtin_amdgcn_*`, inline asm and anything gfx-specific are not. CI greps for them.
- Same numerics contract as T2. T1 is allowed to be slow; it is not allowed to be approximately right.
- Every T1 kernel takes the generated ABI struct (spec 4 §7) by value; no extra parameters, no globals, no dynamic shared memory sized at launch beyond what the struct states.

### 4.2 Generated fast path (T2)

Emitters live in `r9v-kgen` and produce HIP source. The source they produce is committed and audited from the diff and generated-source checks, so it must read like code a person would write:

- Named constants derived from the descriptor and the tile config appear as `constexpr` at the top of the emitted kernel with a comment naming their source (`// from ArchDescriptor.lds_bytes_per_wg`). No bare `32`, `64`, `16384` in the body.
- Arch instructions only through the leaf wrappers (`kgen/src/leaf/<arch>.rs`). The emitter body calls `wmma_iu8(...)`, never `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` directly.
- No dynamic allocation, no atomics into any output or partial that reaches an output, no data-dependent control flow that changes which lanes write where (predicate instead).
- Split-K and split-KV produce fixed-shape partials into the workspace and a separate reduce kernel combines them in ascending index order. The reduce kernel is generated from the same config so the two agree.
- Fixed-order wave reductions: the pattern in `references/kernel-patterns.md` §Reductions; do not invent a new shuffle tree per kernel.
- Alignment assumptions are stated with `__builtin_assume_aligned` and match the ABI's 256-byte promise. Loads are the width the layout promises (spec 2 §2.2): one 64-bit load per lane for int8 fragments, one 32-bit for int4. If the emitted code needs a gather or a permute to build a fragment, the layout is being used wrong.
- Register and LDS budgets come from the cost model. When promoting a variant, dump the ISA (`--save-temps`) and confirm the inner loop is the instruction sequence you intended; occupancy and spills are read from the compiler's resource usage, not guessed.

### 4.3 Numerics in kernels

- Online softmax: f32 running max `m` and sum `l`; rescale accumulators on max change; ascending block order; the pattern is in `references/kernel-patterns.md`.
- Integer paths: `i32` accumulation over full K for `PerToken`; per 32-block `i32` then f32 sum in ascending block order for `PerBlock32`; `I4_K` uses `s·(x·q − z·Σx)` with `Σx` computed once per token per block.
- fp8 cache: the per-token-head scale is applied to P (spec 1 §6.3), so the WMMA runs on raw e4m3.
- Conversions are explicit; no implicit `half`→`float` promotion inside a reduction; no `__fmul_rn` mixed with plain operators in the same expression.

### 4.4 Testing kernels

Every variant, T1 or T2, goes through the harness: golden vs T0 at spec tolerances on 32 random inputs including edge shapes; batch invariance (alone / padded / embedded); determinism (twice); shape fuzz for T1. Performance is recorded by the profiling hook, never asserted in a test, and gated only by the receipt-based regression check.

---

## 5. Python (`tools/r9v-quant`)

- Typed: `pyright --strict` clean; dataclasses for records; no untyped dict soup.
- Math in pure functions over tensors; I/O and CLI in separate modules. A function that quantizes a tensor takes tensors and returns tensors.
- Torch determinism: `torch.use_deterministic_algorithms(True)`, fixed seeds, fixed thread counts, `CUBLAS_WORKSPACE_CONFIG` set where needed. Two runs are byte-identical and CI checks it.
- Every tensor operation carries a shape comment (`# [N, K] @ [K, T] -> [N, T]`), and dtype promotions are explicit (`.to(torch.float32)` before any accumulation).
- No notebooks, no `print` debugging left in, no global state. CLI via `argparse` with typed defaults that mirror spec 13 §13.
- Tests use small synthetic fixtures (the 30M model from card A1.13) and finish in minutes on CPU.
- Dependencies pinned in the lockfile; adding one is justified in the PR.

---

## 6. Testing

Tests are the executable form of the spec. Write them as if the reader has the card and nothing else.

- **Name the behavior**: `commit_with_partial_accept_keeps_verified_prefix`, not `test_commit_2`.
- **One behavior per test**, arrange–act–assert, no shared mutable fixtures between tests.
- **Property tests for laws**: permute∘inverse = identity (layouts), dequant(source) = dequant(repack) (schemes), reserve/commit/rollback equivalence (state), chunked = recurrent (scan). Laws catch what examples miss.
- **Golden files for structure**: partitioner output, `LayerSpec` lists, load reports. Stored as data, regenerated only through an explicit command, and diffed by the implementing agent during self-review.
- **Fixtures are seeded and deterministic**, in the directories `CONVENTIONS.md` names. No network, no clock, no environment, no files outside those directories.
- **Failure paths get tests**: refusals with the exact numbers, malformed files, missing tensors, mismatched shapes. The error text is part of the contract.
- **Fake, don't mock**: a `FakeDevice` implementing the real `Device` trait beats a mock that asserts call sequences. Test outcomes, not choreography.
- **Tiers are honest**: CPU tests run under `cargo test`; GPU tests live under `tests/gpu/` and are skipped only when no device is present, never silently.
- **A test that compares an implementation to itself proves nothing.** The oracle is T0, a torch reference, a law, or a golden file.
- **Do not chase coverage.** The card's "done when" lines are the target; a test that exists to raise a percentage is noise.

---

## 7. Performance work

Only where the trace says, only one variable at a time, only with numbers from the same machine:

1. Profile (`profile.mode = kernel` or the trace) and identify the launch or phase that dominates.
2. Change one thing. Explain in the PR why the cost model predicts the change helps; if the cost model can't explain it, the change is not understood yet.
3. Measure before and after with the same command, fingerprint recorded, thermal state recorded.
4. Report it as an observation, or produce a receipt (spec 11 §9) if the claim is public.
5. Never optimize T0 or T1 beyond "runs at a usable speed"; their job is to be obviously correct.

A regression over 3% against the baseline fails CI. A change that speeds one variant and slows another is judged on the receipt, not on the intent.

---

## 8. Documentation and communication

- Doc comments say what, why and which spec section. `/// Spec 3 §3.6. Advances `ctx_len` by `accepted`; positions beyond it stay allocated and are overwritten by the next reserve.`
- `DECISION(<card>)` comments for every choice the spec leaves open; listed in the PR. Write the alternative rejected in one clause so the acceptance audit does not have to reconstruct it.
- `SPEC-ISSUES.md` entries for every place the spec is wrong or unclear, with a proposed resolution. An entry without a proposal is half an entry.
- Commit messages explain *why*: `A2.8: write the cache manifest last so a crashed repack leaves no partial cache`. Not `misc fixes`, not `wip`.
- PR bodies use the template and are literal about what ran where.
- Write for the next agent, who has the specs, the card and your PR, and no memory of your reasoning.

---

## 9. Self-review before handing in

Do this with the diff open, line by line, before running `check_card.py`:

1. Re-read the card. Does every "done when" clause map to an artifact in the diff?
2. Re-read the spec sections. Is there any sentence in them this code contradicts?
3. Grep the diff for: `HashMap`, `HashSet`, `unwrap(`, `expect(`, `panic!`, `thread_rng`, `TODO`, `todo!`, `unsafe`, `asm`, `32`, `64`, `640`, `gfx1201`, `println!`. Justify every hit or remove it.
4. Run the tests twice. Same bytes?
5. Run the batch-invariance test if any op or executor changed.
6. Trigger every error path once. Does each message carry the numbers and all failures?
7. Read every public item's doc comment. Does it name its spec section?
8. Compare the diff size to the card's size class.
9. Read the emitted kernel source (if any) as an adversarial acceptance audit would. Does it satisfy every cited rule?
10. Ask: would the author of the spec be surprised by anything here? If yes, it's a `SPEC-ISSUES.md` entry, not a surprise.

Then run `scripts/check_card.py` from `r9v-card-work` and walk its acceptance checklist.

---

## 10. What right looks like

```rust
// Wrong: silent, partial, nondeterministic.
let scale = tensors.get(name).unwrap().scale;
for (id, blk) in blocks.iter() { total += blk.bytes; }      // HashMap iteration order
if vram < needed { return Err(LoaderError::Budget); }

// Right: reported, complete, ordered.
let scale = tensors.get(name)
    .ok_or_else(|| FormatError::MissingTensor { name: name.to_owned() })?
    .scale;
let total: u64 = block_ids.iter().map(|id| blocks[id].bytes).sum();   // ascending id
if vram < needed {
    return Err(LoaderError::Budget {
        device, required: needed, available: vram, shortfall: needed - vram,
        largest: contributors.top(5),
        suggestion: Suggestion::LowerMaxCtx { to: fitting_ctx },
    });
}
```

```rust
// Wrong: a hardware fact in code.
const WAVE: u32 = 32;

// Right: the descriptor is the only source.
let wave = arch.wave_size;   // Spec 1 App. A
```

```rust
// Wrong: proves nothing.
assert_eq!(gemv_t2(&x, &w), gemv_t2(&x, &w));

// Right: an oracle, a tolerance from data, and invariance.
harness::golden(&op, &t2_impl, Tier::T0, Tolerance::for_op(op.id()));
harness::batch_invariant(&op, &t2_impl);
harness::deterministic(&op, &t2_impl);
```

```rust
// Right: a decision the spec leaves open, marked and reversible.
// DECISION(A2.8): manifest written after the last tensor, not incrementally;
// a crash mid-repack leaves no manifest and forces a clean re-repack. Rejected:
// incremental manifest (would need a validity marker per tensor). Spec 9 §5.3 is silent.
```

```text
Right: a SPEC-ISSUES entry with a resolution.
## SI-14 — A2.3 — spec 2 §3.3
What: Q5_1's per-32 record is listed as "as Q4" but Q4_1 has a min field and Q4_0 does not.
Why it misleads: the record size differs by 2 bytes; the SoA region stride depends on it.
Option taken: I5_B32FM uses the Q4_1 record (scale + min), I5_B32F uses the Q4_0 record.
Proposed resolution: replace "as Q4" with "as I4_B32F / as I4_B32FM respectively".
```

---

## 11. When you are stuck

1. Read the §1 principles of the spec involved; most gaps are decided there.
2. Read T0 for the op; it is the oracle and often the clearest statement of the formula.
3. Read how the sibling op or the neighboring crate solved the same shape of problem; match it.
4. If the gap is about an interface, numerics or layout: stop, file `SPEC-ISSUES.md`, hand the card back with what you found. That is a good outcome. Guessing at those three is the bad outcome.
5. If the gap is about anything else: take the simplest option, `DECISION` it, continue.

## References

Read these when the section above points to them, or before writing code in that area:

- `references/rust-patterns.md` — error types, newtypes, exhaustive matching, arenas, `FakeDevice`, `SAFETY` comments, builders, tracing fields.
- `references/kernel-patterns.md` — fixed-order reductions, deterministic split-K, online softmax, fragment loads from `L1`, ABI usage, tree masks, cache scale application.
- `references/testing-patterns.md` — harness invocation, property tests, golden files, GPU test skeleton, failure-path tests.
- `references/python-standards.md` — the quant tool's typing, determinism, layout and testing rules.
