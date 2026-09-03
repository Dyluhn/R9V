# R9V Repository Conventions

Status: active (2026-09-03). Authority: Spec 14 §2, Spec 15 §3, Spec 11 §11, and `phase-a-agent-breakdown.md` (Card A0.4).

This document defines repository-wide conventions for all crates, kernels, tools, and documentation in R9V. Read this before writing any code. Do not improvise error types, tracing fields, naming schemes, test layouts, or fixture locations.

---

## 1. Error Handling

### 1.1 Per-Crate Error Types
- Every workspace crate defines its own domain-specific error enum using `#[derive(thiserror::Error)]` (e.g. `IrError` in `r9v-ir`, `FormatError` in `r9v-format`, `StateError` in `r9v-state`, `LoaderError` in `r9v-loader`).
- Crate errors must be public (`pub enum <Crate>Error`) and derive `Debug`, implementing `std::error::Error` via `thiserror`.
- Where an operation depends on a lower crate in the downward dependency graph (Spec 14 §2), wrap the lower error using `#[error(transparent)]` and `#[from]`:
  ```rust
  #[derive(Debug, thiserror::Error)]
  pub enum LoaderError {
      #[error(transparent)]
      Format(#[from] r9v_format::FormatError),
      // ...
  }
  ```

### 1.2 Top-Level `R9vError`
- The top-level engine error enum is `r9v_common::R9vError` defined in `crates/r9v-common`.
- `R9vError` is strictly typed and owned; it avoids stringly-typed messages or generic boxed errors. At this foundation stage, it wraps shared fundamental errors: `ByteSize(#[from] ByteSizeError)` and `Io(#[from] std::io::Error)`.
- Downstream crates define explicit domain error enums (`IrError`, `FormatError`, etc.) and compose down the dependency graph via typed variants.

### 1.3 Errors Carry Numbers and Complete Context
- A refusal or validation failure must never emit a bare message. It must report what was required, what was available, the shortfall, and **every** failing item—never just the first one:
  ```rust
  #[derive(Debug, thiserror::Error)]
  pub enum LoaderError {
      #[error("device {device}: required {required} B, available {available} B, shortfall {shortfall} B; largest: {largest:?}; suggestion: {suggestion}")]
      Budget {
          device: u32,
          required: u64,
          available: u64,
          shortfall: u64,
          largest: Vec<(String, u64)>,
          suggestion: String,
      },

      #[error("{} tensor(s) missing or mis-shaped: {details:?}", details.len())]
      Tensors { details: Vec<TensorProblem> },
  }
  ```

### 1.4 Collect-All Validation Pattern
- Validation logic must accumulate all problems before returning an error:
  ```rust
  let mut problems = Vec::new();
  for item in items {
      if let Err(e) = validate_item(item) {
          problems.push(e);
      }
  }
  if !problems.is_empty() {
      return Err(ValidationError::Multiple { problems });
  }
  ```

### 1.5 Prohibition of Panics on Untrusted Input
- Never call `.unwrap()`, `.expect()`, or `panic!` on data originating from a file, network, request, environment, or device.
- All untrusted input must be parsed at system boundaries into typed structures via `Result`.
- `.expect(...)` is permitted only for internal invariant violations (unreachable logic verified by typing or invariants), and must carry a comment or message explaining why the invariant holds.

---

## 2. Tracing and Logging

Governed by Spec 11 §11.

### 2.1 Structured Output Format
- Engine logs are structured JSON lines: `{"ts": "...", "level": "...", "target": "...", "msg": "...", "fields": {...}}`.
- Standard levels: `error`, `warn`, `info`, `debug`, `trace`.
  - `info`: High-level operational events (model loaded, request admitted, step completed).
  - `debug`: Detailed inline execution records per step.
  - `trace`: Per-launch records for single-step performance and kernel investigations.

### 2.2 Mandatory Correlation Fields
- **Request-scoped logs**: Every log line associated with a user or client request must carry the structured field `req_id`:
  ```rust
  tracing::info!(req_id = %req.id(), "request admitted into queue");
  ```
- **Step-scoped logs**: Every log line associated with a scheduler step must carry the structured field `step_id`:
  ```rust
  tracing::info!(
      step_id = %step.id(),
      s = step.s,
      t_dec = step.t_dec,
      t_pre = step.t_pre,
      "step completed"
  );
  ```
- Combined step and request events include both fields where applicable:
  ```rust
  tracing::debug!(req_id = %req.id(), step_id = %step.id(), "token emitted");
  ```

### 2.3 Prompt and Token Privacy Rule
- Per Spec 11 §1 and §11: Log messages at level `info` or higher **must never** contain prompt text, completion text, or raw token IDs.
- Log token counts, sequence lengths, and content hashes (`xxh3_64`) instead. Raw tokens are visible only in specialized debugging sessions under explicit opt-in.

---

## 3. Naming Conventions

### 3.1 Identifiers and Types
- **Types and Traits**: `UpperCamelCase` (e.g. `SeqId`, `TensorLayout`, `QuantScheme`).
- **Functions, Methods, Variables, Modules**: `snake_case` (e.g. `parse_byte_size`, `xxh3_64`).
- **Constants and Statics**: `SCREAMING_SNAKE_CASE` (e.g. `DEFAULT_ALIGNMENT`, `MAX_WAVE_SIZE`).
- **Opaque IDs**: Distinct minimal newtypes wrapping private `u64` fields (`SeqId`, `ReqId`, `StepId`), with constructor `new(u64)` and accessor `as_u64()`. Never pass bare primitive integers (`u32`, `u64`) for semantic identifiers in public interfaces, and avoid lossy truncation or implicit conversions.

### 3.2 Closed Sets
- Closed sets defined in specs (ops, dtypes, schemes, layouts, state kinds, verify methods, proposer kinds) are Rust enums:
  - Exhaustive matching (no `_ =>` wildcard catch-alls in internal logic).
  - `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]` (and `serde::Serialize`, `serde::Deserialize` where applicable).
  - Serialized to stable lowercase `snake_case` strings, never raw discriminants.

### 3.3 Test Naming
- Name tests by the exact behavior and invariant being proven, not implementation details:
  - Good: `commit_with_partial_accept_keeps_verified_prefix()`
  - Good: `budget_refusal_reports_shortfall_and_all_contenders()`
  - Bad: `test_commit_2()`, `test_loader()`

---

## 4. Test and Fixture Layout

### 4.1 Unit and Integration Tests
- **Unit tests**: Internal private logic is tested in `tests` submodules (`mod tests { ... }`) within the respective file.
- **Integration tests**: Public crate interfaces are tested in `tests/*.rs` within each crate directory.
- **API shape tests**: API-bearing cards deliver `tests/api_shape.rs` checking visibility, `Send`/`Sync` markers, and trait implementations.

### 4.2 Fixture Locations
- **Rust fixtures**: Synthetic inputs and test helpers live under `tests/fixtures/<crate>/` or in test harness modules.
- **Golden files**: Structured outputs (partitioner graphs, summaries, load reports) are stored under `tests/golden/<crate>/<name>.<json|toml>`.
  - Golden files use normalized serialization: sorted keys, fixed spacing, no timestamps.
  - Regenerated only via `R9V_UPDATE_GOLDEN=1 cargo test -p <crate>`.
- **Quant tool fixtures**: Synthetic models live under `tools/r9v-quant/tests/fixtures/` and are generated deterministically by `tests/gen_fixture.py`.

### 4.3 Determinism and Property Testing
- Tests must be reproducible: no dependence on wall-clock time, environment variables, network access, or files outside the repository.
- Property-based tests use `proptest` with fixed seeds or `r9v_common::rng::SeededRng`.
- Determinism tests must execute the operation twice and verify bit-identical output.
- Numeric tolerances must come from the Spec 1 §6.1 table loaded as data, never hard-coded literal floats in test assertions.

### 4.4 GPU Test Placement
- GPU tests live in `tests/gpu/` and run only on the `gpu/gfx1201` runner.
- When hardware is absent, GPU tests must report the skip explicitly (e.g. `gpu::device_or_skip(...)`) and never pass silently as a dummy success.

---

## 5. DECISION Comments

When the spec leaves an implementation detail open, follow this procedure:

1. Select the simplest option that satisfies all governing principles in the relevant specs.
2. Mark the choice in code with a comment in the following format:
   ```rust
   // DECISION(<card-id>): <summary of choice>; rejected: <alternative rejected>. <Spec citation context>.
   ```
   Example:
   ```rust
   // DECISION(A0.4): byte-size parsing treats B, K/KB/KiB, M/MB/MiB, G/GB/GiB, T/TB/TiB as binary powers of 1024; rejected decimal 1000-based multipliers because memory/VRAM budgets in Spec 12 §3 and Spec 9 use standard binary allocation sizes.
   ```
3. List every `DECISION(<card-id>)` comment from implementation files in the PR body under the `## Decisions` heading with its `file:line` location.
4. **Constraint**: A `DECISION` comment is valid only for open details. It must never override or contradict clear spec text.

---

## 6. SPEC-ISSUES Procedure

When a spec contains a contradiction, omission, or error:

1. Never edit files under `specs/`. Specs are Dylan's authoritative decisions.
2. File an entry in `SPEC-ISSUES.md` using the standard format:
   ```markdown
   ## SI-<n> — <card id> — spec <n> §<x>
   What: <the sentence or gap, quoted or precisely located>
   Why it blocks or misleads: <one paragraph>
   Option taken: <what you did, or "stopped">
   Proposed resolution: <the spec edit you'd make, in one or two sentences>
   ```
3. If continuing work, cite the entry in code (`// DECISION(<card-id>): ... per SI-<n>`).
4. The issue blocks only the directly affected dependency line; independent cards and crates continue.
