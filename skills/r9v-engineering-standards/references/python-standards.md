# Python standards (`tools/r9v-quant`)

The quant tool is a second stack with the same bar. It produces files the engine loads zero-copy, so a sloppy byte here becomes a wrong logit there.

## Layout

```
tools/r9v-quant/
  pyproject.toml, uv.lock           # pinned; extras: [cuda], [rocm], [cpu]
  src/r9v_quant/
    cli.py                          # argparse only; no logic
    io/gguf.py                      # container read/write via the gguf package; the only place bytes are touched
    families/<family>.py            # torch forward per spec 8 family; pure modules
    calib/                          # manifest schema, build, tokenization (engine tokenizer via r9v CLI)
    stats.py  smooth.py  sensitivity.py  assign.py  rounding.py  actmode.py  emit.py  verify.py
  tests/                            # small synthetic fixtures; minutes on CPU
```

Math modules are pure: tensors in, tensors out, no I/O, no prints, no globals.

## Typing

- `pyright --strict` clean. `from __future__ import annotations`.
- `@dataclass(frozen=True)` for records (`SchemeAssignment`, `CalibManifest`, `QualityReceipt`).
- Tensors are annotated with a shape comment at every operation: `h = x @ w.T  # [T, K] @ [K, N] -> [T, N]`.
- Enums mirror the closed sets by name (`class SchemeId(str, Enum)`), serialized by name.

## Determinism

At process start, once, in `cli.py`:

```python
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(args.threads)          # fixed; CPU reductions depend on it
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

- No `dict` iteration into output order without `sorted()`.
- No `set` for anything that reaches a file.
- Layer-at-a-time device residency is explicit (`.to(device)` / `.to("cpu")`), never left to the allocator.
- Two runs with the same inputs produce byte-identical output files; CI checks this on the 30M fixture.

## Numerics

- Accumulate in `float32` explicitly: `.float()` before any sum, matmul or Hessian update.
- Scale and min fitting for `I4_K` follows spec 13 §7; the fitted fields are packed by `io/gguf.py` into the exact `Q4_K` record layout and round-tripped through the engine's reference decoder in tests.
- Never rely on torch's default dtype; set it once and pass dtypes explicitly.

## CLI

- `argparse` with typed defaults matching spec 13 §13; `--help` text quotes the spec defaults.
- Exit codes: 0 success, 1 usage, 2 verification ceiling exceeded (file still written), 3 internal error with traceback.
- Progress to stderr; results to stdout as a table plus `--json` for machines.

## Tests

- `pytest`, fixtures under `tests/fixtures/`, generated deterministically by `tests/gen_fixture.py` (the 30M model shared with card A1.13).
- One test per pipeline stage on the fixture; one end-to-end test that quantizes, verifies, and compares two presets.
- Byte-identity test: quantize twice, `filecmp`.
- Round-trip test: emitted native file loads through `r9v eval` and produces logits within spec 1 §6.1 of the tool's own dequant.

## Hygiene

- `ruff` and `black` configured in `pyproject.toml`; CI runs both.
- No notebooks, no `print` debugging, no `# type: ignore` without a reason.
- New dependencies justified in the PR and added to the lockfile in the same commit.
