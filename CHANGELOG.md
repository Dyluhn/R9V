# Changelog

## v0.4.2 (unreleased)

### `qwen38-mtp4-uncensored`: CED quality mode (opt-in)

- `--ced quality` in setup or start runs a multi-source CED projector: the
  layer-16 state plus the inputs of full-attention layers 3, 7, 11 and 15.
  `on` stays the default; `off` and `on` launch exactly as in v0.4.1.
- The tradeoff, from one GPU grade of both projectors (both loaded as int8,
  eager grading server) on the 16 prompts that depend on their long context:

  | `--ced` | Perplexity | Long-context gain lost | Prefill speedup (median) |
  |---|---|---|---|
  | `on` | ×1.049 | 17% | 1.68× |
  | `quality` | ×1.029 | 10% | 1.55× |

  The projector math costs 56 ms per 1K approximated tokens instead of 24 ms.
- The quality projector ships stored as int8: 1.79 GiB per GPU, about what the
  default bf16 projector takes. In bf16 it would need 3.5 GiB per GPU and did
  not fit on GPU 1 in the grade. The int8 file loads bit-identically to what
  the grade ran.
- It is an optional file in the model package (1.8 GB). Setup downloads it
  only with `--ced quality`; `start --ced quality` before that is refused and
  says to run setup with it. The doctor checks it against its pinned SHA-256
  and counts it in the VRAM budget.
- Not yet run in the clean-install GPU test (compiled server, CUDA graphs).

### `./r9v doctor --runtime`

- Right after a first start, `runtime-kv-pressure` no longer fails because of
  qualification's own 130,941-token prompt, which is preempted about 7 times
  before it completes. Start records that count with the qualification
  receipt, and the doctor discounts it for the same container only. Any
  rewind after qualification still fails.

### Upgrading from v0.4.1

- The runtime descriptor changed (it pins the new model file), so an existing
  install qualifies again once on its next start, whatever its CED mode.

## v0.4.1 (2026-09-24)

### `qwen38-mtp4-uncensored`: smaller host expert copy

- Rank 1 now keeps its 400 most-used experts per layer in VRAM for good, so
  its host (RAM) copy of the experts holds only the other 112. The host copy
  drops from 55.4 GiB to 40.3 GiB.
- Start now needs **56.3 GiB** of available RAM (`R9V_MIN_HOST_AVAILABLE_BYTES`
  = 60,424,720,384) instead of 71.4 GiB: the host copy plus the 16 GiB PLE
  reserve. In the GPU test the most available RAM fell was 49 GiB (41 GiB
  shared host copy plus the workers), about 7 GiB under the floor.
- GPU test on the reference host against the 1.3.0 runtime, back to back:
  bit-identical prompt logprobs on 5 prompts (3K–15K tokens) and identical
  greedy probes on 2; decode 34.87 vs 34.83 ms/step on short prompts; a
  127,238-token prompt ran; peak shared host memory 41 GiB instead of 56 GiB.
- The pin list ships with the runtime, SHA-256 pinned twice (runtime descriptor
  and runtime). `tools/pin_sim/` regenerates it from the routing trace it
  records.
- Nothing on the host has to change: the host copy is not locked in memory,
  and the profile runs on stock rootless Docker.

### `./r9v doctor`

New checks, each with a fix that needs no root or host changes:

- `disk-space`: what fetch and setup still have to write, per filesystem.
- `vram-other-processes`: which programs hold VRAM on the R9V GPUs.
- `api-exposure`: warns when `R9V_HOST_BIND` publishes the unauthenticated API
  beyond this machine.
- `runtime-overlays`: every file mounted over the image matches its SHA-256.
- `ced-projector`: the CED projector is the package's pinned file, for the
  profile's split; its VRAM counts in the budget check.
- `expert-limit-consistency`: the full mutable cache's expert ceilings, cache
  slots and pin list match its fixed placement.

### Other

- `start` waits 2,400 s by default for a full mutable expert cache
  (`qwen38-mtp4-uncensored`).
- `./r9v soak PROFILE -- --decode-speed FILE` measures decode ms/step right
  after start and once warm and saves both as JSON, so a clean-install test
  keeps its warm decode numbers.
- `tools/image_bundle.py --verify-only` downloads and verifies an image
  bundle without loading it.

- `./r9v COMMAND PROFILE --state-dir DIR -- ARGS` no longer passes the `--` on to
  the tool; before, `soak ... -- --decode-speed FILE` and
  `doctor ... -- --runtime` failed with "unrecognized arguments".

### Upgrading from v0.4.0

- The runtime changed, so an existing install qualifies again once on its
  first start (a few minutes); later restarts reuse the new receipt.

### Clean-install GPU test (reference host, 2026-09-24)

Fresh clone, fetch, verify, setup, first start with CED on and off, and an
unchanged restart all passed:

- First start (cold compile, CED on): ready in 387 s; qualification passed
  all 7 checks, including a 130,941-token prompt. Peak shared host memory
  41.7 GiB.
- Warm decode 36.8 ms/step on the soak prompt mix (prose about 35.8).
- CED prefill speedup 1.52x at 12,970 tokens and 1.82x at 32,145 tokens. An
  exact request after a CED request was bitwise identical to a fresh exact run.
- Unchanged restart reused the receipt and was ready in 177 s; the
  running-server doctor had 0 FAIL.

### Known issue

- Right after a first start, `./r9v doctor --runtime` can report
  `runtime-kv-pressure` FAIL: qualification's own 130,941-token prompt is
  preempted a few times (it still completes and passes). The counter resets on
  restart, and the doctor is then clean. Planned for v0.4.2.
