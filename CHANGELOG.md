# Changelog

## v0.4.3 (2026-09-24)

### `qwen38-mtp4-uncensored`: `--ced quality` now fits next to the vision encoder

- `--ced quality` now passes first-start qualification on the reference host,
  with image support kept. In v0.4.2 it failed there: GPU 0 fell to 0.90 GiB
  free against the 1.5 GiB target.
- Measured on the compiled server, two things made the difference:
  - The vision encoder's weights (0.42 GiB per GPU) and the quality projector
    (1.79 GiB per GPU) were both in VRAM all the time, although no step ever
    uses both: image and video prompts never use CED. They now take turns in
    one VRAM region per GPU, sized for the projector. Both stay in pinned host
    RAM, and the region is refilled when the other one is needed: the
    projector before an approximate prefill chunk, the vision encoder before
    an image is encoded. A swap takes 36 ms (projector) and 9 ms (vision) on
    GPU 0 (PCIe Gen5 x16), and 266 ms and 63 ms on GPU 1 (Gen4 x4); it happens
    only when a CED prompt follows an image or the other way round.
  - Each approximate chunk made two 400 MiB bf16 copies of the projector's
    int8 "final" map. The maps are now dequantized at most 2,560 rows at a
    time, in place. The KV and GDN state CED writes is bitwise unchanged;
    the "final" output, which only the MTP drafter reads, differs in rounding.

  The region alone brought GPU 0 from 0.84 to 1.25 GiB free; the smaller
  dequantization buffers did the rest.
- Clean-install GPU test on the reference host (fresh clone, setup and first
  start with `--ced quality`, desktop apps holding 1.23 GiB of GPU 0):

  | | v0.4.3 `quality` | v0.4.2 `quality` | v0.4.3 `on` |
  |---|---|---|---|
  | Min free VRAM in qualification, GPU 0 / GPU 1 | **2.04 / 2.49 GiB** (passes) | 0.90 / 1.77 GiB (failed) | 1.65 / 2.12 GiB |
  | Prefill speedup, ~12.8K tokens | 1.47× | 1.49× | 1.58× |
  | Prefill speedup, ~32K tokens | 1.62× | 1.64× | 1.76× |
  | Warm decode (ms/step) | 35.8–41.6 | 38.4–39.1 | 36.0–43.2 |
  | MTP tokens/step, answer after a CED prefill vs exact | −6% | −6% | |

  Decode is the same path in every mode; the spread is host noise. 24
  alternating long-CED and image requests (a new image each time) swapped 24
  times with VRAM flat (GPU 0 free 2,088 MiB throughout) and read every
  image's text correctly. A long CED request with 3 image requests sent
  during it: all completed (the images waited for the prefill, about 11 s).
  The vision encoder's output was bitwise identical to `--ced on`'s (no
  shared region) for 8 test images, before and after swaps (checked with a
  development build that logs a hash of it); the text answers to the same
  image vary slightly between server starts in every mode, as exact outputs
  already did. Exact requests after a CED request were bitwise identical to
  fresh exact runs, repeated greedy runs were identical, and nothing compiled
  after the first CED request. `on` and `off` qualified in the same session
  and an unchanged restart reused its receipt (ready in 177 s).
- Start needs 4.50 GiB more available RAM with `--ced quality` (60.8 GiB
  instead of 56.3): every TP rank keeps pinned host copies of the projector
  and of its share of the vision encoder, in 64 MiB slabs. The doctor adds
  them to the start check and says so. Measured: 4.2 GiB less available RAM
  at peak than `on`.
- `on` and `off` are unchanged: the shared region is only in the CED quality
  model file, which is mounted only with `--ced quality`.
- Upgrading: the runtime descriptor changed, so an existing install qualifies
  again once on its next start, in any CED mode.

## v0.4.2 (2026-09-24)

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
- **Not recommended (tested after release on the compiled server).** In the
  clean-install GPU test (fresh clone of v0.4.2, setup and first start with
  `--ced quality`), the server loaded the projector with its five sources and
  ran all seven qualification checks, including the 130,941-token prompt, but
  its peak VRAM is about 0.75 GiB higher on GPU 0 and 0.35 GiB higher on GPU 1
  than `on`. With desktop apps holding 1.24 GiB of GPU 0, the minimum free
  VRAM was 0.90 GiB on GPU 0 and 1.77 GiB on GPU 1 (target 1.5 GiB), so
  qualification failed and `start` exited with an error. `on` passed in the
  same session with 1.65 / 2.12 GiB free. Quality can pass only when other
  programs use less than about 0.6 GiB of GPU 0.
- Measured on the compiled server, same session:

  | | `quality` | `on` |
  |---|---|---|
  | Prefill speedup, 12,900 tokens | 1.49× | 1.54× |
  | Prefill speedup, 32,253 tokens | 1.64× | 1.74× |
  | Warm decode (ms/step, 3 runs) | 38.4–39.1 | 39.0–41.6 |
  | MTP tokens/step, answer after a CED prefill vs exact | −6% | −7% |

  Decode is the same path in both (short prompts do not use CED); the spread
  is host noise. In both modes an exact request after a CED request was
  bitwise identical to a fresh exact run, repeated greedy runs were identical,
  and nothing recompiled after the first CED request. `on` stays the default
  and is unchanged from v0.4.1.
- The model package moves to revision `8112610745a8ddc3a19cc659314af245820ee728`,
  which adds the quality projector; the 24 files published before are
  unchanged, so an existing install downloads nothing new unless it picks
  `--ced quality`.

### `./r9v doctor --runtime`

- Right after a first start, `runtime-kv-pressure` no longer fails because of
  qualification's own 130,941-token prompt, which is preempted about 7 times
  before it completes. Start records that count with the qualification
  receipt, and the doctor discounts it for the same container only. Any
  rewind after qualification still fails.

### Upgrading from v0.4.1

- The runtime descriptor changed (it pins the new model file), so an existing
  install qualifies again once on its next start, whatever its CED mode.

### Testing

- CPU tests, static checks and the launch parity tests (`on` and `off` launch
  exactly as the GPU-tested v0.4.1 container) pass. The clean-install GPU test
  was not rerun for v0.4.2.

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
