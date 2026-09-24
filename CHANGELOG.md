# Changelog

## v0.4.1 (unreleased)

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
- `tools/image_bundle.py --verify-only` downloads and verifies an image
  bundle without loading it.

### Before release

- The clean-install GPU test of this runtime (fetch, setup, first start with
  CED on and off, restart) has not run yet.
