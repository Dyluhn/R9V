# Uncensored IQ4_XS on the consolidated 1.3.0 runtime (R9V v0.4.0)

Profile `qwen38-mtp4-uncensored`. Status: **experimental**. The runtime it
launches has served as the reference host's deployed service, and the public
fetch → setup → first start → restart flow from a clean checkout passed on the
reference host with CED on and off ([results](#clean-host-qualification-passed)).

## What the profile runs

- Model: [Dyluhn/Qwen3.8-Flash-Next-Uncensored-R9V-IQ4_XS](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-Uncensored-R9V-IQ4_XS)
  at revision `b06687cb2f83ea38039cadd249341a7bd5b76fa3`. All 24 published
  files were remotely size- and SHA-256-verified at that revision. It includes
  the split-16 CED projector.
- Runtime `qwen38-flash-next-gfx1201-mtp4-v3`: image
  `sha256:2dac17a2…` (the `v0.3.0-rc1-images` bundle) plus the consolidated
  1.3.0 overlays, byte-identical and SHA-256 pinned: nine Python files, the CED
  model file and four kernels (the mutable expert cache and the Q8 WMMA
  prefill kernels).
- Placement: fixed, manifest SHA-256 `d38c5ff3…`, hot experts 62 / 428,
  160 cache slots on rank 0. The mutable expert cache works only with this
  placement, so the profile refuses `--headroom`, `--calibration` and
  `--expert-catalog`.
- CED on by default: prompts of 8,192 tokens or more, 2,048-token exact tail,
  bf16 projector. `--ced off` in setup or start turns it off; a request opts
  out with `"vllm_xargs": {"r9v_ced": false}`.

## Proven on CPU (tests in this repository)

- **Launch parity.** `tests/test_launch_contract.py` runs `scripts/launch.sh`
  with this profile against a fake Docker and compares the container with the
  deployed service's create payload (normalized fixture in
  `tests/golden/launch/`). With the deployed CED default (off), the image,
  vLLM command, environment, mounts, host settings and published address
  (127.0.0.1) are identical. The
  release default differs only in `R9V_CED_DEFAULT=on`. With CED off, the
  container is the deployed one without the CED model file and the
  `R9V_CED_*` settings.
- **Pins.** The overlay hashes equal the 1.3.0 package manifest. The
  placement manifest equals the one the service mounts. The package's
  projector hash equals the projector the service loaded.
- **Refusals.** A modified, missing or extra overlay file stops `./r9v validate`
  and the launch. Invalid CED settings are all reported together before any
  container exists. Re-planning options are refused on this profile.

## Measured on the deployed service

Reference host: two Radeon AI PRO R9700 (rank 0 on PCIe Gen5 x16, rank 1 on
Gen4 x4), 128 GiB RAM, TP2, MTP4, one request at a time. Sources: the
consolidated 1.3.0 package's performance notes and the
[CED projector model card](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-Uncensored-CED-Projector).

| Measurement | Result |
|---|---|
| CED prefill speedup, 12.5K–19.5K-token prompts | 1.70× median (1.62–1.76×, n=6) |
| ~12K-token prompts, CED off → on | 1,868 → 3,018 tok/s |
| ~4K-token prompts (below the 8,192 threshold), CED server vs CED-off server | 1,783 vs 1,789 tok/s (unchanged) |
| Quality cost on long-context-dependent prompts | ×1.051 perplexity (ΔNLL +0.050 ± 0.037 nats/token, 15 prompts) |
| MTP tokens per step, first answer after a CED prefill | 0.895× of exact (0.78–0.95×, n=6); follow-up turn 1.02× (n=2) |
| Decode with the projector loaded vs not, ms/step | short prose 34.56 vs 34.76, short code 37.63 vs 37.78, ~8K 38.98 vs 38.46 (3–4 runs each) |
| Projector VRAM | 1.76 GiB per GPU (bf16) |
| Free VRAM with the projector loaded, idle | about 1.7–1.8 GiB (rank 0) and 2.1–2.2 GiB (rank 1) |

The speedup depends on prompt length, not just on crossing the threshold. In
the release config on a clean install it was 1.47× at 12,960 tokens, where the
exact tail rounds up to 3,264 tokens, and 1.80× at 31,987 tokens: about 1.5× at
~13K tokens, rising to about 1.8× at 32K and above.

An exact request after a CED request of the same prompt matched a fresh exact
run bit for bit, and there were no recompiles after the first CED request.
With CED off the service loads the image's own model file, so its compiled
model and rounding differ from the CED-on build: both are deterministic, but
not bitwise equal to each other.

## Clean-host qualification: passed

On 2026-09-23, on the reference host with the deployed service stopped, a
fresh recursive clone at commit `34285b2` ran the public flow with new model,
data, cache and state directories. The commits after it change only the
published address (now 127.0.0.1 by default) and documentation.
Summary with hashes:
[`results/uncensored-public-userflow-20260923.json`](results/uncensored-public-userflow-20260923.json).

| Check | Result |
|---|---|
| `fetch`, `verify --hash` | all 20 package artifacts downloaded without a token and SHA-256 verified |
| `setup` | passed in 133 s: PLE extracted, pinned image loaded, host checks passed |
| First start, CED on, cold compile cache | ready in 337 s including qualification, well inside `--timeout 2400` |
| First-start qualification, CED on | passed, including the 130,941-token prompt at 131,072 context; minimum free VRAM 1.771 / 2.087 GiB against the 1.5 / 1.5 GiB target |
| Runtime doctor | no failures; the expert ceilings (222 / 428) pass on the running server |
| CED on the running server | projector loaded on both GPUs; CED engaged on 12,960- and 31,987-token prompts; an exact request after a CED request matched a fresh exact run bit for bit; repeated greedy requests matched in text and logprobs |
| CED prefill speedup | 1.47× at 12,960 tokens (exact tail rounded up to 3,264), 1.80× at 31,987 tokens (2 runs each) |
| `--ced off` | qualified again on the next start and passed (minimum free VRAM 3.824 / 4.115 GiB); no projector loaded, no CED on the 12,960-token prompt |
| Unchanged restart | reused the receipt: no second qualification, ready in 167 s |

The first ~1,200 decode tokens after a fresh start run slower while caches warm
up (about 45–49 ms/step with CED on, 38–41 with CED off). Warm decode with CED
on measured 34.3 ms/step, matching the deployed service, where loading the
projector showed no steady-state decode cost. (The first three 400-token
answers after start and qualification took 47.18 / 48.88 / 44.78 ms/step with
CED on and 40.52 / 37.97 / 39.53 with CED off; the same three prompts, run
twice more in the CED-on container with identical tokens, took 34.05–34.76,
median 34.30. That re-run was not saved to a file, and warm CED-off decode was
not re-measured in the clean install.)

## Known limitations

- CED is text-only: prompts with images, and requests for prompt logprobs, run
  exactly. The projector was fitted on English-heavy code, docs and prose of up
  to ~20K tokens, and only a 2,048-token exact tail was graded.
- `int8` projector precision is accepted but was never run on the GPU.
- Decode warm-up after start: the first ~1,200 decode tokens after a fresh
  start run slower (about 45–49 ms/step with CED on, 38–41 with CED off) until
  caches warm up.
- The API has no authentication. R9V publishes it on 127.0.0.1 only unless
  `R9V_HOST_BIND` names another address; do not expose this model without
  authentication and moderation in front of it.
- The model is abliterated: its refusal behavior was removed. Add your own
  moderation before exposing it to anyone.
