# R9V setup guide for AI assistants

This file is for LLM agents helping a user install and run R9V. It tells you
what to check, what order to do things in, and what you must not work around.
Human-readable background is in [README.md](README.md); the authoritative
step-by-step is [docs/installation.md](docs/installation.md).

## What you are setting up

R9V is a catalog of pinned, model-specific inference profiles for AMD RDNA4.
One profile is currently runnable end to end:

- `qwen38-flash-next/ud-iq4-xs/dual-r9700-128k` (alias `qwen38`) — an
  OpenAI-compatible server for Qwen3.8 Flash Next on two Radeon AI PRO R9700
  GPUs. This guide covers that profile.
- `muse-glimmer-30b/v1/single-r9700` (alias `muse`) — `fetch` works, but
  `build` and `run` fail closed on purpose. Do not try to bypass that; tell
  the user the runtime is not released yet.

The `./r9v` CLI at the repo root drives everything. It is stdlib-only
Python 3.10+; there is nothing to `pip install`.

## Rules

1. **Check hardware before downloading.** The model package is 90.36 GiB. Run
   `./r9v doctor qwen38` first and stop if the host does not qualify.
2. **Do not update the submodules.** The pinned commits are release inputs.
   `git submodule update --remote` or checking out branch heads forfeits the
   qualified behavior.
3. **Do not skip hash verification.** If `verify --hash` fails, the bundle is
   wrong; re-fetch rather than proceeding.
4. **Ask the user before accepting the model license.** `fetch` requires
   `--accept-model-license` (Qwen Community License 1.0). That acceptance is
   the user's decision, not yours.
5. **Fail-closed errors are intentional.** If a profile refuses to run because
   of GPU architecture, device count, VRAM, or status checks, report it. Do
   not patch checks out.
6. **Device order is semantic.** TP rank 1 holds the larger dynamic cache. Do
   not shuffle `R9V_VISIBLE_DEVICES` to make an error go away without
   checking `amd-smi list` first.

## Host requirements

Verify all of these before step 1:

| Requirement | How to check |
|---|---|
| Two 32 GiB Radeon AI PRO R9700 (`gfx1201`) | `amd-smi list` |
| ROCm driver, `/dev/kfd` and `/dev/dri` access | `ls /dev/kfd /dev/dri` |
| ≥ 128 GiB host RAM | `free -g` |
| Docker daemon + Buildx plugin | `docker info`, `docker buildx version` |
| Python ≥ 3.10, Git, `curl` | `python3 --version` |
| Hugging Face CLI | `hf version` (needed by `fetch`) |
| ~150 GiB free disk | 90.36 GiB model + 26.82 GiB PLE + build/caches |
| User in `render` and `video` groups | `id` |

`./r9v doctor qwen38` automates most of this. Trust its output over your own
guesses.

## Setup sequence

Follow [docs/installation.md](docs/installation.md) exactly. Summary of the
happy path:

```bash
# 1. Clone with submodules (required — vendor forks and kernels live there)
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V

# 2. Sanity checks (no GPU or download needed)
./r9v validate qwen38
./r9v doctor qwen38

# 3. Download the model package (90.36 GiB, pinned HF revision).
#    Confirm license acceptance with the user first.
export MODEL_DIR=/path/with/100GiB/free/qwen38-r9v
./r9v fetch qwen38 --model-dir "$MODEL_DIR" --accept-model-license
./r9v verify qwen38 --model-dir "$MODEL_DIR" -- --hash

# 4. Build the runtime image. This source-builds PyTorch, Triton, AITER, the
#    vLLM fork, and the R9V kernels. Expect hours of CPU time and tens of GiB
#    of Docker build storage on the first run. Later runs reuse layers.
R9V_MAX_JOBS=8 ./r9v build qwen38

# 5. Derive the PLE payload (26.82 GiB, goes on a fast SSD).
#    Exact docker command: docs/installation.md step 4.
#    Verify: the output file must be exactly 28800138240 bytes.

# 6. Launch
export R9V_DATA_DIR=/fast-ssd/r9v
export R9V_PLE_PATH="$R9V_DATA_DIR/per_layer_token_embd.iq4_nl.bin"
export R9V_CACHE_DIR="$R9V_DATA_DIR/cache"
export R9V_VISIBLE_DEVICES=0,1        # confirm order with `amd-smi list` first
./r9v doctor qwen38 --model-dir "$MODEL_DIR"
./r9v run qwen38 --model-dir "$MODEL_DIR"
```

Long operations (fetch, build, PLE derivation) should be run in a way that
survives your session — suggest `tmux`/`screen` or run them detached and poll.

## Verifying it works

Startup takes several minutes. Poll rather than assuming readiness — up to
~15 minutes before concluding failure:

```bash
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8004/v1/models
```

Then issue a real completion. The served model name defaults to
`qwen3.8-flash-next` (confirm against `/v1/models`):

```bash
curl -fsS http://127.0.0.1:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-flash-next", "max_tokens": 32,
       "messages": [{"role": "user", "content": "Say hello."}]}'
```

If the health check never passes, read the container logs:
`docker logs --tail 200 r9v-qwen38-flash-next`.

## Environment variables

Set by the user or you; the launcher validates them. Caller values override
profile defaults.

| Variable | Default | Meaning |
|---|---|---|
| `R9V_MODEL_DIR` | required | Root of the verified model package |
| `R9V_PLE_PATH` | required | Path to the derived PLE payload |
| `R9V_CACHE_DIR` | `.cache` in repo | Runtime cache directory |
| `R9V_VISIBLE_DEVICES` | `0,1` | ROCm device order; rank order is semantic |
| `R9V_HOST_PORT` | `8004` | Host port for the OpenAI endpoint |
| `R9V_CONTAINER_NAME` | `r9v-qwen38-flash-next` | Container name |
| `R9V_MAX_JOBS` | — | Parallel build jobs for `build` |

## Common failure modes

- **`doctor` fails on GPU checks** — the host is not the reference topology.
  This profile only supports two `gfx1201` R9700s. Report it; do not force.
- **`fetch` fails** — check that the `hf` CLI is installed and the disk has
  space. The download targets a pinned revision, so a moved branch is never
  the cause.
- **`build` fails partway** — usually disk exhaustion or Docker/Buildx
  missing. Re-running reuses completed layers.
- **`run` refuses to start** — an existing container with the profile name is
  present. Stop and remove it deliberately:
  `docker stop r9v-qwen38-flash-next && docker rm r9v-qwen38-flash-next`.
- **PLE size mismatch** — re-derive; do not launch with a wrong-size file.
- **Server up but slow** — confirm device order with `amd-smi list` and that
  the PLE file is on a fast SSD.

## Where the details live

- [docs/installation.md](docs/installation.md) — full install and launch
- [docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md](docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md) — what was tested, on what hardware
- [docs/pi.md](docs/pi.md) — wiring the server into the Pi coding agent
- [docs/licensing.md](docs/licensing.md) — license boundaries
- [profiles/README.md](profiles/README.md) — profile lifecycle rules
