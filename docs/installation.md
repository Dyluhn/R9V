# Install and run Qwen3.8 Flash Next

This guide covers the dual-R9700 MTP4 profiles: `qwen38-mtp4`
(UD-IQ4_XS, experimental), `qwen38-q4-xl` (UD-Q4_K_XL, experimental) and
`qwen38-mtp4-uncensored` ([its own section](#uncensored-profile-qwen38-mtp4-uncensored)).
Both require two 32 GiB `gfx1201` Radeon AI PRO R9700 GPUs, ROCm device access,
Docker, Python 3.10+, Git, `curl`, and storage for the model, 28,800,138,240-byte
PLE payload, image layers, and runtime cache. Device order is semantic.
Reserve at least **70 GiB** for image and cache import space; the public image bundle plus its containerd image-store footprint measured roughly **50 GiB**. This is in addition to the model, PLE payload and runtime cache.

Both profiles completed ordinary setup, first-start qualification and unchanged-receipt restart on the reference machine. See [qualification scope and reports](qwen-release-candidate.md). Each new machine or changed placement still runs its own checks.

Model download pages and pinned shard directories are linked in the [README](../README.md#model-downloads). For a versioned installation, use `git clone --recursive --branch v0.2.0 https://github.com/Dyluhn/R9V.git`; the commands below follow the current main branch.

## Check the host

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
./r9v list --by-topology
./r9v validate qwen38-mtp4
./r9v validate qwen38-q4-xl
./r9v doctor qwen38-mtp4 -- --host-only
./r9v doctor qwen38-q4-xl -- --host-only
docker info >/dev/null
docker info -f '{{ .DriverStatus }}'
```

Keep recursive submodule revisions pinned. Host-only checks do not qualify
serving or memory headroom.

Docker Engine 29 uses the containerd image store by default on fresh installs;
upgraded daemons may still use the legacy store. The output of the second
command should identify `io.containerd.snapshotter.v1`. If it does not, follow
the official [containerd image-store guide](https://docs.docker.com/engine/storage/containerd/)
and enable `"containerd-snapshotter": true` under `features` in the rootful
daemon's `/etc/docker/daemon.json`, then restart Docker. For rootless Docker,
use `~/.config/docker/daemon.json`, or `$XDG_CONFIG_HOME/docker/daemon.json`
when `XDG_CONFIG_HOME` is set, and restart the rootless daemon. The [Docker daemon configuration
reference](https://docs.docker.com/engine/daemon/) documents storage
locations. If you use rootless Docker, follow the official
[rootless mode guide](https://docs.docker.com/engine/security/rootless/),
confirm the intended context/socket with `docker info`, and run the same
containerd image-store check. Switching stores temporarily hides images and
containers created in the other store; revert to the prior configuration to
access them again, and preserve the existing store/workload context before
changing it.

## Fetch and setup

Read the Qwen Community License before accepting it. Fetch one profile into its
own model directory; never interchange the IQ4 and Q4 packages or placements.
Install the Hugging Face CLI if model artifacts are not already present:

```bash
python3 -m venv ~/.local/share/r9v/download-tools
~/.local/share/r9v/download-tools/bin/pip install -U huggingface_hub
export PATH="$HOME/.local/share/r9v/download-tools/bin:$PATH"
```

```bash
export MODEL_DIR=/fast-storage/qwen38-r9v
./r9v fetch qwen38-mtp4 --model-dir "$MODEL_DIR" --accept-model-license
./r9v verify qwen38-mtp4 --model-dir "$MODEL_DIR" -- --hash
```

Use `qwen38-q4-xl` in both commands for Q4. Setup selects the profile's
[`release/image-bundle-exact-host-20260912.json`](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc2-images), downloads and SHA-256 verifies its parts,
and loads the exact original image ID. Docker 29 must use the containerd image
store so the loaded image keeps its exact ID; check `docker info` as described
above before setup:

```bash
./r9v setup qwen38-mtp4 --model-dir "$MODEL_DIR" -- \
  --accept-model-license
```

For Q4, substitute `qwen38-q4-xl`. Use
`--gpu-bdfs BDF0,BDF1` for rank order, `--data-dir` for SSD data, `--ple-path`
to reuse a PLE file, and `--state-dir` for isolated resumable state. `--build`
is an explicit source-build choice and cannot accompany image options.

Setup verifies model artifacts, prepares/checks the PLE payload, saves machine
identity, and runs preflight. Repeat it after interruption.

## Start and qualify

```bash
./r9v start qwen38-mtp4
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8004/v1/models
```

The API has no authentication, so it is published on 127.0.0.1 only. To reach
it from other machines, run setup with `R9V_HOST_BIND=0.0.0.0` (or one
interface's IPv4/IPv6 address); setup saves it with `R9V_HOST_PORT`. An invalid
address stops the launch before anything starts. See
[API address](../profiles/qwen38-flash-next/dual-r9700/README.md#api-address).

Use `qwen38-q4-xl` for Q4. Start waits for health, runs the runtime doctor, and
performs the bounded workload qualification for the selected placement. A
reference seed is an estimate and still requires local validation. A different
headroom target can be planned from the portable release seed; matching local
calibration is an alternative when available:

```bash
./r9v start qwen38-mtp4 -- --headroom 5,5
```

Use `./r9v doctor PROFILE --state-dir DIR` for configuration and runtime
evidence, and `./r9v support PROFILE --state-dir DIR` for private diagnostics.
Keep the state directory the same across setup, start, doctor and support. Do not
publish prompts, completions, raw token IDs, or logs. Start refuses to replace
an existing profile container; inspect and deliberately stop that exact
container before retrying.

## Uncensored profile (`qwen38-mtp4-uncensored`)

This profile serves an abliterated model: its refusal behavior was removed and
it will comply with harmful requests that the original model refuses. Add your
own moderation before exposing it to anyone. R9V serves it on 127.0.0.1 only;
do not set `R9V_HOST_BIND` for this profile unless authentication and
moderation stand in front of the port. It is experimental; its public
setup, first start and restart passed on the reference host
([results](qualification/uncensored-v040.md)).

It uses its own model package (about 92.4 GiB, including the CED projector),
the consolidated 1.3.0 runtime with host expert dedupe (the IQ4 profile's image
plus SHA-256-pinned overlays) and a fixed expert placement. Start needs
56.3 GiB of RAM available: the host copy of the experts takes 40.3 GiB (rank 1
keeps its 400 most-used experts per layer in VRAM, so the host holds only the
rest), plus a 16 GiB reserve for the PLE table. Nothing on the host has to be
changed for it; it runs on stock rootless Docker. Use new model and state
directories:

```bash
export MODEL_DIR=/fast-storage/qwen38-uncensored
export STATE_DIR=/fast-storage/r9v-state/uncensored
./r9v setup qwen38-mtp4-uncensored --model-dir "$MODEL_DIR" \
  --state-dir "$STATE_DIR" --accept-model-license
./r9v start qwen38-mtp4-uncensored --state-dir "$STATE_DIR"
```

Setup downloads and SHA-256 verifies every package file, extracts the PLE
table, loads the pinned image and checks the host; `fetch` and `verify` work as
for the other profiles if you prefer to download first. The first start
compiles the model, so for this profile start waits up to 2,400 seconds by
default instead of 900 (`--timeout SECONDS` changes it). It then qualifies the
placement once; an unchanged restart reuses the receipt.

To record decode speed, run this right after start; it saves the numbers
right after start and once warm as JSON (a new file):

```bash
./r9v soak qwen38-mtp4-uncensored --state-dir "$STATE_DIR" -- --decode-speed decode-speed.json
```

The first ~1,200 decode tokens after a fresh start run slower while caches
warm up, so compare the warm median, not the first round.

**CED is on by default.** On prompts of 8,192 tokens or more, layers 0–15 run
exactly and the split-16 projector predicts the later layers for all but the
last ~2K prompt tokens. Decode stays exact. Measured on the reference host:
about 1.5× faster prefill at ~13K tokens, rising to about 1.8× at 32K tokens
and above, about ×1.051 perplexity
on prompts that depend on their long context, and about 10% fewer MTP tokens
per step on the first answer after a CED prefill. Prompts with images and
requests for prompt logprobs always run exactly.

- Turn CED off for the server: add `--ced off` to setup or start. The choice is
  saved; `--ced on` turns it back on. Switching qualifies the placement again
  on the next start.
- Keep one request exact: send `"vllm_xargs": {"r9v_ced": false}` (OpenAI
  Python client: `extra_body={"vllm_xargs": {"r9v_ced": False}}`).
- Trade some speed for quality: `--ced quality` (opt-in) uses a multi-source
  projector that also reads the inputs of layers 3, 7, 11 and 15. In one GPU
  grade of both it lost 10% of the long-context gain (×1.029 perplexity)
  where the default lost 17% (×1.049), for 1.55× instead of 1.68× prefill.
  Run `setup ... --ced quality` once: it downloads that projector (1.8 GB),
  which a default setup skips. It takes 1.79 GiB of VRAM per GPU, stored as
  int8, in a VRAM region it shares with the vision encoder's weights (image
  prompts never use CED; the region is refilled from RAM when the other is
  needed). Since v0.4.3 it passes first-start qualification on the reference
  host with 2.04 GiB free on GPU 0 (target 1.5 GiB). Start needs 60.8 GiB of
  available RAM with it instead of 56.3 GiB. `on` stays the default.

The projector takes 1.76 GiB of VRAM per GPU, so the profile keeps a 1.5 GiB
free-VRAM target per card instead of 3 GiB. The expert placement is fixed
because the runtime's mutable expert cache only works with it: `--headroom`,
`--calibration` and `--expert-catalog` are refused.

## Development builds

The public installation path uses the GitHub Release image bundle. Source builds
are a developer workflow requiring the repository's private pinned base and all
dependency inputs; they are outside the clean public reproduction path and are
not a substitute for the released image identity.

## Troubleshooting

| Symptom | Action |
|---|---|
| Download or hash verification fails | Check disk and `hf` access, repair or re-fetch the exact package, and do not continue with a failed hash. |
| Wrong GPU order or BDF mismatch | Run `amd-smi list`, then rerun setup with `--gpu-bdfs BDF0,BDF1`. |
| Insufficient requested headroom | Preserve the per-rank shortfalls. Adjust the target deliberately or let the release seed plan it; complete local workload qualification afterward. |
| Host normal-zone pressure | Free host memory or reduce CPU-offloaded residency; swap does not satisfy pinned-RAM requirements. |
| Startup/JIT timeout | Start waits 900 seconds by default, 2,400 for the uncensored profile. Increase `--timeout`, inspect retained Docker logs, and check image/cache space before retrying. |
| `runtime overlay problem` at launch | An overlay file under `runtimes/` no longer matches its pinned SHA-256. Restore the checkout (`git status`, `git checkout -- runtimes/`) instead of editing the file. |
| Existing container | Inspect the exact named container, save diagnostics, then deliberately stop/remove it before retrying. |
| Support needed | Run `./r9v support PROFILE --state-dir DIR`; keep the bundle private. |
