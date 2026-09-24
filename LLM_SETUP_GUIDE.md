# Setup guide for AI assistants

This guide covers `qwen38-mtp4` (UD-IQ4_XS, MTP4, 128K, experimental), `qwen38-q4-xl`
(UD-Q4_K_XL, MTP4, 128K) and `qwen38-mtp4-uncensored` (see the end of this guide) on
two 32 GiB `gfx1201` Radeon AI PRO R9700 GPUs.
Keep their model packages, catalogs, manifests, runtime descriptors, and
calibration records separate.

Check ROCm, `/dev/kfd`, render access, Docker, Python 3.10+, Git, `curl`, and
storage first:

```bash
./r9v list --by-topology
./r9v validate qwen38-mtp4
./r9v validate qwen38-q4-xl
./r9v doctor qwen38-mtp4 -- --host-only
```

If model files are not already present, install the Hugging Face CLI in an
isolated environment: `python3 -m venv /tmp/r9v-tools && /tmp/r9v-tools/bin/pip
install -U huggingface_hub`, then add `/tmp/r9v-tools/bin` to `PATH`.

After the user reads and accepts the Qwen Community License:

```bash
export MODEL_DIR=/fast-storage/qwen38-r9v
./r9v fetch qwen38-mtp4 --model-dir "$MODEL_DIR" --accept-model-license
./r9v verify qwen38-mtp4 --model-dir "$MODEL_DIR" -- --hash
./r9v setup qwen38-mtp4 --model-dir "$MODEL_DIR" -- \
  --accept-model-license
```

For Q4, substitute `qwen38-q4-xl` in every command. Setup automatically reads
[`release/image-bundle-exact-host-20260912.json`](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc2-images), verifies every SHA-256 part, and loads the
exact image ID from the GitHub Release bundle. Docker 29 must use the containerd
image store; verify this with `docker info`. `--build` is the explicit source
build alternative. Never mix IQ4 and Q4 placement artifacts.

Start and check readiness:

```bash
./r9v start qwen38-mtp4
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8004/v1/models
```

The API has no authentication and is published on 127.0.0.1 only. Expose it
only if the user asks: run setup with `R9V_HOST_BIND=0.0.0.0` (or one
interface's address), and tell them to put authentication or a firewall in front.

Use `qwen38-q4-xl` for Q4. Start runs runtime checks and bounded qualification;
health alone is not qualification. A reference memory seed remains an estimate
and each new placement needs local workload validation. The portable release
seed can plan another headroom target; matching local calibration is also
accepted:

```bash
./r9v start qwen38-mtp4 -- --headroom 5,5
```

Do not expose prompts, completions, raw token IDs, or private logs. Ordinary
public setup/start/restart and the planned BetterBench speed/latency evaluation
remain pending. Use
`./r9v support PROFILE --state-dir DIR` for private diagnostics. Preserve the
fail-closed checks and pinned recursive submodules.

## Uncensored profile

`qwen38-mtp4-uncensored` (experimental) serves an abliterated model: its
refusal behavior was removed and it will comply with harmful requests. Confirm
the user wants that model before installing it, and tell them to add their own
moderation before exposing it to anyone. Leave it on 127.0.0.1: do not set
`R9V_HOST_BIND` for this profile unless authentication and moderation stand in
front of the port. It needs its own model directory
(about 92.4 GiB) and state directory. Setup downloads and verifies the package
itself; the first start compiles the model, so give it a long timeout:

```bash
./r9v setup qwen38-mtp4-uncensored --model-dir "$MODEL_DIR" \
  --state-dir "$STATE_DIR" --accept-model-license
./r9v start qwen38-mtp4-uncensored --state-dir "$STATE_DIR" --timeout 2400
```

It runs the consolidated 1.3.0 runtime with CED (approximate long-prompt
prefill) on by default: about 1.70x faster prefill on prompts of 12K+ tokens,
about x1.051 perplexity on prompts that depend on long context, and about 10%
fewer MTP tokens per step on the first answer after a CED prefill. Decode is
exact. `--ced off` in setup or start turns CED off; one request can stay exact
with `"vllm_xargs": {"r9v_ced": false}`. The profile uses a fixed expert
placement: do not pass `--headroom`, `--calibration` or `--expert-catalog`;
they are refused. Its public setup/start qualification is still pending; see
[docs/qualification/uncensored-v040.md](docs/qualification/uncensored-v040.md).

Full command details: [docs/installation.md](docs/installation.md).
