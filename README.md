# R9V

[![CI](https://github.com/Dyluhn/R9V/actions/workflows/ci.yml/badge.svg)](https://github.com/Dyluhn/R9V/actions/workflows/ci.yml)

R9V runs Qwen3.8 Flash Next on two AMD Radeon AI PRO R9700 GPUs. It combines a pinned vLLM fork, GGUF loading, specialized `gfx1201` kernels, expert offloading and four-token MTP speculative decoding. The server exposes an OpenAI-compatible API for text, tool calls and single-image requests.

Each profile binds a model package, runtime, hardware layout and expert placement. Downloads are checked against pinned revisions and file hashes. Setup records the selected configuration, and first start qualifies its workload and memory headroom before reporting ready.

**Current status:** both IQ4_XS and Q4_K_XL MTP4 profiles passed ordinary public setup, first-start workload qualification and unchanged-receipt restart on the dual-R9700 reference host. Both profiles remain experimental. Each first start passed all seven checks at 131,072 context, including a 130,941-token prompt, with at least 3 GiB free VRAM per GPU. Setup selects the profile's runtime image bundle, verifies every part and loads the exact image ID. The Q4 profile uses the [v0.2.0-rc2 bundle](https://github.com/Dyluhn/R9V/releases/tag/v0.2.0-rc2-images); the IQ4 profile's WMMA-prefill image (`release/image-bundle-wmma-prefill-20260915.json`) is published under the [`v0.3.0-rc1-images`](https://github.com/Dyluhn/R9V/releases/tag/v0.3.0-rc1-images) release tag. See [release status and evidence](docs/qwen-release-candidate.md).

**New in v0.4.3:** `qwen38-mtp4-uncensored --ced quality` now passes first-start qualification on the reference host, with image support kept: the vision encoder's weights and the quality projector take turns in one VRAM region per GPU, and the projector's work buffers are smaller. GPU 0 kept 2.04 GiB free (v0.4.2: 0.90, target 1.5), more than `on` (1.65). See the [changelog](CHANGELOG.md).

**New in v0.4.2:** `qwen38-mtp4-uncensored` has an opt-in CED quality mode, `--ced quality`, that gives up less long-context quality for less prefill speedup (it did not fit on the reference host until v0.4.3). Right after a first start, `./r9v doctor --runtime` no longer fails its KV-pressure check because of qualification's own long prompt.

**New in v0.4.1:** `qwen38-mtp4-uncensored` keeps rank 1's 400 most-used experts per layer in VRAM for good, so its host copy of the experts drops from 55.4 to 40.3 GiB and it needs 56.3 GiB of free RAM at start instead of 71.4. In the GPU test its outputs matched 1.3.0 bit for bit on the test prompts and decode speed was unchanged. `./r9v doctor` also checks disk space, VRAM held by other programs, API exposure, the runtime's pinned files and the CED projector. See the [changelog](CHANGELOG.md).

**New in v0.4.0:** `qwen38-mtp4-uncensored` runs an uncensored (abliterated) IQ4_XS model on the consolidated 1.3.0 runtime with CED long-prompt prefill on by default. Its public fetch, setup, first-start qualification and restart passed from a clean checkout on the reference host, with CED on and off. See [its qualification note](docs/qualification/uncensored-v040.md).

## Profiles and features

| Alias | Model package | Runtime | Status |
|---|---|---|---|
| `qwen38-mtp4` | [IQ4_XS model bundle](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS) | MTP4, dual R9700, 128K context | Reference setup/start/restart passed |
| `qwen38-q4-xl` | [Q4_K_XL weights](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb/UD-Q4_K_XL) | MTP4, dual R9700, 128K context | Reference setup/start/restart passed |
| `qwen38-mtp4-uncensored` | [Uncensored IQ4_XS bundle](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-Uncensored-R9V-IQ4_XS) (abliterated, **no refusals**) | Consolidated 1.3.0 runtime with host expert dedupe: MTP4, dual R9700, 128K context, CED on; needs 56.3 GiB free RAM at start (60.8 GiB with `--ced quality`) | v0.4.1 clean install (fetch, setup, first start CED on/off, restart) passed on the reference host; v0.4.3 clean install with `--ced quality` passed there too (below) |

**About `qwen38-mtp4-uncensored`.** Its model had its refusal behavior removed and will comply with harmful requests the original model refuses. Use it for research, and add your own moderation before exposing it to anyone. R9V serves it on 127.0.0.1 only; setting `R9V_HOST_BIND` to expose it gives anyone who can reach the port an unauthenticated model with no refusals.

It has CED (approximate long-prompt prefill) on by default. On prompts of 8,192 tokens or more, layers 0–15 run exactly and a split-16 projector stands in for the later layers on all but the last ~2K prompt tokens. Decode stays exact. The tradeoff, measured on the reference host:

- About **1.5× faster prefill** at ~13K tokens, rising to about **1.8×** at 32K tokens and above.
- About **×1.051 perplexity** on prompts that depend on their long context.
- About **10% fewer MTP tokens per step** on the first answer after a CED prefill; later turns are normal.
- The projector takes **1.76 GiB of VRAM per GPU**, so the profile's free-VRAM target is 1.5 GiB per card instead of 3 GiB.

Turn CED off for the server with `--ced off` in setup or start; keep a single request exact with `"vllm_xargs": {"r9v_ced": false}`.

**CED quality (opt-in, `--ced quality`).** A multi-source projector reads the layer-16 state plus the inputs of full-attention layers 3, 7, 11 and 15. In one GPU grade of both projectors (both loaded as int8; the default ships bf16), on the 16 prompts that depend on their long context:

| `--ced` | Perplexity | Long-context gain lost | Prefill speedup (median) | Projector VRAM per GPU |
|---|---|---|---|---|
| `on` (default) | ×1.049 | 17% | 1.68× | 1.76 GiB (bf16) |
| `quality` | ×1.029 | 10% | 1.55× | 1.79 GiB (stored int8), shared with the vision encoder's 0.42 GiB |

The projector math costs 56 ms per 1K approximated tokens instead of 24 ms. Setup downloads the quality projector (1.8 GB) only when you pick it: run `setup --ced quality` once, then `start` keeps the choice. `on` stays the default and is unchanged from v0.4.1.

**Tested on the compiled server (v0.4.3).** The quality projector shares one VRAM region per GPU with the vision encoder's weights: image prompts never use CED, so the region holds whichever the next step needs and is refilled from pinned host RAM (36 ms / 9 ms on GPU 0, 266 ms / 63 ms on GPU 1; only when a CED prompt follows an image or the other way round). In the clean-install GPU test, first start with `--ced quality` passed all seven checks with 2.04 / 2.49 GiB free (target 1.5; `on`: 1.65 / 2.12) while desktop apps held 1.23 GiB of GPU 0. Prefill was 1.47× at 12.8K tokens and 1.62× at 32K (`on`: 1.58× and 1.76×), decode unchanged, exact requests bitwise identical after a CED request, no recompiles; 24 alternating CED and image requests kept VRAM flat and read every image. Quality needs 4.50 GiB more free RAM at start (60.8 GiB) for the pinned copies. v0.4.2 did not fit: GPU 0 fell to 0.90 GiB free. See the [changelog](CHANGELOG.md). The profile uses a fixed expert placement, so it does not accept `--headroom`.

Use the explicit MTP4 aliases for the current workflow.

- **Per-card headroom:** choose the free VRAM to retain on each card with `--headroom 3,3` or an asymmetric budget such as `--headroom 5,3`.
- **Measured expert maps:** separate complete catalogs for IQ4 and Q4 rank all 512 experts across 48 layers per rank, using training and separate held-out routing captures. Existing hot prefixes are preserved, while cold-to-hot arrays use measured counts, with deterministic ties for zero-count experts. The planner retains frequently used experts in VRAM within the memory budget.
- **Q4_K_XL support:** a dedicated package, packed expert costs, ranked placement and streaming loader. Q4 uses its upstream model bytes and original Q8_0 target head; IQ4 retains its original Q6_K target head.
- **Resumable setup and restart receipts:** reuse matching model assets, save configuration and reuse its verified receipt on an unchanged restart after the workflow has qualified.
- **Doctor:** inspect model/runtime/placement compatibility, GPU ordering, memory pressure, PCIe information, cache identity and available worker evidence.
- **Support bundles:** collect configuration summaries, source identities, worker records, memory and GPU diagnostics, logs and capture tails into a bounded local archive with file hashes.

Unobserved experts are explicit ties in the maps. Routing frequency depends on workload; a different prompt mix can change the best placement. Requested headroom is checked against the qualification workload, and cannot prevent another application from allocating VRAM later.

## Model downloads

- **IQ4_XS:** [R9V model page](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS) · [files at the revision used by setup](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS/tree/bf836f0c20b6c92fcad4226ad3115eb8a19f7582). This bundle includes all three target GGUF shards, the MTP checkpoint, vision projector, tokenizer and configuration files.
- **Uncensored IQ4_XS:** [R9V model page](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-Uncensored-R9V-IQ4_XS) · [files at the revision used by setup](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-Uncensored-R9V-IQ4_XS/tree/8112610745a8ddc3a19cc659314af245820ee728). The same layout as the IQ4_XS bundle, with orcarouter's abliterated target and F16 vision projector, plus the CED projector.
- **Q4_K_XL:** [all four target GGUF shards at the revision used by setup](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb/UD-Q4_K_XL). The Q4 profile also uses the shared [MTP checkpoint and configuration](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS/tree/bf836f0c20b6c92fcad4226ad3115eb8a19f7582/mtp), [Q8_0 vision projector](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS/tree/bf836f0c20b6c92fcad4226ad3115eb8a19f7582/vision), and [tokenizer and model metadata](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS/tree/bf836f0c20b6c92fcad4226ad3115eb8a19f7582/metadata) from the IQ4 bundle.

The `./r9v setup` commands below download and verify the required files automatically. For manual downloads, keep every shard and the package directory layout; use the [IQ4 package manifest](packages/models/qwen38-flash-next/ud-iq4-xs--mtp-blockfp8--mmproj-q8/package.json) or [Q4 package manifest](packages/models/qwen38-flash-next/ud-q4-k-xl--mtp-blockfp8--mmproj-q8/package.json) for exact paths, revisions and hashes. The PLE table is extracted locally from the target GGUF; it is not a separate model download.

## Hardware and storage

The reference system uses:

- Two **32 GiB Radeon AI PRO R9700** cards (`gfx1201`).
- Linux with working AMD GPU drivers, `amd-smi`, `/dev/kfd` and `/dev/dri` access.
- **128 GiB host RAM**. Smaller hosts are untested; cold expert allocations use host memory. `qwen38-mtp4-uncensored` checks for **56.3 GiB available** before start (was 71.4 GiB in v0.4.0): its host expert copy is 40.3 GiB plus a 16 GiB PLE reserve.
- An asymmetric PCIe layout: rank 0 on Gen5 x16 and rank 1 across Gen4 x4. GPU ordering matters to placement and performance.
- Git, Python 3.10+, Docker and the Hugging Face CLI described in the [installation guide](docs/installation.md).

The public image bundle plus its containerd image-store footprint measured roughly **50 GiB**; reserve at least **70 GiB** for image and cache import space. The IQ4 package occupies approximately **90.36 GiB** and the uncensored package **92.39 GiB**. The four Q4 target shards alone occupy **103.69 GiB**, with auxiliary assets additional. The derived PLE file occupies **26.82 GiB**. Leave further room for compilation caches and diagnostics. Reuse verified assets instead of duplicating model files.

## Setup and start

Clone the source and its exact dependencies:

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
./r9v show qwen38-mtp4
./r9v doctor qwen38-mtp4 -- --host-only
```

The profile distribution entries select the matching GitHub Release image bundle and exact image IDs. Docker 29 must use the containerd image store so `docker load` preserves those IDs. See Docker's [containerd image store guide](https://docs.docker.com/engine/storage/containerd/) and [daemon configuration reference](https://docs.docker.com/engine/daemon/). Verify the store before setup:

```bash
docker info -f '{{ .DriverStatus }}'
```

The output should identify `io.containerd.snapshotter.v1`. If it does not, follow the configuration steps in the [installation guide](docs/installation.md). For a non-root daemon, see Docker's [rootless mode guide](https://docs.docker.com/engine/security/rootless/) and confirm the selected context/socket with `docker info`; the same containerd image-store check applies. These are the reference identities:

| Profile | Tested local image ID |
|---|---|
| `qwen38-mtp4` | `sha256:2dac17a215fb5b0e3461e4c3e36a2981eec8ac3d6021e73183d247e819740c03` |
| `qwen38-q4-xl` | `sha256:2e50016cfcc9cd22f15d3f69ccf001e4877236e12ebb4ab458cc9c16caaef9e3` |
| `qwen38-mtp4-uncensored` | `sha256:2dac17a215fb5b0e3461e4c3e36a2981eec8ac3d6021e73183d247e819740c03` plus pinned overlays |

Install the download CLI in an isolated environment if it is not already available:

```bash
python3 -m venv ~/.local/share/r9v/download-tools
~/.local/share/r9v/download-tools/bin/pip install huggingface_hub
export PATH="$HOME/.local/share/r9v/download-tools/bin:$PATH"
```

For IQ4, choose persistent SSD directories with enough free space:

```bash
MODEL_DIR=/path/to/qwen-iq4
STATE_DIR=/path/to/r9v-state/iq4
./r9v setup qwen38-mtp4 \
  --model-dir "$MODEL_DIR" --state-dir "$STATE_DIR" \
  --headroom 3,3 --accept-model-license
./r9v start qwen38-mtp4 --state-dir "$STATE_DIR"
```

For Q4, use its own model and state directories:

```bash
./r9v setup qwen38-q4-xl --model-dir /path/to/qwen-q4 \
  --state-dir /path/to/r9v-state/q4 --headroom 3,3 --accept-model-license
./r9v start qwen38-q4-xl --state-dir /path/to/r9v-state/q4
```

For the uncensored profile, use its own directories. Its first start compiles the model, so start waits up to 2,400 seconds for it by default (`--timeout` changes that):

```bash
./r9v setup qwen38-mtp4-uncensored --model-dir /path/to/qwen-uncensored \
  --state-dir /path/to/r9v-state/uncensored --accept-model-license
./r9v start qwen38-mtp4-uncensored --state-dir /path/to/r9v-state/uncensored
```

Run one profile at a time. Before switching, inspect `docker ps`, save any needed support evidence, and stop the selected R9V container by its exact name using `docker stop NAME`. Setup supports `--reuse-from /path/to/existing/assets` for matching assets and `--ple-path /path/to/existing/ple.bin` for an existing verified PLE file. The profiles select their corresponding expert catalog and reference memory seed automatically.

First start runs the local workload qualification, including context and headroom checks. An unchanged restart can reuse its verified receipt. To change the budget, stop the selected container, then run:

```bash
./r9v start qwen38-mtp4 --state-dir "$STATE_DIR" --headroom 5,3
```

The new placement must qualify. Card order follows the saved GPU selection; inspect it with doctor. The planner reports per-card shortfalls when a request cannot fit, while retaining the configured context.

The default API endpoint is `http://127.0.0.1:8004/v1`. The API has no authentication, so R9V publishes it on 127.0.0.1 only, in every profile. To reach it from other machines, run setup with `R9V_HOST_BIND=0.0.0.0` (or one interface's IPv4/IPv6 address); setup saves it with the port. See [API address](profiles/qwen38-flash-next/dual-r9700/README.md#api-address). Use the address recorded by your selected setup if you override the port. See the [release guide](docs/qwen-release-candidate.md), [installation guide](docs/installation.md), and [troubleshooting guide](docs/troubleshooting.md). Rebuilding an image does not reproduce a qualified image identity automatically.

## Measured results

### Prompt processing

Measured with the original release’s corpus, script, warmup and 8K/32K/64K procedure. Rates below are mean prompt tokens/s; every trial and exact reproduction command are linked below.

| Profile | 8K (10 runs) | 32K (3 runs) | 64K (2 runs) | Raw trials |
|---|---:|---:|---:|---|
| IQ4_XS, WMMA prefill (current `qwen38-mtp4`) | **1,684.6** | **1,640.6** | **1,605.1** | [JSON](docs/qualification/results/iq4-wmma-prefill-20260915.json) |
| IQ4_XS, v0.2.0 image | 989.3 | 982.0 | 968.2 | [JSON](docs/qualification/results/iq4-v020-pp-20260914.json) |
| Q4_K_XL | 552.9 | 542.5 | 536.4 | [JSON](docs/qualification/results/q4-v020-pp-20260914.json) |

The current IQ4 profile runs prompt chunks of 4096 tokens on a gfx12 int8 WMMA grouped MoE prefill kernel; decode kernels, placement policy and the model are unchanged, and the placement assets were re-derived and qualified for the new image. [Why v0.2.0 regressed, kernel parity, method and commands](docs/qualification/wmma-prefill-20260914.md); [v0.2.0 method](docs/qualification/v020-prefill-20260914.md). Prefix-cache hits were zero. The older generation references below use different placements and are separate results.

### Earlier fixed-prompt generation references

The following fixed-prompt reference samples used MTP4 on the dual-R9700 system:

| Profile / placement | Static experts, ranks 0/1 | Generation tokens/s |
|---|---:|---:|
| IQ4 image7 streaming reference | 71 / 450 | **89.45196** |
| Q4 measured ranked placement | 97 / 349 | **53.431** |
| Q4 initial bootstrap placement | 64 / 320 | **25.285** |

These are fixed-prompt reference samples with MTP4; they do not measure mixed traffic or generation at full context. Actual user placements depend on the requested headroom and must qualify locally.

The IQ4 image7 streaming reference retained **131,072 context tokens** and passed seven bounded workload checks, including an actual **130,941-token prompt**, text, tools, three image shapes and idle resume. Its median was **89.45196 TG tok/s**, with measured free VRAM of 4,253,020,160 and 4,090,036,224 bytes and a minimum Normal-zone free value of 450,269,184 bytes. The run had a clean 90-second aftermath and GPU reclaim. The supervisor incorrectly reported failure because its cleanup check required exact VRAM equality: rank 0 had 185.203 MiB more free and rank 1 was unchanged. Independent review confirmed no per-card shortfall throughout the aftermath. These reference measurements are separate from the completed ordinary user-flow checks below and do not establish answer quality or a 100 tok/s qualification.

The [current IQ4 reference evidence](docs/qualification/results/iq4-image7-exact-host-reference-20260912.json) records the measured result and independently verified archive commitments. Historical prefill and comparator results remain in the [earlier Qwen qualification](docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md); they should not be substituted for measurements of the new placements.

## Verified public setup and restart

Both profiles passed ordinary setup into new state directories, first-start text/tool/vision/context/idle-resume checks and restart with the same verified receipt. The first starts used an actual 130,941-token prompt at a 131,072 context limit. Model, head and PLE bytes stayed unchanged.

| Profile | Static experts, ranks 0/1 | Cache slots, ranks 0/1 | Minimum free VRAM, ranks 0/1 | Evidence |
|---|---:|---:|---:|---|
| IQ4 image7 | 76 / 451 | 160 / 0 | 3,828,301,824 / 4,120,670,208 bytes | [User-flow report](docs/qualification/results/iq4-public-userflow-20260912.json) |
| Q4 image6 | 99 / 348 | 80 / 0 | 4,072,394,752 / 4,013,797,376 bytes | [User-flow report](docs/qualification/results/q4-public-userflow-20260912.json) |

These checks used a fresh anonymous source checkout and new setup state, reusing previously public-downloaded, hash-verified assets and images. The separate anonymous image import recovered from an interrupted load using unchanged verified parts. This was not a second fresh download or a full new-machine download in one uninterrupted run. Both user flows stopped cleanly, reclaimed GPU allocations and completed at least 90 seconds of post-stop observation. Q4 recorded one low Normal-zone memory sample without reaching the unchanged consecutive-sample stop threshold; this is not evidence of a wide host-memory margin.

## Diagnostics and reporting a problem

Run doctor for the selected profile:

```bash
./r9v doctor qwen38-mtp4 --state-dir "$STATE_DIR"
```

Collect support evidence before removing a failed container, using the same state directory as setup:

```bash
SUPPORT_DIR=/path/to/private/r9v-support
./r9v support qwen38-mtp4 --state-dir "$STATE_DIR" \
  --output "$SUPPORT_DIR/run-001" \
  --archive "$SUPPORT_DIR/run-001.tar.gz"
```

Collection stays local and never uploads automatically. Configuration summaries hide credentials and personal paths, but raw application and kernel logs can contain identifying information or request content. Review the archive before attaching it to a GitHub issue. Include the profile, failed command, approximate failure time and whether the server, container or whole host stopped responding. Do not attach model weights or private prompts.

Doctor distinguishes configured settings from observed execution. Missing live metrics after a container stops are reported as unavailable; a missing kernel marker alone does not prove the wrong kernel ran. See the [configuration reference](profiles/qwen38-flash-next/dual-r9700/README.md) for available controls and corrective actions.

## Source and development

```text
profiles/             model/runtime/hardware compositions and launch settings
packages/models/      pinned model sources, artifact hashes and licenses
packages/placements/  expert maps, memory seeds and placement manifests
runtimes/             runtime descriptors and retained kernel/source overlays
hardware/             GPU, RAM, PCIe and rank contracts
kernels/              pinned R9V kernel submodule
vendor/               pinned vLLM and GGUF-plugin forks
tools/                setup, planning, qualification, doctor and support
tests/                CPU checks and explicit GPU qualification tests
```

The kernels are specialized for supported shapes, quantizations and `gfx1201`. A different model, GPU, topology or runtime image requires its own validation. Current reference qualification covers one active sequence at a time.

Run the CPU and static checks with:

```bash
python -m pip install --requirement requirements-ci.txt
python -m pytest -q tests
./scripts/ci-static.sh
```

CPU CI checks tooling and source contracts. GPU parity, graph replay, full-model qualification and throughput measurements require the matching hardware. Both ordinary public setup/start/restart flows passed on the reference machine; a new machine or changed placement still requires local qualification.

Read [CONVENTIONS.md](CONVENTIONS.md) before changing code. Dependency gitlinks are release inputs: use the committed revisions rather than replacing them with moving branch heads.

## License and provenance

R9V-owned code and kernels use Apache-2.0. The vLLM and GGUF-plugin forks retain their licenses; llama.cpp/ggml-derived quantization primitives retain their MIT notices. Qwen model weights are separately governed by the Qwen Community License 1.0.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [licensing](docs/licensing.md) and the [provenance audit](docs/provenance-audit.md). Runtime source publication, public image distribution and complete installation qualification are tracked separately; this README reports only the gates that have actually passed.
