# Installation and launch

This page covers the dual-R9700 Qwen release candidate. The runtime is locally
qualified, but the public package revision and clean-checkout release benchmark
are pending. A new user can validate and build the source today, but cannot
complete `r9v fetch` until that immutable model revision is published.

Discover every packaged setup from the repository root:

```bash
./r9v list
./r9v show qwen38
./r9v doctor qwen38
```

For topology-first browsing:

```bash
./r9v list --by-topology
```

## 1. Clone the pinned source graph

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
```

The submodule commits are the release inputs. Do not replace them with branch
heads if you want the reported reference behavior.

The build requires Docker with the official Buildx CLI plugin. Confirm it
before starting the source build:

```bash
docker buildx version
```

## 2. Build the image

The first stage rebuilds PyTorch 2.11, Triton 3.6, AITER, and the Apache-2.0
vLLM fork against the immutable official ROCm 7.14 base used for
qualification. The ROCm base is digest-pinned because the mutable
`rocm/vllm-dev:base` moved to a stack that cannot establish HIP IPC on the
reference host. The second stage installs the pinned GGUF plugin and compiles
the three R9V kernel extensions for `gfx1201`.

```bash
R9V_MAX_JOBS=8 ./scripts/build-image.sh
```

This is an intentionally expensive source build. Budget tens of GiB of Docker
build storage and substantial CPU time on the first run; BuildKit reuses the
component layers afterward. It does not depend on an unpublished Radiance
image or any R4D component.

## 3. Fetch or arrange the model bundle

After the public model revision is frozen, use:

```bash
./r9v fetch qwen38 \
  --model-dir /path/to/MODEL_DIR \
  --accept-model-license
```

Until then, that command fails closed rather than downloading an incomplete or
moving package. Existing local package owners can arrange the exact files
below and validate them with `./r9v verify qwen38 --model-dir ... -- --hash`.

```text
MODEL_DIR/
  LICENSE
  target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
  target/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf
  target/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf
  metadata/config.json ...
  mtp/config.json
  mtp/model.safetensors
  mtp/mtp-fp8-block-manifest.json
  vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
  manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json
```

Model artifacts are licensed separately under Qwen Community License 1.0,
which must remain in the bundle as `LICENSE`. The exact source revisions and
hashes are in the package descriptor and `release/sources.lock.json`.

Derive the redundant 26.82 GiB PLE payload onto the fast SSD. Run the
extractor inside the built image so the host does not need a separate GGUF
Python environment:

```bash
export MODEL_DIR=/path/to/MODEL_DIR
export R9V_DATA_DIR=/fast-ssd/r9v
mkdir -p "$R9V_DATA_DIR"

docker run --rm --network none --entrypoint python3 \
  --user "$(id -u):$(id -g)" \
  --security-opt label=disable \
  --volume "$PWD:/r9v:ro" \
  --volume "$MODEL_DIR:/models:ro" \
  --volume "$R9V_DATA_DIR:/r9v-data" \
  r9v-qwen38-flash-next:latest \
  /r9v/tools/prepare_ple.py \
  /models/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  /models/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf \
  /models/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf \
  --output /r9v-data/per_layer_token_embd.iq4_nl.bin
```

## 4. Launch

Confirm that ROCm device indices `0,1` map to the intended display/headless
cards. The reference manifest puts the larger dynamic cache on TP rank 1.

```bash
export R9V_PLE_PATH="$R9V_DATA_DIR/per_layer_token_embd.iq4_nl.bin"
export R9V_CACHE_DIR="$R9V_DATA_DIR/cache"
./r9v run qwen38 --model-dir "$MODEL_DIR"
```

`./scripts/launch.sh` remains the compatibility entry point. Environment
values supplied by the caller override profile defaults.

The server exposes OpenAI-compatible text, tool-call, and image inputs at
`http://127.0.0.1:8004/v1`. See `docs/pi.md` for the Pi coding-agent setup.

Confirm readiness without GPU telemetry polling:

```bash
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8004/v1/models
```

## Reference-only assumptions

- Two 32 GiB Radeon R9700 (`gfx1201`) GPUs in TP2.
- Qwen3.8 in this branch is currently dual-R9700 only.
- Rank 0 is the display GPU; rank 1 is the more tightly packed card.
- 128 GiB host RAM.
- 128K BF16 QSA KV configuration and one concurrent sequence.
- Expert placement is prompt-profile dependent. Other workloads should collect
  a route corpus and regenerate the manifest rather than assuming this ranking
  is optimal.

## Release acceptance still pending

Before this page becomes a claim of one-command public installation, R9V must:

1. publish the package at the immutable descriptor revision;
2. perform `fetch → verify → build → run` from a recursive clone on a clean
   host after that package is public.

The pinned source graph, local source image build, OpenAI text/vision smoke,
and public-runtime PP/TG benchmark have passed on the reference machine.
