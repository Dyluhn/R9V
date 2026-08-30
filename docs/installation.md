# Installation and launch

This page covers the dual-R9700 Qwen release candidate. The immutable model
package is public and remotely hash-verified. The runtime is locally qualified;
a clean-host package installation is the remaining release gate.

## 1. Clone the pinned source graph

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
```

The submodule commits are release inputs. Do not replace them with branch heads
if you want the reported reference behavior.

The host prerequisites are:

- Linux with two 32 GiB Radeon AI PRO R9700 (`gfx1201`) GPUs, a working ROCm
  host driver and `amd-smi` inventory CLI, and access to `/dev/kfd` and
  `/dev/dri`;
- at least 128 GiB host RAM;
- Git, Python 3.10 or newer, `curl`, Docker with daemon access, and the official
  Docker Buildx CLI plugin;
- host `render` and `video` group records for the device GIDs passed into the
  container;
- the Hugging Face `hf` CLI for the public package download; and
- storage for 90.36 GiB of model files, a 26.82 GiB derived PLE file, the image
  build, and runtime caches.

Confirm the source graph and host-facing prerequisites before the expensive
build:

```bash
./r9v list --by-topology
./r9v show qwen38
./r9v validate qwen38
docker info >/dev/null
docker buildx version
./r9v doctor qwen38
```

## 2. Fetch or arrange the model bundle

Fetch the exact verified package revision with:

```bash
export MODEL_DIR=/path/to/qwen38-r9v

./r9v fetch qwen38 \
  --model-dir "$MODEL_DIR" \
  --accept-model-license
./r9v verify qwen38 --model-dir "$MODEL_DIR" -- --hash
```

The package descriptor pins Hugging Face revision
`bf836f0c20b6c92fcad4226ad3115eb8a19f7582`; `fetch` does not follow a moving
branch. Existing package owners can instead arrange the exact files below, set
`MODEL_DIR` to that package root, and run the same hash-verification command.
The descriptor is authoritative; the abbreviated tree is only a navigation aid.

```text
MODEL_DIR/
  LICENSE
  THIRD_PARTY_NOTICES.md
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

Model artifacts are licensed separately under Qwen Community License 1.0.
`LICENSE` and `THIRD_PARTY_NOTICES.md` must remain in the bundle. The exact
source revisions and hashes are in the package descriptor and
`release/sources.lock.json`.

## 3. Build the image

The first stage rebuilds PyTorch 2.11, Triton 3.6, AITER, and the Apache-2.0
vLLM fork against the immutable official ROCm 7.14 base used for
qualification. The ROCm base is digest-pinned because the mutable
`rocm/vllm-dev:base` moved to a stack that cannot establish HIP IPC on the
reference host. The second stage installs the pinned GGUF plugin and compiles
the three R9V kernel extensions for `gfx1201`.

```bash
R9V_MAX_JOBS=8 ./r9v build qwen38
```

This is an intentionally expensive source build. Budget tens of GiB of Docker
build storage and substantial CPU time on the first run; BuildKit reuses the
component layers afterward. It does not depend on an unpublished Radiance
image or any R4D component. The source build can also be validated independently
of the model download.

## 4. Derive the PLE payload

Derive the redundant 26.82 GiB PLE payload onto the fast SSD. Run the
extractor inside the built image so the host does not need a separate GGUF
Python environment:

```bash
export R9V_DATA_DIR=/fast-ssd/r9v
export R9V_PLE_PATH="$R9V_DATA_DIR/per_layer_token_embd.iq4_nl.bin"
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

test "$(stat -c %s "$R9V_PLE_PATH")" -eq 28800138240
```

## 5. Launch

Confirm that ROCm device indices `0,1` map to the intended display/headless
cards with the host's ROCm inventory tool before launch. Device order is
semantic: the reference manifest puts the larger dynamic cache on TP rank 1.
If the required order is not `0,1`, set `R9V_VISIBLE_DEVICES` explicitly.

```bash
amd-smi list

export R9V_CACHE_DIR="$R9V_DATA_DIR/cache"
export R9V_VISIBLE_DEVICES=0,1
./r9v doctor qwen38 --model-dir "$MODEL_DIR"
./r9v run qwen38 --model-dir "$MODEL_DIR"
```

`./scripts/launch.sh` remains the compatibility entry point. Environment
values supplied by the caller override profile defaults.

By default, the server exposes OpenAI-compatible text, tool-call, and image
inputs at `http://127.0.0.1:8004/v1`. See `docs/pi.md` for the Pi coding-agent
setup.

Startup can take several minutes. Poll readiness with a bounded wait rather
than assuming the detached container is immediately ready:

```bash
(
host_port="${R9V_HOST_PORT:-8004}"
container="${R9V_CONTAINER_NAME:-r9v-qwen38-flash-next}"
ready=0
for attempt in {1..180}; do
  if curl -fsS "http://127.0.0.1:${host_port}/health"; then
    ready=1
    break
  fi
  sleep 5
done
if (( ! ready )); then
  docker logs --tail 200 "$container"
  exit 1
fi
curl -fsS "http://127.0.0.1:${host_port}/health"
curl -fsS "http://127.0.0.1:${host_port}/v1/models"
)
```

The launcher refuses to replace an existing container. Stop and remove the
known profile container before a deliberate relaunch:

```bash
container="${R9V_CONTAINER_NAME:-r9v-qwen38-flash-next}"
docker stop "$container"
docker rm "$container"
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

## Final release acceptance gate

Before the profile moves from `release-candidate` to `qualified`, R9V must
perform `fetch → verify → build → run` from a recursive clone on a clean host.

The package publication and remote hashes, pinned source graph, local source
image build, OpenAI text/vision smoke, and public-runtime PP/TG benchmark have
passed.
