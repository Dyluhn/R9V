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
- host RAM: the profile is qualified on a 128 GiB reference host. Smaller
  hosts are untested, not rejected. Cold expert allocations are pinned and do
  not spill to SSD; less RAM instead reduces startup headroom and the
  filesystem cache available to the SSD-backed PLE table. On a smaller host,
  keep `R9V_PLE_RESIDENCY_MODE=ssd`;
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

Confirm that ROCm device indices map to the intended TP ranks before launch.
Device order is semantic: the reference manifest gives rank 1 more static
experts and a 16-slot LRU cache. Set the visible order, then lock the resolved
PCI addresses so a driver or hardware change cannot silently reverse it.

For persistent machine settings, copy the provided template and point both the
doctor and launcher at it. Entries use conditional defaults, so a value
exported directly in the shell still takes precedence.

The complete meaning, discovery command, safe range, and correction procedure
for every setting is in the
[dual-R9700 configuration reference](../profiles/qwen38-flash-next/dual-r9700/README.md).

```bash
cp profiles/qwen38-flash-next/dual-r9700/user-config.example.env \
  /path/to/my-r9v-qwen.env
export R9V_CONFIG_FILE=/path/to/my-r9v-qwen.env
```

```bash
amd-smi list

export R9V_CACHE_DIR="$R9V_DATA_DIR/cache"
export R9V_VISIBLE_DEVICES=0,1
export R9V_EXPECTED_GPU_BDFS=0000:03:00.0,0000:13:00.0  # use your amd-smi BDFs
export R9V_EXPECTED_PCIE_LINKS=Gen5x16,Gen4x4          # use path bottlenecks
./r9v doctor qwen38 --model-dir "$MODEL_DIR"
./r9v run qwen38 --model-dir "$MODEL_DIR"
```

`./scripts/launch.sh` remains the compatibility entry point. Environment
values supplied by the caller override profile defaults. Launch runs the same
read-only preflight automatically and fails on broken requirements. For
debugging only, `R9V_PREFLIGHT=0` bypasses that gate and prints a warning.

The host contract is configurable rather than tied to one motherboard:

| Setting | Meaning | Qualified default |
|---|---|---|
| `R9V_VISIBLE_DEVICES` | HIP devices in TP-rank order | `0,1` |
| `R9V_EXPECTED_GPU_BDFS` | Optional PCI-address lock in rank order | unset; doctor warns |
| `R9V_EXPECTED_PCIE_LINKS` | Optional exact device-to-root capacity bottlenecks in rank order | unset; doctor warns |
| `R9V_MIN_PCIE_BANDWIDTH_GBPS` | Minimum theoretical payload per rank | `15,7` |
| `R9V_MIN_HOST_RAM_BYTES` | Hard total-RAM floor; `0` only reports | `0` |
| `R9V_MIN_HOST_AVAILABLE_BYTES` | Hard pre-launch available-RAM floor | `0` |
| `R9V_TIERED_EXPERT_CACHE_RANKS` | Ranks receiving the dynamic expert cache | `1` |
| `R9V_TIERED_EXPERT_CACHE_SLOTS` | LRU slots per selected rank | `16` |
| `R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK` | Static-manifest plus cache VRAM ceiling | `329,385` |
| `R9V_PLE_RESIDENCY_MODE` | `ssd`, `bounded`, or fully `pinned` | `ssd` |
| `R9V_REQUIRE_PLE_NONROTATIONAL` | Reject a PLE file backed by rotating media | `1` |

The exact PCIe setting accepts forms such as `Gen5x16,Gen4x4` or
`32x16,16x4`. It verifies configured path capacity (maximum speed with
negotiated width); it cannot change
the link. The independent bandwidth check walks every hop from the device to
the root port and scores the slowest-capacity link, so equivalent or faster
paths satisfy the performance floor even when their generation/width differs
from the reference host, and a card that negotiates x16 behind a narrower
upstream bridge is scored at the bridge's capacity.
The `112.5` CPU-offload values are logical loader-accounting budgets, not
112.5 GiB RAM allocations; do not lower them merely to match installed RAM.

After the server is ready, run the runtime half of the doctor:

```bash
./r9v doctor qwen38 --model-dir "$MODEL_DIR" --runtime
```

It verifies the container environment, materialized tiered experts on both TP
ranks, the `reuse3v2` decode path, grouped prefill evidence, MTP counters, and
PLE timing availability. For a low-TG diagnosis, relaunch once with
`R9V_PLE_WORKER_TIMING=1`; the runtime report then prints the latest PLE
latency split. This adds timing logs but does not move the n-gram table to RAM.
`--json` emits the same checks as a support-bundle-friendly document.

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
- Rank roles are configurable; the published placement gives rank 1 the
  larger static placement and dynamic cache.
- 128 GiB host RAM on the reference machine (not an enforced minimum).
- The reference run used asymmetric links. Preflight checks configured payload
  floors rather than requiring those exact link labels.
- 128K BF16 QSA KV configuration and one concurrent sequence.
- The published expert placement was ranked from a multiprompt route corpus.
  Any workload runs correctly on it; a heavily divergent prompt mix may see
  more cold-expert traffic and lower decode throughput than the published
  figure. A route-corpus/manifest regeneration tool is planned but not yet
  shipped.

## Final release acceptance gate

Before the profile moves from `release-candidate` to `qualified`, R9V must
perform `fetch → verify → build → run` from a recursive clone on a clean host.

The package publication and remote hashes, pinned source graph, local source
image build, OpenAI text/vision smoke, and public-runtime PP/TG benchmark have
passed.
