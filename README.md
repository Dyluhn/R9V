# R9V

R9V is a shape-specialized ROCm inference stack for running Qwen3.8 Flash Next
on RDNA4 GPUs from GGUF weights. The first profile targets two 32 GiB Radeon
R9700 cards and combines vision, MTP speculative decoding, 128K context,
SSD-backed PLE, and tiered VRAM/RAM experts behind an OpenAI-compatible API.

This is a clean release tree. It does not redistribute the unlicensed Radiance
launcher repository and does not include the experimental R4D kernels that
caused GPU hangs during development.

## Reference result

| Prompt context | PP | Sustained TG |
|---:|---:|---:|
| 8,202 + 256 | 90.77 tok/s | 78.38 tok/s |
| 32,778 + 256 | 105.44 tok/s | 79.27 tok/s |
| 65,546 + 256 | 133.95 tok/s | 78.57 tok/s |
| 120,010 + 234 | 153.24 tok/s | 77.82 tok/s |

The 120K run completed at 234 generated tokens. These are single-request
measurements on Dylan's dual-R9700 reference system; they are not general GPU
claims. Route distribution, PCIe topology, clocks, memory configuration, and
prompt acceptance all matter.

## What is published

- [R9V gfx1201 kernels](https://github.com/Dyluhn/r9v-gfx1201-kernels):
  Apache-2.0 production kernels for dense M=3 reuse, HyperConnection, GDN, and
  mixed VRAM/UVA MoE with LRU caching.
- [R9V vLLM branch](https://github.com/Dyluhn/vllm/tree/r9v/qwen38-flash-next):
  Qwen3.8 model, MTP/vision, PLE, ROCm, and profiling integration.
- [R9V GGUF-plugin branch](https://github.com/Dyluhn/vllm-gguf-plugin/tree/r9v/qwen38-flash-next):
  Qwen4Exp GGUF loading, tiered experts, RDNA4 quant paths, and multimodal
  adapter support.
- This repository: pinned source graph, image/launch tooling, model provenance,
  PLE extraction, and Pi integration.

## Quick start

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
R9V_MAX_JOBS=8 ./scripts/build-image.sh

export R9V_MODEL_DIR=/path/to/r9v-qwen38-model
export R9V_PLE_PATH=/fast-ssd/r9v/per_layer_token_embd.iq4_nl.bin
export R9V_CACHE_DIR=/fast-ssd/r9v/cache
./scripts/launch.sh
```

The first image build compiles the pinned vLLM ROCm stack and is expensive.
See [installation and launch](docs/installation.md) for the exact model layout,
PLE preparation, device-order requirement, and safety notes.

## Pi coding agent

The API supports text, tool calls, and images. Pi configuration and the
per-model-call PP/TG extension are documented in [docs/pi.md](docs/pi.md).

## Model artifacts

The ready-to-use model bundle is distributed separately because its three GGUF
shards are roughly 90 GiB. It combines:

- Unsloth's `UD-IQ4_XS` target quantization;
- ggml-org's Q8_0 vision projector;
- an R9V-assembled minimal MTP checkpoint using official Qwen BF16 nonexpert
  tensors and official Qwen block-FP8 routed experts;
- official tokenizer/processor metadata; and
- the reference expert-placement manifest.

The 26.82 GiB PLE table is not uploaded a second time. `tools/prepare_ple.py`
extracts its packed IQ4_NL bytes directly from the target GGUF to the chosen
SSD, with metadata, size, hash, and sample validation.

## Licensing

R9V-owned code in this repository and the kernel repository is Apache-2.0.
The vLLM and GGUF-plugin forks retain Apache-2.0. llama.cpp/ggml-derived code
retains its MIT notice. Model artifacts are not Apache-2.0; they remain under
Qwen Community License 1.0 with Unsloth and ggml-org attribution. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[release/sources.lock.json](release/sources.lock.json).
