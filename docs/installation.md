# Installation and launch

## 1. Clone the pinned source graph

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V
```

The submodule commits are the release inputs. Do not replace them with branch
heads if you want the reported reference behavior.

## 2. Build the image

The first stage uses vLLM's Apache-2.0 ROCm Dockerfile on the pinned R9V fork.
The second stage installs the pinned GGUF plugin and compiles the three R9V
kernel extensions for `gfx1201`.

```bash
R9V_MAX_JOBS=8 ./scripts/build-image.sh
```

This is a source build and is intentionally expensive. It avoids depending on
an unpublished local Radiance image or any R4D component.

## 3. Arrange the model bundle

```text
MODEL_DIR/
  target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
  target/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf
  target/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf
  metadata/config.json ...
  mtp/config.json
  mtp/model.safetensors
  vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
  manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json
```

Model artifacts are licensed separately under Qwen Community License 1.0.
The exact source revisions and hashes are in `release/sources.lock.json`.

Derive the redundant 26.82 GiB PLE payload onto the fast SSD:

```bash
python tools/prepare_ple.py "$MODEL_DIR"/target/*.gguf \
  --output /fast-ssd/r9v/per_layer_token_embd.iq4_nl.bin
```

## 4. Launch

Confirm that ROCm device indices `0,1` map to the intended display/headless
cards. The reference manifest puts the larger dynamic cache on TP rank 1.

```bash
export R9V_MODEL_DIR=/path/to/MODEL_DIR
export R9V_PLE_PATH=/fast-ssd/r9v/per_layer_token_embd.iq4_nl.bin
export R9V_CACHE_DIR=/fast-ssd/r9v/cache
./scripts/launch.sh
```

The server exposes OpenAI-compatible text, tool-call, and image inputs at
`http://127.0.0.1:8004/v1`. See `docs/pi.md` for the Pi coding-agent setup.

## Reference-only assumptions

- Two 32 GiB Radeon R9700 (`gfx1201`) GPUs in TP2.
- Rank 0 is the display GPU; rank 1 is the more tightly packed card.
- 128 GiB host RAM.
- 128K BF16 QSA KV configuration and one concurrent sequence.
- Expert placement is prompt-profile dependent. Other workloads should collect
  a route corpus and regenerate the manifest rather than assuming this ranking
  is optimal.
