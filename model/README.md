---
license: other
license_name: qwen-community-license-1.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - rocm
  - vision
  - speculative-decoding
  - qwen3.8
---

# Qwen3.8 Flash Next R9V IQ4_XS

Ready-to-arrange model bundle for the
[R9V](https://github.com/Dyluhn/R9V) dual-RDNA4 inference profile.

## Contents and provenance

- `target/`: Unsloth `UD-IQ4_XS` target GGUF shards from
  `unsloth/Qwen3.8-Flash-Next-GGUF`.
- `vision/`: ggml-org Q8_0 vision projector. This projector was not quantized
  by R9V.
- `mtp/`: R9V-assembled minimal MTP checkpoint. Dense/nonexpert tensors come
  from the official BF16 checkpoint; routed experts come from the official
  block-FP8 checkpoint. R9V did not train these weights.
- `metadata/`: official Qwen tokenizer, processor, and model configuration.
- `manifests/`: the reference dual-R9700 hot-expert placement.
- `sources.lock.json`: exact upstream revisions, sizes, and hashes.

Unsloth and ggml-org are credited for the target quantization and vision
projector. Qwen remains the model author and upstream rights holder.

## PLE table

The 26.82 GiB `per_layer_token_embd.weight` payload is already present inside
target shard 2 and is intentionally not uploaded again. Extract it to the fast
SSD with R9V's metadata-driven tool:

```bash
python tools/prepare_ple.py target/*.gguf \
  --output /fast-ssd/r9v/per_layer_token_embd.iq4_nl.bin
```

## Reference configuration

- Two Radeon R9700 32 GiB GPUs, TP2.
- 128 GiB DDR5.
- MTP depth 2.
- 131,072-token capacity, BF16 QSA KV.
- Q8 vision input, one image/request.
- SSD PLE and tiered expert placement.

Measured single-request TG was 78.38 tok/s at 8K context, 79.27 at 32K,
78.57 at 64K, and 77.82 at a 120,010-token prompt on the reference machine.
See the R9V repository for the exact code revisions and qualification notes.

## License

These model artifacts are distributed under Qwen Community License 1.0; see
`LICENSE`. The R9V Apache-2.0 code license does not apply to model weights.
Users are responsible for reviewing the Qwen license, including its separate
terms for certain commercial MaaS/AI-work-assistant uses and scale thresholds.
