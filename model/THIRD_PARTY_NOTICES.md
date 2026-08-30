# Third-party notices and model provenance

This file records the attribution and source lineage of the Qwen3.8 Flash Next
R9V IQ4_XS model package. Exact artifact hashes are in `package.json` and
`sources.lock.json`.

## Qwen3.8 Flash Next

Qwen is the model author and upstream rights holder. The target, MTP,
projector, tokenizer and processor metadata, and derived PLE representation
remain under Qwen Community License 1.0, included in this package as
`LICENSE`. R9V's Apache-2.0 source license does not apply to these artifacts.

- Project: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- BF16 revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- FP8 revision: `970c569adaca6b35532111fd6b27351b2baefe50`

## Unsloth IQ4_XS target

The three target shards are Unsloth's `UD-IQ4_XS` quantization of Qwen3.8
Flash Next.

- Project: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
- Revision: `8bdc666649440e9bdc97e16f3f75782c98478ff5`

Unsloth is credited for producing and publishing the target quantization.

## ggml-org vision projector

The Q8_0 vision projector is distributed by ggml-org and was produced with
llama.cpp conversion tooling. R9V did not independently quantize it.

- Project: https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF
- Revision: `01534bc2e1877d5de995b73d247d4459d273e688`

## R9V MTP and package assembly

R9V assembled the minimal MTP checkpoint from the official Qwen BF16 and FP8
checkpoints. R9V did not train these weights. The PLE payload is already
embedded in target shard 2 and is derived locally rather than uploaded a
second time.
