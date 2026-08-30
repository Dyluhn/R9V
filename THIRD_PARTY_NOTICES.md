# Third-party notices

These notices describe the exact pinned source graph in
`release/sources.lock.json`.

## R9V gfx1201 kernels

The R9V-owned kernel repository is licensed under Apache License 2.0 and
retains the llama.cpp/ggml MIT attribution for the GGUF quant primitives it
consumes from the plugin.

Project: https://github.com/Dyluhn/r9v-gfx1201-kernels

## vLLM

R9V integrates and modifies [vLLM](https://github.com/vllm-project/vllm), licensed under the Apache License, Version 2.0. The distributed vLLM fork retains the upstream license and marks modified files.

## vLLM GGUF plugin

R9V integrates and modifies the [vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin), licensed under the Apache License, Version 2.0. The distributed plugin fork retains the upstream license and marks modified files.

## llama.cpp / ggml

The GGUF quantization implementation contains source copied or adapted from llama.cpp/ggml. Current source annotations identify historical llama.cpp revision `b2899`, including material from `mmq.cu`, `mmvq.cu`, `convert.cu`, `vecdotq.cuh`, and `ggml-common.h`.

Project: https://github.com/ggml-org/llama.cpp

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Qwen model artifacts

The target GGUF, vision projector, MTP checkpoint, tokenizer, processor metadata, and extracted PLE representation are model artifacts derived from Qwen3.8 Flash Next. They are not covered by R9V's Apache-2.0 code license.

Project: https://huggingface.co/Qwen/Qwen3.8-Flash-Next

License: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE)

## Unsloth IQ4_XS target

The target shards use Unsloth's `UD-IQ4_XS` quantization of Qwen3.8 Flash Next.

Project: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF

The exact source revision and file hashes are pinned in `release/sources.lock.json`. The model remains under Qwen Community License 1.0; Unsloth is credited for the quantized artifact.

## ggml-org vision projector

The current Q8_0 vision projector is distributed by ggml-org and was produced with llama.cpp conversion tooling; it was not independently quantized by R9V.

Project: https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF

The exact source revision and projector hash are pinned in `release/sources.lock.json`.

## Radiance exclusion

The inspected `StillDeadcode/vllm-radiance` upstream tree does not contain an explicit license. Radiance source is not included in this repository. R9V release tooling is independently implemented against the licensed vLLM and vLLM GGUF-plugin bases.
