# Third-party notices: runtime qwen38-flash-next-gfx1201-mtp4-v3

## vLLM and the vLLM GGUF plugin

Most files in `overlays/` are modified copies of files from the pinned vLLM
fork and vLLM GGUF plugin fork (Apache License 2.0). Their SPDX headers name
the vLLM contributors and mark the R9V modification. `full_mutable_cache.py`
and `iq4nl.py` are R9V code under the repository's Apache License 2.0.

- vLLM: https://github.com/vllm-project/vllm
- vLLM GGUF plugin: https://github.com/vllm-project/vllm-gguf-plugin

## llama.cpp / ggml

The `csrc/gguf/` headers and kernels under `sources/` (and the kernels compiled
from them into `overlays/q8_*.so`) contain source copied or adapted from
llama.cpp/ggml. Their annotations identify historical llama.cpp revision
`b2899`, including material from `ggml-common.h`, `mmq.cu`, `mmvq.cu` and
`vecdotq.cuh`.

Project: https://github.com/ggml-org/llama.cpp

MIT License

Copyright (c) 2023-2024 The ggml authors

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
