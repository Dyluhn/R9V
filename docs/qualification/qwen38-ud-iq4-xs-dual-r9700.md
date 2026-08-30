# Qwen3.8 UD-IQ4_XS dual-R9700 qualification

The release candidate exposes OpenAI-compatible text, tools, and one-image
inputs with MTP2 and 128K context. The public-source ROCm 7.14 image was built
locally from the pinned repository revisions and tested through its streaming
OpenAI chat endpoint with thinking disabled.

| Prompt + completion | PP | TG | Result |
|---:|---:|---:|---|
| 278 + 256, three runs | 61.41 mean | 78.11 mean | 76.48–80.57 TG |
| 8,206 + 256 | 55.75 | 76.47 | completed at length |
| 32,779 + 256 | 57.01 | 72.85 | completed at length |

PP is `prompt_tokens / time-to-first-streamed-token`. TG is the 255-token
inter-token interval after the first streamed token. Long prompts were built
in memory with `tools/benchmark_openai.py --repeat-count`, and each used a
different first block so prefix caching could not reuse another benchmark.

Rank 0 was the display card; rank 1 used the synchronous LRU16 expert cache.
PLE used SSD residency with periodic mmap trimming. R4D was disabled. The
production launch also pins RCCL to `Ring/Simple`, keeps AITER unified
attention enabled, and disables the AITER
linear/MHA/MLA/MoE/RMSNorm/FP8BMM/FP4BMM sub-backends. That dispatch policy is
part of the measured profile, not a portable default.

The same server passed text generation and a one-image OpenAI request; the
image response identified the fixture as a water lily. `/v1/models` reported
`max_model_len=131072`.

These measurements qualify the public source/runtime on the exact local model
bundle and reference topology. Public package upload followed by a clean-host
`fetch → verify → build → run` check is still required before the profile can
move from `release-candidate` to `qualified`.
