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

## Grouped-prefill V1 qualification

The V1 profile now selects the bit-exact group-16 GGUF MoE prefill kernel.
Distinct prompt slices included a unique nonce, forcing zero prefix-cache
hits. The unprofiled OpenAI completions endpoint measured:

| Target prompt | Runs | Mean PP | Median PP | Range |
|---:|---:|---:|---:|---:|
| 8K | 10 | 1,512.01 | 1,510.20 | 1,286.32–1,688.07 |
| 32K | 3 | 1,401.83 | 1,365.25 | 1,358.90–1,481.34 |
| 64K | 2 | 1,357.02 | 1,357.02 | 1,351.96–1,362.07 |

Nine 8K requests contained exactly 8,192 tokens and the final corpus slice
contained 8,136. All 32K and 64K requests matched their target exactly. The
server remained healthy throughout and was stopped cleanly. A clean image was
then rebuilt from the same source tree and launched without Python or kernel
development overlays. Three additional 8K cache-miss requests measured
1,277.22, 1,484.00, and 1,566.80 PP tok/s (1,442.67 mean; 1,484.00 median).
The server reported a 0.0% prefix-cache hit rate and was again stopped cleanly.
The verified image ID is
`sha256:09411bb3e4782eff8c45fd90be620a8d4f808bfb55b8210045c106eef8b3e23a`.
The prompt-free raw trial data and pinned corpus provenance are published in
[qwen38-group16-pp-v1.json](results/qwen38-group16-pp-v1.json).

## llama.cpp ROCm target-only comparison

The same IQ4_XS target was also measured with the Qwen4Exp llama.cpp reference
branch at commit `6c5afc86a`. This comparison intentionally disabled MTP and
vision. The 28,800,138,240-byte `per_layer_token_embd.weight` PLE table was
overridden to the CPU with `--load-mode none`, which populated an anonymous
RAM allocation instead of leaving the table mmap-backed on SSD. The process
held about 29 GiB of anonymous resident memory during load.

llama.cpp used ROCm 7.14, F16 KV, flash attention, a 2,048-token logical batch,
a 256-token physical batch, and `layer` splitting over the same two R9700s.
Three native `llama-bench` repetitions produced:

| Runtime cell | PLE residency | MTP | Mean tok/s | Samples |
|---|---|---:|---:|---|
| llama.cpp PP8192 | RAM | 0 | 534.33 | 535.45, 536.26, 531.28 |
| llama.cpp TG256 | RAM | 0 | 28.44 | 28.40, 28.47, 28.45 |
| R9V V1 PP8192 | SSD | 2 | 1,512.01 | ten cache-miss requests |
| R9V V1 TG256 | SSD | 2 | 78.11 | qualified OpenAI reference |

The qualified R9V cells are 2.83x faster for PP8192 and 2.75x faster for
TG256. This is a requested backend comparison rather than a feature-matched
ablation: MTP accounts for part of the generation advantage, while giving
llama.cpp the full PLE table in RAM favors its prefill result. A separate
three-request OpenAI cache-miss check measured llama.cpp at 489.53 PP and
28.32 TG with 271 input and 256 output tokens.

The only working llama.cpp dual-ROCm placement was `layer` split. Experimental
`tensor` split asserted while creating Qwen hybrid memory, `row` split is not
supported by the ROCm backend, and large server KV reservations exhausted the
remaining VRAM. The native 8K benchmark succeeded by allocating only the
context and compute buffers needed for that cell. Raw samples, the complete
command shape, residency evidence, and failed-mode diagnostics are recorded in
[qwen38-llamacpp-rocm-no-mtp-ram-ple.json](results/qwen38-llamacpp-rocm-no-mtp-ram-ple.json).

These measurements qualify the public source/runtime on the exact local model
bundle and reference topology. Public package upload followed by a clean-host
`fetch → verify → build → run` check is still required before the profile can
move from `release-candidate` to `qualified`.
