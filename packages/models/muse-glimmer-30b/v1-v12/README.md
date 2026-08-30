# Muse Glimmer 30B R9V V1

> **Rough-draft release.** This is the first public R9V model package, not the
> best available Muse Glimmer quant. It currently loses badly to Unsloth's
> higher-fidelity Q5/Q6 work in the quality evaluation used here. Do not infer
> quality superiority from the R9V name or the V1 label.

`V1` is the public release name. `V12` is the internal research lineage of the
exact GGUF and remains in manifests and hashes for reproducibility.

## What it is

- 24,554,611,392-byte GGUF
- SHA256 `f4870ff4ac316c1dbf50a55501f4c00e16070336fc40e119ff1167e43382856a`
- 731 tensors: 313 F32, 351 Q8_0, and 67 Q4_K
- seven gate/up pairs promoted to Q8_0 at layers 11, 15, 19, 23, 27, 31,
  and 35
- derived from Unsloth's pinned Q8_0 Muse GGUF with llama.cpp at the pinned
  revision in `sources.lock.json`

## Quality disclosure

All figures below use the same native-BF16 teacher, evaluator, 480 chunks,
512 tokens per chunk, and 122,400 evaluated positions.

| Quant | Bytes | Mean KLD (lower is better) | Same top prediction |
|---|---:|---:|---:|
| R9V V1 / V12 | 24,554,611,392 | 0.006121 | 96.879% |
| Unsloth Q4 comparison | 15,878,222,368 | 0.016883 | 94.806% |
| Unsloth UD-Q5_K_XL | 21,789,618,976 | 0.003071 | 97.724% |
| Unsloth UD-Q6_K_XL | 26,265,362,976 | 0.001034 | 98.752% |

On this evaluator V1 beats the compared Unsloth Q4, but that is not the useful
headline: V1 has about **1.99x the KLD of Unsloth Q5 despite being roughly
2.76 GB larger**, and about **5.92x the KLD of Unsloth Q6**. This package is a
speed-oriented engineering draft and a reproducible starting point for R9V,
not a recommendation over Unsloth's quality quants.

## Current product limitations

- The published R9V user runtime is not complete yet; current speed figures
  come from the frozen raw-token proof engine.
- No OpenAI-compatible API or chat template is qualified for this profile.
- Vision and DFlash artifacts are optional package components, but the speed
  comparison does not exercise them.
- The proof engine is specialized for gfx1201 and the exact tensor layout.
- The TG run reports 208 attention-pin fallbacks per 256-token sample; this is
  disclosed in the benchmark record rather than hidden.

See the profile [benchmark report](../../../../profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md)
and [qualification report](../../../../profiles/muse-glimmer-30b/v1-r9700/QUALIFICATION.md).

## License and provenance

The model artifact remains under Apache License 2.0. It is derived immediately
from Unsloth's pinned Q8_0 GGUF of Meta's Muse Glimmer 30B model and was
requantized with llama.cpp/ggml tooling. Exact revisions and hashes are in
`sources.lock.json`; complete attribution and Meta usage-policy guidance are in
[`MODEL_LICENSES/Muse-Glimmer-30B.md`](../../../../MODEL_LICENSES/Muse-Glimmer-30B.md).
