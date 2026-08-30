# Muse Glimmer 30B R9V V1 qualification

Status: **experimental rough draft**

This is the canonical first R9V release profile. The exact model bytes are the
internal V12 research artifact, but the public package is named R9V V1.
Canonical does not mean quality-qualified: the profile deliberately fails
closed on a normal user launch until a curated runtime is published.

## What is frozen

- model: 24,554,611,392 bytes,
  `f4870ff4ac316c1dbf50a55501f4c00e16070336fc40e119ff1167e43382856a`
- proof executable: 2,061,048 bytes,
  `6d04e63d33e97469d6737f045f8a50f051c2eb3a58f5a9702a14df311b39b429`
- gfx1201 code object: 2,095,040 bytes,
  `46273cac83ac41f5e26cd06f29286eb47ad761b3d7d283c6757daa4c58b87974`
- same-model speed comparison against llama.cpp ROCm and Vulkan
- recipe/source identities and a direct 122,400-position quality comparison

## Quality gate: not passed

V1 is not quality-competitive with Unsloth's Q5/Q6 quants in the evaluator
used for this release. Its mean KLD is 0.006121, versus 0.003071 for Unsloth
UD-Q5_K_XL and 0.001034 for UD-Q6_K_XL. V1 is also larger than Unsloth Q5.
The model card must retain this warning until a replacement quant is measured
with the same evaluation protocol.

## User-runtime gate: not passed

The frozen engine is a raw-token HIP proof executable specialized for gfx1201.
It is not yet a distributable chat runtime, OpenAI-compatible server, or
vision-capable user setup. The source tree from the research campaign is not
imported wholesale because its file-level provenance and license boundary are
not release-clean.

## Performance evidence

The same GGUF was benchmarked on the same headless Radeon AI PRO R9700 against
llama.cpp ROCm and Vulkan. R9V is effectively tied at PP512, about 1.47x faster
than ROCm and 1.84x faster than Vulkan at PP2048/8192, and 1.10x/1.08x faster
at TG256. See [BENCHMARKS.md](BENCHMARKS.md) for exact samples and caveats.

## Gates before `release-candidate`

1. Curate and license-audit the minimal engine source required to reproduce
   the accepted executable and code object.
2. Rebuild both artifacts from the published tree and match correctness and
   speed within documented tolerances.
3. Publish the exact model package with Meta, Unsloth, and llama.cpp notices.
4. Provide at least a tokenizer-aware native CLI and an end-to-end smoke test.
5. Preserve the rough-draft quality disclosure; status promotion is not a
   claim that the quant beats Unsloth.
