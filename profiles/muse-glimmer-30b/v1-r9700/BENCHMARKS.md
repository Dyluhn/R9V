# Muse Glimmer R9V V1 backend speed comparison

This compares the exact V1/V12 GGUF on one isolated headless Radeon AI PRO
R9700. It is a speed record, not a quality comparison or a claim that every
R9700 system will reproduce these numbers.

Primary values are arithmetic means of three measured samples after one
warmup. Full samples and binary identities are in
[`benchmarks.json`](benchmarks.json).

| Cell | R9V HIP proof | llama.cpp ROCm | llama.cpp Vulkan | R9V / ROCm | R9V / Vulkan |
|---|---:|---:|---:|---:|---:|
| PP512 | 1,500.68 | 1,477.87 | 1,204.85 | 1.015x | 1.245x |
| PP2048 | 2,175.17 | 1,477.57 | 1,182.54 | 1.472x | 1.839x |
| PP8192 | 2,078.20 | 1,419.88 | 1,126.46 | 1.464x | 1.845x |
| TG256 | 26.84 | 24.30 | 24.92 | 1.105x | 1.077x |

PP512 is effectively tied with llama.cpp ROCm: the means favor R9V by 1.5%,
while the medians favor ROCm by 0.4% because the first ROCm sample was lower.
The material gains begin at PP2048. Vulkan is slightly faster than llama.cpp
ROCm for TG256 on this quant, so the table preserves that result.

## Protocol

- model: 24,554,611,392-byte V1/V12 GGUF, SHA256
  `f4870ff4ac316c1dbf50a55501f4c00e16070336fc40e119ff1167e43382856a`
- GPU: one Radeon AI PRO R9700, gfx1201, BDF `0000:13:00.0`
- host: Ryzen 5 9600X, 126.4 GiB RAM
- warmups: one per cell; measured repetitions: three
- llama.cpp revision: `dd1ea524333b1e697489067d7a4c39c60d32beee`
- llama.cpp: all layers on GPU, flash attention enabled, split mode none,
  main GPU 0, batch 2048, ubatch 512, F16 KV
- ROCm path: ROCm 7.14.0
- Vulkan path: Mesa RADV 26.1.6
- R9V: one graph replay, raw canonical GGUF, no prepared package, no vision,
  no DFlash sidecar

R9V PP runs a single output token after timing the prompt; llama.cpp PP uses
`-n 0`. R9V TG uses an 8-token seed and times 256 outputs; llama.cpp TG uses
pure `-p 0 -n 256`. This is the closest supported same-work comparison, but it
is not instruction-for-instruction identical.

## Preserved reproduction commands

R9V's checked-in harness records hashes and compact raw reports. The accepted
proof executable and code object are not public user-runtime artifacts, so this
command documents the frozen research protocol rather than a clean-clone user
workflow:

```bash
python3 tools/benchmark_muse_v1_r9v.py \
  --binary /path/to/muse_full_decode-v12-final-scratchshare \
  --hsaco /path/to/a6-dflash2-mrows.hsaco \
  --model /path/to/Muse-Glimmer-30B-R9V-V1.gguf \
  --repetitions 3
```

The llama.cpp cells use:

```bash
llama-bench -m model.gguf -p 512,2048,8192 -n 0 -r 3 \
  -ngl 99 -sm none -mg 0 -fa 1 -o json
llama-bench -m model.gguf -p 0 -n 256 -r 3 \
  -ngl 99 -sm none -mg 0 -fa 1 -o json
```

Run each command once with the ROCm build and once with the Vulkan build,
exposing only the intended GPU.

## Caveats

- This is a frozen research proof engine, not yet the distributable R9V user
  runtime.
- The R9V TG path reports 208 attention-pin fallbacks per 256-token sample for
  two small bulk shapes. The run is valid, but this remains an optimization
  and qualification gap.
- R9V uses about 25.04 GB peak allocated VRAM in the TG cell.
- Vision, chat templating, and API overhead are excluded. The TG cells above
  are plain autoregressive decode; DFlash2-assisted decode is measured
  separately below.
- Quality is addressed separately and is the larger release caveat: V1 is
  substantially worse than Unsloth Q5/Q6 on the recorded evaluator.

## DFlash2-assisted decode (measured 2026-08-30)

Both runtimes support DFlash2 speculative decoding with the package's
`draft/dflash-kquant.gguf` sidecar (SHA256
`27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc`).
These cells measure decode with the sidecar active, on the same model bytes
and the same isolated R9700.

Protocol: greedy, 128 new tokens per sample, one warmup plus three measured
samples per arm with arm order alternated, at two prompt depths — a 136-token
repetitive prompt and an 8,192-token corpus slice. This is a different cell
family from the raw TG256 table above (which decodes 256 tokens from an
8-token seed), so the raw columns here are re-measured under the same
protocol as the assisted columns.

- R9V: the frozen V12 CLI
  (`r9v-frozen-cli-lifecycle-freeze-v12-final-20260821-final1`,
  SHA256 `4cf4d966…`), `a6-dflash2-mrows.hsaco`
  (`46273cac…`), engine object pin verified, strict attention pin with zero
  fallbacks on every sample. Assisted arm: `R9V_DFLASH2_SPEC=1`,
  `N_MAX=15`, anchor fold and prompt lookup enabled. Timing is the engine's
  decode-phase clock.
- llama.cpp: the DFlash2-capable fork (`llama.cpp-dflash2` at `5ecbe1ac`
  plus a small local diff), `build-dflash2-hip`, gfx1201, ROCm 7.14.0.
  Both arms run through `llama-server` (`-ngl 99 -fa on`; assisted adds
  `-md <sidecar> --spec-draft-n-max 15`, spec type auto-detected as
  `draft-dflash`). Timing is `timings.predicted_per_second` from
  `/completion`. This fork is a newer revision than the `dd1ea524` baseline
  used for the raw TG256 table, so its raw column is measured on the fork
  itself.

Values are means of three samples (tok/s):

| Depth | llama.cpp raw | llama.cpp +DFlash2 | R9V raw | R9V +DFlash2 |
|---|---:|---:|---:|---:|
| 136 | 24.10 | 232.65 (9.66x) | 26.05 | 110.69 (4.25x) |
| 8192 | 23.55 | 40.55 (1.72x) | 24.39 | 59.65 (2.45x) |

Draft acceptance:

| Depth | llama.cpp accepted/drafted | R9V accepted/drafted |
|---|---|---|
| 136 | 119/119 | 119/133 |
| 8192 | 81/684 (12%) | 115/190 (61%) |

Reading the numbers:

- The depth-136 prompt is a repeated sentence, so greedy continuation is
  nearly fully predictable and acceptance saturates. Both assisted values
  there are best-case ceilings, not typical throughput; llama.cpp's verify
  path profits most from saturated acceptance.
- The depth-8192 corpus cell is the representative one. Acceptance separates
  the stacks — 61% for R9V versus 12% for llama.cpp — and R9V-assisted
  decodes 1.47x faster than llama.cpp-assisted on the same bytes.
- At depth 8192 the assisted greedy output drifts from raw greedy on both
  engines (verify logits are ground truth; a token can flip at near-ties).
  Depth-136 outputs are byte-equal to raw on both engines.
- Raw columns agree with the published TG256 numbers within protocol
  differences.

Environment note: the locked `laguna-rocm:7.14.0` build container was no
longer present; runs used its exact digest-pinned base image
(`rocm/dev-ubuntu-24.04:7.14.0-full@sha256:439edaa8…`, ROCm 7.14.0). Under
this substitution the R9V depth-136 outputs reproduced the 2026-08-21 freeze
evidence byte-for-byte. Raw sample data, per-run reports, and exact commands
are preserved in the workstation evidence record
(`AI-Work/2026-08-30-muse-dflash-tg`).
