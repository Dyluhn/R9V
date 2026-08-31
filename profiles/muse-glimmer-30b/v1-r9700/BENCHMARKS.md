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
- Vision, DFlash, chat templating, and API overhead are excluded. Both R9V
  and llama.cpp support DFlash2 with this model's sidecar, so the TG cells
  are plain autoregressive decode, not either engine's fastest configuration.
- Quality is addressed separately and is the larger release caveat: V1 is
  substantially worse than Unsloth Q5/Q6 on the recorded evaluator.
