# Spike Result: fp8-wmma

- **Spike ID**: S3 (`fp8-wmma`)
- **Card**: A0.S3
- **Governing Specs**: Spec 4 §8 (inline-asm/intrinsics policy), Spec 4 §5.1 (GEMM inner loop `fp8_fp8`), Spec 4 §10 (gates analogue), Spec 1 §4 (matmul `e4m3×e4m3 → f32` fast path) + §6 (numerics), Roadmap §A0
- **Status**: PASS

## Hardware Fingerprint

- GPU: gfx1201 (AMD Radeon AI PRO R9700, DEVICE_ID 0x7551, BDF 0000:03:00.0, device 0)
- Compute units: 64 (HIP `multiProcessorCount` reports 32 WGPs)
- Clocks: runtime `clockRate` 2350.00 MHz = SYS LEVEL 2 (levels 500/1378/2350 MHz); MEM levels 96/456/772/875 MHz (via SDK `amd-smi static -g 0`)
- Memory: 31.86 GiB global
- Driver: in-kernel amdgpu under kernel `7.1.5-ogc5.1.fc44.x86_64` (`/sys/module/amdgpu/version` absent = built-in driver)
- Compiler: HIP 7.14.60850-0000000 / AMD clang 23.0.0git (`46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6`), SDK at `/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core`
- Note: Spec 4 §4.3's literal flag `-ffast-math=off` is rejected by this pinned clang (`unknown argument`); `-fno-fast-math` used instead (same intent; fast-math stays off). `-fno-gpu-approx-transcendentals` accepted as written. Filed as SI-1.

## Execution

- Build: `/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core/bin/hipcc -O3 --offload-arch=gfx1201 -fno-fast-math -fno-gpu-approx-transcendentals -Wl,-rpath,.../_rocm_sdk_core/lib spikes/fp8-wmma/fp8_wmma.hip -o /tmp/fp8run/fp8_wmma` (binary kept outside the repo; no generated files in `spikes/`)
- Run: `/tmp/fp8run/fp8_wmma [trips]` (default trips 20000), device 0, exit code 0 on all acceptance conditions, nonzero otherwise
- Disassembly evidence: `clang++ -x hip --offload-arch=gfx1201 -O3 -fno-fast-math -fno-gpu-approx-transcendentals --cuda-device-only -S spikes/fp8-wmma/fp8_wmma.hip -o /tmp/fp8run/fp8_wmma.s` → 10 static `v_wmma_f32_16x16x16_fp8_fp8` sites (1 single-tile + 1 K-loop + 8 unrolled bench accs; bench trip loop verified as a real counted loop, not unrolled/eliminated). LLVM IR (`-S -emit-llvm`) shows the exact intrinsic `llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32`.
- Inline-asm audit: `grep -cE '__asm__|asm volatile|__asm\b' spikes/fp8-wmma/fp8_wmma.hip` → 0.

## Raw Measurements

| Check | Observed |
|---|---|
| Builtin compilation (`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, 3-arg `(i32x2, i32x2, f32x8) → f32x8`) | yes, first try after signature probe (the fp8 form takes 3 args, no bool selects, unlike the iu8 6-arg form) |
| Lowers to intended instruction | yes: `v_wmma_f32_16x16x16_fp8_fp8` in device ISA (see above) |
| [T1] exact-int 16×16×16 tile vs f64 oracle | PASSED, 0/256 mism, max_abs 0, max_rel 0 (tol abs 1e-4 / rel 1e-4) |
| [T2] exact-fraction 16×16×16 tile vs f64 oracle | PASSED, 0/256 mism, max_abs 0, max_rel 0 (same tol) |
| [T3] extremes/subnormals tile (±448, ±240, min normal 2^-6, subnormals 2^-7/2^-9, ±0) | PASSED, 0/256 mism, max_abs 0.00781, max_rel 5.53e-08 (tol abs 1e-2 / rel 1e-3; residual is f32 accumulation rounding at large magnitude) |
| [T4] 64×64×64 multi-tile GEMM, seeded LCG values in [-100, 100] | PASSED, 0/4096 mism, max_abs 0, max_rel 0 (tol abs 1e-2 / rel 1e-3) |
| [T5] determinism (T4 repeated, bit-compare) | PASSED, 0/4096 bit-mismatches |
| [T6] peak register-bound rate (per-wave accounting, S1 convention) | 355.48 TFLOPS (median 1.888 ms, 7 reps, range 1.749–1.939 ms); reruns 358.43 / 353.39 TFLOPS — in line with S1's iu8 350 TOPS for the same 16×16×16 shape |
| Output checksum (FNV over all outputs incl. bench) | `0x5b416d62ff22c0d0`, identical across all three runs |
| Leaf wrapper requires inline asm fallback | no |
| Rerun validation | two extra runs, exit 0, T1–T5 numbers and checksum bit-identical, T6 within timing jitter |

Oracle/method notes: E4M3 codec documented in the source header (sign/exp-bias-7/mantissa; subnormal 2^-6·m/8; exp-15 m<7 valid to 448; only 0x7F/0xFF NaN, never generated as input). Encode is nearest-even over the 254 non-NaN codes with saturation to ±448. The oracle decodes the same bytes and accumulates in f64, independent of the device f32 path. Pass criterion `|d−r| ≤ abs_tol + rel_tol·|r|`. T1/T2 patterns are lane-indexed exact values, so any operand-lane or fragment-layout permutation moves outputs by O(1) against the 1e-4 tol; T4's 4096 seeded outputs catch tile-index/K-loop errors. NaN inputs excluded (not part of the matmul numerics contract). Bench `trips` comes from argv and all outputs are checksummed on host, so the compiler cannot eliminate the WMMA loop (ISA confirms a real counted loop).

## Judgment Against Spec Claim

- Claim (Spec 4 §8, Roadmap §A0): Builtin FP8 WMMA compiler intrinsics compile and run correctly on pinned ROCm; leaf wrapper uses builtins without inline asm unless a miscompile is demonstrated.
- Builtin compilation: yes
- Numerical output matches reference: yes (0 mismatches across 4864 checked outputs + bit-exact determinism)
- Inline-asm fallback required: no
- Pass/Fail Judgment: PASS
- Notes:
  - No builtin miscompile was demonstrated. SI-1 covers only the unsupported fast-math flag spelling. The `wmma_fp8` leaf wrapper (card A3.x) should use `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` directly, no asm switch needed.
  - The native WMMA 16×16×16 lane map assumed from S1 (lane l = row l%16, k-group (l/16)*8; acc lane l = rows (l/16)*8+v, col l%16) re-verified here for fp8 with zero mismatches — dtype-agnostic for this shape on gfx1201.
  - Sibling `bf8` (e5m2) builtins exist in this clang (`..._bf8_bf8/..._bf8_fp8/..._fp8_bf8_w32_gfx12`) but were not exercised; the second-operand-only e5m2 case remains for the A3 leaf-wrapper agreement tests.
  - Accounting caution found during the spike: WMMA throughput must be counted per-wave, not per-thread (32× factor); the program documents this and matches the S1 convention.
