# Spike Result: wmma-l1

- **Spike ID**: S1 (`wmma-l1`)
- **Card**: A0.S1
- **Governing Specs**: Spec 1 App. A, Spec 2 §2, Roadmap §A0
- **Status**: PASS

## Hardware Fingerprint
- GPU: gfx1201 (AMD Radeon AI PRO R9700, BDF 0000:03:00.0)
- Driver Version: 7.1.5-ogc5.1.fc44.x86_64
- ROCm Version: HIP 7.14.60850-0000000 / Clang 23.0.0git (ROCm SDK at `/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core`)
- Engine / Memory Clock: 2350.00 MHz sclk (level 2: 2350 MHz), 1258 MHz mclk (2128 MHz fclk)

## Execution
- Command: `/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core/bin/hipcc -O3 --offload-arch=gfx1201 -Wl,-rpath,/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core/lib spikes/wmma-l1/wmma_l1.hip -o spikes/wmma-l1/wmma_l1 && ./spikes/wmma-l1/wmma_l1`

## Raw Measurements
| Metric | Measured |
|---|---|
| Fragment order match L1 (yes/no) | yes (0 / 256 mismatches across all single-tile probes; 0 / 4096 mismatches across multi-tile tests) |
| iu8 throughput (TFLOPS) | 350.09 integer TOPS (2 ops/MAC; median 4.792 ms, 50,000 trips, 512 waves, 8 accs/thread, 15 timed repetitions) |
| iu4 throughput (TFLOPS) | 724.20 integer TOPS (2 ops/MAC; native `16x16x32`, median 4.633 ms); 377.75 integer TOPS (`16x16x16`, median 4.441 ms) |
| iu4 / iu8 throughput ratio | 2.069x (native `16x16x32` vs `iu8`); 1.079x (`16x16x16` vs `iu8`) |
| Independent rerun | iu8 351.08 TOPS (4.779 ms median, 4.413–4.905 ms); native iu4 721.06 TOPS (4.653 ms median, 4.533–4.790 ms); ratio 2.054x |

## Judgment Against Spec Claim
- Claim (Spec 1 App. A, Spec 2 §2): B fragments load directly from global memory in L1 lane order without register shuffle; measure real iu4 rate against iu8.
- Fragment order match: yes
- Measured iu4 / iu8 ratio: 2.069x (native `16x16x32` iu4 / `16x16x16` iu8) ; 1.079x (`16x16x16` iu4 / `16x16x16` iu8)
- Pass/Fail Judgment: PASS
- Notes:
  - **L1 Direct Load**: Empirically verified on device 0 (gfx1201). When weight tensor $W[N, K]$ is laid out in Spec 2 §2.2 L1 lane order (`lane = kgroup*16 + n`, `elem = lane*8 + j`, `value = W[n_base + n, k_base + kgroup*8 + j]`), each lane directly loads its B operand register (`i32x2` for iu8, `int` for iu4 16x16x16, `i32x2` for iu4 16x16x32) straight from global memory without LDS staging or register permutation. Numerical outputs verified bit-identically against CPU references across single-tile and multi-tile GEMM ($64\times 64\times 64$) with 0 mismatches.
  - **iu4 vs iu8 Throughput**: In gfx1201 (RDNA4), the primary native instruction for iu4 matrix multiplication is `v_wmma_i32_16x16x32_iu4` (`__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12`), which doubles $K$ to 32 at identical issue latency, yielding 724.20 TOPS (a 2.069x speedup over iu8). The legacy `v_wmma_i32_16x16x16_iu4` (`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`) achieves 377.75 TOPS (1.079x of iu8), validating the nominal 2x relative to fp16 mentioned in Spec 1 Appendix A.
  - **Rerun Stability**: Validated over repeated invocations; iu4/iu8 throughput ratio remains stable within 2.05x - 2.08x.
