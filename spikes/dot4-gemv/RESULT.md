# Spike Result: dot4-gemv

- **Spike ID**: S2 (`dot4-gemv`)
- **Card**: A0.S2
- **Governing Specs**: Spec 4 §5, Roadmap §A0
- **Status**: [PENDING_RUNNER | PASS | FAIL]

## Hardware Fingerprint
- GPU: [e.g. gfx1201]
- Driver Version:
- ROCm Version:
- Device Memory Bandwidth Spec (GB/s):

## Execution
- Command: `hipcc -O3 --offload-arch=gfx1201 spikes/dot4-gemv/dot4_gemv.hip -o spikes/dot4-gemv/dot4_gemv && ./spikes/dot4-gemv/dot4_gemv`

## Raw Measurements
| Batch M | Achieved Bandwidth (GB/s) | Measured Latency (µs) |
|---|---|---|
| 1 | | |
| 4 | | |
| 8 | | |

## Judgment Against Spec Claim
- Claim (Roadmap §A0, Spec 4 §5): `v_dot4_i32_i8` GEMV over L1 achieves streaming memory read throughput approaching device memory bandwidth ceiling at small batch M in {1, 4, 8}.
- Pass/Fail Judgment: [PASS / FAIL]
- Notes:
