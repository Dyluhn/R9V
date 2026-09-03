# Spike Result: wmma-l1

- **Spike ID**: S1 (`wmma-l1`)
- **Card**: A0.S1
- **Governing Specs**: Spec 1 App. A, Spec 2 §2, Roadmap §A0
- **Status**: [PENDING_RUNNER | PASS | FAIL]

## Hardware Fingerprint
- GPU: [e.g. gfx1201]
- Driver Version:
- ROCm Version:
- Engine / Memory Clock:

## Execution
- Command: `hipcc -O3 --offload-arch=gfx1201 spikes/wmma-l1/wmma_l1.hip -o spikes/wmma-l1/wmma_l1 && ./spikes/wmma-l1/wmma_l1`

## Raw Measurements
| Metric | Measured |
|---|---|
| Fragment order match L1 (yes/no) | |
| iu8 throughput (TFLOPS) | |
| iu4 throughput (TFLOPS) | |
| iu4 / iu8 throughput ratio | |

## Judgment Against Spec Claim
- Claim (Spec 1 App. A, Spec 2 §2): B fragments load directly from global memory in L1 lane order without register shuffle; measure real iu4 rate against iu8.
- Fragment order match: [yes / no]
- Measured iu4 / iu8 ratio: [ratio]
- Pass/Fail Judgment: [PASS / FAIL]
- Notes:
