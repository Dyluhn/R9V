# Spike Result: hipgraph

- **Spike ID**: S4 (`hipgraph`)
- **Card**: A0.S4
- **Governing Specs**: Spec 6 §2, Spec 14, Roadmap §A0
- **Status**: [PENDING_RUNNER | PASS | FAIL]

## Hardware Fingerprint
- GPU: [e.g. gfx1201]
- Driver Version:
- ROCm Version:

## Execution
- Command: `hipcc -O3 --offload-arch=gfx1201 spikes/hipgraph/hipgraph.hip -o spikes/hipgraph/hipgraph && ./spikes/hipgraph/hipgraph`

## Raw Measurements
| Metric | Value |
|---|---|
| Plain launch list total time for 400 launches (µs) | |
| hipGraph replay total time for 400 launches (µs) | |
| Dispatch overhead per launch (µs) | |
| Replay stability across repeated iterations | |

## Judgment Against Spec Claim
- Claim (Roadmap §A0, Spec 6 §2): Capture and replay of 400-launch list on gfx1201 is stable and reduces dispatch overhead compared to sequential kernel launches.
- Pass/Fail Judgment: [PASS / FAIL]
- Notes:
