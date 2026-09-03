# Spike Result: direct-io

- **Spike ID**: S5 (`direct-io`)
- **Card**: A0.S5
- **Governing Specs**: Spec 9 §5.1, Roadmap §A0
- **Status**: [PENDING_RUNNER | PASS | FAIL]

## Hardware Fingerprint
- Storage Device: [NVMe model, PCIe Gen/Lanes]
- Filesystem: [e.g. ext4, 4 KiB alignment]
- OS Kernel:

## Execution
- Command: `cargo run --release --manifest-path spikes/direct-io/Cargo.toml`

## Raw Measurements
| Metric | Measured |
|---|---|
| Queue Depth | 8 |
| Chunk Size | 16 MB |
| Sustained Direct I/O Read Throughput (GB/s) | |
| Pipelined H2D Transfer Throughput (GB/s) | |

## Judgment Against Spec Claim
- Spec Floor (Spec 9 §5.1): O_DIRECT read throughput achieves NVMe line rate (≥ 5 GB/s on PCIe Gen4) at queue depth 8 into pinned staging memory.
- Measured Read Throughput: [GB/s]
- Floor Met (≥ 5 GB/s on Gen4): [yes / no]
- Pass/Fail Judgment: [PASS / FAIL]
- Notes:
