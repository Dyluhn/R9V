# Spike Result: p2p

- **Spike ID**: S6 (`p2p`)
- **Card**: A0.S6
- **Governing Specs**: Spec 5 §2, Spec 5 §6, Roadmap §A0
- **Status**: [PENDING_RUNNER | PASS | FAIL]

## Hardware Fingerprint
- Topology: Dual AMD GPUs [e.g. R9700]
- Host Board / PCIe Topology:
- IOMMU / Peer Access Support:

## Execution
- Command: `hipcc -O3 spikes/p2p/p2p.hip -o spikes/p2p/p2p && ./spikes/p2p/p2p`

## Raw Measurements
| Metric | Measured |
|---|---|
| Peer access supported (`hipDeviceCanAccessPeer`) (yes/no) | |
| Direct P2P transfer latency at 16 KB (µs) | |
| Host-staged transfer latency at 16 KB (µs) | |
| Achieved P2P throughput (GB/s) | |

## Judgment Against Spec Claim
- Claim (Spec 5 §2, Roadmap §A0): Determine whether dual GPUs can peer-map directly on the reference rig; record direct vs host-staged 16 KB latency to populate `ArchDescriptor.p2p` link matrix.
- Peer access available: [yes / no]
- Direct latency vs host-staged latency: [direct µs vs host-staged µs]
- Pass/Fail Judgment: [PASS / FAIL]
- Notes:
