# Spike Result: p2p

- **Spike ID**: S6 (`p2p`)
- **Card**: A0.S6
- **Governing Specs**: Spec 5 §1, §2, §6; Spec 11 §7; Roadmap §A0
- **Status**: PASS for measured peer transport; PCIe fingerprint corrected 2026-09-03

## Hardware Fingerprint

- Board: ASRock B850M-C; AMD Ryzen 5 9600X; 12 logical CPUs; one NUMA node.
- Discrete rank 0: AMD Radeon AI PRO R9700, gfx1201, 32624 MiB, PCI
  `0000:03:00.0`; the complete upstream path has Gen5 x16 capacity; IOMMU
  group 15.
- Discrete rank 1: AMD Radeon AI PRO R9700, gfx1201, 32624 MiB, PCI
  `0000:13:00.0`; its endpoint reports Gen5 x16, but upstream root port
  `0000:00:02.2` is physically capped at Gen4 x4; IOMMU group 31.
- An integrated gfx1036 device is HIP device 2 and is outside the measured
  two-rank topology.
- Kernel: `7.1.5-ogc5.1.fc44.x86_64`; HIP runtime/driver version reported by
  the container: `71460850`; ROCm runtime from `r9v-ci:test`.
- The original result incorrectly treated endpoint sysfs as the complete PCIe
  path and consequently reported x16/x16. Endpoint capability does not expose
  an upstream bridge or root-port bottleneck. The corrected fingerprint walks
  every PCI ancestor and selects the lowest configured payload capacity using
  maximum speed with negotiated width. Current link speed is retained only as
  a transient diagnostic because idle links may downshift.

## Execution

- Compile:
  `docker run --rm -v /var/home/dylan/projects/inference/r9v-engine-spec:/src:ro -v /tmp/r9v-a0s6:/out:rw --entrypoint /opt/rocm/bin/hipcc r9v-ci:test -O3 -std=c++17 -Wall -Wextra -Werror --offload-arch=gfx1201 /src/spikes/p2p/p2p.hip -o /out/p2p`
- Run:
  `docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined -e LD_LIBRARY_PATH=/opt/rocm/lib -v /tmp/r9v-a0s6:/work:ro --entrypoint /work/p2p r9v-ci:test`
- Method: 16 KiB deterministic payload; 10 warmups and 50 timed trials per
  direction and transport. Each direct trial is a genuine
  `hipMemcpyPeerAsync` followed by source-stream completion, and runs only
  after `hipDeviceCanAccessPeer` and peer-enable succeed. Each host-staged
  trial performs and completes D2H into a pinned bounce buffer, switches to
  the destination device, then performs and completes H2D on that device's
  own stream. Latency is end-to-end monotonic wall time; GB/s is decimal
  payload bytes per second. Destination bytes are exhaustively compared with
  the deterministic source after each trial set.
- PCIe topology method: resolve each HIP device's BDF under
  `/sys/bus/pci/devices`, walk the endpoint and every PCI ancestor to the root,
  read both `current_link_*` and `max_link_*`, and choose the path-capacity
  bottleneck. P2P capability and measured peer-copy performance are recorded
  separately and never used to infer lane width.

## Measurements

Both independent executions of the final binary returned `A0.S6 PASS`:

| Direction | Peer query / enable | Direct latency, Run A / B | Direct GB/s, A / B | Host-staged latency, A / B | Host-staged GB/s, A / B |
|---|---|---:|---:|---:|---:|
| 0 → 1 | yes / yes | 21.8 / 21.5 µs | 0.7513 / 0.7625 | 91.5 / 81.3 µs | 0.1792 / 0.2014 |
| 1 → 0 | yes / yes | 27.3 / 31.3 µs | 0.6001 / 0.5240 | 79.6 / 79.7 µs | 0.2057 / 0.2057 |

Run B raw latency samples in milliseconds:

- Direct 0→1: `0.0506 0.0219 0.0331 0.0219 0.0222 0.0221 0.0220 0.0220 0.0219 0.0221 0.0222 0.0226 0.0223 0.0218 0.0216 0.0215 0.0213 0.0215 0.0214 0.0222 0.0220 0.0216 0.0214 0.0218 0.0215 0.0214 0.0215 0.0214 0.0214 0.0214 0.0216 0.0214 0.0214 0.0213 0.0211 0.0214 0.0214 0.0213 0.0216 0.0214 0.0213 0.0215 0.0214 0.0214 0.0215 0.0215 0.0214 0.0214 0.0214 0.0214`
- Host-staged 0→1: `0.0927 0.0841 0.0752 0.0771 0.0760 0.0818 0.0822 0.0823 0.0815 0.0821 0.0802 0.0813 0.0803 0.0755 0.0801 0.0808 0.0799 0.0810 0.0816 0.0812 0.0813 0.0808 0.1500 0.0798 0.0780 0.0860 0.0537 0.0793 0.0773 0.1015 0.0906 0.0830 0.0797 0.0797 0.0792 0.0785 0.0789 0.0839 0.0485 0.0976 0.1785 0.1020 0.0934 0.0936 0.1026 0.1991 0.0987 0.0922 0.0937 0.0753`
- Direct 1→0: `0.0398 0.0400 0.0356 0.0311 0.0315 0.0314 0.0315 0.0311 0.0312 0.0316 0.0316 0.0312 0.0314 0.0311 0.0312 0.0314 0.0312 0.0314 0.0311 0.0315 0.0311 0.0313 0.0317 0.0314 0.0317 0.0312 0.0313 0.0312 0.0314 0.0316 0.0312 0.0313 0.0312 0.0313 0.0311 0.0313 0.0312 0.0313 0.0317 0.0314 0.0317 0.0312 0.0314 0.0296 0.0271 0.0273 0.0273 0.0273 0.0270 0.0278`
- Host-staged 1→0: `0.0787 0.0801 0.0707 0.0796 0.0802 0.0799 0.0796 0.0793 0.1479 0.0812 0.0788 0.0794 0.0829 0.0799 0.0712 0.0800 0.0793 0.0793 0.0809 0.0803 0.1528 0.0793 0.0711 0.0585 0.0497 0.0549 0.0797 0.0810 0.0799 0.0800 0.0783 0.0796 0.0800 0.0794 0.0794 0.0727 0.0796 0.0731 0.0800 0.0779 0.0800 0.0800 0.0799 0.0797 0.0802 0.1466 0.0796 0.0797 0.0803 0.0712`

All four content checks passed in both runs. The first timed sample is retained
in every raw series; medians make the initialization outliers visible without
letting them determine the topology coefficient.

## Judgment

- The peer map is bidirectionally available and peer enable succeeds in both
  directions. Direct measurements are therefore genuine rather than HIP
  fallback copies.
- Direct is materially lower latency than explicit host staging at 16 KiB in
  both directions, and every destination byte matches.
- Selected topology transport for rank 0↔1: **Direct**.
- **Result: PASS.** The peer-map and copy claim is established for the measured
  pair. It does not claim that the paths are symmetric: the physical reference
  topology remains Gen5 x16 plus upstream-capped Gen4 x4.
