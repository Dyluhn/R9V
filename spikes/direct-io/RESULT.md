# Spike Result: direct-io

- **Spike ID**: S5 (`direct-io`)
- **Card**: A0.S5
- **Governing Specs**: Spec 9 §5.1, Roadmap §A0
- **Status**: PASS

## Hardware Fingerprint

- Storage: Samsung SSD 990 EVO Plus 1TB (`/dev/nvme0n1p3`, PCI
  `0000:04:00.0`) at `16.0 GT/s x4` (Gen4 x4). A hard runtime gate rejects
  any other device, endpoint, or link state. The Gen3 Crucial P5 volume was
  not used.
- Filesystem: btrfs on `/var`; test file under `/var/tmp`.
- OS: kernel `7.1.5-ogc5.1.fc44.x86_64`; CPU governor `powersave`, cpuidle
  `acpi_idle`, PCIe ASPM `performance`, NVMe APST primary timeout 100 ms.
- GPU: AMD Radeon AI PRO R9700, gfx1201, 32624 MiB.
- HIP: `/opt/rocm/lib/libamdhip64.so.7` from `r9v-ci:test`, loaded through
  `r9v-hip`.
- Compiler: `rustc 1.96.0 (ac68faa20 2026-05-25)`, release build.

## Execution

- Build:
  `CARGO_TARGET_DIR=/tmp/r9v-a0s5-target cargo build --release --manifest-path spikes/direct-io/Cargo.toml --offline`
- Run:
  `docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined -v /tmp/r9v-a0s5-target/release:/work:ro -v /var/tmp:/var/tmp:rw --entrypoint /work/spike-direct-io r9v-ci:test /var/tmp/r9v-a0s5-qd8.bin`
- Input: 4 GiB deterministic SplitMix64 data, 256 x 16 MiB chunks, fsync'd,
  fully allocated, and prepared outside timed regions.
- Staging: eight 16 MiB `hipHostMalloc` buffers, all asserted 4 KiB aligned;
  one 4 GiB device allocation and eight HIP streams.
- Direct-I/O engines:
  - E0: eight blocking `pread` workers, one pinned slot per worker.
  - E1: single-threaded io_uring QD8, registered buffers/file, SQPOLL, and a
    CPU-pinned submitter.
  - E2: single-threaded io_uring QD8 with plain buffers/file and no SQPOLL.
- Every read must complete exactly 16 MiB; total byte accounting must equal
  4 GiB; every timed chunk checksum is verified; observed peak QD must equal
  8. Warmups compare every word. The pipeline additionally copies all chunks
  back from the GPU and compares every word in all 256 chunks against the
  generated file contents. Any failed gate exits nonzero.
- Rate is decimal GB/s (`bytes / seconds / 1e9`). Reported rates use medians.

## Raw Measurements

Two independent executions of the final binary passed:

| Metric | Run A | Run B |
|---|---:|---:|
| E1 fixed+SQPOLL raw seconds | 0.642, 0.643, 0.641 | 0.635, 0.633, 0.634 |
| E1 median direct read | **6.70 GB/s** | **6.77 GB/s** |
| E2 plain raw seconds | 0.693, 0.705, 0.820 | 0.685, 0.679, 0.675 |
| E2 median direct read | **6.09 GB/s** | **6.33 GB/s** |
| E0 blocking raw seconds | 0.992, 0.977, 0.999, 0.996, 0.979 | 0.953, 0.963, 0.971, 0.943, 0.962 |
| E0 median direct read | 4.33 GB/s | 4.46 GB/s |
| H2D-only raw seconds | 0.089, 0.078, 0.078 | 0.089, 0.078, 0.078 |
| H2D-only median | 55.02 GB/s | 55.05 GB/s |
| Pipelined read+H2D raw seconds | 0.775, 0.761, 0.774, 0.798, 0.781 | 0.761, 0.764, 0.770, 0.764, 0.762 |
| Pipelined median | 5.54 GB/s | 5.62 GB/s |
| Serial E0 read + H2D vs pipeline | 1.070 s vs 0.775 s | 1.040 s vs 0.764 s |
| Observed peak direct-read QD | 8 | 8 |
| Device copy-back verification | 256/256 match | 256/256 match |

Run B also recorded `fixed=true sqpoll=true` for every E1 sample and
`fixed=false sqpoll=false` for every E2 sample. The runtime fingerprint
reported endpoint `0000:04:00.0` at `16.0 GT/s x4`.

## Judgment

- Spec floor: O_DIRECT read throughput at QD8 into pinned staging memory is
  at least 5 GB/s on the reference Gen4 NVMe path.
- Measured: E1 sustained 6.70 and 6.77 GB/s median in independent runs. E2
  independently sustained 6.09 and 6.33 GB/s.
- Pipeline: H2D overlap is demonstrated in both runs, and end-to-end
  throughput is 5.54–5.62 GB/s.
- Correctness: alignment, O_DIRECT, exact-read, total-byte, QD8, host-content,
  overlap, GPU-content, reference-device, and cleanup gates all passed.
- **Result: PASS.** The immutable 5 GB/s floor is exceeded without changing
  the target. The test file was removed after each run; no generated binary
  or test data remains in the repository.
