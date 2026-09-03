# Spike Result: hipgraph

- **Spike ID**: S4 (`hipgraph`)
- **Card**: A0.S4
- **Governing Specs**: Spec 6 §5.2, Spec 14, Roadmap §A0
- **Status**: PASS

## Hardware Fingerprint

- GPU: AMD Radeon AI PRO R9700, `arch=gfx1201` (HIP device 0; Card Model 0x7551, SKU APM107573, Subsystem ID 0x5413, Device Rev 0xc0, Node ID 1, GUID 23334)
- Driver Version: kernel driver 7.1.5-ogc5.1.fc44.x86_64; HIP driver/runtime 71460850 (7.14.60850)
- ROCm Version: pinned SDK `/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core`, HIP 7.14.60850, AMD clang 23.0.0git
- Engine / Memory Clock (observed, demand-scaled): sclk level 1 at 797 MHz pre-run and 1091 MHz post-run, fclk 2128 MHz, socclk 1371 MHz, dcefclk 243 MHz, PCIe 32.0 GT/s x16
- Second GPU in system (rocm-smi GPU[1]) was idle and untouched; program pins HIP device 0

## Execution

- Build: `/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core/bin/hipcc -O3 --offload-arch=gfx1201 --rocm-device-lib-path=/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core/lib/llvm/amdgcn/bitcode -Wl,-rpath,/var/mnt/qwen-storage/projects/r9v/engine-src/.pydeps-native-quant-rocm/_rocm_sdk_core/lib spikes/hipgraph/hipgraph.hip -o /tmp/hipgraph_s4` (binary kept outside the repo at `/tmp/hipgraph_s4`)
- Note: `--rocm-device-lib-path` is required because the pinned SDK lays out the amdgcn bitcode outside hipcc's default search path; first attempt without it failed at compile time with "cannot find ROCm device library".
- Run: `LD_LIBRARY_PATH=<sdk>/lib /tmp/hipgraph_s4` (env `ROCM_PATH`/`HIP_PATH` set to the pinned SDK at build time)
- Runs: two independent executions, exit 0 both times; run logs at `/tmp/hipgraph_s4_run1.log`, `/tmp/hipgraph_s4_run2.log` (outside repo)

## Method (what the program does)

- Exact 400-kernel launch list: one `step_kernel` launch per chunk over 400 chunks of 2048 uint32 words (3.2 MiB total). Each launch uses 8 blocks x 256 threads so each of the 2048 threads owns exactly one chunk word via a single global index plus one bounds check (no overlapping writes, no out-of-range loop iterations). Each thread does an 8-iteration integer multiply-add chain with a per-launch constant and writes its word (full 8 KiB chunk per launch).
- Integer arithmetic is bit-exact on host and device, so the host-computed reference buffer is a strict oracle. (An earlier float version showed a 1-ulp host/device mismatch from FMA contraction differences and was replaced — the oracle is not the implementation compared to itself.)
- Every timed iteration on both paths is verified with a full-buffer `memcmp` against the reference (bit-exact, 100/100 iterations per run), proving launches are real, observable, and semantically equivalent across paths.
- Graph path: `hipStreamBeginCapture` / 400 launches / `hipStreamEndCapture` / `hipGraphGetNodes` asserted == 400 / `hipGraphInstantiate` — all once, outside replay timing. Replay is `hipGraphLaunch` only.
- Timing: output buffer reset + stream sync before the start event, so only replay time is measured; `hipEventRecord` start/stop on the same stream, `hipEventSynchronize` before reading. 10 warmups per path, then 50 timed samples per path alternating seq/graph to share thermal conditions.
- Strict gates, nonzero exit on any violation: any HIP error exits 2; node count != 400, any `memcmp` mismatch, graph max > 2x median, graph CV > 10%, or graph median not strictly below sequential median each print `FAIL:` and exit 1. An earlier failing build (float oracle) exited 1 as designed.

## Raw Measurements

### Run 1 (exit 0)

- Sequential 400-launch total (µs): median 1048.57, min 1045.33, max 1170.49, mean 1057.27, std 24.61
- Graph replay 400-launch total (µs): median 881.93, min 879.69, max 980.09, mean 890.29, std 23.93
- Per-launch dispatch-inclusive: seq 2.621 µs, graph 2.205 µs, saved 0.417 µs/launch, speedup 1.189
- Graph stability: max/median 1.111, CV 0.0271
- Correctness: 100/100 timed iterations bit-exact on both paths; captured nodes 400/400
- Seq raw (µs, 50): 1128.3, 1048.8, 1049.5, 1048.6, 1047.4, 1048.1, 1048.9, 1170.5, 1047.5, 1047.7, 1047.9, 1050.6, 1049.5, 1048.6, 1107.0, 1049.6, 1047.8, 1049.7, 1050.7, 1049.0, 1049.1, 1106.7, 1046.1, 1053.4, 1047.0, 1047.8, 1047.5, 1048.1, 1115.9, 1047.5, 1049.8, 1048.8, 1048.1, 1047.6, 1050.1, 1080.2, 1047.5, 1049.0, 1047.1, 1045.3, 1047.7, 1048.5, 1070.0, 1047.6, 1050.0, 1048.2, 1048.3, 1048.6, 1048.5, 1048.0
- Graph raw (µs, 50): 883.4, 881.7, 880.1, 882.5, 883.4, 881.0, 880.8, 900.7, 880.4, 880.0, 883.0, 882.6, 880.0, 911.4, 884.4, 881.3, 881.9, 884.1, 882.0, 880.8, 959.2, 882.2, 881.5, 880.6, 881.5, 880.2, 879.9, 934.0, 882.2, 879.7, 882.0, 882.4, 879.8, 881.2, 963.3, 882.5, 880.9, 883.8, 882.8, 881.3, 882.1, 958.6, 880.0, 880.2, 882.5, 880.7, 880.2, 882.6, 980.1, 880.3

### Run 2 — independent rerun, same binary (exit 0)

- Sequential 400-launch total (µs): median 1048.29, min 1045.17, max 1091.05, mean 1051.21, std 9.07
- Graph replay 400-launch total (µs): median 881.47, min 879.13, max 975.81, mean 892.40, std 27.14
- Per-launch dispatch-inclusive: seq 2.621 µs, graph 2.204 µs, saved 0.417 µs/launch, speedup 1.189
- Graph stability: max/median 1.107, CV 0.0308
- Correctness: 100/100 timed iterations bit-exact on both paths; captured nodes 400/400
- Seq raw (µs, 50): 1050.1, 1048.7, 1049.4, 1048.0, 1091.1, 1048.3, 1047.6, 1047.8, 1048.3, 1047.3, 1048.1, 1085.6, 1046.9, 1046.2, 1047.4, 1047.0, 1048.4, 1049.4, 1066.1, 1047.8, 1047.5, 1048.3, 1048.3, 1051.8, 1047.7, 1066.3, 1047.1, 1046.7, 1047.7, 1048.6, 1047.0, 1047.5, 1049.4, 1048.3, 1049.4, 1048.9, 1051.1, 1050.5, 1049.0, 1063.6, 1045.2, 1046.8, 1050.7, 1045.9, 1047.5, 1049.9, 1067.4, 1046.9, 1048.6, 1047.7
- Graph raw (µs, 50): 882.6, 881.4, 879.5, 932.1, 883.6, 881.3, 882.2, 883.7, 881.4, 880.7, 973.6, 881.5, 896.4, 881.9, 882.7, 880.6, 880.0, 947.3, 881.7, 879.3, 880.9, 883.3, 880.6, 884.4, 975.8, 880.9, 879.1, 881.6, 882.3, 880.6, 881.3, 961.6, 883.1, 881.4, 882.8, 882.6, 880.2, 881.1, 972.3, 880.7, 879.8, 882.8, 881.2, 879.8, 881.2, 941.1, 880.1, 880.9, 881.9, 881.2

| Metric | Run 1 | Run 2 |
|---|---|---|
| Sequential 400-launch median (µs) | 1048.57 | 1048.29 |
| Graph replay 400-launch median (µs) | 881.93 | 881.47 |
| Per-launch seq / graph (µs) | 2.621 / 2.205 | 2.621 / 2.204 |
| Dispatch saved per launch (µs) | 0.417 | 0.417 |
| Speedup (seq/graph) | 1.189 | 1.189 |
| Graph max/median, CV | 1.111, 0.0271 | 1.107, 0.0308 |
| Bit-exact iterations (seq+graph) | 100/100 | 100/100 |
| Captured nodes | 400/400 | 400/400 |

## Judgment Against Spec Claim

- Claim (Roadmap §A0, Spec 6 §5.2): capture and replay of a 400-launch list on gfx1201 is stable and reduces dispatch overhead versus sequential kernel launches.
- Pass/Fail Judgment: **PASS**
- Notes:
  - Capture is exact: `hipGraphGetNodes` reports 400 nodes for the 400-launch list, and every replay on both paths reproduces the host reference bit-exactly, so the graph is semantically the same list, not an optimized-away version.
  - Replay reduces dispatch overhead: graph median is ~167 µs (~0.42 µs/launch, speedup ~1.19) below sequential in both runs, ~6+ stddevs of separation; run-to-run medians agree within ~0.1%.
  - Replay is stable: graph CV ~0.03, max/median ~1.11 in both runs, far inside the fail gates (2.0x max, 0.10 CV).
  - Scope limit: kernels here are small integer compute kernels (~3.2 MiB working set), chosen so launch dispatch is a visible fraction of total time. The result supports Spec 6 §5.2's warmup comparison (graph eligible as an accelerator) but does not claim any model-step speedup; per Roadmap §7 risks, the launch list remains primary and hipGraph a non-dependency accelerator.
  - No SPEC-ISSUES entry: capture was stable and faster, so nothing contradicts the spec. Had it been slower or unstable, the program would have exited 1 with FAIL recorded here.
