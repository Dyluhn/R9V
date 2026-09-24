# Runtime qwen38-flash-next-gfx1201-mtp4-v3 (consolidated 1.3.0 with host expert dedupe)

This runtime is the pinned image `sha256:2dac17a2…` plus read-only overlays.
The overlays are the consolidated 1.3.0 files that run as the reference host's
deployed service, copied byte for byte, except the mutable expert cache (see
[Host expert dedupe](#host-expert-dedupe)).

- `overlays/`: nine Python files that replace image files on every launch, the
  CED model file (`model.py`, mounted only with CED on), and four compiled
  kernels loaded from the directory mount at `/r9v-full-mutable`: the mutable
  expert cache (`candidate.so`) and the Q8 WMMA prefill kernels (`q8_*.so`).
  `full_mutable_pins.json` lists the experts the cache keeps in VRAM for good.
  `runtime.json` pins every file's SHA-256, its mount target and the
  environment switches that turn the kernels on and name the pin list.
  `tools/runtime_overlays.py` checks all of it before `scripts/launch.sh`
  creates a container.
- `sources/`: the sources, build scripts and build records of the four kernels.
  `sources/SHA256SUMS` pins every file.

The mutable expert cache only works with the fixed placement in
`packages/placements/qwen38-flash-next/uncensored-iq4-xs/dual-r9700/mtp4-full-mutable.json`.

## Host expert dedupe

In 1.3.0 the mutable expert cache could evict any expert, so the host (RAM)
copy held all 512 experts per layer for both GPUs: 55.4 GiB. Now rank 1 pins
its 400 most-routed experts per layer in its first 400 VRAM slots. The planner
never evicts those slots, so their experts are never read from host, and rank
1's host copy holds only the other 112 experts: 4.24 GiB instead of 19.40 GiB.
Rank 0 is unchanged. The host copy is 40.3 GiB in total.

- `overlays/full_mutable_pins.json`: the pins per layer, with the routing
  trace (by SHA-256) and split they were chosen from.
  `tools/pin_sim/make_pins.py` regenerates it from that trace;
  `tools/pin_sim/run.py` replays routing at other pin counts.
  `full_mutable_cache.py` refuses a pin list with any other SHA-256.
- The host copy is indexed through a host-row map (expert to row, -1 for a
  pinned expert) in the planner's copy kernel and in the MoE kernel.
  Startup checks that every pinned expert sits in its slot with no host row.
- The host copy is not locked in memory. Locking would need a raised memlock
  limit, which rootless Docker cannot grant without root.
- `candidate.so` is the binary that was GPU-tested (its SHA-256 is in
  `sources/cache/BUILD_RESULT.json`), built from the sources here.

GPU test on the reference host, 2026-09-24, against 1.3.0 in back-to-back
sessions: bitwise-identical prompt logprobs on 5 prompts (3K–15K tokens) and
identical 48-token greedy probes with top-5 logprobs on 2; short-prompt decode
34.87 vs 34.83 ms/step; a 127,238-token prompt ran; peak shared host memory
41 GiB instead of 56 GiB.

## Differences from the 1.3.0 package

The mutable expert cache files differ as described above
(`candidate.so`, `full_mutable_cache.py`, `tiered_compaction.py`,
`tiered_experts.py`, the new `full_mutable_pins.json` and the cache sources).
The 1.3.0 GPU planner replay tool (`sources/cache/test_planner_equivalence.py`)
was dropped: it assumes a 512-row host copy. `tests/test_full_mutable_pins.py`
replays routing through the planner on the CPU instead.

Four build records had absolute paths of the build machine. Those paths now
read `<build-workdir>/…`; nothing else changed. Their original SHA-256 values
before that edit:

| File | Original SHA-256 |
|---|---|
| `sources/cache/BUILD_SPEC.json` | `68b5f5f87b59b57ebfe9f2d9e439568754cdf7a322a53b1ce752a131c4c2b3a9` (1.3.0); `bc7e08a971caec19942f88ce3f3e671ee79460b25d6d7aa4cecf7ce1b4ca61b0` (host-dedupe build) |
| `sources/small-k-wmma/BUILD_RESULT.json` | `599a543775abb9e0ba19bfb6be81d71f01847e9800086b55b7d84b457984dfb3` |
| `sources/streamed-token32-fallback/BUILD_RESULT.json` | `1017bf88c4d24bbace6b40eaf160a23f1d27a2836519cc5aa8fd34c32736fbd5` |
| `sources/streamed-token64/BUILD_RESULT.json` | `3bc553bc64302d7e3404ce5850e4a6c10cbbe212453259dc346b53462aabb362` |

The 1.3.0 package's optional `perf/` overlays are not included.

Licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
