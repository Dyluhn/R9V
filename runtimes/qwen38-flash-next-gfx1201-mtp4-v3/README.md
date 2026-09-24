# Runtime qwen38-flash-next-gfx1201-mtp4-v3 (consolidated 1.3.0 with host expert dedupe)

This runtime is the pinned image `sha256:2dac17a2…` plus read-only overlays.
The overlays are the consolidated 1.3.0 files that run as the reference host's
deployed service, copied byte for byte, except the mutable expert cache (see
[Host expert dedupe](#host-expert-dedupe)).

- `overlays/`: nine Python files that replace image files on every launch, the
  CED model file (`model.py`, mounted only with CED on), the CED quality model
  file (`model_ced_quality.py`, mounted instead with CED quality), and four compiled
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

## CED quality model file

`model_ced_quality.py` replaces the same image file as `model.py`, and only
with `--ced quality`. It reads the multi-source projector: the split-16
boundary state plus the block inputs of the full-attention layers 3, 7, 11
and 15 (the projector file's `sources` metadata). During an approximate chunk
it keeps those earlier block inputs as they are computed and joins them with
the boundary state in the declared order. A single-source projector takes the
unchanged path.

It is the file the multi-source projector was GPU-graded with on 2026-09-24
(research branch `ced-multisource`, commit `df3d457`, SHA-256
`d31b7b7f27050df7e98175bedd36a63726d3f14103c0a870497a024e8e7c30f2`), with one
change in `_ced_load`: a projector file already stored as int8 (int8 maps plus
`scale.*` and `bias.*`, as `kva/quantize_projector.py` writes them) loads as
stored instead of being quantized again. The grade loaded the bf16 file and
quantized it at load. For the shipped int8 file, every tensor the new
`_ced_load` puts on the GPU is bit-identical to what the graded one did (all
99 checked on the CPU in the pinned image). `tests/test_ced_quality_overlay.py`
checks the same on a small projector.

Its other differences from `model.py`:

- **Shared VRAM region (v0.4.3).** The vision encoder's weights (0.42 GiB per
  GPU, TP-sharded) and the projector (1.79 GiB) take turns in one VRAM region
  per GPU, sized for the larger. No step needs both: image and video prompts
  never get a CED plan, and a step is approximate only when it holds a single
  request. Both sets stay in pinned host RAM; `embed_multimodal` and the
  approximate chunk make theirs resident on demand (copied in only, the
  weights never change). Every tensor is a fixed view into the region, so its
  address is the same each time; the encoder and the approximate chunk run
  eagerly. All TP ranks see the same encoder calls and chunks and swap in the
  same step; a swap synchronizes the device before and after and is logged
  (`CED/vision shared VRAM swap N on cuda:R: A -> B, … in … ms`). The region is
  set up at vLLM's encoder profiling during startup, after the weights are
  final, and the projector loads then instead of in `load_weights`.
  `tests/test_ced_vision_swap.py` checks the region on the CPU.
- **Dequantization in blocks (v0.4.3).** int8 maps are dequantized at most
  2,560 rows at a time, in place: one late layer's map is one block, so the
  KV and GDN state CED writes is bitwise what the graded file wrote; "final"
  (4 blocks) differs only in rounding and feeds only the MTP drafter and the
  discarded logits of approximated positions.
- The research capture's `layers` option (inactive unless
  `R9V_KVA_CAPTURE_DIR` is set, which the launcher never sets).

Measured on the reference host with `--ced quality` (compiled server, desktop
apps holding 1.23 GiB of GPU 0), the minimum free VRAM during first-start
qualification went from 0.84–0.90 / 1.71–1.77 GiB (v0.4.2, fails the 1.5 GiB
target) to 2.04 / 2.49 GiB. The region alone gave 1.25 / 1.67 GiB; the rest
came from the smaller dequantization buffers, which also stop leaving 400 MiB
cached blocks behind.

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
