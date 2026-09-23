# Runtime qwen38-flash-next-gfx1201-mtp4-v3 (consolidated 1.3.0)

This runtime is the pinned image `sha256:2dac17a2…` plus read-only overlays.
The overlays are the consolidated 1.3.0 files that run as the reference host's
deployed service, copied byte for byte.

- `overlays/`: nine Python files that replace image files on every launch, the
  CED model file (`model.py`, mounted only with CED on), and four compiled
  kernels loaded from the directory mount at `/r9v-full-mutable`: the mutable
  expert cache (`candidate.so`) and the Q8 WMMA prefill kernels (`q8_*.so`).
  `runtime.json` pins every file's SHA-256, its mount target and the four
  environment switches that turn the kernels on.
  `tools/runtime_overlays.py` checks all of it before `scripts/launch.sh`
  creates a container.
- `sources/`: the sources, build scripts and build records of the four kernels.
  `sources/SHA256SUMS` pins every file.

The mutable expert cache only works with the fixed placement in
`packages/placements/qwen38-flash-next/uncensored-iq4-xs/dual-r9700/mtp4-full-mutable.json`.

## Differences from the 1.3.0 package

Four build records had absolute paths of the build machine. Those paths now
read `<build-workdir>/…`; nothing else changed. Their original SHA-256 values
in the 1.3.0 package manifest:

| File | 1.3.0 SHA-256 |
|---|---|
| `sources/cache/BUILD_SPEC.json` | `68b5f5f87b59b57ebfe9f2d9e439568754cdf7a322a53b1ce752a131c4c2b3a9` |
| `sources/small-k-wmma/BUILD_RESULT.json` | `599a543775abb9e0ba19bfb6be81d71f01847e9800086b55b7d84b457984dfb3` |
| `sources/streamed-token32-fallback/BUILD_RESULT.json` | `1017bf88c4d24bbace6b40eaf160a23f1d27a2836519cc5aa8fd34c32736fbd5` |
| `sources/streamed-token64/BUILD_RESULT.json` | `3bc553bc64302d7e3404ce5850e4a6c10cbbe212453259dc346b53462aabb362` |

The 1.3.0 package's optional `perf/` overlays are not included.

Licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
