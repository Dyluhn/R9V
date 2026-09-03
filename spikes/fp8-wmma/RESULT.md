# Spike Result: fp8-wmma

- **Spike ID**: S3 (`fp8-wmma`)
- **Card**: A0.S3
- **Governing Specs**: Spec 4 §8, Roadmap §A0
- **Status**: [PENDING_RUNNER | PASS | FAIL]

## Hardware Fingerprint
- GPU: [e.g. gfx1201]
- Compiler: clang++ (ROCm LLVM)
- ROCm Version:

## Execution
- Command: `hipcc -O3 --offload-arch=gfx1201 spikes/fp8-wmma/fp8_wmma.hip -o spikes/fp8-wmma/fp8_wmma && ./spikes/fp8-wmma/fp8_wmma`

## Raw Measurements
| Check | Observed |
|---|---|
| Builtin compilation (`__builtin_amdgcn_wmma_...`) | [yes / no] |
| Numerical output matches reference | [yes / no] |
| Leaf wrapper requires inline asm fallback | [yes / no] |

## Judgment Against Spec Claim
- Claim (Spec 4 §8, Roadmap §A0): Builtin FP8 WMMA compiler intrinsics compile and run correctly on pinned ROCm; leaf wrapper uses builtins without inline asm unless a miscompile is demonstrated.
- Pass/Fail Judgment: [PASS / FAIL]
- Notes:
