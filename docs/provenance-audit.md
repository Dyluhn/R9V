# Fork provenance audit

Audit date: 2026-08-29

This is an engineering source-provenance review, not legal advice. It records
the source boundaries examined for the R9V release candidate and the frozen
source checks repeated against the final submodule revisions.

## Revisions examined

| Component | Compared range | Intended license boundary |
|---|---|---|
| vLLM fork | `d8d2b86cb88c91bbfad7fde09271d20147b8d50c..0c4efde1635edd52624bc60e52655a471b1bdc8a` | Apache-2.0 vLLM work plus R9V-authored changes |
| GGUF plugin fork | `fb973ad784f38b98b054e136bec3414b7cd8494d..566bbc0cd0fc873c425e9a2c4112cb21b30354b1` | Apache-2.0 plugin work; identified ggml portions remain MIT |
| R9V gfx1201 kernels | `4b825dc642cb6eb9a060e54bf8d69288fbee4904..0a466fb35bf29323f29f55c3800c405141299865` (empty tree to pin) | R9V Apache-2.0 code with identified ggml MIT inputs |
| Radiance comparison tree | `620a59d9e00df26571f60618291bf2dc6a9174fe` | Comparison only; not redistributed |

The vLLM ancestry includes the official Qwen model and PLE commits before the
R9V commits. The release lock records those revisions rather than presenting
the complete fork as a clean-room engine.

## Radiance comparison

The immutable upstream Radiance tree at
`620a59d9e00df26571f60618291bf2dc6a9174fe` contains 70 tracked files and no
identified license file. The local comparison checkout had later and dirty
work, so the frozen rerun read blobs directly from that upstream tree object
and did not scan local working-tree changes. R9V does not distribute that
tree, its launcher, or its R4D implementation.

The audit compared normalized, consecutive four-line non-comment windows from
every added code run against all 38 tracked Python, shell, C/C++, CUDA, and
HIP source files in the Radiance tree. Normalization removed whitespace and
full-line comments. Candidates shorter than 120 normalized characters were
discarded. A qualifying run is one that contains at least one retained
four-line window.

| Frozen source range | Non-comment added runs | Qualifying runs | Windows checked | Exact matches |
|---|---:|---:|---:|---:|
| vLLM `d8d2b86cb88c..0c4efde1635e` | 357 | 105 | 1,206 | 0 |
| GGUF plugin `fb973ad784f3..566bbc0cd0fc` | 102 | 51 | 5,231 | 0 |
| gfx1201 kernels, empty tree to `0a466fb35bf2` | 21 | 21 | 4,279 | 0 |

A full-file scan initially found four matches in the autoregressive
speculator. `git blame` traced every matching line to pre-existing upstream
vLLM commits, not to an R9V change. The previously identified multimodal
draft-mask snippet was removed and replaced with the independently specified
`align_draft_multimodal_mask` helper and invariant-based unit tests.

This similarity scan is evidence against an overlooked verbatim copy; it is
not proof that two implementations share no general ideas or interfaces.

## Frozen license and identifier checks

Case-insensitive `git grep` checks found no `SELECT24` reference in the root
tree or any of the three pinned submodule trees.

The SPDX pass covered added or modified Python, shell, C/C++, CUDA, and HIP
files in the two fork ranges, plus every tracked source file in the kernel
tree. All 59 of 59 vLLM files, 29 of 29 GGUF plugin files, and 21 of 21 kernel
files contain an `SPDX-License-Identifier`. The newly added vLLM ROCm 7.14
Dockerfile also carries `SPDX-License-Identifier: Apache-2.0`.

The frozen vLLM and plugin trees each contain their Apache-2.0 `LICENSE` and
`THIRD_PARTY_NOTICES.md`; both source manifests and project metadata include
those files in distributions. The kernel tree contains its Apache-2.0
`LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`. The root, plugin, and kernel
notices each retain the exact ggml MIT copyright line
`Copyright (c) 2023-2024 The ggml authors`. No SPDX or source-license payload
gap was found in this frozen scope.

## llama.cpp and ggml boundary

The GGUF plugin explicitly marks the files copied or adapted from historical
llama.cpp revision `b2899`, including `ggml-common`, dequantization, MMQ,
MMVQ, MoE-vector, and vector-dot sources. The exact historical MIT notice,
including `Copyright (c) 2023-2024 The ggml authors`, is present in:

- the R9V root third-party notices;
- the independently distributable plugin source package; and
- the R9V kernel source package.

The plugin build metadata includes `THIRD_PARTY_NOTICES.md` in source and
wheel distributions. R9V-generated GGUF kernel translation units and the
standalone fused-GDN adapter carry explicit license/modification headers.

## Final-tag requirements

1. Keep the root and all three submodules pinned to the audited revisions, with
   no dirty or untracked release inputs.
2. Update the release lock to those exact commits.
3. Retain this frozen comparison and its exact ranges with the release.
4. Inspect the built source, wheel, image, and kernel artifacts for their
   expected `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES` files.
5. Have the human publisher review every fork delta and this report before
   publishing the tag.
