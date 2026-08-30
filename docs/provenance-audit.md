# Fork provenance audit

Audit date: 2026-08-29

This is an engineering source-provenance review, not legal advice. It records
the source boundaries examined for the R9V release candidate and the checks
that must be repeated against the final committed revisions.

## Revisions examined

| Component | Compared range | Intended license boundary |
|---|---|---|
| vLLM fork | `d8d2b86cb8..dee01d0dea37b3ccca019ca65112ff0594846aa6` | Apache-2.0 vLLM work plus R9V-authored changes |
| GGUF plugin fork | `fb973ad..cd301d332c33befce8eb53359a94e8710c745140` | Apache-2.0 plugin work; identified ggml portions remain MIT |
| R9V gfx1201 kernels | `28f40ebb813d5b228fb6bf59672ca23bd8ffb063` | R9V Apache-2.0 code with identified ggml MIT inputs |
| Radiance comparison tree | `620a59d` | Comparison only; not redistributed |

The vLLM ancestry includes the official Qwen model and PLE commits before the
R9V commits. The release lock records those revisions rather than presenting
the complete fork as a clean-room engine.

## Radiance comparison

The upstream Radiance tree contains 70 tracked files and no identified
license file. R9V therefore does not distribute that tree, its launcher, or
its R4D implementation.

The audit compared normalized, consecutive four-line non-comment windows from
every R9V-added code run against every tracked Python, shell, C++, CUDA, and
HIP source file in Radiance `620a59d`. The minimum candidate length was 120
characters.

- vLLM: 152 added code runs checked; zero matches.
- GGUF plugin: 70 added code runs checked; zero matches.

A full-file scan initially found four matches in the autoregressive
speculator. `git blame` traced every matching line to pre-existing upstream
vLLM commits, not to an R9V change. The previously identified multimodal
draft-mask snippet was removed and replaced with the independently specified
`align_draft_multimodal_mask` helper and invariant-based unit tests.

This similarity scan is evidence against an overlooked verbatim copy; it is
not proof that two implementations share no general ideas or interfaces.

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
3. Repeat this comparison against the frozen tree and retain its output.
4. Inspect the built source, wheel, image, and kernel artifacts for their
   expected `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES` files.
5. Have the human publisher review every fork delta and this report before
   publishing the tag.
