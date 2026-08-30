# Public release gates

The catalog rework is intentionally stricter than the original research tree.
The following items block the next public tag:

## Repository-wide

- Retain the independently implemented multimodal draft-mask alignment helper
  and its invariant-based tests in the pinned vLLM revision; the historical
  copied hunk must remain absent.
- Repeat the file/hunk provenance pass in `docs/provenance-audit.md` against
  the final committed revisions and retain the result with the release.
- Preserve the exact llama.cpp `b2899` MIT notice in the wrapper, plugin fork,
  kernel source package, and binaries that inline those primitives. The
  working tree now does this; the submodule changes still need commits.
- Add SPDX/change provenance to remaining R9V-generated or modified source
  files, including generated HIP code where the generator allows it.
- Create resolved, immutable release locks and test a recursive clone before
  tagging.

## Qwen profile

- Publish the release-candidate package to an immutable model revision and replace the
  descriptor's `revision: null`.
- Test `r9v fetch → verify → build → run` from a clean machine.
- Keep the Qwen Community License separate from root Apache-2.0.

## Muse R9V V1 profile

- Publish the exact 24,554,611,392-byte V1/V12 GGUF to an immutable model
  revision with the rough-draft quality warning intact.
- Include Apache License 2.0 and the Meta/Unsloth/llama.cpp attribution notice
  in that model package, and pin both files in its package descriptor.
- Curate and license-audit the minimal proof-engine source; do not import the
  legacy research workspace wholesale.
- Rebuild and hash-match the accepted executable and gfx1201 code object.
- Add a tokenizer-aware native CLI and clean-checkout end-to-end test.
- Keep the profile experimental until the advertised user path works. A speed
  result does not erase the quality gap versus Unsloth Q5/Q6.

This document is an engineering provenance checklist, not legal advice.
