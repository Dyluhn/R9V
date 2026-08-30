# Third-party notices and model provenance

This file records the attribution and source lineage of Muse Glimmer 30B R9V
V1. Exact artifact hashes are in `package.json` and `sources.lock.json`.

## Meta Muse Glimmer 30B

Muse Glimmer 30B is published by Meta under Apache License 2.0. This package
includes the license as `LICENSE` and retains Meta as the base-model owner.

- Project: https://huggingface.co/meta-models/Muse-Glimmer-30B
- Revision: `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`
- Usage policy: https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/USAGE_POLICY.md

Meta's usage policy is separate guidance that users should review alongside
the Apache license.

## Unsloth Q8_0 quantization

The immediate quantization input is Unsloth's Q8_0 GGUF of Muse Glimmer 30B.

- Project: https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF
- Revision: `faa5b025c584459c13febfa5c59883516710ae39`
- File: `Muse-Glimmer-30B-Q8_0.gguf`
- SHA256: `f2c087d694ca8242a4a436076df7c041703ab051ac4b72bb1bfe2698299b0e86`

Unsloth is credited for producing and publishing that immediate Q8_0 source.
When included, the optional `mmproj-kquant.gguf` vision projector and
`dflash-kquant.gguf` draft model are also distributed unchanged from the same
pinned Unsloth repository and revision.

## llama.cpp and ggml

R9V used llama.cpp/ggml quantization tooling at revision
`dd1ea524333b1e697489067d7a4c39c60d32beee`. That tooling is MIT-licensed.
No llama.cpp or ggml source code is included in this model package.

- Project: https://github.com/ggml-org/llama.cpp
- License: https://github.com/ggml-org/llama.cpp/blob/dd1ea524333b1e697489067d7a4c39c60d32beee/LICENSE

## R9V V1 / internal V12 lineage

R9V supplies the V1/V12 tensor assignments, packaging, runtime specialization,
and explicit quality and qualification disclosures. `V1` is the public release
name; `V12` identifies the frozen internal lineage retained for
reproducibility. R9V does not claim authorship of the Meta model or Unsloth's
immediate Q8_0 quantization.
