# Licensing and distribution boundaries

This is an engineering provenance summary, not legal advice.

## Source code

| Component | License in an R9V release | Boundary |
|---|---|---|
| R9V catalog, scripts, profiles, and original kernels | Apache-2.0 | Covered by the root `LICENSE` and `NOTICE`. |
| vLLM fork | Apache-2.0 | Upstream license retained; R9V modifications are recorded in commits and release notices. |
| vLLM GGUF plugin fork | Apache-2.0 plus llama.cpp MIT notice | The source and built distribution include `THIRD_PARTY_NOTICES.md`. |
| llama.cpp/ggml-derived quant primitives | MIT | The exact historical `b2899` notice is retained in every source package that incorporates them. |
| Radiance | Not redistributed | The upstream tree has no identified license. R9V may describe the development lineage, but does not ship its launcher or R4D source. |
| Muse native proof engine | Not published yet | Only a curated R9V-owned/third-party-audited subset may be released; the legacy research workspace is excluded. |

The root Apache-2.0 license applies only to code Dylan owns or is entitled to
distribute under that license. It does not relicense separable third-party
components or model weights.

## Model artifacts

| Model package | License | Release handling |
|---|---|---|
| Qwen3.8 Flash Next target, MTP, projector, metadata, and derived PLE | Qwen Community License 1.0 | Distributed separately with the exact Qwen license. Users must review the commercial MaaS/AI-work-assistant and scale clauses. |
| Muse Glimmer 30B R9V V1 | Apache-2.0 | The modified quantization retains Meta's license and identifies Meta, the pinned Unsloth Q8 source, and llama.cpp tooling. Meta's usage policy is linked as additional use guidance. |

The Qwen package's staged `LICENSE` preserves the official license text with
normalized line endings, and its exact bytes are pinned in the package
descriptor. The Muse source locks record the immediate Unsloth revision and
the full Meta base-model revision.

## Release checklist

Before a public tag:

1. Commit and pin every submodule notice/provenance change.
2. Run the file-level fork-delta review recorded in `docs/release-gates.md`.
3. Build source and wheel artifacts and confirm their license-file contents.
4. Publish model packages under their own license and immutable revision.
5. Test `fetch`, `verify`, `build`, and `run` from a clean recursive clone.

Authoritative references:

- Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>
- vLLM license: <https://github.com/vllm-project/vllm/blob/main/LICENSE>
- vLLM GGUF plugin: <https://github.com/vllm-project/vllm-gguf-plugin>
- llama.cpp license: <https://github.com/ggml-org/llama.cpp/blob/master/LICENSE>
- Qwen3.8 Flash Next: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- Muse Glimmer GGUF: <https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF>
- Muse usage policy: <https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/USAGE_POLICY.md>
