# R9V

R9V is a topology-first catalog of exact, model-specific inference systems for
AMD RDNA4. It is not one universal engine: every profile freezes a model and
quant package, runtime, kernel set, hardware contract, placement policy, and
qualification record.

The project was originally inspired by [antirez's DS4](https://github.com/antirez/ds4)
and [Neroued's ninfer](https://github.com/Neroued/ninfer): narrow inference
engines built around exact model and hardware targets instead of treating every
checkpoint as a generic workload. R9V is an independent implementation; this
acknowledges the design inspiration, not shared code provenance.

The catalog currently has two runtime families:

- **Single GPU:** fully custom native R9V engines specialized for an exact
  model, quant, and `gfx1201` kernel set.
- **Dual GPU:** an adapted vLLM deployment architecture informed by the
  Radiance profiling workflow, with a pinned Apache-2.0 vLLM fork, GGUF plugin,
  and R9V kernels.

Radiance was a development and profiling reference. This repository does not
redistribute its unlicensed launcher or R4D source. Every R4D path is
hard-disabled in the R9V profile; the separately built public Radiance
comparator used only its exact TP2 BF16 all-reduce.

## Supported profiles

| Topology | Profile | Runtime | Status | User surface |
|---|---|---|---|---|
| Single R9700 | `muse-glimmer-30b/v1/single-r9700` | Custom native HIP proof engine | Experimental | Frozen raw-token Muse V12/V1 proof; curated user runtime pending |
| Dual R9700 | `qwen38-flash-next/ud-iq4-xs/dual-r9700-128k` | Adapted vLLM stack | Release candidate | OpenAI-compatible text, tools, and one-image inputs |

`experimental` means the exact artifact or proof is preserved but the public
user path is incomplete. `release-candidate` means the runtime is qualified on
the reference topology but a release gate remains. A profile becomes
`qualified` only after its advertised download-to-user path passes end to end.

## Model downloads

| Model package | Direct download | Package contents | Runtime status |
|---|---|---|---|
| Qwen3.8 Flash Next R9V IQ4_XS | **[Hugging Face model repository](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS)** | Target GGUF shards, block-FP8 MTP, Q8 vision projector, metadata, and placement manifest | Dual-R9700 OpenAI runtime is a release candidate |
| Muse Glimmer 30B R9V V1 | **[Hugging Face model repository](https://huggingface.co/Dyluhn/Muse-Glimmer-30B-R9V-V1)** | V1/V12 GGUF, projector, DFlash sidecar, license, and provenance | Model is public; custom single-R9700 user runtime is still experimental |

Both repositories are public and immutable artifact revisions are pinned in
their R9V package descriptors. The model license shown in each repository
applies to its weights; downloading a model does not imply that its runtime has
passed every profile qualification gate.

Browse the catalog from a recursive clone:

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V

./r9v list --by-topology
./r9v validate
./r9v show qwen38
./r9v show muse
```

Topology indexes are under [`profiles/topology/`](profiles/topology/README.md).
Aliases such as `qwen38` and `muse` resolve to exact profile IDs; they do not
bypass a profile's fail-closed status or hardware checks.

## Profile lifecycle

The common lifecycle is:

`list → show → doctor → fetch → verify → build → run`

`show` and `validate` inspect the catalog without using a GPU. `doctor` checks
the selected machine and model bundle. The remaining actions execute only when
the profile exposes that stage and its release inputs are available.

- **Dual R9700 / Qwen:** the immutable 90.36 GiB model package is public and
  remotely hash-verified. The source image and local OpenAI serving path are
  qualified; the clean-host package installation test remains open. Follow
  [installation and launch](docs/installation.md) and the
  [qualification report](docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md).
- **Single R9700 / Muse:** the complete V12 research artifact is published as
  R9V V1, including the optional projector and DFlash sidecar. The curated
  native runtime is still pending: `fetch` is available, while `build` and
  `run` intentionally fail closed. See the
  [clean-clone status and artifact requirements](docs/muse-v1.md),
  [benchmark report](profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md),
  [qualification](profiles/muse-glimmer-30b/v1-r9700/QUALIFICATION.md), and
  [model card](packages/models/muse-glimmer-30b/v1-v12/README.md).

## Kernel scope

R9V kernels are deliberately narrow rather than generic ROCm replacements.
Their host APIs reject incompatible dtypes, shapes, qtypes, layouts, and GPU
targets; supported fallbacks remain owned by the selected runtime.

The Qwen profile ships three production `gfx1201` kernel families:

- `dense_mmvq_hip` for MTP multi-row GGUF GEMV reuse, exact Q8 attention M=3,
  and HyperConnection down/up paths;
- `tiered_iq_moe_hip` for mixed VRAM/UVA expert GEMV, graph-safe LRU caching,
  and the bit-exact group-16 MoE prefill path;
- `fused_gdn_mtp_hip` for the TP2 speculative GDN core.

These kernels are qualified only as part of their exact profile. Dense-prefill
tuning outside the selected group-16 path remains experimental and is not a V1
requirement. Porting to another model or GPU requires new parity, graph-replay,
placement, and end-to-end qualification. See the
[kernel scope and safety contract](kernels/r9v-gfx1201/README.md).

## Benchmarks

Benchmark numbers apply only to the named profile, reference hardware, and
recorded protocol. They are not blanket performance claims for every R9700.

### Muse Glimmer 30B — internal V12, public R9V V1

The exact 24,554,611,392-byte V12/V1 GGUF was measured on one isolated Radeon
AI PRO R9700. Values are arithmetic means of three samples after one warmup.
Both llama.cpp baselines use revision
`dd1ea524333b1e697489067d7a4c39c60d32beee` and the same model bytes.

| Runtime | PP512 (tok/s) | PP2048 (tok/s) | PP8192 (tok/s) | TG256 (tok/s) | Notes |
|---|---:|---:|---:|---:|---|
| R9V custom HIP | **1,500.68** (+1.54%) | **2,175.17** (+47.21%) | **2,078.20** (+46.36%) | **26.84** (+7.70%) | Exact V1/V12 engine |
| llama.cpp ROCm | 1,477.87 | 1,477.57 | 1,419.88 | 24.30 | Same model bytes |
| llama.cpp Vulkan | 1,204.85 | 1,182.54 | 1,126.46 | 24.92 | Same model bytes |

Each percentage is the winning result's advantage over the fastest alternative
backend in that benchmark category.

This is a speed result from a frozen raw-token proof engine, not a quality or
product-readiness endorsement. R9V V1/V12 records mean KLD `0.006121`, versus
`0.003071` for Unsloth UD-Q5_K_XL and `0.001034` for UD-Q6_K_XL on the same
122,400-position evaluator. It is therefore not quality-competitive with the
Q5/Q6 quants. Exact samples, binary hashes, protocol differences, and caveats
are in the [Muse benchmark record](profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md).

### Qwen3.8 Flash Next — R9V versus stock/public Radiance

The comparison target is the same dual-R9700 topology and benchmark cells. The
R9V row is the qualified V1 reference. The comparator was built separately
from public Radiance, public vLLM PR #53899, and the public GGUF plugin. Its
compatibility overlay only made Qwen4Exp GGUF, MTP, PLE, and UVA loading work;
it contained no R9V performance kernels or placement code.

| Runtime | PP8192 (tok/s) | TG256 (tok/s) | Notes |
|---|---:|---:|---|
| R9V Qwen V1 | **1,512.01** (+3,239.98%) | **78.11** (+197.90%) | Qualified R9V reference |
| Stock/public Radiance | 45.27 | 26.22 | Public stack, MTP2, CPU-RAM PLE, exact TP2 R4D all-reduce |

Each percentage is the winning result's advantage over the other runtime in
that benchmark category.

Both PP cells are means of ten forced prefix-cache-miss requests over the same
Aider corpus slices; both TG cells are means of three 278+256 OpenAI requests
with thinking disabled. R9V is 33.40x faster on PP8192 and 2.98x faster on
TG256 in these exact cells. Radiance used its public exact, non-quantized TP2
all-reduce, AITER unified attention, public vLLM UVA expert offload, and a
fully materialized CPU-RAM PLE table. Exact samples, revisions, overlay scope,
and launch policy are in the
[Qwen qualification](docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md) and
[machine-readable comparator record](docs/qualification/results/qwen38-public-radiance-dual-r9700.json).

## Repository layout

```text
profiles/topology/   hardware-first profile indexes
profiles/            exact runnable compositions and qualification reports
packages/models/     model artifacts, hashes, sources, and model licenses
packages/placements/ workload- and topology-specific residency manifests
runtimes/            engine capabilities and pinned source ABI
hardware/            GPU, RAM, PCIe, and rank contracts
kernels/             pinned R9V kernel sources
vendor/              pinned vLLM and GGUF-plugin forks
schemas/             fail-closed descriptor formats
```

See [profiles/README.md](profiles/README.md) for profile composition and
lifecycle rules.

## Licensing and provenance

R9V-owned catalog code and original kernels are Apache-2.0. The vLLM and GGUF
plugin forks retain Apache-2.0; llama.cpp/ggml-derived quant primitives retain
their MIT notice. Model packages remain separate:

- Qwen3.8 artifacts use Qwen Community License 1.0.
- Muse Glimmer artifacts use Apache-2.0 with Meta, Unsloth, and llama.cpp
  provenance recorded.
- Radiance's unlicensed source is not redistributed.

The exact boundaries and remaining release gates are documented in
[licensing.md](docs/licensing.md),
[the provenance audit](docs/provenance-audit.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[release-gates.md](docs/release-gates.md).

## Safety

- Current production kernels target `gfx1201`; unsupported architectures fail
  the profile contract.
- Device ordering and expert placement are semantic for the dual-R9700 Qwen
  profile; other layouts may require a new placement profile.
- Every R4D path remains hard-disabled in R9V. The separately identified
  public Radiance comparator enabled only its exact TP2 BF16 all-reduce; R4D
  attention, GDN, quantized all-reduce, and skinny GEMM were not used.
- Model hashes, runtime revisions, topology, and benchmark protocol are part of
  every performance claim.
