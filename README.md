# R9V

[![CI](https://github.com/Dyluhn/R9V/actions/workflows/ci.yml/badge.svg)](https://github.com/Dyluhn/R9V/actions/workflows/ci.yml)

R9V is a catalog of model-specific inference systems for AMD RDNA4. There is
no universal engine here. Each profile pins one model and quant package to a
runtime, a kernel set, a hardware contract, a placement policy, and a
qualification record, and it refuses to run anywhere else.

The approach was inspired by [antirez's DS4](https://github.com/antirez/ds4)
and [Neroued's ninfer](https://github.com/Neroued/ninfer): narrow inference
engines built around an exact model and machine instead of treating every
checkpoint as a generic workload. This is inspiration, not shared code provenance:
R9V is an independent implementation and shares no code with either project.

Two runtime families exist today:

- **Single GPU** — fully custom native R9V engines using HIP, specialized for
  one model, one quant, and the `gfx1201` kernel set.
- **Dual GPU** — an adapted vLLM deployment architecture informed by the
  Radiance profiling workflow, built from a pinned Apache-2.0 vLLM fork, a
  GGUF plugin, and R9V kernels.

Radiance was a development and profiling reference for the dual-GPU work. Its
unlicensed launcher and R4D source are not redistributed here, and every R4D
path is hard-disabled in the R9V profile. The public Radiance comparator used
in the benchmarks below was built separately and enabled only its TP2 BF16
all-reduce.

If you are pointing an AI assistant at this repository, hand it
[LLM_SETUP_GUIDE.md](LLM_SETUP_GUIDE.md). Repository coding conventions, error
types, logging fields, and test layouts are defined in
[CONVENTIONS.md](CONVENTIONS.md).

## Profiles

| Topology | Profile | Runtime | Status |
|---|---|---|---|
| Dual R9700 | `qwen38-flash-next/ud-iq4-xs/dual-r9700-128k` | Adapted vLLM stack | Release candidate |
| Single R9700 | `muse-glimmer-30b/v1/single-r9700` | Custom native HIP proof engine | Experimental |

Status meanings:

- **Release candidate** — qualified on the reference topology, with one release
  gate still open. For Qwen, that gate is a clean-host package installation test;
  the serving path itself is qualified.
- **Experimental** — the artifact and its proof are preserved, but the public
  user path is incomplete. The Muse profile currently ships a frozen raw-token
  proof engine; a curated user runtime is pending.
- **Qualified** — the advertised download-to-run path has passed end to end on
  a clean host. No profile has reached this yet.

## Running the Qwen engine

The dual-R9700 Qwen profile serves OpenAI-compatible text, tool-call, and
single-image requests at `http://127.0.0.1:8004/v1`.
The immutable 90.36 GiB model package is public on Hugging Face.

Host requirements:

- Linux with a working ROCm driver, `amd-smi`, and access to `/dev/kfd` and
  `/dev/dri`
- two 32 GiB Radeon AI PRO R9700 (`gfx1201`) GPUs
- host RAM: qualified on a 128 GiB reference host. Less is untested but not
  rejected. Cold expert allocations are pinned and do not silently spill to
  SSD; less RAM instead reduces startup headroom and the filesystem cache
  available to the SSD-backed PLE table
- Git, Python 3.10+, `curl`, Docker with the Buildx plugin, and the
  Hugging Face `hf` CLI
- roughly 150 GiB of free storage: a 90.36 GiB model package, a 26.82 GiB
  derived PLE file, plus the image build and runtime caches

Everything runs through the `r9v` command, which is plain Python with no
dependencies to install:

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V

./r9v doctor qwen38                     # check the host before downloading anything
./r9v fetch qwen38 --model-dir "$MODEL_DIR" --accept-model-license
./r9v verify qwen38 --model-dir "$MODEL_DIR" -- --hash
R9V_MAX_JOBS=8 ./r9v build qwen38       # full source build; slow the first time
# derive the PLE payload — docs/installation.md, step 4
./r9v run qwen38 --model-dir "$MODEL_DIR"
```

[docs/installation.md](docs/installation.md) walks through each step,
including the PLE extraction between build and run, GPU device ordering (the
order is semantic for this profile), configurable topology/RAM/storage checks,
and how to poll for readiness. The
submodule commits are release inputs — replacing them with branch heads
forfeits the reported reference behavior.

Every portable setting, discovery command, doctor result, and corrective
action is documented in the
[dual-R9700 Qwen configuration reference](profiles/qwen38-flash-next/dual-r9700/README.md).

For hooking the server up to the Pi coding agent, see [docs/pi.md](docs/pi.md).

## Model downloads

| Model package | Download | Contents |
|---|---|---|
| Qwen3.8 Flash Next R9V IQ4_XS | [Hugging Face](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS) | Target GGUF shards, block-FP8 MTP, Q8 vision projector, metadata, placement manifest |
| Muse Glimmer 30B R9V V1 | [Hugging Face](https://huggingface.co/Dyluhn/Muse-Glimmer-30B-R9V-V1) | V1/V12 GGUF, projector, DFlash sidecar, license, provenance |

Both repositories are public, and the R9V package descriptors pin immutable
artifact revisions — `fetch` never follows a moving branch. Each repository
shows the license that applies to its weights; downloading a model does not
mean its runtime has passed every qualification gate.

To browse the catalog without a GPU:

```bash
./r9v list --by-topology
./r9v validate
./r9v show qwen38
./r9v show muse
```

Aliases like `qwen38` and `muse` resolve to exact profile IDs. They do not
bypass a profile's fail-closed status or hardware checks. Topology indexes
live under [`profiles/topology/`](profiles/topology/README.md).

## The Muse profile

The complete V12 research artifact is published as R9V V1, including the
optional projector and DFlash sidecar. `fetch` works today; `build` and `run`
intentionally fail closed until the curated native runtime lands. See the
[clean-clone status](docs/muse-v1.md),
[benchmark record](profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md),
[qualification](profiles/muse-glimmer-30b/v1-r9700/QUALIFICATION.md), and
[model card](packages/models/muse-glimmer-30b/v1-v12/README.md).

## Kernel scope

R9V kernels are narrow by design, not generic ROCm replacements. Their host
APIs reject incompatible dtypes, shapes, qtypes, layouts, and GPU targets;
supported fallbacks stay owned by the selected runtime.

The Qwen profile ships three production `gfx1201` kernel families:

- `dense_mmvq_hip` — MTP multi-row GGUF GEMV reuse, exact Q8 attention M=3,
  and the HyperConnection down/up paths
- `tiered_iq_moe_hip` — mixed VRAM/UVA expert GEMV, graph-safe LRU caching,
  and the bit-exact group-16 MoE prefill path
- `fused_gdn_mtp_hip` — the TP2 speculative GDN core

These are qualified only as part of their profile. Porting them to another
model or GPU means redoing parity, graph-replay, placement, and end-to-end
qualification. Details in the
[kernel scope and safety contract](kernels/r9v-gfx1201/README.md).

## Benchmarks

Numbers apply to the named profile, reference hardware, and recorded protocol
only — they are not blanket claims for every R9700.

### Qwen3.8 Flash Next — R9V versus stock/public Radiance

Same dual-R9700 topology, same benchmark cells. The comparator was built from
public Radiance, public vLLM PR #53899, and the public GGUF plugin; its
compatibility overlay only made Qwen4Exp GGUF, MTP, PLE, and UVA loading work
and contained no R9V performance kernels or placement code.

| Runtime | PP8192 (tok/s) | TG256 (tok/s) |
|---|---:|---:|
| R9V Qwen V1 | **1,512.01** (+3,239.98%) | **78.11** (+197.90%) |
| Stock/public Radiance | 45.27 | 26.22 |

That is 33.4x on prefill and 2.98x on decode in these cells.

The reference host is asymmetric: rank 0 sits on PCIe Gen5 x16, rank 1 on
Gen4 x4, so all TP2 all-reduce and UVA expert traffic for rank 1 crosses a
Gen4 x4 link. Treat these numbers as a floor for better-connected hosts — a
symmetric Gen5 x8/x8 layout is expected to improve on them significantly,
though that has not been measured yet.

PP cells are
means of ten forced prefix-cache-miss requests over the same Aider corpus
slices; TG cells are means of three 278+256 OpenAI requests with thinking
disabled. Radiance ran with its public non-quantized TP2 all-reduce, AITER
unified attention, public vLLM UVA expert offload, and a fully materialized
CPU-RAM PLE table. Full protocol and revisions are in the
[Qwen qualification](docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md) and
the [machine-readable comparator record](docs/qualification/results/qwen38-public-radiance-dual-r9700.json).

### Muse Glimmer 30B — internal V12, public R9V V1

The exact 24,554,611,392-byte V12/V1 GGUF, measured on one isolated Radeon AI
PRO R9700. Values are means of three samples after one warmup. Both llama.cpp
baselines use revision `dd1ea524333b1e697489067d7a4c39c60d32beee` and the same
model bytes. Percentages are R9V's margin over the faster llama.cpp backend in
that column.

| Runtime | PP512 (tok/s) | PP2048 (tok/s) | PP8192 (tok/s) | TG256 (tok/s) |
|---|---:|---:|---:|---:|
| R9V custom HIP | **1,500.68** (+1.54%) | **2,175.17** (+47.21%) | **2,078.20** (+46.36%) | **26.84** (+7.70%) |
| llama.cpp ROCm | 1,477.87 | 1,477.57 | 1,419.88 | 24.30 |
| llama.cpp Vulkan | 1,204.85 | 1,182.54 | 1,126.46 | 24.92 |

The TG256 column is plain autoregressive decode. Both engines also support
DFlash2 speculative decoding with the package's sidecar; measured separately
(greedy, 128 new tokens, means of three, same GPU and model bytes):

| Prompt depth | llama.cpp raw | llama.cpp +DFlash2 | R9V raw | R9V +DFlash2 |
|---|---:|---:|---:|---:|
| 136 (repetitive) | 24.10 | 232.65 | 26.05 | 110.69 |
| 8192 (corpus) | 23.55 | 40.55 | 24.39 | 59.65 |

The repetitive short prompt saturates draft acceptance on both engines, so
those cells are best-case ceilings. The 8K corpus cell is the representative
one: R9V accepts 61% of drafts against llama.cpp's 12%, decoding 1.47x
faster assisted. Protocol, acceptance counts, and caveats are in the
[Muse benchmark record](profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md).

This is a speed result from a frozen proof engine, not a quality endorsement.
The V1/V12 quant records mean KLD 0.006121 versus 0.003071 for Unsloth
UD-Q5_K_XL and 0.001034 for UD-Q6_K_XL on the same 122,400-position evaluator,
so it is not quality-competitive with the Q5/Q6 quants. Exact samples, binary
hashes, and caveats are in the
[Muse benchmark record](profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md).

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

## Development and CI

Every push and pull request runs the dependency-light CPU suite on Python 3.10
and 3.14. A separate static job checks Ruff, shell syntax and ShellCheck,
JSON descriptors, the profile graph, and exact submodule gitlinks. It does not
pretend to qualify GPU performance: model downloads, ROCm compilation, kernel
parity, graph replay, and benchmark reproduction remain explicit qualification
gates on a matching R9700 host.

Run the same checks locally with:

```bash
python -m pip install --requirement requirements-ci.txt
python -m pytest -q tests
./scripts/ci-static.sh
```

Planned portability work — prebuilt images, measured qualification, wider
R9700 topologies, and groundwork for other MoE architectures — is tracked in
[docs/portability-roadmap.md](docs/portability-roadmap.md).

## Licensing and provenance

R9V-owned catalog code and original kernels are Apache-2.0. The vLLM and GGUF
plugin forks stay Apache-2.0; llama.cpp/ggml-derived quant primitives keep
their MIT notice. Model packages are licensed separately:

- Qwen3.8 artifacts: Qwen Community License 1.0
- Muse Glimmer artifacts: Apache-2.0, with Meta, Unsloth, and llama.cpp
  provenance recorded
- Radiance's unlicensed source is not redistributed

Boundaries and remaining release gates are documented in
[licensing.md](docs/licensing.md),
[the provenance audit](docs/provenance-audit.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[release-gates.md](docs/release-gates.md).

## Caveats

- Production kernels target `gfx1201` only; other architectures fail the
  profile contract.
- Device ordering and expert placement are semantic for the dual-R9700 Qwen
  profile. Other layouts need a new placement profile.
- Dual-GPU benchmarks were taken on an asymmetric PCIe layout (Gen5 x16 +
  Gen4 x4); hosts with more interconnect bandwidth should do better.
- The 128 GiB host RAM figure is the qualified reference machine, not an
  enforced minimum. Smaller hosts are untested.
- The published expert placement was ranked from a multiprompt route corpus.
  It is correct for any workload; a heavily divergent prompt mix may see more
  cold-expert traffic and lower decode throughput than the published figure.
- Model hashes, runtime revisions, topology, and benchmark protocol are part
  of every performance claim here.

## Field position (scanned 2026-08-30)

A comparison of R9V against the model-specific narrow-engine field — the
cluster defined by [antirez/ds4](https://github.com/antirez/ds4) (~22k stars,
DeepSeek V4 on Metal/CUDA/ROCm) and
[Neroued/ninfer](https://github.com/Neroued/ninfer) (pinned Qwen checkpoints
on a single RTX 5090). R9V is currently the only public member of this
family targeting AMD RDNA4.

Where R9V matches or leads its peers: fail-closed exact-artifact catalogs,
hash-verified model downloads, speculative decoding in the shipped runtime,
and benchmark provenance (hashes, protocols, and raw samples attached to
every claim).

Where the peers are ahead, in priority order:

1. **Concurrency** — ds4 batches up to 16 concurrent sessions and ninfer
   publishes concurrent-decode numbers; the R9V Qwen profile currently
   qualifies one sequence at a time.
2. **Task-quality evals** — both peers publish GPQA/AIME-class scores; R9V
   publishes distributional quality (KLD) only.
3. **Anthropic Messages API** — both peers serve `/v1/messages` alongside
   OpenAI; R9V is OpenAI-only.

On the same silicon, a handful of small public repos deploy general stacks
on dual-R9700/gfx1201 (SGLang and vLLM recipes). None carry qualification
records; R9V's benchmarks are the citable datapoints for this topology.

The full rubric and scan record live in [`.sota/`](.sota/); the scan was
produced with the [sota-scan](https://github.com/MerlijnW70/sota-scan)
methodology plus a [DeepGit](https://github.com/zamalali/DeepGit)-style
discovery pass.
