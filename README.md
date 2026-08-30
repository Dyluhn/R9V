# R9V

R9V is a collection of exact, model-specific inference systems for AMD RDNA4.
It is not one universal engine. Each profile freezes a model/quant package,
runtime, kernel set, hardware contract, placement policy, and qualification
record so the complete setup can be reproduced instead of reconstructed from
benchmark fragments.

The project currently has two distinct runtime families:

- **Single GPU:** fully custom native R9V engines. The current profile targets
  the exact Muse Glimmer V1 quant and `gfx1201` kernel set.
- **Dual GPU:** an adapted vLLM deployment architecture informed by the
  Radiance profiling workflow, using an Apache-2.0 vLLM fork, the GGUF plugin,
  and R9V kernels. The release tree does not redistribute the unlicensed
  Radiance launcher or R4D source.

## Current status

| Topology | Supported model/profile | Engine | Public status | User surface |
|---|---|---|---|---|
| Single R9700 | `muse-glimmer-30b/v1/single-r9700` | Custom native HIP engine | Experimental canonical V1 | Frozen raw-token proof; curated user runtime pending |
| Dual R9700 | `qwen38-flash-next/ud-iq4-xs/dual-r9700-128k` | Adapted vLLM stack; Radiance-informed workflow | Release candidate | OpenAI-compatible text, tools, and vision |

The Qwen runtime works end-to-end on the reference machine, but its public
model revision and clean-checkout release benchmark are still pending. The
Muse has frozen model and benchmark identities, but the curated
source-complete user runtime has not yet been published. Those distinctions
are deliberate: R9V does not label a local proof as a downloadable release.

## Browse the catalog

```bash
git clone --recursive https://github.com/Dyluhn/R9V.git
cd R9V

./r9v list --by-topology
./r9v validate
./r9v show qwen38
./r9v show muse
./r9v doctor qwen38
```

Aliases such as `qwen38` and `muse` work with every command. `muse` resolves
to the canonical V1/V12 profile.

## Running a profile

### Dual R9700: Qwen3.8 Flash Next

This is the only current profile with a complete serving path. It requires two
32 GiB R9700 cards, 128 GiB host RAM, a fast SSD for the PLE table, and the
exact packaged target/MTP/projector/placement files.

The package upload is not public yet, so a fresh user cannot complete the
download step today. `./r9v fetch qwen38` intentionally fails closed until the
descriptor contains an immutable revision. Once that revision is published,
the supported path is:

```bash
export MODEL_DIR=/path/to/qwen38-r9v
export R9V_DATA_DIR=/fast-ssd/r9v
mkdir -p "$R9V_DATA_DIR"

R9V_MAX_JOBS=8 ./r9v build qwen38

./r9v fetch qwen38 \
  --model-dir "$MODEL_DIR" \
  --accept-model-license

docker run --rm --network none --entrypoint python3 \
  --user "$(id -u):$(id -g)" \
  --security-opt label=disable \
  --volume "$PWD:/r9v:ro" \
  --volume "$MODEL_DIR:/models:ro" \
  --volume "$R9V_DATA_DIR:/r9v-data" \
  r9v-qwen38-flash-next:latest \
  /r9v/tools/prepare_ple.py \
  /models/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  /models/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf \
  /models/target/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf \
  --output /r9v-data/per_layer_token_embd.iq4_nl.bin

export R9V_PLE_PATH="$R9V_DATA_DIR/per_layer_token_embd.iq4_nl.bin"
export R9V_CACHE_DIR="$R9V_DATA_DIR/cache"
./r9v run qwen38 --model-dir "$MODEL_DIR"
```

The server listens at `http://127.0.0.1:8004/v1`. See
[installation.md](docs/installation.md) for package layout, device ordering,
and health checks, and [pi.md](docs/pi.md) for Pi coding-agent and per-call
PP/TG integration.

### Single R9700: Muse Glimmer

The single-card path is a completely custom R9V engine rather than vLLM. The
Muse profile currently fails closed at `build` and `run` because publishing the
legacy research workspace would not be reproducible or license-clean.

What users can verify now:

```bash
./r9v show muse
./r9v doctor muse
```

The release gate is a curated native runtime that rebuilds the accepted
executable and code object, then adds tokenizer-aware text generation. Vision,
DFlash, and an OpenAI-compatible API are not part of the current Muse V1
claim.

## Benchmarks

### Muse Glimmer 30B R9V V1

The exact V1/V12 GGUF was measured on one isolated R9700 against llama.cpp at
the same source revision. Values are means of three measured samples after one
warmup.

| Cell | R9V custom HIP | llama.cpp ROCm | llama.cpp Vulkan |
|---|---:|---:|---:|
| PP512 | 1,500.68 | 1,477.87 | 1,204.85 |
| PP2048 | 2,175.17 | 1,477.57 | 1,182.54 |
| PP8192 | 2,078.20 | 1,419.88 | 1,126.46 |
| TG256 | 26.84 | 24.30 | 24.92 |

This is a speed result, not a quality endorsement. V1 is a rough draft and is
substantially worse than Unsloth Q5/Q6 in the recorded quality evaluation. See
the full [benchmark protocol](profiles/muse-glimmer-30b/v1-r9700/BENCHMARKS.md)
and [model card](packages/models/muse-glimmer-30b/v1-v12/README.md).

### Qwen3.8 Flash Next

The final release benchmark will be run last, after the model revision,
submodule commits, and launch contract are frozen. Development qualification
on the reference dual-R9700 machine measured 77.82–79.27 TG from 8K through
120K context, but those numbers are not being presented as the public release
result. The provisional record is retained in
[the development qualification](docs/qualification/qwen38-ud-iq4-xs-dual-r9700.md).

## Repository layout

Human navigation is topology-first:

```text
profiles/topology/
  single-r9700/       # Muse custom-engine profiles
  dual-r9700/         # Qwen adapted-vLLM profiles
```

The implementation remains normalized so model files and runtimes are not
copied between profiles:

```text
packages/models/      exact artifacts, hashes, sources, model licenses
packages/placements/  hardware/workload-specific residency manifests
runtimes/              engine capabilities and pinned source ABI
hardware/              GPU count, architecture, RAM, PCIe and rank contract
profiles/              runnable compositions and qualification reports
kernels/               pinned R9V kernel source
vendor/                pinned vLLM and GGUF-plugin forks
schemas/               fail-closed descriptor formats
```

Use [profiles/README.md](profiles/README.md) when adding another exact quant or
hardware topology.

## Licensing

R9V-owned catalog code and original kernels are Apache-2.0. The vLLM and GGUF
plugin forks retain Apache-2.0; llama.cpp/ggml-derived quant primitives retain
their MIT notice. Model packages remain separate:

- Qwen3.8 artifacts use Qwen Community License 1.0.
- Muse Glimmer artifacts use Apache-2.0 with Meta, Unsloth, and llama.cpp
  provenance recorded.
- Radiance's unlicensed source is not redistributed.

The exact boundaries and remaining public-tag gates are documented in
[licensing.md](docs/licensing.md),
[the provenance audit](docs/provenance-audit.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[release-gates.md](docs/release-gates.md).

## Safety and scope

- Current kernels are specialized for `gfx1201`; unsupported architectures
  fail the profile contract.
- Qwen device ordering and expert placement are semantic. Other PCIe/RAM
  layouts may require regenerated placement settings.
- Every R4D path remains hard-disabled. R4D was not used for the published
  development results.
- Benchmark numbers describe exact profiles and reference machines, not all
  R9700 systems.
