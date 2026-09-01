# Portability roadmap

Planned work to widen who can run the catalog and what it can host, ordered
roughly by leverage. Nothing here is a release gate unless a profile's own
qualification record says so.

## Near-term items

1. **Startup pinned-memory reduction (top priority).** The loader currently
   pins the full expert set (~27.7 GiB per rank) as host masters, then
   compacts hot experts to VRAM and cold experts to a second pinned buffer,
   and only then drops the masters — a startup peak of roughly 55-75 GiB
   pinned for a steady state that needs about 18 GiB. In the default
   allocation mode the freed masters also stay in the PyTorch
   pinned-allocator cache rather than returning to the OS. Masters do not
   need to be pinned: they exist only to be copied from. Loading expert
   masters pageable (ideally mmap-backed from the package file), compacting
   per layer, and pinning only the cold-owner buffers cuts the startup peak
   to roughly the cold set plus one layer of scratch, and is the change that
   makes 64 GiB hosts realistic. Validate with byte-identical compaction
   output, a measured cgroup peak through startup, and unchanged decode.
   This lands in the vendored plugin/loader, so it carries submodule pin,
   `runtime.json`, and `sources.lock` updates plus an image rebuild.

2. **Prebuilt runtime image.** Publish the dual-R9700 Qwen image to a
   registry, digest-pinned to the exact vLLM/plugin/kernel submodule
   revisions. This turns the slowest install step (a full ROCm source build)
   into a pull, and the digest becomes part of the qualification evidence.
   The runtime doctor already records the container image ID, so
   verification slots in unchanged.

3. **`r9v qualify`.** A measured counterpart to the read-only doctor: run the
   offline PLE random-read benchmark plus a short PP/TG sample against the
   published envelope and write a machine qualification record as JSON. This
   lets a non-reference host demonstrate the published numbers instead of
   trusting them, and is what allows new machines and topologies to be
   qualified with evidence rather than proximity to the reference bench.

4. **Route-corpus collection and manifest generation.** The published
   manifest ships in the model package and is correct on any host; the tool
   that ranked it does not ship. Providing corpus collection and manifest
   generation lets a user with a heavily divergent prompt mix rebuild the
   hot/cold split for their workload, and is a prerequisite for new GPU-count
   placements and for new MoE models.

5. **PLE payload hash.** The derived PLE file is verified by size only. Add
   its sha256 to the package manifest, or distribute the derived file
   directly, so a bad extraction fails at doctor time instead of surfacing as
   unexplained decode latency.

6. **UD-Q4_K_XL quant arm.** A second Unsloth quant at ~4.8 bpw as the
   quality ceiling on the existing dual-R9700 host contract. Host RAM is
   nearly flat across quants (cold experts are the only large pinned
   allocation), so the real work is placement arithmetic: larger expert
   bytes lower the hot counts inside fixed VRAM, which climbs the cold-route
   tail and taxes decode through the slower PCIe rank. Run that arithmetic
   against the published manifest's cold-route curve before committing.
   Shipping it means a K-quant format family in the tiered kernel path, a
   regenerated placement manifest (item 4), and a full qualification arm —
   gated on KLD and the complete quality-domain suite, never a single
   metric, with its own benchmark record.

Smaller: a `r9v support-bundle` command (doctor JSON, versions, log tails)
and a podman docker-shim compatibility pass.

## Topology scaling: one to four R9700s

The catalog already separates single- and dual-GPU runtime families and keys
placement manifests by TP rank. Generalizing to one, three, or four cards is
mostly removing places where "two" is implied:

- Drive the doctor's expected GPU count from the profile's hardware
  descriptor (`gpu_count`) instead of the profile-id prefix. The per-rank
  settings (`R9V_EXPECTED_GPU_BDFS`, link floors, expert maxima) are already
  comma-separated lists and generalize as-is.
- Placement manifests are rank-keyed JSON. Each GPU count needs its own
  generated manifest (item 3 above) and its own VRAM budget arithmetic.
- TP width is constrained by the model's head and expert divisibility. TP3 is
  unlikely to be viable; a three-card host probably wants an expert-parallel
  or offload-tier arm rather than TP.
- A single 32 GiB card cannot hold the Qwen package resident. A single-card
  arm is an offload-heavy configuration — a small static hot set with a
  large cold tier over PCIe — and is a separate qualification, not a
  configuration change.
- More cards mean more resident experts and less cold traffic, so a four-card
  TP arm is mostly placement and qualification work rather than kernel work.

Community-qualified topologies become practical once `r9v qualify` exists:
someone with a 4x R9700 host can submit a qualification record instead of
every topology needing to exist on the reference bench.

## Groundwork for other MoE architectures

The catalog thesis stays: one engine per model, no universal engine. The
groundwork for a DeepSeek-class profile is extracting the machinery that is
already model-agnostic:

- **Descriptor-driven tiered experts.** The expert loader currently validates
  a hardcoded 48-layer, 512-expert shape. Moving layer and expert counts
  (plus shared-expert and routing facts) into the package descriptor lets the
  same static-hot/LRU/cold machinery load any expert-count model.
- **Catalog-driven runtime evidence.** The runtime doctor greps Qwen-specific
  startup markers and environment names. Declaring the expected log markers,
  environment mappings, and metrics in the profile descriptor makes the
  doctor work unchanged for a new model profile.
- **Reusable placement pipeline.** Route statistics to ranked manifest is
  architecture-independent; fine-grained expert designs (more and smaller
  experts, an always-on shared expert) change the numbers, not the pipeline.
- **Per-shape kernel pinning tables.** rocBLAS/Tensile solution selection per
  GEMM shape as a generated, shipped per-model asset.
- **Decoupled speculation machinery.** Keep MTP/draft verification wiring
  separate from the base model so a new profile can reuse it.

What stays per-model: attention kernels, quant layout decisions, PLE (a
Qwen3.8 architecture feature), and every performance claim.

The first step for a new MoE profile is not kernels: write the package and
hardware descriptors and do the placement arithmetic — expert bytes per GPU
count — to determine which R9700 topologies are viable at all.
