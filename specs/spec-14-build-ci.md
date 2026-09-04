# Spec 14 — Build, Toolchain and CI

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 4, 11, 12, 13. Constrains: spec 15.

## 0. Purpose and scope

The repository layout, pinned toolchains, how the kernel bundle and tune files are built, what GitHub-hosted CI is allowed to claim versus what only the RDNA4 runner can, the runner's isolation, nightly and release procedure, versioning, and platform scope.

## 1. Principles

1. **Hosted CI never claims GPU correctness.** It runs the CPU tiers, compiles everything, regenerates and diffs, and builds docs. Its status name says `cpu-only`.
2. **The runner is the gate.** Golden, invariance, determinism and perf tests on real gfx1201 hardware are required for merge to `main`, and they run only on trusted code.
3. **Everything is pinned and reproducible.** Rust toolchain, ROCm LLVM, Python lockfile, calibration manifests, reference checkpoints by fingerprint. Two builds of the same commit produce the same bundle bytes.
4. **The binary runs without ROCm.** HIP is loaded at runtime; no HIP runtime or zero discovered GPUs selects the full scalar CPU tier. The helper, doctor and config tooling use the same no-GPU path.
5. **Linux x86_64 only in v1.** Stated, not implied.
6. **The CPU binary is architectural-baseline code.** Official x86_64 artifacts are compiled for baseline x86_64, never for the build host. T0 contains no optional SIMD requirement; T0v may use runtime-dispatched features only and must always fall back to T0.

## 2. Repository layout

```
r9v/
  Cargo.toml                 workspace
  rust-toolchain.toml        pinned stable
  toolchain.toml             pinned ROCm / LLVM version and the compile-test matrix
  crates/
    r9v-common    shared error type, ids, hashing and byte-size helpers; CONVENTIONS.md lives at the root
    r9v-ir        spec 1: types, ops, graph, sharding tables, arch descriptor
    r9v-format    spec 2: layouts, schemes, container, repack rules
    r9v-state     spec 3
    r9v-kgen      spec 4: generator, cost model, leaf wrappers per arch
    r9v-registry  spec 4: bundle, tune files, resolution, validation
    r9v-t0        spec 4: CPU reference (scalar) and T0v (SIMD)
    r9v-hip       thin HIP runtime binding, dlopen'd; streams, events, modules, memcpy
    r9v-part      spec 5: partitioner, planner, comms
    r9v-sched     spec 6
    r9v-spec      spec 7: proposers
    r9v-models    spec 8: builder and families
    r9v-loader    spec 9
    r9v-serve     spec 10
    r9v-obs       spec 11: metrics, tracing, doctor, bench
    r9v-config    spec 12
    r9v-helper    spec 12 §7
    r9v           the binary: CLI, ties the crates together
  kernels/
    gen/<arch>/<op>/<hash>.hip   committed generated source (spec 4 §1.6)
    reference/<op>.hip           T1 portable source
    bundle/                      built code objects (release artifact, not committed)
  tune/<arch>/<gen_version>.toml
  bench/baselines/<arch>/<model_fp>.json
  support/<family>/<model>.json  verify-arch outputs
  tools/r9v-quant/               Python, own pyproject and lockfile
  cal/                           calibration manifests
  docs/                          mdBook; config.md and SUPPORT.md are generated
  specs/                         these documents
  xtask/                         cargo xtask commands
```

Crate dependencies point strictly downward in the order listed. `cargo deny` enforces the license allowlist and that no crate above `r9v-hip` links HIP directly.

## 3. Toolchain

- **Rust**: stable, pinned in `rust-toolchain.toml`; bumped by PR. `cargo build --locked` everywhere.
- **ROCm**: the pinned release in `toolchain.toml` (a 7.x line release; the file is the truth). Its `clang++` is the only compiler used for kernels (spec 4 §4.3). The compile-test matrix is the pinned release plus the previous minor.
- **HIP at runtime**: `r9v-hip` `dlopen`s `libamdhip64` on first GPU use and resolves only the entry points the engine needs. No GPU or no ROCm → the CPU device is the only device and the loader says so. Enumeration obtains the complete BDF, including PCI function, from `hipDeviceGetPCIBusId` and rejects disagreement with `hipDeviceProp_t` before constructing a stable identity.
- **Spoof pre-queue contract**: on a spoof-constrained plan (spec 1 App. A), `r9v-ir` supplies the `PreQueueLaunchContract` as data and validation only. Exact-CU hardware (physical CUs equal the profile bound) needs no `ROC_GLOBAL_CU_MASK` assignment; reduced-CU targets use the deterministic lowest-N-bits mask, and the launcher must validate the caller-supplied or process value (`validate_env_value` / `validate_process_env`: absent → `MissingCuMask`, wrong count → `CuMaskMismatch`, unexpected on exact-CU → `UnexpectedCuMask`, malformed → `InvalidCuMask`) before HIP queue creation. Library code never writes the process environment. The mask narrows CU visibility only and enforces no VRAM allocation — the VRAM bound is a planning budget enforced by loader refusal (spec 9 §4.3) — and applying the assignment plus the pre-queue validation belongs to the separate `r9v-hip` integration, which is not yet implemented: this branch performs no hard allocation enforcement and claims none.
- **CPU target**: `.cargo/config.toml` pins official Linux x86_64 Rust code to `target-cpu=x86-64`. The `r9v-t0` build gate rejects optional crate-level x86 features and host-native compilation. Optimized T0v functions use runtime detection and a baseline T0 fallback; release correctness never depends on AVX, AVX2, AVX-512, VNNI, AMX or a particular CPU vendor.
- **Python** (quant tool): 3.11+, `uv` lockfile, torch pinned per accelerator extra (`[cuda]`, `[rocm]`, `[cpu]`).
- **Container**: `ci/Dockerfile` with the pinned ROCm and Rust; hosted CI and the runner both use it, so "works in CI" and "works on the runner" mean the same environment.

## 4. Building the bundle

```
cargo xtask gen   [--arch gfx1201]     regenerate kernels/gen from r9v-kgen and tune/ (fails if dirty in CI)
cargo xtask tune  --arch gfx1201       run autotune on this machine for the static set; writes tune/ (runner only)
cargo xtask bundle --arch gfx1201      compile kernels/gen + kernels/reference into kernels/bundle/<arch>/*.co and manifest
cargo xtask verify-bundle              build twice, compare bytes
```

- The **static set** (which `OpStatic` combinations get shipped) is the union of every op instance in every warm bucket of the reference model set (§6), plus the T1 variant of every op at every bucket. Recorded in `tune/<arch>/static-set.json`.
- Bundle reproducibility is verified in CI by building twice in clean containers.
- `gen_version` bumps invalidate `tune/` and `kernels/gen/` for the affected ops; the PR that bumps it regenerates and re-tunes on the runner before merge.

## 5. CI tiers

### 5.1 Hosted (`cpu-only`)

On every PR and push:

- `cargo fmt --check`, `cargo clippy -D warnings`, `cargo deny`
- portability policy: reject developer-home paths and native-CPU flags in engine/build sources; compile T0 at the baseline target; stub HIP discovery at 0, 1, 2 and 3 devices
- `cargo test --workspace`: T0 and T0v tests; format repack round-trips on synthetic tensors (spec 2 §10); state manager commit/prefix logic against an in-memory pool (spec 3 §8); partitioner as a pure function against golden per-rank graph summaries (spec 5 §4.1); scheduler simulation against a fake `C(S, T_dec, T_pre)` and `C_draft(k)` table (spec 6, since every decision is a function of those tables); config schema round-trip and settings-index consistency (spec 12 §3); loader on tiny synthetic GGUFs; serve routes against a CPU engine running a 30M-parameter test model
- `xtask gen` and diff against `kernels/gen/` (spec 4 §1.6)
- compile T1 and T2 for every arch in `toolchain.toml` (compile only)
- inline-asm grep outside `kgen/src/leaf/` (spec 4 §8)
- docs build; fails if a setting is missing from its spec index or a spec references a nonexistent setting
- `tools/r9v-quant`: unit tests and a determinism check (quantize a 30M synthetic model twice on CPU, compare bytes)

Status check name: `ci/cpu-only`. It is required for merge but explicitly does not gate GPU correctness.

### 5.2 Runner (`gpu/gfx1201`)

A self-hosted runner on the reference machine (two R9700). Runs:

- spec 4 §10 gates for every variant in the static set: golden, batch invariance, determinism, perf regression, achieved bandwidth recorded
- end-to-end L0 golden prompts for the reference model set (spec 8 §8)
- spec 3 §8 tests on real pools; spec 5 §7 cross-rank hash test under TP2 and PP2
- spec 11 §10 baselines for `decode`, `prefill`, `multi`
- measurement-pass freshness: if the hardware fingerprint changed (driver, ROCm, kernel), re-measure and fail with a diff until the implementing agent records and commits the new measurement evidence

Triggers: every push to `main`; PRs from branches in the main repository automatically; PRs from forks only after a maintainer applies the `gpu-approved` label to that specific commit. Results are posted as a PR comment with the receipt table and a link to the bundle.

Status check name: `gpu/gfx1201`. Required for merge to `main`.

### 5.3 Runner isolation

- GPU jobs run in the pinned OCI image under the ordinary Linux container runtime. The container receives only `/dev/kfd` and the required `/dev/dri/renderD*` nodes; it does not receive a privileged container or the host filesystem.
- A syscall-interposition sandbox such as gVisor may be an additional CPU-only/untrusted-source tier, but it is not the GPU correctness environment: the GPU path must exercise the real KFD/DRM ioctl and memory-mapping ABI against the host kernel and driver.
- Dedicated non-privileged user; the runner has no secrets beyond its registration token; no deploy keys, no package registry credentials.
- Network egress limited to GitHub and the package mirrors the container needs; models are pre-staged read-only on local disk by fingerprint, never downloaded by jobs.
- Each job runs in a fresh container with the workspace wiped; GPU access via device passthrough only.
- The runner does not run on `pull_request_target`; fork code executes only after the explicit security-authorization label on that exact commit.

### 5.4 Nightly

On `main`: the full bench suites including `depth` and `accept`, golden prompts for every checkpoint in the support matrix, the tune coverage report (how many graph instances resolve to shipped / local / partial / T1), and a quant-tool run on the smallest reference model with `verify`. Regressions open an issue automatically with the receipt attached.

## 6. Reference model set

Checkpoints pinned by `model_fp`, staged on the runner:

| role | example class |
|---|---|
| small dense | ~4B, for fast gates |
| large dense | ~27–30B, the primary receipt model |
| MoE | a mid-size MoE with host experts exercised |
| hybrid | a gated-delta-net + attention model with MTP |
| draft | a ~1B model paired with the large dense |

Both the standard GGUF (Q4_K_M and Q8_0) and the native file for each are staged, so GGUF-parity and native paths are both gated. A checkpoint whose fingerprint changes on disk fails every job that uses it until `support/` is updated.

## 7. Release

`cargo xtask release <version>`:

1. Requires green `ci/cpu-only` and `gpu/gfx1201` on the commit and a green nightly within 24 h.
   A release advertised as generally supporting gfx1201 also requires a green correctness/installation qualification from a second clean gfx1201 host whose stable hardware/topology fingerprint differs from the reference runner. This second-host gate does not weaken or replace the immutable performance floors, which remain gated on the reference receipt protocol.
2. Builds the bundle for every arch in the support matrix; verifies reproducibility.
3. Produces `r9v-<version>-linux-x86_64.tar.gz` (binary, `kernels/bundle`, `tune/`, `docs/`, generated `SUPPORT.md`), `r9v-quant-<version>.whl`, `SHA256SUMS`, and release notes that embed the reference receipts (spec 11 §9.4) and the version table (§8).
4. Tags; publishes the wheel.

A release never ships a bundle built outside CI.

## 8. Versioning

| number | meaning | bump rule |
|---|---|---|
| engine `MAJOR.MINOR.PATCH` | the binary and API | semver on the HTTP API and config |
| `ir_version` (spec 1) | op set and signatures | minor on additive change; major on signature change |
| `format_version` (spec 2) | native file readability | only on a breaking container change |
| `gen_version` (spec 4) | kernel generator, bundle and tune compatibility | any change to emitted code for an existing variant |
| `config_version` (spec 12) | config file schema | on a rename or removal |

All five appear in `r9v --version`, the load report, the doctor bundle and every receipt.

## 9. Platform scope

- **v1**: Linux x86_64 architectural baseline, glibc; scalar T0 execution on every CPU in that platform scope; ROCm on gfx1201 for the fast path; any gfx9+ for the GPU reference tier. Optional SIMD is an upgrade, never a requirement.
- **Not in v1**: Windows (ROCm on Windows exists for gfx12, but direct I/O, pinned-memory handling and the Unix socket need porting and a second runner), macOS, aarch64. Each is a tracked issue with the list of what would need to change, so a contributor can pick it up without archaeology.

## 10. Developer workflow

```
cargo xtask dev shape-test --op matmul --static '...' --tier t2      # one variant vs T0 on this machine
r9v serve --config dev.toml --warmup.enabled false --kernels.allow_jit true
r9v eval --logits --model ... --prompts ...                          # spec 13 uses this
cargo xtask docs                                                     # local docs incl. generated pages
```

Debug builds enable the spec 2 §10 repack verification and the spec 5 §7 cross-rank hashing automatically; release builds gate both behind `--debug-checks`.
