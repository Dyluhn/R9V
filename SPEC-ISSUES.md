# Specification Issues

Issues and ambiguities encountered during implementation that require spec clarification (Spec 15 §1, §9; r9v-card-work §5).

<!--
Format for entries:

## SI-<n> — <card id> — spec <n> §<x>
What: <the sentence or gap, quoted or precisely located>
Why it blocks or misleads: <one paragraph>
Option taken: <what you did, or "stopped">
Proposed resolution: <the spec edit you'd make, in one or two sentences>
-->

## SI-1 — A0.S3 — spec 4 §4.3
What: The pinned compilation command specifies `-ffast-math=off`.
Why it blocks or misleads: The pinned ROCm Clang 23 rejects that spelling as an unknown argument, so the specified command cannot compile any kernel even though the intended requirement—keeping fast-math disabled—is supported.
Option taken: Used `-fno-fast-math`, the Clang spelling that enforces the stated intent, for the A0.S3 probe; the FP8 builtin compiled, passed its numerical checks, and lowered to the intended instruction.
Proposed resolution: Replace `-ffast-math=off` with `-fno-fast-math` in spec 4 §4.3 while retaining `-fno-gpu-approx-transcendentals`.

## SI-2 — A0.S6 — spec 5 §2 / hardware topology
What: Spec 5 §2 says the current rig has one R9700 on x16 and one on x4, and `hardware/dual-r9700/hardware.json` records rank 1 as `Gen4 x4`.
Why it blocks or misleads: The A0.S6 reference-rig measurement resolves the two discrete R9700 endpoints as PCI `0000:03:00.0` and `0000:13:00.0`; sysfs reports `32.0 GT/s PCIe`, width `16` for both endpoints, and they occupy separate IOMMU groups 15 and 31. Keeping the stale x4 description would make the topology fingerprint and every calibrated communication-cost estimate disagree with the measured machine.
Option taken: Recorded the live topology and P2P receipt in `spikes/p2p/RESULT.md`; A0.S6 uses the measured link and selects `Direct`. Work that consumes the seeded x4 topology must use the measured result rather than treating the JSON value as observed fact.
Proposed resolution: Update spec 5 §2 and `hardware/dual-r9700/hardware.json` to record both discrete endpoints at their measured Gen5 x16 link state on the current reference rig, and populate the 0↔1 transport as `Direct` from the A0.S6 receipt.

## SI-3 — A1.1 — spec 1 §2.3, App. A / spec 2 §2 / spec 4 §13
What: Spec 1 requires `Tensor.layout`, `ArchDescriptor.fragment_layout`, and `ArchDescriptor.attention_layout` to carry `LayoutId`, while spec 2 defines only the names `L0`, `L1`, and `L1S`, card A2.1 assigns their Rust type to downstream crate `r9v-format`, and spec 4 describes a distinct gfx1201 attention order without assigning it an id.
Why it blocks or misleads: `r9v-format` depends on `r9v-ir` in the mandated downward dependency graph, so the IR cannot name a `LayoutId` owned by `r9v-format` without a cycle. Reusing `L1` for the described K/V cache order would falsely identify a weight permutation as an attention-state layout, while leaving the field unset would violate the non-optional ArchDescriptor surface.
Option taken: Defined an opaque `LayoutId` code newtype in `r9v-ir` with provisional constants for `Contiguous`, `L0`, `L1`, `L1S`, and the spec 4 §5.3 gfx1201 attention order. Card A2.1 must make `r9v-format` map or re-export these identities instead of declaring a second incompatible type.
Proposed resolution: Name `r9v-ir` as the canonical owner of layout identity codes, assign stable values (including the gfx1201 attention layout), and amend card A2.1 to define layout semantics and format mappings over that shared identity type.

## SI-4 — A1.1 — spec 1 §2.2
What: The `QuantScheme` pseudocode writes `PerRow { scale: f16 }`, `PerToken { scale: f32 }`, and `PerBlock32 { scale: f32 }` but does not define whether `scale` is a scalar enum payload, a tensor handle, or a declaration of the associated scale tensor's dtype.
Why it blocks or misleads: Each scheme requires an array of scales, not one scalar: spec 2 stores PerRow records per output row, `quant_act` emits `scale [T] f32`, and PerBlock32 requires one value per token per K block. Embedding one `f16` or `f32` in the enum cannot represent those shapes, while the actual scale arrays already travel in format records or op tensors.
Option taken: Represented these as closed marker variants and exposed `scale_dtype()`; actual scale arrays remain separate data owned by the format or op signature. `Scheme(SchemeId)` retains its id payload because that value selects structure rather than storing tensor data.
Proposed resolution: Clarify that the braces in §2.2 declare associated scale-record element types, or replace each `scale` field with an explicitly shaped tensor/reference type if the IR is intended to carry scale data directly.

## SI-5 — A1.1 — spec 1 §2.3 / spec 2 §5
What: Spec 1 permits `Host` and `Tiered` only for `Weight` tensors whose semantic role is expert weight, n-gram table, or embedding, but the specified `Tensor` fields contain only the broad five-way `Class` and no weight-role identity.
Why it blocks or misleads: `Tensor::new` can enforce the `Weight` half of the rule but cannot distinguish an allowed expert/table/embedding weight from a forbidden dense weight without adding a field outside the specified surface. Pretending the broad class check is the whole rule would allow an illegal placement to reach execution.
Option taken: `Tensor::new` rejects non-Weight host/tiered tensors; card A2.6 loader binding and the placement planner must enforce the semantic-role half where tensor names/model roles are available.
Proposed resolution: State that role-specific placement legality is a loader/planner validation over model weight bindings, or add a closed weight-role field to the IR Tensor surface.

## SI-6 — A1.1 — spec 1 §2.1–§2.2 / §4.A `ngram_gather`
What: The core-type notes call `i4`, `PerRow`, and `Scheme` weights-only, while the closed `ngram_gather` signature consumes `gather_staging [T, Np, Dn] (i4|i8, Block)`, which is a `Staging` tensor holding copied quantized table rows.
Why it blocks or misleads: Enforcing “weights-only” as `Tensor.class == Weight` makes the specified n-gram op impossible to construct, even though its staging bytes remain quantized weight-origin data with their scale records. Treating the staging tensor as unquantized would misdescribe its bytes and break reference dequantization.
Option taken: Allowed weight-side quant schemes and `i4` on `Class::Staging` as well as `Class::Weight`; `PerToken` and `PerBlock32` remain activation-only. The exception is limited to the class explicitly used by `gather_staging`.
Proposed resolution: Clarify in §2.1–§2.2 that “weights-only” includes quantized weight-origin rows in `Staging` tensors, or give `gather_staging` a distinct closed scheme/class representation.

## SI-7 — A1.2 — spec 1 §4.A `quant_act`
What: Spec 1 §4.A specifies `quant_act` as `x [T, N] (f16|bf16|f32) -> xq [T, N] (i8|e4m3) PerToken, scale [T] f32` with attrs `target: i8|e4m3, smoothing: None|Folded`, omitting `PerBlock32` and rank-2 scale even though Spec 2 §3.4 and Card A1.5 require `i8 PerBlock32` with scale `[T, N/32] f32`.
Why it blocks or misleads: Standard GGUF models without `r9v.*` metadata require `i8 PerBlock32` activation quantization for llama.cpp MMQ parity. Strictly enforcing Spec 1 §4.A's closed signature prevents the IR from expressing block-scaled activation quantization.
Option taken: Added a `scheme: QuantScheme` attribute to `QuantActOp` supporting `PerToken` and `PerBlock32`. Validated that `PerBlock32` requires target `i8`, `N` divisible by 32, and scale rank 2 `[T, N/32] f32`, while rejecting `e4m3 PerBlock32`.
Proposed resolution: Update Spec 1 §4.A to add `scheme: PerToken | PerBlock32` to `attrs` and document the conditional `scale` shape (`[T]` for `PerToken`, `[T, N/32]` for `PerBlock32`).

## SI-8 — A1.2 — spec 1 §4.A `ngram_gather`
What: Spec 1 §4.A specifies `ngram_gather` consuming `gather_staging [T, Np, Dn] (i4|i8, Block), row_scales -> x [T, Np·Dn] act_dtype`, notes a `Device` table mode where hashing occurs on device and `gather_staging` is unused, but provides no signature for device-table mode.
Why it blocks or misleads: Without a signature for the device-table case, the op cannot validate its inputs when gathering directly from a device table, and ignoring input tensors leads to silent signature mismatch between graph and execution.
Option taken: Added a closed `source: NgramSource` attribute (`Staged` vs `Device`). Validated an exact two-tensor signature for each mode: `(gather_staging, row_scales)` for staged mode, and `(token_ids [T] u32, table [TotalEntries, Dn])` with `Device` placement and `Weight` class for device-table mode.
Proposed resolution: Explicitly document the two input tensor signatures for staged mode and device-table mode in Spec 1 §4.A.

## SI-9 — A1.2 — spec 1 §4.C `matmul`
What: Spec 1 §4.C lists `epilogue: None|Bias|Residual|Act(act)` and includes `bias? [N] f32` in the input signature, but omits the `residual` activation tensor `[M, N]` from the signature.
Why it blocks or misleads: Spec 1 §3.4 and §5.2 explicitly permit matmul residual epilogues and partial sum flow through them. Without a formal input tensor for the residual operand, the step graph cannot represent or validate fused matmul-residual nodes.
Option taken: Validated input tensors conditionally based on `epilogue`: `None` and `Act` require exactly 2 inputs `(x, w)`, `Bias` requires 3 inputs `(x, w, bias [N] f32)`, and `Residual` requires 3 inputs `(x, w, residual [M, N])` with matching activation dtype.
Proposed resolution: Update Spec 1 §4.C to list `epilogue_in?` in inputs, specifying `bias [N] f32` for `Bias` and `residual [M, N] act_dtype` for `Residual`.

## SI-10 — A1.2 — spec 1 §4.A `gather_rows` and `scatter_add_rows`
What: Spec 1 §4.A names `gather_rows` and `scatter_add_rows` as ops exposed for reference-tier MoE and tree-verify bookkeeping, but omits their input/output tensor signatures, ranks, and dtypes.
Why it blocks or misleads: Without declared tensor signatures, graph construction, op validation, and sharding tables for these ops cannot be verified against the specification.
Option taken: Defined and validated explicit signatures: `gather_rows(x [N, D], indices [M] u32) -> y [M, D]` and `scatter_add_rows(x [M, D], indices [M] u32, dest? [N, D]) -> y [N, D]`.
Proposed resolution: Add explicit signature blocks and shape/dtype constraints for `gather_rows` and `scatter_add_rows` in Spec 1 §4.A.

## SI-11 — A1.2 — spec 1 §4.G collective operations
What: Spec 1 §4.G lists collective ops `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all(counts)`, `send(peer)`, `recv(peer)`, `barrier` and groups attributes `group: GroupId`, `dtype`, `reduce_in: f32` at the section level, but omits explicit per-op argument mappings.
Why it blocks or misleads: The absence of per-op attribute definitions leaves `send` and `recv` without an explicit communicator `group`, `reduce_scatter` and `all_gather` without a partition axis, `recv` without an expected shape, and `all_to_all` without an explicit counts tensor signature.
Option taken: Added `group: GroupId` to `SendOp` and `RecvOp`, `shape: Box<[Dim]>` to `RecvOp`, `axis: u32` to `AllGatherOp` and `ReduceScatterOp`, `reduce_in == DType::F32` validation to `AllReduceOp` and `ReduceScatterOp`, and validated a two-tensor input signature `(x, counts [P] u32)` for `AllToAllOp`.
Proposed resolution: Provide individual op signatures and attribute listings for each collective operation in Spec 1 §4.G.

## SI-12 — A1.2 — spec 1 §3.1, §3.2, §4.D, §4.F non-Tensor step graph inputs
What: Spec 1 §3.1 defines a graph as a DAG of op instances over tensors, but §3.2, §4.D, and §4.F describe `BatchMeta`, `SamplingParams`, `rng_state`, and `TreeMask` as graph inputs or op arguments without defining them as Tensors.
Why it blocks or misleads: Modeling non-tensor execution metadata as fake `Tensor` instances compromises tensor invariant validation, while omitting them prevents step graphs from capturing complete runtime inputs.
Option taken: Kept `rng_state`, `BatchMeta` (including its optional `TreeMask`), and `SamplingParams` as typed non-Tensor external values because `DType` is closed. Op-level `validate(&self, inputs, outputs)` validates only the tensor portions of signatures, while structured non-tensor metadata and parameters are validated via dedicated typed validation methods (such as `SamplingParams::validate()`).
Proposed resolution: Clarify in Spec 1 §3.1 and §3.2 that step graphs capture structured non-tensor external inputs alongside tensor edges, and specify that op tensor validation applies strictly to tensor portions.

## SI-13 — A1.2 — spec 1 §3.4 `logits_postprocess -> sample` fusion
What: Spec 1 §3.4 lists `logits_postprocess -> sample` as a permitted fusion pattern, but `logits_postprocess` emits `probs [S, q, V] f32` (rank 3) while `sample` consumes `probs [S, V] f32` (rank 2).
Why it blocks or misleads: During decode `q = 1` the query dimension is degenerate, but prefill and speculative verify steps have `q > 1`, creating a structural rank contradiction across the fused edge.
Option taken: Modeled the fusion pattern for decode-class steps where `q = 1` allows squeezing the degenerate dimension into `[S, V]`, and documented that multi-token sampling requires per-token dispatch.
Proposed resolution: Clarify in Spec 1 §3.4 that `logits_postprocess -> sample` fusion applies specifically to decode-class sequences with `q = 1`.

## SI-14 — A2.1 — spec 2 §2.3 / spec 4 §5.1, §8
What: Spec 2 §2.3 defines the `L1S` index region as "per tile, 2 bits per kept element in the lane order SWMMAC expects (spec 4 fixes the exact operand order)", but spec 4 names `swmmac_*` only as a leaf wrapper (§8) and as an inner-loop note (§5.1: "`L1S`: same with `swmmac_*` in the inner loop and the index region streamed alongside") without fixing any index-operand lane order. The referenced order does not exist in the spec.
Why it blocks or misleads: Card A2.1 must emit an index-region byte layout that kernels (A5.4) and the loader (A2.8) will share, but there is no specified order to implement; A0.S1 verified only the dense `L1` lane formula, not any sparse operand order.
Option taken: `L1S` index bytes reuse the A0.S1-verified §2.2 lane formula over the compressed-K tile (lane = kgroup*16+n, eight kept slots and two little-endian index bytes per lane, slot 0 in the lowest 2 bits), recorded as DECISION(A2.1) in `crates/r9v-format/src/sparse.rs`.
Proposed resolution: State the `L1S` index lane order explicitly in spec 2 §2.3 (or point to the verified §2.2 order), removing the forward reference to spec 4.

## SI-15 — A2.1 — spec 2 §2.3 / §8
What: Spec 2 §2.3 requires the `L1S` index region to store 2 bits per kept element. Under 2:4 sparsity that is 4 index bits per four dense weights, or 1.0 index bit per dense weight. The §8 size table instead states `any s24 | ×0.5 + 0.5 bpw indices`, budgeting only 2 index bits per four dense weights.
Why it blocks or misleads: The two rules imply different index-region lengths, tensor offsets, checksums, and whole-model size estimates. A 16×16 compressed-K value tile contains 256 kept values, so §2.3 requires 64 index bytes; the §8 shorthand would permit only 32 bytes and cannot encode a 2-bit position for every kept value.
Option taken: Followed the byte-defining rule in §2.3: 256 kept values × 2 bits = 64 index bytes per compressed-K `L1` tile. Size accounting must therefore use `dense_bpw × 0.5 + 1.0 bpw indices` until the specification resolves the contradiction.
Proposed resolution: Change the §8 shorthand to `×0.5 + 1.0 bpw indices`, or explicitly redefine the §2.3 index encoding to a 2-bit-per-four-dense-values scheme and provide its legal pattern table and SWMMAC operand mapping.

## SI-16 — A1.3 — spec 8 §3
What: Spec 8 §3 defines `NgramSpec { orders, heads, table_sizes, hash, combine, inject_at: layer }` with no row dimension, while Spec 1 §4.A `ngram_gather` requires `Dn` (`gather_staging [T, Np, Dn]` → `x [T, Np·Dn]`) to shape the device table and the projection.
Why it blocks or misleads: Without `Dn` in the model definition, the builder cannot declare the `[TotalEntries, Dn]` table shape or the `heads·Dn`/`Dn` projection width from the spec; any value it picks (including the previously hardcoded 32) is unverifiable and silently fixes a model property the checkpoint family should own.
Option taken: Added an explicit `dim: u32` (Dn) to `NgramSpec`, validated nonzero and bounded, with `orders.len` and `table_sizes.len` each required to equal `heads`; see DECISION(A1.3) at the struct.
Proposed resolution: Add `dim: u32` (Dn) to the Spec 8 §3 `NgramSpec` surface and state that `orders.len == table_sizes.len == heads`.

## SI-20 — A2.2 — spec 2 §3.2
What: The `E4M3_B128` row reads "`E4M3_B128` | e4m3 | 128 | `s: f16` | `w = s·q` | 8.125 | fp8 WMMA" but no section defines the `e4m3` bit encoding (sign/exponent/mantissa widths, bias, NaN patterns, subnormal values, or rounding on encode).
Why it blocks or misleads: The decode formula needs a concrete value for `q`, and kernels (A5.1/A5.4) plus the quant tool (A6.5) must agree with the format on every bit pattern; any two readers that guess differently (bias 7 vs 8, NaN 0x7F/0xFF vs wider) silently disagree on half the grid, and the disagreement only shows up as wrong logits.
Option taken: Implemented OCP E4M3 (1 sign, 4 exponent with bias 7, 3 mantissa; NaN only `0x7F`/`0xFF`; exponent-15 remainder are extended normals at exponent 8, max 448.0; min normal 2^-6, subnormals 2^-9..2^-6; encode round-to-nearest with ties to even, finite overflow saturating to ±448.0, non-finite inputs rejected), cross-checked bit-for-bit against ml_dtypes `float8_e4m3fn` on grid boundaries; see DECISION(A2.2) at `E4m3::from_f32` and SI-20 references in `crates/r9v-format/src/scales.rs`.
Proposed resolution: State the OCP E4M3 bit layout, NaN patterns, subnormal values, and encode rounding/saturation explicitly in spec 2 §3.2.

## SI-24 — A2.2 — spec 2 §3.2
What: Spec 2 §3.2 and DECISIONS.md D-004 describe `I4_K` as having a "12 B packed" header (`d: f16, dmin: f16, sc[8]: u6, mn[8]: u6`), conflating the 12-byte packed sub-block scale/minimum payload with the entire scale record.
Why it blocks or misleads: Reading "12 B packed header" literally would allocate 12 bytes per 256-block scale record, yielding (12×8 + 128×8)/256 = 4.375 bpw, which conflicts with Spec 2 §3.2 and §8 specifying 4.5 bpw. Furthermore, D-004 specifies that `I4_K` is field-identical to GGUF `Q4_K`, which stores two 2-byte `f16` super-scales (`d` and `dmin`, 4 bytes total) followed by 12 packed bytes for the eight 6-bit `sc` and `mn` entries, totaling 16 bytes per scale record (16 + 128 = 144 bytes per 256 weights = 4.5 bpw).
Option taken: Implemented the 16-byte wire record (`d: f16, dmin: f16` [4 bytes LE] + 12-byte packed `sc`/`mn` payload = 16 bytes) matching GGUF `Q4_K` bit-for-bit and satisfying 4.5 bpw; see DECISION(A2.2) at `I4KSuperblock` in `crates/r9v-format/src/records.rs`.
Proposed resolution: Clarify Spec 2 §3.2 and DECISIONS.md D-004 to state that the scale record is 16 bytes total: 4 bytes for `d`/`dmin` (`f16` little-endian) plus a 12-byte packed payload encoding `sc[8]: u6` and `mn[8]: u6`.

## SI-25 — A2.2 — spec 2 §8
What: Spec 2 §8 lists `I8_R` as "8.0" bits per weight in the size accounting table ("8.0 | native"), omitting the per-row `f16` scale overhead from the finite-row bpw formula.
Why it blocks or misleads: Spec 2 §3.2 specifies that `I8_R` stores one `s: f16` scale per row of `K` weights. For any finite row length `K`, the scale contributes 16 bits of overhead, so the true storage is `8*K + 16` bits for `K` weights, or `(8*K + 16) / K` bits per weight (e.g. 9.0 bpw at K=16, ~8.0039 bpw at K=4096). Stating a flat 8.0 bpw misleads memory sizing and tensor offset calculations for short-to-moderate row lengths.
Option taken: Implemented the exact finite-row rational bits-per-weight formula `(8*K + 16) / K` in `bits_per_weight`; see DECISION(A2.2) at `bits_per_weight` in `crates/r9v-format/src/scheme.rs`.
Proposed resolution: In Spec 2 §8, update the `I8_R` bits-per-weight entry to state `(8K+16)/K (approaches 8.0 as K→∞)` to distinguish exact finite-row sizing from asymptotic weight-only bits.

## SI-26 — A2.2 — spec 2 §3
What: `phase-a-agent-breakdown.md` card A2.3 describes its deliverable as "`ggml_type → SchemeId` mapping; repack rules ... for `Q8_0`, `Q4_0`, ..., `F16`, `BF16`", suggesting that unquantized floating-point representations (`F16`, `BF16`) are variants of `SchemeId`. Spec 1 §2.2 and Spec 2 §3 define quantization schemes as applying only to quantized tensors, with unquantized weights/activations represented by `DType::{F16, BF16}` and `QuantScheme::None`.
Why it blocks or misleads: If `SchemeId` were extended to include `F16` and `BF16`, it would violate the closed 22-scheme set defined in Spec 2 §3, conflate data types with quantization schemes, and break invariant checks across IR and loader components expecting `SchemeId` to denote quantized tensors with scale records.
Option taken: Kept `SchemeId` as the closed set of 22 quantized schemes (4 native in §3.2, 18 repack-only in §3.3) and excluded unquantized dtypes (`F16`, `BF16`), which remain represented via `DType` and `QuantScheme::None`; see DECISION(A2.2) at `SchemeId` in `crates/r9v-format/src/scheme.rs`.
Proposed resolution: Update the card A2.3 description in `phase-a-agent-breakdown.md` to state that `F16` and `BF16` are handled as unquantized tensors (`QuantScheme::None` with their respective `DType`) rather than mapped to variants of `SchemeId`.
## SI-17 — A1.11 — spec 1 §2.5 / spec 3 §3.3, §3.5
What: Three gaps around the `BatchMeta` layout. (1) Spec 3 §3.5 says "The block table for the group is a ring of `ceil(w/32) + 1` entries", while spec 3 §3.3 and spec 1 §2.5 fix `block_table` as `[G, S, max_blocks]` with `max_blocks = ceil(state.max_ctx / 32)` padded with a sentinel — a window-sized ring cannot also be a fixed full-context table, and a ring position cannot survive eviction without renumbering every entry. (2) `slot(s, p) = (block_table[g][s][p / 32], p % 32)` with "`[G, T]` (one flattened slot per new token per group)" does not say whether the flattened value carries the pool-global block id or the within-table position, nor whether ids are shared across groups; a within-table position cannot address the arena once eviction compacts the table. (3) `Sink(n) + Window(w)` retention keeps two ranges (pinned sink plus sliding window) while `BatchMeta.window_start` carries a single `[G, S]` value, so the sink range has no stated encoding.
Why it blocks or misleads: Two managers that both satisfy the text can emit different `slot_map`/`window_start` bytes, breaking the Spec 3 §5 promise that identical request histories produce identical `BatchMeta` for doctor-bundle diffs; the scheduler and attention kernel cannot agree on slot decoding, eviction holes, or the sink range without a stated convention.
Option taken: The ring sentence is read as the eviction policy, not the storage shape: every `block_table` row stays width `max_blocks` with each block id stored at its absolute logical block index and sentinel holes where window eviction released blocks. `slot = pool_block_id * 32 + lane` per group with pool-global (not table-position) ids, so the device derives `base[group] + block_id * block_bytes` directly; ids are per-group-pool, never arena-global across groups. `window_start` reports the window start (`max(0, ctx_len - w)`) with the sink range implicitly `[0, ceil(n/32)*32)` pinned from position 0.
Proposed resolution: State that windowed tables keep full `max_blocks` width with sentinel holes (the ring describes eviction, not storage); state the flattening as per-group `block_id * 32 + lane` with pool-global ids; and state that `window_start` is the window start while the sink length derives from the group's `Retain`.

## SI-18 — A1.6 — spec 1 §4.C `matmul` / spec 1 §4.A `quant_act`
What: Spec 1 §4.C defines `matmul` consuming `x [M, K] (f16|bf16|i8 PerToken|i8 PerBlock32|e4m3 PerToken), w [N, K], bias? [N] f32`, omitting an input tensor for activation scales, while Spec 1 §4.A `quant_act` emits quantized activations `xq [T, N] (i8|e4m3)` alongside a separate scale tensor `scale [T] f32` (or `[T, N/32] f32`).
Why it blocks or misleads: Quantized activation GEMM (`i8 PerToken`, `i8 PerBlock32`, `e4m3 PerToken`) cannot evaluate without the activation scale factors emitted by `quant_act`. Without an input tensor slot in `MatmulOp`'s graph arity, step graphs cannot express or validate the data-flow edge connecting `quant_act`'s scale output to `matmul`, forcing engines to either invent an unstandardized IR contract or carry scales out-of-band.
Option taken: Retained the closed Spec 1 §4.C / `r9v-ir::MatmulOp` arity for graph validation, while allowing T0 execution (`matmul_with_scales` or `TensorView::scale`) to accept activation scales attached out-of-band or passed explicitly; rejected modifying `r9v-ir::MatmulOp` without spec authorization.
Proposed resolution: Update Spec 1 §4.C to add `x_scale?` to `matmul` inputs (conditional on `x` carrying `PerToken` or `PerBlock32` quantization), specifying shape `[M] f32` for `PerToken` and `[M, K/32] f32` for `PerBlock32`.

## SI-27 — A1.14 — spec 8 §3 `residual_scale`
What: Spec 8 §3 defines `LayerSpec.residual_scale: f32` ("1.0 unless the family scales residuals") but gives no formula, and Spec 1 §4.B `residual_add` is specified as `a + b` with no scale attribute, so a non-unit factor has no honest lowering.
Why it blocks or misleads: Any choice of scaled form (scale the stream, scale the branch, scale both) changes numerics; folding the factor into the sublayer's output projection instead would silently rewrite loader-bound weights and is not bit-identical in fp arithmetic.
Option taken: `y = a + scale * b` computed in f32 (the added residual branch is scaled; the stream `a` keeps the identity path), via a new `ResidualAddOp.scale` attribute defaulting to 1.0, validated finite and nonzero at both the spec and op layers; `scale = 1.0` reproduces the A1.3 graph exactly. Scalar T0 semantics (`a + scale * b` in f32) and golden-vs-f64 tests ship in this card.
Proposed resolution: State the residual formula `y = a + scale * b` in f32 in spec 8 §3.1 and add `scale: f32 = 1.0` to the Spec 1 §4.B `residual_add` signature.

## SI-28 — A1.14 — spec 8 §3 `final_logit_softcap` / spec 1 §4
What: Spec 8 §3 defines `ModelSpec.final_logit_softcap: Option<f32>` with no formula, and Spec 1 has no final-logit softcap operation: §4.B `activation` offers only a hard upper `clamp`, and §6.3 `logit_softcap` is softmax-internal (applied in f32 before the max), not a post-`lm_head` transform.
Why it blocks or misleads: A hard clamp is not a softcap (different values everywhere except the rails), and reusing the attention-internal attribute would misdescribe where and how the transform applies; either substitution fakes the effect with an unrelated op.
Option taken: New `LogitSoftcapOp { cap: f32 }` lowering exactly one op on the `lm_head` output when the field is set (`y = cap * tanh(x / cap)` in f32 over `[T, V]` f32, the Gemma-family convention for final softcaps); `None` lowers to no op, reproducing the A1.3 graph exactly. Scalar T0 semantics and golden-vs-f64 tests ship in this card.
Proposed resolution: State the final-softcap formula `y = cap * tanh(x / cap)` in f32 in spec 8 §3 and add the `logit_softcap` op to the Spec 1 §4.B catalog.

## SI-29 — A1.14 — spec 8 §3.1 MLA channel mapping / spec 1 §4 `split`+`concat`
What: Spec 8 §3.1 says `mla` "changes the q/k/v projections to the low-rank form" without defining the channel layout, which half RoPE applies to, what the state write carries, or how `qk_norm` (defined for the per-head q/k pair) applies when there is no per-head k; Spec 1 §4 has no channel split or concatenation op, so the latent/rotary ranges cannot be named as edges.
Why it blocks or misleads: Without a split primitive the builder must either rotate the compressed latent as if it were positional (numerically wrong) or fake the separation with an unrelated op; without a qk_norm rule the combination must stay rejected or be silently dropped.
Option taken: New rank-3 last-axis `SplitOp { first }` / `ConcatOp` (pure data movement, Replicated sharding, scalar T0 semantics and tests in this card). The query splits at `qk_nope_dim` and the KV rows at `kv_lora_rank`; rope applies to the rotary parts only with `rot_dim` set to the rotary width itself (a narrower standard `rot_dim` would leave rotary channels unrotated, and validation rejects `rot_dim > D` rather than clamping); the query is concatenated back to `[T, H, nope + rope]`; the state write carries the exact canonical `(c_kv [T, H, kv_lora_rank], k_rope [T, H, rope_dim])` pair (combined-form acceptance retained for A1.2-era graphs); `qk_norm` lowers as the standard after-projection/before-rope pair with `Head(nope + rope)` over the query rows and `Last` over the head-less KV rows. See DECISION(A1.14) notes at the lowering sites.
Proposed resolution: Define the MLA channel layout (per-head `[nope | rope]`, KV rows `[latent | rope]`), the decoupled-rotary `rot_dim` rule, the exact state-write pair, and the qk_norm lowering in spec 8 §3.1, and add `split`/`concat` to the Spec 1 §4.A data-movement catalog.

## SI-30 — A1.14 — spec 1 §2.5 / §4.B rope positions binding
What: Spec 1 §2.5 carries `positions` inside the structured `BatchMeta` input while §4.B `rope` consumes a `positions` tensor edge, but no rule binds the two; the A1.3 builder aliased `token_ids` (scalar) or recorded a token edge (MRoPE) as the positions source.
Why it blocks or misleads: Token IDs are not positions (they coincide only for trivial(offset-0, no-pad, no-vision) batches), so every rope in every A1.3 graph reads a fake alias; graph validation cannot distinguish a legitimate positions edge from any other U32 edge.
Option taken: `Graph::bind_positions` projects `BatchMeta.positions` as one typed edge per graph (`[T] u32` scalar, `[T, 3] u32` MRoPE), bound at most once (duplicate or conflicting rebinds report `PositionsConflict`); graph validation requires every rope's positions input to be exactly that edge (`GraphPositionsMissing` / `GraphRopePositionsMismatch` otherwise). The structured `BatchMeta` remains the single external input.
Proposed resolution: State in spec 1 §2.5 that `BatchMeta.positions` projects to one typed graph edge of the scalar or triplet form, and require in §4.B that `rope` consume exactly that edge.

## SI-31 — A1.14 — spec 8 §2 builder value identity
What: Spec 8 §2 threads plain `Tensor` descriptors through the builder API with no identity rule, so structurally identical descriptors resolve to one edge: cloning aliases, and two different weights with the same shape silently share an edge.
Why it blocks or misleads: Descriptor-equality lookup makes "which value does this op read" depend on insertion order rather than construction; the MRoPE and MTP paths additionally recorded edges that did not produce the descriptor (fake aliases).
Option taken: Opaque `Value { tensor, edge }` is the builder currency: every binder and op output mints a fresh edge, cloning preserves identity, structurally identical values never alias, and descriptor-to-edge lookup is removed (its error variant goes with it). `BoundWeight` keeps the structural descriptor for the loader contract.
Proposed resolution: State in spec 8 §2 that graph values carry SSA identity separate from structural descriptors, that binding and op outputs mint fresh edges, and that no descriptor-equality lookup exists.

## SI-32 — A1.14 — spec 8 §2, §5 MTP capture / spec 1 §3.2 subgraph inputs
What: Spec 8 §5 binds MTP weights in a subgraph but never says which parent value feeds it (the A1.3 builder ignored its `hidden` argument and fed heads from the multimodal override input), and the Spec 1 §3.2 closed external-input set has no subgraph hidden-state input.
Why it blocks or misleads: Without an explicit capture the child graph's input is whatever the builder happened to register, heads cannot be shown to restart from the chosen parent value, and reusing `EmbedOverride` mislabels an MTP capture as multimodal input.
Option taken: `GraphBuilder::subgraph_with_capture` binds a fresh `ExternalInputKind::SubgraphHidden` edge mirroring the parent value's shape/dtype and records `SubgraphCapture { parent_edge, child_edge }` on the finished child; `Layer(n)`/`Last` selection actually flows into the capture, and every head restarts from `capture_value()`.
Proposed resolution: Add the parent-hidden capture binding to spec 8 §5 and a subgraph-hidden external input to the spec 1 §3.2 set (or scope the set to step graphs and define the subgraph input surface separately).

## SI-33 — A1.14 — spec 1 §7 / phase-a-agent-breakdown A1.14 vs A3.3
What: Spec 1 §7 requires both CPU and portable HIP references whenever a new IR operation lands, while phase-a-agent-breakdown and the external A1.14 card define A1.14 as a CPU-only card (`GPU: no`) depending only on A1.2, A1.3, and A1.5, with portable HIP kernel implementations deferred until cards A3.3–A3.7 following the kernel ABI definition in A3.2.
Why it blocks or misleads: Implementing portable HIP references for new ops (`Split`, `Concat`, `LogitSoftcap`, scaled `ResidualAdd`) in A1.14 would require inventing or stubbing an ad-hoc kernel launch ABI before card A3.2 defines the uniform ABI struct and launch conventions, or producing fake implementations that violate the A1.14 card scope and crate boundaries (`r9v-hip` is not in A1.14's allowed crates).
Option taken: Implemented complete scalar CPU T0 references and tests in `r9v-t0` under card A1.14, and explicitly assigned portable HIP kernel implementations for `SplitOp`, `ConcatOp`, `LogitSoftcapOp`, and scaled `ResidualAddOp` to card A3.3 (elementwise and data-movement portable HIP references) after the A3.2 ABI is established.
Proposed resolution: Clarify Spec 1 §7 and the phase breakdown to specify that IR-defining cards prior to A3.2 deliver scalar CPU (T0) references, with portable HIP (T1) implementations delivered in the designated Phase A3 kernel cards once the ABI is frozen.

## SI-40 — A1.15 — spec 1 §2.5 / spec 3 §5
What: Spec 1 §2.5 defines `BatchMeta.seq_ids: [S] u32` for device execution (and Philox RNG keying in §4.F requiring batch composition invariance), while `r9v_common::SeqId` wraps `u64` and Spec 3 §5 / Spec 6 §2.1 thread host sequence IDs across long-running service lifetimes. The specs do not specify whether `seq_ids` are checked global IDs or batch-local slot indices, nor how to reconcile 64-bit host IDs with 32-bit device fields without lossy truncation.
Why it blocks or misleads: A blind `as u32` cast would silently truncate at 2^32 sequences, causing rollover, ID collision, and non-deterministic RNG corruption across long-running server instances. Reusing batch-local indices `0..S` would break the Spec 1 §4.F guarantee that Philox sampling is reproducible regardless of batch composition. An unbounded mapping table from u64 to u32 leaks memory over time.
Option taken: Represented device `seq_ids` as strictly checked global IDs: `StateManager::batch_meta` losslessly converts each `SeqId` via `u32::try_from(seq.as_u64())`, failing with a typed error (`StateError::SeqIdOverflow`) if any ID exceeds `u32::MAX`. In `StateManager::new_seq`, sequence allocation checks that `next_seq <= u32::MAX as u64` before incrementing or mutating any state, failing before mutation with a typed `StateError::SeqIdOverflow` if the 32-bit device address space would be exhausted. Sequences beyond `u32::MAX` are thus guaranteed never to reach device execution or corrupt RNG, while preserving batch-invariance for all valid IDs; see DECISION(A1.15) at `StateManager::batch_meta_with_options` and `StateManager::new_seq`.
Proposed resolution: Clarify in Spec 1 §2.5 and Spec 3 §5 that `BatchMeta.seq_ids` carries checked global sequence identifiers matching `SeqId`, and either update `seq_ids` to `[S] u64` in the IR and Philox ABI or document that the device sequence address space is bounded at `u32::MAX` with typed refusal on overflow before mutation.
