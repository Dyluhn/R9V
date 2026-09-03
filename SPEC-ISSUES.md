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
