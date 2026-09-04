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
Why it blocks or misleads: The original A0.S6 fingerprint read only the GPU endpoint files. Both endpoints report Gen5 x16, but the `0000:13:00.0` path traverses root port `0000:00:02.2`, whose maximum link is Gen4 x4. Endpoint-only discovery therefore produced a false x16/x16 topology and would poison topology fingerprints and communication-cost evidence.
Option taken: Retained the authoritative x16/x4 hardware description, corrected `spikes/p2p/RESULT.md`, and changed spec 5 §2 to require endpoint-to-root ancestry with a capacity bottleneck. P2P remains `Direct` because that is an independent measured transport result; it is not evidence of lane width.
Proposed resolution: Resolved in spec 5 §2 and the corrected A0.S6 receipt. The implementation gate is a synthetic endpoint-x16/upstream-x4 test plus live doctor output naming the capping hop.

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

## SI-41 — A1.4 — spec 8 §3, §4
What: Spec 8 §4 lists `window pattern` as a switch read from metadata for the `llama` family, while Spec 8 §3 defines `window: Option<u32>, sinks: u32` per layer without specifying how global GGUF metadata keys map to heterogeneous per-layer sliding window patterns (such as Gemma 2's 1:1 alternating pattern or Gemma 3's 5:1 interleaving).
Why it blocks or misleads: Checkpoints in GGUF format supply a scalar `attention.sliding_window` (e.g. 4096 or 1024) without embedding explicit layer assignment arrays, so two implementations may disagree on layer phase or stride.
Option taken: Derived the canonical alternating 1:1 pattern for `gemma2` and 5:1 6-layer period pattern for `gemma3` from family identity when `attention.sliding_window_pattern` is absent; documented under DECISION(A1.4).
Proposed resolution: Explicitly document the canonical per-layer window derivation rules for `gemma2` (1:1 local/global) and `gemma3` (5:1 local/global) in Spec 8 §4.

## SI-42 — A1.4 — spec 8 §4
What: Spec 8 §4 lists `softcaps` and `tied embeddings` as metadata switches for the `llama` family without defining the architecture-specific fallback defaults when `<arch>.tie_word_embeddings`, `<arch>.attention.logit_softcapping`, or `<arch>.final_logit_softcapping` keys are omitted from GGUF metadata.
Why it blocks or misleads: Gemma 2 requires attention (50.0) and final (30.0) logit softcapping and tied embeddings by design even when legacy GGUF exporters omit explicit softcap/tied keys, while LLaMA/Mistral/Qwen models require untied embeddings and no softcapping by default. An engine that defaults all models identically will mis-shape Gemma graphs or corrupt LLaMA logits.
Option taken: In `families/llama.rs`, applied architecture-specific defaults when metadata keys are absent: Gemma 2 defaults to attention softcap 50.0, final softcap 30.0, and tied embeddings `true`; Gemma 3 defaults to tied embeddings `true`; all other architectures default to `None` for softcaps and `false` for tied embeddings unless overridden by metadata. Documented under DECISION(A1.4).
Proposed resolution: Document architecture-specific metadata fallback defaults for softcaps and tied embeddings in Spec 8 §4.

## SI-43 — A1.7 — spec 1 §6.3 / spec 1 §4.D `sinks` / spec 3 §2, §3.5
What: Spec 1 §6.3 states "Sinks are extra learned logits prepended to the softmax denominator," but the Spec 1 §4.D `attention` signature carries only `sinks: u32` with no sink-value operand, no sink-score input, and no learned parameter of any kind; Spec 3 §2 defines `Retain::Sink(n) + Window(w)` purely as a retention policy (pin the first `ceil(n/32)` blocks in addition to the window) whose kernel effect is receiving "both ranges" (§3.5).
Why it blocks or misleads: A reader implementing §6.3 literally would add trainable sink logits and extra softmax terms that no graph can feed (nothing in §3.2 storage, §3.3 addressing, or the op arity holds them), silently changing numerics versus an implementation that treats sinks as retained prefix positions; the two agree only when the learned logits are all `-inf`, which defeats their stated purpose.
Option taken: Treated `AttentionOp.sinks` as the count `n` of retained prefix token positions admitted in addition to the window (`p < n`), so the visible set under `CausalWindow(w)` is `(p < n) ∪ (p >= window_start)` intersected with the causal/tree mask; under `Causal` the union is trivially everything, so `sinks > 0` runs instead of failing closed. See DECISION(A1.7) at `is_retained` in `crates/r9v-t0/src/attention.rs`.
Proposed resolution: Replace the §6.3 sentence with "Sinks are retained prefix positions admitted in addition to the window (`p < sinks`), with no extra logits or parameters," and state in §4.D that `sinks` counts prefix positions pinned by `Retain::SinkWindow`.

## SI-44 — A1.7 — spec 1 §4.D `state_write_kv` latent form / spec 3 §2, §3.2
What: Spec 1 §4.D says the MLA write carries "the compressed latent + rope part" without fixing operand order, head count, or what the second operand holds when the first already holds `kv_lora_rank + rope_dim`; Spec 3 §3.2 states the `KvLatent` regions have no head dimension while every producing graph writes rank-3 `[T, H, *]` tensors; Spec 3 §2 requires rope-always-f16 storage but never says whether the rope half of a combined operand needs separate f16 handling.
Why it blocks or misleads: Without a form rule one writer can emit rope-first while another emits latent-first (both satisfy the text) and their caches are mutually unreadable; without a head rule a multi-head write has no defined destaggering into head-less storage; without a rope rule a combined writer may quantize the rope half with the latent scale instead of f16.
Option taken: Honored the landed A1.14 canonical forms (SI-29): the exact split pair (`c_kv [T, 1, kv_lora_rank]`, `k_rope [T, 1, rope_dim]`) or the combined form (operand 0 `[T, 1, kv_lora_rank + rope_dim]` split at `kv_lora_rank`; operand 1 must match on T/H but its values are not stored). Writes require `H == 1` with a typed failure otherwise, and the rope part is always stored as f16 regardless of the latent cache dtype. See DECISION(A1.7) notes at `state_write_kv_latent` and `check_latent_heads` in `crates/r9v-t0/src/attention.rs`.
Proposed resolution: State in Spec 1 §4.D that the latent write is the exact split pair (or the combined operand 0 split at `kv_lora_rank`), that both operands carry `H == 1`, that a combined operand 1 is not stored, and that the rope region is always f16 per Spec 3 §2.

## SI-45 — A1.7 — spec 1 §4.D `scale_granularity: PerBlock`
What: Spec 1 §4.D lists `scale_granularity: PerTokenHead|PerBlock` as an attribute of `state_write_kv`, but no record shape, block size, scale dtype, or SoA placement for a per-block KV-cache scale exists anywhere in Spec 1 §4.D, Spec 3 §2–§3, or Spec 2 §3 (whose block-scale machinery covers weights, not KV state); Spec 3 §2–§3.2 define only the per-token-head f16 scales (two per head for `KvPaged`, one per slot for `KvLatent`).
Why it blocks or misleads: Any per-block layout T0 invented (block size 32? 64? one scale per block per head?) would be a guess about storage that kernels (A5.2/A5.5), the loader, and prefix-cache hashing must agree on bit-for-bit; two guessers that differ corrupt each other's caches silently.
Option taken: Failed `PerBlock` closed with a typed `InvalidAttribute` error (naming SI-45) while fully supporting the required `PerTokenHead` geometry for f16/i8/e4m3 `KvPaged` and `KvLatent` caches; see the granularity check in `validate_write_common` in `crates/r9v-t0/src/attention.rs`.
Proposed resolution: Either define the PerBlock scale record (block size, dtype, per-head/per-slot placement in §3.2 terms) in Spec 3 §2–§3.2, or remove `PerBlock` from the Spec 1 §4.D attribute until a card owns it.

## SI-46 — A1.7 — spec 1 §4.D `MlaAttentionSpec` non-absorbed dims
What: `MlaAttentionSpec` admits `qk_nope_dim != kv_lora_rank` and `v_dim != kv_lora_rank` (the combination the A1.14 builder emits for DeepSeek-style models), but the `attention` op carries no projection operands (`W_UK`, `W_UV`) to map between the query/output dims and the latent width, so scores of the form `q_nope . c_kv` and values of width `v_dim` from a `kv_lora_rank`-wide cache are dimensionally undefined.
Why it blocks or misleads: Any T0 lowering of a non-absorbed graph (truncate the latent, pad the query, invent a projection) invents numerics that the kernel cards would then have to match without a contract; a test that passes by such invention proves nothing about the real model path.
Option taken: Implemented the absorbed MLA form (`qk_nope_dim == kv_lora_rank`, `qk_rope_dim == rope_dim`, `v_dim == kv_lora_rank`) and failed all other combinations closed with typed errors naming SI-46; see `validate_mla_dims` in `crates/r9v-t0/src/attention.rs`.
Proposed resolution: State in Spec 1 §4.D whether the attention op consumes absorbed projections (and which graph ops produce the absorbed query) or add the missing projection operands to the op signature so non-absorbed dims have a defined lowering.

## SI-47 — A1.9 — spec 1 §4.E (scan kind-specific equations absent)
What: Spec 1 §4.E fixes the common `q/k/v/alpha/beta` signature and the `S_t` state for `linear_attn_scan` but states no per-kind equations for `GatedDeltaNet`, `GLA`, or `Mamba2` (gate application, normalization, decay placement, output projection order); `w_gate_up` normalization `NRMS` and the `LinearAttnKind` effects are unstated.
Why it blocks or misleads: Without kind-specific equations, any per-kind T0 behavior would be invented; two implementations can both satisfy the signature while computing different recurrences.
Option taken: Implemented one shared `alpha/beta/k/v` gated outer-product recurrence (`S = alpha*S + beta*(k⊗v)`, `o = q·S`, all `f32` ascending) for all three kinds — the narrowest contract consistent with the signature — in `crates/r9v-t0/src/linear_attn_scan.rs`; chunked and recurrent forms share it bit-exactly. No architecture-specific claim beyond that contract. Documented under DECISION(A1.9).
Proposed resolution: State the kind-specific update equations in Spec 1 §4.E or confirm the three kinds share one recurrence.

## SI-48 — A1.9 — spec 1 §4.E (sequence-boundary inputs missing from state/scan signatures)
What: `causal_conv1d`/`linear_attn_scan` carry `[T, ...]` token-major tensors with no sequence-boundary input, yet require per-sequence state threading and reset; rows are not self-delimiting, `DType` is closed (SI-12), and IR arity cannot grow without touching every producer.
Why it blocks or misleads: Without a boundary channel, multi-sequence batches cannot reset recurrences or address per-sequence slots; threading boundaries through fake tensor edges would corrupt the dtype system.
Option taken: Added an explicit execution-only `SeqLayout` descriptor plus leading-`S` state slots (`[S, Wk-1, C]` conv, `[S, H, D, Dv]` scan) in `crates/r9v-t0/src/segments.rs`, consistent across conv/scan without changing IR arity. Documented under DECISION(A1.9).
Proposed resolution: Confirm boundaries travel out-of-band and bless the `[S, ...]` state convention in Spec 1 §4.E / Spec 3 §4.

## SI-49 — A1.9 — spec 1 §4.C (MoE top-K selection and routing-weight semantics)
What: Spec 1 §4.C fixes MoE shapes (`[T, K]`) but not the selection contract: is `top_k = 0` legal, may one expert occupy several of a token's K slots, are routing `weights` probabilities or opaque multipliers, is `expert_ids` order significant, and which location scheme reports a bad routing value (`InvalidLogit` names sampling positions only)?
Why it blocks or misleads: An empty routing (zero output) vs a refusal, duplicate-slot double counting vs dedup, and renormalized vs raw weights all change numerics silently.
Option taken: In `crates/r9v-t0/src/moe_route.rs` / `moe_ffn.rs`: reject `top_k = 0`; allow duplicate expert slots (each slot combines independently from one shared expert row); treat `weights` as opaque multipliers combined in ascending `(expert, token, slot)` order with `y` zero-initialized first; report non-finite logits per `(t, e)` as `InvalidAttribute`. Documented under DECISION(A1.9).
Proposed resolution: Fix the top-K selection contract, weight semantics, and routing error locations in Spec 1 §4.C.

## SI-50 — A1.9 — spec 1 §4.C + spec 4 §5.6 (MoE gate/up row order unstated)
What: Spec 1 §4.C fixes the `[E, 2·Dff, Dm]` gate/up shape but not the row order; Spec 4 §5.6 says "gate/up interleaved" about kernel access without fixing storage, and names a plain (fused) epilogue.
Why it blocks or misleads: Gate-major halves vs interleaved rows select different weight rows for the same token — a silent numerics fork across implementations.
Option taken: In `crates/r9v-t0/src/moe_ffn.rs`, read gate rows `[0, Dff)` and up rows `[Dff, 2·Dff)` (gate-major halves, contiguous per projection); L1 expert tiles are interpreted over the flattened `[E·R, K]` row space (no tiled rank-3 expert layout is specified). Documented under DECISION(A1.9).
Proposed resolution: State the gate/up row order in Spec 1 §4.C.

## SI-51 — A1.9 — spec 1 §4.C (router scoring, renormalization, scale, grouping gaps)
What: Spec 1 §4.C names the router knobs but not their composition: softmax/sigmoid placement vs bias, what `renormalize` divides by, where `scale` multiplies, the grouped-selection algorithm for `group`, and whether output row order is observable.
Why it blocks or misleads: Post-renorm scaling silently un-normalizes; full-row denominators contradict selected-weight renormalization; an unstated group algorithm cannot be implemented without invention.
Option taken: In `crates/r9v-t0/src/moe_route.rs`: scores from `logits + bias`, stable softmax / sigmoid in `f32`, top-K by `(-score, index)` with lowest-index tie-break, `weights = score * scale` with renormalization dividing by the selected-K sum, rows in descending-score order (presentation only — the combine is order-insensitive); `group.is_some()` fails closed. Documented under DECISION(A1.9).
Proposed resolution: Fix scoring/renormalize/scale composition and the grouped-selection algorithm (or remove `group`) in Spec 1 §4.C.

## SI-52 — A1.9 — spec 1 §4.C/§4.E (MoE/scan scale carriers and the `x` Livness rule)
What: Spec 1 §4.C/§4.E require scales without fixing their carrier: explicit parameters vs attached views, L0 inline-scale geometry for expert weights, and the meaning of the `x` Livness rule for A1.9 executors.
Why it blocks or misleads: Divergent carriers across cards would fork every quantized A1.9 graph between A1.6 scale tooling and new code.
Option taken: In `crates/r9v-t0/src/moe_ffn.rs` / `linear_attn_scan.rs`, mirrored the `matmul`/`matmul_with_scales` carrier contract exactly: explicit scale parameter else attached view (SI-18 pattern), L0 inline strides (`K+2`, `K+K_blocks·2`, `K/2+K_superblocks·16`) permitted for expert weights, and the `x` Livness rule (`Livness::L0` ⇒ inline scales mandatory, `L1` ⇒ explicit scales). Documented under DECISION(A1.9).
Proposed resolution: Bless the shared out-of-band scale convention for A1.9 executors in Spec 1 §4.C/§4.E.

## SI-53 — A1.9 — spec 1 §4.A (n-gram hash families, orders > 1, staged/device carriers)
What: Spec 1 §4.A fixes n-gram shapes but (a) publishes no `HashId` enumeration, (b) leaves device-mode `orders > 1` undefined (order-n context rows are not in the signature — what feeds the hash?), and (c) fixes no staged/device scale carriers.
Why it blocks or misleads: Any hard-coded hash family would be invented; order-n device gather without named hash inputs would fabricate addressing; a scalar cannot name a superblock record, so multi-block staged rows under one scalar have no specified scale application.
Option taken: In `crates/r9v-t0/src/ngram_gather.rs`: never map a hash family — device mode takes `&dyn NgramHash` (scheduler/models scope supplies `NgramSpec.hash`), bounds-checks every hashed row, and rejects `orders != 1` in device mode; staged `I8R` rows take a scalar `row_scales` per `(t, h)`, staged `I8B128` requires `Dn == 128`, staged `I4K` and multi-block staged rows fail closed; quantized device tables require separate carriers (`[entries]`/`[entries, Dn/128]` F16 bytes, `[entries, Dn/256, 4]` U32 bytes); row-major tables only. Documented under DECISION(A1.9).
Proposed resolution: Publish the `HashId` enumeration, define order-n device behavior, and bless the staged/device carriers in Spec 1 §4.A.

## SI-54 — A1.9 — spec 1 §4.G (rank-1 collective semantics and the `recv` refusal)
What: Spec 1 §4.G fixes collective arities but not the accumulation precision or the rank-1 data path: is `all_reduce` an identity at `ranks = 1`, does `all_to_all`'s single `counts[0]` cover all rows, are `send(0)`/`barrier` no-ops, and what sources `recv` data?
Why it blocks or misleads: An `f32` round-trip identity would silently corrupt integer transfers; a `recv` that returns zeros would fabricate data.
Option taken: In `crates/r9v-t0/src/collectives.rs`: `all_reduce(Sum)`/`all_gather`/`reduce_scatter`/`all_to_all` are bit-exact identity transfers through the `copy` core (proven with `u32` values above 2²⁴); `reduce_in` must still be `f32` per spec (accepted, unused at rank 1 — multi-rank ascending-rank `f32` reduction is executor scope); `send(peer = 0)` and `barrier` are no-ops; `recv` validates its descriptor then fails closed (no T0 transport). Documented under DECISION(A1.9).
Proposed resolution: Confirm the rank-1 executor contract in Spec 1 §4.G.

## SI-55 — A1.9 — spec 1 §4.E (causal-conv quantized weights, activations, bias, state dtype)
What: Spec 1 §4.E permits `i8/i4` conv weights but provides no scale input in the signature; `ConvActivation` lists `Silu|Identity` without saying the set is exhaustive; `bias`/`state` width rules are implicit.
Why it blocks or misleads: Scale-less quantized weights cannot be dequantized without invention; a third activation or a wider state type would silently change numerics.
Option taken: In `crates/r9v-t0/src/causal_conv1d.rs`: reject quantized weights with `QuantMismatch`; match `ConvActivation` exhaustively; `bias [C]` optional; state slots `f16` exactly (`[S, Wk-1, C]`, zero rows at `Wk = 1`); every element decoded at use; `y`/state staged before commit. The `f16` carry rounding is the only split-vs-oneshot divergence and is pinned by test, not hidden in tolerance. Documented under DECISION(A1.9).
Proposed resolution: State the conv weight-scale rule (or remove `i8/i4` from the dtype set) and close the activation/bias/state rules in Spec 1 §4.E.

## SI-56 — A2.3 — spec 2 §3.3
What: The §3.3 table lists the `I5_B32F` / `I5_B32FM` scale record as "as Q4", but `Q4_0` stores one `f16` (`d`) while `Q4_1` stores two (`d`, `m`); the record size differs by 2 bytes and the SoA region stride depends on it.
Why it blocks or misleads: A reader implementing "as Q4" literally cannot tell whether `I5_B32F` carries a min field; guessing wrong shifts every subsequent record in the SoA region by 2 bytes per block and silently corrupts all scales.
Option taken: `I5_B32F` uses the `Q4_0` record (`d` only, 2 B) and `I5_B32FM` uses the `Q4_1` record (`d`, `m`, 4 B), matching the GGUF `Q5_0` / `Q5_1` wire blocks both repack.
Proposed resolution: Replace "as Q4" with "as `I4_B32F` / as `I4_B32FM` respectively" in the §3.3 table.

## SI-57 — A2.3 — spec 2 §3.3
What: The §3.3 table lists the `I3_K` / `I2_K` scale record as "as GGUF" without fixing contents or order: `Q3_K` wire carries `[hmask][qs][scales12][d]` while `Q2_K` wire carries `[scales16][qs64][d][dmin]`, so "the scale bytes" are one contiguous slice in the first type and split around the value bytes in the second.
Why it blocks or misleads: Two loaders that both "store as GGUF" can disagree on whether the SoA record is the verbatim wire span (which for `Q2_K` would embed 4 value bytes as `d` / `dmin`) or the gathered scale fields; their scale regions are mutually unreadable.
Option taken: `I3_K` stores `[scales12][d]` in wire order (14 B, contiguous in the wire); `I2_K` stores `[scales16][d][dmin]` gathered across the split wire layout (20 B).
Proposed resolution: State the exact SoA record contents and order for `I3_K` and `I2_K` in §3.3 (field names with wire offsets).

## SI-58 — A2.3 — spec 2 §3.3, §7
What: Neither §3.3 nor §7 pins which GGUF revision defines the wire-block layouts and dequant formulas the repack rules must reproduce bit-exactly ("repack never requantize", §10 round-trip); GGUF layouts are versioned upstream outside this spec.
Why it blocks or misleads: Without a pinned reference, a future GGUF layout change (or a reader built against a different revision) passes the §7 shape checks yet fails the §10 bit-equality test with no way to tell which side moved.
Option taken: Pinned gguf-py 0.19.0 `quants.py dequantize_blocks` as the wire authority (codes from its `GGMLQuantizationType`); fixtures record the version, and K-quant wire bytes that gguf-py cannot quantize are hand-built from that same layout with the provenance stated per case.
Proposed resolution: Name the authoritative GGUF revision (or a vendored layout table) in §7 that repack implementations must reproduce.

## SI-70 — A2.4 — spec 2 §3.3
What: The §3.3 table lists the IQ rows' values as "u4 index into 16-entry i8 LUT" / "codebook indices" and their scale records as "as GGUF", which fixes neither the SoA record contents and order (IQ wire blocks split scales, high-bit planes, sign bytes and indices across non-contiguous spans, and `IQ1_M` packs `d` across scale nibbles) nor the repack granularity (one packed index covers 1, 4 or 8 weights depending on the family).
Why it blocks or misleads: Two loaders that both "store as GGUF" can place different bytes in the SoA region (a verbatim wire prefix vs gathered scale fields for `IQ2_S`/`IQ3_S`, per-weight grid expansion vs packed index bytes for the value region) and stay mutually unreadable while both passing the §7 shape checks.
Option taken: Records follow the gguf-py wire order with the index payload removed (`I4_NL [d]`; `I4_XS [d][scales_h][scales_l]`; `IQ3_XXS [d][scales]`; `IQ3_S [d][qh][signs][scales]`; `IQ2_XXS [d]`; `IQ2_XS [d][scales]`; `IQ2_S [d][signs][qh][scales]`; `IQ1_S [d][qh]`; `IQ1_M [qh][scales]`); value regions hold per-weight nibbles for the IQ4 types and packed index bytes over `[N, K/g]` otherwise; see `crates/r9v-format/src/iq.rs`.
Proposed resolution: State the exact SoA record contents, order and repack granularity per IQ family in §3.3 (field names with wire offsets, as SI-57 did for `I3_K`/`I2_K`).

## SI-71 — A2.4 — spec 2 §8
What: The §8 bits-per-weight table lists no per-type IQ row except `I4_XS` (4.25); the §3.3 table gives only the family range 1.5–3.5 for `IQ2`/`IQ3`/`IQ1` and nothing for `IQ4_NL`.
Why it blocks or misleads: Without exact per-type ratios, two implementations can report different model sizes for the same file and both claim §8 compliance; the load report (spec 2 §10) and budget math depend on these numbers.
Option taken: Exact wire-size ratios for all nine families (`I4_NL` 144/32 = 4.5; `I4_XS` 1088/256 = 4.25; `IQ3_XXS` 784/256 = 3.0625; `IQ3_S` 880/256 = 3.4375; `IQ2_XXS` 528/256 = 2.0625; `IQ2_XS` 592/256 = 2.3125; `IQ2_S` 656/256 = 2.5625; `IQ1_S` 400/256 = 1.5625; `IQ1_M` 448/256 = 1.75), all inside the §3.3 family range where one is stated; see `repack_bits_per_weight`.
Proposed resolution: Add one §8 row per IQ family with the exact wire-size ratio.

## SI-59 — A1.10 — spec 4 §10
What: §10 requires "32 random inputs per shape" for golden tests but names no seed scheme, seed registry, or stream-independence rule; ad-hoc per-test seeds risk cross-op correlation hiding shared-mode bugs.
Why it blocks or misleads: Two op families seeded identically (or with overlapping SplitMix streams) can pass golden while sharing a blind spot; without a stated derivation, "deterministic across runs" is unverifiable by anyone but the test author.
Option taken: `seed = xxh3("a1.10" | op_name | case_idx LE | master LE)` with one fixed `harness::MASTER_SEED`; see DECISION(A1.10) at `MASTER_SEED` in `crates/r9v-t0/src/harness.rs`.
Proposed resolution: State the seed-derivation rule and the single master-seed location in spec 4 §10.

## SI-60 — A1.10 — spec 1 §6.1
What: §6.1 states global floors (f16/bf16 abs 2e-3/rel 1e-2, i8 abs 5e-3/rel 2e-2) but assigns no floor per op/path: pure data movement (`copy`, `split`, `concat`, `gather_rows`), rank-1 collective identity, `sample` token ids, and `verify` decisions are bit-exact, while `scatter_add_rows` accumulates in f32 and cache round-trips carry grid error (f16/i8/e4m3).
Why it blocks or misleads: A harness that checks everything at f16/bf16 tolerance proves nothing about L0 identity paths; one that checks everything exact fails honest f32 accumulation and lossy cache stores.
Option taken: `Tolerance::for_op` maps exactly: `copy`, `split`, `concat`, `gather_rows`, `sample`, `verify` and all seven collectives to `exact()`; `logits_postprocess` and `scatter_add_rows` to `f32()`; `matmul`, `moe_ffn`, `embed_gather` to `i8_weight()`; and the remaining fourteen (`norm`, `rope`, `activation`, `act_mul`, `cast`, `residual_add`, `logit_softcap`, `quant_act`, `state_write_kv`, `attention`, `causal_conv1d`, `linear_attn_scan`, `ngram_gather`, `moe_route`) to `f16_bf16()`. Every row is an existing floor; none widens. The F16 `state_write_kv` gate uses its table row for cache-grid values while separately requiring slot-exact addressing and typed refusal; other cache dtypes use their existing `i8_weight()`/`e4m3_cache()` grid floors in dtype-specific tests. Gate cases resolve floors through fail-closed `tolerance_for` (`UnknownOp` on unknown names).
Proposed resolution: Add the per-op floor assignment to the §6.1 table (exact for movement/identity, f32 for f32 accumulation, grid bounds for cache stores).

## SI-61 — A1.10 — spec 1 §6.1 / App. B
What: §6.1 requires outputs "bit-identical regardless of which other tokens share the batch ... or the token's row index", but several ops are only defined relative to stable identity: rope needs the same position value, sampling needs the same `(seq, step, draw)` Philox keying, and attention/scan need the same slots and seq ids. Moving a row without its identity changes the correct answer.
Why it blocks or misleads: A literal "permute rows, expect identical bytes" test manufactures false L0 failures on correct code; an implementation that ignores positions/ids to satisfy it would be wrong.
Option taken: Batch-invariance runs fix the logical row's identity (positions, seq ids, slots, query bytes, RNG state) across the alone/padded/embedded runs and compare only that row's bytes; see `logical_row_bytes` in `crates/r9v-t0/src/harness.rs`.
Proposed resolution: State in §6.1/App. B that batch-invariance holds token identity fixed and compares the same logical row/sequence across batch shapes.

## SI-62 — A1.10 — spec 4 §10
What: §10 gates 4–5 (perf regression vs stored baseline, bandwidth/rate vs spec 11 §9.5 floors) and the T1 shape-fuzz draw count are written for device variants; T0-vs-self has no baseline, no device, and no stated T0-self fuzz count, while the card's done-when requires the harness to run T0 against itself for every op.
Why it blocks or misleads: Asserting a >3% regression with no baseline, or rate floors with no device, fails every T0-self run; inventing a T0 fuzz count invites under-testing.
Option taken: The harness runs T0-self through the L0/L1 gates only via one cohesive engine: each op family implements `GateCase` (fresh seeded inputs/state, impl call, independent-oracle comparison, logical-row extraction, explicit illegal inputs) and `run_gates` runs golden (exactly 32 seeds per legal shape), batch invariance (pinned logical row at differing row indices), determinism (twice from fresh state), and shape fuzz (deterministic legal shapes run; explicit illegal inputs must refuse typed). Every gate's legal shapes include single-token, padding-row, and max-bucket (4096) edges with tiny non-bucket dimensions (`bucket_edge_counts`, `MAX_BUCKET`); the operand-free `barrier` is the only max-bucket exemption, and illegal fuzz is explicit refusal cases rather than 32-per-shape. No timing is asserted; see DECISION(A1.10) at `CASES_PER_SHAPE` and `run_gates` in `crates/r9v-t0/src/harness.rs`. Mechanical proof: `gate_engine_covers_all_32_op_variants` (gate names match the tolerance rows exactly; max-bucket and illegal-case presence asserted per gate).
Proposed resolution: State that the perf/rate gates apply to T1/T2 only and fix the T0-self fuzz count and edge-shape list in §10.

## SI-72 — spoof-foundation — spec 1 App. A / spec 14 §3 — RESOLVED
What: App. A assigns ISA-invariant facts to `ArchDescriptor` and per-device quantities to `DeviceFacts` populated from discovery, and states "there is intentionally no checked-in R9700 constructor", but it never defines a reduced planning view (planning against a smaller card than the bench hardware), a provenance record distinguishing physical from constrained plans, or a canonical `ROC_GLOBAL_CU_MASK` derivation for reduced-CU launches.
Why it blocks or misleads: Without a stated home for these, each consumer (partitioner, loader, launcher) would invent its own reduction, masking, and naming rules, and a spoofed number could be mistaken for a measured fact in receipts.
Option taken: Added a typed foundation in `r9v-ir` (`SpoofProfileId` catalog, `EffectiveDeviceView` planning view beside the immutable physical descriptor, `Provenance` with qualified `MODEL (SPOOF)` display, `CuMask` lowest-N-bits canonical form, `PreQueueLaunchContract` as launcher-applied data plus pre-queue validation); hardened so the view is never a `DeviceDescriptor`, carries no measured/P2P facts, refuses official qualification use with typed `SpoofQualificationRefused`, and assigns no mask on exact-CU hardware. Recorded as D-007 (revised).
Proposed resolution: Resolved in spec 1 App. A ("Constrained planning views"), spec 5 §5.1 (planner consumes the view, records provenance), spec 9 §4.3/§10 (budgets refuse against the view bound; load report carries provenance), spec 11 §7/§9.4 (measurement fills physical descriptors only; receipts carry provenance and the disclaimer), spec 12 §4 (mask is launcher data, not config), and spec 14 §3 (pre-queue contract with the `r9v-hip` integration named as separate future work; no hard enforcement claimed). The `r9v-ir` surface is the standing contract.
