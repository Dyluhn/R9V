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
