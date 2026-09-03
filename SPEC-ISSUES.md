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
