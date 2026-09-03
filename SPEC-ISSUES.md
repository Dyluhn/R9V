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
