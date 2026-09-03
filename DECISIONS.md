# Architectural and Specification Decisions

Decisions that establish or modify principles across specs, recorded with date, reason, and alternative rejected per Spec 15 §9.

## Seeded Decisions (Spec 15 §9)

### D-001: Placement resolved per load
- **Date**: 2026-09-02
- **Scope**: Spec 5 §3.4, Spec 9 §1
- **Decision**: Placement (`Device(rank)`, `Host`, `Tiered`) is resolved at load time by the planner based on discovered topology, available VRAM, and target model shape, rather than baked into model artifacts.
- **Reason**: Allows a single model checkpoint or weight file to execute across differing hardware configurations (e.g. single R9700, dual R9700 TP2/PP2, or host-assisted offload) without requiring format conversion or re-export.
- **Alternative rejected**: Statically encoding device or host placement decisions into the GGUF/native model file or graph builder definitions.

### D-002: Pipeline Parallel (PP) only bit-identity
- **Date**: 2026-09-02
- **Scope**: Spec 1 §6.1, Spec 5 §7
- **Decision**: Bit-identical output invariance across rank topologies is guaranteed under Pipeline Parallelism (PP), but not across varying Tensor Parallelism (TP) rank configurations.
- **Reason**: PP maintains identical reduction order and sequential arithmetic across layers. TP splits matrix multiplications and performs cross-rank all-reduce summations whose associative reordering causes minor floating-point divergence.
- **Alternative rejected**: Mandating bit-identity across varying TP topologies, which would require software-emulated deterministic reduction trees that severely compromise RDNA4 throughput.

### D-003: Host-computed cold experts for MoE
- **Date**: 2026-09-02
- **Scope**: Spec 5 §3.4, Spec 6 §3.2
- **Decision**: Cold experts that exceed device VRAM are computed on the host CPU using T0v SIMD worker threads via segment pipelining (D2H router output -> CPU compute -> H2D recombine), running concurrently with hot device experts.
- **Reason**: Enables running large Mixture-of-Experts models whose aggregate parameter count exceeds 32GB/64GB VRAM without silent OOM or high-latency PCIe weight-swapping thrashing.
- **Alternative rejected**: Hard-failing models that do not fit entirely in VRAM, or dynamically page-swapping expert weights over PCIe on each token step.

### D-004: I4_K field-identical to GGUF Q4_K
- **Date**: 2026-09-02
- **Scope**: Spec 2 §3.3, Spec 13 §2
- **Decision**: The native `I4_K` block format is field-identical to GGUF `Q4_K` (256-superblock with eight 32-blocks, packed 12-byte header: `d: f16`, `dmin: f16`, `sc[8]: u6`, `mn[8]: u6`).
- **Reason**: Enables a single optimized GPU kernel to serve both native R9V format and repacked GGUF checkpoints, ensuring that quality improvements from R9V quantization (Hessian-weighted GPTQ rounding, folded smoothing) can be cleanly A/B tested against llama.cpp Q4_K_M in the exact same kernel.
- **Alternative rejected**: Defining a novel, proprietary 4-bit block layout for R9V that would prevent zero-repack parity testing.

### D-005: Unified step graph with decode and prefill classes
- **Date**: 2026-09-02
- **Scope**: Spec 1 §3.1, Spec 6 §2
- **Decision**: Decode and prefill tokens execute through a single unified step graph bucketed on `(S, T_dec, T_pre)` using execution classes, rather than maintaining two independent graph representations.
- **Reason**: Greatly simplifies scheduler state, memory allocation, and hipGraph lifecycle; enables mixed batches containing both ongoing decodes and chunked prefills in a single step launch.
- **Alternative rejected**: Constructing and maintaining separate prefill and decode graph builders, executors, and captured hipGraphs.

### D-006: Speculative decoding draft cost charged to spec-decode budget
- **Date**: 2026-09-02
- **Scope**: Spec 6 §2.4, Spec 7 §4
- **Decision**: Draft model compute and latency cost is charged dynamically against the overall speculative decoding budget rather than enforcing a rigid, fixed pre-step budget cap.
- **Reason**: Enables adaptive draft length based on runtime acceptance rates and step latency headroom, maximizing speculative speedup across varying prompt and context lengths.
- **Alternative rejected**: Imposing a fixed per-step pre-allocated execution cap for draft steps.
