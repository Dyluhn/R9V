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

### D-007: Spoof-constrained planning views (not product qualification)
- **Date**: 2026-09-04 (hardened same day: view type, refusal gate, exact/reduced contract)
- **Scope**: Spec 1 App. A, Spec 5 §2/§5.1, Spec 9 §4.3/§10, Spec 11 §7/§9.4, Spec 12 §4, Spec 14 §3
- **Decision**: Planning against a smaller card than the bench hardware uses a distinct typed effective view (`r9v_ir::EffectiveDeviceView`) beside the truthful physical `DeviceDescriptor`, paired as `r9v_ir::ConstrainedDevice`. The view carries the shared ISA descriptor, the unchanged physical identity, and the reduced CU/VRAM bounds — and nothing else: it is not a `DeviceDescriptor`, carries no `measured` block and no `p2p` links, and offers no conversion into a bare descriptor, so physical measured performance can never travel as a spoof fact and provenance cannot be dropped. Construction refuses wrong-arch or under-resourced physical devices with collect-all errors. Provenance (`r9v_ir::Provenance`) is `Physical` or `Spoof`; spoof targets render only as `MODEL (SPOOF)`, preserve the physical identity, and any attempt to use a spoof result for official product qualification or a performance claim fails with the typed `SpoofQualificationRefused` refusal (`check_official_claim`) — a disclaimer string alone authorizes nothing. The pre-queue launch contract (`r9v_ir::PreQueueLaunchContract`) is data plus validation: exact-CU hardware needs no `ROC_GLOBAL_CU_MASK`, reduced-CU targets use the deterministic lowest-N-bits mask, the launcher applies it before starting the engine, and the engine validates it before HIP initialization. The mask narrows CU visibility only. The loader independently creates one shared `r9v_hip::AllocationBudget` per constrained physical identity from the effective VRAM bound and exclusively uses `BudgetedDeviceBuffer`; atomic reservations precede `hipMalloc`, roll back on failure, and return capacity only after `hipFree`, making the VRAM cap hard under concurrency. Initial catalog only: RX 9070 XT (SPOOF) 16 GiB/64 CU and RX 9070 (SPOOF) 16 GiB/56 CU, both gfx1201, dispatched by enum id/stable id with no product-name branches.
- **Reason**: Lets planner/loader work target purchasable-card bounds on larger bench hardware without forking discovery facts, faking identities, or scattering ad-hoc env-var logic through the engine, while making spoof-numbers-as-measured-facts a type error rather than a wording convention.
- **Alternative rejected**: Mutating discovered descriptors in place; cloning the descriptor as the "effective" view (the clone carried `measured`/`p2p` across and let callers drop provenance); branching on marketing-name strings at launch time; an empty-string mask sentinel (replaced by `Option`, forcing the launcher to branch explicitly).
