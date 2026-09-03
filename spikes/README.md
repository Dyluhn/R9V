# R9V Phase A0 Hardware Spikes

Companion programs to Phase A0 (Roadmap §A0, phase-a-agent-breakdown §A0.S1–S6).
Each spike program validates a foundational hardware or architectural assumption before dependent cards are executed.

## Spike Summary
1. **`wmma-l1`**: IU8 and IU4 WMMA direct global fragment loads in L1 lane order (Spec 1 App. A, Spec 2 §2).
2. **`dot4-gemv`**: `v_dot4_i32_i8` streaming memory bandwidth saturation at M in {1, 4, 8} (Spec 4 §5).
3. **`fp8-wmma`**: FP8 compiler builtin support and leaf wrapper asm requirements on ROCm 7.0.0 (Spec 4 §8).
4. **`hipgraph`**: Capture, instantiation, and replay stability/overhead for 400-launch graphs (Spec 6 §2).
5. **`direct-io`**: Pipelined O_DIRECT disk read to pinned buffer to H2D copy at queue depth 8 (Spec 9 §3).
6. **`p2p`**: Dual R9700 peer-to-peer memory access and 16 KB latency check (Spec 5 §6).
