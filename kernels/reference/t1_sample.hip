// SPDX-License-Identifier: Apache-2.0
// T1 portable `sample` reference kernel (Spec 1 §4.F, §6.5; Spec 4 §2, §5.8).
//
// One workgroup per sequence, grid-strided over S. Counter-based Philox4x32
// draw at the stored draw index, inverse-CDF over the fixed-order probs
// (ascending scan, first index with cumsum >= u, default V-1 on rounding
// fallback exactly like T0), then the stored draw advances by one. Valid
// inputs are guaranteed by the host wrapper (mirrors T0 validation),
// including draw-index overflow refusal. Portable HIP only.
//
// The compiland is assembled by the runner test: generated ABI struct +
// `t1_sampling_common.hip` + this body, with `R9V_ARGS` replaced by the
// canonical A3.2 struct name and `R9V_V` by the exact static vocab size.

#include <hip/hip_runtime.h>

extern "C" __global__ void t1_sample(const R9V_ARGS args) {
    // Assume 256-byte alignment on device pointers (Spec 4 §7).
    const float *probs =
        (const float *)__builtin_assume_aligned(args.probs, 256);
    R9vRngWords *rng =
        (R9vRngWords *)__builtin_assume_aligned(args.rng_state, 256);
    const unsigned int *seq_ids =
        (const unsigned int *)__builtin_assume_aligned(args.seq_ids, 256);
    unsigned int *tokens =
        (unsigned int *)__builtin_assume_aligned(args.tokens, 256);
    const unsigned int S = args.s;
    const unsigned int V = (unsigned int)R9V_V;

    const unsigned int tid = threadIdx.x;

    for (unsigned int s_idx = blockIdx.x; s_idx < S;
         s_idx += gridDim.x) {
        if (tid == 0) {
            const unsigned long long seed = rng[s_idx].seed;
            const unsigned long long step = rng[s_idx].step;
            const unsigned int draw = (unsigned int)rng[s_idx].draw;
            const unsigned int seq_word = seq_ids[s_idx];
            const float u = r9v_draw_at(seed, seq_word, step, draw);

            // Inverse-CDF in ascending token order (Spec 1 §4.F). The
            // default is V-1 when rounding keeps cumsum below u, exactly
            // like T0 `sample` (not last-nonzero; that is `verify`'s
            // helper and differs).
            const float *row = probs + (unsigned long long)s_idx * V;
            float cumsum = 0.0f;
            unsigned int chosen = V - 1u;
            for (unsigned int tok = 0u; tok < V; ++tok) {
                cumsum += row[tok];
                if (cumsum >= u) {
                    chosen = tok;
                    break;
                }
            }
            tokens[s_idx] = chosen;
            // The host wrapper refused draw == u32::MAX before launch, so
            // the advance below cannot overflow (mirrors T0's checked add).
            rng[s_idx].draw = (unsigned long long)(draw + 1u);
        }
    }
}
