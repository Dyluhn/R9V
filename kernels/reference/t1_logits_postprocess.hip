// SPDX-License-Identifier: Apache-2.0
// T1 portable `logits_postprocess` reference kernel (Spec 1 §4.F, §6.5; Spec 4 §2, §5.8).
//
// One workgroup per (sequence, query) row, grid-strided over S*q rows. f32
// end to end; transform order bias → penalties → temperature → grammar mask
// → softmax; stable candidate order by (-logit, index) via a global-memory
// bitonic sort on the BitonicSort workspace; top-k/top-p/min-p filtering and
// renormalization single-threaded exactly like T0. Temperature zero takes the
// exact argmax path with lowest-index ties. Portable HIP only (see the
// common header for the intrinsic policy).
//
// The compiland is assembled by the runner test: generated ABI struct +
// `t1_sampling_common.hip` + this body, with `R9V_ARGS` replaced by the
// canonical A3.2 struct name and `R9V_V` by the exact static vocab size.
// `R9V_PP_HAS_HISTORY` / `R9V_PP_HAS_MASK` mirror the static presence flags.

#include <hip/hip_runtime.h>
#include <math.h>

struct R9vSortPair {
    float key;
    unsigned int idx;
};

extern "C" __global__ void t1_logits_postprocess(const R9V_ARGS args) {
    // Assume 256-byte alignment on device pointers (Spec 4 §7).
    const float *logits =
        (const float *)__builtin_assume_aligned(args.logits, 256);
    const R9vSamplingParams *params =
        (const R9vSamplingParams *)__builtin_assume_aligned(args.params, 256);
    const unsigned int *history = (R9V_PP_HAS_HISTORY && args.history_counts)
        ? (const unsigned int *)__builtin_assume_aligned(args.history_counts,
                                                         256)
        : nullptr;
    const unsigned char *mask = (R9V_PP_HAS_MASK && args.grammar_mask)
        ? (const unsigned char *)__builtin_assume_aligned(args.grammar_mask,
                                                          256)
        : nullptr;
    float *probs = (float *)__builtin_assume_aligned(args.probs, 256);
    unsigned char *workspace =
        (unsigned char *)__builtin_assume_aligned(args.workspace, 256);
    const unsigned int S = args.s;
    const unsigned int Q = args.q;
    const unsigned int V = (unsigned int)R9V_V;

    const unsigned int tid = threadIdx.x;
    const unsigned int B = blockDim.x;
    // Sort scratch: pad V up to a power of two; padded slots sort last.
    unsigned int npad = 1u;
    while (npad < V) {
        npad <<= 1u;
    }
    const unsigned long long row_stride = (unsigned long long)V;
    const unsigned long long mask_bytes =
        ((unsigned long long)V + 7ull) & ~7ull;
    const unsigned long long ws_row =
        (unsigned long long)npad * 8ull + mask_bytes;

    // Sized for any block up to 1024 threads; the runner test launches 256.
    __shared__ float lds[1024];
    __shared__ unsigned int lds_idx[1024];

    const unsigned long long rows = (unsigned long long)S * (unsigned long long)Q;
    for (unsigned long long row = (unsigned long long)blockIdx.x; row < rows;
         row += (unsigned long long)gridDim.x) {
        const unsigned int s_idx = (unsigned int)(row / (unsigned long long)Q);
        const R9vSamplingParams P = params[s_idx];
        const float *logit_row = logits + row * row_stride;
        float *out = probs + row * row_stride;
        R9vSortPair *pairs =
            (R9vSortPair *)(workspace + row * ws_row);
        unsigned char *cand_mask =
            workspace + row * ws_row + (unsigned long long)npad * 8ull;

        // Phase A: copy logits, then sparse bias (single thread), penalties,
        // temperature scaling, and the grammar mask. Valid inputs are
        // guaranteed by the host wrapper (mirrors T0 validation).
        for (unsigned int i = tid; i < V; i += B) {
            out[i] = logit_row[i];
        }
        __syncthreads();
        if (tid == 0) {
            const R9vBiasPair *bpairs =
                (const R9vBiasPair *)(params + S);
            for (unsigned int b = 0; b < P.bias_count; ++b) {
                const R9vBiasPair bp = bpairs[P.bias_start + b];
                out[bp.token] += bp.bias;
            }
        }
        __syncthreads();
        const float inv_temp =
            (P.temperature > 0.0f) ? (1.0f / P.temperature) : 1.0f;
        for (unsigned int i = tid; i < V; i += B) {
            float l = out[i];
            if (R9V_PP_HAS_HISTORY && history != nullptr &&
                history[(unsigned long long)s_idx * (unsigned long long)V + i] >
                    0u) {
                const unsigned int count =
                    history[(unsigned long long)s_idx *
                                (unsigned long long)V +
                            i];
                if (P.repetition_penalty != 1.0f) {
                    l = (l > 0.0f) ? (l / P.repetition_penalty)
                                   : (l * P.repetition_penalty);
                }
                if (P.presence_penalty != 0.0f) {
                    l -= P.presence_penalty;
                }
                if (P.frequency_penalty != 0.0f) {
                    l -= (float)count * P.frequency_penalty;
                }
            }
            if (P.temperature > 0.0f) {
                l *= inv_temp;
            }
            if (R9V_PP_HAS_MASK && mask != nullptr &&
                mask[row * row_stride + i] == 0u) {
                l = -INFINITY;
            }
            out[i] = l;
        }
        __syncthreads();

        if (P.temperature == 0.0f) {
            // Exact argmax with stable lowest-index ties (Spec 1 §4.F).
            float best = -INFINITY;
            unsigned int best_idx = 0u;
            for (unsigned int i = tid; i < V; i += B) {
                const float l = out[i];
                if (l > best) {
                    best = l;
                    best_idx = i;
                }
            }
            lds[tid] = best;
            lds_idx[tid] = best_idx;
            __syncthreads();
            for (unsigned int stride = B >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    const float a = lds[tid];
                    const float b = lds[tid + stride];
                    const unsigned int ia = lds_idx[tid];
                    const unsigned int ib = lds_idx[tid + stride];
                    if (b > a || (b == a && ib < ia)) {
                        lds[tid] = b;
                        lds_idx[tid] = ib;
                    }
                }
                __syncthreads();
            }
            const unsigned int winner = lds_idx[0];
            for (unsigned int i = tid; i < V; i += B) {
                out[i] = 0.0f;
            }
            __syncthreads();
            if (tid == 0) {
                out[winner] = 1.0f;
            }
            __syncthreads();
            continue;
        }

        // Candidate keys captured from the transformed logits BEFORE softmax
        // (monotonicity of expf does not imply identical tie structure after
        // rounding); masked tokens take +Inf so they sort after candidates.
        for (unsigned int i = tid; i < npad; i += B) {
            R9vSortPair e;
            if (i < V) {
                const float l = out[i];
                e.key = (l > -INFINITY) ? -l : INFINITY;
                e.idx = i;
            } else {
                e.key = INFINITY;
                e.idx = 0xFFFFFFFFu;
            }
            pairs[i] = e;
        }
        __syncthreads();

        // Softmax: fixed-order parallel max, parallel exp, fixed-order
        // parallel sum, parallel normalize. Reductions are deterministic;
        // per-element ops match T0 bit for bit.
        float part_max = -INFINITY;
        for (unsigned int i = tid; i < V; i += B) {
            const float l = out[i];
            if (l > -INFINITY && l > part_max) {
                part_max = l;
            }
        }
        lds[tid] = part_max;
        __syncthreads();
        for (unsigned int stride = B >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float b = lds[tid + stride];
                if (b > lds[tid]) {
                    lds[tid] = b;
                }
            }
            __syncthreads();
        }
        const float row_max = lds[0];
        __syncthreads();
        for (unsigned int i = tid; i < V; i += B) {
            const float l = out[i];
            out[i] = (l > -INFINITY) ? expf(l - row_max) : 0.0f;
        }
        __syncthreads();
        float part_sum = 0.0f;
        for (unsigned int i = tid; i < V; i += B) {
            part_sum += out[i];
        }
        lds[tid] = part_sum;
        __syncthreads();
        for (unsigned int stride = B >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                lds[tid] += lds[tid + stride];
            }
            __syncthreads();
        }
        const float inv_sum = 1.0f / lds[0];
        __syncthreads();
        for (unsigned int i = tid; i < V; i += B) {
            out[i] *= inv_sum;
        }
        __syncthreads();

        // Bitonic sort of the captured (-logit, index) keys over the padded
        // row. The comparator reproduces T0's `sort_by((-logit, index))`
        // exactly, including -0.0/+0.0 ties.
        for (unsigned int k = 2u; k <= npad; k <<= 1u) {
            for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
                for (unsigned int i = tid; i < npad; i += B) {
                    const unsigned int l = i ^ j;
                    if (l > i) {
                        const R9vSortPair a = pairs[i];
                        const R9vSortPair b = pairs[l];
                        bool do_swap;
                        if ((i & k) == 0u) {
                            do_swap = r9v_pair_less(b.key, b.idx, a.key,
                                                   a.idx);
                        } else {
                            do_swap = r9v_pair_less(a.key, a.idx, b.key,
                                                   b.idx);
                        }
                        if (do_swap) {
                            pairs[i] = b;
                            pairs[l] = a;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Filtering + renormalization, single-threaded exactly like T0.
        if (tid == 0) {
            unsigned int n = 0u;
            for (unsigned int t = 0u; t < npad; ++t) {
                const R9vSortPair e = pairs[t];
                if (e.idx < V && e.key < INFINITY) {
                    pairs[n].idx = e.idx;
                    ++n;
                }
            }
            if (P.top_k > 0u && P.top_k < n) {
                n = P.top_k;
            }
            if (P.top_p < 1.0f) {
                float cumsum = 0.0f;
                unsigned int keep = 0u;
                for (unsigned int t = 0u; t < n; ++t) {
                    cumsum += out[pairs[t].idx];
                    ++keep;
                    if (cumsum >= P.top_p) {
                        break;
                    }
                }
                n = (keep > 1u) ? keep : 1u;
            }
            if (P.min_p > 0.0f && n > 0u) {
                const float p_max = out[pairs[0].idx];
                const float thresh = P.min_p * p_max;
                unsigned int m = 0u;
                for (unsigned int t = 0u; t < n; ++t) {
                    if (out[pairs[t].idx] >= thresh) {
                        pairs[m].idx = pairs[t].idx;
                        ++m;
                    }
                }
                n = (m > 0u) ? m : 1u;
            }
            float cand_sum = 0.0f;
            for (unsigned int t = 0u; t < n; ++t) {
                cand_sum += out[pairs[t].idx];
            }
            if (cand_sum > 0.0f && isfinite(cand_sum)) {
                const float inv_cand = 1.0f / cand_sum;
                for (unsigned int i = 0u; i < V; ++i) {
                    cand_mask[i] = 0u;
                }
                for (unsigned int t = 0u; t < n; ++t) {
                    cand_mask[pairs[t].idx] = 1u;
                }
                for (unsigned int i = 0u; i < V; ++i) {
                    out[i] = (cand_mask[i] != 0u) ? (out[i] * inv_cand)
                                                  : 0.0f;
                }
            } else {
                for (unsigned int i = 0u; i < V; ++i) {
                    out[i] = 0.0f;
                }
                out[pairs[0].idx] = 1.0f;
            }
        }
        __syncthreads();
    }
}
