// SPDX-License-Identifier: Apache-2.0
// T1 portable `verify` reference kernel (Spec 1 §4.F, Spec 7 §4, §5; Spec 4
// §2, §5.8).
//
// One workgroup per sequence, grid-strided over S; the per-sequence walk is
// single-threaded and mirrors T0 op for op: rejection (`min(1, p/q)` with
// `norm(max(0, p-q))` resampling), greedy argmax match, and typical
// acceptance (`p > min(eps, delta * exp(-H)))`). Draw accounting is
// structural and distribution-exact: position i consumes Philox draw i, the
// terminal (replacement or bonus) token consumes draw k, and the stored draw
// advances by k+1. Tree drafts enumerate root-to-leaf paths in ascending
// lexicographic order (equivalent to T0's sort) and commit the longest
// accepted path, ties to the lowest first-token index. Valid inputs are
// guaranteed by the host wrapper (mirrors T0 validation). Portable HIP only.
//
// The compiland is assembled by the runner test: generated ABI struct +
// `t1_sampling_common.hip` + this body, with `R9V_ARGS` replaced by the
// canonical A3.2 struct name, `R9V_V` by the exact static vocab size,
// `R9V_VERIFY_METHOD` (0 rejection, 1 greedy, 2 typical),
// `R9V_VERIFY_TREE` (0/1), `R9V_HAS_DRAFT_PROBS` (0/1), and
// `R9V_VERIFY_EPS_BITS` / `R9V_VERIFY_DELTA_BITS` carrying the
// TypicalAcceptance floats bit-exact from the static.

#include <hip/hip_runtime.h>
#include <math.h>

// Local node cap: the spec-7 contract bounds k <= 16; the kernel supports
// wider trees and the host wrapper refuses beyond this cap with a typed
// error (mirrors T0 shape checks at the launch boundary).
#define R9V_VERIFY_MAXK 64

__device__ __forceinline__ unsigned int r9v_argmax_stable(const float *dist,
                                                         unsigned int v) {
    unsigned int best_idx = 0u;
    float best_val = -INFINITY;
    for (unsigned int tok = 0u; tok < v; ++tok) {
        const float val = dist[tok];
        if (val > best_val) {
            best_val = val;
            best_idx = tok;
        }
    }
    return best_idx;
}

// Inverse-CDF draw with T0 `sample_from_distribution` fallback semantics:
// ascending scan for cumsum >= u, else the last nonzero token, else 0.
__device__ __forceinline__ unsigned int r9v_inv_cdf(const float *dist,
                                                   unsigned int v, float u) {
    float cumsum = 0.0f;
    for (unsigned int tok = 0u; tok < v; ++tok) {
        cumsum += dist[tok];
        if (cumsum >= u) {
            return tok;
        }
    }
    for (unsigned int tok = v; tok > 0u; --tok) {
        if (dist[tok - 1u] > 0.0f) {
            return tok - 1u;
        }
    }
    return 0u;
}

extern "C" __global__ void t1_verify(const R9V_ARGS args) {
    // Assume 256-byte alignment on device pointers (Spec 4 §7).
    const unsigned int *draft_tokens =
        (const unsigned int *)__builtin_assume_aligned(args.draft_tokens, 256);
    const float *draft_probs = (R9V_HAS_DRAFT_PROBS && args.draft_probs)
        ? (const float *)__builtin_assume_aligned(args.draft_probs, 256)
        : nullptr;
    const float *target_probs =
        (const float *)__builtin_assume_aligned(args.target_probs, 256);
    R9vRngWords *rng =
        (R9vRngWords *)__builtin_assume_aligned(args.rng_state, 256);
    const unsigned int *seq_ids =
        (const unsigned int *)__builtin_assume_aligned(args.seq_ids, 256);
    // `tree_ancestors` rides the ABI struct but is not read here: T0 verify
    // consumes only `parents` (ancestors serve attention), so the reference
    // kernel binds the pointer without reading it.
    const int *tree_parents = nullptr;
#if R9V_VERIFY_TREE
    tree_parents =
        (const int *)__builtin_assume_aligned(args.tree_parents, 256);
#endif
    unsigned int *accepted =
        (unsigned int *)__builtin_assume_aligned(args.accepted, 256);
    unsigned int *accept_len =
        (unsigned int *)__builtin_assume_aligned(args.accept_len, 256);
    const unsigned int S = args.s;
    const unsigned int K = args.k;
    const unsigned int V = (unsigned int)R9V_V;
    const float eps = __uint_as_float((unsigned int)R9V_VERIFY_EPS_BITS);
    const float delta = __uint_as_float((unsigned int)R9V_VERIFY_DELTA_BITS);

    const unsigned int tid = threadIdx.x;
    const unsigned long long k1 = (unsigned long long)K + 1ull;

    for (unsigned int s_idx = blockIdx.x; s_idx < S;
         s_idx += gridDim.x) {
        if (tid != 0) {
            continue;
        }
        const unsigned long long seed = rng[s_idx].seed;
        const unsigned long long step = rng[s_idx].step;
        const unsigned int seq_word = seq_ids[s_idx];
        const unsigned int *d_tokens =
            draft_tokens + (unsigned long long)s_idx * K;
        const float *t_row = target_probs +
            (unsigned long long)s_idx * k1 * (unsigned long long)V;
        const float *d_row = (draft_probs != nullptr)
            ? draft_probs +
                (unsigned long long)s_idx * (unsigned long long)K * V
            : nullptr;
        unsigned int *out_acc =
            accepted + (unsigned long long)s_idx * k1;

        if (K == 0u) {
            // Degenerate k=0: single continuation token from the root
            // distribution (Spec 1 §4.F).
            const float *root = t_row;
            unsigned int bonus;
            if (R9V_VERIFY_METHOD == 1) {
                bonus = r9v_argmax_stable(root, V);
            } else {
                bonus = r9v_inv_cdf(
                    root, V, r9v_draw_at(seed, seq_word, step, 0u));
            }
            out_acc[0] = bonus;
            accept_len[s_idx] = 0u;
            rng[s_idx].draw += 1ull;
            continue;
        }

        // Root-to-leaf paths in ascending lexicographic order via iterative
        // DFS (roots ascending, children ascending): identical to folding
        // T0's sorted path list with the strictly-better rule, so the
        // longest accepted path wins with ties to the lowest first-token
        // index (Spec 7 §5).
        unsigned int best_len = 0u;
        unsigned int best_first = 0xFFFFFFFFu;
        unsigned int best_nodes[R9V_VERIFY_MAXK];
        unsigned int best_terminal = 0u;

        // Explicit DFS stack over (node, next scan cursor).
        int stack_node[R9V_VERIFY_MAXK + 1];
        unsigned int stack_cur[R9V_VERIFY_MAXK + 1];
        int stack_depth = -1;
        unsigned int path[R9V_VERIFY_MAXK];
        unsigned int path_len = 0u;
        // Seed the stack with roots in ascending order (push reversed).
        for (unsigned int j = K; j > 0u; --j) {
            bool is_root = (R9V_VERIFY_TREE == 0) ? (j - 1u == 0u)
                                                  : (tree_parents[j - 1u] < 0);
            if (is_root) {
                ++stack_depth;
                stack_node[stack_depth] = (int)(j - 1u);
                stack_cur[stack_depth] = 0u;
            }
        }

        while (stack_depth >= 0) {
            const int node = stack_node[stack_depth];
            // First visit extends the path; revisits (after backtracking)
            // only resume the child scan.
            const bool first = (stack_cur[stack_depth] == 0u);
            if (first) {
                path[path_len] = (unsigned int)node;
                ++path_len;
            }
            // Find the next child of `node` from the cursor.
            unsigned int child = K;
            if (R9V_VERIFY_TREE) {
                for (unsigned int j = stack_cur[stack_depth]; j < K; ++j) {
                    if (tree_parents[j] == node) {
                        child = j;
                        break;
                    }
                }
            } else {
                if (stack_cur[stack_depth] == 0u &&
                    (unsigned int)node + 1u < K) {
                    child = (unsigned int)node + 1u;
                }
            }
            if (child < K) {
                stack_cur[stack_depth] = child + 1u;
                ++stack_depth;
                stack_node[stack_depth] = (int)child;
                stack_cur[stack_depth] = 0u;
                continue;
            }
            // No (more) children from this node. Only a first visit with no
            // children at all is a root-to-leaf path; revisits are internal
            // nodes whose subtree is exhausted and must not be re-evaluated.
            if (!first) {
                --path_len;
                --stack_depth;
                continue;
            }
            // Leaf: evaluate the root-to-leaf `path`.
            {
                unsigned int acc_nodes[R9V_VERIFY_MAXK];
                unsigned int acc_count = 0u;
                unsigned int terminal = 0u;
                bool all_accepted = true;
                for (unsigned int pi = 0u; pi < path_len; ++pi) {
                    const unsigned int nidx = path[pi];
                    const unsigned int d = d_tokens[nidx];
                    const float *tdist;
                    if (pi == 0u) {
                        tdist = t_row;
                    } else {
                        const unsigned int parent = path[pi - 1u];
                        tdist = t_row +
                            ((unsigned long long)parent + 1ull) * V;
                    }
                    const float tp = tdist[d];
                    bool node_accepted = false;
                    if (R9V_VERIFY_METHOD == 0) {
                        float q;
                        if (d_row != nullptr) {
                            q = d_row[(unsigned long long)nidx * V + d];
                        } else {
                            q = 1.0f;
                        }
                        float alpha;
                        if (q > 0.0f) {
                            const float r = tp / q;
                            alpha = (r < 1.0f) ? r : 1.0f;
                        } else {
                            alpha = (tp > 0.0f) ? 1.0f : 0.0f;
                        }
                        const float u = r9v_draw_at(seed, seq_word, step,
                                                    nidx);
                        if (u < alpha) {
                            node_accepted = true;
                        } else {
                            // Replacement from norm(max(0, p-q)) with the
                            // reserved terminal draw k (Spec 1 §4.F). Three
                            // sequential passes over V with no staging
                            // (the accepted row owns only k+1 tokens):
                            // sum, then inverse-CDF over the normalized
                            // residual recomputed on the fly. Every float
                            // op runs in the same order as T0, so the
                            // result is bit-identical.
                            float rsum = 0.0f;
                            for (unsigned int tok = 0u; tok < V; ++tok) {
                                float qv;
                                if (d_row != nullptr) {
                                    qv = d_row[(unsigned long long)nidx *
                                                   V +
                                               tok];
                                } else {
                                    qv = (tok == d) ? 1.0f : 0.0f;
                                }
                                float rv = tdist[tok] - qv;
                                if (rv < 0.0f) {
                                    rv = 0.0f;
                                }
                                rsum += rv;
                            }
                            if (rsum > 0.0f && isfinite(rsum)) {
                                const float inv = 1.0f / rsum;
                                const float ru = r9v_draw_at(seed, seq_word,
                                                             step, K);
                                float cumsum = 0.0f;
                                terminal = V;
                                for (unsigned int tok = 0u; tok < V; ++tok) {
                                    float qv;
                                    if (d_row != nullptr) {
                                        qv = d_row[(unsigned long long)nidx *
                                                       V +
                                                   tok];
                                    } else {
                                        qv = (tok == d) ? 1.0f : 0.0f;
                                    }
                                    float rv = tdist[tok] - qv;
                                    if (rv < 0.0f) {
                                        rv = 0.0f;
                                    }
                                    cumsum += rv * inv;
                                    if (cumsum >= ru) {
                                        terminal = tok;
                                        break;
                                    }
                                }
                                if (terminal == V) {
                                    // Last-nonzero fallback, mirroring T0
                                    // `sample_from_distribution` (inv > 0,
                                    // so nonzero is decided by rv alone).
                                    terminal = 0u;
                                    for (unsigned int tok = V; tok > 0u;
                                         --tok) {
                                        float qv;
                                        if (d_row != nullptr) {
                                            qv = d_row[(unsigned long long)
                                                           nidx *
                                                           V +
                                                       tok - 1u];
                                        } else {
                                            qv = (tok - 1u == d) ? 1.0f
                                                                 : 0.0f;
                                        }
                                        if (tdist[tok - 1u] - qv > 0.0f) {
                                            terminal = tok - 1u;
                                            break;
                                        }
                                    }
                                }
                            } else {
                                terminal = r9v_inv_cdf(
                                    tdist, V,
                                    r9v_draw_at(seed, seq_word, step, K));
                            }
                        }
                    } else if (R9V_VERIFY_METHOD == 1) {
                        const unsigned int targ =
                            r9v_argmax_stable(tdist, V);
                        if (targ == d) {
                            node_accepted = true;
                        } else {
                            terminal = targ;
                        }
                    } else {
                        float h = 0.0f;
                        for (unsigned int tok = 0u; tok < V; ++tok) {
                            const float p = tdist[tok];
                            if (p > 0.0f) {
                                h -= p * logf(p);
                            }
                        }
                        float thr = delta * expf(-h);
                        if (eps < thr) {
                            thr = eps;
                        }
                        if (tp > thr) {
                            node_accepted = true;
                        } else {
                            terminal = r9v_inv_cdf(
                                tdist, V,
                                r9v_draw_at(seed, seq_word, step, K));
                        }
                    }
                    if (node_accepted) {
                        acc_nodes[acc_count] = nidx;
                        ++acc_count;
                    } else {
                        all_accepted = false;
                        break;
                    }
                }
                if (all_accepted) {
                    const unsigned int last = path[path_len - 1u];
                    const float *bdist =
                        t_row + ((unsigned long long)last + 1ull) * V;
                    if (R9V_VERIFY_METHOD == 1) {
                        terminal = r9v_argmax_stable(bdist, V);
                    } else {
                        terminal = r9v_inv_cdf(
                            bdist, V,
                            r9v_draw_at(seed, seq_word, step, K));
                    }
                }
                const unsigned int first = path[0];
                const bool better = (acc_count > best_len) ||
                    (acc_count == best_len && first < best_first);
                if (better) {
                    best_len = acc_count;
                    best_first = first;
                    for (unsigned int t = 0u; t < acc_count; ++t) {
                        best_nodes[t] = acc_nodes[t];
                    }
                    best_terminal = terminal;
                }
            }
            // Backtrack.
            --path_len;
            --stack_depth;
        }

        // Commit the winning path (Spec 7 §5). Evaluation above writes no
        // outputs (rejection resampling is staging-free), so the commit is
        // the first touch of `out_acc`.
        for (unsigned int t = 0u; t < best_len; ++t) {
            out_acc[t] = d_tokens[best_nodes[t]];
        }
        out_acc[best_len] = best_terminal;
        accept_len[s_idx] = best_len;
        // The host wrapper refused draw bases that would overflow u32 on a
        // k+1 advance, so this increment cannot wrap (mirrors T0's checked
        // add exactly over the states the wrapper admits).
        rng[s_idx].draw += k1;
    }
}
