// SPDX-License-Identifier: Apache-2.0
//! Deterministic T0 reference implementation of sampling and verification operations
//! (Spec 1 §4.F, §6.5, Spec 7 §4, §5).

use r9v_ir::{SamplingParams, TreeMask, VerifyMethod};

use crate::error::T0Error;
use crate::philox::RngState;

/// Output of speculative verification (Spec 1 §4.F, Spec 7 §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyOutput {
    /// Accepted draft tokens followed by the terminal bonus or replacement token `[S, k+1]`.
    pub accepted: Vec<u32>,
    /// Number of accepted draft tokens per sequence `[S]`.
    pub accept_len: Vec<u32>,
}

// DECISION(A1.8): logits_postprocess applies top-k, top-p, and min-p filtering to candidate probabilities in stable sorted order by (-logit, index), renormalizing surviving candidates to sum to 1.0; at temperature == 0, argmax with stable tie-break by lowest index receives 1.0 and all other tokens 0.0; rejected non-renormalized probabilities because downstream sample expects a valid CDF. Spec 1 §4.F, Spec 1 §6.5.
/// Post-processes raw logits into normalized probabilities per sequence (Spec 1 §4.F, §6.5).
///
/// Applies `logit_bias` (before temperature), `history_counts` penalties (repetition, presence,
/// frequency), `temperature`, `grammar_mask` (before softmax), stable sort by `(-logit, index)`,
/// `top_k`, `top_p`, and `min_p` filtering, returning normalized probabilities `[S, q, V]`.
// Spec 1 §4.F defines logits_postprocess over (logits, S, q, V, params, history_counts?, grammar_mask?, out_probs).
#[allow(clippy::too_many_arguments)]
pub fn logits_postprocess(
    logits: &[f32],
    s: usize,
    q: usize,
    v: usize,
    params: &[SamplingParams],
    history_counts: Option<&[u32]>,
    grammar_mask: Option<&[bool]>,
    out_probs: &mut [f32],
) -> Result<(), T0Error> {
    let expected_len = s
        .checked_mul(q)
        .and_then(|sq| sq.checked_mul(v))
        .ok_or_else(|| T0Error::ShapeLengthMismatch {
            op: "logits_postprocess",
            tensor: "logits",
            expected: usize::MAX,
            got: logits.len(),
            detail: "overflow computing S * q * V".to_string(),
        })?;

    if logits.len() != expected_len {
        return Err(T0Error::ShapeLengthMismatch {
            op: "logits_postprocess",
            tensor: "logits",
            expected: expected_len,
            got: logits.len(),
            detail: format!("logits length != S({s}) * q({q}) * V({v})"),
        });
    }

    if out_probs.len() != expected_len {
        return Err(T0Error::ShapeLengthMismatch {
            op: "logits_postprocess",
            tensor: "out_probs",
            expected: expected_len,
            got: out_probs.len(),
            detail: format!("out_probs length != S({s}) * q({q}) * V({v})"),
        });
    }

    if params.len() != s {
        return Err(T0Error::ShapeLengthMismatch {
            op: "logits_postprocess",
            tensor: "params",
            expected: s,
            got: params.len(),
            detail: format!("SamplingParams count != batch size S({s})"),
        });
    }

    if s == 0 || q == 0 || v == 0 {
        return Err(T0Error::EmptyInput {
            op: "logits_postprocess",
            tensor: if s == 0 {
                "S"
            } else if q == 0 {
                "q"
            } else {
                "V"
            },
        });
    }

    // Validate all sampling params first (collect all violations per CONVENTIONS.md §1.4).
    let mut parameter_problems = Vec::new();
    for p in params {
        if let Err(error) = p.validate() {
            parameter_problems.push(T0Error::Ir(error));
        }
        for (position, &(token, _)) in p.logit_bias.iter().enumerate() {
            if token as usize >= v {
                parameter_problems.push(T0Error::TokenOutOfRange {
                    op: "logits_postprocess",
                    tensor: "logit_bias",
                    position,
                    token,
                    vocab_size: v,
                });
            }
        }
    }
    T0Error::from_typed_problems(parameter_problems)?;

    if let Some(history) = history_counts {
        let expected_hist = s * v;
        if history.len() != expected_hist {
            return Err(T0Error::ShapeLengthMismatch {
                op: "logits_postprocess",
                tensor: "history_counts",
                expected: expected_hist,
                got: history.len(),
                detail: format!("history_counts length != S({s}) * V({v})"),
            });
        }
    }

    if let Some(mask) = grammar_mask {
        if mask.len() != expected_len {
            return Err(T0Error::ShapeLengthMismatch {
                op: "logits_postprocess",
                tensor: "grammar_mask",
                expected: expected_len,
                got: mask.len(),
                detail: format!("grammar_mask length != S({s}) * q({q}) * V({v})"),
            });
        }
    }

    let mut work_logits = vec![0.0f32; v];
    let mut exp_vals = vec![0.0f32; v];
    let mut cand_mask = vec![false; v];

    for s_idx in 0..s {
        let p = &params[s_idx];
        let hist_row = history_counts.map(|h| &h[s_idx * v..(s_idx + 1) * v]);

        for q_idx in 0..q {
            let offset = (s_idx * q + q_idx) * v;
            let logit_slice = &logits[offset..offset + v];
            let out_slice = &mut out_probs[offset..offset + v];
            let mask_slice = grammar_mask.map(|m| &m[offset..offset + v]);

            work_logits.copy_from_slice(logit_slice);

            // 1. Sparse logit bias added before temperature (Spec 1 §4.F)
            for &(token_id, bias) in &p.logit_bias {
                let idx = token_id as usize;
                work_logits[idx] += bias;
            }

            // 2. Penalties applied to tokens with non-zero history counts (Spec 1 §4.F)
            if let Some(hist) = hist_row {
                for (tok, &count) in hist.iter().enumerate().take(v) {
                    if count > 0 {
                        let l = work_logits[tok];
                        // Repetition penalty
                        if p.repetition_penalty != 1.0 {
                            work_logits[tok] = if l > 0.0 {
                                l / p.repetition_penalty
                            } else {
                                l * p.repetition_penalty
                            };
                        }
                        // Presence penalty (additive per token present in history)
                        if p.presence_penalty != 0.0 {
                            work_logits[tok] -= p.presence_penalty;
                        }
                        // Frequency penalty (additive proportional to occurrence count)
                        if p.frequency_penalty != 0.0 {
                            work_logits[tok] -= (count as f32) * p.frequency_penalty;
                        }
                    }
                }
            }

            // 3. Temperature scaling (Spec 1 §4.F)
            if p.temperature > 0.0 {
                let inv_temp = 1.0 / p.temperature;
                for l in &mut work_logits {
                    *l *= inv_temp;
                }
            }

            // 4. Grammar mask applied after penalties and before softmax (Spec 1 §4.F)
            if let Some(mask) = mask_slice {
                for (tok, &allowed) in mask.iter().enumerate().take(v) {
                    if !allowed {
                        work_logits[tok] = f32::NEG_INFINITY;
                    }
                }
            }

            // Verify at least one unmasked token exists
            let unmasked_count = work_logits
                .iter()
                .filter(|&&l| l > f32::NEG_INFINITY)
                .count();
            if unmasked_count == 0 {
                return Err(T0Error::AllTokensMasked {
                    seq: s_idx,
                    query: q_idx,
                    vocab_size: v,
                });
            }

            // 5. Softmax & filtering
            if p.temperature == 0.0 {
                // Greedy argmax with fixed stable tie-break by lowest index (Spec 1 §4.F, §6.5)
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for (tok, &l) in work_logits.iter().enumerate() {
                    if l > best_val {
                        best_val = l;
                        best_idx = tok;
                    }
                }
                out_slice.fill(0.0);
                out_slice[best_idx] = 1.0;
            } else {
                // Numerically stable softmax
                let max_l = work_logits
                    .iter()
                    .copied()
                    .filter(|&l| l > f32::NEG_INFINITY)
                    .fold(f32::NEG_INFINITY, f32::max);

                let mut sum_exp = 0.0f32;
                for tok in 0..v {
                    let l = work_logits[tok];
                    if l > f32::NEG_INFINITY {
                        let e = (l - max_l).exp();
                        exp_vals[tok] = e;
                        sum_exp += e;
                    } else {
                        exp_vals[tok] = 0.0;
                    }
                }

                if sum_exp <= 0.0 || !sum_exp.is_finite() {
                    return Err(T0Error::InvalidDistribution {
                        op: "logits_postprocess",
                        seq: s_idx,
                        pos: q_idx,
                        sum: sum_exp,
                    });
                }

                let inv_sum = 1.0 / sum_exp;
                for tok in 0..v {
                    out_slice[tok] = exp_vals[tok] * inv_sum;
                }

                // Fixed stable sort by (-logit, index) (Spec 1 §4.F, §6.5)
                let mut candidates: Vec<usize> = (0..v)
                    .filter(|&i| work_logits[i] > f32::NEG_INFINITY)
                    .collect();

                candidates.sort_by(|&a, &b| {
                    (-work_logits[a], a)
                        .partial_cmp(&(-work_logits[b], b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Top-k filtering (Spec 1 §4.F)
                if p.top_k > 0 && (p.top_k as usize) < candidates.len() {
                    candidates.truncate(p.top_k as usize);
                }

                // Top-p nucleus filtering (Spec 1 §4.F)
                if p.top_p < 1.0 {
                    let mut cumsum = 0.0f32;
                    let mut keep = 0;
                    for &idx in &candidates {
                        cumsum += out_slice[idx];
                        keep += 1;
                        if cumsum >= p.top_p {
                            break;
                        }
                    }
                    candidates.truncate(keep.max(1));
                }

                // Min-p filtering (Spec 1 §4.F)
                if p.min_p > 0.0 && !candidates.is_empty() {
                    let p_max = out_slice[candidates[0]];
                    let thresh = p.min_p * p_max;
                    let filtered: Vec<usize> = candidates
                        .iter()
                        .copied()
                        .filter(|&idx| out_slice[idx] >= thresh)
                        .collect();
                    if !filtered.is_empty() {
                        candidates = filtered;
                    } else {
                        candidates.truncate(1);
                    }
                }

                // Renormalize surviving candidates
                cand_mask.fill(false);
                let mut cand_sum = 0.0f32;
                for &idx in &candidates {
                    cand_mask[idx] = true;
                    cand_sum += out_slice[idx];
                }

                if cand_sum > 0.0 && cand_sum.is_finite() {
                    let inv_cand_sum = 1.0 / cand_sum;
                    for tok in 0..v {
                        if cand_mask[tok] {
                            out_slice[tok] *= inv_cand_sum;
                        } else {
                            out_slice[tok] = 0.0;
                        }
                    }
                } else {
                    out_slice.fill(0.0);
                    out_slice[candidates[0]] = 1.0;
                }
            }
        }
    }

    Ok(())
}

/// Stochastic or greedy inverse-CDF token sampling op (Spec 1 §4.F).
///
/// Draws tokens from `probs [S, V]` using Philox4x32 keyed by `(seq_id, step, draw_index)`
/// via inverse-CDF. Advances each sequence's RNG draw counter by 1.
pub fn sample(
    probs: &[f32],
    s: usize,
    v: usize,
    rng_states: &mut [RngState],
) -> Result<Vec<u32>, T0Error> {
    if s == 0 || v == 0 {
        return Err(T0Error::EmptyInput {
            op: "sample",
            tensor: if s == 0 { "S" } else { "V" },
        });
    }

    let expected_len = s
        .checked_mul(v)
        .ok_or_else(|| T0Error::ShapeLengthMismatch {
            op: "sample",
            tensor: "probs",
            expected: usize::MAX,
            got: probs.len(),
            detail: "overflow computing S * V".to_string(),
        })?;

    if probs.len() != expected_len {
        return Err(T0Error::ShapeLengthMismatch {
            op: "sample",
            tensor: "probs",
            expected: expected_len,
            got: probs.len(),
            detail: format!("probs length != S({s}) * V({v})"),
        });
    }

    if rng_states.len() != s {
        return Err(T0Error::ShapeLengthMismatch {
            op: "sample",
            tensor: "rng_states",
            expected: s,
            got: rng_states.len(),
            detail: format!("rng_states count != S({s})"),
        });
    }

    for s_idx in 0..s {
        validate_distribution(&probs[s_idx * v..(s_idx + 1) * v], "sample", s_idx, 0)?;
    }

    let mut tokens = Vec::with_capacity(s);

    for (s_idx, rng) in rng_states.iter_mut().enumerate().take(s) {
        let p_row = &probs[s_idx * v..(s_idx + 1) * v];

        // Draw uniform random float u in (0, 1) and advance RNG state by 1
        let u = rng.draw_uniform_f32();

        let mut cumsum = 0.0f32;
        let mut chosen = v - 1;
        for (tok, &p) in p_row.iter().enumerate() {
            cumsum += p;
            if cumsum >= u {
                chosen = tok;
                break;
            }
        }
        tokens.push(chosen as u32);
    }

    Ok(tokens)
}

// DECISION(A1.8): verify uses Philox draw index i for candidate acceptance test at position i in 0..k-1, and draw index k for the terminal token (bonus token if all accepted, or residual replacement token if rejected at i); rejected using draw i+1 for replacement because candidate i+1's uniform draw is reserved for position i+1 per Spec 7 §4. Spec 1 §4.F, Spec 4 §5.8, Spec 7 §4.
// DECISION(A1.8): In tree verify, draft node j is tested against target_probs[0] if parents[j] == -1, or target_probs[p + 1] if parents[j] == p >= 0; bonus token after node j is sampled from target_probs[j + 1]; rejected separate tree probability indexing because target_probs[k+1, V] aligns root at index 0 and node j output at j+1 identically to linear verify. Spec 1 §4.F, Spec 7 §4, §5.
// DECISION(A1.8): Typical acceptance accepts d_i if p_i[d_i] > min(eps, delta * exp(-H(p_i))) per Spec 7 §4; on rejection at i, replacement is sampled from target_probs[i] using Philox draw k; bonus token at k is sampled from target_probs[k]; rejected discarding target_probs[i] because speculative decode must emit a verified continuation token. Spec 7 §4.
/// Speculative decoding acceptance verification op (Spec 1 §4.F, Spec 7 §4, §5).
///
/// Evaluates draft candidates against target probabilities using `Rejection`, `Greedy`,
/// or `TypicalAcceptance` policies. Supports both linear verification chains and tree drafts
/// via `TreeMask`. Commits the longest accepted path (breaking ties by lowest first-token index)
/// and samples the terminal replacement or bonus token.
// Spec 1 §4.F and Spec 7 §4 define verify over (draft_tokens, draft_probs?, target_probs, S, k, V, method, rng_states, tree?).
#[allow(clippy::too_many_arguments)]
pub fn verify(
    draft_tokens: &[u32],
    draft_probs: Option<&[f32]>,
    target_probs: &[f32],
    s: usize,
    k: usize,
    v: usize,
    method: &VerifyMethod,
    rng_states: &mut [RngState],
    tree: Option<&TreeMask>,
) -> Result<VerifyOutput, T0Error> {
    if s == 0 || v == 0 {
        return Err(T0Error::EmptyInput {
            op: "verify",
            tensor: if s == 0 { "S" } else { "V" },
        });
    }

    let k_plus_one = k
        .checked_add(1)
        .ok_or_else(|| T0Error::ShapeLengthMismatch {
            op: "verify",
            tensor: "k",
            expected: u32::MAX as usize,
            got: k,
            detail: "k + 1 overflows usize".to_string(),
        })?;
    let terminal_draw = u32::try_from(k).map_err(|_| T0Error::ShapeLengthMismatch {
        op: "verify",
        tensor: "k",
        expected: u32::MAX as usize,
        got: k,
        detail: "k cannot be represented by the Philox draw index".to_string(),
    })?;
    let draw_advance =
        terminal_draw
            .checked_add(1)
            .ok_or_else(|| T0Error::ShapeLengthMismatch {
                op: "verify",
                tensor: "k",
                expected: (u32::MAX - 1) as usize,
                got: k,
                detail: "k + 1 cannot be represented by the Philox draw index".to_string(),
            })?;

    let expected_draft = s
        .checked_mul(k)
        .ok_or_else(|| T0Error::ShapeLengthMismatch {
            op: "verify",
            tensor: "draft_tokens",
            expected: usize::MAX,
            got: draft_tokens.len(),
            detail: "overflow computing S * k".to_string(),
        })?;

    if draft_tokens.len() != expected_draft {
        return Err(T0Error::ShapeLengthMismatch {
            op: "verify",
            tensor: "draft_tokens",
            expected: expected_draft,
            got: draft_tokens.len(),
            detail: format!("draft_tokens length != S({s}) * k({k})"),
        });
    }

    for (position, &token) in draft_tokens.iter().enumerate() {
        if token as usize >= v {
            return Err(T0Error::TokenOutOfRange {
                op: "verify",
                tensor: "draft_tokens",
                position,
                token,
                vocab_size: v,
            });
        }
    }

    if let Some(dp) = draft_probs {
        let expected_dp =
            expected_draft
                .checked_mul(v)
                .ok_or_else(|| T0Error::ShapeLengthMismatch {
                    op: "verify",
                    tensor: "draft_probs",
                    expected: usize::MAX,
                    got: dp.len(),
                    detail: "overflow computing S * k * V".to_string(),
                })?;
        if dp.len() != expected_dp {
            return Err(T0Error::ShapeLengthMismatch {
                op: "verify",
                tensor: "draft_probs",
                expected: expected_dp,
                got: dp.len(),
                detail: format!("draft_probs length != S({s}) * k({k}) * V({v})"),
            });
        }
    }

    let expected_target = s
        .checked_mul(k_plus_one)
        .and_then(|sk1| sk1.checked_mul(v))
        .ok_or_else(|| T0Error::ShapeLengthMismatch {
            op: "verify",
            tensor: "target_probs",
            expected: usize::MAX,
            got: target_probs.len(),
            detail: "overflow computing S * (k+1) * V".to_string(),
        })?;

    if target_probs.len() != expected_target {
        return Err(T0Error::ShapeLengthMismatch {
            op: "verify",
            tensor: "target_probs",
            expected: expected_target,
            got: target_probs.len(),
            detail: format!(
                "target_probs length != S({s}) * (k+1)({sk1}) * V({v})",
                sk1 = k_plus_one
            ),
        });
    }

    if rng_states.len() != s {
        return Err(T0Error::ShapeLengthMismatch {
            op: "verify",
            tensor: "rng_states",
            expected: s,
            got: rng_states.len(),
            detail: format!("rng_states count != S({s})"),
        });
    }

    if let Some(tree_mask) = tree {
        if tree_mask.t() != k {
            return Err(T0Error::ShapeLengthMismatch {
                op: "verify",
                tensor: "tree",
                expected: k,
                got: tree_mask.t(),
                detail: format!("TreeMask token count T({}) != k({k})", tree_mask.t()),
            });
        }
    }

    if let Some(dp) = draft_probs {
        for s_idx in 0..s {
            for pos in 0..k {
                let offset = (s_idx * k + pos) * v;
                validate_distribution(&dp[offset..offset + v], "verify", s_idx, pos)?;
            }
        }
    }
    for s_idx in 0..s {
        for pos in 0..k_plus_one {
            let offset = (s_idx * k_plus_one + pos) * v;
            validate_distribution(&target_probs[offset..offset + v], "verify", s_idx, pos)?;
        }
    }

    // Precompute root-to-leaf paths for tree drafts (Spec 7 §5)
    let paths: Vec<Vec<usize>> = if let Some(tree_mask) = tree {
        if k == 0 {
            vec![]
        } else {
            let parents = tree_mask.parents();
            // Identify leaf nodes: nodes that never appear as a parent of another node
            let mut is_parent = vec![false; k];
            for &p in parents {
                if p >= 0 && (p as usize) < k {
                    is_parent[p as usize] = true;
                }
            }

            let mut candidate_paths = Vec::new();
            for (node_idx, &has_child) in is_parent.iter().enumerate().take(k) {
                if !has_child {
                    // Trace leaf to root
                    let mut path = Vec::new();
                    let mut curr = node_idx as i32;
                    while curr >= 0 && (curr as usize) < k {
                        path.push(curr as usize);
                        curr = parents[curr as usize];
                    }
                    path.reverse();
                    candidate_paths.push(path);
                }
            }
            // Sort paths: primary tie-breaker is lowest first-token index (Spec 7 §5)
            candidate_paths.sort();
            candidate_paths
        }
    } else {
        // Linear chain: 0 -> 1 -> ... -> k-1
        if k > 0 {
            vec![(0..k).collect()]
        } else {
            vec![]
        }
    };

    let mut out_accepted = vec![0u32; s * k_plus_one];
    let mut out_accept_len = vec![0u32; s];

    for s_idx in 0..s {
        let d_tokens = &draft_tokens[s_idx * k..(s_idx + 1) * k];
        let d_probs_row = draft_probs.map(|dp| &dp[s_idx * k * v..(s_idx + 1) * k * v]);
        let t_probs_row = &target_probs[s_idx * k_plus_one * v..(s_idx + 1) * k_plus_one * v];
        let rng = &mut rng_states[s_idx];

        let out_acc_slice = &mut out_accepted[s_idx * k_plus_one..(s_idx + 1) * k_plus_one];

        if k == 0 || paths.is_empty() {
            // Degenerate k=0: sample single continuation token directly from root distribution
            let root_dist = &t_probs_row[..v];
            let bonus_tok = match method {
                VerifyMethod::Greedy => argmax_stable(root_dist) as u32,
                VerifyMethod::Rejection | VerifyMethod::TypicalAcceptance { .. } => {
                    sample_from_distribution(root_dist, rng.draw_at(0))?
                }
            };
            out_acc_slice[0] = bonus_tok;
            out_accept_len[s_idx] = 0;
            rng.advance(1);
            continue;
        }

        // Evaluate all candidate paths and find the winning path (Spec 7 §5)
        let mut best_path_len = 0usize;
        let mut best_first_token = usize::MAX;
        let mut best_accepted_nodes: Vec<usize> = Vec::new();
        let mut best_terminal_token: u32 = 0;

        for path in &paths {
            let mut accepted_nodes = Vec::with_capacity(path.len());
            let mut terminal_token = 0u32;
            let mut all_path_accepted = true;

            for (pos_idx, &node_idx) in path.iter().enumerate() {
                let d = d_tokens[node_idx];

                // Target distribution predicting node_idx:
                // If pos_idx == 0 (root), distribution is target_probs[0]
                // If pos_idx > 0, distribution is target_probs[parent_node + 1]
                let target_dist = if pos_idx == 0 {
                    &t_probs_row[..v]
                } else {
                    let parent_node = path[pos_idx - 1];
                    &t_probs_row[(parent_node + 1) * v..(parent_node + 2) * v]
                };

                let target_p = target_dist[d as usize];

                let accepted = match method {
                    VerifyMethod::Rejection => {
                        let draft_q = if let Some(dp) = d_probs_row {
                            dp[node_idx * v + d as usize]
                        } else {
                            // Proposer was deterministic (q is one-hot)
                            1.0
                        };

                        let alpha = if draft_q > 0.0 {
                            (target_p / draft_q).min(1.0)
                        } else if target_p > 0.0 {
                            1.0
                        } else {
                            0.0
                        };

                        let u = rng.draw_at(node_idx as u32);
                        if u < alpha {
                            true
                        } else {
                            // Rejection at position node_idx: sample replacement from norm(max(0, p - q))
                            let mut residual = vec![0.0f32; v];
                            for tok in 0..v {
                                let q_val = if let Some(dp) = d_probs_row {
                                    dp[node_idx * v + tok]
                                } else if tok == d as usize {
                                    1.0
                                } else {
                                    0.0
                                };
                                residual[tok] = (target_dist[tok] - q_val).max(0.0);
                            }

                            let sum_res: f32 = residual.iter().sum();
                            if sum_res > 0.0 && sum_res.is_finite() {
                                let inv = 1.0 / sum_res;
                                for r in &mut residual {
                                    *r *= inv;
                                }
                                terminal_token = sample_from_distribution(
                                    &residual,
                                    rng.draw_at(terminal_draw),
                                )?;
                            } else {
                                // Fallback to target argmax if residual is zeroed
                                terminal_token = argmax_stable(target_dist) as u32;
                            }
                            false
                        }
                    }
                    VerifyMethod::Greedy => {
                        let target_argmax = argmax_stable(target_dist) as u32;
                        if target_argmax == d {
                            true
                        } else {
                            terminal_token = target_argmax;
                            false
                        }
                    }
                    VerifyMethod::TypicalAcceptance { eps, delta } => {
                        // Entropy H(p) = -sum(p * ln(p))
                        let mut h = 0.0f32;
                        for &p in target_dist {
                            if p > 0.0 {
                                h -= p * p.ln();
                            }
                        }
                        let threshold = eps.min(delta * (-h).exp());
                        if target_p > threshold {
                            true
                        } else {
                            // Typical replacement sampled from target_probs[pos]
                            terminal_token =
                                sample_from_distribution(target_dist, rng.draw_at(terminal_draw))?;
                            false
                        }
                    }
                };

                if accepted {
                    accepted_nodes.push(node_idx);
                } else {
                    all_path_accepted = false;
                    break;
                }
            }

            if all_path_accepted {
                // All tokens on path accepted: sample bonus token from target_probs[last_node + 1]
                let last_node = *path.last().unwrap();
                let bonus_dist = &t_probs_row[(last_node + 1) * v..(last_node + 2) * v];
                terminal_token = match method {
                    VerifyMethod::Greedy => argmax_stable(bonus_dist) as u32,
                    VerifyMethod::Rejection | VerifyMethod::TypicalAcceptance { .. } => {
                        sample_from_distribution(bonus_dist, rng.draw_at(terminal_draw))?
                    }
                };
            }

            let path_len = accepted_nodes.len();
            let first_token = path.first().copied().unwrap_or(0);

            // Longest accepted path rule; ties go to lowest first-token index (Spec 7 §5)
            let is_better = if path_len > best_path_len {
                true
            } else if path_len == best_path_len {
                first_token < best_first_token
            } else {
                false
            };

            if is_better {
                best_path_len = path_len;
                best_first_token = first_token;
                best_accepted_nodes = accepted_nodes;
                best_terminal_token = terminal_token;
            }
        }

        // Commit winning path to output (Spec 7 §5)
        out_accept_len[s_idx] = best_path_len as u32;
        for (i, &node_idx) in best_accepted_nodes.iter().enumerate() {
            out_acc_slice[i] = d_tokens[node_idx];
        }
        out_acc_slice[best_path_len] = best_terminal_token;

        // Advance RNG state by k + 1 for clean non-overlapping step keying (Spec 1 §4.F, Spec 7 §4)
        rng.advance(draw_advance);
    }

    Ok(VerifyOutput {
        accepted: out_accepted,
        accept_len: out_accept_len,
    })
}

/// Helper: inverse-CDF draw on a probability distribution with a given uniform float u in (0, 1).
#[inline]
fn sample_from_distribution(probs: &[f32], u: f32) -> Result<u32, T0Error> {
    let mut cumsum = 0.0f32;
    for (tok, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= u {
            return Ok(tok as u32);
        }
    }
    // Fallback on numerical rounding: find last non-zero token or 0
    for tok in (0..probs.len()).rev() {
        if probs[tok] > 0.0 {
            return Ok(tok as u32);
        }
    }
    Ok(0)
}

/// Validates one normalized probability row before sampling or verification.
fn validate_distribution(
    probs: &[f32],
    op: &'static str,
    seq: usize,
    pos: usize,
) -> Result<(), T0Error> {
    for (token, &value) in probs.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(T0Error::InvalidProbability {
                op,
                seq,
                pos,
                token,
                value,
            });
        }
    }

    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 || (sum - 1.0).abs() > 1.0e-3 {
        return Err(T0Error::InvalidDistribution { op, seq, pos, sum });
    }
    Ok(())
}

/// Helper: argmax with stable tie-break by lowest index (Spec 1 §4.F, §6.5).
#[inline]
fn argmax_stable(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in slice.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use r9v_ir::SamplingParams;

    fn default_params() -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: vec![],
        }
    }

    #[test]
    fn test_logits_postprocess_basic_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let params = vec![default_params()];
        let mut probs = vec![0.0; 3];
        logits_postprocess(&logits, 1, 1, 3, &params, None, None, &mut probs).unwrap();

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_logits_postprocess_temperature_zero() {
        let logits = vec![1.0, 5.0, 2.0];
        let mut p = default_params();
        p.temperature = 0.0;
        let params = vec![p];
        let mut probs = vec![0.0; 3];
        logits_postprocess(&logits, 1, 1, 3, &params, None, None, &mut probs).unwrap();

        assert_eq!(probs, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_logits_postprocess_grammar_mask() {
        let logits = vec![10.0, 2.0, 1.0];
        let params = vec![default_params()];
        let mask = vec![false, true, true];
        let mut probs = vec![0.0; 3];
        logits_postprocess(&logits, 1, 1, 3, &params, None, Some(&mask), &mut probs).unwrap();

        assert_eq!(probs[0], 0.0);
        assert!((probs[1] + probs[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_greedy_verify_chain() {
        let draft_tokens = vec![2, 4];
        let target_probs = vec![
            // pos 0: token 2 matches
            0.0, 0.0, 1.0, 0.0, 0.0, // pos 1: token 4 matches
            0.0, 0.0, 0.0, 0.0, 1.0, // pos 2: bonus token is 1
            0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let mut rng_states = vec![RngState::new(42, 1, 0)];

        let out = verify(
            &draft_tokens,
            None,
            &target_probs,
            1,
            2,
            5,
            &VerifyMethod::Greedy,
            &mut rng_states,
            None,
        )
        .unwrap();

        assert_eq!(out.accept_len, vec![2]);
        assert_eq!(out.accepted, vec![2, 4, 1]);
    }
}
