// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 implementation of `moe_route` op (Spec 1 §4.C, §6.1, Card A1.9).
//!
//! Routes each token to its top-K experts with deterministic tie-breaking
//! (lowest expert index wins) and order-stable output rows.

use r9v_ir::{DType, LayoutId, MoeRouteOp, MoeScoring};

use crate::buffer::{TensorDataMut, TensorView, TensorViewMut};
use crate::error::T0Error;

/// Writes one expert id bit-exactly to a `u32` output view (Spec 1 §4.C).
///
/// Accepts both typed `U32` and raw-byte `U32` backings; anything else is a
/// backing mismatch (the `u32` dtype check upstream already passed).
pub(crate) fn write_id(
    view: &mut TensorViewMut<'_>,
    index: usize,
    val: u32,
) -> Result<(), T0Error> {
    match view.data {
        TensorDataMut::U32(ref mut slice) => {
            if index >= slice.len() {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "expert_ids",
                    buffer_len: slice.len(),
                    expected_len: index + 1,
                    shape: view.shape().to_vec(),
                });
            }
            slice[index] = val;
            Ok(())
        }
        TensorDataMut::Bytes(dtype, ref mut slice) => {
            let end = index
                .checked_mul(4)
                .and_then(|off| off.checked_add(4))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "moe_route",
                    detail: format!("u32 byte range for index {index} overflows usize"),
                })?;
            if end > slice.len() {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "expert_ids",
                    buffer_len: slice.len(),
                    expected_len: end,
                    shape: view.shape().to_vec(),
                });
            }
            let _ = dtype;
            slice[index * 4..end].copy_from_slice(&val.to_le_bytes());
            Ok(())
        }
        _ => Err(T0Error::BackingRepresentationMismatch {
            op: "moe_route",
            dtype: view.dtype(),
        }),
    }
}

/// Routes tokens to experts: `logits [T, E] f32` (+ optional `bias [E]`)
/// → `expert_ids [T, K] u32`, `weights [T, K] f32` (Spec 1 §4.C, Card A1.9).
///
/// Algorithm in `f32`, ascending-token outer loop for determinism:
/// 1. `s = logits + bias` (when present);
/// 2. scores: stable softmax in `f32` (row max, ascending exp-sum) or
///    `1/(1+exp(-s))` in `f32` for sigmoid;
/// 3. stable sort of row indices by `(-score, index)` — ties resolve to the
///    lowest expert index per spec — and take the first K;
/// 4. `weights = score[selected] * scale`;
/// 5. when `renormalize`, divide by the `f32` sum over the selected K.
///
/// Output rows list experts in descending score order, ties by ascending
/// expert id. The weighted combine in `moe_ffn` is order-insensitive, so this
/// order is a presentation choice, not a numerics input.
///
/// Numerics: `Numerics::f32(AscendingIndex)`; no IR change.
///
/// Fail-closed (SI-51): `group.is_some()` is rejected — the spec states no
/// grouped-selection algorithm. Non-finite logits are collected per `(t, e)`.
/// After scoring, non-finite post-bias scores, a zero/non-finite
/// renormalize divisor, and non-finite staged weights are each rejected per
/// row before either output mutates, so no NaN/Inf weight is ever committed.
/// DECISION(A1.9): scale multiplies selected scores before renormalization
/// and renormalization divides by the selected-K sum; rejected post-renorm
/// scaling (would silently un-normalize) and full-row denominators (would
/// contradict `renormalize` naming the selected weights). Per SI-51.
pub fn moe_route(
    op: &MoeRouteOp,
    logits: &TensorView<'_>,
    bias: Option<&TensorView<'_>>,
    expert_ids: &mut TensorViewMut<'_>,
    weights: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    logits.validate_backing("logits")?;
    if let Some(b) = bias {
        b.validate_backing("bias")?;
    }
    expert_ids.validate_backing("expert_ids")?;
    weights.validate_backing("weights")?;

    let mut problems = Vec::new();

    if logits.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "logits",
            expected: 2,
            got: logits.rank(),
            shape: logits.shape().to_vec(),
        });
    }
    if logits.dtype() != DType::F32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "logits",
            expected: vec![DType::F32],
            got: logits.dtype(),
        });
    }
    if logits.layout() != LayoutId::CONTIGUOUS && logits.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "logits",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: logits.layout(),
        });
    }

    if let Some(b) = bias {
        if b.rank() != 1 {
            problems.push(T0Error::RankMismatch {
                tensor: "bias",
                expected: 1,
                got: b.rank(),
                shape: b.shape().to_vec(),
            });
        }
        if b.dtype() != DType::F32 {
            problems.push(T0Error::DTypeMismatch {
                tensor: "bias",
                expected: vec![DType::F32],
                got: b.dtype(),
            });
        }
        if b.layout() != LayoutId::CONTIGUOUS && b.layout() != LayoutId::L0 {
            problems.push(T0Error::LayoutMismatch {
                tensor: "bias",
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
                got: b.layout(),
            });
        }
    }

    if expert_ids.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "expert_ids",
            expected: 2,
            got: expert_ids.rank(),
            shape: expert_ids.shape().to_vec(),
        });
    }
    if expert_ids.dtype() != DType::U32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "expert_ids",
            expected: vec![DType::U32],
            got: expert_ids.dtype(),
        });
    }
    if weights.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "weights",
            expected: 2,
            got: weights.rank(),
            shape: weights.shape().to_vec(),
        });
    }
    if weights.dtype() != DType::F32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "weights",
            expected: vec![DType::F32],
            got: weights.dtype(),
        });
    }

    if op.top_k == 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "moe_route",
            attribute: "top_k",
            reason: "top_k must be > 0".to_string(),
        });
    }
    if !op.scale.is_finite() || op.scale <= 0.0 {
        problems.push(T0Error::InvalidAttribute {
            op: "moe_route",
            attribute: "scale",
            reason: format!("scale must be finite and > 0, got {}", op.scale),
        });
    }
    if op.group.is_some() {
        problems.push(T0Error::InvalidAttribute {
            op: "moe_route",
            attribute: "group",
            reason: "grouped expert routing has no specified selection algorithm (SI-51)"
                .to_string(),
        });
    }
    match op.scoring {
        MoeScoring::Softmax | MoeScoring::Sigmoid => {}
    }

    T0Error::from_problems(problems)?;

    let t = logits.shape()[0];
    let e = logits.shape()[1];
    let k = op.top_k as usize;

    if t == 0 || e == 0 {
        return Err(T0Error::EmptyInput {
            op: "moe_route",
            tensor: "logits",
        });
    }
    if k > e {
        return Err(T0Error::InvalidAttribute {
            op: "moe_route",
            attribute: "top_k",
            reason: format!("top_k {k} cannot exceed number of experts E={e}"),
        });
    }
    let _ = u32::try_from(t).map_err(|_| T0Error::ArithmeticOverflow {
        op: "moe_route",
        detail: format!("dimension T exceeds u32: {t}"),
    })?;
    let _ = u32::try_from(e).map_err(|_| T0Error::ArithmeticOverflow {
        op: "moe_route",
        detail: format!("dimension E exceeds u32: {e}"),
    })?;

    let mut problems = Vec::new();
    if let Some(b) = bias {
        if b.shape()[0] != e {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "E",
                expected_from: "logits",
                expected: e,
                tensor: "bias",
                got: b.shape()[0],
            });
        }
    }
    if expert_ids.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "logits",
            expected: t,
            tensor: "expert_ids",
            got: expert_ids.shape()[0],
        });
    }
    if expert_ids.shape()[1] != k {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "K",
            expected_from: "top_k",
            expected: k,
            tensor: "expert_ids",
            got: expert_ids.shape()[1],
        });
    }
    if weights.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "logits",
            expected: t,
            tensor: "weights",
            got: weights.shape()[0],
        });
    }
    if weights.shape()[1] != k {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "K",
            expected_from: "top_k",
            expected: k,
            tensor: "weights",
            got: weights.shape()[1],
        });
    }
    // Collect every non-finite logit before touching any output.
    for row in 0..t {
        for col in 0..e {
            let v = logits.read_f32(row * e + col);
            if !v.is_finite() {
                problems.push(T0Error::InvalidAttribute {
                    op: "moe_route",
                    attribute: "logits",
                    reason: format!("non-finite logit at (t={row}, e={col}): {v}"),
                });
            }
        }
    }
    if let Some(b) = bias {
        for col in 0..e.min(b.backing_len()) {
            let v = b.read_f32(col);
            if !v.is_finite() {
                problems.push(T0Error::InvalidAttribute {
                    op: "moe_route",
                    attribute: "bias",
                    reason: format!("non-finite bias at (e={col}): {v}"),
                });
            }
        }
    }
    T0Error::from_problems(problems)?;

    // Validated: compute into staged buffers, then commit to outputs.
    // NaN/Inf guards (SI-51): post-bias scores, the renormalize divisor, and
    // staged weights are all checked per row before either output mutates, so
    // a degenerate row is a typed refusal rather than silent NaN weights.
    let mut staged_ids = vec![0u32; t * k];
    let mut staged_w = vec![0.0f32; t * k];
    let mut problems = Vec::new();
    for row in 0..t {
        let mut scores = vec![0.0f32; e];
        for col in 0..e {
            let mut s = logits.read_f32(row * e + col);
            if let Some(b) = bias {
                s += b.read_f32(col);
            }
            scores[col] = s;
        }
        match op.scoring {
            MoeScoring::Softmax => {
                let mut max = scores[0];
                for c in 1..e {
                    if scores[c] > max {
                        max = scores[c];
                    }
                }
                let mut sum = 0.0f32;
                for c in 0..e {
                    let v = (scores[c] - max).exp();
                    scores[c] = v;
                    sum += v;
                }
                for c in 0..e {
                    scores[c] /= sum;
                }
            }
            MoeScoring::Sigmoid => {
                for c in 0..e {
                    scores[c] = 1.0 / (1.0 + (-scores[c]).exp());
                }
            }
        }
        // Finite logits/bias can still overflow their f32 sum or scoring
        // transform; no NaN/Inf may reach the comparator or the outputs.
        let mut row_finite = true;
        for c in 0..e {
            if !scores[c].is_finite() {
                problems.push(T0Error::InvalidAttribute {
                    op: "moe_route",
                    attribute: "logits",
                    reason: format!(
                        "non-finite post-bias score at (t={row}, e={c}): {} (SI-51)",
                        scores[c]
                    ),
                });
                row_finite = false;
            }
        }
        if !row_finite {
            continue;
        }
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
        let mut selected_sum = 0.0f32;
        for j in 0..k {
            let col = order[j];
            let w = scores[col] * op.scale;
            staged_ids[row * k + j] = col as u32;
            staged_w[row * k + j] = w;
            selected_sum += w;
        }
        if op.renormalize {
            // A degenerate sigmoid row (e.g. all extreme-negative logits)
            // selects all-zero scores; dividing would emit NaN weights.
            if !selected_sum.is_finite() || selected_sum == 0.0 {
                problems.push(T0Error::InvalidAttribute {
                    op: "moe_route",
                    attribute: "weights",
                    reason: format!(
                        "non-finite or zero selected-K sum {selected_sum} at row {row} with renormalize (SI-51)"
                    ),
                });
                continue;
            }
            for j in 0..k {
                staged_w[row * k + j] /= selected_sum;
            }
        }
        for j in 0..k {
            let w = staged_w[row * k + j];
            if !w.is_finite() {
                problems.push(T0Error::InvalidAttribute {
                    op: "moe_route",
                    attribute: "weights",
                    reason: format!("non-finite staged weight at (t={row}, slot={j}): {w} (SI-51)"),
                });
            }
        }
    }
    T0Error::from_problems(problems)?;

    for (i, &id) in staged_ids.iter().enumerate() {
        write_id(expert_ids, i, id)?;
    }
    for (i, &w) in staged_w.iter().enumerate() {
        weights.write_f32(i, w);
    }
    Ok(())
}

/// 64-bit reference router for testing (Spec 1 §4.C, §6.1).
///
/// Independent `f64` path: never calls [`moe_route`]. Returns
/// `(expert_ids [T, K], weights [T, K])` mirroring the T0 algorithm in `f64`.
/// Slice lengths and `top_k` are validated with typed errors; checked
/// products guard every extent, so no silent empty fallback exists.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_f64_reference(
    logits: &[f64],
    t: usize,
    e: usize,
    bias: Option<&[f64]>,
    top_k: u32,
    scoring: MoeScoring,
    renormalize: bool,
    scale: f64,
) -> Result<(Vec<u32>, Vec<f64>), T0Error> {
    const OP: &str = "moe_route";
    let k = top_k as usize;
    let mut problems = Vec::new();
    if k == 0 || k > e {
        problems.push(T0Error::InvalidAttribute {
            op: OP,
            attribute: "top_k",
            reason: format!("top_k {k} must satisfy 0 < top_k <= E={e}"),
        });
    }
    if t == 0 || e == 0 {
        problems.push(T0Error::EmptyInput {
            op: OP,
            tensor: "logits",
        });
    }
    let te = t
        .checked_mul(e)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * E overflows usize for T={t}, E={e}"),
        })?;
    if logits.len() != te {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "logits",
            expected: te,
            got: logits.len(),
            detail: "logits length must equal T * E".to_string(),
        });
    }
    if let Some(b) = bias {
        if b.len() != e {
            problems.push(T0Error::ShapeLengthMismatch {
                op: OP,
                tensor: "bias",
                expected: e,
                got: b.len(),
                detail: "bias length must equal E".to_string(),
            });
        }
    }
    T0Error::from_problems(problems)?;
    let tk = t
        .checked_mul(k)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * K overflows usize for T={t}, K={k}"),
        })?;
    let mut ids = vec![0u32; tk];
    let mut ws = vec![0.0f64; tk];
    for row in 0..t {
        let mut scores = vec![0.0f64; e];
        for col in 0..e {
            let mut s = logits[row * e + col];
            if let Some(b) = bias {
                s += b[col];
            }
            scores[col] = s;
        }
        match scoring {
            MoeScoring::Softmax => {
                let mut max = scores[0];
                for c in 1..e {
                    if scores[c] > max {
                        max = scores[c];
                    }
                }
                let mut sum = 0.0f64;
                for c in 0..e {
                    let v = (scores[c] - max).exp();
                    scores[c] = v;
                    sum += v;
                }
                for c in 0..e {
                    scores[c] /= sum;
                }
            }
            MoeScoring::Sigmoid => {
                for c in 0..e {
                    scores[c] = 1.0 / (1.0 + (-scores[c]).exp());
                }
            }
        }
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
        let mut selected_sum = 0.0f64;
        for j in 0..k {
            let col = order[j];
            let w = scores[col] * scale;
            ids[row * k + j] = col as u32;
            ws[row * k + j] = w;
            selected_sum += w;
        }
        if renormalize {
            for j in 0..k {
                ws[row * k + j] /= selected_sum;
            }
        }
    }
    Ok((ids, ws))
}
