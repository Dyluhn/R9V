// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 implementations of `linear_attn_scan` (Spec 1 §4.E, Card A1.9).
//!
//! Chunked and recurrent forms over one shared scalar core with identical
//! per-token operation order, so the two forms agree bit-exactly (L0).
//!
//! DECISION(A1.9): all three scan kinds share the gated outer-product
//! recurrence and chunking is T0-level loop blocking only; rejected per-kind
//! branches and device WMMA tiling in T0 scope. Per SI-47.

use r9v_ir::{DType, LayoutId, LinearAttnKind, LinearAttnScanOp};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;
use crate::segments::SeqLayout;

/// Validated scan dimensions shared by both forms.
struct ScanDims {
    t: usize,
    h: usize,
    d: usize,
    dv: usize,
    s: usize,
}

/// Validates a scan invocation without touching outputs or state (Spec 1 §4.E).
///
/// Collects every structural, dtype, layout, dimension, and gate-finiteness
/// problem before any byte is written. Returns the resolved dimensions.
#[allow(clippy::too_many_arguments)]
fn validate_scan(
    op: &LinearAttnScanOp,
    q: &TensorView<'_>,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    alpha: &TensorView<'_>,
    beta: &TensorView<'_>,
    state_in: &TensorView<'_>,
    seq: &SeqLayout,
    o: &TensorViewMut<'_>,
    state_out: &TensorViewMut<'_>,
) -> Result<ScanDims, T0Error> {
    q.validate_backing("q")?;
    k.validate_backing("k")?;
    v.validate_backing("v")?;
    alpha.validate_backing("alpha")?;
    beta.validate_backing("beta")?;
    state_in.validate_backing("state_in")?;
    o.validate_backing("o")?;
    state_out.validate_backing("state_out")?;

    let mut problems = Vec::new();

    for (name, view, rank) in [
        ("q", q, 3),
        ("k", k, 3),
        ("v", v, 3),
        ("alpha", alpha, 2),
        ("beta", beta, 2),
        ("state_in", state_in, 4),
    ] {
        if view.rank() != rank {
            problems.push(T0Error::RankMismatch {
                tensor: name,
                expected: rank,
                got: view.rank(),
                shape: view.shape().to_vec(),
            });
        }
    }
    if o.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "o",
            expected: 3,
            got: o.rank(),
            shape: o.shape().to_vec(),
        });
    }
    if state_out.rank() != 4 {
        problems.push(T0Error::RankMismatch {
            tensor: "state_out",
            expected: 4,
            got: state_out.rank(),
            shape: state_out.shape().to_vec(),
        });
    }

    for (name, view) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("alpha", alpha),
        ("beta", beta),
        ("state_in", state_in),
    ] {
        if view.layout() != LayoutId::CONTIGUOUS && view.layout() != LayoutId::L0 {
            problems.push(T0Error::LayoutMismatch {
                tensor: name,
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
                got: view.layout(),
            });
        }
    }
    if o.layout() != LayoutId::CONTIGUOUS && o.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "o",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: o.layout(),
        });
    }
    if state_out.layout() != LayoutId::CONTIGUOUS && state_out.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "state_out",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: state_out.layout(),
        });
    }

    for (name, view) in [("q", q), ("k", k), ("v", v)] {
        if !matches!(view.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(T0Error::DTypeMismatch {
                tensor: name,
                expected: vec![DType::F16, DType::Bf16, DType::F32],
                got: view.dtype(),
            });
        }
    }
    if k.dtype() != q.dtype() {
        problems.push(T0Error::DTypeMismatch {
            tensor: "k",
            expected: vec![q.dtype()],
            got: k.dtype(),
        });
    }
    if v.dtype() != q.dtype() {
        problems.push(T0Error::DTypeMismatch {
            tensor: "v",
            expected: vec![q.dtype()],
            got: v.dtype(),
        });
    }
    for (name, view) in [("alpha", alpha), ("beta", beta), ("state_in", state_in)] {
        if view.dtype() != DType::F32 {
            problems.push(T0Error::DTypeMismatch {
                tensor: name,
                expected: vec![DType::F32],
                got: view.dtype(),
            });
        }
    }
    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::InvalidAttribute {
            op: "linear_attn_scan",
            attribute: "out_dtype",
            reason: format!("must be f16, bf16, or f32, got {:?}", op.out_dtype),
        });
    }
    if o.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "o",
            expected: vec![op.out_dtype],
            got: o.dtype(),
        });
    }
    if state_out.dtype() != DType::F32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "state_out",
            expected: vec![DType::F32],
            got: state_out.dtype(),
        });
    }
    if op.chunk == 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "linear_attn_scan",
            attribute: "chunk",
            reason: "chunk must be > 0".to_string(),
        });
    }
    // All three kinds share the T0 recurrence (SI-47): accepted, not branched.
    match op.kind {
        LinearAttnKind::GatedDeltaNet | LinearAttnKind::GLA | LinearAttnKind::Mamba2 => {}
    }
    match op.handle.kind() {
        r9v_ir::StateKind::Recurrent => {}
        other => problems.push(T0Error::InvalidAttribute {
            op: "linear_attn_scan",
            attribute: "handle",
            reason: format!("state handle must be Recurrent, got {other:?}"),
        }),
    }

    T0Error::from_problems(problems)?;

    let t = q.shape()[0];
    let h = q.shape()[1];
    let d = q.shape()[2];
    let dv = v.shape()[2];
    let s = seq.seq_count();

    if h == 0 {
        return Err(T0Error::EmptyInput {
            op: "linear_attn_scan",
            tensor: "q",
        });
    }
    if d == 0 || dv == 0 {
        return Err(T0Error::EmptyInput {
            op: "linear_attn_scan",
            tensor: "q",
        });
    }
    if t == 0 {
        return Err(T0Error::EmptyInput {
            op: "linear_attn_scan",
            tensor: "q",
        });
    }
    for dim in [t, h, d, dv, s] {
        u32::try_from(dim).map_err(|_| T0Error::ArithmeticOverflow {
            op: "linear_attn_scan",
            detail: format!("dimension exceeds u32: {dim}"),
        })?;
    }
    seq.check_total("q", t)?;

    let mut problems = Vec::new();
    // T agreement.
    for (name, view) in [("k", k), ("v", v)] {
        if view.shape()[0] != t {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "T",
                expected_from: "q",
                expected: t,
                tensor: name,
                got: view.shape()[0],
            });
        }
    }
    for (name, view) in [("alpha", alpha), ("beta", beta)] {
        if view.shape()[0] != t {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "T",
                expected_from: "q",
                expected: t,
                tensor: name,
                got: view.shape()[0],
            });
        }
    }
    if o.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "q",
            expected: t,
            tensor: "o",
            got: o.shape()[0],
        });
    }
    // H agreement.
    if k.shape()[1] != h {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "q",
            expected: h,
            tensor: "k",
            got: k.shape()[1],
        });
    }
    if v.shape()[1] != h {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "q",
            expected: h,
            tensor: "v",
            got: v.shape()[1],
        });
    }
    if alpha.shape()[1] != h {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "q",
            expected: h,
            tensor: "alpha",
            got: alpha.shape()[1],
        });
    }
    if beta.shape()[1] != h {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "q",
            expected: h,
            tensor: "beta",
            got: beta.shape()[1],
        });
    }
    if o.shape()[1] != h {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "q",
            expected: h,
            tensor: "o",
            got: o.shape()[1],
        });
    }
    // D / Dv agreement.
    if k.shape()[2] != d {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "D",
            expected_from: "q",
            expected: d,
            tensor: "k",
            got: k.shape()[2],
        });
    }
    if o.shape()[2] != dv {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dv",
            expected_from: "v",
            expected: dv,
            tensor: "o",
            got: o.shape()[2],
        });
    }
    // State [S, H, D, Dv].
    if state_in.shape()[0] != s {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "S",
            expected_from: "seq",
            expected: s,
            tensor: "state_in",
            got: state_in.shape()[0],
        });
    }
    if state_in.shape()[1] != h {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "q",
            expected: h,
            tensor: "state_in",
            got: state_in.shape()[1],
        });
    }
    if state_in.shape()[2] != d {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "D",
            expected_from: "q",
            expected: d,
            tensor: "state_in",
            got: state_in.shape()[2],
        });
    }
    if state_in.shape()[3] != dv {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dv",
            expected_from: "v",
            expected: dv,
            tensor: "state_in",
            got: state_in.shape()[3],
        });
    }
    if state_out.shape()[0] != s {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "S",
            expected_from: "seq",
            expected: s,
            tensor: "state_out",
            got: state_out.shape()[0],
        });
    }
    if state_out.shape()[1] != h {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "q",
            expected: h,
            tensor: "state_out",
            got: state_out.shape()[1],
        });
    }
    if state_out.shape()[2] != d {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "D",
            expected_from: "q",
            expected: d,
            tensor: "state_out",
            got: state_out.shape()[2],
        });
    }
    if state_out.shape()[3] != dv {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dv",
            expected_from: "v",
            expected: dv,
            tensor: "state_out",
            got: state_out.shape()[3],
        });
    }
    // Gate finiteness: no spec authority to clamp, so non-finite gates are
    // collected and rejected rather than silently propagated.
    for row in 0..t {
        for head in 0..h {
            let a = alpha.read_f32(row * h + head);
            if !a.is_finite() {
                problems.push(T0Error::InvalidAttribute {
                    op: "linear_attn_scan",
                    attribute: "alpha",
                    reason: format!("non-finite gate at (t={row}, h={head}): {a}"),
                });
            }
            let b = beta.read_f32(row * h + head);
            if !b.is_finite() {
                problems.push(T0Error::InvalidAttribute {
                    op: "linear_attn_scan",
                    attribute: "beta",
                    reason: format!("non-finite gate at (t={row}, h={head}): {b}"),
                });
            }
        }
    }
    T0Error::from_problems(problems)?;

    Ok(ScanDims { t, h, d, dv, s })
}

/// Stages one full scan run: `o_tmp [T·H·Dv]` and `s_tmp [S·H·D·Dv]` in `f32`.
///
/// `chunked` selects the loop nest: `false` runs the recurrent token-at-a-time
/// nest, `true` blocks the token loop over `chunk` with identical per-token
/// operation order (T0-level loop blocking; device WMMA chunking is T1/T2
/// scope per spec 4 §5.5). Both nests share the scalar step below.
#[allow(clippy::too_many_arguments)]
fn run_scan(
    q: &TensorView<'_>,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    alpha: &TensorView<'_>,
    beta: &TensorView<'_>,
    state_in: &TensorView<'_>,
    seq: &SeqLayout,
    dims: &ScanDims,
    chunk: usize,
    chunked: bool,
    o_tmp: &mut [f32],
    s_tmp: &mut [f32],
) {
    let (h, d, dv) = (dims.h, dims.d, dims.dv);
    let mut base = 0usize;
    for (slot, &len_u32) in seq.seq_lens().iter().enumerate() {
        let len = len_u32 as usize;
        for head in 0..h {
            let mut s_mat = vec![0.0f32; d * dv];
            for i in 0..d {
                for j in 0..dv {
                    s_mat[i * dv + j] = state_in.read_f32((slot * h + head) * d * dv + i * dv + j);
                }
            }
            if chunked {
                let mut cs = 0usize;
                while cs < len {
                    let ce = (cs + chunk).min(len);
                    for r in cs..ce {
                        let gt = base + r;
                        scan_step_into(q, k, v, alpha, beta, dims, gt, head, &mut s_mat, o_tmp);
                    }
                    cs = ce;
                }
            } else {
                for r in 0..len {
                    let gt = base + r;
                    scan_step_into(q, k, v, alpha, beta, dims, gt, head, &mut s_mat, o_tmp);
                }
            }
            for i in 0..d {
                for j in 0..dv {
                    s_tmp[(slot * h + head) * d * dv + i * dv + j] = s_mat[i * dv + j];
                }
            }
        }
        base += len;
    }
}

/// Executes the chunked scan form (Spec 1 §4.E, spec 4 §5.5, Card A1.9).
///
/// Tokens run in per-`(sequence, head)` chunks of `op.chunk`; state reads slot
/// A (`state_in`) and writes slot B (`state_out`) without mutating the input.
/// Bit-exact with [`linear_attn_scan_recurrent`] by shared operation order.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_scan_chunked(
    op: &LinearAttnScanOp,
    q: &TensorView<'_>,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    alpha: &TensorView<'_>,
    beta: &TensorView<'_>,
    state_in: &TensorView<'_>,
    seq: &SeqLayout,
    o: &mut TensorViewMut<'_>,
    state_out: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let dims = validate_scan(op, q, k, v, alpha, beta, state_in, seq, o, state_out)?;
    run_and_commit(
        q,
        k,
        v,
        alpha,
        beta,
        state_in,
        seq,
        &dims,
        op.chunk as usize,
        true,
        o,
        state_out,
    )
}

/// Executes the recurrent scan form (Spec 1 §4.E, spec 4 §5.5, Card A1.9).
///
/// One token step at a time per `(sequence, head)`; used for short queries and
/// for accepted-prefix recompute from slot A into slot B (spec 3 §4.2).
/// Bit-exact with [`linear_attn_scan_chunked`] by shared operation order.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_scan_recurrent(
    op: &LinearAttnScanOp,
    q: &TensorView<'_>,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    alpha: &TensorView<'_>,
    beta: &TensorView<'_>,
    state_in: &TensorView<'_>,
    seq: &SeqLayout,
    o: &mut TensorViewMut<'_>,
    state_out: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let dims = validate_scan(op, q, k, v, alpha, beta, state_in, seq, o, state_out)?;
    run_and_commit(
        q,
        k,
        v,
        alpha,
        beta,
        state_in,
        seq,
        &dims,
        op.chunk as usize,
        false,
        o,
        state_out,
    )
}

/// Stages a full run and commits outputs and state (Spec 1 §4.E).
///
/// `o` (cast once to `out_dtype`) and `state_out` are written only after the
/// whole run stages successfully, preserving the A/B discipline where slot A
/// is never mutated.
#[allow(clippy::too_many_arguments)]
fn run_and_commit(
    q: &TensorView<'_>,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    alpha: &TensorView<'_>,
    beta: &TensorView<'_>,
    state_in: &TensorView<'_>,
    seq: &SeqLayout,
    dims: &ScanDims,
    chunk: usize,
    chunked: bool,
    o: &mut TensorViewMut<'_>,
    state_out: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let o_len = dims
        .t
        .checked_mul(dims.h)
        .and_then(|v| v.checked_mul(dims.dv))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "linear_attn_scan",
            detail: "output buffer size overflows usize".to_string(),
        })?;
    let s_len = dims
        .s
        .checked_mul(dims.h)
        .and_then(|v| v.checked_mul(dims.d))
        .and_then(|v| v.checked_mul(dims.dv))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "linear_attn_scan",
            detail: "state buffer size overflows usize".to_string(),
        })?;
    let mut o_tmp = vec![0.0f32; o_len];
    let mut s_tmp = vec![0.0f32; s_len];
    run_scan(
        q, k, v, alpha, beta, state_in, seq, dims, chunk, chunked, &mut o_tmp, &mut s_tmp,
    );
    for (idx, &val) in o_tmp.iter().enumerate() {
        o.write_f32(idx, val);
    }
    for (idx, &val) in s_tmp.iter().enumerate() {
        state_out.write_f32(idx, val);
    }
    Ok(())
}

/// 64-bit reference linear-attention scan for testing (Spec 1 §4.E).
///
/// Independent `f64` path: plain nested loops over `&[f64]` slices, never
/// calling either T0 form. Mirrors the shared recurrence in `f64` with
/// per-segment state slots. Returns `(o [T, H, Dv], state_out [S, H, D, Dv])`.
/// Slice lengths and every extent product are validated with typed errors;
/// there is no silent empty fallback.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_scan_f64_reference(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    alpha: &[f64],
    beta: &[f64],
    t: usize,
    h: usize,
    d: usize,
    dv: usize,
    state_in: &[f64],
    s: usize,
    seq_lens: &[u32],
) -> Result<(Vec<f64>, Vec<f64>), T0Error> {
    const OP: &str = "linear_attn_scan";
    let thd = t
        .checked_mul(h)
        .and_then(|v| v.checked_mul(d))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "T * H * D overflows usize".to_string(),
        })?;
    let thdv = t
        .checked_mul(h)
        .and_then(|v| v.checked_mul(dv))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "T * H * Dv overflows usize".to_string(),
        })?;
    let th = t
        .checked_mul(h)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * H overflows usize for T={t}, H={h}"),
        })?;
    let shddv = s
        .checked_mul(h)
        .and_then(|v| v.checked_mul(d))
        .and_then(|v| v.checked_mul(dv))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "S * H * D * Dv overflows usize".to_string(),
        })?;
    let mut problems = Vec::new();
    for (name, slice, expected, detail) in [
        ("q", q, thd, "q length must equal T * H * D"),
        ("k", k, thd, "k length must equal T * H * D"),
        ("v", v, thdv, "v length must equal T * H * Dv"),
        ("alpha", alpha, th, "alpha length must equal T * H"),
        ("beta", beta, th, "beta length must equal T * H"),
        (
            "state_in",
            state_in,
            shddv,
            "state_in length must equal S * H * D * Dv",
        ),
    ] {
        if slice.len() != expected {
            problems.push(T0Error::ShapeLengthMismatch {
                op: OP,
                tensor: name,
                expected,
                got: slice.len(),
                detail: detail.to_string(),
            });
        }
    }
    if seq_lens.len() != s {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "seq_lens",
            expected: s,
            got: seq_lens.len(),
            detail: "seq_lens length must equal S".to_string(),
        });
    }
    T0Error::from_problems(problems)?;
    let mut o = vec![0.0f64; thdv];
    let mut s_out = vec![0.0f64; shddv];
    let mut base = 0usize;
    for (slot, &len_u32) in seq_lens.iter().enumerate() {
        let len = len_u32 as usize;
        for head in 0..h {
            let mut s_mat = vec![0.0f64; d * dv];
            for i in 0..d {
                for j in 0..dv {
                    s_mat[i * dv + j] = state_in[(slot * h + head) * d * dv + i * dv + j];
                }
            }
            for r in 0..len {
                let gt = base + r;
                let a = alpha[gt * h + head];
                let b = beta[gt * h + head];
                for i in 0..d {
                    let ki = k[(gt * h + head) * d + i];
                    for j in 0..dv {
                        let vj = v[(gt * h + head) * dv + j];
                        s_mat[i * dv + j] = a * s_mat[i * dv + j] + b * ki * vj;
                    }
                }
                for j in 0..dv {
                    let mut acc = 0.0f64;
                    for i in 0..d {
                        acc += q[(gt * h + head) * d + i] * s_mat[i * dv + j];
                    }
                    o[(gt * h + head) * dv + j] = acc;
                }
            }
            for i in 0..d {
                for j in 0..dv {
                    s_out[(slot * h + head) * d * dv + i * dv + j] = s_mat[i * dv + j];
                }
            }
        }
        base += len;
    }
    Ok((o, s_out))
}

/// Token step writing directly into the staged output buffer.
///
/// `S_t = alpha·S_{t-1} + beta·(k_t ⊗ v_t)` over `[D, Dv]`, then
/// `o_t[j] = Σ_i q_t[i]·S_t[i,j]`, all ascending in `f32`.
#[allow(clippy::too_many_arguments)]
fn scan_step_into(
    q: &TensorView<'_>,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    alpha: &TensorView<'_>,
    beta: &TensorView<'_>,
    dims: &ScanDims,
    gt: usize,
    head: usize,
    s_mat: &mut [f32],
    o_tmp: &mut [f32],
) {
    let (h, d, dv) = (dims.h, dims.d, dims.dv);
    let a = alpha.read_f32(gt * h + head);
    let b = beta.read_f32(gt * h + head);
    for i in 0..d {
        let ki = k.read_f32((gt * h + head) * d + i);
        for j in 0..dv {
            let vj = v.read_f32((gt * h + head) * dv + j);
            s_mat[i * dv + j] = a * s_mat[i * dv + j] + b * ki * vj;
        }
    }
    let base = (gt * h + head) * dv;
    for j in 0..dv {
        let mut acc = 0.0f32;
        for i in 0..d {
            acc += q.read_f32((gt * h + head) * d + i) * s_mat[i * dv + j];
        }
        o_tmp[base + j] = acc;
    }
}
