// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 implementation of `moe_ffn` op (Spec 1 §4.C, §6.2, Card A1.9).
//!
//! Sorts `(expert, token)` triples deterministically, runs a grouped GEMM per
//! expert, applies `act_mul`, projects down, and combines with routing weights
//! in sorted order (all in `f32`, ascending index order per Spec 1 §6.1).

use r9v_format::records::{E4M3Block128Scale, I4KSuperblock, I8Block128Scale, I8RowScale};
use r9v_format::scales::E4m3;
use r9v_format::{
    l0_region_bytes, l0_row_stride_bytes, l1_forward_index, scale_geometry, FormatError, Layout,
    PaddedDims, SchemeId,
};
use r9v_ir::{
    ActMulOp, ActivationKind, DType, Epilogue, LayoutId, MatmulOp, MoeFfnOp, QuantScheme,
};

use crate::act_mul::act_mul;
use crate::buffer::{TensorData, TensorView, TensorViewMut};
use crate::dtype::f16_to_f32;
use crate::error::{u64_to_usize, T0Error};
use crate::matmul::matmul_with_scales;

/// Executes scalar T0 mixture-of-experts feed-forward (Spec 1 §4.C, §6.2, Card A1.9).
///
/// Signature:
/// - `x`: `[T, Dm]` (matmul activation family: f16|bf16|i8 PerToken|i8 PerBlock32|e4m3 PerToken)
/// - `expert_ids`: `[T, K]` (`u32`, every id `< E`)
/// - `weights`: `[T, K]` (`f32` routing weights)
/// - `w_gate_up`: `[E, 2·Dff, Dm]` (`Weight`, `Device|Tiered`)
/// - `w_down`: `[E, Dm, Dff]` (`Weight`, `Device|Tiered`)
/// - `y`: `[T, Dm]` (`out_dtype`)
///
/// Scales travel out-of-band per SI-18/SI-52: each explicit scale parameter
/// falls back to the attached view (`x.scale()` and friends, mirroring
/// `matmul`) when `None`.
///
/// Algorithm:
/// (a) validate everything and collect every bound failure before touching
///     `y`; (b) stable sort `(expert, token, k)` triples by expert then token
///     (mirrors spec 4 §5.6 pre-pass and the `scatter_add_rows` sorted
///     precedent); (c) per expert with ≥ 1 token, run the gathered rows
///     through `matmul_with_scales` over the full gate/up matrices (per-row
///     results are batch-independent, so full-matrix GEMM sliced to the
///     expert's columns is bit-identical to a per-expert sub-matrix GEMM),
///     then `act_mul` in `f32`, then the down projection; (d) combine
///     `y[t] += w[t,k]·out` in `f32` in ascending `(expert, token)` order
///     with `y` zero-initialized first.
///
/// Gate/up row layout: rows `[0, Dff)` are gate, `[Dff, 2·Dff)` are up
/// (gate-major halves; the spec's `[E, 2·Dff, Dm]` is silent on order —
/// SI-50; spec 4 §5.6 "gate/up interleaved" describes kernel access, not
/// storage).
///
/// Down-projection numerics mirror matmul Branch D (f32 accumulation over
/// per-element dequantized weights in ascending-K order); the equivalence is
/// proven by bit-exact tests against `matmul_with_scales` itself.
///
/// `shared_experts` is ignored for compute: the spec assigns the shared path
/// to a plain graph-level `matmul`, and this attr only sizes internal buffers
/// (of which T0 has none).
///
/// DECISION(A1.9): accept `shared_experts <= E`, ignore it for compute;
/// rejected wiring a second GEMM path because the graph already owns it.
/// Per SI-49.
///
/// Numerics: same as `matmul` per expert (`moe_ffn_gemm_numerics`); combine in
/// `f32` in sorted order. Batch invariant by construction.
///
/// DECISION(A1.9): gate rows are `[0, Dff)` and up rows `[Dff, 2·Dff)`;
/// rejected interleaved row alternation because no storage order is stated and
/// gate-major halves keep each projection a contiguous row range. Per SI-50.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn(
    op: &MoeFfnOp,
    x: &TensorView<'_>,
    expert_ids: &TensorView<'_>,
    weights: &TensorView<'_>,
    w_gate_up: &TensorView<'_>,
    w_gate_up_scale: Option<&TensorView<'_>>,
    w_down: &TensorView<'_>,
    w_down_scale: Option<&TensorView<'_>>,
    x_scale: Option<&TensorView<'_>>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    expert_ids.validate_backing("expert_ids")?;
    weights.validate_backing("weights")?;
    w_gate_up.validate_backing("w_gate_up")?;
    w_down.validate_backing("w_down")?;
    y.validate_backing("y")?;
    if let Some(s) = w_gate_up_scale {
        s.validate_backing("w_gate_up_scale")?;
    }
    if let Some(s) = w_down_scale {
        s.validate_backing("w_down_scale")?;
    }
    if let Some(s) = x_scale {
        s.validate_backing("x_scale")?;
    }

    // Scales fall back to attached views, mirroring `matmul` (SI-18, SI-52).
    let x_scale = x_scale.or_else(|| x.scale());
    let w_gate_up_scale = w_gate_up_scale.or_else(|| w_gate_up.scale());
    let w_down_scale = w_down_scale.or_else(|| w_down.scale());
    if let Some(s) = x_scale {
        s.validate_backing("x_scale")?;
    }
    if let Some(s) = w_gate_up_scale {
        s.validate_backing("w_gate_up_scale")?;
    }
    if let Some(s) = w_down_scale {
        s.validate_backing("w_down_scale")?;
    }

    let mut problems = Vec::new();

    if x.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "x",
            expected: 2,
            got: x.rank(),
            shape: x.shape().to_vec(),
        });
    }
    if expert_ids.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "expert_ids",
            expected: 2,
            got: expert_ids.rank(),
            shape: expert_ids.shape().to_vec(),
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
    if w_gate_up.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "w_gate_up",
            expected: 3,
            got: w_gate_up.rank(),
            shape: w_gate_up.shape().to_vec(),
        });
    }
    if w_down.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "w_down",
            expected: 3,
            got: w_down.rank(),
            shape: w_down.shape().to_vec(),
        });
    }
    if y.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "y",
            expected: 2,
            got: y.rank(),
            shape: y.shape().to_vec(),
        });
    }

    // Layouts mirror matmul: activations CONTIGUOUS|L0, weights L0|L1|CONTIGUOUS.
    for (name, v) in [("x", x), ("expert_ids", expert_ids), ("weights", weights)] {
        if v.layout() != LayoutId::CONTIGUOUS && v.layout() != LayoutId::L0 {
            problems.push(T0Error::LayoutMismatch {
                tensor: name,
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
                got: v.layout(),
            });
        }
    }
    for (name, v) in [("w_gate_up", w_gate_up), ("w_down", w_down)] {
        if v.layout() == LayoutId::L1S
            || (v.layout() != LayoutId::L0
                && v.layout() != LayoutId::L1
                && v.layout() != LayoutId::CONTIGUOUS)
        {
            problems.push(T0Error::LayoutMismatch {
                tensor: name,
                expected: vec![LayoutId::L0, LayoutId::L1],
                got: v.layout(),
            });
        }
    }
    if y.layout() != LayoutId::CONTIGUOUS && y.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "y",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: y.layout(),
        });
    }
    if let Some(xs) = x_scale {
        if xs.layout() != LayoutId::CONTIGUOUS && xs.layout() != LayoutId::L0 {
            problems.push(T0Error::LayoutMismatch {
                tensor: "x_scale",
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
                got: xs.layout(),
            });
        }
    }
    if let Some(ws) = w_down_scale {
        if ws.layout() == LayoutId::L1S
            || (ws.layout() != LayoutId::CONTIGUOUS
                && ws.layout() != LayoutId::L0
                && ws.layout() != LayoutId::L1)
        {
            problems.push(T0Error::LayoutMismatch {
                tensor: "w_down_scale",
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0, LayoutId::L1],
                got: ws.layout(),
            });
        }
    }

    // Activation dtype/quant family mirrors check_gemm_activation_operand.
    match (x.dtype(), x.quant()) {
        (DType::F16 | DType::Bf16, QuantScheme::None) => {}
        (DType::I8, QuantScheme::PerToken | QuantScheme::PerBlock32) => {}
        (DType::E4m3, QuantScheme::PerToken) => {}
        (DType::F32, _) => problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![DType::F16, DType::Bf16, DType::I8, DType::E4m3],
            got: x.dtype(),
        }),
        _ => problems.push(T0Error::QuantMismatch {
            tensor: "x",
            expected: vec![
                QuantScheme::None,
                QuantScheme::PerToken,
                QuantScheme::PerBlock32,
            ],
            got: x.quant(),
        }),
    }
    if expert_ids.dtype() != DType::U32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "expert_ids",
            expected: vec![DType::U32],
            got: expert_ids.dtype(),
        });
    }
    if weights.dtype() != DType::F32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "weights",
            expected: vec![DType::F32],
            got: weights.dtype(),
        });
    }
    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::InvalidAttribute {
            op: "moe_ffn",
            attribute: "out_dtype",
            reason: format!("must be f16, bf16, or f32, got {:?}", op.out_dtype),
        });
    }
    if y.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.out_dtype],
            got: y.dtype(),
        });
    }
    match op.act {
        ActivationKind::Silu
        | ActivationKind::Gelu
        | ActivationKind::GeluTanh
        | ActivationKind::Relu2
        | ActivationKind::Identity => {}
    }

    // Reserved schemes fail closed before dtype matching (mirrors matmul).
    for (name, v) in [("x", x), ("w_gate_up", w_gate_up), ("w_down", w_down)] {
        if let QuantScheme::Scheme(ir_s) = v.quant() {
            let sid = SchemeId::from_ir(ir_s)?;
            if !sid.is_native() {
                return Err(T0Error::Format(FormatError::ReservedScheme {
                    scheme: sid.name(),
                    owner: sid.owner_card(),
                }));
            }
        }
        let _ = name;
    }

    // Weight dtype/quant families mirror check_gemm_weight_operand.
    for (name, v) in [("w_gate_up", w_gate_up), ("w_down", w_down)] {
        if matches!(v.dtype(), DType::F32 | DType::Bf16) {
            problems.push(T0Error::DTypeMismatch {
                tensor: name,
                expected: vec![DType::F16, DType::I8, DType::I4, DType::E4m3],
                got: v.dtype(),
            });
        }
    }

    T0Error::from_problems(problems)?;

    // Resolve down-projection weight scheme (mirrors matmul Branch-D gating).
    let down_scheme = resolve_weight_scheme("w_down", w_down)?;

    let t = x.shape()[0];
    let dm = x.shape()[1];
    let k_dim = expert_ids.shape()[1];
    let e = w_gate_up.shape()[0];
    let dff = w_down.shape()[2];

    if t == 0 {
        return Err(T0Error::EmptyInput {
            op: "moe_ffn",
            tensor: "x",
        });
    }
    if dm == 0 {
        return Err(T0Error::EmptyInput {
            op: "moe_ffn",
            tensor: "x",
        });
    }
    if e == 0 {
        return Err(T0Error::EmptyInput {
            op: "moe_ffn",
            tensor: "w_gate_up",
        });
    }
    if dff == 0 {
        return Err(T0Error::EmptyInput {
            op: "moe_ffn",
            tensor: "w_down",
        });
    }
    if k_dim == 0 {
        return Err(T0Error::EmptyInput {
            op: "moe_ffn",
            tensor: "expert_ids",
        });
    }
    if op.shared_experts > e as u32 {
        return Err(T0Error::InvalidAttribute {
            op: "moe_ffn",
            attribute: "shared_experts",
            reason: format!(
                "shared_experts {} exceeds expert count E={e}",
                op.shared_experts
            ),
        });
    }
    for v in [t, dm, e, dff, k_dim] {
        u32::try_from(v).map_err(|_| T0Error::ArithmeticOverflow {
            op: "moe_ffn",
            detail: format!("dimension exceeds u32: {v}"),
        })?;
    }

    let mut problems = Vec::new();
    if weights.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "x",
            expected: t,
            tensor: "weights",
            got: weights.shape()[0],
        });
    }
    if weights.shape()[1] != k_dim {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "K",
            expected_from: "expert_ids",
            expected: k_dim,
            tensor: "weights",
            got: weights.shape()[1],
        });
    }
    if w_gate_up.shape()[1]
        != dff
            .checked_mul(2)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: format!("2 * Dff overflows usize for Dff={dff}"),
            })?
    {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "2Dff",
            expected_from: "w_down",
            expected: dff * 2,
            tensor: "w_gate_up",
            got: w_gate_up.shape()[1],
        });
    }
    if w_gate_up.shape()[2] != dm {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dm",
            expected_from: "x",
            expected: dm,
            tensor: "w_gate_up",
            got: w_gate_up.shape()[2],
        });
    }
    if w_down.shape()[0] != e {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "E",
            expected_from: "w_gate_up",
            expected: e,
            tensor: "w_down",
            got: w_down.shape()[0],
        });
    }
    if w_down.shape()[1] != dm {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dm",
            expected_from: "x",
            expected: dm,
            tensor: "w_down",
            got: w_down.shape()[1],
        });
    }
    if y.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "x",
            expected: t,
            tensor: "y",
            got: y.shape()[0],
        });
    }
    if y.shape()[1] != dm {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dm",
            expected_from: "x",
            expected: dm,
            tensor: "y",
            got: y.shape()[1],
        });
    }
    // Every expert id is bounds-checked before any output byte is touched.
    for row in 0..t {
        for slot in 0..k_dim {
            let pos = row * k_dim + slot;
            let id = expert_ids.try_read_u32(pos, "expert_ids").map_err(|_| {
                T0Error::BufferLengthMismatch {
                    tensor: "expert_ids",
                    buffer_len: expert_ids.backing_len(),
                    expected_len: pos + 1,
                    shape: expert_ids.shape().to_vec(),
                }
            })?;
            if (id as usize) >= e {
                problems.push(T0Error::RowIndexOutOfRange {
                    op: "moe_ffn",
                    tensor: "expert_ids",
                    position: pos,
                    index: id,
                    upper_bound: e,
                });
            }
        }
    }
    validate_moe_x_scale(x, x_scale, t, dm, &mut problems);
    validate_down_scales(w_down, w_down_scale, down_scheme, e, dff, &mut problems)?;
    T0Error::from_problems(problems)?;

    // Validated: stage all outputs; `y` is committed once at the end.
    let y_len = t
        .checked_mul(dm)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "moe_ffn",
            detail: format!("T * Dm overflows usize for T={t}, Dm={dm}"),
        })?;
    let mut y_tmp = vec![0.0f32; y_len];

    // Stable sort of (expert, token, k-slot) triples by expert then token.
    // `sort_by` is stable, and triples are pushed with ascending k slots, so
    // duplicate (expert, token) pairs keep ascending slot order.
    let mut triples: Vec<(u32, usize, usize)> = Vec::new();
    for row in 0..t {
        for slot in 0..k_dim {
            let id = expert_ids
                .try_read_u32(row * k_dim + slot, "expert_ids")
                .map_err(|_| T0Error::BufferLengthMismatch {
                    tensor: "expert_ids",
                    buffer_len: expert_ids.backing_len(),
                    expected_len: row * k_dim + slot + 1,
                    shape: expert_ids.shape().to_vec(),
                })?;
            triples.push((id, row, slot));
        }
    }
    triples.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Flattened expert-row view [E·2Dff, Dm] over the rank-3 bytes for the
    // inner matmul (row-major flattening preserves L0/CONTIGUOUS rows;
    // L1 tiles are interpreted over the flattened row space — DECISION(A1.9)
    // alongside SI-50, since no tiled rank-3 expert layout is specified).
    let gu_rows = e
        .checked_mul(dff)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "moe_ffn",
            detail: format!("E * 2Dff overflows usize for E={e}"),
        })?;
    let gu_bytes: &[u8] = match &w_gate_up.data {
        TensorData::Bytes(_, slice) => slice,
        _ => {
            return Err(T0Error::BackingRepresentationMismatch {
                op: "moe_ffn",
                dtype: w_gate_up.dtype(),
            });
        }
    };
    let gu_flat = TensorView::from_bytes(&[gu_rows, dm], w_gate_up.dtype(), gu_bytes)
        .with_quant(w_gate_up.quant())
        .with_layout(w_gate_up.layout());
    let gu_n = gu_rows;
    let gemm_op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    let act_op = ActMulOp {
        act: op.act,
        clamp: None,
    };

    let mut i = 0;
    while i < triples.len() {
        let expert = triples[i].0 as usize;
        let mut j = i + 1;
        while j < triples.len() && triples[j].0 == triples[i].0 {
            j += 1;
        }
        let run = &triples[i..j];
        // Unique tokens in ascending order (duplicate routing slots for one
        // token share a single gathered row; slots combine separately below).
        let mut tokens: Vec<usize> = run.iter().map(|&(_, tok, _)| tok).collect();
        tokens.sort();
        tokens.dedup();
        let position = |tok: usize| {
            tokens
                .iter()
                .position(|&v| v == tok)
                .ok_or(T0Error::RowIndexOutOfRange {
                    op: "moe_ffn",
                    tensor: "expert_ids",
                    position: tok,
                    index: 0,
                    upper_bound: t,
                })
        };
        let rows: Vec<(usize, usize)> = tokens.iter().map(|&tok| (tok, 0)).collect();
        let le = tokens.len();

        // Gather expert token rows (bit-exact, backing-preserving).
        let gathered = gather_activation_rows(x, &rows, dm, "x")?;
        let gx_view = gathered.view(&[le, dm], x.quant(), x.layout());
        let gx_scale_vec = gather_scale_rows(x, x_scale, &rows, t, dm)?;
        let gx_scale_view = gx_scale_vec.as_ref().map(|v| {
            let shape: &[usize] = match x.quant() {
                QuantScheme::PerBlock32 => &[le, dm / 32],
                _ => &[le],
            };
            TensorView::from_f32_slice(shape, v)
        });

        // Gate/up GEMM over the full matrices (row-independent, bit-identical
        // to a per-expert sub-matrix GEMM), then slice expert columns.
        let h_width = gu_n;
        let h_len = le
            .checked_mul(h_width)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: format!("Le * 2Dff overflows usize for Le={le}"),
            })?;
        let mut h_all = vec![0.0f32; h_len];
        {
            let mut h_all_view = TensorViewMut::from_f32_slice(&[le, h_width], &mut h_all);
            matmul_with_scales(
                &gemm_op,
                &gx_view,
                gx_scale_view.as_ref(),
                &gu_flat,
                w_gate_up_scale,
                None,
                None,
                &mut h_all_view,
            )?;
        }
        let base = expert
            .checked_mul(2)
            .and_then(|v| v.checked_mul(dff))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: format!("expert column base overflows usize for expert={expert}"),
            })?;
        let hd_len = le
            .checked_mul(dff)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: format!("Le * Dff overflows usize for Le={le}"),
            })?;
        let mut gate = vec![0.0f32; hd_len];
        let mut up = vec![0.0f32; hd_len];
        for r in 0..le {
            for d in 0..dff {
                gate[r * dff + d] = h_all[r * h_width + base + d];
                up[r * dff + d] = h_all[r * h_width + base + dff + d];
            }
        }
        let gate_view = TensorView::from_f32_slice(&[le, dff], &gate);
        let up_view = TensorView::from_f32_slice(&[le, dff], &up);
        let mut h = vec![0.0f32; hd_len];
        {
            let mut h_view = TensorViewMut::from_f32_slice(&[le, dff], &mut h);
            act_mul(&act_op, &gate_view, &up_view, &mut h_view)?;
        }

        // Down projection mirrors matmul Branch D over global expert rows.
        let dm_len = le
            .checked_mul(dm)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: format!("Le * Dm overflows usize for Le={le}"),
            })?;
        let mut down_out = vec![0.0f32; dm_len];
        down_gemm(
            &h,
            le,
            expert,
            w_down,
            w_down_scale,
            down_scheme,
            dm,
            dff,
            &mut down_out,
        )?;

        // Weighted combine in ascending (expert, token) order: every routing
        // slot contributes, sharing one expert output row per token.
        let mut slots: Vec<(usize, usize)> =
            run.iter().map(|&(_, tok, slot)| (tok, slot)).collect();
        slots.sort();
        for (tok, slot) in slots {
            let r = position(tok)?;
            let w = weights.read_f32(tok * k_dim + slot);
            for d in 0..dm {
                y_tmp[tok * dm + d] += w * down_out[r * dm + d];
            }
        }

        i = j;
    }

    for (idx, &v) in y_tmp.iter().enumerate() {
        y.write_f32(idx, v);
    }
    Ok(())
}

/// Resolves the weight quant scheme for a MoE expert matrix (Spec 1 §4.C, §6.2).
///
/// Mirrors the matmul `w_scheme` gate: `F16+None` is unquantized, `I8+PerRow`
/// is `I8R`, and `Scheme(_)` ids must name a supported native scheme.
/// Anything else is a [`T0Error::QuantMismatch`]; reserved schemes fail closed
/// through [`FormatError::ReservedScheme`].
fn resolve_weight_scheme(
    tensor: &'static str,
    w: &TensorView<'_>,
) -> Result<Option<SchemeId>, T0Error> {
    match (w.dtype(), w.quant()) {
        (DType::F16, QuantScheme::None) => Ok(None),
        (DType::I8, QuantScheme::PerRow) => Ok(Some(SchemeId::I8R)),
        (DType::I8, QuantScheme::Scheme(ir_s)) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if sid != SchemeId::I8R && sid != SchemeId::I8B128 {
                return Err(T0Error::QuantMismatch {
                    tensor,
                    expected: vec![
                        QuantScheme::PerRow,
                        QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                        QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                    ],
                    got: w.quant(),
                });
            }
            Ok(Some(sid))
        }
        (DType::I4, QuantScheme::Scheme(ir_s)) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if sid != SchemeId::I4K {
                return Err(T0Error::QuantMismatch {
                    tensor,
                    expected: vec![QuantScheme::Scheme(SchemeId::I4K.to_ir())],
                    got: w.quant(),
                });
            }
            Ok(Some(sid))
        }
        (DType::E4m3, QuantScheme::Scheme(ir_s)) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if sid != SchemeId::E4M3B128 {
                return Err(T0Error::QuantMismatch {
                    tensor,
                    expected: vec![QuantScheme::Scheme(SchemeId::E4M3B128.to_ir())],
                    got: w.quant(),
                });
            }
            Ok(Some(sid))
        }
        _ => Err(T0Error::QuantMismatch {
            tensor,
            expected: vec![
                QuantScheme::None,
                QuantScheme::PerRow,
                QuantScheme::Scheme(SchemeId::I8R.to_ir()),
            ],
            got: w.quant(),
        }),
    }
}

/// Validates the MoE activation scales (mirrors matmul `x_scale` rules).
///
/// `PerToken` requires `[T] f32`; `PerBlock32` requires `Dm % 32 == 0` and
/// `[T, Dm/32] f32`; unquantized activations reject any scale. Scales must be
/// finite and non-negative.
fn validate_moe_x_scale(
    x: &TensorView<'_>,
    x_scale: Option<&TensorView<'_>>,
    t: usize,
    dm: usize,
    problems: &mut Vec<T0Error>,
) {
    match x.quant() {
        QuantScheme::PerToken => {
            if let Some(xs) = x_scale {
                if xs.rank() != 1 || xs.shape()[0] != t {
                    problems.push(T0Error::DimensionMismatch {
                        dim_name: "T",
                        expected_from: "x",
                        expected: t,
                        tensor: "x_scale",
                        got: xs.shape().first().copied().unwrap_or(0),
                    });
                }
                if xs.dtype() != DType::F32 {
                    problems.push(T0Error::DTypeMismatch {
                        tensor: "x_scale",
                        expected: vec![DType::F32],
                        got: xs.dtype(),
                    });
                }
                if xs.num_elements() < t || xs.backing_len() < t {
                    problems.push(T0Error::BufferLengthMismatch {
                        tensor: "x_scale",
                        buffer_len: xs.backing_len(),
                        expected_len: t,
                        shape: xs.shape().to_vec(),
                    });
                }
            } else {
                problems.push(T0Error::MissingOperand {
                    op: "moe_ffn",
                    operand: "x_scale",
                    detail: "x_scale required for PerToken activations".to_string(),
                });
            }
        }
        QuantScheme::PerBlock32 => {
            if !dm.is_multiple_of(32) {
                problems.push(T0Error::DimensionMismatch {
                    dim_name: "Dm",
                    expected_from: "block_size_32",
                    expected: 32,
                    tensor: "x",
                    got: dm,
                });
            }
            if let Some(xs) = x_scale {
                let expected_blocks = dm / 32;
                if xs.rank() != 2 || xs.shape()[0] != t || xs.shape()[1] != expected_blocks {
                    problems.push(T0Error::DimensionMismatch {
                        dim_name: "Dm_blocks",
                        expected_from: "dm_div_32",
                        expected: expected_blocks,
                        tensor: "x_scale",
                        got: if xs.rank() > 1 { xs.shape()[1] } else { 0 },
                    });
                }
                if xs.dtype() != DType::F32 {
                    problems.push(T0Error::DTypeMismatch {
                        tensor: "x_scale",
                        expected: vec![DType::F32],
                        got: xs.dtype(),
                    });
                }
            } else {
                problems.push(T0Error::MissingOperand {
                    op: "moe_ffn",
                    operand: "x_scale",
                    detail: "x_scale required for PerBlock32 activations".to_string(),
                });
            }
        }
        QuantScheme::None if x_scale.is_some() => {
            problems.push(T0Error::InvalidAttribute {
                op: "moe_ffn",
                attribute: "x_scale",
                reason: "x_scale provided for unquantized activations".to_string(),
            });
        }
        _ => {}
    }
    if let Some(xs) = x_scale {
        for idx in 0..xs.num_elements() {
            let s = xs.read_f32(idx);
            if !s.is_finite() || s < 0.0 {
                problems.push(T0Error::InvalidAttribute {
                    op: "moe_ffn",
                    attribute: "x_scale",
                    reason: format!(
                        "activation scale at index {idx} must be finite and non-negative, got {s}"
                    ),
                });
            }
        }
    }
}

/// Owned bit-exact row gather of an activation matrix (Spec 1 §4.C).
///
/// Preserves the source backing variant so the gathered view carries the same
/// dtype without any float round-trip.
enum OwnedRows {
    F16(Vec<u16>),
    Bf16(Vec<u16>),
    I8(Vec<i8>),
    Bytes(DType, Vec<u8>),
}

impl OwnedRows {
    /// Borrows the gathered rows as a view with the given shape and metadata.
    fn view(&self, shape: &[usize], quant: QuantScheme, layout: LayoutId) -> TensorView<'_> {
        match self {
            OwnedRows::F16(v) => TensorView::from_f16_slice(shape, v)
                .with_quant(quant)
                .with_layout(layout),
            OwnedRows::Bf16(v) => TensorView::from_bf16_slice(shape, v)
                .with_quant(quant)
                .with_layout(layout),
            OwnedRows::I8(v) => TensorView::from_i8_slice(shape, v)
                .with_quant(quant)
                .with_layout(layout),
            OwnedRows::Bytes(dtype, v) => TensorView::from_bytes(shape, *dtype, v)
                .with_quant(quant)
                .with_layout(layout),
        }
    }
}

/// Gathers `rows` (token indices) of `x` into an owned bit-exact buffer.
fn gather_activation_rows(
    x: &TensorView<'_>,
    rows: &[(usize, usize)],
    dm: usize,
    tensor: &'static str,
) -> Result<OwnedRows, T0Error> {
    let check_len = |have: usize, need: usize| {
        if have < need {
            return Err(T0Error::BufferLengthMismatch {
                tensor,
                buffer_len: have,
                expected_len: need,
                shape: x.shape().to_vec(),
            });
        }
        Ok(())
    };
    let row_elems = rows
        .len()
        .checked_mul(dm)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "moe_ffn",
            detail: "gather row count overflows usize".to_string(),
        })?;
    match &x.data {
        TensorData::F16(s) => {
            check_len(s.len(), row_elems)?;
            let mut out = vec![0u16; row_elems];
            for (r, &(tok, _)) in rows.iter().enumerate() {
                out[r * dm..(r + 1) * dm].copy_from_slice(&s[tok * dm..(tok + 1) * dm]);
            }
            Ok(OwnedRows::F16(out))
        }
        TensorData::Bf16(s) => {
            check_len(s.len(), row_elems)?;
            let mut out = vec![0u16; row_elems];
            for (r, &(tok, _)) in rows.iter().enumerate() {
                out[r * dm..(r + 1) * dm].copy_from_slice(&s[tok * dm..(tok + 1) * dm]);
            }
            Ok(OwnedRows::Bf16(out))
        }
        TensorData::I8(s) => {
            check_len(s.len(), row_elems)?;
            let mut out = vec![0i8; row_elems];
            for (r, &(tok, _)) in rows.iter().enumerate() {
                out[r * dm..(r + 1) * dm].copy_from_slice(&s[tok * dm..(tok + 1) * dm]);
            }
            Ok(OwnedRows::I8(out))
        }
        TensorData::Bytes(dtype, s) => {
            let stride = dm
                .checked_mul(crate::dtype::dtype_element_size(*dtype))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "moe_ffn",
                    detail: format!("gather row stride overflows usize for Dm={dm}"),
                })?;
            let row_bytes =
                rows.len()
                    .checked_mul(stride)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "moe_ffn",
                        detail: "gather byte count overflows usize".to_string(),
                    })?;
            check_len(s.len(), row_bytes)?;
            let mut out = vec![0u8; row_bytes];
            for (r, &(tok, _)) in rows.iter().enumerate() {
                out[r * stride..(r + 1) * stride]
                    .copy_from_slice(&s[tok * stride..(tok + 1) * stride]);
            }
            Ok(OwnedRows::Bytes(*dtype, out))
        }
        _ => Err(T0Error::DTypeMismatch {
            tensor,
            expected: vec![DType::F16, DType::Bf16, DType::I8, DType::E4m3],
            got: x.dtype(),
        }),
    }
}

/// Gathers activation scale rows for the expert's tokens (exact `f32` copy).
///
/// Returns `None` when no scale view is present (valid only for unquantized
/// activations, checked by [`validate_moe_x_scale`]).
fn gather_scale_rows(
    x: &TensorView<'_>,
    x_scale: Option<&TensorView<'_>>,
    rows: &[(usize, usize)],
    t: usize,
    dm: usize,
) -> Result<Option<Vec<f32>>, T0Error> {
    let Some(xs) = x_scale else {
        return Ok(None);
    };
    match x.quant() {
        QuantScheme::PerToken => {
            if xs.shape().first().copied().unwrap_or(0) < t {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "x_scale",
                    buffer_len: xs.backing_len(),
                    expected_len: t,
                    shape: xs.shape().to_vec(),
                });
            }
            let mut out = vec![0.0f32; rows.len()];
            for (r, &(tok, _)) in rows.iter().enumerate() {
                out[r] = xs.read_f32(tok);
            }
            Ok(Some(out))
        }
        QuantScheme::PerBlock32 => {
            let blocks = dm / 32;
            if xs.shape().first().copied().unwrap_or(0) < t {
                let need = t.saturating_mul(blocks);
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "x_scale",
                    buffer_len: xs.backing_len(),
                    expected_len: need,
                    shape: xs.shape().to_vec(),
                });
            }
            let out_len =
                rows.len()
                    .checked_mul(blocks)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "moe_ffn",
                        detail: "gathered block-scale count overflows usize".to_string(),
                    })?;
            let mut out = vec![0.0f32; out_len];
            for (r, &(tok, _)) in rows.iter().enumerate() {
                for b in 0..blocks {
                    out[r * blocks + b] = xs.read_f32(tok * blocks + b);
                }
            }
            Ok(Some(out))
        }
        _ => Ok(None),
    }
}

/// Validates the down-projection weight buffer geometry and scale carrier.
///
/// Mirrors the matmul separate/inline scale contract over the full
/// `[E·Dm, Dff]` row space (`n = E·Dm`, `k = Dff`): L0 inline strides,
/// separate-carrier shapes, and L1 region sizes. Tensor names are
/// `w_down`/`w_down_scale` under op `moe_ffn`.
fn validate_down_scales(
    w_down: &TensorView<'_>,
    w_down_scale: Option<&TensorView<'_>>,
    scheme: Option<SchemeId>,
    e: usize,
    dff: usize,
    problems: &mut Vec<T0Error>,
) -> Result<(), T0Error> {
    // Full row count comes from the validated [E, Dm, Dff] shape.
    let dm = w_down.shape()[1];
    let n_rows = e
        .checked_mul(dm)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "moe_ffn",
            detail: format!("E * Dm overflows usize for E={e}, Dm={dm}"),
        })?;
    let k = dff;
    let n_u32 = u32::try_from(n_rows).map_err(|_| T0Error::ArithmeticOverflow {
        op: "moe_ffn",
        detail: format!("down-projection row count exceeds u32: {n_rows}"),
    })?;
    let k_u32 = u32::try_from(k).map_err(|_| T0Error::ArithmeticOverflow {
        op: "moe_ffn",
        detail: format!("dimension Dff exceeds u32: {k}"),
    })?;

    match scheme {
        Some(SchemeId::I4K) if !k.is_multiple_of(256) => {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Dff",
                expected_from: "superblock_256",
                expected: 256,
                tensor: "w_down",
                got: k,
            });
        }
        Some(SchemeId::I8B128) | Some(SchemeId::E4M3B128) if !k.is_multiple_of(128) => {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Dff",
                expected_from: "block_size_128",
                expected: 128,
                tensor: "w_down",
                got: k,
            });
        }
        _ => {}
    }

    let is_l1 = w_down.layout() == LayoutId::L1;
    let w_bytes_len = match &w_down.data {
        TensorData::Bytes(_, slice) => slice.len(),
        _ => {
            problems.push(T0Error::BackingRepresentationMismatch {
                op: "moe_ffn",
                dtype: w_down.dtype(),
            });
            0
        }
    };

    let superblock_k = match scheme {
        Some(SchemeId::I4K) => 256,
        Some(SchemeId::I8B128) | Some(SchemeId::E4M3B128) => 128,
        _ => 16,
    };
    let w_dims = PaddedDims::new(n_u32, k_u32, Some(superblock_k))?;
    let elem_bytes = match w_down.dtype() {
        DType::F16 => 2,
        DType::I8 | DType::E4m3 => 1,
        DType::I4 => 1,
        _ => 1,
    };
    let values_bytes = if w_down.dtype() == DType::I4 {
        n_rows
            .checked_mul(k)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: "down-projection I4 values size overflows usize".to_string(),
            })?
            / 2
    } else {
        n_rows
            .checked_mul(k)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: "down-projection values size overflows usize".to_string(),
            })?
    };

    if let Some(ws) = w_down_scale {
        if ws.layout() != LayoutId::CONTIGUOUS {
            return Err(T0Error::LayoutMismatch {
                tensor: "w_down_scale",
                expected: vec![LayoutId::CONTIGUOUS],
                got: ws.layout(),
            });
        }
        let ws_bytes_len = match &ws.data {
            TensorData::Bytes(_, slice) => slice.len(),
            _ => {
                return Err(T0Error::BackingRepresentationMismatch {
                    op: "moe_ffn",
                    dtype: ws.dtype(),
                });
            }
        };
        if w_bytes_len < values_bytes {
            return Err(T0Error::BufferLengthMismatch {
                tensor: "w_down",
                buffer_len: w_bytes_len,
                expected_len: values_bytes,
                shape: w_down.shape().to_vec(),
            });
        }
        if let Some(sid) = scheme {
            if is_l1 {
                let geom = scale_geometry(sid, Layout::L1, &w_dims)?;
                let req_bytes = u64_to_usize(geom.region_bytes, "region_bytes")?;
                match sid {
                    SchemeId::I8R | SchemeId::I8B128 | SchemeId::E4M3B128 => {
                        if ws.dtype() != DType::F16 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "w_down_scale",
                                expected: vec![DType::F16],
                                got: ws.dtype(),
                            });
                        }
                        let n_blocks = u64_to_usize(geom.n_blocks, "n_blocks")?;
                        let k_blocks = u64_to_usize(geom.k_blocks, "k_blocks")?;
                        if ws.shape() != [n_blocks, k_blocks, 16] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "scale_shape",
                                expected_from: "scale_geometry",
                                expected: geom.records as usize,
                                tensor: "w_down_scale",
                                got: ws.num_elements(),
                            });
                        }
                    }
                    SchemeId::I4K => {
                        if ws.dtype() != DType::U32 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "w_down_scale",
                                expected: vec![DType::U32],
                                got: ws.dtype(),
                            });
                        }
                        let n_blocks = u64_to_usize(geom.n_blocks, "n_blocks")?;
                        let k_blocks = u64_to_usize(geom.k_blocks, "k_blocks")?;
                        if ws.shape() != [n_blocks, k_blocks, 16, 4] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "scale_shape",
                                expected_from: "scale_geometry",
                                expected: (geom.records * 4) as usize,
                                tensor: "w_down_scale",
                                got: ws.num_elements(),
                            });
                        }
                    }
                    _ => {}
                }
                if ws_bytes_len != req_bytes {
                    return Err(T0Error::BufferLengthMismatch {
                        tensor: "w_down_scale",
                        buffer_len: ws_bytes_len,
                        expected_len: req_bytes,
                        shape: ws.shape().to_vec(),
                    });
                }
            } else {
                match sid {
                    SchemeId::I8R => {
                        if ws.dtype() != DType::F16 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "w_down_scale",
                                expected: vec![DType::F16],
                                got: ws.dtype(),
                            });
                        }
                        if ws.shape() != [n_rows] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "N",
                                expected_from: "w_down",
                                expected: n_rows,
                                tensor: "w_down_scale",
                                got: ws.shape().first().copied().unwrap_or(0),
                            });
                        }
                        let req_bytes =
                            u64_to_usize(l0_region_bytes(n_u32, 2)?, "w_down_scale req_bytes")?;
                        if ws_bytes_len != req_bytes {
                            return Err(T0Error::BufferLengthMismatch {
                                tensor: "w_down_scale",
                                buffer_len: ws_bytes_len,
                                expected_len: req_bytes,
                                shape: ws.shape().to_vec(),
                            });
                        }
                    }
                    SchemeId::I8B128 | SchemeId::E4M3B128 => {
                        if ws.dtype() != DType::F16 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "w_down_scale",
                                expected: vec![DType::F16],
                                got: ws.dtype(),
                            });
                        }
                        let k_blocks = k / 128;
                        if ws.shape() != [n_rows, k_blocks] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "K_blocks",
                                expected_from: "w_down",
                                expected: k_blocks,
                                tensor: "w_down_scale",
                                got: if ws.rank() > 1 {
                                    ws.shape()[1]
                                } else {
                                    ws.shape().first().copied().unwrap_or(0)
                                },
                            });
                        }
                        let stride_u64 = (k_u32 / 128 * 2) as u64;
                        let req_bytes = u64_to_usize(
                            l0_region_bytes(n_u32, stride_u64)?,
                            "w_down_scale req_bytes",
                        )?;
                        if ws_bytes_len != req_bytes {
                            return Err(T0Error::BufferLengthMismatch {
                                tensor: "w_down_scale",
                                buffer_len: ws_bytes_len,
                                expected_len: req_bytes,
                                shape: ws.shape().to_vec(),
                            });
                        }
                    }
                    SchemeId::I4K => {
                        if ws.dtype() != DType::U32 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "w_down_scale",
                                expected: vec![DType::U32],
                                got: ws.dtype(),
                            });
                        }
                        let k_superblocks = k / 256;
                        if ws.shape() != [n_rows, k_superblocks, 4] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "K_superblocks",
                                expected_from: "w_down",
                                expected: k_superblocks,
                                tensor: "w_down_scale",
                                got: if ws.rank() > 1 {
                                    ws.shape()[1]
                                } else {
                                    ws.shape().first().copied().unwrap_or(0)
                                },
                            });
                        }
                        let stride_u64 = (k_u32 / 256 * 16) as u64;
                        let req_bytes = u64_to_usize(
                            l0_region_bytes(n_u32, stride_u64)?,
                            "w_down_scale req_bytes",
                        )?;
                        if ws_bytes_len != req_bytes {
                            return Err(T0Error::BufferLengthMismatch {
                                tensor: "w_down_scale",
                                buffer_len: ws_bytes_len,
                                expected_len: req_bytes,
                                shape: ws.shape().to_vec(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        } else {
            return Err(T0Error::InvalidAttribute {
                op: "moe_ffn",
                attribute: "w_down_scale",
                reason: "w_down_scale provided for unquantized weights".to_string(),
            });
        }
    } else if let Some(sid) = scheme {
        // Inline scales: length-gate the values region like matmul.
        let req_bytes = if is_l1 {
            let geom = scale_geometry(sid, Layout::L1, &w_dims)?;
            let l1_vals = if w_down.dtype() == DType::I4 {
                (w_dims.n_padded() as usize)
                    .checked_mul(w_dims.k_padded() as usize)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "moe_ffn",
                        detail: "L1 down values size overflows usize".to_string(),
                    })?
                    / 2
            } else {
                (w_dims.n_padded() as usize)
                    .checked_mul(w_dims.k_padded() as usize)
                    .and_then(|v| v.checked_mul(elem_bytes))
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "moe_ffn",
                        detail: "L1 down values size overflows usize".to_string(),
                    })?
            };
            l1_vals
                .checked_add(u64_to_usize(geom.region_bytes, "region_bytes")?)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "moe_ffn",
                    detail: "L1 down values + scales overflow".to_string(),
                })?
        } else {
            let stride = match sid {
                SchemeId::I8R => l0_row_stride_bytes(k_u32, 1, 1, 2)?,
                SchemeId::I8B128 | SchemeId::E4M3B128 => {
                    l0_row_stride_bytes(k_u32, 1, k_u32 / 128, 2)?
                }
                SchemeId::I4K => l0_row_stride_bytes(k_u32 / 2, 1, k_u32 / 256, 16)?,
                _ => k_u32 as u64,
            };
            u64_to_usize(l0_region_bytes(n_u32, stride)?, "l0_region_bytes")?
        };
        if w_bytes_len < req_bytes {
            problems.push(T0Error::BufferLengthMismatch {
                tensor: "w_down",
                buffer_len: w_bytes_len,
                expected_len: req_bytes,
                shape: w_down.shape().to_vec(),
            });
        }
    } else if w_bytes_len < values_bytes {
        problems.push(T0Error::BufferLengthMismatch {
            tensor: "w_down",
            buffer_len: w_bytes_len,
            expected_len: values_bytes,
            shape: w_down.shape().to_vec(),
        });
    }
    Ok(())
}

/// Down-projection GEMM over `f32` hidden rows (Spec 1 §4.C, §6.2).
///
/// Computes `out[r, d] = Σ_i h[r, i] · dequant(w_down[expert, d, i])` in `f32`
/// with ascending-`i` accumulation. Weight decoding mirrors matmul Branch D
/// (the `F16/BF16` general path with dequantization) element-for-element over
/// global expert rows `grow = expert·Dm + d`; bit-exact agreement with
/// `matmul_with_scales` on identical inputs is proven by test.
#[allow(clippy::too_many_arguments)]
fn down_gemm(
    h: &[f32],
    le: usize,
    expert: usize,
    w_down: &TensorView<'_>,
    w_down_scale: Option<&TensorView<'_>>,
    scheme: Option<SchemeId>,
    dm: usize,
    dff: usize,
    out: &mut [f32],
) -> Result<(), T0Error> {
    let w_bytes: &[u8] = match &w_down.data {
        TensorData::Bytes(_, slice) => slice,
        _ => {
            return Err(T0Error::BackingRepresentationMismatch {
                op: "moe_ffn",
                dtype: w_down.dtype(),
            });
        }
    };
    let is_l1 = w_down.layout() == LayoutId::L1;
    let n_rows = w_down.shape()[0]
        .checked_mul(w_down.shape()[1])
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "moe_ffn",
            detail: "down-projection row count overflows usize".to_string(),
        })?;
    let n_u32 = u32::try_from(n_rows).map_err(|_| T0Error::ArithmeticOverflow {
        op: "moe_ffn",
        detail: format!("down-projection row count exceeds u32: {n_rows}"),
    })?;
    let k_u32 = u32::try_from(dff).map_err(|_| T0Error::ArithmeticOverflow {
        op: "moe_ffn",
        detail: format!("dimension Dff exceeds u32: {dff}"),
    })?;
    let superblock_k = match scheme {
        Some(SchemeId::I4K) => 256,
        Some(SchemeId::I8B128) | Some(SchemeId::E4M3B128) => 128,
        _ => 16,
    };
    let w_dims = PaddedDims::new(n_u32, k_u32, Some(superblock_k))?;
    // Separate-scale slice setup mirrors matmul: explicit carrier, else the L1
    // tail region, else empty (L0 inline scales live in the values bytes).
    let w_scales_slice: &[u8] = if let Some(ws) = w_down_scale {
        match &ws.data {
            TensorData::Bytes(_, slice) => slice,
            _ => {
                return Err(T0Error::BackingRepresentationMismatch {
                    op: "moe_ffn",
                    dtype: ws.dtype(),
                });
            }
        }
    } else if is_l1 {
        let l1_vals = l1_values_len(w_down.dtype(), &w_dims)?;
        if w_bytes.len() > l1_vals {
            &w_bytes[l1_vals..]
        } else {
            &[]
        }
    } else {
        &[]
    };

    // Weight rows depend only on (expert, output channel): global row
    // `expert·Dm + d`. The token position `r` selects the hidden row only.
    let expert_base = expert
        .checked_mul(dm)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "moe_ffn",
            detail: format!("expert down base overflows usize for expert={expert}"),
        })?;
    for r in 0..le {
        for d in 0..dm {
            let grow_d = expert_base
                .checked_add(d)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "moe_ffn",
                    detail: "global down row overflows usize".to_string(),
                })?;
            let mut acc = 0.0f32;
            for k_idx in 0..dff {
                let w_val = down_weight_f32(
                    w_bytes,
                    w_scales_slice,
                    w_down_scale.is_some(),
                    is_l1,
                    &w_dims,
                    scheme,
                    grow_d,
                    k_idx,
                    dff,
                )?;
                acc += h[r * dff + k_idx] * w_val;
            }
            out[r * dm + d] = acc;
        }
    }
    Ok(())
}

/// Padded L1 values length in bytes for the down-projection matrix.
fn l1_values_len(dtype: DType, dims: &PaddedDims) -> Result<usize, T0Error> {
    let elem_bytes = match dtype {
        DType::F16 => 2,
        DType::I8 | DType::E4m3 => 1,
        DType::I4 => 1,
        _ => 1,
    };
    if dtype == DType::I4 {
        (dims.n_padded() as usize)
            .checked_mul(dims.k_padded() as usize)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: "L1 down values size overflows usize".to_string(),
            })
            .map(|v| v / 2)
    } else {
        (dims.n_padded() as usize)
            .checked_mul(dims.k_padded() as usize)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "moe_ffn",
                detail: "L1 down values size overflows usize".to_string(),
            })
    }
}

/// Decodes one down-projection weight element to `f32`.
///
/// Element-for-element mirror of the matmul Branch D weight arms over global
/// row `grow` (the expert's row offset is folded into `grow`), with tensor
/// names `w_down`/`w_down_scale` under op `moe_ffn`.
#[allow(clippy::too_many_arguments)]
fn down_weight_f32(
    w_bytes: &[u8],
    w_scales_slice: &[u8],
    has_separate_scales: bool,
    is_l1: bool,
    w_dims: &PaddedDims,
    scheme: Option<SchemeId>,
    grow: usize,
    k_idx: usize,
    k: usize,
) -> Result<f32, T0Error> {
    const OP: &str = "moe_ffn";
    let grow_u32 = u32::try_from(grow).map_err(|_| T0Error::ArithmeticOverflow {
        op: OP,
        detail: format!("global down row exceeds u32: {grow}"),
    })?;
    let k_idx_u32 = u32::try_from(k_idx).map_err(|_| T0Error::ArithmeticOverflow {
        op: OP,
        detail: format!("k index exceeds u32: {k_idx}"),
    })?;
    let byte_at = |offset: usize, tensor: &'static str| {
        w_bytes
            .get(offset)
            .copied()
            .ok_or_else(|| T0Error::BufferLengthMismatch {
                tensor,
                buffer_len: w_bytes.len(),
                expected_len: offset + 1,
                shape: vec![grow, k],
            })
    };
    let bytes_at = |offset: usize, len: usize, tensor: &'static str| {
        w_bytes
            .get(offset..offset + len)
            .ok_or_else(|| T0Error::BufferLengthMismatch {
                tensor,
                buffer_len: w_bytes.len(),
                expected_len: offset + len,
                shape: vec![grow, k],
            })
    };
    let scale_bytes_at = |offset: usize, len: usize| {
        w_scales_slice
            .get(offset..offset + len)
            .ok_or_else(|| T0Error::BufferLengthMismatch {
                tensor: if has_separate_scales {
                    "w_down_scale"
                } else {
                    "w_down"
                },
                buffer_len: w_scales_slice.len(),
                expected_len: offset + len,
                shape: vec![grow, k],
            })
    };
    // Checked `row * stride + col` for L0 addressing (debug builds panic on
    // overflow, so every offset expression goes through here).
    let row_offset = |row: usize, stride: usize, col: usize| {
        row.checked_mul(stride)
            .and_then(|v| v.checked_add(col))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: OP,
                detail: format!("down-projection offset overflows usize for row={row}"),
            })
    };
    match scheme {
        None => {
            if !is_l1 {
                let offset = grow
                    .checked_mul(k)
                    .and_then(|v| v.checked_add(k_idx))
                    .and_then(|v| v.checked_mul(2))
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: OP,
                        detail: "F16 down offset overflows usize".to_string(),
                    })?;
                let raw = bytes_at(offset, 2, "w_down")?;
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                Ok(f16_to_f32(bits))
            } else {
                let elem_idx = l1_forward_index(grow_u32, k_idx_u32, w_dims)?;
                let offset = u64_to_usize(
                    elem_idx
                        .checked_mul(2)
                        .ok_or_else(|| T0Error::ArithmeticOverflow {
                            op: OP,
                            detail: "L1 F16 down offset overflows".to_string(),
                        })?,
                    "l1_f16_offset",
                )?;
                let raw = bytes_at(offset, 2, "w_down")?;
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                Ok(f16_to_f32(bits))
            }
        }
        Some(SchemeId::I8R) => {
            let w_row_stride = if !is_l1 {
                if has_separate_scales {
                    k
                } else {
                    k + 2
                }
            } else {
                0
            };
            let w_s = if !is_l1 {
                if has_separate_scales {
                    let raw = scale_bytes_at(row_offset(grow, 2, 0)?, 2)?;
                    I8RowScale::from_bytes([raw[0], raw[1]]).value(grow as u64)?
                } else {
                    let off = row_offset(grow, w_row_stride, k)?;
                    let raw = bytes_at(off, 2, "w_down")?;
                    I8RowScale::from_bytes([raw[0], raw[1]]).value(grow as u64)?
                }
            } else {
                let geom = scale_geometry(SchemeId::I8R, Layout::L1, w_dims)?;
                let scale_offset = u64_to_usize(
                    geom.record_offset((grow_u32 / 16) as u64, 0, grow_u32 % 16)?,
                    "record_offset",
                )?;
                let raw = scale_bytes_at(scale_offset, 2)?;
                I8RowScale::from_bytes([raw[0], raw[1]]).value(grow as u64)?
            };
            let q = if !is_l1 {
                byte_at(row_offset(grow, w_row_stride, k_idx)?, "w_down")? as i8
            } else {
                let offset = u64_to_usize(
                    l1_forward_index(grow_u32, k_idx_u32, w_dims)?,
                    "l1_forward_index",
                )?;
                byte_at(offset, "w_down")? as i8
            };
            Ok((q as f32) * w_s)
        }
        Some(SchemeId::I8B128) => {
            let b = k_idx / 128;
            let k_blocks = k / 128;
            let w_row_stride = if !is_l1 {
                if has_separate_scales {
                    k
                } else {
                    k + k_blocks * 2
                }
            } else {
                0
            };
            let w_s = if !is_l1 {
                if has_separate_scales {
                    let base = row_offset(grow, k_blocks, b)?;
                    let raw = scale_bytes_at(row_offset(base, 2, 0)?, 2)?;
                    I8Block128Scale::from_bytes([raw[0], raw[1]]).value(b as u64)?
                } else {
                    let off = row_offset(grow, w_row_stride, k + b * 2)?;
                    let raw = bytes_at(off, 2, "w_down")?;
                    I8Block128Scale::from_bytes([raw[0], raw[1]]).value(b as u64)?
                }
            } else {
                let geom = scale_geometry(SchemeId::I8B128, Layout::L1, w_dims)?;
                let scale_offset = u64_to_usize(
                    geom.record_offset((grow_u32 / 16) as u64, b as u64, grow_u32 % 16)?,
                    "record_offset",
                )?;
                let raw = scale_bytes_at(scale_offset, 2)?;
                I8Block128Scale::from_bytes([raw[0], raw[1]]).value(b as u64)?
            };
            let q = if !is_l1 {
                byte_at(row_offset(grow, w_row_stride, k_idx)?, "w_down")? as i8
            } else {
                let offset = u64_to_usize(
                    l1_forward_index(grow_u32, k_idx_u32, w_dims)?,
                    "l1_forward_index",
                )?;
                byte_at(offset, "w_down")? as i8
            };
            Ok((q as f32) * w_s)
        }
        Some(SchemeId::I4K) => {
            let sb = k_idx / 256;
            let sub = (k_idx % 256) / 32;
            let k_superblocks = k / 256;
            let values_bytes_per_row = k / 2;
            let w_row_stride = if !is_l1 {
                if has_separate_scales {
                    values_bytes_per_row
                } else {
                    values_bytes_per_row + k_superblocks * 16
                }
            } else {
                0
            };
            let header = if !is_l1 {
                if has_separate_scales {
                    let base = row_offset(grow, k_superblocks, sb)?;
                    let raw = scale_bytes_at(row_offset(base, 16, 0)?, 16)?;
                    let mut arr = [0u8; 16];
                    arr.copy_from_slice(raw);
                    I4KSuperblock::from_bytes(&arr)
                } else {
                    let off = row_offset(grow, w_row_stride, values_bytes_per_row + sb * 16)?;
                    let raw = bytes_at(off, 16, "w_down")?;
                    let mut arr = [0u8; 16];
                    arr.copy_from_slice(raw);
                    I4KSuperblock::from_bytes(&arr)
                }
            } else {
                let geom = scale_geometry(SchemeId::I4K, Layout::L1, w_dims)?;
                let scale_offset = u64_to_usize(
                    geom.record_offset((grow_u32 / 16) as u64, sb as u64, grow_u32 % 16)?,
                    "record_offset",
                )?;
                let raw = scale_bytes_at(scale_offset, 16)?;
                let mut arr = [0u8; 16];
                arr.copy_from_slice(raw);
                I4KSuperblock::from_bytes(&arr)
            };
            let d = header.d_value(sb as u64)?;
            let dmin = header.dmin_value(sb as u64)?;
            let sc = header.scales();
            let mn = header.mins();
            let s_block = d * (sc[sub] as f32);
            let m_block = dmin * (mn[sub] as f32);
            let q = if !is_l1 {
                let byte = byte_at(row_offset(grow, w_row_stride, k_idx / 2)?, "w_down")?;
                if k_idx.is_multiple_of(2) {
                    (byte & 0x0F) as i32
                } else {
                    ((byte >> 4) & 0x0F) as i32
                }
            } else {
                let elem_idx = l1_forward_index(grow_u32, k_idx_u32, w_dims)?;
                let offset = u64_to_usize(elem_idx / 2, "l1_i4_offset")?;
                let byte = byte_at(offset, "w_down")?;
                if elem_idx % 2 == 0 {
                    (byte & 0x0F) as i32
                } else {
                    ((byte >> 4) & 0x0F) as i32
                }
            };
            Ok(s_block * (q as f32) - m_block)
        }
        Some(SchemeId::E4M3B128) => {
            let b = k_idx / 128;
            let k_blocks = k / 128;
            let w_row_stride = if !is_l1 {
                if has_separate_scales {
                    k
                } else {
                    k + k_blocks * 2
                }
            } else {
                0
            };
            let w_s = if !is_l1 {
                if has_separate_scales {
                    let base = row_offset(grow, k_blocks, b)?;
                    let raw = scale_bytes_at(row_offset(base, 2, 0)?, 2)?;
                    E4M3Block128Scale::from_bytes([raw[0], raw[1]]).value(b as u64)?
                } else {
                    let off = row_offset(grow, w_row_stride, k + b * 2)?;
                    let raw = bytes_at(off, 2, "w_down")?;
                    E4M3Block128Scale::from_bytes([raw[0], raw[1]]).value(b as u64)?
                }
            } else {
                let geom = scale_geometry(SchemeId::E4M3B128, Layout::L1, w_dims)?;
                let scale_offset = u64_to_usize(
                    geom.record_offset((grow_u32 / 16) as u64, b as u64, grow_u32 % 16)?,
                    "record_offset",
                )?;
                let raw = scale_bytes_at(scale_offset, 2)?;
                E4M3Block128Scale::from_bytes([raw[0], raw[1]]).value(b as u64)?
            };
            let byte = if !is_l1 {
                byte_at(row_offset(grow, w_row_stride, k_idx)?, "w_down")?
            } else {
                let offset = u64_to_usize(
                    l1_forward_index(grow_u32, k_idx_u32, w_dims)?,
                    "l1_forward_index",
                )?;
                byte_at(offset, "w_down")?
            };
            let e = E4m3::new(byte);
            e.check(k_idx as u64)?;
            Ok(e.to_f32() * w_s)
        }
        Some(scheme) => Err(T0Error::InvalidAttribute {
            op: OP,
            attribute: "w_quant",
            reason: format!("unsupported quant scheme in moe_ffn down path: {scheme:?}"),
        }),
    }
}

/// 64-bit reference MoE feed-forward for testing (Spec 1 §4.C, §6.1, §6.2).
///
/// Independent `f64` path: plain nested loops over `&[f64]` slices, never
/// calling [`moe_ffn`]. Expert weights arrive already dequantized to `f64`
/// (the test decodes them); the oracle then mirrors the T0 algorithm —
/// grouped per-expert GEMM, `act`, down projection, sorted weighted combine —
/// in `f64`. Slice lengths, expert bounds, and every extent product are
/// validated with typed errors; there is no silent empty fallback.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_f64_reference(
    x: &[f64],
    t: usize,
    dm: usize,
    expert_ids: &[u32],
    weights: &[f64],
    k_dim: usize,
    w_gate_up: &[f64],
    e: usize,
    dff: usize,
    w_down: &[f64],
    act: ActivationKind,
) -> Result<Vec<f64>, T0Error> {
    const OP: &str = "moe_ffn";
    let tdm = t
        .checked_mul(dm)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * Dm overflows usize for T={t}, Dm={dm}"),
        })?;
    let tk = t
        .checked_mul(k_dim)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * K overflows usize for T={t}, K={k_dim}"),
        })?;
    let gu_len = e
        .checked_mul(2)
        .and_then(|v| v.checked_mul(dff))
        .and_then(|v| v.checked_mul(dm))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "E * 2Dff * Dm overflows usize".to_string(),
        })?;
    let wd_len = e
        .checked_mul(dm)
        .and_then(|v| v.checked_mul(dff))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "E * Dm * Dff overflows usize".to_string(),
        })?;
    let mut problems = Vec::new();
    if x.len() != tdm {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "x",
            expected: tdm,
            got: x.len(),
            detail: "x length must equal T * Dm".to_string(),
        });
    }
    if expert_ids.len() != tk {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "expert_ids",
            expected: tk,
            got: expert_ids.len(),
            detail: "expert_ids length must equal T * K".to_string(),
        });
    }
    if weights.len() != tk {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "weights",
            expected: tk,
            got: weights.len(),
            detail: "weights length must equal T * K".to_string(),
        });
    }
    if w_gate_up.len() != gu_len {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "w_gate_up",
            expected: gu_len,
            got: w_gate_up.len(),
            detail: "w_gate_up length must equal E * 2Dff * Dm".to_string(),
        });
    }
    if w_down.len() != wd_len {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "w_down",
            expected: wd_len,
            got: w_down.len(),
            detail: "w_down length must equal E * Dm * Dff".to_string(),
        });
    }
    for (pos, &id) in expert_ids.iter().enumerate() {
        if (id as usize) >= e {
            problems.push(T0Error::RowIndexOutOfRange {
                op: OP,
                tensor: "expert_ids",
                position: pos,
                index: id,
                upper_bound: e,
            });
        }
    }
    T0Error::from_problems(problems)?;

    let mut triples: Vec<(u32, usize, usize)> = Vec::new();
    for row in 0..t {
        for slot in 0..k_dim {
            triples.push((expert_ids[row * k_dim + slot], row, slot));
        }
    }
    triples.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut y = vec![0.0f64; tdm];
    let mut i = 0;
    while i < triples.len() {
        let expert = triples[i].0 as usize;
        if expert >= e {
            return Err(T0Error::RowIndexOutOfRange {
                op: OP,
                tensor: "expert_ids",
                position: i,
                index: triples[i].0,
                upper_bound: e,
            });
        }
        let mut j = i + 1;
        while j < triples.len() && triples[j].0 == triples[i].0 {
            j += 1;
        }
        let mut rows: Vec<(usize, usize)> = triples[i..j]
            .iter()
            .map(|&(_, tok, slot)| (tok, slot))
            .collect();
        rows.sort_by_key(|&(tok, _)| tok);
        let gu_base = expert * 2 * dff * dm;
        let d_base = expert * dm * dff;
        for &(tok, slot) in &rows {
            let mut hidden = vec![0.0f64; dff];
            for d in 0..dff {
                let mut g = 0.0f64;
                let mut u = 0.0f64;
                for m in 0..dm {
                    g += x[tok * dm + m] * w_gate_up[gu_base + d * dm + m];
                    u += x[tok * dm + m] * w_gate_up[gu_base + (dff + d) * dm + m];
                }
                hidden[d] = crate::activation::eval_activation_f64(g, act) * u;
            }
            let w = weights[tok * k_dim + slot];
            for d in 0..dm {
                let mut acc = 0.0f64;
                for f in 0..dff {
                    acc += hidden[f] * w_down[d_base + d * dff + f];
                }
                y[tok * dm + d] += w * acc;
            }
        }
        i = j;
    }
    Ok(y)
}
