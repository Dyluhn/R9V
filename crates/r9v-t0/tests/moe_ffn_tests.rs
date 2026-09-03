// SPDX-License-Identifier: Apache-2.0
//! Tests for scalar T0 `moe_ffn` (Spec 1 §4.C, §6.2, Card A1.9).

use r9v_common::SeededRng;
use r9v_format::E4m3;
use r9v_format::{
    encode_e4m3_block128, encode_halfs_le, encode_i4k_superblock, encode_i8_block128,
    l1_forward_elems, l1_forward_index, scale_geometry, Layout, PaddedDims, SchemeId,
};
use r9v_ir::{ActivationKind, DType, Epilogue, LayoutId, MatmulOp, MoeFfnOp, Op, QuantScheme};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::{f16_to_f32, f32_to_f16};
use r9v_t0::error::T0Error;
use r9v_t0::matmul::matmul_with_scales;
use r9v_t0::{execute_moe_op, moe_ffn, moe_ffn_f64_reference, Tolerance};

fn next_f32(rng: &mut SeededRng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
    lo + (u as f32) * (hi - lo)
}

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

/// Builds an inline-I8R `[rows, cols]` weight buffer: values plus `f16` scale per row.
fn i8r_inline(values: &[i8], rows: usize, cols: usize, scale: f32) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    let sbits = f32_to_f16(scale);
    let mut out = vec![0u8; rows * (cols + 2)];
    for r in 0..rows {
        for c in 0..cols {
            out[r * (cols + 2) + c] = values[r * cols + c] as u8;
        }
        out[r * (cols + 2) + cols..r * (cols + 2) + cols + 2].copy_from_slice(&sbits.to_le_bytes());
    }
    out
}

fn ffn_op(act: ActivationKind) -> MoeFfnOp {
    MoeFfnOp {
        act,
        out_dtype: DType::F32,
        shared_experts: 0,
    }
}

#[test]
fn hand_computed_single_token_identity_pipeline() {
    // T=1, Dm=2, Dff=2, E=1, K=1, act=Identity, all F16.
    // x=[1,0]; gate rows [2,3] picked by x; up rows [4,5]; h=[8,15];
    // Wd=[[6,7],[8,9]]; y=[153,199] exactly.
    let op = ffn_op(ActivationKind::Identity);
    let x_buf = TypedBuffer::from_f16(&[1, 2], &[f32_to_f16(1.0), f32_to_f16(0.0)]);
    let ids_buf = TypedBuffer::from_u32(&[1, 1], &[0]);
    let wgt_buf = TypedBuffer::from_f32(&[1, 1], &[1.0]);
    let gu = f16_bytes(&[2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
    let gu_buf = TypedBuffer::from_bytes(&[1, 4, 2], DType::F16, &gu);
    let wd = f16_bytes(&[6.0, 7.0, 8.0, 9.0]);
    let wd_buf = TypedBuffer::from_bytes(&[1, 2, 2], DType::F16, &wd);
    let mut y_buf = TypedBuffer::zeros(&[1, 2], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.to_f32_vec(), vec![153.0, 199.0]);
}

#[test]
fn down_path_differential_matches_matmul_branch_d_bit_exact() {
    // Real differential guard for the Branch-D mirror: the same rigged
    // down-projection inputs run through `moe_ffn` (T=1, K=1, weight 1.0,
    // gate/up rigged so the hidden row is exactly [8,15]) and through
    // `matmul_with_scales` Branch D (F16 x=[8,15] against the same Wd
    // bytes) must agree bit-exactly, and both must equal [153,199].
    let op = ffn_op(ActivationKind::Identity);
    let x_buf = TypedBuffer::from_f16(&[1, 2], &[f32_to_f16(1.0), f32_to_f16(0.0)]);
    let ids_buf = TypedBuffer::from_u32(&[1, 1], &[0]);
    let wgt_buf = TypedBuffer::from_f32(&[1, 1], &[1.0]);
    let gu = f16_bytes(&[2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
    let gu_buf = TypedBuffer::from_bytes(&[1, 4, 2], DType::F16, &gu);
    let wd = f16_bytes(&[6.0, 7.0, 8.0, 9.0]);
    let wd_buf = TypedBuffer::from_bytes(&[1, 2, 2], DType::F16, &wd);
    let mut y_moe_buf = TypedBuffer::zeros(&[1, 2], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        None,
        &mut y_moe_buf.as_view_mut(),
    )
    .unwrap();
    let y_moe = y_moe_buf.to_f32_vec();

    let gemm = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    let h_bits = [f32_to_f16(8.0), f32_to_f16(15.0)];
    let h_buf = TypedBuffer::from_f16(&[1, 2], &h_bits);
    let w_buf = TypedBuffer::from_bytes(&[2, 2], DType::F16, &wd);
    let mut y_mm_buf = TypedBuffer::zeros(&[1, 2], DType::F32);
    matmul_with_scales(
        &gemm,
        &h_buf.as_view(),
        None,
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_mm_buf.as_view_mut(),
    )
    .unwrap();
    let y_mm = y_mm_buf.to_f32_vec();

    assert_eq!(y_moe, vec![153.0, 199.0]);
    assert_eq!(y_moe, y_mm);
}

#[test]
fn f16_experts_match_f64_oracle_within_f16_tolerance() {
    let mut rng = SeededRng::new(0xF19);
    let (t, dm, dff, e, k) = (4, 6, 8, 3, 2);
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x_f16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x_f64: Vec<f64> = x_f16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let gu_f64: Vec<f64> = gu.iter().map(|&v| v as f64).collect();
    let wd: Vec<f32> = (0..e * dm * dff)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let wd_f64: Vec<f64> = wd.iter().map(|&v| v as f64).collect();
    // Fixed routing covering experts 0..2 (expert 2 gets one token).
    let ids = vec![0u32, 1, 1, 2, 2, 0, 0, 2];
    let weights: Vec<f32> = (0..t * k).map(|_| next_f32(&mut rng, 0.2, 1.0)).collect();
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();

    let op = ffn_op(ActivationKind::Silu);
    let x_buf = TypedBuffer::from_f16(&[t, dm], &x_f16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));
    let wd_buf = TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &f16_bytes(&wd));
    let mut y_buf = TypedBuffer::zeros(&[t, dm], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // Oracle decodes the same f16 bits the T0 path consumed.
    let gu_dec: Vec<f64> = gu_f64
        .iter()
        .enumerate()
        .map(|(i, _)| f16_to_f32(f32_to_f16(gu[i])) as f64)
        .collect();
    let wd_dec: Vec<f64> = wd_f64
        .iter()
        .enumerate()
        .map(|(i, _)| f16_to_f32(f32_to_f16(wd[i])) as f64)
        .collect();
    let expected = moe_ffn_f64_reference(
        &x_f64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();
    let tol = Tolerance::f16_bf16();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("y[{i}]"));
    }
}

#[test]
fn i8_per_row_experts_match_f64_oracle_within_i8_tolerance() {
    let mut rng = SeededRng::new(0x1F9);
    let (t, dm, dff, e, k) = (3, 8, 8, 2, 2);
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x_f16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x_f64: Vec<f64> = x_f16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu_q: Vec<i8> = (0..e * 2 * dff * dm)
        .map(|_| (rng.next_u64() % 13) as i8 - 6)
        .collect();
    let wd_q: Vec<i8> = (0..e * dm * dff)
        .map(|_| (rng.next_u64() % 11) as i8 - 5)
        .collect();
    let gu_scale = 0.25f32;
    let wd_scale = 0.125f32;
    let gu_bytes = i8r_inline(&gu_q, e * 2 * dff, dm, gu_scale);
    let wd_bytes = i8r_inline(&wd_q, e * dm, dff, wd_scale);
    let ids = vec![0u32, 1, 1, 0, 0, 1];
    let weights = vec![0.6f32, 0.4, 0.3, 0.7, 0.5, 0.5];

    let op = ffn_op(ActivationKind::Relu2);
    let x_buf = TypedBuffer::from_f16(&[t, dm], &x_f16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::I8, &gu_bytes)
        .with_quant(QuantScheme::PerRow);
    let wd_buf = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &wd_bytes)
        .with_quant(QuantScheme::PerRow);
    let mut y_buf = TypedBuffer::zeros(&[t, dm], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let gu_dec: Vec<f64> = gu_q.iter().map(|&q| q as f64 * gu_scale as f64).collect();
    let wd_dec: Vec<f64> = wd_q.iter().map(|&q| q as f64 * wd_scale as f64).collect();
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x_f64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Relu2,
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("y[{i}]"));
    }
}

#[test]
fn token_permutation_preserves_per_token_outputs() {
    // Permuting input token order with per-token routing fixed must preserve
    // each token's output bits (the done-when combine-order test).
    let mut rng = SeededRng::new(0x2F9);
    let (t, dm, dff, e, k) = (6, 4, 6, 3, 2);
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let wd: Vec<f32> = (0..e * dm * dff)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let ids: Vec<u32> = (0..t * k).map(|i| (i as u32 * 7 + 1) % e as u32).collect();
    let weights: Vec<f32> = (0..t * k).map(|_| next_f32(&mut rng, 0.1, 1.0)).collect();

    // Activations run in F16 end to end (F32 is outside the GEMM family).
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let run16 = |order: &[usize]| {
        let op = ffn_op(ActivationKind::Gelu);
        let mut xr = vec![0u16; t * dm];
        let mut ir = vec![0u32; t * k];
        let mut wr = vec![0f32; t * k];
        for (new_t, &old_t) in order.iter().enumerate() {
            xr[new_t * dm..(new_t + 1) * dm].copy_from_slice(&x16[old_t * dm..(old_t + 1) * dm]);
            ir[new_t * k..(new_t + 1) * k].copy_from_slice(&ids[old_t * k..(old_t + 1) * k]);
            wr[new_t * k..(new_t + 1) * k].copy_from_slice(&weights[old_t * k..(old_t + 1) * k]);
        }
        let xb = TypedBuffer::from_f16(&[t, dm], &xr);
        let ib = TypedBuffer::from_u32(&[t, k], &ir);
        let wb = TypedBuffer::from_f32(&[t, k], &wr);
        let gb = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));
        let db = TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &f16_bytes(&wd));
        let mut yb = TypedBuffer::zeros(&[t, dm], DType::F32);
        moe_ffn(
            &op,
            &xb.as_view(),
            &ib.as_view(),
            &wb.as_view(),
            &gb.as_view(),
            None,
            &db.as_view(),
            None,
            None,
            &mut yb.as_view_mut(),
        )
        .unwrap();
        (yb.to_f32_vec(), order.to_vec())
    };
    let (y_base, _) = run16(&[0, 1, 2, 3, 4, 5]);
    let (y_perm, perm) = run16(&[5, 2, 0, 4, 1, 3]);
    for (new_t, &old_t) in perm.iter().enumerate() {
        assert_eq!(
            &y_perm[new_t * dm..(new_t + 1) * dm],
            &y_base[old_t * dm..(old_t + 1) * dm],
            "token {old_t} moved to row {new_t}"
        );
    }
}

#[test]
fn duplicate_expert_slots_combine_deterministically() {
    // Routing token 0 to expert 1 twice with 0.5/0.5 equals a single 1.0 slot.
    // Dims: T=2, Dm=2, Dff=2, E=2, K=2.
    let op = ffn_op(ActivationKind::Silu);
    let x_buf = TypedBuffer::from_f16(
        &[2, 2],
        &[
            f32_to_f16(0.5),
            f32_to_f16(-0.25),
            f32_to_f16(0.1),
            f32_to_f16(0.2),
        ],
    );
    let gu = f16_bytes(&[
        0.1, -0.2, 0.3, 0.4, 0.5, -0.5, 0.25, -0.25, 0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4,
    ]);
    let wd = f16_bytes(&[0.2, 0.1, -0.1, -0.2, 0.3, 0.4, 0.5, 0.6]);
    let gu_buf = TypedBuffer::from_bytes(&[2, 4, 2], DType::F16, &gu);
    let wd_buf = TypedBuffer::from_bytes(&[2, 2, 2], DType::F16, &wd);

    let run = |ids: &[u32], weights: &[f32]| {
        let ib = TypedBuffer::from_u32(&[2, 2], ids);
        let wb = TypedBuffer::from_f32(&[2, 2], weights);
        let mut yb = TypedBuffer::zeros(&[2, 2], DType::F32);
        moe_ffn(
            &op,
            &x_buf.as_view(),
            &ib.as_view(),
            &wb.as_view(),
            &gu_buf.as_view(),
            None,
            &wd_buf.as_view(),
            None,
            None,
            &mut yb.as_view_mut(),
        )
        .unwrap();
        yb.to_f32_vec()
    };
    // Token 0 -> expert 1 twice (0.5 + 0.5); token 1 -> expert 0 once + expert 1 zero-weighted.
    let y_dup = run(&[1, 1, 0, 1], &[0.5, 0.5, 1.0, 0.0]);
    let y_single = run(&[1, 0, 0, 1], &[1.0, 0.0, 1.0, 0.0]);
    assert_eq!(y_dup, y_single);
}

#[test]
fn shared_experts_ignored_for_compute_but_bounded() {
    let x_buf = TypedBuffer::from_f16(&[1, 2], &[f32_to_f16(1.0), f32_to_f16(0.0)]);
    let ids_buf = TypedBuffer::from_u32(&[1, 1], &[0]);
    let wgt_buf = TypedBuffer::from_f32(&[1, 1], &[1.0]);
    let gu_buf = TypedBuffer::from_bytes(
        &[1, 4, 2],
        DType::F16,
        &f16_bytes(&[2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]),
    );
    let wd_buf = TypedBuffer::from_bytes(&[1, 2, 2], DType::F16, &f16_bytes(&[6.0, 7.0, 8.0, 9.0]));
    let run = |shared: u32| {
        let op = MoeFfnOp {
            act: ActivationKind::Identity,
            out_dtype: DType::F32,
            shared_experts: shared,
        };
        let mut yb = TypedBuffer::zeros(&[1, 2], DType::F32);
        moe_ffn(
            &op,
            &x_buf.as_view(),
            &ids_buf.as_view(),
            &wgt_buf.as_view(),
            &gu_buf.as_view(),
            None,
            &wd_buf.as_view(),
            None,
            None,
            &mut yb.as_view_mut(),
        )
        .map(|()| yb.to_f32_vec())
    };
    assert_eq!(run(0).unwrap(), vec![153.0, 199.0]);
    assert_eq!(run(1).unwrap(), vec![153.0, 199.0]);
    let err = run(2).unwrap_err();
    assert!(matches!(
        err,
        T0Error::InvalidAttribute {
            op: "moe_ffn",
            attribute: "shared_experts",
            ..
        }
    ));
}

#[test]
fn expert_id_out_of_range_collected_before_mutation() {
    let op = ffn_op(ActivationKind::Silu);
    let x_buf = TypedBuffer::from_f16(&[2, 2], &[f32_to_f16(0.5); 4]);
    let ids_buf = TypedBuffer::from_u32(&[2, 2], &[0, 7, 9, 1]);
    let wgt_buf = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let gu_buf = TypedBuffer::from_bytes(&[2, 4, 2], DType::F16, &[0u8; 2 * 4 * 2 * 2]);
    let wd_buf = TypedBuffer::from_bytes(&[2, 2, 2], DType::F16, &[0u8; 2 * 2 * 2 * 2]);
    let mut y_buf = TypedBuffer::from_f32(&[2, 2], &[99.0; 4]);
    let err = moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    match err {
        T0Error::Multiple { problems } => assert_eq!(problems.len(), 2),
        other => panic!("expected Multiple, got {other:?}"),
    }
    // y untouched.
    assert_eq!(y_buf.to_f32_vec(), vec![99.0; 4]);
}

#[test]
fn unused_expert_with_no_tokens_is_skipped() {
    // E=3, expert 2 receives no tokens: still exact zeros for its absence.
    let op = ffn_op(ActivationKind::Identity);
    let x_buf = TypedBuffer::from_f16(
        &[2, 2],
        &[
            f32_to_f16(1.0),
            f32_to_f16(0.0),
            f32_to_f16(0.0),
            f32_to_f16(1.0),
        ],
    );
    let ids_buf = TypedBuffer::from_u32(&[2, 1], &[0, 1]);
    let wgt_buf = TypedBuffer::from_f32(&[2, 1], &[1.0, 1.0]);
    let mut gu_vals = vec![0.0f32; 3 * 4 * 2];
    for v in gu_vals.iter_mut() {
        *v = 0.25;
    }
    let mut wd_vals = vec![0.0f32; 3 * 2 * 2];
    for v in wd_vals.iter_mut() {
        *v = 0.5;
    }
    let gu_buf = TypedBuffer::from_bytes(&[3, 4, 2], DType::F16, &f16_bytes(&gu_vals));
    let wd_buf = TypedBuffer::from_bytes(&[3, 2, 2], DType::F16, &f16_bytes(&wd_vals));
    let mut y_buf = TypedBuffer::zeros(&[2, 2], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    // gate = 0.25+0 = 0.25 (x row [1,0]); up same; h = 0.0625 each of 2;
    // y = 0.0625*0.5 + 0.0625*0.5 = 0.0625 per channel.
    let tol = Tolerance::f32();
    for (i, &v) in y_buf.to_f32_vec().iter().enumerate() {
        tol.assert_within(v as f64, 0.0625, &format!("y[{i}]"));
    }
}

#[test]
fn determinism_and_batch_invariance() {
    let mut rng = SeededRng::new(0x3F9);
    let (t, dm, dff, e, k) = (5, 4, 6, 3, 2);
    let x: Vec<u16> = (0..t * dm)
        .map(|_| f32_to_f16(next_f32(&mut rng, -1.0, 1.0)))
        .collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let wd: Vec<f32> = (0..e * dm * dff)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let ids: Vec<u32> = (0..t * k).map(|i| (i as u32 * 5 + 2) % e as u32).collect();
    let weights: Vec<f32> = (0..t * k).map(|_| next_f32(&mut rng, 0.1, 1.0)).collect();
    let run = |xr: &[u16], ir: &[u32], wr: &[f32], tt: usize| {
        let op = ffn_op(ActivationKind::Silu);
        let xb = TypedBuffer::from_f16(&[tt, dm], xr);
        let ib = TypedBuffer::from_u32(&[tt, k], ir);
        let wb = TypedBuffer::from_f32(&[tt, k], wr);
        let gb = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));
        let db = TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &f16_bytes(&wd));
        let mut yb = TypedBuffer::zeros(&[tt, dm], DType::F32);
        moe_ffn(
            &op,
            &xb.as_view(),
            &ib.as_view(),
            &wb.as_view(),
            &gb.as_view(),
            None,
            &db.as_view(),
            None,
            None,
            &mut yb.as_view_mut(),
        )
        .unwrap();
        yb.to_f32_vec()
    };
    let y1 = run(&x, &ids, &weights, t);
    let y2 = run(&x, &ids, &weights, t);
    assert_eq!(y1, y2);
    // Row 2 alone.
    let xrow = x[2 * dm..3 * dm].to_vec();
    let irow = ids[2 * k..3 * k].to_vec();
    let wrow = weights[2 * k..3 * k].to_vec();
    let yrow = run(&xrow, &irow, &wrow, 1);
    assert_eq!(yrow, y1[2 * dm..3 * dm]);
}

/// Runs `moe_ffn` with F16 gate/up and returns `y`.
#[allow(clippy::too_many_arguments)]
fn run_ffn_down(
    x: &TypedBuffer,
    ids: &TypedBuffer,
    wgt: &TypedBuffer,
    gu: &TypedBuffer,
    wd: &TypedBuffer,
    wd_scale: Option<&TypedBuffer>,
    x_scale: Option<&TypedBuffer>,
    t: usize,
    dm: usize,
) -> Vec<f32> {
    let op = ffn_op(ActivationKind::Silu);
    let mut yb = TypedBuffer::zeros(&[t, dm], DType::F32);
    let wd_s = wd_scale.map(|s| s.as_view());
    let x_s = x_scale.map(|s| s.as_view());
    moe_ffn(
        &op,
        &x.as_view(),
        &ids.as_view(),
        &wgt.as_view(),
        &gu.as_view(),
        None,
        &wd.as_view(),
        wd_s.as_ref(),
        x_s.as_ref(),
        &mut yb.as_view_mut(),
    )
    .unwrap();
    yb.to_f32_vec()
}

#[test]
fn i8b128_down_inline_separate_and_attached_match_f64() {
    // Blocker 2: I8B128 down-projection via the canonical encoder, in all
    // three scale-carrier forms. E=1, T=2, Dm=4, Dff=128 (one block/row).
    let mut rng = SeededRng::new(0x1B28);
    let (t, dm, dff, e, k) = (2usize, 4usize, 128usize, 1usize, 1usize);
    let n_rows = e * dm;
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    // One canonical block per down row.
    let mut q_all = Vec::with_capacity(n_rows * dff);
    let mut s_all = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let row: Vec<f32> = (0..dff)
            .map(|_| next_f32(&mut rng, -0.5, 0.5) + (r as f32) * 0.01)
            .collect();
        let (q, sc) = encode_i8_block128(&row).unwrap();
        q_all.extend_from_slice(&q);
        s_all.extend_from_slice(&sc);
    }
    let wd_dec: Vec<f64> = q_all
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f64 * f16_to_f32(s_all[i / dff].bits()) as f64)
        .collect();
    let ids = vec![0u32; t * k];
    let weights = vec![0.7f32, 1.3];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));

    // Inline: values + trailing f16 block scale per row.
    let mut inline_bytes = vec![0u8; n_rows * (dff + 2)];
    for r in 0..n_rows {
        for c in 0..dff {
            inline_bytes[r * (dff + 2) + c] = q_all[r * dff + c] as u8;
        }
        inline_bytes[r * (dff + 2) + dff..r * (dff + 2) + dff + 2]
            .copy_from_slice(&s_all[r].to_bytes());
    }
    let wd_inline = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &inline_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));
    // Separate explicit carrier: values + [n_rows, 1] F16 view.
    let mut values_bytes = vec![0u8; n_rows * dff];
    for r in 0..n_rows {
        for c in 0..dff {
            values_bytes[r * dff + c] = q_all[r * dff + c] as u8;
        }
    }
    let mut scale_bytes = vec![0u8; n_rows * 2];
    for r in 0..n_rows {
        scale_bytes[r * 2..r * 2 + 2].copy_from_slice(&s_all[r].to_bytes());
    }
    let wd_values = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &values_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));
    let wd_scale = TypedBuffer::from_bytes(&[n_rows, 1], DType::F16, &scale_bytes);
    // Attached carrier: same values buffer with the scale view attached.
    let wd_attached = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &values_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));

    let tol = Tolerance::i8_weight();
    let y_inline = run_ffn_down(
        &x_buf, &ids_buf, &wgt_buf, &gu_buf, &wd_inline, None, None, t, dm,
    );
    let y_sep = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_values,
        Some(&wd_scale),
        None,
        t,
        dm,
    );
    // Attached resolution: pass None and attach via with_scale.
    let attached_view = wd_attached.as_view().with_scale(wd_scale.as_view());
    let op = ffn_op(ActivationKind::Silu);
    let mut yb = TypedBuffer::zeros(&[t, dm], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &attached_view,
        None,
        None,
        &mut yb.as_view_mut(),
    )
    .unwrap();
    let y_att = yb.to_f32_vec();
    for (name, y) in [
        ("inline", y_inline),
        ("separate", y_sep),
        ("attached", y_att),
    ] {
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("i8b128 {name} y[{i}]"));
        }
    }
}

#[test]
fn i4k_down_inline_and_separate_match_f64() {
    // Blocker 2: I4K down-projection via the canonical superblock encoder,
    // inline and separate carriers. E=1, T=2, Dm=2, Dff=256. Nibble parity
    // is even-index-low, matching the T0 decode.
    let mut rng = SeededRng::new(0x14B4);
    let (t, dm, dff, e, k) = (2usize, 2usize, 256usize, 1usize, 1usize);
    let n_rows = e * dm;
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let mut nibbles = vec![0u8; n_rows * dff];
    let mut headers = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut row = [0.0f32; 256];
        for v in row.iter_mut().take(dff) {
            *v = next_f32(&mut rng, -0.5, 0.5);
        }
        let (q, header) = encode_i4k_superblock(&row).unwrap();
        for c in 0..dff {
            nibbles[r * dff + c] = q[c];
        }
        headers.push(header);
    }
    // Pack even-low nibbles.
    let mut packed = vec![0u8; n_rows * dff / 2];
    for r in 0..n_rows {
        for c in 0..dff / 2 {
            packed[r * dff / 2 + c] =
                (nibbles[r * dff + 2 * c] & 0x0F) | ((nibbles[r * dff + 2 * c + 1] & 0x0F) << 4);
        }
    }
    // f64 decode mirroring the T0 zero-point form.
    let mut wd_dec = vec![0.0f64; n_rows * dff];
    for r in 0..n_rows {
        let h = &headers[r];
        let d = h.d_value(0).unwrap() as f64;
        let dmin = h.dmin_value(0).unwrap() as f64;
        let sc = h.scales();
        let mn = h.mins();
        for c in 0..dff {
            let sub = (c % 256) / 32;
            let s_block = d * sc[sub] as f64;
            let m_block = dmin * mn[sub] as f64;
            wd_dec[r * dff + c] = s_block * nibbles[r * dff + c] as f64 - m_block;
        }
    }
    let ids = vec![0u32; t * k];
    let weights = vec![0.8f32, 0.6];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));

    // Inline: packed values + 16-byte header per row.
    let mut inline_bytes = vec![0u8; n_rows * (dff / 2 + 16)];
    for r in 0..n_rows {
        inline_bytes[r * (dff / 2 + 16)..r * (dff / 2 + 16) + dff / 2]
            .copy_from_slice(&packed[r * dff / 2..(r + 1) * dff / 2]);
        inline_bytes[r * (dff / 2 + 16) + dff / 2..(r + 1) * (dff / 2 + 16)]
            .copy_from_slice(&headers[r].to_bytes());
    }
    let wd_inline = TypedBuffer::from_bytes(&[e, dm, dff], DType::I4, &inline_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()));
    // Separate: packed values + [n_rows, 1, 4] U32 header view.
    let wd_values = TypedBuffer::from_bytes(&[e, dm, dff], DType::I4, &packed)
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()));
    let mut header_bytes = vec![0u8; n_rows * 16];
    for r in 0..n_rows {
        header_bytes[r * 16..(r + 1) * 16].copy_from_slice(&headers[r].to_bytes());
    }
    let wd_scale = TypedBuffer::from_bytes(&[n_rows, 1, 4], DType::U32, &header_bytes);

    let tol = Tolerance::i8_weight();
    for (name, y) in [
        (
            "inline",
            run_ffn_down(
                &x_buf, &ids_buf, &wgt_buf, &gu_buf, &wd_inline, None, None, t, dm,
            ),
        ),
        (
            "separate",
            run_ffn_down(
                &x_buf,
                &ids_buf,
                &wgt_buf,
                &gu_buf,
                &wd_values,
                Some(&wd_scale),
                None,
                t,
                dm,
            ),
        ),
    ] {
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("i4k {name} y[{i}]"));
        }
    }
}

#[test]
fn e4m3b128_down_inline_and_separate_match_f64() {
    // Blocker 2: E4M3B128 down-projection via the canonical block encoder.
    // E=1, T=2, Dm=4, Dff=128.
    let mut rng = SeededRng::new(0xE4B3);
    let (t, dm, dff, e, k) = (2usize, 4usize, 128usize, 1usize, 1usize);
    let n_rows = e * dm;
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let mut q_all = Vec::with_capacity(n_rows * dff);
    let mut s_all = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let row: Vec<f32> = (0..dff).map(|_| next_f32(&mut rng, -2.0, 2.0)).collect();
        let (q, sc) = encode_e4m3_block128(&row).unwrap();
        q_all.extend(q.iter().map(|v| v.bits()));
        s_all.extend_from_slice(&sc);
    }
    let wd_dec: Vec<f64> = q_all
        .iter()
        .enumerate()
        .map(|(i, &qb)| E4m3::new(qb).to_f32() as f64 * f16_to_f32(s_all[i / dff].bits()) as f64)
        .collect();
    let ids = vec![0u32; t * k];
    let weights = vec![0.9f32, 1.1];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));

    let mut inline_bytes = vec![0u8; n_rows * (dff + 2)];
    for r in 0..n_rows {
        inline_bytes[r * (dff + 2)..r * (dff + 2) + dff]
            .copy_from_slice(&q_all[r * dff..(r + 1) * dff]);
        inline_bytes[r * (dff + 2) + dff..r * (dff + 2) + dff + 2]
            .copy_from_slice(&s_all[r].to_bytes());
    }
    let wd_inline = TypedBuffer::from_bytes(&[e, dm, dff], DType::E4m3, &inline_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::E4M3B128.to_ir()));
    let mut values_bytes = vec![0u8; n_rows * dff];
    values_bytes.copy_from_slice(&q_all);
    let wd_values = TypedBuffer::from_bytes(&[e, dm, dff], DType::E4m3, &values_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::E4M3B128.to_ir()));
    let mut scale_bytes = vec![0u8; n_rows * 2];
    for r in 0..n_rows {
        scale_bytes[r * 2..r * 2 + 2].copy_from_slice(&s_all[r].to_bytes());
    }
    let wd_scale = TypedBuffer::from_bytes(&[n_rows, 1], DType::F16, &scale_bytes);

    let tol = Tolerance::e4m3_cache();
    for (name, y) in [
        (
            "inline",
            run_ffn_down(
                &x_buf, &ids_buf, &wgt_buf, &gu_buf, &wd_inline, None, None, t, dm,
            ),
        ),
        (
            "separate",
            run_ffn_down(
                &x_buf,
                &ids_buf,
                &wgt_buf,
                &gu_buf,
                &wd_values,
                Some(&wd_scale),
                None,
                t,
                dm,
            ),
        ),
    ] {
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("e4m3 {name} y[{i}]"));
        }
    }
}

#[test]
fn l1_f16_experts_match_l0_bits_and_f64_oracle() {
    // Blocker 2: L1 flattened expert layout via the canonical tile
    // permutation, for both gate/up and down matrices. E=1, T=2, Dm=16,
    // Dff=16 (tile-aligned so padded == logical).
    let mut rng = SeededRng::new(0x5119);
    let (t, dm, dff, e, k) = (2usize, 16usize, 16usize, 1usize, 1usize);
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let wd: Vec<f32> = (0..e * dm * dff)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let gu16: Vec<u16> = gu.iter().map(|&v| f32_to_f16(v)).collect();
    let wd16: Vec<u16> = wd.iter().map(|&v| f32_to_f16(v)).collect();
    let gu_dec: Vec<f64> = gu16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let wd_dec: Vec<f64> = wd16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let ids = vec![0u32; t * k];
    let weights = vec![0.6f32, 1.4];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    // L0 reference buffers.
    let gu_l0 = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &encode_halfs_le(&gu16));
    let wd_l0 = TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &encode_halfs_le(&wd16));
    // L1 tiled buffers over the flattened row spaces.
    let gu_dims = PaddedDims::new((e * 2 * dff) as u32, dm as u32, Some(16)).unwrap();
    let wd_dims = PaddedDims::new((e * dm) as u32, dff as u32, Some(16)).unwrap();
    let gu_l1_bytes = encode_halfs_le(&l1_forward_elems(&gu16, &gu_dims).unwrap());
    let wd_l1_bytes = encode_halfs_le(&l1_forward_elems(&wd16, &wd_dims).unwrap());
    let gu_l1 = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &gu_l1_bytes)
        .with_layout(LayoutId::L1);
    let wd_l1 =
        TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &wd_l1_bytes).with_layout(LayoutId::L1);

    let y_l0 = run_ffn_down(
        &x_buf, &ids_buf, &wgt_buf, &gu_l0, &wd_l0, None, None, t, dm,
    );
    let y_l1 = run_ffn_down(
        &x_buf, &ids_buf, &wgt_buf, &gu_l1, &wd_l1, None, None, t, dm,
    );
    // The flattened L1 interpretation is bit-exact with L0 on tile-aligned shapes.
    assert_eq!(y_l1, y_l0);
    let tol = Tolerance::f16_bf16();
    for (i, (&actual, &exp)) in y_l1.iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("l1 y[{i}]"));
    }
}

#[test]
fn l1_i8r_down_inline_and_separate_match_l0_bits_and_f64() {
    // L1 + I8R down-projection via the canonical A2.1 tiling: values
    // scattered with `l1_forward_index`, row scales in the SoA tail via
    // `scale_geometry` record offsets. Non-tile-aligned down shape
    // (n_rows=6, Dff=20 over E=1, Dm=6) so padding is exercised in both
    // dims. L0-inline is the bit reference; the f64 oracle decodes the
    // same logical (q, scale) pairs independently.
    let mut rng = SeededRng::new(0x1181);
    let (t, dm, dff, e, k) = (2usize, 6usize, 20usize, 1usize, 1usize);
    let n_rows = e * dm;
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let q: Vec<i8> = (0..n_rows * dff)
        .map(|_| (rng.next_u64() % 13) as i8 - 6)
        .collect();
    // Distinct per-row scales so the scale path is load-bearing.
    let scales: Vec<f32> = (0..n_rows).map(|r| 0.1 + 0.05 * r as f32).collect();
    assert!(
        scales.windows(2).any(|w| w[0] != w[1]),
        "row scales must vary"
    );
    assert!(q.iter().min() != q.iter().max(), "quant values must vary");
    let wd_dec: Vec<f64> = q
        .iter()
        .enumerate()
        .map(|(i, &v)| v as f64 * scales[i / dff] as f64)
        .collect();
    let ids = vec![0u32; t * k];
    let weights = vec![0.7f32, 1.3];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));

    // L0-inline bit reference: values plus trailing f16 row scale per row.
    let mut l0_bytes = vec![0u8; n_rows * (dff + 2)];
    for r in 0..n_rows {
        for c in 0..dff {
            l0_bytes[r * (dff + 2) + c] = q[r * dff + c] as u8;
        }
        l0_bytes[r * (dff + 2) + dff..r * (dff + 2) + dff + 2]
            .copy_from_slice(&f32_to_f16(scales[r]).to_le_bytes());
    }
    let wd_l0 = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &l0_bytes)
        .with_quant(QuantScheme::PerRow);

    // L1 tiling over the flattened [E*Dm, Dff] row space (canonical A2.1).
    let sid = SchemeId::I8R;
    let w_dims = PaddedDims::new(n_rows as u32, dff as u32, Some(16)).unwrap();
    let geom = scale_geometry(sid, Layout::L1, &w_dims).unwrap();
    let vals_len = w_dims.n_padded() as usize * w_dims.k_padded() as usize;
    let mut l1_vals = vec![0u8; vals_len];
    for r in 0..n_rows {
        for c in 0..dff {
            let idx = l1_forward_index(r as u32, c as u32, &w_dims).unwrap() as usize;
            l1_vals[idx] = q[r * dff + c] as u8;
        }
    }
    let mut soa = vec![0u8; geom.region_bytes as usize];
    for (r, &scale) in scales.iter().enumerate() {
        let off = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        soa[off..off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
    }
    let mut l1_inline_bytes = l1_vals.clone();
    l1_inline_bytes.extend_from_slice(&soa);
    let wd_l1_inline = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &l1_inline_bytes)
        .with_quant(QuantScheme::PerRow)
        .with_layout(LayoutId::L1);
    let wd_l1_values = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &l1_vals)
        .with_quant(QuantScheme::PerRow)
        .with_layout(LayoutId::L1);
    let wd_l1_scale = TypedBuffer::from_bytes(
        &[geom.n_blocks as usize, geom.k_blocks as usize, 16],
        DType::F16,
        &soa,
    );

    let y_l0 = run_ffn_down(
        &x_buf, &ids_buf, &wgt_buf, &gu_buf, &wd_l0, None, None, t, dm,
    );
    let y_l1_inline = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_inline,
        None,
        None,
        t,
        dm,
    );
    let y_l1_sep = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_values,
        Some(&wd_l1_scale),
        None,
        t,
        dm,
    );
    // The tiled L1 interpretation is bit-exact with L0 on the same logical data.
    assert_eq!(y_l1_inline, y_l0);
    assert_eq!(y_l1_sep, y_l0);
    let tol = Tolerance::i8_weight();
    for (name, y) in [
        ("l0", y_l0),
        ("l1-inline", y_l1_inline),
        ("l1-separate", y_l1_sep),
    ] {
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("l1-i8r {name} y[{i}]"));
        }
    }
}

#[test]
fn l1_i8b128_down_inline_and_separate_match_l0_bits_and_f64() {
    // L1 + I8B128 down-projection via the canonical A2.1 tiling and SoA
    // scale records. E=2, Dm=9 gives n_rows=18, spanning two SoA row-blocks
    // (rows 16-17 land in block 1). L0-inline is the bit reference; the
    // f64 oracle decodes the canonical encoder output independently.
    let mut rng = SeededRng::new(0x1B81);
    let (t, dm, dff, e, k) = (3usize, 9usize, 128usize, 2usize, 1usize);
    let n_rows = e * dm;
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    // One canonical block per down row (Dff=128).
    let mut q_all = Vec::with_capacity(n_rows * dff);
    let mut s_all = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let row: Vec<f32> = (0..dff)
            .map(|_| next_f32(&mut rng, -0.5, 0.5) + (r as f32) * 0.01)
            .collect();
        let (qq, sc) = encode_i8_block128(&row).unwrap();
        q_all.extend_from_slice(&qq);
        s_all.extend_from_slice(&sc);
    }
    // Scales must vary across rows so the SoA path is load-bearing.
    let s0 = s_all[0].bits();
    assert!(
        s_all.iter().any(|s| s.bits() != s0),
        "block scales must vary across rows"
    );
    let wd_dec: Vec<f64> = q_all
        .iter()
        .enumerate()
        .map(|(i, &v)| v as f64 * f16_to_f32(s_all[i / dff].bits()) as f64)
        .collect();
    let ids = vec![0u32, 1, 0];
    let weights = vec![0.7f32, 1.1, 0.9];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));

    // L0-inline bit reference.
    let mut l0_bytes = vec![0u8; n_rows * (dff + 2)];
    for r in 0..n_rows {
        for c in 0..dff {
            l0_bytes[r * (dff + 2) + c] = q_all[r * dff + c] as u8;
        }
        l0_bytes[r * (dff + 2) + dff..r * (dff + 2) + dff + 2]
            .copy_from_slice(&s_all[r].to_bytes());
    }
    let wd_l0 = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &l0_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));

    // L1 tiling over the flattened [E*Dm, Dff] row space (canonical A2.1).
    let sid = SchemeId::I8B128;
    let w_dims = PaddedDims::new(n_rows as u32, dff as u32, Some(128)).unwrap();
    let geom = scale_geometry(sid, Layout::L1, &w_dims).unwrap();
    let vals_len = w_dims.n_padded() as usize * w_dims.k_padded() as usize;
    let mut l1_vals = vec![0u8; vals_len];
    for r in 0..n_rows {
        for c in 0..dff {
            let idx = l1_forward_index(r as u32, c as u32, &w_dims).unwrap() as usize;
            l1_vals[idx] = q_all[r * dff + c] as u8;
        }
    }
    let mut soa = vec![0u8; geom.region_bytes as usize];
    for (r, s) in s_all.iter().enumerate() {
        let off = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        soa[off..off + 2].copy_from_slice(&s.to_bytes());
    }
    let mut l1_inline_bytes = l1_vals.clone();
    l1_inline_bytes.extend_from_slice(&soa);
    let wd_l1_inline = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &l1_inline_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()))
        .with_layout(LayoutId::L1);
    let wd_l1_values = TypedBuffer::from_bytes(&[e, dm, dff], DType::I8, &l1_vals)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()))
        .with_layout(LayoutId::L1);
    let wd_l1_scale = TypedBuffer::from_bytes(
        &[geom.n_blocks as usize, geom.k_blocks as usize, 16],
        DType::F16,
        &soa,
    );

    let y_l0 = run_ffn_down(
        &x_buf, &ids_buf, &wgt_buf, &gu_buf, &wd_l0, None, None, t, dm,
    );
    let y_l1_inline = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_inline,
        None,
        None,
        t,
        dm,
    );
    let y_l1_sep = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_values,
        Some(&wd_l1_scale),
        None,
        t,
        dm,
    );
    assert_eq!(y_l1_inline, y_l0);
    assert_eq!(y_l1_sep, y_l0);
    let tol = Tolerance::i8_weight();
    for (name, y) in [
        ("l0", y_l0),
        ("l1-inline", y_l1_inline),
        ("l1-separate", y_l1_sep),
    ] {
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("l1-i8b128 {name} y[{i}]"));
        }
    }
}

#[test]
fn l1_e4m3b128_down_inline_and_separate_match_l0_bits_and_f64() {
    // L1 + E4M3B128 down-projection via the canonical A2.1 tiling and SoA
    // scale records. L0-inline is the bit reference; the f64 oracle
    // decodes the canonical encoder output independently.
    let mut rng = SeededRng::new(0xE481);
    let (t, dm, dff, e, k) = (2usize, 4usize, 128usize, 1usize, 1usize);
    let n_rows = e * dm;
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let mut q_all = Vec::with_capacity(n_rows * dff);
    let mut s_all = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let row: Vec<f32> = (0..dff).map(|_| next_f32(&mut rng, -2.0, 2.0)).collect();
        let (qq, sc) = encode_e4m3_block128(&row).unwrap();
        q_all.extend(qq.iter().map(|v| v.bits()));
        s_all.extend_from_slice(&sc);
    }
    let s0 = s_all[0].bits();
    assert!(
        s_all.iter().any(|s| s.bits() != s0),
        "block scales must vary across rows"
    );
    let wd_dec: Vec<f64> = q_all
        .iter()
        .enumerate()
        .map(|(i, &qb)| E4m3::new(qb).to_f32() as f64 * f16_to_f32(s_all[i / dff].bits()) as f64)
        .collect();
    let ids = vec![0u32; t * k];
    let weights = vec![0.9f32, 1.1];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));

    // L0-inline bit reference.
    let mut l0_bytes = vec![0u8; n_rows * (dff + 2)];
    for r in 0..n_rows {
        l0_bytes[r * (dff + 2)..r * (dff + 2) + dff]
            .copy_from_slice(&q_all[r * dff..(r + 1) * dff]);
        l0_bytes[r * (dff + 2) + dff..r * (dff + 2) + dff + 2]
            .copy_from_slice(&s_all[r].to_bytes());
    }
    let wd_l0 = TypedBuffer::from_bytes(&[e, dm, dff], DType::E4m3, &l0_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::E4M3B128.to_ir()));

    // L1 tiling over the flattened [E*Dm, Dff] row space (canonical A2.1).
    let sid = SchemeId::E4M3B128;
    let w_dims = PaddedDims::new(n_rows as u32, dff as u32, Some(128)).unwrap();
    let geom = scale_geometry(sid, Layout::L1, &w_dims).unwrap();
    let vals_len = w_dims.n_padded() as usize * w_dims.k_padded() as usize;
    let mut l1_vals = vec![0u8; vals_len];
    for r in 0..n_rows {
        for c in 0..dff {
            let idx = l1_forward_index(r as u32, c as u32, &w_dims).unwrap() as usize;
            l1_vals[idx] = q_all[r * dff + c];
        }
    }
    let mut soa = vec![0u8; geom.region_bytes as usize];
    for (r, s) in s_all.iter().enumerate() {
        let off = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        soa[off..off + 2].copy_from_slice(&s.to_bytes());
    }
    let mut l1_inline_bytes = l1_vals.clone();
    l1_inline_bytes.extend_from_slice(&soa);
    let wd_l1_inline = TypedBuffer::from_bytes(&[e, dm, dff], DType::E4m3, &l1_inline_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::E4M3B128.to_ir()))
        .with_layout(LayoutId::L1);
    let wd_l1_values = TypedBuffer::from_bytes(&[e, dm, dff], DType::E4m3, &l1_vals)
        .with_quant(QuantScheme::Scheme(SchemeId::E4M3B128.to_ir()))
        .with_layout(LayoutId::L1);
    let wd_l1_scale = TypedBuffer::from_bytes(
        &[geom.n_blocks as usize, geom.k_blocks as usize, 16],
        DType::F16,
        &soa,
    );

    let y_l0 = run_ffn_down(
        &x_buf, &ids_buf, &wgt_buf, &gu_buf, &wd_l0, None, None, t, dm,
    );
    let y_l1_inline = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_inline,
        None,
        None,
        t,
        dm,
    );
    let y_l1_sep = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_values,
        Some(&wd_l1_scale),
        None,
        t,
        dm,
    );
    assert_eq!(y_l1_inline, y_l0);
    assert_eq!(y_l1_sep, y_l0);
    let tol = Tolerance::e4m3_cache();
    for (name, y) in [
        ("l0", y_l0),
        ("l1-inline", y_l1_inline),
        ("l1-separate", y_l1_sep),
    ] {
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("l1-e4m3 {name} y[{i}]"));
        }
    }
}

#[test]
fn l1_i4k_down_inline_and_separate_match_l0_bits_and_f64() {
    // L1 + I4K down-projection via the canonical A2.1 tiling (nibble-packed,
    // even-index-low) and SoA superblock records. L0-inline is the bit
    // reference; the f64 oracle uses the independent `s·q−m` formula. The
    // mixed-nibble assertion keeps parity load-bearing.
    let mut rng = SeededRng::new(0x14B1);
    let (t, dm, dff, e, k) = (2usize, 2usize, 256usize, 1usize, 1usize);
    let n_rows = e * dm;
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let mut nibbles = vec![0u8; n_rows * dff];
    let mut headers = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut row = [0.0f32; 256];
        for v in row.iter_mut().take(dff) {
            *v = next_f32(&mut rng, -0.5, 0.5);
        }
        let (qq, header) = encode_i4k_superblock(&row).unwrap();
        for c in 0..dff {
            nibbles[r * dff + c] = qq[c];
        }
        headers.push(header);
    }
    // Nibble parity is load-bearing: the fixture must mix low and high
    // nibble positions (even-low packing distinguishes them).
    let mixed = nibbles
        .iter()
        .enumerate()
        .filter(|(i, &v)| v != nibbles[(i + 1) % nibbles.len()])
        .count();
    assert!(mixed > 0, "i4k fixture must mix nibble values");
    assert!(
        nibbles.iter().min() != nibbles.iter().max(),
        "i4k nibbles must vary"
    );
    // f64 decode mirroring the T0 zero-point form.
    let mut wd_dec = vec![0.0f64; n_rows * dff];
    for r in 0..n_rows {
        let h = &headers[r];
        let d = h.d_value(0).unwrap() as f64;
        let dmin = h.dmin_value(0).unwrap() as f64;
        let sc = h.scales();
        let mn = h.mins();
        for c in 0..dff {
            let sub = (c % 256) / 32;
            let s_block = d * sc[sub] as f64;
            let m_block = dmin * mn[sub] as f64;
            wd_dec[r * dff + c] = s_block * nibbles[r * dff + c] as f64 - m_block;
        }
    }
    let ids = vec![0u32; t * k];
    let weights = vec![0.8f32, 0.6];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));

    // L0-inline bit reference: packed values (even-low) plus 16-byte header.
    let mut packed = vec![0u8; n_rows * dff / 2];
    for r in 0..n_rows {
        for c in 0..dff / 2 {
            packed[r * dff / 2 + c] =
                (nibbles[r * dff + 2 * c] & 0x0F) | ((nibbles[r * dff + 2 * c + 1] & 0x0F) << 4);
        }
    }
    let mut l0_bytes = vec![0u8; n_rows * (dff / 2 + 16)];
    for r in 0..n_rows {
        l0_bytes[r * (dff / 2 + 16)..r * (dff / 2 + 16) + dff / 2]
            .copy_from_slice(&packed[r * dff / 2..(r + 1) * dff / 2]);
        l0_bytes[r * (dff / 2 + 16) + dff / 2..(r + 1) * (dff / 2 + 16)]
            .copy_from_slice(&headers[r].to_bytes());
    }
    let wd_l0 = TypedBuffer::from_bytes(&[e, dm, dff], DType::I4, &l0_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()));

    // L1 tiling over the flattened [E*Dm, Dff] row space (canonical A2.1):
    // element index via `l1_forward_index`, even-low nibble packing.
    let sid = SchemeId::I4K;
    let w_dims = PaddedDims::new(n_rows as u32, dff as u32, Some(256)).unwrap();
    let geom = scale_geometry(sid, Layout::L1, &w_dims).unwrap();
    let vals_len = w_dims.n_padded() as usize * w_dims.k_padded() as usize / 2;
    let mut l1_vals = vec![0u8; vals_len];
    for r in 0..n_rows {
        for c in 0..dff {
            let elem = l1_forward_index(r as u32, c as u32, &w_dims).unwrap() as usize;
            let nib = nibbles[r * dff + c] & 0x0F;
            if elem.is_multiple_of(2) {
                l1_vals[elem / 2] |= nib;
            } else {
                l1_vals[elem / 2] |= nib << 4;
            }
        }
    }
    let mut soa = vec![0u8; geom.region_bytes as usize];
    for (r, h) in headers.iter().enumerate() {
        let off = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        soa[off..off + 16].copy_from_slice(&h.to_bytes());
    }
    let mut l1_inline_bytes = l1_vals.clone();
    l1_inline_bytes.extend_from_slice(&soa);
    let wd_l1_inline = TypedBuffer::from_bytes(&[e, dm, dff], DType::I4, &l1_inline_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()))
        .with_layout(LayoutId::L1);
    let wd_l1_values = TypedBuffer::from_bytes(&[e, dm, dff], DType::I4, &l1_vals)
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()))
        .with_layout(LayoutId::L1);
    let wd_l1_scale = TypedBuffer::from_bytes(
        &[geom.n_blocks as usize, geom.k_blocks as usize, 16, 4],
        DType::U32,
        &soa,
    );

    let y_l0 = run_ffn_down(
        &x_buf, &ids_buf, &wgt_buf, &gu_buf, &wd_l0, None, None, t, dm,
    );
    let y_l1_inline = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_inline,
        None,
        None,
        t,
        dm,
    );
    let y_l1_sep = run_ffn_down(
        &x_buf,
        &ids_buf,
        &wgt_buf,
        &gu_buf,
        &wd_l1_values,
        Some(&wd_l1_scale),
        None,
        t,
        dm,
    );
    assert_eq!(y_l1_inline, y_l0);
    assert_eq!(y_l1_sep, y_l0);
    let tol = Tolerance::i8_weight();
    for (name, y) in [
        ("l0", y_l0),
        ("l1-inline", y_l1_inline),
        ("l1-separate", y_l1_sep),
    ] {
        for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("l1-i4k {name} y[{i}]"));
        }
    }
}

#[test]
fn i8r_gate_up_inline_matches_f64_oracle() {
    // Quantized gate/up matrices forward through `matmul_with_scales`: I8R
    // inline gate/up with an F16 down projection vs the f64 oracle.
    let mut rng = SeededRng::new(0x619);
    let (t, dm, dff, e, k) = (2usize, 8usize, 8usize, 1usize, 1usize);
    let x: Vec<f32> = (0..t * dm).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let x16: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let x64: Vec<f64> = x16.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let gu_q: Vec<i8> = (0..e * 2 * dff * dm)
        .map(|_| (rng.next_u64() % 13) as i8 - 6)
        .collect();
    let gu_scale = 0.2f32;
    let gu_bytes = i8r_inline(&gu_q, e * 2 * dff, dm, gu_scale);
    let wd: Vec<f32> = (0..e * dm * dff)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let wd_dec: Vec<f64> = wd
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let gu_dec: Vec<f64> = gu_q.iter().map(|&q| q as f64 * gu_scale as f64).collect();
    let ids = vec![0u32; t * k];
    let weights = vec![1.0f32, 0.5];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let op = ffn_op(ActivationKind::Silu);
    let x_buf = TypedBuffer::from_f16(&[t, dm], &x16);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::I8, &gu_bytes)
        .with_quant(QuantScheme::PerRow);
    let wd_buf = TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &f16_bytes(&wd));
    let mut y_buf = TypedBuffer::zeros(&[t, dm], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("gu-i8r y[{i}]"));
    }
}

#[test]
fn pertoken_activation_matches_f64_oracle() {
    // PerToken I8 activations with an explicit [T] scale carrier.
    let mut rng = SeededRng::new(0x9E70);
    let (t, dm, dff, e, k) = (3usize, 8usize, 8usize, 2usize, 2usize);
    let x_q: Vec<i8> = (0..t * dm)
        .map(|_| (rng.next_u64() % 17) as i8 - 8)
        .collect();
    let x_s: Vec<f32> = (0..t).map(|_| next_f32(&mut rng, 0.1, 0.5)).collect();
    let x64: Vec<f64> = x_q
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f64 * x_s[i / dm] as f64)
        .collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let wd: Vec<f32> = (0..e * dm * dff)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let wd_dec: Vec<f64> = wd
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let ids = vec![0u32, 1, 1, 0, 0, 1];
    let weights = vec![0.6f32, 0.4, 0.3, 0.7, 0.5, 0.5];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Relu2,
    )
    .unwrap();

    let op = MoeFfnOp {
        act: ActivationKind::Relu2,
        out_dtype: DType::F32,
        shared_experts: 0,
    };
    let x_buf = TypedBuffer::from_i8(&[t, dm], &x_q).with_quant(QuantScheme::PerToken);
    let xs_buf = TypedBuffer::from_f32(&[t], &x_s);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));
    let wd_buf = TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &f16_bytes(&wd));
    let mut y_buf = TypedBuffer::zeros(&[t, dm], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        Some(&xs_buf.as_view()),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("pertoken y[{i}]"));
    }
}

#[test]
fn perblock32_activation_matches_f64_oracle() {
    // PerBlock32 I8 activations with an explicit [T, Dm/32] scale carrier.
    let mut rng = SeededRng::new(0xB322);
    let (t, dm, dff, e, k) = (2usize, 32usize, 8usize, 1usize, 1usize);
    let x_q: Vec<i8> = (0..t * dm)
        .map(|_| (rng.next_u64() % 11) as i8 - 5)
        .collect();
    let x_s: Vec<f32> = (0..t * dm / 32)
        .map(|_| next_f32(&mut rng, 0.1, 0.5))
        .collect();
    let x64: Vec<f64> = x_q
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f64 * x_s[i / dm] as f64)
        .collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let wd: Vec<f32> = (0..e * dm * dff)
        .map(|_| next_f32(&mut rng, -0.25, 0.25))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let wd_dec: Vec<f64> = wd
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let ids = vec![0u32; t * k];
    let weights = vec![1.0f32, 0.5];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Silu,
    )
    .unwrap();

    let op = ffn_op(ActivationKind::Silu);
    let x_buf = TypedBuffer::from_i8(&[t, dm], &x_q).with_quant(QuantScheme::PerBlock32);
    let xs_buf = TypedBuffer::from_f32(&[t, dm / 32], &x_s);
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));
    let wd_buf = TypedBuffer::from_bytes(&[e, dm, dff], DType::F16, &f16_bytes(&wd));
    let mut y_buf = TypedBuffer::zeros(&[t, dm], DType::F32);
    moe_ffn(
        &op,
        &x_buf.as_view(),
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_buf.as_view(),
        None,
        Some(&xs_buf.as_view()),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("perblock32 y[{i}]"));
    }
}

#[test]
fn attached_i8r_down_and_pertoken_x_resolve_without_explicit_scales() {
    // Attached carriers: I8R down values with the row-scale view attached to
    // the weight, and PerToken activation scales attached to `x`; both scale
    // parameters passed as `None` must resolve through the attached views.
    let mut rng = SeededRng::new(0xA77);
    let (t, dm, dff, e, k) = (2usize, 8usize, 8usize, 1usize, 1usize);
    let n_rows = e * dm;
    let x_q: Vec<i8> = (0..t * dm)
        .map(|_| (rng.next_u64() % 13) as i8 - 6)
        .collect();
    let x_s: Vec<f32> = vec![0.3, 0.4];
    let x64: Vec<f64> = x_q
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f64 * x_s[i / dm] as f64)
        .collect();
    let gu: Vec<f32> = (0..e * 2 * dff * dm)
        .map(|_| next_f32(&mut rng, -0.5, 0.5))
        .collect();
    let gu_dec: Vec<f64> = gu
        .iter()
        .map(|&v| f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let wd_q: Vec<i8> = (0..n_rows * dff)
        .map(|_| (rng.next_u64() % 11) as i8 - 5)
        .collect();
    let wd_scale = 0.125f32;
    let wd_dec: Vec<f64> = wd_q.iter().map(|&q| q as f64 * wd_scale as f64).collect();
    let ids = vec![0u32; t * k];
    let weights = vec![1.0f32, 0.5];
    let weights_f64: Vec<f64> = weights.iter().map(|&v| v as f64).collect();
    let expected = moe_ffn_f64_reference(
        &x64,
        t,
        dm,
        &ids,
        &weights_f64,
        k,
        &gu_dec,
        e,
        dff,
        &wd_dec,
        ActivationKind::Identity,
    )
    .unwrap();

    let op = ffn_op(ActivationKind::Identity);
    let xs_buf = TypedBuffer::from_f32(&[t], &x_s);
    let x_buf = TypedBuffer::from_i8(&[t, dm], &x_q).with_quant(QuantScheme::PerToken);
    let x_view = x_buf.as_view().with_scale(xs_buf.as_view());
    let ids_buf = TypedBuffer::from_u32(&[t, k], &ids);
    let wgt_buf = TypedBuffer::from_f32(&[t, k], &weights);
    let gu_buf = TypedBuffer::from_bytes(&[e, 2 * dff, dm], DType::F16, &f16_bytes(&gu));
    // Values-only I8R down rows (no inline scales) with the row scales attached.
    let wd_values = TypedBuffer::from_bytes(
        &[e, dm, dff],
        DType::I8,
        &wd_q.iter().map(|&q| q as u8).collect::<Vec<u8>>(),
    )
    .with_quant(QuantScheme::PerRow);
    let mut scale_bytes = vec![0u8; n_rows * 2];
    let sbits = f32_to_f16(wd_scale);
    for r in 0..n_rows {
        scale_bytes[r * 2..r * 2 + 2].copy_from_slice(&sbits.to_le_bytes());
    }
    let wd_scale_buf = TypedBuffer::from_bytes(&[n_rows], DType::F16, &scale_bytes);
    let wd_view = wd_values.as_view().with_scale(wd_scale_buf.as_view());
    let mut y_buf = TypedBuffer::zeros(&[t, dm], DType::F32);
    moe_ffn(
        &op,
        &x_view,
        &ids_buf.as_view(),
        &wgt_buf.as_view(),
        &gu_buf.as_view(),
        None,
        &wd_view,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("attached y[{i}]"));
    }
}

#[test]
fn dispatch_enforces_ffn_arity() {
    let op = Op::MoeFfn(ffn_op(ActivationKind::Silu));
    let z = TypedBuffer::zeros(&[1, 2], DType::F32);
    let mut y = TypedBuffer::zeros(&[1, 2], DType::F32);
    let four = [z.as_view(), z.as_view(), z.as_view(), z.as_view()];
    assert!(execute_moe_op(&op, &four, &mut [y.as_view_mut()]).is_err());
    let five = [
        z.as_view(),
        z.as_view(),
        z.as_view(),
        z.as_view(),
        z.as_view(),
    ];
    assert!(execute_moe_op(&op, &five, &mut []).is_err());
}
