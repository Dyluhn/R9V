// SPDX-License-Identifier: Apache-2.0
//! Comprehensive integration and unit tests for scalar deterministic T0 matmul (Card A1.6).

use r9v_format::records::{E4M3Block128Scale, I4KSuperblock};
use r9v_format::scales::E4m3;
use r9v_format::{scale_geometry, Layout, PaddedDims, SchemeId};
use r9v_ir::{ActivationKind, DType, Epilogue, LayoutId, MatmulOp, QuantScheme};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::f32_to_f16;
use r9v_t0::error::T0Error;
use r9v_t0::matmul::{matmul, matmul_f64_reference, matmul_with_scales};

#[test]
fn test_matmul_f16_x_f16_matches_f64_reference() {
    let m = 3;
    let k = 16;
    let n = 4;

    let mut x_f64 = vec![0.0f64; m * k];
    let mut x_f16 = vec![0u16; m * k];
    for i in 0..(m * k) {
        let val = (i as f64 * 0.1) - 1.0;
        x_f64[i] = val;
        x_f16[i] = f32_to_f16(val as f32);
    }

    let mut w_f64 = vec![0.0f64; n * k];
    let mut w_bytes = vec![0u8; n * k * 2];
    for r in 0..n {
        for c in 0..k {
            let val = (r as f64 * 0.25) + (c as f64 * 0.05);
            w_f64[r * k + c] = val;
            let bits = f32_to_f16(val as f32);
            let offset = (r * k + c) * 2;
            w_bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }

    let bias_f64 = vec![0.5f64, -0.5, 1.0, -1.0];
    let bias_f32: Vec<f32> = bias_f64.iter().map(|&v| v as f32).collect();

    let expected_f64 = matmul_f64_reference(
        &x_f64,
        m,
        k,
        &w_f64,
        n,
        Some(&bias_f64),
        None,
        Epilogue::Bias,
        false,
    );

    let x_buf = TypedBuffer::from_f16(&[m, k], &x_f16);
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::F16, &w_bytes);
    let bias_buf = TypedBuffer::from_f32(&[n], &bias_f32);
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::Bias,
        transpose_w: false,
    };

    matmul(
        &op,
        &x_buf.as_view(),
        &w_buf.as_view(),
        Some(&bias_buf.as_view()),
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let y_slice = y_buf.to_f32_vec();
    for (actual, &expected) in y_slice.iter().zip(expected_f64.iter()) {
        assert!((actual - expected as f32).abs() < 1e-2);
    }
}

#[test]
fn test_matmul_i8_per_token_x_i8_r_exact_i32_accumulation() {
    let m = 2;
    let k = 64;
    let n = 2;

    let x_vals = vec![2i8; m * k];
    let w_vals = vec![3i8; n * k];
    let x_scale_vals = vec![0.5f32, 0.25f32];
    let w_scale_val = 0.125f32;
    let w_scale_bits = f32_to_f16(w_scale_val);

    // Build L0 weight buffer: [N, K + 2]
    let row_stride = k + 2;
    let mut w_bytes = vec![0u8; n * row_stride];
    for r in 0..n {
        for c in 0..k {
            w_bytes[r * row_stride + c] = w_vals[r * k + c] as u8;
        }
        w_bytes[r * row_stride + k..r * row_stride + k + 2]
            .copy_from_slice(&w_scale_bits.to_le_bytes());
    }

    let x_buf = TypedBuffer::from_i8(&[m, k], &x_vals).with_quant(QuantScheme::PerToken);
    let x_scale_buf = TypedBuffer::from_f32(&[m], &x_scale_vals);
    let w_buf =
        TypedBuffer::from_bytes(&[n, k], DType::I8, &w_bytes).with_quant(QuantScheme::PerRow);
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // Exact math:
    // dot_i32 = sum_{k=0..63} (2 * 3) = 64 * 6 = 384.
    // row 0: 384 * (0.125 * 0.5) = 384 * 0.0625 = 24.0.
    // row 1: 384 * (0.125 * 0.25) = 384 * 0.03125 = 12.0.
    let y_slice = y_buf.to_f32_vec();
    assert_eq!(y_slice[0], 24.0);
    assert_eq!(y_slice[1], 24.0);
    assert_eq!(y_slice[2], 12.0);
    assert_eq!(y_slice[3], 12.0);

    // Also test with QuantScheme::Scheme(SchemeId::I8R.to_ir()) to verify
    // I8_R uses full-K AscendingK rather than treating Scheme ID as block-scaled
    let w_buf_scheme = TypedBuffer::from_bytes(&[n, k], DType::I8, &w_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir()));
    let mut y_buf_scheme = TypedBuffer::zeros(&[m, n], DType::F32);
    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf_scheme.as_view(),
        None,
        None,
        None,
        &mut y_buf_scheme.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf_scheme.to_f32_vec(), y_slice);
}

#[test]
fn test_matmul_i8_per_token_x_i8_b128_matches_math() {
    let m = 1;
    let k = 128;
    let n = 1;

    let x_vals = vec![1i8; k];
    let w_vals = vec![2i8; k];
    let x_scale_vals = vec![1.0f32];
    let w_scale_val = 0.5f32;
    let w_scale_bits = f32_to_f16(w_scale_val);

    let row_stride = k + 2;
    let mut w_bytes = vec![0u8; n * row_stride];
    for c in 0..k {
        w_bytes[c] = w_vals[c] as u8;
    }
    w_bytes[k..k + 2].copy_from_slice(&w_scale_bits.to_le_bytes());

    let x_buf = TypedBuffer::from_i8(&[m, k], &x_vals).with_quant(QuantScheme::PerToken);
    let x_scale_buf = TypedBuffer::from_f32(&[m], &x_scale_vals);
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::I8, &w_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(2)));
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // dot_i32 = 128 * (1 * 2) = 256.
    // scale = 0.5 * 1.0 = 0.5.
    // result = 256 * 0.5 = 128.0.
    let y_slice = y_buf.to_f32_vec();
    assert_eq!(y_slice[0], 128.0);
}

#[test]
fn test_matmul_i8_per_token_x_i4_k_matches_zero_point_formula() {
    let m = 1;
    let k = 256;
    let n = 1;

    let x_vals = vec![1i8; k];
    let x_scale_vals = vec![1.0f32];

    let d_val = 1.0f32;
    let dmin_val = 0.5f32;
    let d_bits = f32_to_f16(d_val);
    let dmin_bits = f32_to_f16(dmin_val);
    let sc = [2u8; 8];
    let mn = [1u8; 8];

    let header = I4KSuperblock::pack(d_bits, dmin_bits, sc, mn).unwrap();
    let header_bytes = header.to_bytes();

    // Each nibble is 3 (so byte is (3 | (3 << 4)) = 0x33)
    let values_bytes = k / 2; // 128
    let mut w_bytes = vec![0x33u8; values_bytes + 16];
    w_bytes[values_bytes..values_bytes + 16].copy_from_slice(&header_bytes);

    let x_buf = TypedBuffer::from_i8(&[m, k], &x_vals).with_quant(QuantScheme::PerToken);
    let x_scale_buf = TypedBuffer::from_f32(&[m], &x_scale_vals);
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::I4, &w_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(3)));
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // In each of the eight 32-blocks:
    // s_block = 1.0 * 2 = 2.0.
    // m_block = 0.5 * 1 = 0.5.
    // dot_xq = sum_{j=0..31} (1 * 3) = 96.
    // sum_x = sum_{j=0..31} 1 = 32.
    // block_val = (2.0 * 96 - 0.5 * 32) * 1.0 = 192 - 16 = 176.
    // Total for 8 blocks = 8 * 176 = 1408.0.
    let y_slice = y_buf.to_f32_vec();
    assert_eq!(y_slice[0], 1408.0);
}

#[test]
fn test_matmul_i8_per_block32_x_i8_r() {
    let m = 1;
    let k = 64;
    let n = 1;

    let x_vals = vec![2i8; k];
    // Two 32-blocks for x: scales [0.5, 0.25]
    let x_scale_vals = vec![0.5f32, 0.25f32];
    let w_vals = vec![3i8; k];
    let w_scale_val = 0.125f32;
    let w_scale_bits = f32_to_f16(w_scale_val);

    let mut w_bytes = vec![0u8; k + 2];
    for c in 0..k {
        w_bytes[c] = w_vals[c] as u8;
    }
    w_bytes[k..k + 2].copy_from_slice(&w_scale_bits.to_le_bytes());

    let x_buf = TypedBuffer::from_i8(&[m, k], &x_vals).with_quant(QuantScheme::PerBlock32);
    let x_scale_buf = TypedBuffer::from_f32(&[m, k / 32], &x_scale_vals);
    let w_buf =
        TypedBuffer::from_bytes(&[n, k], DType::I8, &w_bytes).with_quant(QuantScheme::PerRow);
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // Block 0: 32 * (2 * 3) = 192. scaled by (0.125 * 0.5) = 192 * 0.0625 = 12.0.
    // Block 1: 32 * (2 * 3) = 192. scaled by (0.125 * 0.25) = 192 * 0.03125 = 6.0.
    // Total = 18.0.
    let y_slice = y_buf.to_f32_vec();
    assert_eq!(y_slice[0], 18.0);
}

#[test]
fn test_matmul_e4m3_per_token_x_e4m3_b128() {
    let m = 1;
    let k = 128;
    let n = 1;

    // Value 1.5 in e4m3
    let e_1_5 = E4m3::from_f32(1.5).unwrap().bits();
    let x_bytes = vec![e_1_5; k];
    let w_bytes_data = vec![e_1_5; k];
    let x_scale_vals = vec![1.0f32];
    let w_scale_val = 0.5f32;
    let w_scale_bytes = E4M3Block128Scale::from_bits(f32_to_f16(w_scale_val)).to_bytes();

    let mut w_bytes = vec![0u8; k + 2];
    w_bytes[..k].copy_from_slice(&w_bytes_data);
    w_bytes[k..k + 2].copy_from_slice(&w_scale_bytes);

    let x_buf =
        TypedBuffer::from_bytes(&[m, k], DType::E4m3, &x_bytes).with_quant(QuantScheme::PerToken);
    let x_scale_buf = TypedBuffer::from_f32(&[m], &x_scale_vals);
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::E4m3, &w_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(4)));
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // 128 * (1.5 * 1.5) = 128 * 2.25 = 288.0.
    // scale = 0.5 * 1.0 = 0.5.
    // result = 288.0 * 0.5 = 144.0.
    let y_slice = y_buf.to_f32_vec();
    assert_eq!(y_slice[0], 144.0);
}

#[test]
fn test_matmul_epilogues_bias_residual_activation() {
    let m = 2;
    let k = 16;
    let n = 2;

    let x_buf = TypedBuffer::from_f32(&[m, k], &vec![1.0f32; m * k]);
    let w_bytes: Vec<u8> = (0..(n * k))
        .flat_map(|_| f32_to_f16(0.5f32).to_le_bytes())
        .collect();
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::F16, &w_bytes);

    // Unadorned matmul: 16 * (1.0 * 0.5) = 8.0 for every element
    let mut y_none = TypedBuffer::zeros(&[m, n], DType::F32);
    let op_none = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    matmul(
        &op_none,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_none.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_none.to_f32_vec(), vec![8.0, 8.0, 8.0, 8.0]);

    // Bias epilogue: bias = [2.0, -2.0] => [10.0, 6.0, 10.0, 6.0]
    let bias_buf = TypedBuffer::from_f32(&[n], &[2.0f32, -2.0]);
    let mut y_bias = TypedBuffer::zeros(&[m, n], DType::F32);
    let op_bias = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::Bias,
        transpose_w: false,
    };
    matmul(
        &op_bias,
        &x_buf.as_view(),
        &w_buf.as_view(),
        Some(&bias_buf.as_view()),
        None,
        &mut y_bias.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_bias.to_f32_vec(), vec![10.0, 6.0, 10.0, 6.0]);

    // Residual epilogue: residual = [1.0, 2.0, 3.0, 4.0] => [9.0, 10.0, 11.0, 12.0]
    let res_buf = TypedBuffer::from_f32(&[m, n], &[1.0f32, 2.0, 3.0, 4.0]);
    let mut y_res = TypedBuffer::zeros(&[m, n], DType::F32);
    let op_res = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::Residual,
        transpose_w: false,
    };
    matmul(
        &op_res,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        Some(&res_buf.as_view()),
        &mut y_res.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_res.to_f32_vec(), vec![9.0, 10.0, 11.0, 12.0]);

    // Activation epilogue: Gelu(8.0)
    let mut y_act = TypedBuffer::zeros(&[m, n], DType::F32);
    let op_act = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::Act(ActivationKind::Gelu),
        transpose_w: false,
    };
    matmul(
        &op_act,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_act.as_view_mut(),
    )
    .unwrap();
    let expected_gelu = r9v_t0::activation::eval_activation_f32(8.0, ActivationKind::Gelu);
    for v in y_act.to_f32_vec() {
        assert!((v - expected_gelu).abs() < 1e-6);
    }
}

#[test]
fn test_matmul_batch_invariance_and_sequence_t_invariance() {
    let k = 32;
    let n = 4;

    let w_bytes: Vec<u8> = (0..(n * k))
        .flat_map(|i| f32_to_f16((i as f32 * 0.05) - 0.5).to_le_bytes())
        .collect();
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::F16, &w_bytes);

    let row_data: Vec<f32> = (0..k).map(|i| (i as f32 * 0.1) - 1.5).collect();

    // M=1 alone
    let x_single = TypedBuffer::from_f32(&[1, k], &row_data);
    let mut y_single = TypedBuffer::zeros(&[1, n], DType::F32);
    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    matmul(
        &op,
        &x_single.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_single.as_view_mut(),
    )
    .unwrap();

    // M=5 batched with row_data at index 0, 2, 4
    let mut batched_data = vec![0.0f32; 5 * k];
    batched_data[0..k].copy_from_slice(&row_data);
    batched_data[2 * k..3 * k].copy_from_slice(&row_data);
    batched_data[4 * k..5 * k].copy_from_slice(&row_data);
    let x_batched = TypedBuffer::from_f32(&[5, k], &batched_data);
    let mut y_batched = TypedBuffer::zeros(&[5, n], DType::F32);
    matmul(
        &op,
        &x_batched.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_batched.as_view_mut(),
    )
    .unwrap();

    let s = y_single.to_f32_vec();
    let b = y_batched.to_f32_vec();

    // Bit-identical across T and batch
    assert_eq!(&s[..], &b[0..n]);
    assert_eq!(&s[..], &b[2 * n..3 * n]);
    assert_eq!(&s[..], &b[4 * n..5 * n]);
}

#[test]
fn test_matmul_l0_vs_l1_weight_equivalence() {
    let m = 2;
    let k = 32;
    let n = 16;

    let x_buf = TypedBuffer::from_f32(
        &[m, k],
        &(0..(m * k)).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
    );

    let row_scale_val = 0.25f32;
    let scale_bits = f32_to_f16(row_scale_val);

    // L0 weight buffer: [N, K + 2]
    let row_stride = k + 2;
    let mut l0_bytes = vec![0u8; n * row_stride];
    for r in 0..n {
        for c in 0..k {
            l0_bytes[r * row_stride + c] = ((r as i32 * 3 + c as i32 * 5) % 200 - 100) as i8 as u8;
        }
        l0_bytes[r * row_stride + k..r * row_stride + k + 2]
            .copy_from_slice(&scale_bits.to_le_bytes());
    }

    // L1 weight buffer: tiled 16x16 with trailing scales
    let dims = PaddedDims::new(n as u32, k as u32, Some(16)).unwrap();
    let values_bytes = dims.n_padded() as usize * dims.k_padded() as usize;
    let geom = scale_geometry(SchemeId::I8R, Layout::L1, &dims).unwrap();
    let mut l1_bytes = vec![0u8; values_bytes + geom.region_bytes as usize];

    for r in 0..n {
        for c in 0..k {
            let tile_idx = (r / 16) * dims.k_tiles() as usize + (c / 16);
            let lane = ((c % 16) / 8) * 16 + (r % 16);
            let j = c % 8;
            let offset = tile_idx * 256 + lane * 8 + j;
            l1_bytes[offset] = ((r as i32 * 3 + c as i32 * 5) % 200 - 100) as i8 as u8;
        }
        let scale_offset = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        l1_bytes[values_bytes + scale_offset..values_bytes + scale_offset + 2]
            .copy_from_slice(&scale_bits.to_le_bytes());
    }

    let w_l0 =
        TypedBuffer::from_bytes(&[n, k], DType::I8, &l0_bytes).with_quant(QuantScheme::PerRow);
    let w_l1 = TypedBuffer::from_bytes(&[n, k], DType::I8, &l1_bytes)
        .with_quant(QuantScheme::PerRow)
        .with_layout(LayoutId::L1);

    let mut y_l0 = TypedBuffer::zeros(&[m, n], DType::F32);
    let mut y_l1 = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul(
        &op,
        &x_buf.as_view(),
        &w_l0.as_view(),
        None,
        None,
        &mut y_l0.as_view_mut(),
    )
    .unwrap();
    matmul(
        &op,
        &x_buf.as_view(),
        &w_l1.as_view(),
        None,
        None,
        &mut y_l1.as_view_mut(),
    )
    .unwrap();

    assert_eq!(y_l0.to_f32_vec(), y_l1.to_f32_vec());
}

#[test]
fn test_matmul_adversarial_i32_overflow_returns_typed_error() {
    let m = 1;
    let k = 2;
    let n = 1;

    // Two huge multiplications that overflow i32::MAX
    // i32::MAX is 2,147,483,647.
    // If x = [127, 127] and w = [127, 127], max is 127*127*2 = 32258.
    // But what if K is huge or dot_xq in I4_K / i32 overflows?
    // In Branch A1 (i8 PerToken x I8_R):
    // acc_i32.checked_add(x * w).
    // Let's verify checked_add works without panic.
    let x_buf = TypedBuffer::from_i8(&[m, k], &[127, 127]).with_quant(QuantScheme::PerToken);
    let x_scale = TypedBuffer::from_f32(&[m], &[1.0]);
    let w_bytes = vec![127u8, 127, 0, 0x3c]; // scale = 1.0 in f16
    let w_buf =
        TypedBuffer::from_bytes(&[n, k], DType::I8, &w_bytes).with_quant(QuantScheme::PerRow);
    let mut y = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    // Does not overflow for small K
    assert!(matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y.as_view_mut()
    )
    .is_ok());

    // Adversarial K = 133_150 overflows i32 (133_150 * 127 * 127 = 2,147,576,350 > i32::MAX)
    let k_overflow = 133_150;
    let x_over = TypedBuffer::from_i8(&[1, k_overflow], &vec![127i8; k_overflow])
        .with_quant(QuantScheme::PerToken);
    let mut w_over_bytes = vec![127u8; k_overflow + 2];
    w_over_bytes[k_overflow..k_overflow + 2].copy_from_slice(&[0, 0x3c]);
    let w_over = TypedBuffer::from_bytes(&[1, k_overflow], DType::I8, &w_over_bytes)
        .with_quant(QuantScheme::PerRow);
    let mut y_over = TypedBuffer::zeros(&[1, 1], DType::F32);

    let err = matmul_with_scales(
        &op,
        &x_over.as_view(),
        Some(&x_scale.as_view()),
        &w_over.as_view(),
        None,
        None,
        None,
        &mut y_over.as_view_mut(),
    )
    .unwrap_err();

    assert!(matches!(err, T0Error::ArithmeticOverflow { .. }));
}

#[test]
fn test_matmul_reserved_scheme_fails_closed() {
    let m = 2;
    let k = 16;
    let n = 2;

    let x = TypedBuffer::zeros(&[m, k], DType::F16);
    let w = TypedBuffer::from_bytes(&[n, k], DType::I8, &vec![0u8; n * k])
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(5))); // Reserved Scheme 5 (i8_b32f)
    let mut y = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    let err = matmul(
        &op,
        &x.as_view(),
        &w.as_view(),
        None,
        None,
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    match err {
        T0Error::Format(r9v_format::FormatError::ReservedScheme { scheme, owner }) => {
            assert_eq!(scheme, "i8_b32f");
            assert_eq!(owner, "A2.3");
        }
        other => panic!("expected ReservedScheme, got {other:?}"),
    }
}

#[test]
fn test_matmul_missing_operands_and_dimension_mismatch() {
    let m = 2;
    let k = 16;
    let n = 2;

    let x = TypedBuffer::zeros(&[m, k], DType::F16);
    let w_bytes: Vec<u8> = vec![0u8; n * k * 2];
    let w = TypedBuffer::from_bytes(&[n, k], DType::F16, &w_bytes);
    let mut y = TypedBuffer::zeros(&[m, n], DType::F32);

    // Missing bias when epilogue is Bias
    let op_bias = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::Bias,
        transpose_w: false,
    };
    let err_bias = matmul(
        &op_bias,
        &x.as_view(),
        &w.as_view(),
        None,
        None,
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(
        err_bias,
        T0Error::MissingOperand {
            operand: "bias",
            ..
        }
    ));

    // Missing residual when epilogue is Residual
    let op_res = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::Residual,
        transpose_w: false,
    };
    let err_res = matmul(
        &op_res,
        &x.as_view(),
        &w.as_view(),
        None,
        None,
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(
        err_res,
        T0Error::MissingOperand {
            operand: "residual",
            ..
        }
    ));

    // Dimension mismatch on K
    let w_bad_k = TypedBuffer::from_bytes(&[n, k + 4], DType::F16, &vec![0u8; n * (k + 4) * 2]);
    let op_none = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    let err_k = matmul(
        &op_none,
        &x.as_view(),
        &w_bad_k.as_view(),
        None,
        None,
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(
        err_k,
        T0Error::DimensionMismatch { dim_name: "K", .. }
    ));
}

#[test]
fn test_matmul_bf16_x_f16() {
    let m = 2;
    let k = 8;
    let n = 2;

    let x_vals: Vec<u16> = (0..(m * k))
        .map(|i| r9v_t0::dtype::f32_to_bf16(i as f32 * 0.5))
        .collect();
    let w_bytes: Vec<u8> = (0..(n * k))
        .flat_map(|i| f32_to_f16(i as f32 * 0.25).to_le_bytes())
        .collect();

    let x_buf = TypedBuffer::from_bf16(&[m, k], &x_vals);
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::F16, &w_bytes);
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    assert!(matmul(
        &op,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut()
    )
    .is_ok());
}

#[test]
fn test_matmul_output_dtypes_f16_and_bf16() {
    let m = 2;
    let k = 4;
    let n = 2;

    let x_buf = TypedBuffer::from_f16(&[m, k], &[f32_to_f16(1.0); 8]);
    let w_bytes: Vec<u8> = (0..(n * k))
        .flat_map(|_| f32_to_f16(2.0).to_le_bytes())
        .collect();
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::F16, &w_bytes);

    // Output F16: 4 * (1.0 * 2.0) = 8.0
    let mut y_f16 = TypedBuffer::zeros(&[m, n], DType::F16);
    let op_f16 = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    matmul(
        &op_f16,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_f16.as_view_mut(),
    )
    .unwrap();
    let y_f16_slice = y_f16.as_view().as_f16_slice().unwrap().to_vec();
    assert_eq!(y_f16_slice, vec![f32_to_f16(8.0); 4]);

    // Output Bf16
    let mut y_bf16 = TypedBuffer::zeros(&[m, n], DType::Bf16);
    let op_bf16 = MatmulOp {
        out_dtype: DType::Bf16,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    matmul(
        &op_bf16,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_bf16.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_bf16.to_f32_vec(), vec![8.0, 8.0, 8.0, 8.0]);
}

#[test]
fn test_matmul_transpose_w_true() {
    let m = 2;
    let k = 4;
    let n = 3;

    let x_buf = TypedBuffer::from_f16(&[m, k], &[f32_to_f16(1.0); 8]);
    // When transpose_w is true, w has shape [K, N] = [4, 3]
    let w_bytes: Vec<u8> = (0..(k * n))
        .flat_map(|i| f32_to_f16(i as f32).to_le_bytes())
        .collect();
    let w_buf = TypedBuffer::from_bytes(&[k, n], DType::F16, &w_bytes);
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: true,
    };

    matmul(
        &op,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.shape(), &[m, n]);
    let y = y_buf.to_f32_vec();
    // col 0: 0 + 3 + 6 + 9 = 18.0
    // col 1: 1 + 4 + 7 + 10 = 22.0
    // col 2: 2 + 5 + 8 + 11 = 26.0
    assert_eq!(y, vec![18.0, 22.0, 26.0, 18.0, 22.0, 26.0]);
}

#[test]
fn test_matmul_i8_per_block32_x_i8_b128() {
    let m = 1;
    let k = 128;
    let n = 1;

    let x_vals = vec![1i8; k];
    let x_scales = vec![1.0f32; k / 32]; // 4 scales for 4 blocks of 32
    let w_vals = vec![2i8; k];
    let w_scale_val = 0.5f32;
    let w_scale_bits = f32_to_f16(w_scale_val);

    let row_stride = k + 2;
    let mut w_bytes = vec![0u8; row_stride];
    for c in 0..k {
        w_bytes[c] = w_vals[c] as u8;
    }
    w_bytes[k..k + 2].copy_from_slice(&w_scale_bits.to_le_bytes());

    let x_buf = TypedBuffer::from_i8(&[m, k], &x_vals).with_quant(QuantScheme::PerBlock32);
    let x_scale_buf = TypedBuffer::from_f32(&[m, k / 32], &x_scales);
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::I8, &w_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(2)));
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // 4 blocks of 32, each block: 32 * (1 * 2) = 64.
    // scale for each block: 0.5 * 1.0 = 0.5.
    // each block contributes 64 * 0.5 = 32.0.
    // 4 blocks * 32.0 = 128.0.
    assert_eq!(y_buf.to_f32_vec()[0], 128.0);
}

#[test]
fn test_matmul_i8_per_block32_x_i4_k() {
    let m = 1;
    let k = 256;
    let n = 1;

    let x_vals = vec![1i8; k];
    let x_scales = vec![1.0f32; k / 32]; // 8 scales of 1.0

    let d_val = 1.0f32;
    let dmin_val = 0.5f32;
    let d_bits = f32_to_f16(d_val);
    let dmin_bits = f32_to_f16(dmin_val);
    let sc = [2u8; 8];
    let mn = [1u8; 8];

    let header = I4KSuperblock::pack(d_bits, dmin_bits, sc, mn).unwrap();
    let header_bytes = header.to_bytes();

    let values_bytes = k / 2; // 128
    let mut w_bytes = vec![0x33u8; values_bytes + 16];
    w_bytes[values_bytes..values_bytes + 16].copy_from_slice(&header_bytes);

    let x_buf = TypedBuffer::from_i8(&[m, k], &x_vals).with_quant(QuantScheme::PerBlock32);
    let x_scale_buf = TypedBuffer::from_f32(&[m, k / 32], &x_scales);
    let w_buf = TypedBuffer::from_bytes(&[n, k], DType::I4, &w_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(3)));
    let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    matmul_with_scales(
        &op,
        &x_buf.as_view(),
        Some(&x_scale_buf.as_view()),
        &w_buf.as_view(),
        None,
        None,
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // 8 blocks of 32: each block has (2.0 * 96 - 0.5 * 32) * 1.0 = 176.
    // 8 * 176 = 1408.0.
    assert_eq!(y_buf.to_f32_vec()[0], 1408.0);
}

#[test]
fn test_matmul_non_finite_scale_rejected() {
    let m = 2;
    let k = 32;
    let n = 2;

    let x = TypedBuffer::from_i8(&[m, k], &[1i8; 64]).with_quant(QuantScheme::PerToken);
    let x_scale_bad = TypedBuffer::from_f32(&[m], &[1.0, f32::NAN]);
    let w_bytes = vec![0u8; n * (k + 2)];
    let w = TypedBuffer::from_bytes(&[n, k], DType::I8, &w_bytes).with_quant(QuantScheme::PerRow);
    let mut y = TypedBuffer::zeros(&[m, n], DType::F32);

    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    let err = matmul_with_scales(
        &op,
        &x.as_view(),
        Some(&x_scale_bad.as_view()),
        &w.as_view(),
        None,
        None,
        None,
        &mut y.as_view_mut(),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        T0Error::InvalidAttribute {
            attribute: "x_scale",
            ..
        }
    ));
}
