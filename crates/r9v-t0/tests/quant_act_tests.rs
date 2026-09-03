// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 quant_act (Spec 1 §4.A, Spec 2 §3.4, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{DType, QuantActOp, QuantScheme, Smoothing};
use r9v_t0::{quant_act, quant_act_f64_reference, Tolerance, TypedBuffer};

fn generate_f32_data(rng: &mut SeededRng, len: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let raw = (rng.next_u64() & 0xFFFF_FFFF) as u32;
        let norm_val = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(norm_val * scale);
    }
    out
}

#[test]
fn quant_act_per_token_i8_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5070);
    let tol = Tolerance::f32();
    let (t, n) = (3, 128);

    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 50.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let (ref_xq, ref_scale) = quant_act_f64_reference(&op, &x_f64, [t, n]);

    for row in 0..t {
        let actual_scale = scale_buf.read_f32(row) as f64;
        let expected_scale = ref_scale[row];
        tol.assert_within(
            actual_scale,
            expected_scale,
            &format!("quant_act scale at token {row}"),
        );
    }

    let actual_xq = xq_buf.to_i8_vec();
    for i in 0..(t * n) {
        assert_eq!(
            actual_xq[i] as f64, ref_xq[i],
            "quant_act i8 quantized mismatch at {i}"
        );
        // Check symmetric bound [-127, 127] (i8 cannot be -128)
        assert!(actual_xq[i] >= -127);
    }
}

#[test]
fn quant_act_per_token_e4m3_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5071);
    let tol = Tolerance::f32();
    let (t, n) = (3, 64);

    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::E4m3,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 100.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::E4m3);
    let mut scale_buf = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let (ref_xq, ref_scale) = quant_act_f64_reference(&op, &x_f64, [t, n]);

    for row in 0..t {
        let actual_scale = scale_buf.read_f32(row) as f64;
        let expected_scale = ref_scale[row];
        tol.assert_within(
            actual_scale,
            expected_scale,
            &format!("quant_act e4m3 scale at token {row}"),
        );
    }

    let actual_bytes = xq_buf.to_byte_vec();
    for i in 0..(t * n) {
        assert_eq!(
            actual_bytes[i] as f64, ref_xq[i],
            "quant_act e4m3 byte mismatch at {i}"
        );
    }
}

#[test]
fn quant_act_per_block32_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5072);
    let tol = Tolerance::f32();
    let (t, n) = (2, 128); // 4 blocks of 32 per row

    let op = QuantActOp {
        scheme: QuantScheme::PerBlock32,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 40.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let num_blocks = n / 32;
    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[t, num_blocks], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let (ref_xq, ref_scale) = quant_act_f64_reference(&op, &x_f64, [t, n]);

    for b in 0..(t * num_blocks) {
        let actual_scale = scale_buf.read_f32(b) as f64;
        let expected_scale = ref_scale[b];
        tol.assert_within(
            actual_scale,
            expected_scale,
            &format!("quant_act perblock scale at block {b}"),
        );
    }

    let actual_xq = xq_buf.to_i8_vec();
    for i in 0..(t * n) {
        assert_eq!(
            actual_xq[i] as f64, ref_xq[i],
            "quant_act perblock i8 quantized mismatch at {i}"
        );
    }
}

#[test]
fn quant_act_zero_absmax_emits_zeros() {
    let (t, n) = (2, 64);
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_buf = TypedBuffer::zeros(&[t, n], DType::F32); // All zeros
    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    for row in 0..t {
        assert_eq!(scale_buf.read_f32(row), 0.0f32);
    }
    for val in xq_buf.to_i8_vec() {
        assert_eq!(val, 0);
    }
}

#[test]
fn quant_act_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5073);
    let n = 128;

    let target_row = generate_f32_data(&mut rng, n, 20.0);
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    // 1. Alone (T = 1)
    let x_alone = TypedBuffer::from_f32(&[1, n], &target_row);
    let mut xq_alone = TypedBuffer::zeros(&[1, n], DType::I8);
    let mut scale_alone = TypedBuffer::zeros(&[1], DType::F32);
    quant_act(
        &op,
        &x_alone.as_view(),
        &mut xq_alone.as_view_mut(),
        &mut scale_alone.as_view_mut(),
    )
    .unwrap();
    let out_alone = xq_alone.to_i8_vec();
    let scale_alone_val = scale_alone.read_f32(0);

    // 2. Padded (T = 4)
    let mut x_pad = target_row.clone();
    x_pad.extend(vec![0.0f32; 3 * n]);
    let x_pad_buf = TypedBuffer::from_f32(&[4, n], &x_pad);
    let mut xq_pad = TypedBuffer::zeros(&[4, n], DType::I8);
    let mut scale_pad = TypedBuffer::zeros(&[4], DType::F32);
    quant_act(
        &op,
        &x_pad_buf.as_view(),
        &mut xq_pad.as_view_mut(),
        &mut scale_pad.as_view_mut(),
    )
    .unwrap();
    let out_pad = &xq_pad.to_i8_vec()[..n];
    let scale_pad_val = scale_pad.read_f32(0);

    // 3. Embedded (T = 4, target at index 2)
    let other_tokens = generate_f32_data(&mut rng, 3 * n, 50.0);
    let mut x_emb = Vec::with_capacity(4 * n);
    x_emb.extend_from_slice(&other_tokens[..2 * n]);
    x_emb.extend_from_slice(&target_row);
    x_emb.extend_from_slice(&other_tokens[2 * n..]);
    let x_emb_buf = TypedBuffer::from_f32(&[4, n], &x_emb);
    let mut xq_emb = TypedBuffer::zeros(&[4, n], DType::I8);
    let mut scale_emb = TypedBuffer::zeros(&[4], DType::F32);
    quant_act(
        &op,
        &x_emb_buf.as_view(),
        &mut xq_emb.as_view_mut(),
        &mut scale_emb.as_view_mut(),
    )
    .unwrap();
    let out_emb = &xq_emb.to_i8_vec()[2 * n..3 * n];
    let scale_emb_val = scale_emb.read_f32(2);

    assert_eq!(scale_alone_val.to_bits(), scale_pad_val.to_bits());
    assert_eq!(scale_alone_val.to_bits(), scale_emb_val.to_bits());
    for i in 0..n {
        assert_eq!(out_alone[i], out_pad[i]);
        assert_eq!(out_alone[i], out_emb[i]);
    }
}

#[test]
fn quant_act_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5074);
    let (t, n) = (3, 64);

    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::E4m3,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 30.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let mut xq1 = TypedBuffer::zeros(&[t, n], DType::E4m3);
    let mut s1 = TypedBuffer::zeros(&[t], DType::F32);
    let mut xq2 = TypedBuffer::zeros(&[t, n], DType::E4m3);
    let mut s2 = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq1.as_view_mut(),
        &mut s1.as_view_mut(),
    )
    .unwrap();
    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq2.as_view_mut(),
        &mut s2.as_view_mut(),
    )
    .unwrap();

    assert_eq!(xq1.to_byte_vec(), xq2.to_byte_vec());
    assert_eq!(s1.to_f32_vec(), s2.to_f32_vec());
}

#[test]
fn quant_act_rejects_non_divisible_block_size_and_mismatches() {
    let op = QuantActOp {
        scheme: QuantScheme::PerBlock32,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_buf = TypedBuffer::zeros(&[2, 50], DType::F32); // 50 not divisible by 32
    let mut xq_buf = TypedBuffer::zeros(&[2, 50], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[2, 2], DType::F32);

    let err = quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("divisible by 32"));
}

#[test]
fn quant_act_rejects_invalid_per_token_target_without_panicking() {
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::F32, // Invalid target for quant_act (must be i8 or e4m3)
        smoothing: Smoothing::None,
    };

    let x_buf = TypedBuffer::zeros(&[2, 32], DType::F32);
    let mut xq_buf = TypedBuffer::zeros(&[2, 32], DType::F32);
    let mut scale_buf = TypedBuffer::zeros(&[2], DType::F32);

    let err = quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap_err();

    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("target must be i8 or e4m3") || msg.contains("only supports i8 or e4m3"));
}
