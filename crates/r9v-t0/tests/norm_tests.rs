// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 norm (Spec 1 §4.B, §6.1, §6.4, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{DType, NormAxis, NormKind, NormOp};
use r9v_t0::{norm, norm_f64_reference, Tolerance, TypedBuffer};

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
fn norm_rms_last_axis_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5001);
    let tol = Tolerance::f32();

    for &(t, n) in &[(1, 64), (2, 128), (4, 256), (3, 768)] {
        let op = NormOp {
            kind: NormKind::Rms,
            eps: 1e-5,
            axis: NormAxis::Last,
            weight_offset: 0.0,
            out_dtype: DType::F32,
        };

        let x_data = generate_f32_data(&mut rng, t * n, 5.0);
        let w_data = generate_f32_data(&mut rng, n, 2.0);

        let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);
        let w_buf = TypedBuffer::from_f32(&[n], &w_data);
        let mut y_buf = TypedBuffer::zeros(&[t, n], DType::F32);

        norm(
            &op,
            &x_buf.as_view(),
            &w_buf.as_view(),
            None,
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
        let w_f64: Vec<f64> = w_data.iter().map(|&v| v as f64).collect();
        let ref_f64 = norm_f64_reference(&op, &x_f64, [t, n], &w_f64, None, 0.0, 1e-5);

        for i in 0..(t * n) {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(
                actual,
                expected,
                &format!("norm_rms_last_axis [t={t}, n={n}] at {i}"),
            );
        }
    }
}

#[test]
fn norm_rms_head_axis_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5002);
    let tol = Tolerance::f32();

    let d = 64;
    for &(t, n) in &[(1, 128), (2, 256), (4, 512)] {
        let op = NormOp {
            kind: NormKind::Rms,
            eps: 1e-6,
            axis: NormAxis::Head(d as u32),
            weight_offset: 0.0,
            out_dtype: DType::F32,
        };

        let x_data = generate_f32_data(&mut rng, t * n, 4.0);
        let w_data = generate_f32_data(&mut rng, n, 1.5);

        let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);
        let w_buf = TypedBuffer::from_f32(&[n], &w_data);
        let mut y_buf = TypedBuffer::zeros(&[t, n], DType::F32);

        norm(
            &op,
            &x_buf.as_view(),
            &w_buf.as_view(),
            None,
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
        let w_f64: Vec<f64> = w_data.iter().map(|&v| v as f64).collect();
        let ref_f64 = norm_f64_reference(&op, &x_f64, [t, n], &w_f64, None, 0.0, 1e-6);

        for i in 0..(t * n) {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(
                actual,
                expected,
                &format!("norm_rms_head_axis [t={t}, n={n}] at {i}"),
            );
        }
    }
}

#[test]
fn norm_rms_weight_offset_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5003);
    let tol = Tolerance::f32();

    let (t, n) = (2, 128);
    let op = NormOp {
        kind: NormKind::Rms,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 1.0, // Gemma's (1 + w) parameterization
        out_dtype: DType::F32,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 3.0);
    let w_data = generate_f32_data(&mut rng, n, 0.5);

    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);
    let w_buf = TypedBuffer::from_f32(&[n], &w_data);
    let mut y_buf = TypedBuffer::zeros(&[t, n], DType::F32);

    norm(
        &op,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let w_f64: Vec<f64> = w_data.iter().map(|&v| v as f64).collect();
    let ref_f64 = norm_f64_reference(&op, &x_f64, [t, n], &w_f64, None, 1.0, 1e-5);

    for i in 0..(t * n) {
        let actual = y_buf.read_f32(i) as f64;
        let expected = ref_f64[i];
        tol.assert_within(actual, expected, &format!("norm_rms_weight_offset at {i}"));
    }
}

#[test]
fn norm_layer_last_and_head_axis_with_bias_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5004);
    let tol = Tolerance::f32();

    // 1. Last axis with bias
    let (t, n) = (2, 256);
    let op_last = NormOp {
        kind: NormKind::Layer,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 0.0,
        out_dtype: DType::F32,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 4.0);
    let w_data = generate_f32_data(&mut rng, n, 1.0);
    let b_data = generate_f32_data(&mut rng, n, 0.5);

    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);
    let w_buf = TypedBuffer::from_f32(&[n], &w_data);
    let b_buf = TypedBuffer::from_f32(&[n], &b_data);
    let mut y_buf = TypedBuffer::zeros(&[t, n], DType::F32);

    norm(
        &op_last,
        &x_buf.as_view(),
        &w_buf.as_view(),
        Some(&b_buf.as_view()),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let w_f64: Vec<f64> = w_data.iter().map(|&v| v as f64).collect();
    let b_f64: Vec<f64> = b_data.iter().map(|&v| v as f64).collect();
    let ref_f64 = norm_f64_reference(&op_last, &x_f64, [t, n], &w_f64, Some(&b_f64), 0.0, 1e-5);

    for i in 0..(t * n) {
        let actual = y_buf.read_f32(i) as f64;
        let expected = ref_f64[i];
        tol.assert_within(actual, expected, &format!("norm_layer_last at {i}"));
    }

    // 2. Head axis with bias
    let d = 64;
    let op_head = NormOp {
        kind: NormKind::Layer,
        eps: 1e-5,
        axis: NormAxis::Head(d),
        weight_offset: 0.0,
        out_dtype: DType::F32,
    };

    norm(
        &op_head,
        &x_buf.as_view(),
        &w_buf.as_view(),
        Some(&b_buf.as_view()),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let ref_head_f64 =
        norm_f64_reference(&op_head, &x_f64, [t, n], &w_f64, Some(&b_f64), 0.0, 1e-5);

    for i in 0..(t * n) {
        let actual = y_buf.read_f32(i) as f64;
        let expected = ref_head_f64[i];
        tol.assert_within(actual, expected, &format!("norm_layer_head at {i}"));
    }
}

#[test]
fn norm_f16_and_bf16_precision_paths() {
    let mut rng = SeededRng::new(0xA1_5005);
    let tol = Tolerance::f16_bf16();

    let (t, n) = (2, 64);
    let x_data = generate_f32_data(&mut rng, t * n, 2.0);
    let w_data = generate_f32_data(&mut rng, n, 1.0);

    for &dt in &[DType::F16, DType::Bf16] {
        let op = NormOp {
            kind: NormKind::Rms,
            eps: 1e-4,
            axis: NormAxis::Last,
            weight_offset: 0.0,
            out_dtype: dt,
        };

        let mut x_buf = TypedBuffer::zeros(&[t, n], dt);
        for (i, &v) in x_data.iter().enumerate() {
            x_buf.write_f32(i, v);
        }
        let w_buf = TypedBuffer::from_f32(&[n], &w_data);
        let mut y_buf = TypedBuffer::zeros(&[t, n], dt);

        norm(
            &op,
            &x_buf.as_view(),
            &w_buf.as_view(),
            None,
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let x_f64: Vec<f64> = (0..(t * n)).map(|i| x_buf.read_f32(i) as f64).collect();
        let w_f64: Vec<f64> = w_data.iter().map(|&v| v as f64).collect();
        let ref_f64 = norm_f64_reference(&op, &x_f64, [t, n], &w_f64, None, 0.0, 1e-4);

        for i in 0..(t * n) {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(actual, expected, &format!("norm_{dt} at {i}"));
        }
    }
}

#[test]
fn norm_batch_invariance_rms_and_layer() {
    let mut rng = SeededRng::new(0xA1_5006);
    let n = 128;

    // Single target token row
    let target_row = generate_f32_data(&mut rng, n, 3.0);
    let w_data = generate_f32_data(&mut rng, n, 1.2);
    let b_data = generate_f32_data(&mut rng, n, 0.3);

    for &kind in &[NormKind::Rms, NormKind::Layer] {
        let op = NormOp {
            kind,
            eps: 1e-5,
            axis: NormAxis::Last,
            weight_offset: 0.5,
            out_dtype: DType::F32,
        };

        // 1. Target token alone (T = 1)
        let x_alone = TypedBuffer::from_f32(&[1, n], &target_row);
        let w_buf = TypedBuffer::from_f32(&[n], &w_data);
        let b_buf = TypedBuffer::from_f32(&[n], &b_data);
        let mut y_alone = TypedBuffer::zeros(&[1, n], DType::F32);
        norm(
            &op,
            &x_alone.as_view(),
            &w_buf.as_view(),
            Some(&b_buf.as_view()),
            &mut y_alone.as_view_mut(),
        )
        .unwrap();
        let out_alone = y_alone.to_f32_vec();

        // 2. Padded bucket (T = 4, target token at index 0, rest zeros)
        let mut padded_data = target_row.clone();
        padded_data.extend(vec![0.0f32; 3 * n]);
        let x_padded = TypedBuffer::from_f32(&[4, n], &padded_data);
        let mut y_padded = TypedBuffer::zeros(&[4, n], DType::F32);
        norm(
            &op,
            &x_padded.as_view(),
            &w_buf.as_view(),
            Some(&b_buf.as_view()),
            &mut y_padded.as_view_mut(),
        )
        .unwrap();
        let out_padded = &y_padded.to_f32_vec()[..n];

        // 3. Embedded among random tokens (T = 4, target token at index 2)
        let other_tokens = generate_f32_data(&mut rng, 3 * n, 5.0);
        let mut embedded_data = Vec::with_capacity(4 * n);
        embedded_data.extend_from_slice(&other_tokens[..2 * n]);
        embedded_data.extend_from_slice(&target_row); // token at index 2
        embedded_data.extend_from_slice(&other_tokens[2 * n..]);
        let x_embedded = TypedBuffer::from_f32(&[4, n], &embedded_data);
        let mut y_embedded = TypedBuffer::zeros(&[4, n], DType::F32);
        norm(
            &op,
            &x_embedded.as_view(),
            &w_buf.as_view(),
            Some(&b_buf.as_view()),
            &mut y_embedded.as_view_mut(),
        )
        .unwrap();
        let out_embedded = &y_embedded.to_f32_vec()[2 * n..3 * n];

        // Spec 1 §6.1: batch invariance requires bit-identical L0 equality
        for i in 0..n {
            assert_eq!(
                out_alone[i].to_bits(),
                out_padded[i].to_bits(),
                "batch invariance failed between alone and padded on {kind:?} at element {i}"
            );
            assert_eq!(
                out_alone[i].to_bits(),
                out_embedded[i].to_bits(),
                "batch invariance failed between alone and embedded on {kind:?} at element {i}"
            );
        }
    }
}

#[test]
fn norm_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5007);
    let (t, n) = (4, 128);

    let op = NormOp {
        kind: NormKind::Rms,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 1.0,
        out_dtype: DType::F32,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 4.0);
    let w_data = generate_f32_data(&mut rng, n, 1.0);

    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);
    let w_buf = TypedBuffer::from_f32(&[n], &w_data);

    let mut y1 = TypedBuffer::zeros(&[t, n], DType::F32);
    let mut y2 = TypedBuffer::zeros(&[t, n], DType::F32);

    norm(
        &op,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        &mut y1.as_view_mut(),
    )
    .unwrap();
    norm(
        &op,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        &mut y2.as_view_mut(),
    )
    .unwrap();

    let out1 = y1.to_f32_vec();
    let out2 = y2.to_f32_vec();

    for i in 0..(t * n) {
        assert_eq!(
            out1[i].to_bits(),
            out2[i].to_bits(),
            "determinism failed at {i}"
        );
    }
}

#[test]
fn norm_rejects_malformed_inputs_with_complete_error() {
    let op = NormOp {
        kind: NormKind::Rms,
        eps: 1e-5,
        axis: NormAxis::Head(50), // 128 not divisible by 50
        weight_offset: 0.0,
        out_dtype: DType::F32,
    };

    let x_buf = TypedBuffer::zeros(&[2, 128], DType::F32);
    let w_buf = TypedBuffer::zeros(&[100], DType::F32); // 100 != 128
    let mut y_buf = TypedBuffer::zeros(&[2, 64], DType::F32); // 64 != 128

    let err = norm(
        &op,
        &x_buf.as_view(),
        &w_buf.as_view(),
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("not divisible"));
    assert!(msg.contains("does not match"));
}
