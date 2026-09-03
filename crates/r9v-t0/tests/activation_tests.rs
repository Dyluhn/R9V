// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 activation (Spec 1 §4.B, §6.1, §6.4, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{ActivationKind, ActivationOp, DType};
use r9v_t0::{activation, activation_f64_reference, Tolerance, TypedBuffer};

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
fn activation_all_activations_match_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5030);
    let tol = Tolerance::f32();
    let (t, dff) = (3, 128);
    let num_elem = t * dff;

    let kinds = [
        ActivationKind::Silu,
        ActivationKind::Gelu,
        ActivationKind::GeluTanh,
        ActivationKind::Relu2,
        ActivationKind::Identity,
    ];

    for &kind in &kinds {
        for &clamp in &[None, Some(5.0)] {
            let op = ActivationOp { act: kind, clamp };

            let x_data = generate_f32_data(&mut rng, num_elem, 3.5);
            let x_buf = TypedBuffer::from_f32(&[t, dff], &x_data);
            let mut y_buf = TypedBuffer::zeros(&[t, dff], DType::F32);

            activation(&op, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap();

            let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
            let ref_f64 = activation_f64_reference(&op, &x_f64);

            for i in 0..num_elem {
                let actual = y_buf.read_f32(i) as f64;
                let expected = ref_f64[i];
                tol.assert_within(
                    actual,
                    expected,
                    &format!("activation {kind:?} clamp={clamp:?} at {i}"),
                );
            }
        }
    }
}

#[test]
fn activation_f16_and_bf16_precision_paths() {
    let mut rng = SeededRng::new(0xA1_5031);
    let tol = Tolerance::f16_bf16();
    let (t, dff) = (2, 64);
    let num_elem = t * dff;

    let x_data = generate_f32_data(&mut rng, num_elem, 2.5);

    for &dt in &[DType::F16, DType::Bf16] {
        let op = ActivationOp {
            act: ActivationKind::Gelu,
            clamp: None,
        };

        let mut x_buf = TypedBuffer::zeros(&[t, dff], dt);
        let mut y_buf = TypedBuffer::zeros(&[t, dff], dt);

        for i in 0..num_elem {
            x_buf.write_f32(i, x_data[i]);
        }

        activation(&op, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap();

        let x_f64: Vec<f64> = (0..num_elem).map(|i| x_buf.read_f32(i) as f64).collect();
        let ref_f64 = activation_f64_reference(&op, &x_f64);

        for i in 0..num_elem {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(actual, expected, &format!("activation_{dt} at {i}"));
        }
    }
}

#[test]
fn activation_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5032);
    let dff = 128;

    let target_row = generate_f32_data(&mut rng, dff, 4.0);
    let op = ActivationOp {
        act: ActivationKind::GeluTanh,
        clamp: Some(4.0),
    };

    // 1. Alone (T = 1)
    let x_alone = TypedBuffer::from_f32(&[1, dff], &target_row);
    let mut y_alone = TypedBuffer::zeros(&[1, dff], DType::F32);
    activation(&op, &x_alone.as_view(), &mut y_alone.as_view_mut()).unwrap();
    let out_alone = y_alone.to_f32_vec();

    // 2. Padded (T = 4)
    let mut x_pad = target_row.clone();
    x_pad.extend(vec![0.0f32; 3 * dff]);
    let x_pad_buf = TypedBuffer::from_f32(&[4, dff], &x_pad);
    let mut y_padded = TypedBuffer::zeros(&[4, dff], DType::F32);
    activation(&op, &x_pad_buf.as_view(), &mut y_padded.as_view_mut()).unwrap();
    let out_padded = &y_padded.to_f32_vec()[..dff];

    // 3. Embedded (T = 4, index 2)
    let other_rows = generate_f32_data(&mut rng, 3 * dff, 5.0);
    let mut x_emb = Vec::with_capacity(4 * dff);
    x_emb.extend_from_slice(&other_rows[..2 * dff]);
    x_emb.extend_from_slice(&target_row);
    x_emb.extend_from_slice(&other_rows[2 * dff..]);

    let x_emb_buf = TypedBuffer::from_f32(&[4, dff], &x_emb);
    let mut y_emb = TypedBuffer::zeros(&[4, dff], DType::F32);
    activation(&op, &x_emb_buf.as_view(), &mut y_emb.as_view_mut()).unwrap();
    let out_emb = &y_emb.to_f32_vec()[2 * dff..3 * dff];

    for i in 0..dff {
        assert_eq!(
            out_alone[i].to_bits(),
            out_padded[i].to_bits(),
            "batch invariance alone vs padded at {i}"
        );
        assert_eq!(
            out_alone[i].to_bits(),
            out_emb[i].to_bits(),
            "batch invariance alone vs embedded at {i}"
        );
    }
}

#[test]
fn activation_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5033);
    let (t, dff) = (4, 128);
    let num_elem = t * dff;

    let op = ActivationOp {
        act: ActivationKind::Relu2,
        clamp: Some(6.0),
    };

    let x_data = generate_f32_data(&mut rng, num_elem, 3.0);
    let x_buf = TypedBuffer::from_f32(&[t, dff], &x_data);

    let mut y1 = TypedBuffer::zeros(&[t, dff], DType::F32);
    let mut y2 = TypedBuffer::zeros(&[t, dff], DType::F32);

    activation(&op, &x_buf.as_view(), &mut y1.as_view_mut()).unwrap();
    activation(&op, &x_buf.as_view(), &mut y2.as_view_mut()).unwrap();

    let out1 = y1.to_f32_vec();
    let out2 = y2.to_f32_vec();

    for i in 0..num_elem {
        assert_eq!(
            out1[i].to_bits(),
            out2[i].to_bits(),
            "determinism failed at {i}"
        );
    }
}

#[test]
fn activation_rejects_malformed_operands() {
    let op = ActivationOp {
        act: ActivationKind::Silu,
        clamp: Some(0.0), // invalid clamp <= 0
    };

    let x_buf = TypedBuffer::zeros(&[2, 64], DType::F32);
    let mut y_buf = TypedBuffer::zeros(&[2, 128], DType::F32); // shape mismatch

    let err = activation(&op, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("clamp must be finite and > 0"));
    assert!(msg.contains("does not match"));
}

#[test]
fn erf_f32_all_f32_arithmetic_and_matches_f64_oracle() {
    use r9v_t0::{erf_f32, erf_f64};

    assert_eq!(erf_f32(0.0f32), 0.0f32);
    assert!(erf_f32(f32::NAN).is_nan());

    let test_points = [
        -4.0f32, -2.5, -1.8, -1.5, -1.2, -0.8, -0.5, -0.1, 0.0, 0.1, 0.5, 0.8, 1.2, 1.5, 1.8, 2.5,
        4.0,
    ];

    for &x in &test_points {
        let actual = erf_f32(x);
        let expected = erf_f64(x as f64) as f32;
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1e-5,
            "erf_f32({x}) = {actual}, expected {expected}, diff {diff}"
        );
    }
}
