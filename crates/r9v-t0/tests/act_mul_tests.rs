// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 act_mul (Spec 1 §4.B, §6.1, §6.4, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{ActMulOp, ActivationKind, DType};
use r9v_t0::{act_mul, act_mul_f64_reference, Tolerance, TypedBuffer};

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
fn act_mul_all_activations_match_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5020);
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
        for &clamp in &[None, Some(4.0)] {
            let op = ActMulOp { act: kind, clamp };

            let gate_data = generate_f32_data(&mut rng, num_elem, 3.0);
            let up_data = generate_f32_data(&mut rng, num_elem, 2.0);

            let gate_buf = TypedBuffer::from_f32(&[t, dff], &gate_data);
            let up_buf = TypedBuffer::from_f32(&[t, dff], &up_data);
            let mut y_buf = TypedBuffer::zeros(&[t, dff], DType::F32);

            act_mul(
                &op,
                &gate_buf.as_view(),
                &up_buf.as_view(),
                &mut y_buf.as_view_mut(),
            )
            .unwrap();

            let gate_f64: Vec<f64> = gate_data.iter().map(|&v| v as f64).collect();
            let up_f64: Vec<f64> = up_data.iter().map(|&v| v as f64).collect();
            let ref_f64 = act_mul_f64_reference(&op, &gate_f64, &up_f64);

            for i in 0..num_elem {
                let actual = y_buf.read_f32(i) as f64;
                let expected = ref_f64[i];
                tol.assert_within(
                    actual,
                    expected,
                    &format!("act_mul {kind:?} clamp={clamp:?} at {i}"),
                );
            }
        }
    }
}

#[test]
fn act_mul_f16_and_bf16_precision_paths() {
    let mut rng = SeededRng::new(0xA1_5021);
    let tol = Tolerance::f16_bf16();
    let (t, dff) = (2, 64);
    let num_elem = t * dff;

    let gate_data = generate_f32_data(&mut rng, num_elem, 2.5);
    let up_data = generate_f32_data(&mut rng, num_elem, 1.5);

    for &dt in &[DType::F16, DType::Bf16] {
        let op = ActMulOp {
            act: ActivationKind::Silu,
            clamp: None,
        };

        let mut gate_buf = TypedBuffer::zeros(&[t, dff], dt);
        let mut up_buf = TypedBuffer::zeros(&[t, dff], dt);
        let mut y_buf = TypedBuffer::zeros(&[t, dff], dt);

        for i in 0..num_elem {
            gate_buf.write_f32(i, gate_data[i]);
            up_buf.write_f32(i, up_data[i]);
        }

        act_mul(
            &op,
            &gate_buf.as_view(),
            &up_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let gate_f64: Vec<f64> = (0..num_elem).map(|i| gate_buf.read_f32(i) as f64).collect();
        let up_f64: Vec<f64> = (0..num_elem).map(|i| up_buf.read_f32(i) as f64).collect();
        let ref_f64 = act_mul_f64_reference(&op, &gate_f64, &up_f64);

        for i in 0..num_elem {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(actual, expected, &format!("act_mul_{dt} at {i}"));
        }
    }
}

#[test]
fn act_mul_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5022);
    let dff = 128;

    let target_gate = generate_f32_data(&mut rng, dff, 3.0);
    let target_up = generate_f32_data(&mut rng, dff, 2.0);

    let op = ActMulOp {
        act: ActivationKind::Silu,
        clamp: Some(3.5),
    };

    // 1. Alone (T = 1)
    let g_alone = TypedBuffer::from_f32(&[1, dff], &target_gate);
    let u_alone = TypedBuffer::from_f32(&[1, dff], &target_up);
    let mut y_alone = TypedBuffer::zeros(&[1, dff], DType::F32);
    act_mul(
        &op,
        &g_alone.as_view(),
        &u_alone.as_view(),
        &mut y_alone.as_view_mut(),
    )
    .unwrap();
    let out_alone = y_alone.to_f32_vec();

    // 2. Padded (T = 4)
    let mut g_pad = target_gate.clone();
    g_pad.extend(vec![0.0f32; 3 * dff]);
    let mut u_pad = target_up.clone();
    u_pad.extend(vec![0.0f32; 3 * dff]);
    let g_pad_buf = TypedBuffer::from_f32(&[4, dff], &g_pad);
    let u_pad_buf = TypedBuffer::from_f32(&[4, dff], &u_pad);
    let mut y_padded = TypedBuffer::zeros(&[4, dff], DType::F32);
    act_mul(
        &op,
        &g_pad_buf.as_view(),
        &u_pad_buf.as_view(),
        &mut y_padded.as_view_mut(),
    )
    .unwrap();
    let out_padded = &y_padded.to_f32_vec()[..dff];

    // 3. Embedded (T = 4, index 2)
    let other_g = generate_f32_data(&mut rng, 3 * dff, 4.0);
    let other_u = generate_f32_data(&mut rng, 3 * dff, 4.0);
    let mut g_emb = Vec::with_capacity(4 * dff);
    let mut u_emb = Vec::with_capacity(4 * dff);
    g_emb.extend_from_slice(&other_g[..2 * dff]);
    g_emb.extend_from_slice(&target_gate);
    g_emb.extend_from_slice(&other_g[2 * dff..]);
    u_emb.extend_from_slice(&other_u[..2 * dff]);
    u_emb.extend_from_slice(&target_up);
    u_emb.extend_from_slice(&other_u[2 * dff..]);

    let g_emb_buf = TypedBuffer::from_f32(&[4, dff], &g_emb);
    let u_emb_buf = TypedBuffer::from_f32(&[4, dff], &u_emb);
    let mut y_emb = TypedBuffer::zeros(&[4, dff], DType::F32);
    act_mul(
        &op,
        &g_emb_buf.as_view(),
        &u_emb_buf.as_view(),
        &mut y_emb.as_view_mut(),
    )
    .unwrap();
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
fn act_mul_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5023);
    let (t, dff) = (4, 128);
    let num_elem = t * dff;

    let op = ActMulOp {
        act: ActivationKind::GeluTanh,
        clamp: Some(5.0),
    };

    let gate_data = generate_f32_data(&mut rng, num_elem, 3.0);
    let up_data = generate_f32_data(&mut rng, num_elem, 3.0);

    let gate_buf = TypedBuffer::from_f32(&[t, dff], &gate_data);
    let up_buf = TypedBuffer::from_f32(&[t, dff], &up_data);

    let mut y1 = TypedBuffer::zeros(&[t, dff], DType::F32);
    let mut y2 = TypedBuffer::zeros(&[t, dff], DType::F32);

    act_mul(
        &op,
        &gate_buf.as_view(),
        &up_buf.as_view(),
        &mut y1.as_view_mut(),
    )
    .unwrap();
    act_mul(
        &op,
        &gate_buf.as_view(),
        &up_buf.as_view(),
        &mut y2.as_view_mut(),
    )
    .unwrap();

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
fn act_mul_rejects_malformed_operands() {
    let op = ActMulOp {
        act: ActivationKind::Silu,
        clamp: Some(-1.0), // invalid clamp
    };

    let gate_buf = TypedBuffer::zeros(&[2, 64], DType::F32);
    let up_buf = TypedBuffer::zeros(&[2, 128], DType::F32); // shape mismatch
    let mut y_buf = TypedBuffer::zeros(&[2, 64], DType::F32);

    let err = act_mul(
        &op,
        &gate_buf.as_view(),
        &up_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("clamp must be finite and > 0"));
    assert!(msg.contains("does not match"));
}
