// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 residual_add (Spec 1 §4.B, §6.1, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{DType, ResidualAddOp};
use r9v_t0::{residual_add, residual_add_f64_reference, Tolerance, TypedBuffer};

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
fn residual_add_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5010);
    let tol = Tolerance::f32();

    for &shape in &[&[16][..], &[4, 32][..], &[2, 8, 64][..]] {
        let num_elem: usize = shape.iter().product();
        let op = ResidualAddOp {
            out_dtype: DType::F32,
        };

        let a_data = generate_f32_data(&mut rng, num_elem, 10.0);
        let b_data = generate_f32_data(&mut rng, num_elem, 10.0);

        let a_buf = TypedBuffer::from_f32(shape, &a_data);
        let b_buf = TypedBuffer::from_f32(shape, &b_data);
        let mut y_buf = TypedBuffer::zeros(shape, DType::F32);

        residual_add(
            &op,
            &a_buf.as_view(),
            &b_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let a_f64: Vec<f64> = a_data.iter().map(|&v| v as f64).collect();
        let b_f64: Vec<f64> = b_data.iter().map(|&v| v as f64).collect();
        let ref_f64 = residual_add_f64_reference(&a_f64, &b_f64);

        for i in 0..num_elem {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(
                actual,
                expected,
                &format!("residual_add shape {shape:?} at {i}"),
            );
        }
    }
}

#[test]
fn residual_add_f16_and_bf16_precision_paths() {
    let mut rng = SeededRng::new(0xA1_5011);
    let tol = Tolerance::f16_bf16();
    let shape = [2, 64];
    let num_elem = 128;

    let a_data = generate_f32_data(&mut rng, num_elem, 5.0);
    let b_data = generate_f32_data(&mut rng, num_elem, 5.0);

    for &dt in &[DType::F16, DType::Bf16] {
        let op = ResidualAddOp { out_dtype: dt };

        let mut a_buf = TypedBuffer::zeros(&shape, dt);
        let mut b_buf = TypedBuffer::zeros(&shape, dt);
        let mut y_buf = TypedBuffer::zeros(&shape, dt);

        for i in 0..num_elem {
            a_buf.write_f32(i, a_data[i]);
            b_buf.write_f32(i, b_data[i]);
        }

        residual_add(
            &op,
            &a_buf.as_view(),
            &b_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let a_f64: Vec<f64> = (0..num_elem).map(|i| a_buf.read_f32(i) as f64).collect();
        let b_f64: Vec<f64> = (0..num_elem).map(|i| b_buf.read_f32(i) as f64).collect();
        let ref_f64 = residual_add_f64_reference(&a_f64, &b_f64);

        for i in 0..num_elem {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(actual, expected, &format!("residual_add_{dt} at {i}"));
        }
    }
}

#[test]
fn residual_add_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5012);
    let n = 64;

    let target_a = generate_f32_data(&mut rng, n, 4.0);
    let target_b = generate_f32_data(&mut rng, n, 4.0);

    let op = ResidualAddOp {
        out_dtype: DType::F32,
    };

    // 1. Alone (T = 1)
    let a_alone = TypedBuffer::from_f32(&[1, n], &target_a);
    let b_alone = TypedBuffer::from_f32(&[1, n], &target_b);
    let mut y_alone = TypedBuffer::zeros(&[1, n], DType::F32);
    residual_add(
        &op,
        &a_alone.as_view(),
        &b_alone.as_view(),
        &mut y_alone.as_view_mut(),
    )
    .unwrap();
    let out_alone = y_alone.to_f32_vec();

    // 2. Padded (T = 4)
    let mut a_padded = target_a.clone();
    a_padded.extend(vec![0.0f32; 3 * n]);
    let mut b_padded = target_b.clone();
    b_padded.extend(vec![0.0f32; 3 * n]);
    let a_pad_buf = TypedBuffer::from_f32(&[4, n], &a_padded);
    let b_pad_buf = TypedBuffer::from_f32(&[4, n], &b_padded);
    let mut y_padded = TypedBuffer::zeros(&[4, n], DType::F32);
    residual_add(
        &op,
        &a_pad_buf.as_view(),
        &b_pad_buf.as_view(),
        &mut y_padded.as_view_mut(),
    )
    .unwrap();
    let out_padded = &y_padded.to_f32_vec()[..n];

    // 3. Embedded (T = 4, index 2)
    let other_a = generate_f32_data(&mut rng, 3 * n, 5.0);
    let other_b = generate_f32_data(&mut rng, 3 * n, 5.0);
    let mut a_emb = Vec::with_capacity(4 * n);
    let mut b_emb = Vec::with_capacity(4 * n);
    a_emb.extend_from_slice(&other_a[..2 * n]);
    a_emb.extend_from_slice(&target_a);
    a_emb.extend_from_slice(&other_a[2 * n..]);
    b_emb.extend_from_slice(&other_b[..2 * n]);
    b_emb.extend_from_slice(&target_b);
    b_emb.extend_from_slice(&other_b[2 * n..]);

    let a_emb_buf = TypedBuffer::from_f32(&[4, n], &a_emb);
    let b_emb_buf = TypedBuffer::from_f32(&[4, n], &b_emb);
    let mut y_emb = TypedBuffer::zeros(&[4, n], DType::F32);
    residual_add(
        &op,
        &a_emb_buf.as_view(),
        &b_emb_buf.as_view(),
        &mut y_emb.as_view_mut(),
    )
    .unwrap();
    let out_emb = &y_emb.to_f32_vec()[2 * n..3 * n];

    for i in 0..n {
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
fn residual_add_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5013);
    let shape = [2, 128];
    let num_elem = 256;

    let op = ResidualAddOp {
        out_dtype: DType::F32,
    };

    let a_data = generate_f32_data(&mut rng, num_elem, 3.0);
    let b_data = generate_f32_data(&mut rng, num_elem, 3.0);

    let a_buf = TypedBuffer::from_f32(&shape, &a_data);
    let b_buf = TypedBuffer::from_f32(&shape, &b_data);
    let mut y1 = TypedBuffer::zeros(&shape, DType::F32);
    let mut y2 = TypedBuffer::zeros(&shape, DType::F32);

    residual_add(
        &op,
        &a_buf.as_view(),
        &b_buf.as_view(),
        &mut y1.as_view_mut(),
    )
    .unwrap();
    residual_add(
        &op,
        &a_buf.as_view(),
        &b_buf.as_view(),
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
fn residual_add_rejects_shape_and_dtype_mismatches() {
    let op = ResidualAddOp {
        out_dtype: DType::F32,
    };

    let a_buf = TypedBuffer::zeros(&[2, 64], DType::F32);
    let b_buf = TypedBuffer::zeros(&[2, 128], DType::F32); // Shape mismatch
    let mut y_buf = TypedBuffer::zeros(&[2, 32], DType::F32); // Shape mismatch

    let err = residual_add(
        &op,
        &a_buf.as_view(),
        &b_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("does not match"));
}
