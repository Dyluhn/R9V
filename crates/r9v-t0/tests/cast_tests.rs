// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 cast (Spec 1 §4.A, §6.1, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{CastOp, DType};
use r9v_t0::{cast, cast_f64_reference, Tolerance, TypedBuffer};

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
fn cast_across_dtypes_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5050);
    let tol = Tolerance::f16_bf16();
    let shape = [4, 32];
    let num_elem = 128;

    let test_dtypes = [
        DType::F32,
        DType::F16,
        DType::Bf16,
        DType::I8,
        DType::I32,
        DType::U32,
        DType::Bool,
    ];

    let raw_f32 = generate_f32_data(&mut rng, num_elem, 10.0);

    for &src_dt in &test_dtypes {
        for &dst_dt in &test_dtypes {
            let op = CastOp { dtype: dst_dt };

            let mut src_buf = TypedBuffer::zeros(&shape, src_dt);
            for i in 0..num_elem {
                src_buf.write_f32(i, raw_f32[i]);
            }

            let mut dst_buf = TypedBuffer::zeros(&shape, dst_dt);
            cast(&op, &src_buf.as_view(), &mut dst_buf.as_view_mut()).unwrap();

            let ref_input: Vec<f64> = (0..num_elem).map(|i| src_buf.read_f32(i) as f64).collect();
            let ref_output = cast_f64_reference(&ref_input);

            for i in 0..num_elem {
                let actual = dst_buf.read_f32(i) as f64;
                let expected = ref_output[i];
                if dst_dt == DType::I8
                    || dst_dt == DType::I32
                    || dst_dt == DType::U32
                    || dst_dt == DType::Bool
                {
                    let rounded_expected = if dst_dt == DType::Bool {
                        if expected != 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    } else if dst_dt == DType::I8 {
                        expected.round_ties_even().clamp(-128.0, 127.0)
                    } else if dst_dt == DType::U32 {
                        expected.round_ties_even().max(0.0)
                    } else {
                        expected.round_ties_even()
                    };
                    assert_eq!(
                        actual, rounded_expected,
                        "integer cast {src_dt:?} -> {dst_dt:?} mismatch at {i}"
                    );
                } else {
                    tol.assert_within(
                        actual,
                        expected,
                        &format!("cast {src_dt:?} -> {dst_dt:?} at {i}"),
                    );
                }
            }
        }
    }
}

#[test]
fn cast_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5051);
    let n = 64;

    let target_token = generate_f32_data(&mut rng, n, 5.0);
    let op = CastOp { dtype: DType::F16 };

    // 1. Alone (T = 1)
    let x_alone = TypedBuffer::from_f32(&[1, n], &target_token);
    let mut y_alone = TypedBuffer::zeros(&[1, n], DType::F16);
    cast(&op, &x_alone.as_view(), &mut y_alone.as_view_mut()).unwrap();
    let out_alone = y_alone.to_f32_vec();

    // 2. Padded (T = 4)
    let mut x_pad = target_token.clone();
    x_pad.extend(vec![0.0f32; 3 * n]);
    let x_pad_buf = TypedBuffer::from_f32(&[4, n], &x_pad);
    let mut y_pad_buf = TypedBuffer::zeros(&[4, n], DType::F16);
    cast(&op, &x_pad_buf.as_view(), &mut y_pad_buf.as_view_mut()).unwrap();
    let out_pad = &y_pad_buf.to_f32_vec()[..n];

    // 3. Embedded (T = 4, index 2)
    let other_tokens = generate_f32_data(&mut rng, 3 * n, 5.0);
    let mut x_emb = Vec::with_capacity(4 * n);
    x_emb.extend_from_slice(&other_tokens[..2 * n]);
    x_emb.extend_from_slice(&target_token);
    x_emb.extend_from_slice(&other_tokens[2 * n..]);
    let x_emb_buf = TypedBuffer::from_f32(&[4, n], &x_emb);
    let mut y_emb_buf = TypedBuffer::zeros(&[4, n], DType::F16);
    cast(&op, &x_emb_buf.as_view(), &mut y_emb_buf.as_view_mut()).unwrap();
    let out_emb = &y_emb_buf.to_f32_vec()[2 * n..3 * n];

    for i in 0..n {
        assert_eq!(
            out_alone[i].to_bits(),
            out_pad[i].to_bits(),
            "alone vs padded at {i}"
        );
        assert_eq!(
            out_alone[i].to_bits(),
            out_emb[i].to_bits(),
            "alone vs embedded at {i}"
        );
    }
}

#[test]
fn cast_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5052);
    let shape = [4, 64];
    let num_elem = 256;

    let op = CastOp { dtype: DType::Bf16 };
    let x_data = generate_f32_data(&mut rng, num_elem, 8.0);
    let x_buf = TypedBuffer::from_f32(&shape, &x_data);

    let mut y1 = TypedBuffer::zeros(&shape, DType::Bf16);
    let mut y2 = TypedBuffer::zeros(&shape, DType::Bf16);

    cast(&op, &x_buf.as_view(), &mut y1.as_view_mut()).unwrap();
    cast(&op, &x_buf.as_view(), &mut y2.as_view_mut()).unwrap();

    let out1 = y1.to_f32_vec();
    let out2 = y2.to_f32_vec();
    for i in 0..num_elem {
        assert_eq!(
            out1[i].to_bits(),
            out2[i].to_bits(),
            "cast determinism failed at {i}"
        );
    }
}

#[test]
fn cast_rejects_shape_and_dtype_mismatches() {
    let op = CastOp { dtype: DType::F16 };
    let x_buf = TypedBuffer::zeros(&[2, 64], DType::F32);
    let mut y_buf = TypedBuffer::zeros(&[2, 32], DType::F32); // shape and dtype mismatch

    let err = cast(&op, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("does not match"));
}
