// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 cast (Spec 1 §4.A, §6.1, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{CastOp, DType};
use r9v_t0::{bf16_to_f32, cast, cast_f64_reference, Tolerance, TypedBuffer};

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

/// Identity casts preserve `u32` values beyond 2^24 exactly (Spec 1 §2.1, Spec 1 §6.4).
///
/// `f32` cannot represent every `u32` above 2^24, so a round-trip through `f32` corrupts
/// values like `16_777_217`; the identity path must move raw storage instead.
#[test]
fn cast_identity_u32_beyond_f32_precision_is_bit_exact() {
    let values: Vec<u32> = vec![
        0,
        1,
        16_777_215,
        16_777_216,
        16_777_217,
        123_456_789,
        3_000_000_000,
        u32::MAX - 1,
        u32::MAX,
    ];
    let shape = [1, values.len()];
    let x_buf = TypedBuffer::from_u32(&shape, &values);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::U32);
    cast(
        &CastOp { dtype: DType::U32 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.to_u32_vec(), values);
}

/// Identity casts preserve `i32` magnitudes beyond 2^24 exactly (Spec 1 §2.1, Spec 1 §6.4).
#[test]
fn cast_identity_i32_large_magnitudes_is_bit_exact() {
    let values: Vec<i32> = vec![
        0,
        1,
        -1,
        16_777_217,
        -16_777_217,
        123_456_789,
        -123_456_789,
        i32::MAX,
        i32::MIN,
    ];
    let shape = [1, values.len()];
    let x_buf = TypedBuffer::from_i32(&shape, &values);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::I32);
    cast(
        &CastOp { dtype: DType::I32 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.to_i32_vec(), values);
}

/// Identity casts preserve E4M3 bytes including both NaN payloads (Spec 1 §2.1).
///
/// `0x7F` and `0xFF` both decode to NaN, so a value round-trip normalizes `0xFF` to `0x7F`;
/// the identity path must keep every byte unchanged.
#[test]
fn cast_identity_e4m3_preserves_nan_payloads() {
    let bytes: Vec<u8> = vec![0x00, 0x80, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0xFF, 0x01, 0x07];
    let shape = [1, bytes.len()];
    let x_buf = TypedBuffer::from_bytes(&shape, DType::E4m3, &bytes);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::E4m3);
    cast(
        &CastOp { dtype: DType::E4m3 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.to_byte_vec(), bytes);
}

/// Identity casts preserve E5M2 bytes including infinities and all NaN payloads (Spec 1 §2.1).
#[test]
fn cast_identity_e5m2_preserves_nan_payloads_and_infinities() {
    let bytes: Vec<u8> = vec![
        0x00, 0x80, 0x3C, 0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF, 0x01,
    ];
    let shape = [1, bytes.len()];
    let x_buf = TypedBuffer::from_bytes(&shape, DType::E5m2, &bytes);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::E5m2);
    cast(
        &CastOp { dtype: DType::E5m2 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.to_byte_vec(), bytes);
}

/// Identity casts preserve packed I4 nibbles for even and odd element counts (Spec 1 §2.1).
#[test]
fn cast_identity_i4_packed_nibbles_are_bit_exact() {
    for values in [
        vec![-8, -5, -1, 0, 1, 4, 7, -7],
        vec![-8, -1, 0, 1, 7, -7, 3],
    ] {
        let shape = [1, values.len()];
        let mut x_buf = TypedBuffer::zeros(&shape, DType::I4);
        for (i, &v) in values.iter().enumerate() {
            x_buf.write_f32(i, v as f32);
        }
        let expected = x_buf.to_byte_vec();
        let mut y_buf = TypedBuffer::zeros(&shape, DType::I4);
        cast(
            &CastOp { dtype: DType::I4 },
            &x_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();
        assert_eq!(y_buf.to_byte_vec(), expected);
    }
}

/// Identity casts preserve `f16` bits including NaN payloads (Spec 1 §2.1).
///
/// Compared as whole buffers: `f16` decode folds NaN payload bit 9, so comparing
/// decoded values would not catch a payload change.
#[test]
fn cast_identity_f16_preserves_bits_including_nan_payloads() {
    let bits: Vec<u16> = vec![
        0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7C01, 0x7E00, 0x7FFF, 0x0400,
    ];
    let shape = [1, bits.len()];
    let x_buf = TypedBuffer::from_f16(&shape, &bits);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::F16);
    cast(
        &CastOp { dtype: DType::F16 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf, x_buf);
}

/// Identity casts preserve `bf16`, `f32`, `i8`, and `bool` storage exactly (Spec 1 §2.1).
///
/// `bool` bytes beyond `0`/`1` are included: a value conversion would normalize them to `1`.
#[test]
fn cast_identity_scalar_dtypes_are_bit_exact() {
    let f32_vals: Vec<f32> = vec![0.0, -0.0, 1.5, f32::from_bits(0x7FC0_0001), f32::INFINITY];
    let shape = [1, f32_vals.len()];
    let x_buf = TypedBuffer::from_f32(&shape, &f32_vals);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::F32);
    cast(
        &CastOp { dtype: DType::F32 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let out = y_buf.to_f32_vec();
    for (i, &v) in f32_vals.iter().enumerate() {
        assert_eq!(
            out[i].to_bits(),
            v.to_bits(),
            "f32 identity changed bits at {i}"
        );
    }

    let bf16_bits: Vec<u16> = vec![0x0000, 0x3F80, 0x7F80, 0x7FC1, 0xFF80];
    let shape = [1, bf16_bits.len()];
    let x_buf = TypedBuffer::from_bf16(&shape, &bf16_bits);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::Bf16);
    cast(
        &CastOp { dtype: DType::Bf16 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let out = y_buf.to_f32_vec();
    for (i, &b) in bf16_bits.iter().enumerate() {
        assert_eq!(
            out[i].to_bits(),
            bf16_to_f32(b).to_bits(),
            "bf16 identity changed bits at {i}"
        );
    }

    let i8_vals: Vec<i8> = vec![-128, -1, 0, 1, 127];
    let shape = [1, i8_vals.len()];
    let x_buf = TypedBuffer::from_i8(&shape, &i8_vals);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::I8);
    cast(
        &CastOp { dtype: DType::I8 },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.to_i8_vec(), i8_vals);

    let bool_bytes: Vec<u8> = vec![0, 1, 2, 255];
    let shape = [1, bool_bytes.len()];
    let x_buf = TypedBuffer::from_bytes(&shape, DType::Bool, &bool_bytes);
    let mut y_buf = TypedBuffer::zeros(&shape, DType::Bool);
    cast(
        &CastOp { dtype: DType::Bool },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y_buf.to_byte_vec(), bool_bytes);
}

/// Identity casts keep typed validation: mismatched shapes are still refused (Spec 1 §4.A).
#[test]
fn cast_identity_rejects_shape_mismatch() {
    let op = CastOp { dtype: DType::U32 };
    let x_buf = TypedBuffer::zeros(&[2, 64], DType::U32);
    let mut y_buf = TypedBuffer::zeros(&[2, 32], DType::U32);
    let err = cast(&op, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap_err();
    assert!(err.to_string().contains("does not match"));
}
