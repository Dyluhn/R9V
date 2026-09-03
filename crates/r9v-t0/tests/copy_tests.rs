// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 copy (Spec 1 §4.A, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{CopyKind, CopyOp, DType};
use r9v_t0::{copy, copy_f64_reference, T0Error, TypedBuffer};

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
fn copy_preserves_elements_exactly() {
    let mut rng = SeededRng::new(0xA1_5060);
    let shape = [2, 4, 32];
    let num_elem = 256;

    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };
    let data = generate_f32_data(&mut rng, num_elem, 15.0);

    let src_buf = TypedBuffer::from_f32(&shape, &data);
    let mut dst_buf = TypedBuffer::zeros(&shape, DType::F32);

    copy(&op, &src_buf.as_view(), &mut dst_buf.as_view_mut()).unwrap();

    let out = dst_buf.to_f32_vec();
    for i in 0..num_elem {
        assert_eq!(
            data[i].to_bits(),
            out[i].to_bits(),
            "copy mismatch at element {i}"
        );
    }
}

#[test]
fn copy_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5061);
    let n = 64;

    let target_token = generate_f32_data(&mut rng, n, 4.0);
    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };

    // 1. Alone (T = 1)
    let x_alone = TypedBuffer::from_f32(&[1, n], &target_token);
    let mut y_alone = TypedBuffer::zeros(&[1, n], DType::F32);
    copy(&op, &x_alone.as_view(), &mut y_alone.as_view_mut()).unwrap();
    let out_alone = y_alone.to_f32_vec();

    // 2. Padded (T = 4)
    let mut x_pad = target_token.clone();
    x_pad.extend(vec![0.0f32; 3 * n]);
    let x_pad_buf = TypedBuffer::from_f32(&[4, n], &x_pad);
    let mut y_pad_buf = TypedBuffer::zeros(&[4, n], DType::F32);
    copy(&op, &x_pad_buf.as_view(), &mut y_pad_buf.as_view_mut()).unwrap();
    let out_pad = &y_pad_buf.to_f32_vec()[..n];

    // 3. Embedded (T = 4, index 2)
    let other_tokens = generate_f32_data(&mut rng, 3 * n, 5.0);
    let mut x_emb = Vec::with_capacity(4 * n);
    x_emb.extend_from_slice(&other_tokens[..2 * n]);
    x_emb.extend_from_slice(&target_token);
    x_emb.extend_from_slice(&other_tokens[2 * n..]);
    let x_emb_buf = TypedBuffer::from_f32(&[4, n], &x_emb);
    let mut y_emb_buf = TypedBuffer::zeros(&[4, n], DType::F32);
    copy(&op, &x_emb_buf.as_view(), &mut y_emb_buf.as_view_mut()).unwrap();
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
fn copy_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5062);
    let shape = [2, 128];
    let num_elem = 256;

    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };
    let data = generate_f32_data(&mut rng, num_elem, 7.0);
    let src_buf = TypedBuffer::from_f32(&shape, &data);

    let mut y1 = TypedBuffer::zeros(&shape, DType::F32);
    let mut y2 = TypedBuffer::zeros(&shape, DType::F32);

    copy(&op, &src_buf.as_view(), &mut y1.as_view_mut()).unwrap();
    copy(&op, &src_buf.as_view(), &mut y2.as_view_mut()).unwrap();

    let out1 = y1.to_f32_vec();
    let out2 = y2.to_f32_vec();
    for i in 0..num_elem {
        assert_eq!(
            out1[i].to_bits(),
            out2[i].to_bits(),
            "copy determinism failed at {i}"
        );
    }
}

#[test]
fn copy_rejects_shape_and_dtype_mismatches() {
    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };
    let src = TypedBuffer::zeros(&[2, 64], DType::F32);
    let mut dst = TypedBuffer::zeros(&[2, 32], DType::F32); // shape mismatch

    let err = copy(&op, &src.as_view(), &mut dst.as_view_mut()).unwrap_err();
    assert!(
        matches!(
            err,
            T0Error::DimensionMismatch {
                tensor: "y",
                expected: 64,
                got: 32,
                ..
            }
        ),
        "expected y dimension mismatch, got {err:?}"
    );
}

#[test]
fn copy_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5063);
    let shape = [4, 64];
    let num_elem = 256;
    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };
    let data_f32 = generate_f32_data(&mut rng, num_elem, 25.0);
    let data_f64: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let expected_f64 = copy_f64_reference(&op, &data_f64);

    let src = TypedBuffer::from_f32(&shape, &data_f32);
    let mut dst = TypedBuffer::zeros(&shape, DType::F32);
    copy(&op, &src.as_view(), &mut dst.as_view_mut()).unwrap();

    let out_f32 = dst.to_f32_vec();
    for i in 0..num_elem {
        assert_eq!(out_f32[i] as f64, expected_f64[i]);
    }
}

#[test]
fn copy_bit_exact_u32_and_i32() {
    let shape = [4, 8];
    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };

    // Test U32 bit-exactness with numbers that cannot be represented accurately in f32
    let u32_values: Vec<u32> = vec![
        0xDEAD_BEEF,
        0x8000_0001,
        0xFFFF_FFFF,
        0x1234_5678,
        0x7FFF_FFFF,
        0x0000_0000,
        1,
        42,
        0xABCD_EF01,
        0x1000_0001,
        0x00FF_00FF,
        0xFF00_FF00,
        0x5555_5555,
        0xAAAA_AAAA,
        0x1234_4321,
        0xFEDC_BA98,
        0x0001_0000,
        0x0000_0001,
        0x8000_0000,
        0x7FFF_0000,
        0x0000_7FFF,
        0xCAFE_BABE,
        0x0102_0304,
        0x0506_0708,
        0x090A_0B0C,
        0x0D0E_0F10,
        0x1112_1314,
        0x1516_1718,
        0x191A_1B1C,
        0x1D1E_1F20,
        0x2122_2324,
        0x2526_2728,
    ];
    let u32_src = TypedBuffer::from_u32(&shape, &u32_values);
    let mut u32_dst = TypedBuffer::zeros(&shape, DType::U32);
    copy(&op, &u32_src.as_view(), &mut u32_dst.as_view_mut()).unwrap();
    assert_eq!(u32_dst.to_u32_vec(), u32_values);

    // Test I32 bit-exactness
    let i32_values: Vec<i32> = vec![
        i32::MIN,
        i32::MAX,
        -1,
        0,
        1,
        123456789,
        -123456789,
        0x7FFF_FF01,
        -0x7FFF_FF01,
        100,
        -100,
        2147483640,
        -2147483640,
        42,
        -42,
        1337,
        -1337,
        7777777,
        -7777777,
        1010101,
        -1010101,
        999999999,
        -999999999,
        2,
        -2,
        3,
        -3,
        4,
        -4,
        5,
        -5,
        6,
    ];
    let i32_src = TypedBuffer::from_i32(&shape, &i32_values);
    let mut i32_dst = TypedBuffer::zeros(&shape, DType::I32);
    copy(&op, &i32_src.as_view(), &mut i32_dst.as_view_mut()).unwrap();
    assert_eq!(i32_dst.to_i32_vec(), i32_values);
}

#[test]
fn copy_bit_exact_fp8_nans_and_packed_i4() {
    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };

    // FP8 E4M3 NaNs: 0x7F and 0xFF
    let fp8_bytes = vec![0x7Fu8, 0xFF, 0x00, 0x80, 0x7E, 0xFE, 0x3C, 0xBC];
    let shape = [2, 4];
    let fp8_src = TypedBuffer::from_e4m3_bytes(&shape, &fp8_bytes);
    let mut fp8_dst = TypedBuffer::zeros(&shape, DType::E4m3);
    copy(&op, &fp8_src.as_view(), &mut fp8_dst.as_view_mut()).unwrap();
    assert_eq!(fp8_dst.to_byte_vec(), fp8_bytes);

    // Packed I4
    let i4_shape = [2, 8]; // 16 elements = 8 bytes
    let i4_bytes = vec![0xABu8, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89];
    let i4_src = TypedBuffer::from_bytes(&i4_shape, DType::I4, &i4_bytes);
    let mut i4_dst = TypedBuffer::zeros(&i4_shape, DType::I4);
    copy(&op, &i4_src.as_view(), &mut i4_dst.as_view_mut()).unwrap();
    assert_eq!(i4_dst.byte_data(), &i4_bytes[..]);
}
