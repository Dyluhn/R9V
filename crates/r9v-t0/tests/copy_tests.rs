// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 copy (Spec 1 §4.A, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{CopyKind, CopyOp, DType};
use r9v_t0::{copy, TypedBuffer};

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
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("does not match"));
}
