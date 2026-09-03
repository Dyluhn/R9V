// SPDX-License-Identifier: Apache-2.0
use r9v_ir::{DType, GatherRowsOp};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::f32_to_f16;
use r9v_t0::error::T0Error;
use r9v_t0::gather_rows::{gather_rows, gather_rows_f64_reference};

#[test]
fn test_gather_rows_matches_f64_reference_f32() {
    let n = 8;
    let d = 16;
    let m = 5;
    let indices_data = vec![7u32, 0, 3, 2, 5];

    let mut x_f64 = Vec::with_capacity(n * d);
    let mut x_f32 = Vec::with_capacity(n * d);
    for i in 0..(n * d) {
        let val = (i as f64 * 0.125) - 10.0;
        x_f64.push(val);
        x_f32.push(val as f32);
    }

    let expected_f64 = gather_rows_f64_reference(&x_f64, n, d, &indices_data);

    let x_buf = TypedBuffer::from_f32(&[n, d], &x_f32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices_data);
    let mut y_buf = TypedBuffer::zeros(&[m, d], DType::F32);

    let op = GatherRowsOp;
    gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let y_slice = y_buf.to_f32_vec();
    assert_eq!(y_slice.len(), m * d);
    for (actual, &expected) in y_slice.iter().zip(expected_f64.iter()) {
        assert_eq!(*actual, expected as f32);
    }
}

#[test]
fn test_gather_rows_matches_f64_reference_f16() {
    let n = 6;
    let d = 8;
    let m = 4;
    let indices_data = vec![5u32, 1, 4, 0];

    let mut x_f64 = Vec::with_capacity(n * d);
    let mut x_f16 = Vec::with_capacity(n * d);
    for i in 0..(n * d) {
        let val = (i as f64 * 0.25) - 5.0;
        x_f64.push(val);
        x_f16.push(f32_to_f16(val as f32));
    }

    let expected_f64 = gather_rows_f64_reference(&x_f64, n, d, &indices_data);

    let x_buf = TypedBuffer::from_f16(&[n, d], &x_f16);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices_data);
    let mut y_buf = TypedBuffer::zeros(&[m, d], DType::F16);

    let op = GatherRowsOp;
    gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let y_view = y_buf.as_view();
    let y_slice = y_view.as_f16_slice().unwrap();
    for (actual_bits, &expected) in y_slice.iter().zip(expected_f64.iter()) {
        let expected_bits = f32_to_f16(expected as f32);
        assert_eq!(*actual_bits, expected_bits);
    }
}

#[test]
fn test_gather_rows_batch_invariance() {
    let n = 10;
    let d = 32;
    let mut x_f32 = Vec::with_capacity(n * d);
    for i in 0..(n * d) {
        x_f32.push((i as f32 * 0.5) - 20.0);
    }
    let x_buf = TypedBuffer::from_f32(&[n, d], &x_f32);
    let op = GatherRowsOp;

    // Gather single row 4 alone
    let idx_single = TypedBuffer::from_u32(&[1], &[4u32]);
    let mut y_single = TypedBuffer::zeros(&[1, d], DType::F32);
    gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_single.as_view(),
        &mut y_single.as_view_mut(),
    )
    .unwrap();

    // Gather row 4 embedded among others: [1, 4, 7, 4, 9]
    let idx_multi = TypedBuffer::from_u32(&[5], &[1u32, 4, 7, 4, 9]);
    let mut y_multi = TypedBuffer::zeros(&[5, d], DType::F32);
    gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_multi.as_view(),
        &mut y_multi.as_view_mut(),
    )
    .unwrap();

    let single_slice = y_single.to_f32_vec();
    let multi_slice = y_multi.to_f32_vec();

    // Row 4 in y_multi is at index 1 and index 3
    assert_eq!(&single_slice[..], &multi_slice[d..2 * d]);
    assert_eq!(&single_slice[..], &multi_slice[3 * d..4 * d]);
}

#[test]
fn test_gather_rows_determinism_twice_bit_identical() {
    let n = 12;
    let d = 24;
    let m = 8;
    let indices = vec![11u32, 0, 5, 3, 8, 2, 7, 1];
    let x_f32: Vec<f32> = (0..(n * d)).map(|i| (i as f32 * 0.1) - 5.0).collect();

    let x_buf = TypedBuffer::from_f32(&[n, d], &x_f32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let mut y1 = TypedBuffer::zeros(&[m, d], DType::F32);
    let mut y2 = TypedBuffer::zeros(&[m, d], DType::F32);

    let op = GatherRowsOp;
    gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        &mut y1.as_view_mut(),
    )
    .unwrap();
    gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        &mut y2.as_view_mut(),
    )
    .unwrap();

    assert_eq!(y1.to_f32_vec(), y2.to_f32_vec());
}

#[test]
fn test_gather_rows_rejects_out_of_bounds_index() {
    let n = 5;
    let d = 8;
    let m = 3;
    let indices = vec![0u32, 5, 2]; // 5 is out of range 0..5

    let x_buf = TypedBuffer::zeros(&[n, d], DType::F32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let mut y_buf = TypedBuffer::zeros(&[m, d], DType::F32);

    let op = GatherRowsOp;
    let err = gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();

    match err {
        T0Error::RowIndexOutOfRange {
            index,
            upper_bound,
            position,
            ..
        } => {
            assert_eq!(index, 5);
            assert_eq!(upper_bound, 5);
            assert_eq!(position, 1);
        }
        other => panic!("expected RowIndexOutOfRange, got {other:?}"),
    }
}

#[test]
fn test_gather_rows_rejects_shape_and_dtype_mismatch() {
    let x_buf = TypedBuffer::zeros(&[5, 8], DType::F32);
    let idx_buf = TypedBuffer::zeros(&[3], DType::I32); // Wrong dtype: I32 instead of U32
    let mut y_buf = TypedBuffer::zeros(&[3, 8], DType::F32);

    let op = GatherRowsOp;
    assert!(gather_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        &mut y_buf.as_view_mut()
    )
    .is_err());
}
