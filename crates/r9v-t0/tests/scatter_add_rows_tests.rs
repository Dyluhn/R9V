// SPDX-License-Identifier: Apache-2.0
use r9v_ir::{DType, ScatterAddRowsOp};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::f32_to_f16;
use r9v_t0::error::T0Error;
use r9v_t0::scatter_add_rows::{scatter_add_rows, scatter_add_rows_f64_reference};

#[test]
fn test_scatter_add_rows_matches_f64_reference_without_dest() {
    let m = 7;
    let d = 8;
    let n = 4;
    let indices = vec![2u32, 0, 1, 2, 0, 3, 2];

    let mut x_f64 = Vec::with_capacity(m * d);
    let mut x_f32 = Vec::with_capacity(m * d);
    for i in 0..(m * d) {
        let val = (i as f64 * 0.125) - 2.0;
        x_f64.push(val);
        x_f32.push(val as f32);
    }

    let expected_f64 = scatter_add_rows_f64_reference(&x_f64, m, d, &indices, None, n);

    let x_buf = TypedBuffer::from_f32(&[m, d], &x_f32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let mut y_buf = TypedBuffer::zeros(&[n, d], DType::F32);

    let op = ScatterAddRowsOp;
    scatter_add_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let y_slice = y_buf.to_f32_vec();
    for (actual, &expected) in y_slice.iter().zip(expected_f64.iter()) {
        assert!((actual - expected as f32).abs() < 1e-6);
    }
}

#[test]
fn test_scatter_add_rows_matches_f64_reference_with_dest() {
    let m = 6;
    let d = 4;
    let n = 3;
    let indices = vec![1u32, 0, 2, 1, 2, 0];

    let mut x_f64 = Vec::with_capacity(m * d);
    let mut x_f32 = Vec::with_capacity(m * d);
    for i in 0..(m * d) {
        let val = (i as f64 * 0.5) - 1.0;
        x_f64.push(val);
        x_f32.push(val as f32);
    }

    let dest_f64 = vec![
        10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    ];
    let dest_f32: Vec<f32> = dest_f64.iter().map(|&v| v as f32).collect();

    let expected_f64 = scatter_add_rows_f64_reference(&x_f64, m, d, &indices, Some(&dest_f64), n);

    let x_buf = TypedBuffer::from_f32(&[m, d], &x_f32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let dest_buf = TypedBuffer::from_f32(&[n, d], &dest_f32);
    let mut y_buf = TypedBuffer::zeros(&[n, d], DType::F32);

    let op = ScatterAddRowsOp;
    scatter_add_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        Some(&dest_buf.as_view()),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let y_slice = y_buf.to_f32_vec();
    for (actual, &expected) in y_slice.iter().zip(expected_f64.iter()) {
        assert!((actual - expected as f32).abs() < 1e-6);
    }
}

#[test]
fn test_scatter_add_rows_accumulation_order_ascending_ties() {
    // Test that ties accumulate in strictly ascending source row index order
    // Non-associativity test with values where 1.0 is below the ULP of 1e8:
    let m = 4;
    let d = 1;
    let n = 1;
    let indices = vec![0u32, 0, 0, 0];

    // Non-associative floating point values in f32:
    // With 1e8: ULP is 8.0, so 1e8 + 1.0 == 1e8 in f32.
    // In ascending order: ((1e8 + 1.0) + 1.0) - 1e8 == 0.0.
    let x_f32 = vec![1e8f32, 1.0f32, 1.0f32, -1e8f32];
    let x_buf = TypedBuffer::from_f32(&[m, d], &x_f32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let mut y_buf = TypedBuffer::zeros(&[n, d], DType::F32);

    let op = ScatterAddRowsOp;
    scatter_add_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let y_slice = y_buf.to_f32_vec();
    assert_eq!(y_slice[0], 0.0f32);
}

#[test]
fn test_scatter_add_rows_determinism_twice_bit_identical() {
    let m = 8;
    let d = 16;
    let n = 4;
    let indices = vec![3u32, 1, 0, 2, 1, 3, 0, 2];
    let x_f32: Vec<f32> = (0..(m * d)).map(|i| (i as f32 * 0.25) - 3.0).collect();

    let x_buf = TypedBuffer::from_f32(&[m, d], &x_f32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let mut y1 = TypedBuffer::zeros(&[n, d], DType::F32);
    let mut y2 = TypedBuffer::zeros(&[n, d], DType::F32);

    let op = ScatterAddRowsOp;
    scatter_add_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        None,
        &mut y1.as_view_mut(),
    )
    .unwrap();
    scatter_add_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        None,
        &mut y2.as_view_mut(),
    )
    .unwrap();

    assert_eq!(y1.to_f32_vec(), y2.to_f32_vec());
}

#[test]
fn test_scatter_add_rows_f16() {
    let m = 3;
    let d = 4;
    let n = 2;
    let indices = vec![1u32, 0, 1];
    let x_vals = vec![
        1.5f32, -0.5, 2.0, 3.5, 4.0, 1.0, -1.0, 0.5, 0.5, 1.5, -2.0, -3.0,
    ];
    let x_f16: Vec<u16> = x_vals.iter().map(|&v| f32_to_f16(v)).collect();

    let x_buf = TypedBuffer::from_f16(&[m, d], &x_f16);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let mut y_buf = TypedBuffer::zeros(&[n, d], DType::F16);

    let op = ScatterAddRowsOp;
    scatter_add_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        None,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let y_view = y_buf.as_view();
    let y_slice = y_view.as_f16_slice().unwrap();

    // Row 0: source 1 => [4.0, 1.0, -1.0, 0.5]
    assert_eq!(y_slice[0], f32_to_f16(4.0));
    assert_eq!(y_slice[1], f32_to_f16(1.0));
    assert_eq!(y_slice[2], f32_to_f16(-1.0));
    assert_eq!(y_slice[3], f32_to_f16(0.5));

    // Row 1: source 0 + source 2 => [1.5 + 0.5, -0.5 + 1.5, 2.0 - 2.0, 3.5 - 3.0] = [2.0, 1.0, 0.0, 0.5]
    assert_eq!(y_slice[4], f32_to_f16(2.0));
    assert_eq!(y_slice[5], f32_to_f16(1.0));
    assert_eq!(y_slice[6], f32_to_f16(0.0));
    assert_eq!(y_slice[7], f32_to_f16(0.5));
}

#[test]
fn test_scatter_add_rows_rejects_out_of_bounds_index() {
    let m = 3;
    let d = 4;
    let n = 2; // Valid indices are 0..2
    let indices = vec![0u32, 2, 1]; // 2 is >= n

    let x_buf = TypedBuffer::zeros(&[m, d], DType::F32);
    let idx_buf = TypedBuffer::from_u32(&[m], &indices);
    let mut y_buf = TypedBuffer::zeros(&[n, d], DType::F32);

    let op = ScatterAddRowsOp;
    let err = scatter_add_rows(
        &op,
        &x_buf.as_view(),
        &idx_buf.as_view(),
        None,
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
            assert_eq!(index, 2);
            assert_eq!(upper_bound, 2);
            assert_eq!(position, 1);
        }
        other => panic!("expected RowIndexOutOfRange, got {other:?}"),
    }
}
