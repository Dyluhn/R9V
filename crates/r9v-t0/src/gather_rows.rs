// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of `gather_rows` op (Spec 1 §4.A, §6.1, Card A1.6).

use r9v_ir::{DType, GatherRowsOp};

use crate::buffer::{TensorData, TensorDataMut, TensorView, TensorViewMut};
use crate::error::T0Error;

/// Gathers rows from input `x` according to integer `indices` into output `y` (Spec 1 §4.A, Card A1.6).
///
/// Signature:
/// - `x`: `[N, D]` (`f16`, `bf16`, or `f32`)
/// - `indices`: `[M]` (`u32`)
/// - `y`: `[M, D]` (dtype matches `x.dtype()`)
pub fn gather_rows(
    _op: &GatherRowsOp,
    x: &TensorView<'_>,
    indices: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    indices.validate_backing("indices")?;
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    if x.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "x",
            expected: 2,
            got: x.rank(),
            shape: x.shape().to_vec(),
        });
    }

    if indices.rank() != 1 {
        problems.push(T0Error::RankMismatch {
            tensor: "indices",
            expected: 1,
            got: indices.rank(),
            shape: indices.shape().to_vec(),
        });
    }

    if y.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "y",
            expected: 2,
            got: y.rank(),
            shape: y.shape().to_vec(),
        });
    }

    if !matches!(x.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: x.dtype(),
        });
    }

    if indices.dtype() != DType::U32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "indices",
            expected: vec![DType::U32],
            got: indices.dtype(),
        });
    }

    if y.dtype() != x.dtype() {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![x.dtype()],
            got: y.dtype(),
        });
    }

    T0Error::from_typed_problems(problems)?;

    let n = x.shape()[0];
    let d = x.shape()[1];
    let m = indices.shape()[0];

    let mut problems = Vec::new();

    if y.shape()[0] != m {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "M",
            expected_from: "indices",
            expected: m,
            tensor: "y",
            got: y.shape()[0],
        });
    }

    if y.shape()[1] != d {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "D",
            expected_from: "x",
            expected: d,
            tensor: "y",
            got: y.shape()[1],
        });
    }

    // Bounds check every index before modifying output
    for pos in 0..m {
        let idx = indices.read_u32(pos);
        if idx as usize >= n {
            problems.push(T0Error::RowIndexOutOfRange {
                op: "gather_rows",
                tensor: "indices",
                position: pos,
                index: idx,
                upper_bound: n,
            });
        }
    }

    T0Error::from_typed_problems(problems)?;

    // Execute bit-exact copy for each row
    for row_out in 0..m {
        let src_row = indices.read_u32(row_out) as usize;
        match (&x.data, &mut y.data) {
            (TensorData::F32(x_slice), TensorDataMut::F32(y_slice)) => {
                let src_start = src_row * d;
                let dst_start = row_out * d;
                y_slice[dst_start..dst_start + d]
                    .copy_from_slice(&x_slice[src_start..src_start + d]);
            }
            (TensorData::F16(x_slice), TensorDataMut::F16(y_slice))
            | (TensorData::Bf16(x_slice), TensorDataMut::Bf16(y_slice)) => {
                let src_start = src_row * d;
                let dst_start = row_out * d;
                y_slice[dst_start..dst_start + d]
                    .copy_from_slice(&x_slice[src_start..src_start + d]);
            }
            (TensorData::Bytes(_, x_slice), TensorDataMut::Bytes(_, y_slice)) => {
                let elem_bytes = crate::dtype::dtype_element_size(x.dtype());
                let row_bytes = d * elem_bytes;
                let src_start = src_row * row_bytes;
                let dst_start = row_out * row_bytes;
                y_slice[dst_start..dst_start + row_bytes]
                    .copy_from_slice(&x_slice[src_start..src_start + row_bytes]);
            }
            _ => {
                for col in 0..d {
                    let val = x.read_f32(src_row * d + col);
                    y.write_f32(row_out * d + col, val);
                }
            }
        }
    }

    Ok(())
}

/// 64-bit reference implementation of `gather_rows` for testing (Spec 1 §4.A, §6.1).
pub fn gather_rows_f64_reference(x: &[f64], n: usize, d: usize, indices: &[u32]) -> Vec<f64> {
    assert_eq!(x.len(), n * d);
    let m = indices.len();
    let mut y = Vec::with_capacity(m * d);
    for &idx in indices {
        let r = idx as usize;
        assert!(r < n, "index {r} out of bounds 0..{n}");
        let row = &x[r * d..(r + 1) * d];
        y.extend_from_slice(row);
    }
    y
}
