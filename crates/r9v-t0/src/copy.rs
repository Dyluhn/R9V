// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of memory copy and contiguization op (Spec 1 §4.A, Spec 4 §2).

use r9v_ir::{CopyOp, DType};

use crate::buffer::{TensorData, TensorDataMut, TensorView, TensorViewMut};
use crate::dtype::dtype_element_size;
use crate::error::T0Error;

/// Executes scalar T0 tensor copy / contiguization (Spec 1 §4.A, Spec 4 §2).
///
/// Performs bit-exact copying through raw typed storage without floating point conversions.
pub fn copy(_op: &CopyOp, x: &TensorView<'_>, y: &mut TensorViewMut<'_>) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    if x.shape() != y.shape() {
        problems.push(format!(
            "output y shape {:?} does not match input x shape {:?}",
            y.shape(),
            x.shape()
        ));
    }
    if y.dtype() != x.dtype() {
        problems.push(format!(
            "output y dtype {:?} does not match input x dtype {:?}",
            y.dtype(),
            x.dtype()
        ));
    }

    T0Error::from_problems("copy", problems)?;

    let num_elem = x.num_elements();
    match (&x.data, &mut y.data) {
        (TensorData::F32(src), TensorDataMut::F32(dst)) => {
            dst[..num_elem].copy_from_slice(&src[..num_elem]);
        }
        (TensorData::F16(src), TensorDataMut::F16(dst)) => {
            dst[..num_elem].copy_from_slice(&src[..num_elem]);
        }
        (TensorData::Bf16(src), TensorDataMut::Bf16(dst)) => {
            dst[..num_elem].copy_from_slice(&src[..num_elem]);
        }
        (TensorData::I8(src), TensorDataMut::I8(dst)) => {
            dst[..num_elem].copy_from_slice(&src[..num_elem]);
        }
        (TensorData::U32(src), TensorDataMut::U32(dst)) => {
            dst[..num_elem].copy_from_slice(&src[..num_elem]);
        }
        (TensorData::Bytes(dtype_x, src), TensorDataMut::Bytes(_dtype_y, dst)) => {
            let byte_count = if *dtype_x == DType::I4 {
                num_elem / 2 + (num_elem % 2)
            } else {
                num_elem * dtype_element_size(*dtype_x)
            };
            dst[..byte_count].copy_from_slice(&src[..byte_count]);
        }
        (TensorData::U32(src), TensorDataMut::Bytes(_, dst)) => {
            for (i, &val) in src[..num_elem].iter().enumerate() {
                dst[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        (TensorData::Bytes(_, src), TensorDataMut::U32(dst)) => {
            for (i, item) in dst[..num_elem].iter_mut().enumerate() {
                *item = u32::from_le_bytes(src[i * 4..(i + 1) * 4].try_into().unwrap());
            }
        }
        (TensorData::F32(src), TensorDataMut::Bytes(_, dst)) => {
            for (i, &val) in src[..num_elem].iter().enumerate() {
                dst[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        (TensorData::Bytes(_, src), TensorDataMut::F32(dst)) => {
            for (i, item) in dst[..num_elem].iter_mut().enumerate() {
                *item = f32::from_le_bytes(src[i * 4..(i + 1) * 4].try_into().unwrap());
            }
        }
        (TensorData::F16(src), TensorDataMut::Bytes(_, dst)) => {
            for (i, &val) in src[..num_elem].iter().enumerate() {
                dst[i * 2..(i + 1) * 2].copy_from_slice(&val.to_le_bytes());
            }
        }
        (TensorData::Bytes(_, src), TensorDataMut::F16(dst)) => {
            for (i, item) in dst[..num_elem].iter_mut().enumerate() {
                *item = u16::from_le_bytes(src[i * 2..(i + 1) * 2].try_into().unwrap());
            }
        }
        (TensorData::Bf16(src), TensorDataMut::Bytes(_, dst)) => {
            for (i, &val) in src[..num_elem].iter().enumerate() {
                dst[i * 2..(i + 1) * 2].copy_from_slice(&val.to_le_bytes());
            }
        }
        (TensorData::Bytes(_, src), TensorDataMut::Bf16(dst)) => {
            for (i, item) in dst[..num_elem].iter_mut().enumerate() {
                *item = u16::from_le_bytes(src[i * 2..(i + 1) * 2].try_into().unwrap());
            }
        }
        (TensorData::I8(src), TensorDataMut::Bytes(_, dst)) => {
            for (i, &val) in src[..num_elem].iter().enumerate() {
                dst[i] = val as u8;
            }
        }
        (TensorData::Bytes(_, src), TensorDataMut::I8(dst)) => {
            for (i, item) in dst[..num_elem].iter_mut().enumerate() {
                *item = src[i] as i8;
            }
        }
        _ => {
            return Err(T0Error::BackingRepresentationMismatch {
                op: "copy",
                dtype: x.dtype(),
            });
        }
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (Spec 1 §4.A, Spec 4 §2).
pub fn copy_f64_reference(_op: &CopyOp, x: &[f64]) -> Vec<f64> {
    x.to_vec()
}
