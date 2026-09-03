// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of elementwise residual addition (Spec 1 §4.B, §6.1, Spec 4 §2).

use r9v_ir::{DType, ResidualAddOp};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::{push_shape_agreement, T0Error};

/// Executes scalar T0 residual addition: `a + scale * b` in f32, cast to `out_dtype` (Spec 1 §4.B, Spec 1 §6.1, Spec 1 §6.4, Spec 4 §2; card A1.14, SI-27).
pub fn residual_add(
    op: &ResidualAddOp,
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    a.validate_backing("a")?;
    b.validate_backing("b")?;
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    if !op.scale.is_finite() || op.scale == 0.0 {
        problems.push(T0Error::InvalidAttribute {
            op: "residual_add",
            attribute: "scale",
            reason: format!("must be finite and non-zero, got {}", op.scale),
        });
    }
    if a.shape() != b.shape() {
        push_shape_agreement(&mut problems, "b", "a", b.shape(), a.shape());
    }
    if a.shape() != y.shape() {
        push_shape_agreement(&mut problems, "y", "a", y.shape(), a.shape());
    }
    if !matches!(a.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "a",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: a.dtype(),
        });
    }
    if !matches!(b.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "b",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: b.dtype(),
        });
    }
    if y.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.out_dtype],
            got: y.dtype(),
        });
    }

    T0Error::from_problems(problems)?;

    let num_elem = a.num_elements();
    for i in 0..num_elem {
        let val_a = a.read_f32(i);
        let val_b = b.read_f32(i);
        let sum = val_a + op.scale * val_b;
        y.write_f32(i, sum);
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (Spec 1 §4.B, Spec 4 §2; card A1.14, SI-27).
pub fn residual_add_f64_reference(a: &[f64], b: &[f64], scale: f64) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x + scale * y)
        .collect()
}
