// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of precision cast op (Spec 1 §4.A, §6.4, Spec 4 §2).

use r9v_ir::CastOp;

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Executes scalar T0 precision cast: `x -> y` with `y.dtype == op.dtype` (Spec 1 §4.A).
pub fn cast(op: &CastOp, x: &TensorView<'_>, y: &mut TensorViewMut<'_>) -> Result<(), T0Error> {
    let mut problems = Vec::new();

    if x.shape() != y.shape() {
        problems.push(format!(
            "output y shape {:?} does not match input x shape {:?}",
            y.shape(),
            x.shape()
        ));
    }
    if y.dtype() != op.dtype {
        problems.push(format!(
            "output y dtype {:?} does not match op target dtype {:?}",
            y.dtype(),
            op.dtype
        ));
    }

    T0Error::from_problems("cast", problems)?;

    let num_elem = x.num_elements();
    for i in 0..num_elem {
        let val = x.read_f32(i);
        y.write_f32(i, val);
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0.
pub fn cast_f64_reference(x: &[f64]) -> Vec<f64> {
    x.to_vec()
}
