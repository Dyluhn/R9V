// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of memory copy and contiguization op (Spec 1 §4.A, Spec 4 §2).

use r9v_ir::CopyOp;

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Executes scalar T0 tensor copy / contiguization (Spec 1 §4.A).
pub fn copy(_op: &CopyOp, x: &TensorView<'_>, y: &mut TensorViewMut<'_>) -> Result<(), T0Error> {
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
    for i in 0..num_elem {
        let val = x.read_f32(i);
        y.write_f32(i, val);
    }

    Ok(())
}
