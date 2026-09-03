// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of the final-logit softcap (card A1.14, SI-28, Spec 4 §2).

use r9v_ir::{DType, LogitSoftcapOp};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Executes scalar T0 logit softcap: `y = cap * tanh(x / cap)` in f32 over
/// `x [T, V] f32` (card A1.14, SI-28, Spec 4 §2).
pub fn logit_softcap(
    op: &LogitSoftcapOp,
    x: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    if !op.cap.is_finite() || op.cap <= 0.0 {
        problems.push(format!(
            "attribute cap: must be finite and > 0, got {}",
            op.cap
        ));
    }
    if x.rank() != 2 {
        problems.push(format!(
            "input x: expected rank 2 [T, V], got rank {} with shape {:?}",
            x.rank(),
            x.shape()
        ));
    }
    if y.rank() != 2 {
        problems.push(format!(
            "output y: expected rank 2 [T, V], got rank {} with shape {:?}",
            y.rank(),
            y.shape()
        ));
    }
    if x.dtype() != DType::F32 {
        problems.push(format!("input x: expected f32, got {:?}", x.dtype()));
    }
    if y.dtype() != DType::F32 {
        problems.push(format!("output y: expected f32, got {:?}", y.dtype()));
    }
    if x.shape() != y.shape() {
        problems.push(format!(
            "output y shape {:?} does not match input x shape {:?}",
            y.shape(),
            x.shape()
        ));
    }

    T0Error::from_problems("logit_softcap", problems)?;

    let num_elem = x.num_elements();
    for i in 0..num_elem {
        let scaled = x.read_f32(i) / op.cap;
        y.write_f32(i, op.cap * scaled.tanh());
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (card A1.14, SI-28).
pub fn logit_softcap_f64_reference(x: &[f64], cap: f64) -> Vec<f64> {
    x.iter().map(|&v| cap * (v / cap).tanh()).collect()
}
