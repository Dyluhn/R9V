// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of the final-logit softcap (card A1.14, SI-28, Spec 4 §2).

use r9v_ir::{DType, LogitSoftcapOp};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::{push_shape_agreement, T0Error};

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
        problems.push(T0Error::InvalidAttribute {
            op: "logit_softcap",
            attribute: "cap",
            reason: format!("must be finite and > 0, got {}", op.cap),
        });
    }
    if x.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "x",
            expected: 2,
            got: x.rank(),
            shape: x.shape().to_vec(),
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
    if x.dtype() != DType::F32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![DType::F32],
            got: x.dtype(),
        });
    }
    if y.dtype() != DType::F32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![DType::F32],
            got: y.dtype(),
        });
    }
    if x.shape() != y.shape() {
        push_shape_agreement(&mut problems, "y", "x", y.shape(), x.shape());
    }

    T0Error::from_problems(problems)?;

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
