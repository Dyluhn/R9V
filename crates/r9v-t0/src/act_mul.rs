// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of gated activation multiplication (Spec 1 §4.B, §6.4, Spec 4 §2).

use r9v_ir::{ActMulOp, DType};

use crate::activation::{eval_activation_f32, eval_activation_f64};
use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Executes scalar T0 gated activation product: `y = act(gate) * up` (Spec 1 §4.B, Spec 1 §6.4, Spec 4 §2).
pub fn act_mul(
    op: &ActMulOp,
    gate: &TensorView<'_>,
    up: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    gate.validate_backing("gate")?;
    up.validate_backing("up")?;
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    if gate.rank() != 2 {
        problems.push(format!(
            "operand gate: expected rank 2 [T, Dff], got rank {} with shape {:?}",
            gate.rank(),
            gate.shape()
        ));
    }
    if up.rank() != 2 {
        problems.push(format!(
            "operand up: expected rank 2 [T, Dff], got rank {} with shape {:?}",
            up.rank(),
            up.shape()
        ));
    }
    if y.rank() != 2 {
        problems.push(format!(
            "output y: expected rank 2 [T, Dff], got rank {} with shape {:?}",
            y.rank(),
            y.shape()
        ));
    }
    if !matches!(gate.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(format!(
            "operand gate: expected f16, bf16, or f32, got {:?}",
            gate.dtype()
        ));
    }
    if up.dtype() != gate.dtype() {
        problems.push(format!(
            "operand up dtype {:?} does not match gate dtype {:?}",
            up.dtype(),
            gate.dtype()
        ));
    }
    if y.dtype() != gate.dtype() {
        problems.push(format!(
            "output y dtype {:?} does not match gate dtype {:?}",
            y.dtype(),
            gate.dtype()
        ));
    }
    if gate.rank() == 2 && up.rank() == 2 && gate.shape() != up.shape() {
        problems.push(format!(
            "operand up shape {:?} does not match gate shape {:?}",
            up.shape(),
            gate.shape()
        ));
    }
    if gate.rank() == 2 && y.rank() == 2 && gate.shape() != y.shape() {
        problems.push(format!(
            "output y shape {:?} does not match gate shape {:?}",
            y.shape(),
            gate.shape()
        ));
    }
    if let Some(c) = op.clamp {
        if !c.is_finite() || c <= 0.0 {
            problems.push(format!("clamp must be finite and > 0, got {c}"));
        }
    }

    T0Error::from_problems("act_mul", problems)?;

    let num_elem = gate.num_elements();
    for i in 0..num_elem {
        let g = gate.read_f32(i);
        let u = up.read_f32(i);

        // DECISION(A1.5): in act_mul, clamp is applied to the activated gate value act(gate).min(c) before multiplying with up (matching activation op where y = act(x).min(c)); rejected clamping post-multiplication (act(gate)*up).min(c) because clamp is an activation-level attribute paired with act in both ActMulOp and ActivationOp (Spec 1 §4.B).
        let mut act_g = eval_activation_f32(g, op.act);
        if let Some(c) = op.clamp {
            act_g = act_g.min(c);
        }

        let out = act_g * u;
        y.write_f32(i, out);
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (Spec 1 §4.B, Spec 4 §2).
pub fn act_mul_f64_reference(op: &ActMulOp, gate: &[f64], up: &[f64]) -> Vec<f64> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let mut act_g = eval_activation_f64(g, op.act);
            if let Some(c) = op.clamp {
                act_g = act_g.min(c as f64);
            }
            act_g * u
        })
        .collect()
}
