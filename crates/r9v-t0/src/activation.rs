// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of standalone non-gated activation op (Spec 1 §4.B, §6.4, Spec 4 §2).

use r9v_ir::{ActivationKind, ActivationOp, DType};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

use std::f64::consts::{FRAC_1_SQRT_2 as INV_SQRT_2_F64, FRAC_2_SQRT_PI as TWO_DIV_SQRT_PI_F64};

const SQRT_2_DIV_PI_F64: f64 = 0.79788456080286535587989211986876;
const ONE_DIV_SQRT_PI_F64: f64 = 0.56418958354775628694807945156077;

/// Computes the exact error function `erf(x)` in 64-bit precision without external dependencies.
pub fn erf_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x > 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    if ax > 6.0 {
        return sign;
    }

    if ax < 1.5 {
        // Taylor series expansion for small to medium arguments
        let mut term = ax * TWO_DIV_SQRT_PI_F64;
        let mut total = term;
        let ax2 = ax * ax;
        for n in 1..40 {
            let n_f = n as f64;
            term = -term * ax2 / n_f * (2.0 * n_f - 1.0) / (2.0 * n_f + 1.0);
            total += term;
            if term.abs() < 1e-17 {
                break;
            }
        }
        sign * total
    } else {
        // Continued fraction via modified Lentz's method for larger arguments
        let tiny = 1e-30f64;
        let mut f = ax;
        let mut c = f;
        let mut d = 0.0f64;
        for j in 1..80 {
            let a = (j as f64) * 0.5;
            let b = ax;
            d = b + a * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + a / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        let erfc_val = (-ax * ax).exp() * ONE_DIV_SQRT_PI_F64 / f;
        sign * (1.0 - erfc_val)
    }
}

/// Computes error function in 32-bit float precision.
pub fn erf_f32(x: f32) -> f32 {
    erf_f64(x as f64) as f32
}

/// Evaluates an activation function in 32-bit float precision.
pub fn eval_activation_f32(x: f32, kind: ActivationKind) -> f32 {
    match kind {
        ActivationKind::Silu => {
            // x / (1 + exp(-x))
            x / (1.0f32 + (-x).exp())
        }
        ActivationKind::Gelu => {
            // 0.5 * x * (1 + erf(x / sqrt(2)))
            let scaled_x = x * (INV_SQRT_2_F64 as f32);
            0.5f32 * x * (1.0f32 + erf_f32(scaled_x))
        }
        ActivationKind::GeluTanh => {
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let inner = (SQRT_2_DIV_PI_F64 as f32) * (x + 0.044715f32 * x * x * x);
            0.5f32 * x * (1.0f32 + inner.tanh())
        }
        ActivationKind::Relu2 => {
            let relu = if x > 0.0 { x } else { 0.0 };
            relu * relu
        }
        ActivationKind::Identity => x,
    }
}

/// Evaluates an activation function in 64-bit float precision for testing.
pub fn eval_activation_f64(x: f64, kind: ActivationKind) -> f64 {
    match kind {
        ActivationKind::Silu => x / (1.0f64 + (-x).exp()),
        ActivationKind::Gelu => {
            let scaled_x = x * INV_SQRT_2_F64;
            0.5f64 * x * (1.0f64 + erf_f64(scaled_x))
        }
        ActivationKind::GeluTanh => {
            let inner = SQRT_2_DIV_PI_F64 * (x + 0.044715f64 * x * x * x);
            0.5f64 * x * (1.0f64 + inner.tanh())
        }
        ActivationKind::Relu2 => {
            let relu = if x > 0.0 { x } else { 0.0 };
            relu * relu
        }
        ActivationKind::Identity => x,
    }
}

/// Executes scalar T0 standalone non-gated activation (Spec 1 §4.B, §6.4).
pub fn activation(
    op: &ActivationOp,
    x: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();

    if x.rank() != 2 {
        problems.push(format!(
            "input x: expected rank 2 [T, Dff], got rank {} with shape {:?}",
            x.rank(),
            x.shape()
        ));
    }
    if y.rank() != 2 {
        problems.push(format!(
            "output y: expected rank 2 [T, Dff], got rank {} with shape {:?}",
            y.rank(),
            y.shape()
        ));
    }
    if !matches!(x.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(format!(
            "input x: expected f16, bf16, or f32, got {:?}",
            x.dtype()
        ));
    }
    if y.dtype() != x.dtype() {
        problems.push(format!(
            "output y dtype {:?} does not match input x dtype {:?}",
            y.dtype(),
            x.dtype()
        ));
    }
    if x.rank() == 2 && y.rank() == 2 && x.shape() != y.shape() {
        problems.push(format!(
            "output y shape {:?} does not match input x shape {:?}",
            y.shape(),
            x.shape()
        ));
    }
    if let Some(c) = op.clamp {
        if !c.is_finite() || c <= 0.0 {
            problems.push(format!("clamp must be finite and > 0, got {c}"));
        }
    }

    T0Error::from_problems("activation", problems)?;

    let num_elem = x.num_elements();
    for i in 0..num_elem {
        let val = x.read_f32(i);
        let mut act_val = eval_activation_f32(val, op.act);
        if let Some(c) = op.clamp {
            act_val = act_val.min(c);
        }
        y.write_f32(i, act_val);
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0.
pub fn activation_f64_reference(op: &ActivationOp, x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&val| {
            let mut act_val = eval_activation_f64(val, op.act);
            if let Some(c) = op.clamp {
                act_val = act_val.min(c as f64);
            }
            act_val
        })
        .collect()
}
