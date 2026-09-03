// SPDX-License-Identifier: Apache-2.0
//! R9V CPU reference scalar T0 implementations (Spec 1 §4.B, §6.4, Spec 4 §2, Card A1.5).
//!
//! T0 serves as the primary ground-truth oracle for all engine operations.
//! Implementations are scalar, strictly deterministic, accumulate in f32 (or i32),
//! and cast once on output according to the IR contract.

pub mod act_mul;
pub mod activation;
pub mod buffer;
pub mod cast;
pub mod copy;
pub mod dtype;
pub mod error;
pub mod norm;
pub mod quant_act;
pub mod rope;
pub mod tolerance;

pub use act_mul::{act_mul, act_mul_f64_reference};
pub use activation::{
    activation, activation_f64_reference, erf_f32, erf_f64, eval_activation_f32,
    eval_activation_f64,
};
pub use buffer::{TensorData, TensorDataMut, TensorView, TensorViewMut, TypedBuffer};
pub use cast::{cast, cast_f64_reference};
pub use copy::copy;
pub use dtype::{
    bf16_to_f32, dtype_element_size, f16_to_f32, f32_to_bf16, f32_to_f16, fp8_e4m3_decode,
    fp8_e4m3_encode, fp8_e5m2_decode, fp8_e5m2_encode, read_f32_at, read_f64_at, write_f32_at,
};
pub use error::T0Error;
pub use norm::{norm, norm_f64_reference};
pub use quant_act::{quant_act, quant_act_f64_reference};
pub use residual_add::{residual_add, residual_add_f64_reference};
pub use rope::{rope, rope_f64_reference};
pub use tolerance::Tolerance;

pub mod residual_add;

use r9v_ir::Op;

/// Dispatches and executes an elementwise op using scalar T0 reference implementations.
pub fn execute_elementwise_op(
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<(), T0Error> {
    match op {
        Op::Norm(norm_op) => {
            if inputs.len() < 2 || outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "norm",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "norm requires 2 or 3 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let x = &inputs[0];
            let weight = &inputs[1];
            let bias = if inputs.len() >= 3 {
                Some(&inputs[2])
            } else {
                None
            };
            norm(norm_op, x, weight, bias, &mut outputs[0])
        }
        Op::ResidualAdd(res_op) => {
            if inputs.len() < 2 || outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "residual_add",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "residual_add requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            residual_add(res_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::ActMul(act_mul_op) => {
            if inputs.len() < 2 || outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "act_mul",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "act_mul requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            act_mul(act_mul_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::Activation(act_op) => {
            if inputs.is_empty() || outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "activation",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "activation requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            activation(act_op, &inputs[0], &mut outputs[0])
        }
        Op::Rope(rope_op) => {
            if inputs.len() < 2 || outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "rope",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "rope requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            rope(rope_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::Cast(cast_op) => {
            if inputs.is_empty() || outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "cast",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "cast requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            cast(cast_op, &inputs[0], &mut outputs[0])
        }
        Op::Copy(copy_op) => {
            if inputs.is_empty() || outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "copy",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "copy requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            copy(copy_op, &inputs[0], &mut outputs[0])
        }
        Op::QuantAct(quant_op) => {
            if inputs.is_empty() || outputs.len() < 2 {
                return Err(T0Error::InvalidAttribute {
                    op: "quant_act",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "quant_act requires 1 input and 2 outputs, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let (xq, rest) = outputs.split_at_mut(1);
            quant_act(quant_op, &inputs[0], &mut xq[0], &mut rest[0])
        }
        other => Err(T0Error::InvalidAttribute {
            op: other.op_name(),
            attribute: "op",
            reason: format!(
                "op `{}` is not in the A1.5 elementwise group",
                other.op_name()
            ),
        }),
    }
}
