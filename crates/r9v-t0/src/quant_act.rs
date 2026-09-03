// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of activation quantization op (Spec 1 §4.A, Spec 2 §3.4, Spec 4 §2).

use r9v_ir::{DType, QuantActOp, QuantScheme};

use crate::buffer::{TensorView, TensorViewMut};
use crate::dtype::fp8_e4m3_encode;
use crate::error::T0Error;

/// Executes scalar T0 activation quantization (Spec 1 §4.A, Spec 2 §3.4, Spec 4 §2).
pub fn quant_act(
    op: &QuantActOp,
    x: &TensorView<'_>,
    xq: &mut TensorViewMut<'_>,
    scale: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    xq.validate_backing("xq")?;
    scale.validate_backing("scale")?;

    let mut problems = Vec::new();

    if op.target != DType::I8 && op.target != DType::E4m3 {
        problems.push(format!(
            "quant_act target must be i8 or e4m3, got {:?}",
            op.target
        ));
    }

    if x.rank() != 2 {
        problems.push(format!(
            "input x: expected rank 2 [T, N], got rank {} with shape {:?}",
            x.rank(),
            x.shape()
        ));
    }
    if xq.rank() != 2 {
        problems.push(format!(
            "output xq: expected rank 2 [T, N], got rank {} with shape {:?}",
            xq.rank(),
            xq.shape()
        ));
    }
    if !matches!(x.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(format!(
            "input x: expected f16, bf16, or f32, got {:?}",
            x.dtype()
        ));
    }
    if xq.dtype() != op.target {
        problems.push(format!(
            "output xq dtype {:?} does not match op target {:?}",
            xq.dtype(),
            op.target
        ));
    }
    if scale.dtype() != DType::F32 {
        problems.push(format!(
            "output scale: expected f32, got {:?}",
            scale.dtype()
        ));
    }

    if x.rank() == 2 && xq.rank() == 2 && x.shape() != xq.shape() {
        problems.push(format!(
            "output xq shape {:?} does not match input x shape {:?}",
            xq.shape(),
            x.shape()
        ));
    }

    if x.rank() == 2 {
        let t = x.shape()[0];
        let n = x.shape()[1];

        match op.scheme {
            QuantScheme::PerToken => {
                if op.target != DType::I8 && op.target != DType::E4m3 {
                    problems.push(format!(
                        "PerToken only supports i8 or e4m3 target, got {:?}",
                        op.target
                    ));
                }
                if scale.rank() != 1 || scale.shape()[0] != t {
                    problems.push(format!(
                        "PerToken scale: expected rank 1 [{t}], got rank {} with shape {:?}",
                        scale.rank(),
                        scale.shape()
                    ));
                }
            }
            QuantScheme::PerBlock32 => {
                if op.target != DType::I8 {
                    problems.push(format!(
                        "PerBlock32 only supports i8 target, got {:?}",
                        op.target
                    ));
                }
                if !n.is_multiple_of(32) {
                    problems.push(format!("PerBlock32 requires N={n} divisible by 32"));
                } else {
                    let expected_blocks = n / 32;
                    if scale.rank() != 2 || scale.shape() != [t, expected_blocks] {
                        problems.push(format!(
                            "PerBlock32 scale: expected shape [{t}, {expected_blocks}], got shape {:?}",
                            scale.shape()
                        ));
                    }
                }
            }
            _ => {
                problems.push(format!(
                    "quant_act only supports PerToken or PerBlock32, got {:?}",
                    op.scheme
                ));
            }
        }
    }

    T0Error::from_problems("quant_act", problems)?;

    let t = x.shape()[0];
    let n = x.shape()[1];

    match op.scheme {
        QuantScheme::PerToken => {
            for row in 0..t {
                let row_offset = row * n;
                let mut absmax = 0.0f32;
                for i in 0..n {
                    let v = x.read_f32(row_offset + i).abs();
                    if v > absmax {
                        absmax = v;
                    }
                }

                // DECISION(A1.5): for quant_act with zero absmax (all-zero row or block), scale is emitted as 0.0 and quantized elements as 0 (or 0x00 for e4m3); rejected emitting scale 1.0 or NaN to avoid artificial scale inflation on empty/zero padding tokens.
                match op.target {
                    DType::I8 => {
                        let s = absmax / 127.0f32;
                        scale.write_f32(row, s);

                        // DECISION(A1.5): for symmetric i8 quantization in quant_act, scaled values are rounded with round_ties_even and clamped to [-127, 127]; rejected [-128, 127] because Spec 1 §6.2 specifies accumulation bound 127·127·K assuming symmetric bounds.
                        for i in 0..n {
                            let idx = row_offset + i;
                            if s == 0.0f32 {
                                xq.write_f32(idx, 0.0f32);
                            } else {
                                let val = x.read_f32(idx);
                                let unquant = val / s;
                                let q = unquant.round_ties_even().clamp(-127.0f32, 127.0f32);
                                xq.write_f32(idx, q);
                            }
                        }
                    }
                    DType::E4m3 => {
                        let s = absmax / 448.0f32;
                        scale.write_f32(row, s);

                        for i in 0..n {
                            let idx = row_offset + i;
                            if s == 0.0f32 {
                                xq.write_byte(idx, 0x00);
                            } else {
                                let val = x.read_f32(idx);
                                let unquant = val / s;
                                let code = fp8_e4m3_encode(unquant);
                                xq.write_byte(idx, code);
                            }
                        }
                    }
                    _ => unreachable!("validated target"),
                }
            }
        }
        QuantScheme::PerBlock32 => {
            let num_blocks = n / 32;
            for row in 0..t {
                let row_offset = row * n;
                for b in 0..num_blocks {
                    let block_offset = row_offset + b * 32;
                    let mut absmax = 0.0f32;
                    for j in 0..32 {
                        let v = x.read_f32(block_offset + j).abs();
                        if v > absmax {
                            absmax = v;
                        }
                    }

                    let s = absmax / 127.0f32;
                    scale.write_f32(row * num_blocks + b, s);

                    for j in 0..32 {
                        let idx = block_offset + j;
                        if s == 0.0f32 {
                            xq.write_f32(idx, 0.0f32);
                        } else {
                            let val = x.read_f32(idx);
                            let unquant = val / s;
                            let q = unquant.round_ties_even().clamp(-127.0f32, 127.0f32);
                            xq.write_f32(idx, q);
                        }
                    }
                }
            }
        }
        _ => unreachable!("validated scheme"),
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (Spec 1 §4.A, Spec 2 §3.4, Spec 4 §2).
pub fn quant_act_f64_reference(
    op: &QuantActOp,
    x: &[f64],
    shape: [usize; 2],
) -> (Vec<f64>, Vec<f64>) {
    let [t, n] = shape;
    assert_eq!(x.len(), t * n);

    match op.scheme {
        QuantScheme::PerToken => {
            let mut scales = vec![0.0f64; t];
            let mut xq = vec![0.0f64; t * n];

            for (row, scale_out) in scales.iter_mut().enumerate().take(t) {
                let row_offset = row * n;
                let mut absmax = 0.0f64;
                for i in 0..n {
                    let v = x[row_offset + i].abs();
                    if v > absmax {
                        absmax = v;
                    }
                }

                match op.target {
                    DType::I8 => {
                        let s = absmax / 127.0f64;
                        *scale_out = s;
                        for i in 0..n {
                            let idx = row_offset + i;
                            if s == 0.0f64 {
                                xq[idx] = 0.0f64;
                            } else {
                                let unquant = x[idx] / s;
                                xq[idx] = unquant.round_ties_even().clamp(-127.0f64, 127.0f64);
                            }
                        }
                    }
                    DType::E4m3 => {
                        let s = absmax / 448.0f64;
                        *scale_out = s;
                        for i in 0..n {
                            let idx = row_offset + i;
                            if s == 0.0f64 {
                                xq[idx] = 0.0f64;
                            } else {
                                let unquant = x[idx] / s;
                                let code = fp8_e4m3_encode(unquant as f32);
                                xq[idx] = code as f64;
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            (xq, scales)
        }
        QuantScheme::PerBlock32 => {
            let num_blocks = n / 32;
            let mut scales = vec![0.0f64; t * num_blocks];
            let mut xq = vec![0.0f64; t * n];

            for row in 0..t {
                let row_offset = row * n;
                for b in 0..num_blocks {
                    let block_offset = row_offset + b * 32;
                    let mut absmax = 0.0f64;
                    for j in 0..32 {
                        let v = x[block_offset + j].abs();
                        if v > absmax {
                            absmax = v;
                        }
                    }

                    let s = absmax / 127.0f64;
                    scales[row * num_blocks + b] = s;

                    for j in 0..32 {
                        let idx = block_offset + j;
                        if s == 0.0f64 {
                            xq[idx] = 0.0f64;
                        } else {
                            let unquant = x[idx] / s;
                            xq[idx] = unquant.round_ties_even().clamp(-127.0f64, 127.0f64);
                        }
                    }
                }
            }
            (xq, scales)
        }
        _ => unreachable!(),
    }
}
