// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of normalization ops (RMS and LayerNorm) (Spec 1 §4.B, §6.4, Spec 4 §2).

use r9v_ir::{DType, NormAxis, NormKind, NormOp};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Executes scalar T0 normalization (RMS or LayerNorm) per Spec 1 §4.B, §6.4, Spec 4 §2.
///
/// Accumulates mean/variance in f32 in ascending index order and casts once on output.
pub fn norm(
    op: &NormOp,
    x: &TensorView<'_>,
    weight: &TensorView<'_>,
    bias: Option<&TensorView<'_>>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    weight.validate_backing("weight")?;
    if let Some(b) = bias {
        b.validate_backing("bias")?;
    }
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    if !op.eps.is_finite() || op.eps <= 0.0 {
        problems.push(format!("eps must be finite and > 0, got {}", op.eps));
    }
    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(format!(
            "out_dtype must be f16, bf16, or f32, got {:?}",
            op.out_dtype
        ));
    }
    if !op.weight_offset.is_finite() {
        problems.push(format!(
            "weight_offset must be finite, got {}",
            op.weight_offset
        ));
    }
    if let NormAxis::Head(d) = op.axis {
        if d == 0 {
            problems.push("NormAxis::Head(d): d must be > 0".to_string());
        }
    }

    if x.rank() != 2 {
        problems.push(format!(
            "input x: expected rank 2 [T, N], got rank {} with shape {:?}",
            x.rank(),
            x.shape()
        ));
    }
    if weight.rank() != 1 {
        problems.push(format!(
            "weight: expected rank 1 [N], got rank {} with shape {:?}",
            weight.rank(),
            weight.shape()
        ));
    }
    if y.rank() != 2 {
        problems.push(format!(
            "output y: expected rank 2 [T, N], got rank {} with shape {:?}",
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
    if weight.dtype() != DType::F32 {
        problems.push(format!("weight: expected f32, got {:?}", weight.dtype()));
    }
    if y.dtype() != op.out_dtype {
        problems.push(format!(
            "output y: expected out_dtype {:?}, got {:?}",
            op.out_dtype,
            y.dtype()
        ));
    }

    if let Some(b) = bias {
        if b.rank() != 1 {
            problems.push(format!(
                "bias: expected rank 1 [N], got rank {} with shape {:?}",
                b.rank(),
                b.shape()
            ));
        }
        if b.dtype() != DType::F32 {
            problems.push(format!("bias: expected f32, got {:?}", b.dtype()));
        }
    }

    if x.rank() == 2 && weight.rank() == 1 {
        let n = x.shape()[1];
        if weight.shape()[0] != n {
            problems.push(format!(
                "weight dimension {} does not match x feature dimension N={}",
                weight.shape()[0],
                n
            ));
        }
        if let Some(b) = bias {
            if b.rank() == 1 && b.shape()[0] != n {
                problems.push(format!(
                    "bias dimension {} does not match x feature dimension N={}",
                    b.shape()[0],
                    n
                ));
            }
        }
        if let NormAxis::Head(d) = op.axis {
            if d == 0 {
                problems.push("NormAxis::Head(d): d must be > 0".to_string());
            } else if !n.is_multiple_of(d as usize) {
                problems.push(format!(
                    "feature dimension N={n} is not divisible by head dimension d={d}"
                ));
            }
        }
    }

    if x.rank() == 2 && y.rank() == 2 && y.shape() != x.shape() {
        problems.push(format!(
            "output y shape {:?} does not match input x shape {:?}",
            y.shape(),
            x.shape()
        ));
    }

    T0Error::from_problems("norm", problems)?;

    let t = x.shape()[0];
    let n = x.shape()[1];

    match op.axis {
        NormAxis::Last => {
            for row in 0..t {
                let row_offset = row * n;
                match op.kind {
                    NormKind::Rms => {
                        let mut sum_sq = 0.0f32;
                        for i in 0..n {
                            let val = x.read_f32(row_offset + i);
                            sum_sq += val * val;
                        }
                        let mean_sq = sum_sq / (n as f32);
                        let inv_rms = 1.0f32 / (mean_sq + op.eps).sqrt();

                        for i in 0..n {
                            let val = x.read_f32(row_offset + i);
                            let w = weight.read_f32(i) + op.weight_offset;
                            let b = bias.map(|b_view| b_view.read_f32(i)).unwrap_or(0.0f32);
                            let normalized = val * inv_rms;
                            let out = normalized * w + b;
                            y.write_f32(row_offset + i, out);
                        }
                    }
                    NormKind::Layer => {
                        let mut sum = 0.0f32;
                        for i in 0..n {
                            sum += x.read_f32(row_offset + i);
                        }
                        let mean = sum / (n as f32);

                        let mut var_sum = 0.0f32;
                        for i in 0..n {
                            let diff = x.read_f32(row_offset + i) - mean;
                            var_sum += diff * diff;
                        }
                        let var = var_sum / (n as f32);
                        let inv_sd = 1.0f32 / (var + op.eps).sqrt();

                        for i in 0..n {
                            let diff = x.read_f32(row_offset + i) - mean;
                            let w = weight.read_f32(i) + op.weight_offset;
                            let b = bias.map(|b_view| b_view.read_f32(i)).unwrap_or(0.0f32);
                            let normalized = diff * inv_sd;
                            let out = normalized * w + b;
                            y.write_f32(row_offset + i, out);
                        }
                    }
                }
            }
        }
        NormAxis::Head(d_u32) => {
            let d = d_u32 as usize;
            let num_heads = n / d;

            for row in 0..t {
                let row_offset = row * n;
                for h in 0..num_heads {
                    let head_offset = row_offset + h * d;
                    match op.kind {
                        NormKind::Rms => {
                            let mut sum_sq = 0.0f32;
                            for j in 0..d {
                                let val = x.read_f32(head_offset + j);
                                sum_sq += val * val;
                            }
                            let mean_sq = sum_sq / (d as f32);
                            let inv_rms = 1.0f32 / (mean_sq + op.eps).sqrt();

                            for j in 0..d {
                                let idx = head_offset + j;
                                let w_idx = h * d + j;
                                let val = x.read_f32(idx);
                                let w = weight.read_f32(w_idx) + op.weight_offset;
                                let b = bias.map(|b_view| b_view.read_f32(w_idx)).unwrap_or(0.0f32);
                                let normalized = val * inv_rms;
                                let out = normalized * w + b;
                                y.write_f32(idx, out);
                            }
                        }
                        NormKind::Layer => {
                            let mut sum = 0.0f32;
                            for j in 0..d {
                                sum += x.read_f32(head_offset + j);
                            }
                            let mean = sum / (d as f32);

                            let mut var_sum = 0.0f32;
                            for j in 0..d {
                                let diff = x.read_f32(head_offset + j) - mean;
                                var_sum += diff * diff;
                            }
                            let var = var_sum / (d as f32);
                            let inv_sd = 1.0f32 / (var + op.eps).sqrt();

                            for j in 0..d {
                                let idx = head_offset + j;
                                let w_idx = h * d + j;
                                let diff = x.read_f32(idx) - mean;
                                let w = weight.read_f32(w_idx) + op.weight_offset;
                                let b = bias.map(|b_view| b_view.read_f32(w_idx)).unwrap_or(0.0f32);
                                let normalized = diff * inv_sd;
                                let out = normalized * w + b;
                                y.write_f32(idx, out);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (Spec 1 §4.B, Spec 4 §2).
pub fn norm_f64_reference(
    op: &NormOp,
    x: &[f64],
    shape: [usize; 2],
    weight: &[f64],
    bias: Option<&[f64]>,
    weight_offset: f64,
    eps: f64,
) -> Vec<f64> {
    let [t, n] = shape;
    assert_eq!(x.len(), t * n);
    assert_eq!(weight.len(), n);
    if let Some(b) = bias {
        assert_eq!(b.len(), n);
    }
    let mut out = vec![0.0f64; t * n];

    match op.axis {
        NormAxis::Last => {
            for row in 0..t {
                let row_offset = row * n;
                match op.kind {
                    NormKind::Rms => {
                        let mut sum_sq = 0.0f64;
                        for i in 0..n {
                            let v = x[row_offset + i];
                            sum_sq += v * v;
                        }
                        let mean_sq = sum_sq / (n as f64);
                        let inv_rms = 1.0f64 / (mean_sq + eps).sqrt();
                        for i in 0..n {
                            let v = x[row_offset + i];
                            let w = weight[i] + weight_offset;
                            let b = bias.map(|b_slice| b_slice[i]).unwrap_or(0.0f64);
                            out[row_offset + i] = (v * inv_rms) * w + b;
                        }
                    }
                    NormKind::Layer => {
                        let mut sum = 0.0f64;
                        for i in 0..n {
                            sum += x[row_offset + i];
                        }
                        let mean = sum / (n as f64);
                        let mut var_sum = 0.0f64;
                        for i in 0..n {
                            let diff = x[row_offset + i] - mean;
                            var_sum += diff * diff;
                        }
                        let var = var_sum / (n as f64);
                        let inv_sd = 1.0f64 / (var + eps).sqrt();
                        for i in 0..n {
                            let diff = x[row_offset + i] - mean;
                            let w = weight[i] + weight_offset;
                            let b = bias.map(|b_slice| b_slice[i]).unwrap_or(0.0f64);
                            out[row_offset + i] = (diff * inv_sd) * w + b;
                        }
                    }
                }
            }
        }
        NormAxis::Head(d_u32) => {
            let d = d_u32 as usize;
            let num_heads = n / d;
            for row in 0..t {
                let row_offset = row * n;
                for h in 0..num_heads {
                    let head_offset = row_offset + h * d;
                    match op.kind {
                        NormKind::Rms => {
                            let mut sum_sq = 0.0f64;
                            for j in 0..d {
                                let v = x[head_offset + j];
                                sum_sq += v * v;
                            }
                            let mean_sq = sum_sq / (d as f64);
                            let inv_rms = 1.0f64 / (mean_sq + eps).sqrt();
                            for j in 0..d {
                                let idx = head_offset + j;
                                let w_idx = h * d + j;
                                let v = x[idx];
                                let w = weight[w_idx] + weight_offset;
                                let b = bias.map(|b_slice| b_slice[w_idx]).unwrap_or(0.0f64);
                                out[idx] = (v * inv_rms) * w + b;
                            }
                        }
                        NormKind::Layer => {
                            let mut sum = 0.0f64;
                            for j in 0..d {
                                sum += x[head_offset + j];
                            }
                            let mean = sum / (d as f64);
                            let mut var_sum = 0.0f64;
                            for j in 0..d {
                                let diff = x[head_offset + j] - mean;
                                var_sum += diff * diff;
                            }
                            let var = var_sum / (d as f64);
                            let inv_sd = 1.0f64 / (var + eps).sqrt();
                            for j in 0..d {
                                let idx = head_offset + j;
                                let w_idx = h * d + j;
                                let diff = x[idx] - mean;
                                let w = weight[w_idx] + weight_offset;
                                let b = bias.map(|b_slice| b_slice[w_idx]).unwrap_or(0.0f64);
                                out[idx] = (diff * inv_sd) * w + b;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
