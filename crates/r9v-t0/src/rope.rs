// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of Rotary Position Embedding (RoPE) (Spec 1 §4.B, §6.4, Spec 4 §2).

use r9v_ir::{DType, RopeOp, RopeScaling, RopeStyle};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Precomputes RoPE frequencies for a given token position and rotary dimension.
#[allow(clippy::needless_range_loop)]
fn compute_freqs_and_phases_f32(
    op: &RopeOp,
    token_idx: usize,
    positions: &TensorView<'_>,
) -> (Vec<f32>, f32) {
    let m = (op.rot_dim / 2) as usize;
    let mut phases = vec![0.0f32; m];
    let mut eff_mscale = 1.0f32;

    // DECISION(A1.5): for mrope with mrope_sections [s0, s1, s2], section dimensions represent channel counts where section i allocates si/2 frequency pairs (Spec 1 §4.B, r9v-ir RopeOp); frequency pair k uses coordinate positions[t, 0] for k < s0/2, positions[t, 1] for s0/2 <= k < (s0+s1)/2, positions[t, 2] for (s0+s1)/2 <= k < (s0+s1+s2)/2, and positions[t, 0] for any remaining pairs up to rot_dim/2; rejected resetting frequency indices to 0 per section because RoPE frequencies form a single monotonic spectrum across rot_dim.
    let mrope_bounds = op.mrope_sections.map(|sections| {
        let m0 = (sections[0] / 2) as usize;
        let m1 = (sections[1] / 2) as usize;
        let m2 = (sections[2] / 2) as usize;
        (m0, m0 + m1, m0 + m1 + m2)
    });

    let get_pos = |pair_idx: usize| -> f32 {
        if let Some((b0, b1, b2)) = mrope_bounds {
            let coord_idx = if pair_idx < b0 {
                0
            } else if pair_idx < b1 {
                1
            } else if pair_idx < b2 {
                2
            } else {
                0
            };
            positions.read_u32(token_idx * 3 + coord_idx) as f32
        } else {
            positions.read_u32(token_idx) as f32
        }
    };

    match op.scaling {
        RopeScaling::None => {
            for k in 0..m {
                let inv_freq = 1.0f32 / op.theta.powf((2 * k) as f32 / op.rot_dim as f32);
                let pos = get_pos(k);
                phases[k] = pos * inv_freq;
            }
        }
        RopeScaling::Linear(factor) => {
            for k in 0..m {
                let inv_freq =
                    (1.0f32 / op.theta.powf((2 * k) as f32 / op.rot_dim as f32)) / factor;
                let pos = get_pos(k);
                phases[k] = pos * inv_freq;
            }
        }
        RopeScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
            orig_ctx,
            mscale,
        } => {
            eff_mscale = mscale;
            for k in 0..m {
                let inv_freq = 1.0f32 / op.theta.powf((2 * k) as f32 / op.rot_dim as f32);
                let r_k = (orig_ctx as f32 * inv_freq) / (2.0f32 * std::f32::consts::PI);
                let gamma = if beta_fast == beta_slow {
                    if r_k >= beta_fast {
                        1.0f32
                    } else {
                        0.0f32
                    }
                } else {
                    ((r_k - beta_slow) / (beta_fast - beta_slow)).clamp(0.0f32, 1.0f32)
                };
                let scaled_inv_freq = inv_freq * ((1.0f32 - gamma) / factor + gamma);
                let pos = get_pos(k);
                phases[k] = pos * scaled_inv_freq;
            }
        }
        RopeScaling::Dynamic => {
            // DECISION(A1.5): for RopeScaling::Dynamic, since RopeOp carries no context or threshold parameters in the IR (Spec 1 §4.B, r9v-ir RopeScaling::Dynamic) and Spec 1 §6.1 batch invariance prohibits deriving context from the batch's max sequence length, dynamic NTK scaling uses a reference context length of 2048; for pos < 2048, scale is 1.0 (unscaled), and for pos >= 2048, s = (pos + 1) / 2048.0 with base theta scaled by s^(rot_dim / (rot_dim - 2)); rejected batch-wide max_pos (breaks Spec 1 §6.1 batch invariance) and rejected panic/stub.
            let exponent = (op.rot_dim as f32) / ((op.rot_dim - 2) as f32);
            for k in 0..m {
                let pos = get_pos(k);
                let theta_eff = if pos >= 2048.0 {
                    let s = (pos + 1.0) / 2048.0;
                    op.theta * s.powf(exponent)
                } else {
                    op.theta
                };
                let inv_freq = 1.0f32 / theta_eff.powf((2 * k) as f32 / op.rot_dim as f32);
                phases[k] = pos * inv_freq;
            }
        }
    }

    (phases, eff_mscale)
}

/// Executes scalar T0 Rotary Position Embedding (RoPE) (Spec 1 §4.B, §6.4).
pub fn rope(
    op: &RopeOp,
    x: &TensorView<'_>,
    positions: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();

    if x.rank() != 3 {
        problems.push(format!(
            "input x: expected rank 3 [T, H, D], got rank {} with shape {:?}",
            x.rank(),
            x.shape()
        ));
    }
    if y.rank() != 3 {
        problems.push(format!(
            "output y: expected rank 3 [T, H, D], got rank {} with shape {:?}",
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
    if y.dtype() != op.out_dtype {
        problems.push(format!(
            "output y: expected out_dtype {:?}, got {:?}",
            op.out_dtype,
            y.dtype()
        ));
    }
    if positions.dtype() != DType::U32 {
        problems.push(format!(
            "positions: expected u32, got {:?}",
            positions.dtype()
        ));
    }

    if op.mrope_sections.is_some() {
        if positions.rank() != 2 || (positions.rank() == 2 && positions.shape()[1] != 3) {
            problems.push(format!(
                "mrope requires positions with rank 2 [T, 3], got rank {} with shape {:?}",
                positions.rank(),
                positions.shape()
            ));
        }
    } else if positions.rank() != 1 {
        problems.push(format!(
            "standard rope requires positions with rank 1 [T], got rank {} with shape {:?}",
            positions.rank(),
            positions.shape()
        ));
    }

    if x.rank() == 3 {
        let t = x.shape()[0];
        let d = x.shape()[2];
        if positions.shape()[0] != t {
            problems.push(format!(
                "positions token dimension {} does not match input x token dimension T={}",
                positions.shape()[0],
                t
            ));
        }
        if (op.rot_dim as usize) > d {
            problems.push(format!(
                "rot_dim {} exceeds head dimension D={}",
                op.rot_dim, d
            ));
        }
        if op.rot_dim == 0 || !op.rot_dim.is_multiple_of(2) {
            problems.push(format!(
                "rot_dim must be positive and even, got {}",
                op.rot_dim
            ));
        }
        if y.rank() == 3 && y.shape() != x.shape() {
            problems.push(format!(
                "output y shape {:?} does not match input x shape {:?}",
                y.shape(),
                x.shape()
            ));
        }
    }

    T0Error::from_problems("rope", problems)?;

    let t = x.shape()[0];
    let h = x.shape()[1];
    let d = x.shape()[2];
    let rot_dim = op.rot_dim as usize;
    let m = rot_dim / 2;

    for token in 0..t {
        let (phases, eff_mscale) = compute_freqs_and_phases_f32(op, token, positions);
        let mut cos_table = vec![0.0f32; m];
        let mut sin_table = vec![0.0f32; m];
        for k in 0..m {
            cos_table[k] = eff_mscale * phases[k].cos();
            sin_table[k] = eff_mscale * phases[k].sin();
        }

        for head in 0..h {
            let offset = (token * h + head) * d;

            match op.style {
                RopeStyle::Neox => {
                    // Coordinates [0..m) paired with [m..2m)
                    for k in 0..m {
                        let idx1 = offset + k;
                        let idx2 = offset + k + m;
                        let x1 = x.read_f32(idx1);
                        let x2 = x.read_f32(idx2);
                        let cos = cos_table[k];
                        let sin = sin_table[k];

                        let rotated1 = x1 * cos - x2 * sin;
                        let rotated2 = x2 * cos + x1 * sin;

                        y.write_f32(idx1, rotated1);
                        y.write_f32(idx2, rotated2);
                    }
                }
                RopeStyle::Interleaved => {
                    // Coordinates [2k] paired with [2k+1]
                    for k in 0..m {
                        let idx1 = offset + 2 * k;
                        let idx2 = offset + 2 * k + 1;
                        let x1 = x.read_f32(idx1);
                        let x2 = x.read_f32(idx2);
                        let cos = cos_table[k];
                        let sin = sin_table[k];

                        let rotated1 = x1 * cos - x2 * sin;
                        let rotated2 = x1 * sin + x2 * cos;

                        y.write_f32(idx1, rotated1);
                        y.write_f32(idx2, rotated2);
                    }
                }
            }

            // Unrotated tail: pass through untouched
            for j in rot_dim..d {
                let idx = offset + j;
                let val = x.read_f32(idx);
                y.write_f32(idx, val);
            }
        }
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0.
#[allow(clippy::needless_range_loop)]
pub fn rope_f64_reference(
    op: &RopeOp,
    x: &[f64],
    shape: [usize; 3],
    positions: &[u32],
    positions_is_2d: bool,
) -> Vec<f64> {
    let [t, h, d] = shape;
    assert_eq!(x.len(), t * h * d);
    let mut out = vec![0.0f64; t * h * d];
    let rot_dim = op.rot_dim as usize;
    let m = rot_dim / 2;

    let mrope_bounds = op.mrope_sections.map(|sections| {
        let m0 = (sections[0] / 2) as usize;
        let m1 = (sections[1] / 2) as usize;
        let m2 = (sections[2] / 2) as usize;
        (m0, m0 + m1, m0 + m1 + m2)
    });

    let get_pos = |token_idx: usize, pair_idx: usize| -> f64 {
        if positions_is_2d {
            let coord_idx = if let Some((b0, b1, b2)) = mrope_bounds {
                if pair_idx < b0 {
                    0
                } else if pair_idx < b1 {
                    1
                } else if pair_idx < b2 {
                    2
                } else {
                    0
                }
            } else {
                0
            };
            positions[token_idx * 3 + coord_idx] as f64
        } else {
            positions[token_idx] as f64
        }
    };

    for token in 0..t {
        let mut cos_table = vec![0.0f64; m];
        let mut sin_table = vec![0.0f64; m];

        let (phases, eff_mscale) = match op.scaling {
            RopeScaling::None => {
                let mut ph = vec![0.0f64; m];
                for k in 0..m {
                    let inv_freq =
                        1.0f64 / (op.theta as f64).powf((2 * k) as f64 / op.rot_dim as f64);
                    let pos = get_pos(token, k);
                    ph[k] = pos * inv_freq;
                }
                (ph, 1.0f64)
            }
            RopeScaling::Linear(factor) => {
                let mut ph = vec![0.0f64; m];
                for k in 0..m {
                    let inv_freq = (1.0f64
                        / (op.theta as f64).powf((2 * k) as f64 / op.rot_dim as f64))
                        / (factor as f64);
                    let pos = get_pos(token, k);
                    ph[k] = pos * inv_freq;
                }
                (ph, 1.0f64)
            }
            RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                orig_ctx,
                mscale,
            } => {
                let mut ph = vec![0.0f64; m];
                for k in 0..m {
                    let inv_freq =
                        1.0f64 / (op.theta as f64).powf((2 * k) as f64 / op.rot_dim as f64);
                    let r_k = (orig_ctx as f64 * inv_freq) / (2.0f64 * std::f64::consts::PI);
                    let gamma = if beta_fast == beta_slow {
                        if r_k >= (beta_fast as f64) {
                            1.0f64
                        } else {
                            0.0f64
                        }
                    } else {
                        ((r_k - beta_slow as f64) / (beta_fast as f64 - beta_slow as f64))
                            .clamp(0.0f64, 1.0f64)
                    };
                    let scaled_inv_freq = inv_freq * ((1.0f64 - gamma) / (factor as f64) + gamma);
                    let pos = get_pos(token, k);
                    ph[k] = pos * scaled_inv_freq;
                }
                (ph, mscale as f64)
            }
            RopeScaling::Dynamic => {
                let exponent = (op.rot_dim as f64) / ((op.rot_dim - 2) as f64);
                let mut ph = vec![0.0f64; m];
                for k in 0..m {
                    let pos = get_pos(token, k);
                    let theta_eff = if pos >= 2048.0 {
                        let s = (pos + 1.0) / 2048.0;
                        (op.theta as f64) * s.powf(exponent)
                    } else {
                        op.theta as f64
                    };
                    let inv_freq = 1.0f64 / theta_eff.powf((2 * k) as f64 / op.rot_dim as f64);
                    ph[k] = pos * inv_freq;
                }
                (ph, 1.0f64)
            }
        };

        for k in 0..m {
            cos_table[k] = eff_mscale * phases[k].cos();
            sin_table[k] = eff_mscale * phases[k].sin();
        }

        for head in 0..h {
            let offset = (token * h + head) * d;
            match op.style {
                RopeStyle::Neox => {
                    for k in 0..m {
                        let idx1 = offset + k;
                        let idx2 = offset + k + m;
                        let x1 = x[idx1];
                        let x2 = x[idx2];
                        let cos = cos_table[k];
                        let sin = sin_table[k];
                        out[idx1] = x1 * cos - x2 * sin;
                        out[idx2] = x2 * cos + x1 * sin;
                    }
                }
                RopeStyle::Interleaved => {
                    for k in 0..m {
                        let idx1 = offset + 2 * k;
                        let idx2 = offset + 2 * k + 1;
                        let x1 = x[idx1];
                        let x2 = x[idx2];
                        let cos = cos_table[k];
                        let sin = sin_table[k];
                        out[idx1] = x1 * cos - x2 * sin;
                        out[idx2] = x1 * sin + x2 * cos;
                    }
                }
            }

            for j in rot_dim..d {
                let idx = offset + j;
                out[idx] = x[idx];
            }
        }
    }

    out
}
