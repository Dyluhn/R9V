// SPDX-License-Identifier: Apache-2.0
//! Scalar deterministic CPU T0 matmul implementation (Spec 1 §4.C, §6.1, §6.2, Spec 2 §2–§3, Card A1.6).

use r9v_format::records::{E4M3Block128Scale, I4KSuperblock, I8Block128Scale, I8RowScale};
use r9v_format::scales::E4m3;
use r9v_format::{scale_geometry, FormatError, Layout, PaddedDims, SchemeId};
use r9v_ir::{DType, Epilogue, LayoutId, MatmulOp, QuantScheme};

use crate::activation::eval_activation_f32;
use crate::buffer::{TensorData, TensorView, TensorViewMut};
use crate::dtype::f16_to_f32;
use crate::error::T0Error;

/// Scalar deterministic CPU T0 matrix multiplication (Spec 1 §4.C, §6.1, §6.2, Card A1.6).
///
/// Signature:
/// - `x`: `[M, K]` (f16, bf16, i8 PerToken, i8 PerBlock32, e4m3 PerToken)
/// - `w`: `[N, K]` or `[K, N]` if transpose_w (f16, i8 PerRow, i8 I8_B128, i4 I4_K, e4m3 E4M3_B128)
/// - `bias`: optional `[N]` f32 (required if Epilogue::Bias)
/// - `residual`: optional `[M, N]` (required if Epilogue::Residual)
/// - `y`: `[M, N]` (out_dtype)
pub fn matmul(
    op: &MatmulOp,
    x: &TensorView<'_>,
    w: &TensorView<'_>,
    bias: Option<&TensorView<'_>>,
    residual: Option<&TensorView<'_>>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    matmul_with_scales(op, x, x.scale(), w, w.scale(), bias, residual, y)
}

/// Scalar deterministic CPU T0 matmul with explicit scale views (Spec 1 §4.C, §6.1, §6.2, Card A1.6).
///
/// DECISION(A1.6): matmul accepts scales either attached to TensorView or stored as contiguous trailing
/// scale regions in the weight buffer per Spec 2 §2.1 (for L0) and §6 (for L1), with explicit
/// matmul_with_scales for separate scale views; rejected requiring a single combined buffer format
/// because decoupled activation scales (emitted by quant_act) and unified weight buffers (emitted by
/// loader) must both be accepted.
///
/// DECISION(A1.6): for I4_K weights with PerBlock32 activations, zero-point subtraction
/// s_block * (x·q) - m_block * (Σx) is evaluated per 32-block in f32 and scaled by x_block_scale[m, b]
/// before ascending block accumulation; rejected per-token pre-scaling because PerBlock32 activation
/// scales vary per 32-element K-block. Spec 1 §6.2 specifies the zero-point formula for per-block Σx
/// but is silent on PerBlock32 activation scale binding.
#[allow(clippy::too_many_arguments)]
pub fn matmul_with_scales(
    op: &MatmulOp,
    x: &TensorView<'_>,
    x_scale: Option<&TensorView<'_>>,
    w: &TensorView<'_>,
    w_scale: Option<&TensorView<'_>>,
    bias: Option<&TensorView<'_>>,
    residual: Option<&TensorView<'_>>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    w.validate_backing("w")?;
    y.validate_backing("y")?;
    if let Some(b) = bias {
        b.validate_backing("bias")?;
    }
    if let Some(r) = residual {
        r.validate_backing("residual")?;
    }
    if let Some(xs) = x_scale {
        xs.validate_backing("x_scale")?;
    }
    if let Some(ws) = w_scale {
        ws.validate_backing("w_scale")?;
    }

    let mut problems = Vec::new();

    if x.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "x",
            expected: 2,
            got: x.rank(),
            shape: x.shape().to_vec(),
        });
    }

    if w.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "w",
            expected: 2,
            got: w.rank(),
            shape: w.shape().to_vec(),
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

    match op.epilogue {
        Epilogue::None | Epilogue::Act(_) => {
            if bias.is_some() {
                problems.push(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "bias",
                    reason: "bias provided but epilogue is not Bias".to_string(),
                });
            }
            if residual.is_some() {
                problems.push(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "residual",
                    reason: "residual provided but epilogue is not Residual".to_string(),
                });
            }
        }
        Epilogue::Bias => {
            if bias.is_none() {
                problems.push(T0Error::MissingOperand {
                    op: "matmul",
                    operand: "bias",
                    detail: "bias operand required for Bias epilogue".to_string(),
                });
            } else if let Some(b) = bias {
                if b.rank() != 1 {
                    problems.push(T0Error::RankMismatch {
                        tensor: "bias",
                        expected: 1,
                        got: b.rank(),
                        shape: b.shape().to_vec(),
                    });
                }
                if b.dtype() != DType::F32 {
                    problems.push(T0Error::DTypeMismatch {
                        tensor: "bias",
                        expected: vec![DType::F32],
                        got: b.dtype(),
                    });
                }
            }
            if residual.is_some() {
                problems.push(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "residual",
                    reason: "residual provided but epilogue is Bias".to_string(),
                });
            }
        }
        Epilogue::Residual => {
            if residual.is_none() {
                problems.push(T0Error::MissingOperand {
                    op: "matmul",
                    operand: "residual",
                    detail: "residual operand required for Residual epilogue".to_string(),
                });
            } else if let Some(r) = residual {
                if r.rank() != 2 {
                    problems.push(T0Error::RankMismatch {
                        tensor: "residual",
                        expected: 2,
                        got: r.rank(),
                        shape: r.shape().to_vec(),
                    });
                }
                if !matches!(r.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
                    problems.push(T0Error::DTypeMismatch {
                        tensor: "residual",
                        expected: vec![DType::F16, DType::Bf16, DType::F32],
                        got: r.dtype(),
                    });
                }
            }
            if bias.is_some() {
                problems.push(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "bias",
                    reason: "bias provided but epilogue is Residual".to_string(),
                });
            }
        }
    }

    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "out_dtype",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: op.out_dtype,
        });
    }

    if y.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.out_dtype],
            got: y.dtype(),
        });
    }

    // Validate activation operand dtype and quant scheme
    match (x.dtype(), x.quant()) {
        (DType::F16 | DType::Bf16 | DType::F32, QuantScheme::None) => {}
        (DType::I8, QuantScheme::PerToken | QuantScheme::PerBlock32) => {}
        (DType::E4m3, QuantScheme::PerToken) => {}
        _ => {
            problems.push(T0Error::QuantMismatch {
                tensor: "x",
                expected: vec![
                    QuantScheme::None,
                    QuantScheme::PerToken,
                    QuantScheme::PerBlock32,
                ],
                got: x.quant(),
            });
        }
    }

    // Validate weight operand dtype and quant scheme
    let w_scheme = match (w.dtype(), w.quant()) {
        (DType::F16, QuantScheme::None) => None,
        (DType::I8, QuantScheme::PerRow) => Some(SchemeId::I8R),
        (DType::I8, QuantScheme::Scheme(ir_s)) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if !sid.is_native() {
                return Err(T0Error::Format(FormatError::ReservedScheme {
                    scheme: sid.name(),
                    owner: sid.owner_card(),
                }));
            }
            if sid != SchemeId::I8R && sid != SchemeId::I8B128 {
                problems.push(T0Error::QuantMismatch {
                    tensor: "w",
                    expected: vec![
                        QuantScheme::PerRow,
                        QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                        QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                    ],
                    got: w.quant(),
                });
            }
            Some(sid)
        }
        (DType::I4, QuantScheme::Scheme(ir_s)) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if !sid.is_native() {
                return Err(T0Error::Format(FormatError::ReservedScheme {
                    scheme: sid.name(),
                    owner: sid.owner_card(),
                }));
            }
            if sid != SchemeId::I4K {
                problems.push(T0Error::QuantMismatch {
                    tensor: "w",
                    expected: vec![QuantScheme::Scheme(SchemeId::I4K.to_ir())],
                    got: w.quant(),
                });
            }
            Some(sid)
        }
        (DType::E4m3, QuantScheme::Scheme(ir_s)) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if !sid.is_native() {
                return Err(T0Error::Format(FormatError::ReservedScheme {
                    scheme: sid.name(),
                    owner: sid.owner_card(),
                }));
            }
            if sid != SchemeId::E4M3B128 {
                problems.push(T0Error::QuantMismatch {
                    tensor: "w",
                    expected: vec![QuantScheme::Scheme(SchemeId::E4M3B128.to_ir())],
                    got: w.quant(),
                });
            }
            Some(sid)
        }
        _ => {
            problems.push(T0Error::QuantMismatch {
                tensor: "w",
                expected: vec![
                    QuantScheme::None,
                    QuantScheme::PerRow,
                    QuantScheme::Scheme(r9v_ir::SchemeId::new(1)),
                ],
                got: w.quant(),
            });
            None
        }
    };

    T0Error::from_typed_problems(problems)?;

    let m = x.shape()[0];
    let k_x = x.shape()[1];
    let (n, k_w) = if op.transpose_w {
        (w.shape()[1], w.shape()[0])
    } else {
        (w.shape()[0], w.shape()[1])
    };

    let mut problems = Vec::new();

    if k_x != k_w {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "K",
            expected_from: "x",
            expected: k_x,
            tensor: "w",
            got: k_w,
        });
    }

    let k = k_x;

    if y.shape()[0] != m {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "M",
            expected_from: "x",
            expected: m,
            tensor: "y",
            got: y.shape()[0],
        });
    }

    if y.shape()[1] != n {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "N",
            expected_from: "w",
            expected: n,
            tensor: "y",
            got: y.shape()[1],
        });
    }

    if let Some(b) = bias {
        if b.shape()[0] != n {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "N",
                expected_from: "w",
                expected: n,
                tensor: "bias",
                got: b.shape()[0],
            });
        }
    }

    if let Some(r) = residual {
        if r.shape()[0] != m {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "M",
                expected_from: "x",
                expected: m,
                tensor: "residual",
                got: r.shape()[0],
            });
        }
        if r.shape()[1] != n {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "N",
                expected_from: "w",
                expected: n,
                tensor: "residual",
                got: r.shape()[1],
            });
        }
    }

    // Validate activation scale shapes
    match x.quant() {
        QuantScheme::PerToken => {
            if let Some(xs) = x_scale {
                if xs.shape() != [m] {
                    problems.push(T0Error::DimensionMismatch {
                        dim_name: "M",
                        expected_from: "x",
                        expected: m,
                        tensor: "x_scale",
                        got: xs.shape()[0],
                    });
                }
                if xs.dtype() != DType::F32 {
                    problems.push(T0Error::DTypeMismatch {
                        tensor: "x_scale",
                        expected: vec![DType::F32],
                        got: xs.dtype(),
                    });
                }
            } else {
                problems.push(T0Error::MissingOperand {
                    op: "matmul",
                    operand: "x_scale",
                    detail: "x_scale required for PerToken activations".to_string(),
                });
            }
        }
        QuantScheme::PerBlock32 => {
            if !k.is_multiple_of(32) {
                problems.push(T0Error::DimensionMismatch {
                    dim_name: "K",
                    expected_from: "block_size_32",
                    expected: 32,
                    tensor: "x",
                    got: k,
                });
            }
            if let Some(xs) = x_scale {
                if xs.shape() != [m, k / 32] {
                    problems.push(T0Error::DimensionMismatch {
                        dim_name: "K_blocks",
                        expected_from: "k_div_32",
                        expected: k / 32,
                        tensor: "x_scale",
                        got: if xs.rank() > 1 { xs.shape()[1] } else { 0 },
                    });
                }
                if xs.dtype() != DType::F32 {
                    problems.push(T0Error::DTypeMismatch {
                        tensor: "x_scale",
                        expected: vec![DType::F32],
                        got: xs.dtype(),
                    });
                }
            } else {
                problems.push(T0Error::MissingOperand {
                    op: "matmul",
                    operand: "x_scale",
                    detail: "x_scale required for PerBlock32 activations".to_string(),
                });
            }
        }
        _ => {}
    }

    // Validate weight scheme divisibility
    match w_scheme {
        Some(SchemeId::I4K) if !k.is_multiple_of(256) => {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "K",
                expected_from: "superblock_256",
                expected: 256,
                tensor: "w",
                got: k,
            });
        }
        Some(SchemeId::I8B128) | Some(SchemeId::E4M3B128) if !k.is_multiple_of(128) => {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "K",
                expected_from: "block_size_128",
                expected: 128,
                tensor: "w",
                got: k,
            });
        }
        _ => {}
    }

    T0Error::from_typed_problems(problems)?;

    // Validate scale values are finite and non-negative
    if let Some(xs) = x_scale {
        for i in 0..xs.num_elements() {
            let s = xs.read_f32(i);
            if !s.is_finite() || s < 0.0 {
                return Err(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "x_scale",
                    reason: format!(
                        "activation scale at index {i} must be finite and non-negative, got {s}"
                    ),
                });
            }
        }
    }

    // Extract weight bytes and scales
    let is_l1 = w.layout() == LayoutId::L1;
    let w_bytes = match &w.data {
        TensorData::Bytes(_, b) => *b,
        _ => {
            return Err(T0Error::BackingRepresentationMismatch {
                op: "matmul",
                dtype: w.dtype(),
            });
        }
    };

    let superblock_k = match w_scheme {
        Some(SchemeId::I4K) => 256,
        Some(SchemeId::I8B128) | Some(SchemeId::E4M3B128) => 128,
        _ => 16,
    };
    let w_dims = PaddedDims::new(n as u32, k as u32, Some(superblock_k))?;

    let w_scales_slice: &[u8] = if let Some(ws) = w_scale {
        ws.as_bytes().unwrap_or(&[])
    } else if is_l1 {
        let elem_bytes = match w.dtype() {
            DType::F16 => 2,
            DType::I8 | DType::E4m3 => 1,
            DType::I4 => 1,
            _ => 1,
        };
        let values_bytes = if w.dtype() == DType::I4 {
            (w_dims.n_padded() as usize * w_dims.k_padded() as usize) / 2
        } else {
            w_dims.n_padded() as usize * w_dims.k_padded() as usize * elem_bytes
        };
        if w_bytes.len() > values_bytes {
            &w_bytes[values_bytes..]
        } else {
            &[]
        }
    } else {
        &[]
    };

    // Kernel execution branch dispatch
    match (x.dtype(), x.quant(), w_scheme) {
        // --- Branch A1: I8 PerToken x I8_R (full-K i32 accumulate) ---
        (DType::I8, QuantScheme::PerToken, Some(SchemeId::I8R)) => {
            let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                op: "matmul",
                operand: "x_scale",
                detail: "x_scale required for PerToken i8 matmul".to_string(),
            })?;
            let w_row_stride = if !is_l1 { k + 2 } else { 0 };

            for row_m in 0..m {
                let x_s = xs.read_f32(row_m);

                for col_n in 0..n {
                    // Extract row scale for weight row col_n
                    let w_s = if !is_l1 {
                        let scale_offset = col_n * w_row_stride + k;
                        let scale_bytes: [u8; 2] =
                            [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                        I8RowScale::from_bytes(scale_bytes).value(col_n as u64)?
                    } else {
                        let geom = scale_geometry(SchemeId::I8R, Layout::L1, &w_dims)?;
                        let scale_offset =
                            geom.record_offset((col_n / 16) as u64, 0, (col_n % 16) as u32)?
                                as usize;
                        let scale_bytes: [u8; 2] = [
                            w_scales_slice[scale_offset],
                            w_scales_slice[scale_offset + 1],
                        ];
                        I8RowScale::from_bytes(scale_bytes).value(col_n as u64)?
                    };

                    let mut acc_i32 = 0i32;
                    for k_idx in 0..k {
                        let x_val = x.read_i8(row_m * k + k_idx) as i32;
                        let w_val = if !is_l1 {
                            let offset = if op.transpose_w {
                                k_idx * w_row_stride + col_n
                            } else {
                                col_n * w_row_stride + k_idx
                            };
                            w_bytes[offset] as i8 as i32
                        } else {
                            let tile_idx = (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                            let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                            let j = k_idx % 8;
                            let offset = tile_idx * 256 + lane * 8 + j;
                            w_bytes[offset] as i8 as i32
                        };

                        acc_i32 = acc_i32.checked_add(x_val * w_val).ok_or_else(|| {
                            T0Error::ArithmeticOverflow {
                                op: "matmul",
                                detail: format!(
                                    "i32 accumulation overflow at m={row_m}, n={col_n}, k={k_idx}"
                                ),
                            }
                        })?;
                    }

                    // Single scale multiply in f32 per Spec 1 §6.2
                    let acc_f32 = (acc_i32 as f32) * (w_s * x_s);
                    let final_val =
                        apply_epilogue(acc_f32, op.epilogue, row_m, col_n, n, bias, residual);
                    y.write_f32(row_m * n + col_n, final_val);
                }
            }
        }

        // --- Branch A2: I8 PerToken x I8_B128 (block-wise i32 accumulate summed in f32) ---
        (DType::I8, QuantScheme::PerToken, Some(SchemeId::I8B128)) => {
            let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                op: "matmul",
                operand: "x_scale",
                detail: "x_scale required for PerToken i8 matmul".to_string(),
            })?;
            let k_blocks = k / 128;
            let w_row_stride = if !is_l1 { k + k_blocks * 2 } else { 0 };

            for row_m in 0..m {
                let x_s = xs.read_f32(row_m);

                for col_n in 0..n {
                    let mut acc_f32 = 0.0f32;

                    for b in 0..k_blocks {
                        let w_s = if !is_l1 {
                            let scale_offset = col_n * w_row_stride + k + b * 2;
                            let scale_bytes: [u8; 2] =
                                [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                            I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                        } else {
                            let geom = scale_geometry(SchemeId::I8B128, Layout::L1, &w_dims)?;
                            let scale_offset = geom.record_offset(
                                (col_n / 16) as u64,
                                b as u64,
                                (col_n % 16) as u32,
                            )? as usize;
                            let scale_bytes: [u8; 2] = [
                                w_scales_slice[scale_offset],
                                w_scales_slice[scale_offset + 1],
                            ];
                            I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                        };

                        let mut block_acc_i32 = 0i32;
                        for j in 0..128 {
                            let k_idx = b * 128 + j;
                            let x_val = x.read_i8(row_m * k + k_idx) as i32;
                            let w_val = if !is_l1 {
                                let offset = col_n * w_row_stride + k_idx;
                                w_bytes[offset] as i8 as i32
                            } else {
                                let tile_idx =
                                    (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                let offset = tile_idx * 256 + lane * 8 + (k_idx % 8);
                                w_bytes[offset] as i8 as i32
                            };

                            block_acc_i32 = block_acc_i32.checked_add(x_val * w_val).ok_or_else(|| {
                                T0Error::ArithmeticOverflow {
                                    op: "matmul",
                                    detail: format!("i32 accumulation overflow at m={row_m}, n={col_n}, k={k_idx}"),
                                }
                            })?;
                        }

                        acc_f32 += (block_acc_i32 as f32) * (w_s * x_s);
                    }

                    let final_val =
                        apply_epilogue(acc_f32, op.epilogue, row_m, col_n, n, bias, residual);
                    y.write_f32(row_m * n + col_n, final_val);
                }
            }
        }

        // --- Branch A3: I8 PerToken x I4_K (superblock zero-point form s*(x·q) - m*(Σx)) ---
        (DType::I8, QuantScheme::PerToken, Some(SchemeId::I4K)) => {
            let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                op: "matmul",
                operand: "x_scale",
                detail: "x_scale required for PerToken i8 matmul".to_string(),
            })?;
            let k_subblocks = k / 32;
            let k_superblocks = k / 256;
            let values_bytes_per_row = k / 2;
            let w_row_stride = if !is_l1 {
                values_bytes_per_row + k_superblocks * 16
            } else {
                0
            };

            for row_m in 0..m {
                let x_s = xs.read_f32(row_m);

                for col_n in 0..n {
                    let mut acc_f32 = 0.0f32;

                    for b in 0..k_subblocks {
                        let sb = b / 8;
                        let sub = b % 8;

                        let header = if !is_l1 {
                            let header_offset =
                                col_n * w_row_stride + values_bytes_per_row + sb * 16;
                            let header_slice: [u8; 16] = w_bytes[header_offset..header_offset + 16]
                                .try_into()
                                .map_err(|_| T0Error::BufferLengthMismatch {
                                    tensor: "w",
                                    buffer_len: w_bytes.len(),
                                    expected_len: header_offset + 16,
                                    shape: w.shape().to_vec(),
                                })?;
                            I4KSuperblock::from_bytes(&header_slice)
                        } else {
                            let geom = scale_geometry(SchemeId::I4K, Layout::L1, &w_dims)?;
                            let scale_offset = geom.record_offset(
                                (col_n / 16) as u64,
                                sb as u64,
                                (col_n % 16) as u32,
                            )? as usize;
                            let header_slice: [u8; 16] = w_scales_slice
                                [scale_offset..scale_offset + 16]
                                .try_into()
                                .map_err(|_| T0Error::BufferLengthMismatch {
                                    tensor: "w_scales",
                                    buffer_len: w_scales_slice.len(),
                                    expected_len: scale_offset + 16,
                                    shape: w.shape().to_vec(),
                                })?;
                            I4KSuperblock::from_bytes(&header_slice)
                        };

                        let d = header.d_value(sb as u64)?;
                        let dmin = header.dmin_value(sb as u64)?;
                        let sc = header.scales();
                        let mn = header.mins();

                        let s_block = d * (sc[sub] as f32);
                        let m_block = dmin * (mn[sub] as f32);

                        let mut dot_xq = 0i32;
                        let mut sum_x = 0i32;

                        for j in 0..32 {
                            let k_idx = b * 32 + j;
                            let x_val = x.read_i8(row_m * k + k_idx) as i32;
                            let q_val = if !is_l1 {
                                let byte = w_bytes[col_n * w_row_stride + k_idx / 2];
                                if k_idx % 2 == 0 {
                                    (byte & 0x0F) as i32
                                } else {
                                    ((byte >> 4) & 0x0F) as i32
                                }
                            } else {
                                let tile_idx =
                                    (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                let j_tile = k_idx % 8;
                                let offset = tile_idx * 128 + lane * 4 + j_tile / 2;
                                let byte = w_bytes[offset];
                                if j_tile % 2 == 0 {
                                    (byte & 0x0F) as i32
                                } else {
                                    ((byte >> 4) & 0x0F) as i32
                                }
                            };

                            dot_xq = dot_xq.checked_add(x_val * q_val).ok_or_else(|| {
                                T0Error::ArithmeticOverflow {
                                    op: "matmul",
                                    detail: format!(
                                        "dot_xq overflow at m={row_m}, n={col_n}, k={k_idx}"
                                    ),
                                }
                            })?;
                            sum_x = sum_x.checked_add(x_val).ok_or_else(|| {
                                T0Error::ArithmeticOverflow {
                                    op: "matmul",
                                    detail: format!(
                                        "sum_x overflow at m={row_m}, n={col_n}, k={k_idx}"
                                    ),
                                }
                            })?;
                        }

                        // Block contribution in f32 per Spec 1 §6.2
                        let block_val =
                            (s_block * (dot_xq as f32) - m_block * (sum_x as f32)) * x_s;
                        acc_f32 += block_val;
                    }

                    let final_val =
                        apply_epilogue(acc_f32, op.epilogue, row_m, col_n, n, bias, residual);
                    y.write_f32(row_m * n + col_n, final_val);
                }
            }
        }

        // --- Branch B: I8 PerBlock32 x (I8_R, I8_B128, I4_K) ---
        (DType::I8, QuantScheme::PerBlock32, _) => {
            let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                op: "matmul",
                operand: "x_scale",
                detail: "x_scale required for PerBlock32 i8 matmul".to_string(),
            })?;
            let k_blocks32 = k / 32;

            for row_m in 0..m {
                for col_n in 0..n {
                    let mut acc_f32 = 0.0f32;

                    for b in 0..k_blocks32 {
                        let x_b_scale = xs.read_f32(row_m * k_blocks32 + b);

                        match w_scheme {
                            Some(SchemeId::I8R) => {
                                let w_s = if !is_l1 {
                                    let scale_offset = col_n * (k + 2) + k;
                                    let scale_bytes: [u8; 2] =
                                        [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                                    I8RowScale::from_bytes(scale_bytes).value(col_n as u64)?
                                } else {
                                    let geom = scale_geometry(SchemeId::I8R, Layout::L1, &w_dims)?;
                                    let scale_offset = geom.record_offset(
                                        (col_n / 16) as u64,
                                        0,
                                        (col_n % 16) as u32,
                                    )?
                                        as usize;
                                    let scale_bytes: [u8; 2] = [
                                        w_scales_slice[scale_offset],
                                        w_scales_slice[scale_offset + 1],
                                    ];
                                    I8RowScale::from_bytes(scale_bytes).value(col_n as u64)?
                                };

                                let mut block_acc_i32 = 0i32;
                                for j in 0..32 {
                                    let k_idx = b * 32 + j;
                                    let x_val = x.read_i8(row_m * k + k_idx) as i32;
                                    let w_val = if !is_l1 {
                                        w_bytes[col_n * (k + 2) + k_idx] as i8 as i32
                                    } else {
                                        let tile_idx =
                                            (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                        let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                        w_bytes[tile_idx * 256 + lane * 8 + (k_idx % 8)] as i8
                                            as i32
                                    };
                                    block_acc_i32 = block_acc_i32
                                        .checked_add(x_val * w_val)
                                        .ok_or_else(|| T0Error::ArithmeticOverflow {
                                            op: "matmul",
                                            detail: "PerBlock32 i32 overflow".to_string(),
                                        })?;
                                }
                                acc_f32 += (block_acc_i32 as f32) * (w_s * x_b_scale);
                            }
                            Some(SchemeId::I8B128) => {
                                let b128 = b / 4;
                                let w_s = if !is_l1 {
                                    let scale_offset = col_n * (k + (k / 128) * 2) + k + b128 * 2;
                                    let scale_bytes: [u8; 2] =
                                        [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                                    I8Block128Scale::from_bytes(scale_bytes).value(b128 as u64)?
                                } else {
                                    let geom =
                                        scale_geometry(SchemeId::I8B128, Layout::L1, &w_dims)?;
                                    let scale_offset = geom.record_offset(
                                        (col_n / 16) as u64,
                                        b128 as u64,
                                        (col_n % 16) as u32,
                                    )?
                                        as usize;
                                    let scale_bytes: [u8; 2] = [
                                        w_scales_slice[scale_offset],
                                        w_scales_slice[scale_offset + 1],
                                    ];
                                    I8Block128Scale::from_bytes(scale_bytes).value(b128 as u64)?
                                };

                                let mut block_acc_i32 = 0i32;
                                for j in 0..32 {
                                    let k_idx = b * 32 + j;
                                    let x_val = x.read_i8(row_m * k + k_idx) as i32;
                                    let w_val = if !is_l1 {
                                        w_bytes[col_n * (k + (k / 128) * 2) + k_idx] as i8 as i32
                                    } else {
                                        let tile_idx =
                                            (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                        let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                        w_bytes[tile_idx * 256 + lane * 8 + (k_idx % 8)] as i8
                                            as i32
                                    };
                                    block_acc_i32 = block_acc_i32
                                        .checked_add(x_val * w_val)
                                        .ok_or_else(|| T0Error::ArithmeticOverflow {
                                            op: "matmul",
                                            detail: "PerBlock32 i32 overflow".to_string(),
                                        })?;
                                }
                                acc_f32 += (block_acc_i32 as f32) * (w_s * x_b_scale);
                            }
                            Some(SchemeId::I4K) => {
                                let sb = b / 8;
                                let sub = b % 8;
                                let header = if !is_l1 {
                                    let header_offset =
                                        col_n * (k / 2 + (k / 256) * 16) + k / 2 + sb * 16;
                                    let header_slice: [u8; 16] = w_bytes
                                        [header_offset..header_offset + 16]
                                        .try_into()
                                        .map_err(|_| T0Error::BufferLengthMismatch {
                                            tensor: "w",
                                            buffer_len: w_bytes.len(),
                                            expected_len: header_offset + 16,
                                            shape: w.shape().to_vec(),
                                        })?;
                                    I4KSuperblock::from_bytes(&header_slice)
                                } else {
                                    let geom = scale_geometry(SchemeId::I4K, Layout::L1, &w_dims)?;
                                    let scale_offset = geom.record_offset(
                                        (col_n / 16) as u64,
                                        sb as u64,
                                        (col_n % 16) as u32,
                                    )?
                                        as usize;
                                    let header_slice: [u8; 16] = w_scales_slice
                                        [scale_offset..scale_offset + 16]
                                        .try_into()
                                        .map_err(|_| T0Error::BufferLengthMismatch {
                                            tensor: "w_scales",
                                            buffer_len: w_scales_slice.len(),
                                            expected_len: scale_offset + 16,
                                            shape: w.shape().to_vec(),
                                        })?;
                                    I4KSuperblock::from_bytes(&header_slice)
                                };

                                let d = header.d_value(sb as u64)?;
                                let dmin = header.dmin_value(sb as u64)?;
                                let sc = header.scales();
                                let mn = header.mins();

                                let s_block = d * (sc[sub] as f32);
                                let m_block = dmin * (mn[sub] as f32);

                                let mut dot_xq = 0i32;
                                let mut sum_x = 0i32;

                                for j in 0..32 {
                                    let k_idx = b * 32 + j;
                                    let x_val = x.read_i8(row_m * k + k_idx) as i32;
                                    let q_val = if !is_l1 {
                                        let byte =
                                            w_bytes[col_n * (k / 2 + (k / 256) * 16) + k_idx / 2];
                                        if k_idx % 2 == 0 {
                                            (byte & 0x0F) as i32
                                        } else {
                                            ((byte >> 4) & 0x0F) as i32
                                        }
                                    } else {
                                        let tile_idx =
                                            (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                        let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                        let j_tile = k_idx % 8;
                                        let byte = w_bytes[tile_idx * 128 + lane * 4 + j_tile / 2];
                                        if j_tile % 2 == 0 {
                                            (byte & 0x0F) as i32
                                        } else {
                                            ((byte >> 4) & 0x0F) as i32
                                        }
                                    };

                                    dot_xq =
                                        dot_xq.checked_add(x_val * q_val).ok_or_else(|| {
                                            T0Error::ArithmeticOverflow {
                                                op: "matmul",
                                                detail: "dot_xq overflow".to_string(),
                                            }
                                        })?;
                                    sum_x = sum_x.checked_add(x_val).ok_or_else(|| {
                                        T0Error::ArithmeticOverflow {
                                            op: "matmul",
                                            detail: "sum_x overflow".to_string(),
                                        }
                                    })?;
                                }

                                let block_val = (s_block * (dot_xq as f32)
                                    - m_block * (sum_x as f32))
                                    * x_b_scale;
                                acc_f32 += block_val;
                            }
                            _ => {
                                return Err(T0Error::QuantMismatch {
                                    tensor: "w",
                                    expected: vec![
                                        QuantScheme::PerRow,
                                        QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                                        QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                                        QuantScheme::Scheme(SchemeId::I4K.to_ir()),
                                    ],
                                    got: w.quant(),
                                });
                            }
                        }
                    }

                    let final_val =
                        apply_epilogue(acc_f32, op.epilogue, row_m, col_n, n, bias, residual);
                    y.write_f32(row_m * n + col_n, final_val);
                }
            }
        }

        // --- Branch C: E4M3 PerToken x E4M3_B128 (Spec 1 §4.C) ---
        (DType::E4m3, QuantScheme::PerToken, Some(SchemeId::E4M3B128)) => {
            let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                op: "matmul",
                operand: "x_scale",
                detail: "x_scale required for PerToken e4m3 matmul".to_string(),
            })?;
            let k_blocks = k / 128;
            let w_row_stride = if !is_l1 { k + k_blocks * 2 } else { 0 };

            for row_m in 0..m {
                let x_s = xs.read_f32(row_m);

                for col_n in 0..n {
                    let mut acc_f32 = 0.0f32;

                    for b in 0..k_blocks {
                        let w_s = if !is_l1 {
                            let scale_offset = col_n * w_row_stride + k + b * 2;
                            let scale_bytes: [u8; 2] =
                                [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                            E4M3Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                        } else {
                            let geom = scale_geometry(SchemeId::E4M3B128, Layout::L1, &w_dims)?;
                            let scale_offset = geom.record_offset(
                                (col_n / 16) as u64,
                                b as u64,
                                (col_n % 16) as u32,
                            )? as usize;
                            let scale_bytes: [u8; 2] = [
                                w_scales_slice[scale_offset],
                                w_scales_slice[scale_offset + 1],
                            ];
                            E4M3Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                        };

                        let mut unscaled_dot = 0.0f32;
                        for j in 0..128 {
                            let k_idx = b * 128 + j;
                            let x_byte = x.read_byte(row_m * k + k_idx);
                            let x_e4m3 = E4m3::new(x_byte);
                            x_e4m3.check((row_m * k + k_idx) as u64)?;

                            let w_byte = if !is_l1 {
                                w_bytes[col_n * w_row_stride + k_idx]
                            } else {
                                let tile_idx =
                                    (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                w_bytes[tile_idx * 256 + lane * 8 + (k_idx % 8)]
                            };
                            let w_e4m3 = E4m3::new(w_byte);
                            w_e4m3.check(k_idx as u64)?;

                            unscaled_dot += x_e4m3.to_f32() * w_e4m3.to_f32();
                        }

                        acc_f32 += unscaled_dot * (w_s * x_s);
                    }

                    let final_val =
                        apply_epilogue(acc_f32, op.epilogue, row_m, col_n, n, bias, residual);
                    y.write_f32(row_m * n + col_n, final_val);
                }
            }
        }

        // --- Branch D: F16 / BF16 / Float dequant paths (ascending-K accumulation) ---
        _ => {
            // General floating-point path with dequantization
            for row_m in 0..m {
                for col_n in 0..n {
                    let mut acc_f32 = 0.0f32;

                    for k_idx in 0..k {
                        // Dequantize activation element
                        let x_val = match (x.dtype(), x.quant()) {
                            (DType::F16 | DType::Bf16, QuantScheme::None) => {
                                x.read_f32(row_m * k + k_idx)
                            }
                            (DType::I8, QuantScheme::PerToken) => {
                                let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                                    op: "matmul",
                                    operand: "x_scale",
                                    detail: "x_scale required for PerToken activations".to_string(),
                                })?;
                                (x.read_i8(row_m * k + k_idx) as f32) * xs.read_f32(row_m)
                            }
                            (DType::I8, QuantScheme::PerBlock32) => {
                                let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                                    op: "matmul",
                                    operand: "x_scale",
                                    detail: "x_scale required for PerBlock32 activations"
                                        .to_string(),
                                })?;
                                (x.read_i8(row_m * k + k_idx) as f32)
                                    * xs.read_f32(row_m * (k / 32) + k_idx / 32)
                            }
                            (DType::E4m3, QuantScheme::PerToken) => {
                                let xs = x_scale.ok_or_else(|| T0Error::MissingOperand {
                                    op: "matmul",
                                    operand: "x_scale",
                                    detail: "x_scale required for PerToken activations".to_string(),
                                })?;
                                let byte = x.read_byte(row_m * k + k_idx);
                                let e = E4m3::new(byte);
                                e.check((row_m * k + k_idx) as u64)?;
                                e.to_f32() * xs.read_f32(row_m)
                            }
                            _ => x.read_f32(row_m * k + k_idx),
                        };

                        // Dequantize weight element
                        let w_val = match w_scheme {
                            None => {
                                if !is_l1 {
                                    let offset = if op.transpose_w {
                                        (k_idx * n + col_n) * 2
                                    } else {
                                        (col_n * k + k_idx) * 2
                                    };
                                    let bits =
                                        u16::from_le_bytes([w_bytes[offset], w_bytes[offset + 1]]);
                                    f16_to_f32(bits)
                                } else {
                                    let tile_idx =
                                        (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                    let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                    let offset = tile_idx * 512 + (lane * 8 + (k_idx % 8)) * 2;
                                    let bits =
                                        u16::from_le_bytes([w_bytes[offset], w_bytes[offset + 1]]);
                                    f16_to_f32(bits)
                                }
                            }
                            Some(SchemeId::I8R) => {
                                let w_s = if !is_l1 {
                                    let scale_offset = col_n * (k + 2) + k;
                                    let scale_bytes: [u8; 2] =
                                        [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                                    I8RowScale::from_bytes(scale_bytes).value(col_n as u64)?
                                } else {
                                    let geom = scale_geometry(SchemeId::I8R, Layout::L1, &w_dims)?;
                                    let scale_offset = geom.record_offset(
                                        (col_n / 16) as u64,
                                        0,
                                        (col_n % 16) as u32,
                                    )?
                                        as usize;
                                    let scale_bytes: [u8; 2] = [
                                        w_scales_slice[scale_offset],
                                        w_scales_slice[scale_offset + 1],
                                    ];
                                    I8RowScale::from_bytes(scale_bytes).value(col_n as u64)?
                                };
                                let q = if !is_l1 {
                                    w_bytes[col_n * (k + 2) + k_idx] as i8
                                } else {
                                    let tile_idx =
                                        (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                    let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                    w_bytes[tile_idx * 256 + lane * 8 + (k_idx % 8)] as i8
                                };
                                (q as f32) * w_s
                            }
                            Some(SchemeId::I8B128) => {
                                let b = k_idx / 128;
                                let w_s = if !is_l1 {
                                    let scale_offset = col_n * (k + (k / 128) * 2) + k + b * 2;
                                    let scale_bytes: [u8; 2] =
                                        [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                                    I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                                } else {
                                    let geom =
                                        scale_geometry(SchemeId::I8B128, Layout::L1, &w_dims)?;
                                    let scale_offset = geom.record_offset(
                                        (col_n / 16) as u64,
                                        b as u64,
                                        (col_n % 16) as u32,
                                    )?
                                        as usize;
                                    let scale_bytes: [u8; 2] = [
                                        w_scales_slice[scale_offset],
                                        w_scales_slice[scale_offset + 1],
                                    ];
                                    I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                                };
                                let q = if !is_l1 {
                                    w_bytes[col_n * (k + (k / 128) * 2) + k_idx] as i8
                                } else {
                                    let tile_idx =
                                        (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                    let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                    w_bytes[tile_idx * 256 + lane * 8 + (k_idx % 8)] as i8
                                };
                                (q as f32) * w_s
                            }
                            Some(SchemeId::I4K) => {
                                let sb = k_idx / 256;
                                let sub = (k_idx % 256) / 32;
                                let header = if !is_l1 {
                                    let header_offset =
                                        col_n * (k / 2 + (k / 256) * 16) + k / 2 + sb * 16;
                                    let header_slice: [u8; 16] = w_bytes
                                        [header_offset..header_offset + 16]
                                        .try_into()
                                        .map_err(|_| T0Error::BufferLengthMismatch {
                                            tensor: "w",
                                            buffer_len: w_bytes.len(),
                                            expected_len: header_offset + 16,
                                            shape: w.shape().to_vec(),
                                        })?;
                                    I4KSuperblock::from_bytes(&header_slice)
                                } else {
                                    let geom = scale_geometry(SchemeId::I4K, Layout::L1, &w_dims)?;
                                    let scale_offset = geom.record_offset(
                                        (col_n / 16) as u64,
                                        sb as u64,
                                        (col_n % 16) as u32,
                                    )?
                                        as usize;
                                    let header_slice: [u8; 16] = w_scales_slice
                                        [scale_offset..scale_offset + 16]
                                        .try_into()
                                        .map_err(|_| T0Error::BufferLengthMismatch {
                                            tensor: "w_scales",
                                            buffer_len: w_scales_slice.len(),
                                            expected_len: scale_offset + 16,
                                            shape: w.shape().to_vec(),
                                        })?;
                                    I4KSuperblock::from_bytes(&header_slice)
                                };
                                let d = header.d_value(sb as u64)?;
                                let dmin = header.dmin_value(sb as u64)?;
                                let sc = header.scales();
                                let mn = header.mins();

                                let s_block = d * (sc[sub] as f32);
                                let m_block = dmin * (mn[sub] as f32);
                                let q = if !is_l1 {
                                    let byte =
                                        w_bytes[col_n * (k / 2 + (k / 256) * 16) + k_idx / 2];
                                    if k_idx % 2 == 0 {
                                        (byte & 0x0F) as i32
                                    } else {
                                        ((byte >> 4) & 0x0F) as i32
                                    }
                                } else {
                                    let tile_idx =
                                        (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                    let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                    let j_tile = k_idx % 8;
                                    let byte = w_bytes[tile_idx * 128 + lane * 4 + j_tile / 2];
                                    if j_tile % 2 == 0 {
                                        (byte & 0x0F) as i32
                                    } else {
                                        ((byte >> 4) & 0x0F) as i32
                                    }
                                };
                                s_block * (q as f32) - m_block
                            }
                            Some(SchemeId::E4M3B128) => {
                                let b = k_idx / 128;
                                let w_s = if !is_l1 {
                                    let scale_offset = col_n * (k + (k / 128) * 2) + k + b * 2;
                                    let scale_bytes: [u8; 2] =
                                        [w_bytes[scale_offset], w_bytes[scale_offset + 1]];
                                    E4M3Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                                } else {
                                    let geom =
                                        scale_geometry(SchemeId::E4M3B128, Layout::L1, &w_dims)?;
                                    let scale_offset = geom.record_offset(
                                        (col_n / 16) as u64,
                                        b as u64,
                                        (col_n % 16) as u32,
                                    )?
                                        as usize;
                                    let scale_bytes: [u8; 2] = [
                                        w_scales_slice[scale_offset],
                                        w_scales_slice[scale_offset + 1],
                                    ];
                                    E4M3Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                                };
                                let byte = if !is_l1 {
                                    w_bytes[col_n * (k + (k / 128) * 2) + k_idx]
                                } else {
                                    let tile_idx =
                                        (col_n / 16) * w_dims.k_tiles() as usize + (k_idx / 16);
                                    let lane = ((k_idx % 16) / 8) * 16 + (col_n % 16);
                                    w_bytes[tile_idx * 256 + lane * 8 + (k_idx % 8)]
                                };
                                let e = E4m3::new(byte);
                                e.check(k_idx as u64)?;
                                e.to_f32() * w_s
                            }
                            _ => 0.0f32,
                        };

                        acc_f32 += x_val * w_val;
                    }

                    let final_val =
                        apply_epilogue(acc_f32, op.epilogue, row_m, col_n, n, bias, residual);
                    y.write_f32(row_m * n + col_n, final_val);
                }
            }
        }
    }

    Ok(())
}

#[inline(always)]
fn apply_epilogue(
    acc_f32: f32,
    epilogue: Epilogue,
    m: usize,
    n: usize,
    n_dim: usize,
    bias: Option<&TensorView<'_>>,
    residual: Option<&TensorView<'_>>,
) -> f32 {
    match epilogue {
        Epilogue::None => acc_f32,
        Epilogue::Bias => {
            let b_val = bias.map(|b| b.read_f32(n)).unwrap_or(0.0f32);
            acc_f32 + b_val
        }
        Epilogue::Residual => {
            let r_val = residual
                .map(|r| r.read_f32(m * n_dim + n))
                .unwrap_or(0.0f32);
            acc_f32 + r_val
        }
        Epilogue::Act(kind) => eval_activation_f32(acc_f32, kind),
    }
}

/// 64-bit reference implementation of `matmul` for testing (Spec 1 §4.C, §6.1).
#[allow(clippy::too_many_arguments)]
pub fn matmul_f64_reference(
    x_f64: &[f64],
    m: usize,
    k: usize,
    w_f64: &[f64],
    n: usize,
    bias_f64: Option<&[f64]>,
    residual_f64: Option<&[f64]>,
    epilogue: Epilogue,
    transpose_w: bool,
) -> Vec<f64> {
    assert_eq!(x_f64.len(), m * k);
    assert_eq!(w_f64.len(), n * k);
    if let Some(b) = bias_f64 {
        assert_eq!(b.len(), n);
    }
    if let Some(r) = residual_f64 {
        assert_eq!(r.len(), m * n);
    }

    let mut y = vec![0.0f64; m * n];

    for row_m in 0..m {
        for col_n in 0..n {
            let mut acc = 0.0f64;
            for k_idx in 0..k {
                let x_val = x_f64[row_m * k + k_idx];
                let w_val = if transpose_w {
                    w_f64[k_idx * n + col_n]
                } else {
                    w_f64[col_n * k + k_idx]
                };
                acc += x_val * w_val;
            }

            match epilogue {
                Epilogue::None => {}
                Epilogue::Bias => {
                    if let Some(b) = bias_f64 {
                        acc += b[col_n];
                    }
                }
                Epilogue::Residual => {
                    if let Some(r) = residual_f64 {
                        acc += r[row_m * n + col_n];
                    }
                }
                Epilogue::Act(kind) => {
                    acc = crate::activation::eval_activation_f64(acc, kind);
                }
            }

            y[row_m * n + col_n] = acc;
        }
    }

    y
}
