// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 implementation of `embed_gather` op (Spec 1 §4.A, Spec 2 §2–§4, Card A1.6).

use r9v_format::records::{I4KSuperblock, I8Block128Scale, I8RowScale};
use r9v_format::{
    l0_region_bytes, l0_row_offset_bytes, l0_row_stride_bytes, l1_forward_index, scale_geometry,
    FormatError, Layout, PaddedDims, SchemeId,
};
use r9v_ir::{DType, EmbedGatherOp, LayoutId, QuantScheme};

use crate::buffer::{TensorData, TensorView, TensorViewMut};
use crate::dtype::f16_to_f32;
use crate::error::{u64_to_usize, T0Error};

/// Gathers token embeddings from `table` into output `x` (Spec 1 §4.A, Spec 2 §2–§4, Card A1.6).
///
/// Supports both L0 (row-major) and tiled L1 tables, unquantized F16, and native quantized schemes
/// (I8_R, I8_B128, I4_K). Reserved schemes fail closed.
///
/// Signature:
/// - `token_ids`: `[T]` (`u32`)
/// - `table`: `[V, Dm]` (`f16`, `i8`, or `i4`)
/// - `x`: `[T, Dm]` (`out_dtype`)
pub fn embed_gather(
    op: &EmbedGatherOp,
    token_ids: &TensorView<'_>,
    table: &TensorView<'_>,
    x: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    embed_gather_with_scales(op, token_ids, table, table.scale(), x)
}

/// Gathers token embeddings with an explicit optional scales tensor view (Spec 1 §4.A, Card A1.6).
///
/// DECISION(A1.6): embed_gather on L1 tiled tables reconstructs rows by indexing elements via
/// canonical L1 tile/lane arithmetic and scale offsets via ScaleGeometry, producing bit-identical
/// dequantized embeddings to L0 row-major tables per Spec 2 §4 tied-embedding contract.
pub fn embed_gather_with_scales(
    op: &EmbedGatherOp,
    token_ids: &TensorView<'_>,
    table: &TensorView<'_>,
    table_scales: Option<&TensorView<'_>>,
    x: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    token_ids.validate_backing("token_ids")?;
    table.validate_backing("table")?;
    if let Some(s) = table_scales {
        s.validate_backing("table_scales")?;
    }
    x.validate_backing("x")?;

    let mut problems = Vec::new();

    if token_ids.rank() != 1 {
        problems.push(T0Error::RankMismatch {
            tensor: "token_ids",
            expected: 1,
            got: token_ids.rank(),
            shape: token_ids.shape().to_vec(),
        });
    }

    if table.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "table",
            expected: 2,
            got: table.rank(),
            shape: table.shape().to_vec(),
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

    // Validate exact supported layouts
    if token_ids.layout() != LayoutId::CONTIGUOUS && token_ids.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "token_ids",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: token_ids.layout(),
        });
    }

    if table.layout() == LayoutId::L1S
        || (table.layout() != LayoutId::L0
            && table.layout() != LayoutId::L1
            && table.layout() != LayoutId::CONTIGUOUS)
    {
        problems.push(T0Error::LayoutMismatch {
            tensor: "table",
            expected: vec![LayoutId::L0, LayoutId::L1],
            got: table.layout(),
        });
    }

    if let Some(s) = table_scales {
        if s.layout() == LayoutId::L1S
            || (s.layout() != LayoutId::CONTIGUOUS
                && s.layout() != LayoutId::L0
                && s.layout() != LayoutId::L1)
        {
            problems.push(T0Error::LayoutMismatch {
                tensor: "table_scales",
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0, LayoutId::L1],
                got: s.layout(),
            });
        }
    }

    if x.layout() != LayoutId::CONTIGUOUS && x.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "x",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: x.layout(),
        });
    }

    if token_ids.dtype() != DType::U32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "token_ids",
            expected: vec![DType::U32],
            got: token_ids.dtype(),
        });
    }

    if !matches!(table.dtype(), DType::F16 | DType::I8 | DType::I4) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "table",
            expected: vec![DType::F16, DType::I8, DType::I4],
            got: table.dtype(),
        });
    }

    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "out_dtype",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: op.out_dtype,
        });
    }

    if x.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![op.out_dtype],
            got: x.dtype(),
        });
    }

    if !op.scale.is_finite() || op.scale <= 0.0 {
        problems.push(T0Error::InvalidAttribute {
            op: "embed_gather",
            attribute: "scale",
            reason: format!("scale must be finite and positive, got {}", op.scale),
        });
    }

    // Validate SchemeId::from_ir and is_native before dtype matching so every reserved scheme
    // returns FormatError::ReservedScheme regardless of declared dtype.
    if let QuantScheme::Scheme(ir_s) = table.quant() {
        let sid = SchemeId::from_ir(ir_s)?;
        if !sid.is_native() {
            return Err(T0Error::Format(FormatError::ReservedScheme {
                scheme: sid.name(),
                owner: sid.owner_card(),
            }));
        }
    }

    // Identify layout and quantization scheme
    let is_l1 = table.layout() == LayoutId::L1;
    let scheme_res = match table.quant() {
        QuantScheme::None => {
            if table.dtype() != DType::F16 {
                problems.push(T0Error::QuantMismatch {
                    tensor: "table",
                    expected: vec![
                        QuantScheme::PerRow,
                        QuantScheme::Scheme(r9v_ir::SchemeId::new(1)),
                    ],
                    got: table.quant(),
                });
            }
            None
        }
        QuantScheme::PerRow => {
            if table.dtype() != DType::I8 {
                problems.push(T0Error::DTypeMismatch {
                    tensor: "table",
                    expected: vec![DType::I8],
                    got: table.dtype(),
                });
            }
            Some(SchemeId::I8R)
        }
        QuantScheme::Scheme(ir_scheme) => {
            let sid = SchemeId::from_ir(ir_scheme)?;
            match sid {
                SchemeId::I8R | SchemeId::I8B128 => {
                    if table.dtype() != DType::I8 {
                        problems.push(T0Error::DTypeMismatch {
                            tensor: "table",
                            expected: vec![DType::I8],
                            got: table.dtype(),
                        });
                    }
                }
                SchemeId::I4K => {
                    if table.dtype() != DType::I4 {
                        problems.push(T0Error::DTypeMismatch {
                            tensor: "table",
                            expected: vec![DType::I4],
                            got: table.dtype(),
                        });
                    }
                }
                _ => {
                    problems.push(T0Error::QuantMismatch {
                        tensor: "table",
                        expected: vec![
                            QuantScheme::PerRow,
                            QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                            QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                            QuantScheme::Scheme(SchemeId::I4K.to_ir()),
                        ],
                        got: table.quant(),
                    });
                }
            }
            Some(sid)
        }
        other => {
            problems.push(T0Error::QuantMismatch {
                tensor: "table",
                expected: vec![
                    QuantScheme::None,
                    QuantScheme::PerRow,
                    QuantScheme::Scheme(r9v_ir::SchemeId::new(1)),
                ],
                got: other,
            });
            None
        }
    };

    // Validate table_scales presence and dtype per carrier contract
    if scheme_res.is_none() {
        if table_scales.is_some() {
            problems.push(T0Error::InvalidAttribute {
                op: "embed_gather",
                attribute: "table_scales",
                reason: "table_scales provided for unquantized table".to_string(),
            });
        }
    } else if let Some(s) = table_scales {
        match scheme_res {
            Some(SchemeId::I8R) | Some(SchemeId::I8B128) if s.dtype() != DType::F16 => {
                problems.push(T0Error::DTypeMismatch {
                    tensor: "table_scales",
                    expected: vec![DType::F16],
                    got: s.dtype(),
                });
            }
            Some(SchemeId::I4K) if s.dtype() != DType::U32 => {
                problems.push(T0Error::DTypeMismatch {
                    tensor: "table_scales",
                    expected: vec![DType::U32],
                    got: s.dtype(),
                });
            }
            _ => {}
        }
    }

    T0Error::from_typed_problems(problems)?;

    let t_len = token_ids.shape()[0];
    let v_vocab = table.shape()[0];
    let dm = table.shape()[1];

    if t_len == 0 {
        return Err(T0Error::EmptyInput {
            op: "embed_gather",
            tensor: "token_ids",
        });
    }
    if v_vocab == 0 || dm == 0 {
        return Err(T0Error::EmptyInput {
            op: "embed_gather",
            tensor: "table",
        });
    }

    let _t_len_u32 = u32::try_from(t_len).map_err(|_| T0Error::ArithmeticOverflow {
        op: "embed_gather",
        detail: format!("dimension T exceeds u32: {t_len}"),
    })?;
    let v_vocab_u32 = u32::try_from(v_vocab).map_err(|_| T0Error::ArithmeticOverflow {
        op: "embed_gather",
        detail: format!("dimension V exceeds u32: {v_vocab}"),
    })?;
    let dm_u32 = u32::try_from(dm).map_err(|_| T0Error::ArithmeticOverflow {
        op: "embed_gather",
        detail: format!("dimension Dm exceeds u32: {dm}"),
    })?;

    let mut problems = Vec::new();

    if x.shape()[0] != t_len {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "token_ids",
            expected: t_len,
            tensor: "x",
            got: x.shape()[0],
        });
    }

    if x.shape()[1] != dm {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dm",
            expected_from: "table",
            expected: dm,
            tensor: "x",
            got: x.shape()[1],
        });
    }

    match scheme_res {
        Some(SchemeId::I4K) if !dm.is_multiple_of(256) => {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Dm",
                expected_from: "superblock_256",
                expected: 256,
                tensor: "table",
                got: dm,
            });
        }
        Some(SchemeId::I8B128) if !dm.is_multiple_of(128) => {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Dm",
                expected_from: "block_size_128",
                expected: 128,
                tensor: "table",
                got: dm,
            });
        }
        _ => {}
    }

    // Check bounds of every token_id before running
    for pos in 0..t_len {
        let tok = token_ids.read_u32(pos);
        if tok as usize >= v_vocab {
            problems.push(T0Error::TokenOutOfRange {
                op: "embed_gather",
                tensor: "token_ids",
                position: pos,
                token: tok,
                vocab_size: v_vocab,
            });
        }
    }

    T0Error::from_typed_problems(problems)?;

    let superblock_k = match scheme_res {
        Some(SchemeId::I4K) => 256,
        Some(SchemeId::I8B128) => 128,
        _ => 16,
    };
    let dims = PaddedDims::new(v_vocab_u32, dm_u32, Some(superblock_k))?;

    // Get table backing bytes
    let table_bytes = if let Some(b) = table.as_bytes() {
        b
    } else {
        return Err(T0Error::BackingRepresentationMismatch {
            op: "embed_gather",
            dtype: table.dtype(),
        });
    };

    let elem_bytes = match table.dtype() {
        DType::F16 => 2,
        DType::I8 => 1,
        DType::I4 => 1,
        _ => 1,
    };

    let l1_values_bytes = if table.dtype() == DType::I4 {
        (dims.n_padded() as usize)
            .checked_mul(dims.k_padded() as usize)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "embed_gather",
                detail: "padded dimensions overflow usize".to_string(),
            })?
            / 2
    } else {
        (dims.n_padded() as usize)
            .checked_mul(dims.k_padded() as usize)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "embed_gather",
                detail: "padded dimensions overflow usize".to_string(),
            })?
    };

    let l0_values_bytes = if table.dtype() == DType::I4 {
        v_vocab
            .checked_mul(dm)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "embed_gather",
                detail: "dimensions V*Dm overflow usize".to_string(),
            })?
            / 2
    } else {
        v_vocab
            .checked_mul(dm)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "embed_gather",
                detail: "dimensions V*Dm overflow usize".to_string(),
            })?
    };

    let values_bytes = if is_l1 {
        l1_values_bytes
    } else {
        l0_values_bytes
    };

    // DECISION(A1.6): Table-scale carrier contract (Spec 2 §2–§4).
    // Because Spec 2 defines scale records in serialized byte form but r9v-ir::EmbedGatherOp
    // lacks a scale input tensor signature (SI-18), separate scale tensors must be passed as
    // contiguous record tensors (LayoutId::CONTIGUOUS only) backed by raw serialized bytes (TensorData::Bytes).
    // - I8_R, I8_B128: DType::F16 raw-byte backing with shapes:
    //     L0: [V] (for I8_R) or [V, Dm/128] (for I8_B128)
    //     L1: [n_blocks, k_blocks, 16]
    // - I4_K: DType::U32 raw-byte backing (four u32 words per 16-byte record) with shapes:
    //     L0: [V, Dm/256, 4]
    //     L1: [n_blocks, k_blocks, 16, 4]
    // Backing buffers that do not carry raw serialized bytes, or have extra/truncated bytes,
    // or whose shapes do not exactly match the carrier geometry are rejected.
    if let Some(ts) = table_scales {
        if ts.layout() != LayoutId::CONTIGUOUS {
            return Err(T0Error::LayoutMismatch {
                tensor: "table_scales",
                expected: vec![LayoutId::CONTIGUOUS],
                got: ts.layout(),
            });
        }

        let ts_bytes = match ts.data {
            TensorData::Bytes(_, slice) => slice,
            _ => {
                return Err(T0Error::BackingRepresentationMismatch {
                    op: "embed_gather",
                    dtype: ts.dtype(),
                });
            }
        };

        if table_bytes.len() < values_bytes {
            return Err(T0Error::BufferLengthMismatch {
                tensor: "table",
                buffer_len: table_bytes.len(),
                expected_len: values_bytes,
                shape: table.shape().to_vec(),
            });
        }

        if let Some(scheme) = scheme_res {
            if is_l1 {
                let geom = scale_geometry(scheme, Layout::L1, &dims)?;
                let n_blocks = u64_to_usize(geom.n_blocks, "n_blocks")?;
                let k_blocks = u64_to_usize(geom.k_blocks, "k_blocks")?;
                let req_bytes = u64_to_usize(geom.region_bytes, "region_bytes")?;

                match scheme {
                    SchemeId::I8R | SchemeId::I8B128 => {
                        if ts.dtype() != DType::F16 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "table_scales",
                                expected: vec![DType::F16],
                                got: ts.dtype(),
                            });
                        }
                        let expected_shape = [n_blocks, k_blocks, 16];
                        if ts.shape() != expected_shape {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "scale_shape",
                                expected_from: "scale_geometry",
                                expected: geom.records as usize,
                                tensor: "table_scales",
                                got: ts.num_elements(),
                            });
                        }
                    }
                    SchemeId::I4K => {
                        if ts.dtype() != DType::U32 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "table_scales",
                                expected: vec![DType::U32],
                                got: ts.dtype(),
                            });
                        }
                        let expected_shape = [n_blocks, k_blocks, 16, 4];
                        if ts.shape() != expected_shape {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "scale_shape",
                                expected_from: "scale_geometry",
                                expected: (geom.records * 4) as usize,
                                tensor: "table_scales",
                                got: ts.num_elements(),
                            });
                        }
                    }
                    _ => {}
                }

                if ts_bytes.len() != req_bytes {
                    return Err(T0Error::BufferLengthMismatch {
                        tensor: "table_scales",
                        buffer_len: ts_bytes.len(),
                        expected_len: req_bytes,
                        shape: ts.shape().to_vec(),
                    });
                }
            } else {
                match scheme {
                    SchemeId::I8R => {
                        if ts.dtype() != DType::F16 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "table_scales",
                                expected: vec![DType::F16],
                                got: ts.dtype(),
                            });
                        }
                        if ts.shape() != [v_vocab] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "V",
                                expected_from: "table",
                                expected: v_vocab,
                                tensor: "table_scales",
                                got: ts.shape().first().copied().unwrap_or(0),
                            });
                        }
                        let req_bytes = u64_to_usize(
                            l0_region_bytes(v_vocab_u32, 2)?,
                            "table_scales req_bytes",
                        )?;
                        if ts_bytes.len() != req_bytes {
                            return Err(T0Error::BufferLengthMismatch {
                                tensor: "table_scales",
                                buffer_len: ts_bytes.len(),
                                expected_len: req_bytes,
                                shape: ts.shape().to_vec(),
                            });
                        }
                    }
                    SchemeId::I8B128 => {
                        if ts.dtype() != DType::F16 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "table_scales",
                                expected: vec![DType::F16],
                                got: ts.dtype(),
                            });
                        }
                        let k_blocks = dm / 128;
                        if ts.shape() != [v_vocab, k_blocks] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "K_blocks",
                                expected_from: "table",
                                expected: k_blocks,
                                tensor: "table_scales",
                                got: if ts.rank() > 1 {
                                    ts.shape()[1]
                                } else {
                                    ts.shape().first().copied().unwrap_or(0)
                                },
                            });
                        }
                        let stride_u64 = (dm_u32 / 128 * 2) as u64;
                        let req_bytes = u64_to_usize(
                            l0_region_bytes(v_vocab_u32, stride_u64)?,
                            "table_scales req_bytes",
                        )?;
                        if ts_bytes.len() != req_bytes {
                            return Err(T0Error::BufferLengthMismatch {
                                tensor: "table_scales",
                                buffer_len: ts_bytes.len(),
                                expected_len: req_bytes,
                                shape: ts.shape().to_vec(),
                            });
                        }
                    }
                    SchemeId::I4K => {
                        if ts.dtype() != DType::U32 {
                            return Err(T0Error::DTypeMismatch {
                                tensor: "table_scales",
                                expected: vec![DType::U32],
                                got: ts.dtype(),
                            });
                        }
                        let k_superblocks = dm / 256;
                        if ts.shape() != [v_vocab, k_superblocks, 4] {
                            return Err(T0Error::DimensionMismatch {
                                dim_name: "K_superblocks",
                                expected_from: "table",
                                expected: k_superblocks,
                                tensor: "table_scales",
                                got: if ts.rank() > 1 {
                                    ts.shape()[1]
                                } else {
                                    ts.shape().first().copied().unwrap_or(0)
                                },
                            });
                        }
                        let stride_u64 = (dm_u32 / 256 * 16) as u64;
                        let req_bytes = u64_to_usize(
                            l0_region_bytes(v_vocab_u32, stride_u64)?,
                            "table_scales req_bytes",
                        )?;
                        if ts_bytes.len() != req_bytes {
                            return Err(T0Error::BufferLengthMismatch {
                                tensor: "table_scales",
                                buffer_len: ts_bytes.len(),
                                expected_len: req_bytes,
                                shape: ts.shape().to_vec(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    } else {
        // Inline scales or unquantized
        if let Some(scheme) = scheme_res {
            if is_l1 {
                let geom = scale_geometry(scheme, Layout::L1, &dims)?;
                let req_bytes = l1_values_bytes
                    .checked_add(u64_to_usize(geom.region_bytes, "region_bytes")?)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "embed_gather",
                        detail: "l1_values_bytes + geom.region_bytes overflow".to_string(),
                    })?;
                if table_bytes.len() < req_bytes {
                    return Err(T0Error::BufferLengthMismatch {
                        tensor: "table",
                        buffer_len: table_bytes.len(),
                        expected_len: req_bytes,
                        shape: table.shape().to_vec(),
                    });
                }
            } else {
                let stride = match scheme {
                    SchemeId::I8R => l0_row_stride_bytes(dm_u32, 1, 1, 2)?,
                    SchemeId::I8B128 => l0_row_stride_bytes(dm_u32, 1, dm_u32 / 128, 2)?,
                    SchemeId::I4K => l0_row_stride_bytes(dm_u32 / 2, 1, dm_u32 / 256, 16)?,
                    _ => dm_u32 as u64,
                };
                let req_bytes =
                    u64_to_usize(l0_region_bytes(v_vocab_u32, stride)?, "l0_region_bytes")?;
                if table_bytes.len() < req_bytes {
                    return Err(T0Error::BufferLengthMismatch {
                        tensor: "table",
                        buffer_len: table_bytes.len(),
                        expected_len: req_bytes,
                        shape: table.shape().to_vec(),
                    });
                }
            }
        } else if table_bytes.len() < values_bytes {
            return Err(T0Error::BufferLengthMismatch {
                tensor: "table",
                buffer_len: table_bytes.len(),
                expected_len: values_bytes,
                shape: table.shape().to_vec(),
            });
        }
    }

    let scales_slice: &[u8] = if let Some(s) = table_scales {
        s.as_bytes()
            .expect("table_scales raw-byte backing verified")
    } else if is_l1 {
        if table_bytes.len() > l1_values_bytes {
            &table_bytes[l1_values_bytes..]
        } else {
            &[]
        }
    } else {
        &[]
    };

    let mut row_f32 = vec![0.0f32; dm];

    if !is_l1 {
        // L0 (row-major) layout
        match scheme_res {
            None => {
                let row_stride = dm * 2;
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = u64_to_usize(
                        l0_row_offset_bytes(v as u32, row_stride as u64)?,
                        "row_offset",
                    )?;
                    for d in 0..dm {
                        let offset = row_offset + d * 2;
                        let bytes: [u8; 2] = table_bytes
                            .get(offset..offset + 2)
                            .and_then(|s| s.try_into().ok())
                            .ok_or_else(|| T0Error::BufferLengthMismatch {
                                tensor: "table",
                                buffer_len: table_bytes.len(),
                                expected_len: offset + 2,
                                shape: table.shape().to_vec(),
                            })?;
                        let val = f16_to_f32(u16::from_le_bytes(bytes)) * op.scale;
                        x.write_f32(t * dm + d, val);
                    }
                }
            }
            Some(SchemeId::I8R) => {
                let row_stride = if table_scales.is_some() { dm } else { dm + 2 };
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = u64_to_usize(
                        l0_row_offset_bytes(v as u32, row_stride as u64)?,
                        "row_offset",
                    )?;
                    let scale = if table_scales.is_some() {
                        let scale_offset = v * 2;
                        let scale_bytes: [u8; 2] = scales_slice
                            .get(scale_offset..scale_offset + 2)
                            .and_then(|s| s.try_into().ok())
                            .ok_or_else(|| T0Error::BufferLengthMismatch {
                                tensor: "table_scales",
                                buffer_len: scales_slice.len(),
                                expected_len: scale_offset + 2,
                                shape: table.shape().to_vec(),
                            })?;
                        I8RowScale::from_bytes(scale_bytes).value(v as u64)?
                    } else {
                        let scale_offset = row_offset + dm;
                        let scale_bytes: [u8; 2] = table_bytes
                            .get(scale_offset..scale_offset + 2)
                            .and_then(|s| s.try_into().ok())
                            .ok_or_else(|| T0Error::BufferLengthMismatch {
                                tensor: "table",
                                buffer_len: table_bytes.len(),
                                expected_len: scale_offset + 2,
                                shape: table.shape().to_vec(),
                            })?;
                        I8RowScale::from_bytes(scale_bytes).value(v as u64)?
                    };

                    for d in 0..dm {
                        let offset = row_offset + d;
                        let byte = table_bytes.get(offset).copied().ok_or_else(|| {
                            T0Error::BufferLengthMismatch {
                                tensor: "table",
                                buffer_len: table_bytes.len(),
                                expected_len: offset + 1,
                                shape: table.shape().to_vec(),
                            }
                        })?;
                        let q = byte as i8;
                        let val = (q as f32) * scale * op.scale;
                        x.write_f32(t * dm + d, val);
                    }
                }
            }
            Some(SchemeId::I8B128) => {
                let k_blocks = dm / 128;
                let row_stride = if table_scales.is_some() {
                    dm
                } else {
                    dm + k_blocks * 2
                };
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = u64_to_usize(
                        l0_row_offset_bytes(v as u32, row_stride as u64)?,
                        "row_offset",
                    )?;
                    for b in 0..k_blocks {
                        let scale = if table_scales.is_some() {
                            let scale_offset = (v * k_blocks + b) * 2;
                            let scale_bytes: [u8; 2] = scales_slice
                                .get(scale_offset..scale_offset + 2)
                                .and_then(|s| s.try_into().ok())
                                .ok_or_else(|| T0Error::BufferLengthMismatch {
                                    tensor: "table_scales",
                                    buffer_len: scales_slice.len(),
                                    expected_len: scale_offset + 2,
                                    shape: table.shape().to_vec(),
                                })?;
                            I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                        } else {
                            let scale_offset = row_offset + dm + b * 2;
                            let scale_bytes: [u8; 2] = table_bytes
                                .get(scale_offset..scale_offset + 2)
                                .and_then(|s| s.try_into().ok())
                                .ok_or_else(|| T0Error::BufferLengthMismatch {
                                    tensor: "table",
                                    buffer_len: table_bytes.len(),
                                    expected_len: scale_offset + 2,
                                    shape: table.shape().to_vec(),
                                })?;
                            I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?
                        };

                        for j in 0..128 {
                            let d = b * 128 + j;
                            let offset = row_offset + d;
                            let byte = table_bytes.get(offset).copied().ok_or_else(|| {
                                T0Error::BufferLengthMismatch {
                                    tensor: "table",
                                    buffer_len: table_bytes.len(),
                                    expected_len: offset + 1,
                                    shape: table.shape().to_vec(),
                                }
                            })?;
                            let q = byte as i8;
                            let val = (q as f32) * scale * op.scale;
                            x.write_f32(t * dm + d, val);
                        }
                    }
                }
            }
            Some(SchemeId::I4K) => {
                let k_superblocks = dm / 256;
                let values_bytes_per_row = dm / 2;
                let row_stride = if table_scales.is_some() {
                    values_bytes_per_row
                } else {
                    values_bytes_per_row + k_superblocks * 16
                };
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = u64_to_usize(
                        l0_row_offset_bytes(v as u32, row_stride as u64)?,
                        "row_offset",
                    )?;
                    for sb in 0..k_superblocks {
                        let header = if table_scales.is_some() {
                            let header_offset = (v * k_superblocks + sb) * 16;
                            let header_slice: [u8; 16] = scales_slice
                                .get(header_offset..header_offset + 16)
                                .and_then(|s| s.try_into().ok())
                                .ok_or_else(|| T0Error::BufferLengthMismatch {
                                    tensor: "table_scales",
                                    buffer_len: scales_slice.len(),
                                    expected_len: header_offset + 16,
                                    shape: table.shape().to_vec(),
                                })?;
                            I4KSuperblock::from_bytes(&header_slice)
                        } else {
                            let header_offset = row_offset + values_bytes_per_row + sb * 16;
                            let header_slice: [u8; 16] = table_bytes
                                .get(header_offset..header_offset + 16)
                                .and_then(|s| s.try_into().ok())
                                .ok_or_else(|| T0Error::BufferLengthMismatch {
                                    tensor: "table",
                                    buffer_len: table_bytes.len(),
                                    expected_len: header_offset + 16,
                                    shape: table.shape().to_vec(),
                                })?;
                            I4KSuperblock::from_bytes(&header_slice)
                        };

                        let d = header.d_value(sb as u64)?;
                        let dmin = header.dmin_value(sb as u64)?;
                        let sc = header.scales();
                        let mn = header.mins();
                        for sub in 0..8 {
                            let s_block = d * (sc[sub] as f32);
                            let m_block = dmin * (mn[sub] as f32);
                            for j in 0..32 {
                                let d_idx = sb * 256 + sub * 32 + j;
                                let offset = row_offset + d_idx / 2;
                                let byte = table_bytes.get(offset).copied().ok_or_else(|| {
                                    T0Error::BufferLengthMismatch {
                                        tensor: "table",
                                        buffer_len: table_bytes.len(),
                                        expected_len: offset + 1,
                                        shape: table.shape().to_vec(),
                                    }
                                })?;
                                let q = if d_idx % 2 == 0 {
                                    byte & 0x0F
                                } else {
                                    (byte >> 4) & 0x0F
                                };
                                let val = (s_block * (q as f32) - m_block) * op.scale;
                                x.write_f32(t * dm + d_idx, val);
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(T0Error::QuantMismatch {
                    tensor: "table",
                    expected: vec![
                        QuantScheme::None,
                        QuantScheme::PerRow,
                        QuantScheme::Scheme(r9v_ir::SchemeId::new(1)),
                    ],
                    got: table.quant(),
                });
            }
        }
    } else {
        // L1 (tiled) layout
        for t in 0..t_len {
            let v = token_ids.read_u32(t) as usize;
            let nb = (v / 16) as u64;
            let row_in_tile = (v % 16) as u32;

            match scheme_res {
                None => {
                    // F16 L1
                    for d in 0..dm {
                        let elem_idx = l1_forward_index(v as u32, d as u32, &dims)?;
                        let offset = u64_to_usize(
                            elem_idx
                                .checked_mul(2)
                                .ok_or_else(|| T0Error::ArithmeticOverflow {
                                    op: "embed_gather",
                                    detail: "L1 offset overflow".to_string(),
                                })?,
                            "l1_f16_offset",
                        )?;
                        let bytes: [u8; 2] = table_bytes
                            .get(offset..offset + 2)
                            .and_then(|s| s.try_into().ok())
                            .ok_or_else(|| T0Error::BufferLengthMismatch {
                                tensor: "table",
                                buffer_len: table_bytes.len(),
                                expected_len: offset + 2,
                                shape: table.shape().to_vec(),
                            })?;
                        let bits = u16::from_le_bytes(bytes);
                        row_f32[d] = f16_to_f32(bits);
                    }
                }
                Some(SchemeId::I8R) => {
                    let geom = scale_geometry(SchemeId::I8R, Layout::L1, &dims)?;
                    let scale_offset =
                        u64_to_usize(geom.record_offset(nb, 0, row_in_tile)?, "record_offset")?;
                    let scale_bytes: [u8; 2] = scales_slice
                        .get(scale_offset..scale_offset + 2)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| T0Error::BufferLengthMismatch {
                            tensor: if table_scales.is_some() {
                                "table_scales"
                            } else {
                                "table"
                            },
                            buffer_len: scales_slice.len(),
                            expected_len: scale_offset + 2,
                            shape: table.shape().to_vec(),
                        })?;
                    let scale = I8RowScale::from_bytes(scale_bytes).value(v as u64)?;
                    for d in 0..dm {
                        let offset = u64_to_usize(
                            l1_forward_index(v as u32, d as u32, &dims)?,
                            "l1_i8_offset",
                        )?;
                        let byte = table_bytes.get(offset).copied().ok_or_else(|| {
                            T0Error::BufferLengthMismatch {
                                tensor: "table",
                                buffer_len: table_bytes.len(),
                                expected_len: offset + 1,
                                shape: table.shape().to_vec(),
                            }
                        })?;
                        let q = byte as i8;
                        row_f32[d] = (q as f32) * scale;
                    }
                }
                Some(SchemeId::I8B128) => {
                    let geom = scale_geometry(SchemeId::I8B128, Layout::L1, &dims)?;
                    let k_blocks = dm / 128;
                    for b in 0..k_blocks {
                        let scale_offset = u64_to_usize(
                            geom.record_offset(nb, b as u64, row_in_tile)?,
                            "record_offset",
                        )?;
                        let scale_bytes: [u8; 2] = scales_slice
                            .get(scale_offset..scale_offset + 2)
                            .and_then(|s| s.try_into().ok())
                            .ok_or_else(|| T0Error::BufferLengthMismatch {
                                tensor: if table_scales.is_some() {
                                    "table_scales"
                                } else {
                                    "table"
                                },
                                buffer_len: scales_slice.len(),
                                expected_len: scale_offset + 2,
                                shape: table.shape().to_vec(),
                            })?;
                        let scale = I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?;
                        for j_outer in 0..128 {
                            let d = b * 128 + j_outer;
                            let offset = u64_to_usize(
                                l1_forward_index(v as u32, d as u32, &dims)?,
                                "l1_i8_offset",
                            )?;
                            let byte = table_bytes.get(offset).copied().ok_or_else(|| {
                                T0Error::BufferLengthMismatch {
                                    tensor: "table",
                                    buffer_len: table_bytes.len(),
                                    expected_len: offset + 1,
                                    shape: table.shape().to_vec(),
                                }
                            })?;
                            let q = byte as i8;
                            row_f32[d] = (q as f32) * scale;
                        }
                    }
                }
                Some(SchemeId::I4K) => {
                    let geom = scale_geometry(SchemeId::I4K, Layout::L1, &dims)?;
                    let k_superblocks = dm / 256;
                    for sb in 0..k_superblocks {
                        let scale_offset = u64_to_usize(
                            geom.record_offset(nb, sb as u64, row_in_tile)?,
                            "record_offset",
                        )?;
                        let header_slice: [u8; 16] = scales_slice
                            .get(scale_offset..scale_offset + 16)
                            .and_then(|s| s.try_into().ok())
                            .ok_or_else(|| T0Error::BufferLengthMismatch {
                                tensor: if table_scales.is_some() {
                                    "table_scales"
                                } else {
                                    "table"
                                },
                                buffer_len: scales_slice.len(),
                                expected_len: scale_offset + 16,
                                shape: table.shape().to_vec(),
                            })?;
                        let header = I4KSuperblock::from_bytes(&header_slice);
                        let d = header.d_value(sb as u64)?;
                        let dmin = header.dmin_value(sb as u64)?;
                        let sc = header.scales();
                        let mn = header.mins();
                        for sub in 0..8 {
                            let s_block = d * (sc[sub] as f32);
                            let m_block = dmin * (mn[sub] as f32);
                            for j in 0..32 {
                                let d_idx = sb * 256 + sub * 32 + j;
                                let elem_idx = l1_forward_index(v as u32, d_idx as u32, &dims)?;
                                let offset = u64_to_usize(elem_idx / 2, "l1_i4_offset")?;
                                let byte = table_bytes.get(offset).copied().ok_or_else(|| {
                                    T0Error::BufferLengthMismatch {
                                        tensor: "table",
                                        buffer_len: table_bytes.len(),
                                        expected_len: offset + 1,
                                        shape: table.shape().to_vec(),
                                    }
                                })?;
                                let q = if elem_idx % 2 == 0 {
                                    byte & 0x0F
                                } else {
                                    (byte >> 4) & 0x0F
                                };
                                row_f32[d_idx] = s_block * (q as f32) - m_block;
                            }
                        }
                    }
                }
                _ => {
                    return Err(T0Error::QuantMismatch {
                        tensor: "table",
                        expected: vec![
                            QuantScheme::None,
                            QuantScheme::PerRow,
                            QuantScheme::Scheme(r9v_ir::SchemeId::new(1)),
                        ],
                        got: table.quant(),
                    });
                }
            }

            for d in 0..dm {
                let val = row_f32[d] * op.scale;
                x.write_f32(t * dm + d, val);
            }
        }
    }

    Ok(())
}

/// 64-bit reference implementation of `embed_gather` for testing (Spec 1 §4.A, §6.1).
pub fn embed_gather_f64_reference(
    token_ids: &[u32],
    table_f64: &[f64],
    v: usize,
    dm: usize,
    scale: f64,
) -> Vec<f64> {
    assert_eq!(table_f64.len(), v * dm);
    let t_len = token_ids.len();
    let mut x = Vec::with_capacity(t_len * dm);
    for &tok in token_ids {
        let r = tok as usize;
        assert!(r < v, "token id {r} out of vocab bounds 0..{v}");
        for d in 0..dm {
            x.push(table_f64[r * dm + d] * scale);
        }
    }
    x
}
