// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 implementation of `embed_gather` op (Spec 1 §4.A, Spec 2 §2–§4, Card A1.6).

use r9v_format::records::{I4KSuperblock, I8Block128Scale, I8RowScale};
use r9v_format::{scale_geometry, FormatError, Layout, PaddedDims, SchemeId};
use r9v_ir::{DType, EmbedGatherOp, LayoutId, QuantScheme};

use crate::buffer::{TensorView, TensorViewMut};
use crate::dtype::f16_to_f32;
use crate::error::T0Error;

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

    T0Error::from_typed_problems(problems)?;

    let t_len = token_ids.shape()[0];
    let v_vocab = table.shape()[0];
    let dm = table.shape()[1];

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

    // Identify layout and quantization scheme
    let is_l1 = table.layout() == LayoutId::L1;
    let scheme_res = match table.quant() {
        QuantScheme::None => {
            if table.dtype() != DType::F16 {
                return Err(T0Error::QuantMismatch {
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
        QuantScheme::PerRow => Some(SchemeId::I8R),
        QuantScheme::Scheme(ir_scheme) => {
            let sid = SchemeId::from_ir(ir_scheme)?;
            if !sid.is_native() {
                return Err(T0Error::Format(FormatError::ReservedScheme {
                    scheme: sid.name(),
                    owner: sid.owner_card(),
                }));
            }
            Some(sid)
        }
        other => {
            return Err(T0Error::QuantMismatch {
                tensor: "table",
                expected: vec![
                    QuantScheme::None,
                    QuantScheme::PerRow,
                    QuantScheme::Scheme(r9v_ir::SchemeId::new(1)),
                ],
                got: other,
            });
        }
    };

    // Get table backing bytes
    let table_bytes = if let Some(b) = table.as_bytes() {
        b
    } else {
        return Err(T0Error::BackingRepresentationMismatch {
            op: "embed_gather",
            dtype: table.dtype(),
        });
    };

    // Buffer for decoding one row into f32
    let mut row_f32 = vec![0.0f32; dm];

    if !is_l1 {
        // L0 (row-major) layout
        match scheme_res {
            None => {
                // Unquantized F16
                let row_stride = dm * 2;
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = v * row_stride;
                    for d in 0..dm {
                        let offset = row_offset + d * 2;
                        let bits =
                            u16::from_le_bytes([table_bytes[offset], table_bytes[offset + 1]]);
                        let val = f16_to_f32(bits) * op.scale;
                        x.write_f32(t * dm + d, val);
                    }
                }
            }
            Some(SchemeId::I8R) => {
                let row_stride = dm + 2;
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = v * row_stride;
                    let scale_offset = row_offset + dm;
                    let scale_bytes: [u8; 2] =
                        [table_bytes[scale_offset], table_bytes[scale_offset + 1]];
                    let scale = I8RowScale::from_bytes(scale_bytes).value(v as u64)?;
                    for d in 0..dm {
                        let q = table_bytes[row_offset + d] as i8;
                        let val = (q as f32) * scale * op.scale;
                        x.write_f32(t * dm + d, val);
                    }
                }
            }
            Some(SchemeId::I8B128) => {
                if !dm.is_multiple_of(128) {
                    return Err(T0Error::DimensionMismatch {
                        dim_name: "Dm",
                        expected_from: "block_size_128",
                        expected: 128,
                        tensor: "table",
                        got: dm,
                    });
                }
                let k_blocks = dm / 128;
                let row_stride = dm + k_blocks * 2;
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = v * row_stride;
                    let scales_base = row_offset + dm;
                    for b in 0..k_blocks {
                        let scale_offset = scales_base + b * 2;
                        let scale_bytes: [u8; 2] =
                            [table_bytes[scale_offset], table_bytes[scale_offset + 1]];
                        let scale = I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?;
                        for j in 0..128 {
                            let d = b * 128 + j;
                            let q = table_bytes[row_offset + d] as i8;
                            let val = (q as f32) * scale * op.scale;
                            x.write_f32(t * dm + d, val);
                        }
                    }
                }
            }
            Some(SchemeId::I4K) => {
                if !dm.is_multiple_of(256) {
                    return Err(T0Error::DimensionMismatch {
                        dim_name: "Dm",
                        expected_from: "superblock_256",
                        expected: 256,
                        tensor: "table",
                        got: dm,
                    });
                }
                let k_superblocks = dm / 256;
                let values_bytes_per_row = dm / 2;
                let row_stride = values_bytes_per_row + k_superblocks * 16;
                for t in 0..t_len {
                    let v = token_ids.read_u32(t) as usize;
                    let row_offset = v * row_stride;
                    let scales_base = row_offset + values_bytes_per_row;
                    for sb in 0..k_superblocks {
                        let header_offset = scales_base + sb * 16;
                        let header_slice: [u8; 16] = table_bytes[header_offset..header_offset + 16]
                            .try_into()
                            .map_err(|_| T0Error::BufferLengthMismatch {
                                tensor: "table",
                                buffer_len: table_bytes.len(),
                                expected_len: header_offset + 16,
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
                                let byte = table_bytes[row_offset + d_idx / 2];
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
        let superblock_k = match scheme_res {
            Some(SchemeId::I4K) => 256,
            Some(SchemeId::I8B128) => 128,
            _ => 16,
        };
        let dims = PaddedDims::new(v_vocab as u32, dm as u32, Some(superblock_k))?;

        // Find scale slice
        let scales_slice: &[u8] = if let Some(s) = table_scales {
            s.as_bytes().unwrap_or(&[])
        } else {
            let elem_bytes = match table.dtype() {
                DType::F16 => 2,
                DType::I8 => 1,
                DType::I4 => 1,
                _ => 1,
            };
            let values_bytes = if table.dtype() == DType::I4 {
                (dims.n_padded() as usize * dims.k_padded() as usize) / 2
            } else {
                dims.n_padded() as usize * dims.k_padded() as usize * elem_bytes
            };
            if table_bytes.len() > values_bytes {
                &table_bytes[values_bytes..]
            } else {
                &[]
            }
        };

        for t in 0..t_len {
            let v = token_ids.read_u32(t) as usize;
            let nb = (v / 16) as u64;
            let row_in_tile = (v % 16) as u32;

            match scheme_res {
                None => {
                    // F16 L1
                    for d in 0..dm {
                        let tile_idx = (v / 16) * dims.k_tiles() as usize + (d / 16);
                        let lane = ((d % 16) / 8) * 16 + (v % 16);
                        let j = d % 8;
                        let offset = tile_idx * 512 + (lane * 8 + j) * 2;
                        let bits =
                            u16::from_le_bytes([table_bytes[offset], table_bytes[offset + 1]]);
                        row_f32[d] = f16_to_f32(bits);
                    }
                }
                Some(SchemeId::I8R) => {
                    let geom = scale_geometry(SchemeId::I8R, Layout::L1, &dims)?;
                    let scale_offset = geom.record_offset(nb, 0, row_in_tile)? as usize;
                    let scale_bytes: [u8; 2] =
                        [scales_slice[scale_offset], scales_slice[scale_offset + 1]];
                    let scale = I8RowScale::from_bytes(scale_bytes).value(v as u64)?;
                    for d in 0..dm {
                        let tile_idx = (v / 16) * dims.k_tiles() as usize + (d / 16);
                        let lane = ((d % 16) / 8) * 16 + (v % 16);
                        let j = d % 8;
                        let offset = tile_idx * 256 + lane * 8 + j;
                        let q = table_bytes[offset] as i8;
                        row_f32[d] = (q as f32) * scale;
                    }
                }
                Some(SchemeId::I8B128) => {
                    let geom = scale_geometry(SchemeId::I8B128, Layout::L1, &dims)?;
                    let k_blocks = dm / 128;
                    for b in 0..k_blocks {
                        let scale_offset = geom.record_offset(nb, b as u64, row_in_tile)? as usize;
                        let scale_bytes: [u8; 2] =
                            [scales_slice[scale_offset], scales_slice[scale_offset + 1]];
                        let scale = I8Block128Scale::from_bytes(scale_bytes).value(b as u64)?;
                        for j_outer in 0..128 {
                            let d = b * 128 + j_outer;
                            let tile_idx = (v / 16) * dims.k_tiles() as usize + (d / 16);
                            let lane = ((d % 16) / 8) * 16 + (v % 16);
                            let j = d % 8;
                            let offset = tile_idx * 256 + lane * 8 + j;
                            let q = table_bytes[offset] as i8;
                            row_f32[d] = (q as f32) * scale;
                        }
                    }
                }
                Some(SchemeId::I4K) => {
                    let geom = scale_geometry(SchemeId::I4K, Layout::L1, &dims)?;
                    let k_superblocks = dm / 256;
                    for sb in 0..k_superblocks {
                        let scale_offset = geom.record_offset(nb, sb as u64, row_in_tile)? as usize;
                        let header_slice: [u8; 16] = scales_slice[scale_offset..scale_offset + 16]
                            .try_into()
                            .map_err(|_| T0Error::BufferLengthMismatch {
                                tensor: "table_scales",
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
                                let tile_idx = (v / 16) * dims.k_tiles() as usize + (d_idx / 16);
                                let lane = ((d_idx % 16) / 8) * 16 + (v % 16);
                                let j_tile = d_idx % 8;
                                let offset = tile_idx * 128 + lane * 4 + j_tile / 2;
                                let byte = table_bytes[offset];
                                let q = if j_tile % 2 == 0 {
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
