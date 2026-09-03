// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
use r9v_format::records::I4KSuperblock;
use r9v_format::{scale_geometry, Layout, PaddedDims, SchemeId};
use r9v_ir::{DType, EmbedGatherOp, LayoutId, QuantScheme};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::f32_to_f16;
use r9v_t0::embed_gather::{embed_gather, embed_gather_f64_reference};
use r9v_t0::error::T0Error;

#[test]
fn test_embed_gather_f16_l0_and_l1_equivalence() {
    let v = 32; // 2 tiles of 16
    let dm = 16; // 1 tile of 16
    let t = 4;
    let tokens = vec![5u32, 18, 0, 31];

    // Create table data in f32/f64
    let mut table_f64 = vec![0.0f64; v * dm];
    for r in 0..v {
        for c in 0..dm {
            table_f64[r * dm + c] = (r as f64 * 0.5) - (c as f64 * 0.25);
        }
    }

    // Build L0 byte buffer: row-major [V, Dm] of f16 (2 bytes each)
    let mut l0_bytes = vec![0u8; v * dm * 2];
    for r in 0..v {
        for c in 0..dm {
            let bits = f32_to_f16(table_f64[r * dm + c] as f32);
            let offset = (r * dm + c) * 2;
            l0_bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }

    // Build L1 byte buffer: tiled 16x16
    let dims = PaddedDims::new(v as u32, dm as u32, Some(16)).unwrap();
    let mut l1_bytes = vec![0u8; v * dm * 2];
    for r in 0..v {
        for c in 0..dm {
            let bits = f32_to_f16(table_f64[r * dm + c] as f32);
            let tile_idx = (r / 16) * dims.k_tiles() as usize + (c / 16);
            let lane = ((c % 16) / 8) * 16 + (r % 16);
            let j = c % 8;
            let offset = tile_idx * 512 + (lane * 8 + j) * 2;
            l1_bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }

    let tok_buf = TypedBuffer::from_u32(&[t], &tokens);
    let l0_table = TypedBuffer::from_bytes(&[v, dm], DType::F16, &l0_bytes);
    let l1_table =
        TypedBuffer::from_bytes(&[v, dm], DType::F16, &l1_bytes).with_layout(LayoutId::L1);

    let mut y_l0 = TypedBuffer::zeros(&[t, dm], DType::F32);
    let mut y_l1 = TypedBuffer::zeros(&[t, dm], DType::F32);

    let op = EmbedGatherOp {
        scale: 1.5,
        out_dtype: DType::F32,
    };

    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l0_table.as_view(),
        &mut y_l0.as_view_mut(),
    )
    .unwrap();
    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l1_table.as_view(),
        &mut y_l1.as_view_mut(),
    )
    .unwrap();

    let expected_f64 = embed_gather_f64_reference(&tokens, &table_f64, v, dm, 1.5);

    let s_l0 = y_l0.to_f32_vec();
    let s_l1 = y_l1.to_f32_vec();

    // L0 and L1 must be bit-identical
    assert_eq!(s_l0, s_l1);

    // Matches f64 reference within f16 precision
    for (actual, &expected) in s_l0.iter().zip(expected_f64.iter()) {
        assert!((actual - expected as f32).abs() < 1e-3);
    }
}

#[test]
fn test_embed_gather_i8_r_l0_and_l1_equivalence() {
    let v = 16;
    let dm = 32;
    let t = 3;
    let tokens = vec![2u32, 15, 0];

    let row_scale_val = 0.125f32;
    let scale_bits = f32_to_f16(row_scale_val);

    // Generate table quant data
    let mut table_q = vec![0i8; v * dm];
    for r in 0..v {
        for c in 0..dm {
            table_q[r * dm + c] = ((r as i32 * 3 + c as i32 * 7) % 250 - 125) as i8;
        }
    }

    // Build L0 buffer: each row has dm bytes of values followed by 2 bytes of row scale
    let row_stride_l0 = dm + 2;
    let mut l0_bytes = vec![0u8; v * row_stride_l0];
    for r in 0..v {
        for c in 0..dm {
            l0_bytes[r * row_stride_l0 + c] = table_q[r * dm + c] as u8;
        }
        l0_bytes[r * row_stride_l0 + dm..r * row_stride_l0 + dm + 2]
            .copy_from_slice(&scale_bits.to_le_bytes());
    }

    // Build L1 buffer: values tiled 16x16, then trailing scale records
    let dims = PaddedDims::new(v as u32, dm as u32, Some(16)).unwrap();
    let values_bytes = dims.n_padded() as usize * dims.k_padded() as usize;
    let geom = scale_geometry(SchemeId::I8R, Layout::L1, &dims).unwrap();
    let mut l1_bytes = vec![0u8; values_bytes + geom.region_bytes as usize];

    for r in 0..v {
        for c in 0..dm {
            let tile_idx = (r / 16) * dims.k_tiles() as usize + (c / 16);
            let lane = ((c % 16) / 8) * 16 + (r % 16);
            let j = c % 8;
            let offset = tile_idx * 256 + lane * 8 + j;
            l1_bytes[offset] = table_q[r * dm + c] as u8;
        }
        let scale_offset = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        l1_bytes[values_bytes + scale_offset..values_bytes + scale_offset + 2]
            .copy_from_slice(&scale_bits.to_le_bytes());
    }

    let tok_buf = TypedBuffer::from_u32(&[t], &tokens);
    let l0_table =
        TypedBuffer::from_bytes(&[v, dm], DType::I8, &l0_bytes).with_quant(QuantScheme::PerRow);
    let l1_table = TypedBuffer::from_bytes(&[v, dm], DType::I8, &l1_bytes)
        .with_quant(QuantScheme::PerRow)
        .with_layout(LayoutId::L1);

    let mut y_l0 = TypedBuffer::zeros(&[t, dm], DType::F32);
    let mut y_l1 = TypedBuffer::zeros(&[t, dm], DType::F32);

    let op = EmbedGatherOp {
        scale: 2.0,
        out_dtype: DType::F32,
    };

    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l0_table.as_view(),
        &mut y_l0.as_view_mut(),
    )
    .unwrap();
    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l1_table.as_view(),
        &mut y_l1.as_view_mut(),
    )
    .unwrap();

    assert_eq!(y_l0.to_f32_vec(), y_l1.to_f32_vec());
}

#[test]
fn test_embed_gather_i4_k_l0_and_l1_equivalence() {
    let v = 16;
    let dm = 256;
    let t = 2;
    let tokens = vec![1u32, 15];

    let d_val = 0.5f32;
    let dmin_val = 0.25f32;
    let d_bits = f32_to_f16(d_val);
    let dmin_bits = f32_to_f16(dmin_val);
    let sc = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let mn = [8u8, 7, 6, 5, 4, 3, 2, 1];

    let sb_header = I4KSuperblock::pack(d_bits, dmin_bits, sc, mn).unwrap();
    let header_bytes = sb_header.to_bytes();

    let mut q_nibbles = vec![0u8; v * dm];
    for r in 0..v {
        for c in 0..dm {
            q_nibbles[r * dm + c] = ((r * 7 + c * 3) % 16) as u8;
        }
    }

    // Pack L0: 128 bytes values + 16 bytes header per row
    let mut l0_bytes = vec![0u8; v * (128 + 16)];
    for r in 0..v {
        let row_base = r * 144;
        for c in 0..dm {
            let byte_idx = row_base + c / 2;
            let nibble = q_nibbles[r * dm + c];
            if c % 2 == 0 {
                l0_bytes[byte_idx] |= nibble & 0x0F;
            } else {
                l0_bytes[byte_idx] |= (nibble & 0x0F) << 4;
            }
        }
        l0_bytes[row_base + 128..row_base + 144].copy_from_slice(&header_bytes);
    }

    // Pack L1: tiled 16x16, then trailing scale headers
    let dims = PaddedDims::new(v as u32, dm as u32, Some(256)).unwrap();
    let values_bytes = (dims.n_padded() as usize * dims.k_padded() as usize) / 2;
    let geom = scale_geometry(SchemeId::I4K, Layout::L1, &dims).unwrap();
    let mut l1_bytes = vec![0u8; values_bytes + geom.region_bytes as usize];

    for r in 0..v {
        for c in 0..dm {
            let tile_idx = (r / 16) * dims.k_tiles() as usize + (c / 16);
            let lane = ((c % 16) / 8) * 16 + (r % 16);
            let j_tile = c % 8;
            let offset = tile_idx * 128 + lane * 4 + j_tile / 2;
            let nibble = q_nibbles[r * dm + c];
            if j_tile % 2 == 0 {
                l1_bytes[offset] |= nibble & 0x0F;
            } else {
                l1_bytes[offset] |= (nibble & 0x0F) << 4;
            }
        }
        let scale_offset = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        l1_bytes[values_bytes + scale_offset..values_bytes + scale_offset + 16]
            .copy_from_slice(&header_bytes);
    }

    let tok_buf = TypedBuffer::from_u32(&[t], &tokens);
    let l0_table = TypedBuffer::from_bytes(&[v, dm], DType::I4, &l0_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(3)));
    let l1_table = TypedBuffer::from_bytes(&[v, dm], DType::I4, &l1_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(3)))
        .with_layout(LayoutId::L1);

    let mut y_l0 = TypedBuffer::zeros(&[t, dm], DType::F32);
    let mut y_l1 = TypedBuffer::zeros(&[t, dm], DType::F32);

    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    };

    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l0_table.as_view(),
        &mut y_l0.as_view_mut(),
    )
    .unwrap();
    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l1_table.as_view(),
        &mut y_l1.as_view_mut(),
    )
    .unwrap();

    assert_eq!(y_l0.to_f32_vec(), y_l1.to_f32_vec());
}

#[test]
fn test_embed_gather_batch_invariance() {
    let v = 8;
    let dm = 16;
    let mut table_f32 = vec![0.0f32; v * dm];
    for i in 0..(v * dm) {
        table_f32[i] = (i as f32 * 0.25) - 2.0;
    }
    let mut l0_bytes = vec![0u8; v * dm * 2];
    for i in 0..(v * dm) {
        let bits = f32_to_f16(table_f32[i]);
        l0_bytes[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    let table = TypedBuffer::from_bytes(&[v, dm], DType::F16, &l0_bytes);

    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    };

    // Gather single token 3 alone
    let single_tok = TypedBuffer::from_u32(&[1], &[3u32]);
    let mut y_single = TypedBuffer::zeros(&[1, dm], DType::F32);
    embed_gather(
        &op,
        &single_tok.as_view(),
        &table.as_view(),
        &mut y_single.as_view_mut(),
    )
    .unwrap();

    // Gather token 3 in sequence [0, 3, 5]
    let multi_tok = TypedBuffer::from_u32(&[3], &[0u32, 3, 5]);
    let mut y_multi = TypedBuffer::zeros(&[3, dm], DType::F32);
    embed_gather(
        &op,
        &multi_tok.as_view(),
        &table.as_view(),
        &mut y_multi.as_view_mut(),
    )
    .unwrap();

    let s1 = y_single.to_f32_vec();
    let s2 = y_multi.to_f32_vec();

    assert_eq!(&s1[..], &s2[dm..2 * dm]);
}

#[test]
fn test_embed_gather_rejects_out_of_vocab_token() {
    let v = 10;
    let dm = 16;
    let table = TypedBuffer::zeros(&[v, dm], DType::F16);
    let tokens = TypedBuffer::from_u32(&[3], &[2u32, 10, 4]); // 10 >= 10 is invalid
    let mut y = TypedBuffer::zeros(&[3, dm], DType::F32);

    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    };

    let err = embed_gather(
        &op,
        &tokens.as_view(),
        &table.as_view(),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    match err {
        T0Error::TokenOutOfRange {
            token,
            vocab_size,
            position,
            ..
        } => {
            assert_eq!(token, 10);
            assert_eq!(vocab_size, 10);
            assert_eq!(position, 1);
        }
        other => panic!("expected TokenOutOfRange, got {other:?}"),
    }
}

#[test]
fn test_embed_gather_reserved_scheme_fails_closed() {
    let v = 16;
    let dm = 32;
    let table_bytes = vec![0u8; v * dm];
    let table = TypedBuffer::from_bytes(&[v, dm], DType::I8, &table_bytes)
        .with_quant(QuantScheme::Scheme(r9v_ir::SchemeId::new(5))); // Scheme 5 is I8_B32_F (A2.3 reserved!)
    let tokens = TypedBuffer::from_u32(&[1], &[0u32]);
    let mut y = TypedBuffer::zeros(&[1, dm], DType::F32);

    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    };

    let err = embed_gather(
        &op,
        &tokens.as_view(),
        &table.as_view(),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    match err {
        T0Error::Format(r9v_format::FormatError::ReservedScheme { scheme, owner }) => {
            assert_eq!(scheme, "i8_b32f");
            assert_eq!(owner, "A2.3");
        }
        other => panic!("expected ReservedScheme error, got {other:?}"),
    }
}
