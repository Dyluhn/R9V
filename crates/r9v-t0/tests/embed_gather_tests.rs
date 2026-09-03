// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
use r9v_format::records::I4KSuperblock;
use r9v_format::{scale_geometry, Layout, PaddedDims, SchemeId};
use r9v_ir::{DType, EmbedGatherOp, LayoutId, QuantScheme};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::f32_to_f16;
use r9v_t0::embed_gather::{embed_gather, embed_gather_f64_reference, embed_gather_with_scales};
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

#[test]
fn test_embed_gather_inline_vs_separate_scale_equivalence_l0_and_l1() {
    let v = 16;
    let dm = 32;
    let t = 2;
    let tokens = vec![1u32, 7];
    let tok_buf = TypedBuffer::from_u32(&[t], &tokens);

    let row_scale_val = 0.5f32;
    let scale_bits = f32_to_f16(row_scale_val);

    // Raw values: [V, Dm]
    let mut raw_vals = vec![0i8; v * dm];
    for r in 0..v {
        for c in 0..dm {
            raw_vals[r * dm + c] = ((r as i32 * 7 + c as i32 * 3) % 200 - 100) as i8;
        }
    }

    // 1. L0 inline vs L0 separate
    let mut l0_inline_bytes = vec![0u8; v * (dm + 2)];
    for r in 0..v {
        for c in 0..dm {
            l0_inline_bytes[r * (dm + 2) + c] = raw_vals[r * dm + c] as u8;
        }
        l0_inline_bytes[r * (dm + 2) + dm..r * (dm + 2) + dm + 2]
            .copy_from_slice(&scale_bits.to_le_bytes());
    }
    let l0_inline_table = TypedBuffer::from_bytes(&[v, dm], DType::I8, &l0_inline_bytes)
        .with_quant(QuantScheme::PerRow);

    let raw_u8: Vec<u8> = raw_vals.iter().map(|&x| x as u8).collect();
    let l0_sep_values =
        TypedBuffer::from_bytes(&[v, dm], DType::I8, &raw_u8).with_quant(QuantScheme::PerRow);
    let mut l0_scales_bytes = vec![0u8; v * 2];
    for r in 0..v {
        l0_scales_bytes[r * 2..r * 2 + 2].copy_from_slice(&scale_bits.to_le_bytes());
    }
    let l0_scales_buf = TypedBuffer::from_bytes(&[v], DType::F16, &l0_scales_bytes);

    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    };
    let mut y_l0_inline = TypedBuffer::zeros(&[t, dm], DType::F32);
    let mut y_l0_sep = TypedBuffer::zeros(&[t, dm], DType::F32);

    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l0_inline_table.as_view(),
        &mut y_l0_inline.as_view_mut(),
    )
    .unwrap();
    embed_gather_with_scales(
        &op,
        &tok_buf.as_view(),
        &l0_sep_values.as_view(),
        Some(&l0_scales_buf.as_view()),
        &mut y_l0_sep.as_view_mut(),
    )
    .unwrap();

    assert_eq!(y_l0_inline.to_f32_vec(), y_l0_sep.to_f32_vec());

    // 2. L1 inline vs L1 separate
    let dims = PaddedDims::new(v as u32, dm as u32, Some(16)).unwrap();
    let values_bytes = dims.n_padded() as usize * dims.k_padded() as usize;
    let geom = scale_geometry(SchemeId::I8R, Layout::L1, &dims).unwrap();

    let mut l1_values_bytes = vec![0u8; values_bytes];
    for r in 0..v {
        for c in 0..dm {
            let tile_idx = (r / 16) * dims.k_tiles() as usize + (c / 16);
            let lane = ((c % 16) / 8) * 16 + (r % 16);
            let j = c % 8;
            let offset = tile_idx * 256 + lane * 8 + j;
            l1_values_bytes[offset] = raw_vals[r * dm + c] as u8;
        }
    }

    let mut l1_scales_bytes = vec![0u8; geom.region_bytes as usize];
    for r in 0..v {
        let scale_offset = geom
            .record_offset((r / 16) as u64, 0, (r % 16) as u32)
            .unwrap() as usize;
        l1_scales_bytes[scale_offset..scale_offset + 2].copy_from_slice(&scale_bits.to_le_bytes());
    }

    let mut l1_inline_bytes = l1_values_bytes.clone();
    l1_inline_bytes.extend_from_slice(&l1_scales_bytes);

    let l1_inline_table = TypedBuffer::from_bytes(&[v, dm], DType::I8, &l1_inline_bytes)
        .with_quant(QuantScheme::PerRow)
        .with_layout(LayoutId::L1);
    let l1_sep_table = TypedBuffer::from_bytes(&[v, dm], DType::I8, &l1_values_bytes)
        .with_quant(QuantScheme::PerRow)
        .with_layout(LayoutId::L1);
    let l1_scales_buf = TypedBuffer::from_bytes(
        &[geom.n_blocks as usize, geom.k_blocks as usize, 16],
        DType::F16,
        &l1_scales_bytes,
    );

    let mut y_l1_inline = TypedBuffer::zeros(&[t, dm], DType::F32);
    let mut y_l1_sep = TypedBuffer::zeros(&[t, dm], DType::F32);

    embed_gather(
        &op,
        &tok_buf.as_view(),
        &l1_inline_table.as_view(),
        &mut y_l1_inline.as_view_mut(),
    )
    .unwrap();
    embed_gather_with_scales(
        &op,
        &tok_buf.as_view(),
        &l1_sep_table.as_view(),
        Some(&l1_scales_buf.as_view()),
        &mut y_l1_sep.as_view_mut(),
    )
    .unwrap();

    assert_eq!(y_l1_inline.to_f32_vec(), y_l1_sep.to_f32_vec());
    assert_eq!(y_l0_inline.to_f32_vec(), y_l1_inline.to_f32_vec());
}

#[test]
fn test_embed_gather_adversarial_and_validation() {
    let v = 16;
    let dm = 16;
    let tokens = TypedBuffer::from_u32(&[2], &[0u32, 1]);
    let mut y = TypedBuffer::zeros(&[2, dm], DType::F32);
    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    };

    // 1. Truncated values buffer
    let trunc_bytes = vec![0u8; 10]; // Requires 16*16*2 = 512 bytes for F16
    let trunc_table = r9v_t0::buffer::TensorView::from_bytes(&[v, dm], DType::F16, &trunc_bytes);
    let err = embed_gather(&op, &tokens.as_view(), &trunc_table, &mut y.as_view_mut()).unwrap_err();
    assert!(matches!(err, T0Error::BufferLengthMismatch { .. }));

    // 2. Truncated L0 inline scales
    let trunc_inline = vec![0u8; v * dm]; // Missing the 2 trailing bytes per row for I8_R
    let trunc_inline_table =
        r9v_t0::buffer::TensorView::from_bytes(&[v, dm], DType::I8, &trunc_inline)
            .with_quant(QuantScheme::PerRow);
    let err = embed_gather(
        &op,
        &tokens.as_view(),
        &trunc_inline_table,
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::BufferLengthMismatch { .. }));

    // 3. Shape-wrong separate scales (declared backing valid for shape [1])
    let full_vals = vec![0u8; v * dm];
    let vals_table =
        TypedBuffer::from_bytes(&[v, dm], DType::I8, &full_vals).with_quant(QuantScheme::PerRow);
    let trunc_scales = vec![0u8; 2]; // Valid for shape [1]
    let trunc_scale_buf = r9v_t0::buffer::TensorView::from_bytes(&[1], DType::F16, &trunc_scales);
    let err = embed_gather_with_scales(
        &op,
        &tokens.as_view(),
        &vals_table.as_view(),
        Some(&trunc_scale_buf),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        matches!(err, T0Error::DimensionMismatch { tensor, dim_name, expected, got, .. }
            if tensor == "table_scales" && dim_name == "V" && expected == v && got == 1),
        "expected DimensionMismatch on table_scales with exact fields, got: {:?}",
        err
    );

    // 3b. Separate scale buffer with valid shape [v] but extra bytes:
    // declared backing is valid for [v] (32 bytes), but buffer has 36 bytes.
    let extra_scales = vec![0u8; v * 2 + 4];
    let extra_scale_buf = r9v_t0::buffer::TensorView::from_bytes(&[v], DType::F16, &extra_scales);
    let err = embed_gather_with_scales(
        &op,
        &tokens.as_view(),
        &vals_table.as_view(),
        Some(&extra_scale_buf),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        matches!(err, T0Error::BufferLengthMismatch { tensor, buffer_len, expected_len, .. }
            if tensor == "table_scales" && buffer_len == v * 2 + 4 && expected_len == v * 2),
        "expected BufferLengthMismatch on table_scales with exact fields, got: {:?}",
        err
    );

    // 3c. L1 separate scales: declared backing valid for its shape [1, 1, 16],
    // but table tensor requires [2, 1, 16] (32 records instead of 16).
    let dims_l1 = PaddedDims::new(32, 16, Some(16)).unwrap();
    let geom_l1 = scale_geometry(SchemeId::I8R, Layout::L1, &dims_l1).unwrap();
    let l1_table_bytes = vec![0u8; (dims_l1.n_padded() * dims_l1.k_padded()) as usize];
    let l1_table_vals = TypedBuffer::from_bytes(&[32, 16], DType::I8, &l1_table_bytes)
        .with_quant(QuantScheme::PerRow)
        .with_layout(LayoutId::L1);
    let l1_scale_16_bytes = vec![0u8; 32]; // Valid for [1, 1, 16] (16 elements * 2 bytes)
    let l1_bad_scale =
        r9v_t0::buffer::TensorView::from_bytes(&[1, 1, 16], DType::F16, &l1_scale_16_bytes);
    let err = embed_gather_with_scales(
        &op,
        &tokens.as_view(),
        &l1_table_vals.as_view(),
        Some(&l1_bad_scale),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        matches!(err, T0Error::DimensionMismatch { tensor, dim_name, expected, got, .. }
            if tensor == "table_scales" && dim_name == "scale_shape" && expected == geom_l1.records as usize && got == 16),
        "expected DimensionMismatch on L1 table_scales, got: {:?}",
        err
    );

    // 3d. Regression: forbidden alternate scale dtype (e.g. I8 for I8_R)
    let table_scale_forbidden_dtype = TypedBuffer::from_bytes(&[v], DType::I8, &vec![0u8; v]);
    let err = embed_gather_with_scales(
        &op,
        &tokens.as_view(),
        &vals_table.as_view(),
        Some(&table_scale_forbidden_dtype.as_view()),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        matches!(err, T0Error::DTypeMismatch { tensor, ref expected, got }
            if tensor == "table_scales" && expected == &vec![DType::F16] && got == DType::I8),
        "expected DTypeMismatch for forbidden scale dtype on table_scales, got: {:?}",
        err
    );

    // 3e. Regression: arbitrary exact-byte shape for I8_R L0 (shape [v/2, 2] has v elements and v*2 bytes)
    let table_scale_exact_bytes_wrong_shape =
        TypedBuffer::from_bytes(&[v / 2, 2], DType::F16, &vec![0u8; v * 2]);
    let err = embed_gather_with_scales(
        &op,
        &tokens.as_view(),
        &vals_table.as_view(),
        Some(&table_scale_exact_bytes_wrong_shape.as_view()),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        matches!(err, T0Error::DimensionMismatch { tensor, dim_name, expected, got, .. }
            if tensor == "table_scales" && dim_name == "V" && expected == v && got == v / 2),
        "expected DimensionMismatch for arbitrary exact-byte shape on table_scales, got: {:?}",
        err
    );

    // 4. Invalid layout L1S on table
    let l1s_table = TypedBuffer::zeros(&[v, dm], DType::F16).with_layout(LayoutId::L1S);
    let err = embed_gather(
        &op,
        &tokens.as_view(),
        &l1s_table.as_view(),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::LayoutMismatch { .. }));

    // 5. Separate scales on unquantized table
    let f16_table = TypedBuffer::zeros(&[v, dm], DType::F16);
    let scale_buf = TypedBuffer::zeros(&[v], DType::F16);
    let err = embed_gather_with_scales(
        &op,
        &tokens.as_view(),
        &f16_table.as_view(),
        Some(&scale_buf.as_view()),
        &mut y.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::InvalidAttribute { .. }));

    // 6. Empty token_ids
    let empty_tokens = TypedBuffer::from_u32(&[0], &[]);
    let mut empty_y = TypedBuffer::zeros(&[0, dm], DType::F32);
    let err = embed_gather(
        &op,
        &empty_tokens.as_view(),
        &f16_table.as_view(),
        &mut empty_y.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::EmptyInput { .. }));

    // 7. Undersized padded L1 buffer: v=17, dm=17 logical requires 289*2=578 bytes.
    // Padded L1 (32x32) requires 1024*2 = 2048 bytes.
    let l1_logical_bytes = vec![0u8; 17 * 17 * 2];
    let l1_undersized_table =
        r9v_t0::buffer::TensorView::from_bytes(&[17, 17], DType::F16, &l1_logical_bytes)
            .with_layout(LayoutId::L1);
    let l1_tokens = TypedBuffer::from_u32(&[1], &[0u32]);
    let mut l1_y = TypedBuffer::zeros(&[1, 17], DType::F32);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        embed_gather(
            &op,
            &l1_tokens.as_view(),
            &l1_undersized_table,
            &mut l1_y.as_view_mut(),
        )
    }));
    assert!(res.is_ok(), "must not panic on undersized padded L1 table");
    assert!(matches!(
        res.unwrap(),
        Err(T0Error::BufferLengthMismatch { .. })
    ));

    // 8. Non-divisible superblock Dm
    let i4k_bad_dm = TypedBuffer::zeros(&[16, 128], DType::I4)
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()));
    let mut y_128 = TypedBuffer::zeros(&[2, 128], DType::F32);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        embed_gather(
            &op,
            &tokens.as_view(),
            &i4k_bad_dm.as_view(),
            &mut y_128.as_view_mut(),
        )
    }));
    assert!(res.is_ok(), "must not panic on non-divisible I4K Dm");
    assert!(
        matches!(res.unwrap(), Err(T0Error::DimensionMismatch { dim_name, .. }) if dim_name == "Dm")
    );

    let i8b128_bad_dm = TypedBuffer::zeros(&[16, 64], DType::I8)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));
    let mut y_64 = TypedBuffer::zeros(&[2, 64], DType::F32);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        embed_gather(
            &op,
            &tokens.as_view(),
            &i8b128_bad_dm.as_view(),
            &mut y_64.as_view_mut(),
        )
    }));
    assert!(res.is_ok(), "must not panic on non-divisible I8B128 Dm");
    assert!(
        matches!(res.unwrap(), Err(T0Error::DimensionMismatch { dim_name, .. }) if dim_name == "Dm")
    );
}
