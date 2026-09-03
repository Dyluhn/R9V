// SPDX-License-Identifier: Apache-2.0
//! Tests for scalar T0 `ngram_gather` (Spec 1 §4.A, Card A1.9).

use r9v_common::SeededRng;
use r9v_format::SchemeId;
use r9v_ir::{DType, HashId, NgramCombine, NgramGatherOp, NgramSource, Op, QuantScheme};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::{f16_to_f32, f32_to_f16};
use r9v_t0::error::T0Error;
use r9v_t0::{
    execute_ngram_op, ngram_gather, ngram_gather_device, ngram_gather_f64_reference_rows,
    ngram_gather_f64_reference_staged, NgramHash, Tolerance,
};

fn next_f32(rng: &mut SeededRng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
    lo + (u as f32) * (hi - lo)
}

fn staged_op(heads: u32, combine: NgramCombine) -> NgramGatherOp {
    NgramGatherOp {
        source: NgramSource::Staged,
        orders: vec![1; heads as usize].into_boxed_slice(),
        heads,
        hash: HashId::new(0),
        table_sizes: vec![64; heads as usize].into_boxed_slice(),
        combine,
        out_dtype: DType::F32,
    }
}

fn device_op(heads: u32, sizes: &[u32], combine: NgramCombine) -> NgramGatherOp {
    NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![1; heads as usize].into_boxed_slice(),
        heads,
        hash: HashId::new(7),
        table_sizes: sizes.to_vec().into_boxed_slice(),
        combine,
        out_dtype: DType::F32,
    }
}

/// Fake hash: row = (token + pos + head_salt) mod table_size.
struct FakeHash {
    salts: Vec<u32>,
}

impl NgramHash for FakeHash {
    fn row(&self, tokens: &[u32], pos: usize, _order: u32, table_size: u32) -> u32 {
        let head = (pos * 31) % self.salts.len().max(1);
        tokens[pos]
            .wrapping_add(pos as u32)
            .wrapping_add(self.salts[head % self.salts.len()])
            % table_size
    }
}

#[test]
fn staged_concat_and_sum_match_f64_oracle() {
    for combine in [NgramCombine::Concat, NgramCombine::Sum] {
        let mut rng = SeededRng::new(0xB41);
        let (t, np, dn) = (4, 3, 8);
        let staging: Vec<i8> = (0..t * np * dn)
            .map(|_| (rng.next_u64() % 21) as i8 - 10)
            .collect();
        let scales: Vec<f32> = (0..t * np).map(|_| next_f32(&mut rng, 0.05, 0.5)).collect();
        let op = staged_op(np as u32, combine);

        let st_buf = TypedBuffer::from_i8(&[t, np, dn], &staging)
            .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir()));
        let sc_buf = TypedBuffer::from_f32(&[t, np], &scales);
        let out1 = match combine {
            NgramCombine::Concat => np * dn,
            NgramCombine::Sum => dn,
        };
        let mut y_buf = TypedBuffer::zeros(&[t, out1], DType::F32);
        ngram_gather(
            &op,
            &st_buf.as_view(),
            &sc_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let expected = ngram_gather_f64_reference_staged(
            &staging,
            &scales.iter().map(|&s| s as f64).collect::<Vec<f64>>(),
            t,
            np,
            dn,
            combine,
        )
        .unwrap();
        let tol = Tolerance::i8_weight();
        for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("combine={combine:?} y[{i}]"));
        }
    }
}

#[test]
fn staged_rank1_and_f16_scales_agree() {
    let (t, np, dn) = (3, 2, 6);
    let staging = vec![2i8; t * np * dn];
    let scales_f32 = vec![0.5f32; t];
    let scales_f16: Vec<u16> = vec![0.5f32; np].iter().map(|&s| f32_to_f16(s)).collect();
    let op = staged_op(np as u32, NgramCombine::Sum);
    let st_buf = TypedBuffer::from_i8(&[t, np, dn], &staging)
        .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir()));
    let run = |sc: &TypedBuffer| {
        let mut yb = TypedBuffer::zeros(&[t, dn], DType::F32);
        ngram_gather(&op, &st_buf.as_view(), &sc.as_view(), &mut yb.as_view_mut()).unwrap();
        yb.to_f32_vec()
    };
    let sc1 = TypedBuffer::from_f32(&[t], &scales_f32);
    let sc2 = TypedBuffer::from_f16(&[t, np], &{
        let mut v = Vec::new();
        for _ in 0..t {
            v.extend_from_slice(&scales_f16);
        }
        v
    });
    assert_eq!(run(&sc1), run(&sc2));
    // 2 * 0.5 summed over 2 heads = 2.0 per element.
    assert_eq!(run(&sc1), vec![2.0; t * dn]);
}

#[test]
fn staged_i4_rows_fail_closed() {
    let op = staged_op(2, NgramCombine::Concat);
    let st_buf = TypedBuffer::from_bytes(&[2, 2, 8], DType::I4, &[0u8; 2 * 2 * 4])
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()));
    let sc_buf = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let mut y_buf = TypedBuffer::zeros(&[2, 16], DType::F32);
    let err = ngram_gather(
        &op,
        &st_buf.as_view(),
        &sc_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::QuantMismatch { .. }));
}

#[test]
fn staged_multiblock_rows_fail_closed() {
    // I8B128 with Dn != 128 has no scalar-scale rule: reject, don't invent one.
    let op = staged_op(1, NgramCombine::Concat);
    let st_buf = TypedBuffer::from_i8(&[2, 1, 256], &vec![1i8; 512])
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));
    let sc_buf = TypedBuffer::from_f32(&[2], &[0.5, 0.5]);
    let mut y_buf = TypedBuffer::zeros(&[2, 256], DType::F32);
    let err = ngram_gather(
        &op,
        &st_buf.as_view(),
        &sc_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::InvalidAttribute { .. }));
}

#[test]
fn device_unquantized_table_matches_layout_oracle() {
    for combine in [NgramCombine::Concat, NgramCombine::Sum] {
        let mut rng = SeededRng::new(0xD041);
        let (t, dn) = (5usize, 6usize);
        let heads: u32 = 2;
        let nh = heads as usize;
        let sizes = [11u32, 13u32];
        let entries = 24;
        let table: Vec<f32> = (0..entries * dn)
            .map(|_| next_f32(&mut rng, -1.0, 1.0))
            .collect();
        let tokens: Vec<u32> = (0..t).map(|_| (rng.next_u64() % 1000) as u32).collect();
        let op = device_op(heads, &sizes, combine);
        let hash = FakeHash { salts: vec![3, 5] };

        let tok_buf = TypedBuffer::from_u32(&[t], &tokens);
        let tab_buf = TypedBuffer::from_f32(&[entries, dn], &table);
        let out1 = match combine {
            NgramCombine::Concat => nh * dn,
            NgramCombine::Sum => dn,
        };
        let mut y_buf = TypedBuffer::zeros(&[t, out1], DType::F32);
        ngram_gather_device(
            &op,
            &tok_buf.as_view(),
            &tab_buf.as_view(),
            None,
            &hash,
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        // Independent oracle over rows resolved by the same fake hash.
        let bases = [0usize, 11usize];
        let mut row_ids = vec![0u32; t * nh];
        for row in 0..t {
            for head in 0..nh {
                let prow = hash.row(&tokens, row, 1, sizes[head]);
                row_ids[row * nh + head] = (bases[head] + prow as usize) as u32;
            }
        }
        let expected = ngram_gather_f64_reference_rows(
            &table.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            entries,
            dn,
            &row_ids,
            t,
            nh,
            combine,
        )
        .unwrap();
        let tol = Tolerance::f32();
        for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
            tol.assert_within(actual as f64, exp, &format!("combine={combine:?} y[{i}]"));
        }
    }
}

#[test]
fn device_i8_table_with_scales_matches_f64() {
    let (t, dn) = (3usize, 4usize);
    let heads: u32 = 2;
    let nh = heads as usize;
    let sizes = [5u32, 7u32];
    let entries = 12;
    let table_q: Vec<i8> = (0..entries * dn).map(|i| (i as i8 % 9) - 4).collect();
    let table_bytes: Vec<u8> = table_q.iter().map(|&q| q as u8).collect();
    let scales = vec![0.5f32; entries];
    let scale_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| f32_to_f16(s).to_le_bytes())
        .collect();
    let tokens = vec![1u32, 2, 3];
    let op = device_op(heads, &sizes, NgramCombine::Sum);
    let hash = FakeHash { salts: vec![1, 2] };

    let tok_buf = TypedBuffer::from_u32(&[t], &tokens);
    let tab_buf = TypedBuffer::from_bytes(&[entries, dn], DType::I8, &table_bytes)
        .with_quant(QuantScheme::PerRow);
    let sc_buf = TypedBuffer::from_bytes(&[entries], DType::F16, &scale_bytes);
    let mut y_buf = TypedBuffer::zeros(&[t, dn], DType::F32);
    ngram_gather_device(
        &op,
        &tok_buf.as_view(),
        &tab_buf.as_view(),
        Some(&sc_buf.as_view()),
        &hash,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // Oracle: decode rows in f64, then layout.
    let table_f64: Vec<f64> = table_q.iter().map(|&q| q as f64 * 0.5).collect();
    let bases = [0usize, 5usize];
    let mut row_ids = vec![0u32; t * nh];
    for row in 0..t {
        for head in 0..nh {
            row_ids[row * nh + head] =
                (bases[head] + hash.row(&tokens, row, 1, sizes[head]) as usize) as u32;
        }
    }
    let expected = ngram_gather_f64_reference_rows(
        &table_f64,
        entries,
        dn,
        &row_ids,
        t,
        nh,
        NgramCombine::Sum,
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("y[{i}]"));
    }
}

#[test]
fn staged_i8b128_dn128_accept_matches_f64() {
    // Blocker 3: the staged I8B128 accept path (Dn == 128, one scalar row
    // scale = the block scale) vs the f64 oracle.
    let mut rng = SeededRng::new(0xB128);
    let (t, np, dn) = (3usize, 2usize, 128usize);
    let staging: Vec<i8> = (0..t * np * dn)
        .map(|_| (rng.next_u64() % 21) as i8 - 10)
        .collect();
    let scales: Vec<f32> = (0..t * np).map(|_| next_f32(&mut rng, 0.05, 0.5)).collect();
    let op = staged_op(np as u32, NgramCombine::Concat);
    let st_buf = TypedBuffer::from_i8(&[t, np, dn], &staging)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));
    let sc_buf = TypedBuffer::from_f32(&[t, np], &scales);
    let mut y_buf = TypedBuffer::zeros(&[t, np * dn], DType::F32);
    ngram_gather(
        &op,
        &st_buf.as_view(),
        &sc_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();
    let expected = ngram_gather_f64_reference_staged(
        &staging,
        &scales.iter().map(|&s| s as f64).collect::<Vec<f64>>(),
        t,
        np,
        dn,
        NgramCombine::Concat,
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("staged-i8b128 y[{i}]"));
    }
}

#[test]
fn device_i8b128_multiblock_matches_f64() {
    // Blocker 3: device I8B128 with Dn=256 (two blocks/row) via the canonical
    // encoder, per-block scales in the separate [entries, 2] carrier.
    let mut rng = SeededRng::new(0x1B82);
    let (t, dn) = (3usize, 256usize);
    let heads: u32 = 2;
    let nh = heads as usize;
    let sizes = [5u32, 7u32];
    let entries = 12;
    let blocks = dn / 128;
    let mut table_q = vec![0i8; entries * dn];
    let mut table_s = Vec::with_capacity(entries * blocks);
    for row in 0..entries {
        let vals: Vec<f32> = (0..dn).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
        let (q, sc) = r9v_format::encode_i8_block128(&vals).unwrap();
        table_q[row * dn..(row + 1) * dn].copy_from_slice(&q);
        table_s.extend_from_slice(&sc);
    }
    let table_bytes: Vec<u8> = table_q.iter().map(|&q| q as u8).collect();
    let scale_bytes: Vec<u8> = table_s.iter().flat_map(|s| s.to_bytes()).collect();
    let tokens = vec![4u32, 9, 2];
    let op = device_op(heads, &sizes, NgramCombine::Sum);
    let hash = FakeHash { salts: vec![1, 2] };

    let tok_buf = TypedBuffer::from_u32(&[t], &tokens);
    let tab_buf = TypedBuffer::from_bytes(&[entries, dn], DType::I8, &table_bytes)
        .with_quant(QuantScheme::Scheme(SchemeId::I8B128.to_ir()));
    let sc_buf = TypedBuffer::from_bytes(&[entries, blocks], DType::F16, &scale_bytes);
    let mut y_buf = TypedBuffer::zeros(&[t, dn], DType::F32);
    ngram_gather_device(
        &op,
        &tok_buf.as_view(),
        &tab_buf.as_view(),
        Some(&sc_buf.as_view()),
        &hash,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    // Independent oracle: per-block q*s decode in f64, then layout.
    let table_f64: Vec<f64> = table_q
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let s = table_s[(i / dn) * blocks + (i % dn) / 128];
            q as f64 * f16_to_f32(s.bits()) as f64
        })
        .collect();
    let bases = [0usize, 5usize];
    let mut row_ids = vec![0u32; t * nh];
    for row in 0..t {
        for head in 0..nh {
            row_ids[row * nh + head] =
                (bases[head] + hash.row(&tokens, row, 1, sizes[head]) as usize) as u32;
        }
    }
    let expected = ngram_gather_f64_reference_rows(
        &table_f64,
        entries,
        dn,
        &row_ids,
        t,
        nh,
        NgramCombine::Sum,
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("device-i8b128 y[{i}]"));
    }
}

#[test]
fn device_i4k_nibble_parity_matches_f64() {
    // Blocker 3: device I4K with Dn=256 (one superblock/row). The T0 decode
    // packs even flat indices low; the oracle below is written against the
    // same parity independently, and the fixture asserts mixed nibbles so a
    // parity swap would fail loudly.
    let mut rng = SeededRng::new(0x14B4);
    let (t, dn) = (3usize, 256usize);
    let heads: u32 = 2;
    let nh = heads as usize;
    let sizes = [5u32, 7u32];
    let entries = 12;
    let mut nibbles = vec![0u8; entries * dn];
    let mut headers = Vec::with_capacity(entries);
    for row in 0..entries {
        let mut vals = [0.0f32; 256];
        for v in vals.iter_mut().take(dn) {
            *v = next_f32(&mut rng, -1.0, 1.0);
        }
        let (q, header) = r9v_format::encode_i4k_superblock(&vals).unwrap();
        nibbles[row * dn..(row + 1) * dn].copy_from_slice(&q);
        headers.push(header);
    }
    // Parity load-bearing: require mixed low/high nibbles in the fixture.
    let mut mixed = 0;
    for row in 0..entries {
        for c in 0..dn / 2 {
            if nibbles[row * dn + 2 * c] != nibbles[row * dn + 2 * c + 1] {
                mixed += 1;
            }
        }
    }
    assert!(mixed > 0, "fixture must exercise nibble parity");
    let mut packed = vec![0u8; entries * dn / 2];
    for row in 0..entries {
        for c in 0..dn / 2 {
            packed[row * dn / 2 + c] =
                (nibbles[row * dn + 2 * c] & 0x0F) | ((nibbles[row * dn + 2 * c + 1] & 0x0F) << 4);
        }
    }
    let header_bytes: Vec<u8> = headers.iter().flat_map(|h| h.to_bytes()).collect();
    let tokens = vec![7u32, 1, 11];
    let op = device_op(heads, &sizes, NgramCombine::Concat);
    let hash = FakeHash { salts: vec![2, 3] };

    let tok_buf = TypedBuffer::from_u32(&[t], &tokens);
    let tab_buf = TypedBuffer::from_bytes(&[entries, dn], DType::I4, &packed)
        .with_quant(QuantScheme::Scheme(SchemeId::I4K.to_ir()));
    let sc_buf = TypedBuffer::from_bytes(&[entries, 1, 4], DType::U32, &header_bytes);
    let mut y_buf = TypedBuffer::zeros(&[t, nh * dn], DType::F32);
    ngram_gather_device(
        &op,
        &tok_buf.as_view(),
        &tab_buf.as_view(),
        Some(&sc_buf.as_view()),
        &hash,
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let mut table_f64 = vec![0.0f64; entries * dn];
    for row in 0..entries {
        let h = &headers[row];
        let d = h.d_value(0).unwrap() as f64;
        let dmin = h.dmin_value(0).unwrap() as f64;
        let sc = h.scales();
        let mn = h.mins();
        for c in 0..dn {
            let sub = (c % 256) / 32;
            table_f64[row * dn + c] =
                d * sc[sub] as f64 * nibbles[row * dn + c] as f64 - dmin * mn[sub] as f64;
        }
    }
    let bases = [0usize, 5usize];
    let mut row_ids = vec![0u32; t * nh];
    for row in 0..t {
        for head in 0..nh {
            row_ids[row * nh + head] =
                (bases[head] + hash.row(&tokens, row, 1, sizes[head]) as usize) as u32;
        }
    }
    let expected = ngram_gather_f64_reference_rows(
        &table_f64,
        entries,
        dn,
        &row_ids,
        t,
        nh,
        NgramCombine::Concat,
    )
    .unwrap();
    let tol = Tolerance::i8_weight();
    for (i, (&actual, &exp)) in y_buf.to_f32_vec().iter().zip(expected.iter()).enumerate() {
        tol.assert_within(actual as f64, exp, &format!("device-i4k y[{i}]"));
    }
}

struct BadHash;

impl NgramHash for BadHash {
    fn row(&self, _tokens: &[u32], _pos: usize, _order: u32, table_size: u32) -> u32 {
        table_size + 3
    }
}

#[test]
fn device_hash_out_of_range_collected() {
    let op = device_op(2, &[5, 7], NgramCombine::Concat);
    let tok_buf = TypedBuffer::from_u32(&[2], &[1, 2]);
    let tab_buf = TypedBuffer::from_f32(&[12, 4], &[0.1; 48]);
    let mut y_buf = TypedBuffer::zeros(&[2, 8], DType::F32);
    let err = ngram_gather_device(
        &op,
        &tok_buf.as_view(),
        &tab_buf.as_view(),
        None,
        &BadHash,
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    match err {
        T0Error::Multiple { problems } => assert_eq!(problems.len(), 4),
        other => panic!("expected Multiple, got {other:?}"),
    }
    assert_eq!(y_buf.to_f32_vec(), vec![0.0; 16]);
}

#[test]
fn device_orders_above_one_rejected() {
    let op = NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![2u32, 1].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(7),
        table_sizes: vec![5, 7].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F32,
    };
    let tok_buf = TypedBuffer::from_u32(&[2], &[1, 2]);
    let tab_buf = TypedBuffer::from_f32(&[12, 4], &[0.1; 48]);
    let mut y_buf = TypedBuffer::zeros(&[2, 8], DType::F32);
    let hash = FakeHash { salts: vec![1, 2] };
    let err = ngram_gather_device(
        &op,
        &tok_buf.as_view(),
        &tab_buf.as_view(),
        None,
        &hash,
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        T0Error::InvalidAttribute { .. } | T0Error::Multiple { .. }
    ));
}

#[test]
fn dispatch_enforces_staged_arity_and_source() {
    let staged = Op::NgramGather(staged_op(2, NgramCombine::Sum));
    let st_buf = TypedBuffer::from_i8(&[2, 2, 4], &[1i8; 16])
        .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir()));
    let sc_buf = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let mut y_buf = TypedBuffer::zeros(&[2, 4], DType::F32);
    assert!(execute_ngram_op(&staged, &[st_buf.as_view()], &mut [y_buf.as_view_mut()]).is_err());
    assert!(execute_ngram_op(
        &staged,
        &[st_buf.as_view(), sc_buf.as_view()],
        &mut [y_buf.as_view_mut()]
    )
    .is_ok());
    let device = Op::NgramGather(device_op(2, &[5, 7], NgramCombine::Sum));
    assert!(execute_ngram_op(
        &device,
        &[st_buf.as_view(), sc_buf.as_view()],
        &mut [y_buf.as_view_mut()]
    )
    .is_err());
}

#[test]
fn determinism_and_batch_invariance() {
    let (t, np, dn) = (5, 2, 6);
    let staging: Vec<i8> = (0..t * np * dn).map(|i| (i % 17) as i8 - 8).collect();
    let scales: Vec<f32> = (0..t * np).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let op = staged_op(np as u32, NgramCombine::Concat);
    let run = |st: &[i8], sc: &[f32], tt: usize| {
        let stb = TypedBuffer::from_i8(&[tt, np, dn], st)
            .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir()));
        let scb = TypedBuffer::from_f32(&[tt, np], sc);
        let mut yb = TypedBuffer::zeros(&[tt, np * dn], DType::F32);
        ngram_gather(&op, &stb.as_view(), &scb.as_view(), &mut yb.as_view_mut()).unwrap();
        yb.to_f32_vec()
    };
    let y1 = run(&staging, &scales, t);
    let y2 = run(&staging, &scales, t);
    assert_eq!(y1, y2);
    // Row 3 alone.
    let r3 = staging[3 * np * dn..4 * np * dn].to_vec();
    let s3 = scales[3 * np..4 * np].to_vec();
    assert_eq!(run(&r3, &s3, 1), y1[3 * np * dn..4 * np * dn]);
}
