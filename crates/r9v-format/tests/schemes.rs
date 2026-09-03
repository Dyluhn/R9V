// SPDX-License-Identifier: Apache-2.0
//! Native-scheme tests (Spec 2 §3.1, §3.2, §8; card A2.2).
//!
//! Provenance: `I4_K` fixtures come from `tests/fixtures/r9v-format/q4k_reference.txt`,
//! produced by verbatim llama.cpp `quantize_row_q4_K_ref`/`dequantize_row_q4_K`
//! (`ggml/src/ggml-quants.c` @ master 2026-09-03) and cross-checked bit-exact
//! against gguf-py 0.19.0 dequantize (1536/1536 f32 words). The `f16` codec is
//! validated bit-exact against numpy `float16` over 300k adversarial values
//! (ties, subnormals, infinities) by the scratch check that produced these
//! vectors; `E4M3` vectors are the OCP definition (see SI-20).

use r9v_common::SeededRng;
use r9v_format::{
    bits_per_weight, check_f16_scale, decode, decode_e4m3_block128, decode_i4k_superblock,
    decode_i8_block128, decode_i8_row, encode_e4m3_block128, encode_i4k_superblock,
    encode_i8_block128, encode_i8_row, f16_scale_bits, f16_to_f32, f32_to_f16_bits,
    l0_row_stride_bytes, l1s_value_dims, outer_block, scale_geometry, scale_record_bytes,
    E4M3Block128Scale, E4m3, FormatError, I4KSuperblock, I8Block128Scale, I8RowScale, Layout,
    PaddedDims, QuantValue, ScaleGeometry, ScaleSet, SchemeId,
};

/// One parsed fixture case (see the fixture file header for provenance).
struct Q4KCase {
    name: String,
    block: Vec<u8>,
    d_bits: u16,
    dmin_bits: u16,
    sc: [u8; 8],
    mn: [u8; 8],
    q: Vec<u8>,
    y_bits: Vec<u32>,
}

fn hex_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16).expect("fixture hex valid");
        let lo = (bytes[i + 1] as char)
            .to_digit(16)
            .expect("fixture hex valid");
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    out
}

fn load_q4k_reference() -> Vec<Q4KCase> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/r9v-format/q4k_reference.txt"
    );
    let text = std::fs::read_to_string(path).expect("fixture file present");
    let mut cases = Vec::new();
    let mut cur: Option<Q4KCase> = None;
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let (key, val) = line.split_once(' ').expect("fixture lines are key + value");
        match key {
            "case" => {
                if let Some(c) = cur.take() {
                    cases.push(c);
                }
                cur = Some(Q4KCase {
                    name: val.to_owned(),
                    block: Vec::new(),
                    d_bits: 0,
                    dmin_bits: 0,
                    sc: [0; 8],
                    mn: [0; 8],
                    q: Vec::new(),
                    y_bits: Vec::new(),
                });
            }
            "block" => cur.as_mut().expect("case first").block = hex_bytes(val),
            "d_bits" => {
                cur.as_mut().expect("case first").d_bits =
                    u16::from_str_radix(val, 16).expect("fixture d_bits valid");
            }
            "dmin_bits" => {
                cur.as_mut().expect("case first").dmin_bits =
                    u16::from_str_radix(val, 16).expect("fixture dmin_bits valid");
            }
            "sc" => {
                let parts: Vec<u8> = val
                    .split(' ')
                    .map(|p| p.parse().expect("sc valid"))
                    .collect();
                let sc: [u8; 8] = parts.try_into().expect("eight scales");
                cur.as_mut().expect("case first").sc = sc;
            }
            "mn" => {
                let parts: Vec<u8> = val
                    .split(' ')
                    .map(|p| p.parse().expect("mn valid"))
                    .collect();
                let mn: [u8; 8] = parts.try_into().expect("eight mins");
                cur.as_mut().expect("case first").mn = mn;
            }
            "q" => {
                cur.as_mut().expect("case first").q = val
                    .chars()
                    .map(|c| c.to_digit(16).expect("fixture nibble valid") as u8)
                    .collect();
            }
            "y" => {
                let words: Vec<u32> = val
                    .as_bytes()
                    .chunks_exact(8)
                    .map(|w| {
                        u32::from_str_radix(std::str::from_utf8(w).expect("ascii"), 16)
                            .expect("fixture word valid")
                    })
                    .collect();
                cur.as_mut().expect("case first").y_bits = words;
            }
            _ => panic!("unknown fixture key"),
        }
    }
    if let Some(c) = cur.take() {
        cases.push(c);
    }
    assert_eq!(cases.len(), 6, "six reference cases");
    for c in &cases {
        assert_eq!(c.block.len(), 144, "case {} block size", c.name);
        assert_eq!(c.q.len(), 256, "case {} nibbles", c.name);
        assert_eq!(c.y_bits.len(), 256, "case {} outputs", c.name);
    }
    cases
}

fn rand_f32(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut rng = SeededRng::new(seed);
    (0..count)
        .map(|_| ((rng.next_u64() % 20001) as f32 / 10000.0 - 1.0) * scale)
        .collect()
}

#[test]
fn scheme_ids_are_closed_stable_and_ordered() {
    assert_eq!(SchemeId::ALL.len(), 22);
    for (i, id) in SchemeId::ALL.iter().enumerate() {
        // Codes are contiguous from 1 in spec-table order (see scheme.rs).
        assert_eq!(id.code(), i as u64 + 1, "id {id}");
        assert_eq!(SchemeId::from_code(id.code()).unwrap(), *id);
        assert_eq!(SchemeId::from_name(id.name()).unwrap(), *id);
        assert_eq!(format!("{id}"), id.name());
        assert_eq!(id.to_string().parse::<SchemeId>().unwrap(), *id);
        assert!(id
            .name()
            .chars()
            .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()));
    }
    let natives: Vec<_> = SchemeId::ALL.iter().filter(|id| id.is_native()).collect();
    assert_eq!(natives.len(), 4);
    for id in SchemeId::ALL {
        if id.is_native() {
            assert_eq!(id.owner_card(), "A2.2", "native {id}");
        } else {
            assert!(
                id.owner_card() == "A2.3" || id.owner_card() == "A2.4",
                "reserved {id}"
            );
        }
    }
}

#[test]
fn unknown_scheme_codes_and_names_are_rejected() {
    for code in [0, 23, 100, 1000, u64::MAX] {
        assert!(
            matches!(
                SchemeId::from_code(code),
                Err(FormatError::UnknownScheme { .. })
            ),
            "code {code}"
        );
    }
    for name in ["", "I4_K", "Q4_K", "q4_k", "i8", "l1", "i8_R", " IQ4-xs"] {
        assert!(
            matches!(
                SchemeId::from_name(name),
                Err(FormatError::UnknownScheme { .. })
            ),
            "name {name}"
        );
    }
}

#[test]
fn ir_handle_carries_codes_without_a_second_meaning() {
    // One code table (this crate); r9v-ir only transports the handle.
    for id in SchemeId::ALL {
        assert_eq!(id.to_ir().as_u64(), id.code());
        assert_eq!(SchemeId::from_ir(id.to_ir()).unwrap(), id);
    }
    assert_eq!(
        SchemeId::from_ir(r9v_ir::SchemeId::new(3)).unwrap(),
        SchemeId::I4K
    );
    assert!(matches!(
        SchemeId::from_ir(r9v_ir::SchemeId::new(999)),
        Err(FormatError::UnknownScheme { .. })
    ));
}

#[test]
fn repack_only_ids_fail_closed_with_their_owner() {
    let dims = PaddedDims::new(16, 256, None).unwrap();
    for id in SchemeId::ALL {
        if id.is_native() {
            continue;
        }
        let owner = id.owner_card();
        let expect = |e: FormatError| match e {
            FormatError::ReservedScheme { scheme, owner: o } => {
                assert_eq!(scheme, id.name());
                assert_eq!(o, owner);
            }
            other => panic!("{id}: expected ReservedScheme, got {other:?}"),
        };
        expect(scale_record_bytes(id).unwrap_err());
        expect(outer_block(id).unwrap_err());
        expect(bits_per_weight(id, 256).unwrap_err());
        expect(scale_geometry(id, Layout::L1, &dims).unwrap_err());
        let scales = ScaleSet::f16(SchemeId::I8R, 0, 1.0).unwrap();
        expect(decode(id, QuantValue::I8(1), &scales).unwrap_err());
    }
}

#[test]
fn native_metadata_is_exact() {
    assert_eq!(scale_record_bytes(SchemeId::I8R).unwrap(), 2);
    assert_eq!(scale_record_bytes(SchemeId::I8B128).unwrap(), 2);
    assert_eq!(scale_record_bytes(SchemeId::I4K).unwrap(), 16);
    assert_eq!(scale_record_bytes(SchemeId::E4M3B128).unwrap(), 2);
    assert_eq!(outer_block(SchemeId::I8R).unwrap(), None);
    assert_eq!(outer_block(SchemeId::I8B128).unwrap(), Some(128));
    assert_eq!(outer_block(SchemeId::I4K).unwrap(), Some(256));
    assert_eq!(outer_block(SchemeId::E4M3B128).unwrap(), Some(128));
}

#[test]
fn bits_per_weight_matches_spec_section_8() {
    // I4_K: (2+2+12+128) B per 256 = 4.5 (Spec 2 §8).
    assert_eq!(bits_per_weight(SchemeId::I4K, 256).unwrap(), (1152, 256));
    assert_eq!(bits_per_weight(SchemeId::I4K, 512).unwrap(), (2304, 512));
    // I8_B128 / E4M3_B128: (128 + 2) B per 128 = 8.125 (Spec 2 §8).
    assert_eq!(bits_per_weight(SchemeId::I8B128, 128).unwrap(), (1040, 128));
    assert_eq!(
        bits_per_weight(SchemeId::E4M3B128, 384).unwrap(),
        (3120, 384)
    );
    // I8_R: 8 bits per weight plus one f16 per row (§8 prints 8.0).
    assert_eq!(bits_per_weight(SchemeId::I8R, 16).unwrap(), (144, 16));
    let (bits, weights) = bits_per_weight(SchemeId::I8R, 4096).unwrap();
    assert_eq!((bits, weights), (8 * 4096 + 16, 4096));
    let ratio = bits as f64 / weights as f64;
    assert!((ratio - 8.00390625).abs() < 1e-12, "ratio {ratio}");
    // Block divisibility and empty rows are rejected, never rounded.
    assert!(matches!(
        bits_per_weight(SchemeId::I8R, 0),
        Err(FormatError::InvalidDim { .. })
    ));
    for (scheme, bad) in [
        (SchemeId::I8B128, 0),
        (SchemeId::I8B128, 100),
        (SchemeId::E4M3B128, 127),
        (SchemeId::I4K, 0),
        (SchemeId::I4K, 128),
    ] {
        assert!(
            matches!(
                bits_per_weight(scheme, bad),
                Err(FormatError::InvalidBlock { .. })
            ),
            "{scheme} k={bad}"
        );
    }
}
#[test]
fn scale_soa_geometry_and_offsets_over_row_and_k_blocks() {
    // I4_K over N=32, K=512: 2 row-blocks x 2 K-blocks, 16 records of
    // 16 B each per (nb, kb): 64 records, 1024 B region.
    let dims = PaddedDims::new(32, 512, Some(256)).unwrap();
    let g = scale_geometry(SchemeId::I4K, Layout::L1, &dims).unwrap();
    assert_eq!(g.n_blocks, 2);
    assert_eq!(g.k_blocks, 2);
    assert_eq!(g.records, 64);
    assert_eq!(g.region_bytes, 1024);
    // Hand-computed offsets: ((nb*2 + kb)*16 + row) * 16.
    assert_eq!(g.record_offset(0, 0, 0).unwrap(), 0);
    assert_eq!(g.record_offset(0, 0, 15).unwrap(), 240);
    assert_eq!(g.record_offset(0, 1, 0).unwrap(), 256);
    assert_eq!(g.record_offset(1, 0, 0).unwrap(), 512);
    assert_eq!(g.record_offset(1, 1, 5).unwrap(), 848);
    assert_eq!(g.record_offset(1, 1, 15).unwrap(), 1008);
    // I8_B128 over N=16, K=256: 1 x 2 blocks, 32 records of 2 B.
    let dims = PaddedDims::new(16, 256, None).unwrap();
    let g = scale_geometry(SchemeId::I8B128, Layout::L1, &dims).unwrap();
    assert_eq!(
        (g.n_blocks, g.k_blocks, g.records, g.region_bytes),
        (1, 2, 32, 64)
    );
    assert_eq!(g.record_offset(0, 1, 15).unwrap(), 62);
    // I8_R over N=16, K=128: one K-block per row-block, 16 x 2 B.
    let dims = PaddedDims::new(16, 128, None).unwrap();
    let g = scale_geometry(SchemeId::I8R, Layout::L1, &dims).unwrap();
    assert_eq!(
        (g.n_blocks, g.k_blocks, g.records, g.region_bytes),
        (1, 1, 16, 32)
    );
    assert_eq!(g.record_offset(0, 0, 15).unwrap(), 30);
    // Out-of-range indices are all reported, never just the first.
    let err = g.record_offset(5, 7, 16).unwrap_err();
    match err {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 3),
        other => panic!("expected collect-all, got {other:?}"),
    }
    // SoA grouping does not exist on L0 (Spec 2 §2.1 trailing records).
    assert!(matches!(
        scale_geometry(SchemeId::I8B128, Layout::L0, &dims),
        Err(FormatError::UnsupportedLayout { .. })
    ));
    // K must divide the outer block (Spec 2 §2.2 padding).
    let bad = PaddedDims::new(16, 100, None).unwrap();
    assert!(matches!(
        scale_geometry(SchemeId::I8B128, Layout::L1, &bad),
        Err(FormatError::InvalidBlock { .. })
    ));
}

#[test]
fn l0_rows_compose_with_record_bytes() {
    // L0 rows carry trailing scale records (Spec 2 §2.1); A2.2 supplies
    // the record size, A2.1 the row geometry.
    let record = scale_record_bytes(SchemeId::I8B128).unwrap();
    let stride = l0_row_stride_bytes(64, 1, 2, record).unwrap();
    assert_eq!(stride, 68);
    let record = scale_record_bytes(SchemeId::I4K).unwrap();
    let stride = l0_row_stride_bytes(256, 1, 1, record).unwrap();
    assert_eq!(stride, 272);
}

#[test]
fn l1s_compressed_dims_share_the_soa_grouping() {
    // SI-14/SI-15 preserved: L1S scales group over the compressed-K
    // dims from card A2.1 with the §2.3 index law untouched.
    let dense = PaddedDims::new(32, 512, Some(256)).unwrap();
    let compressed = l1s_value_dims(&dense, Some(256)).unwrap();
    assert_eq!((compressed.n(), compressed.k()), (32, 256));
    let g = scale_geometry(SchemeId::I4K, Layout::L1S, &compressed).unwrap();
    assert_eq!(
        (g.n_blocks, g.k_blocks, g.records, g.region_bytes),
        (2, 1, 32, 512)
    );
    assert_eq!(g.record_offset(1, 0, 15).unwrap(), 496);
}

#[test]
fn f16_codec_matches_known_vectors() {
    assert_eq!(f16_to_f32(0x3C00), 1.0);
    assert_eq!(f16_to_f32(0xBC00), -1.0);
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    assert!(f16_to_f32(0x7C00).is_infinite());
    // Round-to-nearest-even emission, ties verified against numpy.
    assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
    assert_eq!(f32_to_f16_bits(-2.5), 0xC100);
    assert_eq!(f32_to_f16_bits(0.0), 0x0000);
    assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    assert_eq!(f32_to_f16_bits(1.000_488_3), 0x3C00);
    assert_eq!(f32_to_f16_bits(3.000_976_6), 0x4200);
    assert_eq!(f32_to_f16_bits(5.960_464_5e-8), 0x0001);
    assert_eq!(f32_to_f16_bits(1e-8), 0x0000);
    assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
    assert_eq!(f32_to_f16_bits(f32::NAN), 0x7E00);
    assert_eq!(f32_to_f16_bits(1e10), 0x7C00);
    // Checked scale emission rejects what scales must never be.
    assert_eq!(f16_scale_bits(1.0, "i8_r", 0).unwrap(), 0x3C00);
    assert_eq!(f16_scale_bits(0.0, "i8_r", 0).unwrap(), 0x0000);
    assert!(matches!(
        f16_scale_bits(f32::NAN, "i8_r", 3),
        Err(FormatError::InvalidScale { record: 3, .. })
    ));
    assert!(matches!(
        f16_scale_bits(f32::INFINITY, "i8_r", 0),
        Err(FormatError::InvalidScale {
            reason: "infinite",
            ..
        })
    ));
    assert!(matches!(
        f16_scale_bits(-1.0, "i8_r", 0),
        Err(FormatError::InvalidScale {
            reason: "negative",
            ..
        })
    ));
    assert!(matches!(
        f16_scale_bits(1e10, "i8_r", 0),
        Err(FormatError::InvalidScale {
            reason: "unrepresentable_in_f16",
            ..
        })
    ));
    // Exhaustive: every finite pattern round-trips; NaN patterns map
    // to canonical quiet NaN with the sign preserved, never wrapped.
    for bits in 0..=u16::MAX {
        let back = f32_to_f16_bits(f16_to_f32(bits));
        let exp = (bits >> 10) & 0x1F;
        let mant = bits & 0x3FF;
        if exp == 31 && mant != 0 {
            assert_eq!(back, (bits & 0x8000) | 0x7E00, "pattern {bits:#06x}");
        } else {
            assert_eq!(back, bits, "pattern {bits:#06x}");
        }
    }
    // Stored-scale validation carries scheme, record and reason.
    assert_eq!(check_f16_scale("i8_r", 7, 0x3C00).unwrap(), 1.0);
    assert!(matches!(
        check_f16_scale("i8_r", 7, 0x7C00),
        Err(FormatError::InvalidScale { record: 7, .. })
    ));
    assert!(matches!(
        check_f16_scale("i8_r", 0, 0xBC00),
        Err(FormatError::InvalidScale {
            reason: "negative",
            ..
        })
    ));
}

#[test]
fn e4m3_codec_matches_the_ocp_definition() {
    // Exact grid values (OCP E4M3, bias 7; see SI-20; cross-checked
    // against ml_dtypes float8_e4m3fn).
    assert_eq!(E4m3::new(0x00).to_f32(), 0.0);
    assert_eq!(E4m3::new(0x80).to_f32(), -0.0);
    assert_eq!(E4m3::new(0x01).to_f32(), 0.001953125);
    assert_eq!(E4m3::new(0x08).to_f32(), 0.015625);
    assert_eq!(E4m3::new(0x30).to_f32(), 0.5);
    assert_eq!(E4m3::new(0x38).to_f32(), 1.0);
    assert_eq!(E4m3::new(0x40).to_f32(), 2.0);
    assert_eq!(E4m3::new(0xB8).to_f32(), -1.0);
    assert_eq!(E4m3::new(0x6F).to_f32(), 120.0);
    assert_eq!(E4m3::new(0x77).to_f32(), 240.0);
    assert_eq!(E4m3::new(0x78).to_f32(), 256.0);
    assert_eq!(E4m3::new(0x7E).to_f32(), 448.0);
    assert!(E4m3::new(0x7F).is_nan());
    assert!(E4m3::new(0xFF).is_nan());
    assert!(!E4m3::new(0x7E).is_nan());
    assert!(E4m3::new(0x7F).check(9).is_err());
    assert!(E4m3::new(0x00).check(9).unwrap().bits() == 0x00);
    // Grid projection: exact values round-trip, ties go even, finite
    // overflow saturates, non-finite inputs are refused.
    assert_eq!(E4m3::from_f32(1.0).unwrap().bits(), 0x38);
    assert_eq!(E4m3::from_f32(-0.0).unwrap().bits(), 0x80);
    assert_eq!(E4m3::from_f32(248.0).unwrap().bits(), 0x78);
    assert_eq!(E4m3::from_f32(500.0).unwrap().bits(), 0x7E);
    assert_eq!(E4m3::from_f32(-500.0).unwrap().bits(), 0xFE);
    assert!(E4m3::from_f32(f32::NAN).is_none());
    assert!(E4m3::from_f32(f32::INFINITY).is_none());
    // Exhaustive: every finite pattern is a fixed point.
    for bits in 0..=255u8 {
        let v = E4m3::new(bits);
        if v.is_nan() {
            continue;
        }
        assert_eq!(
            E4m3::from_f32(v.to_f32()).unwrap().bits(),
            bits,
            "bits {bits:#04x}"
        );
    }
}
#[test]
fn i4k_header_matches_llamacpp_reference() {
    for case in load_q4k_reference() {
        let raw: &[u8; 16] = case.block[0..16].try_into().expect("header slice");
        let header = I4KSuperblock::from_bytes(raw);
        assert_eq!(header.d_bits(), case.d_bits, "case {}", case.name);
        assert_eq!(header.dmin_bits(), case.dmin_bits, "case {}", case.name);
        assert_eq!(header.scales(), case.sc, "case {}", case.name);
        assert_eq!(header.mins(), case.mn, "case {}", case.name);
        // Repack identity: parse → serialize is the same 12 bytes.
        assert_eq!(header.to_bytes(), *raw, "case {}", case.name);
        // Logical repack round-trips through the 6-bit validation.
        let repacked = I4KSuperblock::pack(
            header.d_bits(),
            header.dmin_bits(),
            header.scales(),
            header.mins(),
        )
        .unwrap();
        assert_eq!(repacked, header, "case {}", case.name);
    }
}

#[test]
fn i4k_decode_matches_reference_bit_exactly() {
    for case in load_q4k_reference() {
        let raw: &[u8; 16] = case.block[0..16].try_into().expect("header slice");
        let header = I4KSuperblock::from_bytes(raw);
        let out = decode_i4k_superblock(&case.q, &header).unwrap();
        assert_eq!(out.len(), 256, "case {}", case.name);
        for (i, (got, want)) in out.iter().zip(case.y_bits.iter()).enumerate() {
            assert_eq!(got.to_bits(), *want, "case {} index {i}", case.name);
        }
    }
}

#[test]
fn decode_dispatch_evaluates_the_exact_formulas() {
    let s = ScaleSet::f16(SchemeId::I8R, 0, 0.5).unwrap();
    assert_eq!(
        decode(SchemeId::I8R, QuantValue::I8(100), &s).unwrap(),
        50.0
    );
    assert_eq!(
        decode(SchemeId::I8B128, QuantValue::I8(-128), &s).unwrap(),
        -64.0
    );
    let e = ScaleSet::f16(SchemeId::E4M3B128, 0, 2.0).unwrap();
    assert_eq!(
        decode(SchemeId::E4M3B128, QuantValue::E4M3(E4m3::new(0x38)), &e).unwrap(),
        2.0
    );
    // w = (d·sc)·q − (dmin·mn): (0.5·2)·7 − (0.25·1) = 6.75.
    let k = ScaleSet::i4k(SchemeId::I4K, 0, 0.5, 2, 0.25, 1).unwrap();
    assert_eq!(decode(SchemeId::I4K, QuantValue::I4(7), &k).unwrap(), 6.75);
    // Wrong kinds are typed errors, never silent reinterpretation.
    assert!(matches!(
        decode(SchemeId::I8R, QuantValue::I4(7), &s),
        Err(FormatError::SchemeMismatch { .. })
    ));
    assert!(matches!(
        decode(SchemeId::I4K, QuantValue::I8(7), &k),
        Err(FormatError::SchemeMismatch { .. })
    ));
    assert!(matches!(
        decode(SchemeId::I4K, QuantValue::I4(16), &k),
        Err(FormatError::ValueOutOfRange { .. })
    ));
    assert!(matches!(
        decode(SchemeId::E4M3B128, QuantValue::E4M3(E4m3::new(0x7F)), &e),
        Err(FormatError::ValueOutOfRange { .. })
    ));
    assert!(matches!(
        ScaleSet::i4k(SchemeId::I4K, 0, 0.5, 64, 0.25, 1),
        Err(FormatError::ValueOutOfRange { .. })
    ));
    assert!(matches!(
        ScaleSet::f16(SchemeId::I8R, 2, f32::NAN),
        Err(FormatError::InvalidScale { record: 2, .. })
    ));
}

#[test]
fn i8_encode_decode_within_expected_error() {
    // Rounding is at most half a step (s/2); the stored f16 scale adds
    // at most s·127/2048 relative error, plus f32 arithmetic slack.
    for (seed, len) in [(0xA220u64, 512usize), (0xA221, 128), (0xA222, 4096)] {
        let x = rand_f32(seed, len, 1.5);
        let (q, scale) = encode_i8_row(&x).unwrap();
        assert_eq!(q.len(), len);
        let out = decode_i8_row(&q, &scale).unwrap();
        let s = f16_to_f32(scale.bits());
        let mut peak = 0.0f32;
        for (got, want) in out.iter().zip(x.iter()) {
            peak = peak.max(want.abs());
            assert!((got - want).abs() <= s * 0.57 + 5e-6, "seed {seed}");
        }
        assert!(peak > 0.5, "seed {seed} must exercise the range");
        // Determinism: the same input encodes to the same bytes.
        let again = encode_i8_row(&x).unwrap();
        assert_eq!(again.0, q);
        assert_eq!(again.1.bits(), scale.bits());
    }
    // Blocked form matches the row form block by block.
    let x = rand_f32(0xA223, 256, 2.0);
    let (q, scales) = encode_i8_block128(&x).unwrap();
    assert_eq!(scales.len(), 2);
    let out = decode_i8_block128(&q, &scales).unwrap();
    for (b, block) in x.chunks_exact(128).enumerate() {
        let s = scales[b].value(b as u64).unwrap();
        for (got, want) in out[b * 128..(b + 1) * 128].iter().zip(block.iter()) {
            assert!((got - want).abs() <= s * 0.57 + 5e-6, "block {b}");
        }
    }
    // Exact pins: symmetric grid keeps zero exact and ±127 maximal.
    let (q, scale) = encode_i8_row(&[1.0, -1.0, 0.0]).unwrap();
    assert_eq!(q, vec![127, -127, 0]);
    assert_eq!(
        scale.bits(),
        f16_scale_bits(1.0 / 127.0, "i8_r", 0).unwrap()
    );
    let (q, scale) = encode_i8_row(&[0.0, -0.0]).unwrap();
    assert_eq!(q, vec![0, 0]);
    assert_eq!(scale.bits(), 0);
}

#[test]
fn i4k_simple_encoder_error_within_expected_bound() {
    // Fine step is at most R/15 for input range R (half-step
    // R/30); super-scale and minimum quantization add at most R/63
    // each on top of it, so R/30 + 2R/63 < 0.05R bounds the error.
    for seed in [0xA224u64, 0xA225, 0xA226] {
        let x = rand_f32(seed, 256, 1.5);
        let hi = x.iter().fold(f32::NEG_INFINITY, |a, v| a.max(*v));
        let lo = x.iter().fold(f32::INFINITY, |a, v| a.min(*v)).min(0.0);
        let range = hi - lo;
        let arr: &[f32; 256] = x.as_slice().try_into().expect("256 inputs");
        let (q, header) = encode_i4k_superblock(arr).unwrap();
        let out = decode_i4k_superblock(&q, &header).unwrap();
        for (got, want) in out.iter().zip(x.iter()) {
            assert!((got - want).abs() <= range * 0.06 + 1e-4, "seed {seed}");
        }
        let again = encode_i4k_superblock(arr).unwrap();
        assert_eq!(again.0, q);
        assert_eq!(again.1, header);
    }
    // Degenerate all-zero superblock decodes exactly (cf. reference).
    let (q, header) = encode_i4k_superblock(&[0.0; 256]).unwrap();
    assert!(q.iter().all(|v| *v == 0));
    assert_eq!((header.scales(), header.mins()), ([0; 8], [0; 8]));
    assert_eq!(decode_i4k_superblock(&q, &header).unwrap(), vec![0.0; 256]);
}

#[test]
fn e4m3_block128_encode_decode_within_expected_error() {
    // Grid steps double per exponent (1/8 relative); the widest step
    // is 32 in the extended range, so rounding is at most 16 grid
    // units (SI-20). The stored f16 scale adds a negligible term.
    let mut x = rand_f32(0xA227, 256, 300.0);
    x[0] = 300.0;
    x[128] = -300.0;
    let (q, scales) = encode_e4m3_block128(&x).unwrap();
    assert_eq!(scales.len(), 2);
    let out = decode_e4m3_block128(&q, &scales).unwrap();
    for (b, block) in x.chunks_exact(128).enumerate() {
        let s = scales[b].value(b as u64).unwrap();
        for (got, want) in out[b * 128..(b + 1) * 128].iter().zip(block.iter()) {
            assert!(
                (got - want).abs() <= s * 16.5 + 1e-4,
                "block {b} want {want}"
            );
        }
    }
    let again = encode_e4m3_block128(&x).unwrap();
    let bits_again: Vec<u8> = again.0.iter().map(|v| v.bits()).collect();
    let bits_first: Vec<u8> = q.iter().map(|v| v.bits()).collect();
    assert_eq!(bits_again, bits_first);
    assert_eq!(again.1[0].bits(), scales[0].bits());
    // Exact pin: full-scale 448.0 hits the grid maximum exactly.
    let (q, scales) = encode_e4m3_block128(&[448.0; 128]).unwrap();
    assert!(q.iter().all(|v| v.bits() == 0x7E));
    assert_eq!(decode_e4m3_block128(&q, &scales).unwrap(), vec![448.0; 128]);
}

#[test]
fn decoders_reject_bad_shapes_and_collect_every_failure() {
    // Length and shape errors name what was required and found.
    assert!(matches!(
        decode_i8_row(&[], &I8RowScale::from_bits(0x3C00)),
        Err(FormatError::InvalidDim { .. })
    ));
    assert!(matches!(
        decode_i8_block128(&[0; 100], &[]),
        Err(FormatError::LengthMismatch {
            expected: 128,
            got: 100,
            ..
        })
    ));
    assert!(matches!(
        decode_i4k_superblock(&[0; 255], &I4KSuperblock::from_bytes(&[0; 16])),
        Err(FormatError::LengthMismatch {
            expected: 256,
            got: 255,
            ..
        })
    ));
    assert!(matches!(
        decode_e4m3_block128(&[], &[]),
        Err(FormatError::LengthMismatch { .. })
    ));
    // Scale-count mismatch plus two invalid scales: all three reported.
    let scales = vec![
        I8Block128Scale::from_bits(0x3C00),
        I8Block128Scale::from_bits(0x7C00),
        I8Block128Scale::from_bits(0xFC00),
    ];
    match decode_i8_block128(&[0; 256], &scales).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 3),
        other => panic!("expected collect-all, got {other:?}"),
    }
    // Three bad nibbles at scattered positions: all reported.
    let mut q = vec![0u8; 256];
    q[0] = 16;
    q[100] = 255;
    q[255] = 17;
    let header = I4KSuperblock::from_bytes(&[0x00, 0x3C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    match decode_i4k_superblock(&q, &header).unwrap_err() {
        FormatError::Multiple { problems } => {
            assert_eq!(problems.len(), 3);
            let positions: Vec<u64> = problems
                .iter()
                .map(|e| match e {
                    FormatError::ValueOutOfRange { position, .. } => *position,
                    other => panic!("unexpected {other:?}"),
                })
                .collect();
            assert_eq!(positions, vec![0, 100, 255]);
        }
        other => panic!("expected collect-all, got {other:?}"),
    }
    // NaN super-scales join nibble failures in one report.
    let bad_header =
        I4KSuperblock::from_bytes(&[0x00, 0x7E, 0x00, 0x7C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    match decode_i4k_superblock(&q, &bad_header).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 5),
        other => panic!("expected collect-all, got {other:?}"),
    }
    // NaN e4m3 values are rejected per position alongside scale errors.
    let mut q: Vec<E4m3> = vec![E4m3::new(0x40); 128];
    q[3] = E4m3::new(0x7F);
    q[120] = E4m3::new(0xFF);
    let scales = vec![E4M3Block128Scale::from_bits(0x7C00)];
    match decode_e4m3_block128(&q, &scales).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 3),
        other => panic!("expected collect-all, got {other:?}"),
    }
}

#[test]
fn encoders_reject_nonfinite_overflowing_and_misshaped_inputs() {
    assert!(matches!(
        encode_i8_row(&[]),
        Err(FormatError::InvalidDim { .. })
    ));
    assert!(matches!(
        encode_i8_block128(&[0.0; 100]),
        Err(FormatError::LengthMismatch { .. })
    ));
    assert!(matches!(
        encode_e4m3_block128(&[0.0; 100]),
        Err(FormatError::LengthMismatch { .. })
    ));
    // Every non-finite input is reported with its position.
    let x = [1.0, f32::NAN, 2.0, f32::INFINITY, 3.0, f32::NEG_INFINITY];
    match encode_i8_row(&x).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 3),
        other => panic!("expected collect-all, got {other:?}"),
    }
    assert!(encode_i4k_superblock(&[f32::NAN; 256]).is_err());
    // f16-overflowing scales are rejected, never infinited.
    assert!(matches!(
        encode_i8_row(&[f32::MAX, 1.0]),
        Err(FormatError::InvalidScale {
            reason: "unrepresentable_in_f16",
            ..
        })
    ));
    assert!(encode_i4k_superblock(&[f32::MAX; 256]).is_err());
    // u6 packing rejects out-of-range fields collectively.
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    sc[0] = 64;
    mn[7] = 100;
    match I4KSuperblock::pack(0, 0, sc, mn).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 2),
        other => panic!("expected collect-all, got {other:?}"),
    }
}

#[test]
fn huge_dims_fail_as_overflow_never_panic() {
    // Padded dims past u32 range fail at construction.
    assert!(matches!(
        PaddedDims::new(u32::MAX, 16, None),
        Err(FormatError::Overflow { .. })
    ));
    // Hand-built saturating geometry still reports overflow, not wrap.
    let g = ScaleGeometry {
        scheme: SchemeId::I4K,
        n_blocks: u64::MAX,
        k_blocks: u64::MAX,
        record_bytes: 12,
        records: u64::MAX,
        region_bytes: u64::MAX,
    };
    assert!(matches!(
        g.record_offset(1, 0, 0),
        Err(FormatError::Overflow { .. })
    ));
    assert!(matches!(
        g.record_offset(u64::MAX, 0, 0),
        Err(FormatError::InvalidDim { .. })
    ));
}
