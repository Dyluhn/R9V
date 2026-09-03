// SPDX-License-Identifier: Apache-2.0
//! IQ repack tests (Spec 2 §3.3, §7, §10;
//! phase-a-agent-breakdown.md card A2.4).
//!
//! Provenance: `tests/fixtures/r9v-format/iq_a24_reference.txt`.
//! gguf-py 0.19.0 exposes NO quantize helper for IQ types (every
//! `quantize_blocks` raises `NotImplementedError`), so wire bytes are
//! hand-built deterministically from the GGML block layout in
//! `quants.py`: seeded index/scale payloads plus all-zero and
//! all-`0xFF`-payload edge blocks with valid scales. This fixture wire
//! was not quantized by llama.cpp; the pinned gguf-py writer lacks IQ
//! quantizers. Expected `f32` words for every case come from gguf-py
//! 0.19.0 `dequantize_blocks`. Independent validation: local llama.cpp
//! source commit dd1ea524333b1e697489067d7a4c39c60d32beee
//! (build-vulkan-muse `libggml-base.so.0.19.0`) dequantized every
//! fixture row bit-exact: 23,680/23,680 `f32` words match `y` across
//! all 9 families. Regenerate with
//! `tests/fixtures/r9v-format/gen_iq_fixtures.py` (run from the
//! workspace root; append `--check` to verify byte-identical without
//! writing).

use r9v_format::{
    ggml_dequantize, repack, repack_bits_per_weight, repack_dequantize, repack_outer_block,
    repack_packing, repack_record_bytes, unpack_repacked, verify_padding_zeros_bytes,
    verify_padding_zeros_nibbles, FormatError, GgmlType, Packing, PaddedDims, SchemeId,
};

struct Case {
    name: String,
    n: u32,
    k: u32,
    wire: Vec<u8>,
    y: Vec<u8>,
}

fn hex_bytes(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
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

fn load_reference() -> Vec<Case> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/r9v-format/iq_a24_reference.txt"
    );
    let text = std::fs::read_to_string(path).expect("fixture file present");
    let mut cases = Vec::new();
    let mut cur: Option<Case> = None;
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
                cur = Some(Case {
                    name: val.to_owned(),
                    n: 0,
                    k: 0,
                    wire: Vec::new(),
                    y: Vec::new(),
                });
            }
            "n" => cur.as_mut().expect("case first").n = val.parse().expect("fixture n valid"),
            "k" => cur.as_mut().expect("case first").k = val.parse().expect("fixture k valid"),
            "wire" => cur.as_mut().expect("case first").wire = hex_bytes(val),
            "y" => cur.as_mut().expect("case first").y = hex_bytes(val),
            _ => panic!("unknown fixture key"),
        }
    }
    if let Some(c) = cur.take() {
        cases.push(c);
    }
    assert_eq!(cases.len(), 9, "one case per IQ family");
    for c in &cases {
        assert_eq!(
            c.y.len(),
            c.n as usize * c.k as usize * 4,
            "case {} words",
            c.name
        );
    }
    cases
}

fn ggml(name: &str) -> GgmlType {
    GgmlType::from_name(name).expect("fixture names are known types")
}

fn f32_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "f32 words need whole words");
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

fn to_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Index granularity: weights covered by one packed index byte
/// (mirrors `crate::iq`; the test recomputes geometry from wire facts
/// rather than trusting the implementation's helper).
fn granularity(name: &str) -> u32 {
    match name {
        "IQ4_NL" | "IQ4_XS" => 1,
        "IQ3_XXS" | "IQ3_S" | "IQ2_XXS" | "IQ2_XS" => 4,
        "IQ2_S" | "IQ1_S" | "IQ1_M" => 8,
        _ => panic!("unknown iq type {name}"),
    }
}

#[test]
fn iq_codes_names_and_schemes_match_gguf_py() {
    // (name, code, block_len, block_bytes, scheme); codes and sizes are
    // gguf-py 0.19.0 GGMLQuantizationType / GGML_QUANT_SIZES facts.
    let table: &[(&str, u32, u32, u64, SchemeId)] = &[
        ("IQ2_XXS", 16, 256, 66, SchemeId::Iq2Xxs),
        ("IQ2_XS", 17, 256, 74, SchemeId::Iq2Xs),
        ("IQ3_XXS", 18, 256, 98, SchemeId::Iq3Xxs),
        ("IQ1_S", 19, 256, 50, SchemeId::Iq1S),
        ("IQ4_NL", 20, 32, 18, SchemeId::I4Nl),
        ("IQ3_S", 21, 256, 110, SchemeId::Iq3S),
        ("IQ2_S", 22, 256, 82, SchemeId::Iq2S),
        ("IQ4_XS", 23, 256, 136, SchemeId::I4Xs),
        ("IQ1_M", 29, 256, 56, SchemeId::Iq1M),
    ];
    assert_eq!(GgmlType::ALL.len(), 21);
    for (name, code, block_len, block_bytes, scheme) in table {
        let ggml = GgmlType::from_name(name).expect("table name known");
        assert_eq!(ggml.code(), *code, "{name}");
        assert_eq!(ggml.block_len(), *block_len, "{name}");
        assert_eq!(ggml.block_bytes(), *block_bytes, "{name}");
        assert_eq!(ggml.scheme(), Some(*scheme), "{name}");
        assert_eq!(GgmlType::from_code(*code).expect("table code known"), ggml);
        assert_eq!(format!("{ggml}"), *name);
        assert_eq!(name.parse::<GgmlType>().expect("table name parses"), ggml);
        assert!(ggml.is_quantized(), "{name}");
        assert!(!scheme.is_native(), "{name} is repack-only");
        assert_eq!(scheme.owner_card(), "A2.4", "{name}");
    }
}

#[test]
fn source_dequant_matches_gguf_py_bit_exact() {
    let mut total = 0;
    for case in load_reference() {
        let ggml = ggml(&case.name);
        let got = ggml_dequantize(ggml, &case.wire, case.n, case.k).expect("fixture wire is valid");
        assert_eq!(got.len() as u32, case.n * case.k, "case {}", case.name);
        assert_eq!(to_bits(&got), f32_words(&case.y), "case {}", case.name);
        total += got.len();
    }
    // 640 (IQ4_NL 5x128) + 1536 (IQ4_XS 3x512) + 7 x 3072 (3x1024).
    assert_eq!(total, 640 + 1536 + 7 * 3072);
}

#[test]
fn repack_round_trip_matches_source_bit_exact() {
    for case in load_reference() {
        let ggml = ggml(&case.name);
        let source =
            ggml_dequantize(ggml, &case.wire, case.n, case.k).expect("fixture wire is valid");
        let t = repack(ggml, &case.wire, case.n, case.k).expect("fixture repacks");
        assert_eq!(t.ggml, ggml, "case {}", case.name);
        assert_eq!(t.scheme, ggml.scheme(), "case {}", case.name);
        // Repacked decode equals the source decode bit-exact (Spec 2 §10).
        let repacked = repack_dequantize(&t).expect("repacked decodes");
        assert_eq!(to_bits(&repacked), to_bits(&source), "case {}", case.name);
        assert_eq!(to_bits(&repacked), f32_words(&case.y), "case {}", case.name);
        // The inverse reproduces the wire bytes exactly (bijective).
        assert_eq!(
            unpack_repacked(&t).expect("repacked unpacks"),
            case.wire,
            "case {}",
            case.name
        );
        // Deterministic: the same input repacks to the same bytes.
        let again = repack(ggml, &case.wire, case.n, case.k).expect("fixture repacks");
        assert_eq!(t, again, "case {}", case.name);
    }
}

#[test]
fn repacked_regions_follow_canonical_geometry() {
    for case in load_reference() {
        let ggml = ggml(&case.name);
        let t = repack(ggml, &case.wire, case.n, case.k).expect("fixture repacks");
        let dims = PaddedDims::new(case.n, case.k, ggml.superblock_k()).expect("valid dims");
        assert_eq!(t.dims, dims, "case {}", case.name);
        // Value region: nibbles over the weight dims for the IQ4
        // types, index bytes over [N, K/g] for the grid types.
        let gran = granularity(&case.name);
        let value_dims = if gran == 1 {
            dims
        } else {
            PaddedDims::new(case.n, case.k / gran, None).expect("valid index dims")
        };
        let packing = match t.scheme {
            Some(s) => repack_packing(s).expect("iq packing known"),
            None => panic!("iq case {} must map to a scheme", case.name),
        };
        assert_eq!(
            packing,
            if gran == 1 {
                Packing::Nibble4
            } else {
                Packing::Byte
            },
            "case {}",
            case.name
        );
        assert_eq!(
            t.values.len() as u64,
            value_dims
                .value_region_bytes(packing)
                .expect("region sizes"),
            "case {}",
            case.name
        );
        // Scale region is exactly the §3.1 SoA grouping over 256-blocks.
        let scheme = t.scheme.expect("iq scheme known");
        let record = repack_record_bytes(scheme).expect("record known") as usize;
        let outer = repack_outer_block(scheme)
            .expect("outer known")
            .expect("no row-wise iq scheme") as usize;
        let n_blocks = dims.n_padded() as usize / 16;
        let k_blocks = dims.k_padded() as usize / outer;
        assert_eq!(
            t.scales.len(),
            n_blocks * k_blocks * 16 * record,
            "case {}",
            case.name
        );
        // Padding is zero in the value region (odd N in the IQ4_NL
        // case exercises padded rows; K is a multiple of 256
        // everywhere so K padding only shows past n).
        if gran == 1 {
            verify_padding_zeros_nibbles(&t.values, &dims).expect("zero padding");
        } else {
            verify_padding_zeros_bytes(&t.values, &value_dims).expect("zero padding");
        }
        // Padding rows carry zero scale records.
        if case.n % 16 != 0 {
            let pad_row = case.n as usize;
            for kb in 0..k_blocks {
                let base = (kb * 16 + pad_row) * record;
                assert!(
                    t.scales[base..base + record].iter().all(|b| *b == 0),
                    "case {} kb {kb}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn soa_records_follow_the_documented_wire_layouts() {
    // Row 0's SoA record equals the documented wire bytes for every
    // family (the record layout each `repack_record_bytes` size pins).
    // Offsets below are gguf-py quants.py block-layout facts.
    for case in load_reference() {
        let ggml = ggml(&case.name);
        let t = repack(ggml, &case.wire, case.n, case.k).expect("fixture repacks");
        let scheme = t.scheme.expect("iq scheme known");
        let record = repack_record_bytes(scheme).expect("record known") as usize;
        let wire0 = &case.wire[..ggml.block_bytes() as usize];
        let rec0 = &t.scales[..record];
        let mut expected = Vec::new();
        match case.name.as_str() {
            "IQ4_NL" => expected.extend_from_slice(&wire0[0..2]),
            "IQ4_XS" => expected.extend_from_slice(&wire0[0..8]),
            "IQ3_XXS" => {
                expected.extend_from_slice(&wire0[0..2]);
                expected.extend_from_slice(&wire0[66..98]);
            }
            "IQ3_S" => {
                expected.extend_from_slice(&wire0[0..2]);
                expected.extend_from_slice(&wire0[66..74]);
                expected.extend_from_slice(&wire0[74..106]);
                expected.extend_from_slice(&wire0[106..110]);
            }
            "IQ2_XXS" => expected.extend_from_slice(&wire0[0..2]),
            "IQ2_XS" => {
                expected.extend_from_slice(&wire0[0..2]);
                expected.extend_from_slice(&wire0[66..74]);
            }
            "IQ2_S" => {
                expected.extend_from_slice(&wire0[0..2]);
                expected.extend_from_slice(&wire0[34..82]);
            }
            "IQ1_S" => {
                expected.extend_from_slice(&wire0[0..2]);
                expected.extend_from_slice(&wire0[34..50]);
            }
            "IQ1_M" => expected.extend_from_slice(&wire0[32..56]),
            _ => panic!("unknown iq case"),
        }
        assert_eq!(rec0, expected.as_slice(), "case {}", case.name);
    }
}

#[test]
fn exact_bits_per_weight_match_wire_sizes() {
    // (scheme, k, bits, weights): wire bytes * 8 over the block.
    let table: &[(SchemeId, u32, u64, u64)] = &[
        (SchemeId::I4Nl, 32, 144, 32),
        (SchemeId::I4Xs, 256, 1088, 256),
        (SchemeId::Iq3Xxs, 256, 784, 256),
        (SchemeId::Iq3S, 256, 880, 256),
        (SchemeId::Iq2Xxs, 256, 528, 256),
        (SchemeId::Iq2Xs, 256, 592, 256),
        (SchemeId::Iq2S, 256, 656, 256),
        (SchemeId::Iq1S, 256, 400, 256),
        (SchemeId::Iq1M, 256, 448, 256),
    ];
    for (scheme, k, bits, weights) in table {
        assert_eq!(
            repack_bits_per_weight(*scheme, *k).expect("iq bpw known"),
            (*bits, *weights),
            "{scheme}"
        );
        assert_eq!(
            repack_bits_per_weight(*scheme, *k * 2).expect("scaled bpw"),
            (bits * 2, weights * 2),
            "{scheme}"
        );
    }
    assert!(matches!(
        repack_bits_per_weight(SchemeId::I4Nl, 33),
        Err(FormatError::InvalidBlock { .. })
    ));
    assert!(matches!(
        repack_bits_per_weight(SchemeId::Iq2S, 128),
        Err(FormatError::InvalidBlock { .. })
    ));
    assert!(matches!(
        repack_bits_per_weight(SchemeId::Iq1M, 0),
        Err(FormatError::InvalidBlock { .. })
    ));
}

#[test]
fn malformed_inputs_fail_closed_and_collect_all() {
    // Zero dims collect every problem before returning.
    for name in [
        "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M",
    ] {
        let ggml = ggml(name);
        assert!(
            ggml_dequantize(ggml, &[], 0, 0).is_err(),
            "{name} zero dims must fail"
        );
        // Misaligned K names the block.
        assert!(
            matches!(
                ggml_dequantize(ggml, &[], 1, 7),
                Err(FormatError::InvalidBlock { .. })
            ),
            "{name} misaligned k must fail"
        );
        // Truncated wire names both lengths.
        assert!(
            matches!(
                repack(ggml, &[0u8; 3], 1, ggml.block_len()),
                Err(FormatError::LengthMismatch { .. })
            ),
            "{name} truncated wire must fail"
        );
    }
    // u32::MAX dims overflow without allocating.
    assert!(matches!(
        repack(GgmlType::IQ2_S, &[], u32::MAX, u32::MAX),
        Err(FormatError::Overflow { .. }
            | FormatError::Multiple { .. }
            | FormatError::InvalidBlock { .. })
    ));
    // A forged tensor with inconsistent regions fails by length.
    for case in load_reference() {
        let ggml = ggml(&case.name);
        let mut t = repack(ggml, &case.wire, case.n, case.k).expect("fixture repacks");
        t.values.pop();
        assert!(
            unpack_repacked(&t).is_err() && repack_dequantize(&t).is_err(),
            "case {} short values must fail",
            case.name
        );
        let mut t = repack(ggml, &case.wire, case.n, case.k).expect("fixture repacks");
        t.scales.pop();
        assert!(
            unpack_repacked(&t).is_err() && repack_dequantize(&t).is_err(),
            "case {} short scales must fail",
            case.name
        );
    }
    // A forged scheme agreement fails by mismatch.
    let case = load_reference()
        .into_iter()
        .find(|c| c.name == "IQ2_S")
        .expect("iq2_s case present");
    let mut t = repack(GgmlType::IQ2_S, &case.wire, case.n, case.k).expect("fixture repacks");
    t.scheme = Some(SchemeId::Iq2Xs);
    assert!(matches!(
        unpack_repacked(&t),
        Err(FormatError::SchemeMismatch { .. })
    ));
    assert!(matches!(
        repack_dequantize(&t),
        Err(FormatError::SchemeMismatch { .. })
    ));
}

/// Writes `d` (f16 bits) into a zeroed wire block of `name`.
/// `IQ1_M` packs `d` across the top nibbles of the four scale words at
/// bytes 48..56 with word `i` carrying bits `4*i..4*i+3` (mirrors the
/// `crate::iq` decode: `((w0 & 0xF000) >> 12) | ((w1 & 0xF000) >> 8) |
/// ...`); every other family stores `d` little-endian at bytes 0..2.
fn pour_scale(name: &str, wire: &mut [u8], bits: u16) {
    if name == "IQ1_M" {
        for (i, shift) in [0u32, 4, 8, 12].into_iter().enumerate() {
            let w = ((bits >> shift) & 0xF) << 12;
            wire[48 + 2 * i] = (w & 0xFF) as u8;
            wire[49 + 2 * i] = (w >> 8) as u8;
        }
    } else {
        wire[0] = (bits & 0xFF) as u8;
        wire[1] = (bits >> 8) as u8;
    }
}

#[test]
fn non_finite_scales_rejected_and_negative_accepted_on_both_paths() {
    // NaN and Inf d fail with InvalidScale naming the block on both
    // readers, for all nine families; negative finite d decodes (wire
    // reality, as in A2.3).
    for (bits, ok) in [
        (0x7E00u16, false),
        (0x7C00, false),
        (0xFC00, false),
        (0xBC00, true),
    ] {
        for name in [
            "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M",
        ] {
            let ggml = ggml(name);
            let bb = ggml.block_bytes() as usize;
            let bl = ggml.block_len();
            let mut wire = vec![0u8; bb];
            pour_scale(name, &mut wire, bits);
            let source = ggml_dequantize(ggml, &wire, 1, bl);
            let repacked = repack(ggml, &wire, 1, bl).map(|t| repack_dequantize(&t));
            if ok {
                let s = source.expect("negative d decodes");
                assert!(s.iter().all(|v| v.is_finite()), "{name} negative d finite");
                let r = repacked.expect("repack ok").expect("negative d decodes");
                assert_eq!(
                    r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    s.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{name} paths agree"
                );
            } else {
                assert!(
                    matches!(source, Err(FormatError::InvalidScale { .. })),
                    "{name} NaN/Inf must fail on source"
                );
                match repacked {
                    Ok(Ok(_)) => panic!("{name} NaN/Inf must fail on repacked"),
                    Ok(Err(FormatError::InvalidScale { .. })) => {}
                    Ok(Err(other)) => panic!("{name} wrong repacked error: {other:?}"),
                    Err(other) => panic!("{name} repack itself failed: {other:?}"),
                }
            }
        }
    }
}

#[test]
fn lut_data_matches_gguf_py_layout_facts() {
    // Lengths pin the grid shapes; spot values pin the content
    // (gguf-py grid_map/kvalues facts, not recomputed here).
    assert_eq!(r9v_format::IQ4_KVALUES.len(), 16);
    assert_eq!(r9v_format::IQ4_KVALUES[0], -127);
    assert_eq!(r9v_format::IQ4_KVALUES[15], 113);
    assert_eq!(r9v_format::IQ_SIGN_LUT.len(), 128);
    assert_eq!(r9v_format::IQ_SIGN_LUT[0], 0x00);
    assert_eq!(r9v_format::IQ2_XXS_GRID.len(), 256 * 8);
    assert_eq!(r9v_format::IQ2_XS_GRID.len(), 512 * 8);
    assert_eq!(r9v_format::IQ2_S_GRID.len(), 1024 * 8);
    assert_eq!(r9v_format::IQ3_XXS_GRID.len(), 256 * 4);
    assert_eq!(r9v_format::IQ3_S_GRID.len(), 512 * 4);
    assert_eq!(r9v_format::IQ1_GRID.len(), 2048 * 8);
    // Grid slots stay within their documented value sets.
    assert!(r9v_format::IQ2_XXS_GRID
        .iter()
        .all(|v| *v == 8 || *v == 25 || *v == 43));
    assert!(r9v_format::IQ3_XXS_GRID
        .iter()
        .all(|v| [4, 12, 20, 28, 36, 44, 52, 62].contains(v)));
    assert!(r9v_format::IQ3_S_GRID
        .iter()
        .all(|v| *v % 2 == 1 && *v <= 15));
    assert!(r9v_format::IQ1_GRID.iter().all(|v| *v >= -1 && *v <= 1));
    // Totality: all-0xFF index payloads with a valid scale decode on
    // both paths without panic (every index pattern is legal; only
    // scales still reject non-finite).
    for name in [
        "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M",
    ] {
        let ggml = ggml(name);
        let bb = ggml.block_bytes() as usize;
        let bl = ggml.block_len();
        let mut wire = vec![0xFFu8; bb];
        if name == "IQ1_M" {
            // Top nibbles of the four scale words form d (little-endian
            // u16s at 48..56, word i carrying bits 4*i..4*i+3): force a
            // finite d (0x0C03), keep the low bits.
            wire[49] = (wire[49] & 0x0F) | 0x30;
            wire[51] = (wire[51] & 0x0F) | 0xC0;
            wire[53] &= 0x0F;
            wire[55] &= 0x0F;
        } else {
            wire[0] = 0x00;
            wire[1] = 0x3C;
        }
        let source = ggml_dequantize(ggml, &wire, 1, bl).expect("0xFF payload decodes");
        assert_eq!(source.len() as u32, bl, "{name}");
        assert!(source.iter().all(|v| v.is_finite()), "{name}");
        let t = repack(ggml, &wire, 1, bl).expect("0xFF payload repacks");
        let repacked = repack_dequantize(&t).expect("0xFF payload repacked-decodes");
        assert_eq!(
            repacked.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            source.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}"
        );
        assert_eq!(unpack_repacked(&t).expect("inverse"), wire, "{name}");
    }
}
