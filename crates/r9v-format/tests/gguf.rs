// SPDX-License-Identifier: Apache-2.0
//! GGUF repack tests (Spec 2 §3.3, §7, §10; card A2.3).
//!
//! Provenance: `tests/fixtures/r9v-format/gguf_a23_reference.txt`.
//! `Q8_0`/`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`F16`/`BF16` wire bytes come from
//! gguf-py 0.19.0 `quantize` over seeded tensors; `Q2_K`/`Q3_K`/`Q4_K`/
//! `Q5_K`/`Q6_K` wire bytes are hand-built deterministically from the
//! GGML block layout because gguf-py 0.19.0 exposes no quantize helper
//! for K-quants (honest statement, not llama.cpp output). Expected
//! `f32` words for every case come from gguf-py 0.19.0 `dequantize`.
//! Regenerate with `/tmp/gen_a23_fixtures.py`
//! (`PYTHONPATH=/tmp/ggufsrc python3 /tmp/gen_a23_fixtures.py`).

use r9v_format::{
    bf16_to_f32, ggml_dequantize, l1_forward_index, repack, repack_bits_per_weight,
    repack_dequantize, repack_outer_block, repack_packing, repack_record_bytes, unpack_repacked,
    verify_padding_zeros_bytes, verify_padding_zeros_nibbles, verify_padding_zeros_planes,
    FormatError, GgmlType, PaddedDims, SchemeId,
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
        "/tests/fixtures/r9v-format/gguf_a23_reference.txt"
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
    assert_eq!(cases.len(), 12, "one case per A2.3 source type");
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

#[test]
fn ggml_codes_names_and_schemes_match_gguf_py() {
    // (name, code, block_len, block_bytes, scheme)
    let table: &[(&str, u32, u32, u64, Option<SchemeId>)] = &[
        ("F16", 1, 1, 2, None),
        ("Q4_0", 2, 32, 18, Some(SchemeId::I4B32F)),
        ("Q4_1", 3, 32, 20, Some(SchemeId::I4B32FM)),
        ("Q5_0", 6, 32, 22, Some(SchemeId::I5B32F)),
        ("Q5_1", 7, 32, 24, Some(SchemeId::I5B32FM)),
        ("Q8_0", 8, 32, 34, Some(SchemeId::I8B32F)),
        ("Q2_K", 10, 256, 84, Some(SchemeId::I2K)),
        ("Q3_K", 11, 256, 110, Some(SchemeId::I3K)),
        ("Q4_K", 12, 256, 144, Some(SchemeId::I4K)),
        ("Q5_K", 13, 256, 176, Some(SchemeId::I5K)),
        ("Q6_K", 14, 256, 210, Some(SchemeId::I6K)),
        ("IQ2_XXS", 16, 256, 66, Some(SchemeId::Iq2Xxs)),
        ("IQ2_XS", 17, 256, 74, Some(SchemeId::Iq2Xs)),
        ("IQ3_XXS", 18, 256, 98, Some(SchemeId::Iq3Xxs)),
        ("IQ1_S", 19, 256, 50, Some(SchemeId::Iq1S)),
        ("IQ4_NL", 20, 32, 18, Some(SchemeId::I4Nl)),
        ("IQ3_S", 21, 256, 110, Some(SchemeId::Iq3S)),
        ("IQ2_S", 22, 256, 82, Some(SchemeId::Iq2S)),
        ("IQ4_XS", 23, 256, 136, Some(SchemeId::I4Xs)),
        ("IQ1_M", 29, 256, 56, Some(SchemeId::Iq1M)),
        ("BF16", 30, 1, 2, None),
    ];
    assert_eq!(GgmlType::ALL.len(), 21);
    for (name, code, block_len, block_bytes, scheme) in table {
        let ggml = GgmlType::from_name(name).expect("table name known");
        assert_eq!(ggml.code(), *code, "{name}");
        assert_eq!(ggml.block_len(), *block_len, "{name}");
        assert_eq!(ggml.block_bytes(), *block_bytes, "{name}");
        assert_eq!(ggml.scheme(), *scheme, "{name}");
        assert_eq!(GgmlType::from_code(*code).expect("table code known"), ggml);
        assert_eq!(format!("{ggml}"), *name);
        assert_eq!(name.parse::<GgmlType>().expect("table name parses"), ggml);
        assert_eq!(ggml.is_quantized(), scheme.is_some(), "{name}");
    }
    // Q4_K shares the native I4_K record by design (Spec 2 §3.2, D-004).
    assert_eq!(GgmlType::Q4_K.scheme(), Some(SchemeId::I4K));
    assert!(SchemeId::I4K.is_native());
}

#[test]
fn unknown_ggml_codes_are_hard_errors_naming_the_type() {
    // 16-23 and 29 are the card-A2.4 IQ codes; the remaining gaps
    // (including 4/5/9/15/24-28/31) stay hard errors.
    for code in [0, 4, 5, 9, 15, 24, 25, 26, 27, 28, 31, 34, 1000, u32::MAX] {
        match GgmlType::from_code(code) {
            Err(FormatError::UnknownGgmlType { code: got }) => assert_eq!(got, code),
            other => panic!("code {code}: expected UnknownGgmlType, got {other:?}"),
        }
        let text = format!("{}", FormatError::UnknownGgmlType { code });
        assert!(text.contains(&code.to_string()), "error names {code}");
    }
    for name in ["", "q4_0", "Q4_2", "Q8_K", "IQ5_0", "F32", "Q4-0"] {
        assert!(
            GgmlType::from_name(name).is_err(),
            "name {name} must be rejected"
        );
    }
}

#[test]
fn source_dequant_matches_gguf_py_bit_exact() {
    for case in load_reference() {
        let ggml = ggml(&case.name);
        let got = ggml_dequantize(ggml, &case.wire, case.n, case.k).expect("fixture wire is valid");
        assert_eq!(got.len() as u32, case.n * case.k, "case {}", case.name);
        assert_eq!(to_bits(&got), f32_words(&case.y), "case {}", case.name);
    }
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
        // The inverse reproduces the wire bytes exactly (pure permutation).
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
fn repacked_regions_follow_canonical_l1_geometry() {
    for case in load_reference() {
        let ggml = ggml(&case.name);
        let t = repack(ggml, &case.wire, case.n, case.k).expect("fixture repacks");
        let dims = PaddedDims::new(case.n, case.k, ggml.superblock_k()).expect("valid dims");
        assert_eq!(t.dims, dims, "case {}", case.name);
        // Value region is exactly the L1 tile stream for the packing.
        let packing = match t.scheme {
            Some(s) => repack_packing(s).expect("A2.3 packing known"),
            None => r9v_format::Packing::Half16,
        };
        assert_eq!(
            t.values.len() as u64,
            dims.value_region_bytes(packing).expect("region sizes"),
            "case {}",
            case.name
        );
        // Scale region is exactly the §3.1 SoA grouping (empty for halves).
        let (record, outer) = match t.scheme {
            Some(s) => (
                repack_record_bytes(s).expect("record known") as usize,
                repack_outer_block(s)
                    .expect("outer known")
                    .expect("no row-wise A2.3 scheme") as usize,
            ),
            None => (0, dims.k_padded() as usize),
        };
        assert_eq!(
            t.scales.len(),
            dims.n_padded() as usize / 16 * (dims.k_padded() as usize / outer) * 16 * record,
            "case {}",
            case.name
        );
        // Padding beyond (n, k) reads back as zero in tile order.
        match packing {
            r9v_format::Packing::Byte => {
                verify_padding_zeros_bytes(&t.values, &dims).expect("zero padding")
            }
            r9v_format::Packing::Nibble4 => {
                verify_padding_zeros_nibbles(&t.values, &dims).expect("zero padding")
            }
            r9v_format::Packing::Half16 => {
                let tiled = r9v_format::decode_halfs_le(&t.values).expect("even halves");
                let back = r9v_format::l1_unpack_halfs(&tiled, &dims).expect("halves unpack");
                for n in case.n..dims.n_padded() {
                    for k in 0..dims.k_padded() {
                        assert_eq!(back[(n * dims.k_padded() + k) as usize], 0);
                    }
                }
            }
            r9v_format::Packing::BitPlanes { bits } => {
                verify_padding_zeros_planes(&t.values, &dims, bits).expect("zero padding")
            }
        }
    }
}

#[test]
fn lane_order_matches_the_a0s1_formula_on_repacked_bytes() {
    // Q8_0 stores raw bytes, so the L1 lane law is directly visible:
    // wire (row, col) sits at l1_forward_index(row, col).
    let case = load_reference()
        .into_iter()
        .find(|c| c.name == "Q8_0")
        .expect("Q8_0 fixture present");
    let t = repack(GgmlType::Q8_0, &case.wire, case.n, case.k).expect("Q8_0 repacks");
    let dims = t.dims;
    // Byte (0,0) of row 0 block 0 is wire[2] (after the f16 scale).
    let pos = l1_forward_index(0, 0, &dims).expect("index valid");
    assert_eq!(t.values[pos as usize], case.wire[2]);
    // First byte of the second 32-block lands at (0, 32).
    let pos = l1_forward_index(0, 32, &dims).expect("index valid");
    assert_eq!(t.values[pos as usize], case.wire[34 + 2]);
    // Row 1 starts after row 0's four blocks.
    let pos = l1_forward_index(1, 0, &dims).expect("index valid");
    assert_eq!(t.values[pos as usize], case.wire[4 * 34 + 2]);
}

#[test]
fn q4k_q5k_scales_match_the_i4k_record() {
    // The A2.2 I4K record is field-identical to the Q4_K/Q5_K header
    // (D-004, Spec 2 §3.2): parsing each fixture header with the A2.2
    // type must reproduce the SoA record bytes the repack stored.
    for (name, scheme) in [("Q4_K", SchemeId::I4K), ("Q5_K", SchemeId::I5K)] {
        let case = load_reference()
            .into_iter()
            .find(|c| c.name == name)
            .expect("K fixture present");
        let ggml = ggml(&case.name);
        let block_bytes = ggml.block_bytes() as usize;
        let t = repack(ggml, &case.wire, case.n, case.k).expect("K repacks");
        assert_eq!(t.scheme, Some(scheme));
        let k_blocks = case.k as usize / 256;
        for (b, block) in case.wire.chunks_exact(block_bytes).enumerate() {
            let header = r9v_format::I4KSuperblock::from_bytes(
                block[0..16].try_into().expect("header slice"),
            );
            // Repack identity through the A2.2 type (parse → serialize).
            assert_eq!(header.to_bytes(), block[0..16], "{name} block {b}");
            // The SoA record for (row, k-block) holds the same 16 bytes.
            let row = b / k_blocks;
            let kb = b % k_blocks;
            let base = ((row / 16 * k_blocks + kb) * 16 + row % 16) * 16;
            assert_eq!(
                &t.scales[base..base + 16],
                &block[0..16],
                "{name} block {b}"
            );
            // d/dmin bits survive the repack untouched.
            let d_bits = u16::from_le_bytes([t.scales[base], t.scales[base + 1]]);
            assert_eq!(d_bits, header.d_bits(), "{name} block {b}");
            // The ggml scale unpack agrees with the A2.2 record fields.
            let (sc, mn) = r9v_format::unpack_k4_scales(&block[4..16]).expect("payload valid");
            assert_eq!(sc, header.scales(), "{name} block {b}");
            assert_eq!(mn, header.mins(), "{name} block {b}");
        }
    }
}

#[test]
fn negative_wire_scales_decode() {
    // Real writers emit negative `d` when the block extremum is
    // positive (gguf-py Q4_0 `d = max / -8`); the sign is absorbed by
    // the zero-point form, so the reference decode accepts it.
    let mut wire = vec![0u8; 34];
    // -0.5 in f16 is 0xB800 (sign 1, exp 14, mantissa 0).
    wire[0..2].copy_from_slice(&0xB800u16.to_le_bytes());
    for b in &mut wire[2..] {
        *b = 1;
    }
    let got = ggml_dequantize(GgmlType::Q8_0, &wire, 1, 32).expect("negative d decodes");
    assert_eq!(got.len(), 32);
    for v in &got {
        assert_eq!(v.to_bits(), (-0.5f32).to_bits(), "d * 1 with d = -0.5");
    }
    // The repacked side agrees with the source side on the same bytes.
    let t = repack(GgmlType::Q8_0, &wire, 1, 32).expect("negative d repacks");
    assert_eq!(repack_dequantize(&t).expect("repacked decodes"), got);
    // The Q5_K fixture carries a negative dmin by construction.
    let case = load_reference()
        .into_iter()
        .find(|c| c.name == "Q5_K")
        .expect("Q5_K fixture present");
    assert!(ggml_dequantize(GgmlType::Q5_K, &case.wire, case.n, case.k).is_ok());
}

#[test]
fn nonfinite_scales_are_rejected_on_both_paths() {
    for bits in [0x7C00u16, 0xFC00u16, 0x7E00u16] {
        let mut wire = vec![0u8; 34];
        wire[0..2].copy_from_slice(&bits.to_le_bytes());
        for b in &mut wire[2..] {
            *b = 3;
        }
        match ggml_dequantize(GgmlType::Q8_0, &wire, 1, 32) {
            Err(FormatError::InvalidScale { .. }) => {}
            other => panic!("bits {bits:#06x}: expected InvalidScale, got {other:?}"),
        }
        let mut rec_wire = vec![0u8; 18];
        rec_wire[0..2].copy_from_slice(&bits.to_le_bytes());
        match ggml_dequantize(GgmlType::Q4_0, &rec_wire, 1, 32) {
            Err(FormatError::InvalidScale { .. }) => {}
            other => panic!("bits {bits:#06x}: expected InvalidScale, got {other:?}"),
        }
    }
}

#[test]
fn malformed_inputs_collect_every_problem() {
    // Zero dims plus a misaligned K are all reported, never just the first.
    let wire = vec![0u8; 34];
    match ggml_dequantize(GgmlType::Q8_0, &wire, 0, 0) {
        Err(FormatError::Multiple { problems }) => assert!(problems.len() >= 2),
        other => panic!("expected collect-all, got {other:?}"),
    }
    // K must be a multiple of the wire block (Spec 2 §7 step 2).
    match ggml_dequantize(GgmlType::Q8_0, &wire, 1, 33) {
        Err(FormatError::InvalidBlock { .. }) => {}
        other => panic!("expected InvalidBlock, got {other:?}"),
    }
    // Truncated and overlong wires name their lengths.
    match ggml_dequantize(GgmlType::Q8_0, &wire[0..33], 1, 32) {
        Err(FormatError::LengthMismatch {
            expected: 34,
            got: 33,
            ..
        }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
    match repack(GgmlType::Q4_K, &[0u8; 100], 1, 256) {
        Err(FormatError::LengthMismatch {
            expected: 144,
            got: 100,
            ..
        }) => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
    // A repacked tensor whose regions disagree with its dims is rejected.
    let case = load_reference()
        .into_iter()
        .find(|c| c.name == "Q4_0")
        .expect("Q4_0 fixture present");
    let mut t = repack(GgmlType::Q4_0, &case.wire, case.n, case.k).expect("Q4_0 repacks");
    t.values.pop();
    assert!(matches!(
        repack_dequantize(&t),
        Err(FormatError::LengthMismatch { .. })
    ));
    assert!(matches!(
        unpack_repacked(&t),
        Err(FormatError::LengthMismatch { .. })
    ));
    // A repacked tensor whose scheme disagrees with its source is rejected.
    let mut t = repack(GgmlType::Q4_0, &case.wire, case.n, case.k).expect("Q4_0 repacks");
    t.scheme = Some(SchemeId::I8B32F);
    assert!(matches!(
        repack_dequantize(&t),
        Err(FormatError::SchemeMismatch { .. })
    ));
    // Size arithmetic never wraps: astronomical dims are an overflow
    // error, not an allocation attempt.
    assert!(matches!(
        ggml_dequantize(GgmlType::F16, &[], u32::MAX, u32::MAX),
        Err(FormatError::Overflow { .. })
    ));
}

#[test]
fn repack_metadata_is_exact() {
    assert_eq!(repack_record_bytes(SchemeId::I8B32F).expect("record"), 2);
    assert_eq!(repack_record_bytes(SchemeId::I4B32F).expect("record"), 2);
    assert_eq!(repack_record_bytes(SchemeId::I4B32FM).expect("record"), 4);
    assert_eq!(repack_record_bytes(SchemeId::I5B32F).expect("record"), 2);
    assert_eq!(repack_record_bytes(SchemeId::I5B32FM).expect("record"), 4);
    assert_eq!(repack_record_bytes(SchemeId::I4K).expect("record"), 16);
    assert_eq!(repack_record_bytes(SchemeId::I5K).expect("record"), 16);
    assert_eq!(repack_record_bytes(SchemeId::I6K).expect("record"), 18);
    assert_eq!(repack_record_bytes(SchemeId::I3K).expect("record"), 14);
    assert_eq!(repack_record_bytes(SchemeId::I2K).expect("record"), 20);
    assert_eq!(
        repack_outer_block(SchemeId::I8B32F).expect("outer"),
        Some(32)
    );
    assert_eq!(repack_outer_block(SchemeId::I4K).expect("outer"), Some(256));
    assert_eq!(repack_outer_block(SchemeId::I6K).expect("outer"), Some(256));
    assert_eq!(
        repack_packing(SchemeId::I8B32F).expect("packing"),
        r9v_format::Packing::Byte
    );
    assert_eq!(
        repack_packing(SchemeId::I4B32F).expect("packing"),
        r9v_format::Packing::Nibble4
    );
    assert_eq!(
        repack_packing(SchemeId::I5K).expect("packing"),
        r9v_format::Packing::bit_planes(5).expect("planes 5")
    );
    assert_eq!(
        repack_packing(SchemeId::I6K).expect("packing"),
        r9v_format::Packing::bit_planes(6).expect("planes 6")
    );
    assert_eq!(
        repack_packing(SchemeId::I3K).expect("packing"),
        r9v_format::Packing::bit_planes(3).expect("planes 3")
    );
    assert_eq!(
        repack_packing(SchemeId::I2K).expect("packing"),
        r9v_format::Packing::bit_planes(2).expect("planes 2")
    );
    // Exact wire-size ratios (Spec 2 §3.3, §8).
    assert_eq!(
        repack_bits_per_weight(SchemeId::I8B32F, 32).expect("bpw"),
        (272, 32)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I4B32F, 64).expect("bpw"),
        (288, 64)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I4B32FM, 32).expect("bpw"),
        (160, 32)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I5B32F, 32).expect("bpw"),
        (176, 32)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I5B32FM, 32).expect("bpw"),
        (192, 32)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I4K, 256).expect("bpw"),
        (1152, 256)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I5K, 256).expect("bpw"),
        (1408, 256)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I6K, 512).expect("bpw"),
        (3360, 512)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I3K, 256).expect("bpw"),
        (880, 256)
    );
    assert_eq!(
        repack_bits_per_weight(SchemeId::I2K, 256).expect("bpw"),
        (672, 256)
    );
    assert!(matches!(
        repack_bits_per_weight(SchemeId::I8B32F, 33),
        Err(FormatError::InvalidBlock { .. })
    ));
    assert!(matches!(
        repack_bits_per_weight(SchemeId::I6K, 128),
        Err(FormatError::InvalidBlock { .. })
    ));
    // Card-A2.4 ids are implemented (records, grouping, packing and
    // bpw all resolve); the dedicated behavior lives in tests/iq.rs.
    let implemented: &[(SchemeId, u32, u32, bool)] = &[
        (SchemeId::I4Nl, 2, 32, true),
        (SchemeId::I4Xs, 8, 256, true),
        (SchemeId::Iq3Xxs, 34, 256, false),
        (SchemeId::Iq3S, 46, 256, false),
        (SchemeId::Iq2Xxs, 2, 256, false),
        (SchemeId::Iq2Xs, 10, 256, false),
        (SchemeId::Iq2S, 50, 256, false),
        (SchemeId::Iq1S, 18, 256, false),
        (SchemeId::Iq1M, 24, 256, false),
    ];
    for (id, record, outer, nibble) in implemented {
        assert_eq!(
            repack_record_bytes(*id).expect("iq record known"),
            *record,
            "{id}"
        );
        assert_eq!(
            repack_outer_block(*id).expect("iq outer known"),
            Some(*outer),
            "{id}"
        );
        assert_eq!(
            repack_packing(*id).expect("iq packing known"),
            if *nibble {
                r9v_format::Packing::Nibble4
            } else {
                r9v_format::Packing::Byte
            },
            "{id}"
        );
        assert!(repack_bits_per_weight(*id, 256).is_ok(), "{id}");
    }
}

#[test]
fn bf16_codec_is_the_widening_shift() {
    assert_eq!(bf16_to_f32(0x3F80), 1.0);
    assert_eq!(bf16_to_f32(0xBF80), -1.0);
    assert_eq!(bf16_to_f32(0x0000), 0.0);
    assert_eq!(bf16_to_f32(0x8000), -0.0);
    // Every tested pattern equals the bit-shift widening exactly.
    for bits in [0x0001u16, 0x3F81, 0x7BFF, 0x7F80, 0xFF80, 0x7FC0, 0x4780] {
        assert_eq!(
            bf16_to_f32(bits).to_bits(),
            (bits as u32) << 16,
            "bits {bits:#06x}"
        );
    }
}

/// Expected SoA record for one wire `block`, per the documented layouts
/// (Spec 2 §3.3; SI-57 for the two special records). Every arm is
/// explicit: a future variant fails to compile here instead of
/// inheriting a slice.
fn expected_soa_record(ggml: GgmlType, block: &[u8]) -> Vec<u8> {
    match ggml {
        GgmlType::Q8_0 => block[0..2].to_vec(),
        GgmlType::Q4_0 => block[0..2].to_vec(),
        GgmlType::Q4_1 => block[0..4].to_vec(),
        GgmlType::Q5_0 => block[0..2].to_vec(),
        GgmlType::Q5_1 => block[0..4].to_vec(),
        GgmlType::Q4_K => block[0..16].to_vec(),
        GgmlType::Q5_K => block[0..16].to_vec(),
        GgmlType::Q3_K => block[96..110].to_vec(),
        // d-first reorder: wire d@208..210, scales@192..208 (SI-57).
        GgmlType::Q6_K => [block[208..210].to_vec(), block[192..208].to_vec()].concat(),
        // Gathered across the split wire layout (SI-57).
        GgmlType::Q2_K => [block[0..16].to_vec(), block[80..84].to_vec()].concat(),
        // Card-A2.4 records (SI-70): wire order minus the index payload.
        GgmlType::IQ4_NL => block[0..2].to_vec(),
        GgmlType::IQ4_XS => block[0..8].to_vec(),
        GgmlType::IQ3_XXS => [block[0..2].to_vec(), block[66..98].to_vec()].concat(),
        GgmlType::IQ3_S => [
            block[0..2].to_vec(),
            block[66..74].to_vec(),
            block[74..106].to_vec(),
            block[106..110].to_vec(),
        ]
        .concat(),
        GgmlType::IQ2_XXS => block[0..2].to_vec(),
        GgmlType::IQ2_XS => [block[0..2].to_vec(), block[66..74].to_vec()].concat(),
        GgmlType::IQ2_S => [block[0..2].to_vec(), block[34..82].to_vec()].concat(),
        GgmlType::IQ1_S => [block[0..2].to_vec(), block[34..50].to_vec()].concat(),
        GgmlType::IQ1_M => block[32..56].to_vec(),
        GgmlType::F16 => Vec::new(),
        GgmlType::BF16 => Vec::new(),
    }
}

#[test]
fn soa_scale_records_follow_the_documented_wire_layouts() {
    // Sixteen identical patterned blocks: row 0's SoA record must equal
    // the documented wire bytes, not a coincidental span. The pattern
    // keeps every offset distinct so a wrong slice cannot match.
    for ggml in [
        GgmlType::Q8_0,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q3_K,
        GgmlType::Q2_K,
    ] {
        let block_bytes = ggml.block_bytes() as usize;
        let block_len = ggml.block_len();
        let block: Vec<u8> = (0..block_bytes)
            .map(|i| ((i * 7 + 3) % 256) as u8)
            .collect();
        let wire: Vec<u8> = block.repeat(16);
        let t = repack(ggml, &wire, 16, block_len).expect("patterned wire repacks");
        let scheme = ggml.scheme().expect("quantized type has a scheme");
        let record = repack_record_bytes(scheme).expect("record size") as usize;
        assert_eq!(t.scales.len(), 16 * record, "{ggml} scale region");
        assert_eq!(
            t.scales[0..record],
            expected_soa_record(ggml, &block),
            "{ggml} row-0 SoA record",
        );
    }
    // The two pre-repair lies, pinned false: Q6_K's record is not the
    // wire prefix and Q2_K's is not the wire prefix either.
    let q6_block: Vec<u8> = (0..210).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    let q6_wire = q6_block.repeat(16);
    let q6 = repack(GgmlType::Q6_K, &q6_wire, 16, 256).expect("q6 repacks");
    assert_ne!(
        &q6.scales[0..18],
        &q6_wire[0..18],
        "Q6_K record is reordered"
    );
    let q2_block: Vec<u8> = (0..84).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    let q2_wire = q2_block.repeat(16);
    let q2 = repack(GgmlType::Q2_K, &q2_wire, 16, 256).expect("q2 repacks");
    assert_ne!(
        &q2.scales[0..20],
        &q2_wire[0..20],
        "Q2_K record is gathered"
    );
}

#[test]
fn a23_spec_issue_citations_resolve() {
    // B1 lock: every SI-NN cited by the A2.3 sources must name a
    // `## SI-NN` entry in SPEC-ISSUES.md. (This scans its own file too,
    // so the pre-repair dangling number is described, not quoted.)
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let issues = std::fs::read_to_string(format!("{root}/SPEC-ISSUES.md")).expect("issues file");
    let mut cited: Vec<String> = Vec::new();
    for file in [
        "src/ggml.rs",
        "src/repack.rs",
        "src/error.rs",
        "src/lib.rs",
        "src/iq.rs",
        "src/iq_lut.rs",
        "tests/gguf.rs",
        "tests/iq.rs",
        "tests/api_shape.rs",
    ] {
        let text =
            std::fs::read_to_string(format!("{root}/crates/r9v-format/{file}")).expect("src file");
        let mut i = 0;
        while let Some(pos) = text[i..].find("SI-") {
            let start = i + pos + 3;
            let end = text[start..]
                .find(|c: char| !c.is_ascii_digit())
                .map(|e| start + e)
                .unwrap_or(text.len());
            if end > start {
                let num = text[start..end].to_owned();
                if !cited.contains(&num) {
                    cited.push(num);
                }
            }
            i = end.max(start);
        }
    }
    assert!(!cited.is_empty(), "A2.3 sources cite SPEC-ISSUES entries");
    for num in &cited {
        assert!(
            issues.contains(&format!("## SI-{num} ")),
            "SI-{num} cited by A2.3 sources has no SPEC-ISSUES.md entry",
        );
    }
}
