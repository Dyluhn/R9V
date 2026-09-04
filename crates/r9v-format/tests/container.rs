// SPDX-License-Identifier: Apache-2.0
//! GGUF container tests (Spec 2 §6, §9; Spec 9 §3; card A2.5).
//!
//! Provenance:
//! - `a25_standard.gguf`, `a25_split-*.gguf`: written by
//!   `tests/fixtures/r9v-format/gen_container_fixtures.py` with
//!   gguf-py 0.19.0 (pinned, offline); seeded payload bytes cut to
//!   exact `GGML_QUANT_SIZES` lengths. Expectations below are the
//!   gguf-py reader's own values for those files.
//! - `llama_vocab_bert_bge.gguf`: genuine llama.cpp-produced vocab
//!   file (llama.cpp `models/`, sha256
//!   `fbcbe22278fb302694d5f4a41bfe48c5f90e8e3554eab1c0435387dff654a854`);
//!   0 tensors, 20 metadata fields.
//! - `llama_tiny_q80.hex`: genuine llama.cpp-produced quantized model
//!   with tensor table (via `llama-quantize` Q8_0); 4 tensors, 15 metadata fields.

use r9v_format::{
    accept_format_version, entry_regions, model_fp, parse_r9v_meta, r9v_tensor_type_id,
    EntryRegions, FormatError, GgufFile, GgufWriter, KvType, KvValue, Layout, R9vTensorType,
    SchemeId, ShardSet, TensorType,
};

/// Loads a hex-encoded fixture (`*.gguf.hex`; binaries are
/// git-ignored repo-wide, so fixtures commit as hex text per the
/// A2.3 `gguf_a23_reference.txt` precedent and decode here).
fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/r9v-format/{name}.hex",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture file present");
    let text = text.trim();
    let mut bytes = Vec::with_capacity(text.len() / 2);
    let raw = text.as_bytes();
    let mut i = 0;
    while i < raw.len() {
        let hi = (raw[i] as char).to_digit(16).expect("fixture hex valid");
        let lo = (raw[i + 1] as char)
            .to_digit(16)
            .expect("fixture hex valid");
        bytes.push((hi * 16 + lo) as u8);
        i += 2;
    }
    bytes
}

fn standard_bytes() -> Vec<u8> {
    fixture("a25_standard")
}

fn str_items(value: &KvValue) -> Vec<String> {
    match value {
        KvValue::Array { elem, items } => {
            assert_eq!(*elem, KvType::Str);
            items
                .iter()
                .map(|item| match item {
                    KvValue::Str(s) => s.clone(),
                    other => panic!("expected string item, got {:?}", other.kv_type()),
                })
                .collect()
        }
        other => panic!("expected array, got {:?}", other.kv_type()),
    }
}

#[test]
fn standard_fixture_metadata_covers_all_kv_types() {
    let bytes = standard_bytes();
    let file = GgufFile::parse(&bytes).expect("standard fixture parses");
    assert_eq!(file.version(), 3);
    assert_eq!(file.alignment(), 32);
    assert_eq!(file.kvs().len(), 28);
    assert_eq!(file.tensors().len(), 12);
    assert!(file.is_standard_gguf());

    assert_eq!(
        file.kv("general.architecture"),
        Some(&KvValue::Str("llama".to_owned()))
    );
    assert_eq!(
        file.kv("general.name"),
        Some(&KvValue::Str("a2.5-standard-fixture".to_owned()))
    );
    assert_eq!(file.kv("a25.u8"), Some(&KvValue::U8(200)));
    assert_eq!(file.kv("a25.i8"), Some(&KvValue::I8(-5)));
    assert_eq!(file.kv("a25.u16"), Some(&KvValue::U16(60000)));
    assert_eq!(file.kv("a25.i16"), Some(&KvValue::I16(-3000)));
    assert_eq!(file.kv("a25.u32"), Some(&KvValue::U32(3_000_000_000)));
    assert_eq!(file.kv("a25.i32"), Some(&KvValue::I32(-2_000_000_000)));
    assert_eq!(file.kv("a25.f32"), Some(&KvValue::F32(0.5)));
    assert_eq!(file.kv("a25.bool_true"), Some(&KvValue::Bool(true)));
    assert_eq!(file.kv("a25.bool_false"), Some(&KvValue::Bool(false)));
    assert_eq!(
        file.kv("a25.str"),
        Some(&KvValue::Str("r9v-a2.5 ☃".to_owned()))
    );
    assert_eq!(file.kv("a25.u64"), Some(&KvValue::U64((1 << 63) + 123)));
    assert_eq!(file.kv("a25.i64"), Some(&KvValue::I64(-(1 << 62))));
    // Exact f64 bits (gguf-py stores float64 with no conversion).
    match file.kv("a25.f64") {
        Some(KvValue::F64(v)) => assert_eq!(v.to_bits(), 0x4005_BF0A_8B14_5769),
        other => panic!("a25.f64: {other:?}"),
    }
    assert_eq!(
        file.kv("tokenizer.ggml.tokens").map(str_items),
        Some(vec![
            "<unk>".to_owned(),
            "<s>".to_owned(),
            "hello".to_owned()
        ])
    );
    // Homogeneous numeric arrays keep their element types
    // (gguf-py writes Python int arrays as INT32; only `bytes`
    // writes UINT8).
    match file.kv("a25.arr_i32_small") {
        Some(KvValue::Array { elem, items }) => {
            assert_eq!(*elem, KvType::I32);
            assert_eq!(
                items,
                &vec![KvValue::I32(1), KvValue::I32(2), KvValue::I32(3)]
            );
        }
        other => panic!("a25.arr_i32_small: {other:?}"),
    }
    match file.kv("a25.arr_bytes") {
        Some(KvValue::Array { elem, items }) => {
            assert_eq!(*elem, KvType::U8);
            assert_eq!(items, &vec![KvValue::U8(1), KvValue::U8(2), KvValue::U8(3)]);
        }
        other => panic!("a25.arr_bytes: {other:?}"),
    }
    match file.kv("a25.arr_u64") {
        Some(KvValue::Array { elem, items }) => {
            assert_eq!(*elem, KvType::U64);
            assert_eq!(items, &vec![KvValue::U64(1 << 40), KvValue::U64(1 << 63)]);
        }
        other => panic!("a25.arr_u64: {other:?}"),
    }
    match file.kv("a25.arr_f64") {
        Some(KvValue::Array { elem, items }) => {
            assert_eq!(*elem, KvType::F64);
            assert_eq!(items.len(), 1);
        }
        other => panic!("a25.arr_f64: {other:?}"),
    }
    assert_eq!(file.kv("missing.key"), None);
    assert_eq!(parse_r9v_meta(&file).expect("meta parse"), None);
}

#[test]
fn standard_fixture_tensor_table_matches_writer_layout() {
    let bytes = standard_bytes();
    let file = GgufFile::parse(&bytes).expect("standard fixture parses");
    // (name, file-order dims, type code, data bytes); dims are
    // innermost-first as gguf-py writes them.
    let expected: &[(&str, &[u64], u32, u64)] = &[
        ("bias_f32", &[16], 0, 64),
        ("norm_f16", &[32, 16], 1, 1024),
        ("w_q80", &[32, 16], 8, 544),
        ("w_q40", &[32, 16], 2, 288),
        ("w_q41", &[32, 16], 3, 320),
        ("w_q50", &[32, 16], 6, 352),
        ("w_q51", &[32, 16], 7, 384),
        ("w_q2k", &[256, 16], 10, 1344),
        ("w_q3k", &[256, 16], 11, 1760),
        ("w_q4k", &[256, 16], 12, 2304),
        ("w_q5k", &[256, 16], 13, 2816),
        ("w_q6k", &[256, 16], 14, 3360),
    ];
    assert_eq!(file.tensors().len(), expected.len());
    // gguf-py advances each offset by the 32-padded size.
    let mut offset = 0u64;
    for (info, (name, dims, code, nbytes)) in file.tensors().iter().zip(expected.iter()) {
        assert_eq!(&info.name, name);
        assert_eq!(&info.dims, dims);
        assert_eq!(info.dtype.code(), *code);
        assert_eq!(info.offset, offset);
        let data = file.tensor_bytes(name, &bytes).expect("tensor in range");
        assert_eq!(data.len() as u64, *nbytes);
        offset += nbytes.div_ceil(32) * 32;
    }
    // Data section starts at the 32-aligned end of the table.
    assert_eq!(file.data_start() % 32, 0);
    assert!(file.data_start() > file.ti_range().1);
    // Scheme mapping agrees with the repack set for quant types.
    assert_eq!(
        file.tensor("w_q4k").expect("w_q4k").dtype.scheme(),
        Some(SchemeId::I4K)
    );
    assert_eq!(file.tensor("norm_f16").expect("f16").dtype.scheme(), None);
    assert_eq!(
        file.tensor("w_q4k").expect("w_q4k").dtype.ggml(),
        Some(r9v_format::GgmlType::Q4_K)
    );
}

#[test]
fn reads_real_llamacpp_vocab_metadata_and_empty_table() {
    let bytes = fixture("llama_vocab_bert_bge");
    let file = GgufFile::parse(&bytes).expect("llama.cpp vocab file parses");
    assert_eq!(file.version(), 3);
    assert_eq!(file.tensors().len(), 0);
    // 23 fields in gguf-py terms include its 3 pseudo-fields
    // (GGUF.version, tensor_count, kv_count); the file holds 20 KVs.
    assert_eq!(file.kvs().len(), 20);
    assert!(file.is_standard_gguf());
    assert_eq!(
        file.kv("general.architecture"),
        Some(&KvValue::Str("bert".to_owned()))
    );
    assert_eq!(
        file.kv("general.name"),
        Some(&KvValue::Str("bert-bge".to_owned()))
    );
    assert_eq!(file.kv("bert.block_count"), Some(&KvValue::U32(12)));
    assert_eq!(file.kv("bert.context_length"), Some(&KvValue::U32(512)));
    assert_eq!(
        file.kv("bert.attention.causal"),
        Some(&KvValue::Bool(false))
    );
    assert_eq!(
        file.kv("tokenizer.ggml.unknown_token_id"),
        Some(&KvValue::U32(100))
    );
    assert_eq!(
        file.kv("tokenizer.ggml.cls_token_id"),
        Some(&KvValue::U32(101))
    );
    match file.kv("tokenizer.ggml.tokens") {
        Some(KvValue::Array { elem, items }) => {
            assert_eq!(*elem, KvType::Str);
            assert_eq!(items.len(), 30522);
            assert_eq!(items[0], KvValue::Str("[PAD]".to_owned()));
            assert_eq!(items[100], KvValue::Str("[UNK]".to_owned()));
            assert_eq!(items[101], KvValue::Str("[CLS]".to_owned()));
        }
        other => panic!("tokenizer.ggml.tokens: {other:?}"),
    }
    match file.kv("tokenizer.ggml.token_type") {
        Some(KvValue::Array { elem, items }) => {
            assert_eq!(*elem, KvType::I32);
            assert_eq!(items.len(), 30522);
            assert_eq!(items[0], KvValue::I32(3));
        }
        other => panic!("tokenizer.ggml.token_type: {other:?}"),
    }
}

#[test]
fn reads_real_llamacpp_metadata_and_tensor_table() {
    let bytes = fixture("llama_tiny_q80");
    let file = GgufFile::parse(&bytes).expect("llama.cpp model with tensors parses");
    assert_eq!(file.version(), 3);
    assert_eq!(file.tensors().len(), 4);
    assert!(file.is_standard_gguf());
    assert_eq!(
        file.kv("general.architecture"),
        Some(&KvValue::Str("llama".to_owned()))
    );
    assert_eq!(
        file.kv("general.name"),
        Some(&KvValue::Str("tiny-llama".to_owned()))
    );
    assert_eq!(file.kv("llama.context_length"), Some(&KvValue::U32(32)));
    assert_eq!(file.kv("llama.embedding_length"), Some(&KvValue::U32(32)));
    assert_eq!(file.kv("llama.block_count"), Some(&KvValue::U32(1)));

    assert_eq!(file.data_start(), 960);

    // Verify all 4 tensors produced by llama-quantize
    let out = file.tensor("output.weight").expect("output.weight");
    assert_eq!(out.dtype, TensorType::Q8_0);
    assert_eq!(out.dims, vec![32, 3]);
    assert_eq!(out.offset, 0);
    let out_bytes = file
        .tensor_bytes("output.weight", &bytes)
        .expect("output bytes");
    assert_eq!(out_bytes.len(), 102);
    let out_hash = file
        .entry_xxh3("output.weight", &bytes)
        .expect("output xxh3");
    assert_eq!(out_hash, r9v_common::xxh3_64(out_bytes));

    let embd = file.tensor("token_embd.weight").expect("token_embd.weight");
    assert_eq!(embd.dtype, TensorType::Q8_0);
    assert_eq!(embd.dims, vec![32, 3]);
    assert_eq!(embd.offset, 128);
    let embd_bytes = file
        .tensor_bytes("token_embd.weight", &bytes)
        .expect("embd bytes");
    assert_eq!(embd_bytes.len(), 102);
    let embd_hash = file
        .entry_xxh3("token_embd.weight", &bytes)
        .expect("embd xxh3");
    assert_eq!(embd_hash, r9v_common::xxh3_64(embd_bytes));

    let norm = file
        .tensor("blk.0.attn_norm.weight")
        .expect("attn_norm.weight");
    assert_eq!(norm.dtype, TensorType::F32);
    assert_eq!(norm.dims, vec![32]);
    assert_eq!(norm.offset, 256);
    let norm_bytes = file
        .tensor_bytes("blk.0.attn_norm.weight", &bytes)
        .expect("norm bytes");
    assert_eq!(norm_bytes.len(), 128);
    let norm_hash = file
        .entry_xxh3("blk.0.attn_norm.weight", &bytes)
        .expect("norm xxh3");
    assert_eq!(norm_hash, r9v_common::xxh3_64(norm_bytes));

    let q = file.tensor("blk.0.attn_q.weight").expect("attn_q.weight");
    assert_eq!(q.dtype, TensorType::Q8_0);
    assert_eq!(q.dims, vec![32, 32]);
    assert_eq!(q.offset, 384);
    let q_bytes = file
        .tensor_bytes("blk.0.attn_q.weight", &bytes)
        .expect("q bytes");
    assert_eq!(q_bytes.len(), 1088);
    let q_hash = file
        .entry_xxh3("blk.0.attn_q.weight", &bytes)
        .expect("q xxh3");
    assert_eq!(q_hash, r9v_common::xxh3_64(q_bytes));
}

fn all_kv_values() -> Vec<(&'static str, KvValue)> {
    vec![
        ("k.u8", KvValue::U8(7)),
        ("k.i8", KvValue::I8(-7)),
        ("k.u16", KvValue::U16(1000)),
        ("k.i16", KvValue::I16(-1000)),
        ("k.u32", KvValue::U32(99)),
        ("k.i32", KvValue::I32(-99)),
        ("k.f32", KvValue::F32(1.25)),
        ("k.bool", KvValue::Bool(true)),
        ("k.str", KvValue::Str("héllo� замороженный".to_owned())),
        (
            "k.arr",
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("".to_owned()), KvValue::Str("x".to_owned())],
            },
        ),
        ("k.u64", KvValue::U64(u64::MAX)),
        ("k.i64", KvValue::I64(i64::MIN)),
        ("k.f64", KvValue::F64(-0.0)),
    ]
}

#[test]
fn writer_round_trip_preserves_every_kv_type() {
    let mut writer = GgufWriter::new();
    for (key, value) in all_kv_values() {
        writer.add_kv(key, value).expect("kv appends");
    }
    writer
        .add_tensor("w", &[4, 32], TensorType::Q8_0, vec![9u8; 4 * 34])
        .expect("tensor appends");
    let bytes = writer.emit().expect("emit succeeds");
    assert_eq!(&bytes[0..4], b"GGUF");
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("header holds version"));
    assert_eq!(version, 3);
    let file = GgufFile::parse(&bytes).expect("own output parses");
    assert_eq!(file.tensors().len(), 1);
    let info = &file.tensors()[0];
    assert_eq!(info.name, "w");
    assert_eq!(info.dims, vec![32, 4]);
    assert_eq!(info.shape(), vec![4, 32]);
    for (key, value) in all_kv_values() {
        assert_eq!(file.kv(key), Some(&value), "key {key}");
    }
    // -0.0 survives as bits, not as float equality.
    match file.kv("k.f64") {
        Some(KvValue::F64(v)) => assert_eq!(v.to_bits(), (-0.0f64).to_bits()),
        other => panic!("k.f64: {other:?}"),
    }
    assert!(file.is_standard_gguf());
}

#[test]
fn writer_rejects_duplicates_and_bad_lengths() {
    let mut writer = GgufWriter::new();
    writer.add_kv("k", KvValue::U8(1)).expect("first kv");
    assert!(matches!(
        writer.add_kv("k", KvValue::U8(2)),
        Err(FormatError::DuplicateKey { .. })
    ));
    writer
        .add_tensor("w", &[2, 32], TensorType::Q4_0, vec![0u8; 2 * 18])
        .expect("first tensor");
    assert!(matches!(
        writer.add_tensor("w", &[2, 32], TensorType::Q4_0, vec![0u8; 2 * 18]),
        Err(FormatError::DuplicateTensor { .. })
    ));
    assert!(matches!(
        writer.add_tensor("short", &[2, 32], TensorType::Q4_0, vec![0u8; 10]),
        Err(FormatError::LengthMismatch { .. })
    ));
    assert!(matches!(
        GgufWriter::new().with_alignment(48),
        Err(FormatError::InvalidAlignment { .. })
    ));
    assert!(matches!(
        GgufWriter::new().with_alignment(0),
        Err(FormatError::InvalidAlignment { .. })
    ));
}

fn native_test_bytes() -> (Vec<u8>, Vec<u64>) {
    // One I4_K tile-row tensor plus one f16 vector, native layout.
    let regions = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).expect("regions");
    assert_eq!(regions.offsets(), [0, 2048, 4096]);
    assert_eq!(regions.entry_bytes, 4096);
    let mut entry = vec![0u8; regions.entry_bytes as usize];
    for (i, b) in entry.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let entry_hash = r9v_common::xxh3_64(&entry);
    let mut writer = GgufWriter::new().with_alignment(4096).expect("alignment");
    writer
        .add_kv("general.architecture", KvValue::Str("llama".to_owned()))
        .expect("kv");
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .expect("kv");
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .expect("kv");
    writer
        .add_kv("r9v.layout_id", KvValue::Str("L1".to_owned()))
        .expect("kv");
    writer
        .add_kv("r9v.arch_hint", KvValue::Str("gfx1201".to_owned()))
        .expect("kv");
    writer
        .add_kv(
            "r9v.quant_tool.version",
            KvValue::Str("r9v-quant 0.1.0".to_owned()),
        )
        .expect("kv");
    writer
        .add_kv("r9v.quant_tool.seed", KvValue::U64(42))
        .expect("kv");
    writer
        .add_kv("r9v.quant_tool.preset", KvValue::Str("balanced".to_owned()))
        .expect("kv");
    writer
        .add_kv("r9v.calibration.name", KvValue::Str("calib-a".to_owned()))
        .expect("kv");
    writer
        .add_kv("r9v.calibration.hash", KvValue::Str("abc123".to_owned()))
        .expect("kv");
    writer
        .add_kv("r9v.calibration.tokens", KvValue::U64(1_000_000))
        .expect("kv");
    writer
        .add_kv("r9v.smoothing.folded", KvValue::Bool(true))
        .expect("kv");
    writer
        .add_kv("r9v.smoothing.alpha", KvValue::F32(0.5))
        .expect("kv");
    writer
        .add_kv("r9v.quality.top1", KvValue::F32(0.9))
        .expect("kv");
    writer
        .add_kv("r9v.quality.ppl", KvValue::F32(7.25))
        .expect("kv");
    writer
        .add_kv(
            "r9v.quality.holdout_hash",
            KvValue::Str("holdout-9".to_owned()),
        )
        .expect("kv");
    let name = "blk.0.attn_q.weight";
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.scheme"),
            KvValue::Str("i4_k".to_owned()),
        )
        .expect("kv");
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.act"),
            KvValue::Str("i8/PerToken".to_owned()),
        )
        .expect("kv");
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("matmul".to_owned())],
            },
        )
        .expect("kv");
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.interleave"),
            KvValue::Str("none".to_owned()),
        )
        .expect("kv");
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.sparse"),
            KvValue::Str("none".to_owned()),
        )
        .expect("kv");
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.placement_hint"),
            KvValue::Str("device".to_owned()),
        )
        .expect("kv");
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.residency_unit"),
            KvValue::Str("tensor".to_owned()),
        )
        .expect("kv");
    writer
        .add_kv(
            &format!("r9v.tensor.{name}.regions"),
            KvValue::Array {
                elem: KvType::U64,
                items: regions.offsets().iter().map(|o| KvValue::U64(*o)).collect(),
            },
        )
        .expect("kv");
    writer
        .add_kv(&format!("r9v.tensor.{name}.xxh3"), KvValue::U64(entry_hash))
        .expect("kv");
    writer
        .add_kv(&format!("r9v.tensor.{name}.eps_int4"), KvValue::F32(0.01))
        .expect("kv");
    // Forward-compat probe: unknown keys must not break the parse.
    writer
        .add_kv(
            "r9v.tensor.blk.0.attn_q.weight.future_flag",
            KvValue::Bool(true),
        )
        .expect("kv");
    writer
        .add_kv("r9v.future_global", KvValue::U32(7))
        .expect("kv");
    writer
        .add_tensor(
            name,
            &[16, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            entry,
        )
        .expect("tensor");
    let bytes = writer.emit().expect("emit");
    (bytes, vec![entry_hash])
}

#[test]
fn native_file_round_trip_with_full_r9v_keys() {
    let (bytes, hashes) = native_test_bytes();
    let file = GgufFile::parse(&bytes).expect("native file parses");
    assert_eq!(file.alignment(), 4096);
    assert_eq!(file.data_start() % 4096, 0);
    assert!(!file.is_standard_gguf());
    let info = file.tensor("blk.0.attn_q.weight").expect("tensor row");
    assert_eq!(
        info.dtype,
        TensorType::R9v(R9vTensorType::new(SchemeId::I4K))
    );
    assert_eq!(info.dtype.code(), 1003);
    assert_eq!(info.dtype.scheme(), Some(SchemeId::I4K));

    let meta = parse_r9v_meta(&file)
        .expect("meta parses")
        .expect("r9v present");
    assert_eq!(meta.format_version, 1);
    assert_eq!(meta.layout_id, Layout::L1);
    assert_eq!(meta.arch_hint.as_deref(), Some("gfx1201"));
    assert_eq!(meta.tool_seed, Some(42));
    assert_eq!(meta.calibration.tokens, Some(1_000_000));
    assert_eq!(meta.smoothing.folded, Some(true));
    assert_eq!(meta.quality.top1, Some(0.9));
    let tensor = meta.tensor("blk.0.attn_q.weight").expect("tensor meta");
    assert_eq!(tensor.scheme, Some(SchemeId::I4K));
    let act = tensor.act.expect("act");
    assert_eq!(act.dtype, r9v_format::ActDtype::I8);
    assert_eq!(act.scheme, r9v_format::ActScheme::PerToken);
    assert_eq!(tensor.roles, vec![r9v_format::Role::Matmul]);
    assert_eq!(tensor.regions, Some([0, 2048, 4096]));
    assert_eq!(tensor.xxh3, Some(hashes[0]));
    assert_eq!(tensor.eps_int4, Some(0.01));

    // Per-entry xxh3 in metadata matches the bytes on disk.
    let entry_bytes = file
        .tensor_bytes("blk.0.attn_q.weight", &bytes)
        .expect("tensor bytes");
    assert_eq!(entry_bytes.len(), 4096);
    let computed_hash = file
        .entry_xxh3("blk.0.attn_q.weight", &bytes)
        .expect("entry xxh3");
    assert_eq!(computed_hash, hashes[0]);
    assert_eq!(Some(computed_hash), tensor.xxh3);

    let fp = file.file_fp(&bytes, 1);
    let model = model_fp(fp, &hashes);
    assert_eq!(model, model_fp(file.file_fp(&bytes, 1), &hashes));
    assert_ne!(model, fp);
}

#[test]
fn writes_native_file_parsed_by_gguf_py_at_header_and_metadata_level() {
    let (bytes, _) = native_test_bytes();
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join(format!("r9v_test_native_{}.gguf", std::process::id()));
    std::fs::write(&file_path, &bytes).expect("write temp gguf");

    let py_script = r#"
import sys, gguf

class NativeHeaderReader(gguf.GGUFReader):
    def _build_tensors(self, start_offs: int, fields: list) -> None:
        # Card A2.5: gguf-py parses native files at header and metadata level;
        # R9V scheme type IDs (1000-1099) are outside gguf-py's GGML enum.
        self.raw_tensor_fields = fields

path = sys.argv[1]
reader = NativeHeaderReader(path)
arch = reader.fields['general.architecture'].parts[-1].tobytes().decode()
assert arch == 'llama', f"unexpected arch: {arch}"
align = reader.fields['general.alignment'].parts[-1][0]
assert align == 4096, f"unexpected alignment: {align}"
fmt_v = reader.fields['r9v.format_version'].parts[-1][0]
assert fmt_v == 1, f"unexpected format_version: {fmt_v}"
layout = reader.fields['r9v.layout_id'].parts[-1].tobytes().decode()
assert layout == 'L1', f"unexpected layout: {layout}"
scheme = reader.fields['r9v.tensor.blk.0.attn_q.weight.scheme'].parts[-1].tobytes().decode()
assert scheme == 'i4_k', f"unexpected tensor scheme: {scheme}"
assert len(reader.raw_tensor_fields) == 1, f"expected 1 tensor field, got {len(reader.raw_tensor_fields)}"
tf = reader.raw_tensor_fields[0]
name = str(bytes(tf.parts[1]), encoding='utf-8')
assert name == 'blk.0.attn_q.weight', f"unexpected tensor name: {name}"
dims = list(tf.parts[3])
assert dims == [256, 16], f"unexpected tensor dims: {dims}"
dtype = tf.parts[4][0]
assert dtype == 1003, f"unexpected tensor dtype: {dtype}"
"#;

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(py_script)
        .arg(&file_path)
        .output();

    let _ = std::fs::remove_file(&file_path);

    let output = output.expect("python3 execution failed");
    if !output.status.success() {
        panic!(
            "gguf-py reader validation failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn metadata_only_operations_do_not_read_tensor_payloads() {
    let (orig_bytes, _) = native_test_bytes();
    let orig_file = GgufFile::parse(&orig_bytes).expect("orig file parses");
    let data_start = orig_file.data_start();
    assert!(data_start < orig_bytes.len() as u64);

    // Poison all tensor payload bytes in the data section
    let mut poisoned = orig_bytes.clone();
    for b in &mut poisoned[data_start as usize..] {
        *b ^= 0xAA;
    }

    // Metadata parsing must succeed identically on the poisoned buffer
    let poisoned_file = GgufFile::parse(&poisoned).expect("poisoned parse succeeds");
    assert_eq!(poisoned_file.version(), orig_file.version());
    assert_eq!(poisoned_file.alignment(), orig_file.alignment());
    assert_eq!(poisoned_file.data_start(), orig_file.data_start());
    assert_eq!(poisoned_file.kvs(), orig_file.kvs());
    assert_eq!(poisoned_file.tensors(), orig_file.tensors());
    assert_eq!(poisoned_file.header_range(), orig_file.header_range());
    assert_eq!(poisoned_file.kv_range(), orig_file.kv_range());
    assert_eq!(poisoned_file.ti_range(), orig_file.ti_range());

    // File fingerprint (Spec 9 §3) covers header, table, metadata, size, shards;
    // it never reads tensor payloads, so it must be identical.
    assert_eq!(
        poisoned_file.file_fp(&poisoned, 1),
        orig_file.file_fp(&orig_bytes, 1)
    );

    // R9V metadata accessors never read tensor payloads
    assert_eq!(
        parse_r9v_meta(&poisoned_file).unwrap(),
        parse_r9v_meta(&orig_file).unwrap()
    );

    // Conversely, payload-reading operations MUST observe the poisoned bytes
    let orig_tensor_bytes = orig_file
        .tensor_bytes("blk.0.attn_q.weight", &orig_bytes)
        .expect("orig tensor bytes");
    let poisoned_tensor_bytes = poisoned_file
        .tensor_bytes("blk.0.attn_q.weight", &poisoned)
        .expect("poisoned tensor bytes");
    assert_ne!(orig_tensor_bytes, poisoned_tensor_bytes);
    assert_ne!(
        orig_file
            .entry_xxh3("blk.0.attn_q.weight", &orig_bytes)
            .unwrap(),
        poisoned_file
            .entry_xxh3("blk.0.attn_q.weight", &poisoned)
            .unwrap()
    );
}

#[test]
fn r9v_type_ids_span_schemes_without_colliding() {
    for scheme in SchemeId::ALL {
        let id = r9v_tensor_type_id(scheme);
        assert!((1001..=1099).contains(&id), "id {id} in range");
        let back = R9vTensorType::from_code(id).expect("round trips");
        assert_eq!(back.scheme(), scheme);
        assert_eq!(TensorType::from_code(id), TensorType::R9v(back));
    }
    assert_eq!(R9vTensorType::from_code(1000), None);
    assert_eq!(R9vTensorType::from_code(1023), None);
    assert_eq!(R9vTensorType::from_code(8), None);
    assert!(matches!(TensorType::from_code(4), TensorType::Unknown(4)));
}

#[test]
fn tensor_type_codes_cover_the_ggufpy_table() {
    // (code, block_len, block_bytes) transcribed from gguf-py 0.19.0
    // GGML_QUANT_SIZES.
    let expected: &[(u32, u32, u64)] = &[
        (0, 1, 4),
        (1, 1, 2),
        (2, 32, 18),
        (3, 32, 20),
        (6, 32, 22),
        (7, 32, 24),
        (8, 32, 34),
        (9, 32, 40),
        (10, 256, 84),
        (11, 256, 110),
        (12, 256, 144),
        (13, 256, 176),
        (14, 256, 210),
        (15, 256, 292),
        (16, 256, 66),
        (17, 256, 74),
        (18, 256, 98),
        (19, 256, 50),
        (20, 32, 18),
        (21, 256, 110),
        (22, 256, 82),
        (23, 256, 136),
        (24, 1, 1),
        (25, 1, 2),
        (26, 1, 4),
        (27, 1, 8),
        (28, 1, 8),
        (29, 256, 56),
        (30, 1, 2),
        (34, 256, 54),
        (35, 256, 66),
        (39, 32, 17),
        (40, 64, 36),
        (41, 128, 18),
    ];
    assert_eq!(TensorType::ALL.len(), expected.len());
    for (code, block, size) in expected {
        let ty = TensorType::from_code(*code);
        assert_eq!(ty.code(), *code, "code {code}");
        assert_eq!(ty.quant_size(), Some((*block, *size)), "code {code}");
    }
}

#[test]
fn file_fp_and_model_fp_are_stable_and_sensitive() {
    let bytes = standard_bytes();
    let file = GgufFile::parse(&bytes).expect("parses");
    let fp1 = file.file_fp(&bytes, 1);
    let file2 = GgufFile::parse(&bytes).expect("parses again");
    assert_eq!(fp1, file2.file_fp(&bytes, 1));
    // Shard count feeds the fingerprint.
    assert_ne!(fp1, file.file_fp(&bytes, 2));
    // One flipped key byte (kept ASCII so the file still parses)
    // changes file_fp.
    let mut changed = bytes.clone();
    let kv_off = file.kv_range().0 as usize;
    changed[kv_off + 20] ^= 0x01;
    let file_changed = GgufFile::parse(&changed).expect("still parses");
    assert_ne!(fp1, file_changed.file_fp(&changed, 1));

    let hashes: Vec<u64> = file
        .tensors()
        .iter()
        .map(|t| file.entry_xxh3(&t.name, &bytes).expect("entry hash"))
        .collect();
    for (t, h) in file.tensors().iter().zip(hashes.iter()) {
        let direct = r9v_common::xxh3_64(file.tensor_bytes(&t.name, &bytes).expect("bytes"));
        assert_eq!(*h, direct);
    }
    let model = model_fp(fp1, &hashes);
    assert_eq!(model, model_fp(fp1, &hashes));
    // One flipped weight byte changes model_fp but not file_fp.
    let mut wchanged = bytes.clone();
    let w = file.tensor_bytes("w_q80", &bytes).expect("bytes");
    let start = w.as_ptr() as usize - bytes.as_ptr() as usize;
    wchanged[start] ^= 0x01;
    let wfile = GgufFile::parse(&wchanged).expect("parses");
    assert_eq!(fp1, wfile.file_fp(&wchanged, 1));
    let whashes: Vec<u64> = wfile
        .tensors()
        .iter()
        .map(|t| wfile.entry_xxh3(&t.name, &wchanged).expect("entry hash"))
        .collect();
    assert_ne!(model, model_fp(fp1, &whashes));
}

#[test]
fn format_version_acceptance_rule() {
    assert!(accept_format_version(None).is_ok());
    assert!(accept_format_version(Some(0)).is_ok());
    assert!(accept_format_version(Some(1)).is_ok());
    match accept_format_version(Some(2)) {
        Err(FormatError::FormatVersion { found, max }) => {
            assert_eq!(found, 2);
            assert_eq!(max, 1);
        }
        other => panic!("expected FormatVersion, got {other:?}"),
    }
    // The typed parser enforces the same rule on real keys.
    let mut writer = GgufWriter::new();
    writer
        .add_kv("r9v.format_version", KvValue::U32(9))
        .expect("kv");
    writer
        .add_kv("r9v.layout_id", KvValue::Str("L1".to_owned()))
        .expect("kv");
    let bytes = writer.emit().expect("emit");
    let file = GgufFile::parse(&bytes).expect("parses");
    assert!(matches!(
        parse_r9v_meta(&file),
        Err(FormatError::FormatVersion { .. })
    ));
}

#[test]
fn split_shards_merge_in_table_order() {
    let a = fixture("a25_split-00001-of-00002");
    let b = fixture("a25_split-00002-of-00002");
    let fa = GgufFile::parse(&a).expect("shard 0 parses");
    let fb = GgufFile::parse(&b).expect("shard 1 parses");
    assert_eq!(fa.kv("split.no"), Some(&KvValue::U16(0)));
    assert_eq!(fb.kv("split.no"), Some(&KvValue::U16(1)));
    assert_eq!(fa.kv("split.count"), Some(&KvValue::U16(2)));
    assert_eq!(fa.kv("split.tensors.count"), Some(&KvValue::I32(3)));
    let set = ShardSet::open(vec![fa, fb]).expect("shards merge");
    assert_eq!(set.len(), 3);
    assert!(!set.is_empty());
    let names: Vec<String> = (0..set.len())
        .map(|i| set.tensor_at(i).expect("merged row").1.name.clone())
        .collect();
    assert_eq!(names, vec!["shard.a_q80", "shard.b_f16", "shard.c_q40"]);
    assert_eq!(set.tensor("shard.c_q40").expect("find").0, 1);
    assert_eq!(set.tensor("absent"), None);
    assert_eq!(set.tensor_at(99), None);
}

#[test]
fn split_mismatch_reports_every_problem() {
    let a = fixture("a25_split-00001-of-00002");
    let b = fixture("a25_split-00002-of-00002");
    let fa = GgufFile::parse(&a).expect("shard 0 parses");
    // Same shard twice: duplicate tensors plus a split.no mismatch.
    match ShardSet::open(vec![fa.clone(), fa]) {
        Err(FormatError::Multiple { problems }) => {
            assert!(problems.len() >= 3, "got {problems:?}");
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
    assert!(ShardSet::open(vec![]).is_err());
    let fb = GgufFile::parse(&b).expect("shard 1 parses");
    assert_eq!(fb.tensors().len(), 1);
    assert_eq!(fb.tensor("shard.c_q40").expect("row").dtype.code(), 2);
}

/// Minimal hand encoder for corrupt-input tests (bypasses the
/// writer's own validity checks).
struct Raw {
    bytes: Vec<u8>,
}

impl Raw {
    fn new(n_tensors: u64, n_kv: u64) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&n_tensors.to_le_bytes());
        bytes.extend_from_slice(&n_kv.to_le_bytes());
        Self { bytes }
    }

    fn str_raw(&mut self, s: &[u8]) {
        self.bytes
            .extend_from_slice(&(s.len() as u64).to_le_bytes());
        self.bytes.extend_from_slice(s);
    }

    fn kv(&mut self, key: &str, ty: u32, value: &[u8]) {
        self.str_raw(key.as_bytes());
        self.bytes.extend_from_slice(&ty.to_le_bytes());
        self.bytes.extend_from_slice(value);
    }

    fn tensor(&mut self, name: &str, dims: &[u64], ty: u32, offset: u64) {
        self.str_raw(name.as_bytes());
        self.bytes
            .extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.bytes.extend_from_slice(&d.to_le_bytes());
        }
        self.bytes.extend_from_slice(&ty.to_le_bytes());
        self.bytes.extend_from_slice(&offset.to_le_bytes());
    }
}

#[test]
fn corrupt_headers_fail_with_offsets() {
    assert!(matches!(
        GgufFile::parse(b""),
        Err(FormatError::Truncated { .. })
    ));
    assert!(matches!(
        GgufFile::parse(b"GG"),
        Err(FormatError::Truncated { .. })
    ));
    let mut bad = vec![0u8; 32];
    bad[0..4].copy_from_slice(b"XXXX");
    assert!(matches!(
        GgufFile::parse(&bad),
        Err(FormatError::BadMagic { .. })
    ));
    let mut version = Raw::new(0, 0);
    version.bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
    match GgufFile::parse(&version.bytes) {
        Err(FormatError::UnsupportedVersion { found, .. }) => assert_eq!(found, 99),
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }
    // Truncation at every 7th byte is always a Truncated error, never
    // a panic.
    let bytes = standard_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match GgufFile::parse(&bytes[..i]) {
            Err(FormatError::Truncated { .. })
            | Err(FormatError::BadMagic { .. })
            | Err(FormatError::UnsupportedVersion { .. })
            | Err(FormatError::Malformed { .. })
            | Err(FormatError::BadTensorRange { .. })
            | Err(FormatError::UnknownTensorType { .. })
            | Err(FormatError::Multiple { .. }) => {}
            Ok(_) => panic!("prefix of length {i} unexpectedly parses"),
            Err(other) => panic!("prefix {i}: unexpected error {other:?}"),
        }
        i += 7;
    }
    // BOOL outside 0/1 and nested arrays are malformed.
    let mut raw = Raw::new(0, 1);
    raw.kv("flag", 7, &[0x02]);
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::Malformed { .. })
    ));
    let mut raw = Raw::new(0, 1);
    raw.str_raw(b"arr");
    raw.bytes.extend_from_slice(&9u32.to_le_bytes());
    raw.bytes.extend_from_slice(&9u32.to_le_bytes());
    raw.bytes.extend_from_slice(&1u64.to_le_bytes());
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::Malformed { .. })
    ));
    // Invalid UTF-8 in a key is malformed, not a panic.
    let mut raw = Raw::new(0, 1);
    raw.kv("X", 4, &1u32.to_le_bytes());
    raw.bytes[32] = 0xFF;
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::Malformed { .. })
    ));
    // Unknown KV type code is malformed.
    let mut raw = Raw::new(0, 1);
    raw.kv("mystery", 13, &[]);
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::Malformed { .. })
    ));
}

#[test]
fn table_validation_collects_all_problems() {
    // Duplicate keys, duplicate tensors, bad alignment, unknown
    // tensor type, and an out-of-file range in one buffer.
    let mut raw = Raw::new(3, 3);
    raw.kv("dup", 4, &1u32.to_le_bytes());
    raw.kv("dup", 4, &2u32.to_le_bytes());
    raw.kv("general.alignment", 4, &48u32.to_le_bytes());
    raw.tensor("t", &[32], 8, 0);
    raw.tensor("t", &[32], 8, 0);
    raw.tensor("mystery", &[4], 4, 0);
    raw.bytes.extend_from_slice(&[0u8; 64]);
    match GgufFile::parse(&raw.bytes) {
        Err(FormatError::Multiple { problems }) => {
            let texts: Vec<String> = problems.iter().map(|e| e.to_string()).collect();
            assert!(
                texts.iter().any(|t| t.contains("duplicate metadata key")),
                "{texts:?}"
            );
            assert!(
                texts.iter().any(|t| t.contains("duplicate tensor name")),
                "{texts:?}"
            );
            assert!(
                texts.iter().any(|t| t.contains("invalid alignment")),
                "{texts:?}"
            );
            assert!(
                texts
                    .iter()
                    .any(|t| t.contains("unknown tensor type code 4")),
                "{texts:?}"
            );
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
    // Overlapping ranges name both tensors (a lone problem is
    // returned singly, per FormatError::collect).
    let mut raw = Raw::new(2, 0);
    raw.tensor("first", &[32], 8, 0);
    raw.tensor("second", &[32], 8, 16);
    raw.bytes.extend_from_slice(&[0u8; 128]);
    match GgufFile::parse(&raw.bytes) {
        Err(e) => assert!(e.to_string().contains("overlaps"), "{e:?}"),
        Ok(_) => panic!("overlapping ranges unexpectedly parse"),
    }
}

#[test]
fn entry_regions_match_hand_computed_layouts() {
    // I4_K over one row-block of one superblock: 16 nibble tiles
    // (128 B each) then sixteen 16 B records.
    let regions = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).expect("regions");
    assert_eq!(
        regions,
        EntryRegions {
            values_offset: 0,
            values_bytes: 2048,
            scales_offset: 2048,
            scales_bytes: 256,
            indices_offset: 4096,
            indices_bytes: 0,
            entry_bytes: 4096,
        }
    );
    // I8_B128 over 32x128: 16 byte-tiles (256 B) + 32 f16 scales.
    let regions = entry_regions(SchemeId::I8B128, Layout::L1, 32, 128).expect("regions");
    assert_eq!(regions.values_bytes, 16 * 256);
    assert_eq!(regions.scales_bytes, 2 * 16 * 2);
    assert_eq!(regions.scales_offset % 256, 0);
    assert_eq!(regions.entry_bytes % 4096, 0);
    // L1S compresses K: values shrink to the kept half plus indices.
    let dense = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).expect("dense");
    let sparse = entry_regions(SchemeId::I4K, Layout::L1S, 16, 256).expect("sparse");
    assert_eq!(sparse.values_bytes, dense.values_bytes / 2);
    assert!(sparse.indices_bytes > 0);
    assert_eq!(
        sparse.indices_offset,
        sparse.scales_offset + sparse.scales_bytes
    );
    assert_eq!(sparse.entry_bytes % 4096, 0);
    // L0 vectors are values-only entries.
    let l0 = entry_regions(SchemeId::I8B128, Layout::L0, 8, 64).expect("l0");
    assert_eq!(l0.values_offset, 0);
    assert_eq!(l0.scales_bytes, 0);
    assert_eq!(l0.indices_bytes, 0);
    assert_eq!(l0.scales_offset, l0.entry_bytes);
    // Nibble packing is not an L0 layout.
    assert!(matches!(
        entry_regions(SchemeId::I4K, Layout::L0, 8, 64),
        Err(FormatError::UnsupportedLayout { .. })
    ));
    // Zero dims are refused, not wrapped.
    assert!(entry_regions(SchemeId::I4K, Layout::L1, 0, 256).is_err());
}

#[test]
fn r9v_meta_typing_collects_all_failures() {
    let mut writer = GgufWriter::new();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .expect("kv");
    writer
        .add_kv("r9v.layout_id", KvValue::Str("L1".to_owned()))
        .expect("kv");
    writer
        .add_kv("r9v.smoothing.folded", KvValue::U32(3))
        .expect("kv");
    writer
        .add_kv("r9v.tensor.ghost.scheme", KvValue::Str("i4_k".to_owned()))
        .expect("kv");
    writer
        .add_tensor("real", &[2, 32], TensorType::Q4_0, vec![0u8; 2 * 18])
        .expect("tensor");
    writer
        .add_kv("r9v.tensor.real.act", KvValue::Str("bogus".to_owned()))
        .expect("kv");
    writer
        .add_kv(
            "r9v.tensor.real.regions",
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(0)],
            },
        )
        .expect("kv");
    let bytes = writer.emit().expect("emit");
    let file = GgufFile::parse(&bytes).expect("parses");
    match parse_r9v_meta(&file) {
        Err(FormatError::Multiple { problems }) => {
            let texts: Vec<String> = problems.iter().map(|e| e.to_string()).collect();
            assert_eq!(problems.len(), 4, "{texts:?}");
            assert!(
                texts.iter().any(|t| t.contains("r9v.smoothing.folded")),
                "{texts:?}"
            );
            assert!(texts.iter().any(|t| t.contains("ghost")), "{texts:?}");
            assert!(texts.iter().any(|t| t.contains("bogus")), "{texts:?}");
            assert!(
                texts.iter().any(|t| t.contains("3 region offsets")),
                "{texts:?}"
            );
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
}
