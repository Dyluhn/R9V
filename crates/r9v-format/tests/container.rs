// SPDX-License-Identifier: Apache-2.0
//! GGUF container tests (Spec 2 §6, §9; Spec 9 §3; card A2.5).
//!
//! Provenance:
//! - `a25_standard.gguf`, `a25_split-*.gguf`: written by
//!   `tests/fixtures/r9v-format/gen_container_fixtures.py` with
//!   gguf-py 0.19.0 (pinned, offline); seeded payload bytes cut to
//!   exact `GGML_QUANT_SIZES` lengths. Expectations below are the
//!   gguf-py reader's own values for those files.
//! - `a25_native_writer.hex`: native R9V GGUF writer golden fixture with
//!   full `r9v.*` metadata keys (alignment 4096, format_version 1, layout_id L1,
//!   i4_k tensor), verified offline at header and metadata level by pinned
//!   gguf-py 0.19.0.
//! - `llama_vocab_bert_bge.gguf`: genuine llama.cpp-produced vocab
//!   file (llama.cpp `models/`, sha256
//!   `fbcbe22278fb302694d5f4a41bfe48c5f90e8e3554eab1c0435387dff654a854`);
//!   0 tensors, 20 metadata fields.
//! - `llama_tiny_q80.hex`: genuine llama.cpp-produced quantized model
//!   with tensor table (via `llama-quantize` Q8_0); 4 tensors, 15 metadata fields.

use r9v_format::{
    accept_format_version, entry_regions, model_fp, parse_r9v_meta, r9v_tensor_type_id,
    EntryRegions, FormatError, GgufFile, GgufWriter, Interleave, KvType, KvValue, Layout,
    R9vTensorType, Role, SchemeId, ShardSet, Sparse, TensorType,
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
fn native_writer_output_matches_golden_oracle_and_expected_metadata() {
    let (bytes, hashes) = native_test_bytes();
    let oracle = fixture("a25_native_writer");
    assert_eq!(
        bytes, oracle,
        "native writer output must match committed golden oracle byte-identically"
    );

    let file = GgufFile::parse(&bytes).expect("golden native fixture parses");
    assert_eq!(
        file.kv("general.architecture"),
        Some(&KvValue::Str("llama".to_owned()))
    );
    assert_eq!(file.alignment(), 4096);
    assert_eq!(file.version(), 3);
    assert_eq!(file.tensors().len(), 1);

    let info = file.tensor("blk.0.attn_q.weight").expect("tensor present");
    assert_eq!(info.dims, [256, 16]);
    assert_eq!(
        info.dtype,
        TensorType::R9v(R9vTensorType::new(SchemeId::I4K))
    );
    assert_eq!(info.dtype.code(), 1003);
    assert_eq!(info.dtype.scheme(), Some(SchemeId::I4K));

    let meta = parse_r9v_meta(&file)
        .expect("r9v metadata parses")
        .expect("r9v metadata present");
    assert_eq!(meta.format_version, 1);
    assert_eq!(meta.layout_id, Layout::L1);
    assert_eq!(meta.arch_hint.as_deref(), Some("gfx1201"));

    let tm = meta
        .tensor("blk.0.attn_q.weight")
        .expect("tensor meta present");
    assert_eq!(tm.scheme, Some(SchemeId::I4K));
    assert_eq!(tm.roles, vec![Role::Matmul]);
    assert_eq!(tm.interleave, Interleave::None);
    assert_eq!(tm.sparse, Sparse::None);
    assert_eq!(tm.regions, Some([0, 2048, 4096]));
    assert_eq!(tm.xxh3, Some(hashes[0]));
    assert_eq!(tm.eps_int4, Some(0.01));
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

fn make_native_fixture_with_layout(
    layout_id: Option<&str>,
    extra_kvs: &[(&str, KvValue)],
    tensors: &[(&str, &[u64], SchemeId, Vec<u8>)],
) -> Vec<u8> {
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    if let Some(layout) = layout_id {
        writer
            .add_kv("r9v.layout_id", KvValue::Str(layout.to_owned()))
            .unwrap();
    }
    for (k, v) in extra_kvs {
        writer.add_kv(k, v.clone()).unwrap();
    }
    for (name, dims, scheme, data) in tensors {
        writer
            .add_tensor(
                name,
                dims,
                TensorType::R9v(R9vTensorType::new(*scheme)),
                data.clone(),
            )
            .unwrap();
    }
    writer.emit().unwrap()
}

fn make_single_native_fixture(
    extra_kvs: &[(&str, KvValue)],
    name: &str,
    dims: &[u64],
    scheme: SchemeId,
    data_len: usize,
) -> Vec<u8> {
    make_native_fixture_with_layout(
        Some("L1"),
        extra_kvs,
        &[(name, dims, scheme, vec![0u8; data_len])],
    )
}

#[test]
fn native_tied_embed_lm_head_derives_l1_entry_length() {
    // Spec 2 §4: "a tensor with both embed and lm_head roles is tied.
    // It is stored once, in L1 at the scheme the tool chose for the head role."
    // Closed set specifies exact [embed, lm_head].
    let expected_l1 = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).expect("l1 regions");
    assert_eq!(expected_l1.entry_bytes, 4096);

    let name = "tied.weight";
    // Exact [embed, lm_head] derives Layout::L1
    let bytes = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![
                    KvValue::Str("embed".to_owned()),
                    KvValue::Str("lm_head".to_owned()),
                ],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        expected_l1.entry_bytes as usize,
    );
    let file = GgufFile::parse(&bytes).expect("tied embed/lm_head parses in L1");
    let info = file.tensor(name).unwrap();
    assert_eq!(file.tensor_nbytes(info).unwrap(), expected_l1.entry_bytes);

    // Spec 2 §4 defines the closed set as exactly [embed, lm_head];
    // reversed [lm_head, embed] MUST be rejected instead of accepted.
    let bytes_rev = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![
                    KvValue::Str("lm_head".to_owned()),
                    KvValue::Str("embed".to_owned()),
                ],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        expected_l1.entry_bytes as usize,
    );
    assert!(matches!(
        GgufFile::parse(&bytes_rev),
        Err(FormatError::Malformed { .. })
    ));

    // Untied embed alone resolves to L0 (for a scheme supporting L0 like I8_B128)
    let expected_l0 = entry_regions(SchemeId::I8B128, Layout::L0, 32, 128).expect("l0");
    let embed_name = "untied_embed.weight";
    let bytes_untied = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{embed_name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("embed".to_owned())],
            },
        )],
        embed_name,
        &[32, 128],
        SchemeId::I8B128,
        expected_l0.entry_bytes as usize,
    );
    let file3 = GgufFile::parse(&bytes_untied).expect("untied embed parses in L0");
    assert_eq!(
        file3
            .tensor_nbytes(file3.tensor(embed_name).unwrap())
            .unwrap(),
        expected_l0.entry_bytes
    );
}

#[test]
fn native_1d_vectors_derive_l0_entry_length_even_with_l1_file_layout() {
    // Spec 2 §5: vectors are L0; Spec 2 §7: vectors -> L0.
    // A 1D tensor [K] without an explicit roles key in a file whose r9v.layout_id is "L1"
    // must derive Layout::L0, not Layout::L1.
    let expected_l0 = entry_regions(SchemeId::I8B128, Layout::L0, 1, 256).expect("1d l0");
    assert_eq!(expected_l0.entry_bytes, 4096);

    let vec_name = "blk.0.attn_norm.weight";
    let matmul_name = "blk.0.attn_q.weight";
    let matmul_regions = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).unwrap();

    let bytes = make_native_fixture_with_layout(
        Some("L1"),
        &[],
        &[
            (
                vec_name,
                &[256],
                SchemeId::I8B128,
                vec![0u8; expected_l0.entry_bytes as usize],
            ),
            (
                matmul_name,
                &[16, 256],
                SchemeId::I4K,
                vec![0u8; matmul_regions.entry_bytes as usize],
            ),
        ],
    );
    let file = GgufFile::parse(&bytes).expect("1D vector + 2D matmul cleanly parses");
    let vec_info = file.tensor(vec_name).unwrap();
    assert_eq!(
        file.tensor_nbytes(vec_info).unwrap(),
        expected_l0.entry_bytes
    );

    // 1D tensor cannot be sparse (Spec 2 §4, §5)
    let bytes_sparse_1d = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{vec_name}.sparse"),
            KvValue::Str("s24".to_owned()),
        )],
        vec_name,
        &[256],
        SchemeId::I8B128,
        expected_l0.entry_bytes as usize,
    );
    assert!(matches!(
        GgufFile::parse(&bytes_sparse_1d),
        Err(FormatError::Malformed { .. })
    ));

    // 1D tensor with non-vector role (e.g. matmul) is rejected
    let bytes_bad_role = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{vec_name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("matmul".to_owned())],
            },
        )],
        vec_name,
        &[256],
        SchemeId::I8B128,
        expected_l0.entry_bytes as usize,
    );
    assert!(matches!(
        GgufFile::parse(&bytes_bad_role),
        Err(FormatError::Malformed { .. })
    ));
}

#[test]
fn native_l1s_sparse_entry_length_and_scheme_restrictions() {
    let sparse_regions =
        entry_regions(SchemeId::I4K, Layout::L1S, 16, 256).expect("sparse regions");
    assert_eq!(sparse_regions.entry_bytes, 4096);

    let name = "sparse.weight";
    let bytes = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.sparse"),
            KvValue::Str("s24".to_owned()),
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        sparse_regions.entry_bytes as usize,
    );
    let file = GgufFile::parse(&bytes).expect("l1s parses cleanly");
    let info = file.tensor(name).unwrap();
    assert_eq!(
        file.tensor_nbytes(info).unwrap(),
        sparse_regions.entry_bytes
    );

    // Spec 2 §4: L1S requires a scheme in {I8_R, I8_B128, I4_K, E4M3_B128}.
    // Other schemes (like I8_B32F) fail.
    assert!(matches!(
        entry_regions(SchemeId::I8B32F, Layout::L1S, 16, 256),
        Err(FormatError::UnsupportedLayout { .. })
    ));

    // A file with s24 on an unsupported scheme is rejected
    let bytes_bad_scheme = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.sparse"),
            KvValue::Str("s24".to_owned()),
        )],
        name,
        &[16, 256],
        SchemeId::I8B32F,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes_bad_scheme),
        Err(FormatError::UnsupportedLayout { .. })
    ));
}

#[test]
fn native_metadata_closed_set_adversarial_validation() {
    let name = "test.weight";

    // 1. Invalid per-tensor sparse string (Spec 2 §4: sparse is none | s24)
    let bytes1 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.sparse"),
            KvValue::Str("dense".to_owned()),
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes1),
        Err(FormatError::Malformed { .. })
    ));

    // 2. Sparse flag wrong type (Spec 2 §4: sparse is none | s24 string)
    let bytes2 = make_single_native_fixture(
        &[(&format!("r9v.tensor.{name}.sparse"), KvValue::U32(1))],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes2),
        Err(FormatError::KvTypeMismatch { .. })
    ));

    // 3. Roles array empty
    let bytes3 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes3),
        Err(FormatError::Malformed { .. })
    ));

    // 4. Invalid roles combination [matmul, embed]
    let bytes4 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![
                    KvValue::Str("matmul".to_owned()),
                    KvValue::Str("embed".to_owned()),
                ],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes4),
        Err(FormatError::Malformed { .. })
    ));

    // 5. Unknown role string
    let bytes5 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("unknown_role".to_owned())],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes5),
        Err(FormatError::Malformed { .. })
    ));

    // 6. Invalid tied role order [lm_head, embed] (Spec 2 §4 closed set requires exact [embed, lm_head])
    let bytes6 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.roles"),
            KvValue::Array {
                elem: KvType::Str,
                items: vec![
                    KvValue::Str("lm_head".to_owned()),
                    KvValue::Str("embed".to_owned()),
                ],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes6),
        Err(FormatError::Malformed { .. })
    ));

    // 7. Unknown file-level r9v.layout_id (Spec 2 §6; note: layout_id is file-level only, no per-tensor layout_id)
    let bytes7 = make_native_fixture_with_layout(
        Some("L99"),
        &[],
        &[(name, &[16, 256], SchemeId::I4K, vec![0u8; 4096])],
    );
    assert!(matches!(
        GgufFile::parse(&bytes7),
        Err(FormatError::UnknownLayout { .. })
    ));
}

#[test]
fn explicit_regions_metadata_adversarial_validation() {
    let name = "blk.0.attn_q.weight";
    let expected_regions = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).unwrap();
    assert_eq!(expected_regions.offsets(), [0, 2048, 4096]);

    // 1. Forged offsets disagreeing with geometry [0, 2048, 8192]
    let bytes1 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.regions"),
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(0), KvValue::U64(2048), KvValue::U64(8192)],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes1),
        Err(FormatError::Malformed { .. })
    ));

    // 2. Non-zero values_offset [16, 2048, 4096]
    let bytes2 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.regions"),
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(16), KvValue::U64(2048), KvValue::U64(4096)],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes2),
        Err(FormatError::Malformed { .. })
    ));

    // 3. Unaligned scale offset [0, 2000, 4096] (2000 % 256 != 0)
    let bytes3 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.regions"),
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(0), KvValue::U64(2000), KvValue::U64(4096)],
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes3),
        Err(FormatError::Malformed { .. })
    ));

    // 4. Exact matching regions succeed
    let bytes4 = make_single_native_fixture(
        &[(
            &format!("r9v.tensor.{name}.regions"),
            KvValue::Array {
                elem: KvType::U64,
                items: expected_regions
                    .offsets()
                    .iter()
                    .map(|o| KvValue::U64(*o))
                    .collect(),
            },
        )],
        name,
        &[16, 256],
        SchemeId::I4K,
        4096,
    );
    let file4 = GgufFile::parse(&bytes4).expect("matching explicit regions parse");
    assert_eq!(
        file4.tensor_nbytes(file4.tensor(name).unwrap()).unwrap(),
        4096
    );
}

#[test]
fn native_multi_tensor_table_order_and_offsets_validation() {
    let t0 = "blk.0.attn_norm.weight";
    let reg0 = entry_regions(SchemeId::I8B128, Layout::L0, 1, 256).unwrap();
    assert_eq!(reg0.entry_bytes, 4096);
    let mut data0 = vec![0u8; 4096];
    data0[0] = 0x11;

    let t1 = "blk.0.attn_q.weight";
    let reg1 = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).unwrap();
    assert_eq!(reg1.entry_bytes, 4096);
    let mut data1 = vec![0u8; 4096];
    data1[0] = 0x22;

    let t2 = "token_embd.weight";
    let reg2 = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).unwrap();
    assert_eq!(reg2.entry_bytes, 4096);
    let mut data2 = vec![0u8; 4096];
    data2[0] = 0x33;

    let bytes = make_native_fixture_with_layout(
        Some("L1"),
        &[
            ("general.alignment", KvValue::U32(4096)),
            (
                &format!("r9v.tensor.{t2}.roles"),
                KvValue::Array {
                    elem: KvType::Str,
                    items: vec![
                        KvValue::Str("embed".to_owned()),
                        KvValue::Str("lm_head".to_owned()),
                    ],
                },
            ),
        ],
        &[
            (t0, &[256], SchemeId::I8B128, data0),
            (t1, &[16, 256], SchemeId::I4K, data1),
            (t2, &[16, 256], SchemeId::I4K, data2),
        ],
    );

    let file = GgufFile::parse(&bytes).expect("multi-tensor native file parses cleanly");
    assert_eq!(file.tensors().len(), 3);

    // Verify slicing and distinct byte content
    let b0 = file.tensor_bytes(t0, &bytes).unwrap();
    let b1 = file.tensor_bytes(t1, &bytes).unwrap();
    let b2 = file.tensor_bytes(t2, &bytes).unwrap();
    assert_eq!(b0.len(), 4096);
    assert_eq!(b1.len(), 4096);
    assert_eq!(b2.len(), 4096);
    assert_eq!(b0[0], 0x11);
    assert_eq!(b1[0], 0x22);
    assert_eq!(b2[0], 0x33);

    // Verify per-entry xxh3
    assert_eq!(
        file.entry_xxh3(t0, &bytes).unwrap(),
        r9v_common::xxh3_64(b0)
    );
    assert_eq!(
        file.entry_xxh3(t1, &bytes).unwrap(),
        r9v_common::xxh3_64(b1)
    );
    assert_eq!(
        file.entry_xxh3(t2, &bytes).unwrap(),
        r9v_common::xxh3_64(b2)
    );
}
