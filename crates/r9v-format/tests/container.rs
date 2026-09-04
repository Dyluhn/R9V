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
    accept_format_version, entry_regions, l0_region_bytes, model_fp, parse_r9v_meta,
    r9v_tensor_type_id, repack, EntryRegions, FormatError, GgmlType, GgufFile, GgufWriter,
    Interleave, KvType, KvValue, Layout, R9vTensorType, Role, SchemeId, ShardSet, Sparse,
    TensorType,
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

    let fp = file.file_fp(&bytes, 1).expect("file_fp");
    let model = model_fp(fp, &hashes);

    // Independent golden computation of file_fp and model_fp (Spec 9 §3):
    let mut manual_fp_input = Vec::new();
    manual_fp_input
        .extend_from_slice(&bytes[file.header_range().0 as usize..file.header_range().1 as usize]);
    manual_fp_input
        .extend_from_slice(&bytes[file.ti_range().0 as usize..file.ti_range().1 as usize]);
    manual_fp_input
        .extend_from_slice(&bytes[file.kv_range().0 as usize..file.kv_range().1 as usize]);
    manual_fp_input.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    manual_fp_input.extend_from_slice(&1u64.to_le_bytes());
    let expected_file_fp = r9v_common::xxh3_128(&manual_fp_input);
    assert_eq!(fp, expected_file_fp);

    let mut manual_model_input = Vec::new();
    manual_model_input.extend_from_slice(&expected_file_fp.to_le_bytes());
    for h in &hashes {
        manual_model_input.extend_from_slice(&h.to_le_bytes());
    }
    let expected_model_fp = r9v_common::xxh3_128(&manual_model_input);
    assert_eq!(model, expected_model_fp);
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
        poisoned_file.file_fp(&poisoned, 1).unwrap(),
        orig_file.file_fp(&orig_bytes, 1).unwrap()
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
    const GOLDEN_STANDARD_FILE_FP: u128 = 0x16236d2d1f887bbfc508ac93de6966df;
    const GOLDEN_STANDARD_MODEL_FP: u128 = 0xd4328ed0b095fdb0d59f0d69edb95951;

    let bytes = standard_bytes();
    let file = GgufFile::parse(&bytes).expect("parses");
    let fp1 = file.file_fp(&bytes, 1).unwrap();
    assert_eq!(fp1, GOLDEN_STANDARD_FILE_FP);
    let file2 = GgufFile::parse(&bytes).expect("parses again");
    assert_eq!(fp1, file2.file_fp(&bytes, 1).unwrap());
    // Shard count feeds the fingerprint.
    assert_ne!(fp1, file.file_fp(&bytes, 2).unwrap());
    // One flipped key byte (kept ASCII so the file still parses)
    // changes file_fp.
    let mut changed = bytes.clone();
    let kv_off = file.kv_range().0 as usize;
    changed[kv_off + 20] ^= 0x01;
    let file_changed = GgufFile::parse(&changed).expect("still parses");
    assert_ne!(fp1, file_changed.file_fp(&changed, 1).unwrap());

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
    // Non-tautological golden verification: independently assemble slices and hash with xxh3_128
    let mut manual_model_input = Vec::new();
    manual_model_input.extend_from_slice(&fp1.to_le_bytes());
    for h in &hashes {
        manual_model_input.extend_from_slice(&h.to_le_bytes());
    }
    let expected_model_fp = r9v_common::xxh3_128(&manual_model_input);
    assert_eq!(model, expected_model_fp);
    assert_eq!(model, GOLDEN_STANDARD_MODEL_FP);

    // One flipped weight byte changes model_fp but not file_fp.
    let mut wchanged = bytes.clone();
    let w = file.tensor_bytes("w_q80", &bytes).expect("bytes");
    let start = w.as_ptr() as usize - bytes.as_ptr() as usize;
    wchanged[start] ^= 0x01;
    let wfile = GgufFile::parse(&wchanged).expect("parses");
    assert_eq!(fp1, wfile.file_fp(&wchanged, 1).unwrap());
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
    // The writer gate refuses to emit a future format version.
    let mut writer = GgufWriter::new().with_alignment(4096).expect("align");
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .expect("kv");
    writer
        .add_kv("r9v.format_version", KvValue::U32(9))
        .expect("kv");
    writer
        .add_kv("r9v.layout_id", KvValue::Str("L1".to_owned()))
        .expect("kv");
    assert!(matches!(
        writer.emit(),
        Err(FormatError::FormatVersion { found: 9, max: 1 })
    ));
    // The parse gate enforces the same rule on foreign bytes.
    let mut raw = Raw::new(0, 3);
    raw.u32_kv("general.alignment", 4096);
    raw.u32_kv("r9v.format_version", 9);
    raw.str_kv("r9v.layout_id", "L1");
    // Pad so the zero-tensor native data-section start sits inside
    // the buffer: the rejection below must come from the version alone.
    raw.pad_to(4096);
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::FormatVersion { found: 9, max: 1 })
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

    fn u32_kv(&mut self, key: &str, v: u32) {
        self.str_raw(key.as_bytes());
        self.bytes.extend_from_slice(&4u32.to_le_bytes());
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    fn str_kv(&mut self, key: &str, v: &str) {
        self.str_raw(key.as_bytes());
        self.bytes.extend_from_slice(&8u32.to_le_bytes());
        self.str_raw(v.as_bytes());
    }

    /// Pads with zeros to a multiple of `align` (the writer's data-section rule).
    fn pad_to(&mut self, align: u64) {
        let rem = self.bytes.len() as u64 % align;
        if rem != 0 {
            self.bytes
                .extend_from_slice(&vec![0u8; (align - rem) as usize]);
        }
    }
}

/// Builds one single-tensor native file by hand, bypassing the
/// writer gate so parse-time rejection stays testable (Spec 2 §6;
/// card A2.5). `dims` are logical (outer-last, as in
/// `GgufWriter::add_tensor`); `extra_kvs` are encoded after the
/// required `general.alignment`, `r9v.format_version`, and
/// `r9v.layout_id` keys.
fn native_raw_bytes(
    layout_id: &str,
    extra_kvs: &[(&str, u32, Vec<u8>)],
    tensor_name: &str,
    dims: &[u64],
    ty: u32,
    payload_len: usize,
) -> Vec<u8> {
    let n_kv = 3 + extra_kvs.len() as u64;
    let mut raw = Raw::new(1, n_kv);
    raw.u32_kv("general.alignment", 4096);
    raw.u32_kv("r9v.format_version", 1);
    raw.str_kv("r9v.layout_id", layout_id);
    for (key, ty, value) in extra_kvs {
        raw.str_raw(key.as_bytes());
        raw.bytes.extend_from_slice(&ty.to_le_bytes());
        raw.bytes.extend_from_slice(value);
    }
    let mut wire: Vec<u64> = dims.to_vec();
    wire.reverse();
    raw.tensor(tensor_name, &wire, ty, 0);
    raw.pad_to(4096);
    raw.bytes.extend_from_slice(&vec![0u8; payload_len]);
    raw.bytes
}

fn enc_str_value(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    out
}

fn enc_str_array_value(items: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for item in items {
        out.extend_from_slice(&(item.len() as u64).to_le_bytes());
        out.extend_from_slice(item.as_bytes());
    }
    out
}

fn enc_u64_array_value(items: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&10u32.to_le_bytes());
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for item in items {
        out.extend_from_slice(&item.to_le_bytes());
    }
    out
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
    let l0 = entry_regions(SchemeId::I8B128, Layout::L0, 8, 128).expect("l0");
    assert_eq!(l0.values_offset, 0);
    assert_eq!(l0.scales_bytes, 0);
    assert_eq!(l0.indices_bytes, 0);
    assert_eq!(l0.scales_offset, l0.entry_bytes);
    // Invalid block divisibility for scheme fails closed.
    assert!(matches!(
        entry_regions(SchemeId::I8B128, Layout::L0, 8, 64),
        Err(FormatError::InvalidBlock { .. })
    ));
    // L0 supports all schemes where block divides k, including I4K at k=256.
    let l0_i4k = entry_regions(SchemeId::I4K, Layout::L0, 8, 256).expect("l0 i4k");
    assert_eq!(l0_i4k.values_offset, 0);
    assert_eq!(l0_i4k.scales_bytes, 0);
    assert_eq!(l0_i4k.indices_bytes, 0);
    assert_eq!(l0_i4k.entry_bytes % 4096, 0);
    // Zero dims are refused, not wrapped.
    assert!(entry_regions(SchemeId::I4K, Layout::L1, 0, 256).is_err());
}

#[test]
fn r9v_meta_typing_collects_all_failures() {
    let mut writer = GgufWriter::new().with_alignment(4096).expect("align");
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
        .add_kv("r9v.smoothing.folded", KvValue::U32(3))
        .expect("kv");
    writer
        .add_kv("r9v.tensor.ghost.scheme", KvValue::Str("i4_k".to_owned()))
        .expect("kv");
    // NOTE: the payload must be the exact derived L1 entry length
    // (4096): the writer gate enforces exact native lengths at emit.
    writer
        .add_tensor("real", &[2, 32], TensorType::F16, vec![0u8; 4096])
        .expect("tensor");
    writer
        .add_kv("r9v.tensor.real.act", KvValue::Str("bogus".to_owned()))
        .expect("kv");
    // NOTE: a malformed `regions` arity never reaches `parse_r9v_meta`:
    // both the writer gate and the container parse gate reject anything
    // but exactly 3 region offsets first (Spec 2 §6). Region arity is
    // covered by `test_adversarial_explicit_regions_arity` instead.
    let bytes = writer.emit().expect("emit");
    let file = GgufFile::parse(&bytes).expect("parses");
    match parse_r9v_meta(&file) {
        Err(FormatError::Multiple { problems }) => {
            let texts: Vec<String> = problems.iter().map(|e| e.to_string()).collect();
            assert_eq!(problems.len(), 3, "{texts:?}");
            assert!(
                texts.iter().any(|t| t.contains("r9v.smoothing.folded")),
                "{texts:?}"
            );
            assert!(texts.iter().any(|t| t.contains("ghost")), "{texts:?}");
            assert!(texts.iter().any(|t| t.contains("bogus")), "{texts:?}");
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
    if !extra_kvs.iter().any(|(k, _)| *k == "general.alignment") {
        writer
            .add_kv("general.alignment", KvValue::U32(4096))
            .unwrap();
    }
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
    // Hand-encoded so the parse gate is tested (the writer gate refuses
    // to emit it; see `test_adversarial_writer_rejects_invalid_metadata`).
    let i4k_code = TensorType::R9v(R9vTensorType::new(SchemeId::I4K)).code();
    let bytes_rev = native_raw_bytes(
        "L1",
        &[(
            &format!("r9v.tensor.{name}.roles"),
            9,
            enc_str_array_value(&["lm_head", "embed"]),
        )],
        name,
        &[16, 256],
        i4k_code,
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

    // 1D tensor cannot be sparse (Spec 2 §4, §5). Hand-encoded: the
    // writer gate refuses to emit this (see
    // `test_adversarial_writer_rejects_invalid_metadata`).
    let i8b128_code = TensorType::R9v(R9vTensorType::new(SchemeId::I8B128)).code();
    let bytes_sparse_1d = native_raw_bytes(
        "L1",
        &[(
            &format!("r9v.tensor.{vec_name}.sparse"),
            8,
            enc_str_value("s24"),
        )],
        vec_name,
        &[256],
        i8b128_code,
        expected_l0.entry_bytes as usize,
    );
    assert!(matches!(
        GgufFile::parse(&bytes_sparse_1d),
        Err(FormatError::Malformed { .. })
    ));

    // 1D tensor with non-vector role (e.g. matmul) is rejected
    let bytes_bad_role = native_raw_bytes(
        "L1",
        &[(
            &format!("r9v.tensor.{vec_name}.roles"),
            9,
            enc_str_array_value(&["matmul"]),
        )],
        vec_name,
        &[256],
        i8b128_code,
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

    // A file with s24 on an unsupported scheme is rejected.
    // Hand-encoded: the writer gate refuses to emit it.
    let i8b32f_code = TensorType::R9v(R9vTensorType::new(SchemeId::I8B32F)).code();
    let bytes_bad_scheme = native_raw_bytes(
        "L1",
        &[(
            &format!("r9v.tensor.{name}.sparse"),
            8,
            enc_str_value("s24"),
        )],
        name,
        &[16, 256],
        i8b32f_code,
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
    let i4k_code = TensorType::R9v(R9vTensorType::new(SchemeId::I4K)).code();
    // All cases are hand-encoded: the writer gate refuses to emit any
    // of them (see `test_adversarial_writer_rejects_invalid_metadata`),
    // so the writer cannot build parse-time fixtures for them.
    let sparse_key = format!("r9v.tensor.{name}.sparse");
    let roles_key = format!("r9v.tensor.{name}.roles");

    // 0. Valid baseline: the hand encoder itself produces a clean file.
    let baseline = native_raw_bytes("L1", &[], name, &[16, 256], i4k_code, 4096);
    assert!(GgufFile::parse(&baseline).is_ok());

    // 1. Invalid per-tensor sparse string (Spec 2 §4: sparse is none | s24)
    let bytes1 = native_raw_bytes(
        "L1",
        &[(&sparse_key, 8, enc_str_value("dense"))],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes1),
        Err(FormatError::Malformed { .. })
    ));

    // 2. Sparse flag wrong type (Spec 2 §4: sparse is none | s24 string)
    let mut sparse_u32 = Vec::new();
    sparse_u32.extend_from_slice(&1u32.to_le_bytes());
    let bytes2 = native_raw_bytes(
        "L1",
        &[(&sparse_key, 4, sparse_u32)],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes2),
        Err(FormatError::KvTypeMismatch { .. })
    ));

    // 3. Roles array empty
    let bytes3 = native_raw_bytes(
        "L1",
        &[(&roles_key, 9, enc_str_array_value(&[]))],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes3),
        Err(FormatError::Malformed { .. })
    ));

    // 4. Invalid roles combination [matmul, embed]
    let bytes4 = native_raw_bytes(
        "L1",
        &[(&roles_key, 9, enc_str_array_value(&["matmul", "embed"]))],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes4),
        Err(FormatError::Malformed { .. })
    ));

    // 5. Unknown role string
    let bytes5 = native_raw_bytes(
        "L1",
        &[(&roles_key, 9, enc_str_array_value(&["unknown_role"]))],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes5),
        Err(FormatError::Malformed { .. })
    ));

    // 6. Invalid tied role order [lm_head, embed] (Spec 2 §4 closed set requires exact [embed, lm_head])
    let bytes6 = native_raw_bytes(
        "L1",
        &[(&roles_key, 9, enc_str_array_value(&["lm_head", "embed"]))],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes6),
        Err(FormatError::Malformed { .. })
    ));

    // 7. Unknown file-level r9v.layout_id (Spec 2 §6; note: layout_id is file-level only, no per-tensor layout_id).
    // Both the file-level check and the per-tensor derivation report
    // it, so the collected error is a Multiple of UnknownLayout.
    let bytes7 = native_raw_bytes("L99", &[], name, &[16, 256], i4k_code, 4096);
    match GgufFile::parse(&bytes7) {
        Err(FormatError::Multiple { problems }) => {
            assert!(!problems.is_empty());
            assert!(
                problems
                    .iter()
                    .all(|e| matches!(e, FormatError::UnknownLayout { .. })),
                "got {problems:?}"
            );
        }
        other => panic!("expected Multiple of UnknownLayout, got {other:?}"),
    }
}

#[test]
fn explicit_regions_metadata_adversarial_validation() {
    let name = "blk.0.attn_q.weight";
    let expected_regions = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).unwrap();
    assert_eq!(expected_regions.offsets(), [0, 2048, 4096]);

    // Negative cases are hand-encoded: the writer gate refuses to emit
    // forged regions, so the writer cannot build these fixtures.
    let i4k_code = TensorType::R9v(R9vTensorType::new(SchemeId::I4K)).code();
    let regions_key = format!("r9v.tensor.{name}.regions");
    // 1. Forged offsets disagreeing with geometry [0, 2048, 8192]
    let bytes1 = native_raw_bytes(
        "L1",
        &[(&regions_key, 9, enc_u64_array_value(&[0, 2048, 8192]))],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes1),
        Err(FormatError::Malformed { .. })
    ));

    // 2. Non-zero values_offset [16, 2048, 4096]
    let bytes2 = native_raw_bytes(
        "L1",
        &[(&regions_key, 9, enc_u64_array_value(&[16, 2048, 4096]))],
        name,
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&bytes2),
        Err(FormatError::Malformed { .. })
    ));

    // 3. Unaligned scale offset [0, 2000, 4096] (2000 % 256 != 0)
    let bytes3 = native_raw_bytes(
        "L1",
        &[(&regions_key, 9, enc_u64_array_value(&[0, 2000, 4096]))],
        name,
        &[16, 256],
        i4k_code,
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

#[test]
fn test_gguf_wire_sizing_validates_nonempty_nonzero_and_dims0_block_divisibility() {
    // 1. TensorType::data_nbytes validates dims
    // Empty shape
    assert_eq!(TensorType::Q4_0.data_nbytes(&[]), None);
    assert_eq!(TensorType::F16.data_nbytes(&[]), None);
    // Zero dimension anywhere in shape
    assert_eq!(TensorType::Q4_0.data_nbytes(&[0, 32]), None);
    assert_eq!(TensorType::Q4_0.data_nbytes(&[32, 0]), None);
    assert_eq!(TensorType::F16.data_nbytes(&[0]), None);
    assert_eq!(TensorType::F16.data_nbytes(&[16, 0]), None);
    // dims[0] block divisibility for quantized wire types
    // Q4_0 block_len is 32. dims[0] must be a multiple of 32.
    assert_eq!(TensorType::Q4_0.data_nbytes(&[16, 2]), None);
    assert_eq!(TensorType::Q4_0.data_nbytes(&[31, 1]), None);
    assert_eq!(TensorType::Q4_0.data_nbytes(&[32, 1]), Some(18));
    assert_eq!(TensorType::Q4_0.data_nbytes(&[64, 3]), Some(36 * 3));
    // Q4_K block_len is 256.
    assert_eq!(TensorType::Q4_K.data_nbytes(&[128, 1]), None);
    assert_eq!(TensorType::Q4_K.data_nbytes(&[256, 1]), Some(144));
    assert_eq!(TensorType::Q4_K.data_nbytes(&[512, 2]), Some(288 * 2));

    // 2. GgufWriter rejects empty shape, zero dimensions, and indivisible dim[0]
    let mut writer = GgufWriter::new();
    assert!(matches!(
        writer.add_tensor("bad_empty", &[], TensorType::Q4_0, vec![]),
        Err(FormatError::Malformed { .. })
    ));
    assert!(matches!(
        writer.add_tensor("bad_zero", &[0, 32], TensorType::Q4_0, vec![]),
        Err(FormatError::Malformed { .. })
    ));
    assert!(matches!(
        writer.add_tensor("bad_div", &[16, 1], TensorType::Q4_0, vec![0u8; 9]),
        Err(FormatError::Malformed { .. })
    ));

    // 3. Raw standard GGUF container parser rejects empty dims, zero dims, and indivisible dim[0]
    // Raw empty dims
    let mut raw = Raw::new(1, 0);
    raw.tensor("empty_dims", &[], 2, 0); // 2 = Q4_0
    raw.bytes.extend_from_slice(&[0u8; 64]);
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::Malformed { .. })
    ));

    // Raw zero dim
    let mut raw = Raw::new(1, 0);
    raw.tensor("zero_dim", &[0, 32], 2, 0);
    raw.bytes.extend_from_slice(&[0u8; 64]);
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::Malformed { .. })
    ));

    // Raw indivisible dim[0]
    let mut raw = Raw::new(1, 0);
    raw.tensor("indivisible", &[16, 1], 2, 0);
    raw.bytes.extend_from_slice(&[0u8; 64]);
    assert!(matches!(
        GgufFile::parse(&raw.bytes),
        Err(FormatError::Malformed { .. })
    ));
}

#[test]
fn test_every_tensor_offset_alignment() {
    // Default alignment is 32.
    // If a tensor offset is not a multiple of alignment, validate() rejects it with BadTensorRange.
    let mut raw = Raw::new(1, 0);
    // offset 1 is not a multiple of 32
    raw.tensor("misaligned", &[32, 1], 2, 1); // Q4_0, dims [32, 1], offset 1
    raw.bytes.extend_from_slice(&[0u8; 128]);
    match GgufFile::parse(&raw.bytes) {
        Err(FormatError::BadTensorRange { name, reason, .. }) => {
            assert_eq!(name, "misaligned");
            assert!(
                reason.contains("not a multiple of alignment 32"),
                "{reason}"
            );
        }
        other => panic!("expected BadTensorRange, got {other:?}"),
    }

    // Custom alignment 64: offset 32 is aligned to 32 but NOT 64.
    let mut raw = Raw::new(1, 1);
    raw.kv("general.alignment", 4, &64u32.to_le_bytes());
    raw.tensor("misaligned64", &[32, 1], 2, 32); // offset 32
    raw.bytes.extend_from_slice(&[0u8; 256]);
    match GgufFile::parse(&raw.bytes) {
        Err(FormatError::BadTensorRange { name, reason, .. }) => {
            assert_eq!(name, "misaligned64");
            assert!(
                reason.contains("not a multiple of alignment 64"),
                "{reason}"
            );
        }
        other => panic!("expected BadTensorRange, got {other:?}"),
    }
}

#[test]
fn test_split_metadata_coherent_declarations_and_single_nonsplit_works() {
    // 1. Single non-split file without any split.* keys parses cleanly and works with ShardSet
    let mut writer = GgufWriter::new();
    writer
        .add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("nonsplit file parses");
    let shards = ShardSet::open(vec![file]).expect("single nonsplit shard opens in ShardSet");
    assert_eq!(shards.shards().len(), 1);

    // 2. Single file declaring partial split keys fails closed
    // Missing split.count and split.tensors.count
    let mut writer = GgufWriter::new();
    writer.add_kv("split.no", KvValue::U16(0)).unwrap();
    writer
        .add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let bytes = writer.emit().unwrap();
    match GgufFile::parse(&bytes) {
        Err(FormatError::Multiple { problems }) => {
            assert!(problems
                .iter()
                .any(|p| matches!(p, FormatError::MissingKey { key } if key == "split.count")));
            assert!(problems.iter().any(
                |p| matches!(p, FormatError::MissingKey { key } if key == "split.tensors.count")
            ));
        }
        other => panic!("expected Multiple MissingKey, got {other:?}"),
    }

    // Wrong type: split.no as U32
    let mut writer = GgufWriter::new();
    writer.add_kv("split.no", KvValue::U32(0)).unwrap();
    writer.add_kv("split.count", KvValue::U16(1)).unwrap();
    writer
        .add_kv("split.tensors.count", KvValue::I32(1))
        .unwrap();
    writer
        .add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let bytes = writer.emit().unwrap();
    assert!(matches!(
        GgufFile::parse(&bytes),
        Err(FormatError::KvTypeMismatch { key, expected, .. }) if key == "split.no" && expected == "UINT16"
    ));

    // Incoherent values: split.count == 0
    let mut writer = GgufWriter::new();
    writer.add_kv("split.no", KvValue::U16(0)).unwrap();
    writer.add_kv("split.count", KvValue::U16(0)).unwrap();
    writer
        .add_kv("split.tensors.count", KvValue::I32(1))
        .unwrap();
    writer
        .add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let bytes = writer.emit().unwrap();
    assert!(matches!(
        GgufFile::parse(&bytes),
        Err(FormatError::Malformed { .. })
    ));

    // Incoherent values: split.no >= split.count
    let mut writer = GgufWriter::new();
    writer.add_kv("split.no", KvValue::U16(2)).unwrap();
    writer.add_kv("split.count", KvValue::U16(2)).unwrap();
    writer
        .add_kv("split.tensors.count", KvValue::I32(1))
        .unwrap();
    writer
        .add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let bytes = writer.emit().unwrap();
    assert!(matches!(
        GgufFile::parse(&bytes),
        Err(FormatError::Malformed { .. })
    ));

    // Single shard: split.count == 1 but split.tensors.count != self.tensors.len()
    let mut writer = GgufWriter::new();
    writer.add_kv("split.no", KvValue::U16(0)).unwrap();
    writer.add_kv("split.count", KvValue::U16(1)).unwrap();
    writer
        .add_kv("split.tensors.count", KvValue::I32(5))
        .unwrap(); // 5 != 1
    writer
        .add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let bytes = writer.emit().unwrap();
    assert!(matches!(
        GgufFile::parse(&bytes),
        Err(FormatError::Malformed { .. })
    ));

    // Complete coherent single-shard split parses cleanly
    let mut writer = GgufWriter::new();
    writer.add_kv("split.no", KvValue::U16(0)).unwrap();
    writer.add_kv("split.count", KvValue::U16(1)).unwrap();
    writer
        .add_kv("split.tensors.count", KvValue::I32(1))
        .unwrap();
    writer
        .add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("coherent single-shard split parses");
    let shards = ShardSet::open(vec![file]).expect("coherent split ShardSet opens");
    assert_eq!(shards.shards().len(), 1);

    // 3. Multi-shard: ShardSet::open enforces complete coherent split keys across shards
    let mut w0 = GgufWriter::new();
    w0.add_kv("split.no", KvValue::U16(0)).unwrap();
    w0.add_kv("split.count", KvValue::U16(2)).unwrap();
    w0.add_kv("split.tensors.count", KvValue::I32(2)).unwrap();
    w0.add_tensor("t0", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let b0 = w0.emit().unwrap();

    let mut w1 = GgufWriter::new();
    w1.add_kv("split.no", KvValue::U16(1)).unwrap();
    w1.add_kv("split.count", KvValue::U16(2)).unwrap();
    w1.add_kv("split.tensors.count", KvValue::I32(2)).unwrap();
    w1.add_tensor("t1", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let b1 = w1.emit().unwrap();

    let f0 = GgufFile::parse(&b0).unwrap();
    let f1 = GgufFile::parse(&b1).unwrap();
    let set = ShardSet::open(vec![f0.clone(), f1.clone()]).expect("coherent multi-shard opens");
    assert_eq!(set.shards().len(), 2);
    assert!(set.tensor("t0").is_some());
    assert!(set.tensor("t1").is_some());

    // Multi-shard with missing key on shard 1 fails closed
    let mut w1_bad = GgufWriter::new();
    w1_bad.add_kv("split.no", KvValue::U16(1)).unwrap();
    w1_bad.add_kv("split.count", KvValue::U16(2)).unwrap();
    // omit split.tensors.count
    w1_bad
        .add_tensor("t1", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let b1_bad = w1_bad.emit().unwrap();
    let f1_bad = GgufFile::parse(&b1_bad);
    assert!(f1_bad.is_err()); // fails validate during GgufFile::parse

    // Multi-shard with count mismatch across shards fails closed in ShardSet::open
    let mut w1_mismatch = GgufWriter::new();
    w1_mismatch.add_kv("split.no", KvValue::U16(1)).unwrap();
    w1_mismatch.add_kv("split.count", KvValue::U16(3)).unwrap(); // declares 3 instead of 2
    w1_mismatch
        .add_kv("split.tensors.count", KvValue::I32(2))
        .unwrap();
    w1_mismatch
        .add_tensor("t1", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    let b1_mismatch = w1_mismatch.emit().unwrap();
    let f1_mismatch = GgufFile::parse(&b1_mismatch).unwrap();
    assert!(matches!(
        ShardSet::open(vec![f0, f1_mismatch]),
        Err(FormatError::Malformed { .. })
    ));
}

#[test]
fn test_writer_arrays_reject_nested_and_mismatched_declared_element_types() {
    let mut writer = GgufWriter::new();

    // 1. Array with elem == KvType::Array is rejected
    let res1 = writer.add_kv(
        "nested_array",
        KvValue::Array {
            elem: KvType::Array,
            items: vec![],
        },
    );
    assert!(matches!(res1, Err(FormatError::Malformed { .. })));

    // 2. Array containing a nested KvValue::Array item is rejected
    let res2 = writer.add_kv(
        "nested_item",
        KvValue::Array {
            elem: KvType::U32,
            items: vec![KvValue::Array {
                elem: KvType::U32,
                items: vec![],
            }],
        },
    );
    assert!(matches!(res2, Err(FormatError::Malformed { .. })));

    // 3. Array containing an item whose type does not match declared elem is rejected
    let res3 = writer.add_kv(
        "type_mismatch",
        KvValue::Array {
            elem: KvType::U32,
            items: vec![KvValue::Str("not_a_u32".to_owned())],
        },
    );
    assert!(matches!(
        res3,
        Err(FormatError::KvTypeMismatch { key, found, expected })
            if key == "type_mismatch" && found == "STRING" && expected == "UINT32"
    ));
}

#[test]
fn test_native_typed_metadata_enforces_4096_alignment_and_native_scheme_agreement() {
    // 1. Native file with alignment != 4096 is rejected at emit ...
    let mut writer = GgufWriter::new();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    // Default alignment is 32 (not 4096)
    assert!(matches!(
        writer.emit(),
        Err(FormatError::InvalidAlignment { value: 32 })
    ));
    // ... and foreign bytes with default alignment fail at parse.
    let mut raw_align = Raw::new(0, 2);
    raw_align.u32_kv("r9v.format_version", 1);
    raw_align.str_kv("r9v.layout_id", "l1");
    // Pad so the zero-tensor data-section start sits inside the
    // buffer: the rejection below must come from alignment alone.
    raw_align.pad_to(32);
    assert!(matches!(
        GgufFile::parse(&raw_align.bytes),
        Err(FormatError::InvalidAlignment { value: 32 })
    ));

    // 2. Scheme agreement: meta.scheme does not match R9v tensor scheme.
    // The writer gate refuses to emit the mismatch ...
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    let regions = entry_regions(SchemeId::I4K, Layout::L1, 16, 256).unwrap();
    writer
        .add_tensor(
            "weight",
            &[16, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            vec![0u8; regions.entry_bytes as usize],
        )
        .unwrap();
    // Intentionally declare mismatched scheme "i8_r" for i4_k tensor
    writer
        .add_kv("r9v.tensor.weight.scheme", KvValue::Str("i8_r".to_owned()))
        .unwrap();
    assert!(matches!(
        writer.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("declared scheme")
    ));
    // ... while a foreign file with the same mismatch is rejected
    // by the container parser itself, before the typed layer.
    let i4k_code = TensorType::R9v(R9vTensorType::new(SchemeId::I4K)).code();
    let scheme_key = "r9v.tensor.weight.scheme".to_owned();
    let mismatch_bytes = native_raw_bytes(
        "L1",
        &[(&scheme_key, 8, enc_str_value("i8_r"))],
        "weight",
        &[16, 256],
        i4k_code,
        regions.entry_bytes as usize,
    );
    match GgufFile::parse(&mismatch_bytes) {
        Err(FormatError::SchemeMismatch { expected, got, .. }) => {
            assert_eq!(expected, "i4_k");
            assert_eq!(got, "i8_r");
        }
        other => panic!("expected SchemeMismatch, got {other:?}"),
    }

    // 3. Unquantized F16/BF16/F32 cannot declare a quantization scheme.
    // Rejected at emit ...
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_tensor("f16_weight", &[2, 32], TensorType::F16, vec![0u8; 128])
        .unwrap();
    writer
        .add_kv(
            "r9v.tensor.f16_weight.scheme",
            KvValue::Str("i4_k".to_owned()),
        )
        .unwrap();
    assert!(matches!(writer.emit(), Err(FormatError::Malformed { .. })));
    // ... and at parse on foreign bytes.
    let f16_scheme_key = "r9v.tensor.f16_weight.scheme".to_owned();
    let f16_scheme_bytes = native_raw_bytes(
        "L1",
        &[(&f16_scheme_key, 8, enc_str_value("i4_k"))],
        "f16_weight",
        &[2, 32],
        TensorType::F16.code(),
        128,
    );
    assert!(matches!(
        GgufFile::parse(&f16_scheme_bytes),
        Err(FormatError::Malformed { .. })
    ));

    // 4. Native F32 must be a 1D vector (Spec 2 §3.3): a 2D F32 matrix
    // is rejected at emit ...
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    // 2D F32 tensor with default file layout L1
    writer
        .add_tensor(
            "f32_matrix",
            &[16, 256],
            TensorType::F32,
            vec![0u8; 16 * 256 * 4],
        )
        .unwrap();
    assert!(matches!(
        writer.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("must be 1D")
    ));
    // ... and at parse on foreign bytes.
    let f32_matrix_bytes = native_raw_bytes(
        "L1",
        &[],
        "f32_matrix",
        &[16, 256],
        TensorType::F32.code(),
        16 * 256 * 4,
    );
    assert!(matches!(
        GgufFile::parse(&f32_matrix_bytes),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("must be 1D")
    ));
}

#[test]
fn test_l0_entry_geometry_exact_per_row_bits_for_every_scheme() {
    const BLOCK_SCHEMES_ORACLE: [(SchemeId, u32, u64); 21] = [
        (SchemeId::I8B128, 128, 1040),
        (SchemeId::I4K, 256, 1152),
        (SchemeId::E4M3B128, 128, 1040),
        (SchemeId::I8B32F, 32, 272),
        (SchemeId::I4B32F, 32, 144),
        (SchemeId::I4B32FM, 32, 160),
        (SchemeId::I5B32F, 32, 176),
        (SchemeId::I5B32FM, 32, 192),
        (SchemeId::I4Nl, 32, 144),
        (SchemeId::I5K, 256, 1408),
        (SchemeId::I6K, 256, 1680),
        (SchemeId::I3K, 256, 880),
        (SchemeId::I2K, 256, 672),
        (SchemeId::I4Xs, 256, 1088),
        (SchemeId::Iq3Xxs, 256, 784),
        (SchemeId::Iq3S, 256, 880),
        (SchemeId::Iq2Xxs, 256, 528),
        (SchemeId::Iq2Xs, 256, 592),
        (SchemeId::Iq2S, 256, 656),
        (SchemeId::Iq1S, 256, 400),
        (SchemeId::Iq1M, 256, 448),
    ];

    for (scheme, block_k, bits_per_block) in BLOCK_SCHEMES_ORACLE {
        for num_blocks in [1, 2, 4] {
            let k = block_k * num_blocks;
            let n = 8;
            let regions = entry_regions(scheme, Layout::L0, n, k).unwrap_or_else(|e| {
                panic!("entry_regions failed for {scheme:?} L0 with k={k}: {e:?}")
            });
            let expected_bits_per_row = (num_blocks as u64) * bits_per_block;
            assert_eq!(expected_bits_per_row % 8, 0);
            let expected_row_bytes = expected_bits_per_row / 8;
            let expected_values_bytes = l0_region_bytes(n, expected_row_bytes).unwrap();
            assert_eq!(regions.values_offset, 0);
            assert_eq!(regions.values_bytes, expected_values_bytes);
            assert_eq!(regions.scales_bytes, 0);
            assert_eq!(regions.indices_bytes, 0);
            assert_eq!(regions.scales_offset, regions.entry_bytes);
            assert_eq!(regions.indices_offset, regions.entry_bytes);
            assert_eq!(regions.entry_bytes % 4096, 0);
        }

        let bad_k = block_k - 1;
        assert!(
            matches!(
                entry_regions(scheme, Layout::L0, 8, bad_k),
                Err(FormatError::InvalidBlock { .. })
            ),
            "expected InvalidBlock for {scheme:?} with k={bad_k}"
        );
    }

    // Row-wise I8_R independent formula: 8*k + 16 bits per row (accounting for 16-bit f16 row scale)
    for k in [16, 32, 64, 128, 256] {
        let n = 8;
        let regions = entry_regions(SchemeId::I8R, Layout::L0, n, k).unwrap();
        let expected_bits_per_row = 8 * (k as u64) + 16;
        assert_eq!(expected_bits_per_row % 8, 0);
        let expected_row_bytes = expected_bits_per_row / 8;
        let expected_values_bytes = l0_region_bytes(n, expected_row_bytes).unwrap();
        assert_eq!(regions.values_offset, 0);
        assert_eq!(regions.values_bytes, expected_values_bytes);
        assert_eq!(regions.scales_bytes, 0);
        assert_eq!(regions.indices_bytes, 0);
        assert_eq!(regions.scales_offset, regions.entry_bytes);
        assert_eq!(regions.indices_offset, regions.entry_bytes);
        assert_eq!(regions.entry_bytes % 4096, 0);
    }

    // Specifically verify I4_K, bitplanes (I3Xs, I2Xs), and multiple B128 blocks
    let i4k_l0 = entry_regions(SchemeId::I4K, Layout::L0, 8, 512).unwrap();
    assert_eq!(i4k_l0.values_bytes, 8 * 288); // 2 blocks of 256 -> 2 * 144 B = 288 B/row
    assert_eq!(i4k_l0.entry_bytes, 4096);

    let i8b128_l0 = entry_regions(SchemeId::I8B128, Layout::L0, 8, 384).unwrap(); // 3 blocks of 128
    assert_eq!(i8b128_l0.values_bytes, 8 * (3 * 130)); // 3 * (128 + 2) = 390 B/row -> 8 * 390 = 3120 B
    assert_eq!(i8b128_l0.entry_bytes, 4096);
}

#[test]
fn test_l1_grid_iq_packed_index_dims_proven_equal_to_actual_repack_regions() {
    let cases = [
        ("IQ4_NL", GgmlType::IQ4_NL, SchemeId::I4Nl),
        ("IQ4_XS", GgmlType::IQ4_XS, SchemeId::I4Xs),
        ("IQ3_XXS", GgmlType::IQ3_XXS, SchemeId::Iq3Xxs),
        ("IQ3_S", GgmlType::IQ3_S, SchemeId::Iq3S),
        ("IQ2_XXS", GgmlType::IQ2_XXS, SchemeId::Iq2Xxs),
        ("IQ2_XS", GgmlType::IQ2_XS, SchemeId::Iq2Xs),
        ("IQ2_S", GgmlType::IQ2_S, SchemeId::Iq2S),
        ("IQ1_S", GgmlType::IQ1_S, SchemeId::Iq1S),
        ("IQ1_M", GgmlType::IQ1_M, SchemeId::Iq1M),
    ];

    for (name, ggml, scheme) in cases {
        let bl = ggml.block_len();
        let bb = ggml.block_bytes() as usize;

        // Test with unpadded N (n=1, n=17) and padded N (n=32), with single and multiple blocks of K
        for (n, k) in [(1, 256), (17, 256), (32, 512)] {
            let n_blocks = (n as usize) * (k as usize / bl as usize);
            let mut wire = vec![0xFFu8; n_blocks * bb];

            // For each block, ensure finite scales so repack succeeds
            for block_idx in 0..n_blocks {
                let block_slice = &mut wire[block_idx * bb..(block_idx + 1) * bb];
                if name == "IQ1_M" {
                    block_slice[49] = (block_slice[49] & 0x0F) | 0x30;
                    block_slice[51] = (block_slice[51] & 0x0F) | 0xC0;
                    block_slice[53] &= 0x0F;
                    block_slice[55] &= 0x0F;
                } else {
                    block_slice[0] = 0x00;
                    block_slice[1] = 0x3C; // f16 positive finite 1.0
                }
            }

            let repacked = repack(ggml, &wire, n, k)
                .unwrap_or_else(|e| panic!("repack failed for {name} ({n}, {k}): {e:?}"));
            let regions = entry_regions(scheme, Layout::L1, n, k)
                .unwrap_or_else(|e| panic!("entry_regions failed for {name} ({n}, {k}): {e:?}"));

            // PROOF OF EQUALITY: packed index dims derived in entry_regions exactly match
            // actual repack values length and scale length!
            assert_eq!(
                repacked.values.len() as u64,
                regions.values_bytes,
                "{name} values_bytes mismatch for shape ({n}, {k})"
            );
            assert_eq!(
                repacked.scales.len() as u64,
                regions.scales_bytes,
                "{name} scales_bytes mismatch for shape ({n}, {k})"
            );
            assert_eq!(regions.indices_bytes, 0);
            assert_eq!(regions.entry_bytes % 4096, 0);
        }
    }
}

#[test]
fn test_native_retained_unquantized_types_and_unsupported_standard_fail_closed() {
    // 1. Native file retains F16 in L1 padded to 4096
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    // [1, 64] L1 entry bytes = 4096
    writer
        .add_tensor("f16_small", &[1, 64], TensorType::F16, vec![0u8; 4096])
        .unwrap();
    // [16, 256] L1 entry bytes = 8192
    writer
        .add_tensor("f16_large", &[16, 256], TensorType::F16, vec![0u8; 8192])
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("parses native f16");
    assert_eq!(
        file.tensor_nbytes(file.tensor("f16_small").unwrap())
            .unwrap(),
        4096
    );
    assert_eq!(
        file.tensor_nbytes(file.tensor("f16_large").unwrap())
            .unwrap(),
        8192
    );

    // 2. Native file retains BF16 in L1 padded to 4096
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_tensor("bf16_small", &[1, 64], TensorType::BF16, vec![0u8; 4096])
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("parses native bf16");
    assert_eq!(
        file.tensor_nbytes(file.tensor("bf16_small").unwrap())
            .unwrap(),
        4096
    );

    // 3. Native file retains F32 in L0 padded to 4096 (1D vector with role [vector])
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    writer
        .add_kv(
            "r9v.tensor.f32_vec.roles",
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("vector".to_owned())],
            },
        )
        .unwrap();
    writer
        .add_tensor("f32_vec", &[64], TensorType::F32, vec![0u8; 4096])
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("parses native f32 l0");
    assert_eq!(
        file.tensor_nbytes(file.tensor("f32_vec").unwrap()).unwrap(),
        4096
    );

    // 4. Native file rejects unsupported standard types (e.g. Q4_0, Q8_0) at emit
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_tensor("q4_0_bad", &[1, 32], TensorType::Q4_0, vec![0u8; 18])
        .unwrap();
    assert!(matches!(
        writer.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("unsupported tensor type \"Q4_0\"")
    ));

    // 5. Standard no-r9v GGUF retains wire semantics
    let mut writer = GgufWriter::new(); // alignment 32, no r9v.* keys
    writer
        .add_tensor("q4_0_wire", &[10, 32], TensorType::Q4_0, vec![0u8; 180])
        .unwrap();
    writer
        .add_tensor("f16_wire", &[1, 64], TensorType::F16, vec![0u8; 128])
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("standard gguf parses");
    assert!(file.is_standard_gguf());
    assert_eq!(
        file.tensor_nbytes(file.tensor("q4_0_wire").unwrap())
            .unwrap(),
        180
    );
    assert_eq!(
        file.tensor_nbytes(file.tensor("f16_wire").unwrap())
            .unwrap(),
        128
    );
}

#[test]
fn test_sweep_overlap_enclosing_interval() {
    // Deterministic sweep overlap check:
    // Tensor A: range [0, 128)
    // Tensor B: range [32, 64)
    // Tensor C: range [64, 96)
    // Adjacent-only checking compared B against A (32 < 128, overlap),
    // but then C against B (64 == 64, no overlap!), missing that C is inside A.
    // Deterministic sweep tracks max enclosing end (128) and catches both B and C!
    let mut raw = Raw::new(3, 0);
    raw.tensor("tensor_a", &[32, 1], 0, 0); // F32, dims [32, 1] -> 128 B, offset 0 -> [0, 128)
    raw.tensor("tensor_b", &[16, 1], 1, 32); // F16, dims [16, 1] -> 32 B, offset 32 -> [32, 64)
    raw.tensor("tensor_c", &[16, 1], 1, 64); // F16, dims [16, 1] -> 32 B, offset 64 -> [64, 96)
    raw.bytes.extend_from_slice(&[0u8; 512]);

    match GgufFile::parse(&raw.bytes) {
        Err(FormatError::Multiple { problems }) => {
            let overlaps: Vec<String> = problems
                .iter()
                .filter_map(|e| match e {
                    FormatError::BadTensorRange { name, reason, .. }
                        if reason.contains("overlaps") =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                })
                .collect();
            assert!(
                overlaps.contains(&"tensor_b".to_owned()),
                "tensor_b must be reported as overlapping tensor_a"
            );
            assert!(
                overlaps.contains(&"tensor_c".to_owned()),
                "tensor_c must be reported as overlapping tensor_a via sweep check"
            );
        }
        other => panic!("expected FormatError::Multiple, got {other:?}"),
    }

    // Verify sweep overlap check also runs under parse_metadata_only on truncated buffer
    let header_and_ti_len = raw.bytes.len() - 512;
    let truncated = &raw.bytes[..header_and_ti_len];
    match GgufFile::parse_metadata_only(truncated, raw.bytes.len() as u64) {
        Err(FormatError::Multiple { problems }) => {
            let overlaps: Vec<String> = problems
                .iter()
                .filter_map(|e| match e {
                    FormatError::BadTensorRange { name, reason, .. }
                        if reason.contains("overlaps") =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                })
                .collect();
            assert!(overlaps.contains(&"tensor_b".to_owned()));
            assert!(overlaps.contains(&"tensor_c".to_owned()));
        }
        other => panic!("expected FormatError::Multiple from parse_metadata_only, got {other:?}"),
    }
}

#[test]
fn test_retained_f16_bf16_l1_tile_geometry_irregular_boundary() {
    // Irregular boundary test: N=17, K=256
    // Independent tile math:
    // N padded to 16: 32 (2 tiles)
    // K padded to 16: 256 (16 tiles)
    // Total tiles: 2 * 16 = 32 tiles
    // Packing Half16: 512 bytes per tile
    // Expected value region: 32 * 512 = 16384 bytes
    // Entry bytes: align_up(16384, 4096) = 16384 bytes
    // (In contrast, naive unpadded elements: 17 * 256 * 2 = 8704 bytes, align_up = 12288 bytes)
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_tensor("w_f16", &[17, 256], TensorType::F16, vec![0u8; 16384])
        .unwrap();
    writer
        .add_tensor("w_bf16", &[17, 256], TensorType::BF16, vec![0u8; 16384])
        .unwrap();
    // Also test an L0 tensor (e.g. role embed) with N=17, K=256:
    // L0 row-major: 17 * 256 * 2 = 8704 bytes, align_up(8704, 4096) = 12288 bytes
    writer
        .add_kv(
            "r9v.tensor.w_embed.roles",
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("embed".to_owned())],
            },
        )
        .unwrap();
    writer
        .add_tensor("w_embed", &[17, 256], TensorType::F16, vec![0u8; 12288])
        .unwrap();

    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("parses cleanly");

    let info_f16 = file.tensor("w_f16").unwrap();
    assert_eq!(file.tensor_nbytes(info_f16).unwrap(), 16384);

    let info_bf16 = file.tensor("w_bf16").unwrap();
    assert_eq!(file.tensor_nbytes(info_bf16).unwrap(), 16384);

    let info_embed = file.tensor("w_embed").unwrap();
    assert_eq!(file.tensor_nbytes(info_embed).unwrap(), 12288);
}

#[test]
fn test_retained_explicit_regions_validation() {
    // Retained unquantized types have only values: explicit regions must be [0, entry_bytes, entry_bytes]
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_kv(
            "r9v.tensor.w_f16.regions",
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(0), KvValue::U64(16384), KvValue::U64(16384)],
            },
        )
        .unwrap();
    writer
        .add_tensor("w_f16", &[17, 256], TensorType::F16, vec![0u8; 16384])
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("valid explicit regions parse cleanly");
    assert_eq!(
        file.tensor_nbytes(file.tensor("w_f16").unwrap()).unwrap(),
        16384
    );

    // Corrupted explicit regions: scales_offset not equal to entry_bytes.
    // Hand-encoded (the writer gate refuses to emit forged regions);
    // the writer-gate half is covered by
    // `test_adversarial_writer_rejects_invalid_metadata`.
    let regions_key = "r9v.tensor.w_f16.regions".to_owned();
    let bytes_bad = native_raw_bytes(
        "L1",
        &[(&regions_key, 9, enc_u64_array_value(&[0, 8192, 16384]))],
        "w_f16",
        &[17, 256],
        TensorType::F16.code(),
        16384,
    );
    assert!(matches!(
        GgufFile::parse(&bytes_bad),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("do not match derived regions")
    ));

    // F32 1D vector explicit regions: entry_bytes = 4096.
    // Native F32 requires the explicit role [vector] (Spec 2 §3.3, §4).
    let mut writer_f32 = GgufWriter::new().with_alignment(4096).unwrap();
    writer_f32
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer_f32
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer_f32
        .add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    writer_f32
        .add_kv(
            "r9v.tensor.v_f32.roles",
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("vector".to_owned())],
            },
        )
        .unwrap();
    writer_f32
        .add_kv(
            "r9v.tensor.v_f32.regions",
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(0), KvValue::U64(4096), KvValue::U64(4096)],
            },
        )
        .unwrap();
    writer_f32
        .add_tensor("v_f32", &[256], TensorType::F32, vec![0u8; 4096])
        .unwrap();
    let bytes_f32 = writer_f32.emit().unwrap();
    assert!(GgufFile::parse(&bytes_f32).is_ok());
}

#[test]
fn test_gguf_writer_order_independent_native_lengths_and_strict_standard() {
    // 1. Adding tensor BEFORE metadata KVs works identically to adding KVs before tensor
    let mut w_tensor_first = GgufWriter::new().with_alignment(4096).unwrap();
    w_tensor_first
        .add_tensor("w", &[17, 256], TensorType::F16, vec![0u8; 16384])
        .unwrap();
    w_tensor_first
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_tensor_first
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_tensor_first
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    let bytes_tensor_first = w_tensor_first.emit().unwrap();

    let mut w_kv_first = GgufWriter::new().with_alignment(4096).unwrap();
    w_kv_first
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_kv_first
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_kv_first
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    w_kv_first
        .add_tensor("w", &[17, 256], TensorType::F16, vec![0u8; 16384])
        .unwrap();
    let bytes_kv_first = w_kv_first.emit().unwrap();
    assert_eq!(bytes_tensor_first, bytes_kv_first);

    // 2. Standard GGUF file rejects native-padded payload at emit
    let mut w_standard = GgufWriter::new();
    w_standard
        .add_tensor("w", &[17, 256], TensorType::F16, vec![0u8; 16384])
        .unwrap();
    assert!(matches!(
        w_standard.emit(),
        Err(FormatError::LengthMismatch { what, expected, got })
            if what == "tensor data" && expected == 8704 && got == 16384
    ));

    // 3. Invalid arbitrary payload length fails closed at add_tensor
    let mut w_bad_len = GgufWriter::new();
    assert!(matches!(
        w_bad_len.add_tensor("w", &[17, 256], TensorType::F16, vec![0u8; 5000]),
        Err(FormatError::LengthMismatch { .. })
    ));
}

#[test]
fn test_native_classification_and_agreement() {
    // 1. Standard GGUF with unknown upstream type code (e.g. 9999) fails with UnknownTensorType, NOT InvalidAlignment
    let mut raw_unknown = Raw::new(1, 0);
    raw_unknown.tensor("unk_tensor", &[32, 1], 9999, 0);
    raw_unknown.bytes.extend_from_slice(&[0u8; 256]);
    match GgufFile::parse(&raw_unknown.bytes) {
        Err(FormatError::UnknownTensorType { code, tensor }) => {
            assert_eq!(code, 9999);
            assert_eq!(tensor, "unk_tensor");
        }
        other => panic!("expected UnknownTensorType, got {other:?}"),
    }

    // Standard GGUF file with standard types is identified as standard GGUF
    let mut raw_std = Raw::new(1, 0);
    raw_std.tensor("tensor_std", &[32, 1], 0, 0); // F32
    raw_std.bytes.extend_from_slice(&[0u8; 256]);
    let file_std = GgufFile::parse_metadata_only(&raw_std.bytes, raw_std.bytes.len() as u64)
        .expect("parses standard metadata");
    assert!(file_std.is_standard_gguf());
    assert!(!file_std.is_native());

    // 2. R9v tensor type on file with no r9v.* metadata is classified as native, and both parse and parse_metadata_only fail with MissingKey
    let mut raw_r9v_no_meta = Raw::new(1, 1);
    raw_r9v_no_meta.kv(
        "general.alignment",
        KvType::U32.code(),
        &4096u32.to_le_bytes(),
    );
    raw_r9v_no_meta.tensor("r9v_tensor", &[256, 16], 1003, 0); // 1003 = I4K
    raw_r9v_no_meta.bytes.extend_from_slice(&[0u8; 4096]);
    assert!(matches!(
        GgufFile::parse_metadata_only(&raw_r9v_no_meta.bytes, raw_r9v_no_meta.bytes.len() as u64),
        Err(FormatError::Multiple { problems: ref errs }) if errs.iter().any(|e| matches!(e, FormatError::MissingKey { key } if key == "r9v.format_version"))
    ));
    assert!(matches!(
        GgufFile::parse(&raw_r9v_no_meta.bytes),
        Err(FormatError::Multiple { problems: ref errs }) if errs.iter().any(|e| matches!(e, FormatError::MissingKey { key } if key == "r9v.format_version"))
    ));

    // 3. Type/scheme agreement: if meta.scheme is absent, parse_r9v_meta populates it from r9v_type.scheme()
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_tensor(
            "weight",
            &[16, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            vec![0u8; 4096],
        )
        .unwrap();
    let bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&bytes).unwrap();
    let meta = parse_r9v_meta(&file).unwrap().unwrap();
    assert_eq!(meta.tensor("weight").unwrap().scheme, Some(SchemeId::I4K));

    // 4. Mismatched scheme declaration: the writer gate refuses to emit
    // it, and a foreign file with the same mismatch is rejected by
    // `GgufFile::parse` itself with SchemeMismatch.
    let mut writer_mismatch = GgufWriter::new().with_alignment(4096).unwrap();
    writer_mismatch
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer_mismatch
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer_mismatch
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer_mismatch
        .add_kv("r9v.tensor.weight.scheme", KvValue::Str("i8_r".to_owned()))
        .unwrap();
    writer_mismatch
        .add_tensor(
            "weight",
            &[16, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            vec![0u8; 4096],
        )
        .unwrap();
    assert!(matches!(
        writer_mismatch.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("declared scheme")
    ));
    let i4k_code = TensorType::R9v(R9vTensorType::new(SchemeId::I4K)).code();
    let scheme_key = "r9v.tensor.weight.scheme".to_owned();
    let raw_mismatch = native_raw_bytes(
        "L1",
        &[(&scheme_key, 8, enc_str_value("i8_r"))],
        "weight",
        &[16, 256],
        i4k_code,
        4096,
    );
    assert!(matches!(
        GgufFile::parse(&raw_mismatch),
        Err(FormatError::SchemeMismatch { .. })
    ));
}

#[test]
fn test_f32_native_enforcement() {
    // 1. Native F32 2D matrix is rejected at emit (must be 1D)
    let mut w_2d = GgufWriter::new().with_alignment(4096).unwrap();
    w_2d.add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_2d.add_kv("r9v.format_version", KvValue::U32(1)).unwrap();
    w_2d.add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    assert!(matches!(
        w_2d.add_tensor(
            "f32_2d",
            &[16, 256],
            TensorType::F32,
            vec![0u8; 16 * 256 * 4]
        ),
        Ok(())
    ));
    assert!(matches!(
        w_2d.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("must be 1D")
    ));

    // 2. Native F32 with non-vector role (e.g. matmul) is rejected at emit
    let mut w_role = GgufWriter::new().with_alignment(4096).unwrap();
    w_role
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_role
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_role
        .add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    w_role
        .add_kv(
            "r9v.tensor.v.roles",
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("matmul".to_owned())],
            },
        )
        .unwrap();
    w_role
        .add_tensor("v", &[256], TensorType::F32, vec![0u8; 4096])
        .unwrap();
    assert!(matches!(
        w_role.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("expected explicitly [vector]")
    ));

    // 2b. Native F32 with missing role is rejected at emit regardless of layout_id = "l0"
    let mut w_no_role = GgufWriter::new().with_alignment(4096).unwrap();
    w_no_role
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_no_role
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_no_role
        .add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    w_no_role
        .add_tensor("v", &[256], TensorType::F32, vec![0u8; 4096])
        .unwrap();
    assert!(matches!(
        w_no_role.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("missing required explicit role [vector]")
    ));

    // 3. Native F32 1D vector with role [vector] is accepted with L0 layout
    let mut w_ok = GgufWriter::new().with_alignment(4096).unwrap();
    w_ok.add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_ok.add_kv("r9v.format_version", KvValue::U32(1)).unwrap();
    w_ok.add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    w_ok.add_kv(
        "r9v.tensor.v.roles",
        KvValue::Array {
            elem: KvType::Str,
            items: vec![KvValue::Str("vector".to_owned())],
        },
    )
    .unwrap();
    w_ok.add_tensor("v", &[256], TensorType::F32, vec![0u8; 4096])
        .unwrap();
    let bytes = w_ok.emit().unwrap();
    let file = GgufFile::parse(&bytes).expect("parses cleanly");
    assert_eq!(file.tensor_nbytes(file.tensor("v").unwrap()).unwrap(), 4096);
}

#[test]
fn test_parse_metadata_only() {
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_tensor("w", &[17, 256], TensorType::F16, vec![0u8; 16384])
        .unwrap();
    let full_bytes = writer.emit().unwrap();

    let full_file = GgufFile::parse(&full_bytes).unwrap();
    let ti_end = full_file.ti_range().1 as usize;

    // Buffer truncated before data section (no payload bytes present)
    let truncated = &full_bytes[..ti_end];

    // Strict parse rejects truncated buffer because tensor range end exceeds file size
    assert!(matches!(
        GgufFile::parse(truncated),
        Err(FormatError::BadTensorRange { reason, .. }) if reason.contains("beyond file size")
    ));

    let full_size = full_bytes.len() as u64;

    // parse_metadata_only succeeds and never attempts to read data payload
    let meta_file =
        GgufFile::parse_metadata_only(truncated, full_size).expect("parse_metadata_only succeeds");
    assert_eq!(meta_file.tensors().len(), 1);
    assert_eq!(meta_file.tensor("w").unwrap().name, "w");
    assert_eq!(meta_file.file_size(), full_size);

    // parse_table_only alias works identically
    let table_file =
        GgufFile::parse_table_only(truncated, full_size).expect("parse_table_only succeeds");
    assert_eq!(table_file.tensors().len(), 1);
    assert_eq!(table_file.file_size(), full_size);

    // Full and metadata-only parse with same logical size produce identical file_fp
    let fp_full = full_file.file_fp(&full_bytes, 1).unwrap();
    let fp_meta = meta_file.file_fp(truncated, 1).unwrap();
    assert_eq!(fp_full, fp_meta);

    // Verify rejection of impossible sizes:
    // 1. File size smaller than data section start
    assert!(matches!(
        GgufFile::parse_metadata_only(truncated, 0),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("smaller than data section start")
    ));
    assert!(matches!(
        GgufFile::parse_metadata_only(truncated, meta_file.data_start() - 1),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("smaller than data section start")
    ));
    // 2. File size smaller than tensor data end
    assert!(matches!(
        GgufFile::parse_metadata_only(truncated, full_size - 1),
        Err(FormatError::BadTensorRange { reason, .. }) if reason.contains("beyond file size")
    ));
    // 3. Native file size larger than tensor data end (dead gap)
    assert!(matches!(
        GgufFile::parse_metadata_only(truncated, full_size + 4096),
        Err(FormatError::BadTensorRange { reason, .. }) if reason.contains("dead gap")
    ));

    // Reading payload from truncated buffer fails closed safely
    assert!(matches!(
        meta_file.tensor_bytes("w", truncated),
        Err(FormatError::BadTensorRange { .. })
    ));
}

#[test]
fn test_parse_rejects_forged_tensor_scheme_strings() {
    // A valid native file whose only flaw is a forged
    // `r9v.tensor.weight.scheme` value must be rejected by
    // `GgufFile::parse` itself, not just by the writer gate or
    // `parse_r9v_meta`. The forgeries patch the valid fixture's
    // bytes in place with same-length strings, so all offsets stay
    // intact and the scheme value is the only suspect.
    let scheme_key = "r9v.tensor.weight.scheme".to_owned();
    let valid = native_raw_bytes(
        "L1",
        &[(&scheme_key, 8, enc_str_value("i4_k"))],
        "weight",
        &[16, 256],
        TensorType::R9v(R9vTensorType::new(SchemeId::I4K)).code(),
        4096,
    );
    GgufFile::parse(&valid).expect("valid agreed scheme parses");
    assert_eq!(valid.windows(4).filter(|w| *w == b"i4_k").count(), 1);

    // Unknown same-length scheme string: closed-set rejection.
    let mut unknown = valid.clone();
    let pos = unknown
        .windows(4)
        .position(|w| w == b"i4_k")
        .expect("scheme value present");
    unknown[pos..pos + 4].copy_from_slice(b"zz_z");
    assert!(matches!(
        GgufFile::parse(&unknown),
        Err(FormatError::UnknownScheme { value }) if value == "zz_z"
    ));

    // Valid-but-disagreeing same-length scheme string: the type id
    // says `i4_k` while the metadata declares `i8_r`.
    let mut disagree = valid.clone();
    let pos = disagree
        .windows(4)
        .position(|w| w == b"i4_k")
        .expect("scheme value present");
    disagree[pos..pos + 4].copy_from_slice(b"i8_r");
    match GgufFile::parse(&disagree) {
        Err(FormatError::SchemeMismatch { expected, got, .. }) => {
            assert_eq!(expected, "i4_k");
            assert_eq!(got, "i8_r");
        }
        other => panic!("expected SchemeMismatch, got {other:?}"),
    }
}

#[test]
fn test_parse_rejects_impossible_zero_tensor_file() {
    // Native zero-tensor buffer with no data section: the 4096
    // alignment pushes the data-section start to 4096 while the
    // buffer ends after the tensor-info table. A full parse must
    // reject it even though no per-tensor range exists to blame.
    // (Standard zero-tensor files stay accepted: real vocab-only
    // GGUF ends the same way with no padding to align.)
    let mut raw = Raw::new(0, 3);
    raw.u32_kv("general.alignment", 4096);
    raw.u32_kv("r9v.format_version", 1);
    raw.str_kv("r9v.layout_id", "l1");
    let len = raw.bytes.len() as u64;
    assert!(len < 4096);
    match GgufFile::parse(&raw.bytes) {
        Err(FormatError::BadTensorRange {
            start, end, reason, ..
        }) => {
            assert_eq!(start, len);
            assert_eq!(end, 4096);
            assert!(reason.contains("beyond file size"), "{reason:?}");
        }
        other => panic!("expected BadTensorRange, got {other:?}"),
    }

    // The same bytes with an explicit logical size keep the
    // metadata-only `Malformed` semantics.
    assert!(matches!(
        GgufFile::parse_metadata_only(&raw.bytes, len),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("smaller than data section start")
    ));

    // Control: padding the data section into existence parses cleanly.
    raw.pad_to(4096);
    GgufFile::parse(&raw.bytes).expect("padded zero-tensor native file parses");
}

#[test]
fn test_file_fp_truncated() {
    let (bytes, _) = native_test_bytes();
    let file = GgufFile::parse(&bytes).unwrap();

    // Passing a buffer shorter than needed metadata bytes fails with FormatError::Truncated
    let cut = (file.ti_range().1 - 1) as usize;
    let truncated = &bytes[..cut];
    match file.file_fp(truncated, 1) {
        Err(FormatError::Truncated { what, need, .. }) => {
            assert_eq!(what, "file_fp metadata bytes");
            assert!(need >= 1);
        }
        other => panic!("expected FormatError::Truncated, got {other:?}"),
    }
}

#[test]
fn test_adversarial_explicit_regions_arity() {
    // 1. Explicit regions with 2 elements on retained F16 tensor fails at emit
    let mut w_f16_2 = GgufWriter::new().with_alignment(4096).unwrap();
    w_f16_2
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_f16_2
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_f16_2
        .add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    w_f16_2
        .add_kv(
            "r9v.tensor.w.regions",
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(0), KvValue::U64(4096)],
            },
        )
        .unwrap();
    w_f16_2
        .add_tensor("w", &[16, 1], TensorType::F16, vec![0u8; 4096])
        .unwrap();
    assert!(matches!(
        w_f16_2.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("has 2 item(s), expected 3")
    ));

    // 2. Explicit regions with 4 elements on retained F16 tensor fails at emit
    let mut w_f16_4 = GgufWriter::new().with_alignment(4096).unwrap();
    w_f16_4
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_f16_4
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_f16_4
        .add_kv("r9v.layout_id", KvValue::Str("l0".to_owned()))
        .unwrap();
    w_f16_4
        .add_kv(
            "r9v.tensor.w.regions",
            KvValue::Array {
                elem: KvType::U64,
                items: vec![
                    KvValue::U64(0),
                    KvValue::U64(4096),
                    KvValue::U64(4096),
                    KvValue::U64(4096),
                ],
            },
        )
        .unwrap();
    w_f16_4
        .add_tensor("w", &[16, 1], TensorType::F16, vec![0u8; 4096])
        .unwrap();
    assert!(matches!(
        w_f16_4.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("has 4 item(s), expected 3")
    ));

    // 3. Explicit regions with 2 elements on R9V tensor fails at emit
    let mut w_r9v_2 = GgufWriter::new().with_alignment(4096).unwrap();
    w_r9v_2
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_r9v_2
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_r9v_2
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    w_r9v_2
        .add_kv(
            "r9v.tensor.weight.regions",
            KvValue::Array {
                elem: KvType::U64,
                items: vec![KvValue::U64(0), KvValue::U64(4096)],
            },
        )
        .unwrap();
    w_r9v_2
        .add_tensor(
            "weight",
            &[16, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            vec![0u8; 4096],
        )
        .unwrap();
    assert!(matches!(
        w_r9v_2.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("has 2 item(s), expected 3")
    ));
}

#[test]
fn test_adversarial_writer_emit_validation() {
    // 1. F16 candidate-layout mismatch: role forces L0, but L1 payload length provided
    let mut writer_l0_mismatch = GgufWriter::new().with_alignment(4096).unwrap();
    writer_l0_mismatch
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer_l0_mismatch
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer_l0_mismatch
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer_l0_mismatch
        .add_kv(
            "r9v.tensor.w.roles",
            KvValue::Array {
                elem: KvType::Str,
                items: vec![KvValue::Str("embed".to_owned())],
            },
        )
        .unwrap();
    // [17, 256] with role "embed" resolves to L0 (value bytes = 17 * 256 * 2 = 8704, aligned to 4096 = 12288)
    // Passing 16384 (which is valid for L1 tile math: 2*16 tiles * 512 = 16384) must fail closed at emit
    writer_l0_mismatch
        .add_tensor("w", &[17, 256], TensorType::F16, vec![0u8; 16384])
        .unwrap();
    assert!(matches!(
        writer_l0_mismatch.emit(),
        Err(FormatError::LengthMismatch {
            what,
            expected: 12288,
            got: 16384,
        }) if what == "native tensor data"
    ));

    // 2. Oversized R9v payload fails closed at emit
    let mut w_over = GgufWriter::new().with_alignment(4096).unwrap();
    w_over
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_over
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_over
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    // [16, 256] I4K in L1 requires 4096 bytes. Passing 8192 bytes.
    w_over
        .add_tensor(
            "weight",
            &[16, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            vec![0u8; 8192],
        )
        .unwrap();
    assert!(matches!(
        w_over.emit(),
        Err(FormatError::LengthMismatch {
            what,
            expected: 4096,
            got: 8192,
        }) if what == "native tensor data"
    ));

    // 3. Undersized R9v payload fails closed at emit
    let mut w_under = GgufWriter::new().with_alignment(4096).unwrap();
    w_under
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    w_under
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    w_under
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    // [32, 256] I4K in L1 requires 8192 bytes. Passing 4096 bytes.
    w_under
        .add_tensor(
            "weight",
            &[32, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            vec![0u8; 4096],
        )
        .unwrap();
    assert!(matches!(
        w_under.emit(),
        Err(FormatError::LengthMismatch {
            what,
            expected: 8192,
            got: 4096,
        }) if what == "native tensor data"
    ));
}

#[test]
fn test_adversarial_writer_rejects_invalid_metadata() {
    // The writer resolves final metadata at emit: fixtures the parse
    // gate rejects must also fail here, so no invalid native file can
    // be produced (Spec 2 §4, §6; card A2.5).
    fn native_writer() -> GgufWriter {
        let mut w = GgufWriter::new().with_alignment(4096).unwrap();
        w.add_kv("general.alignment", KvValue::U32(4096)).unwrap();
        w.add_kv("r9v.format_version", KvValue::U32(1)).unwrap();
        w.add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
            .unwrap();
        w
    }
    fn i4k_tensor(w: &mut GgufWriter, name: &str) {
        w.add_tensor(
            name,
            &[16, 256],
            TensorType::R9v(R9vTensorType::new(SchemeId::I4K)),
            vec![0u8; 4096],
        )
        .unwrap();
    }

    // 1. Invalid sparse string.
    let mut w = native_writer();
    i4k_tensor(&mut w, "weight");
    w.add_kv("r9v.tensor.weight.sparse", KvValue::Str("dense".to_owned()))
        .unwrap();
    assert!(matches!(
        w.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("sparse value")
    ));

    // 2. Invalid roles combination [matmul, embed].
    let mut w = native_writer();
    i4k_tensor(&mut w, "weight");
    w.add_kv(
        "r9v.tensor.weight.roles",
        KvValue::Array {
            elem: KvType::Str,
            items: vec![
                KvValue::Str("matmul".to_owned()),
                KvValue::Str("embed".to_owned()),
            ],
        },
    )
    .unwrap();
    assert!(matches!(
        w.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("invalid roles combination")
    ));

    // 3. Forged explicit regions disagreeing with geometry.
    let mut w = native_writer();
    i4k_tensor(&mut w, "weight");
    w.add_kv(
        "r9v.tensor.weight.regions",
        KvValue::Array {
            elem: KvType::U64,
            items: vec![KvValue::U64(0), KvValue::U64(2048), KvValue::U64(8192)],
        },
    )
    .unwrap();
    assert!(matches!(
        w.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("do not match derived regions")
    ));

    // 4. Mismatched scheme declaration.
    let mut w = native_writer();
    i4k_tensor(&mut w, "weight");
    w.add_kv("r9v.tensor.weight.scheme", KvValue::Str("i8_r".to_owned()))
        .unwrap();
    assert!(matches!(
        w.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("declared scheme")
    ));

    // 5. Sparse flag on a 1D vector.
    let mut w = native_writer();
    w.add_tensor(
        "vec",
        &[256],
        TensorType::R9v(R9vTensorType::new(SchemeId::I8B128)),
        vec![0u8; 4096],
    )
    .unwrap();
    w.add_kv("r9v.tensor.vec.sparse", KvValue::Str("s24".to_owned()))
        .unwrap();
    assert!(matches!(
        w.emit(),
        Err(FormatError::Malformed { detail, .. }) if detail.contains("cannot be sparse")
    ));
}

#[test]
fn test_adversarial_native_sequencing_and_dead_gaps() {
    let mut writer = GgufWriter::new().with_alignment(4096).unwrap();
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .unwrap();
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .unwrap();
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_owned()))
        .unwrap();
    writer
        .add_tensor("t0", &[1, 64], TensorType::F16, vec![0u8; 4096])
        .unwrap();
    writer
        .add_tensor("t1", &[1, 64], TensorType::F16, vec![0u8; 4096])
        .unwrap();
    let valid_bytes = writer.emit().unwrap();
    let file = GgufFile::parse(&valid_bytes).expect("valid native file parses");
    assert_eq!(file.tensors().len(), 2);

    // 1. Trailing dead 4KiB gap at EOF
    let mut with_trailing_gap = valid_bytes.clone();
    with_trailing_gap.extend_from_slice(&[0u8; 4096]);
    assert!(matches!(
        GgufFile::parse(&with_trailing_gap),
        Err(FormatError::BadTensorRange { reason, .. }) if reason.contains("dead gap of 4096 bytes")
    ));

    // 2. Dead 4KiB gap between tensors (corrupt offset of tensor 1)
    let t1_pos = valid_bytes.windows(2).position(|w| w == b"t1").unwrap();
    let offset_pos = t1_pos + 2 + 4 + 16 + 4;
    let mut with_gap_between = valid_bytes.clone();
    let current_off = u64::from_le_bytes(
        with_gap_between[offset_pos..offset_pos + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(current_off, 4096);
    // Shift t1 offset forward by 4096 (leaving a 4096 dead gap between t0 and t1)
    with_gap_between[offset_pos..offset_pos + 8]
        .copy_from_slice(&(current_off + 4096).to_le_bytes());
    with_gap_between.extend_from_slice(&[0u8; 4096]);
    assert!(matches!(
        GgufFile::parse(&with_gap_between),
        Err(FormatError::BadTensorRange { reason, .. }) if reason.contains("dead gap of 4096 bytes")
    ));
}
