// SPDX-License-Identifier: Apache-2.0
//! Malformed metadata and templates fail closed (Spec 9 §12; card A2.9).
//!
//! Every case builds GGUF bytes in-memory with `r9v-format`'s writer
//! (deterministic, no fixture files) and asserts the exact fail-closed
//! error.

use r9v_format::container::{GgufFile, GgufWriter, KvType, KvValue};
use r9v_loader::{LoaderError, Tokenizer};

fn parse(bytes: &[u8]) -> GgufFile {
    GgufFile::parse_metadata_only(bytes, bytes.len() as u64 + 4096).unwrap()
}

fn str_array(items: &[&str]) -> KvValue {
    KvValue::Array {
        elem: KvType::Str,
        items: items
            .iter()
            .map(|s| KvValue::Str((*s).to_owned()))
            .collect(),
    }
}

fn base_bpe() -> GgufWriter {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.model", KvValue::Str("gpt2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.pre", KvValue::Str("gpt-2".to_owned()))
        .unwrap();
    w.add_kv(
        "tokenizer.ggml.tokens",
        str_array(&["a", "b", "<|endoftext|>"]),
    )
    .unwrap();
    w.add_kv(
        "tokenizer.ggml.token_type",
        KvValue::Array {
            elem: KvType::I32,
            items: vec![KvValue::I32(1), KvValue::I32(1), KvValue::I32(3)],
        },
    )
    .unwrap();
    w.add_kv("tokenizer.ggml.merges", str_array(&["a b"]))
        .unwrap();
    w.add_kv("tokenizer.ggml.bos_token_id", KvValue::U32(2))
        .unwrap();
    w.add_kv("tokenizer.ggml.eos_token_id", KvValue::U32(2))
        .unwrap();
    w
}

#[test]
fn missing_model_key_fails() {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.pre", KvValue::Str("gpt-2".to_owned()))
        .unwrap();
    let file = parse(&w.emit().unwrap());
    assert!(matches!(
        Tokenizer::from_gguf(&file),
        Err(LoaderError::TokenizerMeta { .. })
    ));
}

#[test]
fn unknown_model_fails_closed() {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.model", KvValue::Str("t5".to_owned()))
        .unwrap();
    let file = parse(&w.emit().unwrap());
    match Tokenizer::from_gguf(&file) {
        Err(LoaderError::UnsupportedTokenizer { model, .. }) => assert_eq!(model, "t5"),
        other => panic!("expected UnsupportedTokenizer, got {other:?}"),
    }
}

#[test]
fn unknown_bpe_pre_fails_closed() {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.model", KvValue::Str("gpt2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.pre", KvValue::Str("llama3".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.tokens", str_array(&["a"]))
        .unwrap();
    w.add_kv("tokenizer.ggml.merges", str_array(&[])).unwrap();
    let file = parse(&w.emit().unwrap());
    match Tokenizer::from_gguf(&file) {
        Err(LoaderError::UnsupportedPreTokenizer { pre, .. }) => assert_eq!(pre, "llama3"),
        other => panic!("expected UnsupportedPreTokenizer, got {other:?}"),
    }
}

#[test]
fn mistyped_tokens_array_collects_problems() {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.model", KvValue::Str("gpt2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.pre", KvValue::Str("gpt-2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.tokens", KvValue::U32(7)).unwrap();
    let file = parse(&w.emit().unwrap());
    match Tokenizer::from_gguf(&file) {
        Err(LoaderError::TokenizerMeta { details }) => {
            assert!(
                details.iter().any(|d| d.contains("tokenizer.ggml.tokens")),
                "{details:?}"
            );
        }
        other => panic!("expected TokenizerMeta, got {other:?}"),
    }
}

#[test]
fn duplicate_token_text_fails() {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.model", KvValue::Str("gpt2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.pre", KvValue::Str("gpt-2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.tokens", str_array(&["a", "a"]))
        .unwrap();
    w.add_kv("tokenizer.ggml.merges", str_array(&[])).unwrap();
    let file = parse(&w.emit().unwrap());
    assert!(matches!(
        Tokenizer::from_gguf(&file),
        Err(LoaderError::TokenizerMeta { .. })
    ));
}

#[test]
fn malformed_merge_fails() {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.model", KvValue::Str("gpt2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.pre", KvValue::Str("gpt-2".to_owned()))
        .unwrap();
    w.add_kv("tokenizer.ggml.tokens", str_array(&["a"]))
        .unwrap();
    w.add_kv("tokenizer.ggml.merges", str_array(&["no-space-here"]))
        .unwrap();
    let file = parse(&w.emit().unwrap());
    assert!(matches!(
        Tokenizer::from_gguf(&file),
        Err(LoaderError::TokenizerMeta { .. })
    ));
}

#[test]
fn short_scores_array_fails() {
    let mut w = GgufWriter::new();
    w.add_kv("tokenizer.ggml.model", KvValue::Str("llama".to_owned()))
        .unwrap();
    w.add_kv(
        "tokenizer.ggml.tokens",
        str_array(&["<unk>", "<s>", "</s>"]),
    )
    .unwrap();
    w.add_kv(
        "tokenizer.ggml.scores",
        KvValue::Array {
            elem: KvType::F32,
            items: vec![KvValue::F32(0.0)],
        },
    )
    .unwrap();
    let file = parse(&w.emit().unwrap());
    assert!(matches!(
        Tokenizer::from_gguf(&file),
        Err(LoaderError::TokenizerMeta { .. })
    ));
}

#[test]
fn valid_base_fixture_builds() {
    let file = parse(&base_bpe().emit().unwrap());
    let tok = Tokenizer::from_gguf(&file).unwrap();
    assert_eq!(tok.vocab_size(), 3);
    assert_eq!(tok.encode("ab", false, false).unwrap(), vec![0, 1]);
}

#[test]
fn decode_rejects_out_of_range_id() {
    let file = parse(&base_bpe().emit().unwrap());
    let tok = Tokenizer::from_gguf(&file).unwrap();
    assert!(matches!(
        tok.decode(&[99]),
        Err(LoaderError::TokenIdOutOfRange { id: 99, .. })
    ));
}
