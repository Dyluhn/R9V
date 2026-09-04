// SPDX-License-Identifier: Apache-2.0
//! Corpus parity: `encode` matches the pinned llama.cpp oracle on every
//! reference tokenizer type (Spec 9 §7; card A2.9).
//!
//! Fixtures (`tests/fixtures/`): synthetic GGUFs built deterministically
//! by `gen_tokenizer_fixtures.py` (gguf-py, version in `meta.json`) with
//! path-covering vocabs; `golden-*.json` produced by `gen_goldens.sh`
//! from the pinned `llama-tokenize` oracle (commit recorded per file) over
//! `corpus.json` × {add_special, parse_special}.

use r9v_format::container::GgufFile;
use r9v_loader::Tokenizer;
use serde_json::Value;

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|_| panic!("missing fixture {path}"))
}

fn load_tokenizer(name: &str) -> Tokenizer {
    let bytes = fixture(name);
    let file =
        GgufFile::parse_metadata_only(&bytes, bytes.len() as u64 + 4096).expect("fixture parses");
    Tokenizer::from_gguf(&file).expect("fixture tokenizer builds")
}

fn golden_cases(name: &str) -> Vec<(String, bool, bool, Vec<u32>)> {
    let path = format!(
        "{}/tests/fixtures/golden-{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing {path}"));
    let golden: Value = serde_json::from_str(&text).expect("golden parses");
    let oracle = &golden["oracle"];
    assert_eq!(
        oracle["commit"].as_str().unwrap(),
        "dd1ea524333b1e697489067d7a4c39c60d32beee",
        "oracle commit pinned"
    );
    golden["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            (
                case["input"].as_str().unwrap().to_owned(),
                case["add_special"].as_bool().unwrap(),
                case["parse_special"].as_bool().unwrap(),
                case["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|id| id.as_u64().unwrap() as u32)
                    .collect(),
            )
        })
        .collect()
}

fn check_parity(fixture: &str, golden: &str) {
    let tok = load_tokenizer(fixture);
    let cases = golden_cases(golden);
    assert!(!cases.is_empty(), "golden has cases");
    let mut fails = 0;
    for (input, add_special, parse_special, expected) in &cases {
        match tok.encode(input, *add_special, *parse_special) {
            Ok(got) if &got == expected => {}
            Ok(got) => {
                fails += 1;
                eprintln!(
                    "MISMATCH input={input:?} add_special={add_special} parse_special={parse_special}\n  exp={expected:?}\n  got={got:?}"
                );
            }
            Err(err) => {
                fails += 1;
                eprintln!("ERROR input={input:?}: {err}");
            }
        }
    }
    assert_eq!(fails, 0, "{fails}/{} parity cases failed", cases.len());
    // Repeatability: identical output across runs (Spec 9 §7).
    for (input, add_special, parse_special, _) in cases.iter().take(8) {
        let first = tok.encode(input, *add_special, *parse_special).unwrap();
        let second = tok.encode(input, *add_special, *parse_special).unwrap();
        assert_eq!(first, second, "repeatability for {input:?}");
    }
}

#[test]
fn parity_bpe_gpt2() {
    check_parity("fixture-bpe.gguf", "bpe");
}

#[test]
fn parity_spm_llama() {
    check_parity("fixture-spm.gguf", "spm");
}

#[test]
fn parity_wpm_bert() {
    check_parity("fixture-bert.gguf", "bert");
}

#[test]
fn special_token_ids_match_metadata() {
    let bpe = load_tokenizer("fixture-bpe.gguf");
    assert_eq!(bpe.bos_id(), bpe.eos_id());
    assert!(bpe.bos_id().is_some());

    let spm = load_tokenizer("fixture-spm.gguf");
    assert_eq!(spm.bos_id(), Some(1));
    assert_eq!(spm.eos_ids(), vec![2]);
    assert_eq!(spm.unk_id(), Some(0));

    let bert = load_tokenizer("fixture-bert.gguf");
    assert_eq!(bert.bos_id(), Some(2));
    assert_eq!(bert.sep_id(), Some(3));
    assert_eq!(bert.pad_id(), Some(0));
    assert_eq!(bert.unk_id(), Some(1));
}

#[test]
fn chat_template_accessor() {
    let bpe = load_tokenizer("fixture-bpe.gguf");
    assert_eq!(
        bpe.chat_template(),
        Some("bpe says {{ messages[0]['content'] }}")
    );
    let spm = load_tokenizer("fixture-spm.gguf");
    assert_eq!(spm.chat_template(), None);
}
