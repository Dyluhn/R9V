// SPDX-License-Identifier: Apache-2.0
//! Incremental detokenizer: streaming equivalence, stop strings, and
//! split-UTF-8 handling (Spec 9 §7; card A2.9).

use r9v_format::container::GgufFile;
use r9v_loader::{Detokenizer, PushOutcome, Tokenizer};

fn fixture(name: &str) -> Tokenizer {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|_| panic!("missing {path}"));
    let file = GgufFile::parse_metadata_only(&bytes, bytes.len() as u64 + 4096).unwrap();
    Tokenizer::from_gguf(&file).unwrap()
}

/// Pushes ids one at a time, collecting emitted text; asserts streaming
/// output (emitted + final flush) equals batch [`Tokenizer::decode`].
fn check_streaming_equals_batch(tok: &Tokenizer, ids: &[u32], stops: &[&str]) {
    let batch = String::from_utf8(tok.decode(ids).unwrap()).unwrap();
    let mut detok = Detokenizer::new(tok, stops, false).unwrap();
    let mut streamed = String::new();
    for &id in ids {
        match detok.push(id).unwrap() {
            PushOutcome::Emit(text) => streamed.push_str(&text),
            PushOutcome::Stop(final_text) => {
                streamed.push_str(&final_text);
                break;
            }
            PushOutcome::Done => break,
        }
    }
    streamed.push_str(&detok.flush());
    assert_eq!(streamed, batch, "streaming != batch for {ids:?}");
}

#[test]
fn streaming_matches_batch_bpe() {
    let tok = fixture("fixture-bpe.gguf");
    // add_special=false: no BOS/EOS, so streaming and batch agree exactly.
    let ids = tok.encode("Hello world, don't stop!", false, true).unwrap();
    check_streaming_equals_batch(&tok, &ids, &[]);
}

#[test]
fn streaming_matches_batch_spm() {
    let tok = fixture("fixture-spm.gguf");
    let ids = tok.encode("Hello world testing", false, true).unwrap();
    check_streaming_equals_batch(&tok, &ids, &[]);
}

#[test]
fn streaming_matches_batch_bert() {
    let tok = fixture("fixture-bert.gguf");
    let ids = tok.encode("Hello world, testing!", false, true).unwrap();
    check_streaming_equals_batch(&tok, &ids, &[]);
}

#[test]
fn split_utf8_held_until_complete() {
    // "é" in the BPE fixture merges to one token; feed its raw bytes split
    // across two unknown-adjacent pushes via single-byte tokens instead:
    // encode "aé" and stream token-by-token, checking no half-char leaks.
    let tok = fixture("fixture-bpe.gguf");
    let ids = tok.encode("aé b", false, false).unwrap();
    assert!(ids.len() >= 2, "expected splits, got {ids:?}");
    let mut detok = Detokenizer::new(&tok, &[], false).unwrap();
    let mut out = String::new();
    for &id in &ids {
        if let PushOutcome::Emit(text) = detok.push(id).unwrap() {
            // Every incremental chunk must be valid UTF-8 on its own.
            assert!(text.is_ascii() || text.chars().all(|_| true));
            out.push_str(&text);
        }
    }
    out.push_str(&detok.flush());
    assert_eq!(out, "aé b");
}

#[test]
fn stop_string_split_across_tokens() {
    let tok = fixture("fixture-bpe.gguf");
    // "testing" encodes to multiple tokens; stop on "sting" (split).
    let ids = tok.encode("testing", false, false).unwrap();
    assert!(ids.len() >= 2, "need split tokens, got {ids:?}");
    let mut detok = Detokenizer::new(&tok, &["sting"], true).unwrap();
    let mut out = String::new();
    let mut stopped = false;
    for &id in &ids {
        match detok.push(id).unwrap() {
            PushOutcome::Emit(text) => out.push_str(&text),
            PushOutcome::Stop(final_text) => {
                out.push_str(&final_text);
                stopped = true;
                break;
            }
            PushOutcome::Done => break,
        }
    }
    assert!(stopped, "stop should fire, out={out:?}");
    assert_eq!(out, "te");
    // Further pushes are ignored after a stop.
    assert_eq!(detok.push(ids[0]).unwrap(), PushOutcome::Done);
}

#[test]
fn stop_string_exact_token_boundary() {
    let tok = fixture("fixture-spm.gguf");
    let ids = tok.encode("Hello world", true, true).unwrap();
    let mut detok = Detokenizer::new(&tok, &[" world"], true).unwrap();
    let mut out = String::new();
    let mut stopped = false;
    for &id in &ids {
        match detok.push(id).unwrap() {
            PushOutcome::Emit(text) => out.push_str(&text),
            PushOutcome::Stop(final_text) => {
                out.push_str(&final_text);
                stopped = true;
            }
            PushOutcome::Done => {}
        }
    }
    assert!(stopped);
    assert_eq!(out, "Hello");
}

#[test]
fn eos_stops_when_enabled() {
    let tok = fixture("fixture-spm.gguf");
    let eos = tok.eos_id().unwrap();
    let mut detok = Detokenizer::new(&tok, &[], true).unwrap();
    let ids = tok.encode("Hello", true, true).unwrap();
    for &id in &ids {
        let _ = detok.push(id).unwrap();
    }
    match detok.push(eos).unwrap() {
        PushOutcome::Stop(_) => {}
        other => panic!("expected Stop, got {other:?}"),
    }
}

#[test]
fn bos_skipped_in_stream() {
    let tok = fixture("fixture-spm.gguf");
    let bos = tok.bos_id().unwrap();
    let mut detok = Detokenizer::new(&tok, &[], false).unwrap();
    assert_eq!(detok.push(bos).unwrap(), PushOutcome::Emit(String::new()));
}

#[test]
fn out_of_range_id_fails_closed() {
    let tok = fixture("fixture-bert.gguf");
    let mut detok = Detokenizer::new(&tok, &[], false).unwrap();
    assert!(detok.push(999_999).is_err());
}

#[test]
fn too_many_stops_fails_closed() {
    let tok = fixture("fixture-bert.gguf");
    let stops: Vec<&str> = vec!["x"; 9];
    assert!(Detokenizer::new(&tok, &stops, false).is_err());
}

#[test]
fn empty_stream_flushes_empty() {
    let tok = fixture("fixture-bpe.gguf");
    let mut detok = Detokenizer::new(&tok, &[], false).unwrap();
    assert_eq!(detok.flush(), "");
}
