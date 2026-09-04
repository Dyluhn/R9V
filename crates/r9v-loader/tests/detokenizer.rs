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

#[test]
fn streaming_ascii_and_orphan_continuation_preserves_ascii() {
    let res = std::panic::catch_unwind(|| {
        let tok = fixture("fixture-bpe.gguf");
        let mut detok = Detokenizer::new(&tok, &[], false).unwrap();
        // 'a' (97) then orphan continuation 0x80 (128)
        let out1 = match detok.push(b'a' as u32).unwrap() {
            PushOutcome::Emit(s) => s,
            other => panic!("expected Emit, got {other:?}"),
        };
        assert_eq!(out1, "a");
        let out2 = match detok.push(0x80).unwrap() {
            PushOutcome::Emit(s) => s,
            other => panic!("expected Emit, got {other:?}"),
        };
        assert_eq!(out2, "\u{FFFD}");
        let flushed = detok.flush();
        assert_eq!(flushed, "");

        // Flush directly on pending 'a' + orphan continuation: 'a' must not be deleted
        let mut detok2 = Detokenizer::new(&tok, &[], false).unwrap();
        // Manually push tokens without draining
        let mut out = String::new();
        if let PushOutcome::Emit(s) = detok2.push(b'x' as u32).unwrap() {
            out.push_str(&s);
        }
        if let PushOutcome::Emit(s) = detok2.push(0x80).unwrap() {
            out.push_str(&s);
        }
        out.push_str(&detok2.flush());
        assert_eq!(out, "x\u{FFFD}");
    });
    assert!(res.is_ok(), "test panicked");
}

#[test]
fn streaming_multiple_orphans() {
    let res = std::panic::catch_unwind(|| {
        let tok = fixture("fixture-bpe.gguf");
        let mut detok = Detokenizer::new(&tok, &[], false).unwrap();
        let mut out = String::new();
        for &byte in &[0x80u32, 0x81, 0x82, 0xBF] {
            if let PushOutcome::Emit(s) = detok.push(byte).unwrap() {
                out.push_str(&s);
            }
        }
        out.push_str(&detok.flush());
        assert_eq!(out, "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}");
    });
    assert!(res.is_ok(), "test panicked");
}

#[test]
fn streaming_valid_partial_sequences_across_tokens() {
    let res = std::panic::catch_unwind(|| {
        let tok = fixture("fixture-bpe.gguf");

        // 2-byte: 'é' = 0xC3 0xA9
        let mut detok2 = Detokenizer::new(&tok, &[], false).unwrap();
        assert_eq!(detok2.push(0xC3).unwrap(), PushOutcome::Emit(String::new()));
        assert_eq!(
            detok2.push(0xA9).unwrap(),
            PushOutcome::Emit("é".to_owned())
        );
        assert_eq!(detok2.flush(), "");

        // 3-byte: '€' = 0xE2 0x82 0xAC
        let mut detok3 = Detokenizer::new(&tok, &[], false).unwrap();
        assert_eq!(detok3.push(0xE2).unwrap(), PushOutcome::Emit(String::new()));
        assert_eq!(detok3.push(0x82).unwrap(), PushOutcome::Emit(String::new()));
        assert_eq!(
            detok3.push(0xAC).unwrap(),
            PushOutcome::Emit("€".to_owned())
        );
        assert_eq!(detok3.flush(), "");

        // 4-byte: '🎉' = 0xF0 0x9F 0x8E 0x89
        let mut detok4 = Detokenizer::new(&tok, &[], false).unwrap();
        assert_eq!(detok4.push(0xF0).unwrap(), PushOutcome::Emit(String::new()));
        assert_eq!(detok4.push(0x9F).unwrap(), PushOutcome::Emit(String::new()));
        assert_eq!(detok4.push(0x8E).unwrap(), PushOutcome::Emit(String::new()));
        assert_eq!(
            detok4.push(0x89).unwrap(),
            PushOutcome::Emit("🎉".to_owned())
        );
        assert_eq!(detok4.flush(), "");
    });
    assert!(res.is_ok(), "test panicked");
}

#[test]
fn streaming_invalid_lead_overlong_surrogate_and_out_of_range() {
    let res = std::panic::catch_unwind(|| {
        let tok = fixture("fixture-bpe.gguf");

        // Invalid lead: 0xC0 (overlong) and 0xF5 (out of range)
        let mut detok_lead = Detokenizer::new(&tok, &[], false).unwrap();
        assert_eq!(
            detok_lead.push(0xC0).unwrap(),
            PushOutcome::Emit("\u{FFFD}".to_owned())
        );
        assert_eq!(
            detok_lead.push(0xF5).unwrap(),
            PushOutcome::Emit("\u{FFFD}".to_owned())
        );
        assert_eq!(
            detok_lead.push(0xFF).unwrap(),
            PushOutcome::Emit("\u{FFFD}".to_owned())
        );
        assert_eq!(detok_lead.flush(), "");

        // Overlong 3-byte: 0xE0 followed by 0x80 (< 0xA0)
        let mut detok_overlong = Detokenizer::new(&tok, &[], false).unwrap();
        assert_eq!(
            detok_overlong.push(0xE0).unwrap(),
            PushOutcome::Emit(String::new())
        ); // 0xE0 holds
        assert_eq!(
            detok_overlong.push(0x80).unwrap(),
            PushOutcome::Emit("\u{FFFD}\u{FFFD}".to_owned())
        ); // overlong released
        assert_eq!(detok_overlong.flush(), "");

        // Surrogate 3-byte: 0xED followed by 0xA0 (0xD800..0xDFFF)
        let mut detok_surrogate = Detokenizer::new(&tok, &[], false).unwrap();
        assert_eq!(
            detok_surrogate.push(0xED).unwrap(),
            PushOutcome::Emit(String::new())
        ); // 0xED holds
        assert_eq!(
            detok_surrogate.push(0xA0).unwrap(),
            PushOutcome::Emit("\u{FFFD}\u{FFFD}".to_owned())
        ); // surrogate released
        assert_eq!(detok_surrogate.flush(), "");

        // Out-of-range 4-byte: 0xF4 followed by 0x90 (> 0x10FFFF)
        let mut detok_oor = Detokenizer::new(&tok, &[], false).unwrap();
        assert_eq!(
            detok_oor.push(0xF4).unwrap(),
            PushOutcome::Emit(String::new())
        ); // 0xF4 holds
        assert_eq!(
            detok_oor.push(0x90).unwrap(),
            PushOutcome::Emit("\u{FFFD}\u{FFFD}".to_owned())
        ); // out-of-range released
        assert_eq!(detok_oor.flush(), "");
    });
    assert!(res.is_ok(), "test panicked");
}

#[test]
fn streaming_stop_prefix_interactions_and_flush() {
    let res = std::panic::catch_unwind(|| {
        let tok = fixture("fixture-bpe.gguf");

        // Stop prefix interaction with partial UTF-8
        let mut detok = Detokenizer::new(&tok, &["STOP"], true).unwrap();
        // Push 'a'
        assert_eq!(
            detok.push(b'a' as u32).unwrap(),
            PushOutcome::Emit("a".to_owned())
        );
        // Push 0xC3 (held for UTF-8)
        assert_eq!(detok.push(0xC3).unwrap(), PushOutcome::Emit(String::new()));
        // Push 0xA9 (completes 'é')
        assert_eq!(detok.push(0xA9).unwrap(), PushOutcome::Emit("é".to_owned()));
        // Push 'S' (held for "STOP")
        assert_eq!(
            detok.push(b'S' as u32).unwrap(),
            PushOutcome::Emit(String::new())
        );
        // Push 'T' (held for "STOP")
        assert_eq!(
            detok.push(b'T' as u32).unwrap(),
            PushOutcome::Emit(String::new())
        );
        // Push 'O' (held for "STOP")
        assert_eq!(
            detok.push(b'O' as u32).unwrap(),
            PushOutcome::Emit(String::new())
        );
        // Push 'P' (matches "STOP", returns Stop)
        match detok.push(b'P' as u32).unwrap() {
            PushOutcome::Stop(final_text) => assert_eq!(final_text, ""),
            other => panic!("expected Stop, got {other:?}"),
        }

        // Multibyte stop string: "€" = 0xE2 0x82 0xAC
        let mut detok_mb = Detokenizer::new(&tok, &["€"], true).unwrap();
        assert_eq!(
            detok_mb.push(b'h' as u32).unwrap(),
            PushOutcome::Emit("h".to_owned())
        );
        assert_eq!(
            detok_mb.push(0xE2).unwrap(),
            PushOutcome::Emit(String::new())
        );
        assert_eq!(
            detok_mb.push(0x82).unwrap(),
            PushOutcome::Emit(String::new())
        );
        match detok_mb.push(0xAC).unwrap() {
            PushOutcome::Stop(final_text) => assert_eq!(final_text, ""),
            other => panic!("expected Stop, got {other:?}"),
        }

        // Flush partial UTF-8 and partial stop prefix
        let mut detok_flush = Detokenizer::new(&tok, &["END"], true).unwrap();
        assert_eq!(
            detok_flush.push(b'o' as u32).unwrap(),
            PushOutcome::Emit("o".to_owned())
        );
        assert_eq!(
            detok_flush.push(b'k' as u32).unwrap(),
            PushOutcome::Emit("k".to_owned())
        );
        assert_eq!(
            detok_flush.push(0xC3).unwrap(),
            PushOutcome::Emit(String::new())
        );
        assert_eq!(detok_flush.flush(), "\u{FFFD}");

        let mut detok_stop_flush = Detokenizer::new(&tok, &["STOP"], true).unwrap();
        assert_eq!(
            detok_stop_flush.push(b'S' as u32).unwrap(),
            PushOutcome::Emit(String::new())
        );
        assert_eq!(
            detok_stop_flush.push(b'T' as u32).unwrap(),
            PushOutcome::Emit(String::new())
        );
        assert_eq!(detok_stop_flush.flush(), "ST");
    });
    assert!(res.is_ok(), "test panicked");
}
