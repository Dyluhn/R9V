// SPDX-License-Identifier: Apache-2.0
//! Incremental detokenizer with stop-string handling (Spec 9 §7; card A2.9).
//!
//! [`Detokenizer`] consumes one token id at a time and emits display text
//! as soon as it is stable:
//!
//! * piece bytes come from [`Tokenizer::piece_bytes`](crate::Tokenizer)
//!   with `special=true`, so control/user-defined pieces pass through;
//! * a trailing incomplete UTF-8 sequence is held until completed (or the
//!   stream ends, at which point [`Detokenizer::flush`] emits it with
//!   U+FFFD substitution per codepoint, mirroring the reference decoder);
//! * when a stop string (or EOS id) completes, pushing returns
//!   [`PushOutcome::Stop`] and no further text is emitted;
//! * a trailing partial stop-string prefix is held back so a stop match is
//!   never emitted piecemeal.
//!
//! Allocation-conscious: one reusable byte buffer plus one small pending
//! buffer; no per-token heap churn beyond the emitted string.

use crate::error::LoaderError;
use crate::tokenizer::Tokenizer;
use crate::unicode::{clean_spaces, decode_one};

/// Maximum stop strings (matches the serving cap of 8, Spec 10 §3.2).
pub const MAX_STOP_STRINGS: usize = 8;
/// Maximum bytes of one stop string.
pub const MAX_STOP_LEN: usize = 512;

/// What one [`Detokenizer::push`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    /// Stable text ready to stream (may be empty while holding).
    Emit(String),
    /// A stop string or EOS id completed; carries the final text before
    /// the stop (the stop text itself is withheld).
    Stop(String),
    /// The stream already stopped; further pushes are ignored.
    Done,
}

/// Incremental detokenizer over one [`Tokenizer`] (Spec 9 §7).
pub struct Detokenizer<'a> {
    tok: &'a Tokenizer,
    stops: Vec<Vec<u8>>,
    stop_on_eos: bool,
    stopped: bool,
    /// Decoded-but-unemitted bytes (split UTF-8 tail / partial stop).
    pending: Vec<u8>,
    /// Whether the WordPiece `clean_spaces` pass applies at flush.
    clean_wpm: bool,
    first_piece: bool,
}

impl<'a> Detokenizer<'a> {
    /// Creates a detokenizer for `tok` with `stop_strings` (raw text) and
    /// EOS stopping when `stop_on_eos` (Spec 9 §7: `eos_ids` end the
    /// sequence; Spec 10 §3.2: `stop` strings end it).
    pub fn new(
        tok: &'a Tokenizer,
        stop_strings: &[&str],
        stop_on_eos: bool,
    ) -> Result<Self, LoaderError> {
        if stop_strings.len() > MAX_STOP_STRINGS {
            return Err(LoaderError::Limit {
                what: "stop strings",
                limit: MAX_STOP_STRINGS,
                got: stop_strings.len(),
            });
        }
        let mut stops = Vec::with_capacity(stop_strings.len());
        for s in stop_strings {
            if s.len() > MAX_STOP_LEN {
                return Err(LoaderError::Limit {
                    what: "stop string bytes",
                    limit: MAX_STOP_LEN,
                    got: s.len(),
                });
            }
            if !s.is_empty() {
                stops.push(s.as_bytes().to_vec());
            }
        }
        Ok(Self {
            tok,
            stops,
            stop_on_eos,
            stopped: false,
            pending: Vec::new(),
            clean_wpm: tok.kind() == crate::tokenizer::TokenizerKind::WordPiece,
            first_piece: true,
        })
    }

    /// Pushes one token id; returns stable text or a stop signal.
    pub fn push(&mut self, id: u32) -> Result<PushOutcome, LoaderError> {
        if self.stopped {
            return Ok(PushOutcome::Done);
        }
        if id as usize >= self.tok.vocab_size() {
            return Err(LoaderError::TokenIdOutOfRange {
                id,
                vocab_size: self.tok.vocab_size(),
            });
        }
        if self.stop_on_eos && self.tok.eos_ids().contains(&id) {
            self.stopped = true;
            let head = std::mem::take(&mut self.pending);
            return Ok(PushOutcome::Stop(sanitize(&head)));
        }
        // DECISION(A2.9): BOS is skipped by the streaming detokenizer (it
        // is a control prefix, never display text); rejected emitting its
        // piece, which would leak "<s>"-style markers into streams. The
        // batch `decode` path still returns it verbatim. Spec 9 §7 silent.
        if self.tok.bos_id() == Some(id) {
            return Ok(PushOutcome::Emit(String::new()));
        }
        let remove_space = self.first_piece && self.tok.add_space_prefix();
        self.first_piece = false;
        let piece = self.tok.piece_bytes(id, true, remove_space);
        self.pending.extend_from_slice(&piece);

        // Stop-string scan over the pending buffer.
        if let Some(pos) = self.find_stop() {
            self.stopped = true;
            let head = self.pending[..pos].to_vec();
            self.pending.clear();
            return Ok(PushOutcome::Stop(sanitize(&head)));
        }
        Ok(PushOutcome::Emit(self.take_stable()))
    }

    /// Emits remaining bytes: completes split UTF-8 with U+FFFD per
    /// truncated codepoint and applies the WPM clean pass.
    pub fn flush(&mut self) -> String {
        let mut tail = std::mem::take(&mut self.pending);
        complete_utf8(&mut tail);
        if self.clean_wpm {
            clean_spaces(&mut tail);
        }
        String::from_utf8(tail).unwrap_or_default()
    }

    /// Finds the earliest stop-string occurrence in `pending`.
    fn find_stop(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for stop in &self.stops {
            if let Some(pos) = find_bytes(&self.pending, stop) {
                best = Some(best.map_or(pos, |b: usize| b.min(pos)));
            }
        }
        best
    }

    /// Splits `pending` into stable output + heldback tail. The tail is
    /// the longest suffix that is either an incomplete UTF-8 sequence or
    /// a strict prefix of a stop string (a full stop is handled by
    /// `find_stop` before this runs).
    fn take_stable(&mut self) -> String {
        let hold_utf8 = incomplete_tail_len(&self.pending);
        let mut hold_stop = 0;
        for stop in &self.stops {
            hold_stop = hold_stop.max(partial_prefix_len(&self.pending, stop));
        }
        let hold = hold_utf8.max(hold_stop).min(self.pending.len());
        let split_at = self.pending.len() - hold;
        let stable = self.pending[..split_at].to_vec();
        self.pending.drain(..split_at);
        // The WPM `clean_spaces` pass applies at `flush` (the reference
        // runs it over the whole text at the end; already-emitted text
        // cannot be rewritten incrementally).
        let _ = self.clean_wpm;
        sanitize(&stable)
    }
}

/// Replaces every invalid UTF-8 sequence with U+FFFD (lossy but total;
/// pieces are valid by construction, so this only fires on adversarial
/// byte-token splits).
fn sanitize(bytes: &[u8]) -> String {
    let mut clean = Vec::with_capacity(bytes.len());
    let mut pos = 0;
    while pos < bytes.len() {
        let (cpt, len) = decode_one(bytes, pos);
        if cpt == 0xFFFD && bytes[pos] >= 0x80 {
            clean.extend_from_slice("�".as_bytes());
            pos += len;
        } else {
            clean.extend_from_slice(&bytes[pos..pos + len]);
            pos += len;
        }
    }
    String::from_utf8(clean).unwrap_or_default()
}

/// Finds the first occurrence of `needle` in `haystack`.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Length of a trailing incomplete UTF-8 sequence (0 when the buffer ends
/// on a codepoint boundary).
pub(crate) fn incomplete_tail_len(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    // Walk back over continuation bytes (at most 3).
    let mut start = bytes.len() - 1;
    let mut cont = 0;
    while start > 0 && bytes[start] & 0xC0 == 0x80 && cont < 3 {
        start -= 1;
        cont += 1;
    }
    let lead = bytes[start];
    let need = if lead < 0x80 {
        // `start` points at a continuation run with no lead in range:
        // treat the run itself as incomplete.
        return if cont > 0 { cont + 1 } else { 0 };
    } else if lead >= 0xF0 {
        4
    } else if lead >= 0xE0 {
        3
    } else if lead >= 0xC0 {
        2
    } else {
        // Stray continuation byte with no lead: hold 1 and let flush
        // replace it with U+FFFD.
        return 1;
    };
    let have = bytes.len() - start;
    if have < need {
        have
    } else {
        0
    }
}

/// Length of the longest strict-prefix-of-`stop` suffix of `bytes`
/// (0 when none; a full match is not a prefix).
fn partial_prefix_len(bytes: &[u8], stop: &[u8]) -> usize {
    let max = bytes.len().min(stop.len().saturating_sub(1));
    for len in (1..=max).rev() {
        if bytes[bytes.len() - len..] == stop[..len] {
            return len;
        }
    }
    0
}

/// Drops a trailing incomplete sequence, sanitizes the rest, and appends
/// one U+FFFD for the truncated tail (mirrors the reference decoder).
fn complete_utf8(tail: &mut Vec<u8>) {
    let hold = incomplete_tail_len(tail);
    tail.truncate(tail.len() - hold);
    let mut done = sanitize(tail);
    if hold > 0 {
        done.push('�');
    }
    *tail = done.into_bytes();
}
