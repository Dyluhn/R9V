// SPDX-License-Identifier: Apache-2.0
//! Pre-tokenization rules by `tokenizer.ggml.pre` (Spec 9 §7; card A2.9).
//!
//! Only the pre-tokenizers required by the card's reference set are
//! implemented: `gpt-2` for BPE (`unicode_regex_split_custom_gpt2` in the
//! reference). SentencePiece and WordPiece need no regex pre-splitting.
//! Any other `tokenizer.ggml.pre` value fails closed with
//! [`LoaderError::UnsupportedPreTokenizer`](crate::LoaderError).

use crate::unicode::{cpts_from_str, encode_utf8, flags_of, is_letter, is_number, is_whitespace};

/// Pre-tokenizer selected from `tokenizer.ggml.pre` (Spec 9 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreType {
    /// GPT-2 regex pre-splitting (BPE reference).
    Gpt2,
    /// No pre-splitting (SentencePiece / WordPiece reference).
    None,
}

/// Out-of-range codepoint marker (mirrors the reference `OUT_OF_RANGE`).
const OUT_OF_RANGE: u32 = 0xFFFF_FFFF;

/// Splits `text` into pre-token words per the GPT-2 system regex
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// via a direct port of the reference custom splitter (codepoints +
/// flags, no regex engine).
pub(crate) fn split_gpt2(text: &str) -> Vec<String> {
    let cpts = cpts_from_str(text);
    let flags: Vec<u16> = cpts.iter().map(|&c| flags_of(c)).collect();
    let end = cpts.len();

    // Local, bounds-safe views mirroring the reference closures.
    let get_cpt = |pos: usize| -> u32 {
        if pos < end {
            cpts[pos]
        } else {
            OUT_OF_RANGE
        }
    };
    let get_flags = |pos: usize| -> u16 {
        if pos < end {
            flags[pos]
        } else {
            0
        }
    };
    let at_letter = |pos: usize| -> bool { pos < end && is_letter(cpts[pos]) };
    let at_number = |pos: usize| -> bool { pos < end && is_number(cpts[pos]) };
    let is_ws = |pos: usize| -> bool { pos < end && is_whitespace(cpts[pos]) };
    let any_flag = |pos: usize| -> bool { get_flags(pos) != 0 };

    let mut words: Vec<String> = Vec::new();
    let mut token_start = 0usize;
    let mut add_token = |end_pos: usize, words: &mut Vec<String>| {
        if end_pos > token_start {
            let mut bytes = Vec::new();
            for &c in &cpts[token_start..end_pos] {
                encode_utf8(c, &mut bytes);
            }
            words.push(String::from_utf8(bytes).unwrap_or_default());
            token_start = end_pos;
        }
    };

    let mut pos = 0;
    while pos < end {
        let cpt = get_cpt(pos);

        // 's|'t|'re|'ve|'m|'ll|'d
        if cpt == '\'' as u32 && pos + 1 < end {
            let next = get_cpt(pos + 1);
            if next == 's' as u32 || next == 't' as u32 || next == 'm' as u32 || next == 'd' as u32
            {
                pos += 2;
                add_token(pos, &mut words);
                continue;
            }
            if pos + 2 < end {
                let next_next = get_cpt(pos + 2);
                if (next_next == 'e' as u32 && (next == 'r' as u32 || next == 'v' as u32))
                    || (next == 'l' as u32 && next_next == 'l' as u32)
                {
                    pos += 3;
                    add_token(pos, &mut words);
                    continue;
                }
            }
        }

        // Optional leading space attaches to the letter/number/other run.
        let space = if cpt == ' ' as u32 { 1 } else { 0 };
        let probe = pos + space;
        // <space>?\p{L}+
        if at_letter(probe) {
            pos = probe;
            while at_letter(pos) {
                pos += 1;
            }
            add_token(pos, &mut words);
            continue;
        }
        // <space>?\p{N}+
        if at_number(probe) {
            pos = probe;
            while at_number(pos) {
                pos += 1;
            }
            add_token(pos, &mut words);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+
        if !is_ws(probe) && !at_letter(probe) && !at_number(probe) && any_flag(probe) {
            pos = probe;
            while !is_ws(pos) && !at_letter(pos) && !at_number(pos) && any_flag(pos) {
                pos += 1;
            }
            add_token(pos, &mut words);
            continue;
        }

        let mut num_ws = 0;
        while is_ws(pos + num_ws) {
            num_ws += 1;
        }
        // \s+(?!\S): trailing run keeps its last char for the next token.
        if num_ws > 1 && get_cpt(pos + num_ws) != OUT_OF_RANGE {
            pos += num_ws - 1;
            add_token(pos, &mut words);
            continue;
        }
        // \s+
        if num_ws > 0 {
            pos += num_ws;
            add_token(pos, &mut words);
            continue;
        }

        // No match: single codepoint.
        pos += 1;
        add_token(pos, &mut words);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_contractions_and_words() {
        assert_eq!(split_gpt2("don't"), vec!["don", "'t"]);
        assert_eq!(split_gpt2(" hello world"), vec![" hello", " world"]);
    }

    #[test]
    fn gpt2_trailing_space_splits_last_char() {
        // `\s+(?!\S)` at end of input: "hello " -> ["hello", " "].
        assert_eq!(split_gpt2("hello "), vec!["hello", " "]);
    }

    #[test]
    fn gpt2_digits_group() {
        assert_eq!(split_gpt2("abc 123"), vec!["abc", " 123"]);
    }
}
