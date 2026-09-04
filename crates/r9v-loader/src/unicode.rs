// SPDX-License-Identifier: Apache-2.0
//! Unicode primitives for tokenization (Spec 9 §7; card A2.9).
//!
//! Mirrors the llama.cpp reference (`src/unicode.cpp`, `src/unicode.h` at
//! the commit pinned in [`crate::unicode_tables`]) so pre-tokenization,
//! BERT normalization, and GPT-2 byte coding agree bit-for-bit:
//! codepoint flags, case maps, the lossy single-codepoint NFD map, the
//! collapsed-text pre-tokenizer view, and GPT-2 `bytes↔unicode` coding.

use crate::unicode_tables::{FLAG_RANGES, LOWERCASE_MAP, NFD_RANGES, WHITESPACE_SET};

/// llama.cpp flag bits (`unicode.h`).
pub(crate) const FLAG_NUMBER: u16 = 0x2;
/// llama.cpp flag bits (`unicode.h`).
pub(crate) const FLAG_LETTER: u16 = 0x4;
/// llama.cpp flag bits (`unicode.h`).
pub(crate) const FLAG_MARK: u16 = 0x10;
/// llama.cpp flag bits (`unicode.h`).
pub(crate) const FLAG_PUNCTUATION: u16 = 0x20;
/// llama.cpp flag bits (`unicode.h`).
pub(crate) const FLAG_SYMBOL: u16 = 0x40;
/// llama.cpp flag bits (`unicode.h`).
pub(crate) const FLAG_CONTROL: u16 = 0x80;

/// Looks up one codepoint's flag bits by binary search over
/// `FLAG_RANGES` (mirrors `unicode_cpt_flags_from_cpt`).
pub(crate) fn flags_of(cpt: u32) -> u16 {
    // Out-of-table reads yield the reference `undef` (zero bits).
    if cpt >= 0x11_0000 {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = FLAG_RANGES.len();
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if FLAG_RANGES[mid].0 <= cpt {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    FLAG_RANGES[lo].1
}

/// True when the codepoint carries the llama.cpp whitespace flag, which
/// is exactly the 25-entry explicit set (mirrors `unicode_set_whitespace`).
pub(crate) fn is_whitespace(cpt: u32) -> bool {
    WHITESPACE_SET.binary_search(&cpt).is_ok()
}

/// Lowercase map lookup (mirrors `unicode_tolower`).
pub(crate) fn to_lower(cpt: u32) -> u32 {
    match LOWERCASE_MAP.binary_search_by_key(&cpt, |(from, _)| *from) {
        Ok(i) => LOWERCASE_MAP[i].1,
        Err(_) => cpt,
    }
}

/// Uppercase map lookup (mirrors `unicode_toupper`). No current caller;
/// kept beside the tables as reference (the table itself carries the
/// allow).
#[allow(dead_code)]
pub(crate) fn to_upper(cpt: u32) -> u32 {
    use crate::unicode_tables::UPPERCASE_MAP;
    match UPPERCASE_MAP.binary_search_by_key(&cpt, |(from, _)| *from) {
        Ok(i) => UPPERCASE_MAP[i].1,
        Err(_) => cpt,
    }
}

/// Lossy single-codepoint NFD base map (mirrors
/// `unicode_cpts_normalize_nfd`): precomposed letters map to their ASCII
/// base, everything else maps to itself. Combining marks are dropped by
/// the caller via [`is_accent_mark`].
pub(crate) fn nfd_base(cpt: u32) -> u32 {
    let mut lo = 0usize;
    let mut hi = NFD_RANGES.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (first, last, _) = NFD_RANGES[mid];
        if cpt < first {
            hi = mid;
        } else if cpt > last {
            lo = mid + 1;
        } else {
            return NFD_RANGES[mid].2;
        }
    }
    cpt
}

/// True for combining marks (`\p{M}`), dropped under `strip_accents`.
pub(crate) fn is_accent_mark(cpt: u32) -> bool {
    flags_of(cpt) & FLAG_MARK != 0
}

/// True for punctuation (`\p{P}`).
pub(crate) fn is_punctuation(cpt: u32) -> bool {
    flags_of(cpt) & FLAG_PUNCTUATION != 0
}

/// True for symbols (`\p{S}`), plus ASCII `< 0x7F` symbol handling done by
/// the caller to mirror the reference exactly.
pub(crate) fn is_symbol(cpt: u32) -> bool {
    flags_of(cpt) & FLAG_SYMBOL != 0
}

/// True for control/format/private-use/surrogate (`\p{C}`).
pub(crate) fn is_control(cpt: u32) -> bool {
    flags_of(cpt) & FLAG_CONTROL != 0
}

/// True for letters (`\p{L}`).
pub(crate) fn is_letter(cpt: u32) -> bool {
    flags_of(cpt) & FLAG_LETTER != 0
}

/// True for numbers (`\p{N}`).
pub(crate) fn is_number(cpt: u32) -> bool {
    flags_of(cpt) & FLAG_NUMBER != 0
}

/// CJK ranges from the reference WPM preprocessor (copied verbatim).
pub(crate) fn is_chinese_char(cpt: u32) -> bool {
    (0x04E00..=0x09FFF).contains(&cpt)
        || (0x03400..=0x04DBF).contains(&cpt)
        || (0x20000..=0x2A6DF).contains(&cpt)
        || (0x2A700..=0x2B73F).contains(&cpt)
        || (0x2B740..=0x2B81F).contains(&cpt)
        || (0x2B920..=0x2CEAF).contains(&cpt)
        || (0x0F900..=0x0FAFF).contains(&cpt)
        || (0x2F800..=0x2FA1F).contains(&cpt)
}

/// Decodes one UTF-8 sequence starting at `bytes[pos]`; returns the
/// codepoint and its byte length. Invalid sequences decode to U+FFFD
/// consuming one byte (mirrors the reference decoder's ASCII fallback).
pub(crate) fn decode_one(bytes: &[u8], pos: usize) -> (u32, usize) {
    let b0 = bytes[pos];
    if b0 < 0x80 {
        return (b0 as u32, 1);
    }
    let (min, len) = if b0 >= 0xF0 {
        (0x10000, 4)
    } else if b0 >= 0xE0 {
        (0x800, 3)
    } else if b0 >= 0xC0 {
        (0x80, 2)
    } else {
        return (0xFFFD, 1);
    };
    if bytes.len() - pos < len {
        return (0xFFFD, 1);
    }
    let mut cpt: u32 = (b0 & (0xFF >> (len + 1))) as u32;
    for i in 1..len {
        let b = bytes[i + pos];
        if b & 0xC0 != 0x80 {
            return (0xFFFD, 1);
        }
        cpt = (cpt << 6) | (b & 0x3F) as u32;
    }
    if cpt < min || cpt > 0x10FFFF || (0xD800..0xE000).contains(&cpt) {
        return (0xFFFD, 1);
    }
    (cpt, len)
}

/// Decodes a whole string into codepoints.
pub(crate) fn cpts_from_str(text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(text.len());
    let mut pos = 0;
    while pos < bytes.len() {
        let (cpt, len) = decode_one(bytes, pos);
        out.push(cpt);
        pos += len;
    }
    out
}

/// Encodes one codepoint as UTF-8 into `out`.
pub(crate) fn encode_utf8(cpt: u32, out: &mut Vec<u8>) {
    if cpt < 0x80 {
        out.push(cpt as u8);
    } else if cpt < 0x800 {
        out.push(0xC0 | (cpt >> 6) as u8);
        out.push(0x80 | (cpt & 0x3F) as u8);
    } else if cpt < 0x10000 {
        out.push(0xE0 | (cpt >> 12) as u8);
        out.push(0x80 | ((cpt >> 6) & 0x3F) as u8);
        out.push(0x80 | (cpt & 0x3F) as u8);
    } else {
        out.push(0xF0 | (cpt >> 18) as u8);
        out.push(0x80 | ((cpt >> 12) & 0x3F) as u8);
        out.push(0x80 | ((cpt >> 6) & 0x3F) as u8);
        out.push(0x80 | (cpt & 0x3F) as u8);
    }
}

/// GPT-2 byte→unicode map (mirrors `unicode_byte_to_utf8_map`): printable
/// ASCII plus `¡–¬`/`®–ÿ` map to themselves; the remaining 68 bytes map to
/// U+0100.. so every byte is representable as a single unicode char.
pub(crate) fn byte_to_unicode_char(byte: u8) -> char {
    let b = byte as u32;
    let cpt =
        if (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b) {
            b
        } else {
            let mut n = 0u32;
            for candidate in 0u32..=255 {
                let printable = (0x21..=0x7E).contains(&candidate)
                    || (0xA1..=0xAC).contains(&candidate)
                    || (0xAE..=0xFF).contains(&candidate);
                if !printable {
                    if candidate == b {
                        break;
                    }
                    n += 1;
                }
            }
            256 + n
        };
    char::from_u32(cpt).unwrap_or('\u{FFFD}')
}

/// Inverse of [`byte_to_unicode_char`]: maps one GPT-2 unicode char back to
/// its byte. Returns `None` for chars outside the 256-entry image.
pub(crate) fn unicode_char_to_byte(c: char) -> Option<u8> {
    let cpt = c as u32;
    if (0x21..=0x7E).contains(&cpt) || (0xA1..=0xAC).contains(&cpt) || (0xAE..=0xFF).contains(&cpt)
    {
        return u8::try_from(cpt).ok();
    }
    if (256..512).contains(&cpt) {
        let mut n = 256u32;
        for candidate in 0u32..=255 {
            let printable = (0x21..=0x7E).contains(&candidate)
                || (0xA1..=0xAC).contains(&candidate)
                || (0xAE..=0xFF).contains(&candidate);
            if !printable {
                if n == cpt {
                    return u8::try_from(candidate).ok();
                }
                n += 1;
            }
        }
    }
    None
}

/// Encodes raw bytes with the GPT-2 byte map (mirrors the reference
/// `unicode_byte_to_utf8` over each input byte).
pub(crate) fn byte_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        out.push(byte_to_unicode_char(b));
    }
    out
}

/// Decodes a GPT-2 byte-encoded piece back to raw bytes (mirrors
/// `llama_decode_text`): each unicode char maps back to its byte; chars
/// outside the image are passed through as UTF-8 (never fail: the piece
/// came from our own vocab).
pub(crate) fn byte_decode_piece(piece: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(piece.len());
    for c in piece.chars() {
        match unicode_char_to_byte(c) {
            Some(b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

/// WPM `clean_spaces` post-pass from the reference detokenizer: removes
/// the space before `? ! . ,` and collapses other runs per the two passes.
pub(crate) fn clean_spaces(text: &mut Vec<u8>) {
    // First pass: drop a space before ? ! . ,
    let mut first: Vec<u8> = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if i + 1 < text.len() && text[i] == b' ' && matches!(text[i + 1], b'?' | b'!' | b'.' | b',')
        {
            i += 1;
            continue;
        }
        first.push(text[i]);
        i += 1;
    }
    // Second pass (mirrors the reference): collapse whitespace runs to one
    // space except around CJK, where spaces are removed entirely.
    let mut second: Vec<u8> = Vec::with_capacity(first.len());
    let mut j = 0;
    while j < first.len() {
        if first[j] == b' ' {
            // Peek at neighboring codepoints.
            let prev_cpt = decode_prev(&first, j);
            let mut k = j;
            while k < first.len() && first[k] == b' ' {
                k += 1;
            }
            let next_cpt = decode_next(&first, k);
            if prev_cpt.map(is_chinese_char).unwrap_or(false)
                || next_cpt.map(is_chinese_char).unwrap_or(false)
            {
                j = k;
                continue;
            }
            second.push(b' ');
            j = k;
        } else {
            second.push(first[j]);
            j += 1;
        }
    }
    *text = second;
}

fn decode_prev(bytes: &[u8], pos: usize) -> Option<u32> {
    if pos == 0 {
        return None;
    }
    let mut start = pos - 1;
    while start > 0 && bytes[start] & 0xC0 == 0x80 {
        start -= 1;
    }
    let (cpt, _) = decode_one(bytes, start);
    Some(cpt)
}

fn decode_next(bytes: &[u8], pos: usize) -> Option<u32> {
    if pos >= bytes.len() {
        return None;
    }
    let (cpt, _) = decode_one(bytes, pos);
    Some(cpt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_round_trips_all_256_values() {
        for b in 0u8..=255 {
            let c = byte_to_unicode_char(b);
            assert_eq!(unicode_char_to_byte(c), Some(b), "byte {b}");
        }
    }

    #[test]
    fn flags_match_ascii_expectations() {
        assert!(is_letter('A' as u32));
        assert!(is_number('3' as u32));
        assert!(is_punctuation('!' as u32));
        assert!(is_symbol('$' as u32));
        assert!(is_whitespace(' ' as u32));
        assert!(is_control(0x00));
        assert!(!is_letter(' ' as u32));
    }

    #[test]
    fn flags_match_non_ascii_categories() {
        // 'é' is a letter, '٣' a number, U+00A0 whitespace.
        assert!(is_letter('é' as u32));
        assert!(is_number('٣' as u32));
        assert!(is_whitespace(0xA0));
    }
}
