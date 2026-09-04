// SPDX-License-Identifier: Apache-2.0
//! Tokenizers built from `tokenizer.ggml.*` metadata (Spec 9 §7; card A2.9).
//!
//! Three families, chosen by `tokenizer.ggml.model`, with behavior ported
//! from the pinned llama.cpp reference (`src/llama-vocab.cpp`,
//! `src/unicode.cpp` at the commit in [`crate::unicode_tables`]):
//!
//! | `tokenizer.ggml.model` | family | pre-tokenizer |
//! |------------------------|--------|----------------|
//! | `"llama"` | SentencePiece (unigram Viterbi) | none (space→▁, optional prefix space) |
//! | `"gpt2"` | byte-level BPE (merge-rank queue) | `tokenizer.ggml.pre == "gpt-2"` GPT-2 regex split |
//! | `"bert"` | WordPiece (greedy longest match) | BERT basic normalization |
//!
//! Special tokens, `add_bos`/`add_eos`/`add_sep` defaults, and per-key
//! overrides follow the reference exactly. Anything else fails closed with
//! [`LoaderError::UnsupportedTokenizer`] or
//! [`LoaderError::UnsupportedPreTokenizer`](crate::LoaderError); malformed
//! metadata collects every problem into
//! [`LoaderError::TokenizerMeta`](crate::LoaderError).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use r9v_format::container::{GgufFile, KvValue};

use crate::error::LoaderError;
use crate::pretokenizer::{split_gpt2, PreType};
use crate::unicode::{
    byte_decode_piece, byte_encode, clean_spaces, cpts_from_str, decode_one, encode_utf8,
    is_accent_mark, is_chinese_char, is_control, is_punctuation, is_symbol, is_whitespace,
    nfd_base, to_lower,
};

/// One BPE merge-candidate queue entry: ordering key, merged text at push
/// time (for stale-entry validation), and the symbol pair indices.
type BpeQueueEntry = (Reverse<(u32, usize, usize)>, Vec<u8>, usize, usize);

/// Maximum vocabulary entries (matches `r9v-models` `MAX_VOCAB_SIZE`).
pub const MAX_VOCAB: usize = 1 << 24;
/// Maximum BPE merge entries.
pub const MAX_MERGES: usize = 1 << 24;
/// Maximum input text bytes per `encode` call.
pub const MAX_ENCODE_BYTES: usize = 1 << 26;
/// Maximum tokens emitted per `encode` call.
pub const MAX_ENCODE_TOKENS: usize = 1 << 24;
/// Maximum pre-tokenizer words per `encode` call.
pub const MAX_PRE_WORDS: usize = 1 << 22;

/// Tokenizer family (Spec 9 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// Byte-level BPE (`tokenizer.ggml.model == "gpt2"`).
    Bpe,
    /// SentencePiece unigram (`tokenizer.ggml.model == "llama"`).
    SentencePiece,
    /// WordPiece (`tokenizer.ggml.model == "bert"`).
    WordPiece,
}

/// Token attribute bits (mirror `LLAMA_TOKEN_ATTR_*`).
pub(crate) const ATTR_UNKNOWN: u32 = 1 << 0;
/// Token attribute bits (mirror `LLAMA_TOKEN_ATTR_*`).
pub(crate) const ATTR_UNUSED: u32 = 1 << 1;
/// Token attribute bits (mirror `LLAMA_TOKEN_ATTR_*`).
pub(crate) const ATTR_NORMAL: u32 = 1 << 2;
/// Token attribute bits (mirror `LLAMA_TOKEN_ATTR_*`).
pub(crate) const ATTR_CONTROL: u32 = 1 << 3;
/// Token attribute bits (mirror `LLAMA_TOKEN_ATTR_*`).
pub(crate) const ATTR_USER_DEFINED: u32 = 1 << 4;
/// Token attribute bits (mirror `LLAMA_TOKEN_ATTR_*`).
pub(crate) const ATTR_BYTE: u32 = 1 << 5;

/// `tokenizer.ggml.token_type` values (mirror `LLAMA_TOKEN_TYPE_*`).
const TYPE_NORMAL: i32 = 1;
/// `tokenizer.ggml.token_type` values (mirror `LLAMA_TOKEN_TYPE_*`).
const TYPE_UNKNOWN: i32 = 2;
/// `tokenizer.ggml.token_type` values (mirror `LLAMA_TOKEN_TYPE_*`).
const TYPE_CONTROL: i32 = 3;
/// `tokenizer.ggml.token_type` values (mirror `LLAMA_TOKEN_TYPE_*`).
const TYPE_USER_DEFINED: i32 = 4;
/// `tokenizer.ggml.token_type` values (mirror `LLAMA_TOKEN_TYPE_*`).
const TYPE_UNUSED: i32 = 5;
/// `tokenizer.ggml.token_type` values (mirror `LLAMA_TOKEN_TYPE_*`).
const TYPE_BYTE: i32 = 6;

/// One vocabulary entry.
#[derive(Debug, Clone)]
struct TokenEntry {
    /// Raw token text (bytes as stored in metadata).
    text: Vec<u8>,
    /// Unigram score (SPM only; 0.0 otherwise).
    score: f32,
    /// Attribute bits.
    attr: u32,
}

/// Tokenizer built from `tokenizer.ggml.*` metadata (Spec 9 §7).
#[derive(Debug, Clone)]
pub struct Tokenizer {
    kind: TokenizerKind,
    tokens: Vec<TokenEntry>,
    text_to_id: HashMap<Vec<u8>, u32>,
    /// BPE merge ranks keyed by `(left, right)` piece pair.
    bpe_ranks: HashMap<(Vec<u8>, Vec<u8>), u32>,
    /// Special-token ids in longest-match-first partition order
    /// (control | user-defined | unknown, text length descending,
    /// id ascending for determinism).
    special_order: Vec<u32>,
    max_token_len: usize,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    sep_id: Option<u32>,
    pad_id: Option<u32>,
    cls_id: Option<u32>,
    mask_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_sep: bool,
    add_space_prefix: bool,
    escape_spaces: bool,
    lowercase: bool,
    strip_accents: bool,
    pre: PreType,
    /// `tokenizer.chat_template` verbatim, when present.
    chat_template: Option<String>,
}

/// One SPM merge symbol: a byte span with doubly-linked neighbors.
#[derive(Debug, Clone)]
struct SpmSymbol {
    /// Byte offset of the span start.
    start: usize,
    /// Byte length of the span (0 once merged away).
    len: usize,
    /// Previous live symbol.
    prev: Option<usize>,
    /// Next live symbol.
    next: Option<usize>,
}

/// Total-order score key for the SPM merge queue (`f32::total_cmp` keeps
/// ordering deterministic; scores are finite in practice).
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoreKey(f32);

impl Eq for ScoreKey {}

impl PartialOrd for ScoreKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Emits one merged SPM symbol (mirrors `resegment`): direct token hit,
/// else recorded split, else per-byte `<0xXX>` fallback.
fn spm_resegment(
    tok: &Tokenizer,
    symbols: &[SpmSymbol],
    index: usize,
    bytes: &[u8],
    rev_merge: &HashMap<Vec<u8>, (usize, usize)>,
    output: &mut Vec<u32>,
) -> Result<(), LoaderError> {
    let text = &bytes[symbols[index].start..symbols[index].start + symbols[index].len];
    if let Some(id) = tok.text_to_token(text) {
        output.push(id);
        return Ok(());
    }
    if let Some(&(first, second)) = rev_merge.get(text) {
        spm_resegment(tok, symbols, first, bytes, rev_merge, output)?;
        spm_resegment(tok, symbols, second, bytes, rev_merge, output)?;
        return Ok(());
    }
    for &b in text {
        output.push(tok.spm_byte_token(b)?);
    }
    Ok(())
}

impl Tokenizer {
    /// Tokenizer family.
    pub fn kind(&self) -> TokenizerKind {
        self.kind
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// BOS id handed to the scheduler via `ModelSpec` (Spec 9 §7).
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// EOS ids handed to the scheduler via `ModelSpec` (Spec 9 §7).
    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos_id.into_iter().collect()
    }

    /// Raw EOS id (singular; metadata carries one `eos_token_id`).
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// UNK id, when the family defines one.
    pub fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }

    /// SEP id (WordPiece), when defined.
    pub fn sep_id(&self) -> Option<u32> {
        self.sep_id
    }

    /// PAD id, when defined.
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }

    /// CLS id, when defined.
    pub fn cls_id(&self) -> Option<u32> {
        self.cls_id
    }

    /// MASK id, when defined.
    pub fn mask_id(&self) -> Option<u32> {
        self.mask_id
    }

    /// Whether `encode(..., add_special)` prepends BOS.
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Whether the SPM space prefix applies (first-piece `lstrip` source).
    pub(crate) fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }

    /// Whether `encode(..., add_special)` appends EOS.
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    /// Whether `encode(..., add_special)` appends SEP (WordPiece).
    pub fn add_sep(&self) -> bool {
        self.add_sep
    }

    /// Embedded `tokenizer.chat_template`, when present (Spec 10 §3.1).
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Looks up a token id by exact piece bytes.
    pub(crate) fn text_to_token(&self, piece: &[u8]) -> Option<u32> {
        self.text_to_id.get(piece).copied()
    }

    /// Builds a tokenizer from parsed GGUF metadata (Spec 9 §7).
    ///
    /// Reads only `tokenizer.ggml.*` / `tokenizer.chat_template` keys;
    /// every problem is collected before returning (CONVENTIONS.md §1.4).
    pub fn from_gguf(file: &GgufFile) -> Result<Self, LoaderError> {
        TokenizerBuilder::build(file)
    }

    /// Encodes `text` to token ids (mirrors `llama_vocab::tokenize` with
    /// `add_special` / `parse_special`).
    ///
    /// With `parse_special`, control/unknown/user-defined special pieces
    /// in the text split out to their ids; otherwise only user-defined
    /// pieces split (control/unknown are encoded as regular text).
    pub fn encode(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<u32>, LoaderError> {
        if text.len() > MAX_ENCODE_BYTES {
            return Err(LoaderError::Limit {
                what: "encode input bytes",
                limit: MAX_ENCODE_BYTES,
                got: text.len(),
            });
        }
        let mut output: Vec<u32> = Vec::new();
        // Affixes apply even to empty input (mirror the reference: the
        // fragment loop over an empty buffer emits nothing, but BOS/EOS
        // handling still runs, e.g. WPM encode('') == [BOS, SEP]).
        let fragments = self.partition(text, parse_special);
        match self.kind {
            TokenizerKind::SentencePiece => {
                self.encode_spm(&fragments, add_special, &mut output)?;
            }
            TokenizerKind::Bpe => {
                self.encode_bpe(&fragments, add_special, &mut output)?;
            }
            TokenizerKind::WordPiece => {
                self.encode_wpm(&fragments, add_special, &mut output)?;
            }
        }
        if output.len() > MAX_ENCODE_TOKENS {
            return Err(LoaderError::Limit {
                what: "encode output tokens",
                limit: MAX_ENCODE_TOKENS,
                got: output.len(),
            });
        }
        Ok(output)
    }

    /// Decodes one token to its piece bytes. With `special == false`,
    /// unknown/control pieces decode to empty (mirrors `token_to_piece`
    /// with `special=false`; the detokenizer passes `true`).
    /// `remove_space` strips one leading space (first-piece `lstrip`).
    pub(crate) fn piece_bytes(&self, id: u32, special: bool, remove_space: bool) -> Vec<u8> {
        let entry = match self.tokens.get(id as usize) {
            Some(e) => e,
            None => return Vec::new(),
        };
        if !special && entry.attr & (ATTR_UNKNOWN | ATTR_CONTROL) != 0 {
            return Vec::new();
        }
        let mut piece: Vec<u8> =
            if entry.attr & (ATTR_UNKNOWN | ATTR_CONTROL | ATTR_USER_DEFINED) != 0 {
                entry.text.clone()
            } else if entry.attr & ATTR_NORMAL != 0 {
                if self.kind == TokenizerKind::Bpe && !self.escape_spaces {
                    byte_decode_piece(piece_str(&entry.text))
                } else {
                    unescape_spaces(&entry.text)
                }
            } else if entry.attr & ATTR_BYTE != 0 {
                match parse_byte_token(&entry.text) {
                    Some(b) => vec![b],
                    None => Vec::new(),
                }
            } else {
                // UNUSED and undefined attrs decode to nothing, mirroring the
                // reference suppressing them like control tokens.
                Vec::new()
            };
        if remove_space && piece.first() == Some(&b' ') {
            piece.remove(0);
        }
        piece
    }

    /// Full decode of a token sequence (mirrors `llama_vocab::detokenize`
    /// with `remove_special=false` and no BOS/EOS stripping).
    pub fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, LoaderError> {
        for &id in ids {
            if id as usize >= self.tokens.len() {
                return Err(LoaderError::TokenIdOutOfRange {
                    id,
                    vocab_size: self.tokens.len(),
                });
            }
        }
        let mut out = Vec::new();
        // The reference strips one leading space from the first piece when
        // `add_space_prefix` (SPM); WPM/BPE never strip here.
        let mut remove_space = self.kind == TokenizerKind::SentencePiece && self.add_space_prefix;
        for &id in ids {
            out.extend_from_slice(&self.piece_bytes(id, true, remove_space));
            remove_space = false;
        }
        if self.kind == TokenizerKind::WordPiece {
            clean_spaces(&mut out);
        }
        Ok(out)
    }

    fn require_bos(&self) -> Result<u32, LoaderError> {
        self.bos_id.ok_or_else(|| LoaderError::TokenizerMeta {
            details: vec![
                "encode with add_special requires a BOS id but none is defined".to_owned(),
            ],
        })
    }

    fn require_eos(&self) -> Result<u32, LoaderError> {
        self.eos_id.ok_or_else(|| LoaderError::TokenizerMeta {
            details: vec![
                "encode with add_special requires an EOS id but none is defined".to_owned(),
            ],
        })
    }

    fn require_sep(&self) -> Result<u32, LoaderError> {
        self.sep_id.ok_or_else(|| LoaderError::TokenizerMeta {
            details: vec![
                "encode with add_special requires a SEP id but none is defined".to_owned(),
            ],
        })
    }

    /// Splits raw text around special-token occurrences (mirrors
    /// `tokenizer_st_partition`): longest special pieces first; with
    /// `parse_special == false`, control/unknown pieces stay as text.
    fn partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut frags: Vec<Fragment> = vec![Fragment::Raw(text.to_owned())];
        for &special_id in &self.special_order {
            let entry = &self.tokens[special_id as usize];
            if !parse_special && entry.attr & (ATTR_CONTROL | ATTR_UNKNOWN) != 0 {
                continue;
            }
            let needle = entry.text.clone();
            if needle.is_empty() {
                continue;
            }
            let mut next: Vec<Fragment> = Vec::with_capacity(frags.len() + 1);
            for frag in frags {
                match frag {
                    Fragment::Token(id) => next.push(Fragment::Token(id)),
                    Fragment::Raw(raw) => {
                        let bytes = raw.as_bytes();
                        let mut base = 0;
                        loop {
                            match find_bytes(&bytes[base..], &needle) {
                                None => {
                                    if base < bytes.len() {
                                        next.push(Fragment::Raw(raw[base..].to_owned()));
                                    }
                                    break;
                                }
                                Some(m) => {
                                    let abs_m = base + m;
                                    if abs_m > base {
                                        next.push(Fragment::Raw(raw[base..abs_m].to_owned()));
                                    }
                                    next.push(Fragment::Token(special_id));
                                    base = abs_m + needle.len();
                                    if base >= bytes.len() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            frags = next;
        }
        frags
    }

    /// SPM path (mirrors the `LLAMA_VOCAB_TYPE_SPM` branch).
    fn encode_spm(
        &self,
        fragments: &[Fragment],
        add_special: bool,
        output: &mut Vec<u32>,
    ) -> Result<(), LoaderError> {
        let mut is_prev_special = true;
        if add_special && self.add_bos {
            output.push(self.require_bos()?);
            is_prev_special = true;
        }
        for frag in fragments {
            match frag {
                Fragment::Token(id) => {
                    output.push(*id);
                    is_prev_special = true;
                }
                Fragment::Raw(raw) => {
                    let mut text = String::new();
                    if self.add_space_prefix && is_prev_special {
                        text.push(' ');
                    }
                    text.push_str(raw);
                    let escaped = escape_spaces_str(&text);
                    self.tokenize_spm(&escaped, output)?;
                    is_prev_special = false;
                }
            }
        }
        if add_special && self.add_eos {
            output.push(self.require_eos()?);
        }
        Ok(())
    }

    /// Score-ordered greedy merging (mirrors
    /// `llm_tokenizer_spm_session::tokenize`): the input is already
    /// space-prefixed and ▁-escaped by [`Self::encode_spm`].
    fn tokenize_spm(&self, text: &str, output: &mut Vec<u32>) -> Result<(), LoaderError> {
        let bytes = text.as_bytes();
        let mut symbols: Vec<SpmSymbol> = Vec::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let len = len_utf8_at(bytes, pos);
            let index = symbols.len();
            symbols.push(SpmSymbol {
                start: pos,
                len,
                prev: index.checked_sub(1),
                next: None,
            });
            if index > 0 {
                symbols[index - 1].next = Some(index);
            }
            pos += len;
        }
        // Work queue: highest score first, ties prefer smaller left
        // (mirrors `llm_bigram_spm::comparator`); the sequence number only
        // breaks remaining ties deterministically.
        // DECISION(A2.9): equal (score, left) pairs are processed FIFO
        // here; the reference heap order among such equivalents is
        // unspecified, so exact agreement there is not achievable by any
        // implementation. Real-vocab scores are effectively distinct, and
        // the parity loop validates the reference set. Rejected
        // replicating libstdc++ heap internals.
        let mut queue: BinaryHeap<(ScoreKey, std::cmp::Reverse<usize>, usize, usize, usize)> =
            BinaryHeap::new();
        let mut seq: usize = 0;
        // rev_merge: merged text -> forming pair (overwritten, like the
        // reference `std::map` assignment).
        let mut rev_merge: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();
        let try_add =
            |left: usize,
             right: usize,
             symbols: &[SpmSymbol],
             queue: &mut BinaryHeap<(ScoreKey, std::cmp::Reverse<usize>, usize, usize, usize)>,
             rev_merge: &mut HashMap<Vec<u8>, (usize, usize)>,
             seq: &mut usize| {
                let text =
                    bytes[symbols[left].start..symbols[right].start + symbols[right].len].to_vec();
                let Some(id) = self.text_to_token(&text) else {
                    return;
                };
                if id as usize >= self.tokens.len() {
                    return;
                }
                let score = self.tokens[id as usize].score;
                let size = text.len();
                *seq += 1;
                queue.push((ScoreKey(score), std::cmp::Reverse(left), size, left, right));
                rev_merge.insert(text, (left, right));
            };
        for i in 1..symbols.len() {
            try_add(i - 1, i, &symbols, &mut queue, &mut rev_merge, &mut seq);
        }
        while let Some((_, _, size, left, right)) = queue.pop() {
            let (llen, rlen) = (symbols[left].len, symbols[right].len);
            if llen == 0 || rlen == 0 {
                continue;
            }
            // Stale-size check (mirrors `left.n + right.n != bigram.size`).
            if llen + rlen != size {
                continue;
            }
            // Merge right into left.
            symbols[left].len += symbols[right].len;
            symbols[right].len = 0;
            let right_next = symbols[right].next;
            symbols[left].next = right_next;
            if let Some(rn) = right_next {
                symbols[rn].prev = Some(left);
            }
            if let Some(p) = symbols[left].prev {
                try_add(p, left, &symbols, &mut queue, &mut rev_merge, &mut seq);
            }
            if let Some(n) = symbols[left].next {
                try_add(left, n, &symbols, &mut queue, &mut rev_merge, &mut seq);
            }
        }
        // Emit surviving symbols in order via resegment.
        let mut cur = symbols.iter().position(|s| s.len > 0);
        while let Some(i) = cur {
            spm_resegment(self, &symbols, i, bytes, &rev_merge, output)?;
            cur = symbols[i].next;
        }
        Ok(())
    }

    /// SPM `<0xXX>` byte token with single-byte-string fallback (mirrors
    /// `byte_to_token`); fails closed when neither exists.
    fn spm_byte_token(&self, byte: u8) -> Result<u32, LoaderError> {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let name = [
            b'<',
            b'0',
            b'x',
            HEX[(byte >> 4) as usize],
            HEX[(byte & 15) as usize],
            b'>',
        ];
        if let Some(id) = self.text_to_id.get(&name[..]) {
            return Ok(*id);
        }
        if let Some(id) = self.text_to_id.get(&[byte][..]) {
            return Ok(*id);
        }
        Err(LoaderError::TokenizerMeta {
            details: vec![format!("SPM vocabulary has no byte token for 0x{byte:02X}")],
        })
    }

    /// BPE path (mirrors the `LLAMA_VOCAB_TYPE_BPE` branch).
    fn encode_bpe(
        &self,
        fragments: &[Fragment],
        add_special: bool,
        output: &mut Vec<u32>,
    ) -> Result<(), LoaderError> {
        if add_special && self.add_bos {
            output.push(self.require_bos()?);
        }
        for frag in fragments {
            match frag {
                Fragment::Token(id) => output.push(*id),
                Fragment::Raw(raw) => {
                    let text = if self.escape_spaces {
                        escape_spaces_str(raw)
                    } else {
                        raw.clone()
                    };
                    self.tokenize_bpe(&text, output)?;
                }
            }
        }
        if add_special && self.add_eos {
            output.push(self.require_eos()?);
        }
        Ok(())
    }

    /// Byte-level BPE over GPT-2 pre-tokenized words (mirrors
    /// `llm_tokenizer_bpe_session::tokenize` with the `(rank, left)`
    /// merge queue).
    fn tokenize_bpe(&self, text: &str, output: &mut Vec<u32>) -> Result<(), LoaderError> {
        // Only the reference-set pre-tokenizer reaches here; anything else
        // fails closed at load (`UnsupportedPreTokenizer`).
        let words = match self.pre {
            PreType::Gpt2 => split_gpt2(text),
            PreType::None => {
                return Err(LoaderError::UnsupportedPreTokenizer {
                    pre: String::new(),
                    model: "gpt2".to_owned(),
                    supported: ["gpt-2".to_owned()].to_vec(),
                });
            }
        };
        if words.len() > MAX_PRE_WORDS {
            return Err(LoaderError::Limit {
                what: "pre-tokenizer words",
                limit: MAX_PRE_WORDS,
                got: words.len(),
            });
        }
        for word in &words {
            // Byte-encode the word (mirrors `unicode_byte_encoding_process`).
            let encoded = byte_encode(word.as_bytes());
            // Symbols: one per UTF-8 char of the encoded word.
            let mut sym_text: Vec<&str> = Vec::new();
            let mut rest = encoded.as_str();
            while !rest.is_empty() {
                let len = len_utf8_at(rest.as_bytes(), 0);
                let (head, tail) = rest.split_at(len.min(rest.len()));
                sym_text.push(head);
                rest = tail;
            }
            let n = sym_text.len();
            if n == 0 {
                continue;
            }
            // Linked list over symbol indices.
            let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
            let mut next: Vec<Option<usize>> = (0..n)
                .map(|i| if i + 1 < n { Some(i + 1) } else { None })
                .collect();
            let mut alive = vec![true; n];
            // (Reverse(rank, left, seq)) queue with stale-entry skipping.
            // Each entry carries the concatenated text at push time so pops
            // can be validated (mirrors `bigram.text`).
            let mut queue: BinaryHeap<BpeQueueEntry> = BinaryHeap::new();
            // Merged text per surviving symbol.
            let mut merged: Vec<Vec<u8>> = sym_text.iter().map(|s| s.as_bytes().to_vec()).collect();
            let mut seq: usize = 0;
            let push_bigram = |left: usize,
                               right: usize,
                               merged: &[Vec<u8>],
                               queue: &mut BinaryHeap<BpeQueueEntry>,
                               seq: &mut usize| {
                let key = (merged[left].clone(), merged[right].clone());
                if let Some(&rank) = self.bpe_ranks.get(&key) {
                    *seq += 1;
                    let mut text = key.0;
                    text.extend_from_slice(&key.1);
                    queue.push((Reverse((rank, left, *seq)), text, left, right));
                }
            };
            for i in 0..n.saturating_sub(1) {
                push_bigram(i, i + 1, &merged, &mut queue, &mut seq);
            }
            while let Some((_, text, left, right)) = queue.pop() {
                if !alive[left] || !alive[right] {
                    continue;
                }
                if next[left] != Some(right) || prev[right] != Some(left) {
                    continue;
                }
                // Stale-text check (mirrors `left_token + right_token !=
                // bigram.text`).
                if merged[left].len() + merged[right].len() != text.len()
                    || !merged[left]
                        .iter()
                        .chain(merged[right].iter())
                        .zip(text.iter())
                        .all(|(a, b)| a == b)
                {
                    continue;
                }
                // Merge right into left.
                let right_text = std::mem::take(&mut merged[right]);
                merged[left].extend_from_slice(&right_text);
                alive[right] = false;
                let right_next = next[right];
                next[left] = right_next;
                if let Some(rn) = right_next {
                    prev[rn] = Some(left);
                }
                if let Some(p) = prev[left] {
                    if alive[p] {
                        push_bigram(p, left, &merged, &mut queue, &mut seq);
                    }
                }
                if let Some(rn) = next[left] {
                    if alive[rn] {
                        push_bigram(left, rn, &merged, &mut queue, &mut seq);
                    }
                }
            }
            // Emit surviving symbols in order.
            let mut idx = 0;
            while idx < n && !alive[idx] {
                idx += 1;
            }
            let mut cur = if idx < n { Some(idx) } else { None };
            while let Some(i) = cur {
                let piece = &merged[i];
                match self.text_to_token(piece) {
                    Some(id) => output.push(id),
                    None => {
                        // Byte fallback (mirrors the reference): each raw
                        // byte looked up directly; misses are dropped.
                        for &b in piece {
                            if let Some(id) = self.text_to_token(&[b]) {
                                output.push(id);
                            }
                        }
                    }
                }
                cur = next[i];
            }
        }
        Ok(())
    }

    /// WPM path (mirrors the `LLAMA_VOCAB_TYPE_WPM` branch).
    fn encode_wpm(
        &self,
        fragments: &[Fragment],
        add_special: bool,
        output: &mut Vec<u32>,
    ) -> Result<(), LoaderError> {
        if add_special {
            output.push(self.require_bos()?);
        }
        for frag in fragments {
            match frag {
                Fragment::Token(id) => output.push(*id),
                Fragment::Raw(raw) => {
                    self.tokenize_wpm(raw, output)?;
                }
            }
        }
        if add_special {
            output.push(self.require_sep()?);
        }
        Ok(())
    }

    fn require_unk(&self) -> Result<u32, LoaderError> {
        self.unk_id.ok_or_else(|| LoaderError::TokenizerMeta {
            details: vec!["tokenizer requires an UNK id but none is defined".to_owned()],
        })
    }

    /// BERT normalization + greedy longest-match (mirrors
    /// `llm_tokenizer_wpm_session::{preprocess, tokenize}`).
    fn tokenize_wpm(&self, text: &str, output: &mut Vec<u32>) -> Result<(), LoaderError> {
        let words = wpm_preprocess(text, self.lowercase, self.strip_accents);
        let unk = self.require_unk()?;
        for word in &words {
            if word.is_empty() {
                continue;
            }
            // Prepend phantom space ▁ (U+2581).
            let mut word1 = "▁".as_bytes().to_vec();
            word1.extend_from_slice(word.as_bytes());
            let before = output.len();
            let n = word1.len();
            let mut i = 0;
            while i < n {
                let mut matched = false;
                let max_len = (i + self.max_token_len + 1).min(n);
                let mut j = max_len;
                while j > i {
                    if self.text_to_id.contains_key(&word1[i..j]) {
                        output.push(self.text_to_id[&word1[i..j]]);
                        matched = true;
                        i = j - 1;
                        break;
                    }
                    j -= 1;
                }
                if !matched {
                    output.truncate(before);
                    break;
                }
                i += 1;
            }
            if output.len() == before {
                output.push(unk);
            }
        }
        Ok(())
    }
}

/// Text or token fragment from [`Tokenizer::partition`].
#[derive(Debug, Clone)]
enum Fragment {
    /// Raw text slice.
    Raw(String),
    /// A special token id.
    Token(u32),
}

/// Finds the first occurrence of `needle` in `haystack`.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Replaces ASCII space with U+2581 (mirrors `llama_escape_whitespace`).
fn escape_spaces_str(text: &str) -> String {
    if !text.contains(' ') {
        return text.to_owned();
    }
    text.replace(' ', "▁")
}

/// Replaces U+2581 with ASCII space (mirrors `llama_unescape_whitespace`).
fn unescape_spaces(text: &[u8]) -> Vec<u8> {
    let marker = "▁".as_bytes();
    if find_bytes(text, marker).is_none() {
        return text.to_vec();
    }
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(marker) {
            out.push(b' ');
            i += marker.len();
        } else {
            out.push(text[i]);
            i += 1;
        }
    }
    out
}

/// Interprets token text as UTF-8 lossily for byte-decode paths.
fn piece_str(text: &[u8]) -> &str {
    std::str::from_utf8(text).unwrap_or("")
}

/// Parses SPM `<0xXX>` byte tokens (mirrors `token_to_byte`).
fn parse_byte_token(text: &[u8]) -> Option<u8> {
    if text.len() == 6 && text.starts_with(b"<0x") && text.ends_with(b">") {
        let hex = std::str::from_utf8(&text[3..5]).ok()?;
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// UTF-8 length of the sequence starting at `bytes[pos]`; invalid lead
/// bytes and truncated tails consume 1 (mirrors the reference advancing
/// past undecodable bytes singly).
fn len_utf8_at(bytes: &[u8], pos: usize) -> usize {
    if pos >= bytes.len() {
        return 0;
    }
    let (_, len) = decode_one(bytes, pos);
    len
}

/// BERT basic normalization + whitespace/punctuation split (mirrors
/// `llm_tokenizer_wpm_session::preprocess`).
fn wpm_preprocess(text: &str, lowercase: bool, strip_accents: bool) -> Vec<String> {
    let mut cpts = cpts_from_str(text);
    if strip_accents {
        cpts = cpts.into_iter().map(nfd_base).collect();
    }
    let mut words: Vec<String> = vec![String::new()];
    for cpt in cpts {
        if is_whitespace(cpt) {
            if !words.last().map(|w| w.is_empty()).unwrap_or(true) {
                words.push(String::new());
            }
            continue;
        }
        if cpt == 0 || cpt == 0xFFFD || is_control(cpt) {
            continue;
        }
        if strip_accents && is_accent_mark(cpt) {
            continue;
        }
        let c = if lowercase { to_lower(cpt) } else { cpt };
        let mut buf = Vec::new();
        encode_utf8(c, &mut buf);
        let s = String::from_utf8(buf).unwrap_or_default();
        if is_punctuation(c) || (c < 0x7F && is_symbol(c)) || is_chinese_char(c) {
            if words.last().is_none_or(|w| !w.is_empty()) {
                words.push(String::new());
            }
            if let Some(last) = words.last_mut() {
                *last = s;
            }
            words.push(String::new());
        } else {
            // `words` always has at least one entry.
            words.last_mut().expect("words nonempty").push_str(&s);
        }
    }
    if words.last().map(|w| w.is_empty()).unwrap_or(false) {
        words.pop();
    }
    words
}

/// Metadata reader: builds a [`Tokenizer`] from `tokenizer.ggml.*` keys.
struct TokenizerBuilder<'a> {
    file: &'a GgufFile,
    problems: Vec<String>,
}

impl<'a> TokenizerBuilder<'a> {
    fn build(file: &'a GgufFile) -> Result<Tokenizer, LoaderError> {
        // Fail-closed family selection first: an unknown
        // `tokenizer.ggml.model` is a dedicated error, never a guess.
        let model = match file.kv("tokenizer.ggml.model") {
            Some(KvValue::Str(s)) => s.clone(),
            Some(other) => {
                return Err(LoaderError::TokenizerMeta {
                    details: vec![format!(
                        "tokenizer.ggml.model: expected STRING, found {}",
                        other.kv_type().name()
                    )],
                });
            }
            None => {
                return Err(LoaderError::TokenizerMeta {
                    details: vec!["tokenizer.ggml.model: missing required key".to_owned()],
                });
            }
        };
        if !matches!(model.as_str(), "llama" | "gpt2" | "bert") {
            return Err(LoaderError::UnsupportedTokenizer {
                model,
                supported: ["llama".to_owned(), "gpt2".to_owned(), "bert".to_owned()].to_vec(),
            });
        }
        let mut b = Self {
            file,
            problems: Vec::new(),
        };
        match b.build_inner(&model) {
            Err(fail_closed) => Err(fail_closed),
            Ok(None) => Err(LoaderError::TokenizerMeta {
                details: if b.problems.is_empty() {
                    vec!["tokenizer metadata missing".to_owned()]
                } else {
                    b.problems
                },
            }),
            Ok(Some(t)) => {
                if b.problems.is_empty() {
                    Ok(t)
                } else {
                    Err(LoaderError::TokenizerMeta {
                        details: b.problems,
                    })
                }
            }
        }
    }

    fn kv_str(&mut self, key: &str) -> Option<String> {
        match self.file.kv(key) {
            None => None,
            Some(KvValue::Str(s)) => Some(s.clone()),
            Some(other) => {
                self.problems.push(format!(
                    "{key}: expected STRING, found {}",
                    other.kv_type().name()
                ));
                None
            }
        }
    }

    fn kv_u32(&mut self, key: &str) -> Option<u32> {
        match self.file.kv(key) {
            None => None,
            Some(KvValue::U32(v)) => Some(*v),
            Some(KvValue::I32(v)) if *v >= 0 => Some(*v as u32),
            Some(other) => {
                self.problems.push(format!(
                    "{key}: expected UINT32, found {}",
                    other.kv_type().name()
                ));
                None
            }
        }
    }

    fn kv_bool(&mut self, key: &str) -> Option<bool> {
        match self.file.kv(key) {
            None => None,
            Some(KvValue::Bool(v)) => Some(*v),
            Some(other) => {
                self.problems.push(format!(
                    "{key}: expected BOOL, found {}",
                    other.kv_type().name()
                ));
                None
            }
        }
    }

    fn kv_str_array(&mut self, key: &str) -> Option<Vec<Vec<u8>>> {
        match self.file.kv(key) {
            None => None,
            Some(KvValue::Array { elem, items }) => {
                let kind = elem.name();
                if kind != "STRING" {
                    self.problems.push(format!(
                        "{key}: expected ARRAY of STRING, found ARRAY of {kind}"
                    ));
                    return None;
                }
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        KvValue::Str(s) => out.push(s.as_bytes().to_vec()),
                        other => {
                            self.problems.push(format!(
                                "{key}: expected STRING item, found {}",
                                other.kv_type().name()
                            ));
                            return None;
                        }
                    }
                }
                Some(out)
            }
            Some(other) => {
                self.problems.push(format!(
                    "{key}: expected ARRAY, found {}",
                    other.kv_type().name()
                ));
                None
            }
        }
    }

    fn kv_i32_array(&mut self, key: &str) -> Option<Vec<i32>> {
        match self.file.kv(key) {
            None => None,
            Some(KvValue::Array { elem, items }) => {
                let kind = elem.name();
                if kind != "INT32" {
                    self.problems.push(format!(
                        "{key}: expected ARRAY of INT32, found ARRAY of {kind}"
                    ));
                    return None;
                }
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        KvValue::I32(v) => out.push(*v),
                        other => {
                            self.problems.push(format!(
                                "{key}: expected INT32 item, found {}",
                                other.kv_type().name()
                            ));
                            return None;
                        }
                    }
                }
                Some(out)
            }
            Some(other) => {
                self.problems.push(format!(
                    "{key}: expected ARRAY, found {}",
                    other.kv_type().name()
                ));
                None
            }
        }
    }

    fn kv_f32_array(&mut self, key: &str) -> Option<Vec<f32>> {
        match self.file.kv(key) {
            None => None,
            Some(KvValue::Array { elem, items }) => {
                let kind = elem.name();
                if kind != "FLOAT32" {
                    self.problems.push(format!(
                        "{key}: expected ARRAY of FLOAT32, found ARRAY of {kind}"
                    ));
                    return None;
                }
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        KvValue::F32(v) => out.push(*v),
                        other => {
                            self.problems.push(format!(
                                "{key}: expected FLOAT32 item, found {}",
                                other.kv_type().name()
                            ));
                            return None;
                        }
                    }
                }
                Some(out)
            }
            Some(other) => {
                self.problems.push(format!(
                    "{key}: expected ARRAY, found {}",
                    other.kv_type().name()
                ));
                None
            }
        }
    }

    fn build_inner(&mut self, model: &str) -> Result<Option<Tokenizer>, LoaderError> {
        let kind = match model {
            "llama" => TokenizerKind::SentencePiece,
            "gpt2" => TokenizerKind::Bpe,
            "bert" => TokenizerKind::WordPiece,
            // Unreachable: `build` rejects anything else fail-closed.
            _ => {
                return Err(LoaderError::UnsupportedTokenizer {
                    model: model.to_owned(),
                    supported: ["llama".to_owned(), "gpt2".to_owned(), "bert".to_owned()].to_vec(),
                });
            }
        };

        // Pre-tokenizer (mirror the reference: only BPE models consult
        // `tokenizer.ggml.pre`; SPM/WPM ignore it entirely).
        // DECISION(A2.9): BPE supports only the reference-set "gpt-2"
        // rule; missing/unknown BPE pre fails closed instead of falling
        // back to the reference "default" regexes. Rejected implementing
        // "default" (a file without pre is degraded-quality per the
        // reference warning, and the card scopes pre rules to the
        // reference set). Spec 9 §7 is silent on fallbacks.
        let pre = match kind {
            TokenizerKind::Bpe => match self.kv_str("tokenizer.ggml.pre").as_deref() {
                Some("gpt-2") => PreType::Gpt2,
                other => {
                    if !self.problems.is_empty() {
                        // The pre key itself is mistyped; report that.
                        return Ok(None);
                    }
                    return Err(LoaderError::UnsupportedPreTokenizer {
                        pre: other.unwrap_or("").to_owned(),
                        model: model.to_owned(),
                        supported: ["gpt-2".to_owned()].to_vec(),
                    });
                }
            },
            TokenizerKind::SentencePiece | TokenizerKind::WordPiece => PreType::None,
        };

        // Vocabulary.
        let token_texts = match self.req_str_array("tokenizer.ggml.tokens") {
            Some(v) => v,
            None => return Ok(None),
        };
        if token_texts.len() > MAX_VOCAB {
            self.problems.push(format!(
                "tokenizer.ggml.tokens: length {} exceeds limit {MAX_VOCAB}",
                token_texts.len()
            ));
            return Ok(None);
        }
        if token_texts.is_empty() {
            self.problems
                .push("tokenizer.ggml.tokens: empty vocabulary".to_owned());
            return Ok(None);
        }
        let scores = self
            .kv_f32_array("tokenizer.ggml.scores")
            .unwrap_or_default();
        if !scores.is_empty() && scores.len() < token_texts.len() {
            self.problems.push(format!(
                "tokenizer.ggml.scores: length {} < tokens {}",
                scores.len(),
                token_texts.len()
            ));
            return Ok(None);
        }
        let types = self
            .kv_i32_array("tokenizer.ggml.token_type")
            .unwrap_or_default();
        if !types.is_empty() && types.len() < token_texts.len() {
            self.problems.push(format!(
                "tokenizer.ggml.token_type: length {} < tokens {}",
                types.len(),
                token_texts.len()
            ));
            return Ok(None);
        }

        let mut tokens: Vec<TokenEntry> = Vec::with_capacity(token_texts.len());
        let mut text_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(token_texts.len());
        let mut max_token_len = 0usize;
        for (i, text) in token_texts.iter().enumerate() {
            if text.is_empty() {
                self.problems
                    .push(format!("tokenizer.ggml.tokens[{i}]: empty token"));
                continue;
            }
            if text_to_id.contains_key(text) {
                self.problems.push(format!(
                    "tokenizer.ggml.tokens[{i}]: duplicate token text of length {}",
                    text.len()
                ));
                continue;
            }
            let attr = match types.get(i).copied().unwrap_or(TYPE_NORMAL) {
                TYPE_UNKNOWN => ATTR_UNKNOWN,
                TYPE_UNUSED => ATTR_UNUSED,
                TYPE_NORMAL => ATTR_NORMAL,
                TYPE_CONTROL => ATTR_CONTROL,
                TYPE_USER_DEFINED => ATTR_USER_DEFINED,
                TYPE_BYTE => ATTR_BYTE,
                _ => 0,
            };
            max_token_len = max_token_len.max(text.len());
            text_to_id.insert(text.clone(), i as u32);
            tokens.push(TokenEntry {
                text: text.clone(),
                score: scores.get(i).copied().unwrap_or(0.0),
                attr,
            });
        }
        if !self.problems.is_empty() {
            return Ok(None);
        }

        // BPE merges.
        let mut bpe_ranks: HashMap<(Vec<u8>, Vec<u8>), u32> = HashMap::new();
        if kind == TokenizerKind::Bpe {
            let merges = match self.req_str_array("tokenizer.ggml.merges") {
                Some(v) => v,
                None => return Ok(None),
            };
            if merges.len() > MAX_MERGES {
                self.problems.push(format!(
                    "tokenizer.ggml.merges: length {} exceeds limit {MAX_MERGES}",
                    merges.len()
                ));
                return Ok(None);
            }
            for (i, line) in merges.iter().enumerate() {
                // Split at the first space after byte 1 (mirrors
                // `word.find(' ', 1)`); malformed lines yield ("","").
                let (first, second) = match line.iter().skip(1).position(|&b| b == b' ') {
                    Some(rel) => {
                        let pos = rel + 1;
                        (line[..pos].to_vec(), line[pos + 1..].to_vec())
                    }
                    None => (Vec::new(), Vec::new()),
                };
                if first.is_empty() || second.is_empty() {
                    self.problems.push(format!(
                        "tokenizer.ggml.merges[{i}]: malformed merge line of length {}",
                        line.len()
                    ));
                    continue;
                }
                bpe_ranks.entry((first, second)).or_insert(i as u32);
            }
            if !self.problems.is_empty() {
                return Ok(None);
            }
        }

        // Special-token defaults per family (mirror the reference).
        let n = tokens.len() as u32;
        let in_range = |id: u32| -> Option<u32> { (id < n).then_some(id) };
        let (mut bos, mut eos, mut unk, mut sep, mut pad, mut mask) = match kind {
            TokenizerKind::SentencePiece => {
                (in_range(1), in_range(2), in_range(0), None, None, None)
            }
            TokenizerKind::Bpe => (in_range(11), in_range(11), None, None, None, None),
            TokenizerKind::WordPiece => (
                in_range(101),
                None,
                in_range(100),
                in_range(102),
                in_range(0),
                in_range(103),
            ),
        };
        // Per-key overrides (out-of-range values are ignored, mirroring
        // the reference warn-and-keep-default).
        for (key, slot) in [
            ("tokenizer.ggml.bos_token_id", &mut bos),
            ("tokenizer.ggml.eos_token_id", &mut eos),
            ("tokenizer.ggml.unknown_token_id", &mut unk),
            ("tokenizer.ggml.seperator_token_id", &mut sep),
            ("tokenizer.ggml.padding_token_id", &mut pad),
            ("tokenizer.ggml.mask_token_id", &mut mask),
        ] {
            if let Some(id) = self.kv_u32(key) {
                if id < n {
                    *slot = Some(id);
                }
            }
        }
        // `cls_token_id` has no loader-side default (mirror the reference).
        let cls = self
            .kv_u32("tokenizer.ggml.cls_token_id")
            .filter(|&id| id < n);

        // Flags with per-family defaults (mirror the reference).
        let (mut add_bos, mut add_eos, mut add_sep) = match kind {
            TokenizerKind::SentencePiece => (true, false, false),
            TokenizerKind::Bpe => (false, false, false),
            TokenizerKind::WordPiece => (true, false, true),
        };
        let mut add_space_prefix = kind == TokenizerKind::SentencePiece;
        let escape_spaces = kind == TokenizerKind::SentencePiece;
        if let Some(v) = self.kv_bool("tokenizer.ggml.add_bos_token") {
            add_bos = v;
        }
        if let Some(v) = self.kv_bool("tokenizer.ggml.add_eos_token") {
            add_eos = v;
        }
        if let Some(v) = self.kv_bool("tokenizer.ggml.add_sep_token") {
            add_sep = v;
        }
        if let Some(v) = self.kv_bool("tokenizer.ggml.add_space_prefix") {
            add_space_prefix = v;
        }
        // Validated for type errors but not stored: none of the three
        // reference families consult it (mirror the reference branches).
        let _ = self.kv_bool("tokenizer.ggml.remove_extra_whitespaces");
        // WordPiece normalizer options (default lowercase=true,
        // strip_accents follows lowercase unless overridden).
        let mut lowercase = false;
        let mut strip_accents = false;
        if kind == TokenizerKind::WordPiece {
            lowercase = true;
            if let Some(v) = self.kv_bool("tokenizer.ggml.normalizer.lowercase") {
                lowercase = v;
            }
            strip_accents = lowercase;
            if let Some(v) = self.kv_bool("tokenizer.ggml.normalizer.strip_accents") {
                strip_accents = v;
            }
        }
        if kind == TokenizerKind::Bpe && pre != PreType::Gpt2 {
            self.problems
                .push("tokenizer.ggml.pre: BPE requires \"gpt-2\"".to_owned());
            return Ok(None);
        }

        // Special-token partition order: control | user-defined | unknown,
        // longest text first, id ascending for determinism (mirrors the
        // reference sort by descending length).
        let mut special_order: Vec<u32> = (0..n)
            .filter(|&id| {
                tokens[id as usize].attr & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN) != 0
            })
            .collect();
        special_order.sort_by(|&a, &b| {
            tokens[b as usize]
                .text
                .len()
                .cmp(&tokens[a as usize].text.len())
                .then(a.cmp(&b))
        });

        let chat_template = self.kv_str("tokenizer.chat_template");

        Ok(Some(Tokenizer {
            kind,
            tokens,
            text_to_id,
            bpe_ranks,
            special_order,
            max_token_len,
            bos_id: bos,
            eos_id: eos,
            unk_id: unk,
            sep_id: sep,
            pad_id: pad,
            cls_id: cls,
            mask_id: mask,
            add_bos,
            add_eos,
            add_sep,
            add_space_prefix,
            escape_spaces,
            lowercase,
            strip_accents,
            pre,
            chat_template,
        }))
    }

    fn req_str_array(&mut self, key: &str) -> Option<Vec<Vec<u8>>> {
        match self.kv_str_array(key) {
            Some(v) => Some(v),
            None => {
                if self.file.kv(key).is_none() {
                    self.problems.push(format!("{key}: missing required key"));
                }
                None
            }
        }
    }
}
