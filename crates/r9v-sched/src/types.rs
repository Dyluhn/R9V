// SPDX-License-Identifier: Apache-2.0
//! Core request, sequence, step, and stop criteria data types (Spec 6 §2, §7).

use std::collections::BTreeSet;
use std::fmt::Debug;

use r9v_common::{ReqId, SeqId, StepId};
use r9v_ir::{SamplingParams, StepGraphKey};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{SchedError, SchedResult};
use crate::log::ScheduleRecord;

/// Fixed-capacity inline sequence container avoiding heap allocation on hot scheduling paths (Spec 6 §3.3).
pub struct InlineVec<T, const CAP: usize> {
    data: [Option<T>; CAP],
    len: usize,
}

impl<T, const CAP: usize> Default for InlineVec<T, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAP: usize> InlineVec<T, CAP> {
    /// Constructs a new empty inline vector.
    pub const fn new() -> Self {
        Self {
            data: [const { None }; CAP],
            len: 0,
        }
    }

    /// Number of elements stored in the inline vector.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Fixed maximum capacity of this inline vector.
    pub const fn capacity(&self) -> usize {
        CAP
    }

    /// Pushes a new element into the vector, returning an error if capacity is exceeded.
    pub fn push(&mut self, item: T) -> SchedResult<()> {
        if self.len < CAP {
            let slot = self
                .data
                .get_mut(self.len)
                .ok_or_else(|| SchedError::Internal("inline vec push index".to_owned()))?;
            *slot = Some(item);
            self.len += 1;
            Ok(())
        } else {
            Err(SchedError::overflow(
                "inline_vec_push",
                format!("capacity {CAP} exceeded"),
            ))
        }
    }

    /// Pops the last element from the vector if non-empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            self.data.get_mut(self.len)?.take()
        } else {
            None
        }
    }

    /// Clears all elements, resetting length to zero.
    pub fn clear(&mut self) {
        let n = self.len;
        for slot in self.data.iter_mut().take(n) {
            *slot = None;
        }
        self.len = 0;
    }

    /// Returns a reference to the element at the given index, or None if out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            self.data.get(index)?.as_ref()
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index, or None if out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            self.data.get_mut(index)?.as_mut()
        } else {
            None
        }
    }

    /// Returns a reference to the first element.
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns an iterator over references to active elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let n = self.len;
        self.data.iter().take(n).filter_map(|opt| opt.as_ref())
    }

    /// Returns an iterator over mutable references to active elements.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let n = self.len;
        self.data.iter_mut().take(n).filter_map(|opt| opt.as_mut())
    }
}

impl<T: Clone, const CAP: usize> Clone for InlineVec<T, CAP> {
    fn clone(&self) -> Self {
        let mut cloned = Self::new();
        for item in self.iter() {
            let _ = cloned.push(item.clone());
        }
        cloned
    }
}

impl<T: Copy, const CAP: usize> Copy for InlineVec<T, CAP> {}

impl<T: std::fmt::Debug, const CAP: usize> std::fmt::Debug for InlineVec<T, CAP> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq, const CAP: usize> PartialEq for InlineVec<T, CAP> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq, const CAP: usize> Eq for InlineVec<T, CAP> {}

impl<'a, T, const CAP: usize> IntoIterator for &'a InlineVec<T, CAP> {
    type Item = &'a T;
    type IntoIter = std::iter::FilterMap<
        std::iter::Take<std::slice::Iter<'a, Option<T>>>,
        fn(&'a Option<T>) -> Option<&'a T>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        let n = self.len;
        self.data.iter().take(n).filter_map(|opt| opt.as_ref())
    }
}

impl<T, const CAP: usize> IntoIterator for InlineVec<T, CAP> {
    type Item = T;
    type IntoIter = std::iter::Flatten<std::array::IntoIter<Option<T>, CAP>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().flatten()
    }
}

impl<T: Serialize, const CAP: usize> Serialize for InlineVec<T, CAP> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_seq(self.iter())
    }
}

impl<'de, T: Deserialize<'de>, const CAP: usize> Deserialize<'de> for InlineVec<T, CAP> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Vec::<T>::deserialize(deserializer)?;
        if raw.len() > CAP {
            return Err(serde::de::Error::invalid_length(
                raw.len(),
                &"over-capacity inline vector",
            ));
        }
        let mut res = Self::new();
        for item in raw {
            res.push(item)
                .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        }
        Ok(res)
    }
}

/// Trait for incremental detokenization during stop-string evaluation (Spec 6 §7).
///
/// Decodes token IDs into a UTF-8 string to evaluate stop strings against the tail.
pub trait Detokenizer: Send + Sync {
    /// Decodes the provided token ID into UTF-8 text and appends it to `output` for the given sequence,
    /// returning the exact number of bytes appended to `output` for this token (Spec 6 §7).
    ///
    /// Must reject invalid/unsupported byte sequences with an error.
    fn append_token(
        &mut self,
        seq_id: SeqId,
        token: u32,
        output: &mut String,
    ) -> SchedResult<usize>;

    /// Decodes and appends the text of `tokens` into the provided `output` string (Spec 6 §7).
    fn detokenize_to(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        output: &mut String,
    ) -> SchedResult<usize> {
        let mut total: usize = 0;
        for &t in tokens {
            total = total
                .checked_add(self.append_token(seq_id, t, output)?)
                .ok_or_else(|| SchedError::overflow("detokenize_to", "total bytes overflow"))?;
        }
        Ok(total)
    }

    /// Decodes the provided slice of token IDs into text (Spec 6 §7).
    fn detokenize(&mut self, seq_id: SeqId, tokens: &[u32]) -> SchedResult<String> {
        let mut s = String::new();
        self.detokenize_to(seq_id, tokens, &mut s)?;
        Ok(s)
    }

    /// Resets sequence-specific detokenizer state on sequence finish or cancellation (Spec 6 §7).
    fn reset(&mut self, _seq_id: SeqId) {}
}

// DECISION(A3.9): detokenization is abstracted through the Detokenizer trait with a default ByteDetokenizer for incremental stop-string evaluation; rejected hard-coding a tokenizer dependency because tokenizer implementations live in r9v-loader/A1.4 and Spec 6 §7 requires matching against the incrementally detokenized tail.
// DECISION(A3.9): ByteDetokenizer accepts only byte-domain token IDs (0..=255) forming valid UTF-8, buffering multi-byte sequences in fixed [u8; 4] storage with no heap allocation; rejected lossy fallback (from_utf8_lossy) and Latin-1 char pushes because both corrupt the byte stream and break incremental==batch equivalence. Stub-tier samplers must therefore emit byte-domain values.
// DECISION(A3.9): on the final permitted token, MaxTokens takes precedence over a simultaneous EOS or stop-string match (no trim); rejected EOS/stop-wins because the budget bound is the deterministic contract the caller configured and trimming at the boundary would hide it. Spec 6 §7 is silent on precedence.
/// Default deterministic detokenizer treating token IDs as byte values,
/// correctly buffering multi-byte UTF-8 sequences with fixed storage (Spec 6 §7).
#[derive(Debug, Clone, Default)]
pub struct ByteDetokenizer {
    active_seq: Option<SeqId>,
    buf: [u8; 4],
    buf_len: usize,
    expected_len: usize,
}

impl ByteDetokenizer {
    /// Constructs a new byte detokenizer with no active sequence (Spec 6 §7).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Detokenizer for ByteDetokenizer {
    fn append_token(
        &mut self,
        seq_id: SeqId,
        token: u32,
        output: &mut String,
    ) -> SchedResult<usize> {
        if token > 255 {
            return Err(SchedError::DetokenizeError {
                detail: format!("token id {token} exceeds byte range (0..=255)"),
            });
        }
        let b = (token & 0xFF) as u8;
        if self.active_seq != Some(seq_id) {
            self.active_seq = Some(seq_id);
            self.buf_len = 0;
            self.expected_len = 0;
        }

        if self.buf_len == 0 {
            if b <= 0x7F {
                output.push(b as char);
                Ok(1)
            } else if (0xC2..=0xDF).contains(&b) {
                *self
                    .buf
                    .first_mut()
                    .ok_or_else(|| SchedError::DetokenizeError {
                        detail: "detokenizer buffer unavailable".to_owned(),
                    })? = b;
                self.buf_len = 1;
                self.expected_len = 2;
                Ok(0)
            } else if (0xE0..=0xEF).contains(&b) {
                *self
                    .buf
                    .first_mut()
                    .ok_or_else(|| SchedError::DetokenizeError {
                        detail: "detokenizer buffer unavailable".to_owned(),
                    })? = b;
                self.buf_len = 1;
                self.expected_len = 3;
                Ok(0)
            } else if (0xF0..=0xF4).contains(&b) {
                *self
                    .buf
                    .first_mut()
                    .ok_or_else(|| SchedError::DetokenizeError {
                        detail: "detokenizer buffer unavailable".to_owned(),
                    })? = b;
                self.buf_len = 1;
                self.expected_len = 4;
                Ok(0)
            } else {
                Err(SchedError::DetokenizeError {
                    detail: format!("unsupported or invalid UTF-8 start byte: 0x{b:02X}"),
                })
            }
        } else if (0x80..=0xBF).contains(&b) {
            let pos = self.buf_len;
            *self
                .buf
                .get_mut(pos)
                .ok_or_else(|| SchedError::DetokenizeError {
                    detail: format!("detokenizer buffer overflow at position {pos}"),
                })? = b;
            self.buf_len += 1;
            if self.buf_len == self.expected_len {
                let total = self.buf_len;
                self.buf_len = 0;
                self.expected_len = 0;
                let pending = self
                    .buf
                    .get(..total)
                    .ok_or_else(|| SchedError::DetokenizeError {
                        detail: format!("detokenizer buffer slice 0..{total} unavailable"),
                    })?;
                match std::str::from_utf8(pending) {
                    Ok(valid_str) => {
                        output.push_str(valid_str);
                        Ok(total)
                    }
                    Err(e) => Err(SchedError::DetokenizeError {
                        detail: format!("invalid UTF-8 byte sequence: {e}"),
                    }),
                }
            } else {
                Ok(0)
            }
        } else {
            self.buf_len = 0;
            self.expected_len = 0;
            Err(SchedError::DetokenizeError {
                detail: format!("expected UTF-8 continuation byte (0x80..=0xBF), got 0x{b:02X}"),
            })
        }
    }

    fn reset(&mut self, seq_id: SeqId) {
        if self.active_seq == Some(seq_id) {
            self.buf_len = 0;
            self.expected_len = 0;
            self.active_seq = None;
        }
    }
}

/// Stop criteria configuring EOS tokens and stop strings (Spec 6 §2, §7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopCriteria {
    /// EOS and explicit stop token IDs (Spec 6 §7).
    pub eos_token_ids: BTreeSet<u32>,
    /// Stop strings to match against the incrementally detokenized tail (Spec 6 §7).
    pub stop_strings: Vec<String>,
    /// Maximum byte length of the detokenized tail retained for matching (Spec 6 §7).
    pub max_stop_len: usize,
}

impl Default for StopCriteria {
    fn default() -> Self {
        Self {
            eos_token_ids: BTreeSet::new(),
            stop_strings: Vec::new(),
            max_stop_len: 256,
        }
    }
}

impl StopCriteria {
    /// Constructs stop criteria with the given EOS tokens and stop strings (Spec 6 §7).
    pub fn new(eos_token_ids: impl IntoIterator<Item = u32>, stop_strings: Vec<String>) -> Self {
        let max_stop_len = stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);

        Self {
            eos_token_ids: eos_token_ids.into_iter().collect(),
            stop_strings,
            max_stop_len,
        }
    }

    /// Checks whether the given token ID is an EOS or stop token (Spec 6 §7).
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    /// Checks whether any configured stop string appears in the detokenized text (Spec 6 §7).
    ///
    /// If matched, returns `Some((match_start_index, matched_string_slice))`.
    pub fn check_stop_string<'a>(&'a self, text: &str) -> Option<(usize, &'a str)> {
        let mut earliest: Option<(usize, &'a str)> = None;
        for stop_str in &self.stop_strings {
            if stop_str.is_empty() {
                continue;
            }
            if let Some(idx) = text.find(stop_str.as_str()) {
                match earliest {
                    Some((earliest_idx, _)) if idx < earliest_idx => {
                        earliest = Some((idx, stop_str.as_str()));
                    }
                    None => {
                        earliest = Some((idx, stop_str.as_str()));
                    }
                    _ => {}
                }
            }
        }
        earliest
    }
}

/// Request submitted to the scheduler (Spec 6 §2).
///
/// Contains prompt tokens, generation bounds, sampling parameters, stop criteria,
/// and streaming flag.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// Opaque unique request identifier (Spec 6 §2, CONVENTIONS.md §3.1).
    pub id: ReqId,
    /// Prompt token sequence (Spec 6 §2).
    pub tokens: Vec<u32>,
    /// Sampling parameters (Spec 1 §4.F, Spec 6 §2).
    pub sampling: SamplingParams,
    /// Maximum number of tokens to generate (Spec 6 §2).
    pub max_tokens: u32,
    /// EOS tokens and stop strings (Spec 6 §2, §7).
    pub stop: StopCriteria,
    /// Whether output tokens are streamed per-step (Spec 6 §2, §3.3).
    pub stream: bool,
}

impl Request {
    /// Constructs and validates a new Request (Spec 6 §2, CONVENTIONS.md §1.4, §2.2).
    pub fn new(
        id: ReqId,
        tokens: Vec<u32>,
        sampling: SamplingParams,
        max_tokens: u32,
        stop: StopCriteria,
        stream: bool,
    ) -> SchedResult<Self> {
        let mut problems = Vec::new();
        if tokens.is_empty() {
            problems.push("prompt tokens must not be empty".to_owned());
        }
        if max_tokens == 0 {
            problems.push("max_tokens must be >= 1".to_owned());
        }
        if let Err(e) = sampling.validate() {
            match e {
                r9v_ir::IrError::Multiple { problems: errs } => {
                    for err in errs {
                        problems.push(format!("invalid sampling param: {err}"));
                    }
                }
                err => {
                    problems.push(format!("invalid sampling param: {err}"));
                }
            }
        }

        if !problems.is_empty() {
            return Err(SchedError::invalid_request(problems));
        }

        Ok(Self {
            id,
            tokens,
            sampling,
            max_tokens,
            stop,
            stream,
        })
    }
}

/// Execution phase of a sequence in the scheduler lifecycle (Spec 6 §2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequencePhase {
    /// Sequence is queued awaiting initial prefill admission (Spec 6 §2).
    Queued,
    /// Sequence is undergoing chunked prompt prefill (Spec 6 §2, §3.3).
    Prefilling {
        /// Number of prompt tokens already processed through the step graph (Spec 6 §2).
        done: u32,
    },
    /// Sequence prompt is fully ingested and generating tokens one step at a time (Spec 6 §2).
    Decoding,
    /// Sequence has finished generation (Spec 6 §2, §7).
    Finished(FinishReason),
}

/// Reason why a sequence finished generation (Spec 6 §7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FinishReason {
    /// Emitted an EOS or configured stop token ID (Spec 6 §7).
    Eos(u32),
    /// Reached maximum generated tokens limit (Spec 6 §7).
    MaxTokens,
    /// Matched a configured stop string in the detokenized tail (Spec 6 §7).
    StopString(String),
    /// Client or host explicitly cancelled the sequence (Spec 6 §7).
    #[default]
    Cancelled,
}

/// Sequence tracking state across scheduler steps (Spec 6 §2).
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    /// Associated user request (Spec 6 §2).
    pub req: Request,
    /// State manager sequence identifier (Spec 3 §3.3, Spec 6 §2).
    pub seq_id: SeqId,
    /// Current execution phase (Spec 6 §2).
    pub phase: SequencePhase,
    /// Verified tokens currently stored in sequence state (Spec 3 §3.3, Spec 6 §2).
    pub ctx_len: u32,
    /// Tokens generated by this sequence excluding the initial prompt (Spec 6 §2).
    pub generated: Vec<u32>,
    /// Exponential moving average of acceptance rate (Spec 6 §2, §4.2).
    pub accept_ema: f32,
    /// Deterministic monotonically increasing arrival order index (Spec 6 §1, §4.1).
    pub arrival_order: u64,
    /// Total steps spent waiting for admission (Spec 6 §4.1).
    pub wait_steps: u64,
    /// Accumulated resolved budget time while waiting for admission in milliseconds (Spec 6 §4.1).
    pub accumulated_wait_ms: f32,
    /// Incrementally detokenized generated text tail (Spec 6 §7).
    pub detokenized_tail: String,
    /// Byte offset of the start of `detokenized_tail` within the global sequence byte stream.
    pub tail_start_byte: usize,
    /// Recent generated token byte boundaries `(token_idx, start_byte, end_byte)` in the tail.
    pub token_byte_spans: Vec<(usize, usize, usize)>,
    /// First contributing token and start byte of a multi-byte code point still
    /// buffered inside the detokenizer, so a stop match beginning at that code
    /// point truncates from the first contributing token (Spec 6 §7).
    ///
    /// Fixed stack slot, never allocated; cleared when the code point completes
    /// or when tail state resets.
    pub pending_utf8_start: Option<(usize, usize)>,
}

impl Sequence {
    /// Constructs a new sequence in the `Queued` phase (Spec 6 §2).
    pub fn new(req: Request, seq_id: SeqId, arrival_order: u64) -> Self {
        let max_stop = req.stop.max_stop_len;
        let tail_cap = if max_stop > 0 {
            (max_stop
                .checked_mul(2)
                .unwrap_or(max_stop)
                .checked_add(128)
                .unwrap_or(256))
            .max(256)
        } else {
            0
        };
        let spans_cap = if max_stop > 0 {
            (max_stop
                .checked_mul(2)
                .unwrap_or(max_stop)
                .checked_add(128)
                .unwrap_or(256))
            .max(256)
        } else {
            0
        };
        let max_tok = (req.max_tokens as usize).max(32);
        Self {
            req,
            seq_id,
            phase: SequencePhase::Queued,
            ctx_len: 0,
            generated: Vec::with_capacity(max_tok),
            accept_ema: 1.0,
            arrival_order,
            wait_steps: 0,
            accumulated_wait_ms: 0.0,
            detokenized_tail: String::with_capacity(tail_cap),
            tail_start_byte: 0,
            token_byte_spans: Vec::with_capacity(spans_cap),
            pending_utf8_start: None,
        }
    }

    /// Returns `true` if the sequence has completed execution (Spec 6 §2).
    pub fn is_finished(&self) -> bool {
        matches!(self.phase, SequencePhase::Finished(_))
    }

    /// Returns the prompt length in tokens.
    pub fn prompt_len(&self) -> u32 {
        self.req.tokens.len() as u32
    }

    /// Returns remaining prompt tokens to be prefilled (Spec 6 §4.1).
    pub fn remaining_prompt(&self) -> u32 {
        match self.phase {
            SequencePhase::Queued => self.prompt_len(),
            SequencePhase::Prefilling { done } => self.prompt_len().saturating_sub(done),
            _ => 0,
        }
    }

    /// Appends a newly generated token, incrementally evaluating stop criteria with bounded tail (Spec 6 §7).
    ///
    /// Finish precedence is deterministic and uniform: a token filling the final
    /// budget slot reports [`FinishReason::MaxTokens`] with no trim, whether or not
    /// it is also an EOS token or completes a stop string. Below the budget, EOS
    /// wins over stop strings. There is no accidental bypass in either direction.
    ///
    /// Returns `(finish_reason, is_trimmed)` where `is_trimmed` indicates whether the newly generated
    /// token was excluded from output due to a stop-string match.
    pub fn append_generated_token(
        &mut self,
        token: u32,
        detok: &mut dyn Detokenizer,
    ) -> SchedResult<(Option<FinishReason>, bool)> {
        // DECISION(A3.9): MaxTokens takes precedence over a simultaneous EOS or stop-string
        // match on the final permitted token (no trim); rejected EOS/stop-wins because the
        // budget bound is the deterministic contract the caller configured and trimming or
        // re-labeling at the boundary would hide it. Applied uniformly to EOS and stop
        // strings alike. Spec 6 §7 is silent on precedence.
        let fills_budget = self.generated.len().saturating_add(1) >= self.req.max_tokens as usize;
        let is_eos = self.req.stop.is_eos(token);

        // 1. Incrementally detokenize just this token into bounded tail buffer.
        // Detokenization still runs on the final token so invalid byte-domain values
        // are rejected rather than silently accepted at the bound.
        let token_idx = self.generated.len();
        let tail_len_before = self.detokenized_tail.len();
        let token_start_byte = self
            .tail_start_byte
            .checked_add(tail_len_before)
            .ok_or_else(|| SchedError::overflow("tail_bytes", "tail byte offset overflow"))?;

        let appended_bytes = detok.append_token(self.seq_id, token, &mut self.detokenized_tail)?;
        let token_end_byte = token_start_byte
            .checked_add(appended_bytes)
            .ok_or_else(|| SchedError::overflow("tail_bytes", "token end byte overflow"))?;

        self.generated.push(token);
        if appended_bytes == 0 {
            // Byte-domain token buffered as the start or middle of a multi-byte code
            // point: remember the first contributing token so a later stop match
            // truncates from here, not from the final continuation token.
            if self.pending_utf8_start.is_none() {
                self.pending_utf8_start = Some((token_idx, token_start_byte));
            }
        } else if let Some((pending_idx, pending_byte)) = self.pending_utf8_start.take() {
            // Code point completed: one span from the first contributing token.
            self.token_byte_spans
                .push((pending_idx, pending_byte, token_end_byte));
        } else {
            self.token_byte_spans
                .push((token_idx, token_start_byte, token_end_byte));
        }

        // 2. Final budget slot wins over EOS and stop strings alike: no trim (Spec 6 §7).
        if fills_budget {
            return Ok((Some(FinishReason::MaxTokens), false));
        }

        // 3. Below the budget, EOS wins over stop strings.
        if is_eos {
            return Ok((Some(FinishReason::Eos(token)), false));
        }

        // 4. Check stop strings against bounded detokenized tail (Spec 6 §7)
        if !self.req.stop.stop_strings.is_empty() {
            if let Some((match_offset, matched_str)) =
                self.req.stop.check_stop_string(&self.detokenized_tail)
            {
                let absolute_match_byte = self
                    .tail_start_byte
                    .checked_add(match_offset)
                    .ok_or_else(|| SchedError::overflow("match_byte", "match byte overflow"))?;

                // Map byte match to token boundary: find first token whose end_byte > absolute_match_byte
                let cut_token_idx = self
                    .token_byte_spans
                    .iter()
                    .find(|(_, _, end)| *end > absolute_match_byte)
                    .map(|(idx, _, _)| *idx)
                    .unwrap_or(token_idx);

                // Trim generated tokens to match start
                self.generated.truncate(cut_token_idx);

                let trimmed_current = token_idx >= cut_token_idx;
                return Ok((
                    Some(FinishReason::StopString(matched_str.to_owned())),
                    trimmed_current,
                ));
            }
        }

        // 5. Keep tail bounded (up to 2 * max_stop_len)
        if self.req.stop.max_stop_len > 0 {
            let max_keep = self
                .req
                .stop
                .max_stop_len
                .checked_mul(2)
                .unwrap_or(512)
                .max(128);
            if self.detokenized_tail.len() > max_keep {
                let target_keep = self.req.stop.max_stop_len;
                let excess = self.detokenized_tail.len().saturating_sub(target_keep);
                let trim_bytes = self
                    .detokenized_tail
                    .char_indices()
                    .map(|(i, _)| i)
                    .find(|&i| i >= excess)
                    .unwrap_or(excess);
                if trim_bytes > 0 && trim_bytes <= self.detokenized_tail.len() {
                    self.detokenized_tail.drain(..trim_bytes);
                    self.tail_start_byte = self
                        .tail_start_byte
                        .checked_add(trim_bytes)
                        .ok_or_else(|| SchedError::overflow("tail_start_byte", "overflow"))?;
                    self.token_byte_spans
                        .retain(|(_, _, end)| *end > self.tail_start_byte);
                }
            }
        }

        Ok((None, false))
    }

    /// Resets all detokenizer and tail tracking state upon sequence finish or cancellation (Spec 6 §7).
    pub fn reset_tail_state(&mut self) {
        self.detokenized_tail.clear();
        self.tail_start_byte = 0;
        self.token_byte_spans.clear();
        self.pending_utf8_start = None;
    }
}

/// Execution descriptor for a single scheduler step (Spec 6 §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// Step identifier (Spec 6 §2, CONVENTIONS.md §3.1).
    pub step_id: StepId,
    /// Sequences executing decode tokens in this step (Spec 6 §2).
    pub seqs_decode: InlineVec<SeqId, 1>,
    /// Sequences executing prefill chunks in this step, with chunk sizes (Spec 6 §2).
    pub seqs_prefill: InlineVec<(SeqId, u32), 1>,
    /// Number of draft tokens verified per decode sequence (Spec 6 §2; for A3.9 k=0).
    pub k: InlineVec<u32, 1>,
    /// Discrete shape bucket `(S, T_dec, T_pre)` (Spec 1 §3.5, Spec 6 §2).
    pub bucket: (u32, u32, u32),
    /// Captured step graph keys per rank (Spec 6 §2, §5.1).
    pub graphs: InlineVec<StepGraphKey, 1>,
}

/// Outcome of executing a scheduler step (Spec 6 §3.3).
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    /// Public step execution descriptor constructed in real pre-step (Spec 6 §2, §3.1).
    pub step: Step,
    /// Step identifier (Spec 6 §3.3).
    pub step_id: StepId,
    /// Shape bucket executed (Spec 6 §2).
    pub bucket: (u32, u32, u32),
    /// Newly accepted tokens emitted per sequence in this step (Spec 6 §3.3).
    pub accepted_tokens: InlineVec<(SeqId, InlineVec<u32, 1>), 1>,
    /// Sequences that completed generation in this step (Spec 6 §3.3, §7).
    pub finished_sequences: InlineVec<(SeqId, FinishReason), 1>,
    /// Diagnostic telemetry record appended to the schedule log (Spec 6 §9).
    pub record: ScheduleRecord,
}
