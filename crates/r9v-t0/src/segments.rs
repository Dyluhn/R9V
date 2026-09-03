// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Explicit execution-only sequence segmentation descriptor (Spec 1 §4.E, Card A1.9).
//!
//! The IR signatures of `causal_conv1d` and `linear_attn_scan` carry `[T, ...]`
//! token-major tensors with no sequence-boundary input (SI-48): rows of `x`
//! are not self-delimiting. Both T0 entry points therefore take an explicit
//! [`SeqLayout`] execution parameter plus per-sequence state slots, and reset
//! their recurrences at every segment boundary. The IR arity is unchanged.
//!
//! DECISION(A1.9): sequence boundaries travel as an explicit `&SeqLayout`
//! execution parameter with per-sequence state slots carrying a leading `S`
//! axis (`[S, Wk-1, C]` conv, `[S, H, D, Dv]` scan); rejected threading
//! boundary metadata through fake tensor edges because `DType` is closed and
//! SI-12 keeps non-tensor execution metadata out of the tensor signature.
//! Per SI-48.

use crate::error::T0Error;

/// Explicit per-step sequence segmentation over the token axis (Spec 1 §4.E, SI-48).
///
/// `seq_lens[s]` is the token count of sequence `s`; the lengths sum to `T`.
/// Both `causal_conv1d` and `linear_attn_scan` reset their recurrence at every
/// segment boundary and address per-sequence state slot `s` for that segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqLayout {
    seq_lens: Vec<u32>,
    total: usize,
}

impl SeqLayout {
    /// Builds a segmentation from per-sequence token counts (Spec 1 §4.E, SI-48).
    ///
    /// Fails with [`T0Error::EmptyInput`] when no sequences are given or any
    /// sequence is zero-length, and with [`T0Error::ArithmeticOverflow`] when
    /// the token total overflows `usize`.
    pub fn new(seq_lens: &[u32]) -> Result<Self, T0Error> {
        if seq_lens.is_empty() {
            return Err(T0Error::EmptyInput {
                op: "seq_layout",
                tensor: "seq_lens",
            });
        }
        let mut total: usize = 0;
        for (s, &len) in seq_lens.iter().enumerate() {
            if len == 0 {
                return Err(T0Error::EmptyInput {
                    op: "seq_layout",
                    tensor: "seq_lens",
                });
            }
            total = total
                .checked_add(len as usize)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "seq_layout",
                    detail: format!("token total overflows usize at sequence {s}"),
                })?;
        }
        Ok(Self {
            seq_lens: seq_lens.to_vec(),
            total,
        })
    }

    /// Returns the per-sequence token counts.
    pub fn seq_lens(&self) -> &[u32] {
        &self.seq_lens
    }

    /// Returns the number of sequences `S`.
    pub fn seq_count(&self) -> usize {
        self.seq_lens.len()
    }

    /// Returns the total token count `T`.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Checks that the segmentation covers exactly `t` tokens (Spec 1 §4.E, SI-48).
    pub(crate) fn check_total(&self, tensor: &'static str, t: usize) -> Result<(), T0Error> {
        if self.total != t {
            return Err(T0Error::DimensionMismatch {
                dim_name: "T",
                expected_from: "seq_lens_sum",
                expected: t,
                tensor,
                got: self.total,
            });
        }
        Ok(())
    }
}
