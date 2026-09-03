// SPDX-License-Identifier: Apache-2.0
//! Typed errors for the sequence-state manager (Spec 3 §5, CONVENTIONS.md §1).
//!
//! All public untrusted inputs (configs, layer specs, token counts, positions)
//! are validated into these typed variants. Validation collects every problem
//! before returning; arithmetic uses checked ops and reports [`StateError::Overflow`]
//! instead of panicking or saturating.

/// A single config or spec problem found during collect-all validation.
///
/// Spec 3 §2, §6.3, §9.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidItem {
    /// Index of the offending item (layer index, group index, or `u32::MAX`
    /// for whole-config problems).
    pub index: u32,
    /// What was wrong, with the offending values.
    pub reason: String,
}

/// Errors from the sequence-state manager (Spec 3 §5).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateError {
    /// Config, layer specs, or pool sizing failed validation.
    ///
    /// Carries every problem found, not just the first (CONVENTIONS.md §1.4).
    #[error("invalid state config: {problems:?}")]
    InvalidConfig {
        /// All problems found during validation.
        problems: Vec<InvalidItem>,
    },

    /// Operation names an unknown or already-freed sequence.
    #[error("unknown sequence {seq}")]
    UnknownSeq {
        /// The unrecognized sequence id.
        seq: u64,
    },

    /// `new_seq` refused: live-sequence cap reached.
    #[error("sequence limit reached: live {live}, cap {cap}")]
    SeqLimit {
        /// Live sequences at refusal time.
        live: u32,
        /// Configured cap.
        cap: u32,
    },

    /// `reserve` refused: request does not fit the pool.
    ///
    /// Reports what was required, what was available, and the shortfall
    /// (CONVENTIONS.md §1.3).
    #[error("pool exhausted in group {group}: required {required} blocks, available {available}, shortfall {shortfall}, end {end}, max_ctx {max_ctx}")]
    PoolExhausted {
        /// Layer-group index (Spec 3 §6.1).
        group: usize,
        /// Blocks the reservation needs beyond what the sequence holds.
        required: u32,
        /// Free blocks in the group pool.
        available: u32,
        /// `required - available`.
        shortfall: u32,
        /// Requested end position (`ctx_len + n`).
        end: u32,
        /// Configured per-sequence token cap.
        max_ctx: u32,
    },

    /// `reserve` refused: request exceeds per-sequence or per-call limits.
    #[error("reservation too large: requested end {end}, max_ctx {max_ctx}, n {n}")]
    ReserveTooLarge {
        /// Requested end position.
        end: u32,
        /// Configured per-sequence token cap.
        max_ctx: u32,
        /// Requested token count.
        n: u32,
    },

    /// `reserve` called with `n == 0` or while a step is still outstanding.
    #[error("invalid reserve: n {n}, outstanding tail {tail}")]
    InvalidReserve {
        /// Requested token count.
        n: u32,
        /// Uncommitted tokens from the previous reserve.
        tail: u32,
    },

    /// `commit` refused: `accepted` exceeds the outstanding tail.
    #[error("commit too large: accepted {accepted}, outstanding tail {tail}")]
    CommitTooLarge {
        /// Accepted token count.
        accepted: u32,
        /// Uncommitted tokens from the reserve.
        tail: u32,
    },

    /// `commit` refused: no outstanding reservation on this sequence.
    ///
    /// Every commit must pair with a prior `reserve` (Spec 3 §3.6); even a
    /// zero accept with no open tail is a caller bug, not a no-op.
    #[error("commit with no outstanding reservation on sequence {seq}")]
    NoReservation {
        /// The sequence id from the caller's `commit` argument.
        seq: u64,
    },

    /// Slot lookup found no block for a retained position.
    ///
    /// `reserve` allocates every block touched by `ctx_len .. ctx_len + n`
    /// before setting the tail, so a missing mapping here means the caller
    /// asked for a position outside the reservation or the manager is
    /// inconsistent; both fail closed instead of clamping to a neighbor.
    #[error("no block mapped for group {group} position {pos} (reserved end {end})")]
    UnmappedPosition {
        /// Layer-group index (Spec 3 §6.1).
        group: usize,
        /// Absolute logical position with no mapped block.
        pos: u32,
        /// End of the reserved range (`ctx_len + tail_len`).
        end: u32,
    },

    /// `compact` refused: positions are out of range, duplicated, or there
    /// is no outstanding tail (Spec 3 §3.6 tree verify).
    #[error("invalid compact: len {len}, tail {tail}, detail {detail}")]
    InvalidCompact {
        /// Number of accepted positions supplied.
        len: usize,
        /// Outstanding tail at call time.
        tail: u32,
        /// Which check failed, with values.
        detail: String,
    },

    /// `batch_meta` refused: malformed batch (length mismatch, empty batch,
    /// bad query length, or a sequence with no outstanding reservation).
    #[error("invalid batch: {detail}")]
    InvalidBatch {
        /// Which check failed, with values.
        detail: String,
    },

    /// Host-side write/read outside the reserved range (Spec 3 §8 test support).
    #[error("state access out of range: start {start}, len {len}, reserved end {end}")]
    OutOfRange {
        /// Requested start position.
        start: u32,
        /// Requested length.
        len: usize,
        /// End of the reserved range (`ctx_len + tail_len`).
        end: u32,
    },

    /// Checked arithmetic overflowed; no state was mutated.
    #[error("arithmetic overflow computing {what}")]
    Overflow {
        /// The quantity being computed.
        what: String,
    },

    /// Sequence ID exceeds 32-bit device address space (Spec 1 §2.5, SI-40).
    #[error("sequence id {seq} exceeds device u32 width {max}")]
    SeqIdOverflow {
        /// 64-bit host sequence id.
        seq: u64,
        /// Maximum 32-bit device id (`u32::MAX`).
        max: u32,
    },

    /// Underlying Op IR error (CONVENTIONS.md §1.1).
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),
}

impl StateError {
    /// Builds an [`StateError::InvalidConfig`] from collected problems.
    pub fn invalid(problems: Vec<InvalidItem>) -> Self {
        Self::InvalidConfig { problems }
    }
}

/// Convenience alias for state results (Spec 3 §5).
pub type StateResult<T> = std::result::Result<T, StateError>;
