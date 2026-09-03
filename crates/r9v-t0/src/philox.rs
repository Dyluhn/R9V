// SPDX-License-Identifier: Apache-2.0
//! Philox4x32-10 counter-based PRNG and RNG state (Spec 1 §4.F, Spec 4 §5.8, Spec 7 §4).

/// Round multiplier constant M0 (Random123 PHILOX_M4x32_0).
const PHILOX_M0: u32 = 0xD251_1F53;
/// Round multiplier constant M1 (Random123 PHILOX_M4x32_1).
const PHILOX_M1: u32 = 0xCD9E_8D57;
/// Per-round Weyl key bump constant W0 (Random123 PHILOX_W32_0).
const PHILOX_W0: u32 = 0x9E37_79B9;
/// Per-round Weyl key bump constant W1 (Random123 PHILOX_W32_1).
const PHILOX_W1: u32 = 0xBB67_AE85;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32)
}

#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(PHILOX_M0, ctr[0]);
    let (hi1, lo1) = mulhilo(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// 10-round Philox4x32 block function (Spec 1 §4.F, Salmon et al. 2011).
///
/// Maps a 128-bit counter `ctr` and 64-bit `key` to four pseudo-random `u32` words.
/// Round 0 uses the initial key; rounds 1..9 bump key words by Weyl constants before the round.
#[inline]
#[must_use]
pub fn philox4x32_10(ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    let mut c = round(ctr, key);
    for _ in 1..10 {
        key[0] = key[0].wrapping_add(PHILOX_W0);
        key[1] = key[1].wrapping_add(PHILOX_W1);
        c = round(c, key);
    }
    c
}

use r9v_common::{SeqId, StepId};

use crate::error::T0Error;

// DECISION(A1.8): Philox4x32 maps 64-bit key [seed as u32, (seed >> 32) as u32] and 128-bit counter [draw_index, step_lo, step_hi, seq_id_u32]; step is encoded losslessly as 64-bit (step_lo, step_hi) to prevent collision across steps > 2^32, and seq_id is bound to canonical BatchMeta u32 global sequence ids, rejecting seq_id > u32::MAX before mutation; uniform f32 draws map word >> 9 centered by +0.5 in (0, 1) to avoid boundary saturation in inverse-CDF; rejected uncentered float conversion because boundary 0.0/1.0 corrupts rejection and CDF thresholds. Spec 1 §4.F, Spec 4 §5.8, Spec 7 §4.
// DECISION(A1.8): draw_index arithmetic is checked, never wrapping: advance and draw_uniform_f32 fail loudly on u32 overflow instead of silently reusing draws; rejected wrapping_add because a wrapped draw counter replays earlier uniforms and breaks the (seq, step, draw) uniqueness the Philox counter mapping promises. Spec 1 §4.F.
/// Maps a 32-bit random word to an `f32` uniform on the open interval `(0, 1)` (Spec 1 §4.F).
///
/// Uses the upper 23 bits (`word >> 9`) centered by `+0.5` scaled by `2^-23`.
/// This guarantees the result is strictly within `(0.0, 1.0)`, avoiding exact 0.0 or 1.0
/// boundary saturation in inverse-CDF or rejection sampling.
#[inline]
#[must_use]
pub fn u32_to_unit_f32(word: u32) -> f32 {
    ((word >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

/// Explicit RNG state for a sequence (Spec 1 §4.F, Spec 7 §4).
///
/// Holds the seed, sequence identifier, execution step, and draw counter.
/// Because Philox4x32 is counter-based and keyed by `(seq_id, step, draw_index)`,
/// every draw is reproducible across arbitrary batch sizes and topologies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RngState {
    seed: u64,
    seq_id: SeqId,
    seq_word: u32,
    step: StepId,
    draw_index: u32,
}

impl RngState {
    /// Creates a new RNG state with `draw_index = 0` (Spec 1 §4.F).
    pub fn new(seed: u64, seq_id: SeqId, step: StepId) -> Result<Self, T0Error> {
        let seq_word = u32::try_from(seq_id.as_u64()).map_err(|_| T0Error::SeqIdOutOfRange {
            op: "RngState::new",
            seq_id: seq_id.as_u64(),
            max: u32::MAX as u64,
        })?;
        Ok(Self {
            seed,
            seq_id,
            seq_word,
            step,
            draw_index: 0,
        })
    }

    /// Creates an RNG state with an explicit initial `draw_index` (Spec 1 §4.F).
    pub fn with_draw(
        seed: u64,
        seq_id: SeqId,
        step: StepId,
        draw_index: u32,
    ) -> Result<Self, T0Error> {
        let mut state = Self::new(seed, seq_id, step)?;
        state.draw_index = draw_index;
        Ok(state)
    }

    /// Creates an RNG state from raw 64-bit integer values (Spec 1 §4.F).
    pub fn from_u64(seed: u64, seq_id: u64, step: u64) -> Result<Self, T0Error> {
        Self::new(seed, SeqId::new(seq_id), StepId::new(step))
    }

    /// Returns the RNG seed (Spec 1 §4.F, Spec 10 §4).
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the sequence identifier (Spec 1 §4.F, Spec 3 §2).
    pub const fn seq_id(&self) -> SeqId {
        self.seq_id
    }

    /// Returns the sequence id as the canonical u32 Philox counter word (Spec 1 §4.F).
    ///
    /// The 128-bit Philox counter carries the sequence id in one 32-bit word,
    /// so ids above `u32::MAX` cannot be represented without truncation
    /// collision. Sampling ops call this for every state at the op boundary,
    /// before any RNG or output mutation, so rejection leaves state untouched.
    pub const fn seq_id_u32(&self) -> u32 {
        self.seq_word
    }

    /// Returns the step identifier/counter (Spec 1 §4.F, Spec 6 §9).
    pub const fn step(&self) -> StepId {
        self.step
    }

    /// Returns the current draw index within the step (Spec 1 §4.F).
    pub const fn draw_index(&self) -> u32 {
        self.draw_index
    }

    /// Updates the execution step and resets the draw counter to 0 (Spec 1 §4.F).
    pub fn set_step(&mut self, step: StepId) {
        self.step = step;
        self.draw_index = 0;
    }

    /// Advances the draw index by `count` (Spec 1 §4.F).
    ///
    /// Fails loudly on `u32` overflow: a wrapped counter would replay earlier
    /// uniforms for the same `(seq, step)`.
    pub fn advance(&mut self, count: u32) -> Result<(), T0Error> {
        let next = self.checked_advance(count, "RngState::advance")?;
        self.draw_index = next;
        Ok(())
    }

    /// Checks an advance without mutating this state.
    pub(crate) fn ensure_can_advance(&self, count: u32, op: &'static str) -> Result<(), T0Error> {
        self.checked_advance(count, op).map(|_| ())
    }

    fn checked_advance(&self, count: u32, op: &'static str) -> Result<u32, T0Error> {
        self.draw_index
            .checked_add(count)
            .ok_or(T0Error::DrawIndexOverflow {
                op,
                draw_index: self.draw_index,
                advance: count,
            })
    }

    /// Draws a uniform `f32` in `(0, 1)` at a specified draw index without mutating the state (Spec 1 §4.F, Spec 7 §4).
    ///
    /// Construction establishes `seq_id <= u32::MAX`, so the counter word is
    /// always lossless and this pure accessor cannot fail.
    #[must_use]
    pub fn draw_at(&self, draw: u32) -> f32 {
        let step_val = self.step.as_u64();
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        let ctr = [
            draw,
            step_val as u32,
            (step_val >> 32) as u32,
            self.seq_word,
        ];
        let words = philox4x32_10(ctr, key);
        u32_to_unit_f32(words[0])
    }

    /// Draws the next uniform `f32` in `(0, 1)` and increments `draw_index` by 1 (Spec 1 §4.F).
    ///
    /// Fails loudly on `u32` overflow rather than wrapping onto reused draws.
    pub fn draw_uniform_f32(&mut self) -> Result<f32, T0Error> {
        let next = self.checked_advance(1, "RngState::draw_uniform_f32")?;
        let val = self.draw_at(self.draw_index);
        self.draw_index = next;
        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_philox_known_answer_vectors() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([0xffff_ffff; 4], [0xffff_ffff, 0xffff_ffff]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn test_u32_to_unit_f32_strictly_in_open_interval() {
        let min_val = u32_to_unit_f32(0);
        let max_val = u32_to_unit_f32(0xffff_ffff);
        assert!(min_val > 0.0);
        assert!(max_val < 1.0);
        assert!(min_val < max_val);
    }

    #[test]
    fn test_rng_state_draw_reproducibility() {
        let mut rng1 = RngState::new(42, SeqId::new(1), StepId::new(0)).unwrap();
        let mut rng2 = RngState::new(42, SeqId::new(1), StepId::new(0)).unwrap();
        for _ in 0..100 {
            assert_eq!(
                rng1.draw_uniform_f32().unwrap(),
                rng2.draw_uniform_f32().unwrap()
            );
        }
    }

    #[test]
    fn test_rng_state_step_upper_bits_no_collision() {
        // Prove that steps differing only in upper 32 bits produce distinct draws (no truncation)
        let rng_low = RngState::new(42, SeqId::new(1), StepId::new(1)).unwrap();
        let rng_high = RngState::new(42, SeqId::new(1), StepId::new(1 | (1u64 << 32))).unwrap();
        assert_ne!(rng_low.draw_at(0), rng_high.draw_at(0));

        let rng_zero = RngState::new(42, SeqId::new(1), StepId::new(0)).unwrap();
        let rng_step_rollover = RngState::new(42, SeqId::new(1), StepId::new(1u64 << 32)).unwrap();
        assert_ne!(rng_zero.draw_at(0), rng_step_rollover.draw_at(0));
    }

    #[test]
    fn test_rng_state_boundary_values() {
        let rng_max =
            RngState::new(u64::MAX, SeqId::new(u32::MAX as u64), StepId::new(u64::MAX)).unwrap();
        let val = rng_max.draw_at(u32::MAX);
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_rng_state_seq_id_u32_checked() {
        let ok = RngState::new(7, SeqId::new(u32::MAX as u64), StepId::new(0)).unwrap();
        assert_eq!(ok.seq_id_u32(), u32::MAX);
        let err = RngState::new(7, SeqId::new(u32::MAX as u64 + 1), StepId::new(0)).unwrap_err();
        assert!(
            matches!(err, crate::error::T0Error::SeqIdOutOfRange { seq_id, max, .. }
                if seq_id == u32::MAX as u64 + 1 && max == u32::MAX as u64)
        );
    }

    #[test]
    fn test_rng_state_draw_at_is_pure_over_advance() {
        // draw_at(i) is a pure function of (seed, seq, step, i): advancing the
        // mutable cursor never changes what draw_at reports for any index.
        let mut rng = RngState::new(99, SeqId::new(3), StepId::new(4)).unwrap();
        let before: Vec<f32> = (0..8).map(|i| rng.draw_at(i)).collect();
        for _ in 0..8 {
            rng.draw_uniform_f32().unwrap();
        }
        let after: Vec<f32> = (0..8).map(|i| rng.draw_at(i)).collect();
        assert_eq!(before, after);
        // Sequential draws consume draw_at(0), draw_at(1), ... in order.
        let mut fresh = RngState::new(99, SeqId::new(3), StepId::new(4)).unwrap();
        for (i, &expected) in before.iter().enumerate() {
            assert_eq!(fresh.draw_uniform_f32().unwrap(), expected, "draw {i}");
        }
    }

    #[test]
    fn test_rng_state_advance_overflow_is_typed_and_non_mutating() {
        let mut rng = RngState::with_draw(1, SeqId::new(0), StepId::new(0), u32::MAX).unwrap();
        let err = rng.advance(1).unwrap_err();
        assert!(matches!(
            err,
            T0Error::DrawIndexOverflow {
                op: "RngState::advance",
                draw_index: u32::MAX,
                advance: 1
            }
        ));
        assert_eq!(rng.draw_index(), u32::MAX);

        let err = rng.draw_uniform_f32().unwrap_err();
        assert!(matches!(
            err,
            T0Error::DrawIndexOverflow {
                op: "RngState::draw_uniform_f32",
                draw_index: u32::MAX,
                advance: 1
            }
        ));
        assert_eq!(rng.draw_index(), u32::MAX);
    }
}
