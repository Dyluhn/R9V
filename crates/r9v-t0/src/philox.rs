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

// DECISION(A1.8): Philox4x32 maps key [seed as u32, (seed >> 32) as u32] and counter [draw_index, step as u32, seq_id as u32, (seq_id >> 32) as u32]; uniform f32 draws map word >> 9 centered by +0.5 in (0, 1) to avoid boundary saturation in inverse-CDF; rejected uncentered float conversion because boundary 0.0/1.0 corrupts rejection and CDF thresholds. Spec 1 §4.F, Spec 7 §4.
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
    seq_id: u64,
    step: u64,
    draw_index: u32,
}

impl RngState {
    /// Creates a new RNG state with `draw_index = 0` (Spec 1 §4.F).
    pub const fn new(seed: u64, seq_id: u64, step: u64) -> Self {
        Self {
            seed,
            seq_id,
            step,
            draw_index: 0,
        }
    }

    /// Creates an RNG state with an explicit initial `draw_index` (Spec 1 §4.F).
    pub const fn with_draw(seed: u64, seq_id: u64, step: u64, draw_index: u32) -> Self {
        Self {
            seed,
            seq_id,
            step,
            draw_index,
        }
    }

    /// Returns the RNG seed (Spec 1 §4.F, Spec 10 §4).
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the sequence identifier (Spec 1 §4.F, Spec 3 §2).
    pub const fn seq_id(&self) -> u64 {
        self.seq_id
    }

    /// Returns the step identifier/counter (Spec 1 §4.F, Spec 6 §9).
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Returns the current draw index within the step (Spec 1 §4.F).
    pub const fn draw_index(&self) -> u32 {
        self.draw_index
    }

    /// Updates the execution step and resets the draw counter to 0 (Spec 1 §4.F).
    pub fn set_step(&mut self, step: u64) {
        self.step = step;
        self.draw_index = 0;
    }

    /// Advances the draw index by `count` (Spec 1 §4.F).
    pub fn advance(&mut self, count: u32) {
        self.draw_index = self.draw_index.wrapping_add(count);
    }

    /// Draws a uniform `f32` in `(0, 1)` at a specified draw index without mutating the state (Spec 1 §4.F, Spec 7 §4).
    #[must_use]
    pub fn draw_at(&self, draw: u32) -> f32 {
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        let ctr = [
            draw,
            self.step as u32,
            self.seq_id as u32,
            (self.seq_id >> 32) as u32,
        ];
        let words = philox4x32_10(ctr, key);
        u32_to_unit_f32(words[0])
    }

    /// Draws the next uniform `f32` in `(0, 1)` and increments `draw_index` by 1 (Spec 1 §4.F).
    pub fn draw_uniform_f32(&mut self) -> f32 {
        let val = self.draw_at(self.draw_index);
        self.draw_index = self.draw_index.wrapping_add(1);
        val
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
        let mut rng1 = RngState::new(42, 1, 0);
        let mut rng2 = RngState::new(42, 1, 0);
        for _ in 0..100 {
            assert_eq!(rng1.draw_uniform_f32(), rng2.draw_uniform_f32());
        }
    }
}
