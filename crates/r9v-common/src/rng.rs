// SPDX-License-Identifier: Apache-2.0
//! Deterministic seeded random number generator (Spec 1 §6.1, Spec 14 §2, CONVENTIONS.md §4.3).

// DECISION(A0.4): SeededRng implements Xoshiro256++ seeded via SplitMix64; rejected external rand dependency to maintain zero-dependency determinism and pass license audit under cargo deny.

/// Deterministic, reproducible pseudorandom number generator for test fixtures (Spec 1 §6.1, Spec 14 §2).
///
/// Implemented with the Xoshiro256++ algorithm (David Blackman and Sebastiano Vigna, 2019)
/// seeded through SplitMix64. Zero-allocation, platform-independent, and bit-exact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeededRng {
    s: [u64; 4],
}

impl SeededRng {
    /// Creates a new [`SeededRng`] initialized from a 64-bit seed using SplitMix64 (Spec 14 §2).
    pub const fn new(seed: u64) -> Self {
        let mut x = seed;

        // SplitMix64 generator to populate 256 bits of state
        let s0 = splitmix64_step(&mut x);
        let s1 = splitmix64_step(&mut x);
        let s2 = splitmix64_step(&mut x);
        let s3 = splitmix64_step(&mut x);

        // State must not be all zeros
        let (s0, s1, s2, s3) = if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
            (
                0x9E3779B97F4A7C15,
                0xBF58476D1CE4E5B9,
                0x94D049BB133111EB,
                0xD6E8FEB86659FD93,
            )
        } else {
            (s0, s1, s2, s3)
        };

        Self {
            s: [s0, s1, s2, s3],
        }
    }

    /// Generates the next pseudorandom `u64` for a deterministic fixture (Spec 14 §2).
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Fills a fixture byte slice deterministically (Spec 14 §2).
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let val = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&val);
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let val = self.next_u64().to_le_bytes();
            remainder.copy_from_slice(&val[..remainder.len()]);
        }
    }
}

const fn splitmix64_step(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the internal step matches the upstream reference implementation of
    /// Xoshiro256++ (David Blackman & Sebastiano Vigna, 2019) on arbitrary raw state [1, 2, 3, 4].
    #[test]
    fn reference_algorithm_raw_state() {
        let mut rng = SeededRng { s: [1, 2, 3, 4] };
        assert_eq!(rng.next_u64(), 0x0000000002800001);
        assert_eq!(rng.next_u64(), 0x0000000003800067);
        assert_eq!(rng.next_u64(), 0x000cc00003800067);
        assert_eq!(rng.next_u64(), 0x000cc201994400b2);
        assert_eq!(rng.next_u64(), 0x8012a2019ac433cd);
    }
}
