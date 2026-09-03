// SPDX-License-Identifier: Apache-2.0
//! XXH3 hashing helpers (Spec 2 §6, Spec 3 §2, Spec 4 §1.6, Spec 9 §1, Spec 14 §2).

pub use twox_hash::XxHash3_128 as Xxh3Hasher128;
pub use twox_hash::XxHash3_64 as Xxh3Hasher;

/// Computes the 64-bit XXH3 checksum of `data` (Spec 2 §6, Spec 3 §2, Spec 4 §1.6).
///
/// Used for tensor integrity checksums (`r9v.tensor.<name>.xxh3`), block content-addressing
/// in the prefix cache, and kernel variant identification.
#[inline]
pub fn xxh3_64(data: &[u8]) -> u64 {
    twox_hash::XxHash3_64::oneshot(data)
}

/// Computes the 64-bit XXH3 checksum of `data` with a custom 64-bit `seed` (Spec 14 §2).
#[inline]
pub fn xxh3_64_with_seed(data: &[u8], seed: u64) -> u64 {
    twox_hash::XxHash3_64::oneshot_with_seed(seed, data)
}

/// Computes the 128-bit XXH3 hash of `data` (Spec 9 §1).
///
/// Used for compound file fingerprints (`file_fp`) and model fingerprints (`model_fp`).
#[inline]
pub fn xxh3_128(data: &[u8]) -> u128 {
    twox_hash::XxHash3_128::oneshot(data)
}

/// Computes the 128-bit XXH3 hash of `data` with a custom 64-bit `seed` (Spec 14 §2).
#[inline]
pub fn xxh3_128_with_seed(data: &[u8], seed: u64) -> u128 {
    twox_hash::XxHash3_128::oneshot_with_seed(seed, data)
}
