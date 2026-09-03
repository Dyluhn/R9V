// SPDX-License-Identifier: Apache-2.0
//! Tests for XXH3 helpers (Spec 2 §6, Spec 3 §2, Spec 4 §1.6, Spec 9 §1, Spec 14 §2).

use std::hash::Hasher;

use r9v_common::{
    xxh3_128, xxh3_128_with_seed, xxh3_64, xxh3_64_with_seed, Xxh3Hasher, Xxh3Hasher128,
};

/// Test known-answer vectors published by the authoritative xxHash specification (Yann Collet).
#[test]
fn xxh3_64_authoritative_known_answer_vectors() {
    // Empty input: official xxHash vector
    assert_eq!(xxh3_64(b""), 0x2d06800538d394c2);

    // Single byte input
    assert_eq!(xxh3_64(b"a"), 0xe6c632b61e964e1f);

    // Short string
    assert_eq!(xxh3_64(b"abc"), 0x78af5f94892f3950);

    // Number string
    assert_eq!(xxh3_64(b"1234567890"), 0x80048550fad2b420);

    // Standard pangram: official reference vector
    assert_eq!(
        xxh3_64(b"The quick brown fox jumps over the lazy dog"),
        0xce7d19a5418fb365
    );

    // Seeded vector
    assert_eq!(xxh3_64_with_seed(b"", 42), 0xb029411ff43d84d2);
    assert_eq!(xxh3_64_with_seed(b"a", 42), 0x4c437dd47f0716f4);
}

/// Test known-answer vectors for 128-bit XXH3.
#[test]
fn xxh3_128_authoritative_known_answer_vectors() {
    // Empty input
    assert_eq!(xxh3_128(b""), 0x99aa06d3014798d86001c324468d497f);

    // Single byte input
    assert_eq!(xxh3_128(b"a"), 0xa96faf705af16834e6c632b61e964e1f);

    // Standard pangram
    assert_eq!(
        xxh3_128(b"The quick brown fox jumps over the lazy dog"),
        0xddd650205ca3e7fa24a1cc2e3a8a7651
    );

    // Seeded vector
    assert_eq!(
        xxh3_128_with_seed(b"", 42),
        0x16c20acd33f7af2f3c1d09e9fe249164
    );
}

#[test]
fn streaming_hasher_matches_oneshot() {
    let part1 = b"first chunk of tensor bytes ";
    let part2 = b"second chunk of tensor bytes";
    let mut combined = Vec::new();
    combined.extend_from_slice(part1);
    combined.extend_from_slice(part2);

    let oneshot = xxh3_64(&combined);

    let mut hasher = Xxh3Hasher::default();
    hasher.write(part1);
    hasher.write(part2);
    let streamed = hasher.finish();

    assert_eq!(streamed, oneshot);
}

#[test]
fn streaming_hasher128_matches_oneshot() {
    let part1 = b"fingerprint prefix ";
    let part2 = b"fingerprint suffix";
    let mut combined = Vec::new();
    combined.extend_from_slice(part1);
    combined.extend_from_slice(part2);

    let oneshot = xxh3_128(&combined);

    let mut hasher = Xxh3Hasher128::default();
    hasher.write(part1);
    hasher.write(part2);
    let streamed = hasher.finish_128();

    assert_eq!(streamed, oneshot);
}
