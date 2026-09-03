// SPDX-License-Identifier: Apache-2.0
//! Tests for deterministic seeded RNG (Spec 1 §6.1, Spec 14 §2, CONVENTIONS.md §4.3).

use r9v_common::SeededRng;

/// Test vectors generated from the upstream reference implementation of Xoshiro256++
/// (David Blackman & Sebastiano Vigna, 2019) initialized via SplitMix64.
#[test]
fn xoshiro256plusplus_seed_derived_vectors() {
    // Seed = 1 sequence
    let mut rng_seed1 = SeededRng::new(1);
    assert_eq!(rng_seed1.next_u64(), 0xcfc5d07f6f03c29b);
    assert_eq!(rng_seed1.next_u64(), 0xbf424132963fe08d);
    assert_eq!(rng_seed1.next_u64(), 0x19a37d5757aaf520);
    assert_eq!(rng_seed1.next_u64(), 0xbf08119f05cd56d6);
    assert_eq!(rng_seed1.next_u64(), 0x2f47184b86186fa4);

    // Seed = 42 sequence
    let mut rng_seed42 = SeededRng::new(42);
    assert_eq!(rng_seed42.next_u64(), 0xd0764d4f4476689f);
    assert_eq!(rng_seed42.next_u64(), 0x519e4174576f3791);
    assert_eq!(rng_seed42.next_u64(), 0xfbe07cfb0c24ed8c);
    assert_eq!(rng_seed42.next_u64(), 0xb37d9f600cd835b8);
    assert_eq!(rng_seed42.next_u64(), 0xcb231c3874846a73);
}

#[test]
fn determinism_across_identical_seeds() {
    let mut rng1 = SeededRng::new(12345);
    let mut rng2 = SeededRng::new(12345);

    for _ in 0..1000 {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
}

#[test]
fn different_seeds_produce_different_sequences() {
    let mut rng1 = SeededRng::new(42);
    let mut rng2 = SeededRng::new(43);

    let vals1: Vec<u64> = (0..100).map(|_| rng1.next_u64()).collect();
    let vals2: Vec<u64> = (0..100).map(|_| rng2.next_u64()).collect();

    assert_ne!(vals1, vals2);
}

#[test]
fn fill_bytes_is_deterministic() {
    let mut rng1 = SeededRng::new(777);
    let mut rng2 = SeededRng::new(777);

    let mut buf1 = [0u8; 127];
    let mut buf2 = [0u8; 127];

    rng1.fill_bytes(&mut buf1);
    rng2.fill_bytes(&mut buf2);

    assert_eq!(buf1, buf2);
}
