// SPDX-License-Identifier: Apache-2.0
//! Determinism tests (Spec 3 §5, §8): block ids and `BatchMeta` are a
//! function of the request history alone.

mod common;

use common::{config_128, kv_all, manager_for, step_all};

fn run_history(m: &mut r9v_state::StateManager) -> r9v_state::BatchMeta {
    let (a, _) = m.new_seq(&[1, 2, 3]).unwrap();
    let (b, _) = m.new_seq(&[4, 5]).unwrap();
    let (c, _) = m.new_seq(&[]).unwrap();
    step_all(m, a, &[10, 11, 12, 13]);
    step_all(m, b, &[20, 21]);
    m.reserve(a, 3).unwrap();
    m.reserve(b, 5).unwrap();
    let meta = m.batch_meta(&[a, b], &[3, 5]).unwrap();
    m.write_tokens(a, 4, &[14, 15, 16]).unwrap();
    m.write_tokens(b, 2, &[22, 23, 24, 25, 26]).unwrap();
    m.commit(a, 3).unwrap();
    m.commit(b, 2).unwrap();
    m.free_seq(c).unwrap();
    meta
}

/// Same request history produces identical block ids (Spec 3 §8).
#[test]
fn same_request_history_produces_identical_block_ids() {
    let mut m1 = manager_for(config_128(), &[kv_all()]);
    let mut m2 = manager_for(config_128(), &[kv_all()]);
    let meta1 = run_history(&mut m1);
    let meta2 = run_history(&mut m2);
    assert_eq!(meta1, meta2);
    assert_eq!(m1.budget(), m2.budget());
    assert_eq!(m1.stats().commits, m2.stats().commits);
}

/// Allocation takes the smallest free id first, so freed blocks are reused
/// deterministically (Spec 3 §5).
#[test]
fn freed_blocks_are_reused_smallest_first() {
    let mut m = manager_for(config_128(), &[kv_all()]);
    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();
    step_all(&mut m, a, &[0; 64]); // blocks 0, 1
    step_all(&mut m, b, &[0; 32]); // block 2
    m.free_seq(a).unwrap();

    let (c, _) = m.new_seq(&[]).unwrap();
    let slots = m.reserve(c, 32).unwrap();
    // Block 0 is the smallest free id: flattened slot 0 * 32 + lane.
    let mut row = vec![0u32; 32];
    m.fill_slots(&slots, 0, &mut row).unwrap();
    assert_eq!(row, (0..32).collect::<Vec<u32>>());
    m.commit(c, 32).unwrap();
}

/// `reserve` twice without an intervening commit is rejected (one
/// outstanding step per sequence, Spec 3 §3.6).
#[test]
fn double_reserve_without_commit_is_rejected() {
    let mut m = manager_for(config_128(), &[kv_all()]);
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();
    let err = m.reserve(a, 4).unwrap_err();
    assert!(
        matches!(err, r9v_state::StateError::InvalidReserve { .. }),
        "{err:?}"
    );
}

/// Allocation depends on request lengths alone: the token slice is only the
/// future prefix-cache key, so different contents with the same lengths
/// produce identical `BatchMeta` (Spec 3 §5).
#[test]
fn allocation_ignores_token_contents() {
    let mut m1 = manager_for(config_128(), &[kv_all()]);
    let mut m2 = manager_for(config_128(), &[kv_all()]);
    let (a1, matched1) = m1.new_seq(&[1, 2, 3]).unwrap();
    let (a2, matched2) = m2.new_seq(&[9, 9, 9]).unwrap();
    assert_eq!(matched1, 0);
    assert_eq!(matched2, 0);
    m1.reserve(a1, 5).unwrap();
    m2.reserve(a2, 5).unwrap();
    let meta1 = m1.batch_meta(&[a1], &[5]).unwrap();
    let meta2 = m2.batch_meta(&[a2], &[5]).unwrap();
    assert_eq!(meta1, meta2);
}
