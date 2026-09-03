// SPDX-License-Identifier: Apache-2.0
//! Exhaustion, overflow, malformed-input, and rollback-atomicity tests.
//! Every refusal is a typed error with the numbers; failed transitions
//! leave all state untouched (Spec 3 §5).

mod common;

use common::{kv_all, manager_for};
use r9v_state::{block_offset, StateConfig, StateError, StateManager, StateSpec};

fn tiny_manager() -> StateManager {
    // max_ctx 64: 2 blocks per sequence in the single group.
    manager_for(
        StateConfig {
            max_ctx: 64,
            max_seqs: 4,
        },
        &[kv_all()],
    )
}

/// Exhaustion names the group, required, available, and shortfall, and the
/// failed reserve mutates nothing (Spec 3 §3.6 admission rule).
#[test]
fn pool_exhaustion_reports_numbers_and_leaves_state_untouched() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();

    let toks = vec![7; 64];
    m.reserve(a, 64).unwrap();
    m.write_tokens(a, 0, &toks).unwrap();
    m.commit(a, 64).unwrap();
    assert_eq!(m.free_blocks(0).unwrap(), 0);

    let free_before = m.free_blocks(0).unwrap();
    let err = m.reserve(b, 32).unwrap_err();
    match err {
        StateError::PoolExhausted {
            group,
            required,
            available,
            shortfall,
            end,
            max_ctx,
        } => {
            assert_eq!(group, 0);
            assert_eq!(required, 1);
            assert_eq!(available, 0);
            assert_eq!(shortfall, 1);
            assert_eq!(end, 32);
            assert_eq!(max_ctx, 64);
        }
        other => panic!("expected PoolExhausted, got {other:?}"),
    }
    // Atomic: no tail opened, no blocks consumed.
    assert_eq!(m.tail_len(b).unwrap(), 0);
    assert_eq!(m.free_blocks(0).unwrap(), free_before);
}

/// Oversized and empty reserves are typed, with the limits attached.
#[test]
fn reserve_limits_are_typed() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();

    let err = m.reserve(a, 65).unwrap_err();
    assert!(matches!(err, StateError::ReserveTooLarge { .. }), "{err:?}");

    let err = m.reserve(a, 0).unwrap_err();
    assert!(matches!(err, StateError::InvalidReserve { .. }), "{err:?}");

    assert_eq!(m.tail_len(a).unwrap(), 0);
}

/// Over-commits are rejected and change nothing.
#[test]
fn commit_beyond_tail_is_rejected_without_mutation() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();
    m.write_tokens(a, 0, &[1, 2, 3, 4]).unwrap();

    let err = m.commit(a, 5).unwrap_err();
    assert!(matches!(err, StateError::CommitTooLarge { .. }), "{err:?}");
    assert_eq!(m.ctx_len(a).unwrap(), 0);
    assert_eq!(m.tail_len(a).unwrap(), 4);

    m.commit(a, 4).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 4);
}

/// Unknown sequences are typed on every path.
#[test]
fn unknown_sequences_are_rejected() {
    use r9v_common::SeqId;
    let mut m = tiny_manager();
    let ghost = SeqId::new(999);
    assert!(matches!(
        m.reserve(ghost, 1).unwrap_err(),
        StateError::UnknownSeq { .. }
    ));
    assert!(matches!(
        m.commit(ghost, 1).unwrap_err(),
        StateError::UnknownSeq { .. }
    ));
    assert!(matches!(
        m.free_seq(ghost).unwrap_err(),
        StateError::UnknownSeq { .. }
    ));
    assert!(matches!(
        m.ctx_len(ghost).unwrap_err(),
        StateError::UnknownSeq { .. }
    ));
    // Double free is also unknown.
    let (a, _) = m.new_seq(&[]).unwrap();
    m.free_seq(a).unwrap();
    assert!(matches!(
        m.free_seq(a).unwrap_err(),
        StateError::UnknownSeq { .. }
    ));
}

/// `compact` validates range, uniqueness, tail presence, and commit match.
#[test]
fn compact_validation_is_typed() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();

    // No outstanding tail.
    let err = m.compact(a, &[0]).unwrap_err();
    assert!(matches!(err, StateError::InvalidCompact { .. }), "{err:?}");

    m.reserve(a, 4).unwrap();
    m.write_tokens(a, 0, &[1, 2, 3, 4]).unwrap();

    // Out of range.
    let err = m.compact(a, &[0, 4]).unwrap_err();
    assert!(matches!(err, StateError::InvalidCompact { .. }), "{err:?}");
    // Duplicate.
    let err = m.compact(a, &[1, 1]).unwrap_err();
    assert!(matches!(err, StateError::InvalidCompact { .. }), "{err:?}");
    // Failed compacts mutate nothing: a valid compact still works.
    let op = m.compact(a, &[3, 1]).unwrap();
    assert_eq!(op.len, 2);
    // Commit must match the compacted length.
    let err = m.commit(a, 4).unwrap_err();
    assert!(matches!(err, StateError::InvalidCompact { .. }), "{err:?}");
    m.commit(a, 2).unwrap();
    assert_eq!(m.read_token(a, 0, 0).unwrap(), Some(4));
    assert_eq!(m.read_token(a, 0, 1).unwrap(), Some(2));
}

/// `batch_meta` collects batch-shape problems into one typed error.
#[test]
fn batch_meta_validation_is_typed() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();

    // Length mismatch.
    let err = m.batch_meta(&[a, b], &[4]).unwrap_err();
    assert!(matches!(err, StateError::InvalidBatch { .. }), "{err:?}");
    // Empty batch.
    let err = m.batch_meta(&[], &[]).unwrap_err();
    assert!(matches!(err, StateError::InvalidBatch { .. }), "{err:?}");
    // Query exceeds the outstanding tail (b reserved nothing).
    let err = m.batch_meta(&[a, b], &[4, 1]).unwrap_err();
    assert!(matches!(err, StateError::InvalidBatch { .. }), "{err:?}");
    // Zero query length.
    let err = m.batch_meta(&[a], &[0]).unwrap_err();
    assert!(matches!(err, StateError::InvalidBatch { .. }), "{err:?}");
}

/// Overflow is a typed error, never a panic or wrap (checked arithmetic).
#[test]
fn arithmetic_overflow_is_typed() {
    let err = block_offset(u64::MAX - 10, u32::MAX, u64::MAX).unwrap_err();
    assert!(matches!(err, StateError::Overflow { .. }), "{err:?}");
    assert_eq!(block_offset(100, 3, 50).unwrap(), 250);

    // Absurd dims are rejected by principled limits before any allocation.
    let cfg = StateConfig {
        max_ctx: 64,
        max_seqs: 1,
    };
    let bad = StateSpec::KvPaged {
        hkv: u32::MAX,
        d: 128,
        dv: 128,
        cache: r9v_state::CacheDtype::E4M3,
        retain: r9v_state::Retain::All,
    };
    let err = StateManager::new(cfg, vec![bad], u64::MAX).unwrap_err();
    assert!(matches!(err, StateError::InvalidConfig { .. }), "{err:?}");
}

/// Config validation collects every problem instead of stopping at the first.
#[test]
fn config_validation_collects_all_problems() {
    let cfg = StateConfig {
        max_ctx: 100,
        max_seqs: 0,
    };
    let err = StateManager::new(cfg, vec![kv_all()], u64::MAX).unwrap_err();
    match err {
        StateError::InvalidConfig { problems } => {
            assert!(problems.len() >= 2, "{problems:?}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }

    // Undersized pools are refused with required and shortfall.
    let cfg = StateConfig {
        max_ctx: 64,
        max_seqs: 1,
    };
    let err = StateManager::new(cfg, vec![kv_all()], 1).unwrap_err();
    match err {
        StateError::InvalidConfig { problems } => {
            assert!(
                problems.iter().any(|p| p.reason.contains("shortfall")),
                "{problems:?}"
            );
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// Oversized prompts and sequence caps are typed at `new_seq`.
#[test]
fn new_seq_limits_are_typed() {
    let mut m = manager_for(
        StateConfig {
            max_ctx: 64,
            max_seqs: 1,
        },
        &[kv_all()],
    );
    let big = vec![0; 65];
    let err = m.new_seq(&big).unwrap_err();
    assert!(matches!(err, StateError::ReserveTooLarge { .. }), "{err:?}");

    let _ = m.new_seq(&[]).unwrap();
    let err = m.new_seq(&[]).unwrap_err();
    assert!(matches!(err, StateError::SeqLimit { .. }), "{err:?}");
}

/// `free_seq` returns every block to the pool (Spec 3 §5).
#[test]
fn free_seq_returns_blocks_to_the_pool() {
    let mut m = tiny_manager();
    let total = m.budget().groups[0].total_blocks;
    let (a, _) = m.new_seq(&[]).unwrap();
    let toks = vec![1; 64];
    m.reserve(a, 64).unwrap();
    m.write_tokens(a, 0, &toks).unwrap();
    m.commit(a, 64).unwrap();
    assert_eq!(m.free_blocks(0).unwrap(), 0);
    m.free_seq(a).unwrap();
    assert_eq!(m.free_blocks(0).unwrap(), total);
    assert_eq!(m.stats().utilization, 0.0);
}
