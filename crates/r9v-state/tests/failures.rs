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

/// `commit` with no outstanding reservation is a typed error — even a zero
/// accept with no open tail (Spec 3 §3.6 pairs every commit with a reserve).
#[test]
fn commit_without_reservation_is_typed() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();

    let err = m.commit(a, 0).unwrap_err();
    assert!(matches!(err, StateError::NoReservation { .. }), "{err:?}");

    m.reserve(a, 4).unwrap();
    m.commit(a, 4).unwrap();
    let err = m.commit(a, 0).unwrap_err();
    assert!(matches!(err, StateError::NoReservation { .. }), "{err:?}");
    // The refused commits mutated nothing.
    assert_eq!(m.ctx_len(a).unwrap(), 4);
    assert_eq!(m.tail_len(a).unwrap(), 0);
}

/// `write_tokens` rejects writes before `ctx_len` (verified history) as well
/// as writes past the reserved end.
#[test]
fn writes_before_ctx_len_are_rejected() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();
    let toks = vec![1; 8];
    m.reserve(a, 8).unwrap();
    m.write_tokens(a, 0, &toks).unwrap();
    m.commit(a, 8).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 8);

    // Rewriting verified history is out of range.
    let err = m.write_tokens(a, 0, &[9]).unwrap_err();
    assert!(matches!(err, StateError::OutOfRange { .. }), "{err:?}");
    let err = m.write_tokens(a, 7, &[9]).unwrap_err();
    assert!(matches!(err, StateError::OutOfRange { .. }), "{err:?}");

    // The open reservation is still writable exactly at `ctx_len`.
    m.reserve(a, 4).unwrap();
    m.write_tokens(a, 8, &[10, 11, 12, 13]).unwrap();
    let err = m.write_tokens(a, 7, &[9]).unwrap_err();
    assert!(matches!(err, StateError::OutOfRange { .. }), "{err:?}");
    m.commit(a, 4).unwrap();
    assert_eq!(m.read_token(a, 0, 8).unwrap(), Some(10));
}

/// The same sequence twice in one batch is a typed error, not a doubled row.
#[test]
fn duplicate_sequences_in_batch_are_rejected() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();

    let err = m.batch_meta(&[a, a], &[2, 2]).unwrap_err();
    match err {
        StateError::InvalidBatch { detail } => {
            assert!(detail.contains("duplicate"), "{detail}");
        }
        other => panic!("expected InvalidBatch, got {other:?}"),
    }
}

/// Public byte geometry rejects invalid dims/policies instead of computing a
/// plausible value (a zero head count must not silently size an empty pool).
#[test]
fn byte_geometry_rejects_invalid_dims() {
    use r9v_state::{CacheDtype, Retain, StateSpec};
    let zero_hkv = StateSpec::KvPaged {
        hkv: 0,
        d: 16,
        dv: 16,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    };
    assert!(matches!(
        zero_hkv.per_token_bytes().unwrap_err(),
        StateError::InvalidConfig { .. }
    ));
    let zero_h = StateSpec::Recurrent { h: 0, d: 8, dv: 8 };
    assert!(matches!(
        zero_h.slot_bytes().unwrap_err(),
        StateError::InvalidConfig { .. }
    ));
    assert!(matches!(
        zero_h.per_token_bytes().unwrap_err(),
        StateError::InvalidConfig { .. }
    ));
    let zero_w = StateSpec::ConvWindow { c: 4, w: 0 };
    assert!(matches!(
        zero_w.slot_bytes().unwrap_err(),
        StateError::InvalidConfig { .. }
    ));
    let bad_window = StateSpec::KvPaged {
        hkv: 2,
        d: 16,
        dv: 16,
        cache: CacheDtype::E4M3,
        retain: Retain::Window { w: 0 },
    };
    assert!(matches!(
        bad_window.per_token_bytes().unwrap_err(),
        StateError::InvalidConfig { .. }
    ));
    // Valid specs still compute exact values.
    assert_eq!(kv_all().per_token_bytes().unwrap(), 2 * (32 + 4));
}

/// Batch token totals are checked, not saturated: an oversized pool lets two
/// large reservations coexist, and their combined total still hits the cap.
#[test]
fn batch_token_total_cap_is_checked() {
    use r9v_state::{group_layers, required_pool_bytes};
    let config = StateConfig {
        max_ctx: 1 << 20,
        max_seqs: 4,
    };
    let groups = group_layers(&[kv_all()]).unwrap();
    let required = required_pool_bytes(config, &groups).expect("pool math is exact");
    let mut m = StateManager::new(config, vec![kv_all()], required * 2).expect("valid config");

    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 600_000).unwrap();
    m.reserve(b, 600_000).unwrap();
    let err = m.batch_meta(&[a, b], &[600_000, 600_000]).unwrap_err();
    match err {
        StateError::InvalidBatch { detail } => {
            assert!(detail.contains("exceed cap"), "{detail}");
        }
        other => panic!("expected InvalidBatch, got {other:?}"),
    }
}

/// Failed `new_seq` calls leave no trace: no slots taken, no id consumed.
#[test]
fn failed_new_seq_leaves_no_trace() {
    use r9v_state::StateConfig;
    let mut m = manager_for(
        StateConfig {
            max_ctx: 64,
            max_seqs: 2,
        },
        &[kv_all()],
    );
    let big = vec![0; 65];
    let err = m.new_seq(&big).unwrap_err();
    assert!(matches!(err, StateError::ReserveTooLarge { .. }), "{err:?}");

    // The refused prompt consumed no sequence id.
    let (a, _) = m.new_seq(&[]).unwrap();
    assert_eq!(a.as_u64(), 0);
    let (b, _) = m.new_seq(&[]).unwrap();
    assert_eq!(b.as_u64(), 1);
    let err = m.new_seq(&[]).unwrap_err();
    assert!(matches!(err, StateError::SeqLimit { .. }), "{err:?}");
}

/// MAX_SLOT_BLOCKS admits at most 134_217_727 blocks so the highest possible
/// flattened slot is strictly less than `SLOT_NONE` (`u32::MAX`). A capacity
/// requesting one more block is rejected at construction without impractical allocation.
#[test]
fn max_slot_blocks_boundary_proves_no_slot_none_and_rejects_one_more_block() {
    use r9v_state::{group_layers, BLOCK_TOKENS, MAX_SLOT_BLOCKS, SLOT_NONE};

    assert_eq!(MAX_SLOT_BLOCKS, 134_217_727);
    let max_block_id = MAX_SLOT_BLOCKS - 1;
    assert_eq!(max_block_id, 134_217_726);

    // Mathematical coverage: every lane in the highest admitted block strictly precedes SLOT_NONE.
    for lane in 0..BLOCK_TOKENS {
        let slot = u64::from(max_block_id) * u64::from(BLOCK_TOKENS) + u64::from(lane);
        assert!(slot < u64::from(SLOT_NONE));
        assert_ne!(slot as u32, SLOT_NONE);
    }
    let highest_real_slot = max_block_id * BLOCK_TOKENS + (BLOCK_TOKENS - 1);
    assert_eq!(highest_real_slot, 4_294_967_263);
    assert_eq!(SLOT_NONE, u32::MAX);
    assert_eq!(u64::from(highest_real_slot) + 32, u64::from(SLOT_NONE));

    // One more block (134_217_728 blocks total) would have admitted block id 134_217_727,
    // whose lane 31 would equal SLOT_NONE (4_294_967_295 == u32::MAX).
    let would_collide_block = MAX_SLOT_BLOCKS;
    let collided_slot =
        u64::from(would_collide_block) * u64::from(BLOCK_TOKENS) + u64::from(BLOCK_TOKENS - 1);
    assert_eq!(collided_slot, u64::from(SLOT_NONE));

    // One-more-block capacity is rejected without impractical allocation.
    let cfg = StateConfig {
        max_ctx: 32,
        max_seqs: 1,
    };
    let spec = kv_all();
    let groups = group_layers(&[spec]).unwrap();
    let block_bytes = groups[0].block_bytes().expect("valid block bytes");
    let pool_bytes = (u64::from(MAX_SLOT_BLOCKS) + 1) * block_bytes;
    let err = StateManager::new(cfg, vec![spec], pool_bytes).unwrap_err();
    match err {
        StateError::InvalidConfig { problems } => {
            assert!(
                problems
                    .iter()
                    .any(|p| p.reason.contains("exceeds u32 slot_map range 134217727")),
                "got {problems:?}"
            );
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// Failed `free_seq` calls are atomic: state, statistics, and resources are unchanged on Err.
#[test]
fn free_seq_failure_leaves_all_state_and_stats_untouched() {
    let mut m = tiny_manager();
    let (a, _) = m.new_seq(&[]).unwrap();
    let toks = vec![3; 64];
    m.reserve(a, 64).unwrap();
    m.write_tokens(a, 0, &toks).unwrap();
    m.commit(a, 64).unwrap();

    let before_stats = m.stats();
    let before_budget = m.budget();
    let before_free = m.free_blocks(0).unwrap();

    // Calling free_seq on an unknown sequence fails and mutates nothing.
    use r9v_common::SeqId;
    let ghost = SeqId::new(9999);
    let err = m.free_seq(ghost).unwrap_err();
    assert!(matches!(err, StateError::UnknownSeq { .. }), "{err:?}");

    assert_eq!(m.ctx_len(a).unwrap(), 64);
    assert_eq!(m.free_blocks(0).unwrap(), before_free);
    assert_eq!(m.stats(), before_stats);
    assert_eq!(m.budget(), before_budget);

    // Clean free_seq works and releases all resources.
    m.free_seq(a).unwrap();
    assert_eq!(
        m.free_blocks(0).unwrap(),
        before_budget.groups[0].total_blocks
    );
}
