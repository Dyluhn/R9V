// SPDX-License-Identifier: Apache-2.0
//! Spec 3 §8 commit-semantics and retention-window tests against the
//! in-memory arena.

mod common;

use common::{config_128, kv_all, kv_window, manager_for, step_all};

/// Spec 3 §8: writing `k + 1` tokens and committing `a` keeps the verified
/// prefix bit-equal to a sequence that never speculated.
#[test]
fn commit_with_partial_accept_keeps_verified_prefix() {
    let mut m = manager_for(config_128(), &[kv_all()]);
    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();

    m.reserve(a, 5).unwrap();
    m.write_tokens(a, 0, &[10, 11, 12, 13, 14]).unwrap();
    m.commit(a, 3).unwrap();

    step_all(&mut m, b, &[10, 11, 12]);

    assert_eq!(m.ctx_len(a).unwrap(), 3);
    assert_eq!(m.ctx_len(b).unwrap(), 3);
    for pos in 0..3 {
        assert_eq!(
            m.read_token(a, 0, pos).unwrap(),
            m.read_token(b, 0, pos).unwrap(),
            "verified prefix must match at {pos}"
        );
    }
}

/// Spec 3 §3.6: over-reserved positions stay allocated and are overwritten by
/// the next reserve (rejection moves no data).
#[test]
fn over_reserved_positions_are_overwritten_by_next_reserve() {
    let mut m = manager_for(config_128(), &[kv_all()]);
    let (a, _) = m.new_seq(&[]).unwrap();

    m.reserve(a, 5).unwrap();
    m.write_tokens(a, 0, &[1, 2, 3, 4, 5]).unwrap();
    m.commit(a, 3).unwrap();

    // Next step overwrites the rejected tail in place.
    let slots = m.reserve(a, 4).unwrap();
    assert_eq!(slots.start, 3);
    m.write_tokens(a, 3, &[30, 31, 32, 33]).unwrap();
    m.commit(a, 4).unwrap();

    let expect = [1, 2, 3, 30, 31, 32, 33];
    for (pos, tok) in expect.iter().enumerate() {
        assert_eq!(
            m.read_token(a, 0, pos as u32).unwrap(),
            Some(*tok),
            "pos {pos}"
        );
    }
}

/// Spec 3 §3.5/§8: after commit, blocks older than the window are released,
/// `window_start` advances, and retained positions still read back.
#[test]
fn windowed_retention_releases_old_blocks_and_keeps_window() {
    let mut m = manager_for(config_128(), &[kv_window(64)]);
    let (a, _) = m.new_seq(&[]).unwrap();

    for chunk in 0..4 {
        let base = chunk * 32;
        let toks: Vec<u32> = (base..base + 32).collect();
        step_all(&mut m, a, &toks);
    }
    assert_eq!(m.ctx_len(a).unwrap(), 128);
    assert_eq!(m.window_start(a, 0).unwrap(), 64);

    // Only the two window blocks (indices 2, 3) are still held.
    assert_eq!(m.free_blocks(0).unwrap(), 2);

    // Evicted positions report absence; retained ones read back exactly.
    assert_eq!(m.read_token(a, 0, 0).unwrap(), None);
    assert_eq!(m.read_token(a, 0, 63).unwrap(), None);
    for pos in 64..128 {
        assert_eq!(m.read_token(a, 0, pos).unwrap(), Some(pos), "pos {pos}");
    }

    // Released blocks are reusable: a second sequence fills the whole pool.
    let (b, _) = m.new_seq(&[]).unwrap();
    let toks: Vec<u32> = (0..64).collect();
    step_all(&mut m, b, &toks);
    assert_eq!(m.ctx_len(b).unwrap(), 64);
}

/// Sink + window pins the first blocks while the window slides (Spec 3 §3.5).
#[test]
fn sink_blocks_stay_pinned_while_window_slides() {
    use r9v_state::{CacheDtype, Retain, StateSpec};
    let spec = StateSpec::KvPaged {
        hkv: 2,
        d: 16,
        dv: 16,
        cache: CacheDtype::E4M3,
        retain: Retain::SinkWindow { n: 32, w: 32 },
    };
    let mut m = manager_for(config_128(), &[spec]);
    let (a, _) = m.new_seq(&[]).unwrap();

    for chunk in 0..4 {
        let base = chunk * 32;
        let toks: Vec<u32> = (base..base + 32).collect();
        step_all(&mut m, a, &toks);
    }
    // Sink block 0 plus window block 3 are held; blocks 1, 2 were released.
    assert_eq!(m.free_blocks(0).unwrap(), 2);
    assert_eq!(m.window_start(a, 0).unwrap(), 96);
    assert_eq!(m.read_token(a, 0, 0).unwrap(), Some(0));
    assert_eq!(m.read_token(a, 0, 31).unwrap(), Some(31));
    assert_eq!(m.read_token(a, 0, 32).unwrap(), None);
    assert_eq!(m.read_token(a, 0, 127).unwrap(), Some(127));
}

/// Tree verify (Spec 3 §3.6): `compact` gathers the accepted path into
/// `ctx_len .. ctx_len + a`, then `commit(a)` verifies it.
#[test]
fn compact_gathers_accepted_path_then_commit_verifies() {
    let mut m = manager_for(config_128(), &[kv_all()]);
    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();

    // Scratch positions hold candidates; the accepted path is [2, 0].
    m.reserve(a, 4).unwrap();
    m.write_tokens(a, 0, &[50, 51, 52, 53]).unwrap();
    let op = m.compact(a, &[2, 0]).unwrap();
    assert_eq!(op.dst_start, 0);
    assert_eq!(op.len, 2);
    assert_eq!(op.src_positions, vec![2, 0]);
    m.commit(a, 2).unwrap();

    step_all(&mut m, b, &[52, 50]);
    for pos in 0..2 {
        assert_eq!(
            m.read_token(a, 0, pos).unwrap(),
            m.read_token(b, 0, pos).unwrap(),
            "compacted position {pos}"
        );
    }
}
