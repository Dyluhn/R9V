// SPDX-License-Identifier: Apache-2.0
//! Recurrent/conv A/B double-buffer tests (Spec 3 §4.2, §8): plain decode
//! swaps, full verify swaps, partial verify defers the swap and records the
//! re-run, and the recomputed path converges with the plain path.

mod common;

use common::{config_128, manager_for};
use r9v_state::{CacheDtype, Retain, StateSpec};

fn hybrid_specs() -> Vec<StateSpec> {
    vec![
        StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        },
        StateSpec::Recurrent { h: 2, d: 8, dv: 8 },
        StateSpec::ConvWindow { c: 4, w: 4 },
    ]
}

/// Plain decode (`query_len = 1`): full accept swaps A↔B on recurrent/conv
/// groups only (Spec 3 §4.2).
#[test]
fn full_accept_swaps_recurrent_slots_only() {
    let mut m = manager_for(config_128(), &hybrid_specs());
    let (a, _) = m.new_seq(&[]).unwrap();

    assert_eq!(m.recurrent_active(a, 1).unwrap(), 0);
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 0);
    m.reserve(a, 1).unwrap();
    m.commit(a, 1).unwrap();
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 1);
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 1);
    // Paged group 0 has no parity concept; it stays at its initial value.
    assert_eq!(m.recurrent_active(a, 0).unwrap(), 0);
    assert_eq!(m.stats().swaps, 1);

    m.reserve(a, 1).unwrap();
    m.commit(a, 1).unwrap();
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 0);
    assert_eq!(m.stats().swaps, 2);
}

/// Partial verify (`a < k + 1`): no swap, and the accepted prefix is recorded
/// for the scheduler's re-run (Spec 3 §4.2).
#[test]
fn partial_accept_defers_swap_and_records_rerun() {
    let mut m = manager_for(config_128(), &hybrid_specs());
    let (a, _) = m.new_seq(&[]).unwrap();

    m.reserve(a, 4).unwrap();
    m.commit(a, 2).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 2);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 0);
    assert_eq!(m.recompute_pending(a).unwrap(), 2);
    assert_eq!(m.stats().swaps, 0);

    // The scheduler re-runs the accepted prefix; the full re-commit swaps.
    m.reserve(a, 2).unwrap();
    assert_eq!(m.recompute_pending(a).unwrap(), 0);
    m.commit(a, 2).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 4);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 1);
    assert_eq!(m.stats().swaps, 1);
}

/// Full rejection keeps the checkpoint: no swap, nothing pending.
#[test]
fn full_rejection_keeps_checkpoint() {
    let mut m = manager_for(config_128(), &hybrid_specs());
    let (a, _) = m.new_seq(&[]).unwrap();

    m.reserve(a, 3).unwrap();
    m.commit(a, 0).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 0);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 0);
    assert_eq!(m.recompute_pending(a).unwrap(), 0);
}

/// Spec 3 §8 double-buffer law: after `commit(a < k+1)` and recompute, the
/// active slot matches a sequence that processed the same tokens without
/// speculation.
#[test]
fn recomputed_path_converges_with_plain_path() {
    let mut m = manager_for(config_128(), &hybrid_specs());
    let (speculative, _) = m.new_seq(&[]).unwrap();
    let (plain, _) = m.new_seq(&[]).unwrap();

    m.reserve(speculative, 3).unwrap();
    m.commit(speculative, 2).unwrap();
    m.reserve(speculative, 2).unwrap();
    m.commit(speculative, 2).unwrap();

    m.reserve(plain, 4).unwrap();
    m.commit(plain, 4).unwrap();

    assert_eq!(m.ctx_len(speculative).unwrap(), 4);
    assert_eq!(m.ctx_len(plain).unwrap(), 4);
    assert_eq!(
        m.recurrent_active(speculative, 1).unwrap(),
        m.recurrent_active(plain, 1).unwrap()
    );
    assert_eq!(
        m.recurrent_active(speculative, 2).unwrap(),
        m.recurrent_active(plain, 2).unwrap()
    );
}
