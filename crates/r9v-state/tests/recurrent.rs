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
/// The scheduler re-runs the accepted prefix from verified A into working B
/// before committing, so the commit publishes it with a swap (Spec 3 §4.2).
#[test]
fn partial_accept_swaps_and_advances_ctx() {
    let mut m = manager_for(config_128(), &hybrid_specs());
    let (a, _) = m.new_seq(&[]).unwrap();

    m.reserve(a, 4).unwrap();
    m.commit(a, 2).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 2);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 1);
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 1);
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
}

/// Exact A/B bookkeeping across a mixed accept/reject history (Spec 3 §4.2,
/// §8): every `accepted > 0` flips the active buffer and counts one swap;
/// rejection flips nothing. The accepted prefix is re-run in place by the
/// scheduler before the commit — never reserved again at later positions.
#[test]
fn mixed_accept_history_flips_active_buffer_exactly() {
    let mut m = manager_for(config_128(), &hybrid_specs());
    let (a, _) = m.new_seq(&[]).unwrap();

    // Spec verify of 4 tokens, 2 accepted: swap publishes the re-run prefix.
    m.reserve(a, 4).unwrap();
    m.commit(a, 2).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 2);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 1);
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 1);
    assert_eq!(m.stats().swaps, 1);

    // Plain decode of 2 more tokens: another swap.
    m.reserve(a, 2).unwrap();
    m.commit(a, 2).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 4);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 0);
    assert_eq!(m.stats().swaps, 2);

    // Full rejection of 3 speculative tokens: checkpoint kept, no swap.
    m.reserve(a, 3).unwrap();
    m.commit(a, 0).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 4);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 0);
    assert_eq!(m.stats().swaps, 2);
    assert_eq!(m.stats().commits, 3);

    // The rejected tail stays allocated and is overwritten in place.
    let slots = m.reserve(a, 1).unwrap();
    assert_eq!(slots.start, 4);
    m.commit(a, 1).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 5);
    assert_eq!(m.recurrent_active(a, 1).unwrap(), 1);
    assert_eq!(m.stats().swaps, 3);
}

/// Fixed pools allocate the smallest free sequence slot per group, hold two
/// buffers per sequence, and reuse released slots deterministically
/// (Spec 3 §4.1, §6.3).
#[test]
fn fixed_slots_are_smallest_free_and_reused() {
    use r9v_state::StateConfig;
    let mut m = manager_for(
        StateConfig {
            max_ctx: 128,
            max_seqs: 4,
        },
        &hybrid_specs(),
    );
    // Groups: 0 paged, 1 recurrent, 2 conv.
    assert_eq!(m.free_slots(1).unwrap(), 4);
    assert_eq!(m.free_slots(2).unwrap(), 4);

    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();
    assert_eq!(m.fixed_slot(a, 1).unwrap(), 0);
    assert_eq!(m.fixed_slot(a, 2).unwrap(), 0);
    assert_eq!(m.fixed_slot(b, 1).unwrap(), 1);
    assert_eq!(m.fixed_slot(b, 2).unwrap(), 1);
    assert_eq!(m.free_slots(1).unwrap(), 2);

    // Two buffers per sequence: buffer ids `2 * slot + parity`, active and
    // working are the two halves of the owned slot.
    assert_eq!(m.recurrent_buffers(a, 1).unwrap(), (0, 1));
    assert_eq!(m.recurrent_buffers(b, 1).unwrap(), (2, 3));
    m.reserve(a, 1).unwrap();
    m.commit(a, 1).unwrap();
    assert_eq!(m.recurrent_buffers(a, 1).unwrap(), (1, 0));
    // Paged groups hold no sequence slots.
    assert!(m.fixed_slot(a, 0).is_err());

    // Release reuses the smallest slot id; buffer offsets are contiguous.
    let off0 = m.fixed_buffer_offset(1, 0).unwrap();
    let off1 = m.fixed_buffer_offset(1, 1).unwrap();
    assert!(off1 > off0);
    assert_eq!(m.fixed_buffer_offset(1, 2).unwrap() - off1, off1 - off0);
    m.free_seq(a).unwrap();
    assert_eq!(m.free_slots(1).unwrap(), 3);
    let (c, _) = m.new_seq(&[]).unwrap();
    assert_eq!(m.fixed_slot(c, 1).unwrap(), 0);
    assert_eq!(m.recurrent_buffers(c, 1).unwrap(), (0, 1));
    // Out-of-range buffers fail closed before any arithmetic.
    assert!(m.fixed_buffer_offset(1, 8).is_err());
    assert!(m.fixed_buffer_offset(0, 0).is_err());
}

/// `new_seq` is atomic across fixed groups: a refused sequence takes no
/// slot, consumes no id, and leaves every pool untouched.
#[test]
fn new_seq_refusal_takes_no_slots_and_consumes_no_id() {
    use r9v_state::{StateConfig, StateError};
    let mut m = manager_for(
        StateConfig {
            max_ctx: 64,
            max_seqs: 1,
        },
        &hybrid_specs(),
    );
    let (a, _) = m.new_seq(&[]).unwrap();
    assert_eq!(a.as_u64(), 0);
    let blocks_before = m.free_blocks(0).unwrap();

    let err = m.new_seq(&[]).unwrap_err();
    assert!(matches!(err, StateError::SeqLimit { .. }), "{err:?}");
    assert_eq!(m.free_slots(1).unwrap(), 0);
    assert_eq!(m.free_slots(2).unwrap(), 0);
    assert_eq!(m.free_blocks(0).unwrap(), blocks_before);

    // No id was consumed by the refusal: the next sequence takes id 1.
    m.free_seq(a).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();
    assert_eq!(b.as_u64(), 1);
    assert_eq!(m.fixed_slot(b, 1).unwrap(), 0);
}
