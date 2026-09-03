// SPDX-License-Identifier: Apache-2.0
//! Sequence lifecycle: `free_seq` removes the sequence and all its mirrors,
//! so long create/free histories reuse ids, blocks, and slots with no
//! unbounded tombstone growth (Spec 3 §5).

mod common;

use common::{config_128, kv_all, manager_for, step_all};
use r9v_state::{StateConfig, StateError};

/// Hundreds of create/reserve/commit/free cycles return every block and slot
/// to the pools with deterministic smallest-first reuse.
#[test]
fn long_create_free_history_reuses_everything_deterministically() {
    let mut m = manager_for(config_128(), &[kv_all()]);
    let total = m.budget().groups[0].total_blocks;
    assert_eq!(total, 4);

    for round in 0..500 {
        let toks = vec![(round as u32) % 251; 32];
        let (a, _) = m.new_seq(&[]).unwrap();
        step_all(&mut m, a, &toks);
        // Every round takes the same smallest-free block id 0.
        let meta = {
            m.reserve(a, 1).unwrap();
            let meta = m.batch_meta(&[a], &[1]).unwrap();
            m.commit(a, 1).unwrap();
            meta
        };
        assert_eq!(meta.block_table[0][0][0], 0);
        m.free_seq(a).unwrap();
        assert_eq!(m.free_blocks(0).unwrap(), total);
        // Freed ids are unknown immediately: no dead entries linger.
        assert!(matches!(
            m.ctx_len(a).unwrap_err(),
            StateError::UnknownSeq { .. }
        ));
    }
    // After 500 cycles the pools are whole and admission is unaffected.
    assert_eq!(m.free_blocks(0).unwrap(), total);
    assert_eq!(m.stats().commits, 500 * 2);
}

/// A full house of live sequences can be freed and refilled repeatedly:
/// the live cap never trips on dead entries and fixed slots recycle.
#[test]
fn full_house_free_and_refill_recycles_slots() {
    let recurrent = r9v_state::StateSpec::Recurrent { h: 2, d: 8, dv: 8 };
    let mut m = manager_for(
        StateConfig {
            max_ctx: 64,
            max_seqs: 2,
        },
        &[kv_all(), recurrent],
    );
    for _ in 0..100 {
        let (a, _) = m.new_seq(&[]).unwrap();
        let (b, _) = m.new_seq(&[]).unwrap();
        assert_eq!(m.free_slots(1).unwrap(), 0);
        // Third sequence still refused while the house is full.
        assert!(matches!(
            m.new_seq(&[]).unwrap_err(),
            StateError::SeqLimit { .. }
        ));
        step_all(&mut m, a, &[1; 32]);
        step_all(&mut m, b, &[2; 32]);
        m.free_seq(a).unwrap();
        m.free_seq(b).unwrap();
        assert_eq!(m.free_slots(1).unwrap(), 2);
        assert_eq!(m.free_blocks(0).unwrap(), 2);
    }
    // Slot ids recycle smallest-first after the churn.
    let (c, _) = m.new_seq(&[]).unwrap();
    assert_eq!(m.fixed_slot(c, 1).unwrap(), 0);
}
