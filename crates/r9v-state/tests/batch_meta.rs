// SPDX-License-Identifier: Apache-2.0
//! Multi-group `BatchMeta` shape and value tests (Spec 1 §2.5, Spec 3 §3.3,
//! §6.1): one table per layer-group, fixed `[G, ...]` shapes.

mod common;

use common::{config_128, manager_for};
use r9v_state::{CacheDtype, Retain, StateSpec, BLOCK_SENTINEL, SLOT_NONE};

fn hybrid_specs() -> Vec<StateSpec> {
    let dense = StateSpec::KvPaged {
        hkv: 4,
        d: 32,
        dv: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    };
    let windowed = StateSpec::KvPaged {
        hkv: 2,
        d: 32,
        dv: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::Window { w: 64 },
    };
    vec![
        dense,
        dense,
        StateSpec::Recurrent {
            h: 4,
            d: 16,
            dv: 16,
        },
        StateSpec::ConvWindow { c: 8, w: 5 },
        windowed,
    ]
}

/// Four layer-groups (dense ×2 layers, recurrent, conv, windowed) produce
/// fixed `[G, ...]` shapes with per-group tables (Spec 3 §6.1).
#[test]
fn multi_group_batch_meta_shapes_and_values() {
    let specs = hybrid_specs();
    let mut m = manager_for(config_128(), &specs);
    assert_eq!(m.groups().len(), 4);
    assert_eq!(m.groups()[0].layers, vec![0, 1]);

    let (a, matched_a) = m.new_seq(&[]).unwrap();
    let (b, matched_b) = m.new_seq(&[]).unwrap();
    assert_eq!(matched_a, 0);
    assert_eq!(matched_b, 0);

    // a: 40 tokens (blocks 0, 1 in paged groups), b: 10 tokens (block 2).
    m.reserve(a, 40).unwrap();
    m.reserve(b, 10).unwrap();
    let meta = m.batch_meta(&[a, b], &[40, 10]).unwrap();

    assert_eq!(meta.num_groups(), 4);
    assert_eq!(meta.num_seqs(), 2);
    assert_eq!(meta.total_tokens(), 50);
    assert_eq!(meta.max_blocks(), 4);
    assert_eq!(meta.query_len(), &[40, 10]);
    assert_eq!(meta.ctx_len(), &[0, 0]);
    assert_eq!(
        meta.positions(),
        &r9v_state::Positions::PerToken((0..40).chain(0..10).collect::<Vec<u32>>())
    );

    // slot_map is [G, T].
    assert_eq!(meta.slot_map().len(), 4 * 50);
    // Dense group: first token of `a` sits in block 0 lane 0; token 32 of
    // `a` sits in block 1 lane 0; first token of `b` sits in block 2 lane 0.
    assert_eq!(meta.slot(0, 0), 0);
    assert_eq!(meta.slot(0, 32), 32);
    assert_eq!(meta.slot(0, 40), 2 * 32);
    // Windowed group (index 3) allocates its own pool: same layout.
    assert_eq!(meta.slot(3, 0), 0);
    assert_eq!(meta.slot(3, 40), 2 * 32);
    // Recurrent/conv groups carry SLOT_NONE (no per-token slots, §4.2).
    for t in 0..50 {
        assert_eq!(meta.slot(1, t), SLOT_NONE);
        assert_eq!(meta.slot(2, t), SLOT_NONE);
    }

    // block_table is [G, S, max_blocks], ascending ids then sentinel.
    assert_eq!(meta.block_table().len(), 4 * 2 * 4);
    assert_eq!(meta.block(0, 0, 0), 0);
    assert_eq!(meta.block(0, 0, 1), 1);
    assert_eq!(meta.block(0, 0, 2), BLOCK_SENTINEL);
    assert_eq!(meta.block(0, 0, 3), BLOCK_SENTINEL);

    assert_eq!(meta.block(0, 1, 0), 2);
    assert_eq!(meta.block(0, 1, 1), BLOCK_SENTINEL);
    assert_eq!(meta.block(0, 1, 2), BLOCK_SENTINEL);
    assert_eq!(meta.block(0, 1, 3), BLOCK_SENTINEL);

    for g in [1, 2] {
        for s in [0, 1] {
            for b in 0..4 {
                assert_eq!(meta.block(g, s, b), BLOCK_SENTINEL);
            }
        }
    }

    // window_start is [G, S]: 0 everywhere at ctx 0.
    assert_eq!(meta.window_start().len(), 4 * 2);
    for g in 0..4 {
        for s in 0..2 {
            assert_eq!(meta.window(g, s), 0);
        }
    }
}

/// After window eviction, each block id sits at its absolute logical block
/// index with sentinel holes where eviction released blocks — never
/// compacted to the front (Spec 3 §3.3, §3.5; SI-17).
#[test]
fn block_table_uses_absolute_indices_with_sentinel_holes() {
    use common::{kv_window, manager_for};
    use r9v_state::{StateConfig, BLOCK_SENTINEL};
    // max_ctx 256: 8-wide rows; w = 64 so two blocks survive eviction.
    let config = StateConfig {
        max_ctx: 256,
        max_seqs: 4,
    };
    let mut m = manager_for(config, &[kv_window(64)]);
    let (a, _) = m.new_seq(&[]).unwrap();

    // Four commits of 32 take pool ids 0, 1, 2, then reuse 0 at index 3;
    // committing 128 with a 64 window evicts indices 0 and 1, holding
    // index 2 (id 2) and index 3 (id 0).
    for chunk in 0..4 {
        let base = chunk * 32;
        let toks: Vec<u32> = (base..base + 32).collect();
        let ctx = m.ctx_len(a).unwrap();
        m.reserve(a, 32).unwrap();
        m.write_tokens(a, ctx, &toks).unwrap();
        m.commit(a, 32).unwrap();
    }
    assert_eq!(m.ctx_len(a).unwrap(), 128);

    // Reserve the next block: smallest free id (1) lands at absolute index 4.
    m.reserve(a, 8).unwrap();
    let meta = m.batch_meta(&[a], &[8]).unwrap();
    assert_eq!(meta.max_blocks(), 8);
    let expected_row = [
        BLOCK_SENTINEL,
        BLOCK_SENTINEL,
        2,
        0,
        1,
        BLOCK_SENTINEL,
        BLOCK_SENTINEL,
        BLOCK_SENTINEL,
    ];
    for (b, &expected) in expected_row.iter().enumerate() {
        assert_eq!(meta.block(0, 0, b as u32), expected);
    }
    // Slots follow the absolute ids, not table positions: positions 128..136
    // sit in pool block 1, so the flattened slots are 32..40.
    for t in 0..8 {
        assert_eq!(meta.slot(0, t), 32 + t);
    }
    assert_eq!(
        meta.positions(),
        &r9v_state::Positions::PerToken((128..136).collect())
    );
    m.commit(a, 8).unwrap();
}

/// After windowed commits, `BatchMeta.window_start` tracks the window while
/// `All` groups still report 0 (Spec 3 §3.5).
#[test]
fn window_start_advances_only_for_windowed_groups() {
    let specs = hybrid_specs();
    let mut m = manager_for(config_128(), &specs);
    let (a, _) = m.new_seq(&[]).unwrap();

    for chunk in 0..3 {
        let base = chunk * 32;
        let toks: Vec<u32> = (base..base + 32).collect();
        let ctx = m.ctx_len(a).unwrap();
        m.reserve(a, 32).unwrap();
        m.write_tokens(a, ctx, &toks).unwrap();
        m.commit(a, 32).unwrap();
    }
    assert_eq!(m.ctx_len(a).unwrap(), 96);

    m.reserve(a, 1).unwrap();
    let meta = m.batch_meta(&[a], &[1]).unwrap();
    // Groups: 0 dense All, 1 recurrent, 2 conv, 3 windowed w=64.
    assert_eq!(meta.window(0, 0), 0);
    assert_eq!(meta.window(1, 0), 0);
    assert_eq!(meta.window(2, 0), 0);
    assert_eq!(meta.window(3, 0), 96 - 64);
    m.commit(a, 1).unwrap();
}
