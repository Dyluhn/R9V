// SPDX-License-Identifier: Apache-2.0
//! Comprehensive contract cohesion tests for Card A1.15 (Spec 1 §2.5, Spec 3 §2-§6, Spec 14 §2).
//!
//! Proves:
//! - Canonical [`BatchMeta`] output preserves scalar positions, MRoPE triplets, and optional [`TreeMask`].
//! - Exact row-major indexing: `meta.slot(g, t)`, `meta.block(g, s, b)`, `meta.window(g, s)`.
//! - Window eviction leaves absolute logical block holes marked with [`BLOCK_TABLE_SENTINEL`].
//! - One shared sentinel definition: [`BLOCK_SENTINEL`] is a compatibility alias for [`BLOCK_TABLE_SENTINEL`].
//! - Device sequence ID boundary handling per SI-40: sequence creation and batch_meta fail before mutation on overflow.
//! - Centralized double-buffered byte accounting for Recurrent and ConvWindow.
//! - Deterministic grouping and nonaliasing of inequivalent state specifications.
//! - All public typed errors fail closed without panics.

mod common;

use common::{config_128, kv_all, kv_window, manager_for};
use r9v_common::SeqId;
use r9v_state::{
    group_layers, CacheDtype, LayerGroup, Positions, Retain, StateConfig, StateError, StateSpec,
    TreeMask, BLOCK_SENTINEL, BLOCK_TABLE_SENTINEL, SLOT_NONE,
};

fn multi_group_specs() -> Vec<StateSpec> {
    vec![
        StateSpec::KvPaged {
            hkv: 4,
            d: 32,
            dv: 32,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        },
        StateSpec::KvLatent {
            latent: 64,
            rope: 32,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        },
        StateSpec::Recurrent {
            h: 4,
            d: 16,
            dv: 16,
        },
        StateSpec::ConvWindow { c: 64, w: 4 },
        StateSpec::KvPaged {
            hkv: 2,
            d: 32,
            dv: 32,
            cache: CacheDtype::F16,
            retain: Retain::Window { w: 64 },
        },
    ]
}

/// Proves canonical `BatchMeta` preserves scalar sequential positions and exact row-major layouts.
#[test]
fn test_canonical_batch_meta_scalar_exactness() {
    let specs = multi_group_specs();
    let mut m = manager_for(config_128(), &specs);
    assert_eq!(m.groups().len(), 5);

    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 32).unwrap();
    m.reserve(b, 16).unwrap();

    let meta = m.batch_meta(&[a, b], &[32, 16]).unwrap();

    assert_eq!(meta.num_groups(), 5);
    assert_eq!(meta.num_seqs(), 2);
    assert_eq!(meta.total_tokens(), 48);
    assert_eq!(meta.max_blocks(), 4);
    assert_eq!(meta.query_len(), &[32, 16]);
    assert_eq!(meta.ctx_len(), &[0, 0]);
    assert_eq!(meta.seq_ids(), &[a.as_u64() as u32, b.as_u64() as u32]);

    // Scalar positions are preserved exactly.
    let expected_positions: Vec<u32> = (0..32).chain(0..16).collect();
    assert_eq!(
        meta.positions(),
        &Positions::PerToken(expected_positions.clone())
    );

    // Row-major slot_map [G, T] indexing matches flat buffer.
    assert_eq!(meta.slot_map().len(), 5 * 48);
    for g in 0..5u32 {
        for t in 0..48u32 {
            let flat_idx = (g as usize) * 48 + (t as usize);
            assert_eq!(meta.slot(g, t), meta.slot_map()[flat_idx]);
        }
    }

    // Groups 2 (Recurrent) and 3 (ConvWindow) carry SLOT_NONE for every token.
    for t in 0..48u32 {
        assert_eq!(meta.slot(2, t), SLOT_NONE);
        assert_eq!(meta.slot(3, t), SLOT_NONE);
    }

    // Row-major block_table [G, S, max_blocks] matches flat buffer.
    assert_eq!(meta.block_table().len(), 5 * 2 * 4);
    for g in 0..5u32 {
        for s in 0..2u32 {
            for b in 0..4u32 {
                let flat_idx = ((g as usize) * 2 + (s as usize)) * 4 + (b as usize);
                assert_eq!(meta.block(g, s, b), meta.block_table()[flat_idx]);
            }
        }
    }

    // Row-major window_start [G, S] matches flat buffer.
    assert_eq!(meta.window_start().len(), 5 * 2);
    for g in 0..5u32 {
        for s in 0..2u32 {
            let flat_idx = (g as usize) * 2 + (s as usize);
            assert_eq!(meta.window(g, s), meta.window_start()[flat_idx]);
        }
    }
}

/// Proves MRoPE positions survive losslessly in canonical `BatchMeta`.
#[test]
fn test_canonical_batch_meta_mrope_exactness() {
    let specs = multi_group_specs();
    let mut m = manager_for(config_128(), &specs);

    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();

    let triplets = vec![[0, 0, 0], [1, 0, 1], [2, 1, 0], [3, 1, 1]];
    let mrope_positions = Positions::Mrope(triplets.clone());

    let meta = m
        .batch_meta_with_options(&[a], &[4], Some(mrope_positions), None)
        .expect("batch_meta with MRoPE must succeed");

    assert_eq!(meta.positions(), &Positions::Mrope(triplets));
}

/// Proves speculative TreeMask survives losslessly in canonical `BatchMeta`.
#[test]
fn test_canonical_batch_meta_tree_mask_exactness() {
    let specs = multi_group_specs();
    let mut m = manager_for(config_128(), &specs);

    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 3).unwrap();

    // Speculative tree: 3 tokens with parent relationships: 0 is root, 1 child of 0, 2 child of 0.
    let tree = TreeMask::new(
        vec![-1, 0, 0],
        3,
        vec![true, false, false, true, true, false, true, false, true],
    )
    .expect("valid tree mask");

    let meta = m
        .batch_meta_with_tree(&[a], &[3], Some(tree.clone()))
        .expect("batch_meta with tree mask must succeed");

    assert_eq!(meta.tree(), Some(&tree));
}

/// Proves sentinel unification and absolute hole preservation after window eviction.
#[test]
fn test_sentinel_unification_and_holes() {
    // Verify one shared sentinel value.
    assert_eq!(BLOCK_SENTINEL, BLOCK_TABLE_SENTINEL);
    assert_eq!(BLOCK_SENTINEL, u32::MAX);

    let cfg = StateConfig {
        max_ctx: 128,
        max_seqs: 2,
    };
    let mut m = manager_for(cfg, &[kv_window(32)]);
    let (a, _) = m.new_seq(&[]).unwrap();

    // Commit 64 tokens in 32-token increments:
    // Chunk 0 allocates block 0.
    m.reserve(a, 32).unwrap();
    m.commit(a, 32).unwrap();
    // Chunk 1 allocates block 1, window eviction releases block 0.
    m.reserve(a, 32).unwrap();
    m.commit(a, 32).unwrap();

    // Query next 8 tokens (allocates block 0 at index 2).
    m.reserve(a, 8).unwrap();
    let meta = m.batch_meta(&[a], &[8]).unwrap();

    // Index 0 was evicted -> must be BLOCK_TABLE_SENTINEL hole.
    assert_eq!(meta.block(0, 0, 0), BLOCK_TABLE_SENTINEL);
    // Index 1 is retained block 1.
    assert_eq!(meta.block(0, 0, 1), 1);
    // Index 2 is newly allocated block 0.
    assert_eq!(meta.block(0, 0, 2), 0);
    // Index 3 is unallocated -> sentinel.
    assert_eq!(meta.block(0, 0, 3), BLOCK_TABLE_SENTINEL);

    m.commit(a, 8).unwrap();
}

/// Proves sequence ID boundary handling per SI-40 without truncation, silent rollover, or leak.
#[test]
fn test_seq_id_boundary_and_overflow_fails_before_mutation() {
    let mut m = manager_for(config_128(), &[kv_all()]);

    // Set next_seq to u32::MAX. This sequence should be allocated with ID u32::MAX.
    m.set_next_seq_for_test(u32::MAX as u64);
    let (seq_max, _) = m.new_seq(&[]).expect("u32::MAX sequence ID must succeed");
    assert_eq!(seq_max.as_u64(), u32::MAX as u64);

    // Next allocation exceeds u32::MAX and must fail BEFORE mutating any state.
    let before_next_seq = m.next_seq_for_test();

    let err = m.new_seq(&[]).unwrap_err();
    assert!(
        matches!(err, StateError::SeqIdOverflow { seq, max } if seq == u32::MAX as u64 + 1 && max == u32::MAX),
        "got {err:?}"
    );

    // State must be completely untouched.
    assert_eq!(m.next_seq_for_test(), before_next_seq);

    // Attempting batch_meta with a sequence ID > u32::MAX fails with typed error before building.
    let huge_seq = SeqId::new(u32::MAX as u64 + 5);
    let err = m.batch_meta(&[huge_seq], &[1]).unwrap_err();
    assert!(
        matches!(
            err,
            StateError::InvalidBatch { .. } | StateError::SeqIdOverflow { .. }
        ),
        "got {err:?}"
    );
}

/// Proves centralized double-buffered per-sequence byte accounting for Recurrent and ConvWindow.
#[test]
fn test_recurrent_and_conv_double_buffered_accounting() {
    let rec = StateSpec::Recurrent {
        h: 4,
        d: 16,
        dv: 16,
    };
    // Recurrent: h * d * dv * 4 = 4096 single slot, 8192 double buffered per sequence.
    assert_eq!(rec.slot_bytes().unwrap(), 4096);
    assert_eq!(rec.per_seq_bytes().unwrap(), 8192);
    assert_eq!(rec.state_per_seq_bytes().unwrap(), 8192);
    assert_eq!(rec.per_token_bytes().unwrap(), 0);

    let conv = StateSpec::ConvWindow { c: 256, w: 4 };
    // ConvWindow: (w - 1) * c * 2 = 1536 single slot, 3072 double buffered per sequence.
    assert_eq!(conv.slot_bytes().unwrap(), 1536);
    assert_eq!(conv.per_seq_bytes().unwrap(), 3072);
    assert_eq!(conv.state_per_seq_bytes().unwrap(), 3072);
    assert_eq!(conv.per_token_bytes().unwrap(), 0);

    let paged = StateSpec::KvPaged {
        hkv: 4,
        d: 32,
        dv: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    };
    assert_eq!(paged.slot_bytes().unwrap(), 0);
    assert_eq!(paged.per_seq_bytes().unwrap(), 0);
    assert_eq!(paged.per_token_bytes().unwrap(), 4 * (64 + 4));

    // LayerGroup slots_bytes_per_seq equals spec.per_seq_bytes() * layers.len().
    let group = LayerGroup {
        index: 0,
        spec: conv,
        layers: vec![0, 1, 2],
    };
    assert_eq!(group.slots_bytes_per_seq().unwrap(), 3072 * 3);
}

/// Proves deterministic grouping and nonaliasing of inequivalent specs.
#[test]
fn test_deterministic_grouping_and_nonaliasing() {
    let s1 = StateSpec::KvPaged {
        hkv: 4,
        d: 32,
        dv: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    };
    let s2 = StateSpec::KvPaged {
        hkv: 4,
        d: 32,
        dv: 32,
        cache: CacheDtype::I8, // different cache dtype
        retain: Retain::All,
    };
    let s3 = StateSpec::KvPaged {
        hkv: 4,
        d: 32,
        dv: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::Window { w: 64 }, // different retain
    };
    let s4 = StateSpec::Recurrent {
        h: 4,
        d: 16,
        dv: 16,
    };
    let s5 = StateSpec::ConvWindow { c: 64, w: 4 };

    let specs = vec![s1, s2, s3, s4, s5, s1, s4];
    let groups = group_layers(&specs);

    // There are 5 distinct specs, so 5 groups.
    assert_eq!(groups.len(), 5);
    assert_eq!(groups[0].spec, s1);
    assert_eq!(groups[0].layers, vec![0, 5]);
    assert_eq!(groups[1].spec, s2);
    assert_eq!(groups[1].layers, vec![1]);
    assert_eq!(groups[2].spec, s3);
    assert_eq!(groups[2].layers, vec![2]);
    assert_eq!(groups[3].spec, s4);
    assert_eq!(groups[3].layers, vec![3, 6]);
    assert_eq!(groups[4].spec, s5);
    assert_eq!(groups[4].layers, vec![4]);

    // None of the inequivalent specs aliased into the same group.
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            assert_ne!(groups[i].spec, groups[j].spec);
        }
    }
}

/// Proves public typed errors without panics on all invalid inputs.
#[test]
fn test_typed_errors_no_panics() {
    // Zero dimensions return InvalidConfig.
    let bad_conv = StateSpec::ConvWindow { c: 0, w: 4 };
    assert!(matches!(
        bad_conv.per_seq_bytes().unwrap_err(),
        StateError::InvalidConfig { .. }
    ));

    let bad_rec = StateSpec::Recurrent { h: 0, d: 8, dv: 8 };
    assert!(matches!(
        bad_rec.per_seq_bytes().unwrap_err(),
        StateError::InvalidConfig { .. }
    ));

    // Retain sink without window is typed error.
    let err = Retain::from_window_sinks(None, 4).unwrap_err();
    assert!(matches!(err, StateError::InvalidConfig { .. }));
}
