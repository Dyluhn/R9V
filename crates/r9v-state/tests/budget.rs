// SPDX-License-Identifier: Apache-2.0
//! Budget and stats exactness (Spec 3 §5, §6.3): proportional paged splits
//! with equal aggregate block capacity, exact free-slot accounting,
//! explicit zero host lines, and accounted remainders.

mod common;

use common::{kv_all, kv_window, manager_for};
use r9v_state::{group_layers, required_pool_bytes, CacheDtype, Retain, StateConfig, StateManager};

/// Minimum pool gives every paged group exactly `max_ctx / 32` aggregate
/// blocks — not `max_seqs` full sequences — while fixed pools take
/// `max_seqs` slots each (Spec 3 §6.3).
#[test]
fn minimum_pool_gives_aggregate_max_ctx_per_paged_group() {
    let dense = kv_all();
    let windowed = kv_window(64);
    let recurrent = r9v_state::StateSpec::Recurrent { h: 2, d: 8, dv: 8 };
    let config = StateConfig {
        max_ctx: 128,
        max_seqs: 8,
    };
    let specs = vec![dense, windowed, recurrent];
    let groups = group_layers(&specs);
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    let m = StateManager::new(config, specs, pool).expect("minimum pool fits");

    // max_ctx 128 / 32 = 4 aggregate blocks per paged group.
    assert_eq!(m.budget().groups[0].total_blocks, 4);
    assert_eq!(m.budget().groups[0].free_blocks, 4);
    assert_eq!(m.budget().groups[1].total_blocks, 4);
    // Fixed group: max_seqs slots, no blocks.
    assert_eq!(m.budget().groups[2].total_blocks, 0);
    assert_eq!(m.budget().groups[2].total_slots, 8);
    assert_eq!(m.budget().groups[2].free_slots, 8);
    // Minimum pool leaves no remainder and no host bytes.
    assert_eq!(m.budget().unusable_bytes, 0);
    assert_eq!(m.budget().host_free, 0);
}

/// Oversized pools split paged bytes in proportion to block costs so every
/// paged group holds the same block count; the sub-block remainder is
/// reported explicitly and arena bases stay contiguous (Spec 3 §6.3).
#[test]
fn oversized_pool_splits_proportionally_with_accounted_remainder() {
    // Two paged groups with different block costs: 8-head dense vs 2-head.
    let big = r9v_state::StateSpec::KvPaged {
        hkv: 8,
        d: 16,
        dv: 16,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    };
    let small = kv_all();
    let config = StateConfig {
        max_ctx: 64,
        max_seqs: 2,
    };
    let specs = vec![big, small];
    let groups = group_layers(&specs);
    let required = required_pool_bytes(config, &groups).expect("pool math is exact");
    // One extra block per group plus 100 unusable remainder bytes.
    let big_block = groups[0].block_bytes().unwrap();
    let small_block = groups[1].block_bytes().unwrap();
    let pool = required + big_block + small_block + 100;
    let m = StateManager::new(config, specs, pool).expect("oversized pool fits");

    let b = m.budget();
    // min 2 blocks + 1 extra = 3 each: equal capacity despite cost ratio.
    assert_eq!(b.groups[0].total_blocks, 3);
    assert_eq!(b.groups[1].total_blocks, 3);
    assert_eq!(b.unusable_bytes, 100);
    assert_eq!(b.host_free, 0);
    // Assigned paged bytes plus remainder reconcile with the supply.
    assert_eq!(
        b.pool_bytes_total + b.unusable_bytes,
        pool - b.fixed_bytes_total
    );
    // Bases are contiguous with no gaps or overlaps.
    assert_eq!(b.groups[0].base_offset, 0);
    assert_eq!(
        b.groups[1].base_offset,
        b.groups[0].base_offset + 3 * big_block
    );
    assert_eq!(b.pool_bytes_total, 3 * (big_block + small_block));
}

/// Free bytes track allocations exactly across paged blocks and fixed slots.
#[test]
fn free_bytes_track_allocations_exactly() {
    let recurrent = r9v_state::StateSpec::Recurrent { h: 2, d: 8, dv: 8 };
    let config = StateConfig {
        max_ctx: 64,
        max_seqs: 2,
    };
    let specs = vec![kv_all(), recurrent];
    let groups = group_layers(&specs);
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    let mut m = StateManager::new(config, specs, pool).expect("minimum pool fits");

    let b0 = m.budget();
    let block_bytes = b0.groups[0].block_bytes;
    // Recurrent per-seq double-buffered bytes: 2*8*8*4*2 = 1024.
    assert_eq!(b0.groups[1].slot_bytes_per_seq, 1024);
    assert_eq!(b0.fixed_bytes_total, 2 * 1024);
    assert_eq!(b0.fixed_bytes_free, 2 * 1024);

    let (a, _) = m.new_seq(&[]).unwrap();
    // One live sequence consumes one fixed slot; paged bytes are untouched.
    assert_eq!(m.budget().fixed_bytes_free, 1024);
    assert_eq!(m.budget().pool_bytes_free, b0.pool_bytes_free);

    let toks = vec![7; 32];
    m.reserve(a, 32).unwrap();
    m.write_tokens(a, 0, &toks).unwrap();
    m.commit(a, 32).unwrap();
    assert_eq!(m.budget().pool_bytes_free, b0.pool_bytes_free - block_bytes);
    assert_eq!(m.free_blocks(0).unwrap(), 1);

    m.free_seq(a).unwrap();
    assert_eq!(m.budget(), b0);
}

/// Stats counters are exact and utilization comes from exact block counts.
#[test]
fn stats_counters_and_utilization_are_exact() {
    let mut m = manager_for(
        StateConfig {
            max_ctx: 64,
            max_seqs: 4,
        },
        &[kv_all()],
    );
    let total = m.budget().groups[0].total_blocks;
    assert_eq!(total, 2);

    let (a, _) = m.new_seq(&[]).unwrap();
    let toks = vec![1; 32];
    m.reserve(a, 32).unwrap();
    m.write_tokens(a, 0, &toks).unwrap();
    m.commit(a, 32).unwrap();
    let stats = m.stats();
    assert_eq!(stats.commits, 1);
    assert_eq!(stats.swaps, 0);
    assert_eq!(stats.prefix_hit_rate, 0.0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.utilization, 0.5);
}
