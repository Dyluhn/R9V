// SPDX-License-Identifier: Apache-2.0
//! A1.16: allocation-free `reserve`/`commit`/metadata hot paths plus behavior
//! regressions (Spec 3 §3.6, §5; Spec 1 §2.5; spec 6 §3.1, §3.3).
//!
//! The counting proof uses real nonempty mixed layer groups and counts heap
//! allocations from the cold first step — no warmup: construction and
//! admission (`new_seq`) size every reusable buffer up front, so the first
//! `reserve` after `new_seq` already allocates nothing, at any width up to
//! the legal maximum, on any thread, with any number of live ranges. A
//! thread-local counter backs the global allocator shim so concurrently
//! running tests on other threads cannot pollute a cycle's delta.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use r9v_common::SeqId;
use r9v_ir::IrError;
use r9v_state::{
    group_layers, required_pool_bytes, BatchWorkspace, CacheDtype, CompactOp, Positions, Retain,
    SlotRange, StateConfig, StateError, StateManager, StateSpec, TreeInput, TreeMask,
    MAX_COMPACT_TOKENS, MAX_RESERVE_HARD, SLOT_NONE,
};

thread_local! {
    static THREAD_ALLOCS: Cell<u64> = const { Cell::new(0) };
}

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get().wrapping_add(1)));
        // SAFETY: delegates to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: delegates to the system allocator with the same layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get().wrapping_add(1)));
        // SAFETY: delegates to the system allocator with the same arguments.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static SHIM: CountingAlloc = CountingAlloc;

/// Heap allocations charged to this thread so far.
fn thread_allocations() -> u64 {
    THREAD_ALLOCS.try_with(|c| c.get()).unwrap_or(0)
}

fn kv_all() -> StateSpec {
    StateSpec::KvPaged {
        hkv: 4,
        d: 32,
        dv: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }
}

fn latent_window(w: u32) -> StateSpec {
    StateSpec::KvLatent {
        latent: 64,
        rope: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::Window { w },
    }
}

fn recurrent() -> StateSpec {
    StateSpec::Recurrent {
        h: 4,
        d: 16,
        dv: 16,
    }
}

fn conv() -> StateSpec {
    StateSpec::ConvWindow { c: 16, w: 8 }
}

/// Single paged group: the wide/full-context manager.
fn dense_manager(max_ctx: u32, max_seqs: u32, pool_factor: u64) -> StateManager {
    let config = StateConfig { max_ctx, max_seqs };
    let specs = vec![kv_all()];
    let groups = group_layers(&specs).expect("valid fixture specs");
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    StateManager::new(config, specs, pool * pool_factor).expect("valid fixture config")
}

/// Four mixed groups: paged KV, windowed latent KV, conv, recurrent.
fn hybrid_manager(max_ctx: u32, max_seqs: u32, pool_factor: u64) -> StateManager {
    let config = StateConfig { max_ctx, max_seqs };
    let specs = vec![kv_all(), latent_window(64), recurrent(), conv()];
    let groups = group_layers(&specs).expect("valid fixture specs");
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    StateManager::new(config, specs, pool * pool_factor).expect("valid fixture config")
}

/// Two paged groups with different retention: window and sink+window.
fn windowed_manager(max_ctx: u32, max_seqs: u32, pool_factor: u64) -> StateManager {
    let config = StateConfig { max_ctx, max_seqs };
    let specs = vec![
        StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::Window { w: 64 },
        },
        StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::SinkWindow { n: 32, w: 64 },
        },
    ];
    let groups = group_layers(&specs).expect("valid fixture specs");
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    StateManager::new(config, specs, pool * pool_factor).expect("valid fixture config")
}

/// One counted hot step on `manager`: reserve `n`, build the scheduler batch
/// view into `ws`, cross-check descriptor slots against the view, commit.
/// Asserts zero heap allocations for the whole step.
fn hot_step(
    manager: &mut StateManager,
    ws: &mut BatchWorkspace,
    seq: SeqId,
    n: u32,
    cycles: &mut u64,
) {
    let before = thread_allocations();
    let ctx = manager.ctx_len(seq).expect("ctx read is infallible here");
    let r = manager.reserve(seq, n).expect("hot reserve succeeds");
    assert_eq!(r.start(), ctx);
    assert_eq!(r.len(), n);
    assert_eq!(r.seq(), seq);
    manager
        .fill_batch_meta(&[seq], &[n], None, ws)
        .expect("hot batch view succeeds");
    assert_eq!(ws.tokens() as usize, n as usize);
    // Descriptor slots agree with the scheduler view, group by group.
    let groups = manager.groups().len();
    let mut row = [0u32; 64];
    if n as usize <= row.len() {
        for gi in 0..groups {
            manager
                .fill_slots(&r, gi, &mut row[..n as usize])
                .expect("row fill succeeds");
            for (k, cell) in row[..n as usize].iter().enumerate() {
                assert_eq!(*cell, r.slot(manager, gi, k as u32).expect("slot resolves"));
                assert_eq!(*cell, ws.slot(gi, k).expect("view covers every token"));
            }
        }
    } else {
        // Wide rows: spot-check first, lane-boundary, and last token.
        for gi in 0..groups {
            for k in [0, 31, 32, n - 1] {
                assert_eq!(
                    r.slot(manager, gi, k).expect("slot resolves"),
                    ws.slot(gi, k as usize).expect("view covers every token"),
                    "group {gi} token {k} of {n}"
                );
            }
        }
    }
    manager.commit(seq, n).expect("hot commit succeeds");
    *cycles += 1;
    assert_eq!(
        thread_allocations(),
        before,
        "hot reserve+view+commit step allocated (n={n})",
    );
}

/// Successful hot-step state work allocates nothing, independent of caller
/// history, width, thread, group mix, or live-range count.
///
/// Counting starts at the cold first step — there is no warmup: admission
/// sizes every reusable buffer, so the first `reserve` after `new_seq`
/// already allocates nothing. Covers the cold first decode, a full-context
/// (`max_ctx`) reservation, a width above the old 4096 row cap, more than 8
/// simultaneous outstanding ranges, three managers with different group
/// shapes, cross-thread execution with a migrated range, and the workspace
/// batch view the scheduler uploads — at least 512 mixed nonempty cycles.
#[test]
fn hot_step_zero_alloc_from_cold_first_step() {
    // Workspace sized once (cold) for the largest batch below: 4 groups,
    // 16 sequences, 8192 tokens, 256 blocks per sequence per group.
    let mut ws = BatchWorkspace::with_capacity(4, 16, 8192, 256);
    let mut cycles: u64 = 0;

    // Cold first decode after `new_seq` on a fresh dense manager: counted.
    let mut dense = dense_manager(8192, 16, 2);
    let (d0, _) = dense.new_seq(&[]).expect("admission succeeds");
    hot_step(&mut dense, &mut ws, d0, 1, &mut cycles);

    // Legal-maximum width: a full-context reservation in one call, counted.
    // d0 is freed (cold teardown) so d1 takes blocks from a whole pool and
    // absolute slot ids are exact: index i -> id i.
    dense.free_seq(d0).expect("teardown succeeds");
    let (d1, _) = dense.new_seq(&[]).expect("admission succeeds");
    {
        let before = thread_allocations();
        let r = dense
            .reserve(d1, 8192)
            .expect("full-context reserve succeeds");
        assert_eq!((r.start(), r.len(), r.end()), (0, 8192, 8192));
        dense
            .fill_batch_meta(&[d1], &[8192], None, &mut ws)
            .expect("full-width view succeeds");
        assert_eq!(ws.slot_map().len(), 8192);
        assert_eq!(ws.block_table().len(), 256);
        // Lane mapping across the first block crossing and the last token.
        assert_eq!(r.slot(&dense, 0, 0).unwrap(), 0);
        assert_eq!(r.slot(&dense, 0, 31).unwrap(), 31);
        assert_eq!(r.slot(&dense, 0, 32).unwrap() % 32, 0);
        assert_eq!(r.slot(&dense, 0, 8191).unwrap() % 32, 31);
        dense.commit(d1, 8192).expect("full commit succeeds");
        cycles += 1;
        assert_eq!(thread_allocations(), before, "full-context step allocated");
    }
    dense.free_seq(d1).expect("teardown succeeds");

    // Width above the old 4096 row cap, counted.
    let (d2, _) = dense.new_seq(&[]).expect("admission succeeds");
    hot_step(&mut dense, &mut ws, d2, 5000, &mut cycles);
    dense.free_seq(d2).expect("teardown succeeds");

    // Twelve sequences across two group shapes; all admissions cold.
    let mut hybrid = hybrid_manager(1024, 16, 8);
    let mut windowed = windowed_manager(2048, 16, 8);
    let mut hseqs = Vec::with_capacity(6);
    let mut wseqs = Vec::with_capacity(6);
    for _ in 0..6 {
        hseqs.push(hybrid.new_seq(&[]).expect("admission succeeds").0);
        wseqs.push(windowed.new_seq(&[]).expect("admission succeeds").0);
    }

    // More than 8 simultaneous outstanding ranges: 12 live descriptors held
    // in a stack array (no heap), each counted.
    let widths = [3u32, 7, 33, 64, 1, 5, 13, 31, 2, 9, 17, 45];
    let mut held: [Option<SlotRange>; 12] = [None; 12];
    {
        let before = thread_allocations();
        for (i, w) in widths.iter().enumerate() {
            let (mgr, seq) = if i < 6 {
                (&mut hybrid, hseqs[i])
            } else {
                (&mut windowed, wseqs[i - 6])
            };
            let ctx = mgr.ctx_len(seq).unwrap();
            let r = mgr.reserve(seq, *w).expect("outstanding reserve succeeds");
            assert_eq!((r.start(), r.len()), (ctx, *w));
            held[i] = Some(r);
            cycles += 1;
        }
        assert_eq!(
            thread_allocations(),
            before,
            "outstanding reserves allocated"
        );
    }
    // One batched scheduler view over all 12 outstanding reservations.
    {
        let before = thread_allocations();
        // Per-manager views (pools differ per manager): hybrid batch first.
        hybrid
            .fill_batch_meta(&hseqs, &widths[..6], None, &mut ws)
            .expect("hybrid batch view succeeds");
        assert_eq!(
            ws.slot_map().len(),
            4 * widths[..6].iter().sum::<u32>() as usize
        );
        for (i, seq) in hseqs.iter().enumerate() {
            let r = held[i].expect("range is live");
            assert_eq!(r.seq(), *seq);
            // The held descriptor resolves while the batch view is live.
            let base: usize = widths[..i].iter().sum::<u32>() as usize;
            for k in 0..widths[i] as usize {
                assert_eq!(
                    r.slot(&hybrid, 0, k as u32).unwrap(),
                    ws.slot(0, base + k).unwrap(),
                    "hybrid seq {i} token {k}"
                );
            }
        }
        windowed
            .fill_batch_meta(&wseqs, &widths[6..], None, &mut ws)
            .expect("windowed batch view succeeds");
        assert_eq!(
            ws.slot_map().len(),
            2 * widths[6..].iter().sum::<u32>() as usize
        );
        cycles += 1;
        assert_eq!(thread_allocations(), before, "batched views allocated");
    }
    // Commit all 12, each counted; ranges stay usable through commit.
    {
        for (i, w) in widths.iter().enumerate() {
            let before = thread_allocations();
            let (mgr, gid) = if i < 6 {
                (&mut hybrid, hseqs[i])
            } else {
                (&mut windowed, wseqs[i - 6])
            };
            mgr.commit(gid, *w).expect("commit succeeds");
            let r = held[i].take();
            assert!(r.is_some());
            cycles += 1;
            assert_eq!(thread_allocations(), before, "commit allocated");
        }
    }

    // 600 mixed nonempty decode cycles across both managers, crossing block
    // boundaries and flowing windowed releases, every cycle counted.
    const MIX: [u32; 12] = [1, 1, 2, 3, 5, 7, 13, 31, 32, 33, 64, 4];
    for it in 0..600u32 {
        let i = (it as usize) % 12;
        let w = MIX[((it as usize) * 7 + (it as usize) / 12) % MIX.len()];
        if i < 6 {
            hot_step(&mut hybrid, &mut ws, hseqs[i], w, &mut cycles);
        } else {
            hot_step(&mut windowed, &mut ws, wseqs[i - 6], w, &mut cycles);
        }
    }

    // Cross-thread execution: a live range migrates into a worker thread
    // with its manager (both `Send`), and the thread's own first steps
    // allocate nothing — no thread-local state anywhere on the hot path.
    let (migr_seq, migr_range) = {
        let (s, _) = hybrid.new_seq(&[]).expect("admission succeeds");
        let r = hybrid.reserve(s, 7).expect("reserve succeeds");
        (s, r)
    };
    let migrated = std::thread::spawn(move || {
        let mut hybrid = hybrid;
        let mut ws = BatchWorkspace::with_capacity(4, 16, 1024, 32);
        let before = thread_allocations();
        assert_eq!(migr_range.seq(), migr_seq);
        assert_eq!(migr_range.len(), 7);
        for k in 0..7 {
            let v = migr_range
                .slot(&hybrid, 0, k)
                .expect("migrated slot resolves");
            assert_eq!(v % 32, (migr_range.start() + k) % 32);
        }
        hybrid
            .fill_batch_meta(&[migr_seq], &[7], None, &mut ws)
            .expect("migrated view succeeds");
        hybrid
            .commit(migr_seq, 7)
            .expect("migrated commit succeeds");
        assert_eq!(thread_allocations(), before, "migrated step allocated");
        let mut n: u64 = 1;
        for it in 0..60u32 {
            let w = MIX[((it as usize) * 7 + (it as usize) / 12) % MIX.len()];
            hot_step(&mut hybrid, &mut ws, migr_seq, w, &mut n);
        }
        n
    });
    // A second worker with its own manager and group shape, fully independent.
    let independent = std::thread::spawn(|| {
        let mut windowed = windowed_manager(1024, 8, 4);
        let mut ws = BatchWorkspace::with_capacity(2, 8, 1024, 32);
        let (s, _) = windowed.new_seq(&[]).expect("admission succeeds");
        let mut n: u64 = 0;
        for it in 0..60u32 {
            let w = MIX[((it as usize) * 7 + (it as usize) / 12) % MIX.len()];
            hot_step(&mut windowed, &mut ws, s, w, &mut n);
        }
        n
    });
    cycles += migrated.join().expect("migrated thread succeeds");
    cycles += independent.join().expect("independent thread succeeds");

    assert!(
        cycles >= 512,
        "proof ran {cycles} mixed nonempty cycles, need at least 512"
    );
}

/// The descriptor resolves every group/token slot with checked agreement:
///
/// slot values match the caller-filled rows, the workspace view, and the
/// owned `BatchMeta` bit for bit across a paged block boundary; recurrent
/// rows are `SLOT_NONE`; an identical run reproduces the same descriptor.
#[test]
fn slot_descriptor_matches_views_across_block_boundary() {
    let config = StateConfig {
        max_ctx: 64,
        max_seqs: 2,
    };
    let specs = vec![
        StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        },
        recurrent(),
    ];
    let groups = group_layers(&specs).expect("valid fixture specs");
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    let mut m = StateManager::new(config, specs.clone(), pool).expect("valid manager");
    let (seq, _) = m.new_seq(&[]).unwrap();
    m.reserve(seq, 31).unwrap();
    m.commit(seq, 31).unwrap();

    let r = m.reserve(seq, 2).unwrap();
    assert_eq!((r.start(), r.len(), r.end()), (31, 2, 33));
    // Position 31 is lane 31 of block 0; position 32 is lane 0 of block 1.
    assert_eq!(r.slot(&m, 0, 0).unwrap() % 32, 31);
    assert_eq!(r.slot(&m, 0, 1).unwrap() % 32, 0);
    assert_ne!(r.slot(&m, 0, 0).unwrap(), r.slot(&m, 0, 1).unwrap());
    // Recurrent group carries no per-token slots.
    assert_eq!(r.slot(&m, 1, 0).unwrap(), SLOT_NONE);
    assert_eq!(r.slot(&m, 1, 1).unwrap(), SLOT_NONE);

    let mut row = [0u32; 2];
    m.fill_slots(&r, 0, &mut row).unwrap();
    assert_eq!(row[0] % 32, 31);
    assert_eq!(row[1] % 32, 0);
    let mut rrow = [9u32; 2];
    m.fill_slots(&r, 1, &mut rrow).unwrap();
    assert_eq!(rrow, [SLOT_NONE; 2]);

    let mut ws = BatchWorkspace::with_capacity(2, 2, 2, 2);
    m.fill_batch_meta(&[seq], &[2], None, &mut ws).unwrap();
    assert_eq!(&ws.slot_map()[..2], &row[..]);
    assert_eq!(&ws.slot_map()[2..], &rrow[..]);
    let meta = m.batch_meta(&[seq], &[2]).expect("batch builds");
    assert_eq!(meta.slot_map(), ws.slot_map());
    assert_eq!(meta.block_table(), ws.block_table());
    assert_eq!(meta.window_start(), ws.window_start());
    // Save live values before commit invalidates `r` (scoped reservation).
    let live_row = row;
    m.commit(seq, 2).unwrap();
    assert!(
        matches!(
            r.slot(&m, 0, 0).unwrap_err(),
            StateError::InvalidBatch { .. }
        ),
        "range is stale after commit"
    );

    // Determinism: a second manager with the same history agrees exactly.
    let mut m2 = StateManager::new(config, specs, pool).expect("valid manager");
    let (seq2, _) = m2.new_seq(&[]).unwrap();
    m2.reserve(seq2, 31).unwrap();
    m2.commit(seq2, 31).unwrap();
    let r2 = m2.reserve(seq2, 2).unwrap();
    assert_eq!(r, r2);
    for k in 0..2 {
        assert_eq!(live_row[k as usize], r2.slot(&m2, 0, k).unwrap());
    }
}

/// Checked descriptor access: out-of-range group/token, short row buffers,
/// and freed sequences are typed errors, never clamps or panics.
#[test]
fn slot_descriptor_access_is_checked() {
    let mut m = hybrid_manager(128, 4, 2);
    let groups = m.groups().len();
    assert_eq!(groups, 4);
    let (a, _) = m.new_seq(&[]).unwrap();
    let r = m.reserve(a, 4).unwrap();

    assert_eq!(
        m.slot(&r, groups, 0).unwrap_err(),
        StateError::InvalidBatch {
            detail: format!("group {groups} out of range {groups}"),
        }
    );
    assert_eq!(
        m.slot(&r, 0, 4).unwrap_err(),
        StateError::OutOfRange {
            start: 4,
            len: 1,
            end: 4,
        }
    );
    let mut short = [0u32; 3];
    assert_eq!(
        m.fill_slots(&r, 0, &mut short).unwrap_err(),
        StateError::OutOfRange {
            start: 0,
            len: 3,
            end: 4,
        }
    );
    // Exact-fit buffers succeed; over-long buffers fill only the prefix.
    let mut exact = [0u32; 4];
    m.fill_slots(&r, 0, &mut exact).unwrap();
    let mut long = [9u32; 6];
    m.fill_slots(&r, 0, &mut long).unwrap();
    assert_eq!(&long[..4], &exact);
    assert_eq!(&long[4..], &[9, 9]);

    // The descriptor is scoped to its open reservation (no borrow, but no
    // read of another step's tail): after commit it is stale typed, and
    // after free the sequence is gone typed `UnknownSeq`.
    m.commit(a, 4).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 4);
    assert!(
        matches!(
            m.slot(&r, 0, 0).unwrap_err(),
            StateError::InvalidBatch { .. }
        ),
        "stale range after commit must be typed InvalidBatch"
    );
    let mut row_after = [0u32; 4];
    assert!(
        matches!(
            m.fill_slots(&r, 0, &mut row_after).unwrap_err(),
            StateError::InvalidBatch { .. }
        ),
        "stale fill after commit must be typed InvalidBatch"
    );
    // A later reserve with a different shape supersedes: the old range stays
    // stale even though a new reservation is open.
    let r2 = m.reserve(a, 2).unwrap();
    assert_ne!((r2.start(), r2.len()), (r.start(), r.len()));
    assert!(
        matches!(
            m.slot(&r, 0, 0).unwrap_err(),
            StateError::InvalidBatch { .. }
        ),
        "superseded range must stay stale"
    );
    assert_eq!(m.slot(&r2, 0, 0).unwrap(), r2.slot(&m, 0, 0).unwrap());
    m.commit(a, 2).unwrap();
    // Identical re-reservation after accepted=0 remains equivalent: the
    // values compare equal and resolve identically (documented, not a hole).
    // ctx is 6 here (4 + 2).
    m.reserve(a, 3).unwrap();
    m.commit(a, 0).unwrap();
    let ra = m.reserve(a, 3).unwrap();
    assert_eq!((ra.start(), ra.len()), (6, 3));
    m.commit(a, 0).unwrap();
    let rb = m.reserve(a, 3).unwrap();
    assert_eq!(ra, rb);
    // The earlier descriptor compares equal, so it resolves identically.
    assert_eq!(m.slot(&ra, 0, 0).unwrap(), m.slot(&rb, 0, 0).unwrap());
    m.commit(a, 3).unwrap();
    m.free_seq(a).unwrap();
    assert_eq!(
        m.slot(&r, 0, 0).unwrap_err(),
        StateError::UnknownSeq { seq: a.as_u64() }
    );
}

/// The descriptor is a compact `Copy` value with no public movable field:
///
/// ranges duplicate by copy (no field-move trap is expressible: fields are
/// private, there is no `Drop`), cross threads as `Send`, and fit in four
/// `u32`s. Undersized workspaces fail typed instead of growing.
#[test]
fn slot_range_has_no_field_move_trap_and_workspace_fails_closed() {
    fn assert_send_sync_copy<T: Send + Sync + Copy>() {}
    assert_send_sync_copy::<SlotRange>();
    assert_send_sync_copy::<SeqId>();
    assert!(std::mem::size_of::<SlotRange>() <= 16);

    let mut m = hybrid_manager(128, 4, 2);
    let (a, _) = m.new_seq(&[]).unwrap();
    let r = m.reserve(a, 4).unwrap();
    // Copy, not move: both spellings stay usable (a moved-out field would
    // not compile here, and neither does a `Drop`-paired field move).
    let r2 = r;
    assert_eq!(
        (r.start(), r.len(), r.end()),
        (r2.start(), r2.len(), r2.end())
    );
    assert_eq!(r.seq(), a);
    assert!(!r.is_empty());
    m.commit(a, 4).unwrap();

    // A workspace sized below the batch fails typed before writing anything.
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(b, 4).unwrap();
    let mut small = BatchWorkspace::with_capacity(4, 1, 1, 4);
    let err = m.fill_batch_meta(&[b], &[4], None, &mut small).unwrap_err();
    assert!(
        matches!(err, StateError::InvalidBatch { .. }),
        "got {err:?}"
    );
    assert_eq!(small.slot_map(), &[] as &[u32]);
    m.commit(b, 4).unwrap();
}

/// Pool exhaustion is typed, transactional, and leaks nothing.
#[test]
fn pool_exhaustion_rollback_leaves_no_partial_mutation() {
    let config = StateConfig {
        max_ctx: 64,
        max_seqs: 2,
    };
    let specs = vec![StateSpec::KvPaged {
        hkv: 2,
        d: 16,
        dv: 16,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }];
    let groups = group_layers(&specs).expect("valid fixture specs");
    // Exactly the minimum pool: two blocks in the single paged group.
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    let mut m = StateManager::new(config, specs, pool).expect("valid manager");
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 64).unwrap();
    m.commit(a, 64).unwrap();
    assert_eq!(m.free_blocks(0).unwrap(), 0);

    let (b, _) = m.new_seq(&[]).unwrap();
    let err = m.reserve(b, 32).unwrap_err();
    assert_eq!(
        err,
        StateError::PoolExhausted {
            group: 0,
            required: 1,
            available: 0,
            shortfall: 1,
            end: 32,
            max_ctx: 64,
        }
    );
    // No partial mutation: no tail, no blocks held, pool untouched.
    assert_eq!(m.tail_len(b).unwrap(), 0);
    assert_eq!(m.ctx_len(b).unwrap(), 0);
    assert_eq!(m.free_blocks(0).unwrap(), 0);
    assert_eq!(m.ctx_len(a).unwrap(), 64);

    // Freeing the holder restores the pool exactly; the refused sequence can
    // then reserve, proving nothing leaked.
    m.free_seq(a).unwrap();
    assert_eq!(m.free_blocks(0).unwrap(), 2);
    let r = m.reserve(b, 32).unwrap();
    assert_eq!((r.start(), r.len()), (0, 32));
    let mut row = vec![0u32; 32];
    m.fill_slots(&r, 0, &mut row).unwrap();
    assert_eq!(row, (0..32).collect::<Vec<u32>>());
}

/// Repeated and invalid reserves are refused with exact typed errors.
#[test]
fn repeated_and_invalid_reserve_rejected() {
    let mut m = hybrid_manager(128, 2, 2);
    let (a, _) = m.new_seq(&[]).unwrap();

    assert_eq!(
        m.reserve(a, 0).unwrap_err(),
        StateError::InvalidReserve { n: 0, tail: 0 }
    );
    assert_eq!(
        m.reserve(a, MAX_RESERVE_HARD + 1).unwrap_err(),
        StateError::InvalidReserve {
            n: MAX_RESERVE_HARD + 1,
            tail: 0,
        }
    );
    assert_eq!(
        m.reserve(a, 129).unwrap_err(),
        StateError::ReserveTooLarge {
            end: 129,
            max_ctx: 128,
            n: 129,
        }
    );
    m.reserve(a, 4).unwrap();
    assert_eq!(
        m.reserve(a, 4).unwrap_err(),
        StateError::InvalidReserve { n: 4, tail: 4 }
    );
    assert_eq!(
        m.commit(a, 5).unwrap_err(),
        StateError::CommitTooLarge {
            accepted: 5,
            tail: 4,
        }
    );
    m.commit(a, 4).unwrap();
    assert_eq!(
        m.commit(a, 1).unwrap_err(),
        StateError::NoReservation { seq: a.as_u64() }
    );
    assert_eq!(
        m.reserve(SeqId::new(9999), 1).unwrap_err(),
        StateError::UnknownSeq { seq: 9999 }
    );
}

/// Double-buffer parity flips on every positive accept (full or partial) and
/// never on full rejection; tree-verify compaction preserves the verified
/// prefix bit for bit.
#[test]
fn double_buffer_swap_and_partial_commit_behavior() {
    let mut m = hybrid_manager(128, 4, 2);
    let (a, _) = m.new_seq(&[]).unwrap();
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 0);
    assert_eq!(m.recurrent_active(a, 3).unwrap(), 0);

    // Full rejection swaps nothing but still closes the reservation.
    m.reserve(a, 4).unwrap();
    m.commit(a, 0).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 0);
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 0);
    assert_eq!(m.recurrent_active(a, 3).unwrap(), 0);
    assert_eq!(m.stats().swaps, 0);
    assert_eq!(m.stats().commits, 1);

    // Partial accept advances ctx and swaps both fixed groups.
    m.reserve(a, 4).unwrap();
    m.commit(a, 2).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 2);
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 1);
    assert_eq!(m.recurrent_active(a, 3).unwrap(), 1);

    // Full accept swaps back.
    m.reserve(a, 2).unwrap();
    m.commit(a, 2).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 4);
    assert_eq!(m.recurrent_active(a, 2).unwrap(), 0);
    assert_eq!(m.stats().swaps, 2);
    assert_eq!(m.stats().commits, 3);

    // Tree verify: accepted path [2, 0] compacts to the verified prefix and
    // matches a sequence that never speculated.
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(b, 4).unwrap();
    m.write_tokens(b, 0, &[10, 11, 12, 13]).unwrap();
    let op = m.compact(b, &[2, 0]).unwrap();
    assert_eq!(op.dst_start(), 0);
    assert_eq!(op.len(), 2);
    m.commit(b, 2).unwrap();

    let (c, _) = m.new_seq(&[]).unwrap();
    m.reserve(c, 2).unwrap();
    m.write_tokens(c, 0, &[12, 10]).unwrap();
    m.commit(c, 2).unwrap();

    for pos in 0..2 {
        assert_eq!(
            m.read_token(b, 0, pos).unwrap(),
            m.read_token(c, 0, pos).unwrap()
        );
    }
    assert_eq!(m.read_token(b, 0, 0).unwrap(), Some(12));
    assert_eq!(m.read_token(b, 0, 1).unwrap(), Some(10));

    // Committing a different count than compacted is refused.
    let (d, _) = m.new_seq(&[]).unwrap();
    m.reserve(d, 4).unwrap();
    m.write_tokens(d, 0, &[1, 2, 3, 4]).unwrap();
    m.compact(d, &[1, 3]).unwrap();
    assert!(m.commit(d, 1).is_err());
}

/// Windowed groups release exactly the aged blocks (sink-pinned blocks stay),
/// evicted mirror positions read back absent, and `free_seq` returns all.
#[test]
fn windowed_releases_and_refcounts_are_exact() {
    let config = StateConfig {
        max_ctx: 256,
        max_seqs: 2,
    };
    let specs = vec![
        StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::Window { w: 64 },
        },
        StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::SinkWindow { n: 32, w: 64 },
        },
    ];
    let groups = group_layers(&specs).expect("valid fixture specs");
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    let mut m = StateManager::new(config, specs, pool).expect("valid manager");
    let total_blocks = m.budget().groups[0].total_blocks;
    assert_eq!(total_blocks, 8);

    let (a, _) = m.new_seq(&[]).unwrap();
    for step in 0..6 {
        let ctx = step * 32;
        m.reserve(a, 32).unwrap();
        m.write_tokens(a, ctx, &[step + 1; 32]).unwrap();
        m.commit(a, 32).unwrap();
    }
    assert_eq!(m.ctx_len(a).unwrap(), 192);
    assert_eq!(m.window_start(a, 0).unwrap(), 128);
    assert_eq!(m.window_start(a, 1).unwrap(), 128);

    // Plain window: blocks 0..=3 aged out (block_end <= 128), blocks 4, 5 held.
    assert_eq!(m.free_blocks(0).unwrap(), total_blocks - 2);
    // Sink+window: block 0 pinned plus blocks 4, 5 held.
    assert_eq!(m.free_blocks(1).unwrap(), total_blocks - 3);

    // Evicted positions read back absent; retained positions still verify.
    assert_eq!(m.read_token(a, 0, 10).unwrap(), None);
    assert_eq!(m.read_token(a, 0, 160).unwrap(), Some(6));
    assert_eq!(m.read_token(a, 1, 10).unwrap(), Some(1));

    m.free_seq(a).unwrap();
    assert_eq!(m.free_blocks(0).unwrap(), total_blocks);
    assert_eq!(m.free_blocks(1).unwrap(), total_blocks);
}

/// Context limits are exact: reserving to `max_ctx` succeeds, one token more
/// is refused, and freeing an unknown sequence is refused.
#[test]
fn maximum_context_bounds_are_exact() {
    let mut m = hybrid_manager(64, 2, 2);
    let before = m.budget();
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 64).unwrap();
    m.commit(a, 64).unwrap();
    assert_eq!(m.ctx_len(a).unwrap(), 64);
    assert_eq!(
        m.reserve(a, 1).unwrap_err(),
        StateError::ReserveTooLarge {
            end: 65,
            max_ctx: 64,
            n: 1,
        }
    );
    assert_eq!(
        m.free_seq(SeqId::new(4242)).unwrap_err(),
        StateError::UnknownSeq { seq: 4242 }
    );
    // Freeing twice is refused the second time; the pool is whole again.
    m.free_seq(a).unwrap();
    assert_eq!(
        m.free_seq(a).unwrap_err(),
        StateError::UnknownSeq { seq: a.as_u64() }
    );
    assert_eq!(m.budget(), before);
}

/// Caller-owned tree staging: pre-sized once (cold), rewritten per cycle
/// with no allocation. The scheduler owns these buffers in production; the
/// test proves the rewrite plus the workspace copy and full validation cost
/// zero heap allocations inside the counted section.
struct TreeStaging {
    parents: Vec<i32>,
    ancestors: Vec<bool>,
    t_max: u32,
}

impl TreeStaging {
    /// Cold staging for at most `max_t` tokens with `max_t_max` columns.
    /// May allocate: call before the counter starts.
    fn cold(max_t: usize, max_t_max: u32) -> Self {
        let mut parents = Vec::new();
        parents
            .try_reserve_exact(max_t)
            .expect("cold staging sizes");
        let mut ancestors = Vec::new();
        ancestors
            .try_reserve_exact(max_t * max_t_max as usize)
            .expect("cold staging sizes");
        Self {
            parents,
            ancestors,
            t_max: 0,
        }
    }

    /// Rewrites the staging as a `t`-token chain with `t_max` columns and
    /// the exact chain ancestor triangle (row `i` marks `0..=i`).
    /// Allocates nothing when the cold caps fit.
    fn rewrite_chain(&mut self, t: usize, t_max: u32) {
        self.parents.clear();
        for i in 0..t {
            self.parents.push(if i == 0 { -1 } else { i as i32 - 1 });
        }
        self.ancestors.clear();
        for i in 0..t {
            for j in 0..t_max as usize {
                self.ancestors.push(j <= i);
            }
        }
        self.t_max = t_max;
    }

    /// Borrows the staged slices for one fill.
    fn input(&self) -> TreeInput<'_> {
        TreeInput::new(&self.parents, self.t_max, &self.ancestors)
    }
}

/// Chain tree mask with `t` tokens: parents `[-1, 0, 1, ..]`, `t_max = t`.
fn chain_tree(t: usize) -> TreeMask {
    let mut parents = Vec::with_capacity(t);
    for i in 0..t {
        parents.push(if i == 0 { -1 } else { (i - 1) as i32 });
    }
    TreeMask::new(parents, t as u32, vec![false; t * t]).expect("chain tree is valid")
}

/// The compaction descriptor is an exact fixed-capacity `Copy` value:
///
/// at most 16 sources inline, slice access returns the exact bytes in exact
/// accepted-path order, and longer paths are refused typed before any
/// mutation.
#[test]
fn compact_descriptor_is_fixed_capacity_copy_with_exact_slice() {
    fn assert_send_sync_copy<T: Send + Sync + Copy>() {}
    assert_send_sync_copy::<CompactOp>();
    assert_eq!(MAX_COMPACT_TOKENS, 16);
    assert!(std::mem::size_of::<CompactOp>() <= 96);

    let mut m = hybrid_manager(128, 4, 2);
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 6).unwrap();
    m.write_tokens(a, 0, &[10, 11, 12, 13, 14, 15]).unwrap();
    // Arbitrary non-prefix accepted path.
    let op = m.compact(a, &[4, 1, 5, 0]).unwrap();
    assert_eq!(op.seq(), a);
    assert_eq!(op.dst_start(), 0);
    assert_eq!(op.len(), 4);
    assert!(!op.is_empty());
    assert_eq!(op.src_positions(), &[4, 1, 5, 0]);
    let op2 = op;
    assert_eq!(op, op2);
    m.commit(a, 4).unwrap();
    assert_eq!(m.read_token(a, 0, 0).unwrap(), Some(14));
    assert_eq!(m.read_token(a, 0, 1).unwrap(), Some(11));
    assert_eq!(m.read_token(a, 0, 2).unwrap(), Some(15));
    assert_eq!(m.read_token(a, 0, 3).unwrap(), Some(10));

    // Longer than the spec/config bound is refused typed, mutating nothing.
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(b, 32).unwrap();
    let long = [0u32; 17];
    let err = m.compact(b, &long).unwrap_err();
    assert!(matches!(err, StateError::InvalidCompact { .. }), "{err:?}");
    assert_eq!(m.tail_len(b).unwrap(), 32);
}

/// Hot tree-verify flow allocates nothing from the cold first step, tree
/// build included.
///
/// Counted per cycle: rewriting the caller-owned parents/ancestor mask,
/// `reserve`, `fill_batch_meta_with_tree_input` (workspace copy plus full
/// validation: shape, parent bounds/self-parent, cycles, batch rules),
/// `compact` of an arbitrary non-prefix accepted path, descriptor consume
/// (`commit(op.len())`). The counter starts BEFORE the tree slices are
/// populated — the old proof built the owned `TreeMask` outside the counted
/// section, hiding one allocation per verify. Device-only (no mirror
/// writes), so the mirror updates are missing-key removes —
/// allocation-free by construction. Covers the cold first verify plus
/// repeated cycles on paged, windowed, sink+window, and hybrid group
/// shapes; descriptor bytes/order are exact and replay-identical.
#[test]
fn hot_tree_verify_zero_alloc_from_cold_first_step() {
    let mut cycles: u64 = 0;

    let configs: [(&str, Vec<StateSpec>); 4] = [
        ("paged", vec![kv_all()]),
        ("window", vec![latent_window(64)]),
        (
            "sink",
            vec![StateSpec::KvPaged {
                hkv: 2,
                d: 16,
                dv: 16,
                cache: CacheDtype::E4M3,
                retain: Retain::SinkWindow { n: 32, w: 64 },
            }],
        ),
        (
            "hybrid",
            vec![kv_all(), latent_window(64), recurrent(), conv()],
        ),
    ];
    for (name, specs) in configs {
        let config = StateConfig {
            max_ctx: 256,
            max_seqs: 8,
        };
        let groups = group_layers(&specs).expect("valid fixture specs");
        let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
        let mut m = StateManager::new(config, specs, pool * 4).expect("valid manager");
        let groups_n = m.groups().len();
        // Cold sizing, once: batch buffers plus tree storage for the widest
        // cycle below (T=6, t_max=6), and caller-owned staging sized the same.
        let mut ws = BatchWorkspace::try_with_capacity_and_tree(groups_n.max(1), 8, 16, 8, 8, 8)
            .expect("workspace sizes");
        let mut staging = TreeStaging::cold(8, 8);
        let (s, _) = m.new_seq(&[]).expect("admission succeeds");
        // Cold first verify: T=5 candidates, accepted non-prefix [3, 1, 0].
        // The counter starts before the tree slices are populated.
        let before = thread_allocations();
        staging.rewrite_chain(5, 5);
        let ctx0 = m.ctx_len(s).unwrap();
        assert_eq!(ctx0, 0);
        m.reserve(s, 5).expect("reserve succeeds");
        m.fill_batch_meta_with_tree_input(&[s], &[5], staging.input(), &mut ws)
            .expect("tree view succeeds");
        assert_eq!(ws.tokens(), 5);
        let view = ws.tree_view().expect("tree stored");
        assert_eq!(view.t(), 5);
        assert_eq!(view.t_max(), 5);
        assert_eq!(view.parents(), &[-1, 0, 1, 2, 3]);
        assert!(view.is_ancestor(4, 0));
        assert!(!view.is_ancestor(0, 4));
        assert!(ws.tree().is_none(), "hot path fabricates no owned mask");
        let op = m.compact(s, &[3, 1, 0]).expect("compact succeeds");
        assert_eq!(op.dst_start(), 0);
        assert_eq!(op.len(), 3);
        assert_eq!(op.src_positions(), &[3, 1, 0]);
        assert_eq!(op.seq(), s);
        m.commit(s, op.len()).expect("commit consumes descriptor");
        assert_eq!(m.ctx_len(s).unwrap(), 3);
        cycles += 1;
        assert_eq!(
            thread_allocations(),
            before,
            "{name}: cold first tree-verify step allocated"
        );

        // Replay-identical descriptor bytes on a twin manager.
        let mut m2 = StateManager::new(config, specs_clone(name), pool * 4).expect("valid manager");
        let (s2, _) = m2.new_seq(&[]).expect("admission succeeds");
        let mut ws2 = BatchWorkspace::try_with_capacity_and_tree(groups_n.max(1), 8, 16, 8, 8, 8)
            .expect("workspace sizes");
        let mut staging2 = TreeStaging::cold(8, 8);
        staging2.rewrite_chain(5, 5);
        m2.reserve(s2, 5).unwrap();
        m2.fill_batch_meta_with_tree_input(&[s2], &[5], staging2.input(), &mut ws2)
            .unwrap();
        let op2 = m2.compact(s2, &[3, 1, 0]).unwrap();
        assert_eq!(op.dst_start(), op2.dst_start());
        assert_eq!(op.len(), op2.len());
        assert_eq!(op.src_positions(), op2.src_positions());
        assert_eq!(
            ws2.tree_view().expect("twin tree stored").parents(),
            view.parents()
        );

        // Repeated verify cycles with rotating non-prefix paths, every cycle
        // counted from before the tree rewrite. Accepted lengths 2-3.
        let paths: [&[u32]; 4] = [&[4, 0], &[2, 0], &[3, 1, 0], &[1, 4]];
        for (i, path) in paths.iter().cycle().take(48).enumerate() {
            let t = if i % 3 == 1 { 6u32 } else { 5u32 };
            let use_path: &[u32] = if t == 6 { &[5, 2, 0] } else { path };
            let before = thread_allocations();
            staging.rewrite_chain(t as usize, t);
            m.reserve(s, t).unwrap();
            m.fill_batch_meta_with_tree_input(&[s], &[t], staging.input(), &mut ws)
                .unwrap();
            let op = m.compact(s, use_path).unwrap();
            assert_eq!(op.len() as usize, use_path.len());
            for (k, p) in use_path.iter().enumerate() {
                assert_eq!(op.src_positions()[k], m.ctx_len(s).unwrap() + p);
            }
            m.commit(s, op.len()).unwrap();
            cycles += 1;
            assert_eq!(
                thread_allocations(),
                before,
                "{name}: tree-verify cycle {i} allocated"
            );
        }
        let _ = name;
    }
    assert!(cycles > 4 * 48, "ran {cycles} tree cycles");
}

fn specs_clone(name: &str) -> Vec<StateSpec> {
    match name {
        "paged" => vec![kv_all()],
        "window" => vec![latent_window(64)],
        "sink" => vec![StateSpec::KvPaged {
            hkv: 2,
            d: 16,
            dv: 16,
            cache: CacheDtype::E4M3,
            retain: Retain::SinkWindow { n: 32, w: 64 },
        }],
        _ => vec![kv_all(), latent_window(64), recurrent(), conv()],
    }
}

/// Mirror-backed compact replaces existing keys with no allocation.
///
/// Setup (reserve + full-tail mirror writes) runs outside the counted
/// section — new BTreeMap keys allocate by construction (test support only).
/// The counted section is tree rewrite + fill + compact + commit, whose
/// mirror updates are replaces: allocation-free and bit-exact against a
/// never-speculated sequence.
#[test]
fn tree_compact_mirror_replace_is_alloc_free_and_bit_exact() {
    let mut m = hybrid_manager(128, 4, 2);
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 5).unwrap();
    m.write_tokens(a, 0, &[10, 11, 12, 13, 14]).unwrap();
    let mut ws =
        BatchWorkspace::try_with_capacity_and_tree(4, 4, 8, 4, 8, 8).expect("workspace sizes");
    let mut staging = TreeStaging::cold(8, 8);
    let before = thread_allocations();
    staging.rewrite_chain(5, 5);
    m.fill_batch_meta_with_tree_input(&[a], &[5], staging.input(), &mut ws)
        .expect("tree view succeeds");
    let op = m.compact(a, &[4, 1, 0]).expect("compact succeeds");
    assert_eq!(op.src_positions(), &[4, 1, 0]);
    m.commit(a, op.len()).expect("commit succeeds");
    assert_eq!(
        thread_allocations(),
        before,
        "mirror-replace fill+compact+commit allocated"
    );
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(b, 3).unwrap();
    m.write_tokens(b, 0, &[14, 11, 10]).unwrap();
    m.commit(b, 3).unwrap();
    for pos in 0..3 {
        assert_eq!(
            m.read_token(a, 0, pos).unwrap(),
            m.read_token(b, 0, pos).unwrap(),
            "compacted position {pos}"
        );
    }
}

/// Hot explicit positions match the owned builder bit for bit.
///
/// Scalar `[T]`, MRoPE `[T,3]`, each with and without a tree mask. The hot
/// `fill_batch_meta_with_options` copies into caller-owned buffers with no
/// allocation; width disambiguates the two position views.
#[test]
fn hot_explicit_positions_match_owned_bit_for_bit() {
    let mut m = hybrid_manager(128, 4, 2);
    let groups = m.groups().len();
    let mut ws = BatchWorkspace::try_with_capacity(groups, 4, 8, 4).expect("workspace sizes");
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();

    // Scalar explicit positions.
    let scalar = Positions::PerToken(vec![100, 101, 102, 103]);
    let tree4 = chain_tree(4);
    let tree4b = chain_tree(4);
    let before = thread_allocations();
    m.fill_batch_meta_with_options(&[a], &[4], Some(&scalar), Some(tree4), &mut ws)
        .expect("hot scalar fill succeeds");
    assert_eq!(
        thread_allocations(),
        before,
        "hot scalar explicit fill allocated"
    );
    assert_eq!(ws.position_width(), 1);
    assert_eq!(ws.positions(), &[100, 101, 102, 103]);
    assert!(ws.positions_mrope().is_empty());
    let owned = m
        .batch_meta_with_options(&[a], &[4], Some(scalar.clone()), Some(tree4b))
        .expect("owned scalar builds");
    assert_eq!(owned.positions(), &scalar);
    assert_eq!(ws.slot_map(), owned.slot_map());
    assert_eq!(ws.block_table(), owned.block_table());
    assert_eq!(ws.window_start(), owned.window_start());
    assert_eq!(
        ws.tree().unwrap().parents(),
        owned.tree().unwrap().parents()
    );

    // MRoPE explicit triplets.
    let triplets = vec![[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]];
    let mrope = Positions::Mrope(triplets.clone());
    let before = thread_allocations();
    m.fill_batch_meta_with_options(&[a], &[4], Some(&mrope), None, &mut ws)
        .expect("hot mrope fill succeeds");
    assert_eq!(
        thread_allocations(),
        before,
        "hot mrope explicit fill allocated"
    );
    assert_eq!(ws.position_width(), 3);
    assert_eq!(ws.positions_mrope(), &triplets[..]);
    assert_eq!(ws.positions_mrope_flat_len(), 12);
    assert_eq!(ws.positions_mrope_flat_value(0), Some(7));
    assert_eq!(ws.positions_mrope_flat_value(5), Some(12));
    assert_eq!(ws.positions_mrope_flat_value(11), Some(18));
    assert_eq!(ws.positions_mrope_flat_value(12), None);
    assert!(ws.positions().is_empty());
    let owned_m = m
        .batch_meta_with_options(&[a], &[4], Some(mrope.clone()), None)
        .expect("owned mrope builds");
    assert_eq!(owned_m.positions(), &mrope);
    assert_eq!(ws.slot_map(), owned_m.slot_map());
    assert_eq!(ws.block_table(), owned_m.block_table());

    // MRoPE plus tree.
    let tree4c = chain_tree(4);
    let tree4d = chain_tree(4);
    m.fill_batch_meta_with_options(&[a], &[4], Some(&mrope), Some(tree4c), &mut ws)
        .expect("hot mrope+tree fill succeeds");
    assert_eq!(ws.position_width(), 3);
    assert_eq!(ws.tree().unwrap().t(), 4);
    let owned_mt = m
        .batch_meta_with_options(&[a], &[4], Some(mrope), Some(tree4d))
        .expect("owned mrope+tree builds");
    assert_eq!(ws.slot_map(), owned_mt.slot_map());
    assert_eq!(
        ws.positions_mrope(),
        match owned_mt.positions() {
            Positions::Mrope(v) => &v[..],
            _ => panic!("owned must be mrope"),
        }
    );

    // Length mismatch is typed on both paths.
    let bad = Positions::PerToken(vec![1, 2]);
    assert!(m
        .fill_batch_meta_with_options(&[a], &[4], Some(&bad), None, &mut ws)
        .is_err());
    assert!(m
        .batch_meta_with_options(&[a], &[4], Some(bad), None)
        .is_err());
    m.commit(a, 4).unwrap();
}

/// The checked workspace constructor refuses overflow/undersize typed.
#[test]
fn batch_workspace_try_with_capacity_is_checked() {
    let ws = BatchWorkspace::try_with_capacity(2, 2, 4, 2).expect("sane dims size");
    assert_eq!(ws.position_width(), 0);
    assert!(ws.seq_ids().is_empty());
    assert!(ws.positions().is_empty());
    assert!(ws.positions_mrope().is_empty());
    let err = BatchWorkspace::try_with_capacity(usize::MAX, usize::MAX, usize::MAX, u32::MAX)
        .unwrap_err();
    assert!(matches!(err, StateError::Overflow { .. }), "{err:?}");

    // Undersized buffers fail typed instead of growing.
    let mut m = hybrid_manager(64, 2, 2);
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();
    let mut small = BatchWorkspace::try_with_capacity(4, 1, 1, 4).expect("small sizes");
    let err = m
        .fill_batch_meta_with_options(
            &[a],
            &[4],
            Some(&Positions::PerToken(vec![0, 1, 2, 3])),
            None,
            &mut small,
        )
        .unwrap_err();
    assert!(matches!(err, StateError::InvalidBatch { .. }), "{err:?}");
    let mut small_m = BatchWorkspace::try_with_capacity(4, 1, 1, 4).expect("small sizes");
    let err = m
        .fill_batch_meta_with_options(
            &[a],
            &[4],
            Some(&Positions::Mrope(vec![[0, 0, 0]; 4])),
            None,
            &mut small_m,
        )
        .unwrap_err();
    assert!(matches!(err, StateError::InvalidBatch { .. }), "{err:?}");
    m.commit(a, 4).unwrap();
}

/// Builds the owned mask from the same slices the staging holds, so parity
/// compares identical inputs through both paths.
fn owned_from_staging(staging: &TreeStaging) -> TreeMask {
    TreeMask::new(
        staging.parents.clone(),
        staging.t_max,
        staging.ancestors.clone(),
    )
    .expect("staging holds a valid chain")
}

/// Hot slices fills match the owned builder bit for bit.
///
/// Default scalar positions, explicit scalar `[T]`, MRoPE `[T,3]`, and the
/// no-tree case: every tensor plus the stored parents/ancestor mask agree
/// exactly, and the slices path leaves `tree()` as `None` (no fabricated
/// owned mask) while `tree_view()` reads the stored slices.
#[test]
fn hot_tree_slices_match_owned_builder_bit_for_bit() {
    let mut m = hybrid_manager(128, 4, 2);
    let groups = m.groups().len();
    let mut ws =
        BatchWorkspace::try_with_capacity_and_tree(groups, 4, 8, 4, 8, 8).expect("workspace sizes");
    let mut staging = TreeStaging::cold(8, 8);
    let (a, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();
    staging.rewrite_chain(4, 4);

    // Default positions plus tree.
    m.fill_batch_meta_with_tree_input(&[a], &[4], staging.input(), &mut ws)
        .expect("slices fill succeeds");
    let owned = m
        .batch_meta_with_tree(&[a], &[4], Some(owned_from_staging(&staging)))
        .expect("owned builds");
    assert_eq!(ws.slot_map(), owned.slot_map());
    assert_eq!(ws.block_table(), owned.block_table());
    assert_eq!(ws.window_start(), owned.window_start());
    assert_eq!(
        owned.positions(),
        &Positions::PerToken(ws.positions().to_vec())
    );
    let view = ws.tree_view().expect("tree stored");
    let otree = owned.tree().expect("owned tree stored");
    assert_eq!(view.parents(), otree.parents());
    assert_eq!(view.ancestors(), otree.ancestors());
    assert_eq!(view.t_max(), otree.t_max());
    assert!(ws.tree().is_none(), "slices fill fabricates no owned mask");

    // Explicit scalar positions plus tree.
    let scalar = Positions::PerToken(vec![100, 101, 102, 103]);
    m.fill_batch_meta_with_options_and_tree_input(
        &[a],
        &[4],
        Some(&scalar),
        Some(staging.input()),
        &mut ws,
    )
    .expect("scalar slices fill succeeds");
    let owned_scalar = m
        .batch_meta_with_options(
            &[a],
            &[4],
            Some(scalar.clone()),
            Some(owned_from_staging(&staging)),
        )
        .expect("owned scalar builds");
    assert_eq!(owned_scalar.positions(), &scalar);
    assert_eq!(ws.positions(), &[100, 101, 102, 103]);
    assert_eq!(ws.slot_map(), owned_scalar.slot_map());
    assert_eq!(ws.block_table(), owned_scalar.block_table());
    assert_eq!(ws.window_start(), owned_scalar.window_start());
    let view = ws.tree_view().expect("tree stored");
    let otree = owned_scalar.tree().expect("owned tree stored");
    assert_eq!(view.parents(), otree.parents());
    assert_eq!(view.ancestors(), otree.ancestors());
    assert_eq!(view.is_ancestor(3, 1), otree.is_ancestor(3, 1));
    assert_eq!(view.is_ancestor(1, 3), otree.is_ancestor(1, 3));

    // MRoPE triplets plus tree.
    let triplets = vec![[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]];
    let mrope = Positions::Mrope(triplets.clone());
    m.fill_batch_meta_with_options_and_tree_input(
        &[a],
        &[4],
        Some(&mrope),
        Some(staging.input()),
        &mut ws,
    )
    .expect("mrope slices fill succeeds");
    let owned_mrope = m
        .batch_meta_with_options(
            &[a],
            &[4],
            Some(mrope.clone()),
            Some(owned_from_staging(&staging)),
        )
        .expect("owned mrope builds");
    assert_eq!(ws.positions_mrope(), &triplets[..]);
    assert_eq!(ws.slot_map(), owned_mrope.slot_map());
    assert_eq!(ws.block_table(), owned_mrope.block_table());
    assert_eq!(ws.tree_view().expect("tree stored").t_max(), 4);

    // No tree on either path.
    m.fill_batch_meta_with_options_and_tree_input(&[a], &[4], Some(&mrope), None, &mut ws)
        .expect("treeless slices fill succeeds");
    let owned_plain = m
        .batch_meta_with_options(&[a], &[4], Some(mrope), None)
        .expect("owned treeless builds");
    assert!(ws.tree_view().is_none());
    assert!(ws.tree().is_none());
    assert!(owned_plain.tree().is_none());
    assert_eq!(ws.slot_map(), owned_plain.slot_map());
    m.commit(a, 4).unwrap();
}

/// Tree slices validation is complete, typed, and fail-closed.
///
/// Intrinsic rules (shape, parent bounds/self-parent, cycles) report the
/// exact [`IrError`] the owned [`TreeMask`] builder reports for the same
/// slices, one for one. Batch rules (`T` match, `t_max` cover, no parent
/// crossing its sequence) report the builder's variants too — on both the
/// slices and the owned fill paths. Undersized buffers fail with the
/// capacity and the requirement, never grow (a fitting fill still fits and
/// an oversized one still fails afterwards), and a failed fill keeps the
/// previous batch tensors while storing no tree.
#[test]
fn tree_slices_failures_are_typed_and_fail_closed() {
    // Intrinsic rule parity with the owned builder, case by case.
    let cases: [(&[i32], u32, &[bool]); 6] = [
        (&[-1, 5], 1, &[true; 2]),
        (&[-2, -1], 1, &[true; 2]),
        (&[0, -1], 1, &[true; 2]),
        (&[-1, 2, 1], 3, &[true; 9]),
        (&[-1], 0, &[]),
        (&[-1, 0], 3, &[true; 5]),
    ];
    for (parents, t_max, ancestors) in cases {
        let mut ws =
            BatchWorkspace::try_with_capacity_and_tree(1, 2, 8, 2, 8, 8).expect("workspace sizes");
        let hot_err = ws
            .fill_tree(TreeInput::new(parents, t_max, ancestors))
            .expect_err("bad tree must be refused");
        let owned_err = TreeMask::new(parents.to_vec(), t_max, ancestors.to_vec())
            .expect_err("owned builder refuses the same slices");
        assert_eq!(hot_err, StateError::Ir(owned_err));
        assert!(ws.tree_view().is_none(), "failed store keeps no tree");
    }
    // Bad parent plus bad ancestor length collects both, like the builder.
    {
        let mut ws =
            BatchWorkspace::try_with_capacity_and_tree(1, 2, 8, 2, 8, 8).expect("workspace sizes");
        let hot_err = ws
            .fill_tree(TreeInput::new(&[1, 1], 3, &[true; 5]))
            .expect_err("two faults collect");
        let owned_err = TreeMask::new(vec![1, 1], 3, vec![true; 5]).expect_err("owned collects");
        assert_eq!(hot_err, StateError::Ir(owned_err));
        assert!(matches!(hot_err, StateError::Ir(IrError::Multiple { .. })));
    }

    // Batch rules on the slices path, with exact builder variants.
    let mut m = hybrid_manager(128, 4, 2);
    let groups = m.groups().len();
    let mut ws =
        BatchWorkspace::try_with_capacity_and_tree(groups, 4, 8, 4, 8, 8).expect("workspace sizes");
    let (a, _) = m.new_seq(&[]).unwrap();
    let (b, _) = m.new_seq(&[]).unwrap();
    m.reserve(a, 4).unwrap();
    m.reserve(b, 2).unwrap();

    // Tree/batch size mismatch.
    let mut staging = TreeStaging::cold(8, 8);
    staging.rewrite_chain(3, 3);
    let err = m
        .fill_batch_meta_with_tree_input(&[a], &[4], staging.input(), &mut ws)
        .unwrap_err();
    assert_eq!(
        err,
        StateError::Ir(IrError::TreeBatchMismatch {
            tree_t: 3,
            batch_t: 4,
        })
    );

    // `t_max` below the longest query.
    staging.rewrite_chain(4, 2);
    let err = m
        .fill_batch_meta_with_tree_input(&[a], &[4], staging.input(), &mut ws)
        .unwrap_err();
    assert_eq!(
        err,
        StateError::Ir(IrError::TreeMaxTooSmall {
            required: 4,
            actual: 2,
        })
    );

    // A parent crossing its sequence: token 2 of the second sequence points
    // at token 0 of the first (same shape as the builder's own test).
    let cross_parents = [-1, 0, 0, -1];
    let cross_anc = [true; 8];
    let err = m
        .fill_batch_meta_with_tree_input(
            &[a, b],
            &[2, 2],
            TreeInput::new(&cross_parents, 2, &cross_anc),
            &mut ws,
        )
        .unwrap_err();
    assert_eq!(
        err,
        StateError::Ir(IrError::TreeParentCrossesSequence {
            token: 2,
            parent: 0,
            seq: 1,
            seq_start: 2,
            seq_end: 4,
        })
    );
    // A parent crossing the other way (first sequence reaching into the
    // second's range) collects alongside the short columns: token 1 points
    // at token 4 while `t_max = 2` covers neither query.
    let cross2_parents = [-1, 4, 1, 1, -1, 4];
    let cross2_anc = [true; 12];
    let err = m
        .fill_batch_meta_with_tree_input(
            &[a, b],
            &[4, 2],
            TreeInput::new(&cross2_parents, 2, &cross2_anc),
            &mut ws,
        )
        .unwrap_err();
    assert!(
        matches!(err, StateError::Ir(IrError::Multiple { .. })),
        "got {err:?}"
    );

    // The owned fill path enforces the same batch rules now (it used to
    // trust the prebuilt mask): cross-sequence and short `t_max` fail there
    // too instead of storing a mask the builder would refuse.
    let bad_owned = TreeMask::new(vec![-1, 0, 0, -1], 2, vec![true; 8]).expect("forest builds");
    let err = m
        .fill_batch_meta(&[a, b], &[2, 2], Some(bad_owned), &mut ws)
        .unwrap_err();
    assert_eq!(
        err,
        StateError::Ir(IrError::TreeParentCrossesSequence {
            token: 2,
            parent: 0,
            seq: 1,
            seq_start: 2,
            seq_end: 4,
        })
    );
    // Short columns on the owned path: `t_max = 2` under a query of 4.
    let short_owned = TreeMask::new(vec![-1, 0, 1, 2], 2, vec![true; 8]).expect("chain builds");
    let err = m
        .fill_batch_meta(&[a], &[4], Some(short_owned), &mut ws)
        .unwrap_err();
    assert_eq!(
        err,
        StateError::Ir(IrError::TreeMaxTooSmall {
            required: 4,
            actual: 2,
        })
    );

    // Undersized tree buffers fail with capacities, never grow: a fitting
    // fill still fits and an oversized one still fails afterwards.
    let mut small =
        BatchWorkspace::try_with_capacity_and_tree(1, 2, 8, 2, 4, 4).expect("small sizes");
    let big_parents = [-1, 0, 1, 2, 3];
    let big_anc = [true; 25];
    let err = small
        .fill_tree(TreeInput::new(&big_parents, 5, &big_anc))
        .unwrap_err();
    assert!(
        matches!(err, StateError::InvalidBatch { .. }),
        "got {err:?}"
    );
    let fit_parents = [-1, 0, 1, 2];
    let fit_anc = [true; 16];
    small
        .fill_tree(TreeInput::new(&fit_parents, 4, &fit_anc))
        .expect("fitting tree still fits");
    assert_eq!(small.tree_view().expect("tree stored").t(), 4);
    assert!(
        small
            .fill_tree(TreeInput::new(&big_parents, 5, &big_anc))
            .is_err(),
        "oversized still fails: no silent growth"
    );

    // A failed fill keeps the previous batch tensors while storing no tree.
    m.commit(a, 0).unwrap();
    m.commit(b, 0).unwrap();
    let (c, _) = m.new_seq(&[]).unwrap();
    m.reserve(c, 4).unwrap();
    staging.rewrite_chain(4, 4);
    m.fill_batch_meta_with_tree_input(&[c], &[4], staging.input(), &mut ws)
        .expect("good fill succeeds");
    assert_eq!(ws.tokens(), 4);
    let good_slots = ws.slot_map().to_vec();
    // Tokens 1 and 2 point at each other; `t_max = 4` covers the query so
    // the cycle is the only fault.
    let cyclic = TreeInput::new(&[-1, 2, 1, 0], 4, &[true; 16]);
    let err = m
        .fill_batch_meta_with_tree_input(&[c], &[4], cyclic, &mut ws)
        .unwrap_err();
    assert_eq!(err, StateError::Ir(IrError::TreeCycle { token: 1 }));
    assert_eq!(ws.tokens(), 4, "batch tensors untouched");
    assert_eq!(ws.slot_map(), &good_slots[..], "batch tensors untouched");
    assert!(ws.tree_view().is_none(), "failed fill stores no tree");
    m.commit(c, 4).unwrap();

    // Cold sizing refuses overflow typed.
    let mut bare = BatchWorkspace::new();
    let err = bare.try_reserve_tree(usize::MAX, u32::MAX).unwrap_err();
    assert!(matches!(err, StateError::Overflow { .. }), "{err:?}");
    let err = BatchWorkspace::try_with_capacity_and_tree(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        u32::MAX,
        8,
        8,
    )
    .unwrap_err();
    assert!(matches!(err, StateError::Overflow { .. }), "{err:?}");
}

/// `try_reserve_tree` regrows reused (nonempty) tree storage to the cold cap.
///
/// Regression: the helper sized the reservation against `capacity`, but
/// `Vec::try_reserve_exact` counts additional space beyond `len`, so a regrow
/// issued after a fill — spare capacity, `len < capacity` — reported `Ok`
/// while keeping the old capacity, and the next maximum-size fill failed
/// closed. This fills the cold cap, regrows cold, and proves the newly
/// promised maximum fills with no hot allocation.
#[test]
fn tree_reserve_regrows_reused_storage_to_cold_cap() {
    let mut ws =
        BatchWorkspace::try_with_capacity_and_tree(1, 2, 8, 2, 8, 8).expect("workspace sizes");
    // Fill T=4: every tree buffer is now nonempty with spare capacity.
    let fit_parents = [-1, 0, 1, 2];
    let fit_anc = [true; 16];
    ws.fill_tree(TreeInput::new(&fit_parents, 4, &fit_anc))
        .expect("fitting tree fills");
    assert_eq!(ws.tree_view().expect("tree stored").t(), 4);
    // Cold regrow to a larger cap must be honored ...
    ws.try_reserve_tree(12, 12).expect("cold regrow succeeds");
    // ... so the newly promised maximum fills with no hot allocation.
    let big_parents: Vec<i32> = (0..12).map(|i| if i == 0 { -1 } else { i - 1 }).collect();
    let big_anc = vec![true; 12 * 12];
    let before = thread_allocations();
    ws.fill_tree(TreeInput::new(&big_parents, 12, &big_anc))
        .expect("promised maximum fills");
    assert_eq!(
        thread_allocations(),
        before,
        "maximum fill after cold regrow allocated"
    );
    assert_eq!(ws.tree_view().expect("tree stored").t(), 12);
}
