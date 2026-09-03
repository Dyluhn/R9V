// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Integration tests for the scalar deterministic T0 attention group
//! (Spec 1 §4.D, §6.3, Spec 3 §2, §3, Card A1.7).
//!
//! Every attention result is checked against the dense independent f64
//! oracles; every cache write is checked by round-trip; validation paths are
//! checked for exact typed failures before mutation.

use r9v_common::rng::SeededRng;
use r9v_ir::{
    AttentionMask, AttentionOp, BatchMeta, CacheScaleGranularity, DType, MlaAttentionSpec,
    MlaLatent, Positions, StateHandle, StateKind, StateWriteKvOp, TreeMask,
};
use r9v_t0::attention::{
    attention_mla, attention_paged, attention_row_f64_reference, mla_row_f64_reference,
    state_write_kv, state_write_kv_latent, state_write_kv_paged, KvCache, KvLatentCache,
    KvPagedCache,
};
use r9v_t0::{Tolerance, TypedBuffer};

const SENTINEL: u32 = u32::MAX;

fn f32_data(rng: &mut SeededRng, len: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let raw = (rng.next_u64() & 0xFFFF_FFFF) as u32;
        let unit = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(unit * scale);
    }
    out
}

/// Builds a single-group paged `BatchMeta` with sequential block ids.
///
/// `holes` lists `(seq, absolute_block)` entries forced to the sentinel so
/// tests can model window eviction without a state manager.
fn build_meta(
    ctx: &[u32],
    qlen: &[u32],
    max_blocks: u32,
    window_start: &[u32],
    holes: &[(usize, u32)],
    tree: Option<TreeMask>,
) -> BatchMeta {
    let s = ctx.len();
    let t: usize = qlen.iter().map(|&q| q as usize).sum();
    let mut table = vec![SENTINEL; s * max_blocks as usize];
    let mut next_block = 0u32;
    for (seq, (&c, &q)) in ctx.iter().zip(qlen.iter()).enumerate() {
        let total = c + q;
        let need = total.div_ceil(32);
        for b in 0..need {
            table[seq * max_blocks as usize + b as usize] = next_block + b;
        }
        next_block += need;
    }
    for &(seq, b) in holes {
        table[seq * max_blocks as usize + b as usize] = SENTINEL;
    }
    let mut slots = Vec::with_capacity(t);
    for (seq, (&c, &q)) in ctx.iter().zip(qlen.iter()).enumerate() {
        for k in 0..q {
            let pos = c + k;
            let b = pos / 32;
            let lane = pos % 32;
            let id = table[seq * max_blocks as usize + b as usize];
            assert_ne!(id, SENTINEL, "test writes only live blocks");
            slots.push(id * 32 + lane);
        }
    }
    BatchMeta::builder(1, s as u32, t as u32, max_blocks)
        .seq_ids((0..s as u32).collect())
        .query_len(qlen.to_vec())
        .ctx_len(ctx.to_vec())
        .positions(Positions::PerToken(vec![0u32; t]))
        .slot_map(slots)
        .block_table(table)
        .window_start(window_start.to_vec())
        .tree(tree)
        .build()
        .unwrap()
}

fn paged_write_op(dtype: DType) -> StateWriteKvOp {
    StateWriteKvOp {
        cache_dtype: dtype,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: None,
        handle: StateHandle::new(0, StateKind::KvPaged),
    }
}

fn paged_attn_op(mask: AttentionMask, sinks: u32, softcap: Option<f32>) -> AttentionOp {
    AttentionOp {
        softmax_scale: 0.25,
        mask,
        sinks,
        logit_softcap: softcap,
        mla: None,
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::KvPaged),
    }
}

/// Fills a cache with `total` rows per (seq-independent) layout, then returns
/// it. The write uses a `ctx=0` meta; step metas with the same totals share
/// the identical block layout by construction.
#[allow(clippy::too_many_arguments)]
fn fill_paged(
    k_full: &[f32],
    v_full: &[f32],
    total: usize,
    hkv: usize,
    d: usize,
    dv: usize,
    dtype: DType,
    max_blocks: usize,
) -> KvPagedCache {
    let mut cache = KvPagedCache::new(max_blocks, hkv, d, dv, dtype).unwrap();
    let meta = build_meta(&[0], &[total as u32], max_blocks as u32, &[0], &[], None);
    let k_buf = TypedBuffer::from_f32(&[total, hkv, d], k_full);
    let v_buf = TypedBuffer::from_f32(&[total, hkv, dv], v_full);
    state_write_kv_paged(
        &paged_write_op(dtype),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap();
    cache
}

/// Gathers dequantized K/V rows for hand-listed absolute positions of one
/// sequence. The position lists are written by hand in each test (never
/// derived from the implementation's own visibility logic).
fn gather_kv(
    cache: &KvPagedCache,
    meta: &BatchMeta,
    seq: usize,
    positions: &[u32],
    kv_head: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut ks = Vec::new();
    let mut vs = Vec::new();
    for &p in positions {
        let b = (p / 32) as usize;
        let lane = (p % 32) as usize;
        let id = meta.block_table()[seq * meta.max_blocks() as usize + b];
        assert_ne!(id, SENTINEL);
        let slot = id as usize * 32 + lane;
        ks.push(
            (0..cache.d())
                .map(|d| cache.read_k_f32(slot, kv_head, d).unwrap() as f64)
                .collect(),
        );
        vs.push(
            (0..cache.dv())
                .map(|d| cache.read_v_f32(slot, kv_head, d).unwrap() as f64)
                .collect(),
        );
    }
    (ks, vs)
}

/// Runs paged attention for one step meta and returns the f32 output.
#[allow(clippy::too_many_arguments)]
fn run_paged(
    op: &AttentionOp,
    q_data: &[f32],
    t: usize,
    h: usize,
    d: usize,
    dv: usize,
    meta: &BatchMeta,
    cache: &KvPagedCache,
) -> Vec<f32> {
    let q_buf = TypedBuffer::from_f32(&[t, h, d], q_data);
    let mut o_buf = TypedBuffer::zeros(&[t, h, dv], DType::F32);
    attention_paged(
        op,
        &q_buf.as_view(),
        meta,
        0,
        cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap();
    o_buf.to_f32_vec()
}

/// Checks one output row against the dense oracle.
fn check_row(actual: &[f32], base: usize, expected: &[f64], tol: &Tolerance, ctx: &str) {
    for (j, &e) in expected.iter().enumerate() {
        tol.assert_within(actual[base + j] as f64, e, &format!("{ctx} dim {j}"));
    }
}

#[test]
fn decode_single_token_matches_dense_f64() {
    let mut rng = SeededRng::new(0xA1_7001);
    let (hkv, h, d, dv) = (2, 2, 8, 8);
    let (ctx, total) = (5u32, 6usize);
    let k_full = f32_data(&mut rng, total * hkv * d, 1.0);
    let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[1], 4, &[0], &[], None);
    assert_eq!(
        meta.block_table(),
        build_meta(&[0], &[6], 4, &[0], &[], None).block_table()
    );
    let q_data = f32_data(&mut rng, h * d, 1.0);
    let out = run_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_data,
        1,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let tol = Tolerance::f16_bf16();
    let group = h / hkv;
    for head in 0..h {
        let q_row: Vec<f64> = q_data[head * d..(head + 1) * d]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (ks, vs) = gather_kv(&cache, &meta, 0, &[0, 1, 2, 3, 4, 5], head / group);
        let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
        check_row(
            &out,
            head * dv,
            &expected,
            &tol,
            &format!("decode head {head}"),
        );
    }
}

#[test]
fn prefill_chunk_matches_dense_f64() {
    let mut rng = SeededRng::new(0xA1_7004);
    let (hkv, h, d, dv) = (2, 2, 8, 8);
    let (ctx, qlen, total) = (3u32, 4u32, 7usize);
    let k_full = f32_data(&mut rng, total * hkv * d, 0.8);
    let v_full = f32_data(&mut rng, total * hkv * dv, 0.8);
    // Prefill keeps write and attend separate: the write lands first.
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[qlen], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, qlen as usize * h * d, 0.8);
    let out = run_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_data,
        qlen as usize,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let tol = Tolerance::f16_bf16();
    for qi in 0..qlen as usize {
        // Hand-listed causal set for chunk query qi at absolute ctx + qi.
        let positions: Vec<u32> = (0..ctx + qi as u32 + 1).collect();
        for head in 0..h {
            let base = (qi * h + head) * d;
            let q_row: Vec<f64> = q_data[base..base + d].iter().map(|&v| v as f64).collect();
            let (ks, vs) = gather_kv(&cache, &meta, 0, &positions, head);
            let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
            check_row(
                &out,
                (qi * h + head) * dv,
                &expected,
                &tol,
                &format!("prefill qi {qi} head {head}"),
            );
        }
    }
}

#[test]
fn spec_verify_grouped_query_matches_dense_f64() {
    let mut rng = SeededRng::new(0xA1_7005);
    let (hkv, h, d, dv) = (2, 4, 8, 8);
    let (ctx, qlen, total) = (6u32, 3u32, 9usize);
    let k_full = f32_data(&mut rng, total * hkv * d, 0.7);
    let v_full = f32_data(&mut rng, total * hkv * dv, 0.7);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[qlen], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, qlen as usize * h * d, 0.7);
    let out = run_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_data,
        qlen as usize,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let tol = Tolerance::f16_bf16();
    let group = h / hkv;
    for qi in 0..qlen as usize {
        let positions: Vec<u32> = (0..ctx + qi as u32 + 1).collect();
        for head in 0..h {
            // GQA: query head h reads KV head h / (H / Hkv); the oracle is
            // fed that head's rows, so a wrong grouping fails the comparison.
            let kv_head = head / group;
            let base = (qi * h + head) * d;
            let q_row: Vec<f64> = q_data[base..base + d].iter().map(|&v| v as f64).collect();
            let (ks, vs) = gather_kv(&cache, &meta, 0, &positions, kv_head);
            let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
            check_row(
                &out,
                (qi * h + head) * dv,
                &expected,
                &tol,
                &format!("verify qi {qi} head {head} (kv {kv_head})"),
            );
        }
    }
}

#[test]
fn mqa_single_kv_head_matches_dense_f64() {
    let mut rng = SeededRng::new(0xA1_7006);
    let (hkv, h, d, dv) = (1, 4, 8, 8);
    let (ctx, total) = (4u32, 5usize);
    let k_full = f32_data(&mut rng, total * hkv * d, 1.0);
    let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[1], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, h * d, 1.0);
    let out = run_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_data,
        1,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let tol = Tolerance::f16_bf16();
    for head in 0..h {
        let q_row: Vec<f64> = q_data[head * d..(head + 1) * d]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (ks, vs) = gather_kv(&cache, &meta, 0, &[0, 1, 2, 3, 4], 0);
        let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
        check_row(
            &out,
            head * dv,
            &expected,
            &tol,
            &format!("mqa head {head}"),
        );
    }
}

#[test]
fn logit_softcap_matches_dense_f64() {
    let mut rng = SeededRng::new(0xA1_7007);
    let (hkv, h, d, dv) = (2, 2, 8, 8);
    let (ctx, total) = (5u32, 6usize);
    // Larger magnitudes so the cap (2.0) actually binds.
    let k_full = f32_data(&mut rng, total * hkv * d, 3.0);
    let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[1], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, h * d, 3.0);
    let out = run_paged(
        &paged_attn_op(AttentionMask::Causal, 0, Some(2.0)),
        &q_data,
        1,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let tol = Tolerance::f16_bf16();
    for head in 0..h {
        let q_row: Vec<f64> = q_data[head * d..(head + 1) * d]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (ks, vs) = gather_kv(&cache, &meta, 0, &[0, 1, 2, 3, 4, 5], head);
        let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, Some(2.0));
        check_row(
            &out,
            head * dv,
            &expected,
            &tol,
            &format!("softcap head {head}"),
        );
    }
}

#[test]
fn causal_window_with_sinks_matches_dense_f64() {
    let mut rng = SeededRng::new(0xA1_7008);
    let (hkv, h, d, dv) = (2, 2, 8, 8);
    // ctx 70, decode one more (total 71, 3 blocks); window 32 keeps
    // positions 38..71, sinks keep 0..4; the survivor set is hand-listed.
    let (ctx, total, window, sinks) = (70u32, 71usize, 32u32, 4u32);
    let window_start = ctx + 1 - window;
    assert_eq!(window_start, 39);
    let k_full = f32_data(&mut rng, total * hkv * d, 0.6);
    let v_full = f32_data(&mut rng, total * hkv * dv, 0.6);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[1], 4, &[window_start], &[], None);
    let q_data = f32_data(&mut rng, h * d, 0.6);
    let out = run_paged(
        &paged_attn_op(AttentionMask::CausalWindow(window), sinks, None),
        &q_data,
        1,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let survivors: Vec<u32> = (0..sinks).chain(window_start..ctx + 1).collect();
    assert!(!survivors.contains(&10));
    assert!(survivors.contains(&2));
    let tol = Tolerance::f16_bf16();
    for head in 0..h {
        let q_row: Vec<f64> = q_data[head * d..(head + 1) * d]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (ks, vs) = gather_kv(&cache, &meta, 0, &survivors, head);
        let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
        check_row(
            &out,
            head * dv,
            &expected,
            &tol,
            &format!("window+sinks head {head}"),
        );
    }
}

#[test]
fn window_ignores_evicted_middle_positions() {
    let mut rng = SeededRng::new(0xA1_7009);
    let (hkv, h, d, dv) = (1, 1, 8, 8);
    let (ctx, total, window, sinks) = (70u32, 71usize, 32u32, 4u32);
    let window_start = ctx + 1 - window;
    let k_full = f32_data(&mut rng, total * hkv * d, 0.6);
    let v_full = f32_data(&mut rng, total * hkv * dv, 0.6);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[1], 4, &[window_start], &[], None);
    let q_data = f32_data(&mut rng, h * d, 0.6);
    let op = paged_attn_op(AttentionMask::CausalWindow(window), sinks, None);
    let out_before = run_paged(&op, &q_data, 1, h, d, dv, &meta, &cache);
    // Overwrite an evicted middle position (p=10) with garbage: the output
    // must not move. Overwriting a sink (p=2) must move it.
    let mut k_evict = k_full.clone();
    let mut v_evict = v_full.clone();
    for dd in 0..d {
        k_evict[10 * hkv * d + dd] = 25.0;
    }
    for dd in 0..dv {
        v_evict[10 * hkv * dv + dd] = -25.0;
    }
    let cache_evict = fill_paged(&k_evict, &v_evict, total, hkv, d, dv, DType::F16, 4);
    let out_evict = run_paged(&op, &q_data, 1, h, d, dv, &meta, &cache_evict);
    assert_eq!(
        out_before, out_evict,
        "evicted position leaked into attention"
    );
    let mut k_sink = k_full.clone();
    for dd in 0..d {
        k_sink[2 * hkv * d + dd] = 25.0;
    }
    let cache_sink = fill_paged(&k_sink, &v_full, total, hkv, d, dv, DType::F16, 4);
    let out_sink = run_paged(&op, &q_data, 1, h, d, dv, &meta, &cache_sink);
    assert_ne!(out_before, out_sink, "sink position had no effect");
}

#[test]
fn sinks_positive_under_causal_never_fails_closed() {
    let mut rng = SeededRng::new(0xA1_700A);
    let (hkv, h, d, dv) = (2, 2, 8, 8);
    let (ctx, total) = (5u32, 6usize);
    let k_full = f32_data(&mut rng, total * hkv * d, 1.0);
    let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[1], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, h * d, 1.0);
    // Under Causal every position is retained, so sinks > 0 is a no-op that
    // must run, not a refusal.
    let out_plain = run_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_data,
        1,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let out_sinks = run_paged(
        &paged_attn_op(AttentionMask::Causal, 3, None),
        &q_data,
        1,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    assert_eq!(out_plain, out_sinks);
}

#[test]
fn tree_mask_matches_explicit_ancestor_sets() {
    let mut rng = SeededRng::new(0xA1_700B);
    let (hkv, h, d, dv) = (1, 1, 8, 8);
    let (ctx, qlen, total) = (2u32, 3u32, 5usize);
    // Flat tokens 0,1,2 are the drafts: parents [-1, 0, 0], i.e. token 0 is
    // the root, tokens 1 and 2 are its children (siblings of each other).
    let parents = vec![-1, 0, 0];
    let ancestors = vec![
        true, false, false, // tok 0 sees itself
        true, true, false, // tok 1 sees root + itself
        true, false, true, // tok 2 sees root + itself (not its sibling)
    ];
    let tree = TreeMask::new(parents, 3, ancestors).unwrap();
    let k_full = f32_data(&mut rng, total * hkv * d, 0.9);
    let v_full = f32_data(&mut rng, total * hkv * dv, 0.9);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[qlen], 4, &[0], &[], Some(tree));
    let q_data = f32_data(&mut rng, qlen as usize * h * d, 0.9);
    let out = run_paged(
        &paged_attn_op(AttentionMask::Tree, 0, None),
        &q_data,
        qlen as usize,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let tol = Tolerance::f16_bf16();
    // Hand-listed visible absolute positions per draft (context 0,1 plus the
    // draft itself at ctx+qi and its ancestors at ctx+col).
    let visible: [Vec<u32>; 3] = [vec![0, 1, 2], vec![0, 1, 2, 3], vec![0, 1, 2, 4]];
    for (qi, positions) in visible.iter().enumerate() {
        let base = qi * d;
        let q_row: Vec<f64> = q_data[base..base + d].iter().map(|&v| v as f64).collect();
        let (ks, vs) = gather_kv(&cache, &meta, 0, positions, 0);
        let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
        check_row(&out, qi * dv, &expected, &tol, &format!("tree qi {qi}"));
    }
}

#[test]
fn tree_ignores_non_ancestor_siblings() {
    let mut rng = SeededRng::new(0xA1_700C);
    let (hkv, h, d, dv) = (1, 1, 8, 8);
    let (ctx, qlen, total) = (2u32, 3u32, 5usize);
    let parents = vec![-1, 0, 0];
    let ancestors = vec![true, false, false, true, true, false, true, false, true];
    let tree = TreeMask::new(parents, 3, ancestors).unwrap();
    let k_full = f32_data(&mut rng, total * hkv * d, 0.9);
    let v_full = f32_data(&mut rng, total * hkv * dv, 0.9);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[ctx], &[qlen], 4, &[0], &[], Some(tree));
    let q_data = f32_data(&mut rng, qlen as usize * h * d, 0.9);
    let op = paged_attn_op(AttentionMask::Tree, 0, None);
    let out_before = run_paged(&op, &q_data, qlen as usize, h, d, dv, &meta, &cache);
    // Token 2 (sibling) is not an ancestor of token 1: rewriting its rows
    // must leave token 1's output bytes identical while moving token 2's.
    let mut k_rw = k_full.clone();
    let mut v_rw = v_full.clone();
    for dd in 0..d {
        k_rw[4 * hkv * d + dd] = 17.0;
    }
    for dd in 0..dv {
        v_rw[4 * hkv * dv + dd] = -17.0;
    }
    let cache_rw = fill_paged(&k_rw, &v_rw, total, hkv, d, dv, DType::F16, 4);
    let out_after = run_paged(&op, &q_data, qlen as usize, h, d, dv, &meta, &cache_rw);
    assert_eq!(&out_before[0..2 * dv], &out_after[0..2 * dv]);
    assert_ne!(&out_before[2 * dv..3 * dv], &out_after[2 * dv..3 * dv]);
}

fn mla_write_op(dtype: DType, rank: u32, rope: u32) -> StateWriteKvOp {
    StateWriteKvOp {
        cache_dtype: dtype,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: Some(MlaLatent {
            kv_lora_rank: rank,
            rope_dim: rope,
        }),
        handle: StateHandle::new(0, StateKind::KvLatent),
    }
}

fn mla_attn_op(rank: u32, rope: u32, sinks: u32) -> AttentionOp {
    AttentionOp {
        softmax_scale: 0.25,
        mask: AttentionMask::Causal,
        sinks,
        logit_softcap: None,
        mla: Some(MlaAttentionSpec {
            q_lora_rank: None,
            kv_lora_rank: rank,
            qk_nope_dim: rank,
            qk_rope_dim: rope,
            v_dim: rank,
        }),
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::KvLatent),
    }
}

/// Fills a latent cache through the exact split form and returns it.
fn fill_latent(
    c_full: &[f32],
    r_full: &[f32],
    total: usize,
    rank: usize,
    rope: usize,
    dtype: DType,
    max_blocks: usize,
) -> KvLatentCache {
    let mut cache = KvLatentCache::new(max_blocks, rank, rope, dtype).unwrap();
    let meta = build_meta(&[0], &[total as u32], max_blocks as u32, &[0], &[], None);
    let c_buf = TypedBuffer::from_f32(&[total, 1, rank], c_full);
    let r_buf = TypedBuffer::from_f32(&[total, 1, rope], r_full);
    state_write_kv_latent(
        &mla_write_op(dtype, rank as u32, rope as u32),
        &c_buf.as_view(),
        &r_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap();
    cache
}

fn gather_latent(
    cache: &KvLatentCache,
    meta: &BatchMeta,
    seq: usize,
    positions: &[u32],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut cs = Vec::new();
    let mut rs = Vec::new();
    for &p in positions {
        let b = (p / 32) as usize;
        let lane = (p % 32) as usize;
        let id = meta.block_table()[seq * meta.max_blocks() as usize + b];
        assert_ne!(id, SENTINEL);
        let slot = id as usize * 32 + lane;
        cs.push(
            (0..cache.latent())
                .map(|d| cache.read_latent_f32(slot, d).unwrap() as f64)
                .collect(),
        );
        rs.push(
            (0..cache.rope())
                .map(|d| cache.read_rope_f32(slot, d).unwrap() as f64)
                .collect(),
        );
    }
    (cs, rs)
}

#[test]
fn mla_absorbed_split_form_matches_dense_f64() {
    let mut rng = SeededRng::new(0xA1_700D);
    let (h, rank, rope) = (2, 8, 4);
    let (ctx, total) = (5u32, 6usize);
    let c_full = f32_data(&mut rng, total * rank, 0.8);
    let r_full = f32_data(&mut rng, total * rope, 0.8);
    let cache = fill_latent(&c_full, &r_full, total, rank, rope, DType::F16, 4);
    let meta = build_meta(&[ctx], &[1], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, h * (rank + rope), 0.8);
    let q_buf = TypedBuffer::from_f32(&[1, h, rank + rope], &q_data);
    let mut o_buf = TypedBuffer::zeros(&[1, h, rank], DType::F32);
    attention_mla(
        &mla_attn_op(rank as u32, rope as u32, 0),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap();
    let out = o_buf.to_f32_vec();
    let tol = Tolerance::f16_bf16();
    for head in 0..h {
        let base = head * (rank + rope);
        let q_nope: Vec<f64> = q_data[base..base + rank]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let q_rope: Vec<f64> = q_data[base + rank..base + rank + rope]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (cs, rs) = gather_latent(&cache, &meta, 0, &[0, 1, 2, 3, 4, 5]);
        let expected = mla_row_f64_reference(&q_nope, &q_rope, &cs, &rs, 0.25, None);
        check_row(
            &out,
            head * rank,
            &expected,
            &tol,
            &format!("mla head {head}"),
        );
    }
}

#[test]
fn mla_combined_form_matches_split_form() {
    let mut rng = SeededRng::new(0xA1_700E);
    let (h, rank, rope) = (2, 8, 4);
    let (ctx, total) = (5u32, 6usize);
    let c_full = f32_data(&mut rng, total * rank, 0.8);
    let r_full = f32_data(&mut rng, total * rope, 0.8);
    let split = fill_latent(&c_full, &r_full, total, rank, rope, DType::F16, 4);
    // Combined operand 0 = [latent | rope] per token; operand 1 is a
    // T/H-matching placeholder whose values are not stored (SI-44).
    let mut combined = vec![0.0f32; total * (rank + rope)];
    for t in 0..total {
        combined[t * (rank + rope)..t * (rank + rope) + rank]
            .copy_from_slice(&c_full[t * rank..(t + 1) * rank]);
        combined[t * (rank + rope) + rank..(t + 1) * (rank + rope)]
            .copy_from_slice(&r_full[t * rope..(t + 1) * rope]);
    }
    let mut cache = KvLatentCache::new(4, rank, rope, DType::F16).unwrap();
    let meta = build_meta(&[0], &[total as u32], 4, &[0], &[], None);
    let c_buf = TypedBuffer::from_f32(&[total, 1, rank + rope], &combined);
    let r_buf = TypedBuffer::from_f32(&[total, 1, rope], &r_full);
    state_write_kv_latent(
        &mla_write_op(DType::F16, rank as u32, rope as u32),
        &c_buf.as_view(),
        &r_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap();
    for slot in 0..total {
        for d in 0..rank {
            assert_eq!(
                cache.read_latent_f32(slot, d).unwrap(),
                split.read_latent_f32(slot, d).unwrap(),
                "combined/split latent slot {slot} dim {d}"
            );
        }
        for d in 0..rope {
            assert_eq!(
                cache.read_rope_f32(slot, d).unwrap(),
                split.read_rope_f32(slot, d).unwrap(),
                "combined/split rope slot {slot} dim {d}"
            );
        }
    }
    // And identical attention outputs through both caches.
    let step = build_meta(&[ctx], &[1], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, h * (rank + rope), 0.8);
    let run = |cache: &KvLatentCache| {
        let q_buf = TypedBuffer::from_f32(&[1, h, rank + rope], &q_data);
        let mut o_buf = TypedBuffer::zeros(&[1, h, rank], DType::F32);
        attention_mla(
            &mla_attn_op(rank as u32, rope as u32, 0),
            &q_buf.as_view(),
            &step,
            0,
            cache,
            &mut o_buf.as_view_mut(),
        )
        .unwrap();
        o_buf.to_f32_vec()
    };
    assert_eq!(run(&split), run(&cache));
}

#[test]
fn quantized_caches_match_dequant_oracle() {
    for (dtype, seed) in [(DType::I8, 0xA1_700Fu64), (DType::E4m3, 0xA1_7010u64)] {
        let mut rng = SeededRng::new(seed);
        let (hkv, h, d, dv) = (2, 4, 16, 16);
        let (ctx, qlen, total) = (9u32, 3u32, 12usize);
        let k_full = f32_data(&mut rng, total * hkv * d, 1.0);
        let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
        let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, dtype, 4);
        let meta = build_meta(&[ctx], &[qlen], 4, &[0], &[], None);
        let q_data = f32_data(&mut rng, qlen as usize * h * d, 1.0);
        let out = run_paged(
            &paged_attn_op(AttentionMask::Causal, 0, None),
            &q_data,
            qlen as usize,
            h,
            d,
            dv,
            &meta,
            &cache,
        );
        // Oracle over dequantized rows isolates the attention math from the
        // quantization grid; round-trip bounds live in their own tests.
        let tol = Tolerance::f32();
        let group = h / hkv;
        for qi in 0..qlen as usize {
            let positions: Vec<u32> = (0..ctx + qi as u32 + 1).collect();
            for head in 0..h {
                let base = (qi * h + head) * d;
                let q_row: Vec<f64> = q_data[base..base + d].iter().map(|&v| v as f64).collect();
                let (ks, vs) = gather_kv(&cache, &meta, 0, &positions, head / group);
                let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
                check_row(
                    &out,
                    (qi * h + head) * dv,
                    &expected,
                    &tol,
                    &format!("{dtype:?} qi {qi} head {head}"),
                );
            }
        }
    }
}

#[test]
fn cache_round_trip_within_tolerance_per_dtype() {
    let mut rng = SeededRng::new(0xA1_7011);
    let (hkv, d, dv, total) = (2, 16, 16, 5usize);
    let k_full = f32_data(&mut rng, total * hkv * d, 1.0);
    let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
    for (dtype, tol) in [
        (DType::F16, Tolerance::f16_bf16()),
        (DType::I8, Tolerance::i8_weight()),
        (DType::E4m3, Tolerance::e4m3_cache()),
    ] {
        let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, dtype, 4);
        for slot in 0..total {
            for head in 0..hkv {
                for dd in 0..d {
                    let expected = k_full[(slot * hkv + head) * d + dd] as f64;
                    let actual = cache.read_k_f32(slot, head, dd).unwrap() as f64;
                    tol.assert_within(
                        actual,
                        expected,
                        &format!("{dtype:?} k slot {slot} head {head} dim {dd}"),
                    );
                }
                for dd in 0..dv {
                    let expected = v_full[(slot * hkv + head) * dv + dd] as f64;
                    let actual = cache.read_v_f32(slot, head, dd).unwrap() as f64;
                    tol.assert_within(
                        actual,
                        expected,
                        &format!("{dtype:?} v slot {slot} head {head} dim {dd}"),
                    );
                }
            }
        }
    }
}

#[test]
fn latent_rope_stored_always_f16() {
    let mut rng = SeededRng::new(0xA1_7012);
    let (rank, rope, total) = (8, 4, 5usize);
    let c_full = f32_data(&mut rng, total * rank, 1.0);
    let r_full = f32_data(&mut rng, total * rope, 1.0);
    // Even with an i8 latent part, the rope part round-trips at f16
    // precision (Spec 3 §2), far tighter than the latent grid.
    let cache = fill_latent(&c_full, &r_full, total, rank, rope, DType::I8, 4);
    let tol = Tolerance::f16_bf16();
    for slot in 0..total {
        for dd in 0..rope {
            tol.assert_within(
                cache.read_rope_f32(slot, dd).unwrap() as f64,
                r_full[slot * rope + dd] as f64,
                &format!("rope slot {slot} dim {dd}"),
            );
        }
    }
    // ... while the latent part only meets the i8 bound.
    let loose = Tolerance::i8_weight();
    for slot in 0..total {
        for dd in 0..rank {
            loose.assert_within(
                cache.read_latent_f32(slot, dd).unwrap() as f64,
                c_full[slot * rank + dd] as f64,
                &format!("latent slot {slot} dim {dd}"),
            );
        }
    }
}

#[test]
fn attention_is_deterministic_across_runs() {
    let mut rng = SeededRng::new(0xA1_7013);
    let (hkv, h, d, dv) = (2, 4, 16, 16);
    let (ctx, qlen, total) = (9u32, 3u32, 12usize);
    let k_full = f32_data(&mut rng, total * hkv * d, 1.0);
    let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::E4m3, 4);
    let meta = build_meta(&[ctx], &[qlen], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, qlen as usize * h * d, 1.0);
    let op = paged_attn_op(AttentionMask::Causal, 0, Some(30.0));
    let run = || run_paged(&op, &q_data, qlen as usize, h, d, dv, &meta, &cache);
    assert_eq!(run(), run(), "two runs diverged");
}

#[test]
fn batch_padding_invariance_alone_padded_embedded() {
    let mut rng = SeededRng::new(0xA1_7014);
    let (hkv, h, d, dv) = (1, 2, 8, 8);
    // Sequence A in isolation.
    let k_a = f32_data(&mut rng, 4 * hkv * d, 1.0);
    let v_a = f32_data(&mut rng, 4 * hkv * dv, 1.0);
    let q_a = f32_data(&mut rng, h * d, 1.0);
    let cache_a = fill_paged(&k_a, &v_a, 4, hkv, d, dv, DType::F16, 4);
    let meta_a = build_meta(&[3], &[1], 4, &[0], &[], None);
    let op = paged_attn_op(AttentionMask::Causal, 0, None);
    let out_alone = run_paged(&op, &q_a, 1, h, d, dv, &meta_a, &cache_a);
    // Sequence A embedded among random neighbors B and C in one batch.
    // Blocks: A uses 0, B uses 1-2, C uses 3; slots stay disjoint.
    let k_b = f32_data(&mut rng, 33 * hkv * d, 2.0);
    let v_b = f32_data(&mut rng, 33 * hkv * dv, 2.0);
    let k_c = f32_data(&mut rng, 2 * hkv * d, 2.0);
    let v_c = f32_data(&mut rng, 2 * hkv * dv, 2.0);
    let mut cache = KvPagedCache::new(4, hkv, d, dv, DType::F16).unwrap();
    let s = 3usize;
    let max_blocks = 2u32;
    // Hand-built 3-sequence meta: A ctx 3 qlen 1, B ctx 32 qlen 1, C ctx 0 qlen 2.
    let table = vec![0, SENTINEL, 1, 2, 3, SENTINEL];
    assert_eq!(table.len(), s * max_blocks as usize);
    let slots = vec![
        3,  // A query at absolute 3 -> block 0 lane 3
        64, // B query at absolute 32 -> block 2 lane 0
        96, // C query 0 at absolute 0 -> block 3 lane 0
        97, // C query 1 at absolute 1 -> block 3 lane 1
    ];
    let meta = BatchMeta::builder(1, s as u32, 4, max_blocks)
        .seq_ids(vec![7, 3, 9])
        .query_len(vec![1, 1, 2])
        .ctx_len(vec![3, 32, 0])
        .positions(Positions::PerToken(vec![0u32; 4]))
        .slot_map(slots)
        .block_table(table)
        .window_start(vec![0, 0, 0])
        .tree(None)
        .build()
        .unwrap();
    // Direct slot writes keep the layout explicit: A rows at slots 0..4, B
    // rows at slots 32..65, C rows at slots 96..98.
    let write_rows = |cache: &mut KvPagedCache, rows_k: &[f32], rows_v: &[f32], base: usize| {
        let n = rows_k.len() / (hkv * d);
        let mut slots_w = Vec::with_capacity(n);
        for i in 0..n {
            slots_w.push((base + i) as u32);
        }
        let wm = BatchMeta::builder(1, 1, n as u32, max_blocks)
            .seq_ids(vec![0])
            .query_len(vec![n as u32])
            .ctx_len(vec![0])
            .positions(Positions::PerToken(vec![0u32; n]))
            .slot_map(slots_w)
            .block_table(vec![0, 1])
            .window_start(vec![0])
            .tree(None)
            .build()
            .unwrap();
        let kb = TypedBuffer::from_f32(&[n, hkv, d], rows_k);
        let vb = TypedBuffer::from_f32(&[n, hkv, dv], rows_v);
        state_write_kv_paged(
            &paged_write_op(DType::F16),
            &kb.as_view(),
            &vb.as_view(),
            &wm,
            0,
            cache,
        )
        .unwrap();
    };
    write_rows(&mut cache, &k_a, &v_a, 0);
    write_rows(&mut cache, &k_b, &v_b, 32);
    write_rows(&mut cache, &k_c, &v_c, 96);
    let q_b = f32_data(&mut rng, h * d, 2.0);
    let q_c = f32_data(&mut rng, 2 * h * d, 2.0);
    let q_all = [q_a.clone(), q_b, q_c].concat();
    let q_buf = TypedBuffer::from_f32(&[4, h, d], &q_all);
    let mut o_buf = TypedBuffer::zeros(&[4, h, dv], DType::F32);
    attention_paged(
        &op,
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap();
    let out = o_buf.to_f32_vec();
    assert_eq!(
        &out[0..h * dv],
        &out_alone[..],
        "batch neighbors changed A's row"
    );
}

#[test]
fn padded_slots_skipped_without_mutation() {
    let mut rng = SeededRng::new(0xA1_7015);
    let (hkv, d, dv) = (2, 8, 8);
    // Two real tokens plus one SLOT_NONE pad token in the same write.
    let k_data = f32_data(&mut rng, 3 * hkv * d, 1.0);
    let v_data = f32_data(&mut rng, 3 * hkv * dv, 1.0);
    let meta = BatchMeta::builder(1, 1, 3, 2)
        .seq_ids(vec![0])
        .query_len(vec![3])
        .ctx_len(vec![0])
        .positions(Positions::PerToken(vec![0u32; 3]))
        .slot_map(vec![0, 1, u32::MAX])
        .block_table(vec![0, SENTINEL])
        .window_start(vec![0])
        .tree(None)
        .build()
        .unwrap();
    let mut cache = KvPagedCache::new(2, hkv, d, dv, DType::F16).unwrap();
    let k_buf = TypedBuffer::from_f32(&[3, hkv, d], &k_data);
    let v_buf = TypedBuffer::from_f32(&[3, hkv, dv], &v_data);
    state_write_kv_paged(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap();
    assert!(cache.is_written(0));
    assert!(cache.is_written(1));
    assert!(!cache.is_written(2), "pad slot must stay unwritten");
    // And the written rows still match their inputs.
    let tol = Tolerance::f16_bf16();
    for slot in 0..2 {
        for head in 0..hkv {
            for dd in 0..d {
                tol.assert_within(
                    cache.read_k_f32(slot, head, dd).unwrap() as f64,
                    k_data[(slot * hkv + head) * d + dd] as f64,
                    &format!("pad-write k slot {slot} head {head} dim {dd}"),
                );
            }
        }
    }
}

#[test]
fn sentinel_holes_skipped_never_read_as_zero() {
    let mut rng = SeededRng::new(0xA1_7016);
    let (hkv, h, d, dv) = (1, 1, 8, 8);
    // Positions 0..64 exist but block 1 (positions 32..64) was evicted: the
    // hole must be skipped even though its slots were never written.
    let total = 96usize;
    let k_full = f32_data(&mut rng, total * hkv * d, 0.5);
    let v_full = f32_data(&mut rng, total * hkv * dv, 0.5);
    let mut cache = KvPagedCache::new(4, hkv, d, dv, DType::F16).unwrap();
    // Write only blocks 0 and 2 (slots 0..32 and 64..96).
    for (base, n) in [(0usize, 32usize), (64usize, 32usize)] {
        let wm = BatchMeta::builder(1, 1, n as u32, 4)
            .seq_ids(vec![0])
            .query_len(vec![n as u32])
            .ctx_len(vec![0])
            .positions(Positions::PerToken(vec![0u32; n]))
            .slot_map((base as u32..base as u32 + n as u32).collect())
            .block_table(vec![0, 1, 2, 3])
            .window_start(vec![0])
            .tree(None)
            .build()
            .unwrap();
        let kb = TypedBuffer::from_f32(&[n, hkv, d], &k_full[base * hkv * d..(base + n) * hkv * d]);
        let vb = TypedBuffer::from_f32(
            &[n, hkv, dv],
            &v_full[base * hkv * dv..(base + n) * hkv * dv],
        );
        state_write_kv_paged(
            &paged_write_op(DType::F16),
            &kb.as_view(),
            &vb.as_view(),
            &wm,
            0,
            &mut cache,
        )
        .unwrap();
    }
    assert!(!cache.is_written(40), "hole slot must stay unwritten");
    // Decode at position 95 with window 32 + sinks 0: survivors are
    // 64..96 only (block 1 is a sentinel hole, block 0 is out of window).
    let meta = build_meta(&[95], &[1], 4, &[64], &[(0, 1)], None);
    let q_data = f32_data(&mut rng, h * d, 0.5);
    let out = run_paged(
        &paged_attn_op(AttentionMask::CausalWindow(32), 0, None),
        &q_data,
        1,
        h,
        d,
        dv,
        &meta,
        &cache,
    );
    let survivors: Vec<u32> = (64..96).collect();
    let (ks, vs) = gather_kv(&cache, &meta, 0, &survivors, 0);
    let q_row: Vec<f64> = q_data.iter().map(|&v| v as f64).collect();
    let expected = attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None);
    check_row(&out, 0, &expected, &Tolerance::f16_bf16(), "hole-skip");
}

#[test]
fn retained_unwritten_slot_fails_typed_not_silent_zero() {
    let (hkv, h, d, dv) = (1, 1, 8, 8);
    let cache = KvPagedCache::new(4, hkv, d, dv, DType::F16).unwrap();
    let meta = build_meta(&[3], &[1], 4, &[0], &[], None);
    let q_buf = TypedBuffer::from_f32(&[1, h, d], &vec![0.5f32; h * d]);
    let mut o_buf = TypedBuffer::zeros(&[1, h, dv], DType::F32);
    let err = attention_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("never written"), "unexpected error: {msg}");
    assert_eq!(
        o_buf.to_f32_vec(),
        vec![0.0; h * dv],
        "output mutated on failure"
    );
}

#[test]
fn h_not_multiple_of_hkv_fails_typed_before_mutation() {
    let mut rng = SeededRng::new(0xA1_7017);
    let (hkv, h, d, dv) = (2, 3, 8, 8);
    let total = 4usize;
    let k_full = f32_data(&mut rng, total * hkv * d, 1.0);
    let v_full = f32_data(&mut rng, total * hkv * dv, 1.0);
    let cache = fill_paged(&k_full, &v_full, total, hkv, d, dv, DType::F16, 4);
    let meta = build_meta(&[3], &[1], 4, &[0], &[], None);
    let q_data = f32_data(&mut rng, h * d, 1.0);
    let q_buf = TypedBuffer::from_f32(&[1, h, d], &q_data);
    let mut o_buf = TypedBuffer::from_f32(&[1, h, dv], &vec![f32::NAN; h * dv]);
    let err = attention_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("H (3)"), "error must carry H: {msg}");
    assert!(msg.contains("Hkv (2)"), "error must carry Hkv: {msg}");
    assert!(
        o_buf.to_f32_vec().iter().all(|v| v.is_nan()),
        "output mutated before validation completed"
    );
}

#[test]
fn per_block_granularity_fails_closed() {
    let op = StateWriteKvOp {
        cache_dtype: DType::I8,
        scale_granularity: r9v_ir::CacheScaleGranularity::PerBlock,
        latent: None,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };
    let k_buf = TypedBuffer::from_f32(&[2, 1, 8], &[0.5f32; 16]);
    let v_buf = TypedBuffer::from_f32(&[2, 1, 8], &[0.5f32; 16]);
    let meta = build_meta(&[0], &[2], 2, &[0], &[], None);
    let mut cache = KvPagedCache::new(2, 1, 8, 8, DType::I8).unwrap();
    let err = state_write_kv_paged(
        &op,
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("PerBlock"), "unexpected error: {msg}");
    assert!(!cache.is_written(0) && !cache.is_written(1));
}

#[test]
fn write_validation_collects_every_problem_before_mutation() {
    // Wrong rank on k, wrong dtype on v, bad group: all three must appear.
    let k_buf = TypedBuffer::from_f32(&[2, 8], &[0.5f32; 16]);
    let v_buf = TypedBuffer::from_u32(&[2, 1, 8], &[1u32; 16]);
    let meta = build_meta(&[0], &[2], 2, &[0], &[], None);
    let mut cache = KvPagedCache::new(2, 1, 8, 8, DType::F16).unwrap();
    let err = state_write_kv_paged(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        5,
        &mut cache,
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("RankMismatch"),
        "missing k rank problem: {msg}"
    );
    assert!(
        msg.contains("DTypeMismatch"),
        "missing v dtype problem: {msg}"
    );
    assert!(msg.contains("group 5"), "missing group problem: {msg}");
    assert!(!cache.is_written(0) && !cache.is_written(1));
}

#[test]
fn nonfinite_quantized_write_rejected_f16_passthrough() {
    let (hkv, d, dv) = (1, 8, 8);
    let mut k_data = vec![0.5f32; hkv * d];
    k_data[0] = f32::INFINITY;
    let v_data = vec![0.5f32; hkv * dv];
    let meta = build_meta(&[0], &[1], 2, &[0], &[], None);
    let k_buf = TypedBuffer::from_f32(&[1, hkv, d], &k_data);
    let v_buf = TypedBuffer::from_f32(&[1, hkv, dv], &v_data);
    let mut cache = KvPagedCache::new(2, hkv, d, dv, DType::I8).unwrap();
    let err = state_write_kv_paged(
        &paged_write_op(DType::I8),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("non-finite"));
    assert!(!cache.is_written(0));
    // F16 passes the bit pattern through instead.
    let mut f16cache = KvPagedCache::new(2, hkv, d, dv, DType::F16).unwrap();
    state_write_kv_paged(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut f16cache,
    )
    .unwrap();
    assert!(f16cache.read_k_f32(0, 0, 0).unwrap().is_infinite());
}

#[test]
fn latent_multihead_write_rejected_and_dims_checked() {
    let op = mla_write_op(DType::F16, 8, 4);
    let c_buf = TypedBuffer::from_f32(&[2, 2, 8], &[0.5f32; 32]);
    let r_buf = TypedBuffer::from_f32(&[2, 2, 4], &[0.5f32; 16]);
    let meta = build_meta(&[0], &[2], 2, &[0], &[], None);
    let mut cache = KvLatentCache::new(2, 8, 4, DType::F16).unwrap();
    let err = state_write_kv_latent(
        &op,
        &c_buf.as_view(),
        &r_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("got: 2"),
        "must report offending H"
    );
    assert!(!cache.is_written(0));
}

#[test]
fn mla_nonabsorbed_dims_fail_closed() {
    let op = AttentionOp {
        softmax_scale: 0.25,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: Some(MlaAttentionSpec {
            q_lora_rank: None,
            kv_lora_rank: 8,
            qk_nope_dim: 4,
            qk_rope_dim: 4,
            v_dim: 8,
        }),
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::KvLatent),
    };
    let cache = KvLatentCache::new(2, 8, 4, DType::F16).unwrap();
    let meta = build_meta(&[0], &[1], 2, &[0], &[], None);
    let q_buf = TypedBuffer::from_f32(&[1, 1, 8], &[0.5f32; 8]);
    let mut o_buf = TypedBuffer::zeros(&[1, 1, 8], DType::F32);
    let err = attention_mla(
        &op,
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("SI-46"));
}

#[test]
fn handle_cache_mismatch_fails_typed() {
    let mut rng = SeededRng::new(0xA1_7018);
    let k_data = f32_data(&mut rng, 8, 1.0);
    let v_data = f32_data(&mut rng, 8, 1.0);
    let meta = build_meta(&[0], &[1], 2, &[0], &[], None);
    let k_buf = TypedBuffer::from_f32(&[1, 1, 8], &k_data);
    let v_buf = TypedBuffer::from_f32(&[1, 1, 8], &v_data);
    let mut latent = KvCache::Latent(KvLatentCache::new(2, 8, 4, DType::F16).unwrap());
    let err = state_write_kv(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut latent,
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("KvPaged"));
}

#[test]
fn slot_out_of_bounds_fails_before_mutation() {
    let meta = BatchMeta::builder(1, 1, 1, 2)
        .seq_ids(vec![0])
        .query_len(vec![1])
        .ctx_len(vec![0])
        .positions(Positions::PerToken(vec![0u32; 1]))
        .slot_map(vec![999])
        .block_table(vec![0, SENTINEL])
        .window_start(vec![0])
        .tree(None)
        .build()
        .unwrap();
    let k_buf = TypedBuffer::from_f32(&[1, 1, 8], &[0.5f32; 8]);
    let v_buf = TypedBuffer::from_f32(&[1, 1, 8], &[0.5f32; 8]);
    let mut cache = KvPagedCache::new(2, 1, 8, 8, DType::F16).unwrap();
    let err = state_write_kv_paged(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("999"));
    assert!(!cache.is_written(0));
}

// ---------------------------------------------------------------------------
// Adversarial repair tests: every fail-closed path reworked by the A1.7
// post-rebase repair (unified `from_problems` API, no panic paths, total
// BatchMeta guards, plan-then-commit writes) is exercised here with exact
// typed errors and no-mutation assertions.
// ---------------------------------------------------------------------------

/// Snapshot helper: a distinctive output fill that a correct refusal must
/// leave untouched.
fn sentinel_output(t: usize, h: usize, dv: usize) -> TypedBuffer {
    TypedBuffer::from_f32(&[t, h, dv], &vec![7.25f32; t * h * dv])
}

#[test]
fn write_group_out_of_range_is_typed_before_mutation() {
    // Group 5 on a single-group meta must refuse without touching BatchMeta
    // asserts or the cache, even with otherwise-valid inputs.
    let meta = build_meta(&[0], &[2], 2, &[0], &[], None);
    let k_buf = TypedBuffer::from_f32(&[2, 1, 8], &[0.5f32; 16]);
    let v_buf = TypedBuffer::from_f32(&[2, 1, 8], &[0.5f32; 16]);
    let mut cache = KvPagedCache::new(2, 1, 8, 8, DType::F16).unwrap();
    let err = state_write_kv_paged(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        5,
        &mut cache,
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("group 5"), "must name the bad group: {msg}");
    assert!(!cache.is_written(0) && !cache.is_written(1));
}

#[test]
fn attention_group_out_of_range_leaves_output_untouched() {
    let mut rng = SeededRng::new(0xA1_7101);
    let (hkv, h, d, dv) = (1, 1, 8, 8);
    let k_full = f32_data(&mut rng, 2 * hkv * d, 1.0);
    let v_full = f32_data(&mut rng, 2 * hkv * dv, 1.0);
    let cache = fill_paged(&k_full, &v_full, 2, hkv, d, dv, DType::F16, 2);
    let meta = build_meta(&[1], &[1], 2, &[0], &[], None);
    let q_buf = TypedBuffer::from_f32(&[1, h, d], &f32_data(&mut rng, h * d, 1.0));
    let mut o_buf = sentinel_output(1, h, dv);
    let before = o_buf.to_f32_vec();
    let err = attention_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_buf.as_view(),
        &meta,
        5,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("group 5"),
        "must name the bad group: {err:?}"
    );
    assert_eq!(o_buf.to_f32_vec(), before, "refused op must not touch o");
}

#[test]
fn block_id_beyond_cache_slots_fails_typed_before_output_mutation() {
    // Block id 5 addresses slots 160..191 but the cache holds 64 slots: the
    // visibility pass must refuse with RowIndexOutOfRange, never panic in
    // BatchMeta::block or read out of bounds.
    let meta = BatchMeta::builder(1, 1, 1, 2)
        .seq_ids(vec![0])
        .query_len(vec![1])
        .ctx_len(vec![0])
        .positions(Positions::PerToken(vec![0u32; 1]))
        .slot_map(vec![0])
        .block_table(vec![5, SENTINEL])
        .window_start(vec![0])
        .tree(None)
        .build()
        .unwrap();
    let mut cache = KvPagedCache::new(2, 1, 8, 8, DType::F16).unwrap();
    let k_buf = TypedBuffer::from_f32(&[1, 1, 8], &[0.5f32; 8]);
    let v_buf = TypedBuffer::from_f32(&[1, 1, 8], &[0.5f32; 8]);
    state_write_kv_paged(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap();
    let q_buf = TypedBuffer::from_f32(&[1, 1, 8], &[0.5f32; 8]);
    let mut o_buf = sentinel_output(1, 1, 8);
    let before = o_buf.to_f32_vec();
    let err = attention_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("RowIndexOutOfRange"),
        "must be a typed range error: {msg}"
    );
    assert_eq!(o_buf.to_f32_vec(), before, "refused op must not touch o");
}

#[test]
fn window_start_beyond_context_reports_empty_visible_set() {
    // Nothing retained (sinks 0, window_start past every position): the
    // pre-pass must report InvalidDistribution, not divide by a zero softmax
    // sum or leave the output half-written.
    let meta = build_meta(&[0], &[2], 2, &[100], &[], None);
    let mut cache = KvPagedCache::new(2, 1, 8, 8, DType::F16).unwrap();
    let k_buf = TypedBuffer::from_f32(&[2, 1, 8], &[0.5f32; 16]);
    let v_buf = TypedBuffer::from_f32(&[2, 1, 8], &[0.5f32; 16]);
    state_write_kv_paged(
        &paged_write_op(DType::F16),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap();
    let q_buf = TypedBuffer::from_f32(&[2, 1, 8], &[0.5f32; 16]);
    let mut o_buf = sentinel_output(2, 1, 8);
    let before = o_buf.to_f32_vec();
    let err = attention_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("InvalidDistribution"),
        "empty visible set must be typed: {err:?}"
    );
    assert_eq!(o_buf.to_f32_vec(), before, "refused op must not touch o");
}

#[test]
fn tree_mask_without_tree_is_typed_before_output_mutation() {
    let meta = build_meta(&[1], &[1], 2, &[0], &[], None);
    let cache = KvPagedCache::new(2, 1, 8, 8, DType::F16).unwrap();
    let q_buf = TypedBuffer::from_f32(&[1, 1, 8], &[0.5f32; 8]);
    let mut o_buf = sentinel_output(1, 1, 8);
    let before = o_buf.to_f32_vec();
    let err = attention_paged(
        &paged_attn_op(AttentionMask::Tree, 0, None),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("BatchMeta.tree"),
        "must demand the tree operand: {err:?}"
    );
    assert_eq!(o_buf.to_f32_vec(), before, "refused op must not touch o");
}

#[test]
fn latent_neither_split_nor_combined_form_fails_before_mutation() {
    // d0 = 7 is neither rank (8) nor rank + rope (12): typed refusal with
    // nothing written, exercising the former debug_assert-only form check.
    let op = mla_write_op(DType::F16, 8, 4);
    let c_buf = TypedBuffer::from_f32(&[2, 1, 7], &[0.5f32; 14]);
    let r_buf = TypedBuffer::from_f32(&[2, 1, 4], &[0.5f32; 8]);
    let meta = build_meta(&[0], &[2], 2, &[0], &[], None);
    let mut cache = KvLatentCache::new(2, 8, 4, DType::F16).unwrap();
    let err = state_write_kv_latent(
        &op,
        &c_buf.as_view(),
        &r_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("combined"), "must name the form rule: {msg}");
    assert!(!cache.is_written(0) && !cache.is_written(1));
}

#[test]
fn latent_combined_form_ignores_operand1_values() {
    // SI-44 lock: combined operand 1 must match on T/H but its values are
    // not stored; the rope region holds operand 0's tail.
    let (rank, rope) = (8usize, 4usize);
    let mut rng = SeededRng::new(0xA1_7107);
    let c_data = f32_data(&mut rng, 2 * (rank + rope), 0.8);
    let garbage = vec![50.0f32; 2 * rope];
    let c_buf = TypedBuffer::from_f32(&[2, 1, rank + rope], &c_data);
    let r_buf = TypedBuffer::from_f32(&[2, 1, rope], &garbage);
    let meta = build_meta(&[0], &[2], 2, &[0], &[], None);
    let mut cache = KvLatentCache::new(2, rank, rope, DType::F16).unwrap();
    state_write_kv_latent(
        &mla_write_op(DType::F16, rank as u32, rope as u32),
        &c_buf.as_view(),
        &r_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap();
    for tok in 0..2 {
        for d in 0..rope {
            let got = cache.read_rope_f32(tok, d).unwrap();
            let want = c_data[tok * (rank + rope) + rank + d];
            assert!(
                (got - want).abs() < 1e-2,
                "rope must come from operand 0 tail, tok {tok} dim {d}: {got} vs {want}"
            );
            assert!(
                (got - 50.0).abs() > 1.0,
                "operand 1 garbage must never be stored, tok {tok} dim {d}: {got}"
            );
        }
    }
}

#[test]
fn e4m3_nonfinite_multitoken_write_leaves_all_slots_unwritten() {
    // Token 1 carries +Inf under an e4m3 cache: the collect-all pass must
    // refuse and leave token 0's slot unwritten too (no partial mutation).
    let (hkv, d, dv) = (1, 8, 8);
    let mut k_data = vec![0.5f32; 2 * hkv * d];
    k_data[hkv * d] = f32::INFINITY;
    let v_data = vec![0.5f32; 2 * hkv * dv];
    let meta = build_meta(&[0], &[2], 2, &[0], &[], None);
    let k_buf = TypedBuffer::from_f32(&[2, hkv, d], &k_data);
    let v_buf = TypedBuffer::from_f32(&[2, hkv, dv], &v_data);
    let mut cache = KvPagedCache::new(2, hkv, d, dv, DType::E4m3).unwrap();
    let err = state_write_kv_paged(
        &paged_write_op(DType::E4m3),
        &k_buf.as_view(),
        &v_buf.as_view(),
        &meta,
        0,
        &mut cache,
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("non-finite"));
    assert!(
        !cache.is_written(0) && !cache.is_written(1),
        "no token may commit when any token fails"
    );
}

#[test]
fn mla_missing_spec_fails_typed_output_untouched() {
    // MLA entry without an MLA spec exercises the former unreachable! path:
    // typed refusal, output untouched.
    let op = AttentionOp {
        softmax_scale: 0.25,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: None,
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::KvLatent),
    };
    let cache = KvLatentCache::new(2, 8, 4, DType::F16).unwrap();
    let meta = build_meta(&[0], &[1], 2, &[0], &[], None);
    let q_buf = TypedBuffer::from_f32(&[1, 1, 12], &[0.5f32; 12]);
    let mut o_buf = sentinel_output(1, 1, 8);
    let before = o_buf.to_f32_vec();
    let err = attention_mla(
        &op,
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("mla"),
        "must demand the MLA spec: {err:?}"
    );
    assert_eq!(o_buf.to_f32_vec(), before, "refused op must not touch o");
}

#[test]
fn retained_unwritten_latent_slot_fails_typed_output_untouched() {
    // Fresh latent cache: position 0 is retained but never written, so MLA
    // attention must refuse instead of reading zeros.
    let cache = KvLatentCache::new(2, 8, 4, DType::F16).unwrap();
    let meta = build_meta(&[0], &[1], 2, &[0], &[], None);
    let q_buf = TypedBuffer::from_f32(&[1, 1, 12], &[0.5f32; 12]);
    let mut o_buf = sentinel_output(1, 1, 8);
    let before = o_buf.to_f32_vec();
    let err = attention_mla(
        &mla_attn_op(8, 4, 0),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("never written"),
        "must name the unwritten slot: {err:?}"
    );
    assert_eq!(o_buf.to_f32_vec(), before, "refused op must not touch o");
}

#[test]
fn h_not_multiple_of_hkv_leaves_output_untouched() {
    // H = 3 over Hkv = 2: typed GQA refusal with the output untouched.
    let meta = build_meta(&[1], &[1], 2, &[0], &[], None);
    let cache = KvPagedCache::new(2, 2, 8, 8, DType::F16).unwrap();
    let q_buf = TypedBuffer::from_f32(&[1, 3, 8], &[0.5f32; 24]);
    let mut o_buf = sentinel_output(1, 3, 8);
    let before = o_buf.to_f32_vec();
    let err = attention_paged(
        &paged_attn_op(AttentionMask::Causal, 0, None),
        &q_buf.as_view(),
        &meta,
        0,
        &cache,
        &mut o_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("multiple"),
        "must name the grouping rule: {err:?}"
    );
    assert_eq!(o_buf.to_f32_vec(), before, "refused op must not touch o");
}

#[test]
fn oracles_truncate_mismatched_rows_without_panic() {
    // The dense oracles are total: mismatched key/value row counts truncate
    // to the shorter side instead of panicking.
    let got = attention_row_f64_reference(
        &[1.0, 0.0],
        &[vec![1.0, 0.0]],
        &[vec![1.0, 0.0], vec![0.0, 1.0]],
        1.0,
        None,
    );
    assert_eq!(got.len(), 2);
    assert!(got.iter().all(|v| v.is_finite()));
    let empty = attention_row_f64_reference(&[1.0], &[], &[], 1.0, None);
    assert!(empty.is_empty());
    let mla = mla_row_f64_reference(
        &[1.0],
        &[1.0],
        &[vec![1.0]],
        &[vec![1.0], vec![2.0]],
        1.0,
        None,
    );
    assert_eq!(mla.len(), 1);
    assert!(mla.iter().all(|v| v.is_finite()));
}
