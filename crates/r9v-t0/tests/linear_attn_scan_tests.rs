// SPDX-License-Identifier: Apache-2.0
//! Tests for scalar T0 `linear_attn_scan` (Spec 1 §4.E, Card A1.9).

use r9v_common::SeededRng;
use r9v_ir::{DType, LinearAttnKind, LinearAttnScanOp, Op, StateHandle, StateKind};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::{
    execute_state_scan_op, linear_attn_scan_chunked, linear_attn_scan_f64_reference,
    linear_attn_scan_recurrent, SeqLayout, Tolerance,
};

fn scan_op(kind: LinearAttnKind, chunk: u32) -> LinearAttnScanOp {
    LinearAttnScanOp {
        kind,
        chunk,
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::Recurrent),
    }
}

fn next_f32(rng: &mut SeededRng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
    lo + (u as f32) * (hi - lo)
}

struct ScanCase {
    t: usize,
    h: usize,
    d: usize,
    dv: usize,
    s: usize,
    lens: Vec<u32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

fn make_case(seed: u64, t: usize, h: usize, d: usize, dv: usize, lens: &[u32]) -> ScanCase {
    let mut rng = SeededRng::new(seed);
    let s = lens.len();
    ScanCase {
        t,
        h,
        d,
        dv,
        s,
        lens: lens.to_vec(),
        q: (0..t * h * d)
            .map(|_| next_f32(&mut rng, -1.0, 1.0))
            .collect(),
        k: (0..t * h * d)
            .map(|_| next_f32(&mut rng, -1.0, 1.0))
            .collect(),
        v: (0..t * h * dv)
            .map(|_| next_f32(&mut rng, -1.0, 1.0))
            .collect(),
        // Gates near 1 / small keep the recurrence bounded and oracle-meaningful.
        alpha: (0..t * h).map(|_| next_f32(&mut rng, 0.8, 1.0)).collect(),
        beta: (0..t * h).map(|_| next_f32(&mut rng, 0.0, 0.5)).collect(),
        state: (0..s * h * d * dv)
            .map(|_| next_f32(&mut rng, -0.25, 0.25))
            .collect(),
    }
}

fn run_form(
    case: &ScanCase,
    kind: LinearAttnKind,
    chunk: u32,
    chunked: bool,
) -> (Vec<f32>, Vec<f32>) {
    let op = scan_op(kind, chunk);
    let qb = TypedBuffer::from_f32(&[case.t, case.h, case.d], &case.q);
    let kb = TypedBuffer::from_f32(&[case.t, case.h, case.d], &case.k);
    let vb = TypedBuffer::from_f32(&[case.t, case.h, case.dv], &case.v);
    let ab = TypedBuffer::from_f32(&[case.t, case.h], &case.alpha);
    let bb = TypedBuffer::from_f32(&[case.t, case.h], &case.beta);
    let sb = TypedBuffer::from_f32(&[case.s, case.h, case.d, case.dv], &case.state);
    let mut ob = TypedBuffer::zeros(&[case.t, case.h, case.dv], DType::F32);
    let mut sob = TypedBuffer::zeros(&[case.s, case.h, case.d, case.dv], DType::F32);
    let seq = SeqLayout::new(&case.lens).unwrap();
    let r = if chunked {
        linear_attn_scan_chunked(
            &op,
            &qb.as_view(),
            &kb.as_view(),
            &vb.as_view(),
            &ab.as_view(),
            &bb.as_view(),
            &sb.as_view(),
            &seq,
            &mut ob.as_view_mut(),
            &mut sob.as_view_mut(),
        )
    } else {
        linear_attn_scan_recurrent(
            &op,
            &qb.as_view(),
            &kb.as_view(),
            &vb.as_view(),
            &ab.as_view(),
            &bb.as_view(),
            &sb.as_view(),
            &seq,
            &mut ob.as_view_mut(),
            &mut sob.as_view_mut(),
        )
    };
    r.unwrap();
    (ob.to_f32_vec(), sob.to_f32_vec())
}

#[test]
fn chunked_and_recurrent_agree_bit_exact_all_kinds() {
    // The done-when scan agreement test across kinds, chunks, and D != Dv.
    for kind in [
        LinearAttnKind::GatedDeltaNet,
        LinearAttnKind::GLA,
        LinearAttnKind::Mamba2,
    ] {
        for chunk in [1u32, 2, 3, 32, 64] {
            for (d, dv) in [(4, 4), (3, 5), (6, 2)] {
                let case = make_case(0x100 + chunk as u64, 9, 2, d, dv, &[4, 5]);
                let (oc, sc) = run_form(&case, kind, chunk, true);
                let (or, sr) = run_form(&case, kind, chunk, false);
                assert_eq!(oc, or, "kind={kind:?} chunk={chunk} d={d} dv={dv} outputs");
                assert_eq!(sc, sr, "kind={kind:?} chunk={chunk} d={d} dv={dv} states");
            }
        }
    }
}

#[test]
fn both_forms_match_f64_oracle_within_f32_tolerance() {
    for kind in [
        LinearAttnKind::GatedDeltaNet,
        LinearAttnKind::GLA,
        LinearAttnKind::Mamba2,
    ] {
        let case = make_case(0x200, 7, 3, 4, 5, &[2, 5]);
        let (oc, sc) = run_form(&case, kind, 2, true);
        let to64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let (exp_o, exp_s) = linear_attn_scan_f64_reference(
            &to64(&case.q),
            &to64(&case.k),
            &to64(&case.v),
            &to64(&case.alpha),
            &to64(&case.beta),
            case.t,
            case.h,
            case.d,
            case.dv,
            &to64(&case.state),
            case.s,
            &case.lens,
        )
        .unwrap();
        let tol = Tolerance::f32();
        for (i, (&actual, &expected)) in oc.iter().zip(exp_o.iter()).enumerate() {
            tol.assert_within(actual as f64, expected, &format!("kind={kind:?} o[{i}]"));
        }
        for (i, (&actual, &expected)) in sc.iter().zip(exp_s.iter()).enumerate() {
            tol.assert_within(actual as f64, expected, &format!("kind={kind:?} s[{i}]"));
        }
    }
}

#[test]
fn state_out_matches_recompute_from_slot_a() {
    // A-slot discipline: recomputing the accepted prefix from A into a fresh B
    // reproduces the carried-B state (spec 3 §4.2 at T0 level).
    let case = make_case(0x300, 6, 2, 4, 4, &[6]);
    let (o_full, s_full) = run_form(&case, LinearAttnKind::GLA, 64, true);
    // Split: first 4 tokens, then tokens 4..6 from the carried state.
    let mut prefix = make_case(0x300, 4, 2, 4, 4, &[4]);
    prefix.q = case.q[..4 * 2 * 4].to_vec();
    prefix.k = case.k[..4 * 2 * 4].to_vec();
    prefix.v = case.v[..4 * 2 * 4].to_vec();
    prefix.alpha = case.alpha[..4 * 2].to_vec();
    prefix.beta = case.beta[..4 * 2].to_vec();
    prefix.state = case.state.clone();
    let (_, s_prefix) = run_form(&prefix, LinearAttnKind::GLA, 64, false);
    // Recompute the accepted prefix (4 tokens) from A into B again: same state.
    let (_, s_recompute) = run_form(&prefix, LinearAttnKind::GLA, 64, true);
    assert_eq!(s_prefix, s_recompute);
    // And the recomputed state advances the suffix identically.
    let mut suffix = make_case(0x300, 2, 2, 4, 4, &[2]);
    suffix.q = case.q[4 * 2 * 4..].to_vec();
    suffix.k = case.k[4 * 2 * 4..].to_vec();
    suffix.v = case.v[4 * 2 * 4..].to_vec();
    suffix.alpha = case.alpha[4 * 2..].to_vec();
    suffix.beta = case.beta[4 * 2..].to_vec();
    suffix.state = s_prefix.clone();
    let (o_suffix, _) = run_form(&suffix, LinearAttnKind::GLA, 64, false);
    assert_eq!(&o_full[4 * 2 * 4..], o_suffix.as_slice());
    let _ = s_full;
}

#[test]
fn determinism_and_segment_batch_invariance() {
    let case = make_case(0x400, 8, 2, 4, 6, &[3, 5]);
    let (o1, s1) = run_form(&case, LinearAttnKind::Mamba2, 3, true);
    let (o2, s2) = run_form(&case, LinearAttnKind::Mamba2, 3, true);
    assert_eq!(o1, o2);
    assert_eq!(s1, s2);
    // Same tokens as one segment with the first slot's state must reproduce
    // the first segment's outputs exactly.
    let mut first = make_case(0x400, 3, 2, 4, 6, &[3]);
    first.q = case.q[..3 * 2 * 4].to_vec();
    first.k = case.k[..3 * 2 * 4].to_vec();
    first.v = case.v[..3 * 2 * 6].to_vec();
    first.alpha = case.alpha[..3 * 2].to_vec();
    first.beta = case.beta[..3 * 2].to_vec();
    first.state = case.state[..2 * 4 * 6].to_vec();
    let (o_first, _) = run_form(&first, LinearAttnKind::Mamba2, 3, false);
    assert_eq!(&o1[..3 * 2 * 6], o_first.as_slice());
}

#[test]
fn nonfinite_gates_rejected_with_locations() {
    let mut case = make_case(0x500, 3, 2, 2, 2, &[3]);
    case.alpha[2] = f32::NAN;
    case.beta[2 * 2 + 1] = f32::INFINITY;
    let op = scan_op(LinearAttnKind::GLA, 2);
    let qb = TypedBuffer::from_f32(&[3, 2, 2], &case.q);
    let kb = TypedBuffer::from_f32(&[3, 2, 2], &case.k);
    let vb = TypedBuffer::from_f32(&[3, 2, 2], &case.v);
    let ab = TypedBuffer::from_f32(&[3, 2], &case.alpha);
    let bb = TypedBuffer::from_f32(&[3, 2], &case.beta);
    let sb = TypedBuffer::from_f32(&[1, 2, 2, 2], &case.state);
    let mut ob = TypedBuffer::zeros(&[3, 2, 2], DType::F32);
    let mut sob = TypedBuffer::zeros(&[1, 2, 2, 2], DType::F32);
    let seq = SeqLayout::new(&[3]).unwrap();
    let err = linear_attn_scan_chunked(
        &op,
        &qb.as_view(),
        &kb.as_view(),
        &vb.as_view(),
        &ab.as_view(),
        &bb.as_view(),
        &sb.as_view(),
        &seq,
        &mut ob.as_view_mut(),
        &mut sob.as_view_mut(),
    )
    .unwrap_err();
    match err {
        r9v_t0::error::T0Error::Multiple { problems } => {
            assert_eq!(problems.len(), 2);
            let text = format!("{problems:?}");
            assert!(text.contains("(t=1, h=0)"));
            assert!(text.contains("(t=2, h=1)"));
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
    assert_eq!(ob.to_f32_vec(), vec![0.0; 12]);
}

#[test]
fn dispatch_selects_both_scan_forms() {
    let case = make_case(0x600, 4, 1, 2, 2, &[4]);
    let op = Op::LinearAttnScan(scan_op(LinearAttnKind::GLA, 2));
    let run = |chunked: bool| {
        let qb = TypedBuffer::from_f32(&[4, 1, 2], &case.q);
        let kb = TypedBuffer::from_f32(&[4, 1, 2], &case.k);
        let vb = TypedBuffer::from_f32(&[4, 1, 2], &case.v);
        let ab = TypedBuffer::from_f32(&[4, 1], &case.alpha);
        let bb = TypedBuffer::from_f32(&[4, 1], &case.beta);
        let sb = TypedBuffer::from_f32(&[1, 1, 2, 2], &case.state);
        let mut ob = TypedBuffer::zeros(&[4, 1, 2], DType::F32);
        let mut sob = TypedBuffer::zeros(&[1, 1, 2, 2], DType::F32);
        let seq = SeqLayout::new(&[4]).unwrap();
        execute_state_scan_op(
            &op,
            &[
                qb.as_view(),
                kb.as_view(),
                vb.as_view(),
                ab.as_view(),
                bb.as_view(),
            ],
            &sb.as_view(),
            &seq,
            &mut [ob.as_view_mut()],
            &mut sob.as_view_mut(),
            chunked,
        )
        .unwrap();
        ob.to_f32_vec()
    };
    assert_eq!(run(true), run(false));
}
