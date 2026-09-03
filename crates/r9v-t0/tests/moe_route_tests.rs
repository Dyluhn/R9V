// SPDX-License-Identifier: Apache-2.0
//! Tests for scalar T0 `moe_route` (Spec 1 §4.C, §6.1, Card A1.9).

use r9v_common::SeededRng;
use r9v_ir::{DType, MoeGroup, MoeRouteOp, MoeScoring, Op};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::error::T0Error;
use r9v_t0::{execute_moe_op, moe_route, moe_route_f64_reference, Tolerance};

fn next_f32(rng: &mut SeededRng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
    lo + (u as f32) * (hi - lo)
}

fn route_op(top_k: u32, scoring: MoeScoring, renormalize: bool, scale: f32) -> MoeRouteOp {
    MoeRouteOp {
        top_k,
        scoring,
        renormalize,
        group: None,
        scale,
    }
}

#[test]
fn softmax_renormalized_weights_match_f64_oracle() {
    let mut rng = SeededRng::new(0xA19);
    let (t, e, k) = (5, 6, 3);
    let logits: Vec<f32> = (0..t * e).map(|_| next_f32(&mut rng, -3.0, 3.0)).collect();
    let logits_f64: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    let op = route_op(k as u32, MoeScoring::Softmax, true, 1.7);

    let l_buf = TypedBuffer::from_f32(&[t, e], &logits);
    let mut ids_buf = TypedBuffer::zeros(&[t, k], DType::U32);
    let mut w_buf = TypedBuffer::zeros(&[t, k], DType::F32);
    moe_route(
        &op,
        &l_buf.as_view(),
        None,
        &mut ids_buf.as_view_mut(),
        &mut w_buf.as_view_mut(),
    )
    .unwrap();

    let (exp_ids, exp_w) = moe_route_f64_reference(
        &logits_f64,
        t,
        e,
        None,
        k as u32,
        MoeScoring::Softmax,
        true,
        1.7,
    )
    .unwrap();
    assert_eq!(ids_buf.to_u32_vec(), exp_ids);
    let tol = Tolerance::f32();
    for (i, (&actual, &expected)) in w_buf.to_f32_vec().iter().zip(exp_w.iter()).enumerate() {
        tol.assert_within(actual as f64, expected, &format!("weight {i}"));
    }
    // Renormalized rows sum to 1.
    for row in 0..t {
        let sum: f32 = w_buf.to_f32_vec()[row * k..(row + 1) * k].iter().sum();
        tol.assert_within(sum as f64, 1.0, &format!("row {row} sum"));
    }
}

#[test]
fn sigmoid_unrenormalized_weights_match_f64_oracle() {
    let mut rng = SeededRng::new(0xB19);
    let (t, e, k) = (4, 5, 2);
    let logits: Vec<f32> = (0..t * e).map(|_| next_f32(&mut rng, -4.0, 4.0)).collect();
    let logits_f64: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    let op = route_op(k as u32, MoeScoring::Sigmoid, false, 0.5);

    let l_buf = TypedBuffer::from_f32(&[t, e], &logits);
    let mut ids_buf = TypedBuffer::zeros(&[t, k], DType::U32);
    let mut w_buf = TypedBuffer::zeros(&[t, k], DType::F32);
    moe_route(
        &op,
        &l_buf.as_view(),
        None,
        &mut ids_buf.as_view_mut(),
        &mut w_buf.as_view_mut(),
    )
    .unwrap();

    let (exp_ids, exp_w) = moe_route_f64_reference(
        &logits_f64,
        t,
        e,
        None,
        k as u32,
        MoeScoring::Sigmoid,
        false,
        0.5,
    )
    .unwrap();
    assert_eq!(ids_buf.to_u32_vec(), exp_ids);
    let tol = Tolerance::f32();
    for (i, (&actual, &expected)) in w_buf.to_f32_vec().iter().zip(exp_w.iter()).enumerate() {
        tol.assert_within(actual as f64, expected, &format!("weight {i}"));
    }
}

#[test]
fn equal_logits_select_lowest_expert_indices() {
    let (t, e, k) = (2, 6, 3);
    let logits = vec![1.0f32; t * e];
    for scoring in [MoeScoring::Softmax, MoeScoring::Sigmoid] {
        let op = route_op(k as u32, scoring, true, 1.0);
        let l_buf = TypedBuffer::from_f32(&[t, e], &logits);
        let mut ids_buf = TypedBuffer::zeros(&[t, k], DType::U32);
        let mut w_buf = TypedBuffer::zeros(&[t, k], DType::F32);
        moe_route(
            &op,
            &l_buf.as_view(),
            None,
            &mut ids_buf.as_view_mut(),
            &mut w_buf.as_view_mut(),
        )
        .unwrap();
        assert_eq!(ids_buf.to_u32_vec(), vec![0, 1, 2, 0, 1, 2]);
    }
}

#[test]
fn bias_param_matches_manual_logit_shift() {
    let mut rng = SeededRng::new(0xC19);
    let (t, e, k) = (3, 5, 2);
    let logits: Vec<f32> = (0..t * e).map(|_| next_f32(&mut rng, -2.0, 2.0)).collect();
    let bias: Vec<f32> = (0..e).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let op = route_op(k as u32, MoeScoring::Softmax, true, 2.0);

    let l_buf = TypedBuffer::from_f32(&[t, e], &logits);
    let b_buf = TypedBuffer::from_f32(&[e], &bias);
    let mut ids_a = TypedBuffer::zeros(&[t, k], DType::U32);
    let mut w_a = TypedBuffer::zeros(&[t, k], DType::F32);
    moe_route(
        &op,
        &l_buf.as_view(),
        Some(&b_buf.as_view()),
        &mut ids_a.as_view_mut(),
        &mut w_a.as_view_mut(),
    )
    .unwrap();

    let shifted: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| v + bias[i % e])
        .collect();
    let s_buf = TypedBuffer::from_f32(&[t, e], &shifted);
    let mut ids_b = TypedBuffer::zeros(&[t, k], DType::U32);
    let mut w_b = TypedBuffer::zeros(&[t, k], DType::F32);
    moe_route(
        &op,
        &s_buf.as_view(),
        None,
        &mut ids_b.as_view_mut(),
        &mut w_b.as_view_mut(),
    )
    .unwrap();

    assert_eq!(ids_a.to_u32_vec(), ids_b.to_u32_vec());
    assert_eq!(w_a.to_f32_vec(), w_b.to_f32_vec());
}

#[test]
fn run_twice_bit_identical() {
    let mut rng = SeededRng::new(0xD19);
    let (t, e, k) = (6, 8, 4);
    let logits: Vec<f32> = (0..t * e).map(|_| next_f32(&mut rng, -5.0, 5.0)).collect();
    let op = route_op(k as u32, MoeScoring::Softmax, true, 1.0);
    let l_buf = TypedBuffer::from_f32(&[t, e], &logits);
    let run = || {
        let mut ids = TypedBuffer::zeros(&[t, k], DType::U32);
        let mut w = TypedBuffer::zeros(&[t, k], DType::F32);
        moe_route(
            &op,
            &l_buf.as_view(),
            None,
            &mut ids.as_view_mut(),
            &mut w.as_view_mut(),
        )
        .unwrap();
        (ids.to_u32_vec(), w.to_f32_vec())
    };
    let (ids1, w1) = run();
    let (ids2, w2) = run();
    assert_eq!(ids1, ids2);
    assert_eq!(w1, w2);
}

#[test]
fn token_rows_batch_invariant_alone_padded_embedded() {
    // One fixed row must route identically alone, padded, and embedded.
    let target = vec![0.5f32, -1.2, 2.1, 0.0, -0.3, 1.4];
    let (e, k) = (6, 2);
    let op = route_op(k as u32, MoeScoring::Softmax, false, 1.0);
    let route_row = |rows: &[f32]| {
        let t = rows.len() / e;
        let buf = TypedBuffer::from_f32(&[t, e], rows);
        let mut ids = TypedBuffer::zeros(&[t, k], DType::U32);
        let mut w = TypedBuffer::zeros(&[t, k], DType::F32);
        moe_route(
            &op,
            &buf.as_view(),
            None,
            &mut ids.as_view_mut(),
            &mut w.as_view_mut(),
        )
        .unwrap();
        (ids.to_u32_vec(), w.to_f32_vec())
    };
    let mut rng = SeededRng::new(0xE19);
    let noise: Vec<f32> = (0..4 * e).map(|_| next_f32(&mut rng, -3.0, 3.0)).collect();

    let (ids_alone, w_alone) = route_row(&target);
    let mut padded = target.clone();
    padded.extend_from_slice(&vec![0.0; 3 * e]);
    let (ids_padded, w_padded) = route_row(&padded);
    let mut embedded = noise[..2 * e].to_vec();
    embedded.extend_from_slice(&target);
    embedded.extend_from_slice(&noise[2 * e..]);
    let (ids_emb, w_emb) = route_row(&embedded);

    assert_eq!(&ids_alone[..k], &ids_padded[..k]);
    assert_eq!(&w_alone[..k], &w_padded[..k]);
    assert_eq!(&ids_alone[..k], &ids_emb[2 * k..3 * k]);
    assert_eq!(&w_alone[..k], &w_emb[2 * k..3 * k]);
}

#[test]
fn grouped_routing_fails_closed() {
    let op = MoeRouteOp {
        top_k: 2,
        scoring: MoeScoring::Softmax,
        renormalize: true,
        group: Some(MoeGroup {
            n_group: 2,
            topk_group: 1,
        }),
        scale: 1.0,
    };
    let l_buf = TypedBuffer::from_f32(&[2, 4], &[0.1; 8]);
    let mut ids = TypedBuffer::zeros(&[2, 2], DType::U32);
    let mut w = TypedBuffer::zeros(&[2, 2], DType::F32);
    let err = moe_route(
        &op,
        &l_buf.as_view(),
        None,
        &mut ids.as_view_mut(),
        &mut w.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        T0Error::InvalidAttribute {
            op: "moe_route",
            attribute: "group",
            ..
        }
    ));
    // Outputs untouched on refusal.
    assert_eq!(ids.to_u32_vec(), vec![0; 4]);
    assert_eq!(w.to_f32_vec(), vec![0.0; 4]);
}

#[test]
fn nonfinite_logits_collected_with_locations() {
    let op = route_op(2, MoeScoring::Softmax, true, 1.0);
    let mut logits = vec![0.1f32; 3 * 4];
    logits[6] = f32::NAN; // (t=1, e=2)
    logits[8] = f32::INFINITY; // (t=2, e=0)
    let l_buf = TypedBuffer::from_f32(&[3, 4], &logits);
    let mut ids = TypedBuffer::zeros(&[3, 2], DType::U32);
    let mut w = TypedBuffer::zeros(&[3, 2], DType::F32);
    let err = moe_route(
        &op,
        &l_buf.as_view(),
        None,
        &mut ids.as_view_mut(),
        &mut w.as_view_mut(),
    )
    .unwrap_err();
    match err {
        T0Error::Multiple { problems } => {
            assert_eq!(problems.len(), 2);
            let text = format!("{problems:?}");
            assert!(text.contains("(t=1, e=2)"));
            assert!(text.contains("(t=2, e=0)"));
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
    assert_eq!(ids.to_u32_vec(), vec![0; 6]);
}

#[test]
fn sigmoid_renormalize_degenerate_zero_sum_refuses_before_mutation() {
    // Blocker 1: all-`-1000.0` finite logits pass input validation, but the
    // sigmoid arm scores every expert `0.0`, so the selected-K sum is `0.0`
    // and renormalizing would emit `NaN` weights. Must refuse with a typed
    // error naming the row, leaving both outputs untouched.
    let (t, e, k) = (3, 4, 2);
    let op = route_op(k as u32, MoeScoring::Sigmoid, true, 1.0);
    let l_buf = TypedBuffer::from_f32(&[t, e], &vec![-1000.0f32; t * e]);
    let mut ids = TypedBuffer::zeros(&[t, k], DType::U32);
    let mut w = TypedBuffer::zeros(&[t, k], DType::F32);
    let err = moe_route(
        &op,
        &l_buf.as_view(),
        None,
        &mut ids.as_view_mut(),
        &mut w.as_view_mut(),
    )
    .unwrap_err();
    let text = format!("{err:?}");
    assert!(
        text.contains("selected-K sum") && text.contains("SI-51"),
        "expected zero-sum refusal, got {err:?}"
    );
    // Neither output mutated.
    assert_eq!(ids.to_u32_vec(), vec![0; t * k]);
    assert_eq!(w.to_f32_vec(), vec![0.0; t * k]);
}

#[test]
fn sigmoid_renormalize_healthy_row_still_passes() {
    // The zero-sum guard must not fire on ordinary finite rows.
    let op = route_op(2, MoeScoring::Sigmoid, true, 1.0);
    let l_buf = TypedBuffer::from_f32(&[2, 4], &[0.5, -0.2, 1.1, 0.0, -0.7, 0.3, 0.9, -1.5]);
    let mut ids = TypedBuffer::zeros(&[2, 2], DType::U32);
    let mut w = TypedBuffer::zeros(&[2, 2], DType::F32);
    moe_route(
        &op,
        &l_buf.as_view(),
        None,
        &mut ids.as_view_mut(),
        &mut w.as_view_mut(),
    )
    .unwrap();
    for v in w.to_f32_vec() {
        assert!(v.is_finite());
    }
}

#[test]
fn post_bias_overflow_to_nonfinite_refuses_before_mutation() {
    // Finite logits + finite bias whose f32 sum overflows to `+Inf` must be
    // rejected as a non-finite post-bias score, not sorted into NaN weights.
    let (t, e, k) = (2, 3, 2);
    let op = route_op(k as u32, MoeScoring::Softmax, false, 1.0);
    let l_buf = TypedBuffer::from_f32(&[t, e], &vec![2.0e38f32; t * e]);
    let b_buf = TypedBuffer::from_f32(&[e], &vec![2.0e38f32; e]);
    let mut ids = TypedBuffer::zeros(&[t, k], DType::U32);
    let mut w = TypedBuffer::zeros(&[t, k], DType::F32);
    let err = moe_route(
        &op,
        &l_buf.as_view(),
        Some(&b_buf.as_view()),
        &mut ids.as_view_mut(),
        &mut w.as_view_mut(),
    )
    .unwrap_err();
    let text = format!("{err:?}");
    assert!(
        text.contains("post-bias score") && text.contains("SI-51"),
        "expected post-bias refusal, got {err:?}"
    );
    assert_eq!(ids.to_u32_vec(), vec![0; t * k]);
    assert_eq!(w.to_f32_vec(), vec![0.0; t * k]);
}

#[test]
fn simultaneous_defects_report_single_multiple() {
    let op = route_op(2, MoeScoring::Softmax, true, 1.0);
    let l_buf = TypedBuffer::from_f32(&[3], &[0.1; 3]);
    let mut ids = TypedBuffer::zeros(&[3, 2], DType::F32);
    let mut w = TypedBuffer::zeros(&[3, 2], DType::F32);
    let err = moe_route(
        &op,
        &l_buf.as_view(),
        None,
        &mut ids.as_view_mut(),
        &mut w.as_view_mut(),
    )
    .unwrap_err();
    match err {
        T0Error::Multiple { problems } => assert!(problems.len() >= 2),
        other => panic!("expected Multiple, got {other:?}"),
    }
}

#[test]
fn dispatch_enforces_route_arity() {
    let op = Op::MoeRoute(route_op(2, MoeScoring::Softmax, true, 1.0));
    let l_buf = TypedBuffer::from_f32(&[2, 4], &[0.1; 8]);
    let mut ids = TypedBuffer::zeros(&[2, 2], DType::U32);
    let mut w = TypedBuffer::zeros(&[2, 2], DType::F32);
    assert!(execute_moe_op(&op, &[], &mut [ids.as_view_mut(), w.as_view_mut()]).is_err());
    assert!(execute_moe_op(&op, &[l_buf.as_view()], &mut [ids.as_view_mut()]).is_err());
    assert!(execute_moe_op(
        &op,
        &[l_buf.as_view()],
        &mut [ids.as_view_mut(), w.as_view_mut()],
    )
    .is_ok());
    let other = Op::Barrier(r9v_ir::BarrierOp {
        group: r9v_ir::GroupId::new(0),
    });
    assert!(execute_moe_op(
        &other,
        &[l_buf.as_view()],
        &mut [ids.as_view_mut(), w.as_view_mut()],
    )
    .is_err());
}
