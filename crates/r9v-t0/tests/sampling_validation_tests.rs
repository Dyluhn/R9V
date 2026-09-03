// SPDX-License-Identifier: Apache-2.0
//! Adversarial validation tests for sampling ops: exact-location logit
//! rejection, overflow, shape/arithmetic/distribution/token/tree/mask refusal,
//! and no output/RNG mutation on failure (Spec 1 §4.F, §6.5, Spec 7 §4,
//! Card A1.8).

use r9v_common::{SeqId, StepId};
use r9v_ir::{IrError, SamplingParams, TreeMask, VerifyMethod};
use r9v_t0::{logits_postprocess, sample, verify, RngState, T0Error};

fn base_params() -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: vec![],
    }
}

#[test]
fn logits_postprocess_rejects_nan_with_exact_location() {
    // Second sequence, second query, token 2 carries the NaN.
    let (s, q, v) = (2, 2, 4);
    let mut logits = vec![0.5f32; s * q * v];
    logits[(q + 1) * v + 2] = f32::NAN;
    let params = vec![base_params(), base_params()];
    let mut probs = vec![0.0f32; s * q * v];
    let err = logits_postprocess(&logits, s, q, v, &params, None, None, &mut probs)
        .expect_err("NaN logit must be rejected");
    assert!(
        matches!(
            err,
            T0Error::InvalidLogit {
                seq: 1,
                query: 1,
                token: 2,
                ..
            }
        ),
        "wrong error or location: {err:?}"
    );
}

#[test]
fn logits_postprocess_rejects_pos_inf_allows_neg_inf() {
    let params = vec![base_params()];
    let mut probs = vec![0.0f32; 3];

    let err = logits_postprocess(
        &[0.0, f32::INFINITY, 1.0],
        1,
        1,
        3,
        &params,
        None,
        None,
        &mut probs,
    )
    .expect_err("+Inf logit must be rejected");
    assert!(
        matches!(err, T0Error::InvalidLogit { token: 1, value, .. } if value == f32::INFINITY),
        "wrong error: {err:?}"
    );

    // Intentional -Inf (impossible token) is legal and gets zero probability.
    logits_postprocess(
        &[f32::NEG_INFINITY, 0.0, 1.0],
        1,
        1,
        3,
        &params,
        None,
        None,
        &mut probs,
    )
    .expect("-Inf logit must be accepted");
    assert_eq!(probs[0], 0.0);
    assert!((probs[1] + probs[2] - 1.0).abs() < 1e-6);
}

#[test]
fn logits_postprocess_rejects_post_transform_overflow_with_location() {
    // Finite inputs that overflow f32 under temperature scaling.
    let params = vec![SamplingParams {
        temperature: 1e-30,
        ..base_params()
    }];
    let mut probs = vec![0.0f32; 2];
    let err = logits_postprocess(&[1e10, 0.0], 1, 1, 2, &params, None, None, &mut probs)
        .expect_err("temperature overflow must be rejected");
    assert!(
        matches!(err, T0Error::InvalidLogit { token: 0, .. }),
        "wrong error or location: {err:?}"
    );

    // Finite logit plus finite bias overflowing to +Inf.
    let mut biased = base_params();
    biased.logit_bias = vec![(0, 3.0e38)];
    let params = vec![biased];
    let err = logits_postprocess(&[3.0e38, 0.0], 1, 1, 2, &params, None, None, &mut probs)
        .expect_err("bias overflow must be rejected");
    assert!(
        matches!(err, T0Error::InvalidLogit { token: 0, .. }),
        "wrong error or location: {err:?}"
    );
}

#[test]
fn logits_postprocess_rejects_non_finite_bias_at_parameter() {
    // Non-finite biases are refused by SamplingParams validation (r9v-ir) with
    // the token and value, surfaced through the typed aggregation path.
    let mut p = base_params();
    p.logit_bias = vec![(1, f32::NAN)];
    let params = vec![p];
    let mut probs = vec![0.0f32; 3];
    let err = logits_postprocess(&[0.0, 0.0, 0.0], 1, 1, 3, &params, None, None, &mut probs)
        .expect_err("NaN bias must be rejected");
    assert!(
        matches!(err, T0Error::Ir(IrError::OpAttributeInvalid { .. })),
        "wrong error: {err:?}"
    );
}

#[test]
fn logits_postprocess_failure_leaves_output_untouched() {
    let params = vec![base_params(), base_params()];
    // Row 0 is valid; row 1 carries a NaN. Staging must not leak row 0.
    let (s, q, v) = (2, 1, 3);
    let logits = vec![1.0f32, 2.0, 3.0, 0.0, f32::NAN, 0.0];
    let mut probs = vec![7.0f32; s * q * v];
    logits_postprocess(&logits, s, q, v, &params, None, None, &mut probs)
        .expect_err("NaN must fail");
    assert_eq!(probs, vec![7.0f32; s * q * v]);

    // Row 0 valid; row 1 fully masked out.
    let logits = vec![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
    let mask = vec![true, true, true, false, false, false];
    let mut probs = vec![7.0f32; s * q * v];
    let err = logits_postprocess(&logits, s, q, v, &params, None, Some(&mask), &mut probs)
        .expect_err("all-masked row must fail");
    assert!(
        matches!(err, T0Error::AllTokensMasked { seq: 1, .. }),
        "{err:?}"
    );
    assert_eq!(probs, vec![7.0f32; s * q * v]);
}

#[test]
fn logits_postprocess_rejects_bad_shapes_masks_and_bias_tokens() {
    let params = vec![base_params()];
    let mut probs = vec![0.0f32; 3];

    // Logits length mismatch.
    let err = logits_postprocess(&[1.0, 2.0], 1, 1, 3, &params, None, None, &mut probs)
        .expect_err("short logits must fail");
    assert!(
        matches!(err, T0Error::ShapeLengthMismatch { tensor, .. } if tensor == "logits"),
        "{err:?}"
    );

    // Grammar mask length mismatch.
    let err = logits_postprocess(
        &[1.0, 2.0, 3.0],
        1,
        1,
        3,
        &params,
        None,
        Some(&[true, true]),
        &mut probs,
    )
    .expect_err("short mask must fail");
    assert!(
        matches!(err, T0Error::ShapeLengthMismatch { tensor, .. } if tensor == "grammar_mask"),
        "{err:?}"
    );

    // Logit bias tokens outside the vocabulary aggregate into one typed
    // Multiple instead of failing on the first token.
    let mut p = base_params();
    p.logit_bias = vec![(9, 1.0), (10, 2.0)];
    let err = logits_postprocess(
        &[1.0, 2.0, 3.0],
        1,
        1,
        3,
        std::slice::from_ref(&p),
        None,
        None,
        &mut probs,
    )
    .expect_err("out-of-range bias tokens must fail");
    assert!(
        matches!(&err, T0Error::Multiple { problems } if problems.len() == 2),
        "expected 2-problem aggregation, got {err:?}"
    );
}

#[test]
fn rng_construction_rejects_bad_seq_id() {
    let probs = vec![0.25f32; 4];
    let err = RngState::from_u64(42, u64::MAX, 0)
        .expect_err("oversized seq_id must fail at construction");
    assert!(
        matches!(err, T0Error::SeqIdOutOfRange { seq_id, .. } if seq_id == u64::MAX),
        "{err:?}"
    );

    // Boundary: u32::MAX is representable and draws fine.
    let mut rng = vec![RngState::from_u64(42, u32::MAX as u64, 0).unwrap()];
    let tokens = sample(&probs, 1, 4, &mut rng).expect("u32::MAX seq_id must pass");
    assert_eq!(tokens.len(), 1);
    assert_eq!(rng[0].draw_index(), 1);
}

#[test]
fn sample_rejects_bad_distribution_without_rng_mutation() {
    let mut rng = vec![RngState::from_u64(7, 1, 0).unwrap()];
    // Negative entry.
    let err =
        sample(&[0.5, -0.25, 0.5, 0.25], 1, 4, &mut rng).expect_err("negative prob must fail");
    assert!(
        matches!(err, T0Error::InvalidProbability { token: 1, .. }),
        "{err:?}"
    );
    // Wrong sum.
    let err = sample(&[0.5, 0.5, 0.5, 0.5], 1, 4, &mut rng).expect_err("unnormalized must fail");
    assert!(
        matches!(err, T0Error::InvalidDistribution { .. }),
        "{err:?}"
    );
    assert_eq!(rng[0].draw_index(), 0);
}

#[test]
fn sample_preflights_draw_overflow_for_the_entire_batch() {
    let probs = vec![0.5f32, 0.5, 0.5, 0.5];
    let mut rng = vec![
        RngState::from_u64(1, 1, 0).unwrap(),
        RngState::with_draw(2, SeqId::new(2), StepId::new(0), u32::MAX).unwrap(),
    ];
    let err = sample(&probs, 2, 2, &mut rng).expect_err("batch draw overflow must fail");
    assert!(matches!(
        err,
        T0Error::DrawIndexOverflow {
            op: "sample",
            draw_index: u32::MAX,
            advance: 1
        }
    ));
    assert_eq!(rng[0].draw_index(), 0);
    assert_eq!(rng[1].draw_index(), u32::MAX);
}

#[test]
fn verify_validates_typical_params_before_mutation() {
    let draft = vec![1u32];
    let target = vec![0.5f32, 0.5, 0.5, 0.5];
    for (eps, delta) in [
        (f32::NAN, 0.5),
        (0.5, f32::INFINITY),
        (-0.1, 0.5),
        (0.5, -2.0),
        (0.0, 0.5),
        (0.5, 0.0),
    ] {
        let method = VerifyMethod::TypicalAcceptance { eps, delta };
        let mut rng = vec![RngState::from_u64(42, 1, 0).unwrap()];
        let err = verify(&draft, None, &target, 1, 1, 2, &method, &mut rng, None)
            .expect_err("bad typical params must fail");
        assert!(
            matches!(err, T0Error::InvalidAttribute { op, .. } if op == "verify"),
            "eps={eps} delta={delta}: {err:?}"
        );
        assert_eq!(rng[0].draw_index(), 0);
    }

    // Valid typical params run and advance k + 1 draws.
    let method = VerifyMethod::TypicalAcceptance {
        eps: 0.1,
        delta: 0.9,
    };
    let mut rng = vec![RngState::from_u64(42, 1, 0).unwrap()];
    let out = verify(&draft, None, &target, 1, 1, 2, &method, &mut rng, None)
        .expect("valid typical params must pass");
    assert_eq!(out.accepted.len(), 2);
    assert_eq!(rng[0].draw_index(), 2);
}

#[test]
fn verify_preflights_draw_overflow_for_the_entire_batch() {
    let draft = vec![0u32, 0];
    let target = vec![0.5f32; 8];
    let mut rng = vec![
        RngState::from_u64(1, 1, 0).unwrap(),
        RngState::with_draw(2, SeqId::new(2), StepId::new(0), u32::MAX - 1).unwrap(),
    ];
    let err = verify(
        &draft,
        None,
        &target,
        2,
        1,
        2,
        &VerifyMethod::Rejection,
        &mut rng,
        None,
    )
    .expect_err("batch draw overflow must fail");
    assert!(matches!(
        err,
        T0Error::DrawIndexOverflow {
            op: "verify",
            draw_index,
            advance: 2
        } if draw_index == u32::MAX - 1
    ));
    assert_eq!(rng[0].draw_index(), 0);
    assert_eq!(rng[1].draw_index(), u32::MAX - 1);
}

#[test]
fn verify_rejects_bad_shapes_without_mutation() {
    let draft = vec![1u32];
    let target = vec![0.5f32, 0.5, 0.5, 0.5];
    let method = VerifyMethod::Greedy;

    let err = RngState::from_u64(1, u64::from(u32::MAX) + 1, 0)
        .expect_err("oversized seq_id must fail at construction");
    assert!(matches!(err, T0Error::SeqIdOutOfRange { .. }), "{err:?}");

    // Draft token outside the vocabulary.
    let mut rng = vec![RngState::from_u64(1, 1, 0).unwrap()];
    let err = verify(&[7u32], None, &target, 1, 1, 2, &method, &mut rng, None)
        .expect_err("out-of-range draft token must fail");
    assert!(
        matches!(err, T0Error::TokenOutOfRange { token: 7, .. }),
        "{err:?}"
    );
    assert_eq!(rng[0].draw_index(), 0);

    // Target length mismatch.
    let mut rng = vec![RngState::from_u64(1, 1, 0).unwrap()];
    let err = verify(&draft, None, &target[..2], 1, 1, 2, &method, &mut rng, None)
        .expect_err("short target must fail");
    assert!(
        matches!(err, T0Error::ShapeLengthMismatch { tensor, .. } if tensor == "target_probs"),
        "{err:?}"
    );

    // Unrepresentable draw index: k = u32::MAX has no terminal draw k.
    let mut rng = vec![RngState::from_u64(1, 1, 0).unwrap()];
    let err = verify(
        &[],
        None,
        &[],
        1,
        u32::MAX as usize,
        2,
        &method,
        &mut rng,
        None,
    )
    .expect_err("u32::MAX k must fail");
    assert!(
        matches!(err, T0Error::ShapeLengthMismatch { tensor, .. } if tensor == "k"),
        "{err:?}"
    );
}

#[test]
fn verify_rejects_mismatched_tree_size() {
    // Tree with T = 2 presented for k = 1: structural refusal, no mutation.
    let tree = TreeMask::new(vec![-1, 0], 2, vec![false; 4]).expect("valid tree");
    let mut rng = vec![RngState::from_u64(1, 1, 0).unwrap()];
    let err = verify(
        &[0u32],
        None,
        &[0.5f32, 0.5, 0.5, 0.5],
        1,
        1,
        2,
        &VerifyMethod::Greedy,
        &mut rng,
        Some(&tree),
    )
    .expect_err("tree/k mismatch must fail");
    assert!(
        matches!(err, T0Error::ShapeLengthMismatch { tensor, .. } if tensor == "tree"),
        "{err:?}"
    );
    assert_eq!(rng[0].draw_index(), 0);
}

#[test]
fn verify_advances_draws_and_reproduces() {
    // k = 2 rejection verify consumes draws {0, 1} for candidates and draw 2
    // for the terminal token, advancing the cursor by k + 1 = 3.
    let draft = vec![0u32, 1];
    let target = vec![
        0.2f32, 0.8, // pos 0
        0.7, 0.3, // pos 1
        0.5, 0.5, // bonus
    ];
    let method = VerifyMethod::Rejection;
    let run = || {
        let mut rng = vec![RngState::from_u64(42, 1, 0).unwrap()];
        let out = verify(&draft, None, &target, 1, 2, 2, &method, &mut rng, None).unwrap();
        (out, rng[0].draw_index())
    };
    let (out_a, adv_a) = run();
    let (out_b, adv_b) = run();
    assert_eq!(out_a, out_b);
    assert_eq!((adv_a, adv_b), (3, 3));

    // Degenerate k = 0 advances exactly one draw.
    let mut rng = vec![RngState::from_u64(42, 1, 0).unwrap()];
    let out = verify(&[], None, &[0.3f32, 0.7], 1, 0, 2, &method, &mut rng, None).unwrap();
    assert_eq!(out.accept_len, vec![0]);
    assert_eq!(rng[0].draw_index(), 1);
}

#[test]
fn typed_seq_and_step_ids_reach_the_philox_counter() {
    // Same (seed, step, draw) with different SeqIds must diverge; the u32
    // word carries the id losslessly below the canonical ceiling.
    let a = RngState::new(11, SeqId::new(1), StepId::new(0)).unwrap();
    let b = RngState::new(11, SeqId::new(2), StepId::new(0)).unwrap();
    assert_ne!(a.draw_at(0), b.draw_at(0));
    // Step and draw words are likewise separated.
    let c = RngState::new(11, SeqId::new(1), StepId::new(1)).unwrap();
    assert_ne!(a.draw_at(0), c.draw_at(0));
    assert_ne!(a.draw_at(0), a.draw_at(1));
}
