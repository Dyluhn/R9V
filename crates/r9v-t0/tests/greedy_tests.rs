// SPDX-License-Identifier: Apache-2.0
//! Temperature-zero greedy equivalence tests (Spec 1 §4.F, Spec 7 §4, Card A1.8).

use r9v_ir::{SamplingParams, VerifyMethod};
use r9v_t0::{logits_postprocess, sample, verify, RngState};

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
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
fn greedy_equivalence_at_temperature_zero() {
    let logits = vec![1.2, 5.8, -0.4, 3.1, 5.8, 2.0];
    let v = logits.len();
    let params = vec![greedy_params()];

    let mut probs = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &params, None, None, &mut probs).unwrap();

    // 1. Logits postprocess at temperature 0 must output a one-hot distribution
    // Notice tokens 1 and 4 have identical max logit 5.8. Stable sort by (-logit, index)
    // selects token 1 (lower index) deterministically.
    assert_eq!(probs[1], 1.0f32);
    assert_eq!(probs[4], 0.0f32);
    for (i, &p) in probs.iter().enumerate() {
        if i == 1 {
            assert_eq!(p, 1.0);
        } else {
            assert_eq!(p, 0.0);
        }
    }

    // 2. Sample on temperature 0 distribution must deterministically return argmax
    for seed in [1, 42, 999, 123456] {
        let mut rng = vec![RngState::new(seed, 1, 0)];
        let token = sample(&probs, 1, v, &mut rng).unwrap()[0];
        assert_eq!(
            token, 1,
            "Sample at temp 0 must yield deterministic argmax token 1"
        );
    }
}

#[test]
fn greedy_equivalence_temperature_zero_rejection_matches_greedy() {
    let v = 6;
    let k = 3;

    // Target logits for 4 positions: pos 0..3 (k=3 draft positions + 1 bonus position)
    let target_logits = vec![
        // pos 0: max at 2
        0.0, 1.0, 5.0, 2.0, 0.5, 0.0, // pos 1: max at 4
        1.0, 0.0, 2.0, 3.0, 6.0, 1.0, // pos 2: max at 0
        4.0, 1.0, 2.0, 0.0, 0.0, 3.0, // pos 3 (bonus): max at 5
        1.0, 2.0, 0.0, 1.0, 3.0, 7.0,
    ];

    let params = vec![greedy_params()];
    let mut target_probs = vec![0.0f32; (k + 1) * v];
    logits_postprocess(
        &target_logits,
        1,
        k + 1,
        v,
        &params,
        None,
        None,
        &mut target_probs,
    )
    .unwrap();

    // Scenario A: all draft tokens match greedy argmax [2, 4, 0]
    let matching_draft = vec![2u32, 4, 0];
    let mut rng_greedy = vec![RngState::new(42, 1, 0)];
    let mut rng_rejection = vec![RngState::new(42, 1, 0)];

    let out_greedy = verify(
        &matching_draft,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Greedy,
        &mut rng_greedy,
        None,
    )
    .unwrap();

    let out_rejection = verify(
        &matching_draft,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Rejection,
        &mut rng_rejection,
        None,
    )
    .unwrap();

    assert_eq!(out_greedy.accept_len, vec![3]);
    assert_eq!(out_greedy.accepted, vec![2, 4, 0, 5]);
    assert_eq!(out_greedy.accept_len, out_rejection.accept_len);
    assert_eq!(out_greedy.accepted, out_rejection.accepted);

    // Scenario B: partial match - second draft token mismatches: [2, 1 (instead of 4), 0]
    let partial_draft = vec![2u32, 1, 0];
    let mut rng_g2 = vec![RngState::new(100, 2, 0)];
    let mut rng_r2 = vec![RngState::new(100, 2, 0)];

    let out_g2 = verify(
        &partial_draft,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Greedy,
        &mut rng_g2,
        None,
    )
    .unwrap();

    let out_r2 = verify(
        &partial_draft,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Rejection,
        &mut rng_r2,
        None,
    )
    .unwrap();

    // In both cases, exactly 1 draft token is accepted (token 2).
    // Replacement token at index 1 is target argmax 4!
    assert_eq!(out_g2.accept_len, vec![1]);
    assert_eq!(out_g2.accepted[0], 2);
    assert_eq!(out_g2.accepted[1], 4);
    assert_eq!(out_g2.accept_len, out_r2.accept_len);
    assert_eq!(out_g2.accepted, out_r2.accepted);
}

#[test]
fn greedy_stable_tie_break_by_lowest_index() {
    let logits = vec![3.0, 3.0, 3.0, 3.0];
    let v = logits.len();
    let mut probs = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[greedy_params()], None, None, &mut probs).unwrap();

    // All logits identical: index 0 must be selected
    assert_eq!(probs, vec![1.0, 0.0, 0.0, 0.0]);
}
