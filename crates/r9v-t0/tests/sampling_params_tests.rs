// SPDX-License-Identifier: Apache-2.0
//! Exhaustive tests for all SamplingParams behaviors (Spec 1 §4.F, Spec 1 §6.5, Card A1.8).

use r9v_ir::SamplingParams;
use r9v_t0::logits_postprocess;

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
fn test_logit_bias_added_before_temperature() {
    let logits = vec![0.0, 0.0];
    let v = logits.len();
    let mut params = base_params();
    params.logit_bias = vec![(0, 2.0)];
    params.temperature = 0.5; // logit 2.0 / 0.5 = 4.0

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], None, None, &mut out).unwrap();

    // With temperature=0.5, effective logit for token 0 is 2.0 / 0.5 = 4.0, token 1 is 0.0
    // Expected prob: e^4 / (e^4 + 1)
    let expected_p0 = 4.0f32.exp() / (4.0f32.exp() + 1.0f32);
    assert!((out[0] - expected_p0).abs() < 1e-5);
}

#[test]
fn test_repetition_penalty_positive_and_negative_logits() {
    // Token 0: positive logit 2.0, count 1 -> 2.0 / 2.0 = 1.0
    // Token 1: negative logit -2.0, count 1 -> -2.0 * 2.0 = -4.0
    // Token 2: logit 1.0, count 0 -> unpenalized 1.0
    let logits = vec![2.0, -2.0, 1.0];
    let v = logits.len();
    let mut params = base_params();
    params.repetition_penalty = 2.0;
    let history = vec![1, 1, 0];

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], Some(&history), None, &mut out).unwrap();

    // After penalty, effective logits: token 0 is 1.0, token 1 is -4.0, token 2 is 1.0
    // Token 0 and Token 2 should have equal probability!
    assert!((out[0] - out[2]).abs() < 1e-6);
    assert!(out[0] > out[1]);
}

#[test]
fn test_presence_and_frequency_penalties() {
    let logits = vec![5.0, 5.0, 5.0];
    let v = logits.len();
    let mut params = base_params();
    params.presence_penalty = 1.0;
    params.frequency_penalty = 0.5;

    // Token 0: count 0 -> unpenalized (5.0)
    // Token 1: count 2 -> penalized: 5.0 - 1.0 - 2 * 0.5 = 3.0
    // Token 2: count 4 -> penalized: 5.0 - 1.0 - 4 * 0.5 = 2.0
    let history = vec![0, 2, 4];

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], Some(&history), None, &mut out).unwrap();

    assert!(out[0] > out[1]);
    assert!(out[1] > out[2]);

    // Check exact softmax ratio: e^5 : e^3 : e^2
    let sum_e = 5.0f32.exp() + 3.0f32.exp() + 2.0f32.exp();
    let expected_p0 = 5.0f32.exp() / sum_e;
    let expected_p1 = 3.0f32.exp() / sum_e;
    let expected_p2 = 2.0f32.exp() / sum_e;

    assert!((out[0] - expected_p0).abs() < 1e-5);
    assert!((out[1] - expected_p1).abs() < 1e-5);
    assert!((out[2] - expected_p2).abs() < 1e-5);
}

#[test]
fn test_grammar_mask_applied_before_softmax() {
    let logits = vec![100.0, 1.0, 2.0];
    let v = logits.len();
    let params = base_params();
    let mask = vec![false, true, true]; // Token 0 has dominant logit 100.0 but is masked

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], None, Some(&mask), &mut out).unwrap();

    assert_eq!(out[0], 0.0);
    assert!(out[2] > out[1]);
    assert!((out[1] + out[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_top_k_filtering() {
    let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
    let v = logits.len();
    let mut params = base_params();
    params.top_k = 2; // Keep top 2 tokens: token 1 (logit 5) and token 3 (logit 4)

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], None, None, &mut out).unwrap();

    assert!(out[1] > 0.0);
    assert!(out[3] > 0.0);
    assert_eq!(out[0], 0.0);
    assert_eq!(out[2], 0.0);
    assert_eq!(out[4], 0.0);
    assert!((out[1] + out[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_top_p_nucleus_filtering() {
    let logits = vec![10.0, 5.0, 1.0, 0.0];
    let v = logits.len();
    let mut params = base_params();
    // With logit 10.0 vs 5.0, token 0 has > 99% of total softmax probability
    params.top_p = 0.95;

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], None, None, &mut out).unwrap();

    // Only token 0 should be retained
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 0.0);
    assert_eq!(out[3], 0.0);
}

#[test]
fn test_min_p_filtering() {
    let logits = vec![4.0, 3.0, 1.0, 0.0];
    let v = logits.len();
    let mut params = base_params();
    params.min_p = 0.25; // Discard tokens with p < 0.25 * p_max

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], None, None, &mut out).unwrap();

    // Token 0 is p_max. Check that surviving tokens have p >= 0.25 * p_max
    assert!(out[0] > 0.0);
    assert!(out[1] > 0.0);
    // Tokens with low logits are filtered out
    assert_eq!(out[2], 0.0);
    assert_eq!(out[3], 0.0);
    assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_stable_sort_tie_breaking_by_lowest_index() {
    // Tokens 0, 1, 2, 3 have identical logits.
    // top_k = 2 must stably keep tokens 0 and 1, breaking ties by lowest index.
    let logits = vec![2.0, 2.0, 2.0, 2.0];
    let v = logits.len();
    let mut params = base_params();
    params.top_k = 2;

    let mut out = vec![0.0f32; v];
    logits_postprocess(&logits, 1, 1, v, &[params], None, None, &mut out).unwrap();

    assert_eq!(out[0], 0.5);
    assert_eq!(out[1], 0.5);
    assert_eq!(out[2], 0.0);
    assert_eq!(out[3], 0.0);
}
