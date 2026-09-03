// SPDX-License-Identifier: Apache-2.0
//! Determinism and batch-invariance tests for sampling operations (Spec 1 §6.1, §6.5, CONVENTIONS.md §4.3, Card A1.8).

use r9v_ir::{SamplingParams, VerifyMethod};
use r9v_t0::{logits_postprocess, sample, verify, RngState};

fn sample_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.8,
        top_k: 4,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.2,
        presence_penalty: 0.1,
        frequency_penalty: 0.1,
        logit_bias: vec![(1, 0.5), (3, -0.2)],
    }
}

#[test]
fn determinism_logits_postprocess_run_to_run_bit_identical() {
    let s = 2;
    let q = 2;
    let v = 8;
    let logits = vec![
        0.1, 2.5, -1.0, 3.2, 0.0, 4.1, -0.5, 1.2, 1.0, -0.2, 0.5, 2.1, 3.3, 0.8, 1.5, -2.0, -1.5,
        0.0, 3.0, 1.2, 2.2, -0.1, 0.9, 1.7, 2.0, 1.1, -0.8, 0.4, 3.5, -1.2, 0.6, 2.8,
    ];
    let params = vec![sample_params(), sample_params()];
    let history = vec![0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 3, 0, 0, 1];
    let mask = vec![
        true, true, false, true, true, true, false, true, true, false, true, true, true, true,
        true, false, false, true, true, true, true, false, true, true, true, true, true, false,
        true, true, false, true,
    ];

    let mut out1 = vec![0.0f32; s * q * v];
    let mut out2 = vec![0.0f32; s * q * v];

    logits_postprocess(
        &logits,
        s,
        q,
        v,
        &params,
        Some(&history),
        Some(&mask),
        &mut out1,
    )
    .unwrap();
    logits_postprocess(
        &logits,
        s,
        q,
        v,
        &params,
        Some(&history),
        Some(&mask),
        &mut out2,
    )
    .unwrap();

    // Check bit-identical float outputs
    for i in 0..out1.len() {
        assert_eq!(
            out1[i].to_bits(),
            out2[i].to_bits(),
            "Mismatch at index {i}"
        );
    }
}

#[test]
fn determinism_sample_run_to_run_bit_identical() {
    let s = 3;
    let v = 5;
    let probs = vec![
        0.1, 0.2, 0.4, 0.2, 0.1, 0.3, 0.3, 0.1, 0.1, 0.2, 0.05, 0.05, 0.1, 0.7, 0.1,
    ];

    let mut rng1 = vec![
        RngState::from_u64(100, 1, 5).unwrap(),
        RngState::from_u64(200, 2, 5).unwrap(),
        RngState::from_u64(300, 3, 5).unwrap(),
    ];
    let mut rng2 = rng1.clone();

    let tokens1 = sample(&probs, s, v, &mut rng1).unwrap();
    let tokens2 = sample(&probs, s, v, &mut rng2).unwrap();

    assert_eq!(tokens1, tokens2);
    assert_eq!(rng1, rng2);
}

#[test]
fn determinism_verify_run_to_run_bit_identical() {
    let s = 2;
    let k = 3;
    let v = 4;
    let draft_tokens = vec![1, 2, 0, 3, 1, 2];
    let draft_probs = vec![
        0.1, 0.6, 0.2, 0.1, 0.0, 0.1, 0.7, 0.2, 0.5, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.7, 0.2, 0.5,
        0.2, 0.1, 0.1, 0.2, 0.6, 0.1,
    ];
    let target_probs = vec![
        // seq 0: pos 0..3
        0.2, 0.5, 0.2, 0.1, 0.1, 0.1, 0.6, 0.2, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4,
        // seq 1: pos 0..3
        0.0, 0.1, 0.2, 0.7, 0.1, 0.6, 0.2, 0.1, 0.2, 0.2, 0.5, 0.1, 0.3, 0.3, 0.2, 0.2,
    ];

    for method in [
        VerifyMethod::Rejection,
        VerifyMethod::Greedy,
        VerifyMethod::TypicalAcceptance {
            eps: 0.1,
            delta: 0.8,
        },
    ] {
        let mut rng1 = vec![
            RngState::from_u64(42, 1, 10).unwrap(),
            RngState::from_u64(43, 2, 10).unwrap(),
        ];
        let mut rng2 = rng1.clone();

        let out1 = verify(
            &draft_tokens,
            Some(&draft_probs),
            &target_probs,
            s,
            k,
            v,
            &method,
            &mut rng1,
            None,
        )
        .unwrap();
        let out2 = verify(
            &draft_tokens,
            Some(&draft_probs),
            &target_probs,
            s,
            k,
            v,
            &method,
            &mut rng2,
            None,
        )
        .unwrap();

        assert_eq!(out1.accepted, out2.accepted);
        assert_eq!(out1.accept_len, out2.accept_len);
        assert_eq!(rng1, rng2);
    }
}

#[test]
fn determinism_batch_invariance_sequence_alone_vs_batched() {
    // Spec 1 §6.1 Batch Invariance: output for a sequence must be bit-identical
    // whether executed alone (S=1) or embedded in a batch (S=3).
    let v = 6;
    let seq0_logits = vec![0.5, 1.2, -0.4, 2.1, 0.8, -1.0];
    let seq1_logits = vec![1.5, 0.2, 0.4, -0.1, 3.0, 0.5];
    let seq2_logits = vec![-0.5, 2.0, 1.1, 0.0, 0.7, 1.8];

    let p0 = sample_params();
    let p1 = sample_params();
    let p2 = sample_params();

    // 1. Run sequence 0 alone
    let mut out_alone = vec![0.0f32; v];
    logits_postprocess(
        &seq0_logits,
        1,
        1,
        v,
        std::slice::from_ref(&p0),
        None,
        None,
        &mut out_alone,
    )
    .unwrap();

    let mut rng_alone = vec![RngState::from_u64(999, 101, 1).unwrap()];
    let token_alone = sample(&out_alone, 1, v, &mut rng_alone).unwrap()[0];

    // 2. Run batched with sequence 1 and sequence 2
    let mut batched_logits = Vec::new();
    batched_logits.extend_from_slice(&seq0_logits);
    batched_logits.extend_from_slice(&seq1_logits);
    batched_logits.extend_from_slice(&seq2_logits);

    let mut out_batched = vec![0.0f32; 3 * v];
    logits_postprocess(
        &batched_logits,
        3,
        1,
        v,
        &[p0, p1, p2],
        None,
        None,
        &mut out_batched,
    )
    .unwrap();

    let mut rng_batched = vec![
        RngState::from_u64(999, 101, 1).unwrap(),
        RngState::from_u64(888, 102, 1).unwrap(),
        RngState::from_u64(777, 103, 1).unwrap(),
    ];
    let tokens_batched = sample(&out_batched, 3, v, &mut rng_batched).unwrap();

    // Verify bit-identical probabilities for sequence 0
    for i in 0..v {
        assert_eq!(
            out_alone[i].to_bits(),
            out_batched[i].to_bits(),
            "Probability mismatch for token {i} between alone and batched"
        );
    }

    // Verify identical sampled token and RNG state advancement
    assert_eq!(token_alone, tokens_batched[0]);
    assert_eq!(rng_alone[0], rng_batched[0]);
}
