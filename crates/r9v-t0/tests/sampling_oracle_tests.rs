// SPDX-License-Identifier: Apache-2.0
//! Cross-tests of f32 `logits_postprocess` against the independent f64 oracle
//! (Spec 1 §4.F, §6.5, Card A1.8).

use r9v_common::rng::SeededRng;
use r9v_ir::SamplingParams;
use r9v_t0::{logits_postprocess, logits_postprocess_f64_reference, Tolerance};

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

fn random_logits(rng: &mut SeededRng, len: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let raw = (rng.next_u64() & 0xFFFF_FFFF) as u32;
        out.push(((raw as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn check_oracle_agreement(
    logits: &[f32],
    s: usize,
    q: usize,
    v: usize,
    params: &[SamplingParams],
    history_counts: Option<&[u32]>,
    grammar_mask: Option<&[bool]>,
    context: &str,
) {
    let mut probs = vec![0.0f32; s * q * v];
    logits_postprocess(
        logits,
        s,
        q,
        v,
        params,
        history_counts,
        grammar_mask,
        &mut probs,
    )
    .unwrap_or_else(|e| panic!("{context}: f32 implementation failed: {e:?}"));

    let logits_f64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let reference = logits_postprocess_f64_reference(
        &logits_f64,
        s,
        q,
        v,
        params,
        history_counts,
        grammar_mask,
    );

    let tol = Tolerance::f32();
    for (i, (&actual, &expected)) in probs.iter().zip(reference.iter()).enumerate() {
        tol.assert_within(
            actual as f64,
            expected,
            &format!("{context} at flat index {i}"),
        );
    }

    // Both pipelines must emit normalized rows.
    for row in 0..s * q {
        let sum: f32 = probs[row * v..(row + 1) * v].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "{context}: f32 row {row} sums to {sum}"
        );
        let ref_sum: f64 = reference[row * v..(row + 1) * v].iter().sum();
        assert!(
            (ref_sum - 1.0).abs() < 1e-12,
            "{context}: oracle row {row} sums to {ref_sum}"
        );
    }
}

#[test]
fn logits_postprocess_matches_f64_oracle_plain_softmax() {
    let mut rng = SeededRng::new(0xA1_8001);
    let (s, q, v) = (2, 3, 32);
    let logits = random_logits(&mut rng, s * q * v, 3.0);
    let params = vec![base_params(), base_params()];
    check_oracle_agreement(&logits, s, q, v, &params, None, None, "plain softmax");
}

#[test]
fn logits_postprocess_matches_f64_oracle_temperature_sweep() {
    let mut rng = SeededRng::new(0xA1_8002);
    let (s, q, v) = (1, 2, 24);
    let logits = random_logits(&mut rng, s * q * v, 2.5);
    for &temp in &[0.2f32, 0.7, 1.0, 2.0] {
        let mut p = base_params();
        p.temperature = temp;
        let context = format!("temperature {temp}");
        check_oracle_agreement(
            &logits,
            s,
            q,
            v,
            std::slice::from_ref(&p),
            None,
            None,
            &context,
        );
    }
}

#[test]
fn logits_postprocess_temperature_zero_is_exact_argmax() {
    let mut rng = SeededRng::new(0xA1_8003);
    let (s, q, v) = (2, 2, 16);
    let logits = random_logits(&mut rng, s * q * v, 4.0);
    let mut p = base_params();
    p.temperature = 0.0;
    let params = vec![p.clone(), p];
    let mut probs = vec![0.0f32; s * q * v];
    logits_postprocess(&logits, s, q, v, &params, None, None, &mut probs).unwrap();

    let logits_f64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let reference = logits_postprocess_f64_reference(&logits_f64, s, q, v, &params, None, None);
    for (i, (&actual, &expected)) in probs.iter().zip(reference.iter()).enumerate() {
        assert_eq!(actual, expected as f32, "temperature-0 mismatch at {i}");
    }
}

#[test]
fn logits_postprocess_matches_f64_oracle_penalties_bias_history() {
    let mut rng = SeededRng::new(0xA1_8004);
    let (s, q, v) = (2, 2, 20);
    let logits = random_logits(&mut rng, s * q * v, 2.0);
    let history: Vec<u32> = (0..s * v).map(|_| (rng.next_u64() % 4) as u32).collect();

    let mut p0 = base_params();
    p0.repetition_penalty = 1.3;
    p0.presence_penalty = 0.2;
    p0.frequency_penalty = 0.1;
    p0.logit_bias = vec![(3, 1.5), (17, -2.0)];
    let mut p1 = base_params();
    p1.temperature = 0.7;
    p1.repetition_penalty = 0.8;
    p1.logit_bias = vec![(0, 0.5)];
    let params = vec![p0, p1];
    check_oracle_agreement(
        &logits,
        s,
        q,
        v,
        &params,
        Some(&history),
        None,
        "penalties+bias+history",
    );
}

#[test]
fn logits_postprocess_matches_f64_oracle_filters_and_mask() {
    let mut rng = SeededRng::new(0xA1_8005);
    let (s, q, v) = (1, 3, 28);
    let logits = random_logits(&mut rng, s * q * v, 2.0);
    // Random mask with token 0 forced allowed so no row is fully masked.
    let mut mask = vec![false; s * q * v];
    for (i, m) in mask.iter_mut().enumerate() {
        *m = i % v == 0 || (rng.next_u64() & 1) == 1;
    }

    for &(top_k, top_p, min_p) in &[
        (3u32, 1.0f32, 0.0f32),
        (0, 0.9, 0.0),
        (0, 1.0, 0.15),
        (5, 0.95, 0.05),
    ] {
        let mut p = base_params();
        p.top_k = top_k;
        p.top_p = top_p;
        p.min_p = min_p;
        let context = format!("top_k={top_k} top_p={top_p} min_p={min_p}");
        check_oracle_agreement(
            &logits,
            s,
            q,
            v,
            std::slice::from_ref(&p),
            None,
            Some(&mask),
            &context,
        );
    }
}

#[test]
fn logits_postprocess_matches_f64_oracle_intentional_neg_inf() {
    let mut rng = SeededRng::new(0xA1_8006);
    let (s, q, v) = (1, 2, 12);
    let mut logits = random_logits(&mut rng, s * q * v, 2.0);
    logits[2] = f32::NEG_INFINITY;
    logits[v + 7] = f32::NEG_INFINITY;
    let params = vec![base_params()];
    check_oracle_agreement(
        &logits,
        s,
        q,
        v,
        &params,
        None,
        None,
        "intentional -Inf logits",
    );
}
