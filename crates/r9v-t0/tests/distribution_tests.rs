// SPDX-License-Identifier: Apache-2.0
//! 100,000-draw distribution tests for speculative rejection sampling (Spec 1 §4.F, Spec 7 §4, Card A1.8).

use r9v_ir::VerifyMethod;
use r9v_t0::{sample, verify, RngState};

/// Helper to compute chi-squared test statistic against expected theoretical probabilities.
fn chi_squared_stat(observed_counts: &[usize], total: usize, expected_probs: &[f32]) -> f64 {
    let mut chi2 = 0.0f64;
    let n = total as f64;
    for (&obs, &exp_p) in observed_counts.iter().zip(expected_probs.iter()) {
        let expected = n * (exp_p as f64);
        let diff = (obs as f64) - expected;
        chi2 += (diff * diff) / expected;
    }
    chi2
}

#[test]
fn rejection_sampling_output_frequency_matches_target_sampling_1e5_draws() {
    // Synthetic target distribution P and draft distribution Q over V = 4
    let target_p = vec![0.10f32, 0.40f32, 0.30f32, 0.20f32];
    let draft_q = vec![0.25f32, 0.25f32, 0.25f32, 0.25f32];
    let vocab_size = 4;
    let draws = 100_000;

    let mut rejection_counts = vec![0usize; vocab_size];
    let mut direct_counts = vec![0usize; vocab_size];

    let mut rng_rej = RngState::new(123456789, 1, 0);
    let mut rng_direct = RngState::new(987654321, 2, 0);
    let mut rng_draft = RngState::new(555555555, 3, 0);

    // Target probs array for verify [S=1, k+1=2, V=4]
    let mut target_probs = Vec::with_capacity(vocab_size * 2);
    target_probs.extend_from_slice(&target_p);
    target_probs.extend_from_slice(&target_p); // bonus distribution

    let mut draft_probs = Vec::with_capacity(vocab_size);
    draft_probs.extend_from_slice(&draft_q);

    let method = VerifyMethod::Rejection;

    for step in 0..draws {
        // 1. Draw a draft token from draft distribution Q
        let draft_tok = sample(&draft_q, 1, vocab_size, &mut [rng_draft.clone()]).unwrap()[0];
        rng_draft.advance(1);

        // 2. Perform speculative rejection sampling verification
        rng_rej.set_step(step as u64);
        let mut rng_slice = [rng_rej.clone()];
        let out = verify(
            &[draft_tok],
            Some(&draft_probs),
            &target_probs,
            1,
            1,
            vocab_size,
            &method,
            &mut rng_slice,
            None,
        )
        .unwrap();
        rng_rej = rng_slice[0].clone();

        // The emitted token at position 0 is either the accepted draft token or the replacement token
        let emitted_token = out.accepted[0] as usize;
        rejection_counts[emitted_token] += 1;

        // 3. Perform direct sampling from target distribution P
        rng_direct.set_step(step as u64);
        let mut rng_dir_slice = [rng_direct.clone()];
        let dir_token = sample(&target_p, 1, vocab_size, &mut rng_dir_slice).unwrap()[0] as usize;
        rng_direct = rng_dir_slice[0].clone();
        direct_counts[dir_token] += 1;
    }

    // Statistical validation: Chi-squared goodness of fit
    // For V=4 (df=3), critical value at alpha=0.001 is 16.27.
    let chi2_rej = chi_squared_stat(&rejection_counts, draws, &target_p);
    let chi2_dir = chi_squared_stat(&direct_counts, draws, &target_p);

    // Assert that rejection sampling matches the target distribution within statistical tolerance
    assert!(
        chi2_rej < 16.27,
        "Rejection sampling chi-squared statistic {chi2_rej} exceeds critical threshold 16.27 (p < 0.001)"
    );
    assert!(
        chi2_dir < 16.27,
        "Direct sampling chi-squared statistic {chi2_dir} exceeds critical threshold 16.27 (p < 0.001)"
    );

    // Total variation distance between rejection sampling empirical frequency and target P
    let mut tvd = 0.0f64;
    for (v, &p) in target_p.iter().enumerate() {
        let freq = (rejection_counts[v] as f64) / (draws as f64);
        tvd += (freq - (p as f64)).abs();
    }
    tvd *= 0.5;
    assert!(
        tvd < 0.005,
        "Total variation distance {tvd} between rejection sampling and target P exceeds 0.005"
    );
}

#[test]
fn rejection_sampling_matches_target_sampling_within_statistical_tolerance_on_synthetic_distributions(
) {
    // Test with one-hot deterministic draft proposer (q_i = None / one-hot)
    let target_p = vec![0.05f32, 0.15f32, 0.50f32, 0.20f32, 0.10f32];
    let vocab_size = 5;
    let draws = 100_000;

    let mut rejection_counts = vec![0usize; vocab_size];
    let mut rng_rej = RngState::new(777, 10, 0);

    let mut target_probs = Vec::with_capacity(vocab_size * 2);
    target_probs.extend_from_slice(&target_p);
    target_probs.extend_from_slice(&target_p);

    let method = VerifyMethod::Rejection;

    // Proposer always drafts token 2 (the mode of target)
    let draft_tok = 2u32;

    for step in 0..draws {
        rng_rej.set_step(step as u64);
        let mut rng_slice = [rng_rej.clone()];
        let out = verify(
            &[draft_tok],
            None, // one-hot
            &target_probs,
            1,
            1,
            vocab_size,
            &method,
            &mut rng_slice,
            None,
        )
        .unwrap();
        rng_rej = rng_slice[0].clone();

        let emitted = out.accepted[0] as usize;
        rejection_counts[emitted] += 1;
    }

    // For V=5 (df=4), critical value at alpha=0.001 is 18.47.
    let chi2 = chi_squared_stat(&rejection_counts, draws, &target_p);
    assert!(
        chi2 < 18.47,
        "Draft-free rejection sampling chi2 {chi2} exceeds critical threshold 18.47"
    );
}
