// SPDX-License-Identifier: Apache-2.0
//! Speculative verification tests: Rejection, Greedy, Typical, and Tree walk (Spec 1 §4.F, Spec 7 §4, §5, Card A1.8).

use r9v_ir::{TreeMask, VerifyMethod};
use r9v_t0::{verify, RngState};

#[test]
fn test_verify_rejection_all_accepted_samples_bonus() {
    let k = 2;
    let v = 4;
    let draft_tokens = vec![1, 3];
    // Target distributions strongly favor draft tokens
    let target_probs = vec![
        0.0, 1.0, 0.0, 0.0, // pos 0 -> token 1 prob 1.0
        0.0, 0.0, 0.0, 1.0, // pos 1 -> token 3 prob 1.0
        0.5, 0.5, 0.0, 0.0, // pos 2 (bonus) -> uniform over 0 and 1
    ];
    let mut rng = vec![RngState::new(42, 1, 0)];

    let out = verify(
        &draft_tokens,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Rejection,
        &mut rng,
        None,
    )
    .unwrap();

    assert_eq!(out.accept_len, vec![2]);
    assert_eq!(out.accepted[0], 1);
    assert_eq!(out.accepted[1], 3);
    // Bonus token must be 0 or 1
    assert!(out.accepted[2] == 0 || out.accepted[2] == 1);
}

#[test]
fn test_verify_rejection_first_rejected_samples_replacement() {
    let k = 2;
    let v = 4;
    let draft_tokens = vec![1, 2];
    // Target pos 0 gives 0.0 to token 1 and 1.0 to token 3
    let target_probs = vec![
        0.0, 0.0, 0.0, 1.0, // pos 0
        0.0, 0.0, 1.0, 0.0, // pos 1
        1.0, 0.0, 0.0, 0.0, // pos 2
    ];
    let mut rng = vec![RngState::new(123, 1, 0)];

    let out = verify(
        &draft_tokens,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Rejection,
        &mut rng,
        None,
    )
    .unwrap();

    assert_eq!(out.accept_len, vec![0]);
    // Replacement token at index 0 must be 3
    assert_eq!(out.accepted[0], 3);
}

#[test]
fn test_verify_typical_acceptance_threshold() {
    let v = 4;
    let draft_tokens = vec![1];
    // Distribution where token 1 has probability 0.8
    let target_probs = vec![
        0.05, 0.80, 0.10, 0.05, 1.0, 0.0, 0.0, 0.0, // bonus
    ];
    let mut rng = vec![RngState::new(42, 1, 0)];

    // 1. With eps = 0.5, p[1] = 0.80 > min(0.5, ...) -> should accept!
    let method_accept = VerifyMethod::TypicalAcceptance {
        eps: 0.5,
        delta: 1.0,
    };
    let out1 = verify(
        &draft_tokens,
        None,
        &target_probs,
        1,
        1,
        v,
        &method_accept,
        &mut rng,
        None,
    )
    .unwrap();
    assert_eq!(out1.accept_len, vec![1]);
    assert_eq!(out1.accepted[0], 1);

    // 2. With eps = 0.9, delta = 2.0 -> min(0.9, 2.0 * 0.492) = 0.9 > 0.80 -> should reject!
    let method_reject = VerifyMethod::TypicalAcceptance {
        eps: 0.9,
        delta: 2.0,
    };
    let out2 = verify(
        &draft_tokens,
        None,
        &target_probs,
        1,
        1,
        v,
        &method_reject,
        &mut rng,
        None,
    )
    .unwrap();
    assert_eq!(out2.accept_len, vec![0]);
}

#[test]
fn test_verify_tree_walk_longest_path_and_tie_breaking() {
    // Construct a tree of k=4 tokens:
    // Roots: node 0 and node 1 (both children of prompt context: parents = -1)
    // Node 2 is child of node 0: parents[2] = 0
    // Node 3 is child of node 1: parents[3] = 1
    // Paths from root to leaves:
    // Path A: 0 -> 2
    // Path B: 1 -> 3
    let parents = vec![-1, -1, 0, 1];
    let k = 4;
    let t_max = 4;
    let ancestors = vec![
        true, false, false, false, false, true, false, false, true, false, true, false, false,
        true, false, true,
    ];
    let tree_mask = TreeMask::new(parents, t_max, ancestors).unwrap();

    let v = 5;
    let draft_tokens = vec![1, 2, 3, 4];

    // Case 1: Path A accepts 2 tokens (0 and 2), Path B accepts 1 token (1)
    // Target distribution:
    // Index 0 (root predicting 0 and 1): predicts token 1 (matching node 0) and token 2 (matching node 1)
    // Index 1 (node 0 output predicting node 2): predicts token 3 (matching node 2)
    // Index 2 (node 1 output predicting node 3): predicts token 0 (mismatching node 3 which is 4)
    // Index 3 (node 2 output - bonus token for Path A): predicts token 0
    // Index 4 (node 3 output - bonus token for Path B): predicts token 0
    let mut target_probs = vec![0.0f32; (k + 1) * v];
    // Root: token 1 has 0.9 (matches node 0), token 2 has 0.9 (matches node 1)
    target_probs[1] = 0.5;
    target_probs[2] = 0.5;
    // Node 0 output: matches node 2 (token 3)
    target_probs[v + 3] = 1.0;
    // Node 1 output: mismatches node 3 (token 4), emits token 0 instead
    target_probs[2 * v] = 1.0;
    // Node 2 output: bonus token 2
    target_probs[3 * v + 2] = 1.0;

    let mut rng = vec![RngState::new(42, 1, 0)];
    let out = verify(
        &draft_tokens,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Greedy,
        &mut rng,
        Some(&tree_mask),
    )
    .unwrap();

    // Path A has 2 accepted tokens (nodes 0 and 2), while Path B has only 1 (node 1).
    // Longest path (Path A) must win!
    assert_eq!(out.accept_len, vec![2]);
    assert_eq!(out.accepted[0], 1); // token of node 0
    assert_eq!(out.accepted[1], 3); // token of node 2
    assert_eq!(out.accepted[2], 2); // bonus token from node 2

    // Case 2: Tie breaking: both Path A and Path B accept 2 tokens!
    // Path A starts with node 0; Path B starts with node 1.
    // Spec 7 §5: "ties go to the path with the lowest first-token index"
    // Node 0 < Node 1, so Path A must win on tie!
    target_probs[2 * v + 4] = 1.0; // Now node 1 output also matches node 3!

    let mut rng_tie = vec![RngState::new(42, 1, 0)];
    let out_tie = verify(
        &draft_tokens,
        None,
        &target_probs,
        1,
        k,
        v,
        &VerifyMethod::Greedy,
        &mut rng_tie,
        Some(&tree_mask),
    )
    .unwrap();

    assert_eq!(out_tie.accept_len, vec![2]);
    // Path A (starting with first-token index 0) wins tie over Path B (first-token index 1)
    assert_eq!(out_tie.accepted[0], 1);
    assert_eq!(out_tie.accepted[1], 3);
}

#[test]
fn test_verify_degenerate_k_zero() {
    let target_probs = vec![0.0, 1.0, 0.0];
    let mut rng = vec![RngState::new(1, 1, 0)];
    let out = verify(
        &[],
        None,
        &target_probs,
        1,
        0,
        3,
        &VerifyMethod::Greedy,
        &mut rng,
        None,
    )
    .unwrap();

    assert_eq!(out.accept_len, vec![0]);
    assert_eq!(out.accepted, vec![1]);
}
