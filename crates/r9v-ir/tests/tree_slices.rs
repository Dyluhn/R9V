// SPDX-License-Identifier: Apache-2.0
//! Borrowed tree-slice validation: undersized scratch and intrinsic parity
//! (Spec 1 §4.D.1).
//!
//! `validate_tree_slices` runs on caller-owned cycle scratch. Short buffers
//! are reachable from caller sizing, so they are typed errors carrying the
//! requirement and the supplied length/capacity — never a panic. With
//! valid-size scratch every intrinsic verdict matches the owned
//! [`TreeMask::new`] builder exactly, which keeps sizing its own scratch and
//! stays compatible.

use std::panic::{catch_unwind, AssertUnwindSafe};

use r9v_ir::{validate_tree_slices, IrError, TreeMask};

/// Short cycle state is a typed error, not a panic.
#[test]
fn short_cycle_state_is_typed_not_a_panic() {
    let parents = [-1, 0, 1];
    let ancestors = [true; 9];
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut state = vec![0u8; 2];
        let mut path = Vec::with_capacity(3);
        validate_tree_slices(&parents, 3, &ancestors, &mut state, &mut path)
    }));
    let outcome = result.expect("short cycle state must not panic");
    assert_eq!(
        outcome.unwrap_err(),
        IrError::TreeCycleStateTooSmall {
            required: 3,
            actual: 2,
        }
    );
}

/// Short cycle path capacity is a typed error, not a panic.
#[test]
fn short_cycle_path_is_typed_not_a_panic() {
    let parents = [-1, 0, 1];
    let ancestors = [true; 9];
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut state = vec![0u8; 3];
        let mut path = Vec::with_capacity(2);
        validate_tree_slices(&parents, 3, &ancestors, &mut state, &mut path)
    }));
    let outcome = result.expect("short cycle path must not panic");
    assert_eq!(
        outcome.unwrap_err(),
        IrError::TreeCyclePathTooSmall {
            required: 3,
            actual: 2,
        }
    );
}

/// Both buffers short reports the state shortage first, deterministically.
#[test]
fn both_buffers_short_reports_state_first() {
    let parents = [-1, 0, 1];
    let ancestors = [true; 9];
    let mut state = vec![0u8; 1];
    let mut path = Vec::with_capacity(1);
    let err = validate_tree_slices(&parents, 3, &ancestors, &mut state, &mut path)
        .expect_err("short scratch must be refused");
    assert_eq!(
        err,
        IrError::TreeCycleStateTooSmall {
            required: 3,
            actual: 1,
        }
    );
}

/// With valid-size scratch every intrinsic verdict matches the owned builder
/// exactly — including success — so [`TreeMask::new`] stays compatible.
#[test]
fn valid_size_scratch_matches_owned_builder_error_for_error() {
    // (parents, t_max, ancestors): one case per intrinsic rule plus a
    // multi-fault collect and a valid chain.
    let cases: [(&[i32], u32, &[bool]); 8] = [
        (&[-1, 5], 1, &[true; 2]),
        (&[-2, -1], 1, &[true; 2]),
        (&[0, -1], 1, &[true; 2]),
        (&[-1, 2, 1], 3, &[true; 9]),
        (&[-1], 0, &[]),
        (&[-1, 0], 3, &[true; 5]),
        (&[1, 1], 3, &[true; 5]),
        (&[-1, 0, 1], 3, &[true; 9]),
    ];
    for (parents, t_max, ancestors) in cases {
        let t = parents.len();
        let mut state = vec![0u8; t];
        let mut path = Vec::with_capacity(t);
        let slices = validate_tree_slices(parents, t_max, ancestors, &mut state, &mut path);
        let owned = TreeMask::new(parents.to_vec(), t_max, ancestors.to_vec()).map(|_| ());
        assert_eq!(
            slices, owned,
            "parity for parents={parents:?} t_max={t_max}"
        );
    }
}

/// Empty trees accept empty scratch on both paths.
#[test]
fn empty_tree_accepts_empty_scratch() {
    let mut state = Vec::new();
    let mut path = Vec::new();
    let slices = validate_tree_slices(&[], 4, &[], &mut state, &mut path);
    let owned = TreeMask::new(Vec::new(), 4, Vec::new()).map(|_| ());
    assert_eq!(slices, owned);
    slices.expect("empty tree with empty scratch validates");
}
