// SPDX-License-Identifier: Apache-2.0
//! Tests for opaque identifier newtypes (Spec 3 §2, Spec 6 §9, Spec 10 §4, Spec 11 §11, Spec 14 §2).

use std::collections::BTreeSet;

use r9v_common::{ReqId, SeqId, StepId};

#[test]
fn seq_id_construction_and_accessors() {
    let id = SeqId::new(42);
    assert_eq!(id.as_u64(), 42);
}

#[test]
fn req_id_construction_and_accessors() {
    let id = ReqId::new(1001);
    assert_eq!(id.as_u64(), 1001);
}

#[test]
fn step_id_construction_and_accessors() {
    let id = StepId::new(9999);
    assert_eq!(id.as_u64(), 9999);
}

#[test]
fn display_formatting() {
    let seq = SeqId::new(12345);
    assert_eq!(format!("{seq}"), "12345");

    let req = ReqId::new(67890);
    assert_eq!(format!("{req}"), "67890");

    let step = StepId::new(54321);
    assert_eq!(format!("{step}"), "54321");
}

#[test]
fn ordering_and_containers() {
    let mut set = BTreeSet::new();
    set.insert(SeqId::new(3));
    set.insert(SeqId::new(1));
    set.insert(SeqId::new(2));

    let ordered: Vec<u64> = set.iter().map(|id| id.as_u64()).collect();
    assert_eq!(ordered, vec![1, 2, 3]);
}
