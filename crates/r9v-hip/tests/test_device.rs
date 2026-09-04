// SPDX-License-Identifier: Apache-2.0
//! Tests proving Device enum behavior and properties (Spec 5 §3.4, Spec 14 §2, §3).

use r9v_hip::Device;
use std::collections::{BTreeSet, HashSet};

#[test]
fn test_device_enum_variants_and_accessors() {
    let cpu = Device::cpu();
    assert!(cpu.is_cpu());
    assert!(!cpu.is_hip());
    assert_eq!(cpu.hip_rank(), None);
    assert_eq!(format!("{cpu}"), "cpu");

    let hip0 = Device::hip(0);
    assert!(!hip0.is_cpu());
    assert!(hip0.is_hip());
    assert_eq!(hip0.hip_rank(), Some(0));
    assert_eq!(format!("{hip0}"), "hip:0");

    let hip1 = Device::hip(1);
    assert!(!hip1.is_cpu());
    assert!(hip1.is_hip());
    assert_eq!(hip1.hip_rank(), Some(1));
    assert_eq!(format!("{hip1}"), "hip:1");
}

#[test]
fn test_device_ordering_and_collections() {
    let cpu = Device::Cpu;
    let hip0 = Device::Hip(r9v_hip::HipOrdinal::new(0));
    let hip1 = Device::Hip(r9v_hip::HipOrdinal::new(1));

    // Ordering: Cpu < Hip(0) < Hip(1)
    assert!(cpu < hip0);
    assert!(hip0 < hip1);

    let mut btree = BTreeSet::new();
    btree.insert(hip1);
    btree.insert(cpu);
    btree.insert(hip0);

    let sorted: Vec<_> = btree.into_iter().collect();
    assert_eq!(
        sorted,
        vec![
            Device::Cpu,
            Device::Hip(r9v_hip::HipOrdinal::new(0)),
            Device::Hip(r9v_hip::HipOrdinal::new(1))
        ]
    );

    let mut set = HashSet::new();
    set.insert(Device::Cpu);
    set.insert(Device::Hip(r9v_hip::HipOrdinal::new(0)));
    assert_eq!(set.len(), 2);
    assert!(set.contains(&Device::Cpu));
    assert!(set.contains(&Device::Hip(r9v_hip::HipOrdinal::new(0))));
}
