// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "x86_64" {
        return;
    }

    let encoded_flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    if encoded_flags
        .split('\u{1f}')
        .any(|flag| flag.replace(' ', "").contains("target-cpu=native"))
    {
        panic!("r9v-t0 forbids target-cpu=native; scalar T0 must run on baseline x86_64");
    }

    let allowed = BTreeSet::from(["fxsr", "sse", "sse2"]);
    let enabled = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let unexpected: Vec<&str> = enabled
        .split(',')
        .filter(|feature| !feature.is_empty() && !allowed.contains(feature))
        .collect();
    if !unexpected.is_empty() {
        panic!(
            "r9v-t0 scalar build enabled non-baseline x86_64 features: {}. Optional SIMD must use runtime dispatch in T0v",
            unexpected.join(",")
        );
    }
}
