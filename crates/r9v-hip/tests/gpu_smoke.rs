// SPDX-License-Identifier: Apache-2.0
//! Integration test entry point for real GPU smoke path (Spec 14 §2, §3).

mod gpu;

#[test]
fn test_real_gpu_smoke_suite() {
    gpu::run_gpu_smoke();
}
