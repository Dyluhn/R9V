// SPDX-License-Identifier: Apache-2.0
//! Deterministic stub-backed tests for the hard GPU allocation budget (Spec 14 §3).
//!
//! No real GPU is required: all HIP calls dispatch to the compiled C stub.

mod common;

use r9v_hip::{AllocationBudget, BudgetedDeviceBuffer, DeviceBuffer, HipError, HipLibrary, Stream};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_clone<T: Clone>() {}

fn load_stub() -> Arc<HipLibrary> {
    let (complete_so, _) = common::get_or_compile_stubs();
    Arc::new(
        HipLibrary::load_from_path(&complete_so).expect("failed to load complete stub libamdhip64"),
    )
}

#[track_caller]
fn expect_refusal(err: HipError, limit: u64, used: u64, requested: u64, available: u64) {
    match err {
        HipError::BudgetExceeded {
            limit: l,
            used: u,
            requested: r,
            available: a,
        } => {
            assert_eq!(l, limit, "limit mismatch");
            assert_eq!(u, used, "used mismatch");
            assert_eq!(r, requested, "requested mismatch");
            assert_eq!(a, available, "available mismatch");
        }
        other => panic!("expected HipError::BudgetExceeded, got: {other:?}"),
    }
}

#[test]
fn test_budget_api_shape() {
    assert_send::<AllocationBudget>();
    assert_sync::<AllocationBudget>();
    assert_clone::<AllocationBudget>();

    assert_send::<BudgetedDeviceBuffer>();
    assert_sync::<BudgetedDeviceBuffer>();

    assert_clone::<HipError>();
    assert_send::<HipError>();
    assert_sync::<HipError>();
}

#[test]
fn test_budget_success_and_shared_clone_accounting() {
    let lib = load_stub();
    let budget = AllocationBudget::new(1024);
    assert_eq!(budget.limit(), 1024);
    assert_eq!(budget.used(), 0);
    assert_eq!(budget.available(), 1024);

    // A clone shares the same ledger: charges aggregate into one counter.
    let clone = budget.clone();
    let mut a = BudgetedDeviceBuffer::allocate(&budget, &lib, 256).expect("allocate a failed");
    assert_eq!(a.size(), 256);
    assert_eq!(a.charged_bytes(), 256);
    assert_eq!(a.budget().limit(), 1024);
    let b = BudgetedDeviceBuffer::allocate(&clone, &lib, 256).expect("allocate b failed");
    assert_eq!(budget.used(), 512);
    assert_eq!(clone.used(), 512);
    assert_eq!(budget.available(), 512);

    // Pointer/size/copy surface works through the wrapper.
    assert!(!a.as_ptr().is_null());
    assert!(!a.as_mut_ptr().is_null());
    let host_src = [0xA5u8; 256];
    a.copy_from_host(&host_src).expect("copy_from_host failed");
    let mut host_dst = [0u8; 256];
    a.copy_to_host(&mut host_dst).expect("copy_to_host failed");
    assert_eq!(host_src, host_dst);

    // Async copies between budgeted buffers stay inside the wrapper API.
    let stream = Stream::new(&lib).expect("Stream::new failed");
    let mut c = BudgetedDeviceBuffer::allocate(&budget, &lib, 256).expect("allocate c failed");
    unsafe {
        a.copy_to_device_async(&mut c, &stream)
            .expect("copy_to_device_async failed");
    }
    stream.synchronize().expect("synchronize failed");
    let mut c_host = [0u8; 256];
    c.copy_to_host(&mut c_host).expect("copy_to_host failed");
    assert_eq!(host_src, c_host);

    assert_eq!(budget.used(), 768);
    drop(a);
    drop(b);
    drop(c);
    assert_eq!(budget.used(), 0);
    assert_eq!(budget.available(), 1024);
}

#[test]
fn test_budget_refusal_happens_before_hip_allocation() {
    let lib = load_stub();
    let budget = AllocationBudget::new(256);
    let full = BudgetedDeviceBuffer::allocate(&budget, &lib, 256).expect("fill budget failed");
    assert_eq!(budget.used(), 256);

    // The stub would satisfy any HIP malloc here, so a typed refusal proves the
    // budget gate runs before hipMalloc and charges nothing.
    let err = match BudgetedDeviceBuffer::allocate(&budget, &lib, 1) {
        Ok(_) => panic!("over-limit must fail"),
        Err(e) => e,
    };
    expect_refusal(err, 256, 256, 1, 0);
    assert_eq!(budget.used(), 256, "refusal must not charge the budget");

    // Releasing restores capacity: the earlier refusal leaked nothing.
    drop(full);
    assert_eq!(budget.used(), 0);
    let _fit = BudgetedDeviceBuffer::allocate(&budget, &lib, 1).expect("must fit after release");
    assert_eq!(budget.used(), 1);
}

#[test]
fn test_budget_release_on_drop_is_exact() {
    let lib = load_stub();
    let budget = AllocationBudget::new(512);
    let a = BudgetedDeviceBuffer::allocate(&budget, &lib, 128).expect("allocate a failed");
    let b = BudgetedDeviceBuffer::allocate(&budget, &lib, 64).expect("allocate b failed");
    assert_eq!(budget.used(), 192);
    drop(a);
    assert_eq!(
        budget.used(),
        64,
        "drop must release exactly the charged bytes"
    );
    drop(b);
    assert_eq!(budget.used(), 0);
    assert_eq!(budget.available(), 512);
}

#[test]
fn test_budget_hip_failure_rolls_back_reservation() {
    let lib = load_stub();
    // Unlimited budget so reservation succeeds; the stub fails hipMalloc for
    // the 0xDEADBEEF sentinel size with hipErrorOutOfMemory.
    let budget = AllocationBudget::new(u64::MAX);
    const HIP_OOM_SENTINEL: usize = 0xDEADBEEF;
    let err = match BudgetedDeviceBuffer::allocate(&budget, &lib, HIP_OOM_SENTINEL) {
        Ok(_) => panic!("sentinel HIP malloc must fail"),
        Err(e) => e,
    };
    match err {
        HipError::ApiError { op, .. } => assert_eq!(op, "hipMalloc"),
        other => panic!("expected HipError::ApiError from hipMalloc, got: {other:?}"),
    }
    assert_eq!(
        budget.used(),
        0,
        "failed HIP allocation must roll back the reservation"
    );
    // Budget remains usable afterwards.
    let _ok = BudgetedDeviceBuffer::allocate(&budget, &lib, 64).expect("must work after rollback");
    assert_eq!(budget.used(), 64);
}

#[test]
fn test_budget_overflow_is_typed_refusal_never_panic() {
    let lib = load_stub();
    let budget = AllocationBudget::new(u64::MAX);
    let _held = BudgetedDeviceBuffer::allocate(&budget, &lib, 64).expect("allocate failed");
    assert_eq!(budget.used(), 64);

    // used (64) + requested (u64::MAX) overflows u64: must be a typed refusal.
    let err = match BudgetedDeviceBuffer::allocate(&budget, &lib, usize::MAX) {
        Ok(_) => panic!("overflow must fail"),
        Err(e) => e,
    };
    expect_refusal(err, u64::MAX, 64, u64::MAX, u64::MAX - 64);
    assert_eq!(
        budget.used(),
        64,
        "overflow refusal must not charge the budget"
    );
}

#[test]
fn test_budget_zero_size_matches_device_buffer() {
    let lib = load_stub();
    // Baseline: direct DeviceBuffer zero-size behavior.
    let direct = DeviceBuffer::allocate(&lib, 0).expect("direct zero-size allocate failed");
    assert_eq!(direct.size(), 0);

    let budget = AllocationBudget::new(64);
    let buf = BudgetedDeviceBuffer::allocate(&budget, &lib, 0).expect("zero-size allocate failed");
    assert_eq!(buf.size(), 0);
    assert_eq!(buf.charged_bytes(), 0);
    assert_eq!(budget.used(), 0, "zero-size must not corrupt accounting");

    // Empty copies round-trip like the direct path.
    let mut buf = buf;
    buf.copy_from_host(&[])
        .expect("empty copy_from_host failed");
    let mut dst: [u8; 0] = [];
    buf.copy_to_host(&mut dst)
        .expect("empty copy_to_host failed");

    drop(buf);
    drop(direct);
    assert_eq!(budget.used(), 0);
}

#[test]
fn test_budget_shared_concurrent_never_overcommits() {
    let lib = load_stub();
    let budget = AllocationBudget::new(1024);
    const THREADS: usize = 8;
    const PER_THREAD: usize = 4;
    const EACH: usize = 64; // 1024 / 64 = exactly 16 slots for 32 attempts.
    let failures = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let lib_clone = Arc::clone(&lib);
        let budget_clone = budget.clone();
        let failures_clone = Arc::clone(&failures);
        handles.push(thread::spawn(move || {
            let mut held = Vec::new();
            for _ in 0..PER_THREAD {
                match BudgetedDeviceBuffer::allocate(&budget_clone, &lib_clone, EACH) {
                    Ok(buf) => held.push(buf),
                    Err(HipError::BudgetExceeded { .. }) => {
                        failures_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(other) => panic!("unexpected allocation error: {other:?}"),
                }
            }
            held
        }));
    }

    // Buffers stay alive in the joined vectors until after the peak assertions.
    let mut held_all = Vec::new();
    for handle in handles {
        held_all.push(handle.join().expect("worker thread panicked"));
    }
    let successes: usize = held_all.iter().map(Vec::len).sum();
    assert_eq!(successes, 16, "exactly 16 of 32 attempts must succeed");
    assert_eq!(failures.load(Ordering::Relaxed), 16);
    assert_eq!(budget.used(), 1024);
    assert!(
        budget.used() <= budget.limit(),
        "budget must never overcommit"
    );
    assert_eq!(budget.available(), 0);

    drop(held_all);
    assert_eq!(
        budget.used(),
        0,
        "all releases must return the budget to zero"
    );
}
