// SPDX-License-Identifier: Apache-2.0
//! Tests proving nonzero hipError_t maps to typed error with operation and numeric code (Spec 14 §2, §3).

mod common;

use r9v_hip::{HipError, HipLibrary};
use std::sync::Arc;

#[test]
fn test_nonzero_hiperror_maps_to_operation_and_numeric_code() {
    let (complete_so, _) = common::get_or_compile_stubs();
    let lib =
        Arc::new(HipLibrary::load_from_path(&complete_so).expect("failed to load complete stub"));

    // 1. Trigger hipErrorInvalidDevice (101) via set_device(999)
    let err1 = lib.set_device(999).expect_err("set_device(999) must fail");
    match &err1 {
        HipError::ApiError { op, code, message } => {
            assert_eq!(*op, "hipSetDevice");
            assert_eq!(*code, 101);
            assert!(
                message.contains("hipErrorInvalidDevice"),
                "message must contain driver description: {message}"
            );
        }
        other => panic!("expected ApiError, got: {:?}", other),
    }

    let display1 = format!("{err1}");
    assert!(
        display1.contains("hipSetDevice"),
        "Display must contain operation name 'hipSetDevice': {display1}"
    );
    assert!(
        display1.contains("101"),
        "Display must contain numeric status code '101': {display1}"
    );

    // 2. Trigger hipErrorOutOfMemory (2) via malloc(0xDEADBEEF)
    let err2 = lib
        .malloc(0xDEADBEEF)
        .expect_err("malloc(0xDEADBEEF) must fail");
    match &err2 {
        HipError::ApiError { op, code, message } => {
            assert_eq!(*op, "hipMalloc");
            assert_eq!(*code, 2);
            assert!(
                message.contains("hipErrorOutOfMemory"),
                "message must contain driver description: {message}"
            );
        }
        other => panic!("expected ApiError, got: {:?}", other),
    }

    let display2 = format!("{err2}");
    assert!(
        display2.contains("hipMalloc"),
        "Display must contain operation name 'hipMalloc': {display2}"
    );
    assert!(
        display2.contains("2"),
        "Display must contain numeric status code '2': {display2}"
    );
}

#[test]
fn test_result_type_alias_interop() {
    fn produces_result() -> r9v_hip::Result<u32> {
        Ok(42)
    }

    let val = produces_result().expect("Result type alias must behave normally");
    assert_eq!(val, 42);
}
