// SPDX-License-Identifier: Apache-2.0
//! Tests proving lazy symbol resolution and cached dispatch (Spec 14 §2, §3).

mod common;

use r9v_hip::{HipError, HipLibrary};
use std::sync::Arc;

#[test]
fn test_symbol_lookup_is_actually_lazy() {
    let (_, missing_so) = common::get_or_compile_stubs();

    // 1. Loading the library succeeds even though hipGraphLaunch is intentionally omitted
    let lib = HipLibrary::load_from_path(&missing_so)
        .expect("library load must succeed even with omitted symbols when lazy");
    let lib_arc = Arc::new(lib);

    // 2. Symbols that ARE present resolve and execute successfully
    let count = lib_arc
        .device_count()
        .expect("present symbol hipGetDeviceCount must succeed");
    assert_eq!(count, 2);

    let dev_id = lib_arc
        .get_device()
        .expect("present symbol hipGetDevice must succeed");
    assert_eq!(dev_id, 0);

    let ptr = lib_arc
        .malloc(128)
        .expect("present symbol hipMalloc must succeed");
    assert!(!ptr.is_null());
    unsafe {
        lib_arc
            .free(ptr)
            .expect("present symbol hipFree must succeed");
    }

    // 3. Invoking the intentionally omitted symbol (hipGraphLaunch) fails with typed SymbolNotFound
    let launch_result = unsafe { lib_arc.graph_launch(std::ptr::null_mut(), std::ptr::null_mut()) };
    match launch_result {
        Err(HipError::SymbolNotFound { symbol, details }) => {
            assert_eq!(symbol, "hipGraphLaunch");
            assert!(
                details.contains("libamdhip64_missing.so"),
                "details must mention the library path: {details}"
            );
        }
        Err(other) => {
            panic!("expected HipError::SymbolNotFound, but got: {:?}", other);
        }
        Ok(_) => {
            panic!("calling missing symbol must fail");
        }
    }
}
