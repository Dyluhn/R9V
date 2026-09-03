// SPDX-License-Identifier: Apache-2.0
//! Tests proving typed error handling for library absence (Spec 14 §2, §3).

use r9v_hip::{HipError, HipLibrary};
use std::path::Path;

#[test]
fn test_library_absence_is_typed() {
    let fake_path = Path::new("/nonexistent/directory/libamdhip64.so.7");
    let result = HipLibrary::load_from_path(fake_path);

    match result {
        Err(HipError::LibraryNotFound { searched }) => {
            assert!(
                !searched.is_empty(),
                "searched paths should record the attempted candidate"
            );
            assert!(
                searched[0].contains("libamdhip64.so.7"),
                "searched record should cite target library: {:?}",
                searched
            );
        }
        Err(other) => {
            panic!("expected HipError::LibraryNotFound, but got: {:?}", other);
        }
        Ok(_) => {
            panic!("loading nonexistent library should not succeed");
        }
    }
}
