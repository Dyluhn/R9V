// SPDX-License-Identifier: Apache-2.0
//! Tests for top-level error conversions and formatting (Spec 14 §2, CONVENTIONS.md §1).

use r9v_common::{parse_byte_size, ByteSizeError, R9vError};

#[test]
fn error_from_byte_size_error() {
    let err = parse_byte_size("invalid").unwrap_err();
    let r9v_err: R9vError = err.into();
    assert!(matches!(
        r9v_err,
        R9vError::ByteSize(ByteSizeError::MissingNumber { .. })
    ));
    assert!(format!("{r9v_err}").contains("byte size parse error:"));
}

#[test]
fn error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let r9v_err: R9vError = io_err.into();
    assert!(matches!(r9v_err, R9vError::Io(_)));
    assert!(format!("{r9v_err}").contains("I/O error: file not found"));
}
