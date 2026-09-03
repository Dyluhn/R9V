// SPDX-License-Identifier: Apache-2.0
//! API shape and trait bound verification for r9v-common (Spec 14 §2, r9v-card-work §6).

use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;

use r9v_common::{
    format_byte_size, parse_byte_size, ByteSize, ByteSizeError, R9vError, ReqId, Result, SeededRng,
    SeqId, StepId,
};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_copy<T: Copy>() {}
fn assert_clone<T: Clone>() {}
fn assert_display<T: Display>() {}
fn assert_hash<T: Hash>() {}
fn assert_from_str<T: FromStr>() {}
fn assert_error<T: std::error::Error>() {}

#[test]
fn api_shape_invariants() {
    assert_send::<SeqId>();
    assert_sync::<SeqId>();
    assert_copy::<SeqId>();
    assert_clone::<SeqId>();
    assert_display::<SeqId>();
    assert_hash::<SeqId>();

    assert_send::<ReqId>();
    assert_sync::<ReqId>();
    assert_copy::<ReqId>();
    assert_clone::<ReqId>();
    assert_display::<ReqId>();
    assert_hash::<ReqId>();

    assert_send::<StepId>();
    assert_sync::<StepId>();
    assert_copy::<StepId>();
    assert_clone::<StepId>();
    assert_display::<StepId>();
    assert_hash::<StepId>();

    assert_send::<ByteSize>();
    assert_sync::<ByteSize>();
    assert_copy::<ByteSize>();
    assert_clone::<ByteSize>();
    assert_display::<ByteSize>();
    assert_hash::<ByteSize>();
    assert_from_str::<ByteSize>();

    assert_send::<ByteSizeError>();
    assert_sync::<ByteSizeError>();
    assert_clone::<ByteSizeError>();
    assert_error::<ByteSizeError>();

    assert_send::<R9vError>();
    assert_sync::<R9vError>();
    assert_error::<R9vError>();

    assert_send::<SeededRng>();
    assert_sync::<SeededRng>();
    assert_clone::<SeededRng>();

    // Verify minimal constructors and accessors
    let s = SeqId::new(1);
    assert_eq!(s.as_u64(), 1);

    let r = ReqId::new(2);
    assert_eq!(r.as_u64(), 2);

    let st = StepId::new(3);
    assert_eq!(st.as_u64(), 3);

    let b = ByteSize::new(1024);
    assert_eq!(b.as_u64(), 1024);

    let _ = parse_byte_size("1024 B");
    let _ = format_byte_size(1024);
    let _: Result<()> = Ok(());
}
