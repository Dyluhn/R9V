// SPDX-License-Identifier: Apache-2.0
//! API shape tests verifying public exports, trait implementations, and visibility (Spec 14 §2, §3, CONVENTIONS.md §4.1).

use r9v_hip::{
    default_library, device_count, is_available, Device, DeviceBuffer, DeviceProperties, Event,
    EventFlags, Function, Graph, GraphExec, HipError, HipLibrary, HostBuffer, MemcpyKind, Module,
    Result, Stream, StreamCaptureMode, StreamFlags,
};
use std::fmt::Display;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_display<T: Display>() {}
fn assert_clone<T: Clone>() {}
fn assert_copy<T: Copy>() {}

#[test]
fn test_api_shape_invariants() {
    // Top-level discovery helpers
    let _: fn() -> Result<std::sync::Arc<HipLibrary>> = default_library;
    let _: fn() -> bool = is_available;
    let _: fn() -> Result<u32> = device_count;

    // Device enum traits
    assert_copy::<Device>();
    assert_clone::<Device>();
    assert_send::<Device>();
    assert_sync::<Device>();
    assert_display::<Device>();

    // Typed parameter enums
    assert_copy::<MemcpyKind>();
    assert_clone::<MemcpyKind>();
    assert_send::<MemcpyKind>();
    assert_sync::<MemcpyKind>();

    assert_copy::<StreamCaptureMode>();
    assert_clone::<StreamCaptureMode>();
    assert_send::<StreamCaptureMode>();
    assert_sync::<StreamCaptureMode>();

    assert_copy::<StreamFlags>();
    assert_clone::<StreamFlags>();
    assert_send::<StreamFlags>();
    assert_sync::<StreamFlags>();

    assert_copy::<EventFlags>();
    assert_clone::<EventFlags>();
    assert_send::<EventFlags>();
    assert_sync::<EventFlags>();

    // DeviceProperties
    assert_clone::<DeviceProperties>();
    assert_send::<DeviceProperties>();
    assert_sync::<DeviceProperties>();

    // RAII handles
    assert_send::<Stream>();
    assert_sync::<Stream>();

    assert_send::<Event>();
    assert_sync::<Event>();

    assert_clone::<Module>();
    assert_send::<Module>();
    assert_sync::<Module>();

    assert_clone::<Function>();
    assert_send::<Function>();
    assert_sync::<Function>();

    assert_send::<Graph>();
    assert_sync::<Graph>();

    assert_send::<GraphExec>();
    assert_sync::<GraphExec>();

    assert_send::<DeviceBuffer>();
    assert_sync::<DeviceBuffer>();

    assert_send::<HostBuffer>();
    assert_sync::<HostBuffer>();

    assert_send::<HipLibrary>();
    assert_sync::<HipLibrary>();

    // HipError: Must implement Clone
    assert_clone::<HipError>();
    assert_send::<HipError>();
    assert_sync::<HipError>();
    assert_display::<HipError>();
}
