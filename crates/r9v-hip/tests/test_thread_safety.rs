// SPDX-License-Identifier: Apache-2.0
//! Tests proving library lifetime and thread safety traits (Spec 14 §2, §3).

mod common;

use r9v_hip::{
    DeviceBuffer, Event, Function, Graph, GraphExec, HipLibrary, HostBuffer, Module, Stream,
};
use std::sync::Arc;
use std::thread;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_thread_traits_statically_enforced() {
    assert_send::<HipLibrary>();
    assert_sync::<HipLibrary>();

    assert_send::<Stream>();
    assert_sync::<Stream>();

    assert_send::<Event>();
    assert_sync::<Event>();

    assert_send::<Module>();
    assert_sync::<Module>();

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
}

#[test]
fn test_concurrent_multithreaded_library_dispatch() {
    let (complete_so, _) = common::get_or_compile_stubs();
    let lib =
        Arc::new(HipLibrary::load_from_path(&complete_so).expect("failed to load complete stub"));

    let mut handles = Vec::new();
    for thread_idx in 0..8 {
        let lib_clone = Arc::clone(&lib);
        let handle = thread::spawn(move || {
            for i in 0..50 {
                let stream = Stream::new(&lib_clone).expect("Stream::new in thread");
                let event = Event::new(&lib_clone).expect("Event::new in thread");
                event.record(&stream).expect("Event::record in thread");
                event.synchronize().expect("Event::sync in thread");

                let mut buf = DeviceBuffer::allocate(&lib_clone, 64)
                    .expect("DeviceBuffer::allocate in thread");
                let host_src = [(thread_idx * 10 + (i % 10)) as u8; 64];
                buf.copy_from_host(&host_src)
                    .expect("copy_from_host in thread");

                let mut host_dst = [0u8; 64];
                buf.copy_to_host(&mut host_dst)
                    .expect("copy_to_host in thread");
                assert_eq!(host_src, host_dst);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("thread panicked");
    }
}
