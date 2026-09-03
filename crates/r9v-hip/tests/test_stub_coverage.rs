// SPDX-License-Identifier: Apache-2.0
//! Tests proving full symbol coverage and signature dispatch against complete stub (Spec 14 §2, §3).

mod common;

use r9v_hip::{
    DeviceBuffer, Event, EventFlags, Graph, HipError, HipLibrary, HostBuffer, MemcpyKind, Module,
    Stream, StreamCaptureMode, StreamFlags,
};
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

#[test]
fn test_every_required_symbol_is_forced_against_complete_stub() {
    let (complete_so, _) = common::get_or_compile_stubs();
    let lib = Arc::new(
        HipLibrary::load_from_path(&complete_so).expect("failed to load complete stub libamdhip64"),
    );

    // 1. Device Count, Device Set/Get, Device Properties
    let count = lib.device_count().expect("device_count failed");
    assert_eq!(count, 2);

    lib.set_device(0).expect("set_device(0) failed");
    let current_dev = lib.get_device().expect("get_device failed");
    assert_eq!(current_dev, 0);

    let props = lib
        .get_device_properties(0)
        .expect("get_device_properties failed");
    assert_eq!(props.name, "Stub AMD Radeon AI PRO R9700");
    assert_eq!(props.gcn_arch_name, "amdgcn-amd-amdhsa--gfx1201");
    assert_eq!(props.total_global_mem, 34359738368);
    assert_eq!(props.warp_size, 32);
    assert_eq!(props.major, 12);
    assert_eq!(props.minor, 0);
    assert_eq!(props.multi_processor_count, 64);
    assert_eq!(props.pci_bus_id, 3);
    assert!(props.can_map_host_memory);
    assert!(props.ecc_enabled);
    assert!(props.cooperative_launch);

    // 2. Malloc / Free / HostMalloc / HostFree
    let dev_ptr = lib.malloc(256).expect("malloc failed");
    assert!(!dev_ptr.is_null());

    let host_ptr = lib.host_malloc(256, 0).expect("host_malloc failed");
    assert!(!host_ptr.is_null());

    // 3. Memcpy (sync), MemcpyAsync, MemcpyPeerAsync with MemcpyKind enum
    let host_data = [0x5Au8; 64];
    unsafe {
        lib.memcpy(
            dev_ptr,
            host_data.as_ptr() as *const c_void,
            64,
            MemcpyKind::Default,
        )
        .expect("memcpy failed");
    }

    // 4. Stream lifecycle & query & wait with StreamFlags enum
    let raw_stream = lib.stream_create().expect("stream_create failed");
    let raw_stream_flags = lib
        .stream_create_with_flags(StreamFlags::NonBlocking)
        .expect("stream_create_with_flags failed");

    unsafe {
        lib.memcpy_async(
            dev_ptr,
            host_data.as_ptr() as *const c_void,
            64,
            MemcpyKind::Default,
            raw_stream,
        )
        .expect("memcpy_async failed");

        lib.memcpy_peer_async(dev_ptr, 0, dev_ptr, 1, 64, raw_stream)
            .expect("memcpy_peer_async failed");

        lib.stream_synchronize(raw_stream)
            .expect("stream_synchronize failed");
        let is_done = lib.stream_query(raw_stream).expect("stream_query failed");
        assert!(is_done);
    }

    // 5. Event lifecycle & timing with EventFlags enum
    let raw_event = lib.event_create().expect("event_create failed");
    let raw_event_flags = lib
        .event_create_with_flags(EventFlags::DisableTiming)
        .expect("event_create_with_flags failed");

    unsafe {
        lib.event_record(raw_event, raw_stream)
            .expect("event_record failed");
        lib.stream_wait_event(raw_stream, raw_event, 0)
            .expect("stream_wait_event failed");
        lib.event_synchronize(raw_event)
            .expect("event_synchronize failed");
        let event_done = lib.event_query(raw_event).expect("event_query failed");
        assert!(event_done);

        let raw_event_end = lib.event_create().expect("event_create 2 failed");
        lib.event_record(raw_event_end, raw_stream)
            .expect("event_record 2 failed");
        let elapsed_ms = lib
            .event_elapsed_time(raw_event, raw_event_end)
            .expect("event_elapsed_time failed");
        assert!((elapsed_ms - 1.25).abs() < 1e-4);

        lib.event_destroy(raw_event).expect("event_destroy failed");
        lib.event_destroy(raw_event_flags)
            .expect("event_destroy flags failed");
        lib.event_destroy(raw_event_end)
            .expect("event_destroy 2 failed");

        lib.stream_destroy(raw_stream)
            .expect("stream_destroy failed");
        lib.stream_destroy(raw_stream_flags)
            .expect("stream_destroy flags failed");

        lib.host_free(host_ptr).expect("host_free failed");
        lib.free(dev_ptr).expect("free failed");
    }

    // 6. Module & Function & Launch
    let raw_mod_file = lib
        .module_load(Path::new("/dummy/empty.co"))
        .expect("module_load failed");
    let dummy_co_bytes = [0x7fu8, b'E', b'L', b'F', 0, 0, 0, 0];
    let raw_mod_data = lib
        .module_load_data(&dummy_co_bytes)
        .expect("module_load_data failed");

    let raw_func = unsafe {
        lib.module_get_function(raw_mod_data, "test_kernel")
            .expect("module_get_function failed")
    };

    let test_stream = lib.stream_create().expect("stream_create failed");
    let mut args: [*mut c_void; 0] = [];
    unsafe {
        lib.module_launch_kernel(raw_func, (1, 1, 1), (32, 1, 1), 0, test_stream, &mut args)
            .expect("module_launch_kernel failed");

        lib.module_unload(raw_mod_file)
            .expect("module_unload failed");
        lib.module_unload(raw_mod_data)
            .expect("module_unload failed");
    }

    // 7. Graph Capture, Instantiate, Launch, Destroy with StreamCaptureMode enum
    unsafe {
        lib.stream_begin_capture(test_stream, StreamCaptureMode::Global)
            .expect("stream_begin_capture failed");
        let raw_graph = lib
            .stream_end_capture(test_stream)
            .expect("stream_end_capture failed");

        let raw_graph_exec = lib
            .graph_instantiate(raw_graph)
            .expect("graph_instantiate failed");

        lib.graph_launch(raw_graph_exec, test_stream)
            .expect("graph_launch failed");
        lib.stream_synchronize(test_stream)
            .expect("stream_synchronize after graph launch failed");

        lib.graph_exec_destroy(raw_graph_exec)
            .expect("graph_exec_destroy failed");
        lib.graph_destroy(raw_graph).expect("graph_destroy failed");
        lib.stream_destroy(test_stream)
            .expect("stream_destroy failed");
    }

    // 8. Peer Access Query / Enable / Disable & Idempotence
    let can_peer = lib
        .device_can_access_peer(0, 1)
        .expect("device_can_access_peer failed");
    assert!(can_peer);

    // First enable succeeds (HIP_SUCCESS)
    lib.device_enable_peer_access(1, 0)
        .expect("first device_enable_peer_access failed");

    // Second enable succeeds idempotently (driver returns 704, library maps to Ok)
    lib.device_enable_peer_access(1, 0)
        .expect("idempotent device_enable_peer_access must succeed");

    // First disable succeeds (HIP_SUCCESS)
    lib.device_disable_peer_access(1)
        .expect("first device_disable_peer_access failed");

    // Second disable succeeds idempotently (driver returns 705, library maps to Ok)
    lib.device_disable_peer_access(1)
        .expect("idempotent device_disable_peer_access must succeed");
}

#[test]
fn test_high_level_raii_handles_against_complete_stub() {
    let (complete_so, _) = common::get_or_compile_stubs();
    let lib = Arc::new(
        HipLibrary::load_from_path(&complete_so).expect("failed to load complete stub libamdhip64"),
    );

    // Stream & Event with typed enums
    let stream = Stream::with_flags(&lib, StreamFlags::Default).expect("Stream::with_flags failed");
    let event = Event::with_flags(&lib, EventFlags::Default).expect("Event::with_flags failed");
    event.record(&stream).expect("event.record failed");
    stream.wait_event(&event).expect("stream.wait_event failed");
    event.synchronize().expect("event.synchronize failed");
    assert!(event.is_complete().expect("is_complete failed"));

    // Buffers & Bounds Verification
    let mut dev_buf = DeviceBuffer::allocate(&lib, 64).expect("DeviceBuffer::allocate failed");
    assert_eq!(dev_buf.size(), 64);
    let host_src = [42u8; 64];
    dev_buf
        .copy_from_host(&host_src)
        .expect("copy_from_host failed");

    let mut host_dst = [0u8; 64];
    dev_buf
        .copy_to_host(&mut host_dst)
        .expect("copy_to_host failed");
    assert_eq!(host_src, host_dst);

    // Bounds checking: buffer too small must return typed HipError::BufferTooSmall
    let oversized = [1u8; 128];
    let err_sync = dev_buf
        .copy_from_host(&oversized)
        .expect_err("oversized host copy must fail");
    match err_sync {
        HipError::BufferTooSmall {
            operation,
            required,
            available,
        } => {
            assert_eq!(operation, "copy_from_host");
            assert_eq!(required, 128);
            assert_eq!(available, 64);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }

    let mut undersized_host = [0u8; 32];
    let mut small_dev = DeviceBuffer::allocate(&lib, 32).expect("allocate small_dev failed");
    unsafe {
        let err_async = dev_buf
            .copy_to_device_async(&mut small_dev, &stream)
            .expect_err("copy_to_device_async into smaller buffer must fail");
        match err_async {
            HipError::BufferTooSmall {
                operation,
                required,
                available,
            } => {
                assert_eq!(operation, "copy_to_device_async");
                assert_eq!(required, 64);
                assert_eq!(available, 32);
            }
            other => panic!("expected BufferTooSmall, got: {:?}", other),
        }

        let err_peer = dev_buf
            .copy_to_peer_async(0, &mut small_dev, 1, &stream)
            .expect_err("copy_to_peer_async into smaller buffer must fail");
        match err_peer {
            HipError::BufferTooSmall {
                operation,
                required,
                available,
            } => {
                assert_eq!(operation, "copy_to_peer_async");
                assert_eq!(required, 64);
                assert_eq!(available, 32);
            }
            other => panic!("expected BufferTooSmall, got: {:?}", other),
        }

        let err_host_async = dev_buf
            .copy_to_host_async(&mut undersized_host, &stream)
            .expect_err("copy_to_host_async into smaller slice must fail");
        match err_host_async {
            HipError::BufferTooSmall {
                operation,
                required,
                available,
            } => {
                assert_eq!(operation, "copy_to_host_async");
                assert_eq!(required, 64);
                assert_eq!(available, 32);
            }
            other => panic!("expected BufferTooSmall, got: {:?}", other),
        }
    }

    let mut host_pinned = HostBuffer::allocate(&lib, 64, 0).expect("HostBuffer::allocate failed");
    host_pinned.as_mut_slice().copy_from_slice(&host_src);
    assert_eq!(host_pinned.as_slice(), &host_src);

    // Module & Function Lifetime: dropping Module keeps Function alive
    let func = {
        let dummy_co = [0x7fu8, b'E', b'L', b'F'];
        let module = Module::load_data(&lib, &dummy_co).expect("Module::load_data failed");
        module
            .get_function("empty_kernel")
            .expect("get_function failed")
        // module is dropped here!
    };

    // Function remains valid and callable because it holds an Arc reference to the module inner allocation
    let mut args: [*mut c_void; 0] = [];
    unsafe {
        func.launch((1, 1, 1), (32, 1, 1), 0, &stream, &mut args)
            .expect("Function must remain callable after original Module handle is dropped");
    }

    // Graph & GraphExec with StreamCaptureMode enum
    Graph::begin_capture(&stream, StreamCaptureMode::Global).expect("begin_capture failed");
    let graph = Graph::end_capture(&stream).expect("end_capture failed");
    let graph_exec = graph.instantiate().expect("instantiate failed");
    unsafe {
        graph_exec.launch(&stream).expect("launch failed");
    }
    stream.synchronize().expect("synchronize failed");
}
