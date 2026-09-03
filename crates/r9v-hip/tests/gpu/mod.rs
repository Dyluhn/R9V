// SPDX-License-Identifier: Apache-2.0
//! Real GPU smoke test implementation (Spec 4 §10, Spec 5 §7, Spec 14 §2, §3).

use std::path::{Path, PathBuf};
use std::process::Command;

use r9v_hip::{
    DeviceBuffer, Event, Graph, HipError, HipLibrary, Module, Stream, StreamCaptureMode,
};

/// Finds the ROCm LLVM bin directory containing `clang++` and `clang-offload-bundler`.
///
/// Searches only explicit `ROCM_PATH` or `/opt/rocm` (Spec 14 §3).
fn find_rocm_llvm_bin() -> Option<PathBuf> {
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        let base = PathBuf::from(&rocm_path);
        let p = base.join("lib/llvm/bin");
        if p.is_dir() {
            return Some(p);
        }
        let p_bin = base.join("bin");
        if p_bin.is_dir() {
            return Some(p_bin);
        }
    }
    let opt_p = PathBuf::from("/opt/rocm/lib/llvm/bin");
    if opt_p.is_dir() {
        return Some(opt_p);
    }
    let opt_bin = PathBuf::from("/opt/rocm/bin");
    if opt_bin.is_dir() {
        return Some(opt_bin);
    }
    None
}

/// Finds the ROCm device bitcode directory if present under `ROCM_PATH` or `/opt/rocm` (Spec 14 §3).
fn find_rocm_device_lib() -> Option<PathBuf> {
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        let base = PathBuf::from(&rocm_path);
        for sub in [
            "lib/llvm/amdgcn/bitcode",
            "amdgcn/bitcode",
            "lib/amdgcn/bitcode",
        ] {
            let p = base.join(sub);
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    for cand in [
        "/opt/rocm/lib/llvm/amdgcn/bitcode",
        "/opt/rocm/amdgcn/bitcode",
        "/opt/rocm/lib/amdgcn/bitcode",
    ] {
        let p = PathBuf::from(cand);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Runs the complete real GPU smoke test sequence on hardware runners (Spec 14 §3).
///
/// When no HIP runtime library or zero GPU devices are present, reports an explicit skip.
/// When hardware is present, any failure panics immediately.
pub fn run_gpu_smoke() {
    let lib = match HipLibrary::default_or_load() {
        Ok(l) => l,
        Err(HipError::LibraryNotFound { searched }) => {
            println!(
                "[SKIP] HIP dynamic library not available on this host; searched: {searched:?}"
            );
            return;
        }
        Err(e) => {
            panic!("failed to load HIP dynamic library: {e}");
        }
    };

    let count = lib
        .device_count()
        .expect("device_count query failed after HIP library load");
    if count == 0 {
        println!("[SKIP] No HIP GPU devices available on this host (device count == 0)");
        return;
    }

    println!("=== Real GPU Smoke Test (Spec 14 §3) ===");
    println!("Discovered {count} HIP device(s)");

    // 1. Device 0 Properties and Configuration
    lib.set_device(0).expect("failed to set active device to 0");
    let current_dev = lib.get_device().expect("failed to get active device");
    assert_eq!(current_dev, 0);

    let props0 = lib
        .get_device_properties(0)
        .expect("failed to get properties for device 0");
    println!(
        "Device 0: {} [arch: {}, VRAM: {} GiB, CUs: {}, PCIe: {:02x}:{:02x}.0]",
        props0.name,
        props0.gcn_arch_name,
        props0.total_global_mem / (1024 * 1024 * 1024),
        props0.multi_processor_count,
        props0.pci_bus_id,
        props0.pci_device_id
    );

    // Derive architecture strictly from device properties without fallback
    let arch = match props0.gcn_arch_name.find("gfx") {
        Some(idx) => {
            let candidate = &props0.gcn_arch_name[idx..];
            let end = candidate
                .find(|c: char| !c.is_alphanumeric())
                .unwrap_or(candidate.len());
            &candidate[..end]
        }
        None => {
            panic!(
                "device 0 gcn_arch_name {:?} does not contain an identifiable gfx architecture",
                props0.gcn_arch_name
            );
        }
    };

    // 2. Linear Memory Allocation on Device 0
    const BUFFER_SIZE: usize = 4096;
    let mut dev_buf0_a = DeviceBuffer::allocate(&lib, BUFFER_SIZE)
        .expect("failed to allocate dev_buf0_a on device 0");
    let mut dev_buf0_b = DeviceBuffer::allocate(&lib, BUFFER_SIZE)
        .expect("failed to allocate dev_buf0_b on device 0");
    assert_eq!(dev_buf0_a.size(), BUFFER_SIZE);
    assert_eq!(dev_buf0_b.size(), BUFFER_SIZE);

    let stream0 = Stream::new(&lib).expect("failed to create HIP stream on device 0");

    // 3. Host-to-Device Copy on Device 0
    let mut host_send = vec![0u8; BUFFER_SIZE];
    for (i, byte) in host_send.iter_mut().enumerate() {
        *byte = ((i * 31 + 7) & 0xFF) as u8;
    }

    unsafe {
        dev_buf0_a
            .copy_from_host_async(&host_send, &stream0)
            .expect("copy_from_host_async on device 0 failed");
    }
    stream0
        .synchronize()
        .expect("stream0 synchronize after H2D failed");

    // 4. Peer Transfer Branch (Device 0 -> Device 1)
    let can_access_peer = if count >= 2 {
        lib.device_can_access_peer(0, 1)
            .expect("failed to query peer-access capability")
    } else {
        false
    };

    if can_access_peer {
        println!(
            "Peer access supported between Device 0 and Device 1; exercising isolated peer path"
        );

        lib.device_enable_peer_access(1, 0)
            .expect("failed to enable peer access on device 0 to device 1");

        lib.set_device(1).expect("failed to switch to device 1");
        let mut peer_buf1 = DeviceBuffer::allocate(&lib, BUFFER_SIZE)
            .expect("failed to allocate peer_buf1 on device 1");

        lib.set_device(0)
            .expect("failed to switch back to device 0");

        // Peer copy on device-0 stream
        unsafe {
            dev_buf0_a
                .copy_to_peer_async(0, &mut peer_buf1, 1, &stream0)
                .expect("copy_to_peer_async from device 0 to device 1 failed");
        }
        stream0
            .synchronize()
            .expect("stream0 synchronize after peer copy failed");

        // Switch to device 1 to read and verify peer buffer using device-1 stream
        lib.set_device(1)
            .expect("failed to switch to device 1 for D2H");
        let stream1 = Stream::new(&lib).expect("failed to create stream1 on device 1");
        let mut host_recv_peer = vec![0u8; BUFFER_SIZE];

        unsafe {
            peer_buf1
                .copy_to_host_async(&mut host_recv_peer, &stream1)
                .expect("peer copy_to_host_async on device 1 failed");
        }
        stream1
            .synchronize()
            .expect("stream1 synchronize after D2H failed");
        assert_eq!(
            host_send, host_recv_peer,
            "peer transfer bytes on device 1 do not match original host data"
        );
        println!("Peer transfer verification passed ({BUFFER_SIZE} bytes bit-identical)");

        // Drop peer resources while active context is device 1
        drop(peer_buf1);
        drop(stream1);

        // Switch back to device 0 for subsequent single-device tests
        lib.set_device(0)
            .expect("failed to switch back to device 0");
    } else {
        println!("Multi-GPU peer access not available; skipping peer transfer test");
    }

    // 5. Device-to-Device Copy on Device 0
    unsafe {
        dev_buf0_a
            .copy_to_device_async(&mut dev_buf0_b, &stream0)
            .expect("copy_to_device_async on device 0 failed");
    }

    let mut host_recv_d2d = vec![0u8; BUFFER_SIZE];
    unsafe {
        dev_buf0_b
            .copy_to_host_async(&mut host_recv_d2d, &stream0)
            .expect("copy_to_host_async after D2D failed");
    }
    stream0
        .synchronize()
        .expect("stream0 synchronize after D2D failed");
    assert_eq!(
        host_send, host_recv_d2d,
        "D2D memory transfer bytes do not match original host data"
    );
    println!("D2D copy verification passed ({BUFFER_SIZE} bytes bit-identical on Device 0)");

    // 6. Event Lifecycle & Timing on Device 0
    let event_start = Event::new(&lib).expect("failed to create start event");
    let event_stop = Event::new(&lib).expect("failed to create stop event");

    event_start
        .record(&stream0)
        .expect("failed to record start event");
    unsafe {
        dev_buf0_a
            .copy_to_device_async(&mut dev_buf0_b, &stream0)
            .expect("copy_to_device_async during event timing failed");
    }
    event_stop
        .record(&stream0)
        .expect("failed to record stop event");
    event_stop
        .synchronize()
        .expect("failed to synchronize stop event");

    let elapsed_ms = event_stop
        .elapsed_since(&event_start)
        .expect("failed to measure elapsed time between events");
    assert!(elapsed_ms >= 0.0);
    println!("Event timing lifecycle passed ({elapsed_ms:.4} ms)");

    // 7. Compile, Load, Get, and Launch empty kernel (Spec 4 §10, Spec 14 §3)
    let llvm_bin = find_rocm_llvm_bin().unwrap_or_else(|| {
        panic!("ROCm LLVM compiler toolchain (clang++, clang-offload-bundler) not found in ROCM_PATH or /opt/rocm");
    });
    let clang = llvm_bin.join("clang++");
    let bundler = llvm_bin.join("clang-offload-bundler");
    assert!(clang.is_file(), "clang++ not found at {}", clang.display());
    assert!(
        bundler.is_file(),
        "clang-offload-bundler not found at {}",
        bundler.display()
    );
    println!(
        "Found ROCm LLVM compiler toolchain at: {}",
        llvm_bin.display()
    );

    let target_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/test-fixtures");
    let _ = std::fs::create_dir_all(&target_dir);
    let hip_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gpu/empty_kernel.hip");
    assert!(
        hip_src.is_file(),
        "committed empty_kernel.hip fixture not found at {}",
        hip_src.display()
    );

    let bundle_path = target_dir.join(format!("empty_bundle_{arch}.o"));
    let co_path = target_dir.join(format!("empty_kernel_{arch}.co"));

    let mut compile_cmd = Command::new(&clang);
    compile_cmd.args([
        "-x",
        "hip",
        &format!("--offload-arch={arch}"),
        "--offload-device-only",
        "-O3",
        "-c",
        "-o",
        bundle_path.to_str().unwrap(),
        hip_src.to_str().unwrap(),
    ]);
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        compile_cmd.arg(format!("--rocm-path={rocm_path}"));
    }
    if let Some(dev_lib) = find_rocm_device_lib() {
        compile_cmd.arg(format!("--rocm-device-lib-path={}", dev_lib.display()));
    }

    let compile_status = compile_cmd
        .status()
        .expect("failed to invoke clang++ to compile empty_kernel.hip");
    assert!(
        compile_status.success(),
        "clang++ failed to compile empty_kernel.hip for architecture {arch}"
    );

    let unbundle_status = Command::new(&bundler)
        .args([
            "--type=o",
            &format!("--targets=hipv4-amdgcn-amd-amdhsa--{arch}"),
            &format!("--input={}", bundle_path.display()),
            &format!("--output={}", co_path.display()),
            "--unbundle",
        ])
        .status()
        .expect("failed to invoke clang-offload-bundler");
    assert!(
        unbundle_status.success(),
        "clang-offload-bundler failed to unbundle {arch} code object"
    );

    let module = Module::load_file(&lib, &co_path)
        .expect("Module::load_file failed for compiled empty kernel");
    let func = module
        .get_function("empty_kernel")
        .expect("Module::get_function failed for 'empty_kernel'");

    // Derive launch dimensions dynamically from device properties
    let block_threads = props0.warp_size.max(1) as u32;
    unsafe {
        func.launch((1, 1, 1), (block_threads, 1, 1), 0, &stream0, &mut [])
            .expect("func.launch failed");
    }
    stream0
        .synchronize()
        .expect("stream0 synchronize after kernel launch failed");
    println!("Empty kernel compiled and launched successfully on {arch} with block size {block_threads}!");

    // 8. Graph capture, instantiate, and launch on Device 0
    Graph::begin_capture(&stream0, StreamCaptureMode::Global).expect("begin_capture failed");
    unsafe {
        dev_buf0_a
            .copy_to_device_async(&mut dev_buf0_b, &stream0)
            .expect("copy in graph capture failed");
    }
    let graph = Graph::end_capture(&stream0).expect("end_capture failed");

    let graph_exec = graph.instantiate().expect("graph.instantiate failed");
    unsafe {
        graph_exec
            .launch(&stream0)
            .expect("graph_exec.launch failed");
    }
    stream0
        .synchronize()
        .expect("stream0 synchronize after graph execution failed");
    println!("Graph capture, instantiate, and launch passed on Device 0!");

    println!("=== Real GPU Smoke Test Succeeded ===");
}
