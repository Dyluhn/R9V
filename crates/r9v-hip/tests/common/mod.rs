// SPDX-License-Identifier: Apache-2.0
//! Test fixtures and stub compilation helpers for r9v-hip (Spec 14 §2, §3).

#![allow(dead_code)]

use std::fs;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

unsafe extern "C" {
    fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int;
}

const LOCK_EX: std::os::raw::c_int = 2;
const LOCK_UN: std::os::raw::c_int = 8;

static STUB_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Returns paths to (complete_stub_so, missing_symbols_stub_so).
///
/// Uses an advisory file lock and atomic rename to avoid cross-test-binary races.
pub fn get_or_compile_stubs() -> (PathBuf, PathBuf) {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stub_src = manifest_dir.join("tests/fixtures/stub_hip.c");
    let target_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/test-fixtures");
    fs::create_dir_all(&target_dir).expect("failed to create target/test-fixtures dir");

    let lock_file_path = target_dir.join(".stub_build.lock");
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_file_path)
        .expect("failed to open stub build lock file");

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_EX);
    }

    let complete_so = target_dir.join("libamdhip64_complete.so");
    let missing_so = target_dir.join("libamdhip64_missing.so");

    // Always rebuild under the lock. Reusing a prior test run's shared object
    // can hide a newly required symbol and makes outcomes depend on local
    // target-directory history.
    compile_so(&stub_src, &complete_so, &target_dir, &[]);
    compile_so(
        &stub_src,
        &missing_so,
        &target_dir,
        &["-DOMIT_HIP_GRAPH_LAUNCH"],
    );

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_UN);
    }

    (complete_so, missing_so)
}

/// Returns the path to a compiled stub with an explicit device count.
pub fn get_or_compile_stub_with_count(count: usize) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stub_src = manifest_dir.join("tests/fixtures/stub_hip.c");
    let target_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/test-fixtures");
    fs::create_dir_all(&target_dir).expect("failed to create target/test-fixtures dir");

    let lock_file_path = target_dir.join(".stub_build.lock");
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_file_path)
        .expect("failed to open stub build lock file");

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_EX);
    }

    let stub_so = target_dir.join(format!("libamdhip64_count_{count}.so"));
    let flag = format!("-DSTUB_DEVICE_COUNT={count}");
    compile_so(&stub_src, &stub_so, &target_dir, &[&flag]);

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_UN);
    }

    stub_so
}

/// Returns the path to a compiled stub that returns runtime errors on initialization.
pub fn get_or_compile_broken_stub() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stub_src = manifest_dir.join("tests/fixtures/stub_hip.c");
    let target_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/test-fixtures");
    fs::create_dir_all(&target_dir).expect("failed to create target/test-fixtures dir");

    let lock_file_path = target_dir.join(".stub_build.lock");
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_file_path)
        .expect("failed to open stub build lock file");

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_EX);
    }

    let stub_so = target_dir.join("libamdhip64_broken.so");
    compile_so(
        &stub_src,
        &stub_so,
        &target_dir,
        &["-DSTUB_DEVICE_COUNT_ERROR"],
    );

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_UN);
    }

    stub_so
}

/// Returns a stub whose `hipGetDeviceCount` follows ROCm's no-device contract:
/// status `hipErrorNoDevice` and count zero.
pub fn get_or_compile_no_device_stub() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stub_src = manifest_dir.join("tests/fixtures/stub_hip.c");
    let target_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/test-fixtures");
    fs::create_dir_all(&target_dir).expect("failed to create target/test-fixtures dir");

    let lock_file_path = target_dir.join(".stub_build.lock");
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_file_path)
        .expect("failed to open stub build lock file");

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_EX);
    }

    let stub_so = target_dir.join("libamdhip64_no_device.so");
    compile_so(
        &stub_src,
        &stub_so,
        &target_dir,
        &["-DSTUB_NO_DEVICE_ERROR"],
    );

    unsafe {
        flock(lock_file.as_raw_fd(), LOCK_UN);
    }

    stub_so
}

fn compile_so(src: &Path, dst: &Path, temp_dir: &Path, extra_flags: &[&str]) {
    let count = STUB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_dst = temp_dir.join(format!(
        "{}.tmp.{}.{}",
        dst.file_name().unwrap().to_string_lossy(),
        pid,
        count
    ));

    let mut cmd = Command::new("gcc");
    cmd.args([
        "-shared",
        "-fPIC",
        "-O2",
        "-o",
        tmp_dst.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    for flag in extra_flags {
        cmd.arg(flag);
    }

    let output = cmd
        .output()
        .expect("failed to invoke gcc to build stub libamdhip64");
    if !output.status.success() {
        panic!(
            "gcc failed to compile stub shared library: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fs::rename(&tmp_dst, dst).expect("failed to atomically rename stub shared library");
}
