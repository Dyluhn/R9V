// SPDX-License-Identifier: Apache-2.0
//! Integration test for real GPU module loading via Registry (Spec 4 §10, §11, Spec 14 §2, §3, CONVENTIONS §4.4).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use r9v_hip::{HipError, HipLibrary};
use r9v_registry::{
    ArchName, ArtifactOrigin, LaunchGeometry, OpId, Registry, RegistryConfig, ResolvedVariant,
    Tier, VariantHash,
};

const EMBEDDED_EMPTY_KERNEL_SOURCE: &str = r#"// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>

extern "C" __global__ void empty_kernel() {
}
"#;

fn is_gpu_lane() -> bool {
    std::env::var("R9V_GPU_LANE").is_ok()
        || std::env::var("GPU_LANE").is_ok()
        || std::env::var("R9V_REQUIRE_GPU").is_ok()
        || std::env::var("R9V_RENDER_NODES").is_ok()
}

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

#[test]
fn test_a3_1_gpu_module_load() {
    let gpu_lane = is_gpu_lane();

    let lib = match HipLibrary::default_or_load() {
        Ok(l) => Arc::new(l),
        Err(HipError::LibraryNotFound { searched }) => {
            if gpu_lane {
                panic!(
                    "GPU lane required but HIP dynamic library not found; searched: {searched:?}"
                );
            }
            println!(
                "[SKIP] HIP dynamic library not available on this host; searched: {searched:?}"
            );
            println!(
                "[HONEST TIER REPORT] Host tier: T0 (Scalar/SIMD reference or Stub only); hardware T1/T2 unavailable."
            );
            return;
        }
        Err(e) => {
            if gpu_lane {
                panic!("failed to load HIP dynamic library: {e}");
            }
            println!("[SKIP] HIP dynamic library failed to load: {e}");
            return;
        }
    };

    let count = match lib.device_count() {
        Ok(c) => c,
        Err(e) => {
            if gpu_lane {
                panic!("device_count query failed after HIP library load: {e}");
            }
            println!("[SKIP] Failed to query device count: {e}");
            return;
        }
    };

    if count == 0 {
        if gpu_lane {
            panic!("GPU lane required but no HIP GPU devices found (device count == 0)");
        }
        println!("[SKIP] No HIP GPU devices available on this host (device count == 0)");
        println!(
            "[HONEST TIER REPORT] Host tier: T0 (Scalar/SIMD reference or Stub only); hardware T1/T2 unavailable."
        );
        return;
    }

    lib.set_device(0).expect("failed to set active device to 0");
    let props0 = lib
        .get_device_properties(0)
        .expect("failed to get properties for device 0");
    println!("=== Real GPU Module Load Test (Spec 4 §10, Spec 14 §3) ===");
    println!("Discovered {count} HIP device(s)");
    println!(
        "Device 0: {} [arch: {}, VRAM: {} GiB, CUs: {}]",
        props0.name,
        props0.gcn_arch_name,
        props0.total_global_mem / (1024 * 1024 * 1024),
        props0.multi_processor_count
    );

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
                "Unable to determine gfx architecture from device 0: {} (device count == {count})",
                props0.gcn_arch_name
            );
        }
    };

    // Locate or compile a valid code object for this arch.
    // Once HIP device count is nonzero, the test must never skip for missing compiler or fixture (Spec 14 §3).
    let co_path: PathBuf = if let Ok(env_co) = std::env::var("R9V_TEST_CODE_OBJECT") {
        PathBuf::from(env_co)
    } else {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_dir = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(manifest_dir);

        let target_root = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace_dir.join("target"));
        let target_dir = target_root.join("test-fixtures");
        std::fs::create_dir_all(&target_dir)
            .expect("failed to create writable GPU test fixture directory");

        // Owned fixture in tests/gpu with embedded fallback
        let fixture_hip = workspace_dir.join("tests/gpu/empty_kernel.hip");
        let source_path = if fixture_hip.is_file() {
            fixture_hip
        } else {
            let embedded_src = target_dir.join("embedded_empty_kernel.hip");
            std::fs::write(&embedded_src, EMBEDDED_EMPTY_KERNEL_SOURCE)
                .expect("failed to write embedded kernel source");
            embedded_src
        };

        let bin_dir = find_rocm_llvm_bin().unwrap_or_else(|| {
            panic!(
                "HIP device count is nonzero ({count}), but ROCm LLVM bin directory was not found; refusing to skip"
            )
        });
        let clang = bin_dir.join("clang++");
        let bundler = bin_dir.join("clang-offload-bundler");
        assert!(
            clang.is_file(),
            "clang++ not found at {} despite nonzero device count ({count})",
            clang.display()
        );
        assert!(
            bundler.is_file(),
            "clang-offload-bundler not found at {} despite nonzero device count ({count})",
            bundler.display()
        );

        let bundle_path = target_dir.join(format!("empty_bundle_{arch}.o"));
        let compiled_co = target_dir.join(format!("empty_kernel_{arch}.co"));

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
            source_path.to_str().unwrap(),
        ]);
        if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
            compile_cmd.arg(format!("--rocm-path={rocm_path}"));
        }
        if let Some(dev_lib) = find_rocm_device_lib() {
            compile_cmd.arg(format!("--rocm-device-lib-path={}", dev_lib.display()));
        }

        let status = compile_cmd.status().expect("clang++ invocation failed");
        assert!(
            status.success(),
            "clang++ failed to compile empty_kernel.hip"
        );

        let unbundle_status = Command::new(&bundler)
            .args([
                "--type=o",
                &format!("--targets=hipv4-amdgcn-amd-amdhsa--{arch}"),
                &format!("--input={}", bundle_path.display()),
                &format!("--output={}", compiled_co.display()),
                "--unbundle",
            ])
            .status()
            .expect("clang-offload-bundler invocation failed");
        assert!(unbundle_status.success(), "clang-offload-bundler failed");

        compiled_co
    };

    assert!(
        co_path.is_file(),
        "code object must exist at {}",
        co_path.display()
    );

    let co_dir = co_path.parent().map(|p| p.to_path_buf());
    let co_file = co_path
        .file_name()
        .expect("code object path must have file name")
        .to_str()
        .expect("valid utf-8 file name")
        .to_string();

    let registry = Registry::new(RegistryConfig::default());
    let variant = ResolvedVariant {
        variant_hash: VariantHash::new(0xa301_0001),
        arch: ArchName::from(arch),
        op: OpId::Matmul,
        tier: Tier::T2,
        entry_symbol: "empty_kernel".to_string(),
        launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        workspace_bytes: 0,
        static_bytes: 0,
        static_flops: 0,
        code_object_path: Some(co_file),
        code_object_bytes: None,
        validated: true,
        artifact_origin: Some(ArtifactOrigin::Local { base_dir: co_dir }),
    };

    let loaded_mod = registry
        .load_module(&lib, &variant)
        .expect("Registry::load_module must succeed with valid code object on GPU runner");
    let func = loaded_mod
        .get_function("empty_kernel")
        .expect("Module::get_function must find empty_kernel");
    drop(func);

    // Verify module caching in Registry
    let cached_mod = registry
        .load_module(&lib, &variant)
        .expect("Registry::load_module cached lookup must succeed");
    assert!(
        Arc::ptr_eq(&loaded_mod, &cached_mod),
        "consecutive load_module calls for the same variant_hash must return the same Arc<Module>"
    );
    println!(
        "Successfully loaded, verified, and cached module for arch {arch} on device {}",
        props0.name
    );
}
