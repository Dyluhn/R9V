// SPDX-License-Identifier: Apache-2.0
//! Pinned ROCm toolchain compilation and layout equality test (Spec 4 §7; card A3.2).
//!
//! Compiles generated HIP structs using the pinned ROCm Clang compiler (ROCm 7.14 / Clang 23)
//! with pinned flags (`-x hip --offload-arch=gfx1201 -O3 -fno-fast-math -fno-gpu-approx-transcendentals`),
//! dumps `sizeof`, `alignof`, and `offsetof` for every field of all 32 closed-set operations,
//! and asserts exact bit-level equality against the Rust `#[repr(C)]` argument layouts.
//!
//! DECISION(A3.2): In tests/pinned_compile_layout.rs, detect the pinned ROCm toolchain either
//! via host ROCm installation (ROCM_PATH, /opt/rocm/llvm/bin/clang++, /opt/rocm/bin/clang++, PATH)
//! or via containerized r9v-ci:test docker image, panicking loudly on missing toolchain when
//! R9V_REQUIRE_ROCM_COMPILE=1 is set, while reporting an explicit honest skip on CPU-only hosted
//! environments when the env var is not set. Rejected silent skip when compiler environment is
//! requested because Spec 4 §7 and Spec 14 §1 require verifiable compiler qualification.
//! Every candidate must report the pinned ROCm 7.14 / Clang 23 signature via --version
//! (AMD clang major 23 plus the 7.14 release marker); any other working compiler is
//! rejected, and the Docker image is verified inside the image before compiling.

mod common;

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use r9v_kgen::abi::{abi_for_op, emit_all_hip_header, emit_all_rust_module, AbiStruct};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct LayoutDump {
    structs: Vec<StructDump>,
}

#[derive(Debug, Deserialize)]
struct StructDump {
    name: String,
    size: usize,
    align: usize,
    fields: Vec<FieldDump>,
}

#[derive(Debug, Deserialize)]
struct FieldDump {
    name: String,
    offset: usize,
    size: usize,
}

enum Toolchain {
    HostClang {
        clang_path: PathBuf,
        ld_path: Option<PathBuf>,
    },
    Docker {
        image: String,
    },
}

/// Pinned toolchain signature (toolchain.toml `7.14.0`, ci/Dockerfile
/// `rocm/dev-ubuntu-24.04:7.14.0-full`, spikes record AMD clang 23.0.0git).
/// ROCm 7.14's `clang++ --version` prints `AMD clang version 23.0.0git (...)`
/// with the release in the install path (`InstalledDir:
/// /opt/rocm/core-7.14/lib/llvm/bin`); both the LLVM major and the 7.14 release
/// marker are required here. Verified against the `r9v-ci:test` image, whose
/// build string carries no `roc-` tag, so the check keys on `7.14`, not it.
const PINNED_CLANG_MAJOR: u32 = 23;
const PINNED_ROCM_MARKER: &str = "7.14";

/// Extracts the clang major version from `clang++ --version` output.
/// Matches `AMD clang version <major>.` (ROCm LLVM) or plain
/// `clang version <major>.`; anything else is `None`.
fn clang_major_from_version_output(output: &str) -> Option<u32> {
    let marker = "clang version ";
    let at = output.find(marker)?;
    let rest = output[at + marker.len()..].trim_start();
    let major: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if major.is_empty() {
        return None;
    }
    major.parse::<u32>().ok()
}

/// Returns true iff `clang++ --version` output is exactly the pinned
/// ROCm 7.14 / Clang 23 toolchain: an AMD (ROCm LLVM) clang, LLVM major
/// 23, carrying the `7.14` release marker. Wrong majors, upstream
/// clang/gcc output, and malformed text all return false.
fn is_pinned_toolchain_version_output(output: &str) -> bool {
    output.contains("AMD clang")
        && output.contains(PINNED_ROCM_MARKER)
        && clang_major_from_version_output(output) == Some(PINNED_CLANG_MAJOR)
}

fn pinned_host_clang(candidate: &Path) -> Option<Toolchain> {
    let out = Command::new(candidate).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    if !is_pinned_toolchain_version_output(&text) {
        eprintln!(
            "REJECTED: {} is not the pinned ROCm 7.14 / Clang 23 toolchain; ignoring.",
            candidate.display()
        );
        return None;
    }
    let ld_path = candidate
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("lib"));
    Some(Toolchain::HostClang {
        clang_path: candidate.to_path_buf(),
        ld_path,
    })
}

/// Runs `clang++ --version` inside the image and accepts it only when the
/// output carries the pinned ROCm 7.14 / Clang 23 signature.
fn pinned_docker_clang(image: &str) -> Option<Toolchain> {
    for clang in ["/opt/rocm/llvm/bin/clang++", "/opt/rocm/bin/clang++"] {
        let out = Command::new("docker")
            .args(["run", "--rm", image, clang, "--version"])
            .output()
            .ok()?;
        if !out.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        if is_pinned_toolchain_version_output(&text) {
            return Some(Toolchain::Docker {
                image: image.to_string(),
            });
        }
        eprintln!("REJECTED: {image}:{clang} is not the pinned ROCm 7.14 / Clang 23 toolchain.");
        return None;
    }
    None
}

fn discover_toolchain() -> Option<Toolchain> {
    // 1. Check host ROCM_PATH or standard locations; each candidate must
    // report the pinned ROCm 7.14 / Clang 23 signature via --version.
    let mut candidates = Vec::new();
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        let p = PathBuf::from(rocm_path);
        candidates.push(p.join("llvm/bin/clang++"));
        candidates.push(p.join("bin/clang++"));
    }
    candidates.push(PathBuf::from("/opt/rocm/llvm/bin/clang++"));
    candidates.push(PathBuf::from("/opt/rocm/bin/clang++"));

    for candidate in candidates {
        if !candidate.is_file() {
            continue;
        }
        if let Some(t) = pinned_host_clang(&candidate) {
            return Some(t);
        }
    }

    // 2. Check docker image r9v-ci:test, verifying the pinned signature
    // inside the image before accepting it for compilation.
    let docker_check = Command::new("docker")
        .args(["image", "inspect", "r9v-ci:test"])
        .output();
    if let Ok(o) = docker_check {
        if o.status.success() {
            if let Some(t) = pinned_docker_clang("r9v-ci:test") {
                return Some(t);
            }
        }
    }

    None
}

fn generate_cpp_dump_source(abis: &[AbiStruct], include_path: &Path) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "#include <stdio.h>");
    let _ = writeln!(out, "#include <stddef.h>");
    let _ = writeln!(out, "#include <stdint.h>");
    let _ = writeln!(out, "#include \"{}\"", include_path.display());
    let _ = writeln!(out);
    let _ = writeln!(out, "int main() {{");
    let _ = writeln!(out, "    printf(\"{{\\n  \\\"structs\\\": [\\n\");");

    for (s_idx, abi) in abis.iter().enumerate() {
        let is_last_struct = s_idx == abis.len() - 1;
        let _ = writeln!(out, "    printf(\"    {{\\n\");");
        let _ = writeln!(
            out,
            "    printf(\"      \\\"name\\\": \\\"{}\\\",\\n\");",
            abi.name
        );
        let _ = writeln!(
            out,
            "    printf(\"      \\\"size\\\": %zu,\\n\", sizeof({}));",
            abi.name
        );
        let _ = writeln!(
            out,
            "    printf(\"      \\\"align\\\": %zu,\\n\", alignof({}));",
            abi.name
        );
        let _ = writeln!(out, "    printf(\"      \\\"fields\\\": [\\n\");");

        for (f_idx, field) in abi.fields.iter().enumerate() {
            let is_last_field = f_idx == abi.fields.len() - 1;
            let comma = if is_last_field { "" } else { "," };
            let _ = writeln!(
                out,
                "        printf(\"        {{\\\"name\\\": \\\"{}\\\", \\\"offset\\\": %zu, \\\"size\\\": %zu}}{}\\n\", offsetof({}, {}), sizeof((({}*)0)->{}));",
                field.name, comma, abi.name, field.name, abi.name, field.name
            );
        }

        let struct_comma = if is_last_struct { "" } else { "," };
        let _ = writeln!(out, "    printf(\"      ]\\n    }}{}\\n\");", struct_comma);
    }

    let _ = writeln!(out, "    printf(\"  ]\\n}}\\n\");");
    let _ = writeln!(out, "    return 0;");
    let _ = writeln!(out, "}}");
    out
}

fn generate_rust_dump_source(abis: &[AbiStruct]) -> Result<String, r9v_kgen::error::KgenError> {
    let mut out = String::new();
    let rust_module = emit_all_rust_module(abis)?;
    out.push_str(&rust_module);
    out.push_str(r#"

fn print_open() {
    println!("{{\n  \"structs\": [");
}

fn print_struct_start(name: &str, size: usize, align: usize) {
    println!("    {{\n      \"name\": \"{name}\",\n      \"size\": {size},\n      \"align\": {align},\n      \"fields\": [");
}

fn print_field(name: &str, offset: usize, size: usize, comma: &str) {
    println!("        {{\"name\": \"{name}\", \"offset\": {offset}, \"size\": {size}}}{comma}");
}

fn print_struct_end(comma: &str) {
    println!("      ]\n    }}{comma}");
}

fn print_close() {
    println!("  ]\n}}");
}

fn main() {
    print_open();
"#);

    for (s_idx, abi) in abis.iter().enumerate() {
        let is_last_struct = s_idx == abis.len() - 1;
        let _ = writeln!(
            out,
            "    print_struct_start(\"{}\", std::mem::size_of::<{}>(), std::mem::align_of::<{}>());",
            abi.name, abi.name, abi.name
        );

        for (f_idx, field) in abi.fields.iter().enumerate() {
            let is_last_field = f_idx == abi.fields.len() - 1;
            let comma = if is_last_field { "" } else { "," };
            let _ = writeln!(
                out,
                "    print_field(\"{}\", std::mem::offset_of!({}, {}), {}, \"{}\");",
                field.name,
                abi.name,
                field.name,
                field.size(),
                comma
            );
        }

        let struct_comma = if is_last_struct { "" } else { "," };
        let _ = writeln!(out, "    print_struct_end(\"{}\");", struct_comma);
    }

    out.push_str("    print_close();\n}\n");
    Ok(out)
}

/// Collects the 32 closed-set ABI structs plus the tree verify and scatter
/// present-form variant shapes.
fn collect_abis() -> Vec<AbiStruct> {
    let mut abis = Vec::new();
    for op in common::ALL_32_OPS {
        let st = common::representative_static_for_op(op);
        abis.push(abi_for_op(op, &st).expect("representative ABI must construct"));
    }
    // The tree verify shape differs from flat verify; qualify its layout too.
    let tree_st = common::representative_verify_static(true);
    abis.push(
        abi_for_op(r9v_registry::OpId::Verify, &tree_st).expect("tree verify ABI must construct"),
    );
    // The scatter present (has_dest) form carries the same dest field with a
    // non-null launch contract; qualify its layout too so generated code is
    // proven to launch both forms.
    let mut scatter_st =
        match common::representative_static_for_op(r9v_registry::OpId::ScatterAddRows) {
            r9v_registry::OpStatic::Elementwise(s) => s,
            _ => panic!("expected elementwise"),
        };
    let mut params = match scatter_st.op_params.clone() {
        r9v_registry::ElementwiseParams::ScatterAddRows(p) => p,
        _ => panic!("expected scatter_add_rows"),
    };
    params.has_dest = true;
    scatter_st.op_params = r9v_registry::ElementwiseParams::ScatterAddRows(params);
    abis.push(
        abi_for_op(
            r9v_registry::OpId::ScatterAddRows,
            &r9v_registry::OpStatic::Elementwise(scatter_st),
        )
        .expect("scatter present-form ABI must construct"),
    );
    abis
}

fn scratch_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let dir = workspace_root.join("target/abi_test");
    fs::create_dir_all(&dir).expect("failed to create scratch directory");
    dir
}

/// Compiles the emitted Rust `#[repr(C)]` layout dump with rustc, runs it, and parses the JSON.
fn rust_layout_dump(abis: &[AbiStruct], scratch_dir: &Path) -> LayoutDump {
    let rust_source = generate_rust_dump_source(abis).expect("Rust dump generation");
    let rust_file = scratch_dir.join("dump_rust_layout.rs");
    fs::write(&rust_file, &rust_source).expect("failed to write dump_rust_layout.rs");

    let rust_bin = scratch_dir.join("dump_rust_layout");
    let rustc_status = Command::new("rustc")
        .args([
            "-O",
            rust_file.to_str().unwrap(),
            "-o",
            rust_bin.to_str().unwrap(),
        ])
        .status()
        .expect("failed to execute rustc");

    assert!(rustc_status.success(), "rustc layout compilation failed");

    let rust_out = Command::new(&rust_bin)
        .output()
        .expect("failed to run rust layout dump binary");
    assert!(
        rust_out.status.success(),
        "rust layout dump binary failed: {}",
        String::from_utf8_lossy(&rust_out.stderr)
    );

    let rust_stdout = String::from_utf8(rust_out.stdout).expect("valid utf8 rust dump");
    let rust_dump: LayoutDump =
        serde_json::from_str(&rust_stdout).expect("Rust dump binary must output valid JSON");

    assert_eq!(
        rust_dump.structs.len(),
        34,
        "Rust dump must contain all 32 closed-set operations plus tree verify and scatter present-form"
    );
    rust_dump
}

/// Asserts exact layout equality between the AbiStruct oracle and one compiler dump.
fn assert_oracle_matches_dump(abis: &[AbiStruct], dump: &LayoutDump, tag: &str) {
    // 32 closed-set operations plus the tree verify and scatter present-form
    // variants (34 struct shapes).
    assert_eq!(
        dump.structs.len(),
        34,
        "{tag} dump must contain all 32 closed-set operations plus tree verify and scatter present-form"
    );
    let structs_by_name: HashMap<&str, &StructDump> =
        dump.structs.iter().map(|s| (s.name.as_str(), s)).collect();
    for abi in abis {
        let dumped = structs_by_name
            .get(abi.name.as_str())
            .unwrap_or_else(|| panic!("missing struct '{}' in {tag} dump", abi.name));

        assert_eq!(
            abi.size(),
            dumped.size,
            "Size mismatch for struct '{}': oracle = {}, {tag} = {}",
            abi.name,
            abi.size(),
            dumped.size
        );
        assert_eq!(
            abi.alignment(),
            dumped.align,
            "Alignment mismatch for struct '{}': oracle = {}, {tag} = {}",
            abi.name,
            abi.alignment(),
            dumped.align
        );
        assert_eq!(
            abi.fields().len(),
            dumped.fields.len(),
            "Field count mismatch for struct '{}': oracle = {}, {tag} = {}",
            abi.name,
            abi.fields().len(),
            dumped.fields.len()
        );
        for (idx, field) in abi.fields().iter().enumerate() {
            let dumped_field = &dumped.fields[idx];
            assert_eq!(
                field.name, dumped_field.name,
                "Field name mismatch in struct '{}' at index {}: oracle = {}, {tag} = {}",
                abi.name, idx, field.name, dumped_field.name
            );
            assert_eq!(
                field.offset(),
                dumped_field.offset,
                "Field offset mismatch in struct '{}' for '{}': oracle = {}, {tag} = {}",
                abi.name,
                field.name,
                field.offset(),
                dumped_field.offset
            );
            assert_eq!(
                field.size(),
                dumped_field.size,
                "Field size mismatch in struct '{}' for '{}': oracle = {}, {tag} = {}",
                abi.name,
                field.name,
                field.size(),
                dumped_field.size
            );
        }
    }
}

/// Rust-side oracle/layout checks: always run, never skipped (Spec 4 §7).
#[test]
fn test_rust_oracle_layout_equality() {
    let abis = collect_abis();
    let dir = scratch_dir();
    let rust_dump = rust_layout_dump(&abis, &dir);
    assert_oracle_matches_dump(&abis, &rust_dump, "Rust");
}

#[test]
fn test_pinned_rocm_compile_and_layout_equality() {
    let require_compile = std::env::var("R9V_REQUIRE_ROCM_COMPILE").as_deref() == Ok("1");

    let toolchain = match discover_toolchain() {
        Some(t) => t,
        None => {
            if require_compile {
                panic!(
                    "FATAL: Pinned ROCm toolchain (ROCm 7.14 / Clang 23) is required by \
                     R9V_REQUIRE_ROCM_COMPILE=1, but no candidate reported the pinned \
                     signature: neither a native clang++ with `AMD clang version 23.*` \
                     plus the 7.14 release marker nor container image `r9v-ci:test` with \
                     that signature inside the image was found (wrong-version compilers \
                     are rejected, not used)!"
                );
            } else {
                eprintln!(
                    "SKIPPED: Pinned ROCm toolchain not available on host. Set \
                     R9V_REQUIRE_ROCM_COMPILE=1 to require it."
                );
                return;
            }
        }
    };

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    let scratch = scratch_dir();

    let abis = collect_abis();

    // 1. Emit and compile HIP header
    let hip_header = emit_all_hip_header(&abis).expect("HIP header emission");
    let header_path = scratch.join("r9v_abi.h");
    fs::write(&header_path, &hip_header).expect("failed to write r9v_abi.h");

    let cpp_source = generate_cpp_dump_source(&abis, &header_path);
    let cpp_file = scratch.join("dump_layout.cpp");
    fs::write(&cpp_file, &cpp_source).expect("failed to write dump_layout.cpp");

    let bin_file = scratch.join("dump_layout");

    let hip_stdout = match toolchain {
        Toolchain::HostClang {
            clang_path,
            ld_path,
        } => {
            let compile_status = Command::new(&clang_path)
                .args([
                    "-x",
                    "hip",
                    "--offload-arch=gfx1201",
                    "-O3",
                    "-fno-fast-math",
                    "-fno-gpu-approx-transcendentals",
                    cpp_file.to_str().unwrap(),
                    "-o",
                    bin_file.to_str().unwrap(),
                ])
                .status()
                .expect("failed to execute native clang++");

            assert!(
                compile_status.success(),
                "native HIP clang++ compilation failed"
            );

            let mut run_cmd = Command::new(&bin_file);
            if let Some(ld) = ld_path {
                run_cmd.env("LD_LIBRARY_PATH", ld);
            }
            let run_out = run_cmd.output().expect("failed to run layout dump binary");
            assert!(
                run_out.status.success(),
                "layout dump binary failed with stderr: {}",
                String::from_utf8_lossy(&run_out.stderr)
            );
            String::from_utf8(run_out.stdout).expect("valid utf8 dump")
        }
        Toolchain::Docker { image } => {
            let ws_str = workspace_root.to_str().unwrap();
            let docker_cmd = format!(
                "/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1201 -O3 -fno-fast-math -fno-gpu-approx-transcendentals \
                 {ws_str}/target/abi_test/dump_layout.cpp -o {ws_str}/target/abi_test/dump_layout && \
                 LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib {ws_str}/target/abi_test/dump_layout"
            );

            let out = Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "-v",
                    &format!("{ws_str}:{ws_str}"),
                    "-w",
                    ws_str,
                    &image,
                    "/bin/bash",
                    "-c",
                    &docker_cmd,
                ])
                .output()
                .expect("failed to execute docker run");

            if !out.status.success() {
                panic!(
                    "docker layout compilation/execution failed:\nstdout: {}\nstderr: {}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            String::from_utf8(out.stdout).expect("valid utf8 dump")
        }
    };

    let hip_dump: LayoutDump =
        serde_json::from_str(&hip_stdout).expect("HIP dump binary must output valid JSON");

    // Oracle vs HIP compiler equality (oracle vs Rust is covered without skipping
    // by test_rust_oracle_layout_equality above).
    assert_oracle_matches_dump(&abis, &hip_dump, "HIP");
}

#[cfg(test)]
mod pinned_signature_tests {
    use super::*;

    // Verbatim `clang++ --version` output observed from the pinned
    // `r9v-ci:test` image (ROCm 7.14.0, AMD clang 23.0.0git).
    const PINNED_SAMPLE: &str = "AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)\nTarget: x86_64-unknown-linux-gnu\nThread model: posix\nInstalledDir: /opt/rocm/core-7.14/lib/llvm/bin";

    #[test]
    fn accepts_pinned_rocm_714_clang_23() {
        assert!(is_pinned_toolchain_version_output(PINNED_SAMPLE));
        assert_eq!(clang_major_from_version_output(PINNED_SAMPLE), Some(23));
    }

    #[test]
    fn rejects_wrong_clang_major() {
        // Previous-minor matrix compiler (toolchain.toml matrix 7.13.0 line):
        // right vendor, wrong LLVM major.
        let other = PINNED_SAMPLE.replace("version 23.", "version 22.");
        assert_eq!(clang_major_from_version_output(&other), Some(22));
        assert!(!is_pinned_toolchain_version_output(&other));
    }

    #[test]
    fn rejects_wrong_rocm_release() {
        // Right LLVM major, previous ROCm release path.
        let other = PINNED_SAMPLE.replace("core-7.14", "core-7.13");
        assert_eq!(clang_major_from_version_output(&other), Some(23));
        assert!(!is_pinned_toolchain_version_output(&other));
    }

    #[test]
    fn rejects_upstream_clang_and_gcc() {
        let upstream = "clang version 19.1.0 (https://github.com/llvm/llvm-project ...)\nTarget: x86_64-unknown-linux-gnu";
        assert_eq!(clang_major_from_version_output(upstream), Some(19));
        assert!(!is_pinned_toolchain_version_output(upstream));
        let gcc = "g++ (GCC) 13.2.0\nCopyright (C) 2023 Free Software Foundation, Inc.";
        assert_eq!(clang_major_from_version_output(gcc), None);
        assert!(!is_pinned_toolchain_version_output(gcc));
    }

    #[test]
    fn rejects_malformed_version_text() {
        for bad in [
            "",
            "not a version string",
            "AMD clang version ",
            "AMD clang version git (https://example.com roc-7.14.0)",
            "AMD clang version 23",
        ] {
            assert!(
                !is_pinned_toolchain_version_output(bad),
                "malformed output must not verify: {bad:?}"
            );
        }
        // Major without the dot still parses (defensive), but the missing
        // ROCm tag alone must fail the signature.
        assert_eq!(
            clang_major_from_version_output("AMD clang version 23"),
            Some(23)
        );
    }

    #[test]
    fn rejects_amd_clang_without_release_marker() {
        // An AMD clang that does not advertise the pinned release (e.g. a
        // dev snapshot build string) must not verify.
        let no_marker = "AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 26001 abcdef)\nTarget: x86_64-unknown-linux-gnu\nThread model: posix\nInstalledDir: /opt/rocm/llvm/bin";
        assert!(!is_pinned_toolchain_version_output(no_marker));
    }
}
