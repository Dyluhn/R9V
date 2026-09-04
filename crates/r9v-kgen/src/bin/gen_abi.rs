// SPDX-License-Identifier: Apache-2.0
//! Generator CLI for R9V kernel ABI argument structs (Spec 4 §7).

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: r9v-gen-abi [--rust <output-path>] [--hip <output-path>]");
        println!(
            "Generates kernel ABI argument struct definitions from input variants (Spec 4 §7)."
        );
        return Ok(());
    }

    let mut rust_out: Option<PathBuf> = None;
    let mut hip_out: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--rust" => {
                if i + 1 < args.len() {
                    rust_out = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Missing path argument for --rust");
                    std::process::exit(1);
                }
            }
            "--hip" => {
                if i + 1 < args.len() {
                    hip_out = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Missing path argument for --hip");
                    std::process::exit(1);
                }
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    let variants: Vec<r9v_kgen::AbiStruct> = Vec::new();

    if let Some(path) = rust_out {
        let code = r9v_kgen::abi::emit_all_rust_module(&variants)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, code)?;
        println!("Wrote Rust ABI structs to {}", path.display());
    }

    if let Some(path) = hip_out {
        let code = r9v_kgen::abi::emit_all_hip_header(&variants)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, code)?;
        println!("Wrote HIP ABI header to {}", path.display());
    }

    Ok(())
}
