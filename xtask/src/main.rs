//! cargo xtask commands (Spec 14 §2, §4).

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(subcmd) = args.next() else {
        eprintln!("Usage: cargo xtask <command> [options]");
        eprintln!("Commands:");
        eprintln!("  gen     Regenerate kernels/gen from r9v-kgen and tune/ (Spec 14 §4)");
        eprintln!(
            "  docs    Build local documentation including generated pages (Spec 14 §4, §10)"
        );
        return ExitCode::FAILURE;
    };

    match subcmd.as_str() {
        "gen" => {
            // Stub for xtask gen (Spec 14 §4)
            println!("xtask gen: stub (kernels/gen up to date)");
            ExitCode::SUCCESS
        }
        "docs" => {
            // Stub for xtask docs (Spec 14 §4, §10)
            println!("xtask docs: stub (docs build)");
            ExitCode::SUCCESS
        }
        "--help" | "-h" => {
            println!("Usage: cargo xtask <command> [options]");
            println!("Commands:");
            println!("  gen     Regenerate kernels/gen from r9v-kgen and tune/ (Spec 14 §4)");
            println!(
                "  docs    Build local documentation including generated pages (Spec 14 §4, §10)"
            );
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("Unknown xtask command: {other}");
            eprintln!("Available commands: gen, docs");
            ExitCode::FAILURE
        }
    }
}
