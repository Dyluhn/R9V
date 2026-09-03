//! R9V main CLI binary connecting the engine crates (Spec 12 §2, Spec 14 §2).

use std::path::PathBuf;

// DECISION(A0.3): the card names the public command `r9v config gen`, so its
// otherwise self-contained r9v-config implementation needs this minimal CLI
// wiring in the existing r9v binary crate. Parsing stays dependency-free; the
// schema, generation, and validation remain owned solely by r9v-config.
fn run(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let args = args.into_iter().collect::<Vec<_>>();
    match args.as_slice() {
        [command, subcommand] if command == "config" && subcommand == "gen" => {
            generate(PathBuf::from("."))
        }
        [command, subcommand, flag, path]
            if command == "config" && subcommand == "gen" && flag == "--output" =>
        {
            generate(PathBuf::from(path))
        }
        [flag] if flag == "--help" || flag == "-h" => {
            println!("usage: r9v config gen [--output <directory>]");
            Ok(())
        }
        _ => Err("usage: r9v config gen [--output <directory>]".to_string()),
    }
}

fn generate(output: PathBuf) -> Result<(), String> {
    r9v_config::write_generated(&output).map_err(|error| error.to_string())?;
    println!("generated {}", output.join("r9v.toml").display());
    println!("generated {}", output.join("docs/config.md").display());
    println!("generated {}", output.join("r9v.schema.json").display());
    Ok(())
}

fn main() {
    if let Err(error) = run(std::env::args().skip(1)) {
        eprintln!("r9v: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::{generate, run};

    #[test]
    fn help_and_invalid_commands_are_deterministic() {
        assert!(run(["--help".to_string()]).is_ok());
        assert_eq!(
            run(["config".to_string()]).unwrap_err(),
            "usage: r9v config gen [--output <directory>]"
        );
    }

    #[test]
    fn config_gen_writes_all_three_artifacts() {
        let output = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/card-tests/r9v-cli-config-gen");
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        generate(output.clone()).unwrap();
        assert!(output.join("r9v.toml").is_file());
        assert!(output.join("docs/config.md").is_file());
        assert!(output.join("r9v.schema.json").is_file());
        fs::remove_dir_all(output).unwrap();
    }
}
