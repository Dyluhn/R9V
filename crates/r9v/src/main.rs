//! R9V main CLI binary connecting the engine crates (Spec 12 §2, Spec 14 §2).

mod eval;

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
        // DECISION(A1.12): `r9v eval` argument order is fixed as
        // `eval --logits --model <path> --tokens <file> [--out <path>]`;
        // rejected free flag permutation because the card names this exact
        // form and one obvious spelling keeps tests and docs aligned.
        // Spec 14 §10, Card A1.12.
        [command, logits, model_flag, model, tokens_flag, tokens]
            if command == "eval"
                && logits == "--logits"
                && model_flag == "--model"
                && tokens_flag == "--tokens" =>
        {
            eval::eval_logits(&PathBuf::from(model), &PathBuf::from(tokens), None)
        }
        [command, logits, model_flag, model, tokens_flag, tokens, out_flag, out]
            if command == "eval"
                && logits == "--logits"
                && model_flag == "--model"
                && tokens_flag == "--tokens"
                && out_flag == "--out" =>
        {
            eval::eval_logits(
                &PathBuf::from(model),
                &PathBuf::from(tokens),
                Some(&PathBuf::from(out)),
            )
        }
        [flag] if flag == "--help" || flag == "-h" => {
            print_help();
            Ok(())
        }
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage: r9v config gen [--output <directory>]\n       r9v eval --logits --model <path> --tokens <file> [--out <path>]"
        .to_string()
}

fn print_help() {
    println!("{}", usage());
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
        assert_eq!(run(["config".to_string()]).unwrap_err(), super::usage(),);
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

    #[test]
    fn eval_logits_round_trip_matches_direct_prefill() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/card-tests/r9v-cli-eval-logits");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("model.json");
        let tokens_path = dir.join("tokens.txt");
        let out_path = dir.join("logits.npy");
        fs::write(
            &model_path,
            serde_json::to_string(&r9v_t0::synthetic::SyntheticSpec::test_default()).unwrap(),
        )
        .unwrap();
        fs::write(&tokens_path, "1 2 3 4 5").unwrap();
        run([
            "eval".to_string(),
            "--logits".to_string(),
            "--model".to_string(),
            model_path.to_str().unwrap().to_string(),
            "--tokens".to_string(),
            tokens_path.to_str().unwrap().to_string(),
            "--out".to_string(),
            out_path.to_str().unwrap().to_string(),
        ])
        .unwrap();
        let (shape, values) = super::eval::read_npy_f32(&out_path).unwrap();
        assert_eq!(shape, vec![5, 64]);
        assert_eq!(values.len(), 5 * 64);

        // Direct prefill through the same model must be bit-identical.
        let model =
            r9v_t0::synthetic::build(&r9v_t0::synthetic::SyntheticSpec::test_default()).unwrap();
        let mut exec = r9v_t0::exec::CpuExecutor::new();
        let max_blocks = r9v_t0::decode::prepare(&mut exec, &model).unwrap();
        let mut rng = Vec::new();
        let direct =
            r9v_t0::decode::run_step(&mut exec, &model, &[1, 2, 3, 4, 5], 0, max_blocks, &mut rng)
                .unwrap();
        assert_eq!(direct.len(), values.len());
        for (index, (&a, &b)) in direct.iter().zip(values.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "logit {index} differs: direct={a} file={b}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn logits_file_round_trips() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/card-tests/r9v-cli-logits-round-trip");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        let npy_path = dir.join("test_round_trip.npy");
        let shape = vec![3, 8];
        let original_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.125 - 1.5).collect();
        super::eval::write_npy_f32(&npy_path, &shape, &original_data).unwrap();
        let (read_shape, read_data) = super::eval::read_npy_f32(&npy_path).unwrap();
        assert_eq!(read_shape, shape);
        assert_eq!(read_data.len(), original_data.len());
        for (i, (&a, &b)) in original_data.iter().zip(read_data.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i} round-trip mismatch: wrote {a}, read {b}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
