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
        // Independent validation with real Python NumPy (Spec 14 §10, Card A1.12).
        validate_with_python_numpy(&out_path, &[5, 64], &direct);
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
        validate_with_python_numpy(&npy_path, &shape, &original_data);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn numpy_interoperability_validates_with_python_numpy() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/card-tests/r9v-cli-numpy-interop");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();

        // Validate multiple shapes: 1D (checks trailing comma), 2D, and 3D.
        let cases: Vec<(Vec<usize>, Vec<f32>)> = vec![
            (vec![7], (0..7).map(|i| (i as f32) * 1.5 - 3.0).collect()),
            (
                vec![3, 8],
                (0..24).map(|i| (i as f32) * 0.125 - 1.5).collect(),
            ),
            (
                vec![2, 3, 4],
                (0..24).map(|i| ((i as f32) + 0.5) / 7.0).collect(),
            ),
        ];

        for (case_idx, (shape, original_data)) in cases.into_iter().enumerate() {
            let npy_path = dir.join(format!("test_interop_{case_idx}.npy"));
            super::eval::write_npy_f32(&npy_path, &shape, &original_data).unwrap();

            // Validate with in-tree reader.
            let (read_shape, read_data) = super::eval::read_npy_f32(&npy_path).unwrap();
            assert_eq!(read_shape, shape);
            assert_eq!(read_data.len(), original_data.len());
            for (i, (&a, &b)) in original_data.iter().zip(read_data.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "case {case_idx} element {i} mismatch"
                );
            }

            // Validate with real Python NumPy.
            validate_with_python_numpy(&npy_path, &shape, &original_data);
        }

        // Validate checked arithmetic on invalid shape/data length.
        let invalid_path = dir.join("invalid.npy");
        let err = super::eval::write_npy_f32(&invalid_path, &[2, 3], &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(err.contains("data length 3 != shape [2, 3] (6)"));

        // Validate checked multiplication on overflowing shape.
        let overflow_err =
            super::eval::write_npy_f32(&invalid_path, &[usize::MAX, 2], &[]).unwrap_err();
        assert!(overflow_err.contains("overflows usize"));

        fs::remove_dir_all(dir).unwrap();
    }

    /// Validates an emitted .npy file with real Python NumPy via subprocess.
    fn validate_with_python_numpy(
        path: &std::path::Path,
        expected_shape: &[usize],
        expected_data: &[f32],
    ) {
        use std::io::Write;
        let py = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
        let shape_tuple = if expected_shape.len() == 1 {
            format!("({},)", expected_shape[0])
        } else {
            let dims = expected_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims})")
        };
        let py_script = format!(
            r#"
import sys, numpy as np
arr = np.load(r"{}")
assert arr.dtype == np.float32, f"unexpected dtype: {{arr.dtype}}"
assert arr.flags.c_contiguous, "array is not C-contiguous"
assert tuple(arr.shape) == {shape_tuple}, f"shape mismatch: {{arr.shape}} vs {shape_tuple}"
expected_bytes = sys.stdin.buffer.read()
assert arr.tobytes() == expected_bytes, "array byte payload mismatch"
"#,
            path.display(),
        );
        let mut child = std::process::Command::new(&py)
            .arg("-c")
            .arg(&py_script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap_or_else(|err| panic!("failed to spawn python interpreter ({py}): {err}"));
        let expected_bytes: Vec<u8> = expected_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        {
            let mut stdin = child.stdin.take().expect("child stdin open");
            stdin.write_all(&expected_bytes).unwrap();
        }
        let output = child
            .wait_with_output()
            .expect("failed waiting for python process");
        assert!(
            output.status.success(),
            "python numpy validation failed for {}: stdout={}\nstderr={}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
