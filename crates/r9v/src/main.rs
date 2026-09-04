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

    #[test]
    fn numpy_emitted_bytes_match_committed_golden_oracle_fixture() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/card-tests/r9v-cli-numpy-golden");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();

        // Cases matching the committed NumPy 2.4.6 golden fixtures under tests/fixtures/r9v/.
        let test_cases: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            (
                "golden_f32_3x8.hex",
                vec![3, 8],
                (0..24).map(|i| (i as f32) * 0.125 - 1.5).collect(),
            ),
            (
                "golden_f32_1d.hex",
                vec![7],
                (0..7).map(|i| (i as f32) * 1.5 - 3.0).collect(),
            ),
        ];

        for (fixture_name, shape, data) in test_cases {
            let golden_bytes = load_fixture_hex(fixture_name);
            let npy_path = dir.join(format!("emitted_{fixture_name}.npy"));
            super::eval::write_npy_f32(&npy_path, &shape, &data).unwrap();

            // 1. Bit-for-bit, byte-for-byte exact equality against NumPy-produced golden oracle.
            let emitted_bytes = fs::read(&npy_path).unwrap();
            assert_eq!(
                emitted_bytes, golden_bytes,
                "emitted .npy bytes differ from NumPy golden oracle for {fixture_name}"
            );

            // 2. In-tree round trip: read_npy_f32 decodes shape and exact float bit patterns.
            let (read_shape, read_data) = super::eval::read_npy_f32(&npy_path).unwrap();
            assert_eq!(read_shape, shape);
            assert_eq!(read_data.len(), data.len());
            for (i, (&a, &b)) in data.iter().zip(read_data.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{fixture_name} element {i} mismatch"
                );
            }

            // 3. Read directly from golden bytes written to disk to prove reader compatibility.
            let direct_golden_path = dir.join(format!("direct_{fixture_name}.npy"));
            fs::write(&direct_golden_path, &golden_bytes).unwrap();
            let (gold_shape, gold_data) = super::eval::read_npy_f32(&direct_golden_path).unwrap();
            assert_eq!(gold_shape, shape);
            assert_eq!(gold_data.len(), data.len());
            for (i, (&a, &b)) in data.iter().zip(gold_data.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{fixture_name} golden decode mismatch at element {i}"
                );
            }
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

    #[test]
    fn untrusted_json_model_arithmetic_overflow_fails_closed_before_allocation() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/card-tests/r9v-cli-untrusted-model");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();

        let tokens_path = dir.join("tokens.txt");
        fs::write(&tokens_path, "1 2 3").unwrap();

        // 1. heads * head_dim overflow in untrusted JSON
        let mut bad_heads = r9v_t0::synthetic::SyntheticSpec::test_default();
        bad_heads.heads = 2147483648;
        bad_heads.kv_heads = 1;
        bad_heads.head_dim = 2;
        let bad_heads_path = dir.join("bad_heads.json");
        fs::write(&bad_heads_path, serde_json::to_string(&bad_heads).unwrap()).unwrap();
        let err = super::eval::eval_logits(&bad_heads_path, &tokens_path, None)
            .unwrap_err()
            .to_lowercase();
        assert!(
            err.contains("arithmetic") && err.contains("overflow") && err.contains("heads"),
            "expected arithmetic overflow on heads * head_dim, got {err}"
        );

        // 2. kv_heads * head_dim overflow in untrusted JSON
        let mut bad_kv = r9v_t0::synthetic::SyntheticSpec::test_default();
        bad_kv.heads = 1;
        bad_kv.kv_heads = 2147483648;
        bad_kv.head_dim = 2;
        let bad_kv_path = dir.join("bad_kv.json");
        fs::write(&bad_kv_path, serde_json::to_string(&bad_kv).unwrap()).unwrap();
        let err = super::eval::eval_logits(&bad_kv_path, &tokens_path, None)
            .unwrap_err()
            .to_lowercase();
        assert!(
            err.contains("arithmetic") && err.contains("overflow") && err.contains("kv_heads"),
            "expected arithmetic overflow on kv_heads * head_dim, got {err}"
        );

        // 3. vocab * dim shape and byte-size product overflow in untrusted JSON
        let mut bad_shape = r9v_t0::synthetic::SyntheticSpec::test_default();
        bad_shape.vocab = u32::MAX;
        bad_shape.dim = u32::MAX;
        let bad_shape_path = dir.join("bad_shape.json");
        fs::write(&bad_shape_path, serde_json::to_string(&bad_shape).unwrap()).unwrap();
        let err = super::eval::eval_logits(&bad_shape_path, &tokens_path, None)
            .unwrap_err()
            .to_lowercase();
        assert!(
            err.contains("arithmetic")
                && err.contains("overflow")
                && err.contains("overflows usize"),
            "expected arithmetic overflow on shape/byte-size products, got {err}"
        );

        fs::remove_dir_all(dir).unwrap();
    }

    /// Loads and decodes a hex-encoded fixture from `crates/r9v/tests/fixtures/r9v/`.
    fn load_fixture_hex(filename: &str) -> Vec<u8> {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/r9v")
            .join(filename);
        let text = fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("cannot read fixture {}: {err}", fixture_path.display()));
        decode_hex(text.trim())
    }

    /// Decodes an ASCII hex string into raw bytes.
    fn decode_hex(hex: &str) -> Vec<u8> {
        let bytes = hex.as_bytes();
        assert_eq!(bytes.len() % 2, 0, "hex length must be even");
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            let hi = (chunk[0] as char).to_digit(16).expect("valid hex digit");
            let lo = (chunk[1] as char).to_digit(16).expect("valid hex digit");
            out.push(((hi << 4) | lo) as u8);
        }
        out
    }
}
