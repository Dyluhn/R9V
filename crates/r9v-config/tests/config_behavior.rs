//! End-to-end configuration machinery tests for card A0.3 (Spec 12 §2, §4–6).

use std::fs;
use std::path::PathBuf;

use r9v_config::{
    check_settings_index, generate_artifacts, render_effective_toml, write_generated, Auto,
    ConfigError, EffectiveConfig, Source,
};
use toml::Value;

fn scratch(name: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/card-tests/r9v-config")
        .join(name);
    if path.exists() {
        fs::remove_dir_all(&path).expect("remove stale test-owned directory");
    }
    fs::create_dir_all(&path).expect("create test-owned directory");
    path
}

#[test]
fn auto_parses_displays_and_resolves() {
    let automatic: Auto<u32> = "auto".parse().unwrap();
    let concrete: Auto<u32> = "12".parse().unwrap();
    assert!(automatic.is_auto());
    assert_eq!(automatic.to_string(), "auto");
    assert_eq!(concrete.as_value(), Some(&12));
    assert_eq!(concrete.resolve(|| 99, |value| *value), 12);

    let mut config = EffectiveConfig::from_defaults();
    config
        .resolve_auto("scheduler.step_budget_ms", Value::Float(4.5))
        .unwrap();
    let setting = config.get("scheduler.step_budget_ms").unwrap();
    assert_eq!(setting.value.as_str(), Some("auto"));
    assert_eq!(
        setting.resolved.as_ref().and_then(Value::as_float),
        Some(4.5)
    );
    assert!(setting
        .auto_rule
        .unwrap()
        .contains("measured single-sequence"));
}

#[test]
fn precedence_sources_and_extensions_round_trip() {
    let dir = scratch("precedence");
    let model = dir.join("model.gguf");
    fs::write(&model, b"fixture").unwrap();
    let input = format!(
        "config_version = 1\n\n[load]\nmodel = {:?}\n\n[io]\nqueue_depth = 9\n\n[scheduler]\nmax_wait_ms = 400\n\n[x-tool]\nnote = \"preserve me\"\n",
        model.display().to_string()
    );

    let mut config = EffectiveConfig::from_defaults();
    config.apply_file_str("fixture.toml", &input).unwrap();
    assert!(matches!(
        &config.get("load.model").unwrap().source,
        Source::File { path, line: 4 } if path == "fixture.toml"
    ));
    config
        .apply_env([
            ("R9V__IO__QUEUE_DEPTH", "10"),
            ("R9V__SCHEDULER__MAX_WAIT_MS", "350"),
        ])
        .unwrap();
    config
        .apply_cli([
            ("--io.queue_depth", "11"),
            ("--scheduler.max_wait_ms", "325"),
        ])
        .unwrap();
    // A late lower-precedence layer cannot overwrite CLI.
    config.apply_env([("R9V__IO__QUEUE_DEPTH", "12")]).unwrap();
    config
        .apply_runtime(
            [("scheduler.max_wait_ms", "300")],
            "helper-approved",
            "2026-09-03T12:00:00Z",
        )
        .unwrap();

    assert_eq!(
        config.get("io.queue_depth").unwrap().value.as_integer(),
        Some(11)
    );
    assert!(matches!(
        config.get("io.queue_depth").unwrap().source,
        Source::Cli { .. }
    ));
    assert_eq!(
        config
            .get("scheduler.max_wait_ms")
            .unwrap()
            .source
            .to_string(),
        "runtime:helper-approved:2026-09-03T12:00:00Z"
    );
    assert_eq!(
        config.extensions()["x-tool"]
            .get("note")
            .and_then(Value::as_str),
        Some("preserve me")
    );

    let rendered = render_effective_toml(&config, true);
    let mut reparsed = EffectiveConfig::from_defaults();
    reparsed
        .apply_file_str("roundtrip.toml", &rendered)
        .unwrap();
    for (key, value) in config.iter() {
        assert_eq!(reparsed.get(key).unwrap().value, value.value, "{key}");
    }
    assert_eq!(reparsed.extensions(), config.extensions());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unknown_key_reports_exact_nearest_key() {
    let mut config = EffectiveConfig::from_defaults();
    let error = config
        .apply_file_str(
            "typo.toml",
            "config_version = 1\n[scheduler]\nmax_wiat_ms = 3\n",
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "unknown setting `scheduler.max_wiat_ms`; did you mean `scheduler.max_wait_ms`?"
    );
}

#[test]
fn type_range_enum_and_path_validation_fail_closed() {
    let mut config = EffectiveConfig::from_defaults();
    assert!(matches!(
        config.apply_cli([("--io.queue_depth", "zero")]),
        Err(ConfigError::Type { .. })
    ));
    assert!(matches!(
        config.apply_cli([("--io.queue_depth", "0")]),
        Err(ConfigError::Range { .. })
    ));
    assert!(matches!(
        config.apply_cli([("--graph.mode", "maybe")]),
        Err(ConfigError::Enum { .. })
    ));
    assert!(matches!(
        config.apply_cli([("--load.model", "/definitely/missing/r9v-model")]),
        Err(ConfigError::MissingPath { .. })
    ));
    assert!(matches!(
        config.apply_cli([("--state.reserve_bytes", "many")]),
        Err(ConfigError::Type { .. })
    ));
    assert!(matches!(
        config.apply_cli([("--warmup.buckets", "{ S = [1], T_dec = [], T_pre = [0] }")]),
        Err(ConfigError::Type { .. })
    ));
    assert!(matches!(
        config.apply_cli([("--warmup.buckets", "{ S = [0], T_dec = [1], T_pre = [0] }")]),
        Err(ConfigError::Type { .. })
    ));
}

#[test]
fn one_layer_reports_every_independent_validation_problem() {
    let mut config = EffectiveConfig::from_defaults();
    let before = config.clone();
    let error = config
        .apply_cli([
            ("--io.queue_depth", "0"),
            ("--graph.mode", "maybe"),
            ("--state.reserve_bytes", "many"),
        ])
        .unwrap_err();
    let ConfigError::Multiple { summary, problems } = error else {
        panic!("expected collected validation problems");
    };
    assert!(summary.contains("io.queue_depth"));
    assert!(summary.contains("graph.mode"));
    assert!(summary.contains("state.reserve_bytes"));
    assert_eq!(problems.len(), 3);
    assert!(matches!(problems[0], ConfigError::Range { .. }));
    assert!(matches!(problems[1], ConfigError::Enum { .. }));
    assert!(matches!(problems[2], ConfigError::Type { .. }));
    assert_eq!(config, before, "a rejected layer must be atomic");
}

#[test]
fn runtime_mutability_and_cross_field_rules_are_atomic() {
    let mut config = EffectiveConfig::from_defaults();
    assert!(matches!(
        config.apply_runtime([("state.max_ctx", "65536")], "test", "now"),
        Err(ConfigError::Mutability { .. })
    ));
    let before = config.clone();
    let error = config
        .apply_cli([
            ("--scheduler.prefill_min_chunk", "3000"),
            ("--scheduler.prefill_max_chunk", "2500"),
            ("--spec.k_max", "15"),
            ("--spec.tree_max", "10"),
            ("--state.max_ctx", "33"),
        ])
        .unwrap_err();
    let ConfigError::CrossField { messages } = error else {
        panic!("expected all cross-field failures");
    };
    assert_eq!(messages.len(), 3, "{messages:?}");
    assert!(messages
        .iter()
        .all(|message| message.contains("schema docs")));
    assert_eq!(config, before, "a rejected layer must not partially apply");
}

#[test]
fn generated_artifacts_are_deterministic_valid_and_writable() {
    let first = generate_artifacts();
    let second = generate_artifacts();
    assert_eq!(first, second);
    assert!(first
        .r9v_toml
        .starts_with("# Generated from r9v-config schema"));
    let mut parsed = EffectiveConfig::from_defaults();
    parsed
        .apply_file_str("generated.toml", &first.r9v_toml)
        .unwrap();
    let schema: serde_json::Value = serde_json::from_str(&first.json_schema).unwrap();
    assert_eq!(schema["properties"]["config_version"]["const"], 1);
    assert_eq!(
        schema["properties"]["scheduler"]["properties"]["max_wait_ms"]["type"],
        "integer"
    );
    assert_eq!(
        schema["properties"]["scheduler"]["properties"]["max_wait_ms"]["default"],
        500
    );
    assert_eq!(
        schema["properties"]["warmup"]["properties"]["buckets"]["type"],
        "object"
    );
    assert_eq!(
        schema["properties"]["warmup"]["properties"]["buckets"]["default"]["T_pre"][3],
        2048
    );
    for (key, _) in parsed.iter() {
        assert!(first.config_markdown.contains(&format!("`{key}`")), "{key}");
    }

    let dir = scratch("generate");
    let written = write_generated(&dir).unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("r9v.toml")).unwrap(),
        written.r9v_toml
    );
    assert_eq!(
        fs::read_to_string(dir.join("docs/config.md")).unwrap(),
        written.config_markdown
    );
    assert_eq!(
        fs::read_to_string(dir.join("r9v.schema.json")).unwrap(),
        written.json_schema
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn committed_artifacts_match_the_generator() {
    let generated = generate_artifacts();
    assert_eq!(generated.r9v_toml, include_str!("../../../r9v.toml"));
    assert_eq!(
        generated.config_markdown,
        include_str!("../../../docs/config.md")
    );
    assert_eq!(
        generated.json_schema,
        include_str!("../../../r9v.schema.json")
    );
}

#[test]
fn settings_index_is_checked_against_the_actual_spec() {
    let spec = include_str!("../../../specs/spec-12-config-and-helper.md");
    check_settings_index(spec).unwrap();
    let missing = spec.replace("`spec.k_max`, ", "");
    let error = check_settings_index(&missing).unwrap_err();
    assert_eq!(error.missing, Vec::<String>::new());
    assert_eq!(error.extra, vec!["spec.k_max"]);
}

#[test]
fn config_version_is_required_and_exact() {
    let mut config = EffectiveConfig::from_defaults();
    assert!(matches!(
        config.apply_file_str("missing.toml", "[io]\nqueue_depth = 8\n"),
        Err(ConfigError::Version { .. })
    ));
    assert!(matches!(
        config.apply_file_str("future.toml", "config_version = 2\n"),
        Err(ConfigError::Version { .. })
    ));
}
