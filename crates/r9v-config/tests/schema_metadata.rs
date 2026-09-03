//! Schema metadata tests: completeness + exact phase-A consistency with
//! Spec 12 §3 (first half of card A0.3).

use r9v_config::{all_settings, find_setting, Mutability};
use std::collections::BTreeSet;

fn expected() -> Vec<(&'static str, &'static str, Mutability)> {
    vec![
        ("load.model", "(none)", Mutability::Load),
        ("load.draft_model", "none", Mutability::Load),
        ("load.eagle_head", "none", Mutability::Load),
        ("load.cache_dir", "auto (beside model)", Mutability::Load),
        ("load.require_fast_path", "false", Mutability::Load),
        ("io.mode", "auto", Mutability::Load),
        ("io.chunk_mb", "16", Mutability::Load),
        ("io.queue_depth", "8", Mutability::Load),
        ("io.repack_threads", "auto (cores-2)", Mutability::Load),
        (
            "host.pinned_budget",
            "auto (min(free-4GB, need))",
            Mutability::Load,
        ),
        ("warmup.enabled", "true", Mutability::Load),
        ("warmup.buckets", "{S:[1,2,4]", Mutability::Load),
        ("state.max_ctx", "32768", Mutability::Reload),
        ("state.max_seqs", "8", Mutability::Reload),
        ("state.cache_dtype", "e4m3", Mutability::Reload),
        ("state.reserve_bytes", "512 MB", Mutability::Reload),
        ("state.host_block_budget", "0", Mutability::Reload),
        ("state.session_cache", "2", Mutability::Runtime),
        ("scheduler.step_budget_ms", "auto", Mutability::Runtime),
        ("scheduler.prefill_min_chunk", "128", Mutability::Runtime),
        ("scheduler.prefill_max_chunk", "2048", Mutability::Runtime),
        ("scheduler.max_wait_ms", "500", Mutability::Runtime),
        ("graph.mode", "auto (measured)", Mutability::Reload),
        ("spec.proposer", "auto", Mutability::Reload),
        ("spec.k_max", "8", Mutability::Runtime),
        ("spec.tree_max", "16", Mutability::Runtime),
        ("spec.min_accept", "0.3", Mutability::Runtime),
        ("spec.lossy", "false", Mutability::Runtime),
        ("spec.ngram.n", "3", Mutability::Runtime),
        ("spec.ngram.min_match", "2", Mutability::Runtime),
        ("kernels.allow_jit", "true", Mutability::Load),
        ("kernels.allow_nondeterministic", "false", Mutability::Load),
        ("kernels.tune_budget_ms", "2000", Mutability::Load),
        ("profile.mode", "step", Mutability::Runtime),
        ("log.level", "info", Mutability::Runtime),
        ("log.file", "none", Mutability::Runtime),
        ("doctor.include_tokens", "false", Mutability::Runtime),
        ("doctor.redact", "true", Mutability::Runtime),
        ("bench.repeats", "5", Mutability::Runtime),
        ("bench.warmup", "2", Mutability::Runtime),
        (
            "bench.suites",
            "[decode, decode-spec, prefill, multi]",
            Mutability::Runtime,
        ),
    ]
}

#[test]
fn metadata_is_complete_one_source_of_truth() {
    let all = all_settings();
    assert!(!all.is_empty());
    let mut seen = BTreeSet::new();
    for s in &all {
        assert!(!s.key.is_empty(), "empty key");
        assert!(s.key.contains('.'), "key without section: {}", s.key);
        assert!(!s.doc.is_empty(), "empty doc: {}", s.key);
        assert!(!s.type_name.is_empty(), "empty type: {}", s.key);
        assert!(!s.default.is_empty(), "empty default: {}", s.key);
        assert_eq!(s.since, 1, "since must be 1 in phase A: {}", s.key);
        assert!(seen.insert(s.key), "duplicate key: {}", s.key);
        assert_eq!(find_setting(s.key).unwrap().key, s.key);
        if s.default.starts_with("auto") {
            assert!(s.doc.contains("auto = "), "auto rule missing: {}", s.key);
        }
    }
    // Enum-typed settings must declare their members.
    for key in [
        "io.mode",
        "state.cache_dtype",
        "graph.mode",
        "profile.mode",
        "log.level",
    ] {
        let s = find_setting(key).unwrap();
        assert!(!s.range_or_enum.is_empty(), "enum missing: {key}");
    }
    // Range-typed settings must declare their range.
    for key in ["scheduler.step_budget_ms", "state.max_ctx", "bench.repeats"] {
        let s = find_setting(key).unwrap();
        assert!(!s.range_or_enum.is_empty(), "range missing: {key}");
    }
    assert_eq!(
        find_setting("scheduler.step_budget_ms").unwrap().interacts,
        ["scheduler.max_wait_ms", "spec.k_max", "parallel.profile"]
    );
}

#[test]
fn phase_a_matches_spec12_section3_exactly() {
    let all = all_settings();
    let keys: BTreeSet<&str> = all.iter().map(|s| s.key).collect();
    let exp = expected();
    let exp_keys: BTreeSet<&str> = exp.iter().map(|e| e.0).collect();
    let missing: Vec<&&str> = exp_keys.difference(&keys).collect();
    let extra: Vec<&&str> = keys.difference(&exp_keys).collect();
    assert!(missing.is_empty(), "missing settings: {missing:?}");
    assert!(extra.is_empty(), "extra settings: {extra:?}");
    for (key, default_part, mutability) in exp {
        let s = find_setting(key).unwrap();
        assert!(
            s.default.contains(default_part),
            "{} default {:?} missing {default_part:?}",
            key,
            s.default
        );
        assert_eq!(s.mutability, mutability, "mutability: {key}");
    }
}

#[test]
fn metadata_is_deterministic() {
    let a: Vec<&str> = all_settings().iter().map(|s| s.key).collect();
    let b: Vec<&str> = all_settings().iter().map(|s| s.key).collect();
    assert_eq!(a, b);
    let mut sorted = a.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(a.len(), sorted.len(), "keys must be unique");
}
