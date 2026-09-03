//! Settings-index consistency checking (Spec 12 §3, Spec 14 §5).

use std::collections::BTreeSet;

use thiserror::Error;

use crate::all_settings;

/// Exact difference between phase-A declarations and the Spec 12 §3 index.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("settings index mismatch; missing from code: {missing:?}; extra in code: {extra:?}")]
pub struct SettingsIndexError {
    /// Keys in the spec index but not code.
    pub missing: Vec<String>,
    /// Keys in code but not the spec index.
    pub extra: Vec<String>,
}

/// Verify that the card's phase-A prefixes exactly match Spec 12 §3.
pub fn check_settings_index(spec_markdown: &str) -> Result<(), SettingsIndexError> {
    let section = spec_markdown
        .split_once("## 3. Settings index")
        .map(|(_, tail)| tail)
        .unwrap_or("")
        .split_once("## 4.")
        .map(|(body, _)| body)
        .unwrap_or("");
    let from_spec: BTreeSet<String> = section
        .lines()
        .filter_map(|line| line.strip_prefix('|'))
        .filter_map(|line| line.split('|').next())
        .flat_map(backticked)
        .filter(|key| is_phase_a_key(key))
        .collect();
    let from_code: BTreeSet<String> = all_settings()
        .into_iter()
        .map(|spec| spec.key.to_string())
        .collect();
    let missing = from_spec
        .difference(&from_code)
        .cloned()
        .collect::<Vec<_>>();
    let extra = from_code
        .difference(&from_spec)
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() && extra.is_empty() {
        Ok(())
    } else {
        Err(SettingsIndexError { missing, extra })
    }
}

fn backticked(cell: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = cell;
    while let Some((_, after_open)) = rest.split_once('`') {
        let Some((value, after_close)) = after_open.split_once('`') else {
            break;
        };
        out.push(value.to_string());
        rest = after_close;
    }
    out
}

fn is_phase_a_key(key: &str) -> bool {
    [
        "load.",
        "io.",
        "host.",
        "warmup.",
        "state.",
        "scheduler.",
        "graph.",
        "kernels.",
        "spec.",
        "profile.",
        "log.",
        "doctor.",
        "bench.",
    ]
    .iter()
    .any(|prefix| key.starts_with(prefix))
}
