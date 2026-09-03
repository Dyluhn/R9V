//! Effective configuration, precedence, sources, and validation (Spec 12 §4–5).

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use thiserror::Error;
use toml::Value;

use r9v_common::parse_byte_size;

use crate::{all_settings, find_setting, Mutability, SettingSpec};

/// Schema version written at the top of every generated config (Spec 12 §6).
pub const CONFIG_VERSION: i64 = 1;

/// Errors produced while loading or changing configuration (Spec 12 §5).
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The TOML document could not be parsed.
    #[error("cannot parse configuration {input}: {message}")]
    Parse {
        /// Input label or path.
        input: String,
        /// Parser detail.
        message: String,
    },
    /// `config_version` was absent, malformed, or unsupported.
    #[error("configuration version must be {required}; got {available}")]
    Version {
        /// Required schema version.
        required: i64,
        /// Supplied value or `missing`.
        available: String,
    },
    /// A setting was not declared by the schema.
    #[error("unknown setting `{key}`; did you mean `{nearest}`?")]
    UnknownKey {
        /// Rejected key.
        key: String,
        /// Nearest declared key by edit distance.
        nearest: String,
    },
    /// A value had the wrong type.
    #[error("setting `{key}` requires {required}; got {available}")]
    Type {
        /// Setting key.
        key: String,
        /// Required schema type.
        required: String,
        /// Supplied TOML type/value.
        available: String,
    },
    /// A value was outside its declared range.
    #[error("setting `{key}` requires range {required}; got {available}")]
    Range {
        /// Setting key.
        key: String,
        /// Declared range.
        required: String,
        /// Supplied number.
        available: String,
    },
    /// A string was not a declared enum member.
    #[error("setting `{key}` requires one of [{required}]; got `{available}`")]
    Enum {
        /// Setting key.
        key: String,
        /// Allowed values.
        required: String,
        /// Supplied value.
        available: String,
    },
    /// A supplied load-time path did not exist.
    #[error("setting `{key}` requires an existing path; `{path}` does not exist")]
    MissingPath {
        /// Setting key.
        key: String,
        /// Missing path.
        path: PathBuf,
    },
    /// A runtime request tried to change a non-runtime setting.
    #[error(
        "setting `{key}` is {mutability}, not Runtime, and cannot change through the runtime API"
    )]
    Mutability {
        /// Setting key.
        key: String,
        /// Declared mutability.
        mutability: Mutability,
    },
    /// One or more cross-field rules failed. All failures are reported.
    #[error("cross-field validation failed: {messages:?}")]
    CrossField {
        /// Complete deterministic failure list.
        messages: Vec<String>,
    },
    /// Multiple independent values in one layer failed validation.
    #[error("configuration validation failed: {summary}")]
    Multiple {
        /// User-facing concatenation of every problem.
        summary: String,
        /// Every validation problem, in deterministic input order.
        problems: Vec<ConfigError>,
    },
    /// A requested auto resolution targeted a concrete value.
    #[error("setting `{key}` is not currently `auto`")]
    NotAuto {
        /// Setting key.
        key: String,
    },
    /// Generated artifacts could not be written.
    #[error("configuration artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Origin of one effective value (Spec 12 §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Schema default.
    Default,
    /// Config file and one-based source line.
    File {
        /// File label/path.
        path: String,
        /// One-based line number.
        line: usize,
    },
    /// Environment variable.
    Env {
        /// Exact variable name.
        variable: String,
    },
    /// Generated command-line flag.
    Cli {
        /// Exact flag spelling.
        flag: String,
    },
    /// Approved runtime update.
    Runtime {
        /// Requester identity supplied by the API boundary.
        requester: String,
        /// Timestamp supplied by the API boundary.
        time: String,
    },
}

impl Source {
    fn precedence(&self) -> u8 {
        match self {
            Source::Default => 0,
            Source::File { .. } => 1,
            Source::Env { .. } => 2,
            Source::Cli { .. } => 3,
            Source::Runtime { .. } => 4,
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::Default => write!(f, "default"),
            Source::File { path, line } => write!(f, "file:{path}:{line}"),
            Source::Env { .. } => write!(f, "env"),
            Source::Cli { .. } => write!(f, "cli"),
            Source::Runtime { requester, time } => {
                write!(f, "runtime:{requester}:{time}")
            }
        }
    }
}

/// Effective raw value, source, and optional resolution of `auto` (Spec 12 §4).
#[derive(Debug, Clone, PartialEq)]
pub struct SourcedValue {
    /// Configured value. The string `"auto"` remains visible after resolution.
    pub value: Value,
    /// Winning precedence source.
    pub source: Source,
    /// Concrete value selected by the documented auto rule, when resolved.
    pub resolved: Option<Value>,
    /// Rule text copied verbatim from the setting's schema doc.
    pub auto_rule: Option<&'static str>,
}

/// Effective phase-A configuration (Spec 12 §4–5).
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveConfig {
    values: BTreeMap<String, SourcedValue>,
    extensions: BTreeMap<String, Value>,
    warnings: Vec<String>,
}

impl Default for EffectiveConfig {
    fn default() -> Self {
        Self::from_defaults()
    }
}

impl EffectiveConfig {
    /// Construct all schema defaults with `Source::Default` (Spec 12 §4).
    pub fn from_defaults() -> Self {
        let values = all_settings()
            .into_iter()
            .map(|spec| {
                let value = default_value(spec);
                (
                    spec.key.to_string(),
                    SourcedValue {
                        auto_rule: is_auto(&value).then(|| auto_rule(spec.doc)),
                        value,
                        source: Source::Default,
                        resolved: None,
                    },
                )
            })
            .collect();
        Self {
            values,
            extensions: BTreeMap::new(),
            warnings: Vec::new(),
        }
    }

    /// Return one effective setting.
    pub fn get(&self, key: &str) -> Option<&SourcedValue> {
        self.values.get(key)
    }

    /// Iterate effective settings in deterministic schema order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SourcedValue)> {
        all_settings()
            .into_iter()
            .filter_map(|spec| self.values.get(spec.key).map(|value| (spec.key, value)))
    }

    /// Preserved top-level `[x-*]` extension sections.
    pub fn extensions(&self) -> &BTreeMap<String, Value> {
        &self.extensions
    }

    /// Migration warnings emitted while loading renamed keys.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Apply a complete TOML file atomically (Spec 12 §4–6).
    pub fn apply_file_str(
        &mut self,
        path: impl Into<String>,
        text: &str,
    ) -> Result<(), ConfigError> {
        let path = path.into();
        let table = text
            .parse::<toml::Table>()
            .map_err(|error| ConfigError::Parse {
                input: path.clone(),
                message: error.to_string(),
            })?;
        let version = table.get("config_version");
        match version.and_then(Value::as_integer) {
            Some(CONFIG_VERSION) => {}
            Some(other) => {
                return Err(ConfigError::Version {
                    required: CONFIG_VERSION,
                    available: other.to_string(),
                })
            }
            None => {
                return Err(ConfigError::Version {
                    required: CONFIG_VERSION,
                    available: version
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "missing".to_string()),
                })
            }
        }

        let lines = key_lines(text);
        let mut flattened = Vec::new();
        let mut problems = Vec::new();
        let mut extensions = BTreeMap::new();
        for (key, value) in &table {
            if key == "config_version" {
                continue;
            }
            if key.starts_with("x-") {
                extensions.insert(key.clone(), value.clone());
                continue;
            }
            flatten_value(key, value, &mut flattened, &mut problems);
        }

        let mut candidate = self.clone();
        candidate.extensions = extensions;
        for (input_key, value) in flattened {
            let (key, warning) = match canonical_key(&input_key) {
                Ok(canonical) => canonical,
                Err(error) => {
                    problems.push(error);
                    continue;
                }
            };
            if let Some(warning) = warning {
                candidate.warnings.push(warning);
            }
            let spec = find_setting(key).expect("canonical key is declared");
            if let Err(error) = validate_value(spec, &value, true) {
                problems.push(error);
                continue;
            }
            let source = Source::File {
                path: path.clone(),
                line: lines.get(&input_key).copied().unwrap_or(1),
            };
            candidate.assign(spec, value, source);
        }
        finish_problems(problems)?;
        candidate.validate_cross_fields()?;
        *self = candidate;
        Ok(())
    }

    /// Apply `R9V__SECTION__KEY=value` variables atomically (Spec 12 §4).
    pub fn apply_env<I, K, V>(&mut self, entries: I) -> Result<(), ConfigError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut candidate = self.clone();
        let mut problems = Vec::new();
        for (name, raw) in entries {
            let name = name.into();
            if !name.starts_with("R9V__") {
                continue;
            }
            let key = name[5..]
                .split("__")
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
                .join(".");
            let (key, warning) = match canonical_key(&key) {
                Ok(canonical) => canonical,
                Err(error) => {
                    problems.push(error);
                    continue;
                }
            };
            if let Some(warning) = warning {
                candidate.warnings.push(warning);
            }
            let spec = find_setting(key).expect("canonical key is declared");
            let value = match parse_text_value(spec, &raw.into()) {
                Ok(value) => value,
                Err(error) => {
                    problems.push(error);
                    continue;
                }
            };
            if let Err(error) = validate_value(spec, &value, true) {
                problems.push(error);
                continue;
            }
            candidate.assign(spec, value, Source::Env { variable: name });
        }
        finish_problems(problems)?;
        candidate.validate_cross_fields()?;
        *self = candidate;
        Ok(())
    }

    /// Apply generated CLI flags (`--section.key value`) atomically (Spec 12 §4).
    pub fn apply_cli<I, K, V>(&mut self, entries: I) -> Result<(), ConfigError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut candidate = self.clone();
        let mut problems = Vec::new();
        for (flag, raw) in entries {
            let flag = flag.into();
            let input_key = flag.strip_prefix("--").unwrap_or(&flag);
            let (key, warning) = match canonical_key(input_key) {
                Ok(canonical) => canonical,
                Err(error) => {
                    problems.push(error);
                    continue;
                }
            };
            if let Some(warning) = warning {
                candidate.warnings.push(warning);
            }
            let spec = find_setting(key).expect("canonical key is declared");
            let value = match parse_text_value(spec, &raw.into()) {
                Ok(value) => value,
                Err(error) => {
                    problems.push(error);
                    continue;
                }
            };
            if let Err(error) = validate_value(spec, &value, true) {
                problems.push(error);
                continue;
            }
            candidate.assign(spec, value, Source::Cli { flag });
        }
        finish_problems(problems)?;
        candidate.validate_cross_fields()?;
        *self = candidate;
        Ok(())
    }

    /// Apply approved runtime values atomically; non-runtime keys are refused (Spec 12 §4–5).
    pub fn apply_runtime<I, K, V>(
        &mut self,
        entries: I,
        requester: impl Into<String>,
        time: impl Into<String>,
    ) -> Result<(), ConfigError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let requester = requester.into();
        let time = time.into();
        let mut candidate = self.clone();
        let mut problems = Vec::new();
        for (input_key, raw) in entries {
            let input_key = input_key.into();
            let (key, warning) = match canonical_key(&input_key) {
                Ok(canonical) => canonical,
                Err(error) => {
                    problems.push(error);
                    continue;
                }
            };
            if let Some(warning) = warning {
                candidate.warnings.push(warning);
            }
            let spec = find_setting(key).expect("canonical key is declared");
            if spec.mutability != Mutability::Runtime {
                problems.push(ConfigError::Mutability {
                    key: key.to_string(),
                    mutability: spec.mutability,
                });
                continue;
            }
            let value = match parse_text_value(spec, &raw.into()) {
                Ok(value) => value,
                Err(error) => {
                    problems.push(error);
                    continue;
                }
            };
            if let Err(error) = validate_value(spec, &value, false) {
                problems.push(error);
                continue;
            }
            candidate.assign(
                spec,
                value,
                Source::Runtime {
                    requester: requester.clone(),
                    time: time.clone(),
                },
            );
        }
        finish_problems(problems)?;
        candidate.validate_cross_fields()?;
        *self = candidate;
        Ok(())
    }

    /// Attach the concrete result of a documented `auto` rule (Spec 12 §1, §4).
    pub fn resolve_auto(&mut self, key: &str, resolved: Value) -> Result<(), ConfigError> {
        let spec = find_setting(key).ok_or_else(|| unknown_key(key))?;
        if is_auto(&resolved) {
            return Err(type_error(spec, resolved));
        }
        validate_value(spec, &resolved, false)?;
        let current = self
            .values
            .get_mut(key)
            .expect("all declared settings have defaults");
        if !is_auto(&current.value) {
            return Err(ConfigError::NotAuto {
                key: key.to_string(),
            });
        }
        current.resolved = Some(resolved);
        Ok(())
    }

    /// Run every applicable phase-A cross-field rule and report all failures (Spec 12 §5).
    pub fn validate_cross_fields(&self) -> Result<(), ConfigError> {
        let mut messages = Vec::new();
        let prefill_min = self.integer("scheduler.prefill_min_chunk");
        let prefill_max = self.integer("scheduler.prefill_max_chunk");
        let max_bucket = self
            .get("warmup.buckets")
            .and_then(|v| max_t_pre_bucket(&v.value))
            .unwrap_or(2048);
        if !(prefill_min <= prefill_max && prefill_max <= max_bucket) {
            messages.push(rule_message(
                &["scheduler.prefill_min_chunk", "scheduler.prefill_max_chunk", "warmup.buckets"],
                format!("required min <= max <= max T_pre bucket; got {prefill_min} <= {prefill_max} <= {max_bucket}"),
            ));
        }

        let k_max = self.integer("spec.k_max");
        let tree_max = self.integer("spec.tree_max");
        if k_max > 15 {
            messages.push(rule_message(
                &["spec.k_max"],
                format!("required k_max <= 15; got {k_max}"),
            ));
        }
        if tree_max > 16 {
            messages.push(rule_message(
                &["spec.tree_max"],
                format!("required tree_max <= 16; got {tree_max}"),
            ));
        }
        if k_max > tree_max {
            messages.push(rule_message(
                &["spec.k_max", "spec.tree_max"],
                format!("required k_max <= tree_max; got {k_max} > {tree_max}"),
            ));
        }

        let max_ctx = self.integer("state.max_ctx");
        if max_ctx % 32 != 0 {
            messages.push(rule_message(
                &["state.max_ctx"],
                format!("required max_ctx % 32 == 0; got {max_ctx}"),
            ));
        }
        if messages.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::CrossField { messages })
        }
    }

    fn integer(&self, key: &str) -> i64 {
        self.values
            .get(key)
            .and_then(|v| v.value.as_integer())
            .expect("integer schema default validated")
    }

    fn assign(&mut self, spec: &'static SettingSpec, value: Value, source: Source) {
        let replace = self
            .values
            .get(spec.key)
            .map(|current| source.precedence() >= current.source.precedence())
            .unwrap_or(true);
        if replace {
            self.values.insert(
                spec.key.to_string(),
                SourcedValue {
                    auto_rule: is_auto(&value).then(|| auto_rule(spec.doc)),
                    value,
                    source,
                    resolved: None,
                },
            );
        }
    }
}

fn canonical_key(input: &str) -> Result<(&'static str, Option<String>), ConfigError> {
    if let Some(spec) = find_setting(input) {
        return Ok((spec.key, None));
    }
    if let Some(spec) = all_settings()
        .into_iter()
        .find(|spec| !spec.renamed_from.is_empty() && spec.renamed_from == input)
    {
        return Ok((
            spec.key,
            Some(format!(
                "setting `{input}` was renamed to `{}`; use the new key",
                spec.key
            )),
        ));
    }
    Err(unknown_key(input))
}

fn unknown_key(input: &str) -> ConfigError {
    let nearest = all_settings()
        .into_iter()
        .min_by_key(|spec| (levenshtein(input, spec.key), spec.key))
        .map(|spec| spec.key)
        .unwrap_or("")
        .to_string();
    ConfigError::UnknownKey {
        key: input.to_string(),
        nearest,
    }
}

fn flatten_value(
    prefix: &str,
    value: &Value,
    out: &mut Vec<(String, Value)>,
    problems: &mut Vec<ConfigError>,
) {
    if find_setting(prefix).is_some()
        || all_settings()
            .into_iter()
            .any(|spec| !spec.renamed_from.is_empty() && spec.renamed_from == prefix)
    {
        out.push((prefix.to_string(), value.clone()));
        return;
    }
    if let Some(table) = value.as_table() {
        let has_descendants = all_settings()
            .into_iter()
            .any(|spec| spec.key.starts_with(&format!("{prefix}.")));
        if has_descendants {
            for (key, child) in table {
                flatten_value(&format!("{prefix}.{key}"), child, out, problems);
            }
            return;
        }
    }
    problems.push(unknown_key(prefix));
}

fn finish_problems(mut problems: Vec<ConfigError>) -> Result<(), ConfigError> {
    match problems.len() {
        0 => Ok(()),
        1 => Err(problems.pop().expect("length checked as one")),
        _ => {
            let summary = problems
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ");
            Err(ConfigError::Multiple { summary, problems })
        }
    }
}

fn key_lines(text: &str) -> BTreeMap<String, usize> {
    let mut section = String::new();
    let mut lines = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        let content = line.split('#').next().unwrap_or("").trim();
        if content.starts_with('[') && content.ends_with(']') {
            section = content
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                .to_string();
            continue;
        }
        if let Some((key, _)) = content.split_once('=') {
            let key = key.trim().trim_matches('"');
            let full = if section.is_empty() {
                key.to_string()
            } else {
                format!("{section}.{key}")
            };
            lines.insert(full, index + 1);
        }
    }
    lines
}

fn parse_text_value(spec: &SettingSpec, raw: &str) -> Result<Value, ConfigError> {
    let raw = raw.trim();
    if spec.type_name.starts_with("Auto<") && raw.eq_ignore_ascii_case("auto") {
        return Ok(Value::String("auto".to_string()));
    }
    let value = match spec.type_name {
        "bool" => bool::from_str(raw)
            .map(Value::Boolean)
            .map_err(|_| type_error(spec, raw))?,
        "u32" | "u64" => i64::from_str(raw)
            .map(Value::Integer)
            .map_err(|_| type_error(spec, raw))?,
        "f32" => f64::from_str(raw)
            .map(Value::Float)
            .map_err(|_| type_error(spec, raw))?,
        "Vec<String>" | "[str]" | "buckets" => {
            parse_fragment(raw).ok_or_else(|| type_error(spec, raw))?
        }
        _ => Value::String(raw.trim_matches('"').to_string()),
    };
    Ok(value)
}

fn parse_fragment(raw: &str) -> Option<Value> {
    format!("value = {raw}")
        .parse::<toml::Table>()
        .ok()?
        .remove("value")
}

fn validate_value(spec: &SettingSpec, value: &Value, check_path: bool) -> Result<(), ConfigError> {
    if spec.type_name.starts_with("Auto<") && is_auto(value) {
        return Ok(());
    }
    validate_concrete_auto(spec, value)?;
    let concrete_type = spec
        .type_name
        .strip_prefix("Auto<")
        .and_then(|name| name.strip_suffix('>'))
        .unwrap_or(spec.type_name);
    if concrete_type == "bytes" {
        let raw = value.as_str().ok_or_else(|| type_error(spec, value))?;
        parse_byte_size(raw).map_err(|_| type_error(spec, value))?;
    }
    if spec.range_or_enum.contains('|') {
        let available = value.as_str().ok_or_else(|| type_error(spec, value))?;
        if !spec.range_or_enum.split('|').any(|item| item == available) {
            return Err(ConfigError::Enum {
                key: spec.key.to_string(),
                required: spec.range_or_enum.replace('|', ", "),
                available: available.to_string(),
            });
        }
    } else if spec.range_or_enum.contains("..=") {
        validate_range(spec, value)?;
    }
    if check_path && is_load_path(spec.key) {
        if let Some(path) = value.as_str() {
            if !matches!(path, "none" | "(none)" | "auto") && !Path::new(path).exists() {
                return Err(ConfigError::MissingPath {
                    key: spec.key.to_string(),
                    path: PathBuf::from(path),
                });
            }
        }
    }
    Ok(())
}

fn validate_concrete_auto(spec: &SettingSpec, value: &Value) -> Result<(), ConfigError> {
    let ty = spec
        .type_name
        .strip_prefix("Auto<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or(spec.type_name);
    let valid = match ty {
        "bool" => value.is_bool(),
        "u32" | "u64" => value.as_integer().is_some_and(|v| v >= 0),
        "f32" => value.is_float() || value.is_integer(),
        "Vec<String>" | "[str]" => value
            .as_array()
            .is_some_and(|items| items.iter().all(Value::is_str)),
        "buckets" => valid_buckets(value),
        _ => value.is_str(),
    };
    if valid {
        Ok(())
    } else {
        Err(type_error(spec, value))
    }
}

fn validate_range(spec: &SettingSpec, value: &Value) -> Result<(), ConfigError> {
    let (low, high) = spec
        .range_or_enum
        .split_once("..=")
        .expect("range metadata was validated by the macro");
    let low = low.parse::<f64>().expect("numeric lower bound");
    let high = high.parse::<f64>().expect("numeric upper bound");
    let available = value
        .as_float()
        .or_else(|| value.as_integer().map(|v| v as f64))
        .ok_or_else(|| type_error(spec, value))?;
    if (low..=high).contains(&available) {
        Ok(())
    } else {
        Err(ConfigError::Range {
            key: spec.key.to_string(),
            required: spec.range_or_enum.to_string(),
            available: available.to_string(),
        })
    }
}

fn type_error(spec: &SettingSpec, available: impl fmt::Display) -> ConfigError {
    ConfigError::Type {
        key: spec.key.to_string(),
        required: spec.type_name.to_string(),
        available: available.to_string(),
    }
}

pub(crate) fn default_value(spec: &SettingSpec) -> Value {
    let default = spec.default;
    if default.starts_with("auto") {
        return Value::String("auto".to_string());
    }
    if spec.key == "warmup.buckets" {
        let mut buckets = toml::Table::new();
        buckets.insert("S".to_string(), integer_array(&[1, 2, 4]));
        buckets.insert("T_dec".to_string(), integer_array(&[1, 2, 4, 8, 16, 32]));
        buckets.insert("T_pre".to_string(), integer_array(&[0, 128, 512, 2048]));
        return Value::Table(buckets);
    }
    match spec.type_name {
        "bool" => Value::Boolean(default.parse().expect("bool schema default")),
        "u32" | "u64" => Value::Integer(default.parse().expect("integer schema default")),
        "f32" => Value::Float(default.parse().expect("float schema default")),
        "Vec<String>" | "[str]" if spec.key == "bench.suites" => Value::Array(
            ["decode", "decode-spec", "prefill", "multi"]
                .into_iter()
                .map(|s| Value::String(s.to_string()))
                .collect(),
        ),
        _ => Value::String(default.to_string()),
    }
}

fn is_auto(value: &Value) -> bool {
    value.as_str() == Some("auto")
}

fn auto_rule(doc: &'static str) -> &'static str {
    doc.split_once("auto = ")
        .map(|(_, rule)| rule.trim_end_matches('.'))
        .unwrap_or("resolved by the owning subsystem")
}

fn is_load_path(key: &str) -> bool {
    matches!(
        key,
        "load.model" | "load.draft_model" | "load.eagle_head" | "load.cache_dir"
    )
}

fn integer_array(values: &[i64]) -> Value {
    Value::Array(values.iter().copied().map(Value::Integer).collect())
}

fn valid_buckets(value: &Value) -> bool {
    let Some(table) = value.as_table() else {
        return false;
    };
    if table.len() != 3
        || !["S", "T_dec", "T_pre"]
            .iter()
            .all(|key| table.contains_key(*key))
    {
        return false;
    }
    valid_bucket_axis(table.get("S"), false)
        && valid_bucket_axis(table.get("T_dec"), false)
        && valid_bucket_axis(table.get("T_pre"), true)
}

fn valid_bucket_axis(value: Option<&Value>, allow_zero: bool) -> bool {
    let Some(items) = value.and_then(Value::as_array) else {
        return false;
    };
    !items.is_empty()
        && items.iter().all(|item| {
            item.as_integer()
                .is_some_and(|number| number >= if allow_zero { 0 } else { 1 })
        })
}

fn max_t_pre_bucket(value: &Value) -> Option<i64> {
    value
        .as_table()?
        .get("T_pre")?
        .as_array()?
        .iter()
        .filter_map(Value::as_integer)
        .max()
}

fn rule_message(keys: &[&str], detail: String) -> String {
    let docs = keys
        .iter()
        .filter_map(|key| find_setting(key).map(|spec| format!("{key}: {:?}", spec.doc)))
        .collect::<Vec<_>>()
        .join("; ");
    format!("{detail}; schema docs: {docs}")
}

fn levenshtein(a: &str, b: &str) -> usize {
    let mut previous: Vec<usize> = (0..=b.chars().count()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut current = vec![i + 1];
        for (j, cb) in b.chars().enumerate() {
            let insert = current[j] + 1;
            let delete = previous[j + 1] + 1;
            let replace = previous[j] + usize::from(ca != cb);
            current.push(insert.min(delete).min(replace));
        }
        previous = current;
    }
    previous[b.chars().count()]
}
