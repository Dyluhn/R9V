//! Deterministic config, documentation, and JSON-schema generation (Spec 12 §2).

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use serde_json::{json, Map, Value as JsonValue};
use toml::Value;

use crate::config::{default_value, ConfigError, EffectiveConfig, CONFIG_VERSION};
use crate::{all_settings, SettingSpec};

/// The three artifacts emitted by `r9v config gen` (Spec 12 §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedArtifacts {
    /// Commented config skeleton.
    pub r9v_toml: String,
    /// Per-setting Markdown reference.
    pub config_markdown: String,
    /// JSON Schema served by `/r9v/schema` later.
    pub json_schema: String,
}

/// Generate every config artifact deterministically from schema declarations.
pub fn generate_artifacts() -> GeneratedArtifacts {
    let defaults = EffectiveConfig::from_defaults();
    GeneratedArtifacts {
        r9v_toml: render_effective_toml(&defaults, false),
        config_markdown: generate_markdown(),
        json_schema: generate_json_schema(),
    }
}

/// Write `r9v.toml`, `docs/config.md`, and `r9v.schema.json` beneath `root`.
pub fn write_generated(root: &Path) -> Result<GeneratedArtifacts, ConfigError> {
    let generated = generate_artifacts();
    let docs = root.join("docs");
    std::fs::create_dir_all(&docs)?;
    std::fs::write(root.join("r9v.toml"), &generated.r9v_toml)?;
    std::fs::write(docs.join("config.md"), &generated.config_markdown)?;
    std::fs::write(root.join("r9v.schema.json"), &generated.json_schema)?;
    Ok(generated)
}

/// Render an effective configuration, optionally including source comments.
/// Preserved `[x-*]` sections are appended unchanged in value semantics.
pub fn render_effective_toml(config: &EffectiveConfig, include_sources: bool) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Generated from r9v-config schema (Spec 12 §2). Do not maintain a parallel setting list."
    )
    .expect("String write");
    writeln!(out, "config_version = {CONFIG_VERSION}\n").expect("String write");

    let mut emitted = BTreeSet::new();
    for spec in all_settings() {
        let (section, leaf) = split_key(spec.key);
        if emitted.insert(section) {
            writeln!(out, "[{section}]").expect("String write");
        }
        writeln!(out, "# {}", spec.doc).expect("String write");
        let mut metadata = format!(
            "type: {}; default: {}; mutability: {}",
            spec.type_name, spec.default, spec.mutability
        );
        if !spec.range_or_enum.is_empty() {
            write!(metadata, "; constraint: {}", spec.range_or_enum).expect("String write");
        }
        if !spec.unit.is_empty() {
            write!(metadata, "; unit: {}", spec.unit).expect("String write");
        }
        writeln!(out, "# {metadata}").expect("String write");
        let sourced = config
            .get(spec.key)
            .expect("all declared settings have effective values");
        if include_sources {
            writeln!(out, "# source: {}", sourced.source).expect("String write");
            if let Some(resolved) = &sourced.resolved {
                writeln!(out, "# resolved: {}", toml_literal(resolved)).expect("String write");
            }
        }
        writeln!(out, "{leaf} = {}\n", toml_literal(&sourced.value)).expect("String write");
    }

    if !config.extensions().is_empty() {
        let extension_table: toml::Table = config
            .extensions()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        out.push_str(
            &toml::to_string(&extension_table)
                .expect("extension values came from a valid TOML document"),
        );
    } else if out.ends_with("\n\n") {
        out.pop();
    }
    out
}

fn generate_markdown() -> String {
    let mut out = String::from(
        "# R9V configuration\n\nGenerated from the `r9v-config` schema (Spec 12 §2).\n\n",
    );
    let mut emitted = BTreeSet::new();
    for spec in all_settings() {
        let (section, _) = split_key(spec.key);
        if emitted.insert(section) {
            writeln!(out, "## `{section}`\n").expect("String write");
        }
        writeln!(out, "### `{}`\n", spec.key).expect("String write");
        writeln!(out, "{}\n", spec.doc).expect("String write");
        writeln!(out, "- Type: `{}`", spec.type_name).expect("String write");
        writeln!(out, "- Default: `{}`", spec.default).expect("String write");
        writeln!(out, "- Mutability: `{}`", spec.mutability).expect("String write");
        if !spec.range_or_enum.is_empty() {
            writeln!(out, "- Range/enum: `{}`", spec.range_or_enum).expect("String write");
        }
        if !spec.unit.is_empty() {
            writeln!(out, "- Unit: `{}`", spec.unit).expect("String write");
        }
        writeln!(out, "- Since schema: `{}`", spec.since).expect("String write");
        if !spec.interacts.is_empty() {
            let keys = spec
                .interacts
                .iter()
                .map(|key| format!("`{key}`"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "- Interacts with: {keys}").expect("String write");
        }
        if !spec.renamed_from.is_empty() {
            writeln!(out, "- Renamed from: `{}`", spec.renamed_from).expect("String write");
        }
        out.push('\n');
    }
    if out.ends_with("\n\n") {
        out.pop();
    }
    out
}

fn generate_json_schema() -> String {
    let mut root_properties = Map::new();
    root_properties.insert(
        "config_version".to_string(),
        json!({"type": "integer", "const": CONFIG_VERSION}),
    );
    for spec in all_settings() {
        insert_schema(&mut root_properties, spec);
    }
    let root = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://r9v.local/schema/config-v1.json",
        "title": "R9V configuration v1",
        "type": "object",
        "required": ["config_version"],
        "properties": root_properties,
        "patternProperties": {"^x-": {"type": "object"}},
        "additionalProperties": false
    });
    let mut text = serde_json::to_string_pretty(&root).expect("schema is JSON-serializable");
    text.push('\n');
    text
}

fn insert_schema(root: &mut Map<String, JsonValue>, spec: &SettingSpec) {
    let parts: Vec<&str> = spec.key.split('.').collect();
    insert_schema_parts(root, &parts, spec);
}

fn insert_schema_parts(
    properties: &mut Map<String, JsonValue>,
    parts: &[&str],
    spec: &SettingSpec,
) {
    if parts.len() == 1 {
        properties.insert(parts[0].to_string(), setting_schema(spec));
        return;
    }
    let entry = properties.entry(parts[0].to_string()).or_insert_with(
        || json!({"type": "object", "properties": {}, "additionalProperties": false}),
    );
    let child = entry
        .as_object_mut()
        .expect("section schema is an object")
        .get_mut("properties")
        .and_then(JsonValue::as_object_mut)
        .expect("section schema has properties");
    insert_schema_parts(child, &parts[1..], spec);
}

fn setting_schema(spec: &SettingSpec) -> JsonValue {
    let concrete = spec
        .type_name
        .strip_prefix("Auto<")
        .and_then(|ty| ty.strip_suffix('>'))
        .unwrap_or(spec.type_name);
    let mut schema = match concrete {
        "bool" => json!({"type": "boolean"}),
        "u32" | "u64" => json!({"type": "integer"}),
        "f32" => json!({"type": "number"}),
        "Vec<String>" | "[str]" => {
            json!({"type": "array", "items": {"type": "string"}})
        }
        "buckets" => json!({
            "type": "object",
            "required": ["S", "T_dec", "T_pre"],
            "properties": {
                "S": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}},
                "T_dec": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}},
                "T_pre": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}}
            },
            "additionalProperties": false
        }),
        _ => json!({"type": "string"}),
    };
    let object = schema.as_object_mut().expect("setting schema is an object");
    if spec.range_or_enum.contains('|') {
        object.insert(
            "enum".to_string(),
            json!(spec.range_or_enum.split('|').collect::<Vec<_>>()),
        );
    } else if let Some((low, high)) = spec.range_or_enum.split_once("..=") {
        if let (Ok(low), Ok(high)) = (low.parse::<f64>(), high.parse::<f64>()) {
            object.insert("minimum".to_string(), json!(low));
            object.insert("maximum".to_string(), json!(high));
        }
    }
    if spec.type_name.starts_with("Auto<") && !spec.range_or_enum.contains("auto") {
        let concrete_schema = JsonValue::Object(object.clone());
        schema = json!({
            "anyOf": [concrete_schema, {"const": "auto"}],
        });
    }
    let object = schema.as_object_mut().expect("setting schema is an object");
    object.insert("description".to_string(), json!(spec.doc));
    object.insert(
        "default".to_string(),
        serde_json::to_value(default_value(spec)).expect("TOML defaults are JSON-serializable"),
    );
    object.insert(
        "x-r9v-mutability".to_string(),
        json!(spec.mutability.to_string()),
    );
    object.insert("x-r9v-since".to_string(), json!(spec.since));
    if !spec.unit.is_empty() {
        object.insert("x-r9v-unit".to_string(), json!(spec.unit));
    }
    schema
}

fn split_key(key: &str) -> (&str, &str) {
    let split = key.rfind('.').expect("setting keys contain a section");
    (&key[..split], &key[split + 1..])
}

fn toml_literal(value: &Value) -> String {
    match value {
        Value::Array(items) => {
            return format!(
                "[{}]",
                items
                    .iter()
                    .map(toml_literal)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Value::Table(entries) => {
            return format!(
                "{{ {} }}",
                entries
                    .iter()
                    .map(|(key, value)| format!("{key} = {}", toml_literal(value)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        _ => {}
    }
    let mut table = toml::Table::new();
    table.insert("value".to_string(), value.clone());
    let rendered = toml::to_string(&table).expect("effective value is TOML-serializable");
    rendered
        .trim()
        .strip_prefix("value = ")
        .expect("single value assignment")
        .to_string()
}
