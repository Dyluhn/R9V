// SPDX-License-Identifier: Apache-2.0
//! Tune-file reader, writer, and merger (Spec 4 §6.2).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{RegistryError, Result};
use crate::types::{ArchName, LaunchGeometry, OpId, TileConfig};

/// Measurement environment metadata recorded in tune files (Spec 4 §6.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuneMeasuredOn {
    /// Driver version string (Spec 4 §6.2).
    pub driver: String,
    /// ROCm runtime version string (Spec 4 §6.2).
    pub rocm: String,
    /// GPU core clock frequency in MHz during measurement (Spec 4 §6.2).
    pub clock_mhz: f32,
}

/// A single autotuned variant configuration entry (Spec 4 §6.1, §6.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuneEntry {
    /// Autotuned tile configuration (Spec 4 §3, §6.2).
    pub config: TileConfig,
    /// Median kernel execution latency in microseconds (Spec 4 §6.1, §6.2).
    pub median_us: f64,
    /// Bytes transferred per launch (Spec 4 §6.2, §12).
    #[serde(default)]
    pub bytes: u64,
    /// Floating-point or integer operations per launch (Spec 4 §6.2, §12).
    #[serde(default)]
    pub flops: u64,
    /// Launch grid and block configuration (Spec 4 §7).
    pub launch_geometry: LaunchGeometry,
    /// Scratch workspace memory required in bytes (Spec 4 §7).
    #[serde(default)]
    pub workspace_bytes: u64,
    /// Optional relative path to local compiled code object; paths are deliberately relative-only (Spec 4 §9.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_object: Option<String>,
    /// Whether this autotuned variant has been validated against T0 (Spec 4 §6.1, §9.3).
    #[serde(default)]
    pub validated: bool,
    /// Whether this entry was produced under online autotune timeout (Spec 4 §6.1).
    #[serde(default)]
    pub partial: bool,
    /// Measurement environment for this entry (Spec 4 §6.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_on: Option<TuneMeasuredOn>,
}

// DECISION(A3.1): tune file entries are keyed by "<op>.<static_hash_hex>" in the [entries] table; rejected nested two-level tables because flat string keys make TOML parsing and inspection simpler and deterministic. Spec 4 §6.2.
/// Tune file storing offline reference tunes or local autotuning results (Spec 4 §6.2).
///
/// Shipped: `tune/<arch>/<gen_version>.toml`.
/// Local: `~/.cache/r9v/tune/<arch>/<gen_version>/<driver_hash>.toml`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TuneFile {
    /// Target GPU architecture (Spec 4 §6.2).
    pub arch: ArchName,
    /// Generator version (Spec 4 §6.2).
    pub gen_version: u32,
    /// Driver environment hash for local additions (Spec 4 §6.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_hash: Option<String>,
    /// Default hardware measurement environment for this tune file (Spec 4 §6.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_on: Option<TuneMeasuredOn>,
    /// Table of entries keyed by `"<op>.<static_hash_hex>"` (Spec 4 §6.2).
    #[serde(default)]
    pub entries: BTreeMap<String, TuneEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
struct TuneEntriesMap(BTreeMap<String, TuneEntry>);

impl<'de> Deserialize<'de> for TuneEntriesMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TEVisitor;
        impl<'de> serde::de::Visitor<'de> for TEVisitor {
            type Value = TuneEntriesMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map of tune entries without duplicate keys")
            }
            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut map = BTreeMap::new();
                while let Some(key) = access.next_key::<String>()? {
                    if map.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate entry key '{key}' in tune file"
                        )));
                    }
                    let val = access.next_value()?;
                    map.insert(key, val);
                }
                Ok(TuneEntriesMap(map))
            }
        }
        deserializer.deserialize_map(TEVisitor)
    }
}

impl<'de> Deserialize<'de> for TuneFile {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TFVisitor;
        impl<'de> serde::de::Visitor<'de> for TFVisitor {
            type Value = TuneFile;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a tune file table")
            }
            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut arch = None;
                let mut gen_version = None;
                let mut driver_hash = None;
                let mut measured_on = None;
                let mut entries = None;

                while let Some(key) = access.next_key::<String>()? {
                    match key.as_str() {
                        "arch" => {
                            if arch.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate key 'arch' in tune file",
                                ));
                            }
                            arch = Some(access.next_value()?);
                        }
                        "gen_version" => {
                            if gen_version.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate key 'gen_version' in tune file",
                                ));
                            }
                            gen_version = Some(access.next_value()?);
                        }
                        "driver_hash" => {
                            if driver_hash.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate key 'driver_hash' in tune file",
                                ));
                            }
                            driver_hash = Some(access.next_value()?);
                        }
                        "measured_on" => {
                            if measured_on.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate key 'measured_on' in tune file",
                                ));
                            }
                            measured_on = Some(access.next_value()?);
                        }
                        "entries" => {
                            if entries.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate key 'entries' in tune file",
                                ));
                            }
                            let emap: TuneEntriesMap = access.next_value()?;
                            entries = Some(emap.0);
                        }
                        other => {
                            return Err(serde::de::Error::custom(format!(
                                "unknown field '{other}' in tune file"
                            )));
                        }
                    }
                }

                let arch = arch.ok_or_else(|| serde::de::Error::missing_field("arch"))?;
                let gen_version =
                    gen_version.ok_or_else(|| serde::de::Error::missing_field("gen_version"))?;
                let entries = entries.unwrap_or_default();

                Ok(TuneFile {
                    arch,
                    gen_version,
                    driver_hash,
                    measured_on,
                    entries,
                })
            }
        }
        deserializer.deserialize_map(TFVisitor)
    }
}

impl TuneFile {
    /// Constructs a new empty tune file.
    pub fn new(arch: ArchName, gen_version: u32) -> Self {
        Self {
            arch,
            gen_version,
            driver_hash: None,
            measured_on: None,
            entries: BTreeMap::new(),
        }
    }

    /// Validates internal consistency, key formats, path security, and numerics (Spec 4 §6.2, CONVENTIONS.md §1.4).
    ///
    /// Collects all validation problems before returning (CONVENTIONS.md §1.4).
    pub fn validate(&self) -> Result<()> {
        let mut problems = Vec::new();

        if self.arch.as_str().trim().is_empty() {
            problems.push("tune file arch cannot be empty".to_owned());
        }

        if let Some(ref mo) = self.measured_on {
            if !mo.clock_mhz.is_finite() || mo.clock_mhz <= 0.0 {
                problems.push(format!(
                    "tune file clock_mhz must be finite and positive, got {}",
                    mo.clock_mhz
                ));
            }
        }

        for (key, entry) in &self.entries {
            // 1. Key format validation
            if let Some((op_str, hash_str)) = key.split_once('.') {
                if OpId::parse_op(op_str).is_none() {
                    problems.push(format!(
                        "malformed tune entry key '{key}': unknown op '{op_str}'"
                    ));
                }
                if hash_str.len() != 16 || u64::from_str_radix(hash_str, 16).is_err() {
                    problems.push(format!(
                        "malformed tune entry key '{key}': static hash must be exactly 16 hex characters"
                    ));
                }
            } else {
                problems.push(format!(
                    "malformed tune entry key '{key}': expected format '<op>.<static_hash_hex>'"
                ));
            }

            // 2. Safe relative code object path
            if let Some(ref co) = entry.code_object {
                let path = Path::new(co);
                if path.is_absolute() || co.starts_with('/') || co.starts_with('\\') {
                    problems.push(format!(
                        "tune entry '{key}': disallowed code object path '{co}' is absolute; only relative paths are permitted"
                    ));
                }
                if path
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
                    || co.contains("..")
                {
                    problems.push(format!(
                        "tune entry '{key}': disallowed code object path '{co}' contains parent traversal ('..')"
                    ));
                }
                if co.trim().is_empty() {
                    problems.push(format!(
                        "tune entry '{key}': code object path cannot be empty"
                    ));
                }
            }

            // 3. Numerics validation
            if !entry.median_us.is_finite() || entry.median_us <= 0.0 {
                problems.push(format!(
                    "tune entry '{key}': median_us must be finite and positive, got {}",
                    entry.median_us
                ));
            }
            if entry.launch_geometry.grid[0] == 0
                || entry.launch_geometry.grid[1] == 0
                || entry.launch_geometry.grid[2] == 0
            {
                problems.push(format!(
                    "tune entry '{key}': launch geometry grid dimensions must be non-zero, got {:?}",
                    entry.launch_geometry.grid
                ));
            }
            if entry.launch_geometry.block[0] == 0
                || entry.launch_geometry.block[1] == 0
                || entry.launch_geometry.block[2] == 0
            {
                problems.push(format!(
                    "tune entry '{key}': launch geometry block dimensions must be non-zero, got {:?}",
                    entry.launch_geometry.block
                ));
            }
            entry
                .config
                .validate(&mut problems, &format!("tune entry '{key}'"));
            if let Some(ref mo) = entry.measured_on {
                if !mo.clock_mhz.is_finite() || mo.clock_mhz <= 0.0 {
                    problems.push(format!(
                        "tune entry '{key}': clock_mhz must be finite and positive, got {}",
                        mo.clock_mhz
                    ));
                }
            }
        }

        if !problems.is_empty() {
            return Err(RegistryError::ValidationFailed { problems });
        }
        Ok(())
    }

    /// Constructs the canonical entry lookup key for `(op, static_hash)`.
    pub fn entry_key(op: OpId, static_hash: u64) -> String {
        format!("{}.{:016x}", op.as_str(), static_hash)
    }

    /// Parses a tune file from a TOML string (Spec 4 §6.2).
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let tune: Self = toml::from_str(s).map_err(|e| RegistryError::TuneParseError {
            path: "<in-memory>".to_owned(),
            detail: e.to_string(),
        })?;
        tune.validate()?;
        Ok(tune)
    }

    /// Serializes this tune file into a TOML string (Spec 4 §6.2).
    pub fn to_toml_string(&self) -> Result<String> {
        self.validate()?;
        toml::to_string_pretty(self).map_err(|e| RegistryError::TuneParseError {
            path: "<serialization>".to_owned(),
            detail: e.to_string(),
        })
    }

    /// Reads and parses a tune file from disk (Spec 4 §6.2).
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(RegistryError::Io)?;
        let tune: Self = toml::from_str(&text).map_err(|e| RegistryError::TuneParseError {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        tune.validate()?;
        Ok(tune)
    }

    /// Serializes and writes this tune file to disk (Spec 4 §6.2).
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let text = self.to_toml_string()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(RegistryError::Io)?;
        }
        fs::write(path, text).map_err(RegistryError::Io)
    }

    /// Merges local tune additions into this tune file (Spec 4 §6.2).
    ///
    /// Per Spec 4 §6.2: "A local entry overrides a shipped one only if measured on the same
    /// gen_version; a bump discards both." Also isolates by target architecture.
    ///
    /// Validates both `self` and `local` before mutation. Returns `Err` if validation fails,
    /// leaving `self` unchanged (atomic refusal). Returns `Ok(false)` if discarded due to
    /// an architecture or generator version mismatch, or `Ok(true)` if successfully merged.
    pub fn merge_local(&mut self, local: &TuneFile) -> Result<bool> {
        self.validate()?;
        local.validate()?;
        if self.gen_version != local.gen_version || self.arch != local.arch {
            return Ok(false);
        }
        for (key, entry) in &local.entries {
            self.entries.insert(key.clone(), entry.clone());
        }
        Ok(true)
    }

    /// Looks up a tune entry for the given op and static parameter hash.
    pub fn get_entry(&self, op: OpId, static_hash: u64) -> Option<&TuneEntry> {
        let key = Self::entry_key(op, static_hash);
        self.entries.get(&key)
    }

    /// Inserts or replaces a tune entry for the given op and static parameter hash.
    pub fn insert_entry(&mut self, op: OpId, static_hash: u64, entry: TuneEntry) {
        let key = Self::entry_key(op, static_hash);
        self.entries.insert(key, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tune_file_toml_roundtrip() {
        let mut tune = TuneFile::new(ArchName::from("gfx942"), 1);
        tune.driver_hash = Some("drv_12345678".to_string());
        tune.measured_on = Some(TuneMeasuredOn {
            driver: "6.2.0".to_string(),
            rocm: "6.2".to_string(),
            clock_mhz: 2100.0,
        });

        tune.insert_entry(
            OpId::Matmul,
            0x1234,
            TuneEntry {
                config: TileConfig::new(64, 64, 32),
                median_us: 42.5,
                bytes: 4096,
                flops: 8192,
                launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
                workspace_bytes: 0,
                code_object: Some("matmul_fast.co".to_string()),
                validated: true,
                partial: false,
                measured_on: None,
            },
        );

        let toml_str = tune.to_toml_string().unwrap();
        let loaded = TuneFile::from_toml_str(&toml_str).unwrap();
        assert_eq!(tune, loaded);

        let entry = loaded.get_entry(OpId::Matmul, 0x1234).unwrap();
        assert_eq!(entry.median_us, 42.5);
        assert_eq!(entry.code_object.as_deref(), Some("matmul_fast.co"));
        assert!(entry.validated);
    }
}
