// SPDX-License-Identifier: Apache-2.0
//! Bundle manifest representation, validation, and fingerprint calculation (Spec 4 §11).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use r9v_common::hash::xxh3_64;
use serde::{Deserialize, Serialize};

use crate::error::{RegistryError, Result};
use crate::types::{ArchName, LaunchGeometry, OpId, Tier, VariantHash};

impl ManifestVariantEntry {
    /// Returns true when this entry may serve a request for `requested` (Spec 4 §9.2).
    ///
    /// Tagged entries match exactly; untagged entries (generic T1 fallbacks)
    /// match any request. Lookup loops skip non-matching entries with this;
    /// direct selection uses [`BundleManifest::check_entry_for`] for a typed error.
    pub fn matches_request(&self, requested: OpId) -> bool {
        match self.op {
            Some(entry_op) => entry_op == requested,
            None => true,
        }
    }
}

/// A single variant entry in the bundle manifest (Spec 4 §11).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// DECISION(A3.1): ManifestVariantEntry explicitly stores typed target architecture arch: ArchName validated against manifest.archs and matched exactly during resolution; rejected implicit arch prefix inference from file path because path conventions are not globally constrained and per-variant hashes are irreversible xxh3_64 hashes. Spec 4 §9.2, §11.
pub struct ManifestVariantEntry {
    /// Target architecture for this compiled variant (Spec 4 §11).
    pub arch: ArchName,
    /// Relative path to compiled code object (`.co`) file within the bundle directory (Spec 4 §11).
    pub file: String,
    /// Execution tier of the compiled variant (Spec 4 §2, §11).
    pub tier: Tier,
    /// Exported entry point kernel symbol name (Spec 4 §9.1).
    pub entry_symbol: String,
    /// Launch geometry parameters (Spec 4 §7).
    pub launch_geometry: LaunchGeometry,
    /// Fixed scratch workspace memory required in bytes (Spec 4 §7).
    pub workspace_bytes: u64,
    /// Static bytes read and written per launch for memory accounting (Spec 4 §12).
    #[serde(default)]
    pub static_bytes: u64,
    /// Static floating-point / integer ops performed per launch (Spec 4 §12).
    #[serde(default)]
    pub static_flops: u64,
    /// Logical operation identity when known (Spec 1 §4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<OpId>,
    /// 64-bit static parameters hash (Spec 4 §6.2, §7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_hash: Option<u64>,
    /// Whether this variant has passed golden and determinism gates on hardware (Spec 4 §9.3).
    #[serde(default)]
    pub validated: bool,
    /// Environment fingerprint identifier on which this variant was validated (Spec 4 §9.3, §11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validated_on: Option<String>,
}

/// Shipped release bundle manifest (`manifest.json`, Spec 4 §11).
///
/// Contains code object mappings for all supported target architectures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BundleManifest {
    /// Generator version used to build this bundle (Spec 4 §11).
    pub gen_version: u32,
    /// List of target architectures included in this bundle (Spec 4 §11).
    pub archs: Vec<ArchName>,
    /// Map of variant hash (16-char hex) to variant entry (Spec 4 §11).
    pub variants: BTreeMap<String, ManifestVariantEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
struct VariantsMap(BTreeMap<String, ManifestVariantEntry>);

impl<'de> Deserialize<'de> for VariantsMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct VVisitor;
        impl<'de> serde::de::Visitor<'de> for VVisitor {
            type Value = VariantsMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map of variant entries without duplicate keys")
            }
            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut map = BTreeMap::new();
                while let Some(key) = access.next_key::<String>()? {
                    if map.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate variant key '{key}' in manifest"
                        )));
                    }
                    let val = access.next_value()?;
                    map.insert(key, val);
                }
                Ok(VariantsMap(map))
            }
        }
        deserializer.deserialize_map(VVisitor)
    }
}

impl<'de> Deserialize<'de> for BundleManifest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ManifestVisitor;
        impl<'de> serde::de::Visitor<'de> for ManifestVisitor {
            type Value = BundleManifest;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a bundle manifest JSON object")
            }
            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: serde::de::MapAccess<'de>,
            {
                let mut gen_version = None;
                let mut archs = None;
                let mut variants = None;

                while let Some(key) = access.next_key::<String>()? {
                    match key.as_str() {
                        "gen_version" => {
                            if gen_version.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate field 'gen_version' in manifest",
                                ));
                            }
                            gen_version = Some(access.next_value()?);
                        }
                        "archs" => {
                            if archs.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate field 'archs' in manifest",
                                ));
                            }
                            archs = Some(access.next_value()?);
                        }
                        "variants" => {
                            if variants.is_some() {
                                return Err(serde::de::Error::custom(
                                    "duplicate field 'variants' in manifest",
                                ));
                            }
                            let vmap: VariantsMap = access.next_value()?;
                            variants = Some(vmap.0);
                        }
                        other => {
                            return Err(serde::de::Error::custom(format!(
                                "unknown field '{other}' in bundle manifest"
                            )));
                        }
                    }
                }

                let gen_version =
                    gen_version.ok_or_else(|| serde::de::Error::missing_field("gen_version"))?;
                let archs = archs.ok_or_else(|| serde::de::Error::missing_field("archs"))?;
                let variants = variants.unwrap_or_default();
                Ok(BundleManifest {
                    gen_version,
                    archs,
                    variants,
                })
            }
        }
        deserializer.deserialize_map(ManifestVisitor)
    }
}

impl BundleManifest {
    /// Constructs a new empty bundle manifest.
    pub fn new(gen_version: u32, archs: Vec<ArchName>) -> Self {
        Self {
            gen_version,
            archs,
            variants: BTreeMap::new(),
        }
    }

    /// Validates internal consistency, key formats, path security, arch targets, and numerics (Spec 4 §11, CONVENTIONS.md §1.4).
    ///
    /// Collects all validation problems before returning (CONVENTIONS.md §1.4).
    pub fn validate(&self) -> Result<()> {
        let mut problems = Vec::new();

        if self.archs.is_empty() {
            problems.push("manifest archs list cannot be empty".to_owned());
        }

        let mut seen_archs = std::collections::BTreeSet::new();
        for arch in &self.archs {
            if arch.as_str().trim().is_empty() {
                problems.push("architecture name in manifest archs cannot be empty".to_owned());
            }
            if !seen_archs.insert(arch.as_str()) {
                problems.push(format!("duplicate architecture '{arch}' in manifest archs"));
            }
        }

        for (hash_str, entry) in &self.variants {
            // 1. Variant hash format
            if hash_str.len() != 16 {
                problems.push(format!(
                    "invalid variant hash '{hash_str}': length must be exactly 16 hex characters, got {}",
                    hash_str.len()
                ));
            }
            if VariantHash::from_hex(hash_str).is_err() {
                problems.push(format!(
                    "invalid variant hash '{hash_str}': must be valid hexadecimal string"
                ));
            }

            // 2. Safe relative code object path
            let path = Path::new(&entry.file);
            if path.is_absolute() || entry.file.starts_with('/') || entry.file.starts_with('\\') {
                problems.push(format!(
                    "variant '{hash_str}': disallowed code object path '{}' is absolute; only relative paths are permitted",
                    entry.file
                ));
            }
            if path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
                || entry.file.contains("..")
            {
                problems.push(format!(
                    "variant '{hash_str}': disallowed code object path '{}' contains parent traversal ('..')",
                    entry.file
                ));
            }
            if entry.file.trim().is_empty() {
                problems.push(format!(
                    "variant '{hash_str}': code object file path cannot be empty"
                ));
            }

            // 3. Architecture validation and mismatch check
            if entry.arch.as_str().trim().is_empty() {
                problems.push(format!(
                    "variant '{hash_str}': architecture name cannot be empty"
                ));
            } else if !self.archs.iter().any(|a| a == &entry.arch) {
                problems.push(format!(
                    "variant '{hash_str}': architecture '{}' is not listed in manifest archs: {:?}",
                    entry.arch,
                    self.archs.iter().map(|a| a.as_str()).collect::<Vec<_>>()
                ));
            }

            if let Some((first_component, _)) = entry.file.split_once('/') {
                if first_component.starts_with("gfx") {
                    if !self.archs.iter().any(|a| a.as_str() == first_component) {
                        problems.push(format!(
                            "variant '{hash_str}': file path '{}' references architecture '{first_component}' not listed in manifest archs: {:?}",
                            entry.file,
                            self.archs.iter().map(|a| a.as_str()).collect::<Vec<_>>()
                        ));
                    }
                    if !entry.arch.as_str().is_empty() && first_component != entry.arch.as_str() {
                        problems.push(format!(
                            "variant '{hash_str}': file path '{}' architecture prefix '{first_component}' does not match entry architecture '{}'",
                            entry.file, entry.arch
                        ));
                    }
                }
            }

            // 4. Numerics check
            if entry.launch_geometry.grid[0] == 0
                || entry.launch_geometry.grid[1] == 0
                || entry.launch_geometry.grid[2] == 0
            {
                problems.push(format!(
                    "variant '{hash_str}': launch geometry grid dimensions must be non-zero, got {:?}",
                    entry.launch_geometry.grid
                ));
            }
            if entry.launch_geometry.block[0] == 0
                || entry.launch_geometry.block[1] == 0
                || entry.launch_geometry.block[2] == 0
            {
                problems.push(format!(
                    "variant '{hash_str}': launch geometry block dimensions must be non-zero, got {:?}",
                    entry.launch_geometry.block
                ));
            }
            if entry.entry_symbol.trim().is_empty() {
                problems.push(format!(
                    "variant '{hash_str}': entry_symbol cannot be empty"
                ));
            }
        }

        if !problems.is_empty() {
            return Err(RegistryError::ValidationFailed { problems });
        }
        Ok(())
    }

    /// Parses a bundle manifest from a JSON string (Spec 4 §11).
    pub fn from_json_str(s: &str) -> Result<Self> {
        let manifest: Self =
            serde_json::from_str(s).map_err(|e| RegistryError::ManifestParseError {
                path: "<in-memory>".to_owned(),
                detail: e.to_string(),
            })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serializes this manifest to a formatted JSON string (Spec 4 §11).
    pub fn to_json_string(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|e| RegistryError::ManifestParseError {
            path: "<serialization>".to_owned(),
            detail: e.to_string(),
        })
    }

    /// Reads and parses a bundle manifest file from disk (Spec 4 §11).
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(RegistryError::Io)?;
        let manifest: Self =
            serde_json::from_str(&content).map_err(|e| RegistryError::ManifestParseError {
                path: path.display().to_string(),
                detail: e.to_string(),
            })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serializes and writes this bundle manifest to disk (Spec 4 §11).
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let text = self.to_json_string()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(RegistryError::Io)?;
        }
        fs::write(path, text).map_err(RegistryError::Io)
    }

    /// Validates that the manifest generator version matches the engine's generator version (Spec 4 §11).
    ///
    /// Per Spec 4 §11: "A bundle built for gen_version = n refuses to load under generator n + 1."
    pub fn validate_version(&self, expected_version: u32, path: Option<&Path>) -> Result<()> {
        if self.gen_version != expected_version {
            return Err(RegistryError::GenVersionMismatch {
                expected: expected_version,
                got: self.gen_version,
                path: path.map(|p| p.display().to_string()),
            });
        }
        Ok(())
    }

    /// Computes the 64-bit manifest fingerprint checksum (Spec 4 §11).
    ///
    /// Per Spec 4 §11: "The manifest hash is part of the doctor fingerprint."
    pub fn manifest_fingerprint(&self) -> u64 {
        let bytes = serde_json::to_vec(self)
            .expect("BundleManifest serialization into canonical JSON must succeed");
        xxh3_64(&bytes)
    }

    /// Checks if a given architecture is in the manifest's supported architecture list.
    pub fn is_arch_supported(&self, arch: &str) -> bool {
        self.archs.iter().any(|a| a.as_str() == arch)
    }

    /// Looks up a variant entry by its [`VariantHash`].
    pub fn get_variant(&self, hash: VariantHash) -> Option<&ManifestVariantEntry> {
        self.variants.get(&hash.to_hex())
    }

    /// Checks a single manifest entry against a requested op with a typed error (Spec 4 §9.2).
    ///
    /// Entries carrying an op tag for a different op are rejected with
    /// [`RegistryError::StaticOpMismatch`], never silently accepted. Untagged
    /// entries (e.g. generic T1 fallbacks) pass. Lookup loops keep skipping
    /// non-matching entries via [`ManifestVariantEntry::matches_request`];
    /// this check guards direct single-entry selection paths.
    pub fn check_entry_for(
        entry: &ManifestVariantEntry,
        requested: OpId,
    ) -> Result<(), RegistryError> {
        match entry.op {
            Some(entry_op) if entry_op != requested => Err(RegistryError::StaticOpMismatch {
                op: requested,
                static_op: entry_op,
            }),
            _ => Ok(()),
        }
    }

    /// Inserts or updates a variant entry in the manifest.
    pub fn insert_variant(&mut self, hash: VariantHash, entry: ManifestVariantEntry) {
        self.variants.insert(hash.to_hex(), entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_manifest_json_roundtrip() {
        let mut manifest =
            BundleManifest::new(1, vec![ArchName::from("gfx942"), ArchName::from("gfx1100")]);
        let vh = VariantHash::new(0xabcdef0123456789);
        manifest.insert_variant(
            vh,
            ManifestVariantEntry {
                arch: ArchName::from("gfx942"),
                file: "kernels/sample.co".to_string(),
                tier: Tier::T2,
                entry_symbol: "sample_kernel".to_string(),
                launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
                workspace_bytes: 128,
                static_bytes: 256,
                static_flops: 512,
                op: Some(OpId::Matmul),
                static_hash: Some(42),
                validated: true,
                validated_on: Some("runner_0".to_string()),
            },
        );

        let json = manifest.to_json_string().unwrap();
        let loaded = BundleManifest::from_json_str(&json).unwrap();
        assert_eq!(manifest, loaded);
        assert_eq!(
            manifest.manifest_fingerprint(),
            loaded.manifest_fingerprint()
        );

        assert!(manifest.is_arch_supported("gfx942"));
        assert!(manifest.is_arch_supported("gfx1100"));
        assert!(!manifest.is_arch_supported("gfx1030"));

        let entry = manifest.get_variant(vh).unwrap();
        assert_eq!(entry.entry_symbol, "sample_kernel");
        assert!(entry.validated);

        assert!(manifest.validate_version(1, None).is_ok());
        assert!(manifest.validate_version(2, None).is_err());
    }
}
