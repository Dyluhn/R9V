// SPDX-License-Identifier: Apache-2.0
//! In-memory kernel registry, resolution order with validation flags, lazy module loading, and allow_jit gating (Spec 4 §9, §11, §12, §14).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use r9v_hip::{HipLibrary, Module};
use serde::{Deserialize, Serialize};

use crate::error::{RegistryError, Result};
use crate::manifest::BundleManifest;
use crate::tune::{TuneEntry, TuneFile};
use crate::types::{ArchName, ArtifactOrigin, LaunchGeometry, OpId, OpStatic, Tier, VariantHash};
use crate::variant::{static_hash, variant_hash, VariantKey};

/// Registry configuration options (Spec 4 §14).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Expected kernel generator version (Spec 4 §3, §11).
    pub gen_version: u32,
    /// Whether to allow online JIT compilation/autotuning of unshipped variants (Spec 4 §9.2, §14).
    pub allow_jit: bool,
    /// Time budget in milliseconds for online autotuning per variant (Spec 4 §6.1, §14).
    pub tune_budget_ms: u32,
    /// Escape hatch allowing nondeterministic kernel variants (Spec 4 §14).
    pub allow_nondeterministic: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            gen_version: 1,
            allow_jit: true,
            tune_budget_ms: 2000,
            allow_nondeterministic: false,
        }
    }
}

/// Resolved runnable kernel variant metadata (Spec 4 §9.1, §9.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedVariant {
    /// 64-bit unique variant hash (Spec 4 §3).
    pub variant_hash: VariantHash,
    /// Target architecture for this variant (Spec 4 §9.2).
    pub arch: ArchName,
    /// Target operation identifier (Spec 4 §9.2).
    pub op: OpId,
    /// Selected execution tier (Spec 4 §2, §9.2).
    pub tier: Tier,
    /// Exported entry point kernel symbol name (Spec 4 §9.1).
    pub entry_symbol: String,
    /// Grid and block launch geometry (Spec 4 §7).
    pub launch_geometry: LaunchGeometry,
    /// Scratch workspace memory required in bytes (Spec 4 §7).
    pub workspace_bytes: u64,
    /// Static bytes transferred per launch (Spec 4 §12).
    pub static_bytes: u64,
    /// Static operations per launch (Spec 4 §12).
    pub static_flops: u64,
    /// Relative path to compiled code object file on disk; paths are deliberately relative-only (Spec 4 §11).
    pub code_object_path: Option<String>,
    /// Origin of the code object (shipped bundle vs local autotune/JIT) (Spec 4 §9.2, §11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_origin: Option<ArtifactOrigin>,
    /// In-memory code object bytes if pre-loaded or generated.
    #[serde(skip)]
    pub code_object_bytes: Option<Vec<u8>>,
    /// Whether this variant has passed golden and determinism validation gates (Spec 4 §9.3).
    pub validated: bool,
}

/// Trait for runtime JIT compilation and autotuning providers (Spec 4 §6.1, §9.2).
///
/// Implemented by `r9v-kgen` in card A3.8.
pub trait JitProvider: Send + Sync {
    /// JIT compiles, autotunes, and validates a kernel variant for the target arch (Spec 4 §6.1, §9.2).
    fn jit_compile_and_validate(
        &self,
        op: OpId,
        arch: &ArchName,
        op_static: &OpStatic,
    ) -> Result<ResolvedVariant>;
}

// DECISION(A3.1): loaded HIP modules are cached in an Arc<RwLock<BTreeMap<VariantHash, Arc<r9v_hip::Module>>>> inside the Registry; rejected loading per-launch or global mutable statics because registry instances must manage their own device context safely and allow multiple concurrent readers. Spec 4 §9.1, §11.
/// In-memory kernel registry managing variant resolution and lazy module loading (Spec 4 §9, §11).
#[derive(Clone)]
pub struct Registry {
    config: RegistryConfig,
    bundle_dir: Option<PathBuf>,
    manifest: Option<BundleManifest>,
    tune_entries: BTreeMap<(ArchName, u32, String), (TuneEntry, Option<PathBuf>)>,
    jit_provider: Option<Arc<dyn JitProvider>>,
    loaded_modules: Arc<RwLock<BTreeMap<VariantHash, Arc<Module>>>>,
}

impl Registry {
    /// Constructs a new empty registry with the specified configuration (Spec 4 §9.1).
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            config,
            bundle_dir: None,
            manifest: None,
            tune_entries: BTreeMap::new(),
            jit_provider: None,
            loaded_modules: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Returns the active registry configuration.
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    /// Returns a mutable reference to the registry configuration.
    pub fn config_mut(&mut self) -> &mut RegistryConfig {
        &mut self.config
    }

    /// Loads a bundle directory containing `manifest.json` and code objects (Spec 4 §11).
    pub fn load_bundle(&mut self, bundle_dir: &Path) -> Result<()> {
        let manifest_path = bundle_dir.join("manifest.json");
        let manifest = BundleManifest::from_file(&manifest_path)?;
        manifest.validate_version(self.config.gen_version, Some(&manifest_path))?;
        self.bundle_dir = Some(bundle_dir.to_path_buf());
        self.manifest = Some(manifest);
        Ok(())
    }

    /// Directly sets an in-memory bundle manifest and optional base directory (Spec 4 §11).
    pub fn set_manifest(
        &mut self,
        manifest: BundleManifest,
        bundle_dir: Option<PathBuf>,
    ) -> Result<()> {
        manifest.validate()?;
        manifest.validate_version(self.config.gen_version, None)?;
        self.bundle_dir = bundle_dir;
        self.manifest = Some(manifest);
        Ok(())
    }

    /// Loads a tune file and merges its entries into the in-memory tune table (Spec 4 §6.2).
    ///
    /// Preserves a per-tune base directory for resolving relative local code objects.
    /// Validates the tune file before loading.
    /// Entries are isolated by both architecture and generator version. Mismatched versions are discarded.
    pub fn load_tune_file(
        &mut self,
        tune_file: &TuneFile,
        base_dir: Option<&Path>,
    ) -> Result<bool> {
        tune_file.validate()?;
        if base_dir.is_none() {
            let mut problems = Vec::new();
            for (key, entry) in &tune_file.entries {
                if let Some(ref co) = entry.code_object {
                    problems.push(format!(
                        "tune entry '{key}': code object '{co}' is relative-only but no base directory was provided"
                    ));
                }
            }
            if !problems.is_empty() {
                return Err(RegistryError::ValidationFailed { problems });
            }
        }
        if tune_file.gen_version != self.config.gen_version {
            return Ok(false);
        }
        let base = base_dir.map(Path::to_path_buf);
        for (key, entry) in &tune_file.entries {
            self.tune_entries.insert(
                (tune_file.arch.clone(), tune_file.gen_version, key.clone()),
                (entry.clone(), base.clone()),
            );
        }
        Ok(true)
    }

    /// Reads, validates, and loads a tune file from disk, preserving its parent directory as the artifact base (Spec 4 §6.2).
    pub fn load_tune_file_from_path(&mut self, path: &Path) -> Result<bool> {
        let tune_file = TuneFile::from_file(path)?;
        let base_dir = path.parent();
        self.load_tune_file(&tune_file, base_dir)
    }

    /// Registers a runtime JIT autotune provider (Spec 4 §6.1, §9.2).
    pub fn register_jit_provider(&mut self, provider: Arc<dyn JitProvider>) {
        self.jit_provider = Some(provider);
    }

    /// Returns the active bundle manifest if loaded.
    pub fn manifest(&self) -> Option<&BundleManifest> {
        self.manifest.as_ref()
    }

    /// Returns the bundle manifest fingerprint if a manifest is loaded (Spec 4 §11).
    pub fn manifest_fingerprint(&self) -> Option<u64> {
        self.manifest.as_ref().map(|m| m.manifest_fingerprint())
    }

    /// Checks whether the target architecture is in the support matrix (Spec 4 §9.2).
    pub fn is_arch_supported(&self, arch: &ArchName) -> bool {
        self.manifest
            .as_ref()
            .map(|m| m.is_arch_supported(arch.as_str()))
            .unwrap_or(false)
    }

    /// Returns the list of supported architectures known to this registry.
    pub fn supported_archs(&self) -> Vec<String> {
        self.manifest
            .as_ref()
            .map(|m| m.archs.iter().map(|a| a.as_str().to_owned()).collect())
            .unwrap_or_default()
    }

    // DECISION(A3.1): resolution on an unlisted arch with no JIT compiler immediately returns RegistryError::UnlistedArchRefused naming the target arch and listing all supported archs from the manifest; rejected silent failure or returning a dummy variant because Spec 4 §9.2 mandates that the engine refuses to start and names the arch. Spec 4 §9.2.
    /// Resolves an op instance to a runnable kernel variant at graph capture time (Spec 4 §9.2).
    ///
    /// Resolution order:
    /// 1. Architecture validation: unlisted arch without a JIT compiler refuses immediately.
    /// 2. Shipped T2: lookup in bundle manifest (`validated == true` required per Spec 4 §9.3).
    /// 3. Local T2: lookup in local tune file (`validated == true` required per Spec 4 §9.3).
    /// 4. JIT compilation: if `allow_jit` is enabled and a JIT provider is registered, autotune.
    /// 5. T1 fallback: portable reference HIP variant for this op and arch.
    ///
    /// An unvalidated variant is NEVER selected (Spec 4 §9.3).
    pub fn resolve(
        &self,
        op: OpId,
        arch: &ArchName,
        op_static: &OpStatic,
    ) -> Result<ResolvedVariant> {
        // Exact OpId-to-nested descriptor agreement: an OpId may never be paired
        // with a wrong nested descriptor (Spec 4 §3). Typed error, never a panic.
        op_static.check_pair(op)?;
        op_static.validate()?;
        let shash = static_hash(op_static);

        // 1. Architecture validation (Spec 4 §9.2)
        if !self.is_arch_supported(arch) {
            // Unlisted arch check: JIT-compile T1 if compiler is present, otherwise refuse (Spec 4 §9.2)
            if self.config.allow_jit {
                if let Some(ref jit) = self.jit_provider {
                    let variant = jit.jit_compile_and_validate(op, arch, op_static)?;
                    if !variant.validated {
                        return Err(RegistryError::VariantNotValidated {
                            hash: variant.variant_hash.as_u64(),
                            op,
                            arch: arch.as_str().to_owned(),
                        });
                    }
                    return Ok(variant);
                }
            }
            return Err(RegistryError::UnlistedArchRefused {
                arch: arch.as_str().to_owned(),
                supported: self.supported_archs(),
            });
        }

        // 2. Shipped T2 entry lookup in bundle manifest (Spec 4 §9.2)
        if let Some(ref manifest) = self.manifest {
            for (hash_str, entry) in &manifest.variants {
                if entry.tier == Tier::T2
                    && entry.arch == *arch
                    && entry.matches_request(op)
                    && entry.static_hash == Some(shash)
                {
                    // Typed op-tag agreement on the selected entry (Spec 4 §3).
                    BundleManifest::check_entry_for(entry, op)?;
                    // Spec 4 §9.3: An unvalidated variant is never selected.
                    if entry.validated {
                        let vhash = VariantHash::from_hex(hash_str).map_err(|e| {
                            RegistryError::ManifestParseError {
                                path: "variant_hash".to_owned(),
                                detail: e.to_string(),
                            }
                        })?;
                        return Ok(ResolvedVariant {
                            variant_hash: vhash,
                            arch: arch.clone(),
                            op,
                            tier: Tier::T2,
                            entry_symbol: entry.entry_symbol.clone(),
                            launch_geometry: entry.launch_geometry,
                            workspace_bytes: entry.workspace_bytes,
                            static_bytes: entry.static_bytes,
                            static_flops: entry.static_flops,
                            code_object_path: Some(entry.file.clone()),
                            artifact_origin: Some(ArtifactOrigin::Shipped),
                            code_object_bytes: None,
                            validated: true,
                        });
                    }
                }
            }
        }

        // 3. Local tune entry lookup (Spec 4 §9.2)
        let tune_key = TuneFile::entry_key(op, shash);
        if let Some((entry, base_dir)) =
            self.tune_entries
                .get(&(arch.clone(), self.config.gen_version, tune_key))
        {
            // Spec 4 §9.3: local variant must be validated and have a code object
            if entry.validated && entry.code_object.is_some() {
                let vkey = VariantKey::new(
                    op,
                    arch.clone(),
                    self.config.gen_version,
                    op_static.clone(),
                    entry.config.clone(),
                );
                let vhash = variant_hash(&vkey);
                return Ok(ResolvedVariant {
                    variant_hash: vhash,
                    arch: arch.clone(),
                    op,
                    tier: Tier::T2,
                    entry_symbol: format!("{}_{:016x}", op.as_str(), vhash.as_u64()),
                    launch_geometry: entry.launch_geometry,
                    workspace_bytes: entry.workspace_bytes,
                    static_bytes: entry.bytes,
                    static_flops: entry.flops,
                    code_object_path: entry.code_object.clone(),
                    artifact_origin: Some(ArtifactOrigin::Local {
                        base_dir: base_dir.clone(),
                    }),
                    code_object_bytes: None,
                    validated: true,
                });
            }
        }

        // 4. JIT compilation path gated behind allow_jit (Spec 4 §9.2, §14)
        if self.config.allow_jit {
            if let Some(ref jit) = self.jit_provider {
                // TODO(A3.8): JIT autotune loop lands with A3.8; invoked here via JitProvider
                if let Ok(mut variant) = jit.jit_compile_and_validate(op, arch, op_static) {
                    if variant.validated {
                        if variant.artifact_origin.is_none() {
                            variant.artifact_origin =
                                Some(ArtifactOrigin::Local { base_dir: None });
                        }
                        return Ok(variant);
                    }
                }
            }
        }

        // 5. T1 fallback (Spec 4 §9.2: "resolution cannot fail on a supported arch")
        if let Some(ref manifest) = self.manifest {
            // First search for an exact (op, static_hash) T1 entry or generic op T1 entry
            for (hash_str, entry) in &manifest.variants {
                if entry.tier == Tier::T1
                    && entry.arch == *arch
                    && entry.op == Some(op)
                    && (entry.static_hash == Some(shash) || entry.static_hash.is_none())
                    && entry.validated
                {
                    // Typed op-tag agreement on the selected entry (Spec 4 §3).
                    BundleManifest::check_entry_for(entry, op)?;
                    let vhash = VariantHash::from_hex(hash_str).map_err(|e| {
                        RegistryError::ManifestParseError {
                            path: "variant_hash".to_owned(),
                            detail: e.to_string(),
                        }
                    })?;
                    return Ok(ResolvedVariant {
                        variant_hash: vhash,
                        arch: arch.clone(),
                        op,
                        tier: Tier::T1,
                        entry_symbol: entry.entry_symbol.clone(),
                        launch_geometry: entry.launch_geometry,
                        workspace_bytes: entry.workspace_bytes,
                        static_bytes: entry.static_bytes,
                        static_flops: entry.static_flops,
                        code_object_path: Some(entry.file.clone()),
                        artifact_origin: Some(ArtifactOrigin::Shipped),
                        code_object_bytes: None,
                        validated: true,
                    });
                }
            }
            return Err(RegistryError::T1FallbackFailed {
                op,
                arch: arch.as_str().to_owned(),
                detail: format!(
                    "no validated T1 entry found in manifest for op '{op}' on arch '{arch}'"
                ),
            });
        }

        Err(RegistryError::VariantNotFound {
            op,
            arch: arch.as_str().to_owned(),
            static_hash: shash,
        })
    }

    /// Lazily loads a code object module into the GPU runtime via `hipModuleLoadData` (Spec 4 §11).
    ///
    /// Loads on demand upon first resolution or launch, caching the loaded [`Module`] by variant hash.
    /// Re-checks safe relative path containment against the appropriate base directory (Spec 4 §11).
    pub fn load_module(
        &self,
        lib: &Arc<HipLibrary>,
        variant: &ResolvedVariant,
    ) -> Result<Arc<Module>> {
        if !variant.validated {
            return Err(RegistryError::VariantNotValidated {
                hash: variant.variant_hash.as_u64(),
                op: variant.op,
                arch: variant.arch.as_str().to_owned(),
            });
        }

        // Fast path: check if already loaded
        {
            let reader = self
                .loaded_modules
                .read()
                .map_err(|_| RegistryError::LockPoisoned {
                    resource: "loaded_modules".to_owned(),
                })?;
            if let Some(module) = reader.get(&variant.variant_hash) {
                return Ok(Arc::clone(module));
            }
        }

        // Fetch code object bytes
        let bytes = if let Some(ref b) = variant.code_object_bytes {
            b.clone()
        } else if let Some(ref path_str) = variant.code_object_path {
            let rel_path = Path::new(path_str);
            if rel_path.is_absolute() || path_str.starts_with('/') || path_str.starts_with('\\') {
                return Err(RegistryError::ModuleLoadError {
                    hash: variant.variant_hash.as_u64(),
                    symbol: variant.entry_symbol.clone(),
                    detail: format!("disallowed code object path '{path_str}': path is absolute"),
                });
            }
            if rel_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
                || path_str.contains("..")
            {
                return Err(RegistryError::ModuleLoadError {
                    hash: variant.variant_hash.as_u64(),
                    symbol: variant.entry_symbol.clone(),
                    detail: format!(
                        "disallowed code object path '{path_str}': parent traversal ('..') is prohibited"
                    ),
                });
            }
            if path_str.trim().is_empty() {
                return Err(RegistryError::ModuleLoadError {
                    hash: variant.variant_hash.as_u64(),
                    symbol: variant.entry_symbol.clone(),
                    detail: "code object path cannot be empty".to_owned(),
                });
            }

            let base = match variant.artifact_origin {
                Some(ArtifactOrigin::Shipped) => {
                    self.bundle_dir
                        .as_deref()
                        .ok_or_else(|| RegistryError::ModuleLoadError {
                            hash: variant.variant_hash.as_u64(),
                            symbol: variant.entry_symbol.clone(),
                            detail: "bundle directory not configured for shipped variant"
                                .to_owned(),
                        })?
                }
                Some(ArtifactOrigin::Local {
                    base_dir: Some(ref b),
                }) => b.as_path(),
                Some(ArtifactOrigin::Local { base_dir: None }) => {
                    return Err(RegistryError::ModuleLoadError {
                        hash: variant.variant_hash.as_u64(),
                        symbol: variant.entry_symbol.clone(),
                        detail: "no base directory configured for relative local code object"
                            .to_owned(),
                    });
                }
                None => {
                    if let Some(ref b) = self.bundle_dir {
                        b.as_path()
                    } else {
                        return Err(RegistryError::ModuleLoadError {
                            hash: variant.variant_hash.as_u64(),
                            symbol: variant.entry_symbol.clone(),
                            detail: "no base directory or bundle directory configured".to_owned(),
                        });
                    }
                }
            };

            let full_path = base.join(rel_path);

            if let (Ok(canon_base), Ok(canon_full)) =
                (base.canonicalize(), full_path.canonicalize())
            {
                if !canon_full.starts_with(&canon_base) {
                    return Err(RegistryError::ModuleLoadError {
                        hash: variant.variant_hash.as_u64(),
                        symbol: variant.entry_symbol.clone(),
                        detail: format!(
                            "path traversal detected: '{path_str}' resolved outside base '{}'",
                            base.display()
                        ),
                    });
                }
            }

            fs::read(&full_path).map_err(RegistryError::Io)?
        } else {
            return Err(RegistryError::ModuleLoadError {
                hash: variant.variant_hash.as_u64(),
                symbol: variant.entry_symbol.clone(),
                detail: "no code object bytes or file path provided for variant".to_owned(),
            });
        };

        // Load into device via hipModuleLoadData (Spec 4 §11)
        let module =
            Module::load_data(lib, &bytes).map_err(|e| RegistryError::ModuleLoadError {
                hash: variant.variant_hash.as_u64(),
                symbol: variant.entry_symbol.clone(),
                detail: e.to_string(),
            })?;
        let arc_mod = Arc::new(module);

        // Store into cache
        let mut writer = self
            .loaded_modules
            .write()
            .map_err(|_| RegistryError::LockPoisoned {
                resource: "loaded_modules".to_owned(),
            })?;
        writer.insert(variant.variant_hash, Arc::clone(&arc_mod));
        Ok(arc_mod)
    }

    /// Injects pre-loaded in-memory module for testing or ahead-of-time loading.
    pub fn insert_loaded_module(&self, hash: VariantHash, module: Arc<Module>) -> Result<()> {
        let mut writer = self
            .loaded_modules
            .write()
            .map_err(|_| RegistryError::LockPoisoned {
                resource: "loaded_modules".to_owned(),
            })?;
        writer.insert(hash, module);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_loaded_modules_lock_poisoning() {
        let registry = Registry::new(RegistryConfig::default());
        let reg_clone = registry.clone();

        let _ = thread::spawn(move || {
            let _guard = reg_clone.loaded_modules.write().unwrap();
            panic!("deliberate panic to poison loaded_modules lock");
        })
        .join();

        // Reading a poisoned loaded_modules lock must return RegistryError::LockPoisoned
        let read_res = registry
            .loaded_modules
            .read()
            .map_err(|_| RegistryError::LockPoisoned {
                resource: "loaded_modules".to_owned(),
            });
        match read_res {
            Err(RegistryError::LockPoisoned { resource }) => {
                assert_eq!(resource, "loaded_modules");
            }
            _ => panic!("expected LockPoisoned for read lock"),
        }

        // Writing a poisoned loaded_modules lock must return RegistryError::LockPoisoned
        let write_res = registry
            .loaded_modules
            .write()
            .map_err(|_| RegistryError::LockPoisoned {
                resource: "loaded_modules".to_owned(),
            });
        match write_res {
            Err(RegistryError::LockPoisoned { resource }) => {
                assert_eq!(resource, "loaded_modules");
            }
            _ => panic!("expected LockPoisoned for write lock"),
        }
    }
}
