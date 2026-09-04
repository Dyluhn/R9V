// SPDX-License-Identifier: Apache-2.0
//! Variant resolution, validation gating, manifest, and tune file tests (Spec 4 §6.2, §9, §11).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use r9v_ir::{DType, Epilogue, LayoutId, QuantScheme};
use r9v_registry::{
    static_hash, variant_hash, ArchName, ArtifactOrigin, BundleManifest, JitProvider,
    LaunchGeometry, ManifestVariantEntry, MatmulStatic, OpId, OpStatic, Registry, RegistryConfig,
    RegistryError, ResolvedVariant, Tier, TileConfig, TuneEntry, TuneFile, VariantHash, VariantKey,
};

fn make_sample_matmul_static() -> OpStatic {
    OpStatic::Matmul(MatmulStatic {
        m_bucket: 128,
        n: 1024,
        k: 512,
        w_scheme: QuantScheme::None,
        w_layout: LayoutId::CONTIGUOUS,
        act_scheme: QuantScheme::None,
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        interleave: false,
        sparse: false,
    })
}

fn add_t1_fallback(manifest: &mut BundleManifest, op: OpId) {
    let arch = manifest
        .archs
        .first()
        .cloned()
        .unwrap_or_else(|| ArchName::from("gfx942"));
    let vhash = VariantHash::new(0x1111222233334444);
    manifest.insert_variant(
        vhash,
        ManifestVariantEntry {
            arch,
            file: format!("reference/{}.co", op.as_str()),
            tier: Tier::T1,
            entry_symbol: format!("t1_{}", op.as_str()),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: Some(op),
            static_hash: None,
            validated: true,
            validated_on: Some("ref".to_string()),
        },
    );
}

struct MockJitProvider {
    target_vhash: VariantHash,
    validated: bool,
}

impl JitProvider for MockJitProvider {
    fn jit_compile_and_validate(
        &self,
        op: OpId,
        arch: &ArchName,
        _op_static: &OpStatic,
    ) -> r9v_registry::Result<ResolvedVariant> {
        Ok(ResolvedVariant {
            variant_hash: self.target_vhash,
            arch: arch.clone(),
            op,
            tier: Tier::T2,
            entry_symbol: format!("jit_{}", op.as_str()),
            launch_geometry: LaunchGeometry::new([4, 1, 1], [256, 1, 1], 1024),
            workspace_bytes: 512,
            static_bytes: 2048,
            static_flops: 4096,
            code_object_path: None,
            code_object_bytes: Some(vec![0x7f, 0x45, 0x4c, 0x46]),
            validated: self.validated,
            artifact_origin: None,
        })
    }
}

#[test]
fn test_unlisted_arch_refusal() {
    let config = RegistryConfig {
        allow_jit: false,
        ..Default::default()
    };
    let mut registry = Registry::new(config);

    let manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    registry.set_manifest(manifest, None).unwrap();

    let op_static = make_sample_matmul_static();
    let unlisted = ArchName::from("gfx1030");

    let err = registry
        .resolve(OpId::Matmul, &unlisted, &op_static)
        .unwrap_err();
    match err {
        RegistryError::UnlistedArchRefused { arch, supported } => {
            assert_eq!(arch, "gfx1030");
            assert_eq!(supported, vec!["gfx942"]);
        }
        other => panic!("expected UnlistedArchRefused, got {other:?}"),
    }
}

#[test]
fn test_generator_version_mismatch_rejection() {
    let config = RegistryConfig {
        gen_version: 1,
        ..Default::default()
    };
    let mut registry = Registry::new(config);

    let manifest = BundleManifest::new(2, vec![ArchName::from("gfx942")]);
    let err = registry.set_manifest(manifest, None).unwrap_err();
    match err {
        RegistryError::GenVersionMismatch { expected, got, .. } => {
            assert_eq!(expected, 1);
            assert_eq!(got, 2);
        }
        other => panic!("expected GenVersionMismatch, got {other:?}"),
    }
}

#[test]
fn test_shipped_t2_resolution() {
    let mut registry = Registry::new(RegistryConfig::default());
    let op_static = make_sample_matmul_static();
    let shash = static_hash(&op_static);

    let mut manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    let key = VariantKey::new(
        OpId::Matmul,
        ArchName::from("gfx942"),
        1,
        op_static.clone(),
        TileConfig::new(64, 64, 32),
    );
    let vhash = variant_hash(&key);

    manifest.insert_variant(
        vhash,
        ManifestVariantEntry {
            arch: ArchName::from("gfx942"),
            file: "kernels/matmul_t2.co".to_string(),
            tier: Tier::T2,
            entry_symbol: "matmul_t2_kernel".to_string(),
            launch_geometry: LaunchGeometry::new([8, 8, 1], [256, 1, 1], 8192),
            workspace_bytes: 4096,
            static_bytes: 16384,
            static_flops: 32768,
            op: Some(OpId::Matmul),
            static_hash: Some(shash),
            validated: true,
            validated_on: Some("mi300x".to_string()),
        },
    );
    registry.set_manifest(manifest, None).unwrap();

    let resolved = registry
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("should resolve shipped T2");
    assert_eq!(resolved.variant_hash, vhash);
    assert_eq!(resolved.tier, Tier::T2);
    assert!(resolved.validated);
    assert_eq!(resolved.entry_symbol, "matmul_t2_kernel");
    assert_eq!(
        resolved.code_object_path.as_deref(),
        Some("kernels/matmul_t2.co")
    );
}

#[test]
fn test_unvalidated_shipped_t2_never_selected() {
    let mut registry = Registry::new(RegistryConfig::default());
    let op_static = make_sample_matmul_static();
    let shash = static_hash(&op_static);

    let mut manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    add_t1_fallback(&mut manifest, OpId::Matmul);
    let vhash = VariantHash::new(0xdeadbeef12345678);

    // Entry exists but is NOT validated
    manifest.insert_variant(
        vhash,
        ManifestVariantEntry {
            arch: ArchName::from("gfx942"),
            file: "kernels/matmul_unvalidated.co".to_string(),
            tier: Tier::T2,
            entry_symbol: "matmul_unvalidated".to_string(),
            launch_geometry: LaunchGeometry::new([8, 8, 1], [256, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: Some(OpId::Matmul),
            static_hash: Some(shash),
            validated: false, // NOT validated
            validated_on: None,
        },
    );
    registry.set_manifest(manifest, None).unwrap();

    let resolved = registry
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("should fall back to T1");
    // Spec 4 §9.3: An unvalidated variant is never selected.
    assert_ne!(resolved.variant_hash, vhash);
    assert_eq!(resolved.tier, Tier::T1);
    assert!(resolved.validated);
}

#[test]
fn test_local_t2_resolution_and_validation() {
    let mut registry = Registry::new(RegistryConfig::default());
    let manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    registry.set_manifest(manifest, None).unwrap();

    let op_static = make_sample_matmul_static();
    let shash = static_hash(&op_static);

    let mut tune = TuneFile::new(ArchName::from("gfx942"), 1);
    tune.insert_entry(
        OpId::Matmul,
        shash,
        TuneEntry {
            config: TileConfig::new(128, 64, 32),
            median_us: 12.5,
            bytes: 8192,
            flops: 16384,
            launch_geometry: LaunchGeometry::new([16, 8, 1], [128, 1, 1], 4096),
            workspace_bytes: 2048,
            code_object: Some("local_cache/matmul.co".to_string()),
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    assert!(registry
        .load_tune_file(&tune, Some(Path::new(".")))
        .unwrap());

    let resolved = registry
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("should resolve local T2");
    assert_eq!(resolved.tier, Tier::T2);
    assert!(resolved.validated);

    // Spec 4 §3: variant hash MUST be computed from VariantKey (not static_hash)
    let expected_vkey = VariantKey::new(
        OpId::Matmul,
        ArchName::from("gfx942"),
        1,
        op_static.clone(),
        TileConfig::new(128, 64, 32),
    );
    assert_eq!(resolved.variant_hash, variant_hash(&expected_vkey));
    assert_ne!(resolved.variant_hash.as_u64(), shash);
    assert_eq!(
        resolved.code_object_path.as_deref(),
        Some("local_cache/matmul.co")
    );

    // Now test with validated == false
    let mut unval_tune = TuneFile::new(ArchName::from("gfx942"), 1);
    unval_tune.insert_entry(
        OpId::Matmul,
        shash,
        TuneEntry {
            config: TileConfig::new(128, 64, 32),
            median_us: 10.0,
            bytes: 8192,
            flops: 16384,
            launch_geometry: LaunchGeometry::new([16, 8, 1], [128, 1, 1], 4096),
            workspace_bytes: 2048,
            code_object: Some("local_cache/unval.co".to_string()),
            validated: false, // NOT validated
            partial: false,
            measured_on: None,
        },
    );
    let mut reg2 = Registry::new(RegistryConfig::default());
    let mut manifest2 = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    add_t1_fallback(&mut manifest2, OpId::Matmul);
    reg2.set_manifest(manifest2, None).unwrap();
    assert!(reg2
        .load_tune_file(&unval_tune, Some(Path::new(".")))
        .unwrap());

    let res2 = reg2
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("should fallback to T1 when local T2 is unvalidated");
    assert_eq!(res2.tier, Tier::T1);
}

#[test]
fn test_allow_jit_gating() {
    let op_static = make_sample_matmul_static();
    let jit_vhash = VariantHash::new(0x9999888877776666);

    // 1. allow_jit = false -> skips JIT and falls back to T1
    let mut reg_no_jit = Registry::new(RegistryConfig {
        allow_jit: false,
        ..Default::default()
    });
    let mut manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    add_t1_fallback(&mut manifest, OpId::Matmul);
    reg_no_jit.set_manifest(manifest, None).unwrap();
    reg_no_jit.register_jit_provider(Arc::new(MockJitProvider {
        target_vhash: jit_vhash,
        validated: true,
    }));

    let res_no_jit = reg_no_jit
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("should fallback to T1");
    assert_eq!(res_no_jit.tier, Tier::T1);
    assert_ne!(res_no_jit.variant_hash, jit_vhash);

    // 2. allow_jit = true -> invokes JIT provider
    let mut reg_jit = Registry::new(RegistryConfig {
        allow_jit: true,
        ..Default::default()
    });
    let manifest2 = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    reg_jit.set_manifest(manifest2, None).unwrap();
    reg_jit.register_jit_provider(Arc::new(MockJitProvider {
        target_vhash: jit_vhash,
        validated: true,
    }));

    let res_jit = reg_jit
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("should invoke JIT");
    assert_eq!(res_jit.tier, Tier::T2);
    assert_eq!(res_jit.variant_hash, jit_vhash);
    assert_eq!(res_jit.entry_symbol, "jit_matmul");
}

#[test]
fn test_tune_file_merging() {
    let mut base = TuneFile::new(ArchName::from("gfx942"), 1);
    base.insert_entry(
        OpId::Matmul,
        100,
        TuneEntry {
            config: TileConfig::new(64, 64, 32),
            median_us: 15.0,
            bytes: 100,
            flops: 200,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );

    let mut local_same_gen = TuneFile::new(ArchName::from("gfx942"), 1);
    local_same_gen.insert_entry(
        OpId::Matmul,
        200,
        TuneEntry {
            config: TileConfig::new(128, 128, 32),
            median_us: 12.0,
            bytes: 200,
            flops: 400,
            launch_geometry: LaunchGeometry::new([2, 2, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );

    // Merge same gen_version should succeed and combine entries
    assert!(base.merge_local(&local_same_gen).unwrap());
    assert!(base.get_entry(OpId::Matmul, 100).is_some());
    assert!(base.get_entry(OpId::Matmul, 200).is_some());

    // Merge different gen_version should fail per Spec 4 §6.2
    let local_diff_gen = TuneFile::new(ArchName::from("gfx942"), 2);
    assert!(!base.merge_local(&local_diff_gen).unwrap());
}

#[test]
fn test_manifest_fingerprint_deterministic() {
    let mut m1 = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    let mut m2 = BundleManifest::new(1, vec![ArchName::from("gfx942")]);

    let vhash = VariantHash::new(0x123456789abcdef0);
    let entry = ManifestVariantEntry {
        arch: ArchName::from("gfx942"),
        file: "k.co".to_string(),
        tier: Tier::T2,
        entry_symbol: "sym".to_string(),
        launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        workspace_bytes: 0,
        static_bytes: 0,
        static_flops: 0,
        op: Some(OpId::Matmul),
        static_hash: Some(42),
        validated: true,
        validated_on: None,
    };

    m1.insert_variant(vhash, entry.clone());
    m2.insert_variant(vhash, entry);

    assert_eq!(m1.manifest_fingerprint(), m2.manifest_fingerprint());

    // Modifying an entry changes fingerprint
    m2.variants.get_mut(&vhash.to_hex()).unwrap().static_bytes = 100;
    assert_ne!(m1.manifest_fingerprint(), m2.manifest_fingerprint());
}

#[test]
fn test_variant_hash_stability() {
    let op_s = make_sample_matmul_static();
    let key1 = VariantKey::new(
        OpId::Matmul,
        ArchName::from("gfx942"),
        1,
        op_s.clone(),
        TileConfig::new(64, 64, 32),
    );
    let key2 = VariantKey::new(
        OpId::Matmul,
        ArchName::from("gfx942"),
        1,
        op_s.clone(),
        TileConfig::new(64, 64, 32),
    );
    let key3 = VariantKey::new(
        OpId::Matmul,
        ArchName::from("gfx942"),
        1,
        op_s,
        TileConfig::new(128, 64, 32), // different tile size
    );

    assert_eq!(key1.hash(), key2.hash());
    assert_ne!(key1.hash(), key3.hash());
}

#[test]
fn test_adversarial_tune_arch_isolation() {
    let mut registry = Registry::new(RegistryConfig::default());
    let mut manifest = BundleManifest::new(1, vec![ArchName::from("gfx1201")]);
    add_t1_fallback(&mut manifest, OpId::Matmul);
    registry.set_manifest(manifest, None).unwrap();

    let op_static = make_sample_matmul_static();
    let shash = static_hash(&op_static);

    // 1. Load tune file for gfx1100 with a validated T2 entry
    let mut tune_1100 = TuneFile::new(ArchName::from("gfx1100"), 1);
    tune_1100.insert_entry(
        OpId::Matmul,
        shash,
        TuneEntry {
            config: TileConfig::new(32, 32, 16),
            median_us: 5.0,
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: Some("gfx1100/matmul.co".to_string()),
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    assert!(registry
        .load_tune_file(&tune_1100, Some(Path::new(".")))
        .unwrap());

    // Resolve for gfx1201: MUST NOT match gfx1100 tune entry; falls back to T1 in manifest!
    let res1201 = registry
        .resolve(OpId::Matmul, &ArchName::from("gfx1201"), &op_static)
        .expect("should fall back to manifest T1 for gfx1201");
    assert_eq!(res1201.tier, Tier::T1);
    assert_ne!(
        res1201.code_object_path.as_deref(),
        Some("gfx1100/matmul.co")
    );

    // 2. Now load tune file for gfx1201 with a validated T2 entry
    let mut tune_1201 = TuneFile::new(ArchName::from("gfx1201"), 1);
    tune_1201.insert_entry(
        OpId::Matmul,
        shash,
        TuneEntry {
            config: TileConfig::new(64, 64, 32),
            median_us: 3.5,
            bytes: 4096,
            flops: 8192,
            launch_geometry: LaunchGeometry::new([2, 2, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: Some("gfx1201/matmul.co".to_string()),
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    assert!(registry
        .load_tune_file(&tune_1201, Some(Path::new(".")))
        .unwrap());

    // Resolve for gfx1201: now resolves the gfx1201 T2 entry!
    let res1201_t2 = registry
        .resolve(OpId::Matmul, &ArchName::from("gfx1201"), &op_static)
        .expect("should resolve gfx1201 T2");
    assert_eq!(res1201_t2.tier, Tier::T2);
    assert_eq!(
        res1201_t2.code_object_path.as_deref(),
        Some("gfx1201/matmul.co")
    );
}

#[test]
fn test_adversarial_no_manifest_and_no_synthetic_t1() {
    let registry = Registry::new(RegistryConfig {
        allow_jit: false,
        ..Default::default()
    });

    // Without manifest, no architecture is supported
    assert!(!registry.is_arch_supported(&ArchName::from("gfx1201")));
    assert!(!registry.is_arch_supported(&ArchName::from("gfx1100")));
    assert!(!registry.is_arch_supported(&ArchName::from("reference")));
    assert!(!registry.is_arch_supported(&ArchName::from("cpu")));
    assert!(registry.supported_archs().is_empty());

    let op_static = make_sample_matmul_static();

    // Resolving without manifest and no JIT must fail with UnlistedArchRefused
    let err = registry
        .resolve(OpId::Matmul, &ArchName::from("gfx1201"), &op_static)
        .unwrap_err();
    match err {
        RegistryError::UnlistedArchRefused { arch, supported } => {
            assert_eq!(arch, "gfx1201");
            assert!(supported.is_empty());
        }
        other => panic!("expected UnlistedArchRefused, got {other:?}"),
    }

    // With manifest supporting gfx1201 but NO variants at all
    let mut reg_empty_manifest = Registry::new(RegistryConfig {
        allow_jit: false,
        ..Default::default()
    });
    let manifest = BundleManifest::new(1, vec![ArchName::from("gfx1201")]);
    reg_empty_manifest.set_manifest(manifest, None).unwrap();
    assert!(reg_empty_manifest.is_arch_supported(&ArchName::from("gfx1201")));

    let err2 = reg_empty_manifest
        .resolve(OpId::Matmul, &ArchName::from("gfx1201"), &op_static)
        .unwrap_err();
    match err2 {
        RegistryError::T1FallbackFailed { op, arch, .. } => {
            assert_eq!(op, OpId::Matmul);
            assert_eq!(arch, "gfx1201");
        }
        other => panic!("expected T1FallbackFailed, got {other:?}"),
    }
}

#[test]
fn test_adversarial_unlisted_arch_jit_unvalidated_rejected() {
    let mut registry = Registry::new(RegistryConfig {
        allow_jit: true,
        ..Default::default()
    });
    let manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    registry.set_manifest(manifest, None).unwrap();

    let jit_vhash = VariantHash::new(0xbad_beef);
    registry.register_jit_provider(Arc::new(MockJitProvider {
        target_vhash: jit_vhash,
        validated: false, // JIT returns UNVALIDATED variant!
    }));

    let op_static = make_sample_matmul_static();
    let unlisted = ArchName::from("gfx1030");

    let err = registry
        .resolve(OpId::Matmul, &unlisted, &op_static)
        .unwrap_err();
    match err {
        RegistryError::VariantNotValidated { hash, op, arch } => {
            assert_eq!(hash, jit_vhash.as_u64());
            assert_eq!(op, OpId::Matmul);
            assert_eq!(arch, "gfx1030");
        }
        other => panic!("expected VariantNotValidated, got {other:?}"),
    }
}

#[test]
fn test_adversarial_manifest_validation() {
    // 1. Duplicate variant key in JSON
    let dup_variant_json = r#"{
        "archs": ["gfx942"],
        "gen_version": 1,
        "variants": {
            "0000000012345678": {
                "arch": "gfx942",
                "file": "kernels/k1.co",
                "tier": "t2",
                "entry_symbol": "sym1",
                "launch_geometry": {"grid": [1, 1, 1], "block": [64, 1, 1], "shared_mem_bytes": 0},
                "workspace_bytes": 0,
                "static_bytes": 0,
                "static_flops": 0,
                "validated": true
            },
            "0000000012345678": {
                "arch": "gfx942",
                "file": "kernels/k2.co",
                "tier": "t2",
                "entry_symbol": "sym2",
                "launch_geometry": {"grid": [1, 1, 1], "block": [64, 1, 1], "shared_mem_bytes": 0},
                "workspace_bytes": 0,
                "static_bytes": 0,
                "static_flops": 0,
                "validated": true
            }
        }
    }"#;
    let err = BundleManifest::from_json_str(dup_variant_json).unwrap_err();
    match err {
        RegistryError::ManifestParseError { detail, .. } => {
            assert!(detail.contains("duplicate variant key"));
        }
        other => panic!("expected ManifestParseError for duplicate variant key, got {other:?}"),
    }

    // 2. Duplicate top-level key in JSON
    let dup_top_json = r#"{
        "archs": ["gfx942"],
        "archs": ["gfx942"],
        "gen_version": 1,
        "variants": {}
    }"#;
    let err_top = BundleManifest::from_json_str(dup_top_json).unwrap_err();
    match err_top {
        RegistryError::ManifestParseError { detail, .. } => {
            assert!(detail.contains("duplicate") && detail.contains("archs"));
        }
        other => panic!("expected ManifestParseError for duplicate top-level key, got {other:?}"),
    }

    // 3. Path traversal in variant file
    let mut manifest_trav = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    manifest_trav.insert_variant(
        VariantHash::new(0x12345678),
        ManifestVariantEntry {
            arch: ArchName::from("gfx942"),
            file: "../../etc/shadow".to_string(),
            tier: Tier::T2,
            entry_symbol: "sym".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: None,
            static_hash: None,
            validated: true,
            validated_on: None,
        },
    );
    let val_err = manifest_trav.validate().unwrap_err();
    match val_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems.iter().any(|p| p.contains("parent traversal")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 4. Absolute path in variant file
    let mut manifest_abs = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    manifest_abs.insert_variant(
        VariantHash::new(0x12345678),
        ManifestVariantEntry {
            arch: ArchName::from("gfx942"),
            file: "/etc/passwd".to_string(),
            tier: Tier::T2,
            entry_symbol: "sym".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: None,
            static_hash: None,
            validated: true,
            validated_on: None,
        },
    );
    let abs_err = manifest_abs.validate().unwrap_err();
    match abs_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems.iter().any(|p| p.contains("is absolute")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 5. Zero dimensions in launch geometry
    let mut manifest_zero = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    manifest_zero.insert_variant(
        VariantHash::new(0x12345678),
        ManifestVariantEntry {
            arch: ArchName::from("gfx942"),
            file: "kernels/good.co".to_string(),
            tier: Tier::T2,
            entry_symbol: "sym".to_string(),
            launch_geometry: LaunchGeometry::new([0, 1, 1], [0, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: None,
            static_hash: None,
            validated: true,
            validated_on: None,
        },
    );
    let zero_err = manifest_zero.validate().unwrap_err();
    match zero_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("grid dimensions must be non-zero")));
            assert!(problems
                .iter()
                .any(|p| p.contains("block dimensions must be non-zero")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }
}

#[test]
fn test_adversarial_tune_validation() {
    // 1. Duplicate entry key in TOML
    let dup_tune_toml = r#"
arch = "gfx942"
gen_version = 1

[entries."matmul.0000000000001234"]
config = { block_m = 64, block_n = 64, block_k = 32, waves_m = 2, waves_n = 2, waves_k = 1, k_splits = 1 }
median_us = 10.0
launch_geometry = { grid = [1, 1, 1], block = [64, 1, 1], shared_mem_bytes = 0 }

[entries."matmul.0000000000001234"]
config = { block_m = 128, block_n = 128, block_k = 32, waves_m = 2, waves_n = 2, waves_k = 1, k_splits = 1 }
median_us = 8.0
launch_geometry = { grid = [2, 2, 1], block = [64, 1, 1], shared_mem_bytes = 0 }
"#;
    let err = TuneFile::from_toml_str(dup_tune_toml).unwrap_err();
    match err {
        RegistryError::TuneParseError { detail, .. } => {
            assert!(
                detail.contains("duplicate")
                    || detail.contains("redefine")
                    || detail.contains("already exists")
            );
        }
        other => panic!("expected TuneParseError for duplicate entry, got {other:?}"),
    }

    // 2. Duplicate top-level key in TOML
    let dup_top_toml = r#"
arch = "gfx942"
arch = "gfx1201"
gen_version = 1
"#;
    let err_top = TuneFile::from_toml_str(dup_top_toml).unwrap_err();
    match err_top {
        RegistryError::TuneParseError { detail, .. } => {
            assert!(detail.contains("duplicate") && detail.contains("arch"));
        }
        other => panic!("expected TuneParseError for duplicate top-level key, got {other:?}"),
    }

    // 3. Path traversal in code_object
    let mut tune_trav = TuneFile::new(ArchName::from("gfx942"), 1);
    tune_trav.insert_entry(
        OpId::Matmul,
        0x1234,
        TuneEntry {
            config: TileConfig::new(64, 64, 32),
            median_us: 10.0,
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: Some("../secret.co".to_string()),
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    let trav_err = tune_trav.validate().unwrap_err();
    match trav_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems.iter().any(|p| p.contains("parent traversal")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 4. Absolute path in code_object
    let mut tune_abs = TuneFile::new(ArchName::from("gfx942"), 1);
    tune_abs.insert_entry(
        OpId::Matmul,
        0x1234,
        TuneEntry {
            config: TileConfig::new(64, 64, 32),
            median_us: 10.0,
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: Some("/tmp/bad.co".to_string()),
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    let abs_err = tune_abs.validate().unwrap_err();
    match abs_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems.iter().any(|p| p.contains("is absolute")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 5. Malformed entry key and non-positive median_us
    let mut tune_bad = TuneFile::new(ArchName::from("gfx942"), 1);
    tune_bad.entries.insert(
        "invalid_op.123".to_string(),
        TuneEntry {
            config: {
                let mut c = TileConfig::new(0, 64, 32);
                c.waves_m = 0;
                c.k_splits = 0;
                c
            },
            median_us: -5.0,
            bytes: 0,
            flops: 0,
            launch_geometry: LaunchGeometry::new([0, 1, 1], [0, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    let bad_err = tune_bad.validate().unwrap_err();
    match bad_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems.iter().any(|p| p.contains("unknown op")));
            assert!(problems
                .iter()
                .any(|p| p.contains("static hash must be exactly 16 hex characters")));
            assert!(problems
                .iter()
                .any(|p| p.contains("median_us must be finite and positive")));
            assert!(problems
                .iter()
                .any(|p| p.contains("tile dimensions must be non-zero")));
            assert!(problems
                .iter()
                .any(|p| p.contains("wave counts must be non-zero")));
            assert!(problems
                .iter()
                .any(|p| p.contains("k_splits must be non-zero")));
            assert!(problems
                .iter()
                .any(|p| p.contains("launch geometry grid dimensions must be non-zero")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(prefix: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), count));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn get_stub_lib() -> Arc<r9v_hip::HipLibrary> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let stub_so = workspace_dir.join("target/test-fixtures/libamdhip64_complete.so");
    if stub_so.is_file() {
        return Arc::new(r9v_hip::HipLibrary::load_from_path(&stub_so).expect("load stub lib"));
    }
    let stub_src = workspace_dir.join("crates/r9v-hip/tests/fixtures/stub_hip.c");
    let target_dir = workspace_dir.join("target/test-fixtures");
    std::fs::create_dir_all(&target_dir).unwrap();
    let status = Command::new("gcc")
        .args(["-shared", "-fPIC", "-O2", "-o"])
        .arg(&stub_so)
        .arg(&stub_src)
        .status()
        .expect("compile stub_hip.c");
    assert!(status.success(), "failed to build stub_hip");
    Arc::new(r9v_hip::HipLibrary::load_from_path(&stub_so).expect("load stub lib"))
}

#[test]
fn test_adversarial_same_op_static_entries_for_two_archs_no_cross_selection() {
    let arch_1100 = ArchName::from("gfx1100");
    let arch_1201 = ArchName::from("gfx1201");
    let op = OpId::Matmul;
    let op_static = make_sample_matmul_static();
    let shash = static_hash(&op_static);

    let mut manifest = BundleManifest::new(1, vec![arch_1100.clone(), arch_1201.clone()]);
    let vhash_1100 = VariantHash::new(0x1100_1100_1100_1100);
    let vhash_1201 = VariantHash::new(0x1201_1201_1201_1201);

    // Adversarial identical op and static_hash, but different arch and file
    manifest.insert_variant(
        vhash_1100,
        ManifestVariantEntry {
            arch: arch_1100.clone(),
            file: "gfx1100/matmul.co".to_string(),
            tier: Tier::T2,
            entry_symbol: "kernel_gfx1100".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 1024,
            static_flops: 2048,
            op: Some(op),
            static_hash: Some(shash),
            validated: true,
            validated_on: None,
        },
    );
    manifest.insert_variant(
        vhash_1201,
        ManifestVariantEntry {
            arch: arch_1201.clone(),
            file: "gfx1201/matmul.co".to_string(),
            tier: Tier::T2,
            entry_symbol: "kernel_gfx1201".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 1024,
            static_flops: 2048,
            op: Some(op),
            static_hash: Some(shash),
            validated: true,
            validated_on: None,
        },
    );

    let mut registry = Registry::new(RegistryConfig {
        allow_jit: false,
        ..Default::default()
    });
    registry
        .set_manifest(manifest, None)
        .expect("valid manifest");

    // Resolving for gfx1100 MUST yield gfx1100 entry and never cross-select gfx1201
    let resolved_1100 = registry
        .resolve(op, &arch_1100, &op_static)
        .expect("resolve gfx1100");
    assert_eq!(resolved_1100.variant_hash, vhash_1100);
    assert_eq!(resolved_1100.entry_symbol, "kernel_gfx1100");
    assert_eq!(
        resolved_1100.code_object_path.as_deref(),
        Some("gfx1100/matmul.co")
    );
    assert_eq!(resolved_1100.artifact_origin, Some(ArtifactOrigin::Shipped));

    // Resolving for gfx1201 MUST yield gfx1201 entry and never cross-select gfx1100
    let resolved_1201 = registry
        .resolve(op, &arch_1201, &op_static)
        .expect("resolve gfx1201");
    assert_eq!(resolved_1201.variant_hash, vhash_1201);
    assert_eq!(resolved_1201.entry_symbol, "kernel_gfx1201");
    assert_eq!(
        resolved_1201.code_object_path.as_deref(),
        Some("gfx1201/matmul.co")
    );
    assert_eq!(resolved_1201.artifact_origin, Some(ArtifactOrigin::Shipped));

    // If gfx1201 variant is removed, resolving gfx1201 must NOT fall back to the gfx1100 variant
    let mut manifest_only_1100 = BundleManifest::new(1, vec![arch_1100.clone(), arch_1201.clone()]);
    manifest_only_1100.insert_variant(
        vhash_1100,
        ManifestVariantEntry {
            arch: arch_1100.clone(),
            file: "gfx1100/matmul.co".to_string(),
            tier: Tier::T2,
            entry_symbol: "kernel_gfx1100".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 1024,
            static_flops: 2048,
            op: Some(op),
            static_hash: Some(shash),
            validated: true,
            validated_on: None,
        },
    );
    registry
        .set_manifest(manifest_only_1100, None)
        .expect("set manifest");
    let err_1201 = registry
        .resolve(op, &arch_1201, &op_static)
        .expect_err("gfx1201 must fail without cross-selecting gfx1100");
    match err_1201 {
        RegistryError::VariantNotFound { arch, .. } => {
            assert_eq!(arch, "gfx1201");
        }
        RegistryError::T1FallbackFailed { arch, .. } => {
            assert_eq!(arch, "gfx1201");
        }
        other => panic!("expected VariantNotFound or T1FallbackFailed, got {other:?}"),
    }

    // Also test T1 resolution with same op across two archs
    let mut manifest_t1 = BundleManifest::new(1, vec![arch_1100.clone(), arch_1201.clone()]);
    let t1_vhash_1100 = VariantHash::new(0x1100_7171_7171_7171);
    let t1_vhash_1201 = VariantHash::new(0x1201_7171_7171_7171);
    manifest_t1.insert_variant(
        t1_vhash_1100,
        ManifestVariantEntry {
            arch: arch_1100.clone(),
            file: "gfx1100/t1.co".to_string(),
            tier: Tier::T1,
            entry_symbol: "t1_gfx1100".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: Some(op),
            static_hash: None,
            validated: true,
            validated_on: None,
        },
    );
    manifest_t1.insert_variant(
        t1_vhash_1201,
        ManifestVariantEntry {
            arch: arch_1201.clone(),
            file: "gfx1201/t1.co".to_string(),
            tier: Tier::T1,
            entry_symbol: "t1_gfx1201".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: Some(op),
            static_hash: None,
            validated: true,
            validated_on: None,
        },
    );

    registry
        .set_manifest(manifest_t1, None)
        .expect("set t1 manifest");
    let res_t1_1100 = registry
        .resolve(op, &arch_1100, &op_static)
        .expect("resolve t1 gfx1100");
    assert_eq!(res_t1_1100.variant_hash, t1_vhash_1100);
    assert_eq!(res_t1_1100.entry_symbol, "t1_gfx1100");

    let res_t1_1201 = registry
        .resolve(op, &arch_1201, &op_static)
        .expect("resolve t1 gfx1201");
    assert_eq!(res_t1_1201.variant_hash, t1_vhash_1201);
    assert_eq!(res_t1_1201.entry_symbol, "t1_gfx1201");
}

#[test]
fn test_distinct_bundle_and_local_temp_roots_and_path_containment() {
    let bundle_temp = TempDir::new("r9v_test_bundle");
    let local_temp = TempDir::new("r9v_test_local");

    let bundle_dir = bundle_temp.path();
    let local_dir = local_temp.path();

    // Create shipped code object in bundle_dir
    let shipped_co_rel = "kernels/shipped.co";
    let shipped_co_abs = bundle_dir.join(shipped_co_rel);
    std::fs::create_dir_all(shipped_co_abs.parent().unwrap()).unwrap();
    std::fs::write(&shipped_co_abs, b"shipped_kernel_bytes").unwrap();

    // Create local code object in local_dir
    let local_co_rel = "custom/local.co";
    let local_co_abs = local_dir.join(local_co_rel);
    std::fs::create_dir_all(local_co_abs.parent().unwrap()).unwrap();
    std::fs::write(&local_co_abs, b"local_kernel_bytes").unwrap();

    let arch = ArchName::from("gfx942");
    let op = OpId::Matmul;
    let op_static = make_sample_matmul_static();
    let shash = static_hash(&op_static);

    let mut manifest = BundleManifest::new(1, vec![arch.clone()]);
    let shipped_vhash = VariantHash::new(0x1111_2222_3333_4444);
    manifest.insert_variant(
        shipped_vhash,
        ManifestVariantEntry {
            arch: arch.clone(),
            file: shipped_co_rel.to_string(),
            tier: Tier::T1,
            entry_symbol: "shipped_kernel".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 1024,
            static_flops: 2048,
            op: Some(op),
            static_hash: Some(shash),
            validated: true,
            validated_on: None,
        },
    );

    let mut tune_file = TuneFile::new(arch.clone(), 1);
    tune_file.insert_entry(
        op,
        shash,
        TuneEntry {
            config: TileConfig::new(128, 128, 32),
            median_us: 5.0,
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([2, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: Some(local_co_rel.to_string()),
            validated: true,
            partial: false,
            measured_on: None,
        },
    );

    let mut registry = Registry::new(RegistryConfig {
        allow_jit: false,
        ..Default::default()
    });
    registry
        .set_manifest(manifest.clone(), Some(bundle_dir.to_path_buf()))
        .unwrap();
    // Load tune file passing explicit base directory
    registry
        .load_tune_file(&tune_file, Some(local_dir))
        .expect("load tune file with local base");

    // Resolving must select local tune entry (median_us 5.0 vs shipped T2)
    let resolved_tune = registry
        .resolve(op, &arch, &op_static)
        .expect("resolve tune");
    assert_eq!(
        resolved_tune.artifact_origin,
        Some(ArtifactOrigin::Local {
            base_dir: Some(local_dir.to_path_buf())
        })
    );
    assert_eq!(
        resolved_tune.code_object_path.as_deref(),
        Some(local_co_rel)
    );

    let lib = get_stub_lib();

    // load_module on local variant must resolve from local_dir (NOT bundle_dir)
    let mod_local = registry
        .load_module(&lib, &resolved_tune)
        .expect("load local module");
    assert!(mod_local.get_function("local_matmul").is_ok());

    // Shipped variant resolution (without tune file)
    let mut registry_shipped = Registry::new(RegistryConfig {
        allow_jit: false,
        ..Default::default()
    });
    registry_shipped
        .set_manifest(manifest, Some(bundle_dir.to_path_buf()))
        .unwrap();
    let resolved_shipped = registry_shipped
        .resolve(op, &arch, &op_static)
        .expect("resolve shipped");
    assert_eq!(
        resolved_shipped.artifact_origin,
        Some(ArtifactOrigin::Shipped)
    );
    assert_eq!(
        resolved_shipped.code_object_path.as_deref(),
        Some(shipped_co_rel)
    );

    // load_module on shipped variant must resolve from bundle_dir (NOT local_dir)
    let mod_shipped = registry_shipped
        .load_module(&lib, &resolved_shipped)
        .expect("load shipped module");
    assert!(mod_shipped.get_function("shipped_kernel").is_ok());

    // Adversarial containment checks:
    // 1. Path traversal attempt in variant code_object_path is rejected by load_module
    let traversal_variant = ResolvedVariant {
        variant_hash: VariantHash::new(0x9999),
        arch: ArchName::from("gfx942"),
        op: OpId::Matmul,
        tier: Tier::T2,
        entry_symbol: "bad_sym".to_string(),
        launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        workspace_bytes: 0,
        static_bytes: 0,
        static_flops: 0,
        code_object_path: Some("../escape.co".to_string()),
        code_object_bytes: None,
        validated: true,
        artifact_origin: Some(ArtifactOrigin::Local {
            base_dir: Some(local_dir.to_path_buf()),
        }),
    };
    let err_trav = match registry.load_module(&lib, &traversal_variant) {
        Err(e) => e,
        Ok(_) => panic!("expected load_module error for parent traversal"),
    };
    match err_trav {
        RegistryError::ModuleLoadError { detail, .. } => {
            assert!(detail.contains("parent traversal"));
        }
        other => panic!("expected ModuleLoadError with parent traversal, got {other:?}"),
    }

    // 2. Absolute path in variant code_object_path is rejected by load_module
    let abs_variant = ResolvedVariant {
        variant_hash: VariantHash::new(0x9998),
        arch: ArchName::from("gfx942"),
        op: OpId::Matmul,
        tier: Tier::T2,
        entry_symbol: "bad_sym".to_string(),
        launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        workspace_bytes: 0,
        static_bytes: 0,
        static_flops: 0,
        code_object_path: Some("/etc/shadow".to_string()),
        code_object_bytes: None,
        validated: true,
        artifact_origin: Some(ArtifactOrigin::Local {
            base_dir: Some(local_dir.to_path_buf()),
        }),
    };
    let err_abs = match registry.load_module(&lib, &abs_variant) {
        Err(e) => e,
        Ok(_) => panic!("expected load_module error for absolute path"),
    };
    match err_abs {
        RegistryError::ModuleLoadError { detail, .. } => {
            assert!(detail.contains("path is absolute"));
        }
        other => panic!("expected ModuleLoadError with path is absolute, got {other:?}"),
    }

    // 3. Symlink escaping the base root is rejected
    #[cfg(unix)]
    {
        let secret_file = TempDir::new("r9v_secret");
        let secret_target = secret_file.path().join("outside.co");
        std::fs::write(&secret_target, b"outside").unwrap();

        let symlink_path = local_dir.join("symlink_escape.co");
        let _ = std::os::unix::fs::symlink(&secret_target, &symlink_path);

        let symlink_variant = ResolvedVariant {
            variant_hash: VariantHash::new(0x9997),
            arch: ArchName::from("gfx942"),
            op: OpId::Matmul,
            tier: Tier::T2,
            entry_symbol: "symlink_escape".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            code_object_path: Some("symlink_escape.co".to_string()),
            code_object_bytes: None,
            validated: true,
            artifact_origin: Some(ArtifactOrigin::Local {
                base_dir: Some(local_dir.to_path_buf()),
            }),
        };
        let err_symlink = match registry.load_module(&lib, &symlink_variant) {
            Err(e) => e,
            Ok(_) => panic!("expected load_module error for escaping symlink"),
        };
        match err_symlink {
            RegistryError::ModuleLoadError { detail, .. } => {
                assert!(detail.contains("resolved outside base"));
            }
            other => panic!("expected ModuleLoadError with resolved outside base, got {other:?}"),
        }
    }
}

#[test]
fn test_validate_manifest_and_tune_on_set_load_and_save() {
    let mut registry = Registry::new(RegistryConfig::default());

    // 1. set_manifest rejects invalid manifest (e.g. entry arch not in manifest.archs)
    let mut invalid_manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    invalid_manifest.insert_variant(
        VariantHash::new(0x1234),
        ManifestVariantEntry {
            arch: ArchName::from("gfx1100"), // NOT in manifest.archs
            file: "k.co".to_string(),
            tier: Tier::T2,
            entry_symbol: "sym".to_string(),
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            static_bytes: 0,
            static_flops: 0,
            op: None,
            static_hash: None,
            validated: true,
            validated_on: None,
        },
    );
    let set_err = registry
        .set_manifest(invalid_manifest.clone(), None)
        .unwrap_err();
    match set_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("architecture 'gfx1100' is not listed")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 2. manifest.to_json_string() validates before serialization
    let to_json_err = invalid_manifest.to_json_string().unwrap_err();
    match to_json_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("architecture 'gfx1100' is not listed")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 3. manifest.save_to_file() validates before writing
    let temp = TempDir::new("r9v_save_test");
    let save_err = invalid_manifest
        .save_to_file(&temp.path().join("manifest.json"))
        .unwrap_err();
    match save_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("architecture 'gfx1100' is not listed")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 4. load_tune_file rejects invalid tune file (e.g. non-positive median_us)
    let mut invalid_tune = TuneFile::new(ArchName::from("gfx942"), 1);
    invalid_tune.insert_entry(
        OpId::Matmul,
        0x5678,
        TuneEntry {
            config: TileConfig::new(64, 64, 32),
            median_us: -1.0, // Invalid!
            bytes: 100,
            flops: 200,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    let tune_err = registry.load_tune_file(&invalid_tune, None).unwrap_err();
    match tune_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("median_us must be finite and positive")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 5. tune_file.to_toml_string() validates before serialization
    let to_toml_err = invalid_tune.to_toml_string().unwrap_err();
    match to_toml_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("median_us must be finite and positive")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 6. tune_file.save_to_file() validates before writing
    let save_tune_err = invalid_tune
        .save_to_file(&temp.path().join("tune.toml"))
        .unwrap_err();
    match save_tune_err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("median_us must be finite and positive")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }
}

#[test]
fn test_tile_config_unknown_fields_rejected() {
    let config = TileConfig::new(128, 64, 32);
    let mut problems = Vec::new();
    config.validate(&mut problems, "test");
    assert!(problems.is_empty(), "clean TileConfig has no problems");

    // 1. Serde JSON deserialization rejects unknown fields
    let bad_json = r#"{
        "block_m": 128,
        "block_n": 64,
        "block_k": 32,
        "waves_m": 1,
        "waves_n": 1,
        "waves_k": 1,
        "k_splits": 1,
        "lds_bytes": 0,
        "vgprs": 0,
        "experimental_fusion": 42
    }"#;
    let err_json = serde_json::from_str::<TileConfig>(bad_json).unwrap_err();
    assert!(
        err_json
            .to_string()
            .contains("unknown field `experimental_fusion`"),
        "unexpected json err: {err_json}"
    );

    // 2. Serde TOML deserialization rejects unknown fields
    let bad_toml = r#"
block_m = 128
block_n = 64
block_k = 32
waves_m = 1
waves_n = 1
waves_k = 1
k_splits = 1
lds_bytes = 0
vgprs = 0
unknown_parameter = 99
"#;
    let err_toml = toml::from_str::<TileConfig>(bad_toml).unwrap_err();
    assert!(
        err_toml
            .to_string()
            .contains("unknown field `unknown_parameter`"),
        "unexpected toml err: {err_toml}"
    );

    // 3. TuneFile TOML containing unknown TileConfig fields is rejected
    let bad_tune_toml = r#"
arch = "gfx942"
gen_version = 1

[entries."matmul.0000000012345678"]
median_us = 12.5
bytes = 1024
flops = 2048
launch_geometry = { grid = [1, 1, 1], block = [64, 1, 1], shared_mem_bytes = 0 }
config = { block_m = 128, block_n = 64, block_k = 32, waves_m = 1, waves_n = 1, waves_k = 1, k_splits = 1, lds_bytes = 0, vgprs = 0, unknown_tile_knob = 123 }
"#;
    let err_tune = toml::from_str::<TuneFile>(bad_tune_toml).unwrap_err();
    assert!(
        err_tune
            .to_string()
            .contains("unknown field `unknown_tile_knob`"),
        "unexpected tune err: {err_tune}"
    );
}

#[test]
fn test_load_module_rejects_unvalidated_variant_and_cached_hash_cannot_bypass() {
    let lib = get_stub_lib();
    let registry = Registry::new(RegistryConfig::default());
    let temp = TempDir::new("r9v_bypass_proof");
    let co_rel = "valid_kernel.co";
    let co_abs = temp.path().join(co_rel);
    std::fs::write(&co_abs, b"dummy co").unwrap();

    let valid_variant = ResolvedVariant {
        variant_hash: VariantHash::new(0xc001_dead),
        arch: ArchName::from("gfx942"),
        op: OpId::Matmul,
        tier: Tier::T2,
        entry_symbol: "local_matmul".to_string(),
        launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        workspace_bytes: 0,
        static_bytes: 0,
        static_flops: 0,
        code_object_path: Some(co_rel.to_string()),
        code_object_bytes: None,
        validated: true,
        artifact_origin: Some(ArtifactOrigin::Local {
            base_dir: Some(temp.path().to_path_buf()),
        }),
    };

    // 1. Load valid variant, populating the loaded_modules cache
    let mod1 = registry
        .load_module(&lib, &valid_variant)
        .expect("load valid module");

    // 2. Unvalidated variant with the SAME variant_hash must be rejected before cache fast path
    let mut unvalidated_variant = valid_variant.clone();
    unvalidated_variant.validated = false;

    let err = match registry.load_module(&lib, &unvalidated_variant) {
        Err(e) => e,
        Ok(_) => panic!("expected load_module error for unvalidated variant"),
    };
    match err {
        RegistryError::VariantNotValidated { hash, op, arch } => {
            assert_eq!(hash, 0xc001_dead);
            assert_eq!(op, OpId::Matmul);
            assert_eq!(arch, "gfx942");
        }
        other => panic!("expected VariantNotValidated, got {other:?}"),
    }

    // 3. Prove that valid variant still hits the cache and returns the identical Arc pointer
    let mod2 = registry
        .load_module(&lib, &valid_variant)
        .expect("load cached valid module");
    assert!(Arc::ptr_eq(&mod1, &mod2));
}

#[test]
fn test_load_tune_file_relative_code_object_requires_base_dir_atomic() {
    let mut registry = Registry::new(RegistryConfig {
        gen_version: 1,
        allow_jit: false,
        ..Default::default()
    });
    let mut manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    add_t1_fallback(&mut manifest, OpId::Matmul);
    registry.set_manifest(manifest, None).unwrap();

    let op_static = make_sample_matmul_static();
    let shash = static_hash(&op_static);

    let mut tune_with_co = TuneFile::new(ArchName::from("gfx942"), 1);
    tune_with_co.insert_entry(
        OpId::Matmul,
        shash,
        TuneEntry {
            config: TileConfig::new(128, 64, 32),
            median_us: 10.0,
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: Some("matmul_local.co".to_string()),
            validated: true,
            partial: false,
            measured_on: None,
        },
    );

    // 1. load_tune_file with base_dir = None MUST fail when code_object is Some
    let err = registry.load_tune_file(&tune_with_co, None).unwrap_err();
    match err {
        RegistryError::ValidationFailed { problems } => {
            assert!(
                problems
                    .iter()
                    .any(|p| p.contains("no base directory was provided")),
                "problems: {problems:?}"
            );
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }

    // 2. Atomic refusal: registry was NOT mutated (still falls back to T1, tune entry not registered)
    let res_t1 = registry
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("resolution falls back to T1");
    assert_eq!(res_t1.tier, Tier::T1);

    // 3. In-memory tune file with code_object = None and base_dir = None succeeds
    let mut in_memory_tune = TuneFile::new(ArchName::from("gfx942"), 1);
    in_memory_tune.insert_entry(
        OpId::Matmul,
        static_hash(&op_static),
        TuneEntry {
            config: TileConfig::new(128, 64, 32),
            median_us: 15.0,
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    assert!(registry.load_tune_file(&in_memory_tune, None).unwrap());

    // 4. load_tune_file_from_path supplies parent directory and succeeds with relative code objects
    let temp = TempDir::new("r9v_tune_disk");
    let tune_path = temp.path().join("tune.toml");
    tune_with_co.save_to_file(&tune_path).unwrap();

    let mut reg_from_path = Registry::new(RegistryConfig::default());
    let mut manifest_path = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    add_t1_fallback(&mut manifest_path, OpId::Matmul);
    reg_from_path.set_manifest(manifest_path, None).unwrap();

    assert!(reg_from_path.load_tune_file_from_path(&tune_path).unwrap());
    let res_disk = reg_from_path
        .resolve(OpId::Matmul, &ArchName::from("gfx942"), &op_static)
        .expect("tune from path resolves to T2");
    assert_eq!(res_disk.tier, Tier::T2);
    assert_eq!(
        res_disk.code_object_path.as_deref(),
        Some("matmul_local.co")
    );
    assert_eq!(
        res_disk.artifact_origin,
        Some(ArtifactOrigin::Local {
            base_dir: Some(temp.path().to_path_buf())
        })
    );
}

#[test]
fn test_tune_file_merge_local_atomic_refusal_and_mismatch() {
    let mut base = TuneFile::new(ArchName::from("gfx942"), 1);
    base.insert_entry(
        OpId::Matmul,
        100,
        TuneEntry {
            config: TileConfig::new(64, 64, 32),
            median_us: 20.0,
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );

    // 1. Invalid local tune file (non-positive median_us) is rejected atomically
    let mut invalid_local = TuneFile::new(ArchName::from("gfx942"), 1);
    invalid_local.insert_entry(
        OpId::Matmul,
        200,
        TuneEntry {
            config: TileConfig::new(64, 64, 32),
            median_us: -5.0, // invalid!
            bytes: 1024,
            flops: 2048,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );

    let err = base.merge_local(&invalid_local).unwrap_err();
    match err {
        RegistryError::ValidationFailed { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("median_us must be finite and positive")));
        }
        other => panic!("expected ValidationFailed, got {other:?}"),
    }
    // Verify atomic refusal: base was NOT mutated
    assert_eq!(base.entries.len(), 1);
    assert!(base.get_entry(OpId::Matmul, 100).is_some());
    assert!(base.get_entry(OpId::Matmul, 200).is_none());

    // 2. If base itself is invalid, merge_local fails before mutation
    let mut invalid_base = TuneFile::new(ArchName::from("gfx942"), 1);
    invalid_base.entries.insert(
        "bad_key".to_string(),
        TuneEntry {
            config: TileConfig::new(0, 0, 0), // invalid tile dimensions
            median_us: 10.0,
            bytes: 0,
            flops: 0,
            launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            workspace_bytes: 0,
            code_object: None,
            validated: true,
            partial: false,
            measured_on: None,
        },
    );
    let valid_local = TuneFile::new(ArchName::from("gfx942"), 1);
    assert!(invalid_base.merge_local(&valid_local).is_err());

    // 3. Arch mismatch returns Ok(false) without mutating base
    let mut base_clean = TuneFile::new(ArchName::from("gfx942"), 1);
    let diff_arch = TuneFile::new(ArchName::from("gfx1100"), 1);
    assert!(!base_clean.merge_local(&diff_arch).unwrap());

    // 4. Generator version mismatch returns Ok(false) without mutating base
    let diff_gen = TuneFile::new(ArchName::from("gfx942"), 2);
    assert!(!base_clean.merge_local(&diff_gen).unwrap());
}
