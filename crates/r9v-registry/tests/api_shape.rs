// SPDX-License-Identifier: Apache-2.0
//! API shape and trait bound verification for r9v-registry (Spec 14 §2, r9v-card-work §6).

use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;

use r9v_registry::{
    dispatch_launch, static_hash, variant_hash, ArchName, ArtifactOrigin, AttentionStatic,
    BundleManifest, CollectivesStatic, ElementwiseStatic, LaunchEntry, LaunchGeometry, LaunchList,
    LaunchRecord, LinearAttnScanStatic, ManifestVariantEntry, MatmulStatic, MoeFfnStatic, OpId,
    OpStatic, PlacementKind, Registry, RegistryConfig, RegistryError, ResolvedVariant,
    SamplingMethod, SamplingStatic, ScanMode, StateWriteKvStatic, StubDevice, Tier, TileConfig,
    TuneEntry, TuneFile, TuneMeasuredOn, VariantHash, VariantKey,
};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_copy<T: Copy>() {}
fn assert_clone<T: Clone>() {}
fn assert_display<T: Display>() {}
fn assert_hash<T: Hash>() {}
fn assert_from_str<T: FromStr>() {}
fn assert_error<T: std::error::Error>() {}

#[test]
fn api_shape_trait_bounds() {
    // Registry and Config
    assert_send::<Registry>();
    assert_sync::<Registry>();

    assert_send::<RegistryConfig>();
    assert_sync::<RegistryConfig>();
    assert_clone::<RegistryConfig>();

    assert_send::<ResolvedVariant>();
    assert_sync::<ResolvedVariant>();
    assert_clone::<ResolvedVariant>();

    assert_send::<ArtifactOrigin>();
    assert_sync::<ArtifactOrigin>();
    assert_clone::<ArtifactOrigin>();

    // Manifest
    assert_send::<BundleManifest>();
    assert_sync::<BundleManifest>();
    assert_clone::<BundleManifest>();

    assert_send::<ManifestVariantEntry>();
    assert_sync::<ManifestVariantEntry>();
    assert_clone::<ManifestVariantEntry>();

    // Tune
    assert_send::<TuneFile>();
    assert_sync::<TuneFile>();
    assert_clone::<TuneFile>();

    assert_send::<TuneEntry>();
    assert_sync::<TuneEntry>();
    assert_clone::<TuneEntry>();

    assert_send::<TuneMeasuredOn>();
    assert_sync::<TuneMeasuredOn>();
    assert_clone::<TuneMeasuredOn>();

    // Launch
    assert_send::<LaunchList>();
    assert_sync::<LaunchList>();
    assert_clone::<LaunchList>();

    assert_send::<LaunchEntry>();
    assert_sync::<LaunchEntry>();
    assert_clone::<LaunchEntry>();

    assert_send::<LaunchRecord>();
    assert_sync::<LaunchRecord>();
    assert_clone::<LaunchRecord>();

    assert_send::<StubDevice>();
    assert_sync::<StubDevice>();

    // Types & Keys
    assert_send::<VariantKey>();
    assert_sync::<VariantKey>();
    assert_clone::<VariantKey>();

    assert_send::<VariantHash>();
    assert_sync::<VariantHash>();
    assert_copy::<VariantHash>();
    assert_clone::<VariantHash>();
    assert_display::<VariantHash>();
    assert_hash::<VariantHash>();
    assert_from_str::<VariantHash>();

    assert_send::<OpId>();
    assert_sync::<OpId>();
    assert_copy::<OpId>();
    assert_clone::<OpId>();
    assert_display::<OpId>();
    assert_hash::<OpId>();
    assert_from_str::<OpId>();

    assert_send::<ArchName>();
    assert_sync::<ArchName>();
    assert_clone::<ArchName>();
    assert_display::<ArchName>();
    assert_hash::<ArchName>();
    assert_from_str::<ArchName>();

    assert_send::<Tier>();
    assert_sync::<Tier>();
    assert_copy::<Tier>();
    assert_clone::<Tier>();
    assert_display::<Tier>();
    assert_hash::<Tier>();
    assert_from_str::<Tier>();

    assert_send::<LaunchGeometry>();
    assert_sync::<LaunchGeometry>();
    assert_copy::<LaunchGeometry>();
    assert_clone::<LaunchGeometry>();
    assert_hash::<LaunchGeometry>();

    assert_send::<TileConfig>();
    assert_sync::<TileConfig>();
    assert_clone::<TileConfig>();
    assert_hash::<TileConfig>();

    assert_send::<ScanMode>();
    assert_sync::<ScanMode>();
    assert_copy::<ScanMode>();
    assert_clone::<ScanMode>();
    assert_hash::<ScanMode>();

    // OpStatic Family Variants
    assert_send::<OpStatic>();
    assert_sync::<OpStatic>();
    assert_clone::<OpStatic>();

    assert_send::<MatmulStatic>();
    assert_sync::<MatmulStatic>();
    assert_clone::<MatmulStatic>();

    assert_send::<AttentionStatic>();
    assert_sync::<AttentionStatic>();
    assert_clone::<AttentionStatic>();

    assert_send::<LinearAttnScanStatic>();
    assert_sync::<LinearAttnScanStatic>();
    assert_clone::<LinearAttnScanStatic>();

    assert_send::<MoeFfnStatic>();
    assert_sync::<MoeFfnStatic>();
    assert_clone::<MoeFfnStatic>();

    assert_send::<ElementwiseStatic>();
    assert_sync::<ElementwiseStatic>();
    assert_clone::<ElementwiseStatic>();

    assert_send::<CollectivesStatic>();
    assert_sync::<CollectivesStatic>();
    assert_clone::<CollectivesStatic>();

    assert_send::<StateWriteKvStatic>();
    assert_sync::<StateWriteKvStatic>();
    assert_clone::<StateWriteKvStatic>();

    assert_send::<SamplingStatic>();
    assert_sync::<SamplingStatic>();
    assert_clone::<SamplingStatic>();

    // Closed enums
    assert_send::<PlacementKind>();
    assert_sync::<PlacementKind>();
    assert_copy::<PlacementKind>();
    assert_clone::<PlacementKind>();
    assert_hash::<PlacementKind>();

    assert_send::<SamplingMethod>();
    assert_sync::<SamplingMethod>();
    assert_copy::<SamplingMethod>();
    assert_clone::<SamplingMethod>();
    assert_hash::<SamplingMethod>();

    // Error
    assert_send::<RegistryError>();
    assert_sync::<RegistryError>();
    assert_display::<RegistryError>();
    assert_error::<RegistryError>();
}

#[test]
fn api_function_signatures() {
    let op_s = OpStatic::Matmul(MatmulStatic {
        m_bucket: 128,
        n: 128,
        k: 64,
        w_scheme: r9v_ir::QuantScheme::None,
        w_layout: r9v_ir::LayoutId::CONTIGUOUS,
        act_scheme: r9v_ir::QuantScheme::None,
        out_dtype: r9v_ir::DType::F16,
        epilogue: r9v_ir::Epilogue::None,
        interleave: false,
        sparse: false,
    });
    let key = VariantKey {
        op: OpId::Matmul,
        arch: ArchName::from("gfx942"),
        gen_version: 1,
        static_params: op_s.clone(),
        config: TileConfig::new(128, 128, 32),
    };
    let vh: VariantHash = variant_hash(&key);
    assert!(!vh.to_hex().is_empty());

    let sh: u64 = static_hash(&op_s);
    assert_ne!(sh, 0);

    let stub = StubDevice::new();
    let entry = LaunchEntry::new(
        vh,
        "matmul_kernel",
        LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        1024,
        4096,
        8192,
        vec![1, 2, 3],
    );
    dispatch_launch(&stub, &entry, None).expect("dispatch to stub should succeed");
    assert_eq!(stub.recorded_launches().unwrap().len(), 1);
}
