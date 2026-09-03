// SPDX-License-Identifier: Apache-2.0
//! API-shape tests for r9v-ir (Spec 1 §2, App. A; r9v-card-work §6).
//!
//! Asserts compilation, visibility boundaries and the `Send`/`Sync` markers
//! downstream crates rely on. Closed-set enums are additionally matched
//! exhaustively (no wildcard) so an RFC-added variant breaks this test until
//! the new surface is handled deliberately.

use std::hash::Hash;

use r9v_ir::{
    ArchDescriptor, ArchFamily, BatchMeta, BatchMetaBuilder, Class, DType, Dim, GraphCapture,
    IrError, IrVersion, LayoutId, MatrixOp, Measured, P2pLink, P2pTransport, Placement, Positions,
    QuantScheme, RelRate, SchemeId, ShapeSymbol, ShardLayout, StateHandle, StateKind, Tensor,
    TreeMask, ValuDot, BLOCK_TABLE_SENTINEL,
};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_copy<T: Copy>() {}
fn assert_clone<T: Clone>() {}
fn assert_debug<T: std::fmt::Debug>() {}
fn assert_display<T: std::fmt::Display>() {}
fn assert_hash<T: Hash>() {}
fn assert_error<T: std::error::Error>() {}
fn assert_ord<T: Ord>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn api_shape_markers_and_errors() {
    assert_send::<DType>();
    assert_sync::<DType>();
    assert_copy::<DType>();
    assert_hash::<DType>();
    assert_display::<DType>();

    assert_send::<SchemeId>();
    assert_sync::<SchemeId>();
    assert_copy::<SchemeId>();
    assert_hash::<SchemeId>();
    assert_display::<SchemeId>();

    assert_send::<QuantScheme>();
    assert_sync::<QuantScheme>();
    assert_copy::<QuantScheme>();
    assert_hash::<QuantScheme>();

    assert_send::<LayoutId>();
    assert_sync::<LayoutId>();
    assert_copy::<LayoutId>();
    assert_hash::<LayoutId>();
    assert_display::<LayoutId>();

    assert_send::<Tensor>();
    assert_sync::<Tensor>();
    assert_clone::<Tensor>();
    assert_debug::<Tensor>();

    assert_send::<BatchMeta>();
    assert_sync::<BatchMeta>();
    assert_clone::<BatchMeta>();

    assert_send::<TreeMask>();
    assert_sync::<TreeMask>();
    assert_clone::<TreeMask>();

    assert_send::<StateHandle>();
    assert_sync::<StateHandle>();
    assert_copy::<StateHandle>();
    assert_hash::<StateHandle>();

    assert_send::<ArchDescriptor>();
    assert_sync::<ArchDescriptor>();
    assert_clone::<ArchDescriptor>();

    assert_send::<IrVersion>();
    assert_sync::<IrVersion>();
    assert_copy::<IrVersion>();
    assert_hash::<IrVersion>();
    assert_display::<IrVersion>();
    assert_ord::<IrVersion>();

    assert_send::<IrError>();
    assert_sync::<IrError>();
    assert_clone::<IrError>();
    assert_error::<IrError>();

    assert_send::<BatchMetaBuilder>();
    assert_sync::<BatchMetaBuilder>();

    assert_send_sync::<ShapeSymbol>();
    assert_send_sync::<Dim>();
    assert_send_sync::<Placement>();
    assert_send_sync::<ShardLayout>();
    assert_send_sync::<Class>();
    assert_send_sync::<Positions>();
    assert_send_sync::<StateKind>();
    assert_send_sync::<ArchFamily>();
    assert_send_sync::<RelRate>();
    assert_send_sync::<MatrixOp>();
    assert_send_sync::<ValuDot>();
    assert_send_sync::<GraphCapture>();
    assert_send_sync::<P2pTransport>();
    assert_send_sync::<P2pLink>();
    assert_send_sync::<Measured>();

    let _ = BLOCK_TABLE_SENTINEL;
}

#[test]
fn api_shape_closed_sets_match_exhaustively() {
    // No wildcard arms: adding a variant fails this test until handled.
    let dtype = DType::F32;
    let _ = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::E4m3 => "e4m3",
        DType::E5m2 => "e5m2",
        DType::I8 => "i8",
        DType::I4 => "i4",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::Bool => "bool",
    };

    let scheme = QuantScheme::None;
    let _ = match scheme {
        QuantScheme::None => 0,
        QuantScheme::PerRow => 1,
        QuantScheme::Scheme(_) => 2,
        QuantScheme::PerToken => 3,
        QuantScheme::PerBlock32 => 4,
    };

    let class = Class::Weight;
    let _ = match class {
        Class::Weight => 0,
        Class::Activation => 1,
        Class::State => 2,
        Class::Staging => 3,
        Class::Param => 4,
    };

    let sharding = ShardLayout::Replicated;
    let _ = match sharding {
        ShardLayout::Replicated => 0,
        ShardLayout::ColShard { .. } => 1,
        ShardLayout::RowShard { .. } => 2,
        ShardLayout::HeadShard { .. } => 3,
        ShardLayout::ExpertShard { .. } => 4,
        ShardLayout::Partial => 5,
    };

    let placement = Placement::Host;
    let _ = match placement {
        Placement::Device { .. } => 0,
        Placement::Host => 1,
        Placement::Tiered => 2,
    };

    let kind = StateKind::KvPaged;
    let _ = match kind {
        StateKind::KvPaged => 0,
        StateKind::KvLatent => 1,
        StateKind::Recurrent => 2,
        StateKind::ConvWindow => 3,
    };

    let family = ArchFamily::Rdna4;
    let _ = match family {
        ArchFamily::Rdna4 => 0,
        ArchFamily::Rdna3 => 1,
        ArchFamily::Cdna3 => 2,
        ArchFamily::Reference => 3,
        ArchFamily::Cpu => 4,
    };

    let dot = ValuDot::Dot4I32I8;
    let _ = match dot {
        ValuDot::Dot4I32I8 => 0,
        ValuDot::Dot2F32F16 => 1,
        ValuDot::Dot2F32Bf16 => 2,
    };

    let capture = GraphCapture::Supported;
    let _ = match capture {
        GraphCapture::Supported => 0,
        GraphCapture::Unstable => 1,
        GraphCapture::None => 2,
    };

    let transport = P2pTransport::Direct;
    let _ = match transport {
        P2pTransport::Direct => 0,
        P2pTransport::HostStaged => 1,
    };

    let dim = Dim::Concrete(1);
    let _ = match dim {
        Dim::Concrete(_) => 0,
        Dim::Symbolic(_) => 1,
    };

    let symbol = ShapeSymbol::T;
    let _ = match symbol {
        ShapeSymbol::T => 0,
        ShapeSymbol::S => 1,
        ShapeSymbol::Dm => 2,
        ShapeSymbol::Dff => 3,
        ShapeSymbol::H => 4,
        ShapeSymbol::Hkv => 5,
        ShapeSymbol::D => 6,
        ShapeSymbol::E => 7,
        ShapeSymbol::K => 8,
        ShapeSymbol::V => 9,
        ShapeSymbol::Np => 10,
        ShapeSymbol::L => 11,
    };

    let pos = Positions::PerToken(vec![]);
    let _ = match pos {
        Positions::PerToken(_) => 0,
        Positions::Mrope(_) => 1,
    };
}

#[test]
fn api_shape_constructors_are_reachable() {
    let scheme = SchemeId::new(7);
    assert_eq!(scheme.as_u64(), 7);

    let layout = LayoutId::new(9);
    assert_eq!(layout.as_u64(), 9);
    assert_eq!(LayoutId::L1, LayoutId::new(2));

    let handle = StateHandle::new(3, StateKind::KvLatent);
    assert_eq!(handle.layer(), 3);
    assert_eq!(handle.kind(), StateKind::KvLatent);

    let tensor = Tensor::new(
        vec![Dim::Concrete(4), Dim::Symbolic(ShapeSymbol::Dm)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect("valid tensor builds");
    assert_eq!(tensor.rank(), 2);

    let rate = RelRate::new(2.0).expect("positive rate builds");
    assert_eq!(rate.as_f32(), 2.0);

    let op = MatrixOp::new([16, 16, 16], DType::I8, DType::I8, DType::I32, rate)
        .expect("valid matrix op builds");
    assert_eq!(op.shape, [16, 16, 16]);

    let measured = Measured::empty();
    assert!(measured.is_empty());

    let link = P2pLink {
        peer_rank: 1,
        transport: P2pTransport::HostStaged,
        measured_gbps: None,
    };
    assert_eq!(link.peer_rank, 1);

    let gfx = ArchDescriptor::gfx1201();
    assert_eq!(gfx.name, "gfx1201");
    let cpu = ArchDescriptor::cpu();
    assert_eq!(cpu.family, ArchFamily::Cpu);

    assert_eq!(IrVersion::CURRENT, IrVersion::new(0, 1, 0));
    assert_eq!(IrVersion::CURRENT.to_string(), "0.1.0");
}
