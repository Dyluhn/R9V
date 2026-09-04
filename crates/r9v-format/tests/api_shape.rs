// SPDX-License-Identifier: Apache-2.0
//! Public-surface shape tests (Spec 2 §2; card A2.1).
//!
//! Visibility, `Send`/`Sync`, exhaustive closed sets, stable names,
//! and IR-handle compatibility.

use r9v_format::{FormatError, L1Regions, L1sRegions, Layout, Packing, PaddedDims};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn gguf_types_are_send_and_sync() {
    assert_send_sync::<r9v_format::GgmlType>();
    assert_send_sync::<r9v_format::RepackedTensor>();
}

#[test]
fn gguf_type_is_plain_data() {
    assert_plain_data::<r9v_format::GgmlType>();
}

#[test]
fn public_types_are_send_and_sync() {
    assert_send_sync::<Layout>();
    assert_send_sync::<Packing>();
    assert_send_sync::<PaddedDims>();
    assert_send_sync::<FormatError>();
    assert_send_sync::<L1Regions>();
    assert_send_sync::<L1sRegions>();
    assert_send_sync::<r9v_ir::LayoutId>();
    assert_send_sync::<r9v_format::SchemeId>();
    assert_send_sync::<r9v_format::ScaleGeometry>();
    assert_send_sync::<r9v_format::I8RowScale>();
    assert_send_sync::<r9v_format::I8Block128Scale>();
    assert_send_sync::<r9v_format::E4M3Block128Scale>();
    assert_send_sync::<r9v_format::I4KSuperblock>();
    assert_send_sync::<r9v_format::E4m3>();
    assert_send_sync::<r9v_format::QuantValue>();
    assert_send_sync::<r9v_format::ScaleSet>();
    assert_send_sync::<r9v_ir::SchemeId>();
}

#[test]
fn closed_sets_match_exhaustively_without_wildcards() {
    fn layout_name(layout: Layout) -> &'static str {
        match layout {
            Layout::L0 => "l0",
            Layout::L1 => "l1",
            Layout::L1S => "l1s",
        }
    }
    fn packing_is_planes(packing: Packing) -> bool {
        match packing {
            Packing::Nibble4 | Packing::Byte | Packing::Half16 => false,
            Packing::BitPlanes { .. } => true,
        }
    }
    assert_eq!(layout_name(Layout::L1S), "l1s");
    assert!(packing_is_planes(Packing::bit_planes(6).unwrap()));
    assert!(!packing_is_planes(Packing::Byte));
}

fn assert_plain_data<
    T: Send + Sync + Copy + Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug,
>() {
}

#[test]
fn scheme_types_are_plain_data() {
    assert_plain_data::<r9v_format::SchemeId>();
    assert_plain_data::<r9v_format::ScaleGeometry>();
    assert_plain_data::<r9v_format::I8RowScale>();
    assert_plain_data::<r9v_format::I8Block128Scale>();
    assert_plain_data::<r9v_format::E4M3Block128Scale>();
    assert_plain_data::<r9v_format::I4KSuperblock>();
    assert_plain_data::<r9v_format::E4m3>();
    assert_plain_data::<r9v_format::QuantValue>();
}

#[test]
fn scheme_id_covers_every_spec_table_row() {
    // Exhaustive with no wildcard: adding a scheme breaks this compile.
    fn code(id: r9v_format::SchemeId) -> u64 {
        match id {
            r9v_format::SchemeId::I8R => 1,
            r9v_format::SchemeId::I8B128 => 2,
            r9v_format::SchemeId::I4K => 3,
            r9v_format::SchemeId::E4M3B128 => 4,
            r9v_format::SchemeId::I8B32F => 5,
            r9v_format::SchemeId::I4B32F => 6,
            r9v_format::SchemeId::I4B32FM => 7,
            r9v_format::SchemeId::I5B32F => 8,
            r9v_format::SchemeId::I5B32FM => 9,
            r9v_format::SchemeId::I5K => 10,
            r9v_format::SchemeId::I6K => 11,
            r9v_format::SchemeId::I3K => 12,
            r9v_format::SchemeId::I2K => 13,
            r9v_format::SchemeId::I4Nl => 14,
            r9v_format::SchemeId::I4Xs => 15,
            r9v_format::SchemeId::Iq3Xxs => 16,
            r9v_format::SchemeId::Iq3S => 17,
            r9v_format::SchemeId::Iq2Xxs => 18,
            r9v_format::SchemeId::Iq2Xs => 19,
            r9v_format::SchemeId::Iq2S => 20,
            r9v_format::SchemeId::Iq1S => 21,
            r9v_format::SchemeId::Iq1M => 22,
        }
    }
    for id in r9v_format::SchemeId::ALL {
        assert_eq!(code(id), id.code());
    }
}

#[test]
fn scheme_errors_carry_scheme_and_reason() {
    let text = format!(
        "{}",
        FormatError::UnknownScheme {
            value: "q9_z".to_owned()
        }
    );
    assert!(text.contains("q9_z"));
    let text = format!(
        "{}",
        FormatError::ReservedScheme {
            scheme: "i8_b32f",
            owner: "A2.3"
        }
    );
    assert!(text.contains("i8_b32f"));
    assert!(text.contains("A2.3"));
    let text = format!(
        "{}",
        FormatError::SchemeMismatch {
            scheme: "i4_k",
            expected: "u4 + i4_k",
            got: "i8 + f16"
        }
    );
    assert!(text.contains("i4_k"));
    let text = format!(
        "{}",
        FormatError::InvalidScale {
            scheme: "i8_b128",
            record: 5,
            bits: f32::NAN.to_bits(),
            reason: "nan"
        }
    );
    assert!(text.contains("i8_b128"));
    let text = format!(
        "{}",
        FormatError::UnsupportedLayout {
            scheme: "i8_b128",
            layout: "l0"
        }
    );
    assert!(text.contains("l0"));
}

#[test]
fn error_type_composes_and_displays() {
    let err = FormatError::LengthMismatch {
        what: "x",
        expected: 1,
        got: 2,
    };
    let text = format!("{err}");
    assert!(text.contains("expected 1 bytes"));
    assert!(text.contains("got 2"));
    // Transparent in `?` chains: Debug + Error + Clone + Eq hold.
    let cloned = err.clone();
    assert_eq!(err, cloned);
    let _: &dyn std::error::Error = &cloned;
}

#[test]
fn container_types_are_send_and_sync() {
    assert_send_sync::<r9v_format::GgufFile>();
    assert_send_sync::<r9v_format::GgufWriter>();
    assert_send_sync::<r9v_format::KvEntry>();
    assert_send_sync::<r9v_format::KvType>();
    assert_send_sync::<r9v_format::KvValue>();
    assert_send_sync::<r9v_format::TensorInfo>();
    assert_send_sync::<r9v_format::TensorType>();
    assert_send_sync::<r9v_format::R9vTensorType>();
    assert_send_sync::<r9v_format::OutTensor>();
    assert_send_sync::<r9v_format::EntryRegions>();
    assert_send_sync::<r9v_format::ShardSet>();
    assert_send_sync::<r9v_format::R9vMeta>();
    assert_send_sync::<r9v_format::TensorMeta>();
}

#[test]
fn container_plain_data_types() {
    assert_plain_data::<r9v_format::KvType>();
    assert_plain_data::<r9v_format::TensorType>();
    assert_plain_data::<r9v_format::R9vTensorType>();
    assert_plain_data::<r9v_format::EntryRegions>();
}

#[test]
fn tensor_type_covers_every_upstream_code() {
    // Exhaustive with no wildcard: a new upstream code breaks this
    // compile and forces a sizing decision.
    fn code(ty: r9v_format::TensorType) -> u32 {
        match ty {
            r9v_format::TensorType::F32 => 0,
            r9v_format::TensorType::F16 => 1,
            r9v_format::TensorType::Q4_0 => 2,
            r9v_format::TensorType::Q4_1 => 3,
            r9v_format::TensorType::Q5_0 => 6,
            r9v_format::TensorType::Q5_1 => 7,
            r9v_format::TensorType::Q8_0 => 8,
            r9v_format::TensorType::Q8_1 => 9,
            r9v_format::TensorType::Q2_K => 10,
            r9v_format::TensorType::Q3_K => 11,
            r9v_format::TensorType::Q4_K => 12,
            r9v_format::TensorType::Q5_K => 13,
            r9v_format::TensorType::Q6_K => 14,
            r9v_format::TensorType::Q8_K => 15,
            r9v_format::TensorType::IQ2_XXS => 16,
            r9v_format::TensorType::IQ2_XS => 17,
            r9v_format::TensorType::IQ3_XXS => 18,
            r9v_format::TensorType::IQ1_S => 19,
            r9v_format::TensorType::IQ4_NL => 20,
            r9v_format::TensorType::IQ3_S => 21,
            r9v_format::TensorType::IQ2_S => 22,
            r9v_format::TensorType::IQ4_XS => 23,
            r9v_format::TensorType::I8 => 24,
            r9v_format::TensorType::I16 => 25,
            r9v_format::TensorType::I32 => 26,
            r9v_format::TensorType::I64 => 27,
            r9v_format::TensorType::F64 => 28,
            r9v_format::TensorType::IQ1_M => 29,
            r9v_format::TensorType::BF16 => 30,
            r9v_format::TensorType::TQ1_0 => 34,
            r9v_format::TensorType::TQ2_0 => 35,
            r9v_format::TensorType::MXFP4 => 39,
            r9v_format::TensorType::NVFP4 => 40,
            r9v_format::TensorType::Q1_0 => 41,
            r9v_format::TensorType::R9v(t) => t.code(),
            r9v_format::TensorType::Unknown(code) => code,
        }
    }
    for ty in r9v_format::TensorType::ALL {
        assert_eq!(code(ty), ty.code());
        assert_eq!(r9v_format::TensorType::from_code(ty.code()), ty);
    }
}

#[test]
fn ir_handle_owns_codes_while_format_owns_semantics() {
    // r9v-ir transports the opaque handle; r9v-format assigns meaning.
    // Both agree on the three spec 2 §2 weight codes.
    for (layout, id) in [
        (Layout::L0, r9v_ir::LayoutId::L0),
        (Layout::L1, r9v_ir::LayoutId::L1),
        (Layout::L1S, r9v_ir::LayoutId::L1S),
    ] {
        assert_eq!(layout.to_ir(), id);
        assert_eq!(Layout::from_ir(id).unwrap(), layout);
    }
}
