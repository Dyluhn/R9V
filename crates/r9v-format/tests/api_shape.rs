// SPDX-License-Identifier: Apache-2.0
//! Public-surface shape tests (Spec 2 §2; card A2.1).
//!
//! Visibility, `Send`/`Sync`, exhaustive closed sets, stable names,
//! and IR-handle compatibility.

use r9v_format::{FormatError, L1Regions, L1sRegions, Layout, Packing, PaddedDims};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_types_are_send_and_sync() {
    assert_send_sync::<Layout>();
    assert_send_sync::<Packing>();
    assert_send_sync::<PaddedDims>();
    assert_send_sync::<FormatError>();
    assert_send_sync::<L1Regions>();
    assert_send_sync::<L1sRegions>();
    assert_send_sync::<r9v_ir::LayoutId>();
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
