// SPDX-License-Identifier: Apache-2.0
//! Layout geometry integration tests (Spec 2 §2; card A2.1).
//!
//! Lane-formula examples, tile/row-block indexing, checked padding,
//! `L0` geometry, scale grouping, region offsets, stable codes with
//! `r9v_ir::LayoutId` compatibility, and collect-all refusal paths.

use r9v_format::{
    l0_region_bytes, l0_row_offset_bytes, l0_row_stride_bytes, l0_row_values_bytes, l1_elem,
    l1_forward_index, l1_inverse_index, l1_lane, row_block_tiles, scale_block_counts,
    scale_record_count, scale_region_bytes, tile_index, tile_origin, FormatError, Layout, Packing,
    PaddedDims, ELEMS_PER_LANE, ELEMS_PER_TILE, LANES_PER_TILE, TILE_K, TILE_N,
};

fn dims(n: u32, k: u32) -> PaddedDims {
    PaddedDims::new(n, k, None).expect("test dims are valid")
}

fn multiple_len(err: &FormatError) -> usize {
    match err {
        FormatError::Multiple { problems } => problems.len(),
        _ => panic!("expected Multiple, got {err:?}"),
    }
}

#[test]
fn lane_formula_matches_spec_2_2_examples() {
    // lane = kgroup * 16 + n
    assert_eq!(l1_lane(0, 0).unwrap(), 0);
    assert_eq!(l1_lane(15, 0).unwrap(), 15);
    assert_eq!(l1_lane(0, 1).unwrap(), 16);
    assert_eq!(l1_lane(5, 1).unwrap(), 21);
    // elem = lane * 8 + j
    assert_eq!(l1_elem(0, 0).unwrap(), 0);
    assert_eq!(l1_elem(21, 3).unwrap(), 171);
    assert_eq!(l1_elem(31, 7).unwrap(), 255);
    // value = W[n_base + n, k_base + kgroup*8 + j]: lane 21 holds row 5,
    // K 8..16, so (n, k) = (5, 11) sits at tile position 171.
    assert_eq!(l1_forward_index(5, 11, &dims(16, 16)).unwrap(), 171);
    assert_eq!(l1_inverse_index(171, &dims(16, 16)).unwrap(), (5, 11));
    // tile_index = (n_base/16) * (K/16) + (k_base/16): (16, 32) in a
    // 64x64 weight is tile 1*4 + 2 = 6.
    assert_eq!(tile_index(16, 32, &dims(64, 64)).unwrap(), 6);
    assert_eq!(tile_origin(6, &dims(64, 64)).unwrap(), (16, 32));
}

#[test]
fn tile_constants_match_spec_2_2() {
    assert_eq!(TILE_N, 16);
    assert_eq!(TILE_K, 16);
    assert_eq!(LANES_PER_TILE, 32);
    assert_eq!(ELEMS_PER_LANE, 8);
    assert_eq!(ELEMS_PER_TILE, 256);
}

#[test]
fn padded_dims_cover_every_shape_class() {
    // Already aligned stays put.
    let d = dims(16, 16);
    assert_eq!((d.n_padded(), d.k_padded()), (16, 16));
    assert_eq!(d.tile_count(), 1);
    // Single element pads to one tile.
    let d = dims(1, 1);
    assert_eq!((d.n_padded(), d.k_padded()), (16, 16));
    // Just over a tile spills to the next.
    let d = dims(17, 17);
    assert_eq!((d.n_padded(), d.k_padded()), (32, 32));
    assert_eq!((d.n_tiles(), d.k_tiles()), (2, 2));
    // Mixed classes.
    let d = dims(33, 300);
    assert_eq!((d.n_padded(), d.k_padded()), (48, 304));
    // Superblock widens K only.
    let d = PaddedDims::new(33, 40, Some(32)).unwrap();
    assert_eq!((d.n_padded(), d.k_padded()), (48, 64));
    let d = PaddedDims::new(33, 300, Some(256)).unwrap();
    assert_eq!((d.n_padded(), d.k_padded()), (48, 512));
    let d = PaddedDims::new(64, 256, Some(256)).unwrap();
    assert_eq!((d.n_padded(), d.k_padded()), (64, 256));
    // Superblock below the tile width still pads K to 16.
    let d = PaddedDims::new(5, 7, None).unwrap();
    assert_eq!((d.n_padded(), d.k_padded()), (16, 16));
}

#[test]
fn tile_origin_inverts_tile_index_for_every_shape_class() {
    for n in [1u32, 7, 15, 16, 17, 31, 32, 33, 48, 64] {
        for k in [1u32, 7, 15, 16, 17, 31, 32, 33, 48, 64] {
            let d = dims(n, k);
            for tile in 0..d.tile_count() {
                let (nb, kb) = tile_origin(tile, &d).unwrap();
                assert_eq!(tile_index(nb, kb, &d).unwrap(), tile, "n={n} k={k}");
            }
        }
    }
}

#[test]
fn forward_index_inverts_inverse_index_for_every_shape_class() {
    for n in [1u32, 15, 16, 17, 33, 64] {
        for k in [1u32, 15, 16, 17, 33, 64] {
            let d = dims(n, k);
            let total = d.n_padded() as u64 * d.k_padded() as u64;
            assert_eq!(total, d.tile_count() * ELEMS_PER_TILE);
            for pos in 0..total {
                let (rn, rk) = l1_inverse_index(pos, &d).unwrap();
                assert_eq!(l1_forward_index(rn, rk, &d).unwrap(), pos, "n={n} k={k}");
            }
        }
    }
}

#[test]
fn row_block_tiles_are_contiguous_streams_over_k() {
    let d = dims(32, 64);
    assert_eq!(d.row_blocks(), 2);
    assert_eq!(row_block_tiles(0, &d).unwrap(), (0, 4));
    assert_eq!(row_block_tiles(1, &d).unwrap(), (4, 8));
    // Every tile belongs to exactly one row-block.
    let mut seen = vec![false; d.tile_count() as usize];
    for nb in 0..d.row_blocks() as u32 {
        let (start, end) = row_block_tiles(nb, &d).unwrap();
        assert_eq!(end - start, d.k_tiles());
        for t in start..end {
            assert!(!seen[t as usize]);
            seen[t as usize] = true;
        }
    }
    assert!(seen.into_iter().all(|s| s));
}

#[test]
fn untrusted_dims_collect_every_problem() {
    // n, k and superblock all bad: all three reported, in order.
    let err = PaddedDims::new(0, 0, Some(7)).unwrap_err();
    assert_eq!(multiple_len(&err), 3);
    // Single failure stays a single error.
    assert!(matches!(
        PaddedDims::new(0, 5, None).unwrap_err(),
        FormatError::InvalidDim { name: "n", .. }
    ));
    assert!(matches!(
        PaddedDims::new(5, 5, Some(0)).unwrap_err(),
        FormatError::InvalidBlock {
            name: "superblock_k",
            ..
        }
    ));
    // Two failures: k and superblock.
    assert_eq!(
        multiple_len(&PaddedDims::new(5, 0, Some(0)).unwrap_err()),
        2
    );
}

#[test]
fn padding_arithmetic_overflows_instead_of_saturating() {
    assert!(matches!(
        PaddedDims::new(u32::MAX, 16, None).unwrap_err(),
        FormatError::Overflow { .. }
    ));
    assert!(matches!(
        PaddedDims::new(16, u32::MAX, None).unwrap_err(),
        FormatError::Overflow { .. }
    ));
    assert!(matches!(
        PaddedDims::new(16, 16, Some(u32::MAX)).unwrap_err(),
        FormatError::InvalidBlock { .. }
    ));
}

#[test]
fn l0_stride_offsets_and_sizes_match_hand_computation() {
    // dim 8 of 2-byte values with 2 K-blocks of 4-byte records:
    // stride = 16 + 8 = 24.
    assert_eq!(l0_row_values_bytes(8, 2).unwrap(), 16);
    assert_eq!(l0_row_stride_bytes(8, 2, 2, 4).unwrap(), 24);
    assert_eq!(l0_row_offset_bytes(0, 24).unwrap(), 0);
    assert_eq!(l0_row_offset_bytes(2, 24).unwrap(), 48);
    assert_eq!(l0_region_bytes(3, 24).unwrap(), 72);
}

#[test]
fn l0_geometry_collects_every_problem() {
    // All four stride inputs bad: values contributes two, blocks two.
    assert_eq!(
        multiple_len(&l0_row_stride_bytes(0, 0, 0, 0).unwrap_err()),
        4
    );
    assert!(matches!(
        l0_region_bytes(0, 24).unwrap_err(),
        FormatError::InvalidDim { name: "rows", .. }
    ));
}

#[test]
fn scale_grouping_matches_spec_3_1_layout() {
    // [N/16][K/B][16 records]: 32x128 with B=128 is 2x1 blocks.
    let d = dims(32, 128);
    assert_eq!(scale_block_counts(&d, 128).unwrap(), (2, 1));
    assert_eq!(scale_record_count(2, 1).unwrap(), 32);
    assert_eq!(scale_region_bytes(32, 2).unwrap(), 64);
    // B=32 over the same tensor is 2x4 blocks.
    assert_eq!(scale_block_counts(&d, 32).unwrap(), (2, 4));
    // B must be a nonzero multiple of 16 dividing padded K.
    assert!(matches!(
        scale_block_counts(&d, 0).unwrap_err(),
        FormatError::InvalidBlock { .. }
    ));
    assert!(matches!(
        scale_block_counts(&d, 24).unwrap_err(),
        FormatError::InvalidBlock { .. }
    ));
    assert!(matches!(
        scale_block_counts(&dims(32, 48), 32).unwrap_err(),
        FormatError::InvalidBlock { .. }
    ));
    assert!(matches!(
        scale_region_bytes(32, 0).unwrap_err(),
        FormatError::InvalidBlock { .. }
    ));
}

#[test]
fn stable_codes_match_ir_handles_and_round_trip() {
    assert_eq!(Layout::L0.code(), r9v_ir::LayoutId::L0.as_u64());
    assert_eq!(Layout::L1.code(), r9v_ir::LayoutId::L1.as_u64());
    assert_eq!(Layout::L1S.code(), r9v_ir::LayoutId::L1S.as_u64());
    assert_eq!(
        (Layout::CODE_L0, Layout::CODE_L1, Layout::CODE_L1S),
        (1, 2, 3)
    );
    for layout in [Layout::L0, Layout::L1, Layout::L1S] {
        assert_eq!(Layout::from_code(layout.code()).unwrap(), layout);
        assert_eq!(Layout::from_ir(layout.to_ir()).unwrap(), layout);
        assert_eq!(layout.to_string(), layout.name());
        assert_eq!(layout.name().parse::<Layout>().unwrap(), layout);
        assert_eq!(Layout::from_name(layout.name()).unwrap(), layout);
    }
    // IR codes outside the spec 2 §2 weight set are errors here.
    assert!(matches!(
        Layout::from_ir(r9v_ir::LayoutId::CONTIGUOUS).unwrap_err(),
        FormatError::UnknownLayout { .. }
    ));
    assert!(matches!(
        Layout::from_ir(r9v_ir::LayoutId::ATTENTION_GFX1201).unwrap_err(),
        FormatError::UnknownLayout { .. }
    ));
    assert!(matches!(
        Layout::from_code(0).unwrap_err(),
        FormatError::UnknownLayout { .. }
    ));
    assert!(matches!(
        Layout::from_code(99).unwrap_err(),
        FormatError::UnknownLayout { .. }
    ));
    assert!(matches!(
        Layout::from_name("L1").unwrap_err(),
        FormatError::UnknownLayout { .. }
    ));
    assert!(matches!(
        Layout::from_name("l2").unwrap_err(),
        FormatError::UnknownLayout { .. }
    ));
}

#[test]
fn packing_table_bytes_match_spec_2_2() {
    assert_eq!(Packing::Nibble4.bytes_per_lane(), 4);
    assert_eq!(Packing::Nibble4.tile_bytes(), 128);
    assert_eq!(Packing::Byte.bytes_per_lane(), 8);
    assert_eq!(Packing::Byte.tile_bytes(), 256);
    assert_eq!(Packing::Half16.bytes_per_lane(), 16);
    assert_eq!(Packing::Half16.tile_bytes(), 512);
    for (bits, bytes) in [(2u8, 2u64), (3, 3), (5, 5), (6, 6)] {
        let packing = Packing::bit_planes(bits).unwrap();
        assert_eq!(packing.bytes_per_lane(), bytes);
        assert_eq!(packing.tile_bytes(), bytes * 32);
    }
    for bits in [0u8, 1, 4, 7, 8, 16] {
        assert!(matches!(
            Packing::bit_planes(bits).unwrap_err(),
            FormatError::InvalidBitWidth { .. }
        ));
    }
}

#[test]
fn index_helpers_reject_out_of_range_coordinates() {
    let d = dims(32, 32);
    assert!(l1_lane(16, 0).is_err());
    assert!(l1_lane(0, 2).is_err());
    assert!(l1_elem(32, 0).is_err());
    assert!(l1_elem(0, 8).is_err());
    // Misaligned bases collect both problems.
    assert_eq!(multiple_len(&tile_index(1, 1, &d).unwrap_err()), 2);
    assert!(tile_origin(d.tile_count(), &d).is_err());
    // Out-of-range forward coordinates collect both problems.
    assert_eq!(
        multiple_len(&l1_forward_index(d.n_padded(), d.k_padded(), &d).unwrap_err()),
        2
    );
    assert!(l1_inverse_index(d.tile_count() * ELEMS_PER_TILE, &d).is_err());
    assert!(row_block_tiles(d.row_blocks() as u32, &d).is_err());
}
