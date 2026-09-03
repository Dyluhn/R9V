// SPDX-License-Identifier: Apache-2.0
//! Permutation round-trip property tests (Spec 2 §2.2, §2.3; card A2.1).
//!
//! Deterministic seeded tensors of every packing and shape class:
//! forward/inverse identity, padding-zero rules, region sizes and
//! offsets, and the `L1S` index region in lane order.

use r9v_common::SeededRng;
use r9v_format::{
    decode_halfs_le, encode_halfs_le, l1_forward_elems, l1_forward_index, l1_inverse_elems,
    l1_inverse_index, l1_pack_bytes, l1_pack_halfs, l1_pack_nibbles, l1_pack_planes,
    l1_unpack_bytes, l1_unpack_halfs, l1_unpack_nibbles, l1_unpack_planes, l1s_index_lane,
    l1s_index_region_bytes, l1s_pack_indices, l1s_unpack_indices, l1s_value_dims, lane_byte_offset,
    pad_row_major_elems, verify_padding_zeros_bytes, verify_padding_zeros_elems,
    verify_padding_zeros_nibbles, verify_padding_zeros_planes, FormatError, L1Regions, L1sRegions,
    Packing, PaddedDims, L1S_INDEX_BYTES_PER_TILE, L1S_KEPT_PER_LANE, L1S_KEPT_PER_TILE,
    LANES_PER_TILE,
};

const NS: [u32; 10] = [1, 7, 15, 16, 17, 31, 32, 33, 48, 64];
const KS: [u32; 10] = [1, 7, 15, 16, 17, 31, 32, 33, 48, 64];

fn all_dims() -> Vec<PaddedDims> {
    let mut out = Vec::new();
    for n in NS {
        for k in KS {
            out.push(PaddedDims::new(n, k, None).expect("test dims are valid"));
            out.push(PaddedDims::new(n, k, Some(32)).expect("test dims are valid"));
        }
    }
    out.push(PaddedDims::new(33, 300, Some(256)).expect("test dims are valid"));
    out
}

fn rand_u16(rng: &mut SeededRng, limit: u32) -> u16 {
    (rng.next_u64() % limit as u64) as u16
}

fn rand_elems(seed: u64, count: usize, limit: u32) -> Vec<u16> {
    let mut rng = SeededRng::new(seed);
    (0..count).map(|_| rand_u16(&mut rng, limit)).collect()
}

fn rand_bytes(seed: u64, count: usize) -> Vec<u8> {
    let mut rng = SeededRng::new(seed);
    (0..count).map(|_| (rng.next_u64() % 256) as u8).collect()
}

#[test]
fn element_permute_inverts_for_every_shape_class() {
    for (i, d) in all_dims().into_iter().enumerate() {
        let total = (d.n_padded() as u64 * d.k_padded() as u64) as usize;
        let src = rand_elems(0xA210_0000 + i as u64, total, 65536);
        let tiled = l1_forward_elems(&src, &d).unwrap();
        assert_eq!(l1_inverse_elems(&tiled, &d).unwrap(), src, "dims {d:?}");
        // Deterministic: the same input permutes to the same bytes.
        assert_eq!(l1_forward_elems(&src, &d).unwrap(), tiled, "dims {d:?}");
    }
}

#[test]
fn byte_permute_inverts_for_every_shape_class() {
    for (i, d) in all_dims().into_iter().enumerate() {
        let total = (d.n_padded() as u64 * d.k_padded() as u64) as usize;
        let src = rand_bytes(0xA210_1000 + i as u64, total);
        let tiled = l1_pack_bytes(&src, &d).unwrap();
        assert_eq!(l1_unpack_bytes(&tiled, &d).unwrap(), src, "dims {d:?}");
        assert_eq!(
            tiled.len() as u64,
            d.value_region_bytes(Packing::Byte).unwrap()
        );
    }
}

#[test]
fn half_permute_and_le_codec_invert_for_every_shape_class() {
    for (i, d) in all_dims().into_iter().enumerate() {
        let total = (d.n_padded() as u64 * d.k_padded() as u64) as usize;
        // Full raw-pattern range: no float math happens in layout code.
        let src = rand_elems(0xA210_2000 + i as u64, total, 65536);
        let tiled = l1_pack_halfs(&src, &d).unwrap();
        assert_eq!(l1_unpack_halfs(&tiled, &d).unwrap(), src, "dims {d:?}");
        let bytes = encode_halfs_le(&tiled);
        assert_eq!(bytes.len(), total * 2);
        assert_eq!(decode_halfs_le(&bytes).unwrap(), tiled, "dims {d:?}");
    }
    assert!(matches!(
        decode_halfs_le(&[0u8; 3]).unwrap_err(),
        FormatError::LengthMismatch { .. }
    ));
}

#[test]
fn nibble_permute_inverts_for_every_shape_class() {
    for (i, d) in all_dims().into_iter().enumerate() {
        let total = (d.n_padded() as u64 * d.k_padded() as u64) as usize;
        let src: Vec<u8> = rand_elems(0xA210_3000 + i as u64, total, 16)
            .into_iter()
            .map(|v| v as u8)
            .collect();
        let tiled = l1_pack_nibbles(&src, &d).unwrap();
        assert_eq!(tiled.len(), total / 2, "dims {d:?}");
        assert_eq!(
            tiled.len() as u64,
            d.value_region_bytes(Packing::Nibble4).unwrap()
        );
        assert_eq!(l1_unpack_nibbles(&tiled, &d).unwrap(), src, "dims {d:?}");
    }
}

#[test]
fn nibble_bytes_hold_low_nibble_lower_k() {
    // Single tile, row 0 = 0..16: lane 0 holds W[0, 0..8], so the first
    // four bytes are (1<<4|0), (3<<4|2), (5<<4|4), (7<<4|6).
    let d = PaddedDims::new(16, 16, None).unwrap();
    let mut src = vec![0u8; 256];
    for k in 0..16u8 {
        src[k as usize] = k;
    }
    let tiled = l1_pack_nibbles(&src, &d).unwrap();
    assert_eq!(&tiled[0..4], &[0x10, 0x32, 0x54, 0x76]);
    // Lane 1 (row 1, all zeros) starts at byte 4.
    assert_eq!(&tiled[4..8], &[0x00, 0x00, 0x00, 0x00]);
    // One 32-bit lane load reads exactly lane 0: bytes 0..4.
    assert_eq!(lane_byte_offset(0, Packing::Nibble4).unwrap(), 0);
    assert_eq!(lane_byte_offset(1, Packing::Nibble4).unwrap(), 4);
}

#[test]
fn nibble_values_outside_4_bits_are_all_reported() {
    let d = PaddedDims::new(16, 16, None).unwrap();
    let mut src = vec![0u8; 256];
    src[0] = 16;
    src[100] = 255;
    src[255] = 17;
    match l1_pack_nibbles(&src, &d).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 3),
        err => panic!("expected Multiple, got {err:?}"),
    }
}

#[test]
fn plane_permute_inverts_for_every_width_and_shape_class() {
    for bits in [2u8, 3, 5, 6] {
        let limit = 1u32 << bits;
        for (i, d) in all_dims().into_iter().enumerate() {
            let total = (d.n_padded() as u64 * d.k_padded() as u64) as usize;
            let src = rand_elems(
                0xA210_4000 + (bits as u64) * 10_000 + i as u64,
                total,
                limit,
            );
            let tiled = l1_pack_planes(&src, &d, bits).unwrap();
            let packing = Packing::bit_planes(bits).unwrap();
            assert_eq!(
                tiled.len() as u64,
                d.value_region_bytes(packing).unwrap(),
                "bits={bits} dims {d:?}"
            );
            assert_eq!(
                l1_unpack_planes(&tiled, &d, bits).unwrap(),
                src,
                "bits={bits} dims {d:?}"
            );
        }
    }
}

#[test]
fn plane_values_outside_width_are_all_reported() {
    let d = PaddedDims::new(16, 16, None).unwrap();
    let mut src = vec![0u16; 256];
    src[0] = 8;
    src[7] = 9;
    match l1_pack_planes(&src, &d, 3).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 2),
        err => panic!("expected Multiple, got {err:?}"),
    }
    assert!(matches!(
        l1_pack_planes(&src, &d, 4).unwrap_err(),
        FormatError::InvalidBitWidth { .. }
    ));
}

#[test]
fn padding_zero_rules_hold_for_every_packing() {
    let cases: [(u32, u32); 5] = [(1, 1), (7, 33), (16, 16), (17, 31), (48, 20)];
    for (case, (n, k)) in cases.into_iter().enumerate() {
        let seed = 0xA210_5000 + case as u64;
        let d = PaddedDims::new(n, k, None).unwrap();
        let unpadded = (n as usize) * (k as usize);
        // Elements and halves.
        let src = rand_elems(seed, unpadded, 65536);
        let padded = pad_row_major_elems(&src, n, k, &d).unwrap();
        let tiled = l1_forward_elems(&padded, &d).unwrap();
        verify_padding_zeros_elems(&tiled, &d).unwrap();
        // Bytes.
        let bsrc: Vec<u8> = src.iter().map(|v| (v % 251) as u8 + 1).collect();
        let mut bpadded = vec![0u8; (d.n_padded() as usize) * (d.k_padded() as usize)];
        for row in 0..n {
            for col in 0..k {
                bpadded[(row * d.k_padded() + col) as usize] = bsrc[(row * k + col) as usize];
            }
        }
        let btiled = l1_pack_bytes(&bpadded, &d).unwrap();
        verify_padding_zeros_bytes(&btiled, &d).unwrap();
        // Nibbles (nonzero data, zero pad).
        let nsrc: Vec<u8> = rand_elems(seed + 100, unpadded, 15)
            .into_iter()
            .map(|v| (v + 1) as u8)
            .collect();
        let mut npadded = vec![0u8; (d.n_padded() as usize) * (d.k_padded() as usize)];
        for row in 0..n {
            for col in 0..k {
                npadded[(row * d.k_padded() + col) as usize] = nsrc[(row * k + col) as usize];
            }
        }
        let ntiled = l1_pack_nibbles(&npadded, &d).unwrap();
        verify_padding_zeros_nibbles(&ntiled, &d).unwrap();
        // Planes at every width.
        for bits in [2u8, 3, 5, 6] {
            let limit = (1u16 << bits) - 1;
            let psrc = rand_elems(seed + 200 + bits as u64, unpadded, limit as u32 + 1);
            let mut ppadded = vec![0u16; (d.n_padded() as usize) * (d.k_padded() as usize)];
            for row in 0..n {
                for col in 0..k {
                    ppadded[(row * d.k_padded() + col) as usize] = psrc[(row * k + col) as usize];
                }
            }
            // Keep padding zero while data may hit the limit: mask data
            // values into range without touching the zero pad.
            let ptiled = l1_pack_planes(&ppadded, &d, bits).unwrap();
            verify_padding_zeros_planes(&ptiled, &d, bits).unwrap();
        }
    }
}

#[test]
fn nonzero_padding_reports_every_position() {
    let d = PaddedDims::new(17, 17, None).unwrap();
    let total = (d.n_padded() as usize) * (d.k_padded() as usize);
    let mut tiled = vec![0u16; total];
    // Corrupt two padding cells: (31, 16) is K-pad, (16, 31) is N-pad.
    tiled[l1_index_of(31, 16, &d)] = 7;
    tiled[l1_index_of(16, 31, &d)] = 9;
    // Row-major scan order: (16, 31) reports before (31, 16).
    match verify_padding_zeros_elems(&tiled, &d).unwrap_err() {
        FormatError::Multiple { problems } => {
            assert_eq!(problems.len(), 2);
            assert!(matches!(
                problems[0],
                FormatError::PaddingNonzero {
                    row: 16,
                    col: 31,
                    value: 9
                }
            ));
            assert!(matches!(
                problems[1],
                FormatError::PaddingNonzero {
                    row: 31,
                    col: 16,
                    value: 7
                }
            ));
        }
        err => panic!("expected Multiple, got {err:?}"),
    }
    // A single nonzero pad cell is a single error.
    let mut one = vec![0u16; total];
    one[l1_index_of(17, 17, &d)] = 1;
    assert!(matches!(
        verify_padding_zeros_elems(&one, &d).unwrap_err(),
        FormatError::PaddingNonzero {
            row: 17,
            col: 17,
            value: 1
        }
    ));
}

fn l1_index_of(n: u32, k: u32, d: &PaddedDims) -> usize {
    r9v_format::l1_forward_index(n, k, d).expect("test positions are valid") as usize
}

#[test]
fn short_and_long_buffers_are_rejected_with_sizes() {
    let d = PaddedDims::new(16, 16, None).unwrap();
    assert!(matches!(
        l1_pack_bytes(&[0u8; 255], &d).unwrap_err(),
        FormatError::LengthMismatch {
            expected: 256,
            got: 255,
            ..
        }
    ));
    assert!(matches!(
        l1_unpack_halfs(&[0u16; 257], &d).unwrap_err(),
        FormatError::LengthMismatch {
            expected: 256,
            got: 257,
            ..
        }
    ));
    assert!(matches!(
        l1_pack_nibbles(&[0u8; 10], &d).unwrap_err(),
        FormatError::LengthMismatch { .. }
    ));
    assert!(matches!(
        l1_pack_planes(&[0u16; 10], &d, 5).unwrap_err(),
        FormatError::LengthMismatch { .. }
    ));
    assert!(matches!(
        pad_row_major_elems(&[0u16; 4], 3, 3, &d).unwrap_err(),
        FormatError::LengthMismatch { .. }
    ));
}

#[test]
fn l1s_index_region_round_trips_in_lane_order() {
    let dense = PaddedDims::new(32, 64, None).unwrap();
    let value = l1s_value_dims(&dense, None).unwrap();
    assert_eq!((value.n(), value.k()), (32, 32));
    assert_eq!((value.n_padded(), value.k_padded()), (32, 32));
    assert_eq!(value.tile_count(), dense.tile_count() / 2);
    assert_eq!(
        l1s_index_region_bytes(value.tile_count()).unwrap(),
        value.tile_count() * 64
    );
    for (i, vd) in [value, PaddedDims::new(16, 16, None).unwrap()]
        .into_iter()
        .enumerate()
    {
        let kept = (vd.tile_count() * 256) as usize;
        let src: Vec<u8> = rand_elems(0xA210_6000 + i as u64, kept, 4)
            .into_iter()
            .map(|v| v as u8)
            .collect();
        let bytes = l1s_pack_indices(&src, &vd).unwrap();
        assert_eq!(
            bytes.len() as u64,
            l1s_index_region_bytes(vd.tile_count()).unwrap()
        );
        assert_eq!(l1s_unpack_indices(&bytes, &vd).unwrap(), src);
    }
}

#[test]
fn l1s_index_bytes_pack_slot_zero_lowest() {
    // One compressed tile. Lane 0 slots [0,1,2,3,3,2,1,0] pin the bit
    // order across the two lane bytes: low = 0b11_10_01_00,
    // high = 0b00_01_10_11. Lane 1 pins the per-lane byte offset.
    let vd = PaddedDims::new(16, 16, None).unwrap();
    let mut kept = vec![0u8; 256];
    kept[0..8].copy_from_slice(&[0, 1, 2, 3, 3, 2, 1, 0]);
    kept[8..16].copy_from_slice(&[1, 1, 1, 1, 2, 2, 2, 2]);
    let bytes = l1s_pack_indices(&kept, &vd).unwrap();
    assert_eq!(bytes.len(), 64);
    assert_eq!(bytes[0], 0b11_10_01_00);
    assert_eq!(bytes[1], 0b00_01_10_11);
    assert_eq!(bytes[2], 0x55);
    assert_eq!(bytes[3], 0xAA);
    assert_eq!(&bytes[4..64], &[0u8; 60]);
}

#[test]
fn l1s_index_values_outside_2_bits_are_all_reported() {
    let vd = PaddedDims::new(16, 16, None).unwrap();
    let mut kept = vec![0u8; 256];
    kept[0] = 4;
    kept[255] = 9;
    match l1s_pack_indices(&kept, &vd).unwrap_err() {
        FormatError::Multiple { problems } => assert_eq!(problems.len(), 2),
        err => panic!("expected Multiple, got {err:?}"),
    }
    assert!(matches!(
        l1s_pack_indices(&[0u8; 10], &vd).unwrap_err(),
        FormatError::LengthMismatch { .. }
    ));
}

#[test]
fn l1_and_l1s_region_offsets_follow_spec_2_6_order() {
    // Dense L1: values at 0, scales after values.
    let d = PaddedDims::new(32, 64, None).unwrap();
    let regions = L1Regions::new(&d, Packing::Byte.tile_bytes(), 64).unwrap();
    assert_eq!(regions.values_bytes, 8 * 256);
    assert_eq!(regions.values_offset(), 0);
    assert_eq!(regions.scales_offset(), 8 * 256);
    assert_eq!(regions.total_bytes().unwrap(), 8 * 256 + 64);
    // L1S: values at 0, scales next, indices last.
    let value = l1s_value_dims(&d, None).unwrap();
    let sparse = L1sRegions::new(&value, Packing::Byte.tile_bytes(), 64).unwrap();
    assert_eq!(sparse.values_bytes, 4 * 256);
    assert_eq!(sparse.indices_bytes, 4 * 64);
    assert_eq!(sparse.values_offset(), 0);
    assert_eq!(sparse.scales_offset(), 4 * 256);
    assert_eq!(sparse.indices_offset().unwrap(), 4 * 256 + 64);
    assert_eq!(sparse.total_bytes().unwrap(), 4 * 256 + 64 + 4 * 64);
}

#[test]
fn l1s_value_dims_halves_k_and_converts_superblock() {
    // Headline case: dense (64, 256, SB 256) maps to value K padded 128.
    let dense = PaddedDims::new(64, 256, Some(256)).unwrap();
    assert_eq!((dense.n_padded(), dense.k_padded()), (64, 256));
    let value = l1s_value_dims(&dense, Some(256)).unwrap();
    assert_eq!((value.n(), value.k()), (64, 128));
    assert_eq!((value.n_padded(), value.k_padded()), (64, 128));
    assert_eq!(value.tile_count(), dense.tile_count() / 2);
    // SB 32 halves to 16 and the dense padded K halves exactly.
    let dense = PaddedDims::new(48, 96, Some(32)).unwrap();
    assert_eq!(dense.k_padded(), 96);
    let value = l1s_value_dims(&dense, Some(32)).unwrap();
    assert_eq!((value.n_padded(), value.k_padded()), (48, 48));
    assert_eq!(value.k_padded(), dense.k_padded() / 2);
    assert_eq!(value.tile_count(), dense.tile_count() / 2);
    // No superblock: compress-then-pad over the unpadded K.
    let dense = PaddedDims::new(32, 64, None).unwrap();
    let value = l1s_value_dims(&dense, None).unwrap();
    assert_eq!((value.n(), value.k()), (32, 32));
    assert_eq!((value.n_padded(), value.k_padded()), (32, 32));
    // Odd dense K is refused: there is no whole compressed column.
    let odd = PaddedDims::new(16, 33, None).unwrap();
    assert!(matches!(
        l1s_value_dims(&odd, None).unwrap_err(),
        FormatError::InvalidDim { name: "k", .. }
    ));
    // Dense superblocks that do not halve to a tile-aligned block are
    // refused, including values PaddedDims alone would accept.
    let dense = PaddedDims::new(64, 256, Some(256)).unwrap();
    for bad in [0u32, 7, 16, 24, 48] {
        assert!(
            matches!(
                l1s_value_dims(&dense, Some(bad)).unwrap_err(),
                FormatError::InvalidBlock {
                    name: "superblock_k",
                    ..
                }
            ),
            "SB {bad}"
        );
    }
    // Odd K plus a bad block reports both problems, never just one.
    assert!(matches!(
        l1s_value_dims(&odd, Some(48)).unwrap_err(),
        FormatError::Multiple { .. }
    ));
}

#[test]
fn l1s_every_stored_slot_maps_to_its_index_bits() {
    // One compressed tile holds 32 lanes x 8 slots = 256 kept values;
    // each slot's 2 bits live at byte lane*2 (+1 for slots 4..8).
    // The (n, k) mapping runs through the verified dense lane law over
    // the compressed tile, not through the pack implementation.
    let vd = PaddedDims::new(16, 16, None).unwrap();
    assert_eq!(vd.tile_count(), 1);
    assert_eq!(L1S_KEPT_PER_TILE, 256);
    assert_eq!(L1S_INDEX_BYTES_PER_TILE, 64);
    let mut seen = vec![false; 256];
    for lane in 0..LANES_PER_TILE {
        for slot in 0..L1S_KEPT_PER_LANE {
            let pos = lane * 8 + slot;
            let (n, k) = l1_inverse_index(pos as u64, &vd).unwrap();
            assert_eq!(l1_forward_index(n, k, &vd).unwrap(), pos as u64);
            // The index lane for this value is its compressed-K lane.
            assert_eq!(l1s_index_lane(n, k / 8).unwrap(), lane);
            assert!(!seen[pos as usize], "lane {lane} slot {slot}");
            seen[pos as usize] = true;
        }
    }
    assert!(seen.into_iter().all(|s| s));
    // A lane-distinct pattern pins every lane's byte pair end to end.
    let mut kept = vec![0u8; L1S_KEPT_PER_TILE as usize];
    for lane in 0..LANES_PER_TILE as usize {
        for slot in 0..L1S_KEPT_PER_LANE as usize {
            kept[lane * L1S_KEPT_PER_LANE as usize + slot] = ((lane + slot) % 4) as u8;
        }
    }
    let bytes = l1s_pack_indices(&kept, &vd).unwrap();
    assert_eq!(bytes.len() as u64, L1S_INDEX_BYTES_PER_TILE);
    for lane in 0..LANES_PER_TILE as usize {
        let mut expected: u16 = 0;
        for slot in 0..L1S_KEPT_PER_LANE as usize {
            expected |=
                ((kept[lane * L1S_KEPT_PER_LANE as usize + slot] & 0x03) as u16) << (slot * 2);
        }
        assert_eq!(
            bytes[lane * 2],
            (expected & 0xFF) as u8,
            "lane {lane} low byte"
        );
        assert_eq!(
            bytes[lane * 2 + 1],
            (expected >> 8) as u8,
            "lane {lane} high byte"
        );
    }
    assert_eq!(l1s_unpack_indices(&bytes, &vd).unwrap(), kept);
}

#[test]
fn l1s_non_multiple_shapes_tile_counts_and_regions() {
    // Dense (18, 36) pads to (32, 48) = 6 tiles; compressed (18, 18)
    // pads to (32, 32) = 4 tiles.
    let dense = PaddedDims::new(18, 36, None).unwrap();
    assert_eq!((dense.n_padded(), dense.k_padded()), (32, 48));
    assert_eq!(dense.tile_count(), 6);
    let value = l1s_value_dims(&dense, None).unwrap();
    assert_eq!((value.n(), value.k()), (18, 18));
    assert_eq!((value.n_padded(), value.k_padded()), (32, 32));
    assert_eq!(value.tile_count(), 4);
    assert_eq!(l1s_index_region_bytes(value.tile_count()).unwrap(), 4 * 64);
    // Region sizes follow Spec 2 section 6 order: values, scales, indices.
    let regions = L1sRegions::new(&value, Packing::Byte.tile_bytes(), 16).unwrap();
    assert_eq!(regions.values_bytes, 4 * 256);
    assert_eq!(regions.indices_bytes, 4 * 64);
    assert_eq!(regions.values_offset(), 0);
    assert_eq!(regions.scales_offset(), 4 * 256);
    assert_eq!(regions.indices_offset().unwrap(), 4 * 256 + 16);
    assert_eq!(regions.total_bytes().unwrap(), 4 * 256 + 16 + 4 * 64);
    // Superblock-256 headline shape regions pin the converted block.
    let dense = PaddedDims::new(64, 256, Some(256)).unwrap();
    let value = l1s_value_dims(&dense, Some(256)).unwrap();
    assert_eq!(value.tile_count(), 32);
    let regions = L1sRegions::new(&value, Packing::Byte.tile_bytes(), 64).unwrap();
    assert_eq!(regions.values_bytes, 32 * 256);
    assert_eq!(regions.indices_bytes, 32 * 64);
    assert_eq!(regions.scales_offset(), 32 * 256);
    assert_eq!(regions.indices_offset().unwrap(), 32 * 256 + 64);
    assert_eq!(regions.total_bytes().unwrap(), 32 * 256 + 64 + 32 * 64);
}
