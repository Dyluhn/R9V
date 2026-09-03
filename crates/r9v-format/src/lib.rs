//! R9V tensor layouts, quantization schemes, container format, and repack rules (Spec 2, Spec 14 §2).
//!
//! Card A2.1 owns the Spec 2 §2 layout half: canonical [`Layout`] ids
//! with stable codes, the [`Packing`] element classes, checked
//! N/K/superblock padding, tile and row-block indexing in the
//! A0.S1-verified lane order, `L1` forward/inverse permutation for
//! every packing, and the `L1S` value/index regions. Schemes (§3),
//! the container (§6) and repack rules (§7) belong to later cards.

pub mod error;
pub mod layout;
pub mod permute;
pub mod sparse;

pub use error::FormatError;
pub use layout::{
    l0_region_bytes, l0_row_offset_bytes, l0_row_stride_bytes, l0_row_values_bytes, l1_elem,
    l1_forward_index, l1_inverse_index, l1_lane, row_block_tiles, scale_block_counts,
    scale_record_count, scale_region_bytes, tile_index, tile_origin, Layout, Packing, PaddedDims,
    ELEMS_PER_LANE, ELEMS_PER_TILE, LANES_PER_TILE, LANE_K, TILE_K, TILE_N,
};
pub use permute::{
    decode_halfs_le, encode_halfs_le, l1_forward_elems, l1_inverse_elems, l1_pack_bytes,
    l1_pack_halfs, l1_pack_nibbles, l1_pack_planes, l1_unpack_bytes, l1_unpack_halfs,
    l1_unpack_nibbles, l1_unpack_planes, lane_byte_offset, pad_row_major_elems,
    verify_padding_zeros_bytes, verify_padding_zeros_elems, verify_padding_zeros_nibbles,
    verify_padding_zeros_planes,
};
pub use sparse::{
    l1s_index_lane, l1s_index_region_bytes, l1s_pack_indices, l1s_unpack_indices, l1s_value_dims,
    L1Regions, L1sRegions, L1S_INDEX_BITS, L1S_INDEX_BYTES_PER_TILE, L1S_KEPT_PER_LANE,
    L1S_KEPT_PER_TILE,
};
