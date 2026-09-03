//! R9V tensor layouts, quantization schemes, container format, and repack rules (Spec 2, Spec 14 §2).
//!
//! Card A2.1 owns the Spec 2 §2 layout half: canonical [`Layout`] ids
//! with stable codes, the [`Packing`] element classes, checked
//! N/K/superblock padding, tile and row-block indexing in the
//! A0.S1-verified lane order, `L1` forward/inverse permutation for
//! every packing, and the `L1S` value/index regions. Schemes (§3),
//! the container (§6) and repack rules (§7) belong to later cards.
//!
//! Card A2.2 owns the Spec 2 §3.1, §3.2 and §8 native-scheme half:
//! the closed [`SchemeId`] set with stable codes, names and IR-handle
//! conversions, exact scale-record structs ([`records`]) with the
//! Q4_K-identical `I4_K` record, SoA placement ([`geometry`]), the
//! `f16`/`E4M3` codecs ([`scales`]), reference decode ([`decode()`]),
//! simple encoders ([`encode`]) and exact bits-per-weight.
//!
//! Card A2.3 owns the Spec 2 §3.3, §7 and §10 GGUF half: the
//! [`GgmlType`] source set with its [`SchemeId`] mapping, wire-block
//! parsing and source-side reference decode ([`ggml`]), pure repack
//! into canonical `L1` plus the exact inverse and the independent
//! repacked-side decode ([`mod@repack`]). The card-A2.2 native decode
//! surface stays native-only by design: its value/record algebra is
//! the §3.2 one, while repack types decode through
//! [`ggml_dequantize`] and [`repack_dequantize`].

pub mod decode;
pub mod encode;
pub mod error;
pub mod geometry;
pub mod ggml;
pub mod layout;
pub mod permute;
pub mod records;
pub mod repack;
pub mod scales;
pub mod scheme;
pub mod sparse;

pub use decode::{
    decode, decode_e4m3_block128, decode_i4k_superblock, decode_i8_block128, decode_i8_row,
    QuantValue, ScaleSet,
};
pub use encode::{encode_e4m3_block128, encode_i4k_superblock, encode_i8_block128, encode_i8_row};
pub use error::FormatError;
pub use geometry::{outer_block, scale_geometry, scale_record_bytes, ScaleGeometry};
pub use ggml::{bf16_to_f32, ggml_dequantize, unpack_k4_scales, unpack_q3_scales, GgmlType};
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
pub use records::{E4M3Block128Scale, I4KSuperblock, I8Block128Scale, I8RowScale};
pub use repack::{
    repack, repack_bits_per_weight, repack_dequantize, repack_outer_block, repack_packing,
    repack_record_bytes, unpack_repacked, RepackedTensor,
};
pub use scales::{
    check_f16_scale, check_f32_scale, f16_scale_bits, f16_to_f32, f32_to_f16_bits, E4m3,
};
pub use scheme::{bits_per_weight, SchemeId};
pub use sparse::{
    l1s_index_lane, l1s_index_region_bytes, l1s_pack_indices, l1s_unpack_indices, l1s_value_dims,
    L1Regions, L1sRegions, L1S_INDEX_BITS, L1S_INDEX_BYTES_PER_LANE, L1S_INDEX_BYTES_PER_TILE,
    L1S_KEPT_PER_LANE, L1S_KEPT_PER_TILE,
};
