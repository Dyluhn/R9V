// SPDX-License-Identifier: Apache-2.0
//! Repack rules from GGUF wire blocks into canonical `L1` (Spec 2 §7;
//! card A2.3).
//!
//! [`repack`] is a pure permutation of bytes plus, for 5/6/3/2-bit
//! types, a bit-plane regrouping (Spec 2 §7 step 4): values move from
//! row-major wire blocks into `L1` tile order via the card-A2.1
//! codecs, scales move verbatim into the §3.1 SoA record region
//! (`[N/16][K/B][16 records]`). No arithmetic happens on values.
//! [`unpack_repacked`] is the exact inverse (repacked bytes back to
//! GGUF wire bytes); [`repack_dequantize`] decodes the repacked form
//! through an independent byte path (tiled values plus SoA records,
//! never wire blocks) so the §10 round-trip compares two genuinely
//! different readers.

use crate::ggml::{bf16_to_f32, block_values, check_wire_f16, unpack_k4_scales, unpack_q3_scales};
use crate::ggml::{wire_geometry, GgmlType};
use crate::layout::{Packing, PaddedDims};
use crate::records::I4KSuperblock;
use crate::scales::f16_to_f32;
use crate::scheme::SchemeId;
use crate::FormatError;

/// Byte length of one SoA scale record for `scheme` (Spec 2 §3.1,
/// §3.3; cards A2.3/A2.4).
///
/// Native ids reuse the card-A2.2 sizes; `I4_K` shares its 16-byte
/// Q4_K-identical record. `Q3_K`/`Q2_K` records gather the GGUF scale
/// payloads ("as GGUF" in §3.3; see SI-57): 12 scale bytes plus `d`
/// for `I3_K` (wire order), 16 scale bytes plus `d`/`dmin` for `I2_K`
/// (reordered: the wire splits them around the value bytes). `I6_K`
/// stores `d` first, then the sixteen `i8` scales (Spec 2 §3.3 table
/// order). IQ records (card A2.4, SI-70) follow the gguf-py wire order
/// with the index payload removed: `I4_NL [d]`, `I4_XS
/// [d][scales_h][scales_l]`, `IQ3_XXS [d][scales]`, `IQ3_S
/// [d][qh][signs][scales]`, `IQ2_XXS [d]`, `IQ2_XS [d][scales]`,
/// `IQ2_S [d][signs][qh][scales]`, `IQ1_S [d][qh]`, `IQ1_M
/// [qh][scales]` (`d` packed in the scales).
pub fn repack_record_bytes(scheme: SchemeId) -> Result<u32, FormatError> {
    match scheme {
        SchemeId::I8R | SchemeId::I8B128 | SchemeId::E4M3B128 => {
            crate::geometry::scale_record_bytes(scheme)
        }
        SchemeId::I4K => Ok(I4KSuperblock::RECORD_BYTES as u32),
        SchemeId::I8B32F | SchemeId::I4B32F | SchemeId::I5B32F => Ok(2),
        SchemeId::I4B32FM | SchemeId::I5B32FM => Ok(4),
        SchemeId::I5K => Ok(16),
        SchemeId::I6K => Ok(18),
        SchemeId::I3K => Ok(14),
        SchemeId::I2K => Ok(20),
        SchemeId::I4Nl | SchemeId::Iq2Xxs => Ok(2),
        SchemeId::I4Xs => Ok(8),
        SchemeId::Iq3Xxs => Ok(34),
        SchemeId::Iq3S => Ok(46),
        SchemeId::Iq2Xs => Ok(10),
        SchemeId::Iq2S => Ok(50),
        SchemeId::Iq1S => Ok(18),
        SchemeId::Iq1M => Ok(24),
    }
}

/// Outer block `B` of the §3.1 SoA grouping for `scheme` (the wire
/// block where one exists). Native ids reuse card-A2.2 answers.
/// `IQ4_NL` groups per 32 like the other 32-block types; every other
/// IQ family groups per 256 (card A2.4).
pub fn repack_outer_block(scheme: SchemeId) -> Result<Option<u32>, FormatError> {
    match scheme {
        SchemeId::I8R | SchemeId::I8B128 | SchemeId::I4K | SchemeId::E4M3B128 => {
            crate::geometry::outer_block(scheme)
        }
        SchemeId::I8B32F
        | SchemeId::I4B32F
        | SchemeId::I4B32FM
        | SchemeId::I5B32F
        | SchemeId::I5B32FM
        | SchemeId::I4Nl => Ok(Some(32)),
        SchemeId::I5K
        | SchemeId::I6K
        | SchemeId::I3K
        | SchemeId::I2K
        | SchemeId::I4Xs
        | SchemeId::Iq3Xxs
        | SchemeId::Iq3S
        | SchemeId::Iq2Xxs
        | SchemeId::Iq2Xs
        | SchemeId::Iq2S
        | SchemeId::Iq1S
        | SchemeId::Iq1M => Ok(Some(256)),
    }
}

/// `L1` value packing for `scheme` (Spec 2 §2.2 table, §3.3).
/// Five/three/two-bit types regroup into per-tile bit planes in lane
/// order; 6-bit the same; everything else keeps its wire granularity.
/// The IQ4 types pack per-weight nibbles; the grid IQ types pack
/// index bytes over the index shape `[N, K/g]` (card A2.4, SI-70), so
/// their `Byte` packing applies to the index dims, not the weight
/// dims (see `crate::iq`).
pub fn repack_packing(scheme: SchemeId) -> Result<Packing, FormatError> {
    match scheme {
        SchemeId::I8R | SchemeId::I8B128 | SchemeId::E4M3B128 | SchemeId::I8B32F => {
            Ok(Packing::Byte)
        }
        SchemeId::I4K | SchemeId::I4B32F | SchemeId::I4B32FM | SchemeId::I4Nl | SchemeId::I4Xs => {
            Ok(Packing::Nibble4)
        }
        SchemeId::I5B32F | SchemeId::I5B32FM | SchemeId::I5K => Packing::bit_planes(5),
        SchemeId::I6K => Packing::bit_planes(6),
        SchemeId::I3K => Packing::bit_planes(3),
        SchemeId::I2K => Packing::bit_planes(2),
        SchemeId::Iq3Xxs
        | SchemeId::Iq3S
        | SchemeId::Iq2Xxs
        | SchemeId::Iq2Xs
        | SchemeId::Iq2S
        | SchemeId::Iq1S
        | SchemeId::Iq1M => Ok(Packing::Byte),
    }
}

/// Exact bits-per-weight as `(bits, weights)` for `k` weights
/// (Spec 2 §8 "including all scale overhead"; card A2.3).
///
/// Repack only regroups bytes, so each ratio equals its GGUF wire
/// size: `I8_B32F` 272/32 = 8.5, `I4_B32F` 144/32 = 4.5, `I4_B32FM`
/// 160/32 = 5.0, `I5_B32F` 176/32 = 5.5, `I5_B32FM` 192/32 = 6.0,
/// `I5_K` 1408/256 = 5.5, `I6_K` 1680/256 = 6.5625, `I3_K` 880/256 =
/// 3.4375, `I2_K` 672/256 = 2.625, `I4_NL` 144/32 = 4.5, `I4_XS`
/// 1088/256 = 4.25, `IQ3_XXS` 784/256 = 3.0625, `IQ3_S` 880/256 =
/// 3.4375, `IQ2_XXS` 528/256 = 2.0625, `IQ2_XS` 592/256 = 2.3125,
/// `IQ2_S` 656/256 = 2.5625, `IQ1_S` 400/256 = 1.5625, `IQ1_M`
/// 448/256 = 1.75. Native ids reuse the card-A2.2 answers.
pub fn repack_bits_per_weight(scheme: SchemeId, k: u32) -> Result<(u64, u64), FormatError> {
    match scheme {
        SchemeId::I8R | SchemeId::I8B128 | SchemeId::I4K | SchemeId::E4M3B128 => {
            crate::scheme::bits_per_weight(scheme, k)
        }
        SchemeId::I8B32F
        | SchemeId::I4B32F
        | SchemeId::I4B32FM
        | SchemeId::I5B32F
        | SchemeId::I5B32FM
        | SchemeId::I4Nl => {
            if k == 0 || !k.is_multiple_of(32) {
                return Err(FormatError::InvalidBlock {
                    name: "k",
                    value: k as u64,
                    reason: "must be a nonzero multiple of 32",
                });
            }
            // No wildcard: every admitted id has an explicit arm and
            // every other id is unreachable here, so it fails closed
            // instead of silently inheriting a bit width.
            let wire_bits: u64 = match scheme {
                SchemeId::I8B32F => 272,
                SchemeId::I4B32F => 144,
                SchemeId::I4B32FM => 160,
                SchemeId::I5B32F => 176,
                SchemeId::I5B32FM => 192,
                SchemeId::I4Nl => 144,
                SchemeId::I8R
                | SchemeId::I8B128
                | SchemeId::I4K
                | SchemeId::E4M3B128
                | SchemeId::I5K
                | SchemeId::I6K
                | SchemeId::I3K
                | SchemeId::I2K
                | SchemeId::I4Xs
                | SchemeId::Iq3Xxs
                | SchemeId::Iq3S
                | SchemeId::Iq2Xxs
                | SchemeId::Iq2Xs
                | SchemeId::Iq2S
                | SchemeId::Iq1S
                | SchemeId::Iq1M => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: scheme.name(),
                        expected: "32-block repack scheme",
                        got: scheme.name(),
                    });
                }
            };
            let bits =
                (k as u64 / 32)
                    .checked_mul(wire_bits)
                    .ok_or_else(|| FormatError::Overflow {
                        what: "repack bits_per_weight",
                        detail: format!("scheme={} k={k}", scheme.name()),
                    })?;
            Ok((bits, k as u64))
        }
        SchemeId::I5K
        | SchemeId::I6K
        | SchemeId::I3K
        | SchemeId::I2K
        | SchemeId::I4Xs
        | SchemeId::Iq3Xxs
        | SchemeId::Iq3S
        | SchemeId::Iq2Xxs
        | SchemeId::Iq2Xs
        | SchemeId::Iq2S
        | SchemeId::Iq1S
        | SchemeId::Iq1M => {
            if k == 0 || !k.is_multiple_of(256) {
                return Err(FormatError::InvalidBlock {
                    name: "k",
                    value: k as u64,
                    reason: "must be a nonzero multiple of 256",
                });
            }
            // No wildcard: every admitted id has an explicit arm and
            // every other id is unreachable here, so it fails closed
            // instead of silently inheriting a bit width.
            let wire_bits: u64 = match scheme {
                SchemeId::I5K => 1408,
                SchemeId::I6K => 1680,
                SchemeId::I3K => 880,
                SchemeId::I2K => 672,
                SchemeId::I4Xs => 1088,
                SchemeId::Iq3Xxs => 784,
                SchemeId::Iq3S => 880,
                SchemeId::Iq2Xxs => 528,
                SchemeId::Iq2Xs => 592,
                SchemeId::Iq2S => 656,
                SchemeId::Iq1S => 400,
                SchemeId::Iq1M => 448,
                SchemeId::I8R
                | SchemeId::I8B128
                | SchemeId::I4K
                | SchemeId::E4M3B128
                | SchemeId::I8B32F
                | SchemeId::I4B32F
                | SchemeId::I4B32FM
                | SchemeId::I5B32F
                | SchemeId::I5B32FM
                | SchemeId::I4Nl => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: scheme.name(),
                        expected: "K-block repack scheme",
                        got: scheme.name(),
                    });
                }
            };
            let bits =
                (k as u64 / 256)
                    .checked_mul(wire_bits)
                    .ok_or_else(|| FormatError::Overflow {
                        what: "repack bits_per_weight",
                        detail: format!("scheme={} k={k}", scheme.name()),
                    })?;
            Ok((bits, k as u64))
        }
    }
}

/// One repacked tensor: `L1` tile-order values plus the §3.1 SoA
/// scale region (Spec 2 §7 steps 3–4; card A2.3).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepackedTensor {
    /// The GGUF source type these bytes came from.
    pub ggml: GgmlType,
    /// The repack scheme, or `None` for unquantized `F16`/`BF16`
    /// halves (SI-26).
    pub scheme: Option<SchemeId>,
    /// Padded `L1` dims both regions are sized for.
    pub dims: PaddedDims,
    /// `L1` tile-order value bytes for [`PaddedDims::value_region_bytes`].
    pub values: Vec<u8>,
    /// SoA scale records in `[N/16][K/B][16]` order, empty for halves.
    pub scales: Vec<u8>,
}

/// Repacks GGUF wire bytes into canonical `L1` (Spec 2 §7; card A2.3).
///
/// `wire` holds `n_rows` rows of `k` values in row-major wire-block
/// order. Values permute into `L1` tile order (bit-plane regrouping
/// for 5/6/3/2-bit types); scale bytes move verbatim into the SoA
/// region. Padding rows and columns are zero in both regions (Spec 2
/// §2.2). All input errors are collected and reported before any
/// output is built (CONVENTIONS.md §1.4).
pub fn repack(
    ggml: GgmlType,
    wire: &[u8],
    n_rows: u32,
    k: u32,
) -> Result<RepackedTensor, FormatError> {
    // Card-A2.4 IQ types repack through the codebook rules in
    // `crate::iq` (packed indices plus LUTs, not per-weight logicals).
    if crate::iq::is_iq(ggml) {
        return crate::iq::iq_repack(ggml, wire, n_rows, k);
    }
    let geo = wire_geometry(ggml, wire, n_rows, k)?;
    let dims = PaddedDims::new(n_rows, k, ggml.superblock_k())?;
    let block_len = ggml.block_len() as usize;
    let block_bytes = ggml.block_bytes() as usize;
    // Row-major logical values over the padded shape, zero-filled so
    // padding is zero by construction (Spec 2 §2.2).
    let n_pad = dims.n_padded() as usize;
    let k_pad = dims.k_padded() as usize;
    let mut logical = vec![0u16; n_pad * k_pad];
    let mut parsed = vec![0u16; block_len];
    let row_blocks = geo.blocks_per_row as usize;
    for row in 0..n_rows as usize {
        for b in 0..row_blocks {
            let base = (row * row_blocks + b) * block_bytes;
            block_values(ggml, &wire[base..base + block_bytes], &mut parsed)?;
            let dest = row * k_pad + b * block_len;
            logical[dest..dest + block_len].copy_from_slice(&parsed);
        }
    }
    let values = pack_logical(ggml, &logical, &dims)?;
    let scales = pack_scales(ggml, wire, &dims, &geo)?;
    Ok(RepackedTensor {
        ggml,
        scheme: ggml.scheme(),
        dims,
        values,
        scales,
    })
}

/// `L1` value region for padded row-major `logical` (Spec 2 §2.2
/// table via the card-A2.1 codecs; §3.3 bit-plane regrouping).
fn pack_logical(
    ggml: GgmlType,
    logical: &[u16],
    dims: &PaddedDims,
) -> Result<Vec<u8>, FormatError> {
    match ggml {
        GgmlType::F16 | GgmlType::BF16 => crate::permute::l1_pack_halfs(logical, dims)
            .map(|tiled| crate::permute::encode_halfs_le(&tiled)),
        GgmlType::Q8_0 => {
            let bytes: Vec<u8> = logical.iter().map(|v| *v as u8).collect();
            crate::permute::l1_pack_bytes(&bytes, dims)
        }
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K => {
            let nibbles: Vec<u8> = logical.iter().map(|v| *v as u8).collect();
            crate::permute::l1_pack_nibbles(&nibbles, dims)
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q5_K => {
            crate::permute::l1_pack_planes(logical, dims, 5)
        }
        GgmlType::Q6_K => crate::permute::l1_pack_planes(logical, dims, 6),
        GgmlType::Q3_K => crate::permute::l1_pack_planes(logical, dims, 3),
        GgmlType::Q2_K => crate::permute::l1_pack_planes(logical, dims, 2),
        GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M => {
            // Unreachable: repack routes IQ types to crate::iq before
            // packing logicals. Fails closed rather than guessing a
            // plane width for packed codebook indices.
            Err(FormatError::SchemeMismatch {
                scheme: ggml.name(),
                expected: "card-A2.3 logical packing (iq packs via crate::iq)",
                got: ggml.name(),
            })
        }
    }
}

/// SoA scale region in `[N/16][K/B][16]` order (Spec 2 §3.1, §7 step
/// 4): record bytes move verbatim from their wire block; padding rows
/// beyond `n_rows` get zero records.
fn pack_scales(
    ggml: GgmlType,
    wire: &[u8],
    dims: &PaddedDims,
    geo: &crate::ggml::WireGeometry,
) -> Result<Vec<u8>, FormatError> {
    let scheme = match ggml.scheme() {
        Some(s) => s,
        None => return Ok(Vec::new()),
    };
    let record = repack_record_bytes(scheme)? as usize;
    let outer = match repack_outer_block(scheme)? {
        Some(b) => b as usize,
        None => dims.k_padded() as usize,
    };
    let n_blocks = dims.n_padded() as usize / 16;
    let k_blocks = dims.k_padded() as usize / outer;
    let block_bytes = ggml.block_bytes() as usize;
    let wire_blocks_per_row = geo.blocks_per_row as usize;
    let mut out = vec![0u8; n_blocks * k_blocks * 16 * record];
    let mut scratch = vec![0u8; record];
    for nb in 0..n_blocks {
        for kb in 0..k_blocks {
            for row in 0..16 {
                let source = nb * 16 + row;
                let dest = ((nb * k_blocks + kb) * 16 + row) * record;
                if source < geo.n_rows as usize {
                    let base = (source * wire_blocks_per_row + kb) * block_bytes;
                    copy_block_record(ggml, &wire[base..base + block_bytes], &mut scratch)?;
                    out[dest..dest + record].copy_from_slice(&scratch);
                }
            }
        }
    }
    Ok(out)
}

/// Copies one wire block's scale record into SoA order.
///
/// Most records are verbatim wire slices at their [`record_span`];
/// `Q6_K` reorders to the §3.3 table order (`d` first, then the sixteen
/// `i8` scales), and `Q2_K` gathers `[scales][d][dmin]` across the split
/// wire layout. The two special cases never consult [`record_span`]:
/// no single wire span holds their record, so the span function returns
/// `None` for them rather than a plausible-looking lie.
// DECISION(A2.3): I6_K record is [d:2][sc:16] and I2_K record is
// [scales:16][d:2][dmin:2] (both reordered from wire order); rejected
// wire-order records because the §3.3 table lists I6_K as d-first and
// SoA readers index by table order. Q3_K keeps wire order. Per SI-57.
fn copy_block_record(ggml: GgmlType, block: &[u8], record: &mut [u8]) -> Result<(), FormatError> {
    match ggml {
        GgmlType::Q6_K => {
            // Record order is d first (Spec 2 §3.3 table); the wire
            // block carries scales first and d last. Length is checked
            // here, not via record_span (which has no span for Q6_K).
            if record.len() != 18 {
                return Err(FormatError::LengthMismatch {
                    what: "repack scale record",
                    expected: 18,
                    got: record.len() as u64,
                });
            }
            let wire_d = match block.get(208..210) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q6_k wire scales",
                        expected: 210,
                        got: block.len() as u64,
                    });
                }
            };
            let wire_sc = match block.get(192..208) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q6_k wire scales",
                        expected: 208,
                        got: block.len() as u64,
                    });
                }
            };
            record[0..2].copy_from_slice(wire_d);
            record[2..18].copy_from_slice(wire_sc);
            Ok(())
        }
        GgmlType::Q2_K => {
            // Record order is scales, d, dmin; the wire block splits
            // them around the 64 value bytes. Length is checked here,
            // not via record_span (which has no span for Q2_K).
            if record.len() != 20 {
                return Err(FormatError::LengthMismatch {
                    what: "repack scale record",
                    expected: 20,
                    got: record.len() as u64,
                });
            }
            let wire_scales = match block.get(0..16) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q2_k wire scales",
                        expected: 16,
                        got: block.len() as u64,
                    });
                }
            };
            let wire_d = match block.get(80..84) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q2_k wire scales",
                        expected: 84,
                        got: block.len() as u64,
                    });
                }
            };
            record[0..16].copy_from_slice(wire_scales);
            record[16..20].copy_from_slice(wire_d);
            Ok(())
        }
        GgmlType::F16 | GgmlType::BF16 => Err(FormatError::SchemeMismatch {
            scheme: ggml.name(),
            expected: "quantized ggml block",
            got: ggml.name(),
        }),
        GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M => Err(FormatError::SchemeMismatch {
            scheme: ggml.name(),
            expected: "card-A2.3 scale record (iq records via crate::iq)",
            got: ggml.name(),
        }),
        GgmlType::Q8_0
        | GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q3_K => {
            // Verbatim wire slice at the record_span. Halves never reach
            // here (pack_scales returns early for scheme None) and fail
            // closed above; Q6_K/Q2_K are handled in their own arms.
            let (start, len) = match record_span(ggml) {
                Some(span) => span,
                None => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "contiguous wire scale span",
                        got: ggml.name(),
                    });
                }
            };
            if record.len() != len {
                return Err(FormatError::LengthMismatch {
                    what: "repack scale record",
                    expected: len as u64,
                    got: record.len() as u64,
                });
            }
            let src = match block.get(start..start + len) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "ggml wire block",
                        expected: (start + len) as u64,
                        got: block.len() as u64,
                    });
                }
            };
            record.copy_from_slice(src);
            Ok(())
        }
    }
}

/// Validates a [`RepackedTensor`] against its own geometry and
/// returns `(packing, record_bytes, outer_block, n_blocks, k_blocks)`.
/// A hand-built tensor with inconsistent regions is a length error,
/// never a panic (CONVENTIONS.md §1.5).
fn repacked_geometry(
    t: &RepackedTensor,
) -> Result<(Packing, usize, usize, usize, usize), FormatError> {
    // Card-A2.4 IQ tensors validate through crate::iq (nibble or
    // index-shape value regions, not A2.3 logicals).
    if crate::iq::is_iq(t.ggml) {
        return Err(FormatError::SchemeMismatch {
            scheme: t.ggml.name(),
            expected: "card-A2.3 repacked tensor (iq validates via crate::iq)",
            got: t.ggml.name(),
        });
    }
    let dims = &t.dims;
    let (packing, record, outer) = match t.scheme {
        Some(scheme) => (
            repack_packing(scheme)?,
            repack_record_bytes(scheme)? as usize,
            match repack_outer_block(scheme)? {
                Some(b) => b as usize,
                None => dims.k_padded() as usize,
            },
        ),
        None => (Packing::Half16, 0, dims.k_padded() as usize),
    };
    if t.ggml.scheme() != t.scheme {
        return Err(FormatError::SchemeMismatch {
            scheme: match t.scheme {
                Some(s) => s.name(),
                None => t.ggml.name(),
            },
            expected: match t.ggml.scheme() {
                Some(s) => s.name(),
                None => "unquantized halves",
            },
            got: match t.scheme {
                Some(s) => s.name(),
                None => "unquantized halves",
            },
        });
    }
    let expected_values = dims.value_region_bytes(packing)?;
    if t.values.len() as u64 != expected_values {
        return Err(FormatError::LengthMismatch {
            what: "repacked value region",
            expected: expected_values,
            got: t.values.len() as u64,
        });
    }
    let n_blocks = dims.n_padded() as usize / 16;
    let k_blocks = dims.k_padded() as usize / outer;
    let expected_scales = n_blocks * k_blocks * 16 * record;
    if t.scales.len() != expected_scales {
        return Err(FormatError::LengthMismatch {
            what: "repacked scale region",
            expected: expected_scales as u64,
            got: t.scales.len() as u64,
        });
    }
    Ok((packing, record, outer, n_blocks, k_blocks))
}

/// Unpacks `L1` values to padded row-major logical values.
fn unpack_logical(t: &RepackedTensor, packing: Packing) -> Result<Vec<u16>, FormatError> {
    let dims = &t.dims;
    match packing {
        Packing::Byte => crate::permute::l1_unpack_bytes(&t.values, dims)
            .map(|v| v.into_iter().map(|b| b as u16).collect()),
        Packing::Nibble4 => crate::permute::l1_unpack_nibbles(&t.values, dims)
            .map(|v| v.into_iter().map(|b| b as u16).collect()),
        Packing::Half16 => {
            let tiled = crate::permute::decode_halfs_le(&t.values)?;
            crate::permute::l1_unpack_halfs(&tiled, dims)
        }
        Packing::BitPlanes { bits } => crate::permute::l1_unpack_planes(&t.values, dims, bits),
    }
}

/// Reads the SoA record for row-block `nb`, K-block `kb`, intra-block
/// row `row` (Spec 2 §3.1 grouping).
fn soa_record(
    scales: &[u8],
    record: usize,
    k_blocks: usize,
    nb: usize,
    kb: usize,
    row: usize,
) -> Result<&[u8], FormatError> {
    let base = ((nb * k_blocks + kb) * 16 + row) * record;
    match scales.get(base..base + record) {
        Some(s) => Ok(s),
        None => Err(FormatError::LengthMismatch {
            what: "repacked scale record",
            expected: (base + record) as u64,
            got: scales.len() as u64,
        }),
    }
}

/// Exact inverse of [`repack`] (Spec 2 §7 step 4): `L1` values unpack
/// to row-major logical values and SoA records regroup to wire-block
/// order, reproducing the input wire bytes exactly. Padding rows and
/// columns are dropped, never emitted.
pub fn unpack_repacked(t: &RepackedTensor) -> Result<Vec<u8>, FormatError> {
    // Card-A2.4 IQ types invert through `crate::iq` (their value
    // region holds nibbles or packed index bytes, not A2.3 logicals).
    if crate::iq::is_iq(t.ggml) {
        return crate::iq::iq_unpack(t);
    }
    let (packing, record, _outer, _n_blocks, k_blocks) = repacked_geometry(t)?;
    let ggml = t.ggml;
    let dims = &t.dims;
    let logical = unpack_logical(t, packing)?;
    let n_rows = dims.n() as usize;
    let k = dims.k() as usize;
    let block_len = ggml.block_len() as usize;
    let block_bytes = ggml.block_bytes() as usize;
    let wire_blocks_per_row = k / block_len;
    let mut wire = vec![0u8; n_rows * wire_blocks_per_row * block_bytes];
    let mut record_buf = vec![0u8; record];
    for row in 0..n_rows {
        for b in 0..wire_blocks_per_row {
            let vals = &logical[row * dims.k_padded() as usize + b * block_len..][..block_len];
            let rec = soa_record(&t.scales, record, k_blocks, row / 16, b, row % 16)?;
            record_buf.copy_from_slice(rec);
            let dest = (row * wire_blocks_per_row + b) * block_bytes;
            write_wire_block(ggml, vals, &record_buf, &mut wire[dest..dest + block_bytes])?;
        }
    }
    Ok(wire)
}

/// Emits one wire block from logical values plus its SoA record: the
/// exact inverse of [`block_values`][crate::ggml] plus
/// [`copy_block_record`].
fn write_wire_block(
    ggml: GgmlType,
    values: &[u16],
    record: &[u8],
    block: &mut [u8],
) -> Result<(), FormatError> {
    if values.len() != ggml.block_len() as usize {
        return Err(FormatError::LengthMismatch {
            what: "repack logical values",
            expected: ggml.block_len() as u64,
            got: values.len() as u64,
        });
    }
    if block.len() != ggml.block_bytes() as usize {
        return Err(FormatError::LengthMismatch {
            what: "ggml wire block",
            expected: ggml.block_bytes(),
            got: block.len() as u64,
        });
    }
    match ggml {
        GgmlType::F16 | GgmlType::BF16 => {
            for (i, v) in values.iter().enumerate() {
                let bytes = v.to_le_bytes();
                block[2 * i] = bytes[0];
                block[2 * i + 1] = bytes[1];
            }
        }
        GgmlType::Q8_0 => {
            let (rec, qs) = match (record.get(0..2), block.get_mut(0..34)) {
                (Some(r), Some(q)) => (r, q),
                _ => {
                    return Err(FormatError::LengthMismatch {
                        what: "q8_0 wire block",
                        expected: 34,
                        got: block.len() as u64,
                    });
                }
            };
            qs[0..2].copy_from_slice(rec);
            for (i, v) in values.iter().enumerate() {
                qs[2 + i] = *v as u8;
            }
        }
        GgmlType::Q4_0 | GgmlType::Q4_1 => {
            // The remaining arms are unreachable here (outer arm admits
            // only Q4_0/Q4_1) and fail closed rather than guessing an
            // offset.
            let base = match ggml {
                GgmlType::Q4_0 => 2,
                GgmlType::Q4_1 => 4,
                GgmlType::F16
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q2_K
                | GgmlType::Q3_K
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::IQ4_NL
                | GgmlType::IQ4_XS
                | GgmlType::IQ3_XXS
                | GgmlType::IQ3_S
                | GgmlType::IQ2_XXS
                | GgmlType::IQ2_XS
                | GgmlType::IQ2_S
                | GgmlType::IQ1_S
                | GgmlType::IQ1_M
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q4_0/1 block",
                        got: ggml.name(),
                    });
                }
            };
            let (rec, rest) = match (record.get(..), block.get_mut(..)) {
                (Some(r), Some(q)) => (r, q),
                _ => {
                    return Err(FormatError::LengthMismatch {
                        what: "q4_0/1 wire block",
                        expected: ggml.block_bytes(),
                        got: block.len() as u64,
                    });
                }
            };
            rest[0..rec.len()].copy_from_slice(rec);
            for j in 0..16 {
                rest[base + j] = values[j] as u8 | (values[j + 16] as u8) << 4;
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            // The remaining arms are unreachable here (outer arm admits
            // only Q5_0/Q5_1) and fail closed rather than guessing
            // offsets.
            let (qh_off, qs_off) = match ggml {
                GgmlType::Q5_0 => (2, 6),
                GgmlType::Q5_1 => (4, 8),
                GgmlType::F16
                | GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q8_0
                | GgmlType::Q2_K
                | GgmlType::Q3_K
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::IQ4_NL
                | GgmlType::IQ4_XS
                | GgmlType::IQ3_XXS
                | GgmlType::IQ3_S
                | GgmlType::IQ2_XXS
                | GgmlType::IQ2_XS
                | GgmlType::IQ2_S
                | GgmlType::IQ1_S
                | GgmlType::IQ1_M
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q5_0/1 block",
                        got: ggml.name(),
                    });
                }
            };
            let (rec, rest) = match (record.get(..), block.get_mut(..)) {
                (Some(r), Some(q)) => (r, q),
                _ => {
                    return Err(FormatError::LengthMismatch {
                        what: "q5_0/1 wire block",
                        expected: ggml.block_bytes(),
                        got: block.len() as u64,
                    });
                }
            };
            rest[0..rec.len()].copy_from_slice(rec);
            let mut qh: u32 = 0;
            for j in 0..16 {
                rest[qs_off + j] = values[j] as u8 & 0x0F | ((values[j + 16] as u8 & 0x0F) << 4);
                qh |= ((values[j] >> 4) as u32) << j;
                qh |= ((values[j + 16] >> 4) as u32) << (j + 16);
            }
            let bytes = qh.to_le_bytes();
            rest[qh_off..qh_off + 4].copy_from_slice(&bytes);
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let (rec, rest) = match (record.get(..), block.get_mut(..)) {
                (Some(r), Some(q)) => (r, q),
                _ => {
                    return Err(FormatError::LengthMismatch {
                        what: "q4_k/q5_k wire block",
                        expected: ggml.block_bytes(),
                        got: block.len() as u64,
                    });
                }
            };
            rest[0..16].copy_from_slice(rec);
            // The remaining arms are unreachable here (outer arm admits
            // only Q4_K/Q5_K) and fail closed rather than guessing the
            // payload offset.
            let qs_off = match ggml {
                GgmlType::Q5_K => 48,
                GgmlType::Q4_K => 16,
                GgmlType::F16
                | GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q2_K
                | GgmlType::Q3_K
                | GgmlType::Q6_K
                | GgmlType::IQ4_NL
                | GgmlType::IQ4_XS
                | GgmlType::IQ3_XXS
                | GgmlType::IQ3_S
                | GgmlType::IQ2_XXS
                | GgmlType::IQ2_XS
                | GgmlType::IQ2_S
                | GgmlType::IQ1_S
                | GgmlType::IQ1_M
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q4_k/q5_k block",
                        got: ggml.name(),
                    });
                }
            };
            for g in 0..4 {
                for b in 0..32 {
                    rest[qs_off + g * 32 + b] = values[(2 * g) * 32 + b] as u8 & 0x0F
                        | ((values[(2 * g + 1) * 32 + b] as u8 & 0x0F) << 4);
                }
            }
            if ggml == GgmlType::Q5_K {
                for p in 0..32 {
                    let mut byte: u8 = 0;
                    for j in 0..8 {
                        byte |= ((values[j * 32 + p] >> 4) as u8) << j;
                    }
                    rest[16 + p] = byte;
                }
            }
        }
        GgmlType::Q6_K => {
            let d = match record.get(0..2) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "i6_k record",
                        expected: 2,
                        got: record.len() as u64,
                    });
                }
            };
            let sc = match record.get(2..18) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "i6_k record",
                        expected: 18,
                        got: record.len() as u64,
                    });
                }
            };
            let rest = match block.get_mut(..) {
                Some(q) => q,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q6_k wire block",
                        expected: 210,
                        got: block.len() as u64,
                    });
                }
            };
            rest[192..208].copy_from_slice(sc);
            rest[208..210].copy_from_slice(d);
            for j in 0..16 {
                let big = j / 2;
                let pos0 = (j % 2) * 16;
                for p in 0..16 {
                    let big_p = pos0 + p;
                    let flat = big * 32 + big_p;
                    let g = flat / 128;
                    let s = (flat % 128) / 64;
                    let b = flat % 64;
                    rest[g * 64 + b] |= (values[j * 16 + p] as u8 & 0x0F) << (s * 4);
                    let group = big / 4;
                    let pair = big % 4;
                    rest[128 + group * 32 + big_p] |=
                        ((values[j * 16 + p] >> 4) as u8 & 0x03) << (pair * 2);
                }
            }
        }
        GgmlType::Q3_K | GgmlType::Q2_K => {
            // The remaining arms are unreachable here (outer arm admits
            // only Q3_K/Q2_K) and fail closed rather than guessing the
            // payload offset.
            let qs_off = match ggml {
                GgmlType::Q3_K => 32,
                GgmlType::Q2_K => 16,
                GgmlType::F16
                | GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::IQ4_NL
                | GgmlType::IQ4_XS
                | GgmlType::IQ3_XXS
                | GgmlType::IQ3_S
                | GgmlType::IQ2_XXS
                | GgmlType::IQ2_XS
                | GgmlType::IQ2_S
                | GgmlType::IQ1_S
                | GgmlType::IQ1_M
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q3_k/q2_k block",
                        got: ggml.name(),
                    });
                }
            };
            let rest = match block.get_mut(..) {
                Some(q) => q,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q3_k/q2_k wire block",
                        expected: ggml.block_bytes(),
                        got: block.len() as u64,
                    });
                }
            };
            if ggml == GgmlType::Q3_K {
                let rec = match record.get(..) {
                    Some(r) => r,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "q3_k record",
                            expected: 14,
                            got: record.len() as u64,
                        });
                    }
                };
                rest[96..110].copy_from_slice(rec);
            } else {
                let scales = match record.get(0..16) {
                    Some(r) => r,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "q2_k record",
                            expected: 16,
                            got: record.len() as u64,
                        });
                    }
                };
                let d = match record.get(16..20) {
                    Some(r) => r,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "q2_k record",
                            expected: 20,
                            got: record.len() as u64,
                        });
                    }
                };
                rest[0..16].copy_from_slice(scales);
                rest[80..84].copy_from_slice(d);
            }
            for j in 0..16 {
                let group = j / 8;
                let half = (j % 8) % 2;
                let shift = ((j % 8) / 2) * 2;
                for p in 0..16 {
                    rest[qs_off + group * 32 + half * 16 + p] |=
                        (values[j * 16 + p] as u8 & 0x03) << shift;
                }
            }
            if ggml == GgmlType::Q3_K {
                for j in 0..16 {
                    for p in 0..16 {
                        // Stored bit is the inverse of the value's
                        // high bit (see block_values).
                        let bit = (((values[j * 16 + p] >> 2) & 1) ^ 1) as u8;
                        rest[(j % 2) * 16 + p] |= bit << (j / 2);
                    }
                }
            }
        }
        GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M => {
            // Unreachable: unpack_repacked routes IQ types to
            // crate::iq before emitting wire blocks. Fails closed
            // rather than guessing a wire layout for packed indices.
            return Err(FormatError::SchemeMismatch {
                scheme: ggml.name(),
                expected: "card-A2.3 wire block (iq inverts via crate::iq)",
                got: ggml.name(),
            });
        }
    }
    Ok(())
}

/// Reference dequantization of repacked bytes (Spec 2 §3.3, §10;
/// card A2.3): decodes the `L1` value region plus the SoA scale
/// region to `n*k` row-major `f32` values, ascending.
///
/// This path never touches GGUF wire blocks: values come from tiles,
/// scales from SoA records, so the §10 round-trip
/// (`ggml_dequantize` vs `repack_dequantize`, bit-exact) compares two
/// independent readers. Formulas and evaluation order match
/// [`crate::ggml::ggml_dequantize`]; any divergence fails that test.
pub fn repack_dequantize(t: &RepackedTensor) -> Result<Vec<f32>, FormatError> {
    // Card-A2.4 IQ types decode through the independent codebook
    // reader in `crate::iq` (tiles plus SoA, never wire blocks).
    if crate::iq::is_iq(t.ggml) {
        return crate::iq::repacked_dequantize(t);
    }
    let (packing, record, _outer, _n_blocks, k_blocks) = repacked_geometry(t)?;
    let logical = unpack_logical(t, packing)?;
    let dims = &t.dims;
    let ggml = t.ggml;
    let n = dims.n() as usize;
    let k = dims.k() as usize;
    let k_pad = dims.k_padded() as usize;
    let block_len = ggml.block_len() as usize;
    let wire_blocks_per_row = k / block_len;
    let scheme_name = match ggml.scheme() {
        Some(s) => s.name(),
        None => ggml.name(),
    };
    let mut out = Vec::with_capacity(n * k);
    let mut problems = Vec::new();
    for row in 0..n {
        for b in 0..wire_blocks_per_row {
            let vals = &logical[row * k_pad + b * block_len..][..block_len];
            let index = (row * wire_blocks_per_row + b) as u64;
            match soa_record(&t.scales, record, k_blocks, row / 16, b, row % 16) {
                Ok(rec) => match dequant_parsed(ggml, scheme_name, index, rec, vals, &mut out) {
                    Ok(()) => {}
                    Err(FormatError::Multiple { problems: inner }) => {
                        problems.extend(inner.into_vec());
                    }
                    Err(single) => problems.push(single),
                },
                Err(e) => problems.push(e),
            }
        }
    }
    FormatError::collect(problems)?;
    Ok(out)
}

/// Decodes parsed logical values with one SoA record, appending to
/// `out` in ascending order. Record layouts are the
/// [`repack_record_bytes`] orderings (`I6_K` carries `d` first unlike
/// its wire block); formulas match the source-side decode exactly.
fn dequant_parsed(
    ggml: GgmlType,
    scheme_name: &'static str,
    index: u64,
    record: &[u8],
    values: &[u16],
    out: &mut Vec<f32>,
) -> Result<(), FormatError> {
    match ggml {
        GgmlType::F16 => {
            for v in values {
                out.push(f16_to_f32(*v));
            }
            Ok(())
        }
        GgmlType::BF16 => {
            for v in values {
                out.push(bf16_to_f32(*v));
            }
            Ok(())
        }
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_0 => {
            let d = check_wire_f16(scheme_name, index, u16_pair(record, 0, "repacked scale")?)?;
            for v in values {
                out.push(dequant_simple(ggml, d, 0.0, *v)?);
            }
            Ok(())
        }
        GgmlType::Q4_1 | GgmlType::Q5_1 => {
            let d = check_wire_f16(scheme_name, index, u16_pair(record, 0, "repacked scale")?)?;
            let m = check_wire_f16(scheme_name, index, u16_pair(record, 2, "repacked min")?)?;
            for v in values {
                out.push(dequant_simple(ggml, d, m, *v)?);
            }
            Ok(())
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let d = check_wire_f16(scheme_name, index, u16_pair(record, 0, "repacked scale")?)?;
            let dmin = check_wire_f16(
                scheme_name,
                index,
                u16_pair(record, 2, "repacked min scale")?,
            )?;
            let payload = match record.get(4..16) {
                Some(p) => p,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "repacked k4 payload",
                        expected: 16,
                        got: record.len() as u64,
                    });
                }
            };
            let (sc, mn) = unpack_k4_scales(payload)?;
            for j in 0..8 {
                let dl = d * sc[j] as f32;
                let ml = dmin * mn[j] as f32;
                for v in &values[j * 32..(j + 1) * 32] {
                    out.push(dl * *v as f32 - ml);
                }
            }
            Ok(())
        }
        GgmlType::Q6_K => {
            let d = check_wire_f16(scheme_name, index, u16_pair(record, 0, "repacked scale")?)?;
            let scales = match record.get(2..18) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "repacked i6_k scales",
                        expected: 18,
                        got: record.len() as u64,
                    });
                }
            };
            for j in 0..16 {
                let dl = d * scales[j] as i8 as f32;
                for v in &values[j * 16..(j + 1) * 16] {
                    out.push(dl * (*v as f32 - 32.0));
                }
            }
            Ok(())
        }
        GgmlType::Q3_K => {
            let payload = match record.get(0..12) {
                Some(p) => p,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "repacked q3_k payload",
                        expected: 12,
                        got: record.len() as u64,
                    });
                }
            };
            let sc = unpack_q3_scales(payload)?;
            let d = check_wire_f16(scheme_name, index, u16_pair(record, 12, "repacked scale")?)?;
            for j in 0..16 {
                let dl = d * sc[j] as f32;
                for v in &values[j * 16..(j + 1) * 16] {
                    let low = (*v & 3) as i8;
                    let high = ((*v >> 2) & 1) as i8;
                    out.push(dl * (low - (high << 2)) as f32);
                }
            }
            Ok(())
        }
        GgmlType::Q2_K => {
            let scales = match record.get(0..16) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "repacked q2_k scales",
                        expected: 16,
                        got: record.len() as u64,
                    });
                }
            };
            let d = check_wire_f16(scheme_name, index, u16_pair(record, 16, "repacked scale")?)?;
            let dmin = check_wire_f16(
                scheme_name,
                index,
                u16_pair(record, 18, "repacked min scale")?,
            )?;
            for j in 0..16 {
                let dl = d * (scales[j] & 0x0F) as f32;
                let ml = dmin * (scales[j] >> 4) as f32;
                for v in &values[j * 16..(j + 1) * 16] {
                    out.push(dl * *v as f32 - ml);
                }
            }
            Ok(())
        }
        GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M => {
            // Unreachable: repack_dequantize routes IQ types to
            // crate::iq before decoding tiles. Fails closed rather
            // than guessing a codebook formula for packed indices.
            Err(FormatError::SchemeMismatch {
                scheme: ggml.name(),
                expected: "card-A2.3 repacked decode (iq decodes via crate::iq)",
                got: ggml.name(),
            })
        }
    }
}

/// Scalar formula for the `_0`/`_1` 32-block types, shared by the
/// repacked decode (the source side inlines the same expressions;
/// both are pinned by the §10 round-trip test).
///
/// Every other [`GgmlType`] fails closed with [`FormatError::SchemeMismatch`]:
/// this helper only ever runs for the five 32-block types, so any other
/// variant here is a logic error, never a value with a default formula.
fn dequant_simple(ggml: GgmlType, d: f32, m: f32, value: u16) -> Result<f32, FormatError> {
    match ggml {
        GgmlType::Q8_0 => Ok(d * (value as u8 as i8) as f32),
        GgmlType::Q4_0 => Ok(d * (value as f32 - 8.0)),
        GgmlType::Q4_1 => Ok(d * value as f32 + m),
        GgmlType::Q5_0 => Ok(d * (value as f32 - 16.0)),
        GgmlType::Q5_1 => Ok(d * value as f32 + m),
        GgmlType::F16
        | GgmlType::BF16
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M => Err(FormatError::SchemeMismatch {
            scheme: ggml.name(),
            expected: "32-block _0/_1 type",
            got: ggml.name(),
        }),
    }
}

/// Reads one little-endian `u16` from a scale record (bounds-checked).
fn u16_pair(record: &[u8], offset: usize, what: &'static str) -> Result<u16, FormatError> {
    match (record.get(offset), record.get(offset + 1)) {
        (Some(lo), Some(hi)) => Ok(u16::from_le_bytes([*lo, *hi])),
        _ => Err(FormatError::LengthMismatch {
            what,
            expected: (offset + 2) as u64,
            got: record.len() as u64,
        }),
    }
}

/// Contiguous wire span holding the scale record: `Some((offset, length))`.
///
/// Returns `None` where no single wire span holds the record, and the
/// caller must not invent one:
///
/// - `F16`/`BF16` carry no scales (unquantized halves, SI-26);
/// - `Q6_K` stores `[d][sc16]` while the wire carries `d` at 208..210
///   and the sixteen `i8` scales at 192..208 (reordered, SI-57);
/// - `Q2_K` stores `[scales16][d][dmin]` gathered from wire 0..16 plus
///   `d`/`dmin` at 80..84, split around the value bytes (SI-57).
///
/// [`copy_block_record`] special-cases those three groups before
/// consulting this function, so a `None` here is never silently
/// treated as an empty or zero span.
///
/// Card-A2.4 spans are layout facts about the gguf-py wire, used by
/// documentation and tests: `IQ4_NL` carries `[d]` at 0..2 and
/// `IQ4_XS` carries `[d][scales_h][scales_l]` at 0..8, both verbatim
/// in the SoA record. The grid families gather non-contiguous fields
/// (`crate::iq::parse_block`), so no single span holds their record.
fn record_span(ggml: GgmlType) -> Option<(usize, usize)> {
    match ggml {
        GgmlType::F16 | GgmlType::BF16 => None,
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_0 => Some((0, 2)),
        GgmlType::Q4_1 | GgmlType::Q5_1 => Some((0, 4)),
        GgmlType::Q4_K | GgmlType::Q5_K => Some((0, 16)),
        GgmlType::Q6_K => None,
        GgmlType::Q3_K => Some((96, 14)),
        GgmlType::Q2_K => None,
        GgmlType::IQ4_NL => Some((0, 2)),
        GgmlType::IQ4_XS => Some((0, 8)),
        GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_span_reports_only_contiguous_wire_spans() {
        // Verbatim records: the span is the exact wire slice
        // copy_block_record copies (B3: these must stay truthful).
        assert_eq!(record_span(GgmlType::Q8_0), Some((0, 2)));
        assert_eq!(record_span(GgmlType::Q4_0), Some((0, 2)));
        assert_eq!(record_span(GgmlType::Q5_0), Some((0, 2)));
        assert_eq!(record_span(GgmlType::Q4_1), Some((0, 4)));
        assert_eq!(record_span(GgmlType::Q5_1), Some((0, 4)));
        assert_eq!(record_span(GgmlType::Q4_K), Some((0, 16)));
        assert_eq!(record_span(GgmlType::Q5_K), Some((0, 16)));
        assert_eq!(record_span(GgmlType::Q3_K), Some((96, 14)));
        // Card-A2.4 verbatim records: IQ4_NL carries [d] at 0..2 and
        // IQ4_XS carries [d][scales_h][scales_l] at 0..8.
        assert_eq!(record_span(GgmlType::IQ4_NL), Some((0, 2)));
        assert_eq!(record_span(GgmlType::IQ4_XS), Some((0, 8)));
        // No single wire span holds these records: Q6_K stores
        // [d][sc16] reordered from wire d@208..210 + sc@192..208, Q2_K
        // gathers wire 0..16 + 80..84, halves carry no scales, and the
        // grid IQ families gather non-contiguous fields. A span here
        // would be a lie that copy_block_record could misuse.
        assert_eq!(record_span(GgmlType::Q6_K), None);
        assert_eq!(record_span(GgmlType::Q2_K), None);
        assert_eq!(record_span(GgmlType::F16), None);
        assert_eq!(record_span(GgmlType::BF16), None);
        for ggml in [
            GgmlType::IQ2_XXS,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ1_S,
            GgmlType::IQ3_S,
            GgmlType::IQ2_S,
            GgmlType::IQ1_M,
        ] {
            assert_eq!(record_span(ggml), None, "{ggml} gathers its record");
        }
    }

    #[test]
    fn dequant_simple_computes_the_five_32_block_formulas() {
        assert_eq!(dequant_simple(GgmlType::Q8_0, 2.0, 0.0, 0xFF), Ok(-2.0));
        assert_eq!(dequant_simple(GgmlType::Q4_0, 2.0, 0.0, 10), Ok(4.0));
        assert_eq!(dequant_simple(GgmlType::Q4_1, 2.0, 1.0, 10), Ok(21.0));
        assert_eq!(dequant_simple(GgmlType::Q5_0, 2.0, 0.0, 20), Ok(8.0));
        assert_eq!(dequant_simple(GgmlType::Q5_1, 2.0, 1.0, 20), Ok(41.0));
    }

    #[test]
    fn dequant_simple_fails_closed_outside_32_block_types() {
        // B2: no silent default formula for logically unreachable
        // closed variants; each names the offending type.
        for ggml in [
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ2_XXS,
            GgmlType::IQ2_XS,
            GgmlType::IQ2_S,
            GgmlType::IQ1_S,
            GgmlType::IQ1_M,
        ] {
            assert_eq!(
                dequant_simple(ggml, 1.0, 0.0, 0),
                Err(FormatError::SchemeMismatch {
                    scheme: ggml.name(),
                    expected: "32-block _0/_1 type",
                    got: ggml.name(),
                }),
                "{ggml} must fail closed",
            );
        }
    }
}
