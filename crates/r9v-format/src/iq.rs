// SPDX-License-Identifier: Apache-2.0
//! IQ repack-only schemes (Spec 2 §3.3, §7 steps 1–4, §10; card A2.4).
//!
//! The nine GGUF codebook types (`IQ4_NL`, `IQ4_XS`, `IQ3_XXS`, `IQ3_S`,
//! `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ1_S`, `IQ1_M`) repack into canonical
//! `L1` exactly like the card-A2.3 types: a pure permutation of bytes,
//! no arithmetic on values (Spec 2 §7 step 4). What differs is what a
//! "value" is:
//!
//! - `IQ4_NL`/`IQ4_XS` store one 4-bit codebook index per weight, so
//!   their `L1` region is per-weight nibbles (`Packing::Nibble4`) and
//!   their scale records are `[d]` / `[d][scales_h][scales_l]`.
//! - The grid types (`IQ3_*`, `IQ2_*`, `IQ1_*`) pack one codebook index
//!   per 4 or 8 weights, so their `L1` region is the packed index bytes
//!   permuted into tile order over the index shape `[N, K/g]`
//!   (`g = 4` or `8`), and the scale record gathers the remaining wire
//!   fields (high-bit planes, sign bytes, per-group scales) verbatim.
//!
//! Codebooks live in [`crate::iq_lut`] as generated auditable data. The
//! source-side decode (`source_block`, over GGUF wire blocks) and the
//! repacked-side decode (`repacked_block`, over `L1` tiles plus SoA
//! records) are two independent readers implementing the same gguf-py
//! 0.19.0 `dequantize_blocks` formulas in the same `f32` evaluation
//! order; the §10 round-trip compares them bit-exact (SI-70).
//!
//! Every packed index pattern decodes to an in-range codebook slot:
//! decoders derive each grid position from fixed-width bit fields
//! (nibble masks, `& 0x1FF` entry masks, `qh` high-bit planes,
//! `field & 7` for `IQ1_*`), so no index payload can address outside
//! its table. Not every byte pattern is legal, however: corrupt
//! lengths are [`FormatError::LengthMismatch`] and non-finite `f16`
//! scales (including `IQ1_M`'s nibble-packed `d`) are
//! [`FormatError::InvalidScale`]. Nothing here panics
//! (CONVENTIONS.md §1.5).
//!
//! The `iq` module is public so its wire-format documentation is auditable,
//! while its operational items remain `pub(crate)`. The callable public
//! surface stays [`GgmlType`] plus the [`crate::iq_lut`] codebook tables;
//! [`crate::ggml::ggml_dequantize`] and the [`mod@crate::repack`] entry
//! points route IQ types here.

use crate::ggml::{check_wire_f16, wire_geometry, GgmlType};
use crate::iq_lut::{
    IQ1_GRID, IQ2_S_GRID, IQ2_XS_GRID, IQ2_XXS_GRID, IQ3_S_GRID, IQ3_XXS_GRID, IQ4_KVALUES,
    IQ_SIGN_LUT,
};
use crate::layout::PaddedDims;
use crate::repack::RepackedTensor;
use crate::FormatError;

// DECISION(A2.4): grid-type L1 values are packed index bytes permuted as
// opaque units over the index shape [N, K/g], not per-weight expansions;
// rejected expanding each index to its 4/8 grid values because the
// repacked region would exceed the wire size and break the §8 wire-size
// bpw identity that repack-only schemes keep. Nibble-per-weight IQ4
// types use the normal Nibble4 path instead. Spec 2 §3.3 is silent on
// IQ repack granularity; see SI-70.

/// Card-A2.4 IQ family classifier (Spec 2 §3.3 IQ rows).
///
/// Crate-private: the public surface stays [`GgmlType`]; this enum lets
/// `iq.rs` match nine exhaustive arms instead of repeating the twelve
/// fail-closed non-IQ variants at every site. [`iq_kind`] is the single
/// 21-arm classification point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IqKind {
    Iq4Nl,
    Iq4Xs,
    Iq3Xxs,
    Iq3S,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq1S,
    Iq1M,
}

/// Classifies `ggml`, returning `None` for the twelve card-A2.3 types.
/// Every arm is explicit: a future variant fails to compile here
/// instead of silently joining a family (CONVENTIONS.md §3.2).
pub(crate) fn iq_kind(ggml: GgmlType) -> Option<IqKind> {
    match ggml {
        GgmlType::IQ4_NL => Some(IqKind::Iq4Nl),
        GgmlType::IQ4_XS => Some(IqKind::Iq4Xs),
        GgmlType::IQ3_XXS => Some(IqKind::Iq3Xxs),
        GgmlType::IQ3_S => Some(IqKind::Iq3S),
        GgmlType::IQ2_XXS => Some(IqKind::Iq2Xxs),
        GgmlType::IQ2_XS => Some(IqKind::Iq2Xs),
        GgmlType::IQ2_S => Some(IqKind::Iq2S),
        GgmlType::IQ1_S => Some(IqKind::Iq1S),
        GgmlType::IQ1_M => Some(IqKind::Iq1M),
        GgmlType::F16
        | GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::BF16 => None,
    }
}

/// Whether `ggml` is one of the nine card-A2.4 IQ types.
pub(crate) fn is_iq(ggml: GgmlType) -> bool {
    iq_kind(ggml).is_some()
}

/// Weights covered by one packed index byte: 1 for the nibble-per-weight
/// IQ4 types (each nibble is one index), 4 where one byte yields four
/// grid values (`IQ3_XXS`, `IQ3_S`, `IQ2_XXS`, `IQ2_XS`), 8 where one
/// byte yields eight (`IQ2_S`, `IQ1_S`, `IQ1_M`).
pub(crate) fn index_granularity(kind: IqKind) -> u32 {
    match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => 1,
        IqKind::Iq3Xxs | IqKind::Iq3S | IqKind::Iq2Xxs | IqKind::Iq2Xs => 4,
        IqKind::Iq2S | IqKind::Iq1S | IqKind::Iq1M => 8,
    }
}

/// Packed index bytes per wire block: nibbles-as-bytes for the IQ4
/// types (32 / 256), packed index bytes for the grid types
/// (64 for `g = 4`, 32 for `g = 8`).
pub(crate) fn wire_index_len(kind: IqKind) -> usize {
    match kind {
        IqKind::Iq4Nl => 32,
        IqKind::Iq4Xs => 256,
        IqKind::Iq3Xxs | IqKind::Iq3S | IqKind::Iq2Xxs | IqKind::Iq2Xs => 64,
        IqKind::Iq2S | IqKind::Iq1S | IqKind::Iq1M => 32,
    }
}

/// SoA scale-record bytes (Spec 2 §3.1 grouping; SI-70 for contents):
/// `Iq4Nl [d]`, `Iq4Xs [d][scales_h][scales_l]`, `Iq3Xxs [d][scales]`,
/// `Iq3S [d][qh][signs][scales]`, `Iq2Xxs [d]`,
/// `Iq2Xs [d][scales]`, `Iq2S [d][signs][qh][scales]`,
/// `Iq1S [d][qh]`, `Iq1M [qh][scales]` (`d` packed in the scales).
pub(crate) fn record_len(kind: IqKind) -> usize {
    match kind {
        IqKind::Iq4Nl => 2,
        IqKind::Iq4Xs => 8,
        IqKind::Iq3Xxs => 34,
        IqKind::Iq3S => 46,
        IqKind::Iq2Xxs => 2,
        IqKind::Iq2Xs => 10,
        IqKind::Iq2S => 50,
        IqKind::Iq1S => 18,
        IqKind::Iq1M => 24,
    }
}

/// Reads one little-endian `u16` at `offset` (bounds-checked).
fn u16_at(block: &[u8], offset: usize, what: &'static str) -> Result<u16, FormatError> {
    match (block.get(offset), block.get(offset + 1)) {
        (Some(lo), Some(hi)) => Ok(u16::from_le_bytes([*lo, *hi])),
        _ => Err(FormatError::LengthMismatch {
            what,
            expected: (offset + 2) as u64,
            got: block.len() as u64,
        }),
    }
}

/// Reads one little-endian `u32` at `offset` (bounds-checked).
fn u32_at(block: &[u8], offset: usize, what: &'static str) -> Result<u32, FormatError> {
    let mut bytes = [0u8; 4];
    for (i, slot) in bytes.iter_mut().enumerate() {
        match block.get(offset + i) {
            Some(b) => *slot = *b,
            None => {
                return Err(FormatError::LengthMismatch {
                    what,
                    expected: (offset + 4) as u64,
                    got: block.len() as u64,
                });
            }
        }
    }
    Ok(u32::from_le_bytes(bytes))
}

/// Copies `len` bytes at `offset` into `dest` (bounds-checked on both).
fn copy_span(
    block: &[u8],
    offset: usize,
    dest: &mut [u8],
    what: &'static str,
) -> Result<(), FormatError> {
    if offset > block.len() || dest.len() > block.len() - offset {
        return Err(FormatError::LengthMismatch {
            what,
            expected: (offset + dest.len()) as u64,
            got: block.len() as u64,
        });
    }
    match block.get(offset..offset + dest.len()) {
        Some(src) => {
            dest.copy_from_slice(src);
            Ok(())
        }
        None => Err(FormatError::LengthMismatch {
            what,
            expected: (offset + dest.len()) as u64,
            got: block.len() as u64,
        }),
    }
}

/// Requires `block` to hold one full wire block of `ggml`.
fn expect_block(block: &[u8], ggml: GgmlType) -> Result<(), FormatError> {
    if block.len() as u64 != ggml.block_bytes() {
        return Err(FormatError::LengthMismatch {
            what: "iq wire block",
            expected: ggml.block_bytes(),
            got: block.len() as u64,
        });
    }
    Ok(())
}

/// Requires `idx`/`rec` to hold one block's parsed payloads.
fn expect_parsed(kind: IqKind, idx: &[u8], rec: &[u8]) -> Result<(), FormatError> {
    if idx.len() != wire_index_len(kind) {
        return Err(FormatError::LengthMismatch {
            what: "iq parsed indices",
            expected: wire_index_len(kind) as u64,
            got: idx.len() as u64,
        });
    }
    if rec.len() != record_len(kind) {
        return Err(FormatError::LengthMismatch {
            what: "iq parsed record",
            expected: record_len(kind) as u64,
            got: rec.len() as u64,
        });
    }
    Ok(())
}

/// Unpacks one wire block's nibbles into per-weight 4-bit indices.
///
/// `Iq4Nl` uses `Q4_0`-style split nibbles (byte `p` holds values `p`
/// low and `p + 16` high); `Iq4Xs` uses group-paired nibbles (byte
/// `g * 16 + p` holds values `g * 32 + p` low and `g * 32 + 16 + p`
/// high). Both orders were proven against gguf-py 0.19.0
/// `dequantize_blocks` with sequential-nibble probes (`/tmp/iq_precheck.py`).
fn unpack_nibbles(kind: IqKind, block: &[u8], nib: &mut [u8]) -> Result<(), FormatError> {
    match kind {
        IqKind::Iq4Nl => {
            if block.len() != 18 || nib.len() != 32 {
                return Err(FormatError::LengthMismatch {
                    what: "iq4_nl nibbles",
                    expected: 18,
                    got: block.len() as u64,
                });
            }
            for (p, slot) in nib.iter_mut().enumerate().take(32) {
                let byte = match block.get(2 + (p % 16)) {
                    Some(b) => *b,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "iq4_nl values",
                            expected: 18,
                            got: block.len() as u64,
                        });
                    }
                };
                *slot = if p < 16 { byte & 0x0F } else { byte >> 4 };
            }
            Ok(())
        }
        IqKind::Iq4Xs => {
            if block.len() != 136 || nib.len() != 256 {
                return Err(FormatError::LengthMismatch {
                    what: "iq4_xs nibbles",
                    expected: 136,
                    got: block.len() as u64,
                });
            }
            for (j, slot) in nib.iter_mut().enumerate().take(256) {
                let byte = match block.get(8 + (j / 32) * 16 + (j % 16)) {
                    Some(b) => *b,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "iq4_xs values",
                            expected: 136,
                            got: block.len() as u64,
                        });
                    }
                };
                *slot = if j % 32 < 16 { byte & 0x0F } else { byte >> 4 };
            }
            Ok(())
        }
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => Err(FormatError::SchemeMismatch {
            scheme: kind.name(),
            expected: "4-bit-per-weight iq type",
            got: kind.name(),
        }),
    }
}

/// Stable lowercase name of an IQ family (mirrors [`GgmlType::name`]).
impl IqKind {
    fn name(self) -> &'static str {
        match self {
            IqKind::Iq4Nl => "IQ4_NL",
            IqKind::Iq4Xs => "IQ4_XS",
            IqKind::Iq3Xxs => "IQ3_XXS",
            IqKind::Iq3S => "IQ3_S",
            IqKind::Iq2Xxs => "IQ2_XXS",
            IqKind::Iq2Xs => "IQ2_XS",
            IqKind::Iq2S => "IQ2_S",
            IqKind::Iq1S => "IQ1_S",
            IqKind::Iq1M => "IQ1_M",
        }
    }
}

/// Splits one wire block into packed index bytes plus the SoA scale
/// record (Spec 2 §7 step 4: pure byte movement, no arithmetic).
///
/// `idx` holds `wire_index_len` bytes, `rec` `record_len` bytes;
/// both length-checked before any byte moves (CONVENTIONS.md §1.4).
/// Record contents follow the gguf-py wire order with the index payload
/// removed; `Iq3S`/`Iq2S` gather non-contiguous fields (SI-70).
pub(crate) fn parse_block(
    kind: IqKind,
    ggml: GgmlType,
    block: &[u8],
    idx: &mut [u8],
    rec: &mut [u8],
) -> Result<(), FormatError> {
    expect_block(block, ggml)?;
    expect_parsed(kind, idx, rec)?;
    match kind {
        IqKind::Iq4Nl => {
            unpack_nibbles(kind, block, idx)?;
            copy_span(block, 0, &mut rec[0..2], "iq4_nl record")?;
            Ok(())
        }
        IqKind::Iq4Xs => {
            unpack_nibbles(kind, block, idx)?;
            copy_span(block, 0, &mut rec[0..8], "iq4_xs record")?;
            Ok(())
        }
        IqKind::Iq3Xxs => {
            copy_span(block, 2, &mut idx[0..64], "iq3_xxs indices")?;
            copy_span(block, 0, &mut rec[0..2], "iq3_xxs scale")?;
            copy_span(block, 66, &mut rec[2..34], "iq3_xxs scales")?;
            Ok(())
        }
        IqKind::Iq3S => {
            copy_span(block, 2, &mut idx[0..64], "iq3_s indices")?;
            copy_span(block, 0, &mut rec[0..2], "iq3_s scale")?;
            copy_span(block, 66, &mut rec[2..10], "iq3_s high bits")?;
            copy_span(block, 74, &mut rec[10..42], "iq3_s signs")?;
            copy_span(block, 106, &mut rec[42..46], "iq3_s scales")?;
            Ok(())
        }
        IqKind::Iq2Xxs => {
            copy_span(block, 2, &mut idx[0..64], "iq2_xxs indices")?;
            copy_span(block, 0, &mut rec[0..2], "iq2_xxs scale")?;
            Ok(())
        }
        IqKind::Iq2Xs => {
            copy_span(block, 2, &mut idx[0..64], "iq2_xs indices")?;
            copy_span(block, 0, &mut rec[0..2], "iq2_xs scale")?;
            copy_span(block, 66, &mut rec[2..10], "iq2_xs scales")?;
            Ok(())
        }
        IqKind::Iq2S => {
            copy_span(block, 2, &mut idx[0..32], "iq2_s indices")?;
            copy_span(block, 0, &mut rec[0..2], "iq2_s scale")?;
            copy_span(block, 34, &mut rec[2..34], "iq2_s signs")?;
            copy_span(block, 66, &mut rec[34..42], "iq2_s high bits")?;
            copy_span(block, 74, &mut rec[42..50], "iq2_s scales")?;
            Ok(())
        }
        IqKind::Iq1S => {
            copy_span(block, 2, &mut idx[0..32], "iq1_s indices")?;
            copy_span(block, 0, &mut rec[0..2], "iq1_s scale")?;
            copy_span(block, 34, &mut rec[2..18], "iq1_s high bits")?;
            Ok(())
        }
        IqKind::Iq1M => {
            copy_span(block, 0, &mut idx[0..32], "iq1_m indices")?;
            copy_span(block, 32, &mut rec[0..16], "iq1_m high bits")?;
            copy_span(block, 48, &mut rec[16..24], "iq1_m scales")?;
            Ok(())
        }
    }
}

/// Per-weight 4-bit logicals for the IQ4 types, widened to `u16` for
/// `block_values`. Grid types have no per-weight
/// logical and fail closed (their indices pack 4–8 weights per byte).
pub(crate) fn nibble_logical(
    kind: IqKind,
    ggml: GgmlType,
    block: &[u8],
    out: &mut [u16],
) -> Result<(), FormatError> {
    match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => {
            if out.len() != ggml.block_len() as usize {
                return Err(FormatError::LengthMismatch {
                    what: "iq logical values",
                    expected: ggml.block_len() as u64,
                    got: out.len() as u64,
                });
            }
            let mut nib = [0u8; 256];
            let (nib, _) = nib.split_at_mut(out.len());
            unpack_nibbles(kind, block, nib)?;
            for (slot, v) in out.iter_mut().zip(nib.iter()) {
                *slot = *v as u16;
            }
            Ok(())
        }
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => Err(FormatError::SchemeMismatch {
            scheme: kind.name(),
            expected: "4-bit-per-weight iq type",
            got: kind.name(),
        }),
    }
}

/// Exact inverse of `parse_block`: packed indices plus the SoA record
/// reproduce the wire block byte-exact (Spec 2 §7 step 4). Nibble
/// packing inverts `unpack_nibbles`; grid payloads copy back to their
/// wire offsets. Bijective on every family, proven by
/// `unpack(repack(wire)) == wire` over all fixtures.
pub(crate) fn write_block(
    kind: IqKind,
    ggml: GgmlType,
    idx: &[u8],
    rec: &[u8],
    block: &mut [u8],
) -> Result<(), FormatError> {
    expect_block(block, ggml)?;
    expect_parsed(kind, idx, rec)?;
    for slot in block.iter_mut() {
        *slot = 0;
    }
    match kind {
        IqKind::Iq4Nl => {
            copy_span(rec, 0, &mut block[0..2], "iq4_nl record")?;
            for p in 0..16 {
                block[2 + p] = (idx[p] & 0x0F) | ((idx[p + 16] & 0x0F) << 4);
            }
            Ok(())
        }
        IqKind::Iq4Xs => {
            copy_span(rec, 0, &mut block[0..8], "iq4_xs record")?;
            for g in 0..8 {
                for p in 0..16 {
                    block[8 + g * 16 + p] =
                        (idx[g * 32 + p] & 0x0F) | ((idx[g * 32 + 16 + p] & 0x0F) << 4);
                }
            }
            Ok(())
        }
        IqKind::Iq3Xxs => {
            copy_span(rec, 0, &mut block[0..2], "iq3_xxs scale")?;
            copy_span(idx, 0, &mut block[2..66], "iq3_xxs indices")?;
            copy_span(rec, 2, &mut block[66..98], "iq3_xxs scales")?;
            Ok(())
        }
        IqKind::Iq3S => {
            copy_span(rec, 0, &mut block[0..2], "iq3_s scale")?;
            copy_span(idx, 0, &mut block[2..66], "iq3_s indices")?;
            copy_span(rec, 2, &mut block[66..74], "iq3_s high bits")?;
            copy_span(rec, 10, &mut block[74..106], "iq3_s signs")?;
            copy_span(rec, 42, &mut block[106..110], "iq3_s scales")?;
            Ok(())
        }
        IqKind::Iq2Xxs => {
            copy_span(rec, 0, &mut block[0..2], "iq2_xxs scale")?;
            copy_span(idx, 0, &mut block[2..66], "iq2_xxs indices")?;
            Ok(())
        }
        IqKind::Iq2Xs => {
            copy_span(rec, 0, &mut block[0..2], "iq2_xs scale")?;
            copy_span(idx, 0, &mut block[2..66], "iq2_xs indices")?;
            copy_span(rec, 2, &mut block[66..74], "iq2_xs scales")?;
            Ok(())
        }
        IqKind::Iq2S => {
            copy_span(rec, 0, &mut block[0..2], "iq2_s scale")?;
            copy_span(idx, 0, &mut block[2..34], "iq2_s indices")?;
            copy_span(rec, 2, &mut block[34..66], "iq2_s signs")?;
            copy_span(rec, 34, &mut block[66..74], "iq2_s high bits")?;
            copy_span(rec, 42, &mut block[74..82], "iq2_s scales")?;
            Ok(())
        }
        IqKind::Iq1S => {
            copy_span(rec, 0, &mut block[0..2], "iq1_s scale")?;
            copy_span(idx, 0, &mut block[2..34], "iq1_s indices")?;
            copy_span(rec, 2, &mut block[34..50], "iq1_s high bits")?;
            Ok(())
        }
        IqKind::Iq1M => {
            copy_span(idx, 0, &mut block[0..32], "iq1_m indices")?;
            copy_span(rec, 0, &mut block[32..48], "iq1_m high bits")?;
            copy_span(rec, 16, &mut block[48..56], "iq1_m scales")?;
            Ok(())
        }
    }
}

/// One sign bit to its `f32` factor: clear bit is `+1.0`, set bit is
/// `-1.0` (gguf-py `np.where(signs == 0, 1, -1)`).
fn sign_factor(bit: u8) -> f32 {
    if bit == 0 {
        1.0
    } else {
        -1.0
    }
}

/// One `ksigns` byte selected by a 7-bit field (the field is masked at
/// extraction, so the index is always in range; no panic path exists).
fn sign_byte(field: u32) -> u8 {
    IQ_SIGN_LUT[(field & 0x7F) as usize]
}

/// Source-side reference decode of one wire block (Spec 2 §3.3, §10):
/// reads GGUF wire bytes directly and appends 32 (`Iq4Nl`) or 256
/// `f32` values in ascending order, matching gguf-py 0.19.0
/// `dequantize_blocks` bit-exact.
///
/// `f32` evaluation order mirrors the numpy expressions exactly
/// (`db = (d * (0.5 + extra)) * 0.25`, `(db * grid) * sign`,
/// `dl * (grid + delta)`), because any reassociation can round
/// differently and fail the bit-exact test.
pub(crate) fn source_block(
    kind: IqKind,
    scheme: &'static str,
    index: u64,
    block: &[u8],
    out: &mut Vec<f32>,
) -> Result<(), FormatError> {
    match kind {
        IqKind::Iq4Nl => {
            if block.len() != 18 {
                return Err(FormatError::LengthMismatch {
                    what: "iq4_nl wire block",
                    expected: 18,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq4_nl scale")?)?;
            for j in 0..32 {
                let byte = block[2 + (j % 16)];
                let q = if j < 16 { byte & 0x0F } else { byte >> 4 };
                out.push(d * IQ4_KVALUES[(q & 0x0F) as usize] as f32);
            }
            Ok(())
        }
        IqKind::Iq4Xs => {
            if block.len() != 136 {
                return Err(FormatError::LengthMismatch {
                    what: "iq4_xs wire block",
                    expected: 136,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq4_xs scale")?)?;
            let scales_h = u16_at(block, 2, "iq4_xs scale high")?;
            for g in 0..8 {
                let lo = (block[4 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let hi = ((scales_h >> (2 * g)) & 0x03) as u8;
                let s = (lo | (hi << 4)) as i8 as f32 - 32.0;
                let dl = d * s;
                for j in 0..32 {
                    let byte = block[8 + g * 16 + (j % 16)];
                    let q = if j < 16 { byte & 0x0F } else { byte >> 4 };
                    out.push(dl * IQ4_KVALUES[(q & 0x0F) as usize] as f32);
                }
            }
            Ok(())
        }
        IqKind::Iq2Xxs => {
            if block.len() != 66 {
                return Err(FormatError::LengthMismatch {
                    what: "iq2_xxs wire block",
                    expected: 66,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq2_xxs scale")?)?;
            for e in 0..8 {
                let u_even = u32_at(block, 2 + 8 * e, "iq2_xxs indices")?;
                let u_odd = u32_at(block, 2 + 8 * e + 4, "iq2_xxs signs")?;
                let extra = (u_odd >> 28) as f32;
                let db = (d * (0.5 + extra)) * 0.25;
                for s in 0..4 {
                    let grid_idx = (u_even >> (8 * s)) & 0xFF;
                    let signs = sign_byte(u_odd >> (7 * s));
                    for b in 0..8 {
                        let grid = IQ2_XXS_GRID[(grid_idx as usize) * 8 + b] as f32;
                        out.push((db * grid) * sign_factor((signs >> b) & 1));
                    }
                }
            }
            Ok(())
        }
        IqKind::Iq2Xs => {
            if block.len() != 74 {
                return Err(FormatError::LengthMismatch {
                    what: "iq2_xs wire block",
                    expected: 74,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq2_xs scale")?)?;
            for t in 0..32 {
                let entry = u16_at(block, 2 + 2 * t, "iq2_xs indices")?;
                let g = t / 2;
                let s = (block[66 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let db = (d * (0.5 + s as f32)) * 0.25;
                let grid_idx = (entry & 0x1FF) as usize;
                let signs = sign_byte(entry as u32 >> 9);
                for b in 0..8 {
                    let grid = IQ2_XS_GRID[grid_idx * 8 + b] as f32;
                    out.push((db * grid) * sign_factor((signs >> b) & 1));
                }
            }
            Ok(())
        }
        IqKind::Iq2S => {
            if block.len() != 82 {
                return Err(FormatError::LengthMismatch {
                    what: "iq2_s wire block",
                    expected: 82,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq2_s scale")?)?;
            for t in 0..32 {
                let g = t / 2;
                let e = t % 2;
                let s = (block[74 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let db = (d * (0.5 + s as f32)) * 0.25;
                let qh = (block[66 + t / 4] >> ((t % 4) * 2)) & 0x03;
                let grid_idx = (block[2 + t] as usize) | ((qh as usize) << 8);
                let sign_row = block[34 + 2 * g + e];
                for b in 0..8 {
                    let grid = IQ2_S_GRID[grid_idx * 8 + b] as f32;
                    out.push((db * grid) * sign_factor((sign_row >> b) & 1));
                }
            }
            Ok(())
        }
        IqKind::Iq3Xxs => {
            if block.len() != 98 {
                return Err(FormatError::LengthMismatch {
                    what: "iq3_xxs wire block",
                    expected: 98,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq3_xxs scale")?)?;
            for g in 0..8 {
                let u = u32_at(block, 66 + 4 * g, "iq3_xxs scales")?;
                let extra = (u >> 28) as f32;
                let db = (d * (0.5 + extra)) * 0.5;
                for t in 0..8 {
                    let grid_idx = block[2 + g * 8 + t] as usize;
                    // Four sign fields per group of 32: index t draws
                    // field t / 2 at bit base (t % 2) * 4.
                    let signs = sign_byte(u >> (7 * (t / 2)));
                    let base = (t % 2) * 4;
                    for b in 0..4 {
                        let grid = IQ3_XXS_GRID[grid_idx * 4 + b] as f32;
                        out.push((db * grid) * sign_factor((signs >> (base + b)) & 1));
                    }
                }
            }
            Ok(())
        }
        IqKind::Iq3S => {
            if block.len() != 110 {
                return Err(FormatError::LengthMismatch {
                    what: "iq3_s wire block",
                    expected: 110,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq3_s scale")?)?;
            for g in 0..8 {
                let s = (block[106 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let db = d * (1.0 + 2.0 * s as f32);
                for sg in 0..8 {
                    let i = g * 8 + sg;
                    let bit = (block[66 + g] >> sg) & 1;
                    let grid_idx = (block[2 + i] as usize) | ((bit as usize) << 8);
                    // Signs pack four indices per group of 32: index sg
                    // draws byte g * 4 + sg / 2 at bit base (sg % 2) * 4.
                    let sign_row = block[74 + g * 4 + sg / 2];
                    let base = (sg % 2) * 4;
                    for bg in 0..4 {
                        let grid = IQ3_S_GRID[grid_idx * 4 + bg] as f32;
                        out.push((db * grid) * sign_factor((sign_row >> (base + bg)) & 1));
                    }
                }
            }
            Ok(())
        }
        IqKind::Iq1S => {
            if block.len() != 50 {
                return Err(FormatError::LengthMismatch {
                    what: "iq1_s wire block",
                    expected: 50,
                    got: block.len() as u64,
                });
            }
            let d = check_wire_f16(scheme, index, u16_at(block, 0, "iq1_s scale")?)?;
            for t in 0..32 {
                let g = t / 4;
                let qh = u16_at(block, 34 + 2 * g, "iq1_s high bits")?;
                let dl = d * ((2 * ((qh >> 12) & 7) + 1) as f32);
                let delta = if qh & 0x8000 == 0 { 0.125 } else { -0.125 };
                let field = (qh >> ((t % 4) * 3)) & 7;
                let grid_idx = (block[2 + t] as usize) | ((field as usize) << 8);
                for b in 0..8 {
                    let grid = IQ1_GRID[grid_idx * 8 + b] as f32;
                    out.push(dl * (grid + delta));
                }
            }
            Ok(())
        }
        IqKind::Iq1M => {
            if block.len() != 56 {
                return Err(FormatError::LengthMismatch {
                    what: "iq1_m wire block",
                    expected: 56,
                    got: block.len() as u64,
                });
            }
            let w0 = u16_at(block, 48, "iq1_m scales")?;
            let w1 = u16_at(block, 50, "iq1_m scales")?;
            let w2 = u16_at(block, 52, "iq1_m scales")?;
            let w3 = u16_at(block, 54, "iq1_m scales")?;
            let d_bits =
                ((w0 & 0xF000) >> 12) | ((w1 & 0xF000) >> 8) | ((w2 & 0xF000) >> 4) | (w3 & 0xF000);
            let d = check_wire_f16(scheme, index, d_bits)?;
            for t in 0..32 {
                let g = t / 4;
                let h = (t / 2) % 2;
                // dl index i = g * 2 + h over the 16 (word, shift) pairs.
                let i = g * 2 + h;
                let dl_word = u16_at(block, 48 + 2 * (i / 4), "iq1_m scales")?;
                let dl = d * ((2 * ((dl_word >> ((i % 4) * 3)) & 7) + 1) as f32);
                // qh fields are nibbles of the 16 qh bytes.
                let field = (block[32 + t / 2] >> ((t % 2) * 4)) & 0x0F;
                let grid_idx = (block[t] as usize) | (((field & 7) as usize) << 8);
                let delta = if field & 8 == 0 { 0.125 } else { -0.125 };
                for b in 0..8 {
                    let grid = IQ1_GRID[grid_idx * 8 + b] as f32;
                    out.push(dl * (grid + delta));
                }
            }
            Ok(())
        }
    }
}

/// Repacked-side reference decode of one block (Spec 2 §3.3, §10):
/// reads the SoA scale record plus already-parsed index bytes and
/// appends 32 (`Iq4Nl`) or 256 `f32` values in ascending order.
///
/// This path never touches GGUF wire blocks: scales come from the SoA
/// record layout (`record_len`) and indices from the tiled index
/// region, so the §10 round-trip compares two genuinely different
/// readers. Formulas and `f32` evaluation order match `source_block`
/// exactly; any divergence fails that test. Nibbles are masked to four
/// bits at use (parse and the `L1` nibble codec only ever produce
/// four-bit fields), so no input can panic here.
pub(crate) fn repacked_block(
    kind: IqKind,
    scheme: &'static str,
    index: u64,
    idx: &[u8],
    rec: &[u8],
    out: &mut Vec<f32>,
) -> Result<(), FormatError> {
    expect_parsed(kind, idx, rec)?;
    match kind {
        IqKind::Iq4Nl => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq4_nl scale")?)?;
            for q in idx.iter().take(32) {
                out.push(d * IQ4_KVALUES[(q & 0x0F) as usize] as f32);
            }
            Ok(())
        }
        IqKind::Iq4Xs => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq4_xs scale")?)?;
            let scales_h = u16_at(rec, 2, "repacked iq4_xs scale high")?;
            for g in 0..8 {
                let lo = (rec[4 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let hi = ((scales_h >> (2 * g)) & 0x03) as u8;
                let s = (lo | (hi << 4)) as i8 as f32 - 32.0;
                let dl = d * s;
                for q in idx.iter().skip(g * 32).take(32) {
                    out.push(dl * IQ4_KVALUES[(q & 0x0F) as usize] as f32);
                }
            }
            Ok(())
        }
        IqKind::Iq2Xxs => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq2_xxs scale")?)?;
            for e in 0..8 {
                let u_even = u32_at(idx, 8 * e, "repacked iq2_xxs indices")?;
                let u_odd = u32_at(idx, 8 * e + 4, "repacked iq2_xxs signs")?;
                let extra = (u_odd >> 28) as f32;
                let db = (d * (0.5 + extra)) * 0.25;
                for s in 0..4 {
                    let grid_idx = (u_even >> (8 * s)) & 0xFF;
                    let signs = sign_byte(u_odd >> (7 * s));
                    for b in 0..8 {
                        let grid = IQ2_XXS_GRID[(grid_idx as usize) * 8 + b] as f32;
                        out.push((db * grid) * sign_factor((signs >> b) & 1));
                    }
                }
            }
            Ok(())
        }
        IqKind::Iq2Xs => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq2_xs scale")?)?;
            for t in 0..32 {
                let entry = u16_at(idx, 2 * t, "repacked iq2_xs indices")?;
                let g = t / 2;
                let s = (rec[2 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let db = (d * (0.5 + s as f32)) * 0.25;
                let grid_idx = (entry & 0x1FF) as usize;
                let signs = sign_byte(entry as u32 >> 9);
                for b in 0..8 {
                    let grid = IQ2_XS_GRID[grid_idx * 8 + b] as f32;
                    out.push((db * grid) * sign_factor((signs >> b) & 1));
                }
            }
            Ok(())
        }
        IqKind::Iq2S => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq2_s scale")?)?;
            for t in 0..32 {
                let g = t / 2;
                let e = t % 2;
                let s = (rec[42 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let db = (d * (0.5 + s as f32)) * 0.25;
                let qh = (rec[34 + t / 4] >> ((t % 4) * 2)) & 0x03;
                let grid_idx = (idx[t] as usize) | ((qh as usize) << 8);
                let sign_row = rec[2 + 2 * g + e];
                for b in 0..8 {
                    let grid = IQ2_S_GRID[grid_idx * 8 + b] as f32;
                    out.push((db * grid) * sign_factor((sign_row >> b) & 1));
                }
            }
            Ok(())
        }
        IqKind::Iq3Xxs => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq3_xxs scale")?)?;
            for g in 0..8 {
                let u = u32_at(rec, 2 + 4 * g, "repacked iq3_xxs scales")?;
                let extra = (u >> 28) as f32;
                let db = (d * (0.5 + extra)) * 0.5;
                for t in 0..8 {
                    let grid_idx = idx[g * 8 + t] as usize;
                    // Four sign fields per group of 32: index t draws
                    // field t / 2 at bit base (t % 2) * 4.
                    let signs = sign_byte(u >> (7 * (t / 2)));
                    let base = (t % 2) * 4;
                    for b in 0..4 {
                        let grid = IQ3_XXS_GRID[grid_idx * 4 + b] as f32;
                        out.push((db * grid) * sign_factor((signs >> (base + b)) & 1));
                    }
                }
            }
            Ok(())
        }
        IqKind::Iq3S => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq3_s scale")?)?;
            for g in 0..8 {
                let s = (rec[42 + g / 2] >> ((g % 2) * 4)) & 0x0F;
                let db = d * (1.0 + 2.0 * s as f32);
                for sg in 0..8 {
                    let i = g * 8 + sg;
                    let bit = (rec[2 + g] >> sg) & 1;
                    let grid_idx = (idx[i] as usize) | ((bit as usize) << 8);
                    // Signs pack four indices per group of 32: index sg
                    // draws byte g * 4 + sg / 2 at bit base (sg % 2) * 4.
                    let sign_row = rec[10 + g * 4 + sg / 2];
                    let base = (sg % 2) * 4;
                    for bg in 0..4 {
                        let grid = IQ3_S_GRID[grid_idx * 4 + bg] as f32;
                        out.push((db * grid) * sign_factor((sign_row >> (base + bg)) & 1));
                    }
                }
            }
            Ok(())
        }
        IqKind::Iq1S => {
            let d = check_wire_f16(scheme, index, u16_at(rec, 0, "repacked iq1_s scale")?)?;
            for (t, qb) in idx.iter().enumerate().take(32) {
                let g = t / 4;
                let qh = u16_at(rec, 2 + 2 * g, "repacked iq1_s high bits")?;
                let dl = d * ((2 * ((qh >> 12) & 7) + 1) as f32);
                let delta = if qh & 0x8000 == 0 { 0.125 } else { -0.125 };
                let field = (qh >> ((t % 4) * 3)) & 7;
                let grid_idx = (*qb as usize) | ((field as usize) << 8);
                for b in 0..8 {
                    let grid = IQ1_GRID[grid_idx * 8 + b] as f32;
                    out.push(dl * (grid + delta));
                }
            }
            Ok(())
        }
        IqKind::Iq1M => {
            let w0 = u16_at(rec, 16, "repacked iq1_m scales")?;
            let w1 = u16_at(rec, 18, "repacked iq1_m scales")?;
            let w2 = u16_at(rec, 20, "repacked iq1_m scales")?;
            let w3 = u16_at(rec, 22, "repacked iq1_m scales")?;
            let d_bits =
                ((w0 & 0xF000) >> 12) | ((w1 & 0xF000) >> 8) | ((w2 & 0xF000) >> 4) | (w3 & 0xF000);
            let d = check_wire_f16(scheme, index, d_bits)?;
            for t in 0..32 {
                let g = t / 4;
                let h = (t / 2) % 2;
                let i = g * 2 + h;
                let dl_word = u16_at(rec, 16 + 2 * (i / 4), "repacked iq1_m scales")?;
                let dl = d * ((2 * ((dl_word >> ((i % 4) * 3)) & 7) + 1) as f32);
                let field = (rec[t / 2] >> ((t % 2) * 4)) & 0x0F;
                let grid_idx = (idx[t] as usize) | (((field & 7) as usize) << 8);
                let delta = if field & 8 == 0 { 0.125 } else { -0.125 };
                for b in 0..8 {
                    let grid = IQ1_GRID[grid_idx * 8 + b] as f32;
                    out.push(dl * (grid + delta));
                }
            }
            Ok(())
        }
    }
}

/// Reference dequantization of IQ GGUF source bytes (Spec 2 §3.3,
/// §10): `wire` holds `n_rows` rows of `k` values in row-major
/// wire-block order; returns `n_rows * k` `f32` values in row-major
/// order, ascending, matching gguf-py 0.19.0 `dequantize_blocks`
/// bit-exact. Called from [`crate::ggml::ggml_dequantize`] for IQ
/// types. Every scale failure is collected per block and reported
/// with its block index (CONVENTIONS.md §1.4).
pub(crate) fn source_dequantize(
    ggml: GgmlType,
    wire: &[u8],
    n_rows: u32,
    k: u32,
) -> Result<Vec<f32>, FormatError> {
    let kind = match iq_kind(ggml) {
        Some(kind) => kind,
        None => {
            return Err(FormatError::SchemeMismatch {
                scheme: ggml.name(),
                expected: "iq ggml type",
                got: ggml.name(),
            });
        }
    };
    let geo = wire_geometry(ggml, wire, n_rows, k)?;
    let block_bytes = ggml.block_bytes() as usize;
    let scheme_name = match ggml.scheme() {
        Some(s) => s.name(),
        None => ggml.name(),
    };
    let mut out = Vec::with_capacity(n_rows as usize * k as usize);
    let mut problems = Vec::new();
    let row_bytes = geo.blocks_per_row as usize * block_bytes;
    // `wire_geometry` fixed the exact length, so every slice below is
    // in range by construction; `source_block` rechecks its own block.
    for row in 0..n_rows as usize {
        for b in 0..geo.blocks_per_row as usize {
            let base = row * row_bytes + b * block_bytes;
            let block = &wire[base..base + block_bytes];
            let record = (row * geo.blocks_per_row as usize + b) as u64;
            match source_block(kind, scheme_name, record, block, &mut out) {
                Ok(()) => {}
                Err(FormatError::Multiple { problems: inner }) => {
                    problems.extend(inner.into_vec());
                }
                Err(single) => problems.push(single),
            }
        }
    }
    FormatError::collect(problems)?;
    Ok(out)
}

/// Index-shape dims for the tiled index region: `[N, K/g]` padded to
/// 16 (no superblock: index blocks already tile-align since `K` is a
/// multiple of 256 and `g` divides 16 evenly).
fn index_dims(dims: &PaddedDims, kind: IqKind) -> Result<PaddedDims, FormatError> {
    let gran = index_granularity(kind);
    if gran <= 1 {
        return Err(FormatError::SchemeMismatch {
            scheme: kind.name(),
            expected: "packed-index iq type",
            got: kind.name(),
        });
    }
    if !dims.k().is_multiple_of(gran) {
        return Err(FormatError::InvalidBlock {
            name: "k",
            value: dims.k() as u64,
            reason: "must be a multiple of the iq index granularity",
        });
    }
    PaddedDims::new(dims.n(), dims.k() / gran, None)
}

/// Reads the SoA record for row-block `nb`, K-block `kb`, intra-block
/// row `row` (Spec 2 §3.1 grouping over wire blocks: 32 weights for
/// `Iq4Nl`, 256 otherwise).
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
            what: "iq soa scale record",
            expected: (base + record) as u64,
            got: scales.len() as u64,
        }),
    }
}

/// Validates a [`RepackedTensor`] holding an IQ tensor against its own
/// geometry: scheme agreement, value-region length over the nibble or
/// index shape, and SoA scale-region length. Returns the parsed kind,
/// the index dims (grid types), the record length and the K-block
/// count. A hand-built tensor with inconsistent regions is a length
/// error, never a panic (CONVENTIONS.md §1.5).
fn repacked_geometry_iq(
    t: &RepackedTensor,
) -> Result<(IqKind, PaddedDims, PaddedDims, usize, usize), FormatError> {
    let kind = match iq_kind(t.ggml) {
        Some(kind) => kind,
        None => {
            return Err(FormatError::SchemeMismatch {
                scheme: t.ggml.name(),
                expected: "iq ggml type",
                got: t.ggml.name(),
            });
        }
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
    let dims = &t.dims;
    let record = record_len(kind);
    // SoA grouping follows the wire block: per 32 for `Iq4Nl`, per
    // 256 for every other family (mirrors `repack_outer_block`).
    let outer: usize = match kind {
        IqKind::Iq4Nl => 32,
        IqKind::Iq4Xs
        | IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => 256,
    };
    let value_dims = match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => *dims,
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => index_dims(dims, kind)?,
    };
    let packing = match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => crate::layout::Packing::Nibble4,
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => crate::layout::Packing::Byte,
    };
    let expected_values = value_dims.value_region_bytes(packing)?;
    if t.values.len() as u64 != expected_values {
        return Err(FormatError::LengthMismatch {
            what: "iq repacked value region",
            expected: expected_values,
            got: t.values.len() as u64,
        });
    }
    let n_blocks = dims.n_padded() as usize / 16;
    let k_blocks = dims.k_padded() as usize / outer;
    let expected_scales = n_blocks * k_blocks * 16 * record;
    if t.scales.len() != expected_scales {
        return Err(FormatError::LengthMismatch {
            what: "iq repacked scale region",
            expected: expected_scales as u64,
            got: t.scales.len() as u64,
        });
    }
    Ok((kind, *dims, value_dims, record, k_blocks))
}

/// Repacks IQ GGUF wire bytes into canonical `L1` (Spec 2 §7; card
/// A2.4): values permute into tile order (nibbles per weight for the
/// IQ4 types, packed index bytes over `[N, K/g]` for the grid types),
/// scale bytes move verbatim into the §3.1 SoA region. Padding rows
/// and columns are zero in both regions (Spec 2 §2.2). Called from
/// [`crate::repack::repack`] for IQ types.
pub(crate) fn iq_repack(
    ggml: GgmlType,
    wire: &[u8],
    n_rows: u32,
    k: u32,
) -> Result<RepackedTensor, FormatError> {
    let kind = match iq_kind(ggml) {
        Some(kind) => kind,
        None => {
            return Err(FormatError::SchemeMismatch {
                scheme: ggml.name(),
                expected: "iq ggml type",
                got: ggml.name(),
            });
        }
    };
    let geo = wire_geometry(ggml, wire, n_rows, k)?;
    let dims = PaddedDims::new(n_rows, k, ggml.superblock_k())?;
    let block_bytes = ggml.block_bytes() as usize;
    let reclen = record_len(kind);
    let row_blocks = geo.blocks_per_row as usize;
    let mut parsed_idx = vec![0u8; wire_index_len(kind)];
    let mut parsed_rec = vec![0u8; reclen];
    // Row-major logical values over the padded value shape, zero-filled
    // so padding is zero by construction (Spec 2 §2.2).
    let n_pad = dims.n_padded() as usize;
    // One wire block contributes exactly wire_index_len bytes: nibbles
    // per weight for the IQ4 types, packed index bytes otherwise.
    let value_count = wire_index_len(kind);
    let (value_dims, value_stride) = match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => (dims, dims.k_padded() as usize),
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => {
            let idims = index_dims(&dims, kind)?;
            (idims, idims.k_padded() as usize)
        }
    };
    let k_pad = value_dims.k_padded() as usize;
    let mut logical = vec![0u8; n_pad * k_pad];
    for row in 0..n_rows as usize {
        for b in 0..row_blocks {
            let base = (row * row_blocks + b) * block_bytes;
            parse_block(
                kind,
                ggml,
                &wire[base..base + block_bytes],
                &mut parsed_idx,
                &mut parsed_rec,
            )?;
            let dest = row * value_stride + b * value_count;
            logical[dest..dest + value_count].copy_from_slice(&parsed_idx[..value_count]);
        }
    }
    let values = match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => crate::permute::l1_pack_nibbles(&logical, &value_dims)?,
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => crate::permute::l1_pack_bytes(&logical, &value_dims)?,
    };
    // SoA scale region in [N/16][K/B][16] order (Spec 2 §3.1, §7
    // step 4, B = 32 for `Iq4Nl` else 256): record bytes move verbatim
    // from their wire block; padding rows beyond n_rows get zero
    // records.
    let outer: usize = match kind {
        IqKind::Iq4Nl => 32,
        IqKind::Iq4Xs
        | IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => 256,
    };
    let n_blocks = n_pad / 16;
    let k_blocks = dims.k_padded() as usize / outer;
    let mut scales = vec![0u8; n_blocks * k_blocks * 16 * reclen];
    for nb in 0..n_blocks {
        for kb in 0..k_blocks {
            for row in 0..16 {
                let source = nb * 16 + row;
                let dest = ((nb * k_blocks + kb) * 16 + row) * reclen;
                if source < geo.n_rows as usize {
                    let base = (source * row_blocks + kb) * block_bytes;
                    parse_block(
                        kind,
                        ggml,
                        &wire[base..base + block_bytes],
                        &mut parsed_idx,
                        &mut parsed_rec,
                    )?;
                    scales[dest..dest + reclen].copy_from_slice(&parsed_rec);
                }
            }
        }
    }
    Ok(RepackedTensor {
        ggml,
        scheme: ggml.scheme(),
        dims,
        values,
        scales,
    })
}

/// Exact inverse of `iq_repack` (Spec 2 §7 step 4): `L1` values
/// unpack to row-major logical indices and SoA records regroup to
/// wire-block order, reproducing the input wire bytes exactly.
/// Padding rows and columns are dropped, never emitted. Called from
/// [`crate::repack::unpack_repacked`] for IQ types.
pub(crate) fn iq_unpack(t: &RepackedTensor) -> Result<Vec<u8>, FormatError> {
    let (kind, dims, value_dims, reclen, k_blocks) = repacked_geometry_iq(t)?;
    let ggml = t.ggml;
    let logical = match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => crate::permute::l1_unpack_nibbles(&t.values, &value_dims)?,
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => crate::permute::l1_unpack_bytes(&t.values, &value_dims)?,
    };
    let n_rows = dims.n() as usize;
    let k = dims.k() as usize;
    let block_len = ggml.block_len() as usize;
    let block_bytes = ggml.block_bytes() as usize;
    let wire_blocks_per_row = k / block_len;
    let value_count = wire_index_len(kind);
    let value_stride = value_dims.k_padded() as usize;
    let mut wire = vec![0u8; n_rows * wire_blocks_per_row * block_bytes];
    let mut record_buf = vec![0u8; reclen];
    for row in 0..n_rows {
        for b in 0..wire_blocks_per_row {
            let vals = &logical[row * value_stride + b * value_count..][..value_count];
            let rec = soa_record(&t.scales, reclen, k_blocks, row / 16, b, row % 16)?;
            record_buf.copy_from_slice(rec);
            let dest = (row * wire_blocks_per_row + b) * block_bytes;
            write_block(
                kind,
                ggml,
                vals,
                &record_buf,
                &mut wire[dest..dest + block_bytes],
            )?;
        }
    }
    Ok(wire)
}

/// Reference dequantization of repacked IQ bytes (Spec 2 §3.3, §10):
/// decodes the `L1` value region plus the SoA scale region to `n*k`
/// row-major `f32` values, ascending, via `repacked_block`. Called
/// from [`crate::repack::repack_dequantize`] for IQ types.
pub(crate) fn repacked_dequantize(t: &RepackedTensor) -> Result<Vec<f32>, FormatError> {
    let (kind, dims, value_dims, reclen, k_blocks) = repacked_geometry_iq(t)?;
    let ggml = t.ggml;
    let logical = match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => crate::permute::l1_unpack_nibbles(&t.values, &value_dims)?,
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => crate::permute::l1_unpack_bytes(&t.values, &value_dims)?,
    };
    let n = dims.n() as usize;
    let k = dims.k() as usize;
    let block_len = ggml.block_len() as usize;
    let wire_blocks_per_row = k / block_len;
    let value_count = wire_index_len(kind);
    let value_stride = value_dims.k_padded() as usize;
    let scheme_name = match ggml.scheme() {
        Some(s) => s.name(),
        None => ggml.name(),
    };
    let mut out = Vec::with_capacity(n * k);
    let mut problems = Vec::new();
    for row in 0..n {
        for b in 0..wire_blocks_per_row {
            let vals = &logical[row * value_stride + b * value_count..][..value_count];
            let index = (row * wire_blocks_per_row + b) as u64;
            match soa_record(&t.scales, reclen, k_blocks, row / 16, b, row % 16) {
                Ok(rec) => match repacked_block(kind, scheme_name, index, vals, rec, &mut out) {
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

/// Maps a [`crate::SchemeId`] to its [`IqKind`] where one exists (card A2.4, card A2.5).
pub(crate) fn scheme_iq_kind(scheme: crate::SchemeId) -> Option<IqKind> {
    match scheme {
        crate::SchemeId::I4Nl => Some(IqKind::Iq4Nl),
        crate::SchemeId::I4Xs => Some(IqKind::Iq4Xs),
        crate::SchemeId::Iq3Xxs => Some(IqKind::Iq3Xxs),
        crate::SchemeId::Iq3S => Some(IqKind::Iq3S),
        crate::SchemeId::Iq2Xxs => Some(IqKind::Iq2Xxs),
        crate::SchemeId::Iq2Xs => Some(IqKind::Iq2Xs),
        crate::SchemeId::Iq2S => Some(IqKind::Iq2S),
        crate::SchemeId::Iq1S => Some(IqKind::Iq1S),
        crate::SchemeId::Iq1M => Some(IqKind::Iq1M),
        crate::SchemeId::I8R
        | crate::SchemeId::I8B128
        | crate::SchemeId::I4K
        | crate::SchemeId::E4M3B128
        | crate::SchemeId::I8B32F
        | crate::SchemeId::I4B32F
        | crate::SchemeId::I4B32FM
        | crate::SchemeId::I5B32F
        | crate::SchemeId::I5B32FM
        | crate::SchemeId::I5K
        | crate::SchemeId::I6K
        | crate::SchemeId::I3K
        | crate::SchemeId::I2K => None,
    }
}

/// Returns the value dimensions for IQ schemes (authoritative packed index dims
/// for grid IQ types; original dims for IQ4 types).
pub(crate) fn iq_value_dims(dims: &PaddedDims, kind: IqKind) -> Result<PaddedDims, FormatError> {
    match kind {
        IqKind::Iq4Nl | IqKind::Iq4Xs => Ok(*dims),
        IqKind::Iq3Xxs
        | IqKind::Iq3S
        | IqKind::Iq2Xxs
        | IqKind::Iq2Xs
        | IqKind::Iq2S
        | IqKind::Iq1S
        | IqKind::Iq1M => index_dims(dims, kind),
    }
}
