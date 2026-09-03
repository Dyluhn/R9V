// SPDX-License-Identifier: Apache-2.0
//! `L1S` tiled 2:4 structured-sparse layout (Spec 2 §2.3; card A2.1).
//!
//! Values are `L1` over the compressed K dimension (`K/2`); the index
//! region holds 2 bits per kept element in lane order. The exact
//! SWMMAC operand order is recorded as SI-14 (spec 2 §2.3 defers to
//! spec 4, which names only the wrapper); this module reuses the
//! A0.S1-verified §2.2 lane formula over the compressed-K tile.

use crate::layout::{PaddedDims, LANES_PER_TILE};
use crate::FormatError;

/// Index bits per kept element (Spec 2 §2.3: 2 bits per kept element).
pub const L1S_INDEX_BITS: u32 = 2;
/// Kept elements per lane in the compressed-K tile: 8 compressed-K
/// columns per lane half-pair become 4 kept slots per lane.
pub const L1S_KEPT_PER_LANE: u32 = 4;
/// Kept elements per compressed tile (16 rows × 8 kept columns).
pub const L1S_KEPT_PER_TILE: u64 = 128;
/// Index bytes per compressed tile (128 kept × 2 bits).
pub const L1S_INDEX_BYTES_PER_TILE: u64 = 32;

/// Compressed-K dims for the `L1S` value region: `L1` over `K/2`
/// (Spec 2 §2.3). Padding keeps the value region a well-formed `L1`
/// tensor: padded K is always a multiple of 16 (hence even), so the
/// halved width pads exactly like the dense half.
pub fn l1s_value_dims(
    dims: &PaddedDims,
    superblock_k: Option<u32>,
) -> Result<PaddedDims, FormatError> {
    PaddedDims::new(dims.n(), dims.k() / 2 + dims.k() % 2, superblock_k)
}

/// Lane slot for kept element `m` of a lane (Spec 2 §2.3 index region
/// in §2.2 lane order per SI-14: `lane = kgroup * 16 + n`, four kept
/// slots per lane).
pub fn l1s_index_lane(n_in_tile: u32, kgroup: u32) -> Result<u32, FormatError> {
    crate::layout::l1_lane(n_in_tile, kgroup)
}

/// Byte length of the `L1S` index region for `tiles` compressed tiles
/// (Spec 2 §2.3: 32 bytes per tile); checked, never saturating.
pub fn l1s_index_region_bytes(tiles: u64) -> Result<u64, FormatError> {
    tiles
        .checked_mul(L1S_INDEX_BYTES_PER_TILE)
        .ok_or_else(|| FormatError::Overflow {
            what: "l1s_index_region_bytes",
            detail: format!("tiles={tiles}"),
        })
}

/// Entry offsets for one `L1S` tensor (Spec 2 §6 region order:
/// values → scales → indices). All offsets are byte offsets within
/// the tensor entry; arithmetic is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L1sRegions {
    /// Byte length of the compressed-value region.
    pub values_bytes: u64,
    /// Byte length of the scale region.
    pub scales_bytes: u64,
    /// Byte length of the index region.
    pub indices_bytes: u64,
}

impl L1sRegions {
    /// Computes region sizes from the compressed-value dims, the
    /// scale-region byte length (card A2.2 owns record contents), and
    /// the compressed tile count (Spec 2 §2.3, §6, §7).
    pub fn new(
        value_dims: &PaddedDims,
        value_tile_bytes: u64,
        scales_bytes: u64,
    ) -> Result<Self, FormatError> {
        let values_bytes = value_dims
            .tile_count()
            .checked_mul(value_tile_bytes)
            .ok_or_else(|| FormatError::Overflow {
                what: "l1s values_bytes",
                detail: format!(
                    "tiles={} tile_bytes={value_tile_bytes}",
                    value_dims.tile_count()
                ),
            })?;
        let indices_bytes = l1s_index_region_bytes(value_dims.tile_count())?;
        Ok(Self {
            values_bytes,
            scales_bytes,
            indices_bytes,
        })
    }

    /// Byte offset of the value region (always zero; Spec 2 §6).
    pub const fn values_offset(self) -> u64 {
        0
    }
    /// Byte offset of the scale region (Spec 2 §6).
    pub const fn scales_offset(self) -> u64 {
        self.values_bytes
    }
    /// Byte offset of the index region (Spec 2 §6).
    pub fn indices_offset(self) -> Result<u64, FormatError> {
        self.values_bytes
            .checked_add(self.scales_bytes)
            .ok_or_else(|| FormatError::Overflow {
                what: "l1s indices_offset",
                detail: format!("values={} scales={}", self.values_bytes, self.scales_bytes),
            })
    }
    /// Total entry bytes (Spec 2 §6, §7).
    pub fn total_bytes(self) -> Result<u64, FormatError> {
        self.indices_offset()?
            .checked_add(self.indices_bytes)
            .ok_or_else(|| FormatError::Overflow {
                what: "l1s total_bytes",
                detail: format!(
                    "indices={} bytes={}",
                    self.indices_offset().unwrap_or(u64::MAX),
                    self.indices_bytes
                ),
            })
    }
}

/// Entry offsets for one dense `L1` tensor (Spec 2 §6 region order:
/// values → scales).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L1Regions {
    /// Byte length of the value region.
    pub values_bytes: u64,
    /// Byte length of the scale region.
    pub scales_bytes: u64,
}

impl L1Regions {
    /// Computes region sizes from validated dims, packing tile bytes,
    /// and the scale-region byte length (Spec 2 §2.2, §6, §7).
    pub fn new(
        dims: &PaddedDims,
        value_tile_bytes: u64,
        scales_bytes: u64,
    ) -> Result<Self, FormatError> {
        let values_bytes = dims
            .tile_count()
            .checked_mul(value_tile_bytes)
            .ok_or_else(|| FormatError::Overflow {
                what: "l1 values_bytes",
                detail: format!("tiles={} tile_bytes={value_tile_bytes}", dims.tile_count()),
            })?;
        Ok(Self {
            values_bytes,
            scales_bytes,
        })
    }

    /// Byte offset of the value region (always zero; Spec 2 §6).
    pub const fn values_offset(self) -> u64 {
        0
    }
    /// Byte offset of the scale region (Spec 2 §6).
    pub const fn scales_offset(self) -> u64 {
        self.values_bytes
    }
    /// Total entry bytes (Spec 2 §6, §7).
    pub fn total_bytes(self) -> Result<u64, FormatError> {
        self.values_bytes
            .checked_add(self.scales_bytes)
            .ok_or_else(|| FormatError::Overflow {
                what: "l1 total_bytes",
                detail: format!("values={} scales={}", self.values_bytes, self.scales_bytes),
            })
    }
}

/// Packs kept-element indices (each 0..4) given in compressed-tile
/// lane order into index bytes (Spec 2 §2.3): one byte per lane holds
/// that lane's 4 kept slots, slot 0 in the lowest 2 bits.
/// All violations are reported (CONVENTIONS.md §1.4).
// DECISION(A2.1): index bytes reuse the §2.2 lane formula over the
// compressed-K tile with slot 0 in the lowest 2 bits; rejected
// inventing a separate SWMMAC order because spec 4 fixes none (SI-14).
pub fn l1s_pack_indices(kept: &[u8], value_dims: &PaddedDims) -> Result<Vec<u8>, FormatError> {
    let tiles = value_dims.tile_count();
    let expected = tiles
        .checked_mul(L1S_KEPT_PER_TILE)
        .ok_or_else(|| FormatError::Overflow {
            what: "l1s kept indices",
            detail: format!("tiles={tiles}"),
        })?;
    if kept.len() as u64 != expected {
        return Err(FormatError::LengthMismatch {
            what: "l1s kept indices",
            expected,
            got: kept.len() as u64,
        });
    }
    let mut problems = Vec::new();
    for (pos, value) in kept.iter().enumerate() {
        if *value >= 4 {
            problems.push(FormatError::ValueOutOfRange {
                what: "l1s index",
                position: pos as u64,
                value: *value as u64,
            });
        }
    }
    FormatError::collect(problems)?;
    let mut out = Vec::with_capacity(tiles as usize * L1S_INDEX_BYTES_PER_TILE as usize);
    for tile in kept.chunks_exact(L1S_KEPT_PER_TILE as usize) {
        for lane in 0..LANES_PER_TILE as usize {
            let base = lane * L1S_KEPT_PER_LANE as usize;
            let mut byte = 0u8;
            for m in 0..L1S_KEPT_PER_LANE as usize {
                byte |= (tile[base + m] & 0x03) << (m * L1S_INDEX_BITS as usize);
            }
            out.push(byte);
        }
    }
    Ok(out)
}

/// Unpacks index bytes to kept-element indices in compressed-tile
/// lane order (inverse of [`l1s_pack_indices`]; Spec 2 §2.3).
pub fn l1s_unpack_indices(bytes: &[u8], value_dims: &PaddedDims) -> Result<Vec<u8>, FormatError> {
    let tiles = value_dims.tile_count();
    let expected = l1s_index_region_bytes(tiles)?;
    if bytes.len() as u64 != expected {
        return Err(FormatError::LengthMismatch {
            what: "l1s index bytes",
            expected,
            got: bytes.len() as u64,
        });
    }
    let mut out = Vec::with_capacity(tiles as usize * L1S_KEPT_PER_TILE as usize);
    for tile in bytes.chunks_exact(L1S_INDEX_BYTES_PER_TILE as usize) {
        for byte in tile {
            for m in 0..L1S_KEPT_PER_LANE as usize {
                out.push((byte >> (m * L1S_INDEX_BITS as usize)) & 0x03);
            }
        }
    }
    Ok(out)
}
