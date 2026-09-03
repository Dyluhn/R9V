// SPDX-License-Identifier: Apache-2.0
//! `L1` forward/inverse permutation and element codecs (Spec 2 §2.2).
//!
//! The lane/element assignment is packing-independent; these functions
//! move logical elements between row-major order and tile order, then
//! encode tiles to the byte forms of the §2.2 table. All entry points
//! validate untrusted buffers and dimensions into [`FormatError`]
//! (CONVENTIONS.md §1.4, §1.5): length mismatches, out-of-range nibbles
//! and nonzero padding are reported with positions, never panicked on.

use crate::layout::{Packing, PaddedDims, ELEMS_PER_LANE, ELEMS_PER_TILE, LANES_PER_TILE};
use crate::FormatError;

/// Tile-order image of padded row-major logical elements (Spec 2 §2.2).
/// `src` holds `n_padded * k_padded` values in row-major order; element
/// values are logical (full-range `u16`) regardless of packing.
pub fn l1_forward_elems(src: &[u16], dims: &PaddedDims) -> Result<Vec<u16>, FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    expect_len(src.len() as u64, total, "l1 row-major elements")?;
    let mut dst = vec![0u16; total as usize];
    for n in 0..dims.n_padded() {
        for k in 0..dims.k_padded() {
            let pos = crate::layout::l1_forward_index(n, k, dims)?;
            // Internal invariant: pos < total by the forward-index law,
            // and both indices are in range by construction above.
            dst[pos as usize] = src[(n as u64 * dims.k_padded() as u64 + k as u64) as usize];
        }
    }
    Ok(dst)
}

/// Row-major image of tile-order logical elements (inverse of
/// [`l1_forward_elems`]; Spec 2 §2.2).
pub fn l1_inverse_elems(tiled: &[u16], dims: &PaddedDims) -> Result<Vec<u16>, FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    expect_len(tiled.len() as u64, total, "l1 tile-order elements")?;
    let mut dst = vec![0u16; total as usize];
    for (pos, value) in tiled.iter().enumerate() {
        let (n, k) = crate::layout::l1_inverse_index(pos as u64, dims)?;
        // Internal invariant: (n, k) is in range by the inverse-index law.
        dst[(n as u64 * dims.k_padded() as u64 + k as u64) as usize] = *value;
    }
    Ok(dst)
}

/// Zero-pads unpadded `(n, k)` row-major elements to the padded shape
/// (Spec 2 §2.2: padding rows and columns are zero).
pub fn pad_row_major_elems(
    src: &[u16],
    n: u32,
    k: u32,
    dims: &PaddedDims,
) -> Result<Vec<u16>, FormatError> {
    if n != dims.n() || k != dims.k() {
        return Err(FormatError::LengthMismatch {
            what: "unpadded row-major elements",
            expected: n as u64 * k as u64,
            got: src.len() as u64,
        });
    }
    expect_len(
        src.len() as u64,
        n as u64 * k as u64,
        "unpadded row-major elements",
    )?;
    let mut dst = vec![0u16; (dims.n_padded() as u64 * dims.k_padded() as u64) as usize];
    for row in 0..n {
        for col in 0..k {
            dst[(row as u64 * dims.k_padded() as u64 + col as u64) as usize] =
                src[(row as u64 * k as u64 + col as u64) as usize];
        }
    }
    Ok(dst)
}

/// Requires `tiled` padding positions to read back as zero (Spec 2
/// §2.2); reports every nonzero position (CONVENTIONS.md §1.4).
pub fn verify_padding_zeros_elems(tiled: &[u16], dims: &PaddedDims) -> Result<(), FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    expect_len(tiled.len() as u64, total, "l1 tile-order elements")?;
    let mut problems = Vec::new();
    for n in 0..dims.n_padded() {
        for k in 0..dims.k_padded() {
            if n < dims.n() && k < dims.k() {
                continue;
            }
            let pos = crate::layout::l1_forward_index(n, k, dims)?;
            let value = tiled[pos as usize];
            if value != 0 {
                problems.push(FormatError::PaddingNonzero {
                    row: n,
                    col: k,
                    value: value as u64,
                });
            }
        }
    }
    FormatError::collect(problems)
}

/// Forward permutation for byte types (`i8`/`e4m3`/`e5m2`; Spec 2
/// §2.2 table): padded row-major bytes to tile-order bytes.
pub fn l1_pack_bytes(src: &[u8], dims: &PaddedDims) -> Result<Vec<u8>, FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    expect_len(src.len() as u64, total, "l1 row-major bytes")?;
    let mut dst = vec![0u8; total as usize];
    for n in 0..dims.n_padded() {
        for k in 0..dims.k_padded() {
            let pos = crate::layout::l1_forward_index(n, k, dims)?;
            dst[pos as usize] = src[(n as u64 * dims.k_padded() as u64 + k as u64) as usize];
        }
    }
    Ok(dst)
}

/// Inverse permutation for byte types (Spec 2 §2.2 table).
pub fn l1_unpack_bytes(tiled: &[u8], dims: &PaddedDims) -> Result<Vec<u8>, FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    expect_len(tiled.len() as u64, total, "l1 tile-order bytes")?;
    let mut dst = vec![0u8; total as usize];
    for (pos, value) in tiled.iter().enumerate() {
        let (n, k) = crate::layout::l1_inverse_index(pos as u64, dims)?;
        dst[(n as u64 * dims.k_padded() as u64 + k as u64) as usize] = *value;
    }
    Ok(dst)
}

/// Requires byte-tile padding positions to read back as zero (Spec 2 §2.2).
pub fn verify_padding_zeros_bytes(tiled: &[u8], dims: &PaddedDims) -> Result<(), FormatError> {
    let as_wide: Vec<u16> = tiled.iter().map(|v| *v as u16).collect();
    verify_padding_zeros_elems(&as_wide, dims)
}

/// Forward permutation for 16-bit types (`f16`/`bf16`; Spec 2 §2.2
/// table): padded row-major raw half bits to tile-order halves.
/// Values are raw 16-bit patterns; no float math happens here.
pub fn l1_pack_halfs(src: &[u16], dims: &PaddedDims) -> Result<Vec<u16>, FormatError> {
    l1_forward_elems(src, dims)
}

/// Inverse permutation for 16-bit types (Spec 2 §2.2 table).
pub fn l1_unpack_halfs(tiled: &[u16], dims: &PaddedDims) -> Result<Vec<u16>, FormatError> {
    l1_inverse_elems(tiled, dims)
}

/// Encodes tile-order halves to little-endian bytes (one 128-bit load
/// per lane; Spec 2 §2.2 table). Length must be a whole tile stream.
pub fn encode_halfs_le(tiled: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tiled.len() * 2);
    for v in tiled {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decodes little-endian bytes to tile-order halves; the byte length
/// must be even (Spec 2 §2.2 table).
pub fn decode_halfs_le(bytes: &[u8]) -> Result<Vec<u16>, FormatError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(FormatError::LengthMismatch {
            what: "half16 little-endian bytes",
            expected: bytes.len() as u64 + 1,
            got: bytes.len() as u64,
        });
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

/// Forward permutation for `i4` nibbles (Spec 2 §2.2 table): padded
/// row-major 4-bit values to tile-order nibble bytes with the low
/// nibble holding the lower k. Every value must fit in 4 bits; all
/// violations are reported (CONVENTIONS.md §1.4).
pub fn l1_pack_nibbles(src: &[u8], dims: &PaddedDims) -> Result<Vec<u8>, FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    expect_len(src.len() as u64, total, "l1 row-major nibbles")?;
    if !total.is_multiple_of(2) {
        return Err(FormatError::Overflow {
            what: "l1_pack_nibbles",
            detail: format!("padded element count {total} is not a nibble pair"),
        });
    }
    let mut problems = Vec::new();
    for (pos, value) in src.iter().enumerate() {
        if *value >= 16 {
            problems.push(FormatError::ValueOutOfRange {
                what: "nibble",
                position: pos as u64,
                value: *value as u64,
            });
        }
    }
    FormatError::collect(problems)?;
    // Packing stays inside one lane: a lane's 8 K-consecutive elements
    // become 4 bytes, so one 32-bit load per lane (Spec 2 §2.2 table)
    // reads exactly one lane, and the low nibble holds the lower k.
    let tiled = l1_pack_bytes(src, dims)?;
    let mut out = Vec::with_capacity(total as usize / 2);
    for tile in tiled.chunks_exact(ELEMS_PER_TILE as usize) {
        for lane in 0..LANES_PER_TILE as usize {
            let base = lane * ELEMS_PER_LANE as usize;
            for m in 0..ELEMS_PER_LANE as usize / 2 {
                out.push(tile[base + 2 * m] | (tile[base + 2 * m + 1] << 4));
            }
        }
    }
    Ok(out)
}

/// Inverse permutation for `i4` nibbles (Spec 2 §2.2 table).
pub fn l1_unpack_nibbles(tiled: &[u8], dims: &PaddedDims) -> Result<Vec<u8>, FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    if !total.is_multiple_of(2) {
        return Err(FormatError::Overflow {
            what: "l1_unpack_nibbles",
            detail: format!("padded element count {total} is not a nibble pair"),
        });
    }
    expect_len(tiled.len() as u64, total / 2, "l1 tile-order nibble bytes")?;
    // Expansion mirrors packing: 4 bytes per lane become that lane's
    // 8 elements, low nibble first (Spec 2 §2.2 table).
    let mut expanded = Vec::with_capacity(total as usize);
    // One tile holds 256 elements as 128 nibble bytes.
    for tile in tiled.chunks_exact(ELEMS_PER_TILE as usize / 2) {
        for lane in 0..LANES_PER_TILE as usize {
            let base = lane * (ELEMS_PER_LANE as usize / 2);
            for m in 0..ELEMS_PER_LANE as usize / 2 {
                let byte = tile[base + m];
                expanded.push(byte & 0x0F);
                expanded.push(byte >> 4);
            }
        }
    }
    l1_unpack_bytes(&expanded, dims)
}

/// Requires nibble-tile padding positions to read back as zero
/// (Spec 2 §2.2).
pub fn verify_padding_zeros_nibbles(tiled: &[u8], dims: &PaddedDims) -> Result<(), FormatError> {
    let expanded = l1_unpack_nibbles_for_check(tiled, dims)?;
    verify_padding_zeros_bytes(&expanded, dims)
}

/// Expands tile-order nibble bytes to the tile-order element stream for
/// padding verification, mirroring [`l1_unpack_nibbles`] lane by lane
/// (Spec 2 §2.2 table).
fn l1_unpack_nibbles_for_check(tiled: &[u8], dims: &PaddedDims) -> Result<Vec<u8>, FormatError> {
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    if !total.is_multiple_of(2) {
        return Err(FormatError::Overflow {
            what: "l1 nibble check",
            detail: format!("padded element count {total} is not a nibble pair"),
        });
    }
    expect_len(tiled.len() as u64, total / 2, "l1 tile-order nibble bytes")?;
    let mut expanded = Vec::with_capacity(total as usize);
    // One tile holds 256 elements as 128 nibble bytes.
    for tile in tiled.chunks_exact(ELEMS_PER_TILE as usize / 2) {
        for lane in 0..LANES_PER_TILE as usize {
            let base = lane * (ELEMS_PER_LANE as usize / 2);
            for m in 0..ELEMS_PER_LANE as usize / 2 {
                let byte = tile[base + m];
                expanded.push(byte & 0x0F);
                expanded.push(byte >> 4);
            }
        }
    }
    Ok(expanded)
}

/// Forward permutation for scheme-defined bit planes (Spec 2 §2.2
/// table, §3.3): padded row-major values to tile-order plane bytes.
/// Each lane's 8 values pack LSB-first (element 0 = lowest bits);
/// every value must fit in `bits` bits and all violations are
/// reported (CONVENTIONS.md §1.4).
pub fn l1_pack_planes(src: &[u16], dims: &PaddedDims, bits: u8) -> Result<Vec<u8>, FormatError> {
    let packing = Packing::bit_planes(bits)?;
    let total = dims.n_padded() as u64 * dims.k_padded() as u64;
    expect_len(src.len() as u64, total, "l1 row-major plane values")?;
    let limit: u16 = 1u16 << bits;
    let mut problems = Vec::new();
    for (pos, value) in src.iter().enumerate() {
        if *value >= limit {
            problems.push(FormatError::ValueOutOfRange {
                what: "bit-plane value",
                position: pos as u64,
                value: *value as u64,
            });
        }
    }
    FormatError::collect(problems)?;
    let tiled = l1_forward_elems(src, dims)?;
    let mut out = Vec::with_capacity(
        dims.tile_count() as usize * packing.bytes_per_lane() as usize * LANES_PER_TILE as usize,
    );
    let per_lane_bytes = packing.bytes_per_lane() as usize;
    for tile in tiled.chunks_exact(ELEMS_PER_TILE as usize) {
        for lane in 0..LANES_PER_TILE as usize {
            let mut acc: u64 = 0;
            for j in 0..ELEMS_PER_LANE as usize {
                acc |= (tile[lane * ELEMS_PER_LANE as usize + j] as u64) << (j * bits as usize);
            }
            for b in 0..per_lane_bytes {
                out.push((acc >> (b * 8)) as u8);
            }
        }
    }
    Ok(out)
}

/// Inverse permutation for scheme-defined bit planes (Spec 2 §2.2
/// table, §3.3).
pub fn l1_unpack_planes(
    tiled: &[u8],
    dims: &PaddedDims,
    bits: u8,
) -> Result<Vec<u16>, FormatError> {
    let packing = Packing::bit_planes(bits)?;
    let per_lane_bytes = packing.bytes_per_lane() as usize;
    let expected = dims.tile_count() * packing.tile_bytes();
    expect_len(tiled.len() as u64, expected, "l1 tile-order plane bytes")?;
    let mask: u64 = (1u64 << bits) - 1;
    let mut elems = Vec::with_capacity((dims.n_padded() as u64 * dims.k_padded() as u64) as usize);
    for tile in tiled.chunks_exact(packing.tile_bytes() as usize) {
        for lane in 0..LANES_PER_TILE as usize {
            let mut acc: u64 = 0;
            for b in 0..per_lane_bytes {
                acc |= (tile[lane * per_lane_bytes + b] as u64) << (b * 8);
            }
            for j in 0..ELEMS_PER_LANE as usize {
                elems.push(((acc >> (j * bits as usize)) & mask) as u16);
            }
        }
    }
    l1_inverse_elems(&elems, dims)
}

/// Requires plane-tile padding positions to read back as zero
/// (Spec 2 §2.2).
pub fn verify_padding_zeros_planes(
    tiled: &[u8],
    dims: &PaddedDims,
    bits: u8,
) -> Result<(), FormatError> {
    let elems = l1_unpack_planes_to_elems(tiled, dims, bits)?;
    verify_padding_zeros_elems(&elems, dims)
}

/// Unpacks tile-order plane bytes to tile-order elements (shared by
/// [`l1_unpack_planes`] verification; Spec 2 §2.2 table, §3.3).
fn l1_unpack_planes_to_elems(
    tiled: &[u8],
    dims: &PaddedDims,
    bits: u8,
) -> Result<Vec<u16>, FormatError> {
    let packing = Packing::bit_planes(bits)?;
    let per_lane_bytes = packing.bytes_per_lane() as usize;
    let expected = dims.tile_count() * packing.tile_bytes();
    expect_len(tiled.len() as u64, expected, "l1 tile-order plane bytes")?;
    let mask: u64 = (1u64 << bits) - 1;
    let mut elems = Vec::with_capacity((dims.n_padded() as u64 * dims.k_padded() as u64) as usize);
    for tile in tiled.chunks_exact(packing.tile_bytes() as usize) {
        for lane in 0..LANES_PER_TILE as usize {
            let mut acc: u64 = 0;
            for b in 0..per_lane_bytes {
                acc |= (tile[lane * per_lane_bytes + b] as u64) << (b * 8);
            }
            for j in 0..ELEMS_PER_LANE as usize {
                elems.push(((acc >> (j * bits as usize)) & mask) as u16);
            }
        }
    }
    Ok(elems)
}

/// Tile-order lane base for K-group halves: the byte offset where
/// lane `lane` starts inside one tile of `packing` (Spec 2 §2.2: one
/// 64-bit load per lane for byte types, one 32-bit load for int4).
pub fn lane_byte_offset(lane: u32, packing: Packing) -> Result<u64, FormatError> {
    if lane >= LANES_PER_TILE {
        return Err(FormatError::InvalidDim {
            name: "lane",
            value: lane as u64,
            reason: "must be below 32",
        });
    }
    Ok(lane as u64 * packing.bytes_per_lane())
}

/// Rejects a length mismatch with required and actual byte counts
/// (CONVENTIONS.md §1.3: errors carry the numbers).
fn expect_len(got: u64, expected: u64, what: &'static str) -> Result<(), FormatError> {
    if got != expected {
        return Err(FormatError::LengthMismatch {
            what,
            expected,
            got,
        });
    }
    Ok(())
}
