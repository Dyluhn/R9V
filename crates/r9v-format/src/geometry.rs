// SPDX-License-Identifier: Apache-2.0
//! Scale-record sizes and SoA placement (Spec 2 §3.1, §3.2; card A2.2).
//!
//! Per-scheme record sizes, outer blocks, and the §3.1 SoA geometry
//! (`[N/16][K/B][16 records]`) over validated [`PaddedDims`].

use crate::layout::{Layout, PaddedDims};
use crate::scheme::{NativeScheme, SchemeId};
use crate::FormatError;

/// Byte length of one scale record (Spec 2 §3.2: 2 B `f16`, 16 B
/// `I4_K` record). Repack-only ids fail closed with their owning card.
pub fn scale_record_bytes(scheme: SchemeId) -> Result<u32, FormatError> {
    match scheme.as_native()? {
        NativeScheme::I8R | NativeScheme::I8B128 | NativeScheme::E4M3B128 => Ok(2),
        NativeScheme::I4K => Ok(16),
    }
}

/// Outer block `B` of the §3.1 SoA grouping (superblock where one
/// exists, else the block). Returns `None` for row-wise
/// [`SchemeId::I8R`], where `B` is the padded row length itself
/// (Spec 2 §3.2: block `row`). Repack-only ids fail closed.
pub fn outer_block(scheme: SchemeId) -> Result<Option<u32>, FormatError> {
    match scheme.as_native()? {
        NativeScheme::I8R => Ok(None),
        NativeScheme::I8B128 | NativeScheme::E4M3B128 => Ok(Some(128)),
        NativeScheme::I4K => Ok(Some(256)),
    }
}

/// SoA scale-region geometry for one native tensor (Spec 2 §3.1:
/// `[N/16][K/B][16 records]`; one wave loads `16 × record_size`
/// contiguous bytes for row-block `nb`, K-block `kb`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScaleGeometry {
    /// The scheme whose records are grouped.
    pub scheme: SchemeId,
    /// Row-blocks (`N/16` over padded N).
    pub n_blocks: u64,
    /// K-blocks (`K/B` over padded K).
    pub k_blocks: u64,
    /// Bytes per scale record (2 for `f16`, 16 for `I4_K`).
    pub record_bytes: u32,
    /// Total records (`n_blocks × k_blocks × 16`).
    pub records: u64,
    /// Total region bytes.
    pub region_bytes: u64,
}

impl ScaleGeometry {
    /// Byte offset of the record for row-block `nb`, K-block `kb`,
    /// intra-block row `row` (Spec 2 §3.1 grouping, ascending order).
    /// All indices are bounds-checked; arithmetic is checked.
    pub fn record_offset(&self, nb: u64, kb: u64, row: u32) -> Result<u64, FormatError> {
        let mut problems = Vec::new();
        if nb >= self.n_blocks {
            problems.push(FormatError::InvalidDim {
                name: "nb",
                value: nb,
                reason: "must be below the row-block count",
            });
        }
        if kb >= self.k_blocks {
            problems.push(FormatError::InvalidDim {
                name: "kb",
                value: kb,
                reason: "must be below the K-block count",
            });
        }
        if row as u64 >= crate::layout::TILE_N as u64 {
            problems.push(FormatError::InvalidDim {
                name: "row",
                value: row as u64,
                reason: "must be below 16",
            });
        }
        FormatError::collect(problems)?;
        // Internal invariant: indices are in range, so the product fits
        // whenever the region itself did; still checked, never wrapping.
        (nb.checked_mul(self.k_blocks)
            .and_then(|v| v.checked_add(kb))
            .and_then(|v| v.checked_mul(crate::layout::TILE_N as u64))
            .and_then(|v| v.checked_add(row as u64))
            .and_then(|v| v.checked_mul(self.record_bytes as u64)))
        .ok_or_else(|| FormatError::Overflow {
            what: "scale record_offset",
            detail: format!(
                "nb={nb} kb={kb} row={row} records={} record_bytes={}",
                self.records, self.record_bytes
            ),
        })
    }
}

/// SoA scale-region geometry for `scheme` over validated `dims`
/// (Spec 2 §3.1; card A2.2).
///
/// `L1` and `L1S` share the grouping; `L1S` callers pass the
/// compressed-K dims from [`crate::sparse::l1s_value_dims`] (Spec 2
/// §2.3). `L0` is rejected: its rows carry trailing scale records via
/// the `l0_*` helpers (Spec 2 §2.1), composed with
/// [`scale_record_bytes`]. Repack-only ids fail closed.
pub fn scale_geometry(
    scheme: SchemeId,
    layout: Layout,
    dims: &PaddedDims,
) -> Result<ScaleGeometry, FormatError> {
    match layout {
        Layout::L1 | Layout::L1S => {}
        Layout::L0 => {
            return Err(FormatError::UnsupportedLayout {
                scheme: scheme.name(),
                layout: layout.name(),
            });
        }
    }
    let record_bytes = scale_record_bytes(scheme)?;
    let block = match outer_block(scheme)? {
        Some(b) => b,
        // DECISION(A2.2): row-wise I8_R groups one K-block per row-block
        // (B = padded K, always divisible); rejected a fixed B because
        // §3.2 defines I8_R's block as the row itself.
        None => dims.k_padded(),
    };
    let (n_blocks, k_blocks) = crate::layout::scale_block_counts(dims, block)?;
    let records = crate::layout::scale_record_count(n_blocks, k_blocks)?;
    let region_bytes = crate::layout::scale_region_bytes(records, record_bytes)?;
    Ok(ScaleGeometry {
        scheme,
        n_blocks,
        k_blocks,
        record_bytes,
        records,
        region_bytes,
    })
}
