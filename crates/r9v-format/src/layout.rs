// SPDX-License-Identifier: Apache-2.0
//! Canonical tensor layouts (Spec 2 §2; card A2.1).
//!
//! This module owns the spec 2 §2 layout semantics: the closed [`Layout`]
//! set with stable codes, the [`Packing`] element classes from the §2.2
//! table, checked N/K/superblock padding, tile and row-block indexing in
//! the A0.S1-verified lane order, `L0` row geometry, and the §3.1
//! scale-record grouping shared by all `L1` schemes. Byte movement lives
//! in [`crate::permute`]; 2:4 sparsity lives in [`crate::sparse`].
//!
//! Stable codes equal the opaque [`r9v_ir::LayoutId`] codes so the two
//! types convert without a dependency cycle (`r9v-format` depends on
//! `r9v-ir` downward per spec 14 §2; `r9v-ir` never depends back).

use std::fmt;
use std::str::FromStr;

use crate::FormatError;

/// Rows per `L1` tile (Spec 2 §2.2: 16 rows × 16 columns).
pub const TILE_N: u32 = 16;
/// Columns per `L1` tile (Spec 2 §2.2).
pub const TILE_K: u32 = 16;
/// Lanes per tile (Spec 2 §2.2: 32 lanes with 8 elements each).
pub const LANES_PER_TILE: u32 = 32;
/// Elements per lane (Spec 2 §2.2).
pub const ELEMS_PER_LANE: u32 = 8;
/// Elements per tile (Spec 2 §2.2: 16 × 16).
pub const ELEMS_PER_TILE: u64 = 256;
/// K columns covered by one lane half (Spec 2 §2.2: `kgroup * 8 + j`).
pub const LANE_K: u32 = 8;

/// Canonical layout id (Spec 2 §2; card A2.1).
///
/// Closed enum: adding a layout is a spec change (Spec 2 §2.4, §9), and
/// every `match` stays exhaustive with no wildcard arm (CONVENTIONS.md
/// §3.2). Stable codes are part of the contract: they equal the opaque
/// [`r9v_ir::LayoutId`] codes, so `L0 == 1`, `L1 == 2`, `L1S == 3` on
/// disk, in metadata, and across crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major layout for lookup tables and vectors (Spec 2 §2.1).
    L0,
    /// Tiled layout for matmul weights; also the gfx12 native
    /// B-fragment order, which is what makes zero-copy load possible
    /// (Spec 2 §2.2; A0.S1 verified the lane formula on gfx1201).
    L1,
    /// Tiled 2:4 structured-sparse layout over compressed K plus an
    /// index region (Spec 2 §2.3).
    L1S,
}

impl Layout {
    /// Stable code for `L0` (Spec 2 §2; equals `r9v_ir::LayoutId::L0`).
    pub const CODE_L0: u64 = 1;
    /// Stable code for `L1` (Spec 2 §2; equals `r9v_ir::LayoutId::L1`).
    pub const CODE_L1: u64 = 2;
    /// Stable code for `L1S` (Spec 2 §2; equals `r9v_ir::LayoutId::L1S`).
    pub const CODE_L1S: u64 = 3;

    /// Returns the stable code (Spec 2 §2.4, §9: ids are immutable).
    pub const fn code(self) -> u64 {
        match self {
            Layout::L0 => Self::CODE_L0,
            Layout::L1 => Self::CODE_L1,
            Layout::L1S => Self::CODE_L1S,
        }
    }

    /// Decodes a stable code; unknown codes are errors, never guesses
    /// (Spec 2 §2.4: a new fragment order is `L2`, not a new meaning
    /// for an old code).
    pub fn from_code(code: u64) -> Result<Self, FormatError> {
        match code {
            Self::CODE_L0 => Ok(Layout::L0),
            Self::CODE_L1 => Ok(Layout::L1),
            Self::CODE_L1S => Ok(Layout::L1S),
            _ => Err(FormatError::UnknownLayout {
                value: code.to_string(),
            }),
        }
    }

    /// Returns the stable lowercase name (CONVENTIONS.md §3.2:
    /// serialization uses names, never discriminants).
    pub const fn name(self) -> &'static str {
        match self {
            Layout::L0 => "l0",
            Layout::L1 => "l1",
            Layout::L1S => "l1s",
        }
    }

    /// Parses a stable name; anything else is an error (Spec 2 §2).
    pub fn from_name(name: &str) -> Result<Self, FormatError> {
        match name {
            "l0" => Ok(Layout::L0),
            "l1" => Ok(Layout::L1),
            "l1s" => Ok(Layout::L1S),
            _ => Err(FormatError::UnknownLayout {
                value: name.to_owned(),
            }),
        }
    }

    /// Converts to the opaque IR handle (Spec 1 §2.3, Spec 2 §2.4).
    pub fn to_ir(self) -> r9v_ir::LayoutId {
        r9v_ir::LayoutId::new(self.code())
    }

    /// Converts from the opaque IR handle (Spec 1 §2.3, Spec 2 §2.4).
    /// IR codes outside `L0`/`L1`/`L1S` (`contiguous`, the gfx1201
    /// attention order, future `L2`) are errors here: this crate owns
    /// only the spec 2 §2 weight layouts.
    pub fn from_ir(id: r9v_ir::LayoutId) -> Result<Self, FormatError> {
        Self::from_code(id.as_u64())
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for Layout {
    type Err = FormatError;
    /// Parses a stable layout name (see [`Layout::from_name`]).
    fn from_str(s: &str) -> Result<Self, FormatError> {
        Self::from_name(s)
    }
}

/// Element packing class from the Spec 2 §2.2 table (card A2.1).
///
/// The lane/element assignment is identical for every class (Spec 2
/// §2.2: the formula has no dtype dependence); only the bytes per lane
/// differ. Closed enum: every `match` stays exhaustive with no wildcard
/// arm (CONVENTIONS.md §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Packing {
    /// `i4`: 2 elements per byte, low nibble = lower k
    /// (Spec 2 §2.2 table; one 32-bit load per lane).
    Nibble4,
    /// `i8`, `e4m3`, `e5m2`: 1 element per byte (Spec 2 §2.2 table;
    /// one 64-bit load per lane).
    Byte,
    /// `f16`, `bf16`: 2 bytes per element (Spec 2 §2.2 table; one
    /// 128-bit load per lane).
    Half16,
    /// Scheme-defined bit planes for 2/3/5/6-bit types (Spec 2 §2.2
    /// table, §3.3): planes follow the same lane order as the low
    /// nibbles, so a lane's eight values stay addressable together.
    BitPlanes {
        /// Bits per element: one of 2, 3, 5, 6.
        bits: u8,
    },
}

impl Packing {
    /// Builds the bit-plane packing, rejecting widths outside the
    /// scheme-defined set (Spec 2 §2.2 table, §3.3).
    pub fn bit_planes(bits: u8) -> Result<Self, FormatError> {
        match bits {
            2 | 3 | 5 | 6 => Ok(Packing::BitPlanes { bits }),
            _ => Err(FormatError::InvalidBitWidth { bits }),
        }
    }

    /// Bytes stored per lane per tile (Spec 2 §2.2 table).
    pub const fn bytes_per_lane(self) -> u64 {
        match self {
            Packing::Nibble4 => 4,
            Packing::Byte => 8,
            Packing::Half16 => 16,
            // DECISION(A2.1): plane bytes per lane are the minimal
            // whole-byte cover of 8 elements LSB-first
            // (ceil(8*bits/8): 2->2, 3->3, 5->5, 6->6 bytes), so a
            // lane stays within two loads per §2.2; rejected mirroring
            // any one GGUF scheme's plane split because plane byte
            // order is scheme detail owned by cards A2.3/A2.4, while
            // the lane-order law here must serve all of them.
            Packing::BitPlanes { bits } => (8_u64 * (bits as u64)).div_ceil(8),
        }
    }

    /// Bytes stored per tile (32 lanes).
    pub const fn tile_bytes(self) -> u64 {
        self.bytes_per_lane() * (LANES_PER_TILE as u64)
    }

    /// Returns the stable lowercase name (CONVENTIONS.md §3.2).
    pub const fn name(self) -> &'static str {
        match self {
            Packing::Nibble4 => "nibble4",
            Packing::Byte => "byte",
            Packing::Half16 => "half16",
            Packing::BitPlanes { .. } => "bitplanes",
        }
    }
}

impl fmt::Display for Packing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Packing::BitPlanes { bits } => write!(f, "bitplanes{}", bits),
            _ => write!(f, "{}", self.name()),
        }
    }
}

/// Validated, padded matmul dims (Spec 2 §2.2; card A2.1).
///
/// Carries the untrusted `(n, k)` extents plus their tile-aligned,
/// superblock-aligned padded form. Construction validates everything
/// and collects all problems (CONVENTIONS.md §1.4); afterward the
/// accessors can be trusted without re-validation (CONVENTIONS.md §2.2
/// boundary rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PaddedDims {
    n: u32,
    k: u32,
    n_padded: u32,
    k_padded: u32,
}

impl PaddedDims {
    /// Validates `(n, k)` and pads to `N % 16 == 0`, `K % 16 == 0`,
    /// with K additionally padded to `superblock_k` where one exists
    /// (Spec 2 §2.2). `superblock_k` itself must be a nonzero multiple
    /// of 16. Reports every problem, never just the first
    /// (CONVENTIONS.md §1.4); arithmetic is checked, never saturating.
    // DECISION(A2.1): zero extents are rejected (a weight with no rows
    // or no columns has no tile representation); rejected accepting
    // them as empty regions because downstream tile counts would be
    // zero and every consumer divides by them. Spec 2 §2 is silent.
    pub fn new(n: u32, k: u32, superblock_k: Option<u32>) -> Result<Self, FormatError> {
        let mut problems = Vec::new();
        if n == 0 {
            problems.push(FormatError::InvalidDim {
                name: "n",
                value: 0,
                reason: "must be at least 1",
            });
        }
        if k == 0 {
            problems.push(FormatError::InvalidDim {
                name: "k",
                value: 0,
                reason: "must be at least 1",
            });
        }
        let mut block: u32 = TILE_K;
        if let Some(s) = superblock_k {
            if s == 0 || !s.is_multiple_of(TILE_K) {
                problems.push(FormatError::InvalidBlock {
                    name: "superblock_k",
                    value: s as u64,
                    reason: "must be a nonzero multiple of 16",
                });
            } else {
                block = s;
            }
        }
        if !problems.is_empty() {
            return Err(collect_problems(problems));
        }
        let n_padded = round_up_u32(n, TILE_N, "padded_n")?;
        let k_padded = round_up_u32(k, block.max(TILE_K), "padded_k")?;
        Ok(Self {
            n,
            k,
            n_padded,
            k_padded,
        })
    }

    /// Untrusted (unpadded) output rows.
    pub const fn n(self) -> u32 {
        self.n
    }
    /// Untrusted (unpadded) reduction columns.
    pub const fn k(self) -> u32 {
        self.k
    }
    /// Padded output rows (`% 16 == 0`).
    pub const fn n_padded(self) -> u32 {
        self.n_padded
    }
    /// Padded reduction columns (`% 16 == 0`, superblock-aligned).
    pub const fn k_padded(self) -> u32 {
        self.k_padded
    }
    /// Tiles along N.
    pub const fn n_tiles(self) -> u64 {
        self.n_padded as u64 / TILE_N as u64
    }
    /// Tiles along K.
    pub const fn k_tiles(self) -> u64 {
        self.k_padded as u64 / TILE_K as u64
    }
    /// Total tiles in row-block-major, K-inner order.
    pub const fn tile_count(self) -> u64 {
        self.n_tiles() * self.k_tiles()
    }
    /// Row-blocks of 16 output rows (Spec 2 §2.2: one contiguous
    /// stream over all of K per row-block).
    pub const fn row_blocks(self) -> u64 {
        self.n_tiles()
    }
    /// Byte length of the value region for `packing` (Spec 2 §2.2,
    /// §7); checked, never saturating.
    pub fn value_region_bytes(self, packing: Packing) -> Result<u64, FormatError> {
        self.tile_count()
            .checked_mul(packing.tile_bytes())
            .ok_or_else(|| FormatError::Overflow {
                what: "value_region_bytes",
                detail: format!(
                    "tiles={} tile_bytes={}",
                    self.tile_count(),
                    packing.tile_bytes()
                ),
            })
    }
}

/// Rounds `value` up to a multiple of `align` in checked arithmetic
/// (Spec 2 §2.2 padding; never saturates on untrusted input).
fn round_up_u32(value: u32, align: u32, what: &'static str) -> Result<u32, FormatError> {
    let widened = value as u64;
    let steps = widened
        .checked_add(align as u64 - 1)
        .ok_or_else(|| FormatError::Overflow {
            what,
            detail: format!("value={value} align={align}"),
        })?;
    let rounded = steps / align as u64 * align as u64;
    u32::try_from(rounded).map_err(|_| FormatError::Overflow {
        what,
        detail: format!("value={value} align={align}"),
    })
}

/// Collapses a validated-nonempty problem list (CONVENTIONS.md §1.4).
fn collect_problems(problems: Vec<FormatError>) -> FormatError {
    debug_assert!(!problems.is_empty());
    if problems.len() == 1 {
        let mut problems = problems;
        // Internal invariant: this branch runs only when len == 1.
        problems.pop().expect("problems holds exactly one entry")
    } else {
        FormatError::Multiple {
            problems: problems.into_boxed_slice(),
        }
    }
}

/// Lane for output row `n_in_tile` and K-group `kgroup` (Spec 2 §2.2:
/// `lane = kgroup * 16 + n`; A0.S1 verified this order on gfx1201).
pub fn l1_lane(n_in_tile: u32, kgroup: u32) -> Result<u32, FormatError> {
    if n_in_tile >= TILE_N {
        return Err(FormatError::InvalidDim {
            name: "n_in_tile",
            value: n_in_tile as u64,
            reason: "must be below 16",
        });
    }
    if kgroup >= 2 {
        return Err(FormatError::InvalidDim {
            name: "kgroup",
            value: kgroup as u64,
            reason: "must be 0 or 1",
        });
    }
    Ok(kgroup * TILE_N + n_in_tile)
}

/// Element slot for `lane` and intra-lane position `j` (Spec 2 §2.2:
/// `elem = lane * 8 + j`).
pub fn l1_elem(lane: u32, j: u32) -> Result<u32, FormatError> {
    if lane >= LANES_PER_TILE {
        return Err(FormatError::InvalidDim {
            name: "lane",
            value: lane as u64,
            reason: "must be below 32",
        });
    }
    if j >= ELEMS_PER_LANE {
        return Err(FormatError::InvalidDim {
            name: "j",
            value: j as u64,
            reason: "must be below 8",
        });
    }
    Ok(lane * ELEMS_PER_LANE + j)
}

/// Tile index in row-block-major, K-inner order (Spec 2 §2.2:
/// `tile_index = (n_base / 16) * (K / 16) + (k_base / 16)`).
/// Both bases must be tile-aligned; arithmetic is checked.
pub fn tile_index(n_base: u32, k_base: u32, dims: &PaddedDims) -> Result<u64, FormatError> {
    let mut problems = Vec::new();
    if !n_base.is_multiple_of(TILE_N) {
        problems.push(FormatError::InvalidDim {
            name: "n_base",
            value: n_base as u64,
            reason: "must be a multiple of 16",
        });
    }
    if !k_base.is_multiple_of(TILE_K) {
        problems.push(FormatError::InvalidDim {
            name: "k_base",
            value: k_base as u64,
            reason: "must be a multiple of 16",
        });
    }
    if n_base >= dims.n_padded() {
        problems.push(FormatError::InvalidDim {
            name: "n_base",
            value: n_base as u64,
            reason: "must be below padded N",
        });
    }
    if k_base >= dims.k_padded() {
        problems.push(FormatError::InvalidDim {
            name: "k_base",
            value: k_base as u64,
            reason: "must be below padded K",
        });
    }
    if !problems.is_empty() {
        return Err(collect_problems(problems));
    }
    (n_base as u64 / TILE_N as u64)
        .checked_mul(dims.k_tiles())
        .and_then(|row| row.checked_add(k_base as u64 / TILE_K as u64))
        .ok_or_else(|| FormatError::Overflow {
            what: "tile_index",
            detail: format!("n_base={n_base} k_base={k_base}"),
        })
}

/// Tile origin `(n_base, k_base)` for a tile index (inverse of
/// [`tile_index`]; Spec 2 §2.2).
pub fn tile_origin(tile: u64, dims: &PaddedDims) -> Result<(u32, u32), FormatError> {
    if tile >= dims.tile_count() {
        return Err(FormatError::InvalidDim {
            name: "tile",
            value: tile,
            reason: "must be below the tile count",
        });
    }
    let k_tiles = dims.k_tiles();
    // Internal invariant: k_tiles >= 1 because padded K >= 16.
    debug_assert!(k_tiles >= 1);
    let n_base = (tile / k_tiles * TILE_N as u64) as u32;
    let k_base = (tile % k_tiles * TILE_K as u64) as u32;
    Ok((n_base, k_base))
}

/// Tile-order element position for padded `(n, k)` (Spec 2 §2.2 lane
/// order flattened: `tile * 256 + lane * 8 + j`).
pub fn l1_forward_index(n: u32, k: u32, dims: &PaddedDims) -> Result<u64, FormatError> {
    let mut problems = Vec::new();
    if n >= dims.n_padded() {
        problems.push(FormatError::InvalidDim {
            name: "n",
            value: n as u64,
            reason: "must be below padded N",
        });
    }
    if k >= dims.k_padded() {
        problems.push(FormatError::InvalidDim {
            name: "k",
            value: k as u64,
            reason: "must be below padded K",
        });
    }
    if !problems.is_empty() {
        return Err(collect_problems(problems));
    }
    let tile = (n as u64 / TILE_N as u64) * dims.k_tiles() + (k as u64 / TILE_K as u64);
    let lane =
        (k as u64 % TILE_K as u64 / LANE_K as u64) * TILE_N as u64 + (n as u64 % TILE_N as u64);
    Ok(tile * ELEMS_PER_TILE + lane * ELEMS_PER_LANE as u64 + (k as u64 % LANE_K as u64))
}

/// Padded `(n, k)` for a tile-order element position (inverse of
/// [`l1_forward_index`]; Spec 2 §2.2).
pub fn l1_inverse_index(pos: u64, dims: &PaddedDims) -> Result<(u32, u32), FormatError> {
    let total = dims.tile_count() * ELEMS_PER_TILE;
    if pos >= total {
        return Err(FormatError::InvalidDim {
            name: "pos",
            value: pos,
            reason: "must be below the tiled element count",
        });
    }
    let tile = pos / ELEMS_PER_TILE;
    let lane = (pos % ELEMS_PER_TILE) / ELEMS_PER_LANE as u64;
    let j = (pos % ELEMS_PER_TILE) % ELEMS_PER_LANE as u64;
    let k_tiles = dims.k_tiles();
    // Internal invariant: k_tiles >= 1 because padded K >= 16.
    debug_assert!(k_tiles >= 1);
    let n = (tile / k_tiles * TILE_N as u64 + lane % TILE_N as u64) as u32;
    let k = (tile % k_tiles * TILE_K as u64 + lane / TILE_N as u64 * LANE_K as u64 + j) as u32;
    Ok((n, k))
}

/// Contiguous tile span of row-block `nb` (Spec 2 §2.2: a row-block of
/// 16 output rows is one contiguous stream over all of K).
pub fn row_block_tiles(nb: u32, dims: &PaddedDims) -> Result<(u64, u64), FormatError> {
    if nb as u64 >= dims.row_blocks() {
        return Err(FormatError::InvalidDim {
            name: "nb",
            value: nb as u64,
            reason: "must be below the row-block count",
        });
    }
    let start = nb as u64 * dims.k_tiles();
    Ok((start, start + dims.k_tiles()))
}

/// Values bytes of one `L0` row before scales (Spec 2 §2.1:
/// contiguous rows); checked, never saturating.
pub fn l0_row_values_bytes(dim_elems: u32, elem_bytes: u32) -> Result<u64, FormatError> {
    let mut problems = Vec::new();
    if dim_elems == 0 {
        problems.push(FormatError::InvalidDim {
            name: "dim_elems",
            value: 0,
            reason: "must be at least 1",
        });
    }
    if elem_bytes == 0 {
        problems.push(FormatError::InvalidDim {
            name: "elem_bytes",
            value: 0,
            reason: "must be at least 1",
        });
    }
    if !problems.is_empty() {
        return Err(collect_problems(problems));
    }
    (dim_elems as u64)
        .checked_mul(elem_bytes as u64)
        .ok_or_else(|| FormatError::Overflow {
            what: "l0_row_values_bytes",
            detail: format!("dim_elems={dim_elems} elem_bytes={elem_bytes}"),
        })
}

/// Stride of one `L0` row: values plus per-(row, K-block) scale records
/// stored immediately after the row's values (Spec 2 §2.1: one row
/// plus its scales is one contiguous region, which is what makes
/// row-granular residency work).
// DECISION(A2.1): every L0 row carries at least one scale record slot
// (`k_blocks >= 1`, `record_bytes >= 1`); rejected a scaleless L0 form
// because §2.1 stores scale records per (row, K-block) unconditionally
// and record contents are scheme detail owned by card A2.2.
pub fn l0_row_stride_bytes(
    dim_elems: u32,
    elem_bytes: u32,
    k_blocks: u32,
    record_bytes: u32,
) -> Result<u64, FormatError> {
    let mut problems = Vec::new();
    let values = match l0_row_values_bytes(dim_elems, elem_bytes) {
        Ok(values) => Some(values),
        Err(FormatError::Multiple { problems: inner }) => {
            problems.extend(inner.into_vec());
            None
        }
        Err(single) => {
            problems.push(single);
            None
        }
    };
    if k_blocks == 0 {
        problems.push(FormatError::InvalidBlock {
            name: "k_blocks",
            value: 0,
            reason: "must be at least 1",
        });
    }
    if record_bytes == 0 {
        problems.push(FormatError::InvalidBlock {
            name: "record_bytes",
            value: 0,
            reason: "must be at least 1",
        });
    }
    if !problems.is_empty() {
        return Err(collect_problems(problems));
    }
    // Internal invariant: values is Some because problems was empty.
    let values = values.expect("values valid when problems is empty");
    (k_blocks as u64)
        .checked_mul(record_bytes as u64)
        .and_then(|scales| scales.checked_add(values))
        .ok_or_else(|| FormatError::Overflow {
            what: "l0_row_stride_bytes",
            detail: format!("k_blocks={k_blocks} record_bytes={record_bytes}"),
        })
}

/// Byte offset of `L0` row `row` (Spec 2 §2.1); checked.
pub fn l0_row_offset_bytes(row: u32, stride_bytes: u64) -> Result<u64, FormatError> {
    (row as u64)
        .checked_mul(stride_bytes)
        .ok_or_else(|| FormatError::Overflow {
            what: "l0_row_offset_bytes",
            detail: format!("row={row} stride={stride_bytes}"),
        })
}

/// Byte length of an `L0` region of `rows` rows (Spec 2 §2.1); checked.
pub fn l0_region_bytes(rows: u32, stride_bytes: u64) -> Result<u64, FormatError> {
    if rows == 0 {
        return Err(FormatError::InvalidDim {
            name: "rows",
            value: 0,
            reason: "must be at least 1",
        });
    }
    (rows as u64)
        .checked_mul(stride_bytes)
        .ok_or_else(|| FormatError::Overflow {
            what: "l0_region_bytes",
            detail: format!("rows={rows} stride={stride_bytes}"),
        })
}

/// Scale-record grouping for one `L1` tensor (Spec 2 §3.1:
/// `[N/16][K/B][16 records]`, so the wave handling row-block `nb` and
/// K-block `kb` loads `16 × record_size` contiguous bytes).
/// Returns `(n_blocks, k_blocks)`; `block_b` must divide padded K.
pub fn scale_block_counts(dims: &PaddedDims, block_b: u32) -> Result<(u64, u64), FormatError> {
    if block_b == 0 || !block_b.is_multiple_of(TILE_K) {
        return Err(FormatError::InvalidBlock {
            name: "block_b",
            value: block_b as u64,
            reason: "must be a nonzero multiple of 16",
        });
    }
    if !dims.k_padded().is_multiple_of(block_b) {
        return Err(FormatError::InvalidBlock {
            name: "block_b",
            value: block_b as u64,
            reason: "must divide padded K",
        });
    }
    Ok((dims.n_tiles(), dims.k_padded() as u64 / block_b as u64))
}

/// Scale-record count for the §3.1 grouping (16 records per block).
pub fn scale_record_count(n_blocks: u64, k_blocks: u64) -> Result<u64, FormatError> {
    n_blocks
        .checked_mul(k_blocks)
        .and_then(|blocks| blocks.checked_mul(TILE_N as u64))
        .ok_or_else(|| FormatError::Overflow {
            what: "scale_record_count",
            detail: format!("n_blocks={n_blocks} k_blocks={k_blocks}"),
        })
}

/// Byte length of the scale region (Spec 2 §3.1, §7); checked.
pub fn scale_region_bytes(record_count: u64, record_bytes: u32) -> Result<u64, FormatError> {
    if record_bytes == 0 {
        return Err(FormatError::InvalidBlock {
            name: "record_bytes",
            value: 0,
            reason: "must be at least 1",
        });
    }
    record_count
        .checked_mul(record_bytes as u64)
        .ok_or_else(|| FormatError::Overflow {
            what: "scale_region_bytes",
            detail: format!("records={record_count} record_bytes={record_bytes}"),
        })
}
