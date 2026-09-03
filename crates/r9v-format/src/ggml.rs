// SPDX-License-Identifier: Apache-2.0
//! GGUF source types and source-side reference dequantization
//! (Spec 2 §3.3, §7 steps 1–2, §10; card A2.3).
//!
//! [`GgmlType`] is the closed set of GGUF source types this card owns:
//! the ten quantized `Q*_0/1/K` types plus unquantized `F16`/`BF16`.
//! Numeric codes and names match `GGMLQuantizationType` in gguf-py
//! 0.19.0 (`constants.py`); wire-block layouts and dequant formulas
//! match its `quants.py` `dequantize_blocks` bit-exact, verified by the
//! `gguf_a23_reference.txt` fixtures. [`ggml_dequantize`] decodes GGUF
//! wire bytes in row-major block order to `f32`; the repacked-side
//! decode lives in [`mod@crate::repack`] and reads `L1`+SoA instead, so the
//! §10 round-trip compares two independent byte paths.
//!
//! `F16`/`BF16` map to no [`SchemeId`]: they are unquantized dtypes
//! (`QuantScheme::None` per Spec 1 §2.2; SI-26), repacked by pure `L1`
//! permutation of their half bits.

use std::fmt;
use std::str::FromStr;

use crate::scales::f16_to_f32;
use crate::scheme::SchemeId;
use crate::FormatError;

/// GGUF source tensor type owned by card A2.3 (Spec 2 §3.3, §7).
///
/// Closed enum: every `match` stays exhaustive with no wildcard arm
/// (CONVENTIONS.md §3.2). Numeric codes are the upstream GGUF
/// `ggml_type` ids, not [`SchemeId`] codes: the mapping between them
/// is [`GgmlType::scheme`]. Variant spellings keep the GGUF form
/// (`Q4_0`, not `Q40`) so names match the spec and the file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    /// 16-bit float, 2 bytes per element, no blocking (GGUF id 1).
    F16,
    /// 32 values in 18 bytes: `d: f16`, split nibbles (GGUF id 2).
    Q4_0,
    /// 32 values in 20 bytes: `d: f16, m: f16`, split nibbles (GGUF id 3).
    Q4_1,
    /// 32 values in 22 bytes: `d: f16`, 4 high-bit bytes, split
    /// nibbles (GGUF id 6).
    Q5_0,
    /// 32 values in 24 bytes: `d: f16, m: f16`, high bits, split
    /// nibbles (GGUF id 7).
    Q5_1,
    /// 32 values in 34 bytes: `d: f16`, 32 `i8` values (GGUF id 8).
    Q8_0,
    /// 256 values in 84 bytes: 16 scale bytes, 64 value bytes,
    /// `d: f16, dmin: f16` (GGUF id 10).
    Q2_K,
    /// 256 values in 110 bytes: 32 high-bit bytes, 64 value bytes,
    /// 12 scale bytes, `d: f16` (GGUF id 11).
    Q3_K,
    /// 256 values in 144 bytes: `d: f16, dmin: f16`, 12 scale bytes,
    /// 128 value bytes (GGUF id 12).
    Q4_K,
    /// 256 values in 176 bytes: `d: f16, dmin: f16`, 12 scale bytes,
    /// 32 high-bit bytes, 128 value bytes (GGUF id 13).
    Q5_K,
    /// 256 values in 210 bytes: 128 low-nibble bytes, 64 2-bit bytes,
    /// 16 `i8` scales, `d: f16` (GGUF id 14).
    Q6_K,
    /// Brain float, 2 bytes per element, no blocking (GGUF id 30).
    BF16,
}

// DECISION(A2.3): numeric codes pin gguf-py 0.19.0 GGMLQuantizationType
// ids (F16=1 .. BF16=30); rejected reusing SchemeId codes because the
// §7 step-1 mapping is from upstream ggml_type space into SchemeId
// space and the two tables have different owners and widths.
impl GgmlType {
    /// All source types in ggml-code order.
    pub const ALL: [GgmlType; 12] = [
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::BF16,
    ];

    /// Returns the upstream `ggml_type` code (Spec 2 §7 step 1).
    pub const fn code(self) -> u32 {
        match self {
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q2_K => 10,
            GgmlType::Q3_K => 11,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            GgmlType::BF16 => 30,
        }
    }

    /// Decodes an upstream code; anything else is a hard error naming
    /// the type (Spec 2 §7 step 1), never a guess.
    pub fn from_code(code: u32) -> Result<Self, FormatError> {
        match code {
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            10 => Ok(GgmlType::Q2_K),
            11 => Ok(GgmlType::Q3_K),
            12 => Ok(GgmlType::Q4_K),
            13 => Ok(GgmlType::Q5_K),
            14 => Ok(GgmlType::Q6_K),
            30 => Ok(GgmlType::BF16),
            _ => Err(FormatError::UnknownGgmlType { code }),
        }
    }

    /// Returns the stable uppercase name matching the GGUF spelling.
    pub const fn name(self) -> &'static str {
        match self {
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q2_K => "Q2_K",
            GgmlType::Q3_K => "Q3_K",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
            GgmlType::BF16 => "BF16",
        }
    }

    /// Parses the GGUF spelling; anything else is an error, never a
    /// guess (Spec 2 §7 step 1).
    pub fn from_name(name: &str) -> Result<Self, FormatError> {
        match name {
            "F16" => Ok(GgmlType::F16),
            "Q4_0" => Ok(GgmlType::Q4_0),
            "Q4_1" => Ok(GgmlType::Q4_1),
            "Q5_0" => Ok(GgmlType::Q5_0),
            "Q5_1" => Ok(GgmlType::Q5_1),
            "Q8_0" => Ok(GgmlType::Q8_0),
            "Q2_K" => Ok(GgmlType::Q2_K),
            "Q3_K" => Ok(GgmlType::Q3_K),
            "Q4_K" => Ok(GgmlType::Q4_K),
            "Q5_K" => Ok(GgmlType::Q5_K),
            "Q6_K" => Ok(GgmlType::Q6_K),
            "BF16" => Ok(GgmlType::BF16),
            _ => Err(FormatError::UnknownScheme {
                value: name.to_owned(),
            }),
        }
    }

    /// Maps `ggml_type` to repack scheme (Spec 2 §7 step 1, §3.3).
    ///
    /// `Q4_K` maps to the native [`SchemeId::I4K`], which is
    /// field-identical to `Q4_K` by design (Spec 2 §3.2,
    /// DECISIONS.md D-004): repacking it only permutes nibbles into
    /// `L1` and regroups headers into SoA. `F16`/`BF16` map to `None`:
    /// they keep their dtype with `QuantScheme::None` (SI-26).
    pub const fn scheme(self) -> Option<SchemeId> {
        match self {
            GgmlType::Q8_0 => Some(SchemeId::I8B32F),
            GgmlType::Q4_0 => Some(SchemeId::I4B32F),
            GgmlType::Q4_1 => Some(SchemeId::I4B32FM),
            GgmlType::Q5_0 => Some(SchemeId::I5B32F),
            GgmlType::Q5_1 => Some(SchemeId::I5B32FM),
            GgmlType::Q4_K => Some(SchemeId::I4K),
            GgmlType::Q5_K => Some(SchemeId::I5K),
            GgmlType::Q6_K => Some(SchemeId::I6K),
            GgmlType::Q3_K => Some(SchemeId::I3K),
            GgmlType::Q2_K => Some(SchemeId::I2K),
            GgmlType::F16 | GgmlType::BF16 => None,
        }
    }

    /// Whether this type carries quantization blocks (`false` for the
    /// unquantized halves).
    pub const fn is_quantized(self) -> bool {
        match self {
            GgmlType::F16 | GgmlType::BF16 => false,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K => true,
        }
    }

    /// Values per wire block: 32 for `_0`/`_1` types, 256 (`QK_K`)
    /// for K types, 1 for unblocked halves (gguf-py `GGML_QUANT_SIZES`).
    pub const fn block_len(self) -> u32 {
        match self {
            GgmlType::F16 | GgmlType::BF16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => {
                32
            }
            GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
                256
            }
        }
    }

    /// Wire bytes per block (gguf-py `GGML_QUANT_SIZES` type sizes).
    pub const fn block_bytes(self) -> u64 {
        match self {
            GgmlType::F16 | GgmlType::BF16 => 2,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q2_K => 84,
            GgmlType::Q3_K => 110,
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
        }
    }

    /// `L1` superblock for [`crate::layout::PaddedDims`]: the wire
    /// block for quantized types, `None` (plain 16-pad) for halves
    /// (Spec 2 §2.2, §7 step 2).
    pub const fn superblock_k(self) -> Option<u32> {
        match self {
            GgmlType::F16 | GgmlType::BF16 => None,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => {
                Some(32)
            }
            GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
                Some(256)
            }
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for GgmlType {
    type Err = FormatError;
    /// Parses the GGUF spelling (see [`GgmlType::from_name`]).
    fn from_str(s: &str) -> Result<Self, FormatError> {
        Self::from_name(s)
    }
}

/// Exact `bf16` bits to `f32` (Spec 2 §3.3 halves): the widening shift
/// is exact for every pattern, including NaN payloads. Total function.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Validates wire `f16` scale bits to `f32` (Spec 2 §7 steps 2/4).
///
/// Only non-finite patterns are rejected: real `Q*_0/1/K` writers
/// (gguf-py `quantize`, llama.cpp) emit negative `d` whenever the
/// block extremum is positive, and the sign is absorbed by the
/// zero-point form, so negativity is wire reality, not corruption.
/// This differs deliberately from the native-scale rule in
/// [`crate::scales::check_f16_scale`] (non-negative multipliers).
// DECISION(A2.3): accept any finite wire f16 scale, including
// negative d; rejected mirroring the native non-negative rule because
// real writers emit negative scales and llama.cpp applies them as-is.
// Spec 2 §3.3 is silent on wire scale validity.
pub(crate) fn check_wire_f16(
    scheme: &'static str,
    record: u64,
    bits: u16,
) -> Result<f32, FormatError> {
    let value = f16_to_f32(bits);
    if value.is_nan() {
        return Err(FormatError::InvalidScale {
            scheme,
            record,
            bits: value.to_bits(),
            reason: "nan",
        });
    }
    if value.is_infinite() {
        return Err(FormatError::InvalidScale {
            scheme,
            record,
            bits: value.to_bits(),
            reason: "infinite",
        });
    }
    Ok(value)
}

/// Validated wire geometry for one tensor's GGUF bytes (Spec 2 §7
/// steps 1–2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct WireGeometry {
    /// Source type being decoded.
    pub(crate) ggml: GgmlType,
    /// Tensor rows.
    pub(crate) n_rows: u32,
    /// Tensor columns (multiple of the wire block).
    pub(crate) k: u32,
    /// Wire blocks per row.
    pub(crate) blocks_per_row: u64,
    /// Total wire bytes expected.
    pub(crate) total_bytes: u64,
}

/// Validates `(n_rows, k)` against the wire block and the byte length,
/// collecting every problem before returning (CONVENTIONS.md §1.4).
/// `K` must be a nonzero multiple of the block (Spec 2 §7 step 2);
/// halves additionally require `2*n*k` to fit the length exactly.
pub(crate) fn wire_geometry(
    ggml: GgmlType,
    wire: &[u8],
    n_rows: u32,
    k: u32,
) -> Result<WireGeometry, FormatError> {
    let mut problems = Vec::new();
    if n_rows == 0 {
        problems.push(FormatError::InvalidDim {
            name: "n_rows",
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
    let block = ggml.block_len();
    if k != 0 && !k.is_multiple_of(block) {
        problems.push(FormatError::InvalidBlock {
            name: "k",
            value: k as u64,
            reason: "must be a multiple of the ggml block length",
        });
    }
    if !problems.is_empty() {
        return Err(single_or_multiple(problems));
    }
    let blocks_per_row = k as u64 / block as u64;
    let expected = (n_rows as u64)
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(ggml.block_bytes()));
    let expected = match expected {
        Some(v) => v,
        None => {
            return Err(FormatError::Overflow {
                what: "ggml wire bytes",
                detail: format!(
                    "n_rows={n_rows} k={k} block={} block_bytes={}",
                    ggml.name(),
                    ggml.block_bytes()
                ),
            });
        }
    };
    if wire.len() as u64 != expected {
        problems.push(FormatError::LengthMismatch {
            what: "ggml wire bytes",
            expected,
            got: wire.len() as u64,
        });
    }
    if !problems.is_empty() {
        return Err(single_or_multiple(problems));
    }
    Ok(WireGeometry {
        ggml,
        n_rows,
        k,
        blocks_per_row,
        total_bytes: expected,
    })
}

/// Collapses a validated-nonempty problem list (CONVENTIONS.md §1.4).
fn single_or_multiple(problems: Vec<FormatError>) -> FormatError {
    if problems.len() == 1 {
        let mut problems = problems;
        match problems.pop() {
            Some(single) => single,
            // Internal invariant: len == 1 so pop always succeeds;
            // unreachable without panic machinery, so fall back to the
            // (empty, unreachable-in-practice) Multiple report.
            None => FormatError::Multiple {
                problems: Vec::new().into_boxed_slice(),
            },
        }
    } else {
        FormatError::Multiple {
            problems: problems.into_boxed_slice(),
        }
    }
}

/// Reads one little-endian `u16` at `offset` (bounds-checked).
fn u16_le(block: &[u8], offset: usize, what: &'static str) -> Result<u16, FormatError> {
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
fn u32_le(block: &[u8], offset: usize, what: &'static str) -> Result<u32, FormatError> {
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

/// Requires `block` to hold one full wire block of `ggml`.
fn expect_block(block: &[u8], ggml: GgmlType) -> Result<(), FormatError> {
    if block.len() as u64 != ggml.block_bytes() {
        return Err(FormatError::LengthMismatch {
            what: "ggml wire block",
            expected: ggml.block_bytes(),
            got: block.len() as u64,
        });
    }
    Ok(())
}

/// Parses one wire block into logical values in ascending order.
///
/// Output encoding per type (what [`crate::repack`] stores in `L1`):
/// raw `i8` bits for `Q8_0`, raw halves for `F16`/`BF16`, unsigned
/// magnitudes otherwise (`u4` for `Q4_*`, `u5` for `Q5_*` including
/// the high bit, `u6`/`u2` raw plane magnitudes for `Q6_K`/`Q2_K`,
/// and `u3` holding low 2 bits plus the decoded high bit for `Q3_K`
/// — the wire inversion is applied once here, at parse).
/// Zero-points and signs belong to the dequant formula, not to the
/// stored values (Spec 2 §7 step 4: pure permutation plus bit-plane
/// regrouping, no arithmetic).
pub(crate) fn block_values(
    ggml: GgmlType,
    block: &[u8],
    out: &mut [u16],
) -> Result<(), FormatError> {
    expect_block(block, ggml)?;
    let want = ggml.block_len() as usize;
    if out.len() != want {
        return Err(FormatError::LengthMismatch {
            what: "ggml logical values",
            expected: want as u64,
            got: out.len() as u64,
        });
    }
    match ggml {
        GgmlType::F16 | GgmlType::BF16 => {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = u16_le(block, 2 * i, "ggml half block")?;
            }
        }
        GgmlType::Q8_0 => {
            for (i, slot) in out.iter_mut().enumerate() {
                match block.get(2 + i) {
                    Some(b) => *slot = *b as u16,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "q8_0 values",
                            expected: ggml.block_bytes(),
                            got: block.len() as u64,
                        });
                    }
                }
            }
        }
        GgmlType::Q4_0 | GgmlType::Q4_1 => {
            // Split nibbles: byte j holds values j (low) and j+16
            // (high), matching gguf-py Q4_0/Q4_1 dequantize_blocks.
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
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q4_0/1 block",
                        got: ggml.name(),
                    });
                }
            };
            for j in 0..16 {
                let byte = match block.get(base + j) {
                    Some(b) => *b,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "q4_0/1 values",
                            expected: ggml.block_bytes(),
                            got: block.len() as u64,
                        });
                    }
                };
                out[j] = (byte & 0x0F) as u16;
                out[j + 16] = (byte >> 4) as u16;
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            // Split nibbles plus a little-endian 32-bit high-bit
            // plane, bit i serving value i (gguf-py Q5_0/Q5_1).
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
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q5_0/1 block",
                        got: ggml.name(),
                    });
                }
            };
            let qh = u32_le(block, qh_off, "q5_0/1 high bits")?;
            for j in 0..16 {
                let byte = match block.get(qs_off + j) {
                    Some(b) => *b,
                    None => {
                        return Err(FormatError::LengthMismatch {
                            what: "q5_0/1 values",
                            expected: ggml.block_bytes(),
                            got: block.len() as u64,
                        });
                    }
                };
                let lo = (byte & 0x0F) as u16;
                let hi = (byte >> 4) as u16;
                out[j] = lo | (((qh >> j) & 1) as u16) << 4;
                out[j + 16] = hi | (((qh >> (j + 16)) & 1) as u16) << 4;
            }
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            // Group-paired nibbles: byte (j/2)*32+p of the 128-byte
            // payload holds sub-block 2g low and 2g+1 high at the same
            // position p (gguf-py Q4_K/Q5_K reshape law, verified
            // bit-exact in the A2.3 pre-check). Q5_K carries its
            // 32 high-bit bytes before the nibbles.
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
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q4_k/q5_k block",
                        got: ggml.name(),
                    });
                }
            };
            for j in 0..8 {
                for p in 0..32 {
                    let byte = match block.get(qs_off + (j / 2) * 32 + p) {
                        Some(b) => *b,
                        None => {
                            return Err(FormatError::LengthMismatch {
                                what: "q4_k/q5_k values",
                                expected: ggml.block_bytes(),
                                got: block.len() as u64,
                            });
                        }
                    };
                    out[j * 32 + p] = ((byte >> ((j % 2) * 4)) & 0x0F) as u16;
                }
            }
            if ggml == GgmlType::Q5_K {
                // Transposed high-bit plane: byte p serves position p
                // of every sub-block (gguf-py Q5_K reshape law).
                for j in 0..8 {
                    for p in 0..32 {
                        let byte = match block.get(16 + p) {
                            Some(b) => *b,
                            None => {
                                return Err(FormatError::LengthMismatch {
                                    what: "q5_k high bits",
                                    expected: ggml.block_bytes(),
                                    got: block.len() as u64,
                                });
                            }
                        };
                        out[j * 32 + p] |= (((byte >> j) & 1) as u16) << 4;
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            // 128 low-nibble bytes shared across sub-block pairs plus
            // 64 2-bit high bytes (gguf-py Q6_K reshape law, verified
            // bit-exact in the A2.3 pre-check).
            for j in 0..16 {
                let big = j / 2;
                let pos0 = (j % 2) * 16;
                for p in 0..16 {
                    let big_p = pos0 + p;
                    let flat = big * 32 + big_p;
                    let g = flat / 128;
                    let s = (flat % 128) / 64;
                    let b = flat % 64;
                    let lo = match block.get(g * 64 + b) {
                        Some(v) => ((v >> (s * 4)) & 0x0F) as u16,
                        None => {
                            return Err(FormatError::LengthMismatch {
                                what: "q6_k low bits",
                                expected: ggml.block_bytes(),
                                got: block.len() as u64,
                            });
                        }
                    };
                    let group = big / 4;
                    let pair = big % 4;
                    let hi = match block.get(128 + group * 32 + big_p) {
                        Some(v) => ((v >> (pair * 2)) & 0x03) as u16,
                        None => {
                            return Err(FormatError::LengthMismatch {
                                what: "q6_k high bits",
                                expected: ggml.block_bytes(),
                                got: block.len() as u64,
                            });
                        }
                    };
                    out[j * 16 + p] = lo | (hi << 4);
                }
            }
        }
        GgmlType::Q3_K | GgmlType::Q2_K => {
            // 64 two-bit bytes: byte (j/8)*32 + ((j%8)%2)*16 + p of
            // the payload serves sub-block j position p at shift
            // ((j%8)/2)*2 (gguf-py Q3_K/Q2_K reshape law).
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
                | GgmlType::BF16 => {
                    return Err(FormatError::SchemeMismatch {
                        scheme: ggml.name(),
                        expected: "q3_k/q2_k block",
                        got: ggml.name(),
                    });
                }
            };
            for j in 0..16 {
                let group = j / 8;
                let half = (j % 8) % 2;
                let shift = ((j % 8) / 2) * 2;
                for p in 0..16 {
                    let byte = match block.get(qs_off + group * 32 + half * 16 + p) {
                        Some(b) => *b,
                        None => {
                            return Err(FormatError::LengthMismatch {
                                what: "q3_k/q2_k values",
                                expected: ggml.block_bytes(),
                                got: block.len() as u64,
                            });
                        }
                    };
                    out[j * 16 + p] = ((byte >> shift) & 0x03) as u16;
                }
            }
            if ggml == GgmlType::Q3_K {
                // Inverted high-bit plane: stored bit 1 means zero
                // offset, stored 0 means −4 (llama.cpp convention
                // mirrored by gguf-py: `qh ^ 1`).
                // DECISION(A2.3): the stored u3 holds the decoded high
                // bit (wire inversion applied once here at parse), so
                // both decodes use ql − (qh<<2) with no second
                // inversion; rejected storing raw wire bits because a
                // reader would then need to know the inversion rule.
                for j in 0..16 {
                    for p in 0..16 {
                        let byte = match block.get((j % 2) * 16 + p) {
                            Some(b) => *b,
                            None => {
                                return Err(FormatError::LengthMismatch {
                                    what: "q3_k high bits",
                                    expected: ggml.block_bytes(),
                                    got: block.len() as u64,
                                });
                            }
                        };
                        out[j * 16 + p] |= ((((byte >> (j / 2)) & 1) ^ 1) as u16) << 2;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Unpacks the 12-byte `Q4_K`/`Q5_K` scale payload into eight 6-bit
/// sub-block scales and minimums (gguf-py `Q4_K.get_scale_min`,
/// llama.cpp `get_scale_min_k4`; byte-identical to
/// [`crate::records::I4KSuperblock`] field order).
pub fn unpack_k4_scales(payload: &[u8]) -> Result<([u8; 8], [u8; 8]), FormatError> {
    if payload.len() != 12 {
        return Err(FormatError::LengthMismatch {
            what: "q4_k/q5_k scale payload",
            expected: 12,
            got: payload.len() as u64,
        });
    }
    let byte = |i: usize| payload[i];
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4 {
        sc[j] = byte(j) & 63;
        mn[j] = byte(j + 4) & 63;
    }
    for j in 4..8 {
        sc[j] = (byte(j + 4) & 0x0F) | ((byte(j - 4) >> 2) & 0x30);
        mn[j] = (byte(j + 4) >> 4) | ((byte(j) >> 2) & 0x30);
    }
    Ok((sc, mn))
}

/// Unpacks the 12-byte `Q3_K` scale payload into sixteen signed
/// per-16-block scales: low nibbles of bytes 0–7 are scales 0–7/8–15,
/// bytes 8–11 hold the 2-bit high parts (gguf-py Q3_K scale law).
pub fn unpack_q3_scales(payload: &[u8]) -> Result<[i8; 16], FormatError> {
    if payload.len() != 12 {
        return Err(FormatError::LengthMismatch {
            what: "q3_k scale payload",
            expected: 12,
            got: payload.len() as u64,
        });
    }
    let mut out = [0i8; 16];
    for j in 0..16 {
        let low = (payload[j % 8] >> ((j / 8) * 4)) & 0x0F;
        let high = (payload[8 + (j % 4)] >> ((j / 4) * 2)) & 0x03;
        let raw = low | (high << 4);
        out[j] = raw as i8 - 32;
    }
    Ok(out)
}

/// Reference dequantization of GGUF source bytes (Spec 2 §3.3, §10;
/// card A2.3): `wire` holds `n_rows` rows of `k` values in row-major
/// wire-block order; returns `n_rows * k` `f32` values in row-major
/// order, ascending, matching gguf-py 0.19.0 `dequantize` bit-exact.
///
/// Every scale failure is collected per block and reported with its
/// block index (CONVENTIONS.md §1.4); non-finite `f16` scales are
/// rejected, negative ones accepted (unlike the native-scale rule,
/// which requires non-negative multipliers).
pub fn ggml_dequantize(
    ggml: GgmlType,
    wire: &[u8],
    n_rows: u32,
    k: u32,
) -> Result<Vec<f32>, FormatError> {
    let geo = wire_geometry(ggml, wire, n_rows, k)?;
    let block_len = ggml.block_len() as usize;
    let block_bytes = ggml.block_bytes() as usize;
    let scheme_name = match ggml.scheme() {
        Some(s) => s.name(),
        None => ggml.name(),
    };
    let mut out = Vec::with_capacity(n_rows as usize * k as usize);
    let mut values = vec![0u16; block_len];
    let mut problems = Vec::new();
    let row_bytes = geo.blocks_per_row as usize * block_bytes;
    for row in 0..n_rows as usize {
        for b in 0..geo.blocks_per_row as usize {
            let base = row * row_bytes + b * block_bytes;
            let block = &wire[base..base + block_bytes];
            let record = (row * geo.blocks_per_row as usize + b) as u64;
            if let Err(e) = block_values(ggml, block, &mut values) {
                problems.push(e);
                continue;
            }
            match dequant_block(ggml, scheme_name, record, block, &values, &mut out) {
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

/// Decodes one parsed block's logical values with its wire scales,
/// appending to `out` in ascending order (gguf-py `dequantize_blocks`
/// evaluation order: `(d·sc)·q − (dmin·mn)` per sub-block for K types,
/// `d·q+m` / `d·(q−zero)` for `_0`/`_1` types).
fn dequant_block(
    ggml: GgmlType,
    scheme_name: &'static str,
    record: u64,
    block: &[u8],
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
        GgmlType::Q8_0 => {
            let d = check_wire_f16(scheme_name, record, u16_le(block, 0, "q8_0 scale")?)?;
            for v in values {
                out.push(d * (*v as u8 as i8) as f32);
            }
            Ok(())
        }
        GgmlType::Q4_0 => {
            let d = check_wire_f16(scheme_name, record, u16_le(block, 0, "q4_0 scale")?)?;
            for v in values {
                out.push(d * (*v as f32 - 8.0));
            }
            Ok(())
        }
        GgmlType::Q4_1 => {
            let d = check_wire_f16(scheme_name, record, u16_le(block, 0, "q4_1 scale")?)?;
            let m = check_wire_f16(scheme_name, record, u16_le(block, 2, "q4_1 min")?)?;
            for v in values {
                out.push(d * *v as f32 + m);
            }
            Ok(())
        }
        GgmlType::Q5_0 => {
            let d = check_wire_f16(scheme_name, record, u16_le(block, 0, "q5_0 scale")?)?;
            for v in values {
                out.push(d * (*v as f32 - 16.0));
            }
            Ok(())
        }
        GgmlType::Q5_1 => {
            let d = check_wire_f16(scheme_name, record, u16_le(block, 0, "q5_1 scale")?)?;
            let m = check_wire_f16(scheme_name, record, u16_le(block, 2, "q5_1 min")?)?;
            for v in values {
                out.push(d * *v as f32 + m);
            }
            Ok(())
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let d = check_wire_f16(scheme_name, record, u16_le(block, 0, "qk scale")?)?;
            let dmin = check_wire_f16(scheme_name, record, u16_le(block, 2, "qk min scale")?)?;
            let payload = match block.get(4..16) {
                Some(p) => p,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "qk scale payload",
                        expected: 16,
                        got: block.len() as u64,
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
            let scales = match block.get(192..208) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q6_k scales",
                        expected: 208,
                        got: block.len() as u64,
                    });
                }
            };
            let d = check_wire_f16(scheme_name, record, u16_le(block, 208, "q6_k scale")?)?;
            for j in 0..16 {
                let dl = d * scales[j] as i8 as f32;
                for v in &values[j * 16..(j + 1) * 16] {
                    out.push(dl * (*v as f32 - 32.0));
                }
            }
            Ok(())
        }
        GgmlType::Q3_K => {
            let payload = match block.get(96..108) {
                Some(p) => p,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q3_k scale payload",
                        expected: 108,
                        got: block.len() as u64,
                    });
                }
            };
            let sc = unpack_q3_scales(payload)?;
            let d = check_wire_f16(scheme_name, record, u16_le(block, 108, "q3_k scale")?)?;
            for j in 0..16 {
                let dl = d * sc[j] as f32;
                for v in &values[j * 16..(j + 1) * 16] {
                    // Stored u3 holds low 2 bits plus the decoded
                    // high bit (wire bit already inverted at parse);
                    // decode is ql − (qh<<2) in −4..3
                    // (gguf-py Q3_K law).
                    let low = (*v & 3) as i8;
                    let high = ((*v >> 2) & 1) as i8;
                    out.push(dl * (low - (high << 2)) as f32);
                }
            }
            Ok(())
        }
        GgmlType::Q2_K => {
            let scales = match block.get(0..16) {
                Some(s) => s,
                None => {
                    return Err(FormatError::LengthMismatch {
                        what: "q2_k scales",
                        expected: 16,
                        got: block.len() as u64,
                    });
                }
            };
            let d = check_wire_f16(scheme_name, record, u16_le(block, 80, "q2_k scale")?)?;
            let dmin = check_wire_f16(scheme_name, record, u16_le(block, 82, "q2_k min scale")?)?;
            for j in 0..16 {
                let dl = d * (scales[j] & 0x0F) as f32;
                let ml = dmin * (scales[j] >> 4) as f32;
                for v in &values[j * 16..(j + 1) * 16] {
                    out.push(dl * *v as f32 - ml);
                }
            }
            Ok(())
        }
    }
}
