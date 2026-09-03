// SPDX-License-Identifier: Apache-2.0
//! Native scale-record structs (Spec 2 §3.2; card A2.2).
//!
//! Exact records for [`SchemeId::I8R`], [`SchemeId::I8B128`],
//! [`SchemeId::I4K`] and [`SchemeId::E4M3B128`]: `f16` scales stored
//! little-endian plus the Q4_K-identical 16-byte `I4_K` record
//! (DECISIONS.md D-004). All structs are plain data with total
//! parse/serialize; validity lives in the `value` accessors.

use crate::scales::check_f16_scale;
use crate::scheme::SchemeId;
use crate::FormatError;

/// `f16` scale for one [`SchemeId::I8R`] row (Spec 2 §3.2: `s: f16`,
/// stored `[N]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct I8RowScale {
    bits: u16,
}

impl I8RowScale {
    /// The owning scheme.
    pub const SCHEME: SchemeId = SchemeId::I8R;

    /// Wraps stored bits without validation; [`I8RowScale::value`]
    /// rejects non-finite and negative scales.
    pub const fn from_bits(bits: u16) -> Self {
        Self { bits }
    }

    /// Parses little-endian stored bytes (Spec 2 §6 alignment: scales
    /// follow values; `f16` is two bytes LE).
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Self {
            bits: u16::from_le_bytes(bytes),
        }
    }

    /// Returns the stored bits.
    pub const fn bits(self) -> u16 {
        self.bits
    }

    /// Serializes to little-endian stored bytes.
    pub const fn to_bytes(self) -> [u8; 2] {
        self.bits.to_le_bytes()
    }

    /// Validates to the `f32` multiplier (Spec 2 §3.2 `w = s·q`).
    pub fn value(self, record: u64) -> Result<f32, FormatError> {
        check_f16_scale(Self::SCHEME.name(), record, self.bits)
    }
}

/// `f16` scale for one 128-block (Spec 2 §3.2 [`SchemeId::I8B128`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct I8Block128Scale {
    bits: u16,
}

impl I8Block128Scale {
    /// The owning scheme.
    pub const SCHEME: SchemeId = SchemeId::I8B128;

    /// Wraps stored bits without validation; see [`I8RowScale`].
    pub const fn from_bits(bits: u16) -> Self {
        Self { bits }
    }

    /// Parses little-endian stored bytes.
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Self {
            bits: u16::from_le_bytes(bytes),
        }
    }

    /// Returns the stored bits.
    pub const fn bits(self) -> u16 {
        self.bits
    }

    /// Serializes to little-endian stored bytes.
    pub const fn to_bytes(self) -> [u8; 2] {
        self.bits.to_le_bytes()
    }

    /// Validates to the `f32` multiplier (Spec 2 §3.2 `w = s·q`).
    pub fn value(self, record: u64) -> Result<f32, FormatError> {
        check_f16_scale(Self::SCHEME.name(), record, self.bits)
    }
}

/// `f16` scale for one 128-block (Spec 2 §3.2 [`SchemeId::E4M3B128`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct E4M3Block128Scale {
    bits: u16,
}

impl E4M3Block128Scale {
    /// The owning scheme.
    pub const SCHEME: SchemeId = SchemeId::E4M3B128;

    /// Wraps stored bits without validation; see [`I8RowScale`].
    pub const fn from_bits(bits: u16) -> Self {
        Self { bits }
    }

    /// Parses little-endian stored bytes.
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Self {
            bits: u16::from_le_bytes(bytes),
        }
    }

    /// Returns the stored bits.
    pub const fn bits(self) -> u16 {
        self.bits
    }

    /// Serializes to little-endian stored bytes.
    pub const fn to_bytes(self) -> [u8; 2] {
        self.bits.to_le_bytes()
    }

    /// Validates to the `f32` multiplier (Spec 2 §3.2 `w = s·q`).
    pub fn value(self, record: u64) -> Result<f32, FormatError> {
        check_f16_scale(Self::SCHEME.name(), record, self.bits)
    }
}

/// `I4_K` superblock header, logical form (Spec 2 §3.2; DECISIONS.md
/// D-004): `d: f16`, `dmin: f16`, eight 6-bit sub-block scales and
/// eight 6-bit sub-block minimums. The 16-byte wire form is
/// bit-identical to GGUF `Q4_K` (`ggml/src/ggml-common.h`
/// `block_q4_K` minus the value bytes; the "12 B packed" of §3.2 is the
/// `sc`/`mn` sub-field, verified against compiled llama.cpp output in
/// `tests/schemes.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct I4KSuperblock {
    d: u16,
    dmin: u16,
    sc: [u8; 8],
    mn: [u8; 8],
}

impl I4KSuperblock {
    /// Wire size of the record: `d` + `dmin` + 12 packed scale bytes
    /// (Spec 2 §3.2; the "12 B packed" is the `sc`/`mn` sub-field).
    // DECISION(A2.2): I4_K scale record wire geometry is 16 bytes total
    // (d: f16 + dmin: f16 [4 B] plus sc/mn packed payload [12 B]), matching
    // field-identical GGUF Q4_K layout and 4.5 bpw; rejected a 12-byte total
    // record (which would yield 4.375 bpw and break Q4_K bit compatibility).
    // Spec 2 §3.2, DECISIONS.md D-004, SI-24.
    pub const RECORD_BYTES: usize = 16;
    /// The owning scheme.
    pub const SCHEME: SchemeId = SchemeId::I4K;

    /// Packs logical fields, validating the 6-bit ranges with every
    /// violation reported (CONVENTIONS.md §1.4). `d`/`dmin` are stored
    /// unchecked; [`I4KSuperblock::d_value`] validates them.
    pub fn pack(d: u16, dmin: u16, sc: [u8; 8], mn: [u8; 8]) -> Result<Self, FormatError> {
        let mut problems = Vec::new();
        for (j, v) in sc.iter().enumerate() {
            if *v >= 64 {
                problems.push(FormatError::ValueOutOfRange {
                    what: "i4_k scale",
                    position: j as u64,
                    value: *v as u64,
                });
            }
        }
        for (j, v) in mn.iter().enumerate() {
            if *v >= 64 {
                problems.push(FormatError::ValueOutOfRange {
                    what: "i4_k min",
                    position: j as u64,
                    value: *v as u64,
                });
            }
        }
        FormatError::collect(problems)?;
        Ok(Self { d, dmin, sc, mn })
    }

    /// Parses the 16-byte wire form: `d`/`dmin` little-endian `f16`,
    /// then the 12 packed scale bytes holding the 6-bit `sc`/`mn`
    /// fields (`get_scale_min_k4` in llama.cpp
    /// `ggml/src/ggml-quants.c`). Every 6-bit field is masked to
    /// range by construction, so parsing is total.
    pub fn from_bytes(bytes: &[u8; Self::RECORD_BYTES]) -> Self {
        let d = u16::from_le_bytes([bytes[0], bytes[1]]);
        let dmin = u16::from_le_bytes([bytes[2], bytes[3]]);
        let q = &bytes[4..16];
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for j in 0..8 {
            if j < 4 {
                sc[j] = q[j] & 63;
                mn[j] = q[j + 4] & 63;
            } else {
                sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
            }
        }
        Self { d, dmin, sc, mn }
    }

    /// Serializes to the 16-byte wire form (inverse of
    /// [`I4KSuperblock::from_bytes`]; repack identity).
    pub fn to_bytes(self) -> [u8; Self::RECORD_BYTES] {
        let mut out = [0u8; Self::RECORD_BYTES];
        out[0..2].copy_from_slice(&self.d.to_le_bytes());
        out[2..4].copy_from_slice(&self.dmin.to_le_bytes());
        out[4..8].copy_from_slice(&self.sc[0..4]);
        out[8..12].copy_from_slice(&self.mn[0..4]);
        for j in 4..8 {
            out[12 + j - 4] = (self.sc[j] & 0xF) | ((self.mn[j] & 0xF) << 4);
            out[4 + j - 4] |= (self.sc[j] >> 4) << 6;
            out[8 + j - 4] |= (self.mn[j] >> 4) << 6;
        }
        out
    }

    /// Returns the stored `d` bits.
    pub const fn d_bits(self) -> u16 {
        self.d
    }

    /// Returns the stored `dmin` bits.
    pub const fn dmin_bits(self) -> u16 {
        self.dmin
    }

    /// Returns the eight 6-bit sub-block scales.
    pub const fn scales(self) -> [u8; 8] {
        self.sc
    }

    /// Returns the eight 6-bit sub-block minimums.
    pub const fn mins(self) -> [u8; 8] {
        self.mn
    }

    /// Validates `d` to the `f32` super-scale (Spec 2 §3.2).
    pub fn d_value(&self, record: u64) -> Result<f32, FormatError> {
        check_f16_scale(Self::SCHEME.name(), record, self.d)
    }

    /// Validates `dmin` to the `f32` super-scale (Spec 2 §3.2).
    pub fn dmin_value(&self, record: u64) -> Result<f32, FormatError> {
        check_f16_scale(Self::SCHEME.name(), record, self.dmin)
    }
}
