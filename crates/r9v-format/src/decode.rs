// SPDX-License-Identifier: Apache-2.0
//! Reference dequantization and simple encoders (Spec 2 §3.2; card A2.2).
//!
//! [`decode`] is the pure checked reference `decode(scheme, values,
//! scales) -> f32` used by T0 and tests: one logical value plus its
//! validated scales, evaluated with the exact §3.2 formulas in
//! ascending value order with no cross-value accumulation (that order
//! belongs to `matmul`, card A1.6). Block helpers decode whole blocks
//! with collect-all validation. Simple encoders live in
//! [`crate::encode`].

use crate::records::{E4M3Block128Scale, I4KSuperblock, I8Block128Scale, I8RowScale};
use crate::scales::{check_f32_scale, E4m3};
use crate::scheme::{NativeScheme, SchemeId};
use crate::FormatError;

/// One logical quantized value (Spec 2 §3.2 value bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantValue {
    /// Signed 8-bit value (`I8_R`, `I8_B128`).
    I8(i8),
    /// Unsigned 4-bit value, 0..15 (`I4_K`).
    I4(u8),
    /// OCP `E4M3` value (`E4M3_B128`).
    E4M3(E4m3),
}

impl QuantValue {
    /// Stable kind name for mismatch reports.
    fn kind(self) -> &'static str {
        match self {
            QuantValue::I8(_) => "i8",
            QuantValue::I4(_) => "u4",
            QuantValue::E4M3(_) => "e4m3",
        }
    }
}

/// Validated scales for one [`decode`] call (Spec 2 §3.2 records).
///
/// Fields are private: the only way to build a [`ScaleSet`] is through
/// the checked constructors, so [`decode`] can trust the type without
/// re-validating (CONVENTIONS.md §2.2 boundary rule).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleSet {
    /// Plain `f16` multiplier (`I8_R`, `I8_B128`, `E4M3_B128`).
    F16 {
        /// Validated non-negative finite multiplier.
        s: f32,
    },
    /// One `I4_K` sub-block's factors: super-scales plus 6-bit
    /// sub-block scale and minimum.
    I4K {
        /// Validated super-scale for quantized scales.
        d: f32,
        /// Sub-block scale, below 64.
        sc: u8,
        /// Validated super-scale for quantized minimums.
        dmin: f32,
        /// Sub-block minimum, below 64.
        mn: u8,
    },
}

impl ScaleSet {
    /// Builds a validated plain multiplier (Spec 2 §3.2 `s: f16`
    /// records as `f32`). `scheme`/`record` name the owner for error
    /// reports; kind agreement is enforced by [`decode`].
    pub fn f16(scheme: SchemeId, record: u64, s: f32) -> Result<Self, FormatError> {
        check_f32_scale(scheme.name(), record, s).map(|s| ScaleSet::F16 { s })
    }

    /// Builds validated `I4_K` sub-block factors, reporting every
    /// invalid field (CONVENTIONS.md §1.4).
    pub fn i4k(
        scheme: SchemeId,
        record: u64,
        d: f32,
        sc: u8,
        dmin: f32,
        mn: u8,
    ) -> Result<Self, FormatError> {
        let mut problems = Vec::new();
        let d = match check_f32_scale(scheme.name(), record, d) {
            Ok(d) => Some(d),
            Err(e) => {
                problems.push(e);
                None
            }
        };
        let dmin = match check_f32_scale(scheme.name(), record, dmin) {
            Ok(dmin) => Some(dmin),
            Err(e) => {
                problems.push(e);
                None
            }
        };
        if sc >= 64 {
            problems.push(FormatError::ValueOutOfRange {
                what: "i4_k scale",
                position: 0,
                value: sc as u64,
            });
        }
        if mn >= 64 {
            problems.push(FormatError::ValueOutOfRange {
                what: "i4_k min",
                position: 0,
                value: mn as u64,
            });
        }
        FormatError::collect(problems)?;
        // Internal invariant: both are Some because problems was empty.
        Ok(ScaleSet::I4K {
            d: d.expect("d valid when problems is empty"),
            sc,
            dmin: dmin.expect("dmin valid when problems is empty"),
            mn,
        })
    }

    /// Stable kind name for mismatch reports.
    fn kind(&self) -> &'static str {
        match self {
            ScaleSet::F16 { .. } => "f16",
            ScaleSet::I4K { .. } => "i4_k",
        }
    }
}

/// Reference dequantization of one logical value (Spec 2 §3.2; card
/// A2.2): `decode(scheme, values, scales) -> f32`.
///
/// Exact formulas, each evaluated left to right with no accumulation
/// across values: `w = s·q` for the `i8`/`e4m3` schemes,
/// `w = (d·sc)·q − (dmin·mn)` for `I4_K` (the llama.cpp
/// `dequantize_row_q4_K` order). Wrong value/scale kinds, out-of-range
/// nibbles and NaN `e4m3` patterns are typed errors; repack-only ids
/// fail closed with their owning card.
pub fn decode(scheme: SchemeId, value: QuantValue, scales: &ScaleSet) -> Result<f32, FormatError> {
    match scheme.as_native()? {
        NativeScheme::I8R | NativeScheme::I8B128 => match (value, scales) {
            (QuantValue::I8(q), ScaleSet::F16 { s }) => Ok(s * q as f32),
            _ => Err(FormatError::SchemeMismatch {
                scheme: scheme.name(),
                expected: "i8 + f16",
                got: pair_name(value, scales),
            }),
        },
        NativeScheme::E4M3B128 => match (value, scales) {
            (QuantValue::E4M3(q), ScaleSet::F16 { s }) => {
                q.check(0)?;
                Ok(s * q.to_f32())
            }
            _ => Err(FormatError::SchemeMismatch {
                scheme: scheme.name(),
                expected: "e4m3 + f16",
                got: pair_name(value, scales),
            }),
        },
        NativeScheme::I4K => match (value, scales) {
            (QuantValue::I4(q), ScaleSet::I4K { d, sc, dmin, mn }) => {
                if q > 15 {
                    return Err(FormatError::ValueOutOfRange {
                        what: "i4_k nibble",
                        position: 0,
                        value: q as u64,
                    });
                }
                // Internal invariant: sc/mn are below 64 by the
                // ScaleSet::i4k constructor, so the type proves it here.
                Ok((d * *sc as f32) * q as f32 - (dmin * *mn as f32))
            }
            _ => Err(FormatError::SchemeMismatch {
                scheme: scheme.name(),
                expected: "u4 + i4_k",
                got: pair_name(value, scales),
            }),
        },
    }
}

/// Names a value/scale pair for mismatch reports without allocating.
fn pair_name(value: QuantValue, scales: &ScaleSet) -> &'static str {
    match (value.kind(), scales.kind()) {
        ("i8", "f16") => "i8 + f16",
        ("u4", "i4_k") => "u4 + i4_k",
        ("e4m3", "f16") => "e4m3 + f16",
        ("i8", "i4_k") => "i8 + i4_k",
        ("u4", "f16") => "u4 + f16",
        ("e4m3", "i4_k") => "e4m3 + i4_k",
        _ => "unknown",
    }
}

/// Decodes one [`SchemeId::I8R`] row in ascending order (Spec 2 §3.2
/// `w = s·q`). Empty rows are rejected; scale errors are reported.
pub fn decode_i8_row(q: &[i8], scale: &I8RowScale) -> Result<Vec<f32>, FormatError> {
    if q.is_empty() {
        return Err(FormatError::InvalidDim {
            name: "k",
            value: 0,
            reason: "must be at least 1",
        });
    }
    let s = scale.value(0)?;
    Ok(q.iter().map(|v| s * *v as f32).collect())
}

/// Decodes 128-blocks in ascending order (Spec 2 §3.2 `w = s·q` for
/// [`SchemeId::I8B128`]). `q.len()` must be a nonzero multiple of 128
/// with one scale per block; every scale failure is reported
/// (CONVENTIONS.md §1.4).
pub fn decode_i8_block128(q: &[i8], scales: &[I8Block128Scale]) -> Result<Vec<f32>, FormatError> {
    if q.is_empty() || !q.len().is_multiple_of(128) {
        return Err(FormatError::LengthMismatch {
            what: "i8_b128 values",
            expected: ((q.len() / 128) + 1) as u64 * 128,
            got: q.len() as u64,
        });
    }
    let mut s_vals = Vec::with_capacity(scales.len());
    let mut problems = Vec::new();
    // Structural count first: it is independent of scale validity,
    // while s_vals shortfall would only repeat a validity failure.
    if scales.len() != q.len() / 128 {
        problems.push(FormatError::LengthMismatch {
            what: "i8_b128 scales",
            expected: (q.len() / 128) as u64,
            got: scales.len() as u64,
        });
    }
    for (i, sc) in scales.iter().enumerate() {
        match sc.value(i as u64) {
            Ok(s) => s_vals.push(s),
            Err(e) => problems.push(e),
        }
    }
    FormatError::collect(problems)?;
    let mut out = Vec::with_capacity(q.len());
    for (block, s) in s_vals.iter().enumerate() {
        for v in &q[block * 128..(block + 1) * 128] {
            out.push(s * *v as f32);
        }
    }
    Ok(out)
}

/// Decodes one 256-superblock in ascending sub-block order
/// (Spec 2 §3.2 `w = (d·sc)·q − (dmin·mn)` per 32-block; the
/// llama.cpp `dequantize_row_q4_K` order). Exactly 256 nibbles
/// (0..15) are required; every out-of-range nibble and every invalid
/// super-scale is reported (CONVENTIONS.md §1.4).
pub fn decode_i4k_superblock(q: &[u8], header: &I4KSuperblock) -> Result<Vec<f32>, FormatError> {
    if q.len() != 256 {
        return Err(FormatError::LengthMismatch {
            what: "i4_k nibbles",
            expected: 256,
            got: q.len() as u64,
        });
    }
    let mut problems = Vec::new();
    for (pos, v) in q.iter().enumerate() {
        if *v > 15 {
            problems.push(FormatError::ValueOutOfRange {
                what: "i4_k nibble",
                position: pos as u64,
                value: *v as u64,
            });
        }
    }
    let d = match header.d_value(0) {
        Ok(d) => Some(d),
        Err(e) => {
            problems.push(e);
            None
        }
    };
    let dmin = match header.dmin_value(0) {
        Ok(dmin) => Some(dmin),
        Err(e) => {
            problems.push(e);
            None
        }
    };
    FormatError::collect(problems)?;
    // Internal invariant: both are Some because problems was empty.
    let (d, dmin) = (
        d.expect("d valid when problems is empty"),
        dmin.expect("dmin valid when problems is empty"),
    );
    let sc = header.scales();
    let mn = header.mins();
    let mut out = Vec::with_capacity(256);
    for (jb, chunk) in q.chunks_exact(32).enumerate() {
        let block_d = d * sc[jb] as f32;
        let block_m = dmin * mn[jb] as f32;
        for v in chunk {
            out.push(block_d * *v as f32 - block_m);
        }
    }
    Ok(out)
}

/// Decodes 128-blocks in ascending order (Spec 2 §3.2 `w = s·q` for
/// [`SchemeId::E4M3B128`]). Length and scale rules match
/// [`decode_i8_block128`]; NaN `e4m3` patterns are rejected per value
/// with positions (CONVENTIONS.md §1.4).
pub fn decode_e4m3_block128(
    q: &[E4m3],
    scales: &[E4M3Block128Scale],
) -> Result<Vec<f32>, FormatError> {
    if q.is_empty() || !q.len().is_multiple_of(128) {
        return Err(FormatError::LengthMismatch {
            what: "e4m3_b128 values",
            expected: ((q.len() / 128) + 1) as u64 * 128,
            got: q.len() as u64,
        });
    }
    let mut s_vals = Vec::with_capacity(scales.len());
    let mut problems = Vec::new();
    // Structural count first: it is independent of scale validity,
    // while s_vals shortfall would only repeat a validity failure.
    if scales.len() != q.len() / 128 {
        problems.push(FormatError::LengthMismatch {
            what: "e4m3_b128 scales",
            expected: (q.len() / 128) as u64,
            got: scales.len() as u64,
        });
    }
    for (i, sc) in scales.iter().enumerate() {
        match sc.value(i as u64) {
            Ok(s) => s_vals.push(s),
            Err(e) => problems.push(e),
        }
    }
    for (pos, v) in q.iter().enumerate() {
        if v.is_nan() {
            problems.push(FormatError::ValueOutOfRange {
                what: "e4m3",
                position: pos as u64,
                value: v.bits() as u64,
            });
        }
    }
    FormatError::collect(problems)?;
    let mut out = Vec::with_capacity(q.len());
    for (block, s) in s_vals.iter().enumerate() {
        for v in &q[block * 128..(block + 1) * 128] {
            out.push(s * v.to_f32());
        }
    }
    Ok(out)
}
