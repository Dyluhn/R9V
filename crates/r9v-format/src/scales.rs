// SPDX-License-Identifier: Apache-2.0
//! Scale and value codecs for native schemes (Spec 2 §3.2; card A2.2).
//!
//! The `f16` bit codec (no new dependency) and the OCP `E4M3` value
//! codec (see SI-20). Record structs live in [`crate::records`], SoA
//! placement in [`crate::geometry`].

use crate::FormatError;

/// Exact `f16` bit pattern to `f32` (Spec 2 §3.2: scales stored as
/// `f16`). Every finite `f16` is exact in `f32`; NaN payloads are
/// quieted canonically. Total function.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) >> 10) & 0x1F;
    let mant = (bits as u32) & 0x3FF;
    let wide = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: renormalize into the f32 exponent.
            let mut m = mant;
            let mut e: i32 = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 112) as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(wide)
}

/// Round-to-nearest-even `f32` to `f16` bits (Spec 2 §3.2: scale
/// emission). Total function: NaN maps to canonical quiet NaN,
/// infinities and `f16` overflow map to infinity, either zero maps to
/// its signed zero. Verified bit-exact against numpy `float16` over
/// 300k adversarial values including ties (see `tests/schemes.rs`
/// provenance note). Scale validity (non-negative, finite, in range)
/// is enforced separately by [`f16_scale_bits`].
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let exp = (x >> 23) & 0xFF;
    let mant = x & 0x7F_FFFF;
    if exp == 0xFF {
        if mant == 0 {
            return ((x >> 16) & 0x8000) as u16 | 0x7C00;
        }
        return ((x >> 16) & 0x8000) as u16 | 0x7E00;
    }
    let e = exp as i32 - 112;
    if e >= 31 {
        return ((x >> 16) & 0x8000) as u16 | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return (x >> 16) as u16 & 0x8000;
        }
        let mant_full = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let kept = mant_full >> shift;
        let rest = mant_full & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let rounded = if rest > half || (rest == half && kept & 1 == 1) {
            kept + 1
        } else {
            kept
        };
        return (((x >> 16) & 0x8000) | (rounded & 0x3FF)) as u16;
    }
    let kept = mant >> 13;
    let rest = mant & 0x1FFF;
    let (mut kept, mut e) = (kept, e);
    if rest > 0x1000 || (rest == 0x1000 && kept & 1 == 1) {
        kept += 1;
        if kept == 0x400 {
            kept = 0;
            e += 1;
            if e >= 31 {
                return ((x >> 16) & 0x8000) as u16 | 0x7C00;
            }
        }
    }
    (((x >> 16) & 0x8000) | ((e as u32) << 10) | kept) as u16
}

/// Checked `f16` scale emission (Spec 2 §3.2): round-to-nearest-even
/// via [`f32_to_f16_bits`], rejecting NaN, infinite, negative and
/// `f16`-overflowing scales with the offending `f32` bits
/// (CONVENTIONS.md §1.3). Zero (either sign) is a legal scale.
pub fn f16_scale_bits(value: f32, scheme: &'static str, record: u64) -> Result<u16, FormatError> {
    check_f32_scale(scheme, record, value)?;
    let bits = f32_to_f16_bits(value);
    // Overflow converts to infinity, which scales must never be.
    if bits & 0x7FFF == 0x7C00 {
        return Err(FormatError::InvalidScale {
            scheme,
            record,
            bits: value.to_bits(),
            reason: "unrepresentable_in_f16",
        });
    }
    Ok(bits)
}

/// Validates an `f32` scale value (Spec 2 §3.2: scales are
/// non-negative finite multipliers). Rejects NaN, infinite and
/// negative scales with position and bits (CONVENTIONS.md §1.3, §1.4).
pub fn check_f32_scale(scheme: &'static str, record: u64, value: f32) -> Result<f32, FormatError> {
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
    if value < 0.0 {
        return Err(FormatError::InvalidScale {
            scheme,
            record,
            bits: value.to_bits(),
            reason: "negative",
        });
    }
    Ok(value)
}

// DECISION(A2.2): the `bits` reported for an `f16` bit pattern that
// decodes to a rejected value (NaN/Inf) is the decoded `f32` bits,
// which equal the widened pattern; rejected reporting raw `u16` bits
// because every other scale error carries `f32::to_bits`.
/// Validates stored `f16` scale bits to `f32` (see [`check_f32_scale`]).
pub fn check_f16_scale(scheme: &'static str, record: u64, bits: u16) -> Result<f32, FormatError> {
    check_f32_scale(scheme, record, f16_to_f32(bits))
}

/// 8-bit floating value in OCP `E4M3` encoding (Spec 2 §3.2 values for
/// `E4M3_B128`; see SI-20 for the encoding decision).
///
/// Bit layout: sign 1, exponent 4 (bias 7), mantissa 3. Exponent 15
/// with mantissa 7 is NaN (`0x7F`/`0xFF`); remaining exponent-15
/// patterns are extended normals at exponent 8 (max 448.0); the
/// smallest normal is 2^-6 (`0x08`), subnormals span 2^-9..2^-6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct E4m3(u8);

impl E4m3 {
    /// Wraps raw bits without validation; [`E4m3::check`] rejects NaN
    /// patterns at decode boundaries.
    pub const fn new(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns the raw bits.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether the bits are an OCP `E4M3` NaN (`0x7F` or `0xFF`).
    pub const fn is_nan(self) -> bool {
        self.0 & 0x7F == 0x7F
    }

    /// Rejects NaN patterns with position (Spec 2 §3.2; NaN has no
    /// decoded value and must not reach arithmetic).
    pub fn check(self, position: u64) -> Result<Self, FormatError> {
        if self.is_nan() {
            return Err(FormatError::ValueOutOfRange {
                what: "e4m3",
                position,
                value: self.0 as u64,
            });
        }
        Ok(self)
    }

    /// Exact decode to `f32` (Spec 2 §3.2 `w = s·q`). Every finite
    /// pattern is exact; NaN patterns yield `f32::NAN` (prefer
    /// [`E4m3::check`] at boundaries).
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits & 0x80) << 24;
        let exp = (bits >> 3) & 0xF;
        let mant = bits & 0x7;
        let wide = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                // Subnormal: 2^-6 * (m/8).
                let mut m = mant;
                let mut e: i32 = 1;
                while m & 0x8 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (((e + 120) as u32) << 23) | ((m & 0x7) << 20)
            }
        } else if exp == 15 {
            if mant == 7 {
                sign | 0x7FC0_0000
            } else {
                // Extended normals at exponent 8 (max 448.0).
                sign | (135 << 23) | (mant << 20)
            }
        } else {
            // Unbiased exponent e - 7 in f32 bias: e + 120.
            sign | ((exp + 120) << 23) | (mant << 20)
        };
        f32::from_bits(wide)
    }

    /// Projects finite `value` onto the `E4M3` grid
    /// (round-to-nearest, ties to the even pattern LSB). Magnitudes
    /// past the 448.0 grid maximum saturate to ±448.0 (OCP convert
    /// semantics); NaN and infinite inputs return `None` and are
    /// reported by the block encoder (CONVENTIONS.md §1.4).
    // DECISION(A2.2): saturating finite overflow (rejected erroring)
    // because clamping to the grid is what quantization means (cf.
    // llama.cpp MIN/MAX clamps), while non-finite inputs are corrupt
    // data, not coarse data. See SI-20.
    pub fn from_f32(value: f32) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let sign = if value.is_sign_negative() { 0x80 } else { 0x00 };
        let target = value.abs() as f64;
        let mut best: u8 = 0x00;
        let mut best_dist = f64::INFINITY;
        for mag in 0..0x7Fu8 {
            let grid = E4m3(mag).to_f32() as f64;
            let dist = (target - grid).abs();
            if dist < best_dist || (dist == best_dist && mag & 1 == 0) {
                best_dist = dist;
                best = mag;
            }
        }
        Some(E4m3(best | sign))
    }
}
