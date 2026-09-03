// SPDX-License-Identifier: Apache-2.0
//! Element data type conversions and scalar operations for T0 (Spec 1 §2.1, Spec 4 §2).

use r9v_ir::DType;

/// Converts IEEE 754 single precision float (`f32`) to IEEE 754 half precision float (`f16`) bits (Spec 1 §2.1, Spec 4 §2).
///
/// Uses round-to-nearest-even and handles subnormals, infinities, and NaNs.
pub fn f32_to_f16(val: f32) -> u16 {
    let u = val.to_bits();
    let sign = ((u >> 31) & 1) as u16;
    let exp = ((u >> 23) & 0xFF) as i32;
    let mant = u & 0x7F_FFFF;

    if exp == 0xFF {
        // NaN or Infinity
        if mant != 0 {
            // Quiet NaN
            (sign << 15) | 0x7E00 | ((mant >> 13) as u16).max(1)
        } else {
            // Infinity
            (sign << 15) | 0x7C00
        }
    } else if exp == 0 {
        // Zero or subnormal f32 -> zero f16
        sign << 15
    } else {
        let unbiased_exp = exp - 127;
        let new_exp = unbiased_exp + 15;

        if new_exp >= 31 {
            // Overflow -> Infinity
            (sign << 15) | 0x7C00
        } else if new_exp <= 0 {
            // Subnormal or underflow in f16
            if new_exp < -10 {
                // Underflow to zero
                sign << 15
            } else {
                let shift = 14 - new_exp;
                let full_mant = mant | 0x80_0000;
                let half = 1 << (shift - 1);
                let lsb = (full_mant >> shift) & 1;
                let rounded = full_mant + half - 1 + lsb;
                let final_mant = (rounded >> shift) as u16;
                (sign << 15) | final_mant
            }
        } else {
            // Normal f16
            let lsb = (mant >> 13) & 1;
            let rounding_bias = 0x0FFF + lsb;
            let rounded_mant = mant + rounding_bias;

            if rounded_mant >= 0x80_0000 {
                // Mantissa overflow
                let final_exp = new_exp + 1;
                if final_exp >= 31 {
                    (sign << 15) | 0x7C00
                } else {
                    (sign << 15) | ((final_exp as u16) << 10)
                }
            } else {
                (sign << 15) | ((new_exp as u16) << 10) | ((rounded_mant >> 13) as u16)
            }
        }
    }
}

/// Converts IEEE 754 half precision float (`f16`) bits to IEEE 754 single precision float (`f32`) (Spec 1 §2.1, Spec 4 §2).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0x1F {
        if mant == 0 {
            // Infinity
            f32::from_bits((sign << 31) | 0x7F80_0000)
        } else {
            // NaN
            f32::from_bits((sign << 31) | 0x7FC0_0000 | (mant << 13))
        }
    } else if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let shift = mant.leading_zeros() - 21;
            let normalized_mant = (mant << shift) & 0x03FF;
            let final_exp = 113 - shift;
            f32::from_bits((sign << 31) | (final_exp << 23) | (normalized_mant << 13))
        }
    } else {
        // Normal
        let final_exp = exp + 112;
        f32::from_bits((sign << 31) | (final_exp << 23) | (mant << 13))
    }
}

/// Converts IEEE 754 single precision float (`f32`) to bfloat16 (`bf16`) bits with round-to-nearest-even (Spec 1 §2.1, Spec 4 §2).
pub fn f32_to_bf16(val: f32) -> u16 {
    let u = val.to_bits();
    if val.is_nan() {
        return ((u >> 16) | 0x0040) as u16;
    }
    let lsb = (u >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    let rounded = u.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

/// Converts bfloat16 (`bf16`) bits to IEEE 754 single precision float (`f32`) (Spec 1 §2.1, Spec 4 §2).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Decodes OCP FP8 E4M3 byte to single precision float (`f32`) (Spec 1 §2.1, Spec 4 §2, spikes/fp8-wmma/fp8_wmma.hip).
pub fn fp8_e4m3_decode(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0x0F;
    let m = b & 0x07;
    let sign = if s != 0 { -1.0f32 } else { 1.0f32 };
    if e == 0 {
        // subnormal: (-1)^s * 2^-6 * (m / 8)
        sign * (m as f32 / 8.0) * 0.015625f32
    } else if e == 15 && m == 7 {
        // NaN
        f32::NAN
    } else {
        // normal
        let exp = (e as i32) - 7;
        let scale = 2.0f32.powi(exp);
        sign * (1.0 + m as f32 / 8.0) * scale
    }
}

/// Encodes single precision float (`f32`) to OCP FP8 E4M3 byte with saturation to ±448 (Spec 1 §2.1, Spec 4 §2, spikes/fp8-wmma/fp8_wmma.hip).
pub fn fp8_e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let av = v.abs();
    if av > 448.0 {
        return if v < 0.0 { 0xFE } else { 0x7E };
    }
    if av == 0.0 {
        return if v.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let mut best: u8 = 0x00;
    let mut best_d = 1e30f32;
    for c in 0u16..256u16 {
        let b = c as u8;
        if b == 0x7F || b == 0xFF {
            continue;
        }
        let dec = fp8_e4m3_decode(b);
        let d = (dec - v).abs();
        if d < best_d || (d == best_d && (b & 1) == 0 && (best & 1) == 1) {
            best_d = d;
            best = b;
        }
    }
    best
}

/// Decodes OCP FP8 E5M2 byte to single precision float (`f32`) (Spec 1 §2.1, Spec 4 §2).
pub fn fp8_e5m2_decode(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 2) & 0x1F;
    let m = b & 0x03;
    let sign = if s != 0 { -1.0f32 } else { 1.0f32 };
    if e == 0 {
        // subnormal: (-1)^s * 2^-14 * (m / 4)
        sign * (m as f32 / 4.0) * (2.0f32.powi(-14))
    } else if e == 31 {
        if m == 0 {
            if s != 0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let exp = (e as i32) - 15;
        let scale = 2.0f32.powi(exp);
        sign * (1.0 + m as f32 / 4.0) * scale
    }
}

/// Encodes single precision float (`f32`) to OCP FP8 E5M2 byte with saturation to ±57344 (Spec 1 §2.1, Spec 4 §2).
pub fn fp8_e5m2_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let av = v.abs();
    if av > 57344.0 {
        return if v < 0.0 { 0xFC } else { 0x7C };
    }
    if av == 0.0 {
        return if v.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let mut best: u8 = 0x00;
    let mut best_d = 1e30f32;
    for c in 0u16..256u16 {
        let b = c as u8;
        if (b & 0x7C) == 0x7C && (b & 0x03) != 0 {
            // NaN codes excluded
            continue;
        }
        let dec = fp8_e5m2_decode(b);
        let d = (dec - v).abs();
        if d < best_d || (d == best_d && (b & 1) == 0 && (best & 1) == 1) {
            best_d = d;
            best = b;
        }
    }
    best
}

/// Reads one element from a raw byte buffer at `index` and returns it as `f32` (Spec 1 §2.1, Spec 4 §2).
pub fn read_f32_at(dtype: DType, slice: &[u8], index: usize) -> f32 {
    match dtype {
        DType::F32 => {
            let offset = index * 4;
            let bytes = [
                slice[offset],
                slice[offset + 1],
                slice[offset + 2],
                slice[offset + 3],
            ];
            f32::from_ne_bytes(bytes)
        }
        DType::F16 => {
            let offset = index * 2;
            let bytes = [slice[offset], slice[offset + 1]];
            let bits = u16::from_ne_bytes(bytes);
            f16_to_f32(bits)
        }
        DType::Bf16 => {
            let offset = index * 2;
            let bytes = [slice[offset], slice[offset + 1]];
            let bits = u16::from_ne_bytes(bytes);
            bf16_to_f32(bits)
        }
        DType::E4m3 => fp8_e4m3_decode(slice[index]),
        DType::E5m2 => fp8_e5m2_decode(slice[index]),
        DType::I8 => slice[index] as i8 as f32,
        DType::I32 => {
            let offset = index * 4;
            let bytes = [
                slice[offset],
                slice[offset + 1],
                slice[offset + 2],
                slice[offset + 3],
            ];
            i32::from_ne_bytes(bytes) as f32
        }
        DType::U32 => {
            let offset = index * 4;
            let bytes = [
                slice[offset],
                slice[offset + 1],
                slice[offset + 2],
                slice[offset + 3],
            ];
            u32::from_ne_bytes(bytes) as f32
        }
        DType::Bool => {
            if slice[index] != 0 {
                1.0f32
            } else {
                0.0f32
            }
        }
        DType::I4 => {
            let byte = slice[index / 2];
            let nibble = if index.is_multiple_of(2) {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            // Sign extend 4-bit signed int
            let val = if nibble >= 8 {
                (nibble as i8) - 16
            } else {
                nibble as i8
            };
            val as f32
        }
    }
}

/// Reads one element from a raw byte buffer at `index` and returns it as `f64` (Spec 1 §2.1, Spec 4 §2).
pub fn read_f64_at(dtype: DType, slice: &[u8], index: usize) -> f64 {
    read_f32_at(dtype, slice, index) as f64
}

/// Writes one `f32` value into a raw byte buffer at `index` converted to `dtype` (Spec 1 §2.1, Spec 4 §2).
pub fn write_f32_at(dtype: DType, slice: &mut [u8], index: usize, val: f32) {
    match dtype {
        DType::F32 => {
            let offset = index * 4;
            let bytes = val.to_ne_bytes();
            slice[offset..offset + 4].copy_from_slice(&bytes);
        }
        DType::F16 => {
            let offset = index * 2;
            let bits = f32_to_f16(val);
            slice[offset..offset + 2].copy_from_slice(&bits.to_ne_bytes());
        }
        DType::Bf16 => {
            let offset = index * 2;
            let bits = f32_to_bf16(val);
            slice[offset..offset + 2].copy_from_slice(&bits.to_ne_bytes());
        }
        DType::E4m3 => {
            slice[index] = fp8_e4m3_encode(val);
        }
        DType::E5m2 => {
            slice[index] = fp8_e5m2_encode(val);
        }
        DType::I8 => {
            let rounded = val.round_ties_even();
            let clamped = rounded.clamp(-128.0, 127.0) as i8;
            slice[index] = clamped as u8;
        }
        DType::I32 => {
            let offset = index * 4;
            let rounded = val.round_ties_even();
            let clamped = rounded.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            slice[offset..offset + 4].copy_from_slice(&clamped.to_ne_bytes());
        }
        DType::U32 => {
            let offset = index * 4;
            let rounded = val.round_ties_even();
            let clamped = rounded.clamp(0.0, u32::MAX as f32) as u32;
            slice[offset..offset + 4].copy_from_slice(&clamped.to_ne_bytes());
        }
        DType::Bool => {
            slice[index] = if val != 0.0 && !val.is_nan() { 1 } else { 0 };
        }
        DType::I4 => {
            let rounded = val.round_ties_even();
            let clamped = rounded.clamp(-8.0, 7.0) as i8;
            let nibble = (clamped as u8) & 0x0F;
            let byte_idx = index / 2;
            if index.is_multiple_of(2) {
                slice[byte_idx] = (slice[byte_idx] & 0xF0) | nibble;
            } else {
                slice[byte_idx] = (slice[byte_idx] & 0x0F) | (nibble << 4);
            }
        }
    }
}

/// Returns the element size in bytes for a `DType`. Returns 1 for sub-byte `I4` (packed) (Spec 1 §2.1, Spec 4 §2).
pub fn dtype_element_size(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::E4m3 | DType::E5m2 | DType::I8 | DType::Bool => 1,
        DType::I4 => 1, // Packed 2 per byte
    }
}
