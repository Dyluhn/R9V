// SPDX-License-Identifier: Apache-2.0
//! Simple reference encoders (Spec 2 §3.2; card A2.2).
//!
//! Round-to-nearest quantizers that pin the card's encode→decode error
//! bounds. No importance weighting, no GPTQ: real rounding quality
//! belongs to the quant tool (card A6.5).

use crate::records::{E4M3Block128Scale, I4KSuperblock, I8Block128Scale, I8RowScale};
use crate::scales::{f16_scale_bits, f16_to_f32, E4m3};
use crate::FormatError;

/// Rejects non-finite inputs with positions (CONVENTIONS.md §1.4).
fn check_inputs(x: &[f32]) -> Result<(), FormatError> {
    let mut problems = Vec::new();
    for (pos, v) in x.iter().enumerate() {
        if !v.is_finite() {
            problems.push(FormatError::ValueOutOfRange {
                what: "input",
                position: pos as u64,
                value: v.to_bits() as u64,
            });
        }
    }
    FormatError::collect(problems)
}

/// Simple reference quantizer for one [`crate::scheme::SchemeId::I8R`] row: symmetric
/// scale `s = max|x|/127`, round-to-nearest, exact zero preserved.
/// Empty and non-finite inputs are rejected; `f16`-overflowing scales
/// are rejected, never infinited.
// DECISION(A2.2): symmetric ±127 grid (rejected −128..127) so zero
// stays exact and the grid matches the Q8_0 convention; `round`
// (half away from zero), rejected the llama.cpp magic-number RNE,
// because A6.5 owns rounding quality and this encoder only pins
// error bounds.
pub fn encode_i8_row(x: &[f32]) -> Result<(Vec<i8>, I8RowScale), FormatError> {
    if x.is_empty() {
        return Err(FormatError::InvalidDim {
            name: "k",
            value: 0,
            reason: "must be at least 1",
        });
    }
    check_inputs(x)?;
    let mut peak = 0.0f32;
    for v in x {
        let a = v.abs();
        if a > peak {
            peak = a;
        }
    }
    if peak == 0.0 {
        return Ok((vec![0i8; x.len()], I8RowScale::from_bits(0)));
    }
    let s = peak / 127.0;
    let bits = f16_scale_bits(s, I8RowScale::SCHEME.name(), 0)?;
    let mut q = Vec::with_capacity(x.len());
    for v in x {
        q.push((v / s).round().clamp(-127.0, 127.0) as i8);
    }
    Ok((q, I8RowScale::from_bits(bits)))
}

/// Simple reference quantizer for [`crate::scheme::SchemeId::I8B128`]: per-128-block
/// [`encode_i8_row`]. Length must be a nonzero multiple of 128.
pub fn encode_i8_block128(x: &[f32]) -> Result<(Vec<i8>, Vec<I8Block128Scale>), FormatError> {
    if x.is_empty() || !x.len().is_multiple_of(128) {
        return Err(FormatError::LengthMismatch {
            what: "i8_b128 inputs",
            expected: ((x.len() / 128) + 1) as u64 * 128,
            got: x.len() as u64,
        });
    }
    check_inputs(x)?;
    let mut q = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len() / 128);
    for (b, block) in x.chunks_exact(128).enumerate() {
        let mut peak = 0.0f32;
        for v in block {
            let a = v.abs();
            if a > peak {
                peak = a;
            }
        }
        if peak == 0.0 {
            q.extend(std::iter::repeat_n(0i8, 128));
            scales.push(I8Block128Scale::from_bits(0));
            continue;
        }
        let s = peak / 127.0;
        let bits = f16_scale_bits(s, I8Block128Scale::SCHEME.name(), b as u64)?;
        for v in block {
            q.push((v / s).round().clamp(-127.0, 127.0) as i8);
        }
        scales.push(I8Block128Scale::from_bits(bits));
    }
    Ok((q, scales))
}

/// Simple reference quantizer for one [`crate::scheme::SchemeId::I4K`] superblock:
/// per-32-block min/max grid with super-scales over the block maxima
/// (`d = max_step/63`, `dmin = max_min/63`), mirroring the
/// `quantize_row_q4_K_ref` structure without its importance weighting
/// (card A6.5 owns real rounding). Non-finite inputs and
/// `f16`-overflowing super-scales are rejected.
pub fn encode_i4k_superblock(x: &[f32; 256]) -> Result<(Vec<u8>, I4KSuperblock), FormatError> {
    check_inputs(x)?;
    let mut steps = [0.0f32; 8];
    let mut floors = [0.0f32; 8];
    for (j, block) in x.chunks_exact(32).enumerate() {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for v in block {
            if *v < lo {
                lo = *v;
            }
            if *v > hi {
                hi = *v;
            }
        }
        // Mirror make_qkx2_quants: minima are non-positive.
        if lo > 0.0 {
            lo = 0.0;
        }
        steps[j] = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
        floors[j] = -lo;
    }
    let mut peak_step = 0.0f32;
    let mut peak_floor = 0.0f32;
    for j in 0..8 {
        if steps[j] > peak_step {
            peak_step = steps[j];
        }
        if floors[j] > peak_floor {
            peak_floor = floors[j];
        }
    }
    let dall = if peak_step > 0.0 {
        peak_step / 63.0
    } else {
        0.0
    };
    let dmins = if peak_floor > 0.0 {
        peak_floor / 63.0
    } else {
        0.0
    };
    let d_bits = f16_scale_bits(dall, I4KSuperblock::SCHEME.name(), 0)?;
    let dmin_bits = f16_scale_bits(dmins, I4KSuperblock::SCHEME.name(), 0)?;
    let d_f32 = f16_to_f32(d_bits);
    let dmin_f32 = f16_to_f32(dmin_bits);
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..8 {
        sc[j] = if dall > 0.0 {
            (steps[j] / dall).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
        mn[j] = if dmins > 0.0 {
            (floors[j] / dmins).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
    }
    let mut q = Vec::with_capacity(256);
    for (j, block) in x.chunks_exact(32).enumerate() {
        let block_d = d_f32 * sc[j] as f32;
        let block_m = dmin_f32 * mn[j] as f32;
        for v in block {
            if block_d > 0.0 {
                q.push(((v + block_m) / block_d).round().clamp(0.0, 15.0) as u8);
            } else {
                q.push(0);
            }
        }
    }
    // Internal invariant: sc/mn are clamped to 0..64 by construction.
    let header =
        I4KSuperblock::pack(d_bits, dmin_bits, sc, mn).expect("encoder clamps sc/mn to 6 bits");
    Ok((q, header))
}

/// Simple reference quantizer for [`crate::scheme::SchemeId::E4M3B128`]: per-128-block
/// scale `s = max|x|/448`, values projected with [`E4m3::from_f32`].
/// Length must be a nonzero multiple of 128; non-finite inputs and
/// `f16`-overflowing scales are rejected.
pub fn encode_e4m3_block128(x: &[f32]) -> Result<(Vec<E4m3>, Vec<E4M3Block128Scale>), FormatError> {
    if x.is_empty() || !x.len().is_multiple_of(128) {
        return Err(FormatError::LengthMismatch {
            what: "e4m3_b128 inputs",
            expected: ((x.len() / 128) + 1) as u64 * 128,
            got: x.len() as u64,
        });
    }
    check_inputs(x)?;
    let mut q = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len() / 128);
    for (b, block) in x.chunks_exact(128).enumerate() {
        let mut peak = 0.0f32;
        for v in block {
            let a = v.abs();
            if a > peak {
                peak = a;
            }
        }
        if peak == 0.0 {
            q.extend(std::iter::repeat_n(E4m3::new(0), 128));
            scales.push(E4M3Block128Scale::from_bits(0));
            continue;
        }
        let s = peak / 448.0;
        let bits = f16_scale_bits(s, E4M3Block128Scale::SCHEME.name(), b as u64)?;
        let s_f32 = f16_to_f32(bits);
        for v in block {
            // Internal invariant: inputs are finite and s_f32 is
            // positive finite, so the quotient is finite and the
            // projection always succeeds.
            q.push(E4m3::from_f32(v / s_f32).expect("finite quotient always projects to E4M3"));
        }
        scales.push(E4M3Block128Scale::from_bits(bits));
    }
    Ok((q, scales))
}
