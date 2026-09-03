// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 quant_act (Spec 1 §4.A, Spec 2 §3.4, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{DType, QuantActOp, QuantScheme, Smoothing};
use r9v_t0::{
    fp8_e4m3_decode, fp8_e4m3_encode, fp8_e4m3_encode_f64_oracle, quant_act,
    quant_act_f64_reference, Tolerance, TypedBuffer,
};

fn generate_f32_data(rng: &mut SeededRng, len: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let raw = (rng.next_u64() & 0xFFFF_FFFF) as u32;
        let norm_val = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(norm_val * scale);
    }
    out
}

#[test]
fn quant_act_per_token_i8_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5070);
    let tol = Tolerance::f32();
    let (t, n) = (3, 128);

    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 50.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let (ref_xq, ref_scale) = quant_act_f64_reference(&op, &x_f64, [t, n]);

    for row in 0..t {
        let actual_scale = scale_buf.read_f32(row) as f64;
        let expected_scale = ref_scale[row];
        tol.assert_within(
            actual_scale,
            expected_scale,
            &format!("quant_act scale at token {row}"),
        );
    }

    let actual_xq = xq_buf.to_i8_vec();
    for i in 0..(t * n) {
        assert_eq!(
            actual_xq[i] as f64, ref_xq[i],
            "quant_act i8 quantized mismatch at {i}"
        );
        // Check symmetric bound [-127, 127] (i8 cannot be -128)
        assert!(actual_xq[i] >= -127);
    }
}

#[test]
fn quant_act_per_token_e4m3_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5071);
    let tol = Tolerance::f32();
    let (t, n) = (3, 64);

    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::E4m3,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 100.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::E4m3);
    let mut scale_buf = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let (ref_xq, ref_scale) = quant_act_f64_reference(&op, &x_f64, [t, n]);

    for row in 0..t {
        let actual_scale = scale_buf.read_f32(row) as f64;
        let expected_scale = ref_scale[row];
        tol.assert_within(
            actual_scale,
            expected_scale,
            &format!("quant_act e4m3 scale at token {row}"),
        );
    }

    let actual_bytes = xq_buf.to_byte_vec();
    for i in 0..(t * n) {
        assert_eq!(
            actual_bytes[i] as f64, ref_xq[i],
            "quant_act e4m3 byte mismatch at {i}"
        );
    }
}

#[test]
fn quant_act_per_block32_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5072);
    let tol = Tolerance::f32();
    let (t, n) = (2, 128); // 4 blocks of 32 per row

    let op = QuantActOp {
        scheme: QuantScheme::PerBlock32,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 40.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let num_blocks = n / 32;
    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[t, num_blocks], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let (ref_xq, ref_scale) = quant_act_f64_reference(&op, &x_f64, [t, n]);

    for b in 0..(t * num_blocks) {
        let actual_scale = scale_buf.read_f32(b) as f64;
        let expected_scale = ref_scale[b];
        tol.assert_within(
            actual_scale,
            expected_scale,
            &format!("quant_act perblock scale at block {b}"),
        );
    }

    let actual_xq = xq_buf.to_i8_vec();
    for i in 0..(t * n) {
        assert_eq!(
            actual_xq[i] as f64, ref_xq[i],
            "quant_act perblock i8 quantized mismatch at {i}"
        );
    }
}

#[test]
fn quant_act_zero_absmax_emits_zeros() {
    let (t, n) = (2, 64);
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_buf = TypedBuffer::zeros(&[t, n], DType::F32); // All zeros
    let mut xq_buf = TypedBuffer::zeros(&[t, n], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap();

    for row in 0..t {
        assert_eq!(scale_buf.read_f32(row), 0.0f32);
    }
    for val in xq_buf.to_i8_vec() {
        assert_eq!(val, 0);
    }
}

#[test]
fn quant_act_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5073);
    let n = 128;

    let target_row = generate_f32_data(&mut rng, n, 20.0);
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    // 1. Alone (T = 1)
    let x_alone = TypedBuffer::from_f32(&[1, n], &target_row);
    let mut xq_alone = TypedBuffer::zeros(&[1, n], DType::I8);
    let mut scale_alone = TypedBuffer::zeros(&[1], DType::F32);
    quant_act(
        &op,
        &x_alone.as_view(),
        &mut xq_alone.as_view_mut(),
        &mut scale_alone.as_view_mut(),
    )
    .unwrap();
    let out_alone = xq_alone.to_i8_vec();
    let scale_alone_val = scale_alone.read_f32(0);

    // 2. Padded (T = 4)
    let mut x_pad = target_row.clone();
    x_pad.extend(vec![0.0f32; 3 * n]);
    let x_pad_buf = TypedBuffer::from_f32(&[4, n], &x_pad);
    let mut xq_pad = TypedBuffer::zeros(&[4, n], DType::I8);
    let mut scale_pad = TypedBuffer::zeros(&[4], DType::F32);
    quant_act(
        &op,
        &x_pad_buf.as_view(),
        &mut xq_pad.as_view_mut(),
        &mut scale_pad.as_view_mut(),
    )
    .unwrap();
    let out_pad = &xq_pad.to_i8_vec()[..n];
    let scale_pad_val = scale_pad.read_f32(0);

    // 3. Embedded (T = 4, target at index 2)
    let other_tokens = generate_f32_data(&mut rng, 3 * n, 50.0);
    let mut x_emb = Vec::with_capacity(4 * n);
    x_emb.extend_from_slice(&other_tokens[..2 * n]);
    x_emb.extend_from_slice(&target_row);
    x_emb.extend_from_slice(&other_tokens[2 * n..]);
    let x_emb_buf = TypedBuffer::from_f32(&[4, n], &x_emb);
    let mut xq_emb = TypedBuffer::zeros(&[4, n], DType::I8);
    let mut scale_emb = TypedBuffer::zeros(&[4], DType::F32);
    quant_act(
        &op,
        &x_emb_buf.as_view(),
        &mut xq_emb.as_view_mut(),
        &mut scale_emb.as_view_mut(),
    )
    .unwrap();
    let out_emb = &xq_emb.to_i8_vec()[2 * n..3 * n];
    let scale_emb_val = scale_emb.read_f32(2);

    assert_eq!(scale_alone_val.to_bits(), scale_pad_val.to_bits());
    assert_eq!(scale_alone_val.to_bits(), scale_emb_val.to_bits());
    for i in 0..n {
        assert_eq!(out_alone[i], out_pad[i]);
        assert_eq!(out_alone[i], out_emb[i]);
    }
}

#[test]
fn quant_act_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5074);
    let (t, n) = (3, 64);

    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::E4m3,
        smoothing: Smoothing::None,
    };

    let x_data = generate_f32_data(&mut rng, t * n, 30.0);
    let x_buf = TypedBuffer::from_f32(&[t, n], &x_data);

    let mut xq1 = TypedBuffer::zeros(&[t, n], DType::E4m3);
    let mut s1 = TypedBuffer::zeros(&[t], DType::F32);
    let mut xq2 = TypedBuffer::zeros(&[t, n], DType::E4m3);
    let mut s2 = TypedBuffer::zeros(&[t], DType::F32);

    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq1.as_view_mut(),
        &mut s1.as_view_mut(),
    )
    .unwrap();
    quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq2.as_view_mut(),
        &mut s2.as_view_mut(),
    )
    .unwrap();

    assert_eq!(xq1.to_byte_vec(), xq2.to_byte_vec());
    assert_eq!(s1.to_f32_vec(), s2.to_f32_vec());
}

#[test]
fn quant_act_rejects_non_divisible_block_size_and_mismatches() {
    let op = QuantActOp {
        scheme: QuantScheme::PerBlock32,
        target: DType::I8,
        smoothing: Smoothing::None,
    };

    let x_buf = TypedBuffer::zeros(&[2, 50], DType::F32); // 50 not divisible by 32
    let mut xq_buf = TypedBuffer::zeros(&[2, 50], DType::I8);
    let mut scale_buf = TypedBuffer::zeros(&[2, 2], DType::F32);

    let err = quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("divisible by 32"));
}

#[test]
fn quant_act_rejects_invalid_per_token_target_without_panicking() {
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::F32, // Invalid target for quant_act (must be i8 or e4m3)
        smoothing: Smoothing::None,
    };

    let x_buf = TypedBuffer::zeros(&[2, 32], DType::F32);
    let mut xq_buf = TypedBuffer::zeros(&[2, 32], DType::F32);
    let mut scale_buf = TypedBuffer::zeros(&[2], DType::F32);

    let err = quant_act(
        &op,
        &x_buf.as_view(),
        &mut xq_buf.as_view_mut(),
        &mut scale_buf.as_view_mut(),
    )
    .unwrap_err();

    let msg = err.to_string();
    assert!(msg.contains("validation error(s)"));
    assert!(msg.contains("target must be i8 or e4m3") || msg.contains("only supports i8 or e4m3"));
}

/// Authoritative E4M3 golden vectors for the independent `f64` oracle (Spec 1 §2.1).
///
/// Hand-derived from the OCP FP8 E4M3 grid definition, not from the production encoder:
/// a regression in either implementation breaks this test from opposite sides.
#[test]
fn e4m3_f64_oracle_golden_vectors() {
    let cases: &[(f64, u8)] = &[
        (0.0, 0x00),
        (-0.0, 0x80),
        (1.0, 0x38),
        (-1.0, 0xB8),
        (2.0, 0x40),
        (0.5, 0x30),
        (0.015625, 0x08),
        (0.001953125, 0x01),
        (0.013671875, 0x07),
        (1.125, 0x39),
        (0.1, 0x1D),
        (1.0625, 0x38),  // exact tie between 1.0 and 1.125 rounds to even
        (-1.0625, 0xB8), // exact tie between -1.0 and -1.125 rounds to even
        (1.1875, 0x3A),  // exact tie between 1.125 and 1.25 rounds to even
        (448.0, 0x7E),
        (-448.0, 0xFE),
        (500.0, 0x7E),  // saturates to +448
        (-500.0, 0xFE), // saturates to -448
        (f64::INFINITY, 0x7E),
        (f64::NEG_INFINITY, 0xFE),
        (f64::NAN, 0x7F),
        (-f64::NAN, 0x7F),
    ];
    for &(input, expected) in cases {
        assert_eq!(
            fp8_e4m3_encode_f64_oracle(input),
            expected,
            "oracle golden mismatch for input {input}"
        );
    }
}

/// The same goldens through the production encoder (Spec 1 §2.1).
///
/// Uses only hard-coded expectations, never the oracle, so a production regression fails here
/// even if the oracle were wrong.
#[test]
fn e4m3_production_encoder_golden_vectors() {
    let cases: &[(f32, u8)] = &[
        (0.0, 0x00),
        (-0.0, 0x80),
        (1.0, 0x38),
        (-1.0, 0xB8),
        (2.0, 0x40),
        (0.5, 0x30),
        (0.015625, 0x08),
        (0.001953125, 0x01),
        (0.013671875, 0x07),
        (1.125, 0x39),
        (0.1, 0x1D),
        (1.0625, 0x38),
        (-1.0625, 0xB8),
        (1.1875, 0x3A),
        (448.0, 0x7E),
        (-448.0, 0xFE),
        (500.0, 0x7E),
        (-500.0, 0xFE),
        (f32::INFINITY, 0x7E),
        (f32::NEG_INFINITY, 0xFE),
        (f32::NAN, 0x7F),
    ];
    for &(input, expected) in cases {
        assert_eq!(
            fp8_e4m3_encode(input),
            expected,
            "production golden mismatch for input {input}"
        );
    }
}

/// Every finite E4M3 code round-trips through the production encoder (Spec 1 §2.1).
///
/// Encodes each exactly-representable grid value and requires the original byte back;
/// a production regression in saturation, NaN skipping, or nearest selection breaks this.
#[test]
fn e4m3_grid_values_roundtrip_through_production_encoder() {
    for code in 0u16..256u16 {
        let b = code as u8;
        if b == 0x7F || b == 0xFF {
            continue;
        }
        let grid = fp8_e4m3_decode(b);
        assert_eq!(
            fp8_e4m3_encode(grid),
            b,
            "roundtrip mismatch for code {b:#04X} (grid value {grid})"
        );
    }
}

/// Exact midpoints between adjacent grid codes round to even in both implementations (Spec 1 §2.1).
///
/// A flipped tie-break in the production encoder fails this while the oracle still passes.
#[test]
fn e4m3_midpoint_ties_round_to_even() {
    for code in 0u16..255u16 {
        let lo = code as u8;
        let hi = (code + 1) as u8;
        if lo == 0x7F || lo == 0xFF || hi == 0x7F || hi == 0xFF {
            continue;
        }
        if lo == 0x80 {
            // Midpoint between -0.0 and the first negative subnormal is a three-way
            // tie with +0.0; pinned separately below.
            continue;
        }
        let mid = (f64::from(fp8_e4m3_decode(lo)) + f64::from(fp8_e4m3_decode(hi))) / 2.0;
        let expected = if lo & 1 == 0 { lo } else { hi };
        assert_eq!(
            fp8_e4m3_encode(mid as f32),
            expected,
            "production tie mismatch between {lo:#04X} and {hi:#04X}"
        );
        assert_eq!(
            fp8_e4m3_encode_f64_oracle(mid),
            expected,
            "oracle tie mismatch between {lo:#04X} and {hi:#04X}"
        );
    }
}

/// The midpoint between -0.0 (`0x80`) and the first negative subnormal (`0x81`) ties
/// three ways with +0.0 (`0x00`); both implementations keep the first even code (Spec 1 §2.1).
#[test]
fn e4m3_negative_zero_midpoint_three_way_tie() {
    let mid = (f64::from(fp8_e4m3_decode(0x80)) + f64::from(fp8_e4m3_decode(0x81))) / 2.0;
    assert_eq!(fp8_e4m3_encode(mid as f32), 0x00);
    assert_eq!(fp8_e4m3_encode_f64_oracle(mid), 0x00);
}

/// Seeded sweep requiring exact production/oracle agreement across the full range (Spec 1 §2.1).
///
/// Covers subnormals, normals, ties, and the saturation zone with a fixed seed, so the test
/// is deterministic; any production regression shows up as a mismatch against the oracle.
#[test]
fn e4m3_production_matches_independent_oracle_on_seeded_sweep() {
    let mut rng = SeededRng::new(0xA1_5075);
    for _ in 0..4096 {
        let raw = (rng.next_u64() & 0xFFFF_FFFF) as u32;
        let v = (f64::from(raw) / f64::from(u32::MAX)) * 1200.0 - 600.0;
        let f = v as f32;
        assert_eq!(
            fp8_e4m3_encode(f),
            fp8_e4m3_encode_f64_oracle(f64::from(f)),
            "production/oracle mismatch at {f}"
        );
    }
    for _ in 0..1024 {
        let raw = (rng.next_u64() & 0xFFFF_FFFF) as u32;
        let v = (f64::from(raw) / f64::from(u32::MAX)) * 0.03 - 0.015;
        let f = v as f32;
        assert_eq!(
            fp8_e4m3_encode(f),
            fp8_e4m3_encode_f64_oracle(f64::from(f)),
            "production/oracle subnormal mismatch at {f}"
        );
    }
}

/// Pins the `quant_act_f64_reference` E4M3 path itself to golden bytes (Spec 1 §4.A).
///
/// With `absmax == 448.0` the scale is exactly 1.0, so each output byte is the oracle
/// encoding of the input value; hard-coded expectations keep the reference honest
/// without involving the production encoder at all.
#[test]
fn quant_act_f64_reference_e4m3_matches_golden_bytes() {
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::E4m3,
        smoothing: Smoothing::None,
    };
    let x = vec![0.0, 1.0, -1.0, 448.0, -448.0, f64::NAN, 0.5, -0.0];
    let (ref_xq, ref_scale) = quant_act_f64_reference(&op, &x, [1, 8]);
    assert_eq!(ref_scale[0], 1.0);
    let expected: Vec<f64> = vec![0x00, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0x30, 0x80]
        .into_iter()
        .map(f64::from)
        .collect();
    assert_eq!(ref_xq, expected);
}
