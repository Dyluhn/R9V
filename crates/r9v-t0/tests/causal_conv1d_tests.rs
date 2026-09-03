// SPDX-License-Identifier: Apache-2.0
//! Tests for scalar T0 `causal_conv1d` (Spec 1 §4.E, Card A1.9).

use r9v_common::SeededRng;
use r9v_ir::{CausalConv1dOp, ConvActivation, DType, Op, StateHandle, StateKind};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::{f16_to_f32, f32_to_f16};
use r9v_t0::error::T0Error;
use r9v_t0::{
    causal_conv1d, causal_conv1d_f64_reference, execute_state_scan_op, SeqLayout, Tolerance,
};

fn conv_op(kernel: u32, act: ConvActivation) -> CausalConv1dOp {
    CausalConv1dOp {
        kernel,
        act,
        handle: StateHandle::new(0, StateKind::ConvWindow),
    }
}

fn next_f32(rng: &mut SeededRng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
    lo + (u as f32) * (hi - lo)
}

fn f16_of(vals: &[f32]) -> Vec<u16> {
    vals.iter().map(|&v| f32_to_f16(v)).collect()
}

fn f64_of_f16(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f16_to_f32(b) as f64).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_conv(
    op: &CausalConv1dOp,
    x: &[f32],
    t: usize,
    c: usize,
    w: &[f32],
    wk: usize,
    bias: Option<&[f32]>,
    state: &[u16],
    s: usize,
    lens: &[u32],
) -> (Vec<f32>, Vec<u16>) {
    let xb = TypedBuffer::from_f32(&[t, c], x);
    let wb = TypedBuffer::from_f32(&[c, wk], w);
    let bb = bias.map(|b| TypedBuffer::from_f32(&[c], b));
    let hist = wk - 1;
    let sb = TypedBuffer::from_f16(&[s, hist, c], state);
    let mut yb = TypedBuffer::zeros(&[t, c], DType::F32);
    let mut sob = TypedBuffer::zeros(&[s, hist, c], DType::F16);
    let seq = SeqLayout::new(lens).unwrap();
    let bias_view = bb.as_ref().map(|b| b.as_view());
    causal_conv1d(
        op,
        &xb.as_view(),
        &wb.as_view(),
        bias_view.as_ref(),
        &sb.as_view(),
        &seq,
        &mut yb.as_view_mut(),
        &mut sob.as_view_mut(),
    )
    .unwrap();
    // Read back f16 state bits exactly.
    let mut state_bits = vec![0u16; s * hist * c];
    for (i, v) in sob.to_f32_vec().iter().enumerate() {
        state_bits[i] = f32_to_f16(*v);
    }
    (yb.to_f32_vec(), state_bits)
}

#[test]
fn split_runs_with_carried_state_match_oneshot_bit_exact() {
    // The done-when continuity test: (T1+T2 with carried state) vs one-shot.
    // Inputs are exactly f16-representable, so the f16 state carry round-trips
    // without loss and the two runs must agree bit-exactly.
    for wk in [1usize, 2, 4, 8] {
        for act in [ConvActivation::Silu, ConvActivation::Identity] {
            for with_bias in [false, true] {
                let mut rng = SeededRng::new(0xC0 + wk as u64 * 2 + with_bias as u64);
                let (t1, t2, c) = (5, 7, 4);
                let t = t1 + t2;
                let exact = |v: f32| f16_to_f32(f32_to_f16(v));
                let x: Vec<f32> = (0..t * c)
                    .map(|_| exact(next_f32(&mut rng, -1.0, 1.0)))
                    .collect();
                let w: Vec<f32> = (0..c * wk)
                    .map(|_| exact(next_f32(&mut rng, -0.5, 0.5)))
                    .collect();
                let bias: Vec<f32> = (0..c)
                    .map(|_| exact(next_f32(&mut rng, -0.25, 0.25)))
                    .collect();
                let hist = wk - 1;
                let init: Vec<f32> = (0..hist * c)
                    .map(|_| exact(next_f32(&mut rng, -1.0, 1.0)))
                    .collect();
                let op = conv_op(wk as u32, act);
                let bias_opt = if with_bias {
                    Some(bias.as_slice())
                } else {
                    None
                };

                let (y_full, _) = run_conv(
                    &op,
                    &x,
                    t,
                    c,
                    &w,
                    wk,
                    bias_opt,
                    &f16_of(&init),
                    1,
                    &[t as u32],
                );

                // Chunked: first T1 from init, then T2 from carried state.
                let (y_a, s_a) = run_conv(
                    &op,
                    &x[..t1 * c],
                    t1,
                    c,
                    &w,
                    wk,
                    bias_opt,
                    &f16_of(&init),
                    1,
                    &[t1 as u32],
                );
                let (y_b, _) = run_conv(
                    &op,
                    &x[t1 * c..],
                    t2,
                    c,
                    &w,
                    wk,
                    bias_opt,
                    &s_a,
                    1,
                    &[t2 as u32],
                );
                let mut y_split = y_a;
                y_split.extend_from_slice(&y_b);

                assert_eq!(y_split, y_full, "wk={wk} act={act:?} bias={with_bias}");
            }
        }
    }
}

#[test]
fn general_values_split_matches_prerounded_oneshot_bit_exact() {
    // With general (non-f16-exact) values the ONLY split-vs-oneshot divergence
    // is the f16 rounding of carry-boundary rows: chunk 1's outputs match the
    // plain one-shot exactly, and chunk 2's outputs match a one-shot whose
    // boundary rows [T1-hist, T1) were pre-rounded to f16. This pins the carry
    // mechanism precisely instead of hiding the rounding in a tolerance.
    let mut rng = SeededRng::new(0xC41);
    let (t1, t2, c, wk) = (5, 7, 4, 4);
    let hist = wk - 1;
    let t = t1 + t2;
    let x: Vec<f32> = (0..t * c).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let w: Vec<f32> = (0..c * wk).map(|_| next_f32(&mut rng, -0.5, 0.5)).collect();
    let init: Vec<f32> = (0..hist * c)
        .map(|_| next_f32(&mut rng, -1.0, 1.0))
        .collect();
    let op = conv_op(wk as u32, ConvActivation::Silu);

    let (y_plain, _) = run_conv(&op, &x, t, c, &w, wk, None, &f16_of(&init), 1, &[t as u32]);
    let (y_a, s_a) = run_conv(
        &op,
        &x[..t1 * c],
        t1,
        c,
        &w,
        wk,
        None,
        &f16_of(&init),
        1,
        &[t1 as u32],
    );
    let (y_b, _) = run_conv(
        &op,
        &x[t1 * c..],
        t2,
        c,
        &w,
        wk,
        None,
        &s_a,
        1,
        &[t2 as u32],
    );

    // Chunk 1 sees no carries in its own computation: exact prefix match.
    assert_eq!(&y_a[..], &y_plain[..t1 * c]);

    // Chunk 2 matches the pre-rounded one-shot on its suffix exactly.
    let mut x_rounded = x.clone();
    for r in (t1 - hist)..t1 {
        for ch in 0..c {
            x_rounded[r * c + ch] = f16_to_f32(f32_to_f16(x[r * c + ch]));
        }
    }
    let (y_pre, _) = run_conv(
        &op,
        &x_rounded,
        t,
        c,
        &w,
        wk,
        None,
        &f16_of(&init),
        1,
        &[t as u32],
    );
    assert_eq!(&y_b[..], &y_pre[t1 * c..]);
}

#[test]
fn general_values_split_continuity_within_f32_tolerance() {
    // General (non-f16-exact) values: split-vs-oneshot is NOT bit-exact, by
    // the specified f16 state quantization of the carry-boundary rows. The
    // divergence is bounded: the split run must match the plain one-shot
    // within the existing named f32 tolerance. No unconditional L0 is
    // claimed; the exact carry mechanism is pinned by the two L0 tests above.
    let mut rng = SeededRng::new(0x6E1);
    let (t1, t2, c, wk) = (5, 7, 4, 4);
    let t = t1 + t2;
    let x: Vec<f32> = (0..t * c).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let w: Vec<f32> = (0..c * wk).map(|_| next_f32(&mut rng, -0.5, 0.5)).collect();
    let bias: Vec<f32> = (0..c).map(|_| next_f32(&mut rng, -0.25, 0.25)).collect();
    let hist = wk - 1;
    let init: Vec<f32> = (0..hist * c)
        .map(|_| next_f32(&mut rng, -1.0, 1.0))
        .collect();
    let op = conv_op(wk as u32, ConvActivation::Silu);

    let (y_full, _) = run_conv(
        &op,
        &x,
        t,
        c,
        &w,
        wk,
        Some(&bias),
        &f16_of(&init),
        1,
        &[t as u32],
    );
    let (y_a, s_a) = run_conv(
        &op,
        &x[..t1 * c],
        t1,
        c,
        &w,
        wk,
        Some(&bias),
        &f16_of(&init),
        1,
        &[t1 as u32],
    );
    let (y_b, _) = run_conv(
        &op,
        &x[t1 * c..],
        t2,
        c,
        &w,
        wk,
        Some(&bias),
        &s_a,
        1,
        &[t2 as u32],
    );
    let mut y_split = y_a;
    y_split.extend_from_slice(&y_b);

    let tol = Tolerance::f32();
    for (i, (&actual, &exp)) in y_split.iter().zip(y_full.iter()).enumerate() {
        tol.assert_within(actual as f64, exp as f64, &format!("split y[{i}]"));
    }
}

#[test]
fn state_tail_matches_last_history_rows_bit_exact() {
    let mut rng = SeededRng::new(0x5A1);
    let (t, c, wk) = (6, 3, 4);
    let hist = wk - 1;
    let x: Vec<f32> = (0..t * c).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let w: Vec<f32> = (0..c * wk).map(|_| next_f32(&mut rng, -0.5, 0.5)).collect();
    let init: Vec<f32> = (0..hist * c)
        .map(|_| next_f32(&mut rng, -1.0, 1.0))
        .collect();
    let op = conv_op(wk as u32, ConvActivation::Identity);
    let (_, s_out) = run_conv(&op, &x, t, c, &w, wk, None, &f16_of(&init), 1, &[t as u32]);
    // Tail is the last `hist` x rows encoded f16 (t >= hist here).
    for h in 0..hist {
        for ch in 0..c {
            let expect = f32_to_f16(x[(t - hist + h) * c + ch]);
            assert_eq!(s_out[h * c + ch], expect, "tail row {h} ch {ch}");
        }
    }
}

#[test]
fn multi_segment_recurrence_resets_per_sequence() {
    // Two segments: the second segment's outputs must equal a lone run of its
    // own tokens from zero state, proving reset at boundaries.
    let mut rng = SeededRng::new(0x9E9);
    let (l0, l1, c, wk) = (4, 5, 3, 3);
    let t = l0 + l1;
    let x: Vec<f32> = (0..t * c).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let w: Vec<f32> = (0..c * wk).map(|_| next_f32(&mut rng, -0.5, 0.5)).collect();
    let op = conv_op(wk as u32, ConvActivation::Silu);
    let zero = vec![0u16; 2 * (wk - 1) * c];
    let (y_both, _) = run_conv(
        &op,
        &x,
        t,
        c,
        &w,
        wk,
        None,
        &zero,
        2,
        &[l0 as u32, l1 as u32],
    );
    let (y_lone, _) = run_conv(
        &op,
        &x[l0 * c..],
        l1,
        c,
        &w,
        wk,
        None,
        &vec![0u16; (wk - 1) * c],
        1,
        &[l1 as u32],
    );
    assert_eq!(&y_both[l0 * c..], y_lone.as_slice());
}

#[test]
fn oneshot_matches_f64_oracle_within_f32_tolerance() {
    let mut rng = SeededRng::new(0xA01);
    let (t, c, wk, s) = (7, 4, 3, 2);
    let lens = [3u32, 4u32];
    let x: Vec<f32> = (0..t * c).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let w: Vec<f32> = (0..c * wk).map(|_| next_f32(&mut rng, -0.5, 0.5)).collect();
    let bias: Vec<f32> = (0..c).map(|_| next_f32(&mut rng, -0.25, 0.25)).collect();
    let hist = wk - 1;
    let init: Vec<f32> = (0..s * hist * c)
        .map(|_| next_f32(&mut rng, -1.0, 1.0))
        .collect();
    let op = conv_op(wk as u32, ConvActivation::Silu);
    let (y, _) = run_conv(&op, &x, t, c, &w, wk, Some(&bias), &f16_of(&init), s, &lens);

    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let w64: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let b64: Vec<f64> = bias.iter().map(|&v| v as f64).collect();
    let (exp_y, _) = causal_conv1d_f64_reference(
        &x64,
        t,
        c,
        &w64,
        wk,
        Some(&b64),
        ConvActivation::Silu,
        &f64_of_f16(&f16_of(&init)),
        s,
        &lens,
    )
    .unwrap();
    let tol = Tolerance::f32();
    for (i, (&actual, &expected)) in y.iter().zip(exp_y.iter()).enumerate() {
        tol.assert_within(actual as f64, expected, &format!("y[{i}]"));
    }
}

#[test]
fn determinism_and_token_batch_invariance() {
    let mut rng = SeededRng::new(0xB01);
    let (t, c, wk) = (5, 3, 4);
    let hist = wk - 1;
    let x: Vec<f32> = (0..t * c).map(|_| next_f32(&mut rng, -1.0, 1.0)).collect();
    let w: Vec<f32> = (0..c * wk).map(|_| next_f32(&mut rng, -0.5, 0.5)).collect();
    let init: Vec<f32> = (0..hist * c)
        .map(|_| next_f32(&mut rng, -1.0, 1.0))
        .collect();
    let op = conv_op(wk as u32, ConvActivation::Silu);
    let (y1, s1) = run_conv(&op, &x, t, c, &w, wk, None, &f16_of(&init), 1, &[t as u32]);
    let (y2, s2) = run_conv(&op, &x, t, c, &w, wk, None, &f16_of(&init), 1, &[t as u32]);
    assert_eq!(y1, y2);
    assert_eq!(s1, s2);
    // Batch invariance across segmentation with f16-exact values: per-token
    // segments with chained state equal the one-shot output bit-exactly
    // (the carry round-trips losslessly on exact inputs).
    let exact = |v: f32| f16_to_f32(f32_to_f16(v));
    let xe: Vec<f32> = x.iter().map(|&v| exact(v)).collect();
    let we: Vec<f32> = w.iter().map(|&v| exact(v)).collect();
    let init_exact: Vec<f32> = init.iter().map(|&v| exact(v)).collect();
    let (ye, _) = run_conv(
        &op,
        &xe,
        t,
        c,
        &we,
        wk,
        None,
        &f16_of(&init_exact),
        1,
        &[t as u32],
    );
    let mut chained_y = Vec::new();
    let mut st = f16_of(&init_exact);
    for row in 0..t {
        let (yr, sr) = run_conv(
            &op,
            &xe[row * c..(row + 1) * c],
            1,
            c,
            &we,
            wk,
            None,
            &st,
            1,
            &[1],
        );
        chained_y.extend_from_slice(&yr);
        st = sr;
    }
    assert_eq!(chained_y, ye);
}

#[test]
fn quantized_weights_fail_closed() {
    let op = conv_op(2, ConvActivation::Identity);
    let xb = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let wb = TypedBuffer::from_bytes(&[2, 2], DType::I8, &[1i8 as u8; 4]);
    let sb = TypedBuffer::from_f16(&[1, 1, 2], &[0; 2]);
    let mut yb = TypedBuffer::zeros(&[2, 2], DType::F32);
    let mut sob = TypedBuffer::zeros(&[1, 1, 2], DType::F16);
    let seq = SeqLayout::new(&[2]).unwrap();
    let err = causal_conv1d(
        &op,
        &xb.as_view(),
        &wb.as_view(),
        None,
        &sb.as_view(),
        &seq,
        &mut yb.as_view_mut(),
        &mut sob.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::QuantMismatch { .. }));
    assert_eq!(yb.to_f32_vec(), vec![0.0; 4]);
}

#[test]
fn dispatch_enforces_conv_arity_and_form() {
    let op = Op::CausalConv1d(conv_op(2, ConvActivation::Identity));
    let xb = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let wb = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let sb = TypedBuffer::from_f16(&[1, 1, 2], &[0; 2]);
    let mut yb = TypedBuffer::zeros(&[2, 2], DType::F32);
    let mut sob = TypedBuffer::zeros(&[1, 1, 2], DType::F16);
    let seq = SeqLayout::new(&[2]).unwrap();
    assert!(execute_state_scan_op(
        &op,
        &[xb.as_view()],
        &sb.as_view(),
        &seq,
        &mut [yb.as_view_mut()],
        &mut sob.as_view_mut(),
        false,
    )
    .is_err());
    assert!(execute_state_scan_op(
        &op,
        &[xb.as_view(), wb.as_view()],
        &sb.as_view(),
        &seq,
        &mut [yb.as_view_mut()],
        &mut sob.as_view_mut(),
        false,
    )
    .is_ok());
}
