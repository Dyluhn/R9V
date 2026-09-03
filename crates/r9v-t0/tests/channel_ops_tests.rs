// SPDX-License-Identifier: Apache-2.0
//! Deterministic tests for scalar T0 split/concat/logit_softcap against their
//! f64 references plus malformed-input rejection (card A1.14, SI-19, SI-20).

use r9v_common::rng::SeededRng;
use r9v_ir::{ConcatOp, DType, LogitSoftcapOp, SplitOp};
use r9v_t0::{
    concat, concat_f64_reference, logit_softcap, logit_softcap_f64_reference, split,
    split_f64_reference, Tolerance, TypedBuffer,
};

fn f32_data(rng: &mut SeededRng, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let raw = (rng.next_u64() & 0xFFFF_FFFF) as u32;
        out.push((raw as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

#[test]
fn split_matches_f64_reference_and_roundtrips_through_concat() {
    let mut rng = SeededRng::new(0xA1_1401);
    let tol = Tolerance::f32();
    let (t, h, d) = (2usize, 3usize, 8usize);
    for first in [1usize, 3, 7] {
        let op = SplitOp {
            first: first as u32,
        };
        let data = f32_data(&mut rng, t * h * d);
        let x_buf = TypedBuffer::from_f32(&[t, h, d], &data);
        let mut a_buf = TypedBuffer::zeros(&[t, h, first], DType::F32);
        let mut b_buf = TypedBuffer::zeros(&[t, h, d - first], DType::F32);
        split(
            &op,
            &x_buf.as_view(),
            &mut a_buf.as_view_mut(),
            &mut b_buf.as_view_mut(),
        )
        .unwrap();

        let x_f64: Vec<f64> = data.iter().map(|&v| v as f64).collect();
        let (exp_a, exp_b) = split_f64_reference(&x_f64, [t, h, d], first);
        for (i, e) in exp_a.iter().enumerate() {
            tol.assert_within(a_buf.read_f32(i) as f64, *e, &format!("split a[{i}]"));
        }
        for (i, e) in exp_b.iter().enumerate() {
            tol.assert_within(b_buf.read_f32(i) as f64, *e, &format!("split b[{i}]"));
        }

        // Concat reconstructs the input bit-for-bit.
        let mut y_buf = TypedBuffer::zeros(&[t, h, d], DType::F32);
        concat(
            &ConcatOp,
            &a_buf.as_view(),
            &b_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();
        for (i, v) in data.iter().enumerate() {
            assert_eq!(
                y_buf.read_f32(i).to_bits(),
                v.to_bits(),
                "split->concat roundtrip at {i}"
            );
        }
    }
}

#[test]
fn concat_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_1402);
    let tol = Tolerance::f32();
    let (t, h, da, db) = (2usize, 2usize, 5usize, 3usize);
    let a_data = f32_data(&mut rng, t * h * da);
    let b_data = f32_data(&mut rng, t * h * db);
    let a_buf = TypedBuffer::from_f32(&[t, h, da], &a_data);
    let b_buf = TypedBuffer::from_f32(&[t, h, db], &b_data);
    let mut y_buf = TypedBuffer::zeros(&[t, h, da + db], DType::F32);
    concat(
        &ConcatOp,
        &a_buf.as_view(),
        &b_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let a_f64: Vec<f64> = a_data.iter().map(|&v| v as f64).collect();
    let b_f64: Vec<f64> = b_data.iter().map(|&v| v as f64).collect();
    let expected = concat_f64_reference(&a_f64, &b_f64, t, h);
    for (i, e) in expected.iter().enumerate() {
        tol.assert_within(y_buf.read_f32(i) as f64, *e, &format!("concat y[{i}]"));
    }
}

#[test]
fn split_and_concat_reject_malformed_operands() {
    let x_buf = TypedBuffer::zeros(&[2, 2, 8], DType::F32);
    let mut a_buf = TypedBuffer::zeros(&[2, 2, 3], DType::F32);
    let mut b_buf = TypedBuffer::zeros(&[2, 2, 5], DType::F32);

    // Zero or full width.
    for first in [0u32, 8] {
        let err = split(
            &SplitOp { first },
            &x_buf.as_view(),
            &mut a_buf.as_view_mut(),
            &mut b_buf.as_view_mut(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("first"),
            "split first={first} must be rejected: {err}"
        );
    }
    // Wrong output widths.
    let mut narrow = TypedBuffer::zeros(&[2, 2, 4], DType::F32);
    let err = split(
        &SplitOp { first: 3 },
        &x_buf.as_view(),
        &mut a_buf.as_view_mut(),
        &mut narrow.as_view_mut(),
    )
    .unwrap_err();
    assert!(!err.to_string().is_empty(), "width mismatch rejected");

    // Concat with mismatched [T, H].
    let c_buf = TypedBuffer::zeros(&[2, 3, 5], DType::F32);
    let mut y_buf = TypedBuffer::zeros(&[2, 2, 8], DType::F32);
    let err = concat(
        &ConcatOp,
        &a_buf.as_view(),
        &c_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(!err.to_string().is_empty(), "head mismatch rejected");

    // Concat with wrong output width.
    let mut short = TypedBuffer::zeros(&[2, 2, 7], DType::F32);
    let err = concat(
        &ConcatOp,
        &a_buf.as_view(),
        &b_buf.as_view(),
        &mut short.as_view_mut(),
    )
    .unwrap_err();
    assert!(
        !err.to_string().is_empty(),
        "output width mismatch rejected"
    );
}

#[test]
fn logit_softcap_matches_reference_and_bounds_output() {
    let mut rng = SeededRng::new(0xA1_1403);
    let tol = Tolerance::f32();
    let cap = 30.0f32;
    let data: Vec<f32> = f32_data(&mut rng, 96).iter().map(|v| v * 100.0).collect();
    let x_buf = TypedBuffer::from_f32(&[4, 24], &data);
    let mut y_buf = TypedBuffer::zeros(&[4, 24], DType::F32);
    logit_softcap(
        &LogitSoftcapOp { cap },
        &x_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let expected = logit_softcap_f64_reference(&x_f64, cap as f64);
    for (i, e) in expected.iter().enumerate() {
        let actual = y_buf.read_f32(i) as f64;
        tol.assert_within(actual, *e, &format!("softcap at {i}"));
        assert!(
            actual.abs() <= cap as f64 + 1e-4,
            "softcap output bounded by cap at {i}: {actual}"
        );
    }
    // Near zero the softcap is the identity (tanh(x/c) ≈ x/c).
    let tiny = vec![0.01f32, -0.02, 0.0];
    let x_tiny = TypedBuffer::from_f32(&[1, 3], &tiny);
    let mut y_tiny = TypedBuffer::zeros(&[1, 3], DType::F32);
    logit_softcap(
        &LogitSoftcapOp { cap },
        &x_tiny.as_view(),
        &mut y_tiny.as_view_mut(),
    )
    .unwrap();
    for (i, v) in tiny.iter().enumerate() {
        assert!(
            (y_tiny.read_f32(i) - v).abs() < 1e-6,
            "softcap near-identity at {i}"
        );
    }
}

#[test]
fn logit_softcap_rejects_bad_cap_and_dtype() {
    let x_buf = TypedBuffer::zeros(&[2, 8], DType::F32);
    let mut y_buf = TypedBuffer::zeros(&[2, 8], DType::F32);
    for cap in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let err = logit_softcap(
            &LogitSoftcapOp { cap },
            &x_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("cap"),
            "cap {cap} must be rejected: {err}"
        );
    }
    let x16 = TypedBuffer::zeros(&[2, 8], DType::F16);
    let err = logit_softcap(
        &LogitSoftcapOp { cap: 30.0 },
        &x16.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(!err.to_string().is_empty(), "non-f32 input rejected");
}
