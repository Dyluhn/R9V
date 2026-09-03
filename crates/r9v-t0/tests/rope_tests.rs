// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Deterministic property and batch-invariance tests for scalar T0 RoPE (Spec 1 §4.B, §6.1, §6.4, Spec 4 §2).

use r9v_common::rng::SeededRng;
use r9v_ir::{DType, RopeOp, RopeScaling, RopeStyle};
use r9v_t0::{rope, rope_f64_reference, T0Error, Tolerance, TypedBuffer};

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
fn rope_both_styles_and_partial_rotary_match_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5040);
    let tol = Tolerance::f32();

    let (t, h, d) = (3, 4, 64);
    let num_elem = t * h * d;

    let x_data = generate_f32_data(&mut rng, num_elem, 3.0);
    let positions = vec![0u32, 17u32, 1024u32];

    for &style in &[RopeStyle::Neox, RopeStyle::Interleaved] {
        for &rot_dim in &[64u32, 32u32] {
            let op = RopeOp {
                rot_dim,
                theta: 10000.0,
                style,
                scaling: RopeScaling::None,
                mrope_sections: None,
                out_dtype: DType::F32,
            };

            let x_buf = TypedBuffer::from_f32(&[t, h, d], &x_data);
            let pos_buf = TypedBuffer::from_u32(&[t], &positions);
            let mut y_buf = TypedBuffer::zeros(&[t, h, d], DType::F32);

            rope(
                &op,
                &x_buf.as_view(),
                &pos_buf.as_view(),
                &mut y_buf.as_view_mut(),
            )
            .unwrap();

            let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
            let ref_f64 = rope_f64_reference(&op, &x_f64, [t, h, d], &positions, false);

            for i in 0..num_elem {
                let actual = y_buf.read_f32(i) as f64;
                let expected = ref_f64[i];
                tol.assert_within(
                    actual,
                    expected,
                    &format!("rope style={style:?} rot_dim={rot_dim} at {i}"),
                );
            }
        }
    }
}

#[test]
fn rope_all_scalings_match_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5041);
    let tol = Tolerance::f32();

    let (t, h, d) = (2, 2, 64);
    let num_elem = t * h * d;
    let x_data = generate_f32_data(&mut rng, num_elem, 2.5);
    let positions = vec![500u32, 3500u32];

    let scalings = [
        RopeScaling::None,
        RopeScaling::Linear(2.5),
        RopeScaling::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx: 4096,
            mscale: 1.1,
        },
        RopeScaling::Dynamic,
    ];

    for &scaling in &scalings {
        let op = RopeOp {
            rot_dim: 64,
            theta: 10000.0,
            style: RopeStyle::Neox,
            scaling,
            mrope_sections: None,
            out_dtype: DType::F32,
        };

        let x_buf = TypedBuffer::from_f32(&[t, h, d], &x_data);
        let pos_buf = TypedBuffer::from_u32(&[t], &positions);
        let mut y_buf = TypedBuffer::zeros(&[t, h, d], DType::F32);

        rope(
            &op,
            &x_buf.as_view(),
            &pos_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
        let ref_f64 = rope_f64_reference(&op, &x_f64, [t, h, d], &positions, false);

        for i in 0..num_elem {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(
                actual,
                expected,
                &format!("rope scaling={scaling:?} at {i}"),
            );
        }
    }
}

#[test]
fn rope_mrope_matches_f64_reference() {
    let mut rng = SeededRng::new(0xA1_5042);
    let tol = Tolerance::f32();

    let (t, h, d) = (2, 2, 64);
    let num_elem = t * h * d;
    let x_data = generate_f32_data(&mut rng, num_elem, 2.0);

    // Positions [T, 3] for temporal, height, width
    let positions_2d = vec![
        10u32, 25u32, 30u32, // token 0
        10u32, 26u32, 31u32, // token 1
    ];

    let op = RopeOp {
        rot_dim: 64,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: Some([16, 24, 24]), // 16 + 24 + 24 = 64
        out_dtype: DType::F32,
    };

    let x_buf = TypedBuffer::from_f32(&[t, h, d], &x_data);
    let pos_buf = TypedBuffer::from_u32(&[t, 3], &positions_2d);
    let mut y_buf = TypedBuffer::zeros(&[t, h, d], DType::F32);

    rope(
        &op,
        &x_buf.as_view(),
        &pos_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap();

    let x_f64: Vec<f64> = x_data.iter().map(|&v| v as f64).collect();
    let ref_f64 = rope_f64_reference(&op, &x_f64, [t, h, d], &positions_2d, true);

    for i in 0..num_elem {
        let actual = y_buf.read_f32(i) as f64;
        let expected = ref_f64[i];
        tol.assert_within(actual, expected, &format!("mrope at {i}"));
    }
}

#[test]
fn rope_f16_and_bf16_precision_paths() {
    let mut rng = SeededRng::new(0xA1_5043);
    let tol = Tolerance::f16_bf16();
    let (t, h, d) = (2, 2, 32);
    let num_elem = t * h * d;

    let x_data = generate_f32_data(&mut rng, num_elem, 2.0);
    let positions = vec![5u32, 100u32];

    for &dt in &[DType::F16, DType::Bf16] {
        let op = RopeOp {
            rot_dim: 32,
            theta: 10000.0,
            style: RopeStyle::Interleaved,
            scaling: RopeScaling::Linear(2.0),
            mrope_sections: None,
            out_dtype: dt,
        };

        let mut x_buf = TypedBuffer::zeros(&[t, h, d], dt);
        for i in 0..num_elem {
            x_buf.write_f32(i, x_data[i]);
        }
        let pos_buf = TypedBuffer::from_u32(&[t], &positions);
        let mut y_buf = TypedBuffer::zeros(&[t, h, d], dt);

        rope(
            &op,
            &x_buf.as_view(),
            &pos_buf.as_view(),
            &mut y_buf.as_view_mut(),
        )
        .unwrap();

        let x_f64: Vec<f64> = (0..num_elem).map(|i| x_buf.read_f32(i) as f64).collect();
        let ref_f64 = rope_f64_reference(&op, &x_f64, [t, h, d], &positions, false);

        for i in 0..num_elem {
            let actual = y_buf.read_f32(i) as f64;
            let expected = ref_f64[i];
            tol.assert_within(actual, expected, &format!("rope_{dt} at {i}"));
        }
    }
}

#[test]
fn rope_batch_invariance() {
    let mut rng = SeededRng::new(0xA1_5044);
    let (h, d) = (2, 64);
    let token_elems = h * d;

    let target_token_data = generate_f32_data(&mut rng, token_elems, 3.0);
    let target_pos = 42u32;

    let op = RopeOp {
        rot_dim: 48,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::Yarn {
            factor: 2.0,
            beta_fast: 16.0,
            beta_slow: 1.0,
            orig_ctx: 2048,
            mscale: 1.05,
        },
        mrope_sections: None,
        out_dtype: DType::F32,
    };

    // 1. Alone (T = 1)
    let x_alone = TypedBuffer::from_f32(&[1, h, d], &target_token_data);
    let pos_alone = TypedBuffer::from_u32(&[1], &[target_pos]);
    let mut y_alone = TypedBuffer::zeros(&[1, h, d], DType::F32);
    rope(
        &op,
        &x_alone.as_view(),
        &pos_alone.as_view(),
        &mut y_alone.as_view_mut(),
    )
    .unwrap();
    let out_alone = y_alone.to_f32_vec();

    // 2. Padded (T = 4)
    let mut x_pad = target_token_data.clone();
    x_pad.extend(vec![0.0f32; 3 * token_elems]);
    let x_pad_buf = TypedBuffer::from_f32(&[4, h, d], &x_pad);
    let pos_pad_buf = TypedBuffer::from_u32(&[4], &[target_pos, 0, 0, 0]);
    let mut y_padded = TypedBuffer::zeros(&[4, h, d], DType::F32);
    rope(
        &op,
        &x_pad_buf.as_view(),
        &pos_pad_buf.as_view(),
        &mut y_padded.as_view_mut(),
    )
    .unwrap();
    let out_padded = &y_padded.to_f32_vec()[..token_elems];

    // 3. Embedded (T = 4, target token at index 2)
    let other_tokens = generate_f32_data(&mut rng, 3 * token_elems, 4.0);
    let mut x_emb = Vec::with_capacity(4 * token_elems);
    x_emb.extend_from_slice(&other_tokens[..2 * token_elems]);
    x_emb.extend_from_slice(&target_token_data);
    x_emb.extend_from_slice(&other_tokens[2 * token_elems..]);
    let x_emb_buf = TypedBuffer::from_f32(&[4, h, d], &x_emb);
    let pos_emb_buf = TypedBuffer::from_u32(&[4], &[100, 200, target_pos, 400]);
    let mut y_emb = TypedBuffer::zeros(&[4, h, d], DType::F32);
    rope(
        &op,
        &x_emb_buf.as_view(),
        &pos_emb_buf.as_view(),
        &mut y_emb.as_view_mut(),
    )
    .unwrap();
    let out_emb = &y_emb.to_f32_vec()[2 * token_elems..3 * token_elems];

    for i in 0..token_elems {
        assert_eq!(
            out_alone[i].to_bits(),
            out_padded[i].to_bits(),
            "batch invariance alone vs padded at {i}"
        );
        assert_eq!(
            out_alone[i].to_bits(),
            out_emb[i].to_bits(),
            "batch invariance alone vs embedded at {i}"
        );
    }
}

#[test]
fn rope_determinism_twice_bit_identical() {
    let mut rng = SeededRng::new(0xA1_5045);
    let (t, h, d) = (3, 2, 64);
    let num_elem = t * h * d;

    let op = RopeOp {
        rot_dim: 64,
        theta: 10000.0,
        style: RopeStyle::Interleaved,
        scaling: RopeScaling::Dynamic,
        mrope_sections: None,
        out_dtype: DType::F32,
    };

    let x_data = generate_f32_data(&mut rng, num_elem, 3.0);
    let positions = vec![10u32, 200u32, 3000u32];

    let x_buf = TypedBuffer::from_f32(&[t, h, d], &x_data);
    let pos_buf = TypedBuffer::from_u32(&[t], &positions);

    let mut y1 = TypedBuffer::zeros(&[t, h, d], DType::F32);
    let mut y2 = TypedBuffer::zeros(&[t, h, d], DType::F32);

    rope(
        &op,
        &x_buf.as_view(),
        &pos_buf.as_view(),
        &mut y1.as_view_mut(),
    )
    .unwrap();
    rope(
        &op,
        &x_buf.as_view(),
        &pos_buf.as_view(),
        &mut y2.as_view_mut(),
    )
    .unwrap();

    let out1 = y1.to_f32_vec();
    let out2 = y2.to_f32_vec();

    for i in 0..num_elem {
        assert_eq!(
            out1[i].to_bits(),
            out2[i].to_bits(),
            "determinism failed at {i}"
        );
    }
}

#[test]
fn rope_rejects_odd_rot_dim_and_dimension_mismatch() {
    let op = RopeOp {
        rot_dim: 33, // odd rot_dim invalid
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F32,
    };

    let x_buf = TypedBuffer::zeros(&[2, 2, 32], DType::F32); // rot_dim 33 > head dim 32
    let pos_buf = TypedBuffer::zeros(&[3], DType::U32); // T=3 != T=2
    let mut y_buf = TypedBuffer::zeros(&[2, 2, 32], DType::F32);

    let err = rope(
        &op,
        &x_buf.as_view(),
        &pos_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    let T0Error::Multiple { problems } = err else {
        panic!("expected aggregated Multiple, got {err:?}");
    };
    assert_eq!(problems.len(), 3);
    assert!(
        problems.iter().any(|e| matches!(
            e,
            T0Error::InvalidAttribute {
                attribute: "rot_dim",
                ..
            }
        )),
        "missing rot_dim problem: {problems:?}"
    );
    assert!(
        problems.iter().any(|e| matches!(
            e,
            T0Error::DimensionMismatch {
                tensor: "positions",
                dim_name: "T",
                ..
            }
        )),
        "missing positions problem: {problems:?}"
    );
}

#[test]
fn rope_rejects_rank_zero_positions_with_typed_error() {
    let op = RopeOp {
        rot_dim: 16,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F32,
    };

    let x_buf = TypedBuffer::zeros(&[2, 2, 16], DType::F32);
    let pos_buf = TypedBuffer::zeros(&[], DType::U32); // rank 0
    let mut y_buf = TypedBuffer::zeros(&[2, 2, 16], DType::F32);

    let err = rope(
        &op,
        &x_buf.as_view(),
        &pos_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();

    match err {
        r9v_t0::T0Error::RankMismatch {
            tensor,
            expected,
            got,
            ..
        } => {
            assert_eq!(tensor, "positions");
            assert_eq!(expected, 1);
            assert_eq!(got, 0);
        }
        other => panic!("expected RankMismatch error, got {other:?}"),
    }
}

#[test]
fn rope_rejects_dynamic_scaling_rot_dim_2() {
    let op = RopeOp {
        rot_dim: 2,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::Dynamic,
        mrope_sections: None,
        out_dtype: DType::F32,
    };

    let x_buf = TypedBuffer::zeros(&[2, 2, 16], DType::F32);
    let pos_buf = TypedBuffer::zeros(&[2], DType::U32);
    let mut y_buf = TypedBuffer::zeros(&[2, 2, 16], DType::F32);

    let err = rope(
        &op,
        &x_buf.as_view(),
        &pos_buf.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();

    let msg = err.to_string();
    assert!(msg.contains("rot_dim 2 is invalid for Dynamic RoPE"));
}
