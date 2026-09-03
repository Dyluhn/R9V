// SPDX-License-Identifier: Apache-2.0
//! Hand-computed byte summary tests for `ModelSummary` (Spec 8 §7; Spec 3 §6.2; card A1.3).

use r9v_ir::op::{
    ActivationKind, HashId, LinearAttnKind, MoeScoring, NgramCombine, RopeScaling, RopeStyle,
};
use r9v_ir::version::IrVersion;
use r9v_models::{
    build_model, CacheDtype, Ffn, Graph, LayerSpec, Mixer, MixerKind, ModelSpec, MoeSharedSpec,
    NgramSpec, NormPlacement, NormSpec, PositionEncoding, RopeSpec, SchemeKey,
};

#[test]
fn test_synthetic_model_byte_summary() {
    let rope = RopeSpec {
        theta: 10000.0,
        rot_dim: 32,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    };

    // Layer 0: Attention (KV cache E4m3) + Gated Dense FFN
    let layer0 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 32,
            dv: 32,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope,
            window: Some(1024),
            sinks: 4,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
            pre_fused: false,
        },
        ffn: Ffn::Dense {
            dff: 512,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
            pre_fused: false,
        },
        residual_scale: 1.0,
    };

    // Layer 1: MoE (4 experts, top-2, shared expert)
    let layer1 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::None,
        ffn: Ffn::Moe {
            e: 4,
            k: 2,
            dff_e: 128,
            act: ActivationKind::Silu,
            scoring: MoeScoring::Softmax,
            renormalize: true,
            group: None,
            route_bias: false,
            route_scale: 1.0,
            shared: Some(MoeSharedSpec { n: 1, dff: 128 }),
            shared_gate: false,
        },
        residual_scale: 1.0,
    };

    // Layer 2: Linear Attention (conv + recurrent scan)
    let layer2 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::LinearAttention {
            kind: LinearAttnKind::GatedDeltaNet,
            h: 4,
            d: 16,
            dv: 16,
            conv: Some(4),
            gate_act: ActivationKind::Silu,
            output_norm: None,
            output_gate: false,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    let model = ModelSpec {
        dm: 256,
        layers: vec![layer0, layer1, layer2],
        vocab: 1000,
        embed_scale: 1.0,
        tied_embeddings: false,
        final_norm: NormSpec::rms(1e-5),
        final_logit_softcap: None,
        positions: PositionEncoding::Scalar,
        ngram: Some(NgramSpec {
            orders: vec![2, 3],
            heads: 2,
            dim: 32,
            table_sizes: vec![64, 64],
            hash: HashId::new(1),
            combine: NgramCombine::Sum,
            inject_at: 0,
        }),
        mtp: None,
        export_hidden: true,
        eos_ids: vec![2],
        bos_id: Some(1),
    };

    let builder = Graph::new(IrVersion::CURRENT, "synthetic-byte-model");
    let model_graph = build_model(builder, &model).expect("model must build");
    let summary = model_graph.summary().expect("summary must compute");

    // 1. Hand-computed global dimensions and flags
    assert_eq!(summary.vocab, 1000);
    assert_eq!(summary.dm, 256);
    assert_eq!(summary.hkv, 4);
    assert_eq!(summary.tp_divisors, vec![1, 2, 4]);
    assert!(!summary.mtp);
    assert!(summary.export_hidden);

    // 2. Embed and Head bytes: vocab * dm * sizeof(F16) = 1000 * 256 * 2 = 512,000 bytes
    assert_eq!(summary.embed_bytes, 512_000);
    assert_eq!(summary.head_bytes, 512_000);

    // 3. N-gram table: 128 entries * 32 dim * sizeof(F16) = 8,192 bytes
    assert_eq!(summary.ngram_table_bytes, 8_192);

    // 4. Layer summaries
    assert_eq!(summary.layers.len(), 3);

    // Layer 0 assertions:
    let l0 = &summary.layers[0];
    assert_eq!(l0.mixer_kind, Some(MixerKind::Attention));
    // state_per_token_bytes: hkv * ((d + dv) * 1 + 4) = 4 * ((32 + 32) * 1 + 4) = 4 * 68 = 272 bytes
    assert_eq!(l0.state_per_token_bytes, 272);
    assert_eq!(l0.state_per_seq_bytes, 0);
    assert_eq!(l0.experts, None);

    // Layer 1 assertions (MoE):
    let l1 = &summary.layers[1];
    assert_eq!(l1.mixer_kind, None);
    assert_eq!(l1.state_per_token_bytes, 0);
    assert_eq!(l1.state_per_seq_bytes, 0);
    let exp = l1
        .experts
        .as_ref()
        .expect("layer 1 must have expert summary");
    assert_eq!(exp.e, 4);
    // bytes_each = gate_up (256 * 256 * 2) + down (256 * 128 * 2) = 131,072 + 65,536 = 196,608 bytes
    assert_eq!(exp.bytes_each, 196_608);

    // Layer 2 assertions (Linear Attention):
    let l2 = &summary.layers[2];
    assert_eq!(l2.mixer_kind, Some(MixerKind::LinearAttention));
    assert_eq!(l2.state_per_token_bytes, 0);
    // state_per_seq_bytes: conv ((4-1)*256*2*2 = 3,072) + recurrent (4 * 16 * 16 * 4 * 2 = 8,192) = 11,264 bytes
    assert_eq!(l2.state_per_seq_bytes, 11_264);

    // 5. Total model state memory budgets
    assert_eq!(
        summary.total_state_per_token_bytes().expect("state totals"),
        272
    );
    assert_eq!(
        summary.total_state_per_seq_bytes().expect("state totals"),
        11_264
    );

    // 6. Total weight bytes is > 0 and bucketed schemes are valid
    let total_weights = summary.total_weight_bytes().expect("weight total");
    assert!(total_weights > 0, "total weight bytes must be positive");
    for (scheme, bytes) in &summary.layers[0].weight_bytes_by_scheme {
        assert!(
            matches!(scheme, SchemeKey::None),
            "unquantized weights belong to SchemeKey::None"
        );
        assert!(*bytes > 0);
    }
}
