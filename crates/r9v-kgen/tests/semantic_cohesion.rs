// SPDX-License-Identifier: Apache-2.0
//! Semantic cohesion tests for the A3.API typed descriptor contract (Spec 4 §3, §7).
//!
//! Root decision: every compile-time kernel semantic is a closed typed descriptor field
//! included in `static_hash`. No family guessing, no opaque strings, no collisions.

mod common;

use r9v_ir::AttentionMask;
use r9v_kgen::abi::{
    abi, abi_for_op, canonical_struct_name, emit_hip_assume_aligned, emit_hip_struct,
    op_static_family, AbiStruct, AbiType, BatchMetaField, FieldRole, PointeeType,
};
use r9v_kgen::error::KgenError;
use r9v_registry::{static_hash, OpId, OpStatic};

/// Every representative static constructs its ABI for every one of the 32 ops.
#[test]
fn test_all_32_ops_construct_canonical_abis() {
    for op in common::ALL_32_OPS {
        let stat = common::representative_static_for_op(op);
        let built = abi_for_op(op, &stat).expect("representative ABI must construct");
        assert_eq!(built.op(), op, "built ABI must carry its op");
        assert_eq!(
            built.name(),
            canonical_struct_name(op, &stat),
            "built ABI must carry the canonical name"
        );
    }

    // Exact 1:1 families also dispatch without an explicit op.
    for stat in [
        common::representative_matmul_static(),
        common::representative_moe_route_static(),
        common::representative_moe_ffn_static(),
        common::representative_attention_static(AttentionMask::Causal, 4),
        common::representative_state_write_kv_static(),
        common::representative_causal_conv1d_static(),
        common::representative_linear_attn_scan_static(),
    ] {
        abi(&stat).expect("exact family must dispatch directly");
    }
}

/// One semantic field flip at a time must change `static_hash`.
#[test]
fn test_each_compile_time_semantic_changes_static_hash() {
    // Matmul transpose_w, activation dtype, and weight dtype.
    let base = common::representative_matmul_static();
    let mut flipped = match base.clone() {
        OpStatic::Matmul(s) => s,
        _ => panic!("expected matmul"),
    };
    flipped.transpose_w = true;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Matmul(flipped.clone())),
        "transpose_w must change static_hash"
    );
    flipped.transpose_w = false;
    flipped.in_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Matmul(flipped.clone())),
        "activation in_dtype must change static_hash"
    );
    flipped.in_dtype = r9v_ir::DType::F16;
    flipped.w_dtype = r9v_ir::DType::I8;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Matmul(flipped.clone())),
        "weight w_dtype must change static_hash"
    );
    flipped.w_dtype = r9v_ir::DType::F16;
    flipped.epilogue = r9v_ir::Epilogue::Residual;
    flipped.residual_dtype = Some(r9v_ir::DType::F16);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Matmul(flipped.clone())),
        "residual epilogue input must change static_hash"
    );
    flipped.residual_dtype = Some(r9v_ir::DType::Bf16);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Matmul(flipped)),
        "residual input dtype must change static_hash"
    );

    // MoeRoute scoring, scale bits, group, and bias presence.
    let base = common::representative_moe_route_static();
    let mut flipped = match base.clone() {
        OpStatic::MoeRoute(s) => s,
        _ => panic!("expected moe_route"),
    };
    flipped.scoring = r9v_ir::MoeScoring::Sigmoid;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeRoute(flipped.clone())),
        "scoring must change static_hash"
    );
    flipped.scoring = r9v_ir::MoeScoring::Softmax;
    flipped.set_scale(2.0);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeRoute(flipped.clone())),
        "scale bits must change static_hash"
    );
    flipped.set_scale(1.0);
    flipped.group = Some(r9v_ir::MoeGroup {
        n_group: 2,
        topk_group: 1,
    });
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeRoute(flipped.clone())),
        "group must change static_hash"
    );
    flipped.group = None;
    flipped.has_bias = true;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeRoute(flipped)),
        "bias presence must change static_hash"
    );

    // MoeFfn act, dtypes, and shared experts.
    let base = common::representative_moe_ffn_static();
    let mut flipped = match base.clone() {
        OpStatic::MoeFfn(s) => s,
        _ => panic!("expected moe_ffn"),
    };
    flipped.act = r9v_ir::ActivationKind::Gelu;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeFfn(flipped.clone())),
        "moe act must change static_hash"
    );
    flipped.act = r9v_ir::ActivationKind::Silu;
    flipped.out_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeFfn(flipped.clone())),
        "moe out_dtype must change static_hash"
    );
    flipped.out_dtype = r9v_ir::DType::F16;
    flipped.shared_experts = 2;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeFfn(flipped.clone())),
        "shared_experts must change static_hash"
    );
    flipped.shared_experts = 0;
    flipped.in_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeFfn(flipped.clone())),
        "moe in_dtype must change static_hash"
    );
    flipped.in_dtype = r9v_ir::DType::F16;
    flipped.gate_up.layout = r9v_ir::LayoutId::L0;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeFfn(flipped.clone())),
        "moe gate_up layout must change static_hash"
    );
    flipped.gate_up.layout = r9v_ir::LayoutId::L1;
    flipped.down.dtype = r9v_ir::DType::I8;
    flipped.down.scheme = r9v_ir::QuantScheme::PerRow;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeFfn(flipped)),
        "moe down dtype/scheme must change static_hash"
    );

    // Attention softmax scale, out dtype, sinks, and MLA descriptor.
    let base = common::representative_attention_static(AttentionMask::Causal, 16);
    let mut flipped = match base.clone() {
        OpStatic::Attention(s) => s,
        _ => panic!("expected attention"),
    };
    flipped.set_softmax_scale(0.5);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Attention(flipped.clone())),
        "softmax scale bits must change static_hash"
    );
    flipped.set_softmax_scale(flipped.softmax_scale());
    flipped.out_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Attention(flipped.clone())),
        "attention out_dtype must change static_hash"
    );
    flipped.out_dtype = r9v_ir::DType::F16;
    flipped.sinks = 4;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Attention(flipped.clone())),
        "sinks must change static_hash"
    );
    flipped.sinks = 0;
    flipped.mla = Some(r9v_registry::MlaAttentionStatic {
        q_lora_rank: Some(64),
        kv_lora_rank: 128,
        qk_nope_dim: 64,
        qk_rope_dim: 64,
        v_dim: 128,
    });
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Attention(flipped.clone())),
        "mla descriptor must change static_hash"
    );
    flipped.mla = None;
    flipped.q_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Attention(flipped)),
        "attention q_dtype must change static_hash"
    );

    // StateWriteKv granularity, latent, and input dtype.
    let base = common::representative_state_write_kv_static();
    let mut flipped = match base.clone() {
        OpStatic::StateWriteKv(s) => s,
        _ => panic!("expected state_write_kv"),
    };
    flipped.scale_granularity = r9v_ir::CacheScaleGranularity::PerBlock;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::StateWriteKv(flipped.clone())),
        "scale granularity must change static_hash"
    );
    flipped.scale_granularity = r9v_ir::CacheScaleGranularity::PerTokenHead;
    flipped.latent = Some(r9v_registry::MlaLatentStatic {
        kv_lora_rank: 128,
        rope_dim: 64,
    });
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::StateWriteKv(flipped.clone())),
        "mla latent must change static_hash"
    );
    flipped.latent = None;
    flipped.in_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::StateWriteKv(flipped)),
        "state_write in_dtype must change static_hash"
    );

    // CausalConv1d bucket and per-tensor dtypes.
    let base = common::representative_causal_conv1d_static();
    let mut flipped = match base.clone() {
        OpStatic::CausalConv1d(s) => s,
        _ => panic!("expected causal_conv1d"),
    };
    flipped.act = r9v_ir::ConvActivation::Identity;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped.clone())),
        "conv activation must change static_hash"
    );
    flipped.act = r9v_ir::ConvActivation::Silu;
    flipped.t_bucket = 32;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped.clone())),
        "conv t_bucket must change static_hash"
    );
    flipped.t_bucket = 16;
    flipped.x_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped.clone())),
        "conv x_dtype must change static_hash"
    );
    flipped.x_dtype = r9v_ir::DType::F16;
    flipped.out_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped.clone())),
        "conv out_dtype must change static_hash"
    );
    flipped.out_dtype = r9v_ir::DType::F16;
    flipped.w_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped.clone())),
        "conv w_dtype must change static_hash"
    );
    flipped.w_dtype = r9v_ir::DType::F16;
    flipped.w_layout = r9v_ir::LayoutId::L0;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped.clone())),
        "conv w_layout must change static_hash"
    );
    flipped.w_layout = r9v_ir::LayoutId::L1;
    flipped.w_dtype = r9v_ir::DType::I8;
    flipped.w_scheme = r9v_ir::QuantScheme::PerRow;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped.clone())),
        "conv w_scheme must change static_hash"
    );
    flipped.w_dtype = r9v_ir::DType::F16;
    flipped.w_scheme = r9v_ir::QuantScheme::None;
    flipped.bias_dtype = Some(r9v_ir::DType::F16);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::CausalConv1d(flipped)),
        "conv bias dtype must change static_hash"
    );

    // LinearAttnScan dtypes.
    let base = common::representative_linear_attn_scan_static();
    let mut flipped = match base.clone() {
        OpStatic::LinearAttnScan(s) => s,
        _ => panic!("expected linear_attn_scan"),
    };
    flipped.out_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::LinearAttnScan(flipped.clone())),
        "scan out_dtype must change static_hash"
    );
    flipped.out_dtype = r9v_ir::DType::F16;
    flipped.in_dtype = r9v_ir::DType::Bf16;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::LinearAttnScan(flipped)),
        "scan in_dtype must change static_hash"
    );

    // Norm epsilon bits inside elementwise params.
    let base = common::representative_elementwise_static();
    let mut flipped = match base.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    let mut norm = match flipped.op_params.clone() {
        r9v_registry::ElementwiseParams::Norm(n) => n,
        _ => panic!("expected norm"),
    };
    norm.set_eps(1e-6);
    flipped.op_params = r9v_registry::ElementwiseParams::Norm(norm);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped)),
        "norm eps bits must change static_hash"
    );

    // ResidualAdd scale bits.
    let base = common::representative_residual_add_static();
    let mut flipped = match base.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    let mut add = match flipped.op_params.clone() {
        r9v_registry::ElementwiseParams::ResidualAdd(a) => a,
        _ => panic!("expected residual_add"),
    };
    add.set_scale(0.5);
    flipped.op_params = r9v_registry::ElementwiseParams::ResidualAdd(add.clone());
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped.clone())),
        "residual scale bits must change static_hash"
    );
    add.a_dtype = r9v_ir::DType::Bf16;
    flipped.op_params = r9v_registry::ElementwiseParams::ResidualAdd(add.clone());
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped.clone())),
        "residual a_dtype must change static_hash"
    );
    add.a_dtype = r9v_ir::DType::F16;
    add.b_dtype = r9v_ir::DType::F32;
    flipped.op_params = r9v_registry::ElementwiseParams::ResidualAdd(add);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped)),
        "residual b_dtype must change static_hash"
    );

    // Norm bias presence.
    let base = common::representative_elementwise_static();
    let mut flipped = match base.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    let mut norm = match flipped.op_params.clone() {
        r9v_registry::ElementwiseParams::Norm(n) => n,
        _ => panic!("expected norm"),
    };
    norm.has_bias = true;
    flipped.op_params = r9v_registry::ElementwiseParams::Norm(norm);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped)),
        "norm bias presence must change static_hash"
    );

    // ScatterAddRows dest presence.
    let base = common::representative_static_for_op(OpId::ScatterAddRows);
    let mut flipped = match base.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    let mut scatter = match flipped.op_params.clone() {
        r9v_registry::ElementwiseParams::ScatterAddRows(p) => p,
        _ => panic!("expected scatter_add_rows"),
    };
    scatter.has_dest = true;
    flipped.op_params = r9v_registry::ElementwiseParams::ScatterAddRows(scatter);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped)),
        "scatter dest presence must change static_hash"
    );

    // Elementwise op variant itself is a semantic.
    let norm_st = common::representative_elementwise_static();
    let residual = common::representative_residual_add_static();
    assert_ne!(
        static_hash(&norm_st),
        static_hash(&residual),
        "elementwise op variant must change static_hash"
    );

    // Ngram source mode and Dn.
    let staged = common::representative_static_for_op(OpId::NgramGather);
    let mut device = match staged.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    let mut params = match device.op_params.clone() {
        r9v_registry::ElementwiseParams::NgramGather(p) => p,
        _ => panic!("expected ngram"),
    };
    params.source = r9v_ir::NgramSource::Device;
    device.op_params = r9v_registry::ElementwiseParams::NgramGather(params.clone());
    assert_ne!(
        static_hash(&staged),
        static_hash(&OpStatic::Elementwise(device.clone())),
        "ngram source must change static_hash"
    );
    params.dn = 256;
    device.op_params = r9v_registry::ElementwiseParams::NgramGather(params.clone());
    assert_ne!(
        static_hash(&staged),
        static_hash(&OpStatic::Elementwise(device.clone())),
        "ngram dn must change static_hash"
    );
    params.dn = 128;
    params.scales_dtype = Some(r9v_ir::DType::F16);
    device.op_params = r9v_registry::ElementwiseParams::NgramGather(params.clone());
    assert_ne!(
        static_hash(&staged),
        static_hash(&OpStatic::Elementwise(device.clone())),
        "ngram scales dtype must change static_hash"
    );
    params.scales_dtype = Some(r9v_ir::DType::F32);
    params.staging_layout = r9v_ir::LayoutId::L1;
    device.op_params = r9v_registry::ElementwiseParams::NgramGather(params);
    assert_ne!(
        static_hash(&staged),
        static_hash(&OpStatic::Elementwise(device)),
        "ngram staging layout must change static_hash"
    );

    // Split first and total widths.
    let base = common::representative_static_for_op(OpId::Split);
    let mut flipped = match base.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    let mut split = match flipped.op_params.clone() {
        r9v_registry::ElementwiseParams::Split(p) => p,
        _ => panic!("expected split"),
    };
    split.first = 1024;
    flipped.op_params = r9v_registry::ElementwiseParams::Split(split.clone());
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped.clone())),
        "split first must change static_hash"
    );
    split.total = 2048;
    split.first = 1024;
    flipped.op_params = r9v_registry::ElementwiseParams::Split(split);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped)),
        "split total must change static_hash"
    );

    // Sampling tree flag, method, and rng.
    let base = common::representative_verify_static(false);
    let tree = common::representative_verify_static(true);
    assert_ne!(
        static_hash(&base),
        static_hash(&tree),
        "tree flag must change static_hash"
    );
    let mut greedy = match base.clone() {
        OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(v)) => v,
        _ => panic!("expected verify"),
    };
    greedy.method = r9v_registry::VerifyMethodStatic::Greedy;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(
            greedy
        ))),
        "verify method must change static_hash"
    );
    let sample = common::representative_sample_static();
    assert_ne!(
        static_hash(&base),
        static_hash(&sample),
        "sampling op variant must change static_hash"
    );
    let mut drafted = match base.clone() {
        OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(v)) => v,
        _ => panic!("expected verify"),
    };
    drafted.has_draft_probs = true;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(
            drafted
        ))),
        "verify draft presence must change static_hash"
    );

    // LogitsPostprocess optional inputs.
    let base = common::representative_logits_postprocess_static();
    let mut flagged = match base.clone() {
        OpStatic::Sampling(r9v_registry::SamplingStatic::LogitsPostprocess(p)) => p,
        _ => panic!("expected logits_postprocess"),
    };
    flagged.has_history_counts = true;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Sampling(
            r9v_registry::SamplingStatic::LogitsPostprocess(flagged)
        )),
        "history presence must change static_hash"
    );
    let mut flagged = match base.clone() {
        OpStatic::Sampling(r9v_registry::SamplingStatic::LogitsPostprocess(p)) => p,
        _ => panic!("expected logits_postprocess"),
    };
    flagged.has_grammar_mask = true;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Sampling(
            r9v_registry::SamplingStatic::LogitsPostprocess(flagged)
        )),
        "grammar presence must change static_hash"
    );

    // Collective rank, group, and peer.
    let base = common::representative_static_for_op(OpId::Send);
    let mut flipped = match base.clone() {
        OpStatic::Collectives(r9v_registry::CollectivesStatic::Send(s)) => s,
        _ => panic!("expected send"),
    };
    flipped.peer = 0;
    flipped.rank = 1;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Collectives(
            r9v_registry::CollectivesStatic::Send(flipped.clone())
        )),
        "send rank/peer must change static_hash"
    );
    flipped.rank = 0;
    flipped.peer = 1;
    flipped.group = 7;
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Collectives(
            r9v_registry::CollectivesStatic::Send(flipped)
        )),
        "send group must change static_hash"
    );

    // Recv shape is a semantic.
    let base = common::representative_static_for_op(OpId::Recv);
    let mut flipped = match base.clone() {
        OpStatic::Collectives(r9v_registry::CollectivesStatic::Recv(s)) => s,
        _ => panic!("expected recv"),
    };
    flipped.shape = vec![64, 4096];
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Collectives(
            r9v_registry::CollectivesStatic::Recv(flipped)
        )),
        "recv shape must change static_hash"
    );
}

/// Bit-identical floats hash equal; any bit difference hashes different.
#[test]
fn test_f32_determinism_is_bitwise() {
    let base = common::representative_moe_route_static();
    let same = common::representative_moe_route_static();
    assert_eq!(
        static_hash(&base),
        static_hash(&same),
        "identical descriptors must hash equal"
    );

    let mut bumped = match base.clone() {
        OpStatic::MoeRoute(s) => s,
        _ => panic!("expected moe_route"),
    };
    bumped.set_scale(f32::from_bits(bumped.scale_bits.wrapping_add(1)));
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::MoeRoute(bumped)),
        "one float bit flip must change static_hash"
    );

    // Typical acceptance epsilon/delta bits are preserved exactly.
    let eps = 0.05f32;
    let delta = 1.25f32;
    let method = r9v_registry::VerifyMethodStatic::typical(eps, delta);
    assert_eq!(method.eps(), Some(eps));
    assert_eq!(method.delta(), Some(delta));
    let neighbor = r9v_registry::VerifyMethodStatic::typical(eps, 1.2500001f32);
    assert_ne!(method, neighbor);
    let h1 = static_hash(&OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(
        r9v_registry::VerifyStatic {
            s_bucket: 2,
            v: 32000,
            q_bucket: 4,
            method,
            tree: false,
            has_draft_probs: false,
        },
    )));
    let h2 = static_hash(&OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(
        r9v_registry::VerifyStatic {
            s_bucket: 2,
            v: 32000,
            q_bucket: 4,
            method: neighbor,
            tree: false,
            has_draft_probs: false,
        },
    )));
    assert_ne!(h1, h2, "one float bit flip must change static_hash");
}

/// Canonical names encode the op and the full 16-hex static hash.
#[test]
fn test_canonical_name_encodes_op_and_hash() {
    for (op, stat) in [
        (OpId::Matmul, common::representative_matmul_static()),
        (OpId::MoeRoute, common::representative_moe_route_static()),
        (
            OpId::CausalConv1d,
            common::representative_causal_conv1d_static(),
        ),
        (OpId::Verify, common::representative_verify_static(false)),
        (
            OpId::AllReduce,
            common::representative_static_for_op(OpId::AllReduce),
        ),
    ] {
        let name = canonical_struct_name(op, &stat);
        let expected = format!("{}_{:016x}_args", op.as_str(), static_hash(&stat));
        assert_eq!(name, expected, "canonical name must encode op and hash");
        let abi = abi_for_op(op, &stat).expect("representative ABI must construct");
        assert_eq!(
            abi.name(),
            expected,
            "built ABI must carry the canonical name"
        );
    }

    // MoeRoute and MoeFfn over different statics never share a name.
    let route_name =
        canonical_struct_name(OpId::MoeRoute, &common::representative_moe_route_static());
    let ffn_name = canonical_struct_name(OpId::MoeFfn, &common::representative_moe_ffn_static());
    assert_ne!(route_name, ffn_name, "distinct families must not collide");
}

/// Family names are exact; no guessing between neighboring families.
#[test]
fn test_family_names_are_exact() {
    assert_eq!(
        op_static_family(&common::representative_moe_route_static()),
        "moe_route"
    );
    assert_eq!(
        op_static_family(&common::representative_moe_ffn_static()),
        "moe_ffn"
    );
    assert_eq!(
        op_static_family(&common::representative_causal_conv1d_static()),
        "causal_conv1d"
    );
    assert_eq!(
        op_static_family(&common::representative_linear_attn_scan_static()),
        "linear_attn_scan"
    );
}

/// Matmul activation pointer follows activation dtype, not out dtype.
#[test]
fn test_matmul_activation_pointer_follows_activation_dtype() {
    let mut inner = match common::representative_matmul_static() {
        OpStatic::Matmul(s) => s,
        _ => panic!("expected matmul"),
    };
    inner.in_dtype = r9v_ir::DType::Bf16;
    inner.out_dtype = r9v_ir::DType::F32;
    let stat = OpStatic::Matmul(inner);
    let built = abi_for_op(OpId::Matmul, &stat).expect("matmul abi builds");
    let x = built.field("x").expect("matmul has x");
    match x.ty {
        r9v_kgen::abi::AbiType::Pointer { pointee, .. } => assert_eq!(
            pointee,
            PointeeType::BF16,
            "matmul x pointer must follow activation dtype"
        ),
        ref other => panic!("matmul x must be a pointer, got {other:?}"),
    }
    let y = built.field("y").expect("matmul has y");
    match y.ty {
        r9v_kgen::abi::AbiType::Pointer { pointee, .. } => assert_eq!(
            pointee,
            PointeeType::F32,
            "matmul y pointer must follow out dtype"
        ),
        ref other => panic!("matmul y must be a pointer, got {other:?}"),
    }
}

/// Every op rejects statics built for another op with a typed error.
#[test]
fn test_all_32_exact_pairing_mismatches_are_typed_errors() {
    let stats: Vec<(OpId, OpStatic)> = common::ALL_32_OPS
        .iter()
        .map(|op| (*op, common::representative_static_for_op(*op)))
        .collect();
    assert_eq!(stats.len(), 32);
    for (op, _) in &stats {
        for (other_op, other_stat) in &stats {
            if op == other_op {
                continue;
            }
            let err = abi_for_op(*op, other_stat).expect_err("cross-op pairing must fail");
            match err {
                KgenError::MismatchedOpFamily { .. } | KgenError::NestedOpMismatch { .. } => {}
                unexpected => panic!(
                    "pairing {op} with {other_op} statics must be a family/nested mismatch, got {unexpected:?}"
                ),
            }
        }
    }

    // Within-family mismatches are nested errors, never panics.
    let sample_stat = common::representative_sample_static();
    match abi_for_op(OpId::Verify, &sample_stat).expect_err("sample static for verify must fail") {
        KgenError::NestedOpMismatch { op, static_op } => {
            assert_eq!(op, OpId::Verify);
            assert_eq!(static_op, OpId::Sample);
        }
        unexpected => panic!("expected nested mismatch, got {unexpected:?}"),
    }
    let send_stat = common::representative_static_for_op(OpId::Send);
    match abi_for_op(OpId::Recv, &send_stat).expect_err("send static for recv must fail") {
        KgenError::NestedOpMismatch { op, static_op } => {
            assert_eq!(op, OpId::Recv);
            assert_eq!(static_op, OpId::Send);
        }
        unexpected => panic!("expected nested mismatch, got {unexpected:?}"),
    }
    let norm_stat = common::representative_elementwise_static();
    match abi_for_op(OpId::Rope, &norm_stat).expect_err("norm static for rope must fail") {
        KgenError::NestedOpMismatch { op, static_op } => {
            assert_eq!(op, OpId::Rope);
            assert_eq!(static_op, OpId::Norm);
        }
        unexpected => panic!("expected nested mismatch, got {unexpected:?}"),
    }
}

/// Tree verify adds tree inputs; flat verify has none.
#[test]
fn test_tree_verify_flag_gates_tree_inputs() {
    let flat = common::representative_verify_static(false);
    let flat_abi = abi_for_op(OpId::Verify, &flat).expect("flat verify abi");
    assert!(
        flat_abi.field("tree_parents").is_none(),
        "flat verify must not have tree_parents"
    );
    assert!(
        flat_abi.field("tree_ancestors").is_none(),
        "flat verify must not have tree_ancestors"
    );

    let tree = common::representative_verify_static(true);
    let tree_abi = abi_for_op(OpId::Verify, &tree).expect("tree verify abi");
    assert!(
        tree_abi.field("tree_parents").is_some(),
        "tree verify must have tree_parents"
    );
    assert!(
        tree_abi.field("tree_ancestors").is_some(),
        "tree verify must have tree_ancestors"
    );
}

/// Collective rank/world/peer/group are static-only; count stays dynamic.
#[test]
fn test_only_count_is_a_dynamic_collective_scalar() {
    for op in [
        OpId::AllReduce,
        OpId::AllGather,
        OpId::ReduceScatter,
        OpId::Send,
        OpId::Recv,
    ] {
        let stat = common::representative_static_for_op(op);
        let abi = abi_for_op(op, &stat).expect("collective abi");
        assert!(abi.field("count").is_some(), "{op} must keep dynamic count");
        assert!(
            abi.field("rank").is_none(),
            "{op} must not have dynamic rank"
        );
        assert!(
            abi.field("world_size").is_none(),
            "{op} must not have dynamic world_size"
        );
        assert!(
            abi.field("peer").is_none(),
            "{op} must not have dynamic peer"
        );
    }
    let barrier_stat = common::representative_static_for_op(OpId::Barrier);
    let barrier_abi = abi_for_op(OpId::Barrier, &barrier_stat).expect("barrier abi");
    assert!(
        barrier_abi.field("flags").is_some(),
        "barrier keeps its flags pointer"
    );
    assert!(
        barrier_abi.field("rank").is_none(),
        "barrier must not have dynamic rank"
    );
}

/// Embed override/mask and both n-gram sources are wired nullable inputs.
#[test]
fn test_embed_override_and_both_ngram_sources() {
    let embed_abi = abi_for_op(
        OpId::EmbedGather,
        &common::representative_static_for_op(OpId::EmbedGather),
    )
    .expect("embed abi");
    let over = embed_abi
        .field("embed_override")
        .expect("embed must have embed_override");
    assert!(over.ty.is_nullable(), "embed_override must be nullable");
    let mask = embed_abi
        .field("embed_mask")
        .expect("embed must have embed_mask");
    assert!(mask.ty.is_nullable(), "embed_mask must be nullable");

    let ngram_abi = abi_for_op(
        OpId::NgramGather,
        &common::representative_static_for_op(OpId::NgramGather),
    )
    .expect("ngram abi");
    for name in ["staging", "row_scales", "token_ids", "table"] {
        let field = ngram_abi.field(name).expect("ngram source field");
        assert!(field.ty.is_nullable(), "ngram {name} must be nullable");
    }
}

/// Every newly closed input semantic changes the canonical variant name, and
/// dtype-dependent ABI pointers follow the static dtype instead of aliasing.
#[test]
fn test_new_semantics_change_canonical_name_and_abi_pointers() {
    let named = |op: OpId, stat: &OpStatic| canonical_struct_name(op, stat);

    // Attention q dtype changes the variant name.
    let base = common::representative_attention_static(AttentionMask::Causal, 16);
    let mut q_bf16 = match base.clone() {
        OpStatic::Attention(s) => s,
        _ => panic!("expected attention"),
    };
    q_bf16.q_dtype = r9v_ir::DType::Bf16;
    let q_bf16 = OpStatic::Attention(q_bf16);
    assert_ne!(
        named(OpId::Attention, &base),
        named(OpId::Attention, &q_bf16),
        "q dtype must change the canonical name"
    );

    // Residual addends change the variant name independently.
    let base = common::representative_residual_add_static();
    let mut a_bf16 = match base.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    if let r9v_registry::ElementwiseParams::ResidualAdd(ref mut r) = a_bf16.op_params {
        r.a_dtype = r9v_ir::DType::Bf16;
    }
    let a_bf16 = OpStatic::Elementwise(a_bf16);
    assert_ne!(
        named(OpId::ResidualAdd, &base),
        named(OpId::ResidualAdd, &a_bf16),
        "residual a dtype must change the canonical name"
    );

    // Conv bias pointer follows the static bias dtype; absent bias keeps the
    // historical f32 pointer shape.
    let conv_none = common::representative_causal_conv1d_static();
    let mut conv_f16 = match conv_none.clone() {
        OpStatic::CausalConv1d(s) => s,
        _ => panic!("expected causal_conv1d"),
    };
    conv_f16.bias_dtype = Some(r9v_ir::DType::F16);
    let conv_f16 = OpStatic::CausalConv1d(conv_f16);
    assert_ne!(
        named(OpId::CausalConv1d, &conv_none),
        named(OpId::CausalConv1d, &conv_f16),
        "conv bias dtype must change the canonical name"
    );
    let abi_none = abi_for_op(OpId::CausalConv1d, &conv_none).expect("conv abi");
    let abi_f16 = abi_for_op(OpId::CausalConv1d, &conv_f16).expect("conv abi");
    let ptr_none = &abi_none.field("bias").expect("conv bias").ty;
    let ptr_f16 = &abi_f16.field("bias").expect("conv bias").ty;
    assert_ne!(
        ptr_none, ptr_f16,
        "f16 bias must not share the f32 bias pointer type"
    );
    assert_eq!(
        *ptr_none,
        r9v_kgen::abi::AbiType::nullable_const_ptr(PointeeType::F32),
        "absent bias keeps the historical f32 pointer shape"
    );

    // Matmul residual pointer follows the static residual dtype.
    let matmul_none = common::representative_matmul_static();
    let mut res_f16 = match matmul_none.clone() {
        OpStatic::Matmul(s) => s,
        _ => panic!("expected matmul"),
    };
    res_f16.epilogue = r9v_ir::Epilogue::Residual;
    res_f16.residual_dtype = Some(r9v_ir::DType::F16);
    let res_f16 = OpStatic::Matmul(res_f16);
    let mut res_bf16 = match res_f16.clone() {
        OpStatic::Matmul(s) => s,
        _ => panic!("expected matmul"),
    };
    res_bf16.residual_dtype = Some(r9v_ir::DType::Bf16);
    let res_bf16 = OpStatic::Matmul(res_bf16);
    assert_ne!(
        named(OpId::Matmul, &matmul_none),
        named(OpId::Matmul, &res_f16),
        "residual epilogue input must change the canonical name"
    );
    assert_ne!(
        named(OpId::Matmul, &res_f16),
        named(OpId::Matmul, &res_bf16),
        "residual dtype must change the canonical name"
    );
    let abi_f16 = abi_for_op(OpId::Matmul, &res_f16).expect("matmul abi");
    let abi_bf16 = abi_for_op(OpId::Matmul, &res_bf16).expect("matmul abi");
    assert_ne!(
        abi_f16.field("residual").expect("residual").ty,
        abi_bf16.field("residual").expect("residual").ty,
        "bf16 residual must not share the f16 residual pointer type"
    );

    // Ngram row-scales pointer follows the static scales dtype; Device mode
    // shares the Staged-f32 pointer shape but keeps a distinct name.
    let staged_f32 = common::representative_static_for_op(OpId::NgramGather);
    let mut staged_f16 = match staged_f32.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    if let r9v_registry::ElementwiseParams::NgramGather(ref mut g) = staged_f16.op_params {
        g.scales_dtype = Some(r9v_ir::DType::F16);
    }
    let staged_f16 = OpStatic::Elementwise(staged_f16);
    assert_ne!(
        named(OpId::NgramGather, &staged_f32),
        named(OpId::NgramGather, &staged_f16),
        "scales dtype must change the canonical name"
    );
    let abi_f32 = abi_for_op(OpId::NgramGather, &staged_f32).expect("ngram abi");
    let abi_f16 = abi_for_op(OpId::NgramGather, &staged_f16).expect("ngram abi");
    assert_ne!(
        abi_f32.field("row_scales").expect("row_scales").ty,
        abi_f16.field("row_scales").expect("row_scales").ty,
        "f16 scales must not share the f32 scales pointer type"
    );
}

/// Asserts one pointer field's exact pointee, nullability, constness, and role.
fn expect_ptr(
    abi: &AbiStruct,
    name: &str,
    pointee: PointeeType,
    nullable: bool,
    is_const: bool,
    role: FieldRole,
) {
    let field = abi
        .field(name)
        .unwrap_or_else(|| panic!("{} ABI must have field '{name}'", abi.op()));
    assert_eq!(
        field.role(),
        role,
        "{} field '{name}' must have role {role:?}",
        abi.op()
    );
    match &field.ty {
        AbiType::Pointer {
            pointee: got,
            is_const: got_const,
            is_nullable: got_nullable,
            ..
        } => {
            assert_eq!(
                *got,
                pointee,
                "{} field '{name}' must point to {pointee:?}",
                abi.op()
            );
            assert_eq!(
                *got_nullable,
                nullable,
                "{} field '{name}' nullability must be {nullable}",
                abi.op()
            );
            assert_eq!(
                *got_const,
                is_const,
                "{} field '{name}' constness must be {is_const}",
                abi.op()
            );
        }
        other => panic!(
            "{} field '{name}' must be a pointer, got {other:?}",
            abi.op()
        ),
    }
}

/// Exhaustive ABI pointee policy over all 32 ops (Spec 4 §7).
///
/// Every activation, parameter, index, or output pointer whose exact element
/// dtype is in OpStatic is typed via `PointeeType::from_dtype` (or the exact
/// U32/U64/F32/U8 spelling when the static carries no dtype, including the
/// spec-fixed batch-meta index buffers); Void survives only for the documented
/// exception classes (packed weights, scale records, state arenas, byte-copy,
/// heterogeneous records).
#[test]
fn test_abi_pointee_policy_is_exhaustive() {
    use FieldRole::{
        ActivationScale, Bias, InputTensor, OutputTensor, Residual, Weight, WeightIndices,
        WeightScale, Workspace,
    };
    use PointeeType::{Void, BF16, F16, F32, I8, U32, U64, U8};

    // One asserted pointer row: (field, pointee, nullable, is_const, role).
    type PtrRow = (&'static str, PointeeType, bool, bool, FieldRole);
    // Every pointer in every ABI, with batch-meta and workspace fields included.
    let table: &[(OpId, &[PtrRow])] = &[
        (
            OpId::EmbedGather,
            &[
                ("token_ids", U32, false, true, InputTensor),
                ("table", Void, false, true, InputTensor),
                ("embed_override", F16, true, true, InputTensor),
                ("embed_mask", U8, true, true, InputTensor),
                ("x", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::NgramGather,
            &[
                ("staging", Void, true, true, InputTensor),
                ("row_scales", F32, true, true, WeightScale),
                ("token_ids", U32, true, true, InputTensor),
                ("table", Void, true, true, InputTensor),
                ("x", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::QuantAct,
            &[
                ("x", F16, false, true, InputTensor),
                ("xq", I8, false, false, OutputTensor),
                ("scale", F32, false, false, ActivationScale),
            ],
        ),
        (
            OpId::Cast,
            &[
                ("x", F16, false, true, InputTensor),
                ("y", BF16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Copy,
            &[
                ("src", Void, false, true, InputTensor),
                ("dst", Void, false, false, OutputTensor),
            ],
        ),
        (
            OpId::GatherRows,
            &[
                ("x", F16, false, true, InputTensor),
                ("indices", U32, false, true, InputTensor),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::ScatterAddRows,
            &[
                ("x", F32, false, true, InputTensor),
                ("indices", U32, false, true, InputTensor),
                ("dest", F32, true, true, InputTensor),
                ("y", F32, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Split,
            &[
                ("x", F16, false, true, InputTensor),
                ("y0", F16, false, false, OutputTensor),
                ("y1", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Concat,
            &[
                ("x0", F16, false, true, InputTensor),
                ("x1", F16, false, true, InputTensor),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Norm,
            &[
                ("x", F16, false, true, InputTensor),
                ("weight", F32, false, true, Weight),
                ("bias", F32, true, true, Bias),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::ResidualAdd,
            &[
                ("a", F16, false, true, InputTensor),
                ("b", F16, false, true, Residual),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::ActMul,
            &[
                ("gate", F16, false, true, InputTensor),
                ("up", F16, false, true, InputTensor),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Activation,
            &[
                ("x", F16, false, true, InputTensor),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::LogitSoftcap,
            &[
                ("x", F32, false, true, InputTensor),
                ("y", F32, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Rope,
            &[
                ("x", F16, false, true, InputTensor),
                (
                    "positions",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::Positions),
                ),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Matmul,
            &[
                ("w", U8, false, true, Weight),
                ("w_scales", Void, true, true, WeightScale),
                ("w_indices", U8, true, true, WeightIndices),
                ("x", F16, false, true, InputTensor),
                ("x_scale", F32, true, true, ActivationScale),
                ("bias", F32, true, true, Bias),
                ("residual", F16, true, true, Residual),
                ("y", F16, false, false, OutputTensor),
                ("workspace", F32, false, false, Workspace),
            ],
        ),
        (
            OpId::MoeRoute,
            &[
                ("logits", F32, false, true, InputTensor),
                ("bias", F32, true, true, Bias),
                ("expert_ids", U32, false, false, OutputTensor),
                ("weights", F32, false, false, OutputTensor),
            ],
        ),
        (
            OpId::MoeFfn,
            &[
                ("x", F16, false, true, InputTensor),
                ("expert_ids", U32, false, true, InputTensor),
                ("weights", F32, false, true, InputTensor),
                ("w_gate_up", Void, false, true, Weight),
                ("w_gate_up_scales", Void, true, true, WeightScale),
                ("w_down", Void, false, true, Weight),
                ("w_down_scales", Void, true, true, WeightScale),
                ("y", F16, false, false, OutputTensor),
                ("sort_workspace", I8, false, false, Workspace),
            ],
        ),
        (
            OpId::Attention,
            &[
                ("q", F16, false, true, InputTensor),
                ("k_cache", Void, false, true, InputTensor),
                ("v_cache", Void, false, true, InputTensor),
                ("o", F16, false, false, OutputTensor),
                (
                    "block_table",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::BlockTable),
                ),
                (
                    "ctx_lens",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::CtxLen),
                ),
                (
                    "query_lens",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::QueryLen),
                ),
                ("workspace", F32, false, false, Workspace),
            ],
        ),
        (
            OpId::StateWriteKv,
            &[
                ("k", F16, false, true, InputTensor),
                ("v", F16, false, true, InputTensor),
                (
                    "slot_map",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::SlotMap),
                ),
                ("k_cache", Void, false, false, OutputTensor),
                ("v_cache", Void, false, false, OutputTensor),
            ],
        ),
        (
            OpId::CausalConv1d,
            &[
                ("x", F16, false, true, InputTensor),
                ("w", Void, false, true, Weight),
                ("bias", F32, true, true, Bias),
                ("conv_state", Void, false, false, InputTensor),
                ("y", F16, false, false, OutputTensor),
            ],
        ),
        (
            OpId::LinearAttnScan,
            &[
                ("q", F16, false, true, InputTensor),
                ("k", F16, false, true, InputTensor),
                ("v", F16, false, true, InputTensor),
                ("alpha", F32, false, true, InputTensor),
                ("beta", F32, false, true, InputTensor),
                ("state", Void, false, false, InputTensor),
                ("o", F16, false, false, OutputTensor),
                (
                    "query_lens",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::QueryLen),
                ),
                ("workspace", F32, false, false, Workspace),
            ],
        ),
        (
            OpId::LogitsPostprocess,
            &[
                ("logits", F32, false, true, InputTensor),
                ("params", Void, false, true, InputTensor),
                ("history_counts", U32, true, true, InputTensor),
                ("grammar_mask", U8, true, true, InputTensor),
                ("probs", F32, false, false, OutputTensor),
                ("workspace", I8, false, false, Workspace),
            ],
        ),
        (
            OpId::Sample,
            &[
                ("probs", F32, false, true, InputTensor),
                ("rng_state", U64, false, false, InputTensor),
                (
                    "seq_ids",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::SeqIds),
                ),
                ("tokens", U32, false, false, OutputTensor),
            ],
        ),
        (
            OpId::Verify,
            &[
                ("draft_tokens", U32, false, true, InputTensor),
                ("draft_probs", F32, true, true, InputTensor),
                ("target_probs", F32, false, true, InputTensor),
                ("rng_state", U64, false, false, InputTensor),
                (
                    "seq_ids",
                    U32,
                    false,
                    true,
                    FieldRole::BatchMeta(BatchMetaField::SeqIds),
                ),
                ("accepted", U32, false, false, OutputTensor),
                ("accept_len", U32, false, false, OutputTensor),
            ],
        ),
        (
            OpId::AllReduce,
            &[
                ("send_buf", F16, false, true, InputTensor),
                ("recv_buf", F16, false, false, OutputTensor),
                ("staging", I8, false, false, Workspace),
            ],
        ),
        (
            OpId::AllGather,
            &[
                ("send_buf", F16, false, true, InputTensor),
                ("recv_buf", F16, false, false, OutputTensor),
                ("staging", I8, false, false, Workspace),
            ],
        ),
        (
            OpId::ReduceScatter,
            &[
                ("send_buf", F16, false, true, InputTensor),
                ("recv_buf", F16, false, false, OutputTensor),
                ("staging", I8, false, false, Workspace),
            ],
        ),
        (
            OpId::AllToAll,
            &[
                ("send_buf", F16, false, true, InputTensor),
                ("recv_buf", F16, false, false, OutputTensor),
                ("counts", U32, false, true, InputTensor),
                ("staging", I8, false, false, Workspace),
            ],
        ),
        (
            OpId::Send,
            &[
                ("send_buf", F16, false, true, InputTensor),
                ("staging", I8, false, false, Workspace),
            ],
        ),
        (
            OpId::Recv,
            &[
                ("recv_buf", F16, false, false, OutputTensor),
                ("staging", I8, false, false, Workspace),
            ],
        ),
        (OpId::Barrier, &[("flags", U32, false, false, InputTensor)]),
    ];
    assert_eq!(table.len(), 32, "pointee table must cover all 32 ops");

    for (op, fields) in table {
        let stat = common::representative_static_for_op(*op);
        let built = abi_for_op(*op, &stat).expect("representative ABI must construct");
        for (name, pointee, nullable, is_const, role) in fields.iter() {
            expect_ptr(&built, name, *pointee, *nullable, *is_const, *role);
        }
        // No unnamed pointer may hide outside the table: every pointer field
        // in the built ABI must be an asserted row above.
        for field in built.fields() {
            if field.ty.is_pointer() {
                assert!(
                    fields.iter().any(|(n, _, _, _, _)| *n == field.name),
                    "{} has unasserted pointer field '{}'",
                    op,
                    field.name
                );
            }
        }
    }
}

/// Scatter absent/present ABI contract (Spec 1 §4.A, SI-10, Spec 4 §3, §7).
///
/// Both legal forms carry the same distinct nullable `dest` input typed by the
/// static dtype; absent-form launches with null, present-form with a base
/// tensor. Hash and canonical name already distinguish the forms.
///
/// Nullable is the only contract available here: A3.1/A3.2 ship no value-level
/// host-argument packing or validation API (LaunchEntry.args_blob is opaque
/// bytes; DeviceExecutor/StubDevice accept entries as-is), so no typed
/// dest-null-iff-has_dest error can be enforced host-side without inventing a
/// launch layer outside A3.1/A3.2 scope. Enforcement is the static has_dest
/// flag, the nullable ABI spelling, and the guarded HIP unpacker below.
#[test]
fn test_scatter_dest_absent_present_contract() {
    let absent = common::representative_static_for_op(OpId::ScatterAddRows);
    let mut present_inner = match absent.clone() {
        OpStatic::Elementwise(s) => s,
        _ => panic!("expected elementwise"),
    };
    let mut scatter = match present_inner.op_params.clone() {
        r9v_registry::ElementwiseParams::ScatterAddRows(p) => p,
        _ => panic!("expected scatter_add_rows"),
    };
    scatter.has_dest = true;
    present_inner.op_params = r9v_registry::ElementwiseParams::ScatterAddRows(scatter);
    let present = OpStatic::Elementwise(present_inner);

    let absent_abi = abi_for_op(OpId::ScatterAddRows, &absent).expect("absent ABI builds");
    let present_abi = abi_for_op(OpId::ScatterAddRows, &present).expect("present ABI builds");

    // Both forms carry the same distinct nullable dest input.
    for abi in [&absent_abi, &present_abi] {
        expect_ptr(
            abi,
            "dest",
            PointeeType::F32,
            true,
            true,
            FieldRole::InputTensor,
        );
        expect_ptr(
            abi,
            "x",
            PointeeType::F32,
            false,
            true,
            FieldRole::InputTensor,
        );
        expect_ptr(
            abi,
            "indices",
            PointeeType::U32,
            false,
            true,
            FieldRole::InputTensor,
        );
        expect_ptr(
            abi,
            "y",
            PointeeType::F32,
            false,
            false,
            FieldRole::OutputTensor,
        );
        abi.validate().expect("scatter ABI must validate");
    }

    // Hash and canonical name already distinguish the forms; statics are untouched.
    assert_ne!(
        static_hash(&absent),
        static_hash(&present),
        "has_dest must change static_hash"
    );
    assert_ne!(
        canonical_struct_name(OpId::ScatterAddRows, &absent),
        canonical_struct_name(OpId::ScatterAddRows, &present),
        "has_dest must change the canonical name"
    );

    // Generated code launches both forms: the HIP struct carries dest and the
    // nullable unpacker guards it, so absent-form null and present-form base
    // tensor both launch through the same guarded load.
    for abi in [&absent_abi, &present_abi] {
        let hip = emit_hip_struct(abi);
        assert!(
            hip.contains("dest;"),
            "emitted HIP struct for {} must carry dest",
            abi.name()
        );
        let unpack = emit_hip_assume_aligned(abi, "args");
        assert!(
            unpack.contains("args.dest ?"),
            "emitted unpacker for {} must guard nullable dest",
            abi.name()
        );
    }
}
