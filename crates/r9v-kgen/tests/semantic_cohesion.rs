// SPDX-License-Identifier: Apache-2.0
//! Semantic cohesion tests for the A3.API typed descriptor contract (Spec 4 §3, §7).
//!
//! Root decision: every compile-time kernel semantic is a closed typed descriptor field
//! included in `static_hash`. No family guessing, no opaque strings, no collisions.

mod common;

use r9v_ir::AttentionMask;
use r9v_kgen::abi::{abi, abi_for_op, canonical_struct_name, op_static_family, PointeeType};
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
        static_hash(&OpStatic::Matmul(flipped)),
        "weight w_dtype must change static_hash"
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
        static_hash(&OpStatic::MoeFfn(flipped)),
        "moe in_dtype must change static_hash"
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
        static_hash(&OpStatic::Attention(flipped)),
        "mla descriptor must change static_hash"
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
        static_hash(&OpStatic::CausalConv1d(flipped)),
        "conv out_dtype must change static_hash"
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
    flipped.op_params = r9v_registry::ElementwiseParams::ResidualAdd(add);
    assert_ne!(
        static_hash(&base),
        static_hash(&OpStatic::Elementwise(flipped)),
        "residual scale bits must change static_hash"
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
    device.op_params = r9v_registry::ElementwiseParams::NgramGather(params);
    assert_ne!(
        static_hash(&staged),
        static_hash(&OpStatic::Elementwise(device)),
        "ngram dn must change static_hash"
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
        },
    )));
    let h2 = static_hash(&OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(
        r9v_registry::VerifyStatic {
            s_bucket: 2,
            v: 32000,
            q_bucket: 4,
            method: neighbor,
            tree: false,
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
