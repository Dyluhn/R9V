// SPDX-License-Identifier: Apache-2.0
//! Deterministic regeneration and contract tests for r9v-kgen ABI generator (Spec 4 §7; card A3.2).

mod common;

use r9v_ir::AttentionMask;
use r9v_kgen::abi::{
    abi, abi_for_op, canonical_struct_name, emit_all_hip_header, emit_all_rust_module,
    emit_hip_assume_aligned, AbiField, AbiType, BatchMetaField, FieldRole,
    KERNEL_PTR_ALIGNMENT_BYTES,
};
use r9v_kgen::error::KgenError;
use r9v_registry::{static_hash, OpId, OpStatic};

#[test]
fn test_golden_variant_struct_names() {
    let matmul_st = common::representative_matmul_static();
    let matmul_hash = static_hash(&matmul_st);
    let expected_matmul_name = format!("matmul_{:016x}_args", matmul_hash);
    let matmul_abi = abi_for_op(OpId::Matmul, &matmul_st).expect("matmul abi");
    assert_eq!(matmul_abi.name(), expected_matmul_name);
    assert_eq!(
        canonical_struct_name(OpId::Matmul, &matmul_st),
        expected_matmul_name
    );

    let moe_st = common::representative_moe_ffn_static();
    let moe_hash = static_hash(&moe_st);
    let expected_moe_name = format!("moe_route_{:016x}_args", moe_hash);
    let moe_route_abi = abi_for_op(OpId::MoeRoute, &moe_st).expect("moe_route abi");
    assert_eq!(moe_route_abi.name(), expected_moe_name);

    let attn_causal_st = common::representative_attention_static(AttentionMask::Causal, 16);
    let attn_causal_hash = static_hash(&attn_causal_st);
    let expected_attn_name = format!("attention_{:016x}_args", attn_causal_hash);
    let attn_abi = abi_for_op(OpId::Attention, &attn_causal_st).expect("attention abi");
    assert_eq!(attn_abi.name(), expected_attn_name);

    let rope_st = common::representative_elementwise_static();
    let rope_hash = static_hash(&rope_st);
    let expected_rope_name = format!("rope_{:016x}_args", rope_hash);
    let rope_abi = abi_for_op(OpId::Rope, &rope_st).expect("rope abi");
    assert_eq!(rope_abi.name(), expected_rope_name);
}

#[test]
fn test_static_field_change_changes_hash_and_name() {
    let matmul1 = match common::representative_matmul_static() {
        OpStatic::Matmul(s) => s,
        _ => unreachable!(),
    };
    let mut matmul2 = matmul1.clone();
    matmul2.n = 2048; // change single field

    let st1 = OpStatic::Matmul(matmul1.clone());
    let st2 = OpStatic::Matmul(matmul2);

    let hash1 = static_hash(&st1);
    let hash2 = static_hash(&st2);
    assert_ne!(
        hash1, hash2,
        "changing static field n must produce different static_hash"
    );

    let name1 = canonical_struct_name(OpId::Matmul, &st1);
    let name2 = canonical_struct_name(OpId::Matmul, &st2);
    assert_ne!(
        name1, name2,
        "changing static field n must produce different canonical struct name"
    );

    // Another field change: interleave
    let mut matmul3 = matmul1;
    matmul3.interleave = true;
    let st3 = OpStatic::Matmul(matmul3);
    let hash3 = static_hash(&st3);
    assert_ne!(
        hash1, hash3,
        "changing interleave flag must change static_hash"
    );
}

#[test]
fn test_hip_assume_aligned_guarded_unpacker() {
    // Test that required pointers unpack directly, nullable pointers unpack with ternary guard,
    // and KERNEL_PTR_ALIGNMENT_BYTES (256) is referenced.
    let matmul_st = common::representative_matmul_static();
    let matmul_abi = abi_for_op(OpId::Matmul, &matmul_st).expect("matmul abi");

    let hip_code = emit_hip_assume_aligned(&matmul_abi, "args");

    // Required pointer "w" should be unconditional __builtin_assume_aligned
    assert!(
        hip_code
            .contains("const uint8_t* w = (const uint8_t*)__builtin_assume_aligned(args.w, 256);"),
        "required pointer 'w' must emit unconditional assume_aligned unpacker, got:\n{hip_code}"
    );

    // Nullable pointer "w_scales" must be ternary guarded
    assert!(
        hip_code.contains("const void* w_scales = args.w_scales ? (const void*)__builtin_assume_aligned(args.w_scales, 256) : nullptr;"),
        "nullable pointer 'w_scales' must emit ternary guarded assume_aligned unpacker, got:\n{hip_code}"
    );

    // Nullable pointer "bias" must be ternary guarded
    assert!(
        hip_code.contains("const float* bias = args.bias ? (const float*)__builtin_assume_aligned(args.bias, 256) : nullptr;"),
        "nullable pointer 'bias' must emit ternary guarded assume_aligned unpacker, got:\n{hip_code}"
    );

    // Verify 256-byte constant
    assert!(
        hip_code.contains(&format!("{KERNEL_PTR_ALIGNMENT_BYTES}")),
        "assume_aligned unpacker must reference 256 bytes"
    );
}

#[test]
fn test_attention_batch_meta_exact_sets() {
    // 1. Causal attention: BlockTable, CtxLen, QueryLen.
    let causal_st = common::representative_attention_static(AttentionMask::Causal, 16);
    let causal_abi = abi_for_op(OpId::Attention, &causal_st).expect("causal attention");
    let causal_bm = causal_abi.batch_meta_fields();
    assert_eq!(
        causal_bm,
        &[
            BatchMetaField::BlockTable,
            BatchMetaField::CtxLen,
            BatchMetaField::QueryLen
        ],
        "Causal attention must have exactly [BlockTable, CtxLen, QueryLen]"
    );
    assert!(
        !causal_bm.contains(&BatchMetaField::WindowStart),
        "Causal attention must never contain WindowStart"
    );
    assert!(
        !causal_bm.contains(&BatchMetaField::TreeParents),
        "Causal attention must never contain TreeParents"
    );
    assert!(
        !causal_bm.contains(&BatchMetaField::TreeAncestors),
        "Causal attention must never contain TreeAncestors"
    );

    // 2. CausalWindow attention: BlockTable, CtxLen, QueryLen, WindowStart.
    let window_st = common::representative_attention_static(AttentionMask::CausalWindow(256), 16);
    let window_abi = abi_for_op(OpId::Attention, &window_st).expect("causal window attention");
    let window_bm = window_abi.batch_meta_fields();
    assert_eq!(
        window_bm,
        &[
            BatchMetaField::BlockTable,
            BatchMetaField::CtxLen,
            BatchMetaField::QueryLen,
            BatchMetaField::WindowStart
        ],
        "CausalWindow attention must have exactly [BlockTable, CtxLen, QueryLen, WindowStart]"
    );
    assert!(
        !window_bm.contains(&BatchMetaField::TreeParents),
        "CausalWindow attention must never contain TreeParents"
    );
    assert!(
        !window_bm.contains(&BatchMetaField::TreeAncestors),
        "CausalWindow attention must never contain TreeAncestors"
    );

    // 3. Tree attention: BlockTable, CtxLen, QueryLen, TreeParents, TreeAncestors.
    let tree_st = common::representative_attention_static(AttentionMask::Tree, 16);
    let tree_abi = abi_for_op(OpId::Attention, &tree_st).expect("tree attention");
    let tree_bm = tree_abi.batch_meta_fields();
    assert_eq!(
        tree_bm,
        &[
            BatchMetaField::BlockTable,
            BatchMetaField::CtxLen,
            BatchMetaField::QueryLen,
            BatchMetaField::TreeParents,
            BatchMetaField::TreeAncestors
        ],
        "Tree attention must have exactly [BlockTable, CtxLen, QueryLen, TreeParents, TreeAncestors]"
    );
    assert!(
        !tree_bm.contains(&BatchMetaField::WindowStart),
        "Tree attention must never contain WindowStart"
    );
}

#[test]
fn test_spec_dynamic_pointers_and_scalars() {
    // Rope: dynamic pointers are input, output, BatchMetaField::Positions. No sin_cos pointer!
    let rope_st = common::representative_elementwise_static();
    let rope_abi = abi_for_op(OpId::Rope, &rope_st).expect("rope abi");
    assert!(
        rope_abi.field("sin_cos").is_none(),
        "Rope must not have sin_cos pointer"
    );
    assert!(rope_abi.field("x").is_some(), "Rope must have input x");
    assert!(rope_abi.field("y").is_some(), "Rope must have output y");
    assert_eq!(
        rope_abi.batch_meta_fields(),
        &[BatchMetaField::Positions],
        "Rope must use BatchMetaField::Positions"
    );
    let rope_scalar = rope_abi.field("t").expect("Rope dynamic scalar t");
    assert_eq!(rope_scalar.role, FieldRole::DynamicScalar);

    // MoeRoute: logits [T, E] f32, optional/nullable bias [E] f32, expert_ids [T, K] u32, weights [T, K] f32, scalar t.
    // No gate_weight pointer, no sort workspace.
    let moe_st = common::representative_moe_ffn_static();
    let route_abi = abi_for_op(OpId::MoeRoute, &moe_st).expect("moe_route abi");
    assert!(
        route_abi.field("gate_weight").is_none(),
        "MoeRoute must not have gate_weight pointer"
    );
    assert!(
        route_abi.workspace_slots().is_empty(),
        "MoeRoute must not have workspace slots"
    );
    let logits = route_abi.field("logits").expect("MoeRoute logits");
    assert!(logits.ty.is_pointer() && !logits.ty.is_nullable());
    let bias = route_abi.field("bias").expect("MoeRoute bias");
    assert!(bias.ty.is_pointer() && bias.ty.is_nullable());
    let expert_ids = route_abi.field("expert_ids").expect("MoeRoute expert_ids");
    assert!(expert_ids.ty.is_pointer() && !expert_ids.ty.is_nullable());
    let weights = route_abi.field("weights").expect("MoeRoute weights");
    assert!(weights.ty.is_pointer() && !weights.ty.is_nullable());
    let route_t = route_abi.field("t").expect("MoeRoute scalar t");
    assert_eq!(route_t.role, FieldRole::DynamicScalar);

    // Matmul: no runtime scalar dimensions in args (m_bucket, n, k are compile-time static constants).
    let matmul_st = common::representative_matmul_static();
    let matmul_abi = abi_for_op(OpId::Matmul, &matmul_st).expect("matmul abi");
    assert!(
        matmul_abi.field("m").is_none(),
        "Matmul must not have runtime scalar m"
    );
    assert!(
        matmul_abi.field("n").is_none(),
        "Matmul must not have runtime scalar n"
    );
    assert!(
        matmul_abi.field("k").is_none(),
        "Matmul must not have runtime scalar k"
    );
}

#[test]
fn test_family_dispatch_and_adversarial_validation() {
    // 1. Unique families dispatch directly via abi(&OpStatic)
    let matmul_st = common::representative_matmul_static();
    let matmul_abi = abi(&matmul_st).expect("matmul dispatches directly");
    assert_eq!(matmul_abi.op(), OpId::Matmul);

    let attn_st = common::representative_attention_static(AttentionMask::Causal, 16);
    let attn_abi = abi(&attn_st).expect("attention dispatches directly");
    assert_eq!(attn_abi.op(), OpId::Attention);

    let kv_st = common::representative_state_write_kv_static();
    let kv_abi = abi(&kv_st).expect("state_write_kv dispatches directly");
    assert_eq!(kv_abi.op(), OpId::StateWriteKv);

    // 2. Shared/ambiguous families fail closed with AmbiguousOpFamily
    let moe_st = common::representative_moe_ffn_static();
    let err_moe = abi(&moe_st).unwrap_err();
    assert!(
        matches!(err_moe, KgenError::AmbiguousOpFamily { family, ref valid_ops } if family == "moe_ffn" && valid_ops.contains(&OpId::MoeRoute) && valid_ops.contains(&OpId::MoeFfn)),
        "expected AmbiguousOpFamily for MoeFfn, got: {err_moe}"
    );

    let elem_st = common::representative_elementwise_static();
    let err_elem = abi(&elem_st).unwrap_err();
    assert!(
        matches!(err_elem, KgenError::AmbiguousOpFamily { family, ref valid_ops } if family == "elementwise" && valid_ops.contains(&OpId::Rope)),
        "expected AmbiguousOpFamily for Elementwise, got: {err_elem}"
    );

    // 3. Mismatched op and static descriptor fail closed with MismatchedOpFamily
    let err_mismatch = abi_for_op(OpId::Matmul, &moe_st).unwrap_err();
    assert!(
        matches!(err_mismatch, KgenError::MismatchedOpFamily { op: OpId::Matmul, family } if family == "moe_ffn"),
        "expected MismatchedOpFamily, got: {err_mismatch}"
    );
}

#[test]
fn test_emission_sorting_and_deduplication() {
    let op1 = OpId::Matmul;
    let st1 = common::representative_matmul_static();
    let abi1 = abi_for_op(op1, &st1).expect("abi1");

    let op2 = OpId::Attention;
    let st2 = common::representative_attention_static(AttentionMask::Causal, 16);
    let abi2 = abi_for_op(op2, &st2).expect("abi2");

    // Shuffled orders must emit identical code
    let list_a = vec![abi1.clone(), abi2.clone()];
    let list_b = vec![abi2.clone(), abi1.clone()];

    let rust_a = emit_all_rust_module(&list_a).expect("rust_a");
    let rust_b = emit_all_rust_module(&list_b).expect("rust_b");
    assert_eq!(rust_a, rust_b, "Rust emission must sort deterministically");

    let hip_a = emit_all_hip_header(&list_a).expect("hip_a");
    let hip_b = emit_all_hip_header(&list_b).expect("hip_b");
    assert_eq!(hip_a, hip_b, "HIP emission must sort deterministically");

    // Duplicate identical structs are deduplicated cleanly
    let list_with_dup = vec![abi1.clone(), abi2.clone(), abi1.clone()];
    let rust_dup = emit_all_rust_module(&list_with_dup).expect("rust_dup");
    assert_eq!(rust_a, rust_dup, "Duplicates must be deduplicated");

    // Inconsistent collision with same struct name must fail closed
    let mut colliding_abi = abi1.clone();
    colliding_abi.fields.push(AbiField {
        name: "extra".to_string(),
        ty: AbiType::u32(),
        role: FieldRole::DynamicScalar,
        offset: colliding_abi.size(),
        doc: "collision".to_string(),
    });
    let colliding_list = vec![abi1.clone(), colliding_abi];
    let err = emit_all_rust_module(&colliding_list).unwrap_err();
    assert!(
        matches!(err, KgenError::InconsistentVariantCollision { ref name, .. } if name == &abi1.name),
        "expected InconsistentVariantCollision, got: {err}"
    );
}

#[test]
fn test_repeated_generation_is_bit_exact() {
    let mut abis = Vec::new();
    for op in common::ALL_32_OPS {
        let st = common::representative_static_for_op(op);
        abis.push(abi_for_op(op, &st).expect("abi"));
    }

    let rust1 = emit_all_rust_module(&abis).expect("rust run 1");
    let rust2 = emit_all_rust_module(&abis).expect("rust run 2");
    assert_eq!(rust1, rust2, "Rust emission must be bit-exact across runs");

    let hip1 = emit_all_hip_header(&abis).expect("hip run 1");
    let hip2 = emit_all_hip_header(&abis).expect("hip run 2");
    assert_eq!(hip1, hip2, "HIP emission must be bit-exact across runs");
}
