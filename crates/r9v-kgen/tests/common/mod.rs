// SPDX-License-Identifier: Apache-2.0
//! Common test helpers and representative op static descriptors for r9v-kgen tests.

use r9v_ir::{AttentionMask, DType, Epilogue, LayoutId, LinearAttnKind, P2pTransport, QuantScheme};
use r9v_registry::{
    AttentionStatic, CollectivesStatic, ElementwiseStatic, LinearAttnScanStatic, MatmulStatic,
    MoeFfnStatic, OpId, OpStatic, PlacementKind, SamplingMethod, SamplingStatic, ScanMode,
    StateWriteKvStatic,
};

pub const ALL_32_OPS: [OpId; 32] = [
    OpId::EmbedGather,
    OpId::NgramGather,
    OpId::QuantAct,
    OpId::Cast,
    OpId::Copy,
    OpId::GatherRows,
    OpId::ScatterAddRows,
    OpId::Split,
    OpId::Concat,
    OpId::Norm,
    OpId::ResidualAdd,
    OpId::ActMul,
    OpId::Activation,
    OpId::LogitSoftcap,
    OpId::Rope,
    OpId::Matmul,
    OpId::MoeRoute,
    OpId::MoeFfn,
    OpId::StateWriteKv,
    OpId::Attention,
    OpId::CausalConv1d,
    OpId::LinearAttnScan,
    OpId::LogitsPostprocess,
    OpId::Sample,
    OpId::Verify,
    OpId::AllReduce,
    OpId::AllGather,
    OpId::ReduceScatter,
    OpId::AllToAll,
    OpId::Send,
    OpId::Recv,
    OpId::Barrier,
];

pub fn representative_matmul_static() -> OpStatic {
    OpStatic::Matmul(MatmulStatic {
        m_bucket: 16,
        n: 4096,
        k: 4096,
        w_scheme: QuantScheme::None,
        w_layout: LayoutId::L1,
        act_scheme: QuantScheme::None,
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        interleave: false,
        sparse: false,
    })
}

pub fn representative_moe_ffn_static() -> OpStatic {
    OpStatic::MoeFfn(MoeFfnStatic {
        t_bucket: 16,
        e_local: 8,
        k_topk: 2,
        dm: 2048,
        dff: 1024,
        schemes: vec![QuantScheme::None],
        act_scheme: QuantScheme::None,
        placement_kind: PlacementKind::Device,
    })
}

pub fn representative_attention_static(mask_kind: AttentionMask, q_bucket: u32) -> OpStatic {
    OpStatic::Attention(AttentionStatic {
        q_bucket,
        h_local: 32,
        hkv_local: 8,
        d: 128,
        dv: 128,
        cache_dtype: DType::F16,
        attention_layout: LayoutId::L1,
        mask_kind,
        latent: None,
        softcap_bits: None,
        sinks: None,
    })
}

pub fn representative_state_write_kv_static() -> OpStatic {
    OpStatic::StateWriteKv(StateWriteKvStatic {
        hkv_local: 8,
        d: 128,
        dv: 128,
        cache_dtype: DType::F16,
        attention_layout: LayoutId::L1,
        latent: None,
    })
}

pub fn representative_linear_attn_scan_static() -> OpStatic {
    OpStatic::LinearAttnScan(LinearAttnScanStatic {
        kind: LinearAttnKind::GatedDeltaNet,
        h_local: 16,
        d: 128,
        dv: 128,
        chunk: 64,
        mode: ScanMode::Chunked,
    })
}

pub fn representative_elementwise_static() -> OpStatic {
    OpStatic::Elementwise(ElementwiseStatic {
        t_bucket: 16,
        dims: vec![16, 4096],
        dtypes: vec![DType::F16],
        fused_with: None,
    })
}

pub fn representative_sampling_static(method: SamplingMethod) -> OpStatic {
    OpStatic::Sampling(SamplingStatic {
        s_bucket: 4,
        v: 32000,
        q_bucket: 16,
        method,
    })
}

pub fn representative_collectives_static() -> OpStatic {
    OpStatic::Collectives(CollectivesStatic {
        bytes_bucket: 65536,
        dtype: DType::F16,
        transport: P2pTransport::Direct,
    })
}

pub fn representative_static_for_op(op: OpId) -> OpStatic {
    match op {
        OpId::Matmul => representative_matmul_static(),
        OpId::MoeRoute | OpId::MoeFfn => representative_moe_ffn_static(),
        OpId::Attention => representative_attention_static(AttentionMask::Causal, 16),
        OpId::StateWriteKv => representative_state_write_kv_static(),
        OpId::CausalConv1d | OpId::LinearAttnScan => representative_linear_attn_scan_static(),
        OpId::EmbedGather
        | OpId::NgramGather
        | OpId::QuantAct
        | OpId::Cast
        | OpId::Copy
        | OpId::GatherRows
        | OpId::ScatterAddRows
        | OpId::Split
        | OpId::Concat
        | OpId::Norm
        | OpId::ResidualAdd
        | OpId::ActMul
        | OpId::Activation
        | OpId::LogitSoftcap
        | OpId::Rope => representative_elementwise_static(),
        OpId::LogitsPostprocess => {
            representative_sampling_static(SamplingMethod::LogitsPostprocess)
        }
        OpId::Sample => representative_sampling_static(SamplingMethod::InverseCdfSample),
        OpId::Verify => representative_sampling_static(SamplingMethod::VerifyRejection),
        OpId::AllReduce
        | OpId::AllGather
        | OpId::ReduceScatter
        | OpId::AllToAll
        | OpId::Send
        | OpId::Recv
        | OpId::Barrier => representative_collectives_static(),
    }
}
