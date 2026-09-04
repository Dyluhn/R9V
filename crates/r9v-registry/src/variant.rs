// SPDX-License-Identifier: Apache-2.0
//! Kernel variant keys, static parameter hashing, and stable variant hashing (Spec 4 §3).

use r9v_common::hash::xxh3_64;
use serde::{Deserialize, Serialize};

use crate::types::{ArchName, OpId, OpStatic, TileConfig, VariantHash};

/// Unique identifier for a compiled kernel variant (Spec 4 §3).
///
/// Combines the logical operation, target architecture, generator version,
/// family-specific static dimensions, and autotuned tile configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VariantKey {
    /// Operation identifier (Spec 1 §4, Spec 4 §3).
    pub op: OpId,
    /// Target architecture name (Spec 4 §3).
    pub arch: ArchName,
    /// Generator version; bumping this version invalidates all cached tunes and bundles (Spec 4 §3).
    pub gen_version: u32,
    /// Static shape, dtype, layout, and epilogue parameters (Spec 4 §3).
    pub static_params: OpStatic,
    /// Autotuned tile and wave launch configuration (Spec 4 §3).
    pub config: TileConfig,
}

impl VariantKey {
    /// Creates a new variant key (Spec 4 §3).
    pub fn new(
        op: OpId,
        arch: ArchName,
        gen_version: u32,
        static_params: OpStatic,
        config: TileConfig,
    ) -> Self {
        Self {
            op,
            arch,
            gen_version,
            static_params,
            config,
        }
    }

    /// Computes the stable 64-bit variant hash (Spec 4 §3).
    ///
    /// The variant key is serialized into canonical JSON with deterministic key ordering
    /// before hashing with `xxh3_64`.
    pub fn hash(&self) -> VariantHash {
        variant_hash(self)
    }

    /// Computes the static hash for the variant's static parameters (Spec 4 §6.2, §7).
    pub fn static_hash(&self) -> u64 {
        static_hash(&self.static_params)
    }
}

// DECISION(A3.1): variant hashing serializes VariantKey into canonical JSON with sorted keys before feeding into xxh3_64; rejected raw memory casting or non-canonical serde formats because cross-compiler and cross-architecture reproducibility require identical serialized bytes. Spec 4 §3.
/// Computes the 64-bit XXH3 variant hash for a given [`VariantKey`] (Spec 4 §3).
pub fn variant_hash(key: &VariantKey) -> VariantHash {
    let serialized =
        serde_json::to_vec(key).expect("VariantKey serialization into JSON must never fail");
    VariantHash::new(xxh3_64(&serialized))
}

// DECISION(A3.1): static_hash serializes OpStatic into canonical JSON with sorted keys before feeding into xxh3_64; rejected ad-hoc hash algorithms because consistency with variant_hash and Spec 4 §6.2/§7 requires bit-identical hashes across tools. Spec 4 §6.2, §7.
/// Computes the 64-bit XXH3 hash for an [`OpStatic`] parameter block (Spec 4 §6.2, §7).
pub fn static_hash(static_params: &OpStatic) -> u64 {
    let serialized = serde_json::to_vec(static_params)
        .expect("OpStatic serialization into JSON must never fail");
    xxh3_64(&serialized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use r9v_ir::{
        AttentionMask, CacheScaleGranularity, DType, Epilogue, LayoutId, LinearAttnKind,
        QuantScheme,
    };

    #[test]
    fn test_all_10_op_static_families_hashing() {
        let arch = ArchName::from("gfx942");

        // 1. Matmul
        let matmul_s = OpStatic::Matmul(MatmulStatic {
            m_bucket: 64,
            n: 1024,
            k: 512,
            w_dtype: DType::F16,
            w_scheme: QuantScheme::None,
            w_layout: LayoutId::CONTIGUOUS,
            in_dtype: DType::F16,
            act_scheme: QuantScheme::None,
            out_dtype: DType::F16,
            epilogue: Epilogue::None,
            transpose_w: false,
            interleave: false,
            sparse: false,
        });
        let k1 = VariantKey::new(
            OpId::Matmul,
            arch.clone(),
            1,
            matmul_s.clone(),
            TileConfig::new(64, 64, 32),
        );
        assert_eq!(variant_hash(&k1), variant_hash(&k1));
        assert_ne!(static_hash(&matmul_s), 0);

        // 2. MoeFfn
        let moe_s = OpStatic::MoeFfn(MoeFfnStatic {
            t_bucket: 16,
            e_local: 8,
            k_topk: 2,
            dm: 2048,
            dff: 5632,
            schemes: vec![QuantScheme::None, QuantScheme::None],
            w_dtype: DType::F16,
            in_dtype: DType::F16,
            act_scheme: QuantScheme::None,
            act: r9v_ir::ActivationKind::Silu,
            out_dtype: DType::F16,
            shared_experts: 0,
            placement_kind: PlacementKind::Device,
        });
        let k2 = VariantKey::new(
            OpId::MoeFfn,
            arch.clone(),
            1,
            moe_s.clone(),
            TileConfig::new(32, 32, 16),
        );
        assert_eq!(variant_hash(&k2), variant_hash(&k2));
        assert_ne!(static_hash(&moe_s), 0);

        // 3. Attention
        let attn_s = OpStatic::Attention(AttentionStatic {
            q_bucket: 128,
            h_local: 32,
            hkv_local: 8,
            d: 128,
            dv: 128,
            cache_dtype: DType::F16,
            attention_layout: LayoutId::CONTIGUOUS,
            mask_kind: AttentionMask::Causal,
            softmax_scale_bits: (1.0f32 / 128.0f32.sqrt()).to_bits(),
            out_dtype: DType::F16,
            mla: None,
            softcap_bits: None,
            sinks: 0,
        });
        let k3 = VariantKey::new(
            OpId::Attention,
            arch.clone(),
            1,
            attn_s.clone(),
            TileConfig::new(16, 16, 16),
        );
        assert_eq!(variant_hash(&k3), variant_hash(&k3));
        assert_ne!(static_hash(&attn_s), 0);

        // 4. StateWriteKv
        let sw_s = OpStatic::StateWriteKv(StateWriteKvStatic {
            hkv_local: 8,
            d: 128,
            dv: 128,
            in_dtype: DType::F16,
            cache_dtype: DType::F16,
            scale_granularity: CacheScaleGranularity::PerTokenHead,
            attention_layout: LayoutId::CONTIGUOUS,
            latent: None,
        });
        let k4 = VariantKey::new(
            OpId::StateWriteKv,
            arch.clone(),
            1,
            sw_s.clone(),
            TileConfig::new(64, 1, 1),
        );
        assert_eq!(variant_hash(&k4), variant_hash(&k4));
        assert_ne!(static_hash(&sw_s), 0);

        // 5. LinearAttnScan
        let lin_s = OpStatic::LinearAttnScan(LinearAttnScanStatic {
            kind: LinearAttnKind::GatedDeltaNet,
            h_local: 16,
            d: 64,
            dv: 128,
            chunk: 64,
            mode: ScanMode::Chunked,
            in_dtype: DType::F16,
            out_dtype: DType::F16,
        });
        let k5 = VariantKey::new(
            OpId::LinearAttnScan,
            arch.clone(),
            1,
            lin_s.clone(),
            TileConfig::new(64, 64, 1),
        );
        assert_eq!(variant_hash(&k5), variant_hash(&k5));
        assert_ne!(static_hash(&lin_s), 0);

        // 6. MoeRoute
        let moe_route_s = OpStatic::MoeRoute(MoeRouteStatic {
            t_bucket: 16,
            e_total: 8,
            top_k: 2,
            scoring: r9v_ir::MoeScoring::Softmax,
            renormalize: true,
            group: None,
            scale_bits: 1.0f32.to_bits(),
            has_bias: false,
        });
        let k_route = VariantKey::new(
            OpId::MoeRoute,
            arch.clone(),
            1,
            moe_route_s.clone(),
            TileConfig::new(256, 1, 1),
        );
        assert_eq!(variant_hash(&k_route), variant_hash(&k_route));
        assert_ne!(static_hash(&moe_route_s), 0);

        // 7. CausalConv1d
        let conv_s = OpStatic::CausalConv1d(CausalConv1dStatic {
            t_bucket: 16,
            channels: 2048,
            kernel: 4,
            act: r9v_ir::ConvActivation::Silu,
            x_dtype: DType::F16,
            w_dtype: DType::F16,
            out_dtype: DType::F16,
            has_bias: false,
        });
        let k_conv = VariantKey::new(
            OpId::CausalConv1d,
            arch.clone(),
            1,
            conv_s.clone(),
            TileConfig::new(256, 1, 1),
        );
        assert_eq!(variant_hash(&k_conv), variant_hash(&k_conv));
        assert_ne!(static_hash(&conv_s), 0);

        // 8. Elementwise
        let elem_s = OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 128,
            fused_with: None,
            op_params: ElementwiseParams::ResidualAdd(ResidualAddStatic {
                out_dtype: DType::F16,
                scale_bits: 1.0f32.to_bits(),
                n: 4096,
            }),
        });
        let k6 = VariantKey::new(
            OpId::ResidualAdd,
            arch.clone(),
            1,
            elem_s.clone(),
            TileConfig::new(256, 1, 1),
        );
        assert_eq!(variant_hash(&k6), variant_hash(&k6));
        assert_ne!(static_hash(&elem_s), 0);

        // 9. Sampling
        let samp_s = OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
            s_bucket: 1,
            v: 32000,
            rng: r9v_ir::RngAlgorithm::Philox4x32,
        }));
        let k7 = VariantKey::new(
            OpId::Sample,
            arch.clone(),
            1,
            samp_s.clone(),
            TileConfig::new(1024, 1, 1),
        );
        assert_eq!(variant_hash(&k7), variant_hash(&k7));
        assert_ne!(static_hash(&samp_s), 0);

        // 10. Collectives
        let coll_s = OpStatic::Collectives(CollectivesStatic::AllReduce(AllReduceStatic {
            group: 0,
            rank: 0,
            world: 1,
            dtype: DType::Bf16,
            reduce_in: DType::F32,
            reduction_op: r9v_ir::ReduceOp::Sum,
            transport: r9v_ir::P2pTransport::Direct,
            bytes_bucket: 1048576,
        }));
        let k8 = VariantKey::new(
            OpId::AllReduce,
            arch,
            1,
            coll_s.clone(),
            TileConfig::new(1, 1, 1),
        );
        assert_eq!(variant_hash(&k8), variant_hash(&k8));
        assert_ne!(static_hash(&coll_s), 0);

        // All 10 hashes must be distinct
        let hashes = [
            k1.hash(),
            k2.hash(),
            k3.hash(),
            k4.hash(),
            k5.hash(),
            k_route.hash(),
            k_conv.hash(),
            k6.hash(),
            k7.hash(),
            k8.hash(),
        ];
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "hashes at {i} and {j} must differ");
            }
        }
    }

    #[test]
    fn test_variant_key_serde_roundtrip() {
        let op_s = OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
            s_bucket: 4,
            v: 128000,
            rng: r9v_ir::RngAlgorithm::Philox4x32,
        }));
        let key = VariantKey::new(
            OpId::Sample,
            ArchName::from("gfx942"),
            2,
            op_s,
            TileConfig::new(256, 1, 1),
        );

        let json = serde_json::to_string(&key).expect("serialize");
        let deserialized: VariantKey = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(key, deserialized);
        assert_eq!(key.hash(), deserialized.hash());
    }

    #[test]
    fn test_new_statics_serde_roundtrip_and_hash_stable() {
        let cases = [
            OpStatic::MoeRoute(MoeRouteStatic {
                t_bucket: 16,
                e_total: 8,
                top_k: 2,
                scoring: r9v_ir::MoeScoring::Sigmoid,
                renormalize: false,
                group: Some(r9v_ir::MoeGroup {
                    n_group: 2,
                    topk_group: 1,
                }),
                scale_bits: 2.0f32.to_bits(),
                has_bias: true,
            }),
            OpStatic::CausalConv1d(CausalConv1dStatic {
                t_bucket: 8,
                channels: 512,
                kernel: 3,
                act: r9v_ir::ConvActivation::Identity,
                x_dtype: r9v_ir::DType::F32,
                w_dtype: r9v_ir::DType::F32,
                out_dtype: r9v_ir::DType::F32,
                has_bias: true,
            }),
            OpStatic::Elementwise(ElementwiseStatic {
                t_bucket: 8,
                fused_with: None,
                op_params: ElementwiseParams::Norm(NormStatic {
                    kind: r9v_ir::NormKind::Layer,
                    eps_bits: 1e-6f32.to_bits(),
                    axis: r9v_ir::NormAxis::Head(8),
                    weight_offset_bits: 1.0f32.to_bits(),
                    in_dtype: r9v_ir::DType::F16,
                    out_dtype: r9v_ir::DType::F16,
                    n: 1024,
                }),
            }),
            OpStatic::Sampling(SamplingStatic::Verify(VerifyStatic::typical(
                2, 32000, 4, 0.05, 1.25, true,
            ))),
            OpStatic::Collectives(CollectivesStatic::Send(SendStatic {
                group: 2,
                rank: 1,
                world: 4,
                peer: 2,
                dtype: r9v_ir::DType::F16,
                transport: r9v_ir::P2pTransport::Direct,
                bytes_bucket: 4096,
            })),
        ];

        for op_s in cases {
            let json = serde_json::to_string(&op_s).expect("static must serialize");
            let back: OpStatic = serde_json::from_str(&json).expect("static must deserialize");
            assert_eq!(op_s, back, "serde roundtrip must preserve {json}");
            assert_eq!(
                static_hash(&op_s),
                static_hash(&back),
                "hash must survive serde roundtrip"
            );
        }
    }

    #[test]
    fn test_verify_method_closed_set_and_ieee_bit_preservation() {
        let eps = 0.05f32;
        let delta = 1.25f32;
        let method = VerifyMethodStatic::typical(eps, delta);
        assert_eq!(method.eps(), Some(eps));
        assert_eq!(method.delta(), Some(delta));

        // Slightly different delta must yield distinct static hash and variant hash
        let method2 = VerifyMethodStatic::typical(eps, 1.2500001f32);
        assert_ne!(method, method2);

        let mk = |m| {
            OpStatic::Sampling(SamplingStatic::Verify(VerifyStatic {
                s_bucket: 2,
                v: 32000,
                q_bucket: 1,
                method: m,
                tree: false,
            }))
        };
        assert_ne!(static_hash(&mk(method)), static_hash(&mk(method2)));

        // From VerifyMethod conversion preserves bits
        let vm = r9v_ir::VerifyMethod::TypicalAcceptance { eps, delta };
        let from_vm = VerifyMethodStatic::from_ir(&vm);
        assert_eq!(from_vm, method);
        assert_eq!(from_vm.to_ir(), vm);

        let vm_greedy = r9v_ir::VerifyMethod::Greedy;
        assert_eq!(
            VerifyMethodStatic::from_ir(&vm_greedy),
            VerifyMethodStatic::Greedy
        );

        let vm_rej = r9v_ir::VerifyMethod::Rejection;
        assert_eq!(
            VerifyMethodStatic::from_ir(&vm_rej),
            VerifyMethodStatic::Rejection
        );
    }
}
