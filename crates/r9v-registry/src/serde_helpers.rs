// SPDX-License-Identifier: Apache-2.0
//! Serde serialization and deserialization adapters for Op IR types (Spec 1, Spec 4 §3).

use r9v_ir::{
    ActivationKind, AttentionMask, CacheScaleGranularity, ConvActivation, CopyKind, DType,
    Epilogue, HashId, LayoutId, LinearAttnKind, MoeGroup, MoeScoring, NgramCombine, NgramSource,
    NormAxis, NormKind, P2pTransport, Placement, QuantScheme, ReduceOp, RngAlgorithm, RopeStyle,
    SchemeId, Smoothing,
};
use serde::de::{self, Deserializer};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};

pub mod serde_p2p_transport {
    use super::*;

    pub fn serialize<S: Serializer>(transport: &P2pTransport, s: S) -> Result<S::Ok, S::Error> {
        let name = match transport {
            P2pTransport::Direct => "direct",
            P2pTransport::HostStaged => "host_staged",
        };
        s.serialize_str(name)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<P2pTransport, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "direct" => Ok(P2pTransport::Direct),
            "host_staged" => Ok(P2pTransport::HostStaged),
            other => Err(de::Error::custom(format!(
                "unknown P2pTransport '{other}', expected 'direct' or 'host_staged'"
            ))),
        }
    }
}

pub mod serde_dtype {
    use super::*;

    pub fn serialize<S: Serializer>(dtype: &DType, s: S) -> Result<S::Ok, S::Error> {
        let name = match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::E4m3 => "e4m3",
            DType::E5m2 => "e5m2",
            DType::I8 => "i8",
            DType::I4 => "i4",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::Bool => "bool",
        };
        s.serialize_str(name)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DType, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "f32" => Ok(DType::F32),
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::Bf16),
            "e4m3" => Ok(DType::E4m3),
            "e5m2" => Ok(DType::E5m2),
            "i8" => Ok(DType::I8),
            "i4" => Ok(DType::I4),
            "i32" => Ok(DType::I32),
            "u32" => Ok(DType::U32),
            "bool" => Ok(DType::Bool),
            other => Err(de::Error::custom(format!("unknown DType '{other}'"))),
        }
    }
}

pub mod serde_dtype_vec {
    use super::*;

    pub fn serialize<S: Serializer>(vec: &[DType], s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(vec.len()))?;
        for item in vec {
            let name = match item {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::Bf16 => "bf16",
                DType::E4m3 => "e4m3",
                DType::E5m2 => "e5m2",
                DType::I8 => "i8",
                DType::I4 => "i4",
                DType::I32 => "i32",
                DType::U32 => "u32",
                DType::Bool => "bool",
            };
            seq.serialize_element(name)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<DType>, D::Error> {
        let raw_list = Vec::<String>::deserialize(d)?;
        let mut result = Vec::with_capacity(raw_list.len());
        for s in raw_list {
            let dt = match s.to_ascii_lowercase().as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "bf16" => DType::Bf16,
                "e4m3" => DType::E4m3,
                "e5m2" => DType::E5m2,
                "i8" => DType::I8,
                "i4" => DType::I4,
                "i32" => DType::I32,
                "u32" => DType::U32,
                "bool" => DType::Bool,
                other => return Err(de::Error::custom(format!("unknown DType '{other}'"))),
            };
            result.push(dt);
        }
        Ok(result)
    }
}

pub mod serde_quant_scheme {
    use super::*;

    pub fn serialize<S: Serializer>(scheme: &QuantScheme, s: S) -> Result<S::Ok, S::Error> {
        match scheme {
            QuantScheme::None => s.serialize_str("none"),
            QuantScheme::PerRow => s.serialize_str("per_row"),
            QuantScheme::Scheme(id) => s.serialize_str(&format!("scheme:{}", id.as_u64())),
            QuantScheme::PerToken => s.serialize_str("per_token"),
            QuantScheme::PerBlock32 => s.serialize_str("per_block32"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<QuantScheme, D::Error> {
        let s = String::deserialize(d)?;
        if s.eq_ignore_ascii_case("none") {
            Ok(QuantScheme::None)
        } else if s.eq_ignore_ascii_case("per_row") {
            Ok(QuantScheme::PerRow)
        } else if s.eq_ignore_ascii_case("per_token") {
            Ok(QuantScheme::PerToken)
        } else if s.eq_ignore_ascii_case("per_block32") {
            Ok(QuantScheme::PerBlock32)
        } else if let Some(code_str) = s.strip_prefix("scheme:") {
            let code = code_str.parse::<u64>().map_err(de::Error::custom)?;
            Ok(QuantScheme::Scheme(SchemeId::new(code)))
        } else {
            Err(de::Error::custom(format!("unknown QuantScheme '{s}'")))
        }
    }
}

/// Canonical nullable dtype encoding for optional input dtypes (`None` for an
/// absent optional input, the stable snake_case dtype name otherwise).
/// Uses the same names as [`serde_dtype`] so `Some(d)` round-trips through it.
pub mod serde_opt_dtype {
    use super::*;

    fn name(dtype: &DType) -> &'static str {
        match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::E4m3 => "e4m3",
            DType::E5m2 => "e5m2",
            DType::I8 => "i8",
            DType::I4 => "i4",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::Bool => "bool",
        }
    }

    pub fn serialize<S: Serializer>(opt: &Option<DType>, s: S) -> Result<S::Ok, S::Error> {
        match opt {
            Some(dtype) => s.serialize_str(name(dtype)),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<DType>, D::Error> {
        let opt = Option::<String>::deserialize(d)?;
        match opt.as_deref() {
            None => Ok(None),
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "f32" => Ok(Some(DType::F32)),
                "f16" => Ok(Some(DType::F16)),
                "bf16" => Ok(Some(DType::Bf16)),
                "e4m3" => Ok(Some(DType::E4m3)),
                "e5m2" => Ok(Some(DType::E5m2)),
                "i8" => Ok(Some(DType::I8)),
                "i4" => Ok(Some(DType::I4)),
                "i32" => Ok(Some(DType::I32)),
                "u32" => Ok(Some(DType::U32)),
                "bool" => Ok(Some(DType::Bool)),
                other => Err(de::Error::custom(format!("unknown DType '{other}'"))),
            },
        }
    }
}

pub mod serde_layout_id {
    use super::*;

    pub fn serialize<S: Serializer>(layout: &LayoutId, s: S) -> Result<S::Ok, S::Error> {
        let name = match *layout {
            LayoutId::CONTIGUOUS => "contiguous",
            LayoutId::L0 => "l0",
            LayoutId::L1 => "l1",
            LayoutId::L1S => "l1s",
            LayoutId::ATTENTION_GFX1201 => "attention_gfx1201",
            _ => return s.serialize_u64(layout.as_u64()),
        };
        s.serialize_str(name)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<LayoutId, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum LayoutRepr {
            Num(u64),
            Str(String),
        }

        match LayoutRepr::deserialize(d)? {
            LayoutRepr::Num(code) => Ok(LayoutId::new(code)),
            LayoutRepr::Str(s) => match s.to_ascii_lowercase().as_str() {
                "contiguous" => Ok(LayoutId::CONTIGUOUS),
                "l0" => Ok(LayoutId::L0),
                "l1" => Ok(LayoutId::L1),
                "l1s" => Ok(LayoutId::L1S),
                "attention_gfx1201" => Ok(LayoutId::ATTENTION_GFX1201),
                other => {
                    if let Ok(val) = other.parse::<u64>() {
                        Ok(LayoutId::new(val))
                    } else {
                        Err(de::Error::custom(format!("unknown LayoutId '{other}'")))
                    }
                }
            },
        }
    }
}

pub mod serde_epilogue {
    use super::*;

    pub fn serialize<S: Serializer>(ep: &Epilogue, s: S) -> Result<S::Ok, S::Error> {
        match ep {
            Epilogue::None => s.serialize_str("none"),
            Epilogue::Bias => s.serialize_str("bias"),
            Epilogue::Residual => s.serialize_str("residual"),
            Epilogue::Act(ActivationKind::Silu) => s.serialize_str("act:silu"),
            Epilogue::Act(ActivationKind::Gelu) => s.serialize_str("act:gelu"),
            Epilogue::Act(ActivationKind::GeluTanh) => s.serialize_str("act:gelu_tanh"),
            Epilogue::Act(ActivationKind::Relu2) => s.serialize_str("act:relu2"),
            Epilogue::Act(ActivationKind::Identity) => s.serialize_str("act:identity"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Epilogue, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Epilogue::None),
            "bias" => Ok(Epilogue::Bias),
            "residual" => Ok(Epilogue::Residual),
            "act:silu" => Ok(Epilogue::Act(ActivationKind::Silu)),
            "act:gelu" => Ok(Epilogue::Act(ActivationKind::Gelu)),
            "act:gelu_tanh" => Ok(Epilogue::Act(ActivationKind::GeluTanh)),
            "act:relu2" => Ok(Epilogue::Act(ActivationKind::Relu2)),
            "act:identity" => Ok(Epilogue::Act(ActivationKind::Identity)),
            other => Err(de::Error::custom(format!("unknown Epilogue '{other}'"))),
        }
    }
}

pub mod serde_placement {
    use super::*;

    pub fn serialize<S: Serializer>(p: &Placement, s: S) -> Result<S::Ok, S::Error> {
        match p {
            Placement::Device { rank } => s.serialize_str(&format!("device:{rank}")),
            Placement::Host => s.serialize_str("host"),
            Placement::Tiered => s.serialize_str("tiered"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Placement, D::Error> {
        let s = String::deserialize(d)?;
        if s.eq_ignore_ascii_case("host") {
            Ok(Placement::Host)
        } else if s.eq_ignore_ascii_case("tiered") {
            Ok(Placement::Tiered)
        } else if let Some(rank_str) = s.strip_prefix("device:") {
            let rank = rank_str.parse::<u32>().map_err(de::Error::custom)?;
            Ok(Placement::Device { rank })
        } else {
            Err(de::Error::custom(format!("unknown Placement '{s}'")))
        }
    }
}

pub mod serde_attention_mask {
    use super::*;

    pub fn serialize<S: Serializer>(m: &AttentionMask, s: S) -> Result<S::Ok, S::Error> {
        match m {
            AttentionMask::Causal => s.serialize_str("causal"),
            AttentionMask::CausalWindow(w) => s.serialize_str(&format!("causal_window:{w}")),
            AttentionMask::Tree => s.serialize_str("tree"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<AttentionMask, D::Error> {
        let s = String::deserialize(d)?;
        if s.eq_ignore_ascii_case("causal") {
            Ok(AttentionMask::Causal)
        } else if s.eq_ignore_ascii_case("tree") {
            Ok(AttentionMask::Tree)
        } else if let Some(w_str) = s.strip_prefix("causal_window:") {
            let w = w_str.parse::<u32>().map_err(de::Error::custom)?;
            Ok(AttentionMask::CausalWindow(w))
        } else {
            Err(de::Error::custom(format!("unknown AttentionMask '{s}'")))
        }
    }
}

pub mod serde_linear_attn_kind {
    use super::*;

    pub fn serialize<S: Serializer>(k: &LinearAttnKind, s: S) -> Result<S::Ok, S::Error> {
        let name = match k {
            LinearAttnKind::GatedDeltaNet => "gated_delta_net",
            LinearAttnKind::GLA => "gla",
            LinearAttnKind::Mamba2 => "mamba2",
        };
        s.serialize_str(name)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<LinearAttnKind, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "gated_delta_net" => Ok(LinearAttnKind::GatedDeltaNet),
            "gla" => Ok(LinearAttnKind::GLA),
            "mamba2" => Ok(LinearAttnKind::Mamba2),
            other => Err(de::Error::custom(format!(
                "unknown LinearAttnKind '{other}'"
            ))),
        }
    }
}

pub mod serde_moe_scoring {
    use super::*;

    pub fn serialize<S: Serializer>(s_mode: &MoeScoring, s: S) -> Result<S::Ok, S::Error> {
        match s_mode {
            MoeScoring::Softmax => s.serialize_str("softmax"),
            MoeScoring::Sigmoid => s.serialize_str("sigmoid"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<MoeScoring, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "softmax" => Ok(MoeScoring::Softmax),
            "sigmoid" => Ok(MoeScoring::Sigmoid),
            other => Err(de::Error::custom(format!("unknown MoeScoring '{other}'"))),
        }
    }
}

pub mod serde_opt_moe_group {
    use super::*;

    #[derive(Serialize, Deserialize)]
    struct MoeGroupRepr {
        n_group: u32,
        topk_group: u32,
    }

    pub fn serialize<S: Serializer>(opt: &Option<MoeGroup>, s: S) -> Result<S::Ok, S::Error> {
        match opt {
            Some(g) => {
                let repr = MoeGroupRepr {
                    n_group: g.n_group,
                    topk_group: g.topk_group,
                };
                repr.serialize(s)
            }
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<MoeGroup>, D::Error> {
        let opt = Option::<MoeGroupRepr>::deserialize(d)?;
        Ok(opt.map(|g| MoeGroup {
            n_group: g.n_group,
            topk_group: g.topk_group,
        }))
    }
}

pub mod serde_conv_activation {
    use super::*;

    pub fn serialize<S: Serializer>(act: &ConvActivation, s: S) -> Result<S::Ok, S::Error> {
        match act {
            ConvActivation::Silu => s.serialize_str("silu"),
            ConvActivation::Identity => s.serialize_str("identity"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ConvActivation, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "silu" => Ok(ConvActivation::Silu),
            "identity" => Ok(ConvActivation::Identity),
            other => Err(de::Error::custom(format!(
                "unknown ConvActivation '{other}'"
            ))),
        }
    }
}

pub mod serde_norm_kind {
    use super::*;

    pub fn serialize<S: Serializer>(kind: &NormKind, s: S) -> Result<S::Ok, S::Error> {
        match kind {
            NormKind::Rms => s.serialize_str("rms"),
            NormKind::Layer => s.serialize_str("layer"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<NormKind, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "rms" => Ok(NormKind::Rms),
            "layer" => Ok(NormKind::Layer),
            other => Err(de::Error::custom(format!("unknown NormKind '{other}'"))),
        }
    }
}

pub mod serde_norm_axis {
    use super::*;

    pub fn serialize<S: Serializer>(axis: &NormAxis, s: S) -> Result<S::Ok, S::Error> {
        match axis {
            NormAxis::Last => s.serialize_str("last"),
            NormAxis::Head(h) => s.serialize_str(&format!("head:{h}")),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<NormAxis, D::Error> {
        let s = String::deserialize(d)?;
        if s.eq_ignore_ascii_case("last") {
            Ok(NormAxis::Last)
        } else if let Some(h_str) = s.strip_prefix("head:") {
            let h = h_str.parse::<u32>().map_err(de::Error::custom)?;
            Ok(NormAxis::Head(h))
        } else {
            Err(de::Error::custom(format!("unknown NormAxis '{s}'")))
        }
    }
}

pub mod serde_activation_kind {
    use super::*;

    pub fn serialize<S: Serializer>(act: &ActivationKind, s: S) -> Result<S::Ok, S::Error> {
        match act {
            ActivationKind::Silu => s.serialize_str("silu"),
            ActivationKind::Gelu => s.serialize_str("gelu"),
            ActivationKind::GeluTanh => s.serialize_str("gelu_tanh"),
            ActivationKind::Relu2 => s.serialize_str("relu2"),
            ActivationKind::Identity => s.serialize_str("identity"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ActivationKind, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "silu" => Ok(ActivationKind::Silu),
            "gelu" => Ok(ActivationKind::Gelu),
            "gelu_tanh" => Ok(ActivationKind::GeluTanh),
            "relu2" => Ok(ActivationKind::Relu2),
            "identity" => Ok(ActivationKind::Identity),
            other => Err(de::Error::custom(format!(
                "unknown ActivationKind '{other}'"
            ))),
        }
    }
}

pub mod serde_rope_style {
    use super::*;

    pub fn serialize<S: Serializer>(style: &RopeStyle, s: S) -> Result<S::Ok, S::Error> {
        match style {
            RopeStyle::Neox => s.serialize_str("neox"),
            RopeStyle::Interleaved => s.serialize_str("interleaved"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<RopeStyle, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "neox" => Ok(RopeStyle::Neox),
            "interleaved" => Ok(RopeStyle::Interleaved),
            other => Err(de::Error::custom(format!("unknown RopeStyle '{other}'"))),
        }
    }
}

pub mod serde_smoothing {
    use super::*;

    pub fn serialize<S: Serializer>(sm: &Smoothing, s: S) -> Result<S::Ok, S::Error> {
        match sm {
            Smoothing::None => s.serialize_str("none"),
            Smoothing::Folded => s.serialize_str("folded"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Smoothing, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Smoothing::None),
            "folded" => Ok(Smoothing::Folded),
            other => Err(de::Error::custom(format!("unknown Smoothing '{other}'"))),
        }
    }
}

pub mod serde_ngram_source {
    use super::*;

    pub fn serialize<S: Serializer>(src: &NgramSource, s: S) -> Result<S::Ok, S::Error> {
        match src {
            NgramSource::Staged => s.serialize_str("staged"),
            NgramSource::Device => s.serialize_str("device"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<NgramSource, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "staged" => Ok(NgramSource::Staged),
            "device" => Ok(NgramSource::Device),
            other => Err(de::Error::custom(format!("unknown NgramSource '{other}'"))),
        }
    }
}

pub mod serde_ngram_combine {
    use super::*;

    pub fn serialize<S: Serializer>(comb: &NgramCombine, s: S) -> Result<S::Ok, S::Error> {
        match comb {
            NgramCombine::Concat => s.serialize_str("concat"),
            NgramCombine::Sum => s.serialize_str("sum"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<NgramCombine, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "concat" => Ok(NgramCombine::Concat),
            "sum" => Ok(NgramCombine::Sum),
            other => Err(de::Error::custom(format!("unknown NgramCombine '{other}'"))),
        }
    }
}

pub mod serde_copy_kind {
    use super::*;

    pub fn serialize<S: Serializer>(kind: &CopyKind, s: S) -> Result<S::Ok, S::Error> {
        match kind {
            CopyKind::Contiguize => s.serialize_str("contiguize"),
            CopyKind::DeviceToDevice => s.serialize_str("device_to_device"),
            CopyKind::HostToDevice => s.serialize_str("host_to_device"),
            CopyKind::DeviceToHost => s.serialize_str("device_to_host"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<CopyKind, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "contiguize" => Ok(CopyKind::Contiguize),
            "device_to_device" => Ok(CopyKind::DeviceToDevice),
            "host_to_device" => Ok(CopyKind::HostToDevice),
            "device_to_host" => Ok(CopyKind::DeviceToHost),
            other => Err(de::Error::custom(format!("unknown CopyKind '{other}'"))),
        }
    }
}

pub mod serde_opt_reduce_op {
    use super::*;

    pub fn serialize<S: Serializer>(opt: &Option<ReduceOp>, s: S) -> Result<S::Ok, S::Error> {
        match opt {
            Some(ReduceOp::Sum) => s.serialize_str("sum"),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<ReduceOp>, D::Error> {
        let opt = Option::<String>::deserialize(d)?;
        match opt.as_deref() {
            Some(s) if s.eq_ignore_ascii_case("sum") => Ok(Some(ReduceOp::Sum)),
            None => Ok(None),
            Some(other) => Err(de::Error::custom(format!("unknown ReduceOp '{other}'"))),
        }
    }
}

pub mod serde_rng_algorithm {
    use super::*;

    pub fn serialize<S: Serializer>(rng: &RngAlgorithm, s: S) -> Result<S::Ok, S::Error> {
        match rng {
            RngAlgorithm::Philox4x32 => s.serialize_str("philox4x32"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<RngAlgorithm, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "philox4x32" => Ok(RngAlgorithm::Philox4x32),
            other => Err(de::Error::custom(format!("unknown RngAlgorithm '{other}'"))),
        }
    }
}

pub mod serde_cache_scale_granularity {
    use super::*;

    pub fn serialize<S: Serializer>(
        granularity: &CacheScaleGranularity,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match granularity {
            CacheScaleGranularity::PerTokenHead => s.serialize_str("per_token_head"),
            CacheScaleGranularity::PerBlock => s.serialize_str("per_block"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<CacheScaleGranularity, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "per_token_head" => Ok(CacheScaleGranularity::PerTokenHead),
            "per_block" => Ok(CacheScaleGranularity::PerBlock),
            other => Err(de::Error::custom(format!(
                "unknown CacheScaleGranularity '{other}'"
            ))),
        }
    }
}

pub mod serde_reduce_op {
    use super::*;

    pub fn serialize<S: Serializer>(op: &ReduceOp, s: S) -> Result<S::Ok, S::Error> {
        match op {
            ReduceOp::Sum => s.serialize_str("sum"),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ReduceOp, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "sum" => Ok(ReduceOp::Sum),
            other => Err(de::Error::custom(format!("unknown ReduceOp '{other}'"))),
        }
    }
}

pub mod serde_hash_id {
    use super::*;

    pub fn serialize<S: Serializer>(hash: &HashId, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(hash.as_u64())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<HashId, D::Error> {
        let val = u64::deserialize(d)?;
        Ok(HashId::new(val))
    }
}
