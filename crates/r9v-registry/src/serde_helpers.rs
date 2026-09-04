// SPDX-License-Identifier: Apache-2.0
//! Serde serialization and deserialization adapters for Op IR types (Spec 1, Spec 4 §3).

use r9v_ir::{
    ActivationKind, AttentionMask, DType, Epilogue, LayoutId, LinearAttnKind, P2pTransport,
    Placement, QuantScheme, SchemeId,
};
use serde::de::{self, Deserializer};
use serde::ser::{SerializeSeq, Serializer};
use serde::Deserialize;

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

pub mod serde_quant_scheme_vec {
    use super::*;

    pub fn serialize<S: Serializer>(vec: &[QuantScheme], s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(vec.len()))?;
        for item in vec {
            match item {
                QuantScheme::None => seq.serialize_element("none")?,
                QuantScheme::PerRow => seq.serialize_element("per_row")?,
                QuantScheme::Scheme(id) => {
                    seq.serialize_element(&format!("scheme:{}", id.as_u64()))?
                }
                QuantScheme::PerToken => seq.serialize_element("per_token")?,
                QuantScheme::PerBlock32 => seq.serialize_element("per_block32")?,
            }
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<QuantScheme>, D::Error> {
        let raw_list = Vec::<String>::deserialize(d)?;
        let mut result = Vec::with_capacity(raw_list.len());
        for s in raw_list {
            let qs = if s.eq_ignore_ascii_case("none") {
                QuantScheme::None
            } else if s.eq_ignore_ascii_case("per_row") {
                QuantScheme::PerRow
            } else if s.eq_ignore_ascii_case("per_token") {
                QuantScheme::PerToken
            } else if s.eq_ignore_ascii_case("per_block32") {
                QuantScheme::PerBlock32
            } else if let Some(code_str) = s.strip_prefix("scheme:") {
                let code = code_str.parse::<u64>().map_err(de::Error::custom)?;
                QuantScheme::Scheme(SchemeId::new(code))
            } else {
                return Err(de::Error::custom(format!("unknown QuantScheme '{s}'")));
            };
            result.push(qs);
        }
        Ok(result)
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
