// SPDX-License-Identifier: Apache-2.0
//! Typed `r9v.*` metadata accessors (Spec 2 §6; card A2.5).
//!
//! [`parse_r9v_meta`] reads the `r9v.*` key schema from a parsed
//! [`crate::container::GgufFile`] into [`R9vMeta`] with all failures
//! collected (CONVENTIONS.md §1.4), enforcing the Spec 2 §9 version
//! rule. Returns `None` when the file carries no `r9v.*` keys (a
//! standard GGUF). Unknown `r9v.*` keys are ignored so minor
//! additions (Spec 2 §9: a new flag with a default does not bump
//! the version) keep loading on older readers.

use crate::container::{GgufFile, KvValue};
use crate::{FormatError, Layout, SchemeId};

/// Activation dtype of [`ActSpec`] (Spec 2 §3.4; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActDtype {
    /// Integer activations.
    I8,
    /// FP8 `e4m3` activations.
    E4M3,
    /// Full-precision activations.
    F16,
}

/// Activation quantization scheme (Spec 2 §3.4; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActScheme {
    /// Per-token scales.
    PerToken,
    /// Per-32-block scales (the llama.cpp MMQ parity path).
    PerBlock32,
    /// No activation quantization.
    None,
}

/// Per-tensor activation metadata, written `"dtype/scheme"`
/// (`"i8/PerToken"`; Spec 2 §3.4, §6; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActSpec {
    /// Activation dtype.
    pub dtype: ActDtype,
    /// Activation scheme.
    pub scheme: ActScheme,
}

impl ActSpec {
    /// Parses the `"dtype/scheme"` form; anything else is an error
    /// naming the value (Spec 2 §3.4).
    pub fn parse(text: &str) -> Result<Self, FormatError> {
        let (dtype, scheme) = text.split_once('/').ok_or_else(|| FormatError::Malformed {
            offset: 0,
            detail: format!("r9v act {text:?}: expected \"dtype/scheme\""),
        })?;
        let dtype = match dtype {
            "i8" => ActDtype::I8,
            "e4m3" => ActDtype::E4M3,
            "f16" => ActDtype::F16,
            _ => {
                return Err(FormatError::Malformed {
                    offset: 0,
                    detail: format!("r9v act {text:?}: unknown dtype {dtype:?}"),
                });
            }
        };
        let scheme = match scheme {
            "PerToken" => ActScheme::PerToken,
            "PerBlock32" => ActScheme::PerBlock32,
            "None" => ActScheme::None,
            _ => {
                return Err(FormatError::Malformed {
                    offset: 0,
                    detail: format!("r9v act {text:?}: unknown scheme {scheme:?}"),
                });
            }
        };
        Ok(Self { dtype, scheme })
    }
}

/// Structural role of a tensor (Spec 2 §4; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Dense matmul weight.
    Matmul,
    /// Embedding table.
    Embed,
    /// Language-model head.
    LmHead,
    /// N-gram table.
    NgramTable,
    /// Vector (norm weight, bias).
    Vector,
}

impl Role {
    /// Parses one role name.
    pub fn parse(text: &str) -> Result<Self, FormatError> {
        match text {
            "matmul" => Ok(Role::Matmul),
            "embed" => Ok(Role::Embed),
            "lm_head" => Ok(Role::LmHead),
            "ngram_table" => Ok(Role::NgramTable),
            "vector" => Ok(Role::Vector),
            _ => Err(FormatError::Malformed {
                offset: 0,
                detail: format!("r9v role {text:?}: expected one of matmul, embed, lm_head, ngram_table, vector"),
            }),
        }
    }
}

/// Tile interleave declaration (Spec 2 §4; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Interleave {
    /// No fusion.
    #[default]
    None,
    /// Alternating gate/up tiles.
    GateUp,
    /// Fused QKV tiles.
    Qkv,
}

impl Interleave {
    /// Parses one interleave name.
    pub fn parse(text: &str) -> Result<Self, FormatError> {
        match text {
            "none" => Ok(Interleave::None),
            "gate_up" => Ok(Interleave::GateUp),
            "qkv" => Ok(Interleave::Qkv),
            _ => Err(FormatError::Malformed {
                offset: 0,
                detail: format!("r9v interleave {text:?}: expected one of none, gate_up, qkv"),
            }),
        }
    }
}

/// Sparsity flag (Spec 2 §4; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Sparse {
    /// Dense.
    #[default]
    None,
    /// 2:4 structured sparse (`L1S`).
    S24,
}

impl Sparse {
    /// Parses one sparsity name.
    pub fn parse(text: &str) -> Result<Self, FormatError> {
        match text {
            "none" => Ok(Sparse::None),
            "s24" => Ok(Sparse::S24),
            _ => Err(FormatError::Malformed {
                offset: 0,
                detail: format!("r9v sparse {text:?}: expected one of none, s24"),
            }),
        }
    }
}

/// Placement recommendation (Spec 2 §4; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlacementHint {
    /// Device-resident.
    Device,
    /// Host-resident.
    Host,
    /// Tiered (row cache).
    Tiered,
}

impl PlacementHint {
    /// Parses one placement name.
    pub fn parse(text: &str) -> Result<Self, FormatError> {
        match text {
            "device" => Ok(PlacementHint::Device),
            "host" => Ok(PlacementHint::Host),
            "tiered" => Ok(PlacementHint::Tiered),
            _ => Err(FormatError::Malformed {
                offset: 0,
                detail: format!(
                    "r9v placement_hint {text:?}: expected one of device, host, tiered"
                ),
            }),
        }
    }
}

/// Cache residency granularity (Spec 2 §4; card A2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidencyUnit {
    /// Whole tensor.
    Tensor,
    /// One expert.
    Expert,
    /// One row.
    Row,
}

impl ResidencyUnit {
    /// Parses one residency name.
    pub fn parse(text: &str) -> Result<Self, FormatError> {
        match text {
            "tensor" => Ok(ResidencyUnit::Tensor),
            "expert" => Ok(ResidencyUnit::Expert),
            "row" => Ok(ResidencyUnit::Row),
            _ => Err(FormatError::Malformed {
                offset: 0,
                detail: format!("r9v residency_unit {text:?}: expected one of tensor, expert, row"),
            }),
        }
    }
}

/// Calibration provenance (Spec 2 §6 `r9v.calibration.*`; card A2.5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Calibration {
    /// Calibration set name.
    pub name: Option<String>,
    /// SHA-256 of the calibration manifest.
    pub hash: Option<String>,
    /// Token count calibrated on.
    pub tokens: Option<u64>,
    /// Domain-mix JSON.
    pub mix: Option<String>,
}

/// Smoothing provenance (Spec 2 §6 `r9v.smoothing.*`; card A2.5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Smoothing {
    /// Whether smoothing was folded into the weights.
    pub folded: Option<bool>,
    /// Smoothing strength.
    pub alpha: Option<f32>,
}

/// Quality report (Spec 2 §6 `r9v.quality.*`, Spec 13 §11; card A2.5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Quality {
    /// Mean KL divergence.
    pub kl_mean: Option<f32>,
    /// P99 KL divergence.
    pub kl_p99: Option<f32>,
    /// Top-1 agreement.
    pub top1: Option<f32>,
    /// Top-5 agreement.
    pub top5: Option<f32>,
    /// Holdout perplexity.
    pub ppl: Option<f32>,
    /// Holdout-set hash.
    pub holdout_hash: Option<String>,
    /// Engine version that measured the report.
    pub engine_version: Option<String>,
}

/// Per-tensor `r9v.tensor.<name>.*` metadata (Spec 2 §6; card A2.5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorMeta {
    /// Tensor name (matches the tensor-info table).
    pub name: String,
    /// Repack scheme.
    pub scheme: Option<SchemeId>,
    /// Activation contract.
    pub act: Option<ActSpec>,
    /// Structural roles.
    pub roles: Vec<Role>,
    /// Fusion interleave (defaults to none).
    pub interleave: Interleave,
    /// Sparsity flag (defaults to none).
    pub sparse: Sparse,
    /// Placement recommendation.
    pub placement_hint: Option<PlacementHint>,
    /// Cache residency granularity.
    pub residency_unit: Option<ResidencyUnit>,
    /// In-entry `[values, scales, indices]` offsets.
    pub regions: Option<[u64; 3]>,
    /// `xxh3` of the entry bytes.
    pub xxh3: Option<u64>,
    /// Per-expert routing frequencies (stacked experts only).
    pub hot_hint: Vec<f32>,
    /// Int4 sensitivity (informational).
    pub eps_int4: Option<f32>,
    /// Int8 sensitivity (informational).
    pub eps_int8: Option<f32>,
}

/// File-level `r9v.*` metadata (Spec 2 §6; card A2.5).
#[derive(Debug, Clone, PartialEq)]
pub struct R9vMeta {
    /// `r9v.format_version` (required whenever `r9v.*` keys exist).
    pub format_version: u32,
    /// `r9v.layout_id` (required whenever `r9v.*` keys exist).
    pub layout_id: Layout,
    /// `r9v.arch_hint` (informational).
    pub arch_hint: Option<String>,
    /// `r9v.quant_tool.version`.
    pub tool_version: Option<String>,
    /// `r9v.quant_tool.seed`.
    pub tool_seed: Option<u64>,
    /// `r9v.quant_tool.preset`.
    pub tool_preset: Option<String>,
    /// `r9v.quant_tool.target`.
    pub tool_target: Option<String>,
    /// `r9v.calibration.*`.
    pub calibration: Calibration,
    /// `r9v.smoothing.*`.
    pub smoothing: Smoothing,
    /// `r9v.quality.*`.
    pub quality: Quality,
    /// Per-tensor entries in tensor-table order.
    pub tensors: Vec<TensorMeta>,
}

impl R9vMeta {
    /// Finds one tensor's metadata by name (`None` when absent).
    pub fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Reads one required string key.
fn req_str(file: &GgufFile, key: &str, problems: &mut Vec<FormatError>) -> Option<String> {
    match file.kv(key) {
        None => {
            problems.push(FormatError::MissingKey {
                key: key.to_owned(),
            });
            None
        }
        Some(KvValue::Str(s)) => Some(s.clone()),
        Some(other) => {
            problems.push(FormatError::KvTypeMismatch {
                key: key.to_owned(),
                found: other.kv_type().name(),
                expected: "STRING",
            });
            None
        }
    }
}

/// Reads one optional scalar key with a type check.
fn opt_scalar<T>(
    file: &GgufFile,
    key: &str,
    expect: &'static str,
    extract: impl Fn(&KvValue) -> Option<T>,
    problems: &mut Vec<FormatError>,
) -> Option<T> {
    match file.kv(key) {
        None => None,
        Some(v) => match extract(v) {
            Some(t) => Some(t),
            None => {
                problems.push(FormatError::KvTypeMismatch {
                    key: key.to_owned(),
                    found: v.kv_type().name(),
                    expected: expect,
                });
                None
            }
        },
    }
}

/// Parses the `r9v.*` key schema from `file` (Spec 2 §6; card A2.5).
///
/// Returns `None` when no `r9v.*` key exists (standard GGUF).
/// Otherwise `r9v.format_version` and `r9v.layout_id` are required,
/// the Spec 2 §9 version rule is enforced, and every present key is
/// strictly type-checked, collecting all failures before returning
/// (CONVENTIONS.md §1.4).
pub fn parse_r9v_meta(file: &GgufFile) -> Result<Option<R9vMeta>, FormatError> {
    if !file.kvs().iter().any(|kv| kv.key.starts_with("r9v.")) {
        return Ok(None);
    }
    let mut problems = Vec::new();

    let format_version = opt_scalar(
        file,
        "r9v.format_version",
        "UINT32",
        |v| match v {
            KvValue::U32(x) => Some(*x),
            _ => None,
        },
        &mut problems,
    );
    let format_version = match format_version {
        Some(v) => v,
        None => {
            if file.kv("r9v.format_version").is_none() {
                problems.push(FormatError::MissingKey {
                    key: "r9v.format_version".to_owned(),
                });
            }
            0
        }
    };
    if let Err(e) = crate::container::accept_format_version(Some(format_version)) {
        problems.push(e);
    }

    // SI-74: spec 2 §6 writes `r9v.layout_id = "L1"` while the
    // Layout closed set serializes lowercase (`l1`); both spellings
    // are accepted on read and lowercase is written.
    let layout_id = match req_str(file, "r9v.layout_id", &mut problems) {
        Some(name) => {
            let canonical = match name.as_str() {
                "L0" => "l0",
                "L1" => "l1",
                "L1S" => "l1s",
                _ => name.as_str(),
            };
            match Layout::from_name(canonical) {
                Ok(layout) => Some(layout),
                Err(e) => {
                    problems.push(e);
                    None
                }
            }
        }
        None => None,
    };

    let arch_hint = opt_scalar(
        file,
        "r9v.arch_hint",
        "STRING",
        |v| match v {
            KvValue::Str(s) => Some(s.clone()),
            _ => None,
        },
        &mut problems,
    );
    let tool_version = opt_scalar(
        file,
        "r9v.quant_tool.version",
        "STRING",
        |v| match v {
            KvValue::Str(s) => Some(s.clone()),
            _ => None,
        },
        &mut problems,
    );
    let tool_seed = opt_scalar(
        file,
        "r9v.quant_tool.seed",
        "UINT64",
        |v| match v {
            KvValue::U64(x) => Some(*x),
            _ => None,
        },
        &mut problems,
    );
    let tool_preset = opt_scalar(
        file,
        "r9v.quant_tool.preset",
        "STRING",
        |v| match v {
            KvValue::Str(s) => Some(s.clone()),
            _ => None,
        },
        &mut problems,
    );
    let tool_target = opt_scalar(
        file,
        "r9v.quant_tool.target",
        "STRING",
        |v| match v {
            KvValue::Str(s) => Some(s.clone()),
            _ => None,
        },
        &mut problems,
    );
    let calibration = Calibration {
        name: opt_scalar(
            file,
            "r9v.calibration.name",
            "STRING",
            |v| match v {
                KvValue::Str(s) => Some(s.clone()),
                _ => None,
            },
            &mut problems,
        ),
        hash: opt_scalar(
            file,
            "r9v.calibration.hash",
            "STRING",
            |v| match v {
                KvValue::Str(s) => Some(s.clone()),
                _ => None,
            },
            &mut problems,
        ),
        tokens: opt_scalar(
            file,
            "r9v.calibration.tokens",
            "UINT64",
            |v| match v {
                KvValue::U64(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
        mix: opt_scalar(
            file,
            "r9v.calibration.mix",
            "STRING",
            |v| match v {
                KvValue::Str(s) => Some(s.clone()),
                _ => None,
            },
            &mut problems,
        ),
    };
    let smoothing = Smoothing {
        folded: opt_scalar(
            file,
            "r9v.smoothing.folded",
            "BOOL",
            |v| match v {
                KvValue::Bool(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
        alpha: opt_scalar(
            file,
            "r9v.smoothing.alpha",
            "FLOAT32",
            |v| match v {
                KvValue::F32(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
    };
    let quality = Quality {
        kl_mean: opt_scalar(
            file,
            "r9v.quality.kl_mean",
            "FLOAT32",
            |v| match v {
                KvValue::F32(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
        kl_p99: opt_scalar(
            file,
            "r9v.quality.kl_p99",
            "FLOAT32",
            |v| match v {
                KvValue::F32(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
        top1: opt_scalar(
            file,
            "r9v.quality.top1",
            "FLOAT32",
            |v| match v {
                KvValue::F32(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
        top5: opt_scalar(
            file,
            "r9v.quality.top5",
            "FLOAT32",
            |v| match v {
                KvValue::F32(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
        ppl: opt_scalar(
            file,
            "r9v.quality.ppl",
            "FLOAT32",
            |v| match v {
                KvValue::F32(x) => Some(*x),
                _ => None,
            },
            &mut problems,
        ),
        holdout_hash: opt_scalar(
            file,
            "r9v.quality.holdout_hash",
            "STRING",
            |v| match v {
                KvValue::Str(s) => Some(s.clone()),
                _ => None,
            },
            &mut problems,
        ),
        engine_version: opt_scalar(
            file,
            "r9v.quality.engine_version",
            "STRING",
            |v| match v {
                KvValue::Str(s) => Some(s.clone()),
                _ => None,
            },
            &mut problems,
        ),
    };

    // Per-tensor keys: `r9v.tensor.<name>.<suffix>` where `<name>`
    // may itself contain dots, so the suffix is matched last.
    const SUFFIXES: [&str; 12] = [
        ".scheme",
        ".act",
        ".roles",
        ".interleave",
        ".sparse",
        ".placement_hint",
        ".residency_unit",
        ".regions",
        ".xxh3",
        ".hot_hint",
        ".eps_int4",
        ".eps_int8",
    ];
    // TensorMeta slots in tensor-table order, plus overflow slots
    // for names the table does not contain (reported below).
    let mut tensors: Vec<TensorMeta> = file
        .tensors()
        .iter()
        .map(|t| TensorMeta {
            name: t.name.clone(),
            ..TensorMeta::default()
        })
        .collect();
    // Internal invariant: `tensors` parallels `file.tensors()`, so
    // position `i` always names the table row `i`.
    let index_of = |file: &GgufFile, name: &str| -> Option<usize> {
        file.tensors().iter().position(|t| t.name == name)
    };
    let mut unknown_tensors: Vec<String> = Vec::new();
    for kv in file.kvs() {
        let Some(rest) = kv.key.strip_prefix("r9v.tensor.") else {
            continue;
        };
        let Some(suffix) = SUFFIXES.iter().find(|s| rest.ends_with(*s)) else {
            // Unknown per-tensor suffix: ignored for forward
            // compatibility (Spec 2 §9 minor additions).
            continue;
        };
        let name = &rest[..rest.len() - suffix.len()];
        let slot = match index_of(file, name) {
            Some(i) => i,
            None => {
                if !unknown_tensors.iter().any(|n| n == name) {
                    unknown_tensors.push(name.to_owned());
                }
                continue;
            }
        };
        let meta = &mut tensors[slot];
        match *suffix {
            ".scheme" => match &kv.value {
                KvValue::Str(s) => match SchemeId::from_name(s) {
                    Ok(scheme) => meta.scheme = Some(scheme),
                    Err(e) => problems.push(e),
                },
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "STRING",
                }),
            },
            ".act" => match &kv.value {
                KvValue::Str(s) => match ActSpec::parse(s) {
                    Ok(act) => meta.act = Some(act),
                    Err(e) => problems.push(e),
                },
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "STRING",
                }),
            },
            ".roles" => match &kv.value {
                KvValue::Array { elem, items } => {
                    if *elem != crate::container::KvType::Str {
                        problems.push(FormatError::KvTypeMismatch {
                            key: kv.key.clone(),
                            found: elem.name(),
                            expected: "STRING",
                        });
                    } else if items.is_empty() {
                        problems.push(FormatError::Malformed {
                            offset: 0,
                            detail: format!("metadata key {:?}: roles array is empty", kv.key),
                        });
                    } else {
                        let mut parsed_roles = Vec::with_capacity(items.len());
                        let mut ok = true;
                        for item in items {
                            match item {
                                KvValue::Str(s) => match Role::parse(s) {
                                    Ok(role) => parsed_roles.push(role),
                                    Err(e) => {
                                        problems.push(e);
                                        ok = false;
                                    }
                                },
                                other => {
                                    problems.push(FormatError::KvTypeMismatch {
                                        key: kv.key.clone(),
                                        found: other.kv_type().name(),
                                        expected: "STRING",
                                    });
                                    ok = false;
                                }
                            }
                        }
                        if ok {
                            let is_valid_combo = matches!(
                                parsed_roles.as_slice(),
                                [Role::Matmul]
                                    | [Role::Embed]
                                    | [Role::LmHead]
                                    | [Role::NgramTable]
                                    | [Role::Vector]
                                    | [Role::Embed, Role::LmHead]
                                    | [Role::LmHead, Role::Embed]
                            );
                            if !is_valid_combo {
                                problems.push(FormatError::Malformed {
                                    offset: 0,
                                    detail: format!(
                                        "metadata key {:?}: invalid roles combination {:?}",
                                        kv.key, parsed_roles
                                    ),
                                });
                            } else {
                                meta.roles = parsed_roles;
                            }
                        }
                    }
                }
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "ARRAY",
                }),
            },
            ".interleave" => match &kv.value {
                KvValue::Str(s) => match Interleave::parse(s) {
                    Ok(v) => meta.interleave = v,
                    Err(e) => problems.push(e),
                },
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "STRING",
                }),
            },
            ".sparse" => match &kv.value {
                KvValue::Str(s) => match Sparse::parse(s) {
                    Ok(v) => meta.sparse = v,
                    Err(e) => problems.push(e),
                },
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "STRING",
                }),
            },
            ".placement_hint" => match &kv.value {
                KvValue::Str(s) => match PlacementHint::parse(s) {
                    Ok(v) => meta.placement_hint = Some(v),
                    Err(e) => problems.push(e),
                },
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "STRING",
                }),
            },
            ".residency_unit" => match &kv.value {
                KvValue::Str(s) => match ResidencyUnit::parse(s) {
                    Ok(v) => meta.residency_unit = Some(v),
                    Err(e) => problems.push(e),
                },
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "STRING",
                }),
            },
            ".regions" => match &kv.value {
                KvValue::Array { elem, items } => {
                    if *elem != crate::container::KvType::U64 {
                        problems.push(FormatError::KvTypeMismatch {
                            key: kv.key.clone(),
                            found: elem.name(),
                            expected: "UINT64",
                        });
                    } else if items.len() != 3 {
                        problems.push(FormatError::Malformed {
                            offset: 0,
                            detail: format!(
                                "metadata key {:?}: expected 3 region offsets, got {}",
                                kv.key,
                                items.len(),
                            ),
                        });
                    } else {
                        let mut offsets = [0u64; 3];
                        let mut ok = true;
                        for (i, item) in items.iter().enumerate() {
                            match item {
                                KvValue::U64(v) => offsets[i] = *v,
                                other => {
                                    problems.push(FormatError::KvTypeMismatch {
                                        key: kv.key.clone(),
                                        found: other.kv_type().name(),
                                        expected: "UINT64",
                                    });
                                    ok = false;
                                }
                            }
                        }
                        if ok {
                            if offsets[0] != 0
                                || offsets[0] > offsets[1]
                                || offsets[1] > offsets[2]
                                || offsets[1] % 256 != 0
                            {
                                problems.push(FormatError::Malformed {
                                    offset: 0,
                                    detail: format!(
                                        "metadata key {:?}: invalid region offsets {:?}",
                                        kv.key, offsets
                                    ),
                                });
                            } else {
                                meta.regions = Some(offsets);
                            }
                        }
                    }
                }
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "ARRAY",
                }),
            },
            ".xxh3" => match &kv.value {
                KvValue::U64(v) => meta.xxh3 = Some(*v),
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "UINT64",
                }),
            },
            ".hot_hint" => match &kv.value {
                KvValue::Array { elem, items } => {
                    if *elem != crate::container::KvType::F32 {
                        problems.push(FormatError::KvTypeMismatch {
                            key: kv.key.clone(),
                            found: elem.name(),
                            expected: "FLOAT32",
                        });
                    } else {
                        for item in items {
                            match item {
                                KvValue::F32(v) => meta.hot_hint.push(*v),
                                other => problems.push(FormatError::KvTypeMismatch {
                                    key: kv.key.clone(),
                                    found: other.kv_type().name(),
                                    expected: "FLOAT32",
                                }),
                            }
                        }
                    }
                }
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "ARRAY",
                }),
            },
            ".eps_int4" => match &kv.value {
                KvValue::F32(v) => meta.eps_int4 = Some(*v),
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "FLOAT32",
                }),
            },
            ".eps_int8" => match &kv.value {
                KvValue::F32(v) => meta.eps_int8 = Some(*v),
                other => problems.push(FormatError::KvTypeMismatch {
                    key: kv.key.clone(),
                    found: other.kv_type().name(),
                    expected: "FLOAT32",
                }),
            },
            _ => {}
        }
    }
    for name in unknown_tensors {
        problems.push(FormatError::Malformed {
            offset: 0,
            detail: format!("r9v metadata names unknown tensor {name:?}"),
        });
    }

    let Some(layout_id) = layout_id else {
        // `req_str` or the name parse above already recorded the
        // failure, so `problems` is nonempty here; the fallback is
        // only a backstop the compiler cannot see past.
        FormatError::collect(problems)?;
        return Err(FormatError::Malformed {
            offset: 0,
            detail: "r9v.layout_id is missing or invalid".to_owned(),
        });
    };
    FormatError::collect(problems)?;
    Ok(Some(R9vMeta {
        format_version,
        layout_id,
        arch_hint,
        tool_version,
        tool_seed,
        tool_preset,
        tool_target,
        calibration,
        smoothing,
        quality,
        tensors,
    }))
}
