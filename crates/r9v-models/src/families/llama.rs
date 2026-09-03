// SPDX-License-Identifier: Apache-2.0
//! Model architecture definition for the `llama` family (Spec 8 §4, §5, §9; card A1.4).
//!
//! Covers the dense transformer architectures registered under the `llama` family:
//! `llama`, `mistral`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, `phi3`, and `olmo2`.
//!
//! Hyperparameter keys follow the GGUF `<arch>.*` conventions. All semantic values
//! are decoded via the typed [`GgufMeta`] interface and validated using collect-all
//! error aggregation ([`ModelsError::Multiple`]) per `CONVENTIONS.md` §1.4 without panics.
//!
//! ### Accepted Metadata Keys Table (Spec 8 §4)
//!
//! | Semantic Value | Scope | Accepted Metadata Key | Expected Type | Required | Default / Behavior |
//! |---|---|---|---|---|---|
//! | Architecture | general | `general.architecture` | string | yes | Must be one of the 8 supported architectures |
//! | Layer count | `<arch>` | `<arch>.block_count` | u32 | yes | Number of transformer layers (> 0) |
//! | Model dim | `<arch>` | `<arch>.embedding_length` | u32 / u64 | yes | Hidden dimension `dm` (> 0) |
//! | Vocab size | `<arch>`/tok | `<arch>.vocab_size` or `tokenizer.ggml.tokens` | u32 / array of string | yes | Vocabulary size (> 0) |
//! | FFN dim | `<arch>` | `<arch>.feed_forward_length` | u32 | yes | Feed-forward intermediate dimension `dff` (> 0) |
//! | Query heads | `<arch>` | `<arch>.attention.head_count` | u32 | yes | Attention query heads `h` (> 0) |
//! | KV heads | `<arch>` | `<arch>.attention.head_count_kv` | u32 | no | Key/value heads `hkv` (defaults to `h`) |
//! | Key dimension | `<arch>` | `<arch>.attention.key_length` | u32 | no | Head key dimension `d` (required for gemma2/gemma3; defaults to `dm / h` for others) |
//! | Value dimension | `<arch>` | `<arch>.attention.value_length` | u32 | no | Head value dimension `dv` (defaults to `d`) |
//! | RMSNorm epsilon | `<arch>` | `<arch>.attention.layer_norm_rms_epsilon` | f32 | yes* | Epsilon floor (> 0); also accepts `layer_norm_epsilon`, `norm_epsilon` |
//! | RoPE base theta | `<arch>` | `<arch>.rope.freq_base` | f32 | no | Base frequency (defaults per architecture) |
//! | RoPE rotary dim | `<arch>` | `<arch>.rope.dimension_count` | u32 | no | Rotary dimension (defaults to `d`) |
//! | RoPE style | `<arch>` | `<arch>.rope.style` | string | no | `neox` (default) or `interleaved` |
//! | RoPE scaling type | `<arch>` | `<arch>.rope.scaling.type` | string | no | `none`, `linear`, `yarn`, `dynamic` |
//! | RoPE scaling factor | `<arch>` | `<arch>.rope.scaling.factor` | f32 | no* | Scaling factor (required if linear or yarn) |
//! | RoPE scaling orig ctx | `<arch>` | `<arch>.rope.scaling.original_context_length` | u32 | no* | Original context length (required if yarn) |
//! | RoPE scaling log mul | `<arch>` | `<arch>.rope.scaling.yarn_log_mul` | f32 | no | YaRN log multiplier (defaults to 1.0) |
//! | RoPE scaling beta fast | `<arch>` | `<arch>.rope.scaling.beta_fast` / `yarn_beta_fast` | f32 | no | YaRN fast beta (defaults to 32.0) |
//! | RoPE scaling beta slow | `<arch>` | `<arch>.rope.scaling.beta_slow` / `yarn_beta_slow` | f32 | no | YaRN slow beta (defaults to 1.0) |
//! | Sliding window size | `<arch>` | `<arch>.attention.sliding_window` | u32 | no | Sliding window size (or `sliding_window_size`) |
//! | Sliding window pattern | `<arch>` | `<arch>.attention.sliding_window_pattern` | u32, array of u32, or array of bool | no | Per-layer pattern: scalar period, u32 array, or bool array (Muse form) |
//! | Attention sinks | `<arch>` | `<arch>.attention.sinks` | u32 | no | Initial sinks (defaults to 0, requires window) |
//! | Attention softcap | `<arch>` | `<arch>.attn_logit_softcapping` | f32 | no | Canonical attention softcap (defaults 50.0 on gemma2; legacy `attention.logit_softcapping` accepted) |
//! | Final softcap | `<arch>` | `<arch>.final_logit_softcapping` | f32 | no | Output logits softcapping (defaults 30.0 on gemma2) |
//! | Tied embeddings | `<arch>`/gen | `<arch>.tie_word_embeddings` / `general.tie_word_embeddings` | bool | no | Defaults true for gemma2/gemma3, false for others |
//! | Embed scale | `<arch>` | `<arch>.embed_scale` | f32 | no | Defaults sqrt(dm) for gemma2/gemma3, 1.0 for others |
//! | QKV bias | `<arch>` | `<arch>.attention.qkv_bias` | bool | no | Defaults true for qwen2, false for others |
//! | Output bias | `<arch>` | `<arch>.attention.o_bias` | bool | no | Defaults false |
//! | QK norm | `<arch>` | `<arch>.attention.qk_norm` | bool | no | Defaults true for qwen3/olmo2/gemma3, false for others |
//! | Output gate | `<arch>` | `<arch>.attention.output_gate` | bool | no | Defaults false |
//! | Activation | `<arch>` | `<arch>.feed_forward_activation` | string | no | Defaults `gelu_tanh` on gemma2/gemma3, `silu` on others |
//! | BOS token ID | tokenizer | `tokenizer.ggml.bos_token_id` | u32 | no | Beginning-of-sequence token ID |
//! | EOS token ID(s) | tokenizer | `tokenizer.ggml.eos_token_id` / `tokenizer.ggml.eos_token_ids` | u32 / array of u32 | no | End-of-sequence token IDs |

use r9v_ir::op::{ActivationKind, RopeScaling, RopeStyle};

use crate::error::ModelsError;
use crate::meta::GgufMeta;
use crate::spec::{
    CacheDtype, Ffn, LayerSpec, Mixer, ModelSpec, NormPlacement, NormSpec, PositionEncoding,
    RopeSpec, MAX_ATTENTION_HEADS, MAX_FEATURE_DIM, MAX_KV_HEADS, MAX_MODEL_LAYERS, MAX_VOCAB_SIZE,
    MAX_WINDOW,
};

/// Definition record for an accepted GGUF metadata key (Spec 8 §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedKeyDef {
    /// Semantic attribute identifier.
    pub semantic: &'static str,
    /// Architecture prefix or scope (`"general"`, `"tokenizer"`, or `"<arch>"`).
    pub arch_scope: &'static str,
    /// Metadata key path.
    pub key: &'static str,
    /// Expected data type description.
    pub expected_type: &'static str,
    /// Whether this metadata key is mandatory for model definition.
    pub required: bool,
    /// Semantic description and default behavior.
    pub description: &'static str,
}

/// Explicit accepted metadata keys table across all supported architectures (Spec 8 §4; card A1.4).
pub const ACCEPTED_METADATA_KEYS: &[AcceptedKeyDef] = &[
    AcceptedKeyDef {
        semantic: "architecture",
        arch_scope: "general",
        key: "general.architecture",
        expected_type: "string",
        required: true,
        description: "Must be one of the eight supported architecture strings",
    },
    AcceptedKeyDef {
        semantic: "layer_count",
        arch_scope: "<arch>",
        key: "<arch>.block_count",
        expected_type: "u32",
        required: true,
        description: "Number of transformer layers",
    },
    AcceptedKeyDef {
        semantic: "embedding_length",
        arch_scope: "<arch>",
        key: "<arch>.embedding_length",
        expected_type: "u32 or u64",
        required: true,
        description: "Model hidden feature dimension dm",
    },
    AcceptedKeyDef {
        semantic: "feed_forward_length",
        arch_scope: "<arch>",
        key: "<arch>.feed_forward_length",
        expected_type: "u32",
        required: true,
        description: "Feed-forward intermediate hidden dimension dff",
    },
    AcceptedKeyDef {
        semantic: "head_count",
        arch_scope: "<arch>",
        key: "<arch>.attention.head_count",
        expected_type: "u32",
        required: true,
        description: "Number of attention query heads h",
    },
    AcceptedKeyDef {
        semantic: "head_count_kv",
        arch_scope: "<arch>",
        key: "<arch>.attention.head_count_kv",
        expected_type: "u32",
        required: false,
        description: "Number of attention key/value heads hkv (defaults to h)",
    },
    AcceptedKeyDef {
        semantic: "key_length",
        arch_scope: "<arch>",
        key: "<arch>.attention.key_length",
        expected_type: "u32",
        required: false,
        description: "Head key dimension d (required for gemma2 and gemma3; defaults to dm / h for others)",
    },
    AcceptedKeyDef {
        semantic: "value_length",
        arch_scope: "<arch>",
        key: "<arch>.attention.value_length",
        expected_type: "u32",
        required: false,
        description: "Head value dimension dv (defaults to d)",
    },
    AcceptedKeyDef {
        semantic: "layer_norm_rms_epsilon",
        arch_scope: "<arch>",
        key: "<arch>.attention.layer_norm_rms_epsilon",
        expected_type: "f32",
        required: true,
        description: "RMSNorm variance epsilon; layer_norm_epsilon accepted as fallback",
    },
    AcceptedKeyDef {
        semantic: "layer_norm_epsilon_legacy",
        arch_scope: "<arch>",
        key: "<arch>.attention.layer_norm_epsilon",
        expected_type: "f32",
        required: false,
        description: "Legacy fallback epsilon when layer_norm_rms_epsilon is absent",
    },
    AcceptedKeyDef {
        semantic: "norm_epsilon_legacy",
        arch_scope: "<arch>",
        key: "<arch>.norm_epsilon",
        expected_type: "f32",
        required: false,
        description: "Legacy fallback epsilon when attention epsilon keys are absent",
    },
    AcceptedKeyDef {
        semantic: "rope_freq_base",
        arch_scope: "<arch>",
        key: "<arch>.rope.freq_base",
        expected_type: "f32",
        required: false,
        description: "RoPE base frequency theta (defaults per family: 1e4 or 1e6 or 5e5)",
    },
    AcceptedKeyDef {
        semantic: "rope_dimension_count",
        arch_scope: "<arch>",
        key: "<arch>.rope.dimension_count",
        expected_type: "u32",
        required: false,
        description: "RoPE rotary dimension rot_dim (defaults to d)",
    },
    AcceptedKeyDef {
        semantic: "rope_style",
        arch_scope: "<arch>",
        key: "<arch>.rope.style",
        expected_type: "string",
        required: false,
        description: "RoPE pairing style: neox (default) or interleaved",
    },
    AcceptedKeyDef {
        semantic: "rope_scaling_type",
        arch_scope: "<arch>",
        key: "<arch>.rope.scaling.type",
        expected_type: "string",
        required: false,
        description: "RoPE frequency scaling kind: none, linear, yarn, dynamic",
    },
    AcceptedKeyDef {
        semantic: "rope_scaling_factor",
        arch_scope: "<arch>",
        key: "<arch>.rope.scaling.factor",
        expected_type: "f32",
        required: false,
        description: "RoPE scaling factor (required for linear or yarn scaling)",
    },
    AcceptedKeyDef {
        semantic: "rope_scaling_orig_ctx",
        arch_scope: "<arch>",
        key: "<arch>.rope.scaling.original_context_length",
        expected_type: "u32",
        required: false,
        description: "Original context length for YaRN scaling",
    },
    AcceptedKeyDef {
        semantic: "rope_scaling_yarn_log_mul",
        arch_scope: "<arch>",
        key: "<arch>.rope.scaling.yarn_log_mul",
        expected_type: "f32",
        required: false,
        description: "YaRN target scale mscale (defaults to 1.0)",
    },
    AcceptedKeyDef {
        semantic: "rope_scaling_beta_fast",
        arch_scope: "<arch>",
        key: "<arch>.rope.scaling.beta_fast",
        expected_type: "f32",
        required: false,
        description: "YaRN fast beta cutoff (defaults to 32.0)",
    },
    AcceptedKeyDef {
        semantic: "rope_scaling_beta_slow",
        arch_scope: "<arch>",
        key: "<arch>.rope.scaling.beta_slow",
        expected_type: "f32",
        required: false,
        description: "YaRN slow beta cutoff (defaults to 1.0)",
    },
    AcceptedKeyDef {
        semantic: "sliding_window",
        arch_scope: "<arch>",
        key: "<arch>.attention.sliding_window",
        expected_type: "u32",
        required: false,
        description: "Sliding window attention size w (or sliding_window_size)",
    },
    AcceptedKeyDef {
        semantic: "sliding_window_size_alt",
        arch_scope: "<arch>",
        key: "<arch>.attention.sliding_window_size",
        expected_type: "u32",
        required: false,
        description: "Alternate sliding window size key when sliding_window is absent",
    },
    AcceptedKeyDef {
        semantic: "context_length_fallback",
        arch_scope: "<arch>",
        key: "<arch>.context_length",
        expected_type: "u32",
        required: false,
        description: "Model context length; fallback for YaRN original context",
    },
    AcceptedKeyDef {
        semantic: "sliding_window_pattern",
        arch_scope: "<arch>",
        key: "<arch>.attention.sliding_window_pattern",
        expected_type: "u32, array of u32, or array of bool",
        required: false,
        description: "Explicit per-layer sliding window pattern (scalar period, array of u32, or array of bool)",
    },
    AcceptedKeyDef {
        semantic: "attention_sinks",
        arch_scope: "<arch>",
        key: "<arch>.attention.sinks",
        expected_type: "u32",
        required: false,
        description: "Initial attention sink token count (requires sliding window)",
    },
    AcceptedKeyDef {
        semantic: "attn_logit_softcapping",
        arch_scope: "<arch>",
        key: "<arch>.attn_logit_softcapping",
        expected_type: "f32",
        required: false,
        description: "Canonical attention logit softcapping threshold (defaults to 50.0 for gemma2)",
    },
    AcceptedKeyDef {
        semantic: "logit_softcapping_legacy",
        arch_scope: "<arch>",
        key: "<arch>.attention.logit_softcapping",
        expected_type: "f32",
        required: false,
        description: "Legacy fallback key for attention logit softcapping",
    },
    AcceptedKeyDef {
        semantic: "final_logit_softcapping",
        arch_scope: "<arch>",
        key: "<arch>.final_logit_softcapping",
        expected_type: "f32",
        required: false,
        description: "Output LM head logit softcapping threshold (defaults to 30.0 for gemma2)",
    },
    AcceptedKeyDef {
        semantic: "tie_word_embeddings",
        arch_scope: "<arch>",
        key: "<arch>.tie_word_embeddings",
        expected_type: "bool",
        required: false,
        description: "Tied input and output embeddings (defaults true on gemma2/gemma3)",
    },
    AcceptedKeyDef {
        semantic: "embed_scale",
        arch_scope: "<arch>",
        key: "<arch>.embed_scale",
        expected_type: "f32",
        required: false,
        description: "Embedding scaling factor (defaults sqrt(dm) on gemma2/gemma3, 1.0 on others)",
    },
    AcceptedKeyDef {
        semantic: "qkv_bias",
        arch_scope: "<arch>",
        key: "<arch>.attention.qkv_bias",
        expected_type: "bool",
        required: false,
        description: "Whether QKV projection includes bias vectors (defaults true on qwen2/qwen3)",
    },
    AcceptedKeyDef {
        semantic: "o_bias",
        arch_scope: "<arch>",
        key: "<arch>.attention.o_bias",
        expected_type: "bool",
        required: false,
        description: "Whether output projection includes bias vector (defaults false)",
    },
    AcceptedKeyDef {
        semantic: "qk_norm",
        arch_scope: "<arch>",
        key: "<arch>.attention.qk_norm",
        expected_type: "bool",
        required: false,
        description: "Whether Q/K projections have RMSNorm (defaults true on qwen3/olmo2/gemma3)",
    },
    AcceptedKeyDef {
        semantic: "output_gate",
        arch_scope: "<arch>",
        key: "<arch>.attention.output_gate",
        expected_type: "bool",
        required: false,
        description: "Gated attention output formulation (defaults false)",
    },
    AcceptedKeyDef {
        semantic: "feed_forward_activation",
        arch_scope: "<arch>",
        key: "<arch>.feed_forward_activation",
        expected_type: "string",
        required: false,
        description: "Activation function (defaults gelu_tanh on gemma2/gemma3, silu on others)",
    },
    AcceptedKeyDef {
        semantic: "vocab_size",
        arch_scope: "<arch>",
        key: "<arch>.vocab_size",
        expected_type: "u32",
        required: false,
        description: "Vocabulary size (falls back to tokenizer.ggml.tokens length)",
    },
    AcceptedKeyDef {
        semantic: "tokens",
        arch_scope: "tokenizer",
        key: "tokenizer.ggml.tokens",
        expected_type: "array of string",
        required: false,
        description: "Vocabulary token strings (used for vocab size if vocab_size missing)",
    },
    AcceptedKeyDef {
        semantic: "bos_token_id",
        arch_scope: "tokenizer",
        key: "tokenizer.ggml.bos_token_id",
        expected_type: "u32",
        required: false,
        description: "Beginning-of-sequence token ID",
    },
    AcceptedKeyDef {
        semantic: "eos_token_id",
        arch_scope: "tokenizer",
        key: "tokenizer.ggml.eos_token_id",
        expected_type: "u32",
        required: false,
        description: "Primary end-of-sequence token ID",
    },
    AcceptedKeyDef {
        semantic: "eos_token_ids",
        arch_scope: "tokenizer",
        key: "tokenizer.ggml.eos_token_ids",
        expected_type: "array of u32",
        required: false,
        description: "Multiple end-of-sequence token IDs",
    },
];

/// Builds a [`ModelSpec`] from typed GGUF metadata for any architecture supported
/// by the `llama` family (Spec 8 §4; card A1.4).
///
/// Reads `general.architecture` from metadata and verifies it matches one of
/// the eight supported architecture strings before building.
pub fn build(meta: &(impl GgufMeta + ?Sized)) -> Result<ModelSpec, ModelsError> {
    let arch = meta.str("general.architecture")?;
    build_for_arch(arch, meta)
}

/// Builds a [`ModelSpec`] for a specific verified architecture string (Spec 8 §4).
///
/// Collects all metadata parsing, typing, dimension, and consistency problems
/// into [`ModelsError::Multiple`] per `CONVENTIONS.md` §1.4 without panics.
pub fn build_for_arch(
    arch: &str,
    meta: &(impl GgufMeta + ?Sized),
) -> Result<ModelSpec, ModelsError> {
    match arch {
        "llama" | "mistral" | "qwen2" | "qwen3" | "gemma2" | "gemma3" | "phi3" | "olmo2" => {}
        other => {
            return Err(ModelsError::UnknownArchitecture {
                arch: other.to_string(),
                nearest: "llama",
            });
        }
    }

    let mut problems = Vec::new();

    // 1. Layer count
    let block_count_key = format!("{arch}.block_count");
    let block_count = if meta.has(&block_count_key) {
        match meta.u32(&block_count_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{block_count_key} must be > 0, got 0"),
                    });
                    None
                } else if v > MAX_MODEL_LAYERS {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{block_count_key} ({v}) exceeds implementation limit {MAX_MODEL_LAYERS}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        problems.push(ModelsError::MissingMetaKey {
            key: block_count_key,
            expected_type: "u32",
        });
        None
    };

    // 2. Model hidden dimension dm
    let dm_key = format!("{arch}.embedding_length");
    let dm = if meta.has(&dm_key) {
        match meta.u32(&dm_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{dm_key} must be > 0, got 0"),
                    });
                    None
                } else if v > MAX_FEATURE_DIM {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{dm_key} ({v}) exceeds implementation limit {MAX_FEATURE_DIM}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        problems.push(ModelsError::MissingMetaKey {
            key: dm_key,
            expected_type: "u32",
        });
        None
    };

    // 3. FFN intermediate dimension dff
    let dff_key = format!("{arch}.feed_forward_length");
    let dff = if meta.has(&dff_key) {
        match meta.u32(&dff_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{dff_key} must be > 0, got 0"),
                    });
                    None
                } else if v > MAX_FEATURE_DIM {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{dff_key} ({v}) exceeds implementation limit {MAX_FEATURE_DIM}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        problems.push(ModelsError::MissingMetaKey {
            key: dff_key,
            expected_type: "u32",
        });
        None
    };

    // 4. Query head count h
    let h_key = format!("{arch}.attention.head_count");
    let h = if meta.has(&h_key) {
        match meta.u32(&h_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{h_key} must be > 0, got 0"),
                    });
                    None
                } else if v > MAX_ATTENTION_HEADS {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{h_key} ({v}) exceeds implementation limit {MAX_ATTENTION_HEADS}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        problems.push(ModelsError::MissingMetaKey {
            key: h_key,
            expected_type: "u32",
        });
        None
    };

    // 5. KV head count hkv (defaults to h)
    let hkv_key = format!("{arch}.attention.head_count_kv");
    let hkv = if meta.has(&hkv_key) {
        match meta.u32(&hkv_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{hkv_key} must be > 0, got 0"),
                    });
                    None
                } else if v > MAX_KV_HEADS {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{hkv_key} ({v}) exceeds implementation limit {MAX_KV_HEADS}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        // Defaults to h when omitted (standard MHA)
        h
    };

    // Check h and hkv relationship
    if let (Some(h_val), Some(hkv_val)) = (h, hkv) {
        if hkv_val > h_val {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!("head_count_kv ({hkv_val}) cannot exceed head_count ({h_val})"),
            });
        } else if !h_val.is_multiple_of(hkv_val) {
            problems.push(ModelsError::InvalidModelSpec {
                reason: format!(
                    "head_count ({h_val}) must be divisible by head_count_kv ({hkv_val})"
                ),
            });
        }
    }

    // 6. Attention key dimension d (head_dim)
    let key_len_key = format!("{arch}.attention.key_length");
    let d = if meta.has(&key_len_key) {
        match meta.u32(&key_len_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{key_len_key} must be > 0, got 0"),
                    });
                    None
                } else if !v.is_multiple_of(16) {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "attention key_length ({v}) must be divisible by 16 per Spec 8 §6"
                        ),
                    });
                    None
                } else if v > MAX_FEATURE_DIM {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{key_len_key} ({v}) exceeds implementation limit {MAX_FEATURE_DIM}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else if arch == "gemma2" || arch == "gemma3" {
        problems.push(ModelsError::MissingMetaKey {
            key: key_len_key.clone(),
            expected_type: "u32",
        });
        None
    } else {
        // Derive from dm / h if both valid
        match (dm, h) {
            (Some(dm_val), Some(h_val)) => {
                if !dm_val.is_multiple_of(h_val) {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "embedding_length ({dm_val}) must be divisible by head_count ({h_val}) when key_length is omitted"
                        ),
                    });
                    None
                } else {
                    let derived_d = dm_val / h_val;
                    if derived_d == 0 {
                        problems.push(ModelsError::InvalidModelSpec {
                            reason: "derived head dimension is 0".to_string(),
                        });
                        None
                    } else if !derived_d.is_multiple_of(16) {
                        problems.push(ModelsError::InvalidModelSpec {
                            reason: format!(
                                "derived head dimension ({derived_d}) must be divisible by 16 per Spec 8 §6"
                            ),
                        });
                        None
                    } else {
                        Some(derived_d)
                    }
                }
            }
            _ => None,
        }
    };

    // 7. Attention value dimension dv (defaults to d)
    let val_len_key = format!("{arch}.attention.value_length");
    let dv = if meta.has(&val_len_key) {
        match meta.u32(&val_len_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{val_len_key} must be > 0, got 0"),
                    });
                    None
                } else if v > MAX_FEATURE_DIM {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{val_len_key} ({v}) exceeds implementation limit {MAX_FEATURE_DIM}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        d
    };

    // 8. Normalization epsilon
    let eps_primary_key = format!("{arch}.attention.layer_norm_rms_epsilon");
    let eps_alt_key = format!("{arch}.attention.layer_norm_epsilon");
    let eps_norm_key = format!("{arch}.norm_epsilon");
    let norm_eps = if meta.has(&eps_primary_key) {
        match meta.f32(&eps_primary_key) {
            Ok(v) => {
                if !v.is_finite() || v <= 0.0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{eps_primary_key} must be finite and > 0, got {v}"),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else if meta.has(&eps_alt_key) {
        match meta.f32(&eps_alt_key) {
            Ok(v) => {
                if !v.is_finite() || v <= 0.0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{eps_alt_key} must be finite and > 0, got {v}"),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else if meta.has(&eps_norm_key) {
        match meta.f32(&eps_norm_key) {
            Ok(v) => {
                if !v.is_finite() || v <= 0.0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{eps_norm_key} must be finite and > 0, got {v}"),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        problems.push(ModelsError::MissingMetaKey {
            key: eps_primary_key,
            expected_type: "f32",
        });
        None
    };

    // 9. RoPE base theta
    // DECISION(A1.4): default RoPE base frequency when omitted from metadata: 10000.0 for llama, mistral, gemma2, gemma3, phi3; 1000000.0 for qwen2 and qwen3; 500000.0 for olmo2; rejected: hardcoding 10000.0 for all families because Qwen and OLMo checkpoints rely on high base theta for context scaling when the metadata key is omitted in older exports. Spec 8 §4.
    let default_theta = match arch {
        "qwen2" | "qwen3" => 1_000_000.0,
        "olmo2" => 500_000.0,
        _ => 10_000.0,
    };
    let freq_base_key = format!("{arch}.rope.freq_base");
    let theta = if meta.has(&freq_base_key) {
        match meta.f32(&freq_base_key) {
            Ok(v) => {
                if !v.is_finite() || v <= 0.0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{freq_base_key} must be finite and > 0, got {v}"),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        Some(default_theta)
    };

    // 10. RoPE rotary dimension rot_dim (defaults to d)
    let rot_dim_key = format!("{arch}.rope.dimension_count");
    let rot_dim = if meta.has(&rot_dim_key) {
        match meta.u32(&rot_dim_key) {
            Ok(v) => {
                if v == 0 || !v.is_multiple_of(2) {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{rot_dim_key} must be positive and even per Spec 8 §3, got {v}"
                        ),
                    });
                    None
                } else {
                    if let Some(d_val) = d {
                        if v > d_val {
                            problems.push(ModelsError::InvalidModelSpec {
                                reason: format!(
                                    "rot_dim ({v}) cannot exceed head dimension ({d_val})"
                                ),
                            });
                        }
                    }
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        d
    };

    // 11. RoPE style (defaults to Neox per standard GGUF convention)
    let rope_style_key = format!("{arch}.rope.style");
    let rope_style = if meta.has(&rope_style_key) {
        match meta.str(&rope_style_key) {
            Ok("neox") => RopeStyle::Neox,
            Ok("interleaved") => RopeStyle::Interleaved,
            Ok(other) => {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("unsupported rope style {other}; expected neox or interleaved"),
                });
                RopeStyle::Neox
            }
            Err(e) => {
                problems.push(e);
                RopeStyle::Neox
            }
        }
    } else {
        RopeStyle::Neox
    };

    // 12. RoPE scaling
    let scaling_type_key = format!("{arch}.rope.scaling.type");
    let rope_scaling = if meta.has(&scaling_type_key) {
        match meta.str(&scaling_type_key) {
            Ok("none") | Ok("") => RopeScaling::None,
            Ok("linear") => {
                let factor_key = format!("{arch}.rope.scaling.factor");
                if meta.has(&factor_key) {
                    match meta.f32(&factor_key) {
                        Ok(f) => {
                            if !f.is_finite() || f <= 0.0 {
                                problems.push(ModelsError::InvalidModelSpec {
                                    reason: format!("{factor_key} must be finite and > 0, got {f}"),
                                });
                                RopeScaling::None
                            } else {
                                RopeScaling::Linear(f)
                            }
                        }
                        Err(e) => {
                            problems.push(e);
                            RopeScaling::None
                        }
                    }
                } else {
                    problems.push(ModelsError::MissingMetaKey {
                        key: factor_key,
                        expected_type: "f32",
                    });
                    RopeScaling::None
                }
            }
            Ok("yarn") => {
                let factor_key = format!("{arch}.rope.scaling.factor");
                let orig_ctx_key = format!("{arch}.rope.scaling.original_context_length");
                let ctx_fallback_key = format!("{arch}.context_length");
                let mscale_key = format!("{arch}.rope.scaling.yarn_log_mul");
                let beta_fast_key = format!("{arch}.rope.scaling.beta_fast");
                let beta_slow_key = format!("{arch}.rope.scaling.beta_slow");

                let factor = if meta.has(&factor_key) {
                    match meta.f32(&factor_key) {
                        Ok(f) if f.is_finite() && f > 0.0 => f,
                        Ok(f) => {
                            problems.push(ModelsError::InvalidModelSpec {
                                reason: format!("{factor_key} must be finite and > 0, got {f}"),
                            });
                            1.0
                        }
                        Err(e) => {
                            problems.push(e);
                            1.0
                        }
                    }
                } else {
                    problems.push(ModelsError::MissingMetaKey {
                        key: factor_key,
                        expected_type: "f32",
                    });
                    1.0
                };

                let orig_ctx = if meta.has(&orig_ctx_key) {
                    match meta.u32(&orig_ctx_key) {
                        Ok(v) if v > 0 => v,
                        Ok(_) => {
                            problems.push(ModelsError::InvalidModelSpec {
                                reason: format!("{orig_ctx_key} must be > 0"),
                            });
                            4096
                        }
                        Err(e) => {
                            problems.push(e);
                            4096
                        }
                    }
                } else if meta.has(&ctx_fallback_key) {
                    match meta.u32(&ctx_fallback_key) {
                        Ok(v) if v > 0 => v,
                        _ => 4096,
                    }
                } else {
                    problems.push(ModelsError::MissingMetaKey {
                        key: orig_ctx_key,
                        expected_type: "u32",
                    });
                    4096
                };

                let mscale = match meta.get_f32(&mscale_key) {
                    Ok(Some(v)) if v.is_finite() && v > 0.0 => v,
                    Ok(Some(v)) => {
                        problems.push(ModelsError::InvalidModelSpec {
                            reason: format!("{mscale_key} must be finite and > 0, got {v}"),
                        });
                        1.0
                    }
                    Ok(None) => 1.0,
                    Err(e) => {
                        problems.push(e);
                        1.0
                    }
                };

                let beta_fast = match meta.get_f32(&beta_fast_key) {
                    Ok(Some(v)) if v.is_finite() && v > 0.0 => v,
                    Ok(Some(v)) => {
                        problems.push(ModelsError::InvalidModelSpec {
                            reason: format!("{beta_fast_key} must be finite and > 0, got {v}"),
                        });
                        32.0
                    }
                    Ok(None) => 32.0,
                    Err(e) => {
                        problems.push(e);
                        32.0
                    }
                };

                let beta_slow = match meta.get_f32(&beta_slow_key) {
                    Ok(Some(v)) if v.is_finite() && v > 0.0 => v,
                    Ok(Some(v)) => {
                        problems.push(ModelsError::InvalidModelSpec {
                            reason: format!("{beta_slow_key} must be finite and > 0, got {v}"),
                        });
                        1.0
                    }
                    Ok(None) => 1.0,
                    Err(e) => {
                        problems.push(e);
                        1.0
                    }
                };

                RopeScaling::Yarn {
                    factor,
                    beta_fast,
                    beta_slow,
                    orig_ctx,
                    mscale,
                }
            }
            Ok("dynamic") => RopeScaling::Dynamic,
            Ok(other) => {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!(
                        "unsupported rope scaling type {other}; expected none, linear, yarn, or dynamic"
                    ),
                });
                RopeScaling::None
            }
            Err(e) => {
                problems.push(e);
                RopeScaling::None
            }
        }
    } else {
        RopeScaling::None
    };

    // 13. Vocab size
    let vocab_key = format!("{arch}.vocab_size");
    let vocab = if meta.has(&vocab_key) {
        match meta.u32(&vocab_key) {
            Ok(v) => {
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{vocab_key} must be > 0, got 0"),
                    });
                    None
                } else if v > MAX_VOCAB_SIZE {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{vocab_key} ({v}) exceeds implementation limit {MAX_VOCAB_SIZE}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else if meta.has("tokenizer.ggml.tokens") {
        match meta.str_array("tokenizer.ggml.tokens") {
            Ok(tokens) => {
                let v = tokens.len() as u32;
                if v == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: "tokenizer.ggml.tokens array is empty".to_string(),
                    });
                    None
                } else if v > MAX_VOCAB_SIZE {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "tokenizer tokens length ({v}) exceeds implementation limit {MAX_VOCAB_SIZE}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        problems.push(ModelsError::MissingMetaKey {
            key: vocab_key,
            expected_type: "u32 or tokenizer.ggml.tokens",
        });
        None
    };

    // 14. Tied embeddings
    // DECISION(A1.4): tied embeddings default: true for gemma2 and gemma3 (standard Gemma weight-sharing contract); false for llama, mistral, qwen2, qwen3, phi3, olmo2 unless <arch>.tie_word_embeddings or general.tie_word_embeddings is true; rejected: defaulting false for all architectures because Gemma checkpoints omit explicit tied flag and rely on architecture contract. Spec 8 §4.
    let default_tied = matches!(arch, "gemma2" | "gemma3");
    let tie_key = format!("{arch}.tie_word_embeddings");
    let tied_embeddings = match meta.get_bool(&tie_key) {
        Ok(Some(b)) => b,
        Ok(None) => match meta.get_bool("general.tie_word_embeddings") {
            Ok(Some(b)) => b,
            Ok(None) => default_tied,
            Err(e) => {
                problems.push(e);
                default_tied
            }
        },
        Err(e) => {
            problems.push(e);
            default_tied
        }
    };

    // 15. Embed scale
    // DECISION(A1.4): embed_scale default: (dm as f32).sqrt() for gemma2 and gemma3 per Spec 8 §3; 1.0 for all other six architectures, unless overridden by <arch>.embed_scale; rejected: defaulting 1.0 for all families because Gemma models scale embeddings by sqrt(dm). Spec 8 §3, §4.
    let embed_scale_key = format!("{arch}.embed_scale");
    let embed_scale = if meta.has(&embed_scale_key) {
        match meta.f32(&embed_scale_key) {
            Ok(s) => {
                if !s.is_finite() || s <= 0.0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{embed_scale_key} must be finite and > 0, got {s}"),
                    });
                    1.0
                } else {
                    s
                }
            }
            Err(e) => {
                problems.push(e);
                1.0
            }
        }
    } else if matches!(arch, "gemma2" | "gemma3") {
        match dm {
            Some(dm_val) => (dm_val as f32).sqrt(),
            None => 1.0,
        }
    } else {
        1.0
    };

    // 16. Softcapping
    // DECISION(A1.4): softcapping defaults: logit_softcap Some(50.0) and final_logit_softcap Some(30.0) for gemma2 per Spec 8 §3; None for all other seven families unless explicitly specified in metadata; rejected: applying softcapping to gemma3 where QK-norm proactively bounds logit growth. Spec 8 §3, §4, SI-42.
    let default_attn_softcap = if arch == "gemma2" { Some(50.0) } else { None };
    let default_final_softcap = if arch == "gemma2" { Some(30.0) } else { None };

    // DECISION(A1.4): probe canonical {arch}.attn_logit_softcapping first; accept legacy {arch}.attention.logit_softcapping for backward compatibility per SI-42; rejected ignoring canonical GGUF key naming. Spec 8 §4, SI-42.
    let canonical_attn_softcap_key = format!("{arch}.attn_logit_softcapping");
    let legacy_attn_softcap_key = format!("{arch}.attention.logit_softcapping");
    let (chosen_attn_softcap_key, raw_attn_softcap) = if meta.has(&canonical_attn_softcap_key) {
        (
            canonical_attn_softcap_key.clone(),
            meta.get_f32(&canonical_attn_softcap_key),
        )
    } else if meta.has(&legacy_attn_softcap_key) {
        (
            legacy_attn_softcap_key.clone(),
            meta.get_f32(&legacy_attn_softcap_key),
        )
    } else {
        (canonical_attn_softcap_key.clone(), Ok(None))
    };

    let attn_softcap = match raw_attn_softcap {
        Ok(Some(cap)) => {
            if !cap.is_finite() || cap <= 0.0 {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("{chosen_attn_softcap_key} must be finite and > 0, got {cap}"),
                });
                None
            } else {
                Some(cap)
            }
        }
        Ok(None) => default_attn_softcap,
        Err(e) => {
            problems.push(e);
            default_attn_softcap
        }
    };

    let final_softcap_key = format!("{arch}.final_logit_softcapping");
    let final_logit_softcap = match meta.get_f32(&final_softcap_key) {
        Ok(Some(cap)) => {
            if !cap.is_finite() || cap <= 0.0 {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("{final_softcap_key} must be finite and > 0, got {cap}"),
                });
                None
            } else {
                Some(cap)
            }
        }
        Ok(None) => default_final_softcap,
        Err(e) => {
            problems.push(e);
            default_final_softcap
        }
    };

    // 17. QKV Bias and Output Bias
    // DECISION(A1.4): bias defaults: qkv_bias true for qwen2 only and false for llama, mistral, qwen3, gemma2, gemma3, phi3, and olmo2, unless <arch>.attention.qkv_bias is present; o_bias is false across all eight families unless explicitly set; rejected: treating qwen3 like qwen2 because production Qwen3 checkpoints omit Q/K/V bias tensors. Spec 8 §3, §4.
    let default_qkv_bias = arch == "qwen2";
    let qkv_bias_key = format!("{arch}.attention.qkv_bias");
    let qkv_bias = match meta.get_bool(&qkv_bias_key) {
        Ok(Some(b)) => b,
        Ok(None) => default_qkv_bias,
        Err(e) => {
            problems.push(e);
            default_qkv_bias
        }
    };

    let o_bias_key = format!("{arch}.attention.o_bias");
    let o_bias = match meta.get_bool(&o_bias_key) {
        Ok(Some(b)) => b,
        Ok(None) => false,
        Err(e) => {
            problems.push(e);
            false
        }
    };

    // 18. QK Norm
    // DECISION(A1.4): QK normalization defaults: Some(NormSpec::rms(eps)) for qwen3, olmo2, and gemma3; None for llama, mistral, phi3, gemma2, and standard qwen2, unless overridden by <arch>.attention.qk_norm; rejected: enabling QK norm on all models or requiring explicit metadata boolean on modern architectures. Spec 8 §4.
    let default_qk_norm = matches!(arch, "qwen3" | "olmo2" | "gemma3");
    let qk_norm_key = format!("{arch}.attention.qk_norm");
    let has_qk_norm = match meta.get_bool(&qk_norm_key) {
        Ok(Some(b)) => b,
        Ok(None) => default_qk_norm,
        Err(e) => {
            problems.push(e);
            default_qk_norm
        }
    };
    let qk_norm_spec = if has_qk_norm {
        norm_eps.map(NormSpec::rms)
    } else {
        None
    };

    // 19. Output Gate
    let out_gate_key = format!("{arch}.attention.output_gate");
    let output_gate = match meta.get_bool(&out_gate_key) {
        Ok(Some(b)) => b,
        Ok(None) => false,
        Err(e) => {
            problems.push(e);
            false
        }
    };

    // 20. Feed-forward activation
    // DECISION(A1.4): activation function defaults when omitted from metadata: Silu for llama, mistral, qwen2, qwen3, phi3, olmo2; GeluTanh for gemma2 and gemma3; rejected: defaulting Silu universally because Gemma architectures require GeLU with tanh approximation. Spec 8 §4.
    let default_act = match arch {
        "gemma2" | "gemma3" => ActivationKind::GeluTanh,
        _ => ActivationKind::Silu,
    };
    let act_key = format!("{arch}.feed_forward_activation");
    let act = if meta.has(&act_key) {
        match meta.str(&act_key) {
            Ok("silu") | Ok("swish") => ActivationKind::Silu,
            Ok("gelu") => ActivationKind::Gelu,
            Ok("gelu_tanh") | Ok("gelu_pytorch_tanh") => ActivationKind::GeluTanh,
            Ok("relu2") | Ok("relu_squared") => ActivationKind::Relu2,
            Ok(other) => {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!(
                        "unsupported activation function {other}; expected silu, gelu, gelu_tanh, or relu2"
                    ),
                });
                default_act
            }
            Err(e) => {
                problems.push(e);
                default_act
            }
        }
    } else {
        default_act
    };

    // 21. Normalization placement and spec
    // DECISION(A1.4): OLMo 2 normalization placement maps to the honest
    // NormPlacement::Post contract: no input pre-norm, then canonical
    // post_attention_norm/post_ffw_norm on each sublayer branch before residual
    // addition. Gemma 2/3 remain Sandwich; the other families remain Pre.
    // Rejected mapping OLMo 2 to Sandwich because that binds nonexistent
    // attn_norm/ffn_norm weights. Spec 8 §3, §4.
    let norm_placement = match arch {
        "olmo2" => NormPlacement::Post,
        "gemma2" | "gemma3" => NormPlacement::Sandwich,
        _ => NormPlacement::Pre,
    };

    let norm_spec = match (arch, norm_eps) {
        ("gemma2" | "gemma3", Some(eps_val)) => NormSpec::gemma(eps_val),
        (_, Some(eps_val)) => NormSpec::rms(eps_val),
        (_, None) => NormSpec::rms(1e-5),
    };

    // 22. Sliding window and heterogeneous window patterns
    // DECISION(A1.4): Gemma 2 heterogeneous sliding window alternates 1:1 (even layers local with sliding window, odd layers global), and Gemma 3 uses 5:1 local-to-global ratio with 6-layer period (every 6th layer global, remaining local), unless overridden by explicit metadata; rejected: requiring explicit per-layer array metadata in every checkpoint file because GGUF checkpoints supply only scalar sliding_window. Spec 8 §4, SI-41.
    let sw_primary_key = format!("{arch}.attention.sliding_window");
    let sw_alt_key = format!("{arch}.attention.sliding_window_size");
    let sliding_window_size = if meta.has(&sw_primary_key) {
        match meta.u32(&sw_primary_key) {
            Ok(v) => {
                if v == 0 {
                    None
                } else if v > MAX_WINDOW {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{sw_primary_key} ({v}) exceeds implementation limit {MAX_WINDOW}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else if meta.has(&sw_alt_key) {
        match meta.u32(&sw_alt_key) {
            Ok(v) => {
                if v == 0 {
                    None
                } else if v > MAX_WINDOW {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!(
                            "{sw_alt_key} ({v}) exceeds implementation limit {MAX_WINDOW}"
                        ),
                    });
                    None
                } else {
                    Some(v)
                }
            }
            Err(e) => {
                problems.push(e);
                None
            }
        }
    } else {
        match arch {
            "gemma2" => Some(4096),
            "gemma3" => Some(1024),
            _ => None,
        }
    };

    #[derive(Debug, Clone, PartialEq)]
    enum SlidingWindowPattern {
        Period(u32),
        BoolArray(Vec<bool>),
        U32Array(Vec<u32>),
    }

    let pattern_key = format!("{arch}.attention.sliding_window_pattern");
    let explicit_pattern = if meta.has(&pattern_key) {
        // Probe scalar u32 period first
        match meta.u32(&pattern_key) {
            Ok(period) => {
                if period == 0 {
                    problems.push(ModelsError::InvalidModelSpec {
                        reason: format!("{pattern_key} period must be > 0, got 0"),
                    });
                    None
                } else {
                    Some(SlidingWindowPattern::Period(period))
                }
            }
            Err(_) => {
                // Probe bool array (e.g. Muse GGUF format: [True, True, True, False, ...])
                match meta.bool_array(&pattern_key) {
                    Ok(bools) => {
                        if bools.is_empty() {
                            problems.push(ModelsError::InvalidModelSpec {
                                reason: format!("{pattern_key} array must not be empty"),
                            });
                            None
                        } else {
                            Some(SlidingWindowPattern::BoolArray(bools))
                        }
                    }
                    Err(_) => {
                        // Probe u32 array (e.g. [1, 1, 0, ...] or explicit window sizes)
                        match meta.u32_array(&pattern_key) {
                            Ok(u32s) => {
                                if u32s.is_empty() {
                                    problems.push(ModelsError::InvalidModelSpec {
                                        reason: format!("{pattern_key} array must not be empty"),
                                    });
                                    None
                                } else {
                                    for &v in &u32s {
                                        if v > MAX_WINDOW {
                                            problems.push(ModelsError::InvalidModelSpec {
                                                reason: format!(
                                                    "{pattern_key} element ({v}) exceeds implementation limit {MAX_WINDOW}"
                                                ),
                                            });
                                        }
                                    }
                                    Some(SlidingWindowPattern::U32Array(u32s))
                                }
                            }
                            Err(_) => {
                                problems.push(ModelsError::MetaTypeMismatch {
                                    key: pattern_key.clone(),
                                    expected: "u32, array of u32, or array of bool",
                                    found: "unsupported or malformed sliding window pattern type"
                                        .to_string(),
                                });
                                None
                            }
                        }
                    }
                }
            }
        }
    } else {
        None
    };

    if explicit_pattern.is_some() && sliding_window_size.is_none() {
        problems.push(ModelsError::InvalidModelSpec {
            reason: format!("{pattern_key} requires a base sliding window size ({sw_primary_key})"),
        });
    }

    // 23. Attention sinks
    let sinks_key = format!("{arch}.attention.sinks");
    let sinks = match meta.get_u32(&sinks_key) {
        Ok(Some(v)) => {
            if v > MAX_WINDOW {
                problems.push(ModelsError::InvalidModelSpec {
                    reason: format!("{sinks_key} ({v}) exceeds implementation limit {MAX_WINDOW}"),
                });
                0
            } else {
                v
            }
        }
        Ok(None) => 0,
        Err(e) => {
            problems.push(e);
            0
        }
    };

    // 24. EOS and BOS token IDs
    let bos_id = match meta.get_u32("tokenizer.ggml.bos_token_id") {
        Ok(Some(id)) => Some(id),
        Ok(None) => None,
        Err(e) => {
            problems.push(e);
            None
        }
    };

    let mut eos_ids = Vec::new();
    if meta.has("tokenizer.ggml.eos_token_ids") {
        match meta.u32_array("tokenizer.ggml.eos_token_ids") {
            Ok(ids) => eos_ids = ids,
            Err(e) => problems.push(e),
        }
    } else if let Ok(Some(id)) = meta.get_u32("tokenizer.ggml.eos_token_id") {
        eos_ids.push(id);
    }

    // 25. Dimension arithmetic overflow validation
    if let (Some(h_val), Some(d_val), Some(hkv_val), Some(dv_val)) = (h, d, hkv, dv) {
        if h_val.checked_mul(d_val).is_none() {
            problems.push(ModelsError::ArithmeticOverflow {
                context: "query projection dimension".to_string(),
                operation: format!("{h_val} * {d_val}"),
            });
        }
        if hkv_val.checked_mul(d_val).is_none() {
            problems.push(ModelsError::ArithmeticOverflow {
                context: "key projection dimension".to_string(),
                operation: format!("{hkv_val} * {d_val}"),
            });
        }
        if hkv_val.checked_mul(dv_val).is_none() {
            problems.push(ModelsError::ArithmeticOverflow {
                context: "value projection dimension".to_string(),
                operation: format!("{hkv_val} * {dv_val}"),
            });
        }
        if h_val.checked_mul(dv_val).is_none() {
            problems.push(ModelsError::ArithmeticOverflow {
                context: "attention output projection dimension".to_string(),
                operation: format!("{h_val} * {dv_val}"),
            });
        }
    }

    // If any problem occurred during metadata decoding and initial dimension checking, return now
    if !problems.is_empty() {
        return Err(ModelsError::from_problems(problems).unwrap_err());
    }

    // Unpack validated primitives
    let num_layers = block_count.unwrap();
    let dm_val = dm.unwrap();
    let dff_val = dff.unwrap();
    let h_val = h.unwrap();
    let hkv_val = hkv.unwrap();
    let d_val = d.unwrap();
    let dv_val = dv.unwrap();
    let theta_val = theta.unwrap();
    let rot_dim_val = rot_dim.unwrap();
    let vocab_val = vocab.unwrap();

    let rope_spec = RopeSpec {
        theta: theta_val,
        rot_dim: rot_dim_val,
        style: rope_style,
        scaling: rope_scaling,
        mrope_sections: None,
    };

    // Derive per-layer LayerSpecs including heterogeneous sliding window
    let mut layers = Vec::with_capacity(num_layers as usize);
    for i in 0..num_layers {
        let layer_window = if let Some(pattern) = &explicit_pattern {
            match pattern {
                SlidingWindowPattern::Period(period) => {
                    let is_local = (i % period) < (period - 1);
                    if is_local {
                        sliding_window_size
                    } else {
                        None
                    }
                }
                SlidingWindowPattern::BoolArray(bools) => {
                    let is_local = bools[(i as usize) % bools.len()];
                    if is_local {
                        sliding_window_size
                    } else {
                        None
                    }
                }
                SlidingWindowPattern::U32Array(u32s) => {
                    let p = u32s[(i as usize) % u32s.len()];
                    if p == 0 {
                        None
                    } else if p == 1 {
                        sliding_window_size
                    } else {
                        Some(p)
                    }
                }
            }
        } else if arch == "gemma2" {
            // Gemma 2: 1:1 alternating pattern (even local, odd global)
            let base_w = sliding_window_size.unwrap_or(4096);
            if i % 2 == 0 {
                Some(base_w)
            } else {
                None
            }
        } else if arch == "gemma3" {
            // Gemma 3: 5:1 interleaving (every 6th layer global, remaining 5 local)
            let base_w = sliding_window_size.unwrap_or(1024);
            if (i + 1) % 6 == 0 {
                None
            } else {
                Some(base_w)
            }
        } else {
            sliding_window_size
        };

        // Check sink constraint per layer
        if sinks > 0 && layer_window.is_none() {
            problems.push(ModelsError::InvalidLayerSpec {
                layer: i,
                reason: format!(
                    "attention sinks ({sinks}) require a sliding window; Spec 3 §2 has no sink-only Retain form"
                ),
            });
        }

        let pre_fused = arch == "phi3";

        let mixer = Mixer::Attention {
            h: h_val,
            hkv: hkv_val,
            d: d_val,
            dv: dv_val,
            qkv_bias,
            o_bias,
            qk_norm: qk_norm_spec,
            rope: rope_spec.clone(),
            window: layer_window,
            sinks,
            logit_softcap: attn_softcap,
            output_gate,
            mla: None,
            cache: CacheDtype::E4M3,
            pre_fused,
        };

        let ffn = Ffn::Dense {
            dff: dff_val,
            act,
            gated: true,
            bias: false,
            pre_fused,
        };

        layers.push(LayerSpec {
            norm: norm_placement,
            norm_kind: norm_spec,
            mixer,
            ffn,
            residual_scale: 1.0,
        });
    }

    if !problems.is_empty() {
        return Err(ModelsError::from_problems(problems).unwrap_err());
    }

    let model_spec = ModelSpec {
        dm: dm_val,
        layers,
        vocab: vocab_val,
        embed_scale,
        tied_embeddings,
        final_norm: norm_spec,
        final_logit_softcap,
        positions: PositionEncoding::Scalar,
        ngram: None,
        mtp: None,
        export_hidden: false,
        eos_ids,
        bos_id,
    };

    // Full Spec 8 §6 validation
    model_spec.validate()?;

    Ok(model_spec)
}
