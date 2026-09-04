// SPDX-License-Identifier: Apache-2.0
//! Single-sequence greedy CPU decode loop for tests (Card A1.12).
//!
//! [`decode_greedy`] drives a [`TinyModel`] step graph through
//! [`CpuExecutor`]: one prefill over the prompt, then one-token steps
//! reading the paged KV caches the executor owns. Each step rebuilds its
//! [`BatchMeta`] (positions, slots, context lengths advance); the next
//! token is the lowest-index argmax of the last logits row, so decoding
//! is fully deterministic with no RNG.

use r9v_ir::{BatchMeta, Positions};

use crate::attention::KvPagedCache;
use crate::buffer::TypedBuffer;
use crate::error::T0Error;
use crate::exec::{CpuExecutor, ExecError, RunArgs};
use crate::synthetic::{TinyModel, CACHE_BLOCK_TOKENS};

/// Greedy decode configuration (Spec 4 §2, Card A1.12).
pub struct DecodeConfig {
    /// Maximum new tokens after the prompt.
    pub max_new_tokens: u32,
    /// Optional stop token (exclusive: never emitted).
    pub eos: Option<u32>,
}

/// Greedy decode output (Spec 4 §2, Card A1.12).
pub struct DecodeResult {
    /// Prefill logits, row-major `[prompt_len, vocab]` f32.
    pub prompt_logits: Vec<f32>,
    /// Prompt token count.
    pub prompt_len: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Generated token ids (prompt excluded).
    pub generated: Vec<u32>,
    /// Logits row that produced each generated token (`[V]` f32 each).
    pub step_logits: Vec<Vec<f32>>,
}

/// Decodes `max_new_tokens` greedily after prefilling `prompt` (Spec 4 §2, Card A1.12).
///
/// Binds the model's weights, registers one paged cache per layer sized
/// from the spec's `max_ctx`, and refuses (typed) when the prompt plus
/// generation would exceed `max_ctx` or when the prompt is empty.
pub fn decode_greedy(
    exec: &mut CpuExecutor,
    model: &TinyModel,
    prompt: &[u32],
    config: &DecodeConfig,
) -> Result<DecodeResult, ExecError> {
    if prompt.is_empty() {
        return Err(ExecError::T0(T0Error::EmptyInput {
            op: "decode",
            tensor: "prompt",
        }));
    }
    let spec = &model.spec;
    let vocab = spec.vocab as usize;
    for token in prompt {
        if *token >= spec.vocab {
            return Err(ExecError::T0(T0Error::TokenOutOfRange {
                op: "decode",
                tensor: "prompt",
                position: 0,
                token: *token,
                vocab_size: vocab,
            }));
        }
    }
    let total = prompt.len() as u64 + config.max_new_tokens as u64;
    if total > spec.max_ctx as u64 {
        return Err(ExecError::T0(T0Error::ArithmeticOverflow {
            op: "decode",
            detail: format!(
                "prompt ({}) + max_new_tokens ({}) exceeds max_ctx ({})",
                prompt.len(),
                config.max_new_tokens,
                spec.max_ctx
            ),
        }));
    }

    let max_blocks = prepare(exec, model)?;

    let mut rng_states = Vec::new();
    let prompt_logits = run_step(exec, model, prompt, 0, max_blocks, &mut rng_states)?;
    let mut generated = Vec::new();
    let mut step_logits = Vec::new();
    let mut next = argmax_last_row(&prompt_logits, prompt.len(), vocab);
    let mut pos = prompt.len() as u32;
    while generated.len() < config.max_new_tokens as usize {
        if config.eos == Some(next) {
            break;
        }
        generated.push(next);
        let logits = run_step(exec, model, &[next], pos, max_blocks, &mut rng_states)?;
        step_logits.push(logits.clone());
        next = argmax_last_row(&logits, 1, vocab);
        pos += 1;
    }
    Ok(DecodeResult {
        prompt_logits,
        prompt_len: prompt.len(),
        vocab,
        generated,
        step_logits,
    })
}

/// Binds a model's weights and registers its paged KV caches (Spec 1 §4.D, Spec 3 §3.3, Spec 4 §2).
///
/// Returns `max_blocks` (`ceil(max_ctx / 32)`, Spec 3 §3.3) for step
/// [`BatchMeta`] construction. Shared by greedy decode and `r9v eval`.
pub fn prepare(exec: &mut CpuExecutor, model: &TinyModel) -> Result<u32, ExecError> {
    let spec = &model.spec;
    for (edge, buffer) in &model.weights {
        exec.bind(*edge, buffer.clone());
    }
    let max_blocks = spec.max_ctx.div_ceil(CACHE_BLOCK_TOKENS);
    for handle in &model.handles {
        exec.register_paged_cache(
            *handle,
            KvPagedCache::new(
                max_blocks as usize,
                spec.kv_heads as usize,
                spec.head_dim as usize,
                spec.head_dim as usize,
                r9v_ir::DType::F16,
            )?,
        );
    }
    Ok(max_blocks)
}

/// Runs one step (prefill or single-token decode) and returns `[T, V]` f32 logits (Spec 1 §3.1, Spec 4 §2).
///
/// Shared by greedy decode and `r9v eval`: binds per-step token and
/// position edges, builds the step [`BatchMeta`], runs the graph, and
/// reads the logits edge.
pub fn run_step(
    exec: &mut CpuExecutor,
    model: &TinyModel,
    tokens: &[u32],
    ctx_pos: u32,
    max_blocks: u32,
    rng_states: &mut Vec<crate::philox::RngState>,
) -> Result<Vec<f32>, ExecError> {
    let t = tokens.len() as u32;
    let positions: Vec<u32> = (ctx_pos..ctx_pos + t).collect();
    exec.bind(
        model.token_edge,
        TypedBuffer::from_u32(&[tokens.len()], tokens),
    );
    exec.bind(
        model.positions_edge,
        TypedBuffer::from_u32(&[tokens.len()], &positions),
    );
    let batch: BatchMeta = BatchMeta::builder(1, 1, t, max_blocks)
        .seq_ids(vec![0])
        .query_len(vec![t])
        .ctx_len(vec![ctx_pos])
        .positions(Positions::PerToken(positions.clone()))
        .slot_map(positions)
        .block_table((0..max_blocks).collect())
        .window_start(vec![0])
        .tree(None)
        .build()?;
    exec.run(
        &model.graph,
        RunArgs {
            batch: &batch,
            params: &[],
            rng: rng_states,
            ngram_hash: None,
        },
    )?;
    exec.edge(model.logits_edge)
        .ok_or(ExecError::MissingOutput {
            node: usize::MAX,
            edge: model.logits_edge.0,
        })
        .map(|buffer| buffer.to_f32_vec())
}

/// Lowest-index argmax of the last row of row-major `[rows, vocab]` logits.
fn argmax_last_row(logits: &[f32], rows: usize, vocab: usize) -> u32 {
    let start = (rows - 1) * vocab;
    let mut best = 0u32;
    let mut best_value = logits[start];
    for (i, &value) in logits[start..start + vocab].iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best = i as u32;
        }
    }
    best
}
