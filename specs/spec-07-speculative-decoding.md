# Spec 7 — Speculative Decoding

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 1, 3, 6. Constrains: specs 8, 10, 11.

## 0. Purpose and scope

The split between *proposing* draft tokens and *verifying* them, so that a new speculative method is a new proposer and never an engine change. Defines the `Proposer` trait, the `Draft` it returns, the verifier contract the engine guarantees, the acceptance rules, tree drafts, the four proposer kinds shipped in v1, and how draft models and heads are loaded.

Out of scope: how `k` is budgeted per step (spec 6 §4.2), the `verify` op's kernel (spec 4 §5.8), state rollback mechanics (spec 3 §3.6, §4.2).

## 1. Principles

1. **Proposers propose; the engine verifies.** A proposer's only outputs are candidate tokens, an optional tree shape, and optional draft probabilities. It never touches state, logits, or sampling.
2. **The verifier is four capabilities**, all already in specs 1, 3 and 6: a forward over several query tokens per sequence with an arbitrary mask, logits at every one of those positions, cheap state truncation, and sampling with explicit RNG. Nothing in this spec adds a fifth.
3. **Distribution-exact by default.** With draft probabilities available, verification is Leviathan/Chen rejection sampling and the output distribution equals plain sampling from the target with the same parameters. Without them, the one-hot form of the same rule applies. Greedy is the temperature-0 special case, and lossy acceptance is opt-in.
4. **Same postprocess on both sides.** Temperature, top-k/p, min-p and penalties are applied to target logits at every verified position and to draft probabilities identically, so the acceptance test compares like with like.
5. **`k` belongs to the scheduler.** A proposer states its maximum; the scheduler picks the actual `k` per step from cost and acceptance history.

## 2. Proposer trait

```
trait Proposer {
  fn kind(&self) -> ProposerKind;
  fn max_k(&self) -> u32;                        # per sequence per step
  fn max_tree(&self) -> u32;                     # 1 for linear drafts
  fn needs_device_pass(&self) -> bool;
  fn provides_probs(&self) -> bool;

  fn on_prefill(&mut self, seq: SeqId, tokens: &[u32], ctx: &PrefillCtx);   # called per prefill chunk
  fn draft(&mut self, seq: SeqId, k: u32, view: &SeqView) -> Draft;         # pre-step
  fn observe(&mut self, seq: SeqId, accepted: &[u32], ctx: &VerifyCtx);     # post-step
  fn reset(&mut self, seq: SeqId);
}

Draft {
  tokens:  Vec<u32>,            # length T_d ≤ max_tree; linear drafts have T_d = k
  parents: Option<Vec<i32>>,    # tree: parent index per token, −1 = root; None = linear chain
  probs:   Option<DraftProbs>,  # a device tensor [T_d, V] f32 of postprocessed draft probabilities (produced on device by
                                # draft/eagle/mtp passes); None means one-hot q. Host proposers never produce probs.
}

SeqView   { tokens: &[u32] (prompt + generated), ctx_len, last_hidden: Option<&DeviceTensor> }
PrefillCtx{ hidden: Option<&DeviceTensor> }      # target's pre-lm_head hidden state for this chunk, if exported
VerifyCtx { target_probs_at_accepted: &[f32], hidden: Option<&DeviceTensor>, accept_len: u32 }
```

The engine calls `draft` in pre-step (spec 6 §3.1 step 5) and `observe` in post-step. A proposer with `needs_device_pass = true` gets its own captured graph on the rank that holds the target's last hidden state (rank 0, or the last PP stage) and runs it on the compute stream before the target graph; its cost per depth `k` is measured at warmup and charged as `C_draft(k)` in the scheduler's spec-decode budget (spec 6 §4.2).

## 3. Verifier contract

What the engine guarantees to every proposer, by reference:

| capability | where |
|---|---|
| forward over `k + 1` query tokens per sequence with `Causal` or `Tree` mask | spec 1 §4.D `attention`, `BatchMeta.tree` |
| logits and postprocessed probabilities at every query position | spec 1 §4.F `logits_postprocess` over `[S, q, V]` |
| accept/resample with explicit RNG | spec 1 §4.F `verify` |
| state truncation by counter (paged) and buffer swap (recurrent) | spec 3 §3.6, §4.2 |
| compaction of a non-contiguous accepted path | spec 3 §3.6 `compact` |
| `k` and tree size chosen per step | spec 6 §4.2 |
| target's pre-`lm_head` hidden state exported as a graph output when a proposer asks for it | spec 1 §3.2 `hidden`; spec 8 sets `export_hidden = true` for `eagle` and block-parallel drafters. `mtp` consumes the hidden state inside the target graph and needs no export. |

## 4. Acceptance

The `verify` op's `method` (spec 1 §4.F):

- **`Rejection`** (default when temperature > 0). For position `i` with draft token `d_i`, target probs `p_i` and draft probs `q_i`: accept if `u_i < min(1, p_i[d_i] / q_i[d_i])`. On the first rejection at `i`, sample the replacement from `norm(max(0, p_i − q_i))` and stop. If all `k` accepted, sample the bonus token from `p_{k+1}`. Draft-free proposers (n-gram, MTP argmax) have `q_i = onehot(d_i)`, which reduces the rule to `u_i < p_i[d_i]` and the replacement to `norm(p_i with d_i zeroed)`.
- **`Greedy`** (temperature = 0): accept while `argmax p_i == d_i`; the replacement is `argmax p_i`.
- **`Typical { eps, delta }`** (opt-in, per request): Medusa-style typical acceptance for trees; accepts `d_i` if `p_i[d_i] > min(eps, delta · exp(−H(p_i)))`. Not distribution-exact; the request must set `spec.lossy = true`.

`u_i` comes from Philox keyed by `(seq_id, step, i)` (spec 1 §4.F), so a run is reproducible for a given seed and proposer. Different proposers legitimately produce different token streams; what is invariant is the distribution.

## 5. Tree drafts

- `parents` defines a forest rooted at the sequence's last verified token. Depth ≤ `k`, size ≤ `max_tree` (engine cap 16, config `tree_max`).
- Pre-step builds the ancestor mask (spec 1 §4.D.1) and assigns each tree token a scratch position (spec 3 §3.6).
- `verify` evaluates every root-to-leaf path with the acceptance rule and commits the longest accepted path; ties go to the path with the lowest first-token index. The bonus token is sampled at the end of that path.
- After verify the scheduler issues `compact` with the accepted positions, then `commit`.
- Recurrent-layer models: the accepted path is recomputed through the recurrent form (spec 3 §4.2). The scheduler adds `depth · c_recur` to the step estimate, which in practice keeps trees shallow on hybrids; that is the intended outcome, not a limitation to fix.

## 6. Proposer kinds shipped in v1

| kind | device pass | probs | tree | notes |
|---|---|---|---|---|
| `ngram` | no | no | no | prompt-lookup: longest suffix match of the last `n` tokens in `SeqView.tokens`, emit the following `k`; `min_match` tokens required. Zero cost; wins on code, repetition, RAG. |
| `mtp` | no (rides the target) | optional | optional | the model's own MTP head(s). The model definition exposes the head as a sub-graph on the target's hidden state at the last accepted position; it runs at the end of the target graph, so the next step's draft is a free by-product of this step. Multi-head models emit a tree. |
| `draft` | yes | yes | no | a second model loaded through the same engine (§7). Full rejection sampling. Best acceptance on general text when a matched small model exists. |
| `eagle` | yes | yes | yes | draft head on `(last_hidden, token_embedding)`, autoregressive over its own tiny state for `k` steps, tree by top-m branching per step. Requires `export_hidden`. |

Block-parallel drafters (DFlash-style, one device pass emitting `k` tokens jointly) are a fifth kind with `needs_device_pass = true` and `probs` as the drafter provides; nothing in the engine distinguishes them from `eagle` except their own graph. They are not in v1 but need no spec change.

`proposer = auto` resolves to: `mtp` if the model has MTP weights, else `eagle` if `eagle_head` is configured, else `draft` if `draft_model` is configured, else `ngram`.

## 7. Draft models and heads

- A `draft` model is a full engine load (spec 9) on the rank that holds the target's logits, with its own plan (single device), tune entries, graph captures, and state pools sized `max_seqs × max_ctx` at its own per-token cost. It uses the target's tokenizer; a vocab mismatch is a load error unless a vocab map is supplied in the model definition.
- Its prefill runs chunk-for-chunk alongside the target's (`on_prefill`), so it is ready at the first decode step. It has its own prefix cache.
- Its `draft` call runs `k` sequential decode steps of the small model (one captured graph replayed `k` times, or one `k`-step unrolled graph when `k` is fixed by the scheduler for the step).
- An `eagle` head is loaded as a partial model definition (a few layers plus embeddings) and needs `export_hidden`. Its state is `KvPaged` with its own small pool.
- VRAM for either is charged against the target's budget before state pools are sized (spec 3 §6.3), so `max_ctx` reflects what is actually left.

## 8. Metrics

Per sequence: `k` chosen, `accept_len`, `accept_ema`, tree size, verify share of step time. Per proposer: global acceptance rate (the seed for new sequences' `accept_ema`), draft cost per step, tokens per step. Exposed through spec 11 and shown in the doctor bundle summary, so "spec decode isn't helping" is answerable from one table.

## 9. Adding a method

1. Implement `Proposer` as a crate behind a feature flag (compile-time; dynamic plugin loading is not in v1).
2. If it needs a device pass, define its graph with the spec 8 builder; it gets a registry, tune entries and captures like any model.
3. If it needs a new acceptance rule, that is a new `verify.method` value and goes through the spec 1 RFC with a proof or reference that the rule is distribution-exact, or a `lossy` flag if it isn't.
4. Add it to the `auto` resolution order if it should be preferred, with the metrics evidence.

No change to attention, state, scheduler or sampling ops is required for steps 1–2. That is the test of this spec: if a new method needs one, this spec has a gap.

## 10. Config

```
[spec]
proposer     = "auto" | "none" | "ngram" | "mtp" | "draft" | "eagle"
k_max        = 8            # spec 6
tree_max     = 16
lossy        = false        # enables Typical acceptance per request when the request asks
[spec.ngram]
n            = 3
min_match    = 2
```

The draft model and eagle head paths are `load.draft_model` and `load.eagle_head` (spec 9 §13), since they are loads.
