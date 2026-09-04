# SPDX-License-Identifier: Apache-2.0
"""Torch reference forward for the A1.13 fixture (spec 13 §12, spec 8 §8).

Implements the fixture's dense Llama-family decoder in torch float32,
mirroring the T0 CPU execution order op by op (spec 1 §6; embed → per-layer
rms-norm/GQA/rope/SwiGLU with residuals → final norm → head). Every op output
is rounded to F16 and back, exactly like T0's F16 edge storage, and every
reduction accumulates in T0's ascending index order (spec 1 §6.2–§6.4), so
remaining differences are libm transcendental rounding (exp/cos/sin/sqrt) at
~1e-7, which the spec 1 §6.1 logits criteria (top-1, KL) absorb.

Weights are read from the fixture GGUF (never regenerated here), so the
comparison genuinely covers the checkpoint bytes. Deterministic: single
thread, deterministic algorithms, no RNG anywhere in this module.
"""

from __future__ import annotations

from pathlib import Path

import gguf
import numpy as np
import torch

# [T, H, D] head views documented per op.


def _f16(x: torch.Tensor) -> torch.Tensor:
    """Round an f32 tensor to F16 storage and back (T0 F16 edge semantics)."""
    return x.to(torch.float16).to(torch.float32)


def _mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Dense matmul with ascending-index f32 accumulation (T0 matmul branch D).

    ``x`` is ``[T, K]``, ``w`` is ``[N, K]`` row-major, result ``[T, N]`` with
    ``out[t, n] = sum_k x[t, k] * w[n, k]`` accumulated in ascending ``k``
    (spec 1 §6.2; the fixed reduction order of the T0 reference). An explicit
    loop: blocked kernels (``@``, ``sum``, ``cumsum``) accumulate in a
    different order, and the f32 summation noise flips F16 rounding
    boundaries downstream, which the end-to-end comparison cannot absorb.
    """
    out = torch.zeros(x.shape[0], w.shape[0], dtype=torch.float32)
    wk = w.unsqueeze(0)  # [1, N, K]
    for k in range(x.shape[1]):
        out += x[:, k].unsqueeze(-1) * wk[:, :, k]
    return out


def _rms_norm_sum_sq(x: torch.Tensor) -> torch.Tensor:
    """Ascending-index sum of squares over the last axis (T0 norm, spec 1 §6.4)."""
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        v = x[..., i]
        acc += v * v
    return acc


def load_weights(gguf_path: Path) -> dict[str, torch.Tensor]:
    """Reads every fixture tensor from the GGUF file as f32 (exact for F16/F32)."""
    reader = gguf.GGUFReader(str(gguf_path))
    out: dict[str, torch.Tensor] = {}
    for i in range(len(reader.tensors)):
        tensor = reader.get_tensor(i)
        # Copy: gguf-py hands out read-only views torch must not alias.
        out[tensor.name] = torch.from_numpy(np.array(tensor.data, copy=True)).to(torch.float32)
    return out


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS norm over the last axis (T0 norm Last/RMS, f32 accumulate)."""
    # [T, D] -> [T, D]
    mean_sq = (_rms_norm_sum_sq(x) / float(x.shape[-1])).unsqueeze(-1)
    # NOTE: `1 / sqrt`, not rsqrt: T0 divides by the rooted sum (two
    # roundings) and hardware rsqrt is a different approximation.
    return _f16(x * (1.0 / torch.sqrt(mean_sq + eps)) * w)


def rope_neox(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Neox RoPE, full rotary dim, positions 0..T-1 (T0 rope, f32 cos/sin)."""
    # [T, H, D] -> [T, H, D]
    t, _, d = x.shape
    half = d // 2
    pos = torch.arange(t, dtype=torch.float32).unsqueeze(1)  # [T, 1]
    freq = torch.arange(half, dtype=torch.float32)  # [D/2]
    inv_freq = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float32), (2.0 * freq) / float(d))
    angle = pos * inv_freq.unsqueeze(0)  # [T, D/2]
    cos, sin = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., :half], x[..., half:]
    first = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
    second = x2 * cos.unsqueeze(1) + x1 * sin.unsqueeze(1)
    return _f16(torch.cat([first, second], dim=-1))


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, groups: int, scale: float
) -> torch.Tensor:
    """Causal scaled dot-product attention with GQA repeat (T0 attention_paged)."""
    # q/k/v: [T, H, D] / [T, Hkv, D] / [T, Hkv, D] -> [T, H, D]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    # Batch over heads so scores run over key positions, not heads.
    qh = q.transpose(0, 1)  # [H, T, D]
    kh = k.transpose(0, 1)  # [H, T, D]
    vh = v.transpose(0, 1)  # [H, T, D]
    t = q.shape[0]
    # Ascending-d f32 dots (T0 attention_paged score loop, spec 1 §6.3),
    # then the f32 scale multiply, exactly like `s *= scale`.
    scores = torch.zeros(qh.shape[0], t, t, dtype=torch.float32)
    for d in range(qh.shape[-1]):
        scores += qh[:, :, d].unsqueeze(-1) * kh[:, :, d].unsqueeze(-2)
    scores = scores * scale
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # Ascending-position probability-weighted sum (T0's online accumulator
    # visits slots in the same order; the rescaling algebra is identical).
    out = torch.zeros_like(vh)
    for s in range(t):
        out += probs[:, :, s].unsqueeze(-1) * vh[:, s, :].unsqueeze(-2)
    return _f16(out.transpose(0, 1))


def forward(weights: dict[str, torch.Tensor], tokens: list[int], params: dict) -> torch.Tensor:
    """Full prefill: token ids -> [T, V] f32 logits."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    heads, kv_heads = params["heads"], params["kv_heads"]
    hd, theta = params["head_dim"], float(params["theta"])
    groups = heads // kv_heads
    scale = 1.0 / (hd**0.5)

    ids = torch.tensor(tokens, dtype=torch.long)
    x = _f16(weights["token_embd.weight"][ids])  # [T, D]
    t = x.shape[0]
    for i in range(params["layers"]):
        p = f"blk.{i}."
        h = rms_norm(x, weights[p + "attn_norm.weight"])  # [T, D]
        q = _f16(_mm(h, weights[p + "attn_q.weight"]))  # [T, H*Dhd]
        k = _f16(_mm(h, weights[p + "attn_k.weight"]))
        v = _f16(_mm(h, weights[p + "attn_v.weight"]))
        qr = rope_neox(q.view(t, heads, hd), theta)
        kr = rope_neox(k.view(t, kv_heads, hd), theta)
        kr, v = _f16(kr), _f16(v.view(t, kv_heads, hd))
        o = attention(qr, kr, v, groups, scale).reshape(t, heads * hd)  # [T, H*Dhd]
        x = _f16(x + _f16(_mm(o, weights[p + "attn_output.weight"])))
        h2 = rms_norm(x, weights[p + "ffn_norm.weight"])  # [T, D]
        g = _f16(_mm(h2, weights[p + "ffn_gate.weight"]))  # [T, FF]
        u = _f16(_mm(h2, weights[p + "ffn_up.weight"]))  # [T, FF]
        down = _f16(_mm(_f16(torch.nn.functional.silu(g) * u), weights[p + "ffn_down.weight"]))
        x = _f16(x + down)
    x = rms_norm(x, weights["output_norm.weight"])
    return _f16(_mm(x, weights["output.weight"]))
