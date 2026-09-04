# SPDX-License-Identifier: Apache-2.0
"""Deterministic ~30M Llama-family F16 GGUF fixture generator (card A1.13).

Generates, from one fixed seed and no other input:
  - ``model.gguf``: an F16 GGUF checkpoint (via ``gguf-py``) whose weights are
    bit-identical to the weights ``r9v eval`` regenerates internally from the
    same seed (mirror of ``r9v-t0`` ``synthetic`` + ``r9v-common`` ``SeededRng``
    + ``A1.10`` ``seed_for``; specs 1 ``§6.1``, 8 ``§8``),
  - ``model.json``: the matching ``SyntheticSpec`` consumed by ``r9v eval``,
  - ``prompts/seq_XX.txt``: 64 fixed token sequences (stdlib RNG only),
  - ``manifest.json``: params plus file hashes for cache validation.

Llama naming follows the ``llama`` family key set (spec 8 §4, card A1.4):
``llama.*`` metadata plus ``token_embd``/``blk.*``/``output*`` tensors, so the
A1.4 family function can bind this file once the GGUF loader path lands.
Norm vectors stay F32, exactly like real F16 checkpoints; everything else is
F16 (spec 13 §12 ``verify-arch`` runs an F16 GGUF).

The file intentionally carries no ``r9v.*`` metadata keys: per spec 2 §6 a
file whose every tensor has a standard type id and no ``r9v.*`` keys is a
standard GGUF that loads through repack, and ``r9v-format``'s ``GgufFile``
treats any ``r9v.*`` key as a native-file marker (requiring
``r9v.format_version``/``r9v.layout_id``). Test provenance (seed, prompt
count, generator identity) lives in ``manifest.json``, never in the GGUF.

Pure Python + ``numpy``/``gguf``/``xxhash``. No wall-clock, no network, no
arch-specific code paths: the same seed yields byte-identical files on any
little-endian host (GGUF is little-endian by format).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import gguf
import numpy as np
import xxhash

# A1.13 fixture shape: a Llama-family dense decoder (~29.4M params).
# Field names mirror SyntheticSpec (crates/r9v-t0/src/synthetic.rs).
FIXTURE_PARAMS = {
    "vocab": 4096,
    "dim": 512,
    "heads": 8,
    "kv_heads": 4,
    "head_dim": 64,
    "ff": 1536,
    "layers": 8,
    "theta": 10000.0,
    "seed": 0xA113,
    "max_ctx": 64,
}
# Prompt-set shape: 64 fixed sequences (spec 8 §8 / spec 13 §12).
N_SEQUENCES = 64
SEQ_LEN = 8
PROMPT_SEED = 0xA113
# DECISION(A1.13): prompts come from a stdlib Random stream, not torch/numpy,
# so sequence generation stays portable even where those packages are absent.
# Rejected torch randint because it would couple the fixed prompt set to the
# reference implementation under test. Spec 13 §12 fixes only the count (64).

# Total-parameter window that counts as "~30M" for this card.
PARAM_LO, PARAM_HI = 25_000_000, 35_000_000

_MASK64 = (1 << 64) - 1
# SplitMix64 increment and synthetic weight-counter multiplier: the same
# 0x9E3779B97F4A7C15 literal both places (r9v-common rng.rs, synthetic.rs).
_GOLDEN64 = 0x9E3779B97F4A7C15
_HARNESS_DOMAIN = b"a1.10"
_SYNTHETIC_DOMAIN = "a1.12-synthetic"


def _splitmix64(state: list[int]) -> int:
    """One SplitMix64 step, mirroring r9v-common SeededRng::new."""
    state[0] = (state[0] + _GOLDEN64) & _MASK64
    z = state[0]
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


class SeededRng:
    """Bit-exact mirror of r9v_common::SeededRng (Xoshiro256++ / SplitMix64)."""

    def __init__(self, seed: int) -> None:
        cell = [seed & _MASK64]
        words = [_splitmix64(cell) for _ in range(4)]
        if all(w == 0 for w in words):
            words = [0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0xD6E8FEB86659FD93]
        self.s = words

    @classmethod
    def from_state(cls, words: list[int]) -> SeededRng:
        """Raw-state constructor for cross-checking against r9v-common vectors."""
        rng = cls.__new__(cls)
        rng.s = [w & _MASK64 for w in words]
        return rng

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self.s
        result = (((((s0 + s3) & _MASK64) << 23) & _MASK64) | (((s0 + s3) & _MASK64) >> 41)) & _MASK64
        result = (result + s0) & _MASK64
        t = (s1 << 17) & _MASK64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = (((s3 << 45) & _MASK64) | (s3 >> 19)) & _MASK64
        self.s = [s0, s1, s2, s3]
        return result


def seed_for(op_name: str, case_idx: int, master: int) -> int:
    """Mirror of r9v-t0 harness seed_for: xxh3(domain | op | case LE | master LE)."""
    payload = (
        _HARNESS_DOMAIN
        + b"\x7c"
        + op_name.encode("ascii")
        + b"\x7c"
        + int(case_idx & _MASK64).to_bytes(8, "little")
        + int(master & _MASK64).to_bytes(8, "little")
    )
    return xxhash.xxh3_64_intdigest(payload)


def weight_counter(ordinal: int, name: str) -> int:
    """Mirror of synthetic.rs weight_counter (ordinal * GOLDEN + len(name))."""
    # NOTE: parentheses are load-bearing; & binds tighter than + in Python.
    return (((ordinal * _GOLDEN64) & _MASK64) + len(name)) & _MASK64


def uniform_f32(op_name: str, counter: int, master: int, n: int, lo: float, hi: float) -> np.ndarray:
    """Mirror of harness uniform_f32: low-24-bit draws mapped to [lo, hi] in f32."""
    rng = SeededRng(seed_for(op_name, counter, master))
    draws = np.fromiter((rng.next_u64() for _ in range(n)), dtype=np.uint64, count=n)
    top = (draws >> np.uint64(40)).astype(np.float32)
    unit = top / np.float32(16777216.0)
    lo32 = np.float32(lo)
    span32 = np.float32(hi) - lo32
    return lo32 + span32 * unit


def f32_to_f16_bits(values: np.ndarray) -> np.ndarray:
    """Bit-exact mirror of r9v-t0 dtype::f32_to_f16 (RNE, all branches).

    Vectorized over a float32 array; returns uint16 half bits.
    """
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    sign = u >> np.uint32(31)
    exp = ((u >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    mant = u & np.uint32(0x7FFFFF)
    out = np.zeros(u.shape, dtype=np.uint32)

    is_inf_nan = exp == 0xFF
    nan = mant != np.uint32(0)
    out = np.where(is_inf_nan & nan, (sign << np.uint32(15)) | np.uint32(0x7E00)
                   | np.maximum(mant >> np.uint32(13), np.uint32(1)).astype(np.uint32), out)
    out = np.where(is_inf_nan & ~nan, (sign << np.uint32(15)) | np.uint32(0x7C00), out)
    out = np.where(exp == 0, sign << np.uint32(15), out)

    finite = ~(is_inf_nan | (exp == 0))
    new_exp = exp - np.int32(127) + np.int32(15)
    overflow = finite & (new_exp >= 31)
    out = np.where(overflow, (sign << np.uint32(15)) | np.uint32(0x7C00), out)
    underflow = finite & (new_exp < -10)
    out = np.where(underflow, sign << np.uint32(15), out)
    sub = finite & (new_exp <= 0) & (new_exp >= -10)
    shift = np.where(sub, 14 - new_exp, 1).astype(np.uint32)
    full = (mant | np.uint32(0x800000)).astype(np.uint64)
    half = (np.uint64(1) << (shift.astype(np.uint64) - np.uint64(1)))
    lsb = ((full >> shift.astype(np.uint64)) & np.uint64(1)).astype(np.uint64)
    rounded = full + half - np.uint64(1) + lsb
    out = np.where(sub, (sign << np.uint32(15)) | (rounded >> shift.astype(np.uint64)).astype(np.uint32), out)
    normal = finite & (new_exp > 0) & (new_exp < 31)
    lsb_n = (mant >> np.uint32(13)) & np.uint32(1)
    bias = np.uint32(0x0FFF) + lsb_n
    rm = mant + bias
    mant_over = rm >= np.uint32(0x800000)
    final_exp = new_exp + 1
    exp_over = final_exp >= 31
    out = np.where(normal & mant_over & exp_over, (sign << np.uint32(15)) | np.uint32(0x7C00), out)
    out = np.where(
        normal & mant_over & ~exp_over,
        (sign << np.uint32(15)) | (final_exp.astype(np.uint32) << np.uint32(10)),
        out,
    )
    out = np.where(
        normal & ~mant_over,
        (sign << np.uint32(15)) | (new_exp.astype(np.uint32) << np.uint32(10)) | (rm >> np.uint32(13)),
        out,
    )
    return out.astype(np.uint16)


def build_weights(params: dict) -> list[tuple[str, np.ndarray, bool]]:
    """Weight table in synthetic.rs build order: (name, f32 values, is_norm).

    Norm vectors draw from [0.5, 1.5] (add_param); everything else from
    [-1, 1] (add_weight). Ordinals follow insertion order, mirroring
    ``weights.len()`` at each ``add_weight``/``add_param`` call.
    """
    vocab, dim = params["vocab"], params["dim"]
    heads, kv_heads = params["heads"], params["kv_heads"]
    hd, ff, layers = params["head_dim"], params["ff"], params["layers"]
    seed = params["seed"]
    table: list[tuple[str, np.ndarray, bool]] = []

    def add(name: str, shape: tuple[int, ...], is_norm: bool) -> None:
        n = 1
        for d in shape:
            n *= d
        counter = weight_counter(len(table), name)
        lo, hi = (0.5, 1.5) if is_norm else (-1.0, 1.0)
        table.append((name, uniform_f32(_SYNTHETIC_DOMAIN, counter, seed, n, lo, hi).reshape(shape), is_norm))

    h = heads * hd
    hkv = kv_heads * hd
    # Push order mirrors build(): embed, per-layer blocks, final norm, head.
    add("embed", (vocab, dim), False)
    for i in range(layers):
        tag = f"l{i}"
        add(f"{tag}_attn_norm", (dim,), True)
        add(f"{tag}_wq", (h, dim), False)
        add(f"{tag}_wk", (hkv, dim), False)
        add(f"{tag}_wv", (hkv, dim), False)
        add(f"{tag}_wo", (dim, h), False)
        add(f"{tag}_ffn_norm", (dim,), True)
        add(f"{tag}_wg", (ff, dim), False)
        add(f"{tag}_wu", (ff, dim), False)
        add(f"{tag}_wd", (dim, ff), False)
    add("final_norm", (dim,), True)
    add("lm_head", (vocab, dim), False)
    return table


# Synthetic name -> llama tensor name (spec 8 §4 weight binding).
def llama_name(synth: str) -> str:
    if synth == "embed":
        return "token_embd.weight"
    if synth == "lm_head":
        return "output.weight"
    if synth == "final_norm":
        return "output_norm.weight"
    layer, _, rest = synth.partition("_")
    idx = layer[1:]
    mapping = {
        "attn_norm": f"blk.{idx}.attn_norm.weight",
        "wq": f"blk.{idx}.attn_q.weight",
        "wk": f"blk.{idx}.attn_k.weight",
        "wv": f"blk.{idx}.attn_v.weight",
        "wo": f"blk.{idx}.attn_output.weight",
        "ffn_norm": f"blk.{idx}.ffn_norm.weight",
        "wg": f"blk.{idx}.ffn_gate.weight",
        "wu": f"blk.{idx}.ffn_up.weight",
        "wd": f"blk.{idx}.ffn_down.weight",
    }
    return mapping[rest]


def total_params(table: list[tuple[str, np.ndarray, bool]]) -> int:
    return sum(a.size for _, a, _ in table)


def _as_f16_array(values: np.ndarray) -> np.ndarray:
    """Exact-half copy of f32 values (mirrors T0 F16 weight storage)."""
    bits = f32_to_f16_bits(np.ascontiguousarray(values, dtype=np.float32))
    return bits.reshape(values.shape).view(np.float16)


def write_gguf(path: Path, params: dict, table: list[tuple[str, np.ndarray, bool]]) -> None:
    """Writes the F16 GGUF checkpoint (gguf-py writer, little-endian)."""
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name(f"r9v-a113-tiny-{total_params(table) // 1_000_000}m")
    writer.add_uint32("llama.block_count", params["layers"])
    writer.add_uint32("llama.embedding_length", params["dim"])
    writer.add_uint32("llama.feed_forward_length", params["ff"])
    writer.add_uint32("llama.attention.head_count", params["heads"])
    writer.add_uint32("llama.attention.head_count_kv", params["kv_heads"])
    writer.add_uint32("llama.attention.key_length", params["head_dim"])
    writer.add_uint32("llama.attention.value_length", params["head_dim"])
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_float32("llama.rope.freq_base", float(params["theta"]))
    writer.add_uint32("llama.rope.dimension_count", params["head_dim"])
    writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 1)
    writer.add_array("tokenizer.ggml.tokens", [f"tok_{i:04d}" for i in range(params["vocab"])])
    # Decode-graph consumption order (spec 13 §10): embed, per-block, head.
    ordered = sorted(table, key=lambda e: (0 if e[0] == "embed" else 2 if e[0] in ("lm_head", "final_norm") else 1, e[0]))
    for name, values, is_norm in ordered:
        if is_norm:
            writer.add_tensor(llama_name(name), np.ascontiguousarray(values, dtype=np.float32))
        else:
            writer.add_tensor(llama_name(name), np.ascontiguousarray(_as_f16_array(values)))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_prompts(prompts_dir: Path, vocab: int) -> list[list[int]]:
    """Writes the 64 fixed token sequences (stdlib RNG; portable)."""
    rng = random.Random(PROMPT_SEED)
    seqs = [[rng.randrange(vocab) for _ in range(SEQ_LEN)] for _ in range(N_SEQUENCES)]
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for i, ids in enumerate(seqs):
        (prompts_dir / f"seq_{i:02d}.txt").write_text(" ".join(str(t) for t in ids) + "\n", encoding="ascii")
    return seqs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> dict[str, str]:
    """sha256 of the generator scripts themselves (cache invalidation).

    A fixture cached from an older generator must never be trusted: the
    digest pins the exact ``gen_fixture.py`` + ``torch_forward.py`` bytes
    that produced it, so any edit to either script invalidates the cache.
    """
    here = Path(__file__).resolve().parent
    return {
        "gen_fixture.py": sha256_file(here / "gen_fixture.py"),
        "torch_forward.py": sha256_file(here / "torch_forward.py"),
    }


def generate(out_dir: Path) -> dict:
    """Generates the full fixture; returns the manifest dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(FIXTURE_PARAMS)
    table = build_weights(params)
    n_params = total_params(table)
    if not PARAM_LO <= n_params <= PARAM_HI:
        raise ValueError(f"fixture has {n_params} params, outside ~30M window [{PARAM_LO}, {PARAM_HI}]")
    gguf_path = out_dir / "model.gguf"
    write_gguf(gguf_path, params, table)
    (out_dir / "model.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="ascii")
    seqs = write_prompts(out_dir / "prompts", params["vocab"])
    manifest = {
        "card": "A1.13",
        "params": params,
        "total_params": n_params,
        "sequences": N_SEQUENCES,
        "seq_len": SEQ_LEN,
        "sources": source_digest(),
        "files": {
            "model.gguf": sha256_file(gguf_path),
            "model.json": sha256_file(out_dir / "model.json"),
        },
        "prompts_sha256": hashlib.sha256(
            "".join(" ".join(str(t) for t in s) for s in seqs).encode("ascii")
        ).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the A1.13 ~30M fixture.")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    manifest = generate(args.out)
    print(f"wrote {args.out}/model.gguf params={manifest['total_params']}")


if __name__ == "__main__":
    main()
