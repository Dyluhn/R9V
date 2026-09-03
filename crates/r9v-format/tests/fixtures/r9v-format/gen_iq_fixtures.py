"""Generate crates/r9v-format/tests/fixtures/r9v-format/iq_a24_reference.txt.

gguf-py 0.19.0 exposes NO quantize helper for IQ types (every
quantize_blocks raises NotImplementedError), so wire bytes are
hand-built deterministically from the GGML block layout documented in
gguf-py 0.19.0 quants.py: seeded index/scale payloads plus all-zero
and all-0xFF edge blocks with valid scales. Expected f32 words for
every case come from gguf-py 0.19.0 dequantize_blocks (f32LE hex) --
the independent oracle. The wire was NOT quantized by llama.cpp.

Independent validation (recorded in the fixture header): local
llama.cpp source commit dd1ea524333b1e697489067d7a4c39c60d32beee,
build-vulkan-muse libggml-base.so.0.19.0, dequantized every fixture
row bit-exact: 23,680/23,680 f32 words match y across all 9 families.

Usage (from the workspace root):
  python3 crates/r9v-format/tests/fixtures/r9v-format/gen_iq_fixtures.py
  python3 crates/r9v-format/tests/fixtures/r9v-format/gen_iq_fixtures.py --check
--check regenerates in memory and fails loudly on any byte difference;
rerunning without --check must reproduce the committed file
byte-identically. Requires gguf-py 0.19.0 exactly (asserted below):
any gguf-py version change must fail here, never silently pass.
"""
import struct
import sys
from pathlib import Path

import numpy as np

from gguf.quants import (
    IQ1_M,
    IQ1_S,
    IQ2_S,
    IQ2_XS,
    IQ2_XXS,
    IQ3_S,
    IQ3_XXS,
    IQ4_NL,
    IQ4_XS,
)

SEED = 0xA240
EXPECTED_GGUF_VERSION = "0.19.0"
# llama.cpp source commit that independently dequantized every fixture
# row bit-exact (23,680/23,680 f32 words); recorded, not executed, here.
LLAMA_CPP_COMMIT = "dd1ea524333b1e697489067d7a4c39c60d32beee"
DS = [0x3C00, 0x3800, 0x4000, 0xBC00, 0x0000]  # 1.0, 0.5, 2.0, -1.0, 0.0

SCRIPT_REL = "crates/r9v-format/tests/fixtures/r9v-format/gen_iq_fixtures.py"


def dbytes(bits):
    return np.frombuffer(struct.pack('<H', bits), dtype=np.uint8)


TYPES = [
    # (name, class, block_bytes, n_rows, k)
    ('IQ4_NL', IQ4_NL, 18, 5, 128),
    ('IQ4_XS', IQ4_XS, 136, 3, 512),
    ('IQ3_XXS', IQ3_XXS, 98, 3, 1024),
    ('IQ3_S', IQ3_S, 110, 3, 1024),
    ('IQ2_XXS', IQ2_XXS, 66, 3, 1024),
    ('IQ2_XS', IQ2_XS, 74, 3, 1024),
    ('IQ2_S', IQ2_S, 82, 3, 1024),
    ('IQ1_S', IQ1_S, 50, 3, 1024),
    ('IQ1_M', IQ1_M, 56, 3, 1024),
]


def pour_d_scale(block, name, bits):
    """Write a valid f16 scale into a block (IQ1_M packs d in scales)."""
    if name == 'IQ1_M':
        nibs = [(bits >> s) & 0xF for s in (12, 8, 4, 0)]
        for i in range(4):
            w = (nibs[i] << 12) | int(block[48 + 2 * i]) | (int(block[49 + 2 * i]) << 8 & 0x0FFF)
            # keep low 12 random bits, force top nibble
            w = (w & 0x0FFF) | (nibs[i] << 12)
            block[48 + 2 * i] = w & 0xFF
            block[49 + 2 * i] = (w >> 8) & 0xFF
    else:
        block[0:2] = dbytes(bits)


def build_lines():
    import importlib.metadata as md

    gguf_version = md.version('gguf')
    assert gguf_version == EXPECTED_GGUF_VERSION, (
        f"gguf-py must be {EXPECTED_GGUF_VERSION}, found {gguf_version}"
    )
    rng = np.random.default_rng(SEED)
    lines = []
    lines.append('# IQ A2.4 reference fixtures (phase-a-agent-breakdown.md card A2.4; tests/iq.rs).')
    lines.append('# gguf-py 0.19.0 exposes NO quantize helper for IQ types, so wire bytes are')
    lines.append('# hand-built deterministically from the GGML block layout (seeded payloads plus')
    lines.append('# all-zero and all-0xFF edge blocks with valid scales). This fixture wire was')
    lines.append('# not quantized by llama.cpp; the pinned gguf-py writer lacks IQ quantizers.')
    lines.append('# y for every case: gguf-py 0.19.0 dequantize_blocks of the wire bytes (f32LE hex).')
    lines.append('# Independent validation: local llama.cpp source commit '
                 + LLAMA_CPP_COMMIT + ',')
    lines.append('# build-vulkan-muse libggml-base.so.0.19.0, dequantized every fixture row')
    lines.append('# bit-exact: 23,680/23,680 f32 words match y across all 9 families.')
    lines.append(f'# Regenerate: python3 {SCRIPT_REL}  (--check verifies byte-identical).')
    lines.append(f'# seed 0x{SEED:x}; gguf {EXPECTED_GGUF_VERSION}; numpy {np.__version__}')
    for name, cls, bb, n, k in TYPES:
        # block counts from GGML_QUANT_SIZES instead of guessing:
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T

        qtype = getattr(T, name)
        blk_vals, blk_bytes = GGML_QUANT_SIZES[qtype]
        assert blk_bytes == bb, (name, blk_bytes, bb)
        nblocks = n * (k // blk_vals)
        wire = np.zeros((nblocks, bb), dtype=np.uint8)
        for i in range(nblocks):
            if i == 0:
                continue  # all-zero edge block (d = +0.0)
            elif i == 1:
                wire[i, :] = 0xFF
                pour_d_scale(wire[i], name, 0x3C00)  # valid d under 0xFF payload
            else:
                wire[i, :] = rng.integers(0, 256, size=bb, dtype=np.uint8)
                pour_d_scale(wire[i], name, DS[i % len(DS)])
        cls.init_grid()
        y = cls.dequantize_blocks(wire).reshape(n, k).astype(np.float32)
        lines.append(f'case {name}')
        lines.append(f'n {n}')
        lines.append(f'k {k}')
        lines.append(f'wire {wire.tobytes().hex()}')
        lines.append(f'y {y.tobytes().hex()}')
        print(name, 'wire', wire.nbytes, 'y', y.size)
    return '\n'.join(lines) + '\n'


def main():
    path = Path(__file__).with_name('iq_a24_reference.txt')
    text = build_lines()
    if '--check' in sys.argv[1:]:
        committed = path.read_text()
        if committed != text:
            for i, (a, b) in enumerate(zip(committed.splitlines(), text.splitlines())):
                if a != b:
                    print(f"--check FAILED at line {i + 1}:\n  committed: {a[:120]}\n  regen:     {b[:120]}")
                    break
            else:
                print("--check FAILED: line counts differ "
                      f"({len(committed.splitlines())} vs {len(text.splitlines())})")
            sys.exit(1)
        print(f"--check OK: regen reproduces {path.name} byte-identically")
    else:
        path.write_text(text)
        print('wrote', path)


main()
