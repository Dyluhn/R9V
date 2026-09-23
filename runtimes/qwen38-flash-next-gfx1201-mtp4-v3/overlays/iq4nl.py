# SPDX-License-Identifier: Apache-2.0
"""IQ4_NL dequantization on CPU in torch, bit-identical to gguf-py's numpy version.

The n-gram (PLE) offload worker dequantizes every gathered table row on CPU. gguf-py's
numpy path costs ~30-60 us per prompt token there (single-threaded, while both GPUs
wait); these torch ops run multi-threaded. Same arithmetic: float32(d) * float32(level),
so the float32 result and any later cast are bit-identical.
"""
import torch

BLOCK_BYTES = 18  # fp16 scale + 16 bytes of 4-bit indices
BLOCK_VALUES = 32
_LEVELS = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.float32,
)
# Level of each byte's low / high nibble: one lookup per byte instead of per nibble.
_LOW = _LEVELS[torch.arange(256) & 0x0F]
_HIGH = _LEVELS[torch.arange(256) >> 4]


def dequantize_iq4_nl(quant: torch.Tensor) -> torch.Tensor:
    """[rows, k*18] uint8 -> [rows, k*32] float32 (gguf layout: low nibbles, then high)."""
    rows = quant.shape[0]
    blocks = quant.reshape(-1, BLOCK_BYTES)
    scale = blocks[:, :2].contiguous().view(torch.float16).float()
    indices = blocks[:, 2:].long()
    out = torch.empty((blocks.shape[0], BLOCK_VALUES), dtype=torch.float32)
    torch.mul(_LOW[indices], scale, out=out[:, :16])
    torch.mul(_HIGH[indices], scale, out=out[:, 16:])
    return out.view(rows, -1)
