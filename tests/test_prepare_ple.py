from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_ple.py"
SPEC = importlib.util.spec_from_file_location("prepare_ple", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_ple = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_ple
SPEC.loader.exec_module(prepare_ple)


def test_sample_offsets_cover_start_middle_and_end() -> None:
    size = 64 * 1024
    offsets = prepare_ple.sample_offsets(size)
    assert offsets == (0, size // 2 - 2048, size - 4096)


def test_validate_samples_uses_tensor_subrange(tmp_path: Path) -> None:
    prefix = b"prefix" * 1000
    payload = bytes(index % 251 for index in range(64 * 1024))
    suffix = b"suffix" * 1000
    source = tmp_path / "source.gguf"
    target = tmp_path / "ple.bin"
    source.write_bytes(prefix + payload + suffix)
    target.write_bytes(payload)

    span = prepare_ple.TensorSpan(
        source=str(source),
        tensor_name="per_layer_token_embd.weight",
        data_offset=len(prefix),
        packed_bytes=len(payload),
        tensor_type="IQ4_NL",
        gguf_shape=(160, 728),
    )
    assert prepare_ple.validate_samples(span, target)

    damaged = bytearray(payload)
    damaged[-1] ^= 0xFF
    target.write_bytes(damaged)
    assert not prepare_ple.validate_samples(span, target)


def test_parse_shape() -> None:
    assert prepare_ple.parse_shape("160,320001536") == (160, 320001536)
