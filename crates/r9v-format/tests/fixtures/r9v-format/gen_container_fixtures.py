#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic GGUF fixtures for card A2.5 container tests.

Provenance: generated with gguf-py 0.19.0 (pinned, local install, no
network) via ``python3 gen_container_fixtures.py`` from this
directory. Tensor payloads are seeded numpy bytes cut to the exact
gguf-py ``GGML_QUANT_SIZES`` wire length, so the files exercise the
reader/writer layout without claiming real quantized values. The
The ``llama_vocab_bert_bge.gguf`` and ``llama_tiny_q80.hex`` fixtures
alongside these are, in contrast, genuine llama.cpp-produced files
(the latter generated via ``llama-quantize`` with 4 tensors).

Outputs (all small, committed as hex text; *.gguf binaries are git-ignored):
  a25_standard.hex           standard GGUF, 12 tensors, all 13 KV types
  a25_split-00001-of-00002.hex / a25_split-00002-of-00002.hex
                             two-shard split with split.* keys
  llama_tiny_q80.hex         genuine llama.cpp llama-quantize model with 4 tensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

try:
    import gguf
    from gguf.constants import GGMLQuantizationType
except ImportError:
    sys.exit("gguf-py 0.19.0 required: pip install gguf==0.19.0")


def wire_bytes(rng: np.random.Generator, ggml: GGMLQuantizationType, n_elems: int) -> np.ndarray:
    from gguf.constants import GGML_QUANT_SIZES

    block, size = GGML_QUANT_SIZES[ggml]
    assert n_elems % block == 0, (ggml, n_elems)
    nbytes = n_elems * size // block
    return rng.integers(0, 256, size=nbytes, dtype=np.uint8)


def add_all_kv_types(writer: gguf.GGUFWriter) -> None:
    writer.add_uint8("a25.u8", 200)
    writer.add_int8("a25.i8", -5)
    writer.add_uint16("a25.u16", 60000)
    writer.add_int16("a25.i16", -3000)
    writer.add_uint32("a25.u32", 3_000_000_000)
    writer.add_int32("a25.i32", -2_000_000_000)
    writer.add_float32("a25.f32", 0.5)
    writer.add_bool("a25.bool_true", True)
    writer.add_bool("a25.bool_false", False)
    writer.add_string("a25.str", "r9v-a2.5 ☃")
    # Note: gguf-py infers every Python int array as INT32 (its
    # documented 64-bit TODO); only `bytes` writes UINT8 arrays.
    writer.add_array("a25.arr_i32_small", [1, 2, 3])
    writer.add_array("a25.arr_bytes", b"\x01\x02\x03")
    writer.add_array("a25.arr_i32", [-1, 0, 1])
    writer.add_array("a25.arr_f32", [0.25, 1.5])
    writer.add_array("a25.arr_bool", [True, False])
    writer.add_array("a25.arr_str", ["aa", "b", "ccc"])
    # 64-bit arrays need explicit types: gguf-py infers every int
    # as INT32 (its documented 64-bit TODO).
    from gguf import GGUFValue
    from gguf.constants import GGUFValueType

    writer.add_key_value(
        "a25.arr_u64", [2**40, 2**63], GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64
    )
    writer.add_key_value(
        "a25.arr_i64", [-(2**40), 2**62], GGUFValueType.ARRAY, sub_type=GGUFValueType.INT64
    )
    writer.add_key_value(
        "a25.arr_f64", [3.14159265358979], GGUFValueType.ARRAY, sub_type=GGUFValueType.FLOAT64
    )
    writer.add_uint64("a25.u64", 2**63 + 123)
    writer.add_int64("a25.i64", -(2**62))
    writer.add_float64("a25.f64", 2.718281828459045)


def build_standard(path: Path) -> None:
    rng = np.random.default_rng(0xA25)
    writer = gguf.GGUFWriter(str(path), arch="llama")

    writer.add_string("general.name", "a2.5-standard-fixture")
    writer.add_uint32("llama.block_count", 2)
    writer.add_uint32("llama.context_length", 128)
    writer.add_array("tokenizer.ggml.tokens", ["<unk>", "<s>", "hello"])
    writer.add_array("tokenizer.ggml.scores", [0.0, 0.0, -1.5])
    add_all_kv_types(writer)
    # (name, type, logical shape); writer takes numpy shape, file stores reversed.
    tensors = [
        ("bias_f32", GGMLQuantizationType.F32, (16,)),
        ("norm_f16", GGMLQuantizationType.F16, (16, 32)),
        ("w_q80", GGMLQuantizationType.Q8_0, (16, 32)),
        ("w_q40", GGMLQuantizationType.Q4_0, (16, 32)),
        ("w_q41", GGMLQuantizationType.Q4_1, (16, 32)),
        ("w_q50", GGMLQuantizationType.Q5_0, (16, 32)),
        ("w_q51", GGMLQuantizationType.Q5_1, (16, 32)),
        ("w_q2k", GGMLQuantizationType.Q2_K, (16, 256)),
        ("w_q3k", GGMLQuantizationType.Q3_K, (16, 256)),
        ("w_q4k", GGMLQuantizationType.Q4_K, (16, 256)),
        ("w_q5k", GGMLQuantizationType.Q5_K, (16, 256)),
        ("w_q6k", GGMLQuantizationType.Q6_K, (16, 256)),
    ]
    for name, gtype, shape in tensors:
        n_elems = int(np.prod(shape))
        raw = wire_bytes(rng, gtype, n_elems)
        if gtype in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
        ):
            arr = raw.view(np.float32 if gtype == GGMLQuantizationType.F32 else np.float16)
            writer.add_tensor(name, arr.reshape(shape))
        else:
            # Raw quantized bytes: gguf-py expects the byte shape
            # (rows, bytes-per-row) and recovers logical dims itself.
            row_bytes = raw.nbytes // shape[0]
            writer.add_tensor(
                name,
                raw.reshape((shape[0], row_bytes)),
                raw_shape=np.asarray((shape[0], row_bytes)),
                raw_dtype=gtype,
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def build_split(out1: Path, out2: Path) -> None:
    rng = np.random.default_rng(0x511)
    for idx, (path, names) in enumerate(
        [(out1, ["shard.a_q80", "shard.b_f16"]), (out2, ["shard.c_q40"])]
    ):
        writer = gguf.GGUFWriter(str(path), arch="llama")
        writer.add_architecture()
        writer.add_string("general.name", "a2.5-split-fixture")
        writer.add_uint16("split.no", idx)
        writer.add_uint16("split.count", 2)
        writer.add_int32("split.tensors.count", 3)
        for name in names:
            gtype = {
                "shard.a_q80": GGMLQuantizationType.Q8_0,
                "shard.b_f16": GGMLQuantizationType.F16,
                "shard.c_q40": GGMLQuantizationType.Q4_0,
            }[name]
            shape = (16, 32)
            raw = wire_bytes(rng, gtype, int(np.prod(shape)))
            if gtype == GGMLQuantizationType.F16:
                writer.add_tensor(name, raw.view(np.float16).reshape(shape))
            else:
                row_bytes = raw.nbytes // shape[0]
                writer.add_tensor(
                    name,
                    raw.reshape((shape[0], row_bytes)),
                    raw_shape=np.asarray((shape[0], row_bytes)),
                    raw_dtype=gtype,
                )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


def to_hex(path: Path) -> None:
    data = path.read_bytes()
    path.with_suffix(".hex").write_text(data.hex() + "\n")
    path.unlink()


def main() -> None:
    std = HERE / "a25_standard.gguf"
    sp1 = HERE / "a25_split-00001-of-00002.gguf"
    sp2 = HERE / "a25_split-00002-of-00002.gguf"
    build_standard(std)
    build_split(sp1, sp2)
    to_hex(std)
    to_hex(sp1)
    to_hex(sp2)
    print("wrote a25 fixtures (.hex)")


if __name__ == "__main__":
    main()
