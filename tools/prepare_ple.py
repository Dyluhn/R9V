#!/usr/bin/env python3
"""Locate and extract Qwen3.8's packed PLE tensor from GGUF metadata.

The extracted file is a byte-for-byte view of the GGUF tensor payload. The
copy is atomic, sample-validated, and accompanied by a JSON provenance file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import tempfile
import time
from typing import Iterable, Sequence


DEFAULT_TENSOR_NAME = "per_layer_token_embd.weight"
DEFAULT_EXPECTED_BYTES = 28_800_138_240
DEFAULT_EXPECTED_TYPE = "IQ4_NL"
DEFAULT_EXPECTED_SHAPE = (160, 320_001_536)
COPY_CHUNK_BYTES = 16 * 1024 * 1024
SAMPLE_BYTES = 4096


@dataclass(frozen=True)
class TensorSpan:
    source: str
    tensor_name: str
    data_offset: int
    packed_bytes: int
    tensor_type: str
    gguf_shape: tuple[int, ...]


def _load_gguf_reader():
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit(
            "The gguf Python package is required. Install llama.cpp's gguf-py "
            "package or run this tool in the R9V/vLLM GGUF environment."
        ) from exc
    return GGUFReader


def locate_tensor(paths: Iterable[Path], tensor_name: str) -> TensorSpan:
    gguf_reader = _load_gguf_reader()
    matches: list[TensorSpan] = []

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"GGUF shard does not exist: {path}")
        reader = gguf_reader(str(path), "r")
        try:
            for tensor in reader.tensors:
                if tensor.name != tensor_name:
                    continue
                matches.append(
                    TensorSpan(
                        source=str(path.resolve()),
                        tensor_name=tensor.name,
                        data_offset=int(tensor.data_offset),
                        packed_bytes=int(tensor.n_bytes),
                        tensor_type=tensor.tensor_type.name,
                        gguf_shape=tuple(int(value) for value in tensor.shape),
                    )
                )
        finally:
            del reader
            gc.collect()

    if not matches:
        raise ValueError(f"Tensor {tensor_name!r} was not found in the supplied shards")
    if len(matches) != 1:
        locations = ", ".join(match.source for match in matches)
        raise ValueError(
            f"Expected exactly one {tensor_name!r} tensor, found {len(matches)}: "
            f"{locations}"
        )
    return matches[0]


def validate_span(
    span: TensorSpan,
    *,
    expected_bytes: int,
    expected_type: str,
    expected_shape: Sequence[int],
) -> None:
    source_size = os.stat(span.source).st_size
    if span.data_offset < 0 or span.packed_bytes <= 0:
        raise ValueError(f"Invalid GGUF tensor range: {span}")
    if span.data_offset + span.packed_bytes > source_size:
        raise ValueError(
            "GGUF tensor range extends beyond its source shard: "
            f"end={span.data_offset + span.packed_bytes}, source={source_size}"
        )
    if span.packed_bytes != expected_bytes:
        raise ValueError(
            f"Packed-size mismatch: metadata={span.packed_bytes}, "
            f"expected={expected_bytes}"
        )
    if span.tensor_type != expected_type:
        raise ValueError(
            f"Qtype mismatch: metadata={span.tensor_type}, expected={expected_type}"
        )
    shape = tuple(expected_shape)
    if span.gguf_shape != shape:
        raise ValueError(
            f"GGUF-shape mismatch: metadata={span.gguf_shape}, expected={shape}"
        )


def sample_offsets(size: int, sample_bytes: int = SAMPLE_BYTES) -> tuple[int, ...]:
    if size < sample_bytes:
        return (0,)
    middle = max(0, size // 2 - sample_bytes // 2)
    end = size - sample_bytes
    return tuple(dict.fromkeys((0, middle, end)))


def validate_samples(span: TensorSpan, output: Path) -> bool:
    if not output.is_file() or output.stat().st_size != span.packed_bytes:
        return False
    with open(span.source, "rb", buffering=0) as source, open(
        output, "rb", buffering=0
    ) as target:
        for relative_offset in sample_offsets(span.packed_bytes):
            source_sample = os.pread(
                source.fileno(),
                SAMPLE_BYTES,
                span.data_offset + relative_offset,
            )
            target_sample = os.pread(
                target.fileno(), SAMPLE_BYTES, relative_offset
            )
            if source_sample != target_sample:
                return False
    return True


def copy_span(span: TensorSpan, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if validate_samples(span, output):
            print(f"Existing PLE file is sample-valid: {output}")
            return "existing-sample-validated"
        raise FileExistsError(
            f"Refusing to overwrite an existing invalid or unexpected file: {output}"
        )

    temp_path: Path | None = None
    digest = hashlib.sha256()
    copied = 0
    next_report = 1 << 30
    started = time.monotonic()

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            buffering=0,
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".partial",
            delete=False,
        ) as target, open(span.source, "rb", buffering=0) as source:
            temp_path = Path(target.name)
            source.seek(span.data_offset)
            remaining = span.packed_bytes
            while remaining:
                chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise OSError(
                        f"Unexpected EOF after {copied} of {span.packed_bytes} bytes"
                    )
                target.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                remaining -= len(chunk)
                if copied >= next_report or not remaining:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        f"copied={copied}/{span.packed_bytes} "
                        f"rate_mib_s={copied / elapsed / (1024 * 1024):.1f}",
                        file=sys.stderr,
                    )
                    next_report += 1 << 30
            os.fsync(target.fileno())

        os.replace(temp_path, output)
        temp_path = None
        if not validate_samples(span, output):
            raise OSError("Extracted PLE file failed post-copy sample validation")
        return digest.hexdigest()
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_manifest(output: Path, span: TensorSpan, payload_sha256: str) -> None:
    manifest = {
        "format": "r9v-gguf-tensor-extract-v1",
        "tensor": asdict(span),
        "output": str(output.resolve()),
        "payload_sha256": payload_sha256,
        "sample_bytes": SAMPLE_BYTES,
        "sample_offsets": list(sample_offsets(span.packed_bytes)),
    }
    manifest_path = output.with_name(f"{output.name}.manifest.json")
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR_NAME)
    parser.add_argument("--expected-bytes", type=int, default=DEFAULT_EXPECTED_BYTES)
    parser.add_argument("--expected-type", default=DEFAULT_EXPECTED_TYPE)
    parser.add_argument(
        "--expected-shape",
        type=parse_shape,
        default=DEFAULT_EXPECTED_SHAPE,
        metavar="D0,D1,...",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="locate and validate metadata without writing the extracted payload",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    span = locate_tensor(args.shards, args.tensor_name)
    validate_span(
        span,
        expected_bytes=args.expected_bytes,
        expected_type=args.expected_type,
        expected_shape=args.expected_shape,
    )
    print(json.dumps(asdict(span), indent=2))
    if args.dry_run:
        return 0

    payload_sha256 = copy_span(span, args.output)
    write_manifest(args.output, span, payload_sha256)
    print(f"Prepared PLE payload: {args.output}")
    print(f"payload_sha256={payload_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
