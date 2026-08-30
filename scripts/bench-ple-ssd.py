#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-page A/B for the SSD-backed packed PLE embedding table."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import gguf
import torch

from vllm_gguf_plugin.quantization.params import (
    _MADV_DONTNEED,
    _MADV_RANDOM,
    _MmapRowReadahead,
    _madvise_tensor,
)
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("--row-bytes", type=int, default=90)
    parser.add_argument("--lookup-rows", type=int, default=16_384)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _drop_file_pages(table: torch.Tensor, fd: int, nbytes: int) -> None:
    _madvise_tensor(table, _MADV_DONTNEED)
    os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_DONTNEED)


def main() -> None:
    args = _parse_args()
    path = args.table.expanduser().resolve(strict=True)
    nbytes = path.stat().st_size
    if args.row_bytes <= 0 or nbytes % args.row_bytes:
        raise ValueError(
            f"{path} has {nbytes} bytes, not packed {args.row_bytes}-byte rows"
        )
    num_rows = nbytes // args.row_bytes
    table = torch.from_file(
        str(path), shared=False, size=nbytes, dtype=torch.uint8
    ).reshape(num_rows, args.row_bytes)
    _madvise_tensor(table, _MADV_RANDOM)
    readahead = _MmapRowReadahead(table)
    generator = torch.Generator().manual_seed(args.seed)
    batches = [
        torch.randint(0, num_rows, (args.lookup_rows,), generator=generator)
        for _ in range(args.iterations)
    ]
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    results: dict[str, list[dict[str, float | int]]] = {
        "plain": [],
        "readahead": [],
    }
    try:
        # Alternate arm order so the second arm is not systematically favored.
        for iteration, row_ids in enumerate(batches):
            arms = ("plain", "readahead") if iteration % 2 == 0 else (
                "readahead",
                "plain",
            )
            for arm in arms:
                _drop_file_pages(table, fd, nbytes)
                gc.collect()
                advice_start = time.perf_counter_ns()
                if arm == "readahead":
                    pages, ranges, advised_bytes = readahead.prepare(row_ids)
                else:
                    pages = ranges = advised_bytes = 0
                advice_end = time.perf_counter_ns()
                quant = torch.index_select(table, dim=0, index=row_ids)
                gather_end = time.perf_counter_ns()
                output = torch.from_numpy(
                    gguf.dequantize(
                        quant.contiguous().numpy(),
                        gguf.GGMLQuantizationType.IQ4_NL,
                    )
                ).to(torch.bfloat16)
                dequant_end = time.perf_counter_ns()
                checksum = float(output[:16].float().sum().item())
                checksum_end = time.perf_counter_ns()
                results[arm].append(
                    {
                        "iteration": iteration,
                        "advice_ms": (advice_end - advice_start) / 1e6,
                        "gather_ms": (gather_end - advice_end) / 1e6,
                        "dequant_ms": (dequant_end - gather_end) / 1e6,
                        "checksum_ms": (checksum_end - dequant_end) / 1e6,
                        "total_ms": (dequant_end - advice_start) / 1e6,
                        "pages": pages,
                        "ranges": ranges,
                        "advised_bytes": advised_bytes,
                        "checksum": checksum,
                    }
                )
                del output, quant
    finally:
        _drop_file_pages(table, fd, nbytes)
        os.close(fd)

    summary = {}
    for arm, samples in results.items():
        summary[arm] = {
            field: statistics.median(float(sample[field]) for sample in samples)
            for field in (
                "advice_ms",
                "gather_ms",
                "dequant_ms",
                "checksum_ms",
                "total_ms",
            )
        }
    plain_ms = summary["plain"]["total_ms"]
    readahead_ms = summary["readahead"]["total_ms"]
    report = {
        "table": str(path),
        "table_bytes": nbytes,
        "row_bytes": args.row_bytes,
        "num_rows": num_rows,
        "lookup_rows": args.lookup_rows,
        "iterations": args.iterations,
        "samples": results,
        "median": summary,
        "speedup": plain_ms / readahead_ms,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
