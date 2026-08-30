#!/usr/bin/env python3
"""Benchmark the frozen Muse V1 R9V proof engine without changing its runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CELLS = {
    "pp512": (512, 1, 1, 1),
    "pp2048": (2048, 1, 1, 1),
    "pp8192": (8192, 1, 1, 1),
    "tg256": (8, 256, 1, 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_once(binary: Path, hsaco: Path, model: Path, argv: tuple[int, ...]) -> dict:
    command = [str(binary), str(hsaco), str(model), *(str(value) for value in argv)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    if report.get("schema") != 1:
        raise RuntimeError("R9V report has an unexpected schema")
    generated_ids = report.pop("generated_ids", None)
    if generated_ids is not None:
        encoded = b"".join(int(token).to_bytes(4, "little") for token in generated_ids)
        report["generated_ids_count"] = len(generated_ids)
        report["generated_ids_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def metric(cell: str, report: dict) -> float:
    if cell.startswith("tg"):
        return float(report["raw_tg"])
    return float(report["prompt_tokens"]) / float(report["prompt_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--hsaco", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--cells",
        default=",".join(CELLS),
        help="comma-separated benchmark cells (default: all)",
    )
    parser.add_argument("--skip-model-hash", action="store_true")
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    selected_cells = [cell.strip() for cell in args.cells.split(",") if cell.strip()]
    unknown_cells = sorted(set(selected_cells).difference(CELLS))
    if not selected_cells or unknown_cells:
        parser.error(f"invalid --cells selection; unknown={unknown_cells}")
    for path in (args.binary, args.hsaco, args.model):
        if not path.is_file():
            parser.error(f"missing file: {path}")

    results: dict[str, dict] = {}
    for cell in selected_cells:
        argv = CELLS[cell]
        print(f"[muse-v1-bench] {cell}: warmup", file=sys.stderr, flush=True)
        run_once(args.binary, args.hsaco, args.model, argv)
        reports = []
        values = []
        for index in range(args.repetitions):
            print(
                f"[muse-v1-bench] {cell}: sample {index + 1}/{args.repetitions}",
                file=sys.stderr,
                flush=True,
            )
            report = run_once(args.binary, args.hsaco, args.model, argv)
            reports.append(report)
            values.append(metric(cell, report))
        results[cell] = {
            "unit": "tok/s",
            "samples": values,
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "raw_reports": reports,
        }

    output = {
        "schema": "r9v.muse-v1-speed-benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "warmups_per_cell": 1,
            "measured_repetitions_per_cell": args.repetitions,
            "prompt_cells": [512, 2048, 8192],
            "tg_generated_tokens": 256,
            "graph_replays": 1,
        },
        "identity": {
            "binary": {
                "path": str(args.binary),
                "bytes": args.binary.stat().st_size,
                "sha256": sha256(args.binary),
            },
            "hsaco": {
                "path": str(args.hsaco),
                "bytes": args.hsaco.stat().st_size,
                "sha256": sha256(args.hsaco),
            },
            "model": {
                "path": str(args.model),
                "bytes": args.model.stat().st_size,
                "sha256": None if args.skip_model_hash else sha256(args.model),
            },
        },
        "results": results,
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
