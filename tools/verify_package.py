#!/usr/bin/env python3
"""Verify an installed R9V model package against its immutable descriptor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _load_package(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read package descriptor {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema") != "r9v.model-package.v1":
        raise SystemExit(f"{path}: expected an r9v.model-package.v1 object")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise SystemExit(f"{path}: artifacts must be a list")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    descriptor: Path,
    model_dir: Path,
    *,
    verify_hashes: bool,
    require_optional: bool,
) -> int:
    package = _load_package(descriptor)
    failures: list[str] = []
    checked = 0
    skipped = 0
    for artifact in package["artifacts"]:
        if not isinstance(artifact, dict):
            failures.append("descriptor contains a non-object artifact")
            continue
        relative = artifact.get("path")
        expected_bytes = artifact.get("bytes")
        expected_sha = artifact.get("sha256")
        required = artifact.get("required", True)
        if not isinstance(relative, str) or not isinstance(expected_bytes, int):
            failures.append(f"invalid artifact entry: {artifact!r}")
            continue
        path = model_dir / relative
        if not path.is_file():
            if required or require_optional:
                failures.append(f"missing: {path}")
            else:
                skipped += 1
            continue
        checked += 1
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            failures.append(
                f"size mismatch: {path} ({actual_bytes} != {expected_bytes})"
            )
            continue
        if verify_hashes:
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                failures.append(f"invalid expected SHA256 for {relative}")
                continue
            actual_sha = _sha256(path)
            if actual_sha != expected_sha:
                failures.append(
                    f"SHA256 mismatch: {path} ({actual_sha} != {expected_sha})"
                )

    if failures:
        print(f"FAIL {package.get('id', descriptor)}")
        for failure in failures:
            print(f"  {failure}")
        return 1
    mode = "size+sha256" if verify_hashes else "size"
    print(
        f"PASS {package['id']}: {checked} artifacts verified ({mode}), "
        f"{skipped} optional absent"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--hash", action="store_true", dest="verify_hashes")
    parser.add_argument("--all", action="store_true", dest="require_optional")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_dir = args.model_dir
    if model_dir is None:
        value = os.environ.get("R9V_MODEL_DIR")
        if not value:
            raise SystemExit("set R9V_MODEL_DIR or pass --model-dir")
        model_dir = Path(value)
    return verify(
        args.descriptor.resolve(),
        model_dir.expanduser().resolve(),
        verify_hashes=args.verify_hashes,
        require_optional=args.require_optional,
    )


if __name__ == "__main__":
    raise SystemExit(main())
