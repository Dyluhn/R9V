#!/usr/bin/env python3
"""Fetch a published immutable R9V model package from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema") != "r9v.model-package.v1":
        raise SystemExit(f"{path}: expected r9v.model-package.v1")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args()

    package = _load(args.descriptor.resolve())
    distribution = package.get("distribution", {})
    repository = distribution.get("repository")
    revision = distribution.get("revision")
    status = distribution.get("status", "unknown")
    if not repository or not revision:
        note = distribution.get("note", "No published immutable revision exists.")
        raise SystemExit(
            f"{package['id']} is {status!r}, not downloadable yet.\n{note}"
        )
    if package.get("license") != "Apache-2.0" and not args.accept_model_license:
        raise SystemExit(
            "this package uses a separate model license; inspect it and rerun with "
            "--accept-model-license"
        )
    model_dir = args.model_dir
    if model_dir is None:
        value = os.environ.get("R9V_MODEL_DIR")
        if not value:
            raise SystemExit("set R9V_MODEL_DIR or pass --model-dir")
        model_dir = Path(value)
    model_dir = model_dir.expanduser().resolve()

    artifacts = [
        item
        for item in package.get("artifacts", [])
        if item.get("required", True) or args.include_optional
    ]
    total = sum(int(item["bytes"]) for item in artifacts)
    print(
        f"Fetching {len(artifacts)} files ({total / 2**30:.2f} GiB) to "
        f"{model_dir} from {repository}@{revision}"
    )
    hf = shutil.which("hf")
    if not hf:
        raise SystemExit("the Hugging Face `hf` CLI is required")
    model_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        subprocess.run(
            [
                hf,
                "download",
                repository,
                artifact["path"],
                "--revision",
                revision,
                "--local-dir",
                str(model_dir),
            ],
            check=True,
        )
    verifier = Path(__file__).with_name("verify_package.py")
    command = [
        str(verifier),
        str(args.descriptor),
        "--model-dir",
        str(model_dir),
        "--hash",
    ]
    if args.include_optional:
        command.append("--all")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
