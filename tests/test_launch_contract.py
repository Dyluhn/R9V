# SPDX-License-Identifier: Apache-2.0
"""Run scripts/launch.sh against a fake Docker and check the container it would create."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILES = [
    "target/a",
    "target/b",
    "target/c",
    "metadata/config.json",
    "mtp/config.json",
    "mtp/model.safetensors",
    "vision/mmproj",
    "manifests/hot",
]
FAKE_DOCKER = """#!/usr/bin/env python3
import json, os, sys
if sys.argv[1] == "container":
    sys.exit(1)
if sys.argv[1] == "info":
    print("name=rootless")
    sys.exit(0)
open(os.environ["TEST_ARGS"], "w").write(json.dumps(sys.argv[1:]))
"""


def run_launcher(tmp_path: Path, env: dict[str, str], files=LAUNCH_FILES):
    """Return (completed process, docker run arguments or None)."""
    model = tmp_path / "models"
    for relative in [*files, "ple"]:
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    docker = tmp_path / "bin/docker"
    docker.parent.mkdir(exist_ok=True)
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    captured = tmp_path / "args.json"
    base = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("R9V_") and not key.startswith("TEST_")
    }
    full_env = {
        **base,
        "PATH": f"{docker.parent}{os.pathsep}{os.environ['PATH']}",
        "TEST_ARGS": str(captured),
        "R9V_MODEL_DIR": str(model),
        "R9V_PLE_PATH": str(model / "ple"),
        "R9V_CACHE_DIR": str(tmp_path / "cache"),
        "R9V_CACHE_NAMESPACE": "test",
        "R9V_PREFLIGHT": "0",
        "R9V_TARGET_REL": "target/a",
        "R9V_TARGET_SHARD2_REL": "target/b",
        "R9V_TARGET_SHARD3_REL": "target/c",
        "R9V_MMPROJ_REL": "vision/mmproj",
        "R9V_MANIFEST_REL": "manifests/hot",
        **env,
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/launch.sh")],
        env=full_env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    args = json.loads(captured.read_text()) if captured.exists() else None
    return result, args


def option_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def test_launcher_passes_prefix_cache_retention_interval(tmp_path):
    result, args = run_launcher(
        tmp_path, {"R9V_PREFIX_CACHE_RETENTION_INTERVAL": "1616"}
    )

    assert result.returncode == 0, result.stderr
    assert option_value(args, "--prefix-cache-retention-interval") == "1616"


def test_launcher_omits_retention_interval_when_prefix_caching_is_off(tmp_path):
    result, args = run_launcher(
        tmp_path,
        {
            "R9V_PREFIX_CACHE_RETENTION_INTERVAL": "1616",
            "R9V_ENABLE_PREFIX_CACHING": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "--prefix-cache-retention-interval" not in args


def test_launcher_refuses_non_integer_retention_interval(tmp_path):
    result, args = run_launcher(
        tmp_path, {"R9V_PREFIX_CACHE_RETENTION_INTERVAL": "16k"}
    )

    assert result.returncode == 2
    assert "R9V_PREFIX_CACHE_RETENTION_INTERVAL must be a non-negative integer" in result.stderr
    assert args is None
