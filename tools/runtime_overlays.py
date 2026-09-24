#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a runtime's pinned image overlays and render their Docker mounts and environment.

A runtime descriptor may carry an "overlays" block: files bind-mounted read-only
over the pinned image. Every file is SHA-256 pinned; the "ced" group (the CED
model file) and the CED environment are added only when R9V_CED=on. Nothing is
mounted unless every file matches and every CED setting is valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath

# CED (approximate long-prompt prefill) settings passed to the container as-is.
# Allowed values, or the minimum for integers. Ported from the 1.3.0 package's
# ced_config(); the projector comes from the model package instead of a host path.
CED_CHOICES = {"R9V_CED_PRECISION": ("bf16", "int8"), "R9V_CED_DEFAULT": ("on", "off")}
CED_MINIMUMS = {"R9V_CED_MIN_PROMPT": 0, "R9V_CED_TAIL": 512}


class OverlayError(ValueError):
    """The runtime overlays or CED settings cannot be launched."""


def load(runtime_path: Path) -> dict | None:
    """Return the runtime's overlays block, or None when it has none."""
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    overlays = runtime.get("overlays")
    if overlays is not None and not isinstance(overlays, dict):
        raise OverlayError(f"{runtime_path}: overlays must be an object")
    return overlays


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(runtime_path: Path, overlays: dict) -> list[str]:
    """Return every problem with the overlay files; an empty list means all match."""
    root = runtime_path.parent.resolve()
    directory = (root / overlays["directory"]).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        return [f"overlay directory {directory} is missing or outside {root}"]
    pinned = overlays["sha256"]
    present = {path.name: path for path in directory.iterdir()}
    problems = [f"unpinned file in {directory}: {name}" for name in sorted(set(present) - set(pinned))]
    problems += [f"missing overlay file: {directory / name}" for name in sorted(set(pinned) - set(present))]
    for name in sorted(set(pinned) & set(present)):
        path = present[name]
        if path.is_symlink() or not path.is_file():
            problems.append(f"overlay is not a regular file: {path}")
        elif _sha256(path) != pinned[name]:
            problems.append(f"SHA-256 mismatch: {path} (expected {pinned[name]})")
    for group, mounts in overlays["mounts"].items():
        problems += [f"mount group {group!r} names unpinned file {name}"
                     for name in mounts if name not in pinned]
    if ":" in str(directory) or "\n" in str(directory):
        problems.append(f"overlay path {directory} contains ':' or a newline, which Docker mounts cannot express")
    return problems


def ced_environment(env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Return (container CED environment, problems). CED off returns no environment."""
    switch = env.get("R9V_CED", "off")
    if switch not in ("on", "off"):
        return {}, [f"R9V_CED must be on or off (got {switch!r})"]
    if switch == "off":
        return {}, []
    problems = []
    for key, allowed in CED_CHOICES.items():
        if env.get(key) not in allowed:
            problems.append(f"{key} must be one of {allowed} (got {env.get(key)!r})")
    for key, minimum in CED_MINIMUMS.items():
        value = env.get(key, "")
        if not (value.isascii() and value.isdigit()) or int(value) < minimum:
            problems.append(f"{key} must be an integer >= {minimum} (got {value!r})")
    relative = env.get("R9V_CED_PROJECTOR_REL", "")
    parts = PurePosixPath(relative).parts
    if not relative or relative.startswith("/") or ".." in parts or ":" in relative or "\n" in relative:
        problems.append(f"R9V_CED_PROJECTOR_REL must be a path inside the model directory (got {relative!r})")
    elif not (Path(env.get("R9V_MODEL_DIR", "")) / relative).is_file():
        problems.append(f"CED projector missing: {Path(env.get('R9V_MODEL_DIR', '')) / relative}; "
                        "rerun setup to fetch it, or turn CED off with --ced off")
    result = {key: env.get(key, "") for key in (*CED_CHOICES, *CED_MINIMUMS)}
    result["R9V_CED_PROJECTOR"] = f"/models/{relative}"
    return result, problems


def docker_args(runtime_path: Path, env: dict[str, str]) -> list[str]:
    """Return the docker run mount and environment arguments for this runtime."""
    overlays = load(runtime_path)
    if overlays is None:
        if env.get("R9V_CED", "off") != "off":
            raise OverlayError(f"R9V_CED={env['R9V_CED']} but runtime {runtime_path} has no CED overlay")
        return []
    ced, problems = ced_environment(env)
    problems = verify(runtime_path, overlays) + problems
    if problems:
        raise OverlayError(f"{len(problems)} runtime overlay problem(s):\n  " + "\n  ".join(problems))
    directory = (runtime_path.parent / overlays["directory"]).resolve()
    groups = ["always", "ced"] if ced else ["always"]
    args = ["--volume", f"{directory}:{overlays['directory_target']}:ro"]
    for group in groups:
        for name, target in overlays["mounts"].get(group, {}).items():
            args += ["--volume", f"{directory / name}:{target}:ro"]
    for key, value in {**overlays["env"].get("always", {}), **ced}.items():
        args += ["--env", f"{key}={value}"]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["docker-args", "verify"])
    parser.add_argument("runtime", type=Path, help="runtime.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            overlays = load(args.runtime)
            problems = verify(args.runtime, overlays) if overlays else []
            if problems:
                raise OverlayError("\n  ".join(["runtime overlay problems:", *problems]))
            print(f"PASS {args.runtime}: {len(overlays['sha256']) if overlays else 0} overlay files match")
        else:
            # One argument per line; verify() refuses paths containing newlines.
            print("\n".join(docker_args(args.runtime, dict(os.environ))))
    except (OSError, KeyError, TypeError, json.JSONDecodeError, OverlayError) as error:
        print(f"{args.runtime}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
