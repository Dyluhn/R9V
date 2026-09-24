#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a runtime's pinned image overlays and render their Docker mounts and environment.

A runtime descriptor may carry an "overlays" block: files bind-mounted read-only
over the pinned image. Every file is SHA-256 pinned. The CED environment and
one CED model file are added only with CED: the "ced" group with R9V_CED=on,
the "ced-quality" group (the multi-source model file) with R9V_CED=quality.
Nothing is mounted unless every file matches and every CED setting is valid.
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
# Overlay groups mounted for each R9V_CED value. "quality" runs the multi-source
# projector (split 16 plus the block inputs of layers 3/7/11/15) with its own
# model file.
CED_GROUPS = {"off": ["always"], "on": ["always", "ced"], "quality": ["always", "ced-quality"]}
# The fix every headroom failure names while the projector is loaded on each GPU.
CED_HEADROOM_FIX = "the CED projector needs VRAM on each GPU; start with --ced off to leave it unloaded"


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


def projector_setting(switch: str) -> str:
    """The setting that names the projector file (in the model directory) CED loads."""
    return "R9V_CED_QUALITY_PROJECTOR_REL" if switch == "quality" else "R9V_CED_PROJECTOR_REL"


def ced_environment(env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Return (container CED environment, problems). CED off returns no environment."""
    switch = env.get("R9V_CED", "off")
    if switch not in CED_GROUPS:
        return {}, [f"R9V_CED must be on, off or quality (got {switch!r})"]
    if switch == "off":
        return {}, []
    if switch == "quality":
        # The quality projector ships stored as int8 and loads as stored: in bf16 it
        # would not fit next to the expert cache. R9V_CED_PRECISION applies to "on".
        env = {**env, "R9V_CED_PRECISION": "int8"}
    problems = []
    for key, allowed in CED_CHOICES.items():
        if env.get(key) not in allowed:
            problems.append(f"{key} must be one of {allowed} (got {env.get(key)!r})")
    for key, minimum in CED_MINIMUMS.items():
        value = env.get(key, "")
        if not (value.isascii() and value.isdigit()) or int(value) < minimum:
            problems.append(f"{key} must be an integer >= {minimum} (got {value!r})")
    setting = projector_setting(switch)
    relative = env.get(setting, "")
    parts = PurePosixPath(relative).parts
    if not relative or relative.startswith("/") or ".." in parts or ":" in relative or "\n" in relative:
        problems.append(f"{setting} must be a path inside the model directory (got {relative!r})")
    elif not (Path(env.get("R9V_MODEL_DIR", "")) / relative).is_file():
        problems.append(f"CED projector missing: {Path(env.get('R9V_MODEL_DIR', '')) / relative}; "
                        f"rerun setup with --ced {switch} to fetch it, or turn CED off with --ced off")
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
    groups = CED_GROUPS.get(env.get("R9V_CED", "off"), ["always"])
    if ced and groups[-1] not in overlays["mounts"]:
        raise OverlayError(f"R9V_CED={env['R9V_CED']} but runtime {runtime_path} has no {groups[-1]!r} overlay group")
    problems = verify(runtime_path, overlays) + problems
    if problems:
        raise OverlayError(f"{len(problems)} runtime overlay problem(s):\n  " + "\n  ".join(problems))
    directory = (runtime_path.parent / overlays["directory"]).resolve()
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
