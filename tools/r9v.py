#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Discover, inspect, validate, and run exact R9V model profiles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA = "r9v.profile.v1"
VALID_STATUSES = {"qualified", "release-candidate", "experimental", "retired"}
DESCRIPTOR_SCHEMAS = {
    "model_package": "r9v.model-package.v1",
    "runtime": "r9v.runtime.v1",
    "hardware": "r9v.hardware.v1",
    "placement": "r9v.placement.v1",
}


class ProfileError(RuntimeError):
    """A profile is missing, ambiguous, or invalid."""


@dataclass(frozen=True)
class Profile:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def aliases(self) -> tuple[str, ...]:
        values = self.data.get("aliases", [])
        return tuple(str(value) for value in values)


def _require_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{source}: {key!r} must be a non-empty string")
    return value


def validate_profile(path: Path, data: dict[str, Any]) -> None:
    if data.get("schema") != PROFILE_SCHEMA:
        raise ProfileError(f"{path}: schema must be {PROFILE_SCHEMA!r}")
    for key in ("id", "name", "status", "model_package", "runtime", "hardware"):
        _require_string(data, key, path)
    if data["status"] not in VALID_STATUSES:
        raise ProfileError(
            f"{path}: unsupported status {data['status']!r}; "
            f"expected one of {sorted(VALID_STATUSES)}"
        )
    aliases = data.get("aliases", [])
    if not isinstance(aliases, list) or not all(
        isinstance(value, str) and value for value in aliases
    ):
        raise ProfileError(f"{path}: aliases must be a list of non-empty strings")
    commands = data.get("commands", {})
    if not isinstance(commands, dict):
        raise ProfileError(f"{path}: commands must be an object")
    for name, command in commands.items():
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) and value for value in command)
        ):
            raise ProfileError(
                f"{path}: command {name!r} must be a non-empty string array"
            )


def discover_profiles(root: Path = REPO_ROOT) -> list[Profile]:
    profiles: list[Profile] = []
    identities: dict[str, Path] = {}
    for path in sorted((root / "profiles").glob("**/profile.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileError(f"cannot read {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ProfileError(f"{path}: top level must be an object")
        validate_profile(path, raw)
        profile = Profile(path=path, data=raw)
        for identity in (profile.id, *profile.aliases):
            previous = identities.get(identity)
            if previous is not None:
                raise ProfileError(
                    f"profile identity {identity!r} is duplicated in {previous} and {path}"
                )
            identities[identity] = path
        profiles.append(profile)
    if not profiles:
        raise ProfileError(f"no profile.json files found under {root / 'profiles'}")
    return profiles


def resolve_profile(name: str, profiles: list[Profile]) -> Profile:
    exact = [profile for profile in profiles if name in (profile.id, *profile.aliases)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ProfileError(f"profile name {name!r} is ambiguous")
    prefix = [profile for profile in profiles if profile.id.startswith(name)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        choices = ", ".join(profile.id for profile in prefix)
        raise ProfileError(f"profile prefix {name!r} is ambiguous: {choices}")
    raise ProfileError(f"unknown profile {name!r}; run './r9v list'")


def _path_from_repo(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def referenced_descriptor(profile: Profile, key: str) -> tuple[Path, dict[str, Any]]:
    relative = profile.data.get("descriptors", {}).get(key)
    if not isinstance(relative, str) or not relative:
        raise ProfileError(f"{profile.path}: descriptors.{key} is required")
    path = _path_from_repo(relative)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ProfileError(f"{path}: top level must be an object")
    expected_schema = DESCRIPTOR_SCHEMAS[key]
    if data.get("schema") != expected_schema:
        raise ProfileError(f"{path}: schema must be {expected_schema!r}")
    expected_id = profile.data[key]
    if data.get("id") != expected_id:
        raise ProfileError(
            f"{path}: id {data.get('id')!r} does not match profile {key} "
            f"{expected_id!r}"
        )
    return path, data


def _validate_model_package(path: Path, package: dict[str, Any]) -> None:
    artifacts = package.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ProfileError(f"{path}: artifacts must be a non-empty list")
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ProfileError(f"{path}: every artifact must be an object")
        relative = artifact.get("path")
        size = artifact.get("bytes")
        digest = artifact.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ProfileError(f"{path}: artifact path must be a non-empty string")
        artifact_path = Path(relative)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ProfileError(f"{path}: artifact path escapes package: {relative}")
        if relative in seen:
            raise ProfileError(f"{path}: duplicate artifact path: {relative}")
        seen.add(relative)
        if not isinstance(size, int) or size < 0:
            raise ProfileError(f"{path}: invalid byte count for {relative}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ProfileError(f"{path}: invalid SHA256 for {relative}")


def _validate_commands(profile: Profile) -> None:
    for name, command in profile.data.get("commands", {}).items():
        executable = _path_from_repo(command[0])
        if not executable.is_file():
            raise ProfileError(
                f"{profile.path}: command {name!r} is missing: {executable}"
            )
        if not os.access(executable, os.X_OK):
            raise ProfileError(
                f"{profile.path}: command {name!r} is not executable: {executable}"
            )


def verify_profile_graph(profile: Profile) -> list[str]:
    checked = [str(profile.path.relative_to(REPO_ROOT))]
    descriptors: dict[str, dict[str, Any]] = {}
    for key in ("model_package", "runtime", "hardware"):
        path, data = referenced_descriptor(profile, key)
        descriptors[key] = data
        checked.append(str(path.relative_to(REPO_ROOT)))
    placement = profile.data.get("placement")
    if placement:
        path, data = referenced_descriptor(profile, "placement")
        descriptors["placement"] = data
        checked.append(str(path.relative_to(REPO_ROOT)))
    _validate_model_package(
        _path_from_repo(profile.data["descriptors"]["model_package"]),
        descriptors["model_package"],
    )
    architecture = descriptors["hardware"].get("architecture")
    targets = descriptors["runtime"].get("targets", [])
    if architecture not in targets:
        raise ProfileError(
            f"{profile.path}: runtime does not support hardware architecture "
            f"{architecture!r}"
        )
    if placement:
        selected = descriptors["placement"]
        if selected.get("model_package") != profile.data["model_package"]:
            raise ProfileError(f"{profile.path}: placement model package mismatch")
        if selected.get("hardware") != profile.data["hardware"]:
            raise ProfileError(f"{profile.path}: placement hardware mismatch")
    _validate_commands(profile)
    return checked


def command_environment(profile: Profile, model_dir: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["R9V_PROFILE_ID"] = profile.id
    env["R9V_PROFILE_ROOT"] = str(profile.root)
    profile_env = profile.data.get("legacy_env")
    if profile_env:
        env["R9V_PROFILE"] = str(_path_from_repo(str(profile_env)))
    if model_dir:
        env["R9V_MODEL_DIR"] = str(Path(model_dir).expanduser().resolve())
    return env


def run_profile_command(
    profile: Profile,
    action: str,
    remainder: list[str],
    *,
    model_dir: str | None,
    dry_run: bool,
) -> int:
    command = profile.data.get("commands", {}).get(action)
    if not command:
        raise ProfileError(f"profile {profile.id!r} does not provide {action!r}")
    resolved = [str(_path_from_repo(command[0])), *command[1:], *remainder]
    env = command_environment(profile, model_dir)
    if dry_run:
        print(
            json.dumps(
                {
                    "profile": profile.id,
                    "action": action,
                    "command": resolved,
                    "environment": {
                        key: env[key]
                        for key in (
                            "R9V_PROFILE_ID",
                            "R9V_PROFILE_ROOT",
                            "R9V_PROFILE",
                            "R9V_MODEL_DIR",
                        )
                        if key in env
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return subprocess.run(resolved, cwd=REPO_ROOT, env=env, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r9v",
        description="Inspect and run exact R9V model/quant profiles with explicit release status.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list available profiles")
    list_parser.add_argument("--json", action="store_true")
    list_parser.add_argument(
        "--by-topology",
        action="store_true",
        help="group profiles by single-, dual-, or multi-GPU topology",
    )

    show_parser = subparsers.add_parser("show", help="show a resolved profile")
    show_parser.add_argument("profile")
    show_parser.add_argument("--json", action="store_true")

    verify_parser = subparsers.add_parser(
        "validate", help="validate profile descriptors without running a model"
    )
    verify_parser.add_argument("profile", nargs="?")

    for action, help_text in (
        ("doctor", "check profile prerequisites"),
        ("fetch", "download and verify profile artifacts"),
        ("build", "build the selected runtime"),
        ("run", "launch the selected profile"),
        ("verify", "verify installed artifacts or run the profile smoke test"),
    ):
        action_parser = subparsers.add_parser(action, help=help_text)
        action_parser.add_argument("profile")
        action_parser.add_argument("--model-dir")
        action_parser.add_argument("--dry-run", action="store_true")
    return parser


def _profile_summary(profile: Profile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.data["name"],
        "status": profile.data["status"],
        "model_package": profile.data["model_package"],
        "runtime": profile.data["runtime"],
        "hardware": profile.data["hardware"],
        "aliases": list(profile.aliases),
    }


def _topology_group(profile: Profile) -> tuple[int, str]:
    path, hardware = referenced_descriptor(profile, "hardware")
    gpu_count = hardware.get("gpu_count")
    if not isinstance(gpu_count, int) or gpu_count < 1:
        raise ProfileError(f"{path}: gpu_count must be a positive integer")
    names = {1: "single-gpu", 2: "dual-gpu"}
    return gpu_count, names.get(gpu_count, f"{gpu_count}-gpu")


def _profiles_by_topology(profiles: list[Profile]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for profile in profiles:
        key = _topology_group(profile)
        grouped.setdefault(key, []).append(_profile_summary(profile))
    return [
        {
            "topology": topology,
            "gpu_count": gpu_count,
            "profiles": grouped[(gpu_count, topology)],
        }
        for gpu_count, topology in sorted(grouped)
    ]


def _print_topology_groups(groups: list[dict[str, Any]]) -> None:
    for index, group in enumerate(groups):
        if index:
            print()
        heading = str(group["topology"]).replace("-", " ").upper()
        print(heading)
        print(f"{'PROFILE':58} {'STATUS':17} HARDWARE")
        for item in group["profiles"]:
            print(f"{item['id']:58} {item['status']:17} {item['hardware']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remainder = parser.parse_known_args(argv)
    try:
        profiles = discover_profiles()
        if args.command == "list":
            if remainder:
                parser.error(f"unrecognized arguments: {' '.join(remainder)}")
            if args.by_topology:
                groups = _profiles_by_topology(profiles)
                if args.json:
                    print(json.dumps(groups, indent=2, sort_keys=True))
                else:
                    _print_topology_groups(groups)
            else:
                summaries = [_profile_summary(profile) for profile in profiles]
                if args.json:
                    print(json.dumps(summaries, indent=2, sort_keys=True))
                else:
                    print(f"{'PROFILE':58} {'STATUS':17} HARDWARE")
                    for item in summaries:
                        print(
                            f"{item['id']:58} {item['status']:17} "
                            f"{item['hardware']}"
                        )
            return 0

        if args.command == "show":
            if remainder:
                parser.error(f"unrecognized arguments: {' '.join(remainder)}")
            profile = resolve_profile(args.profile, profiles)
            checked = verify_profile_graph(profile)
            payload = dict(profile.data)
            payload["descriptor_files"] = checked
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                summary = _profile_summary(profile)
                for key in (
                    "id",
                    "name",
                    "status",
                    "model_package",
                    "runtime",
                    "hardware",
                ):
                    print(f"{key:16} {summary[key]}")
                if profile.aliases:
                    print(f"{'aliases':16} {', '.join(profile.aliases)}")
                print("descriptor files")
                for value in checked:
                    print(f"  {value}")
            return 0

        if args.command == "validate":
            if remainder:
                parser.error(f"unrecognized arguments: {' '.join(remainder)}")
            selected = (
                [resolve_profile(args.profile, profiles)] if args.profile else profiles
            )
            for profile in selected:
                checked = verify_profile_graph(profile)
                print(f"PASS {profile.id} ({len(checked)} descriptors)")
            return 0

        profile = resolve_profile(args.profile, profiles)
        verify_profile_graph(profile)
        if remainder and remainder[0] == "--":
            remainder = remainder[1:]
        return run_profile_command(
            profile,
            args.command,
            remainder,
            model_dir=args.model_dir,
            dry_run=args.dry_run,
        )
    except ProfileError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
