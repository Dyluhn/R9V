# SPDX-License-Identifier: Apache-2.0
"""Mechanical guards against baking the development host into the engine."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    ROOT / ".cargo",
    ROOT / ".github" / "workflows",
    ROOT / "ci",
    ROOT / "crates",
    ROOT / "xtask",
)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hip", ".py", ".rs", ".sh", ".toml", ".yml", ".yaml"}


def engine_source_files() -> list[Path]:
    files: list[Path] = []
    for source_root in SOURCE_ROOTS:
        files.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix in SOURCE_SUFFIXES
        )
    return sorted(files)


def test_engine_sources_contain_no_developer_home_paths() -> None:
    offenders: list[str] = []
    for path in engine_source_files():
        text = path.read_text(encoding="utf-8")
        if "/home/" in text or "/var/home/" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_engine_sources_never_select_the_build_hosts_cpu() -> None:
    # Build the forbidden spellings in pieces so this policy test does not
    # trigger itself when it scans the tests-independent engine roots.
    forbidden = ("target-cpu=" + "native", "-march=" + "native")
    offenders: list[str] = []
    for path in engine_source_files():
        if path == ROOT / "crates" / "r9v-t0" / "build.rs":
            # This is the enforcement point and necessarily names the value it
            # rejects; it is separately inspected below.
            continue
        text = path.read_text(encoding="utf-8")
        if any(value in text for value in forbidden):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_official_x86_64_build_is_pinned_to_the_architectural_baseline() -> None:
    cargo_config = (ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8")
    assert "[target.x86_64-unknown-linux-gnu]" in cargo_config
    assert 'rustflags = ["-C", "target-cpu=x86-64"]' in cargo_config

    t0_gate = (ROOT / "crates" / "r9v-t0" / "build.rs").read_text(
        encoding="utf-8"
    )
    assert 'BTreeSet::from(["fxsr", "sse", "sse2"])' in t0_gate
    assert "CARGO_CFG_TARGET_FEATURE" in t0_gate
