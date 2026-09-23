# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "r9v"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CLI), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_catalog_validates_all_profiles() -> None:
    result = run_cli("validate")
    assert result.returncode == 0, result.stderr
    assert "PASS muse-glimmer-30b/v1/single-r9700" in result.stdout
    assert "PASS qwen38-flash-next/ud-iq4-xs/dual-r9700-128k" in result.stdout


def test_help_does_not_overstate_profile_qualification() -> None:
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "explicit release status" in result.stdout
    assert "qualified R9V model/quant profiles" not in result.stdout


def test_setup_help_exposes_headroom_and_reuse_options() -> None:
    result = run_cli("setup", "qwen38", "--help")
    assert result.returncode == 0
    for option in ("--headroom", "--ced", "--reuse-from", "--calibration", "--expert-catalog", "--state-dir"):
        assert option in result.stdout


def test_support_help_exposes_private_bundle_and_archive_options() -> None:
    result = run_cli("support", "qwen38", "--help")
    assert result.returncode == 0
    for option in ("--state-dir", "--container", "--output", "--archive"):
        assert option in result.stdout


def test_catalog_can_be_grouped_by_topology() -> None:
    result = run_cli("list", "--by-topology", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [group["topology"] for group in payload] == ["single-gpu", "dual-gpu"]

    by_topology = {
        group["topology"]: {profile["id"] for profile in group["profiles"]}
        for group in payload
    }
    assert by_topology["single-gpu"] == {
        "muse-glimmer-30b/v1/single-r9700",
    }
    assert by_topology["dual-gpu"] == {
        "qwen38-flash-next/ud-iq4-xs/dual-r9700-128k",
        "qwen38-flash-next/ud-iq4-xs/dual-r9700-mtp4-128k",
        "qwen38-flash-next/uncensored-iq4-xs/dual-r9700-mtp4-128k",
        "qwen38-flash-next/ud-q4-k-xl/dual-r9700-128k",
    }


def test_topology_text_view_is_hardware_first() -> None:
    result = run_cli("list", "--by-topology")
    assert result.returncode == 0, result.stderr
    single_index = result.stdout.index("SINGLE GPU")
    dual_index = result.stdout.index("DUAL GPU")
    muse_index = result.stdout.index("muse-glimmer-30b/v1/single-r9700")
    qwen_index = result.stdout.index(
        "qwen38-flash-next/ud-iq4-xs/dual-r9700-128k"
    )
    assert single_index < muse_index < dual_index < qwen_index


def test_muse_alias_resolves_to_canonical_v1() -> None:
    result = run_cli("show", "muse", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["id"] == "muse-glimmer-30b/v1/single-r9700"
    assert payload["status"] == "experimental"


def test_action_options_after_profile_are_not_forwarded() -> None:
    result = run_cli(
        "verify",
        "qwen38",
        "--model-dir",
        "/tmp/r9v-test-model",
        "--dry-run",
        "--",
        "--hash",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["environment"]["R9V_MODEL_DIR"] == "/tmp/r9v-test-model"
    assert payload["command"][-1] == "--hash"


def test_unknown_profile_fails_closed() -> None:
    result = run_cli("show", "not-a-real-profile")
    assert result.returncode != 0
    assert "unknown profile" in result.stderr


def test_doctor_uses_setup_state_and_preserves_explicit_overrides(tmp_path, monkeypatch):
    from tools import r9v
    from types import SimpleNamespace
    (tmp_path / 'setup.json').write_text(json.dumps({'config': {
        'R9V_MODEL_DIR': '/saved-model', 'R9V_IMAGE': 'saved-image', 'R9V_HOST_PORT': '8123'}}))
    monkeypatch.delenv('R9V_CONFIG_FILE', raising=False)
    monkeypatch.delenv('R9V_MODEL_DIR', raising=False)
    monkeypatch.delenv('R9V_IMAGE', raising=False)
    monkeypatch.setenv('R9V_HOST_PORT', '8124')
    observed = []
    monkeypatch.setattr(r9v.subprocess, 'run', lambda command, **kwargs:
                        observed.append((command, kwargs['env'])) or SimpleNamespace(returncode=0))
    profile = r9v.resolve_profile('qwen38', r9v.discover_profiles())
    assert r9v.run_profile_command(profile, 'doctor', ['--state-dir', str(tmp_path), '--runtime'],
                                   model_dir=None, dry_run=False) == 0
    command, env = observed[0]
    assert '--state-dir' not in command
    assert env['R9V_MODEL_DIR'] == '/saved-model'
    assert env['R9V_IMAGE'] == 'saved-image'
    assert env['R9V_HOST_PORT'] == '8124'


def test_q4_fetch_and_verify_select_its_own_package():
    for command in ("fetch", "verify"):
        result = run_cli(command, "qwen38-q4-xl", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "ud-q4-k-xl--mtp-blockfp8--mmproj-q8/package.json" in result.stdout
        assert "ud-iq4-xs--mtp-blockfp8--mmproj-q8/package.json" not in result.stdout


UNCENSORED = ROOT / "profiles/qwen38-flash-next/dual-r9700-mtp4-uncensored/profile.json"


def test_uncensored_alias_resolves_to_the_ced_runtime_and_fixed_placement():
    result = run_cli("show", "qwen38-mtp4-uncensored", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["runtime"] == "qwen38-flash-next-gfx1201-mtp4-v3"
    assert payload["placement"] == "qwen38-uncensored-iq4-xs-dual-r9700-mtp4-full-mutable"
    assert payload["features"]["ced"]["default"] == "on"


def test_uncensored_fetch_and_verify_select_its_own_package_and_it_never_replans():
    for command in ("fetch", "verify"):
        result = run_cli(command, "qwen38-mtp4-uncensored", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uncensored-iq4-xs--mtp-blockfp8--mmproj-f16/package.json" in result.stdout
    for command in ("plan", "placement"):
        result = run_cli(command, "qwen38-mtp4-uncensored", "--dry-run")
        assert result.returncode != 0
        assert f"does not provide {command!r}" in result.stderr


def test_validate_refuses_a_runtime_with_a_modified_overlay(tmp_path):
    from tools import r9v
    source = ROOT / "runtimes/qwen38-flash-next-gfx1201-mtp4-v3"
    shutil.copytree(source / "overlays", tmp_path / "overlays")
    shutil.copy(source / "runtime.json", tmp_path / "runtime.json")
    (tmp_path / "overlays/model.py").write_text("# edited\n")
    runtime = json.loads((tmp_path / "runtime.json").read_text())

    with pytest.raises(r9v.ProfileError, match="SHA-256 mismatch.*model.py"):
        r9v._verify_runtime_overlays(tmp_path / "runtime.json", runtime)


def test_validate_refuses_a_placement_whose_pinned_manifest_differs():
    from tools import r9v
    placement = json.loads(
        (ROOT / "packages/placements/qwen38-flash-next/uncensored-iq4-xs/dual-r9700/"
         "mtp4-full-mutable.json").read_text()
    )
    placement["manifest"]["sha256"] = "0" * 64

    with pytest.raises(r9v.ProfileError, match="expected 0000"):
        r9v._verify_pinned_manifest(placement)
