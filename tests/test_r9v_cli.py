from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
        "qwen38-flash-next/ud-iq4-xs/dual-r9700-128k"
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
