# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools import runtime_overlays

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtimes/qwen38-flash-next-gfx1201-mtp4-v3/runtime.json"
PROJECTOR = "ced/ced-projector-split16.safetensors"
CED_ON = {
    "R9V_CED": "on",
    "R9V_CED_PROJECTOR_REL": PROJECTOR,
    "R9V_CED_PRECISION": "bf16",
    "R9V_CED_MIN_PROMPT": "8192",
    "R9V_CED_TAIL": "2048",
    "R9V_CED_DEFAULT": "on",
}


def copied_runtime(tmp_path: Path) -> Path:
    target = tmp_path / "runtime"
    shutil.copytree(RUNTIME.parent / "overlays", target / "overlays")
    shutil.copy(RUNTIME, target / "runtime.json")
    return target / "runtime.json"


def model_dir_with_projector(tmp_path: Path) -> Path:
    model = tmp_path / "models"
    (model / "ced").mkdir(parents=True)
    (model / PROJECTOR).write_bytes(b"projector")
    return model


def pairs(args: list[str], flag: str) -> list[str]:
    return [args[index + 1] for index, value in enumerate(args) if value == flag]


def test_committed_mtp4_v3_overlays_match_their_pins():
    overlays = runtime_overlays.load(RUNTIME)

    assert runtime_overlays.verify(RUNTIME, overlays) == []
    assert len(overlays["sha256"]) == 15
    assert set(overlays["mounts"]) == {"always", "ced"}


def test_tampered_overlays_are_refused_with_every_problem(tmp_path):
    runtime = copied_runtime(tmp_path)
    overlays_dir = runtime.parent / "overlays"
    (overlays_dir / "linear.py").write_text("# changed\n")
    (overlays_dir / "q8_wmma.so").unlink()
    (overlays_dir / "extra.py").write_text("")

    problems = runtime_overlays.verify(runtime, runtime_overlays.load(runtime))

    assert len(problems) == 3
    assert any("SHA-256 mismatch" in p and "linear.py" in p for p in problems)
    assert any("missing overlay file" in p and "q8_wmma.so" in p for p in problems)
    assert any("unpinned file" in p and "extra.py" in p for p in problems)
    with pytest.raises(runtime_overlays.OverlayError, match="3 runtime overlay problem"):
        runtime_overlays.docker_args(runtime, {"R9V_CED": "off"})


def test_symlinked_overlay_is_refused(tmp_path):
    runtime = copied_runtime(tmp_path)
    overlays_dir = runtime.parent / "overlays"
    real = tmp_path / "linear.py"
    (overlays_dir / "linear.py").rename(real)
    (overlays_dir / "linear.py").symlink_to(real)

    problems = runtime_overlays.verify(runtime, runtime_overlays.load(runtime))

    assert problems == [f"overlay is not a regular file: {overlays_dir / 'linear.py'}"]


def test_ced_off_mounts_the_always_group_without_ced_files_or_environment():
    args = runtime_overlays.docker_args(RUNTIME, {"R9V_CED": "off"})
    volumes = pairs(args, "--volume")
    environment = pairs(args, "--env")

    assert len(volumes) == 10
    assert any(volume.endswith(":/r9v-full-mutable:ro") for volume in volumes)
    assert not any("model.py" in volume for volume in volumes)
    assert sorted(environment) == [
        "R9V_FULL_MUTABLE_CACHE=1",
        "R9V_FULL_MUTABLE_PINS=/r9v-full-mutable/full_mutable_pins.json",
        "R9V_FULL_MUTABLE_SO=/r9v-full-mutable/candidate.so",
        "R9V_Q8_PREFILL_COMBINED=1",
        "R9V_Q8_PREFILL_TOKEN64=1",
    ]


def test_unset_ced_switch_means_off():
    assert runtime_overlays.docker_args(RUNTIME, {}) == runtime_overlays.docker_args(
        RUNTIME, {"R9V_CED": "off"}
    )


def test_ced_on_adds_the_model_overlay_and_projector_environment(tmp_path):
    env = {**CED_ON, "R9V_MODEL_DIR": str(model_dir_with_projector(tmp_path))}

    args = runtime_overlays.docker_args(RUNTIME, env)
    volumes = pairs(args, "--volume")
    environment = pairs(args, "--env")

    assert len(volumes) == 11
    assert any(
        volume.endswith(
            "overlays/model.py:/opt/r9v/lib/python3.12/site-packages/"
            "vllm/models/qwen4_exp/amd/model.py:ro"
        )
        for volume in volumes
    )
    assert f"R9V_CED_PROJECTOR=/models/{PROJECTOR}" in environment
    assert "R9V_CED_DEFAULT=on" in environment
    assert "R9V_CED_TAIL=2048" in environment
    assert "R9V_CED_MIN_PROMPT=8192" in environment
    assert "R9V_CED_PRECISION=bf16" in environment


def test_invalid_ced_settings_report_every_problem(tmp_path):
    env = {
        **CED_ON,
        "R9V_MODEL_DIR": str(model_dir_with_projector(tmp_path)),
        "R9V_CED_PRECISION": "fp8",
        "R9V_CED_TAIL": "511",
        "R9V_CED_MIN_PROMPT": "-1",
        "R9V_CED_DEFAULT": "yes",
    }

    with pytest.raises(runtime_overlays.OverlayError) as error:
        runtime_overlays.docker_args(RUNTIME, env)

    message = str(error.value)
    assert "4 runtime overlay problem" in message
    assert "R9V_CED_PRECISION must be one of ('bf16', 'int8') (got 'fp8')" in message
    assert "R9V_CED_TAIL must be an integer >= 512 (got '511')" in message
    assert "R9V_CED_MIN_PROMPT must be an integer >= 0 (got '-1')" in message
    assert "R9V_CED_DEFAULT must be one of ('on', 'off') (got 'yes')" in message


def test_ced_switch_accepts_only_on_or_off():
    with pytest.raises(runtime_overlays.OverlayError, match="R9V_CED must be on or off"):
        runtime_overlays.docker_args(RUNTIME, {"R9V_CED": "1"})


def test_ced_on_without_the_projector_file_is_refused(tmp_path):
    env = {**CED_ON, "R9V_MODEL_DIR": str(tmp_path)}

    with pytest.raises(runtime_overlays.OverlayError, match="CED projector missing"):
        runtime_overlays.docker_args(RUNTIME, env)


@pytest.mark.parametrize("relative", ["", "/abs/projector", "../projector", "ced/a:b"])
def test_ced_projector_path_must_stay_inside_the_model_directory(tmp_path, relative):
    env = {**CED_ON, "R9V_MODEL_DIR": str(tmp_path), "R9V_CED_PROJECTOR_REL": relative}

    with pytest.raises(runtime_overlays.OverlayError, match="R9V_CED_PROJECTOR_REL"):
        runtime_overlays.docker_args(RUNTIME, env)


def test_runtime_without_overlays_renders_nothing_and_refuses_ced(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"schema": "r9v.runtime.v1"}))

    assert runtime_overlays.docker_args(runtime, {}) == []
    with pytest.raises(runtime_overlays.OverlayError, match="has no CED overlay"):
        runtime_overlays.docker_args(runtime, {"R9V_CED": "on"})


def test_cli_prints_one_argument_per_line(capsys, monkeypatch):
    monkeypatch.delenv("R9V_CED", raising=False)

    assert runtime_overlays.main(["docker-args", str(RUNTIME)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "--volume"
    assert lines.count("--volume") == 10


def test_cli_reports_problems_and_exits_2(tmp_path, capsys):
    runtime = copied_runtime(tmp_path)
    (runtime.parent / "overlays/qsa.py").write_text("")

    assert runtime_overlays.main(["verify", str(runtime)]) == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_mtp4_v3_kernel_sources_match_their_checksum_list():
    sources = RUNTIME.parent / "sources"
    listed = {}
    for line in (sources / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    present = {
        str(path.relative_to(sources)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }

    assert present == listed
