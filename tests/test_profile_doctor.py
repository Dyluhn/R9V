# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

import tools.profile_doctor as doctor
from tools.profile_doctor import (
    KfdGpu,
    Reporter,
    _check_manifest_budget,
    _check_ple_hash,
    _check_profile_policy,
    discover_kfd_gpus,
    parse_amd_smi_list,
    parse_expected_pcie_links,
    parse_prometheus_metrics,
    pcie_payload_gbps,
    pcie_upstream_links,
)

# A stand-in for the 26.8 GiB derived PLE table. Tests never hash the real one.
PLE_SAMPLE = b"r9v-ple-test-payload"
PLE_SAMPLE_SHA256 = "e28b313ed08f5e62eb9c36da7775280ad2720c02e1aa1c2a9767bab93ff7b4b0"


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (b"partial output\xff\n", b"partial error\n"),
        (None, b"partial error\xff\n"),
        ("output", "error"),
        (None, None),
    ],
)
def test_timed_out_package_check_still_produces_json_report(
    monkeypatch, stdout, stderr
) -> None:
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 120, output=stdout, stderr=stderr)

    monkeypatch.setattr(doctor.subprocess, "run", timeout)
    monkeypatch.setenv("R9V_MODEL_DIR", "/synthetic-model")
    reporter = Reporter()

    doctor._check_model_package(reporter, Path("/synthetic-repo"), "qwen38")

    checks = json.loads(json.dumps([asdict(check) for check in reporter.checks]))
    assert len(checks) == 1
    assert checks[0]["status"] == "FAIL"
    assert isinstance(checks[0]["message"], str)
    if isinstance(stdout, bytes):
        assert checks[0]["message"] == "partial output\ufffd"


@pytest.mark.parametrize(
    ("exit_code", "oom_killed"), [(0, False), (137, False), (137, True)]
)
def test_stopped_runtime_report_preserves_exit_evidence(
    monkeypatch, exit_code, oom_killed
) -> None:
    state = {
        "Status": "exited",
        "ExitCode": exit_code,
        "OOMKilled": oom_killed,
        "Error": "",
        "StartedAt": "2026-09-02T03:12:50Z",
        "FinishedAt": "2026-09-02T03:47:38Z",
    }
    inspected = [
        {
            "State": state,
            "RestartCount": 2,
            "Image": "sha256:test",
            "Config": {"Env": ["PRIVATE_VALUE=not-for-report"]},
        }
    ]
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(inspected), ""
        ),
    )
    reporter = Reporter()

    doctor._check_runtime(reporter, 2)

    (check,) = reporter.checks
    assert check.name == "runtime-container"
    assert check.status == "FAIL"
    assert check.details is not None
    assert check.details["exit_code"] == exit_code
    assert check.details["oom_killed"] is oom_killed
    assert check.details["started_at"] == state["StartedAt"]
    assert check.details["finished_at"] == state["FinishedAt"]
    assert check.details["restart_count"] == 2
    assert check.details["image"] == "sha256:test"
    assert f"exit code={exit_code}" in check.message
    assert f"OOMKilled={oom_killed}" in check.message
    assert "docker inspect" in check.remediation
    assert "not-for-report" not in json.dumps(asdict(check))


def _sample_ple(tmp_path: Path) -> Path:
    path = tmp_path / "per_layer_token_embd.iq4_nl.bin"
    path.write_bytes(PLE_SAMPLE)
    return path


def test_amd_smi_inventory_preserves_device_order_and_bdf() -> None:
    inventory = parse_amd_smi_list(
        """
GPU: 0
    BDF: 0000:03:00.0
    UUID: first
    NODE_ID: 1

GPU: 1
    BDF: 0000:13:00.0
    UUID: second
    NODE_ID: 2
"""
    )

    assert [(gpu.index, gpu.bdf, gpu.node_id) for gpu in inventory] == [
        (0, "0000:03:00.0", 1),
        (1, "0000:13:00.0", 2),
    ]


def test_kfd_location_id_maps_to_pci_bdf(tmp_path: Path) -> None:
    properties = tmp_path / "class/kfd/kfd/topology/nodes/1/properties"
    properties.parent.mkdir(parents=True)
    properties.write_text(
        "gfx_target_version 120001\nlocation_id 4864\ndomain 0\ndrm_render_minor 129\n",
        encoding="utf-8",
    )

    gpu = discover_kfd_gpus(tmp_path)[0]
    assert gpu.bdf == "0000:13:00.0"
    assert gpu.node_id == 1


def test_selected_gpu_indices_follow_kfd_not_amd_smi_order(
    tmp_path: Path, monkeypatch
) -> None:
    for bdf in ("0000:03:00.0", "0000:ca:00.0", "0000:cd:00.0"):
        device = tmp_path / "bus/pci/devices" / bdf
        device.mkdir(parents=True)
        (device / "current_link_speed").write_text("16.0 GT/s", encoding="utf-8")
        (device / "current_link_width").write_text("16", encoding="utf-8")
    inventory = (
        "GPU: 0\n BDF: 0000:03:00.0\n NODE_ID: 6\n"
        "GPU: 1\n BDF: 0000:ca:00.0\n NODE_ID: 4\n"
        "GPU: 2\n BDF: 0000:cd:00.0\n NODE_ID: 5\n"
    )
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/amd-smi")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, inventory, ""),
    )
    monkeypatch.setattr(
        doctor,
        "discover_kfd_gpus",
        lambda _root: [
            KfdGpu("0000:ca:00.0", 120001, 128, 4),
            KfdGpu("0000:cd:00.0", 120001, 129, 5),
            KfdGpu("0000:03:00.0", 120001, 130, 6),
        ],
    )
    monkeypatch.setenv("R9V_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("R9V_EXPECTED_GPU_BDFS", "0000:ca:00.0,0000:cd:00.0")
    monkeypatch.setenv("R9V_MIN_PCIE_BANDWIDTH_GBPS", "0,0")
    reporter = Reporter()

    selected = doctor._selected_gpus(reporter, 2, tmp_path)

    assert [(rank, gpu.index, gpu.bdf) for rank, gpu, *_ in selected] == [
        (0, 1, "0000:ca:00.0"),
        (1, 2, "0000:cd:00.0"),
    ]
    assert not [check for check in reporter.checks if check.status == "FAIL"]
    architecture = [
        check for check in reporter.checks if check.name == "gpu-architecture"
    ]
    assert "amd-smi device 1" in architecture[0].message
    assert "amd-smi device 2" in architecture[1].message


def test_pcie_payload_estimate_matches_gen4_x4() -> None:
    assert abs(pcie_payload_gbps(16.0, 4) - 7.876923) < 1e-5


def test_pcie_path_rejects_partial_link_bearing_ancestor(tmp_path: Path) -> None:
    bridge = tmp_path / "devices/pci0000:00/0000:00:02.2"
    device = bridge / "0000:13:00.0"
    device.mkdir(parents=True)
    (bridge / "max_link_speed").write_text("16.0 GT/s", encoding="utf-8")
    by_bus = tmp_path / "bus/pci/devices"
    by_bus.mkdir(parents=True)
    (by_bus / "0000:13:00.0").symlink_to(device)

    try:
        pcie_upstream_links(tmp_path, "0000:13:00.0")
    except ValueError as error:
        assert "0000:00:02.2" in str(error)
        assert "complete positive speed/width" in str(error)
    else:
        raise AssertionError("partial upstream PCIe hop was silently omitted")


def test_expected_pcie_links_accept_generation_and_numeric_forms() -> None:
    links = parse_expected_pcie_links("Gen5x16, 16 GT/s x4", 2)

    assert [(link.speed_gts, link.width) for link in links] == [
        (32.0, 16),
        (16.0, 4),
    ]
    assert [link.config_value() for link in links] == ["Gen5x16", "Gen4x4"]


def test_selected_gpu_check_rejects_mismatched_exact_link(
    tmp_path: Path, monkeypatch
) -> None:
    for bdf, speed, width in (
        ("0000:03:00.0", "32.0 GT/s", "16"),
        ("0000:13:00.0", "16.0 GT/s", "4"),
    ):
        device = tmp_path / "bus/pci/devices" / bdf
        device.mkdir(parents=True)
        (device / "current_link_speed").write_text(speed, encoding="utf-8")
        (device / "current_link_width").write_text(width, encoding="utf-8")
    inventory = "GPU: 0\n BDF: 0000:03:00.0\nGPU: 1\n BDF: 0000:13:00.0\n"
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/amd-smi")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, inventory, ""),
    )
    monkeypatch.setattr(
        doctor,
        "discover_kfd_gpus",
        lambda _root: [
            KfdGpu("0000:03:00.0", 120001, 128),
            KfdGpu("0000:13:00.0", 120001, 129),
        ],
    )
    monkeypatch.setenv("R9V_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("R9V_EXPECTED_PCIE_LINKS", "Gen5x16,Gen5x16")
    monkeypatch.setenv("R9V_MIN_PCIE_BANDWIDTH_GBPS", "0,0")
    reporter = Reporter()

    doctor._selected_gpus(reporter, 2, tmp_path)

    exact = [
        check for check in reporter.checks if check.name == "pcie-link-expectation"
    ]
    assert [check.status for check in exact] == ["PASS", "FAIL"]
    assert exact[1].remediation is not None
    assert "cannot change a PCIe path capacity" in exact[1].remediation


def test_pcie_floor_scores_the_slowest_upstream_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    # Endpoint sysfs on both cards reads Gen5x16, but rank 1 sits behind a
    # bridge physically capped at Gen4x4. Its current speed is power-managed
    # down to Gen1, but the stable capacity floor must score Gen4x4 rather than
    # either the endpoint or the transient idle speed.
    for bridge, bdf, hop_speed, hop_width in (
        ("0000:00:01.0", "0000:03:00.0", "32.0 GT/s", "16"),
        ("0000:00:01.1", "0000:13:00.0", "16.0 GT/s", "4"),
    ):
        bridge_dir = tmp_path / "devices/pci0000:00" / bridge
        device_dir = bridge_dir / bdf
        device_dir.mkdir(parents=True)
        (bridge_dir / "current_link_speed").write_text("2.5 GT/s", encoding="utf-8")
        (bridge_dir / "current_link_width").write_text(hop_width, encoding="utf-8")
        (bridge_dir / "max_link_speed").write_text(hop_speed, encoding="utf-8")
        (bridge_dir / "max_link_width").write_text(hop_width, encoding="utf-8")
        (device_dir / "current_link_speed").write_text("32.0 GT/s", encoding="utf-8")
        (device_dir / "current_link_width").write_text("16", encoding="utf-8")
        (device_dir / "max_link_speed").write_text("32.0 GT/s", encoding="utf-8")
        (device_dir / "max_link_width").write_text("16", encoding="utf-8")
        by_bus = tmp_path / "bus/pci/devices"
        by_bus.mkdir(parents=True, exist_ok=True)
        (by_bus / bdf).symlink_to(device_dir)
    inventory = "GPU: 0\n BDF: 0000:03:00.0\nGPU: 1\n BDF: 0000:13:00.0\n"
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/amd-smi")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, inventory, ""),
    )
    monkeypatch.setattr(
        doctor,
        "discover_kfd_gpus",
        lambda _root: [
            KfdGpu("0000:03:00.0", 120001, 128),
            KfdGpu("0000:13:00.0", 120001, 129),
        ],
    )
    monkeypatch.setenv("R9V_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("R9V_MIN_PCIE_BANDWIDTH_GBPS", "15,15")
    monkeypatch.delenv("R9V_EXPECTED_PCIE_LINKS", raising=False)
    reporter = Reporter()

    selected = doctor._selected_gpus(reporter, 2, tmp_path)

    links = [check for check in reporter.checks if check.name == "pcie-link"]
    assert [check.status for check in links] == ["PASS", "FAIL"]
    assert "upstream hop 0000:00:01.1" in links[1].message
    assert "16 GT/s x4" in links[1].message
    assert abs(selected[1][4] - 7.876923) < 1e-5


@pytest.mark.parametrize("hop_width", ["4", None, "0", "-1"])
def test_exact_link_expectation_requires_negotiated_upstream_width(
    tmp_path: Path, monkeypatch, hop_width
) -> None:
    bridge = tmp_path / "devices/pci0000:00/0000:00:02.2"
    device = bridge / "0000:13:00.0"
    device.mkdir(parents=True)
    for node, current_speed, current_width, max_speed, max_width in (
        (bridge, "2.5 GT/s", hop_width, "16.0 GT/s", "4"),
        (device, "32.0 GT/s", "16", "32.0 GT/s", "16"),
    ):
        (node / "current_link_speed").write_text(current_speed, encoding="utf-8")
        if current_width is not None:
            (node / "current_link_width").write_text(current_width, encoding="utf-8")
        (node / "max_link_speed").write_text(max_speed, encoding="utf-8")
        (node / "max_link_width").write_text(max_width, encoding="utf-8")
    by_bus = tmp_path / "bus/pci/devices"
    by_bus.mkdir(parents=True)
    (by_bus / "0000:13:00.0").symlink_to(device)
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/amd-smi")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "GPU: 0\n BDF: 0000:13:00.0\n", ""
        ),
    )
    monkeypatch.setattr(
        doctor,
        "discover_kfd_gpus",
        lambda _root: [KfdGpu("0000:13:00.0", 120001, 128)],
    )
    monkeypatch.setenv("R9V_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("R9V_EXPECTED_PCIE_LINKS", "Gen4x4")
    monkeypatch.setenv("R9V_MIN_PCIE_BANDWIDTH_GBPS", "7")

    reporter = Reporter()
    doctor._selected_gpus(reporter, 1, tmp_path)

    exact = [
        check for check in reporter.checks if check.name == "pcie-link-expectation"
    ]
    if hop_width == "4":
        assert [check.status for check in exact] == ["PASS"]
        assert exact[0].details == {"expected_speed_gts": 16.0, "expected_width": 4}
    else:
        assert [check.status for check in exact] == ["FAIL"]
        monkeypatch.delenv("R9V_EXPECTED_PCIE_LINKS")
        reporter = Reporter()
        selected = doctor._selected_gpus(reporter, 1, tmp_path)
        assert selected[0][2:] == (None, None, None)
        assert not any(check.name == "pcie-link-lock" for check in reporter.checks)


@pytest.mark.parametrize("width", [None, "Unknown", "0", "-1", "8"])
def test_endpoint_capacity_never_substitutes_maximum_lane_width(
    tmp_path: Path, width
) -> None:
    (tmp_path / "max_link_speed").write_text("16.0 GT/s")
    (tmp_path / "max_link_width").write_text("16")
    if width is not None:
        (tmp_path / "current_link_width").write_text(width)
    if width == "8":
        assert doctor._pcie_capacity(tmp_path) == (16.0, 8)
    else:
        with pytest.raises(ValueError, match="speed/width capacity pair"):
            doctor._pcie_capacity(tmp_path)


def test_visible_devices_rejects_duplicate_selection(
    tmp_path: Path, monkeypatch
) -> None:
    inventory = "GPU: 0\n BDF: 0000:03:00.0\nGPU: 1\n BDF: 0000:13:00.0\n"
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/amd-smi")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, inventory, ""),
    )
    monkeypatch.setattr(
        doctor,
        "discover_kfd_gpus",
        lambda _root: [
            KfdGpu("0000:03:00.0", 120001, 128),
            KfdGpu("0000:13:00.0", 120001, 129),
        ],
    )
    monkeypatch.setenv("R9V_VISIBLE_DEVICES", "0,0")
    reporter = Reporter()

    selected = doctor._selected_gpus(reporter, 2, tmp_path)

    assert len(selected) == 1
    assert any(
        check.name == "gpu-order"
        and check.status == "FAIL"
        and "repeats device" in check.message
        for check in reporter.checks
    )


def test_rotational_flag_accepts_boolean_and_string_lsblk_forms() -> None:
    assert doctor._any_rotational([{"rota": True}])
    assert doctor._any_rotational([{"rota": "1"}])
    assert not doctor._any_rotational([{"rota": False}])
    assert not doctor._any_rotational([{"rota": "0"}])
    assert not doctor._any_rotational([{"rota": None}])


def test_prometheus_metrics_sum_engine_labels() -> None:
    metrics = parse_prometheus_metrics(
        """
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0"} 100
vllm:spec_decode_num_drafts_total{engine="1"} 50
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 120
vllm:spec_decode_num_accepted_tokens_total{engine="1"} 80
"""
    )

    assert metrics["vllm:spec_decode_num_drafts_total"] == 150
    assert metrics["vllm:spec_decode_num_accepted_tokens_total"] == 200


def test_cache_policy_warns_when_slowest_rank_has_no_cache(monkeypatch) -> None:
    monkeypatch.setenv("R9V_TIERED_IQ_MOE_VARIANT", "reuse3v2")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_SLOTS", "16")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_RANKS", "0")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_POLICY", "lru")
    monkeypatch.setenv("R9V_MTP_SPEC_TOKENS", "2")
    monkeypatch.setenv("R9V_PLE_RESIDENCY_MODE", "ssd")
    reporter = Reporter()
    fake_gpu = parse_amd_smi_list(
        "GPU: 0\n BDF: 0000:03:00.0\nGPU: 1\n BDF: 0000:13:00.0\n"
    )

    _check_profile_policy(
        reporter,
        2,
        [
            (0, fake_gpu[0], 32.0, 16, 63.0),
            (1, fake_gpu[1], 16.0, 4, 7.88),
        ],
    )

    assert any(
        check.name == "cache-rank-topology" and check.status == "WARN"
        for check in reporter.checks
    )


def test_cache_policy_suppresses_slowest_rank_warning_when_manifest_is_all_hot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("R9V_TIERED_IQ_MOE_VARIANT", "reuse3v2")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_SLOTS", "16")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_RANKS", "0")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_POLICY", "lru")
    monkeypatch.setenv("R9V_MTP_SPEC_TOKENS", "2")
    monkeypatch.setenv("R9V_PLE_RESIDENCY_MODE", "ssd")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 1, "num_layers": 48, "num_experts": 512,
        "ranks": {
            "0": {"hot_experts_by_layer": [[0] for _ in range(48)]},
            "1": {"hot_experts_by_layer": [list(range(512)) for _ in range(48)]},
        },
    }))
    monkeypatch.setenv("R9V_EXPERT_MANIFEST_PATH", str(manifest))
    fake_gpu = parse_amd_smi_list("GPU: 0\n BDF: 0000:03:00.0\nGPU: 1\n BDF: 0000:13:00.0\n")
    reporter = Reporter()
    _check_profile_policy(reporter, 2, [
        (0, fake_gpu[0], 32.0, 16, 63.0),
        (1, fake_gpu[1], 16.0, 4, 7.88),
    ])
    assert not any(check.name == "cache-rank-topology" for check in reporter.checks)


def test_launch_runs_preflight_and_example_config_preserves_caller_values() -> None:
    root = Path(__file__).resolve().parents[1]
    launch = (root / "scripts/launch.sh").read_text(encoding="utf-8")
    example = (
        root / "profiles/qwen38-flash-next/dual-r9700/user-config.example.env"
    ).read_text(encoding="utf-8")

    assert '"$repo_root/scripts/profile-doctor.sh"' in launch
    assert "R9V_PREFLIGHT=0" in launch
    assert "${R9V_VISIBLE_DEVICES:=0,1}" in example
    assert "R9V_EXPECTED_GPU_BDFS" in example


def test_manifest_budget_rejects_cache_overcommit(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifests/placement.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "num_layers": 48,
                "num_experts": 512,
                "ranks": {
                    str(rank): {
                        "hot_count": count,
                        "hot_experts_by_layer": [list(range(count)) for _ in range(48)],
                    }
                    for rank, count in enumerate((329, 385))
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("R9V_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("R9V_MANIFEST_REL", "manifests/placement.json")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_SLOTS", "16")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_RANKS", "1")
    monkeypatch.setenv("R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK", "329,385")
    reporter = Reporter()

    _check_manifest_budget(reporter, 2)

    assert any(
        check.name == "expert-budget" and check.status == "FAIL"
        for check in reporter.checks
    )


def test_runtime_check_proves_decode_path_and_mtp_metrics(monkeypatch) -> None:
    expected_env = {
        "HIP_VISIBLE_DEVICES": "0,1",
        "QWEN38_TIERED_IQ_MOE_VARIANT": "reuse3v2",
        "QWEN38_TIERED_PREFILL_GROUP_SIZE": "16",
        "QWEN38_TIERED_EXPERT_CACHE_SLOTS": "16",
        "QWEN38_TIERED_EXPERT_CACHE_RANKS": "1",
        "QWEN38_TIERED_EXPERT_CACHE_POLICY": "lru",
        "VLLM_PLE_RESIDENCY_MODE": "ssd",
        "VLLM_PLE_WORKER_TIMING": "1",
        "RADIANCE_USE_R4D": "0",
        "RADIANCE_USE_R4D_AR": "0",
    }
    for name, value in {
        "R9V_VISIBLE_DEVICES": "0,1",
        "R9V_TIERED_IQ_MOE_VARIANT": "reuse3v2",
        "R9V_TIERED_PREFILL_GROUP_SIZE": "16",
        "R9V_TIERED_EXPERT_CACHE_SLOTS": "16",
        "R9V_TIERED_EXPERT_CACHE_RANKS": "1",
        "R9V_TIERED_EXPERT_CACHE_POLICY": "lru",
        "R9V_PLE_RESIDENCY_MODE": "ssd",
        "R9V_PLE_WORKER_TIMING": "1",
        "R9V_ENABLE_FUSED_GDN_MTP": "1",
    }.items():
        monkeypatch.setenv(name, value)
    logs = "\n".join(
        (
            "Tiered GGUF experts ready on TP rank 0",
            "Tiered GGUF experts ready on TP rank 1",
            "Using tiered IQ MoE exact-shape variant reuse3v2",
            "Using tiered IQ MoE grouped-16 prefill",
            "Qwen3.8 TP2 fused speculative GDN HIP kernel enabled",
            "PLE worker timing layer=0 total_us=10.000",
        )
    )
    monkeypatch.setattr(
        doctor,
        "_docker_environment",
        lambda _container: ("running", expected_env, "sha256:test", {}),
    )
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, logs, ""),
    )
    monkeypatch.setattr(
        doctor,
        "_fetch_metrics",
        lambda _port: "\n".join(
            (
                "vllm:spec_decode_num_drafts_total 100",
                "vllm:spec_decode_num_draft_tokens_total 200",
                "vllm:spec_decode_num_accepted_tokens_total 150",
            )
        ),
    )
    reporter = Reporter()

    doctor._check_runtime(reporter, 2)

    mtp = next(check for check in reporter.checks if check.name == "mtp-metrics")
    assert mtp.status == "PASS"
    assert mtp.details is not None
    assert mtp.details["mean_emitted_length"] == 2.5
    assert not [check for check in reporter.checks if check.status == "FAIL"]


def test_ple_hash_passes_when_the_payload_matches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("R9V_PLE_EXPECTED_SHA256", PLE_SAMPLE_SHA256)
    reporter = Reporter()

    _check_ple_hash(reporter, _sample_ple(tmp_path), True)

    check = next(check for check in reporter.checks if check.name == "ple-hash")
    assert check.status == "PASS"


def test_ple_hash_mismatch_fails_with_regeneration_remediation(
    tmp_path: Path, monkeypatch
) -> None:
    path = _sample_ple(tmp_path)
    path.write_bytes(PLE_SAMPLE + b"corruption")
    monkeypatch.setenv("R9V_PLE_EXPECTED_SHA256", PLE_SAMPLE_SHA256)
    reporter = Reporter()

    _check_ple_hash(reporter, path, True)

    check = next(check for check in reporter.checks if check.name == "ple-hash")
    assert check.status == "FAIL"
    assert check.details is not None
    assert check.details["expected_sha256"] == PLE_SAMPLE_SHA256
    assert check.details["actual_sha256"] != PLE_SAMPLE_SHA256
    assert check.remediation is not None
    assert "regenerate it from the" in check.remediation
    assert "Never hand-repair" in check.remediation


def test_ple_hash_is_noted_but_skipped_unless_requested(
    tmp_path: Path, monkeypatch
) -> None:
    # A corrupt payload must not be read at all when --hash-ple is absent.
    path = _sample_ple(tmp_path)
    path.write_bytes(PLE_SAMPLE + b"corruption")
    monkeypatch.setenv("R9V_PLE_EXPECTED_SHA256", PLE_SAMPLE_SHA256)
    reporter = Reporter()

    _check_ple_hash(reporter, path, False)

    check = next(check for check in reporter.checks if check.name == "ple-hash")
    assert check.status == "NOTE"
    assert "--hash-ple" in check.message

    monkeypatch.setenv("R9V_PLE_EXPECTED_SHA256", "")
    unset = Reporter()

    _check_ple_hash(unset, path, True)

    check = next(check for check in unset.checks if check.name == "ple-hash")
    assert check.status == "NOTE"
    assert "size only" in check.message


def test_every_actionable_doctor_result_has_remediation() -> None:
    source = Path(doctor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    missing: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"warn", "fail"}:
            continue
        has_keyword = any(keyword.arg == "remediation" for keyword in node.keywords)
        if len(node.args) < 3 and not has_keyword:
            missing.append(node.lineno)

    assert missing == []


def test_scheduler_rewinds_fail_doctor_even_when_public_metrics_are_missing(monkeypatch):
    record = {"total_preemptions": 2, "computed_tokens_before_reset": 129024}
    monkeypatch.setattr(
        doctor, "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, json.dumps(record), ""),
    )
    reporter = Reporter()
    doctor._check_scheduler_pressure(reporter)
    assert reporter.checks[0].status == "FAIL"
    assert "fixed KV allocation" in reporter.checks[0].remediation


def test_missing_scheduler_record_does_not_certify_capacity(monkeypatch):
    monkeypatch.setattr(
        doctor, "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "{}", ""),
    )
    reporter = Reporter()
    doctor._check_scheduler_pressure(reporter)
    assert reporter.checks[0].status == "NOTE"
    assert "does not certify" in reporter.checks[0].message


def scheduler_runtime(monkeypatch, tmp_path, total, recorded, container_id="abc123"):
    """A running container with `total` rewinds whose saved setup recorded `recorded`
    rewinds from first-start qualification in container abc123."""
    (tmp_path / "setup.json").write_text(json.dumps({"qualification": {
        "preemptions": {"container_id": "abc123", "count": recorded}}}))
    monkeypatch.setenv("R9V_STATE_DIR", str(tmp_path))

    def run(command, **kwargs):
        stdout = container_id if command[:2] == ["docker", "inspect"] else json.dumps(
            {"total_preemptions": total})
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(doctor, "_run", run)


def test_rewinds_from_first_start_qualification_do_not_fail_the_runtime_doctor(monkeypatch, tmp_path):
    scheduler_runtime(monkeypatch, tmp_path, total=7, recorded=7)
    reporter = Reporter()

    doctor._check_scheduler_pressure(reporter)

    assert reporter.checks[0].status == "NOTE"
    assert "All 7 scheduler rewinds" in reporter.checks[0].message


def test_rewinds_after_qualification_still_fail_and_say_how_many_are_new(monkeypatch, tmp_path):
    scheduler_runtime(monkeypatch, tmp_path, total=9, recorded=7)
    reporter = Reporter()

    doctor._check_scheduler_pressure(reporter)

    assert reporter.checks[0].status == "FAIL"
    assert "9 times" in reporter.checks[0].message
    assert "2 since first-start qualification, which caused 7" in reporter.checks[0].message


def test_qualification_rewinds_of_an_earlier_container_are_not_discounted(monkeypatch, tmp_path):
    scheduler_runtime(monkeypatch, tmp_path, total=7, recorded=7, container_id="restarted")
    reporter = Reporter()

    doctor._check_scheduler_pressure(reporter)

    assert reporter.checks[0].status == "FAIL"
    assert "since first-start" not in reporter.checks[0].message


def test_qwen_configuration_reference_covers_every_portable_setting() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "profiles/qwen38-flash-next/dual-r9700/README.md").read_text(
        encoding="utf-8"
    )

    for setting in (
        "R9V_CONFIG_FILE",
        "R9V_VISIBLE_DEVICES",
        "R9V_EXPECTED_GPU_BDFS",
        "R9V_EXPECTED_PCIE_LINKS",
        "R9V_MIN_PCIE_BANDWIDTH_GBPS",
        "R9V_MIN_HOST_RAM_BYTES",
        "R9V_MIN_HOST_AVAILABLE_BYTES",
        "R9V_CPU_OFFLOAD_GB",
        "R9V_PLE_PATH",
        "R9V_PLE_EXPECTED_SHA256",
        "R9V_PLE_RESIDENCY_MODE",
        "R9V_REQUIRE_PLE_NONROTATIONAL",
        "R9V_PLE_WORKER_TIMING",
        "R9V_TIERED_EXPERT_CACHE_RANKS",
        "R9V_TIERED_EXPERT_CACHE_SLOTS",
        "R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK",
        "R9V_KV_CACHE_MEMORY_BYTES",
        "R9V_MTP_SPEC_TOKENS",
        "R9V_DOCTOR_STRICT",
        "R9V_PREFLIGHT",
    ):
        assert setting in readme
    for discovery in ("amd-smi list", "/proc/meminfo", "findmnt", "lsblk"):
        assert discovery in readme
    for status in ("`PASS`", "`WARN`", "`FAIL`", "`NOTE`"):
        assert status in readme
    assert "--runtime --json" in readme


def test_cpu_doctor_reports_contention_without_assuming_powersave_disables_boost(tmp_path, monkeypatch):
    from tools import profile_doctor as doctor
    monkeypatch.setattr(doctor.os, 'sched_getaffinity', lambda _: {0, 6})
    for cpu in (0, 6):
        base = tmp_path / 'sys/devices/system/cpu' / f'cpu{cpu}'
        (base / 'topology').mkdir(parents=True)
        (base / 'cpufreq').mkdir()
        (base / 'topology/physical_package_id').write_text('0')
        (base / 'topology/core_id').write_text('0')
        (base / 'cpufreq/scaling_governor').write_text('powersave')
    (tmp_path / 'proc/pressure').mkdir(parents=True)
    (tmp_path / 'proc/pressure/cpu').write_text('some avg10=25.0 avg60=10.0 total=1\n')
    reporter = doctor.Reporter()
    doctor._check_host_cpu(reporter, tmp_path / 'sys', tmp_path / 'proc')
    assert '1 physical cores' in reporter.checks[0].message
    assert reporter.checks[0].status == 'PASS'
    assert reporter.checks[1].status == 'WARN'


def test_normal_zone_warning_is_distinct_from_gpu_oom(tmp_path):
    zoneinfo = tmp_path / "zoneinfo"
    zoneinfo.write_text("Node 0, zone Normal\n  pages free     10\n        min      5\n        low      20\n        high     30\n")
    reporter = Reporter()
    doctor._check_normal_zone_pressure(reporter, tmp_path)
    assert reporter.checks[0].name == "host-normal-zone"
    assert reporter.checks[0].status == "WARN"
    assert "not GPU OOM" in reporter.checks[0].message


def test_empty_normal_zone_telemetry_is_unavailable(tmp_path):
    (tmp_path / "zoneinfo").write_text("Node 0, zone DMA\n  pages free     10\n")
    reporter = Reporter()
    doctor._check_normal_zone_pressure(reporter, tmp_path)
    assert reporter.checks[0].name == "host-normal-zone"
    assert reporter.checks[0].status == "NOTE"
    assert "unavailable" in reporter.checks[0].message


def test_amd_smi_uuid_is_not_accepted_as_rocr_selection(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/amd-smi")
    monkeypatch.setattr(doctor, "_run", lambda *a, **k: subprocess.CompletedProcess(
        [], 0, "GPU: 0\n BDF: 0000:03:00.0\n UUID: first\n NODE_ID: 1\n", ""))
    monkeypatch.setenv("R9V_VISIBLE_DEVICES", "first,second")
    reporter = Reporter()
    assert doctor._selected_gpus(reporter, 2, Path("/unused")) == []
    assert reporter.checks[-1].status == "FAIL"
    assert "not ROCR UUIDs" in reporter.checks[-1].remediation


def test_custom_headroom_setup_phase_reports_pending_but_direct_doctor_fails(monkeypatch):
    monkeypatch.delenv("R9V_PLACEMENT_PLAN", raising=False)
    monkeypatch.setenv("R9V_HEADROOM_SELECTION", "1")
    reporter = Reporter()
    doctor._check_placement_plan(reporter)
    assert reporter.checks[-1].status == "FAIL"
    monkeypatch.setenv("R9V_SETUP_PHASE", "1")
    reporter = Reporter()
    doctor._check_placement_plan(reporter)
    assert reporter.checks[-1].status == "NOTE"
    monkeypatch.setenv("R9V_RUNTIME_PREBUILT", "1")
    reporter = Reporter()
    doctor._check_placement_plan(reporter, runtime=True)
    assert reporter.checks[-1].status == "FAIL"


@pytest.mark.parametrize(
    ("group_size", "status"),
    [("32", "PASS"), ("16", "PASS"), ("64", "FAIL")],
)
def test_decode_policy_accepts_wmma_prefill_group_32(monkeypatch, group_size, status) -> None:
    monkeypatch.setenv("R9V_TIERED_IQ_MOE_VARIANT", "reuse3v2")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_SLOTS", "16")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_RANKS", "1")
    monkeypatch.setenv("R9V_TIERED_EXPERT_CACHE_POLICY", "lru")
    monkeypatch.setenv("R9V_MTP_SPEC_TOKENS", "2")
    monkeypatch.setenv("R9V_PLE_RESIDENCY_MODE", "ssd")
    monkeypatch.setenv("R9V_TIERED_PREFILL_GROUP_SIZE", group_size)
    reporter = Reporter()
    fake_gpu = parse_amd_smi_list("GPU: 0\n BDF: 0000:03:00.0\nGPU: 1\n BDF: 0000:13:00.0\n")

    _check_profile_policy(reporter, 2, [
        (0, fake_gpu[0], 32.0, 16, 63.0),
        (1, fake_gpu[1], 16.0, 4, 7.88),
    ])

    decode = [check for check in reporter.checks if check.name == "decode-policy"]
    assert [check.status for check in decode] == [status]


def test_ced_quality_raises_the_available_ram_start_needs_by_its_host_copies(tmp_path, monkeypatch):
    from tools import profile_doctor as doctor
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(f"MemTotal: {120 * 1024**2} kB\nMemAvailable: {58 * 1024**2} kB\n")
    monkeypatch.setenv("R9V_MIN_HOST_AVAILABLE_BYTES", str(56 * 1024**3))
    monkeypatch.setattr(doctor, "ced_quality_host_bytes", lambda: 4 * 1024**3)
    reporter = doctor.Reporter()

    doctor._check_host_memory(reporter, proc)

    check = next(c for c in reporter.checks if c.name == "host-memory")
    assert check.status == "FAIL"
    assert "below the 60.00 GiB" in check.message and "CED quality's 4.00 GiB" in check.message


def test_ced_quality_host_copies_leave_a_start_with_enough_ram_passing(tmp_path, monkeypatch):
    from tools import profile_doctor as doctor
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(f"MemTotal: {120 * 1024**2} kB\nMemAvailable: {61 * 1024**2} kB\n")
    monkeypatch.setenv("R9V_MIN_HOST_AVAILABLE_BYTES", str(56 * 1024**3))
    monkeypatch.setattr(doctor, "ced_quality_host_bytes", lambda: 4 * 1024**3)
    reporter = doctor.Reporter()

    doctor._check_host_memory(reporter, proc)

    check = next(c for c in reporter.checks if c.name == "host-memory")
    assert check.status == "PASS"
    assert "start needs 60.00 GiB" in check.message
