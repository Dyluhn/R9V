# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

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
)

# A stand-in for the 26.8 GiB derived PLE table. Tests never hash the real one.
PLE_SAMPLE = b"r9v-ple-test-payload"
PLE_SAMPLE_SHA256 = (
    "e28b313ed08f5e62eb9c36da7775280ad2720c02e1aa1c2a9767bab93ff7b4b0"
)


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
        "gfx_target_version 120001\n"
        "location_id 4864\n"
        "domain 0\n"
        "drm_render_minor 129\n",
        encoding="utf-8",
    )

    assert discover_kfd_gpus(tmp_path)[0].bdf == "0000:13:00.0"


def test_pcie_payload_estimate_matches_gen4_x4() -> None:
    assert abs(pcie_payload_gbps(16.0, 4) - 7.876923) < 1e-5


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
    inventory = (
        "GPU: 0\n BDF: 0000:03:00.0\n"
        "GPU: 1\n BDF: 0000:13:00.0\n"
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
        check
        for check in reporter.checks
        if check.name == "pcie-link-expectation"
    ]
    assert [check.status for check in exact] == ["PASS", "FAIL"]
    assert exact[1].remediation is not None
    assert "cannot change a negotiated PCIe link" in exact[1].remediation


def test_pcie_floor_scores_the_slowest_upstream_hop(
    tmp_path: Path, monkeypatch
) -> None:
    # Endpoint sysfs on both cards reads Gen5x16, but rank 1 sits behind a
    # bridge that negotiated Gen4x4. The floor must score the bridge, not the
    # endpoint.
    for bridge, bdf, hop_speed, hop_width in (
        ("0000:00:01.0", "0000:03:00.0", "32.0 GT/s", "16"),
        ("0000:00:01.1", "0000:13:00.0", "16.0 GT/s", "4"),
    ):
        bridge_dir = tmp_path / "devices/pci0000:00" / bridge
        device_dir = bridge_dir / bdf
        device_dir.mkdir(parents=True)
        (bridge_dir / "current_link_speed").write_text(hop_speed, encoding="utf-8")
        (bridge_dir / "current_link_width").write_text(hop_width, encoding="utf-8")
        (device_dir / "current_link_speed").write_text("32.0 GT/s", encoding="utf-8")
        (device_dir / "current_link_width").write_text("16", encoding="utf-8")
        by_bus = tmp_path / "bus/pci/devices"
        by_bus.mkdir(parents=True, exist_ok=True)
        (by_bus / bdf).symlink_to(device_dir)
    inventory = (
        "GPU: 0\n BDF: 0000:03:00.0\n"
        "GPU: 1\n BDF: 0000:13:00.0\n"
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


def test_launch_runs_preflight_and_example_config_preserves_caller_values() -> None:
    root = Path(__file__).resolve().parents[1]
    launch = (root / "scripts/launch.sh").read_text(encoding="utf-8")
    example = (
        root
        / "profiles/qwen38-flash-next/dual-r9700/user-config.example.env"
    ).read_text(encoding="utf-8")

    assert '"$repo_root/scripts/profile-doctor.sh"' in launch
    assert "R9V_PREFLIGHT=0" in launch
    assert '${R9V_VISIBLE_DEVICES:=0,1}' in example
    assert "R9V_EXPECTED_GPU_BDFS" in example


def test_manifest_budget_rejects_cache_overcommit(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifests/placement.json"
    manifest.parent.mkdir()
    manifest.write_text(
        '{"ranks":{"0":{"hot_count":329},"1":{"hot_count":385}}}',
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
        lambda _container: ("running", expected_env, "sha256:test"),
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


def test_ple_hash_passes_when_the_payload_matches(
    tmp_path: Path, monkeypatch
) -> None:
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
        has_keyword = any(
            keyword.arg == "remediation" for keyword in node.keywords
        )
        if len(node.args) < 3 and not has_keyword:
            missing.append(node.lineno)

    assert missing == []


def test_qwen_configuration_reference_covers_every_portable_setting() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (
        root / "profiles/qwen38-flash-next/dual-r9700/README.md"
    ).read_text(encoding="utf-8")

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
