# SPDX-License-Identifier: Apache-2.0
"""Deterministic unit tests for vm/r9v-hw-container.sh.

Tests topology discovery, HSA ordinal derivation, support node identification,
command construction, and fail-closed error handling against synthetic KFD topologies.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HW_CONTAINER = ROOT / "vm" / "r9v-hw-container.sh"


def make_kfd_node(
    nodes_dir: Path,
    node_id: int,
    simd_count: int,
    drm_render_minor: int,
    extra_properties: str = "",
) -> Path:
    node_dir = nodes_dir / str(node_id)
    node_dir.mkdir(parents=True, exist_ok=True)
    props = (
        f"cpu_cores_count 0\n"
        f"simd_count {simd_count}\n"
        f"drm_render_minor {drm_render_minor}\n"
        f"{extra_properties}"
    )
    (node_dir / "properties").write_text(props, encoding="utf-8")
    return node_dir


def run_hw_container(
    args: list[str],
    env_overrides: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "R9V_TEST_SKIP_DEV_CHECK": "1",
        **env_overrides,
    }
    return subprocess.run(
        [str(HW_CONTAINER), *args],
        env=env,
        capture_output=True,
        text=True,
    )


def test_dry_run_multi_gpu_with_support_node(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 0, simd_count=0, drm_render_minor=0)  # CPU
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)  # GPU 0
    make_kfd_node(nodes, 2, simd_count=128, drm_render_minor=129)  # GPU 1
    make_kfd_node(nodes, 3, simd_count=4, drm_render_minor=130)  # GPU 2 (e.g. iGPU)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128 /dev/dri/renderD129",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode == 0, res.stderr
    assert "targets [/dev/dri/renderD128 /dev/dri/renderD129]" in res.stdout
    assert "support [/dev/dri/renderD130]" in res.stdout
    assert "visible ordinals: 0,1" in res.stdout
    assert "target: /dev/dri/renderD128 -> HSA ordinal 0" in res.stdout
    assert "target: /dev/dri/renderD129 -> HSA ordinal 1" in res.stdout
    assert "support: /dev/dri/renderD130 -> HSA ordinal 2" in res.stdout

    dry_line = next(line for line in res.stdout.splitlines() if line.startswith("DRY_RUN:"))
    assert "--device /dev/kfd" in dry_line
    assert "--device /dev/dri/renderD128" in dry_line
    assert "--device /dev/dri/renderD129" in dry_line
    assert "--device /dev/dri/renderD130" in dry_line
    assert "-e ROCR_VISIBLE_DEVICES=0,1" in dry_line
    assert "-e R9V_REQUIRE_GPU=1" in dry_line
    assert "-e R9V_GPU_LANE=1" in dry_line
    assert "--volume /sys/bus/pci:/sys/bus/pci:ro" in dry_line
    assert "./scripts/ci-gates.sh hardware" in dry_line


def test_dry_run_custom_ordinal_order(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)
    make_kfd_node(nodes, 2, simd_count=128, drm_render_minor=129)
    make_kfd_node(nodes, 3, simd_count=4, drm_render_minor=130)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD129 /dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode == 0, res.stderr
    assert "targets [/dev/dri/renderD129 /dev/dri/renderD128]" in res.stdout
    assert "support [/dev/dri/renderD130]" in res.stdout
    assert "visible ordinals: 1,0" in res.stdout
    assert "-e ROCR_VISIBLE_DEVICES=1,0" in res.stdout


def test_dry_run_single_target(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)
    make_kfd_node(nodes, 2, simd_count=128, drm_render_minor=129)
    make_kfd_node(nodes, 3, simd_count=4, drm_render_minor=130)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD130",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode == 0, res.stderr
    assert "targets [/dev/dri/renderD130]" in res.stdout
    assert "support [/dev/dri/renderD128 /dev/dri/renderD129]" in res.stdout
    assert "visible ordinals: 2" in res.stdout
    assert "-e ROCR_VISIBLE_DEVICES=2" in res.stdout


def test_dry_run_no_support_needed(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 0, simd_count=0, drm_render_minor=0)
    make_kfd_node(nodes, 1, simd_count=64, drm_render_minor=128)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode == 0, res.stderr
    assert "targets [/dev/dri/renderD128]" in res.stdout
    assert "support [none]" in res.stdout
    assert "visible ordinals: 0" in res.stdout
    assert "-e ROCR_VISIBLE_DEVICES=0" in res.stdout


def test_dry_run_comma_separated_render_nodes(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)
    make_kfd_node(nodes, 2, simd_count=128, drm_render_minor=129)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128,/dev/dri/renderD129",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode == 0, res.stderr
    assert "targets [/dev/dri/renderD128 /dev/dri/renderD129]" in res.stdout
    assert "visible ordinals: 0,1" in res.stdout


def test_dry_run_custom_gate_args(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)

    res = run_hw_container(
        ["--dry-run", "--", "gpu-smoke"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode == 0, res.stderr
    assert "./scripts/ci-gates.sh gpu-smoke" in res.stdout


def test_numerical_sorting_of_topology_nodes(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    # Create directory numbers out of order and with multi-digit IDs:
    # 0 (CPU), 2, 10, 1.
    make_kfd_node(nodes, 0, simd_count=0, drm_render_minor=0)
    make_kfd_node(nodes, 2, simd_count=64, drm_render_minor=129)
    make_kfd_node(nodes, 10, simd_count=64, drm_render_minor=130)
    make_kfd_node(nodes, 1, simd_count=64, drm_render_minor=128)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128 /dev/dri/renderD129 /dev/dri/renderD130",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode == 0, res.stderr
    assert "target: /dev/dri/renderD128 -> HSA ordinal 0" in res.stdout
    assert "target: /dev/dri/renderD129 -> HSA ordinal 1" in res.stdout
    assert "target: /dev/dri/renderD130 -> HSA ordinal 2" in res.stdout
    assert "visible ordinals: 0,1,2" in res.stdout


def test_fails_when_r9v_render_nodes_missing(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)

    env = {k: v for k, v in os.environ.items() if k != "R9V_RENDER_NODES"}
    env["KFD_TOPOLOGY_ROOT"] = str(nodes)
    env["R9V_TEST_SKIP_DEV_CHECK"] = "1"

    res = subprocess.run(
        [str(HW_CONTAINER), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "R9V_RENDER_NODES is required" in res.stderr


def test_fails_when_r9v_render_nodes_empty(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "   ",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode != 0
    assert "R9V_RENDER_NODES is required" in res.stderr or "at least one render node" in res.stderr


def test_fails_on_invalid_render_node_syntax(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)

    for bad in ("/dev/dri/card0", "/dev/kfd", "renderD128", "/dev/dri/renderD"):
        res = run_hw_container(
            ["--dry-run"],
            {
                "R9V_RENDER_NODES": bad,
                "KFD_TOPOLOGY_ROOT": str(nodes),
            },
        )
        assert res.returncode != 0
        assert f"invalid render node: {bad}" in res.stderr


def test_fails_on_duplicate_target_nodes(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128 /dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode != 0
    assert "duplicate render node in R9V_RENDER_NODES: /dev/dri/renderD128" in res.stderr


def test_fails_on_unresolvable_target_node(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD129",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode != 0
    assert "does not resolve to any KFD GPU topology node" in res.stderr


def test_fails_on_duplicate_drm_render_minor_in_topology(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=128)
    make_kfd_node(nodes, 2, simd_count=128, drm_render_minor=128)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode != 0
    assert "duplicate drm_render_minor 128" in res.stderr


def test_fails_on_missing_topology_directory(tmp_path: Path) -> None:
    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(tmp_path / "nonexistent"),
        },
    )
    assert res.returncode != 0
    assert "KFD topology directory not found" in res.stderr


def test_fails_on_gpu_node_with_invalid_minor(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 1, simd_count=128, drm_render_minor=0)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode != 0
    assert "invalid drm_render_minor: 0" in res.stderr


def test_fails_when_no_gpu_nodes_exist(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    make_kfd_node(nodes, 0, simd_count=0, drm_render_minor=0)

    res = run_hw_container(
        ["--dry-run"],
        {
            "R9V_RENDER_NODES": "/dev/dri/renderD128",
            "KFD_TOPOLOGY_ROOT": str(nodes),
        },
    )
    assert res.returncode != 0
    assert "no GPU nodes found in KFD topology" in res.stderr
