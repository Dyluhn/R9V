#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only host and runtime checks for an R9V profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PLE_EXPECTED_BYTES = 28_800_138_240


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    message: str
    remediation: str | None = None
    details: dict[str, Any] | None = None


class Reporter:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        status: str,
        name: str,
        message: str,
        remediation: str | None = None,
        **details: Any,
    ) -> None:
        self.checks.append(
            Check(status, name, message, remediation, details or None)
        )

    def passed(self, name: str, message: str, **details: Any) -> None:
        self.add("PASS", name, message, **details)

    def warn(
        self,
        name: str,
        message: str,
        remediation: str | None = None,
        **details: Any,
    ) -> None:
        self.add("WARN", name, message, remediation, **details)

    def fail(
        self,
        name: str,
        message: str,
        remediation: str | None = None,
        **details: Any,
    ) -> None:
        self.add("FAIL", name, message, remediation, **details)

    def note(self, name: str, message: str, **details: Any) -> None:
        self.add("NOTE", name, message, **details)

    def counts(self) -> dict[str, int]:
        return {
            status: sum(check.status == status for check in self.checks)
            for status in ("PASS", "WARN", "FAIL", "NOTE")
        }


@dataclass(frozen=True)
class AmdGpu:
    index: int
    bdf: str
    uuid: str | None
    node_id: int | None


@dataclass(frozen=True)
class KfdGpu:
    bdf: str
    gfx_target: int
    render_minor: int | None


@dataclass(frozen=True)
class PcieLink:
    speed_gts: float
    width: int

    @property
    def generation(self) -> int | None:
        for generation, speed in PCIE_GENERATION_SPEED_GTS.items():
            if abs(self.speed_gts - speed) < 0.01:
                return generation
        return None

    def config_value(self) -> str:
        if self.generation is not None:
            return f"Gen{self.generation}x{self.width}"
        return f"{self.speed_gts:g}x{self.width}"


PCIE_GENERATION_SPEED_GTS = {
    1: 2.5,
    2: 5.0,
    3: 8.0,
    4: 16.0,
    5: 32.0,
    6: 64.0,
}


def _run(command: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(
            command,
            124,
            error.stdout or "",
            error.stderr or f"timed out after {timeout:g} seconds",
        )
    except FileNotFoundError as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def _normalize_bdf(value: str) -> str:
    value = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        value = f"0000:{value}"
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        raise ValueError(f"invalid PCI BDF {value!r}")
    return value


def parse_amd_smi_list(text: str) -> list[AmdGpu]:
    records: list[AmdGpu] = []
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"GPU:\s*(\d+)", line)
        if match:
            if current is not None:
                records.append(_amd_gpu_from_fields(current))
            current = {"index": match.group(1)}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        current[key.strip().lower()] = value.strip()
    if current is not None:
        records.append(_amd_gpu_from_fields(current))
    return records


def _amd_gpu_from_fields(fields: dict[str, str]) -> AmdGpu:
    node = fields.get("node_id")
    return AmdGpu(
        index=int(fields["index"]),
        bdf=_normalize_bdf(fields["bdf"]),
        uuid=fields.get("uuid"),
        node_id=int(node) if node is not None else None,
    )


def discover_kfd_gpus(sys_root: Path = Path("/sys")) -> list[KfdGpu]:
    result: list[KfdGpu] = []
    topology = sys_root / "class/kfd/kfd/topology/nodes"
    for properties in sorted(topology.glob("*/properties")):
        fields: dict[str, int] = {}
        try:
            lines = properties.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                fields[parts[0]] = int(parts[1])
        location = fields.get("location_id", 0)
        domain = fields.get("domain", 0)
        if not location or not fields.get("gfx_target_version"):
            continue
        bus = (location >> 8) & 0xFF
        device = (location >> 3) & 0x1F
        function = location & 0x7
        result.append(
            KfdGpu(
                bdf=f"{domain:04x}:{bus:02x}:{device:02x}.{function}",
                gfx_target=fields["gfx_target_version"],
                render_minor=fields.get("drm_render_minor"),
            )
        )
    return result


def _csv(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_float(value: str | None, count: int, name: str) -> list[float]:
    items = _csv(value)
    if not items:
        return []
    if len(items) != count:
        raise ValueError(f"{name} needs {count} comma-separated values")
    try:
        return [float(item) for item in items]
    except ValueError as error:
        raise ValueError(f"{name} values must be numeric") from error


def parse_expected_pcie_links(value: str | None, count: int) -> list[PcieLink]:
    items = _csv(value)
    if not items:
        return []
    if len(items) != count:
        raise ValueError(
            f"R9V_EXPECTED_PCIE_LINKS needs {count} comma-separated values"
        )
    links: list[PcieLink] = []
    for item in items:
        generation = re.fullmatch(
            r"gen\s*([1-6])\s*x\s*(\d+)", item, re.IGNORECASE
        )
        numeric = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*(?:gt/s)?\s*x\s*(\d+)", item, re.IGNORECASE
        )
        if generation:
            speed = PCIE_GENERATION_SPEED_GTS[int(generation.group(1))]
            width = int(generation.group(2))
        elif numeric:
            speed = float(numeric.group(1))
            width = int(numeric.group(2))
        else:
            raise ValueError(
                "R9V_EXPECTED_PCIE_LINKS values must look like "
                "Gen5x16 or 32x16"
            )
        if width not in {1, 2, 4, 8, 12, 16, 32}:
            raise ValueError(
                "R9V_EXPECTED_PCIE_LINKS lane widths must be one of "
                "1, 2, 4, 8, 12, 16, or 32"
            )
        links.append(PcieLink(speed, width))
    return links


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"expected 0 or 1, got {value!r}")
    return value == "1"


def _parse_bytes(value: str | None, name: str) -> int:
    if value is None or not value:
        return 0
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer byte count") from error
    if result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _human_bytes(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def pcie_payload_gbps(speed_gts: float, width: int) -> float:
    encoding_efficiency = 0.8 if speed_gts <= 5.0 else 128.0 / 130.0
    return speed_gts * width * encoding_efficiency / 8.0


def _read_first_float(path: Path) -> float | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _read_first_int(path: Path) -> int | None:
    value = _read_first_float(path)
    return int(value) if value is not None else None


def _meminfo(proc_root: Path = Path("/proc")) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = (proc_root / "meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        match = re.fullmatch(r"([^:]+):\s+(\d+)\s+kB", line)
        if match:
            result[match.group(1)] = int(match.group(2)) * 1024
    return result


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([-+\deE.]+)$", line)
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        totals[match.group(1)] = totals.get(match.group(1), 0.0) + value
    return totals


def _selected_gpus(
    reporter: Reporter,
    expected_count: int,
    sys_root: Path,
) -> list[tuple[int, AmdGpu, float | None, int | None, float | None]]:
    if shutil.which("amd-smi") is None:
        reporter.fail(
            "gpu-inventory",
            "amd-smi is not installed",
            "Install the ROCm amd-smi package, then rerun the doctor.",
        )
        return []
    result = _run(["amd-smi", "list"])
    if result.returncode != 0:
        reporter.fail(
            "gpu-inventory",
            f"amd-smi list failed: {result.stderr.strip()}",
            "Fix the ROCm driver/device permissions until `amd-smi list` works.",
        )
        return []
    try:
        inventory = parse_amd_smi_list(result.stdout)
    except (KeyError, ValueError) as error:
        reporter.fail(
            "gpu-inventory",
            f"cannot parse amd-smi list: {error}",
            "Save `amd-smi list` output and report it as an R9V compatibility issue.",
        )
        return []
    by_index = {gpu.index: gpu for gpu in inventory}
    by_uuid = {
        gpu.uuid.lower(): gpu for gpu in inventory if gpu.uuid is not None
    }
    default_visible = ",".join(str(index) for index in range(expected_count))
    visible = _csv(os.environ.get("R9V_VISIBLE_DEVICES", default_visible))
    if len(visible) != expected_count:
        reporter.fail(
            "gpu-order",
            f"R9V_VISIBLE_DEVICES selects {len(visible)} devices; profile needs {expected_count}",
            f"Set R9V_VISIBLE_DEVICES to exactly {expected_count} comma-separated amd-smi indices.",
            configured=visible,
        )
        return []

    expected_bdfs = _csv(os.environ.get("R9V_EXPECTED_GPU_BDFS"))
    if expected_bdfs and len(expected_bdfs) != expected_count:
        reporter.fail(
            "gpu-order",
            f"R9V_EXPECTED_GPU_BDFS needs {expected_count} entries",
            "Copy the BDFs from `amd-smi list` in the same order as R9V_VISIBLE_DEVICES.",
            configured=expected_bdfs,
        )
        return []
    try:
        expected_bdfs = [_normalize_bdf(value) for value in expected_bdfs]
        minimum_bandwidth = _csv_float(
            os.environ.get("R9V_MIN_PCIE_BANDWIDTH_GBPS"),
            expected_count,
            "R9V_MIN_PCIE_BANDWIDTH_GBPS",
        )
        expected_links = parse_expected_pcie_links(
            os.environ.get("R9V_EXPECTED_PCIE_LINKS"), expected_count
        )
    except ValueError as error:
        reporter.fail(
            "gpu-policy",
            str(error),
            "Correct the named comma-separated setting in R9V_CONFIG_FILE.",
        )
        return []

    kfd = {gpu.bdf: gpu for gpu in discover_kfd_gpus(sys_root)}
    selected: list[tuple[int, AmdGpu, float | None, int | None, float | None]] = []
    for rank, token in enumerate(visible):
        gpu: AmdGpu | None = None
        if token.isdigit():
            gpu = by_index.get(int(token))
        else:
            normalized_uuid = token.lower().removeprefix("gpu-")
            gpu = by_uuid.get(normalized_uuid)
        if gpu is None:
            reporter.fail(
                "gpu-order",
                f"rank {rank} device {token!r} was not found",
                "Choose an index or UUID shown by `amd-smi list` and update R9V_VISIBLE_DEVICES.",
            )
            continue
        pci = sys_root / "bus/pci/devices" / gpu.bdf
        speed = _read_first_float(pci / "current_link_speed")
        width = _read_first_int(pci / "current_link_width")
        bandwidth = (
            pcie_payload_gbps(speed, width)
            if speed is not None and width is not None
            else None
        )
        mapped = kfd.get(gpu.bdf)
        if mapped is None:
            reporter.fail(
                "gpu-architecture",
                f"rank {rank} {gpu.bdf} is absent from KFD",
                "Fix the ROCm/KFD driver binding for this GPU before launching R9V.",
            )
        elif mapped.gfx_target != 120001:
            reporter.fail(
                "gpu-architecture",
                f"rank {rank} {gpu.bdf} is gfx target {mapped.gfx_target}, not gfx1201",
                "Select a Radeon AI PRO R9700/gfx1201 device in R9V_VISIBLE_DEVICES.",
            )
        else:
            reporter.passed(
                "gpu-architecture",
                f"rank {rank} device {gpu.index} {gpu.bdf} is gfx1201",
                rank=rank,
                device_index=gpu.index,
                bdf=gpu.bdf,
                uuid=gpu.uuid,
            )
        if expected_bdfs and gpu.bdf != expected_bdfs[rank]:
            reporter.fail(
                "gpu-order",
                f"rank {rank} resolved to {gpu.bdf}, expected {expected_bdfs[rank]}",
                "Reorder R9V_VISIBLE_DEVICES, or update "
                "R9V_EXPECTED_GPU_BDFS only after confirming the intended "
                "rank placement.",
            )
        if bandwidth is None:
            if expected_links:
                reporter.fail(
                    "pcie-link-expectation",
                    f"rank {rank} {gpu.bdf} link data is unavailable; cannot "
                    f"verify expected {expected_links[rank].config_value()}",
                    "Fix sysfs/driver visibility for current_link_speed and "
                    "current_link_width. An exact link lock cannot be waived "
                    "by guessing the hardware topology.",
                )
            else:
                reporter.warn(
                    "pcie-link",
                    f"rank {rank} {gpu.bdf} link data is unavailable",
                    "Check the PCIe link in sysfs/amd-smi; do not assume reference "
                    "performance until it is visible.",
                )
        else:
            minimum = minimum_bandwidth[rank] if minimum_bandwidth else 0.0
            message = (
                f"rank {rank} {gpu.bdf}: {speed:g} GT/s x{width}, "
                f"~{bandwidth:.2f} GB/s payload"
            )
            if minimum and bandwidth + 1e-9 < minimum:
                reporter.fail(
                    "pcie-link",
                    f"{message}; configured minimum is {minimum:g} GB/s",
                    "Check BIOS lane allocation and card order. Reorder "
                    "R9V_VISIBLE_DEVICES if appropriate; lower the minimum "
                    "only if you accept an unqualified slower topology.",
                )
            else:
                reporter.passed("pcie-link", message, minimum_gbps=minimum)
            if expected_links:
                expected = expected_links[rank]
                speed_matches = abs(speed - expected.speed_gts) < 0.01
                width_matches = width == expected.width
                if not speed_matches or not width_matches:
                    reporter.fail(
                        "pcie-link-expectation",
                        f"rank {rank} {gpu.bdf} negotiated "
                        f"{PcieLink(speed, width).config_value()}, expected "
                        f"{expected.config_value()}",
                        "R9V cannot change a negotiated PCIe link. Check the "
                        "motherboard slot, BIOS lane allocation, riser/cable, "
                        "and competing devices. Update R9V_EXPECTED_PCIE_LINKS "
                        "only if the detected topology is intentional.",
                        expected_speed_gts=expected.speed_gts,
                        expected_width=expected.width,
                        actual_speed_gts=speed,
                        actual_width=width,
                    )
                else:
                    reporter.passed(
                        "pcie-link-expectation",
                        f"rank {rank} {gpu.bdf} matches "
                        f"{expected.config_value()}",
                        expected_speed_gts=expected.speed_gts,
                        expected_width=expected.width,
                    )
        selected.append((rank, gpu, speed, width, bandwidth))

    if not expected_bdfs and len(selected) == expected_count:
        value = ",".join(gpu.bdf for _, gpu, _, _, _ in selected)
        reporter.warn(
            "gpu-order-lock",
            "device order is detected but not locked; set "
            f"R9V_EXPECTED_GPU_BDFS={value} after confirming the ranks",
            f"Add `: \"${{R9V_EXPECTED_GPU_BDFS:={value}}}\"` to R9V_CONFIG_FILE.",
        )
    elif (
        expected_bdfs
        and len(selected) == expected_count
        and all(
            gpu.bdf == expected_bdfs[rank]
            for rank, gpu, _, _, _ in selected
        )
    ):
        reporter.passed("gpu-order-lock", "configured BDF order matches selected ranks")
    if not expected_links and len(selected) == expected_count:
        detected_links = [
            PcieLink(speed, width).config_value()
            for _, _, speed, width, _ in selected
            if speed is not None and width is not None
        ]
        if len(detected_links) == expected_count:
            value = ",".join(detected_links)
            reporter.warn(
                "pcie-link-lock",
                "negotiated PCIe links are detected but not locked; set "
                f"R9V_EXPECTED_PCIE_LINKS={value} after confirming the topology",
                f"Add `: \"${{R9V_EXPECTED_PCIE_LINKS:={value}}}\"` to "
                "R9V_CONFIG_FILE. This records the intended links; it does "
                "not change PCIe negotiation.",
            )
    return selected


def _check_host_memory(reporter: Reporter, proc_root: Path) -> None:
    values = _meminfo(proc_root)
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        reporter.warn(
            "host-memory",
            "cannot read MemTotal/MemAvailable",
            "Check that /proc is mounted and readable, then rerun the doctor.",
        )
        return
    try:
        minimum_total = _parse_bytes(
            os.environ.get("R9V_MIN_HOST_RAM_BYTES"), "R9V_MIN_HOST_RAM_BYTES"
        )
        minimum_available = _parse_bytes(
            os.environ.get("R9V_MIN_HOST_AVAILABLE_BYTES"),
            "R9V_MIN_HOST_AVAILABLE_BYTES",
        )
        reference = _parse_bytes(
            os.environ.get("R9V_REFERENCE_HOST_RAM_BYTES"),
            "R9V_REFERENCE_HOST_RAM_BYTES",
        )
    except ValueError as error:
        reporter.fail(
            "host-memory-policy",
            str(error),
            "Use non-negative integer byte counts for the named RAM setting.",
        )
        return
    message = f"total {_human_bytes(total)}, available {_human_bytes(available)}"
    if minimum_total and total < minimum_total:
        reporter.fail(
            "host-memory",
            f"{message}; total is below configured minimum",
            "Use a host with more RAM, or lower R9V_MIN_HOST_RAM_BYTES only "
            "after validating startup and PLE behavior.",
        )
    elif minimum_available and available < minimum_available:
        reporter.fail(
            "host-memory",
            f"{message}; available is below configured minimum",
            "Stop memory-heavy processes or reduce the explicit "
            "R9V_MIN_HOST_AVAILABLE_BYTES policy.",
        )
    else:
        reporter.passed("host-memory", message)
    if reference and total < reference:
        reporter.warn(
            "host-memory-reference",
            f"host is below the {_human_bytes(reference)} qualified reference; "
            "PLE page-cache misses and startup headroom may differ",
            "Keep PLE residency on SSD, enable PLE timing for qualification, "
            "and do not reduce the logical CPU-offload budget to match RAM.",
        )
    offload = os.environ.get("R9V_CPU_OFFLOAD_GB")
    if offload:
        reporter.note(
            "offload-accounting",
            f"R9V_CPU_OFFLOAD_GB={offload} is loader accounting, not a host-RAM allocation",
        )


def _flatten_lsblk(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in nodes:
        result.append(node)
        result.extend(_flatten_lsblk(node.get("children", [])))
    return result


def _check_ple_storage(reporter: Reporter) -> None:
    raw_path = os.environ.get("R9V_PLE_PATH")
    if not raw_path:
        reporter.warn(
            "ple-path",
            "R9V_PLE_PATH is not set; storage and payload checks were skipped",
            "Set R9V_PLE_PATH to the file produced by tools/prepare_ple.py, then rerun the doctor.",
        )
        return
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        reporter.fail(
            "ple-path",
            f"PLE payload is missing: {path}",
            "Correct R9V_PLE_PATH or derive the payload with the installation "
            "guide's PLE extraction step.",
        )
        return
    try:
        expected = _parse_bytes(
            os.environ.get("R9V_PLE_EXPECTED_BYTES", str(PLE_EXPECTED_BYTES)),
            "R9V_PLE_EXPECTED_BYTES",
        )
    except ValueError as error:
        reporter.fail(
            "ple-path",
            str(error),
            "Set R9V_PLE_EXPECTED_BYTES to a non-negative integer byte count.",
        )
        return
    actual = path.stat().st_size
    if expected and actual != expected:
        reporter.fail(
            "ple-payload",
            f"PLE size is {actual}, expected {expected}",
            "Delete only this derived PLE file and regenerate it from the verified target shards.",
        )
    else:
        reporter.passed("ple-payload", f"{path} is {_human_bytes(actual)}")

    mount = _run(["findmnt", "-J", "-T", str(path), "-o", "SOURCE,FSTYPE,TARGET"])
    if mount.returncode != 0:
        reporter.warn(
            "ple-storage",
            f"findmnt failed for {path}",
            "Install util-linux/findmnt or report the filesystem and physical device manually.",
        )
        return
    try:
        filesystems = json.loads(mount.stdout).get("filesystems", [])
        filesystem = filesystems[0]
    except (IndexError, KeyError, json.JSONDecodeError):
        reporter.warn(
            "ple-storage",
            f"cannot parse findmnt output for {path}",
            "Run `findmnt -T <PLE_PATH>` and include its output in the support report.",
        )
        return
    source = str(filesystem.get("source", ""))
    fstype = str(filesystem.get("fstype", "unknown"))
    if not source.startswith("/dev/"):
        reporter.warn(
            "ple-storage",
            f"PLE is on {source or 'an unknown source'} ({fstype}); block media is unknown",
            "Place the PLE on a directly discoverable non-rotating SSD or "
            "qualify this storage manually.",
        )
        return
    block = _run(
        ["lsblk", "-s", "-J", "-o", "NAME,PATH,TYPE,ROTA,TRAN", source]
    )
    if block.returncode != 0:
        reporter.warn(
            "ple-storage",
            f"lsblk failed for {source}",
            "Install util-linux/lsblk or report the backing physical device manually.",
        )
        return
    try:
        nodes = _flatten_lsblk(json.loads(block.stdout).get("blockdevices", []))
    except json.JSONDecodeError:
        reporter.warn(
            "ple-storage",
            f"cannot parse lsblk output for {source}",
            "Run `lsblk -s <device>` and include its output in the support report.",
        )
        return
    disks_by_path = {
        str(node.get("path")): node
        for node in nodes
        if node.get("type") == "disk"
    }
    disks = list(disks_by_path.values())
    media = ", ".join(
        f"{node.get('path')}:{node.get('tran') or 'unknown'}"
        for node in disks
    ) or "unknown"
    rotating = any(bool(node.get("rota")) for node in disks)
    try:
        require_nonrotational = _parse_bool(
            os.environ.get("R9V_REQUIRE_PLE_NONROTATIONAL"), True
        )
    except ValueError as error:
        reporter.fail(
            "ple-storage-policy",
            str(error),
            "Set R9V_REQUIRE_PLE_NONROTATIONAL to 0 or 1.",
        )
        return
    message = f"{path} -> {source} ({fstype}); physical media {media}"
    if rotating and require_nonrotational:
        reporter.fail(
            "ple-storage",
            f"{message}; rotating media is unsupported",
            "Move/regenerate the PLE payload on a non-rotating SSD and update R9V_PLE_PATH.",
        )
    elif not disks:
        reporter.warn(
            "ple-storage",
            f"{message}; physical media could not be resolved",
            "Resolve the backing device manually before treating SSD decode "
            "performance as qualified.",
        )
    elif not all(node.get("tran") == "nvme" for node in disks):
        reporter.warn(
            "ple-storage",
            f"{message}; SSD random-read latency is unqualified",
            "Use NVMe storage or run the offline PLE benchmark after stopping the model server.",
        )
    else:
        reporter.passed("ple-storage", message)


def _check_profile_policy(
    reporter: Reporter,
    expected_count: int,
    selected: list[tuple[int, AmdGpu, float | None, int | None, float | None]],
) -> None:
    variant = os.environ.get("R9V_TIERED_IQ_MOE_VARIANT")
    cache_slots = os.environ.get("R9V_TIERED_EXPERT_CACHE_SLOTS")
    cache_ranks_raw = os.environ.get("R9V_TIERED_EXPERT_CACHE_RANKS", "")
    cache_policy = os.environ.get("R9V_TIERED_EXPERT_CACHE_POLICY")
    mtp_tokens = os.environ.get("R9V_MTP_SPEC_TOKENS")
    ple_mode = os.environ.get("R9V_PLE_RESIDENCY_MODE")
    if not variant:
        reporter.note("profile-policy", "no tiered MoE policy applies to this profile")
        return
    try:
        slots = int(cache_slots or "0")
        cache_ranks = {int(value) for value in _csv(cache_ranks_raw)}
    except ValueError:
        reporter.fail(
            "cache-policy",
            "cache slots and ranks must be integers",
            "Set R9V_TIERED_EXPERT_CACHE_SLOTS to one integer and CACHE_RANKS "
            "to comma-separated TP ranks.",
        )
        return
    if slots < 0 or slots > 16:
        reporter.fail(
            "cache-policy",
            f"cache slots {slots} are outside supported range 0..16",
            "Choose 0..16 slots; use the published value 16 unless a "
            "different arm has been qualified.",
        )
    elif any(rank < 0 or rank >= expected_count for rank in cache_ranks):
        reporter.fail(
            "cache-policy",
            f"cache ranks {sorted(cache_ranks)} are invalid",
            f"Choose ranks from 0..{expected_count - 1}; the published TP2 placement uses rank 1.",
        )
    elif slots and cache_policy != "lru":
        reporter.warn(
            "cache-policy",
            f"cache policy {cache_policy!r} is not the qualified LRU arm",
            "Set R9V_TIERED_EXPERT_CACHE_POLICY=lru for the published configuration.",
        )
    else:
        reporter.passed(
            "cache-policy",
            f"slots={slots} ranks={sorted(cache_ranks)} policy={cache_policy}",
        )
    reporter.passed(
        "decode-policy",
        f"MoE={variant}, MTP depth={mtp_tokens}, PLE={ple_mode}",
    )
    bandwidths = [item[4] for item in selected]
    if (
        slots
        and len(bandwidths) == expected_count
        and all(value is not None for value in bandwidths)
    ):
        typed = [float(value) for value in bandwidths if value is not None]
        slowest = {index for index, value in enumerate(typed) if value == min(typed)}
        if len(slowest) == 1 and not slowest <= cache_ranks:
            reporter.warn(
                "cache-rank-topology",
                f"slowest PCIe rank {next(iter(slowest))} is not a dynamic-cache rank",
                "Move the cache to the slowest rank and use a matching "
                "manifest, or explicitly qualify the alternate placement.",
            )


def _check_manifest_budget(reporter: Reporter, expected_count: int) -> None:
    model_dir = os.environ.get("R9V_MODEL_DIR")
    if not model_dir or expected_count < 2:
        return
    relative = os.environ.get(
        "R9V_MANIFEST_REL",
        "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
    )
    path = Path(model_dir).expanduser().resolve() / relative
    if not path.is_file():
        reporter.fail(
            "expert-manifest",
            f"manifest is missing: {path}",
            "Use the manifest shipped in the model package or set "
            "R9V_MANIFEST_REL to a compatible packaged manifest.",
        )
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ranks = data["ranks"]
        hot = [int(ranks[str(rank)]["hot_count"]) for rank in range(expected_count)]
        slots = int(os.environ.get("R9V_TIERED_EXPERT_CACHE_SLOTS", "0"))
        cache_ranks = {
            int(value)
            for value in _csv(os.environ.get("R9V_TIERED_EXPERT_CACHE_RANKS"))
        }
        maximum = _csv_float(
            os.environ.get("R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK"),
            expected_count,
            "R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK",
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        reporter.fail(
            "expert-manifest",
            f"cannot validate {path}: {error}",
            "Restore the verified package manifest; do not hand-edit expert IDs or counts.",
        )
        return
    effective = [
        count + (slots if rank in cache_ranks else 0)
        for rank, count in enumerate(hot)
    ]
    if maximum and any(
        effective[rank] > maximum[rank] for rank in range(expected_count)
    ):
        reporter.fail(
            "expert-budget",
            f"static+cache experts {effective} exceed configured maxima {maximum}",
            "Reduce R9V_TIERED_EXPERT_CACHE_SLOTS or select a manifest with "
            "fewer static experts. Do not raise the maximum without VRAM "
            "qualification.",
            static=hot,
            effective=effective,
            maximum=maximum,
        )
    else:
        reporter.passed(
            "expert-budget",
            f"static experts={hot}; static+cache={effective}",
            manifest=str(path),
            maximum=maximum or None,
        )


def _check_model_package(reporter: Reporter, repo_root: Path, profile_id: str) -> None:
    model_dir = os.environ.get("R9V_MODEL_DIR")
    if not model_dir:
        reporter.warn(
            "model-package",
            "R9V_MODEL_DIR is not set; package verification skipped",
            "Pass --model-dir or export R9V_MODEL_DIR after fetching the package.",
        )
        return
    command = [
        str(repo_root / "r9v"),
        "verify",
        profile_id,
        "--model-dir",
        str(Path(model_dir).expanduser().resolve()),
    ]
    result = _run(command, timeout=120.0)
    summary = (result.stdout or result.stderr).strip().splitlines()
    message = summary[0] if summary else "package verifier returned no output"
    if result.returncode == 0:
        reporter.passed("model-package", message)
    else:
        reporter.fail(
            "model-package",
            message,
            "Run `./r9v fetch` or restore the exact published files, then rerun `./r9v verify`.",
            output="\n".join(summary[-20:]),
        )


def _docker_environment(container: str) -> tuple[str | None, dict[str, str], str | None]:
    result = _run(["docker", "inspect", container])
    if result.returncode != 0:
        return None, {}, None
    try:
        info = json.loads(result.stdout)[0]
        values = info.get("Config", {}).get("Env", [])
        env = dict(value.split("=", maxsplit=1) for value in values if "=" in value)
        state = info.get("State", {}).get("Status")
        image = info.get("Image")
        return state, env, image
    except (IndexError, KeyError, ValueError, json.JSONDecodeError):
        return None, {}, None


def _fetch_metrics(port: str) -> str | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=2.0
        ) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return None


def _check_runtime(reporter: Reporter, expected_count: int) -> None:
    container = os.environ.get("R9V_CONTAINER_NAME", "r9v-qwen38-flash-next")
    state, actual, image = _docker_environment(container)
    if state is None:
        reporter.fail(
            "runtime-container",
            f"container {container!r} does not exist",
            "Launch the profile first, or set R9V_CONTAINER_NAME to the running profile container.",
        )
        return
    if state != "running":
        reporter.fail(
            "runtime-container",
            f"container {container!r} is {state}",
            f"Inspect `docker logs {container}`, fix startup, then recreate the container.",
        )
        return
    reporter.passed("runtime-container", f"{container} is running", image=image)
    expected = {
        "HIP_VISIBLE_DEVICES": os.environ.get("R9V_VISIBLE_DEVICES", "0,1"),
        "QWEN38_TIERED_IQ_MOE_VARIANT": os.environ.get("R9V_TIERED_IQ_MOE_VARIANT"),
        "QWEN38_TIERED_PREFILL_GROUP_SIZE": os.environ.get("R9V_TIERED_PREFILL_GROUP_SIZE"),
        "QWEN38_TIERED_EXPERT_CACHE_SLOTS": os.environ.get("R9V_TIERED_EXPERT_CACHE_SLOTS"),
        "QWEN38_TIERED_EXPERT_CACHE_RANKS": os.environ.get("R9V_TIERED_EXPERT_CACHE_RANKS"),
        "QWEN38_TIERED_EXPERT_CACHE_POLICY": os.environ.get("R9V_TIERED_EXPERT_CACHE_POLICY"),
        "VLLM_PLE_RESIDENCY_MODE": os.environ.get("R9V_PLE_RESIDENCY_MODE"),
        "VLLM_PLE_WORKER_TIMING": os.environ.get("R9V_PLE_WORKER_TIMING"),
        "RADIANCE_USE_R4D": "0",
        "RADIANCE_USE_R4D_AR": "0",
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if value is not None and actual.get(key) != value
    }
    if mismatches:
        mismatch_text = "; ".join(
            f"{key}: actual={value['actual']!r}, expected={value['expected']!r}"
            for key, value in sorted(mismatches.items())
        )
        reporter.fail(
            "runtime-environment",
            f"{len(mismatches)} critical settings differ: {mismatch_text}",
            "Stop/remove the container, correct R9V_CONFIG_FILE or exported "
            "values, and relaunch; container environments cannot be repaired "
            "in place.",
            mismatches=mismatches,
        )
    else:
        reporter.passed("runtime-environment", "critical container settings match the profile")

    logs_result = _run(["docker", "logs", "--tail", "4000", container], timeout=20.0)
    logs = logs_result.stdout + logs_result.stderr
    ready_ranks = {
        int(value)
        for value in re.findall(r"Tiered GGUF experts ready on TP rank (\d+)", logs)
    }
    expected_ranks = set(range(expected_count))
    if ready_ranks == expected_ranks:
        reporter.passed("tiered-experts", f"materialized on TP ranks {sorted(ready_ranks)}")
    else:
        reporter.fail(
            "tiered-experts",
            f"startup logs show ranks {sorted(ready_ranks)}, expected {sorted(expected_ranks)}",
            "Inspect startup logs for manifest/materialization errors; verify "
            "the package, rank order, cache settings, and rebuilt image "
            "before relaunching.",
        )
    variant = os.environ.get("R9V_TIERED_IQ_MOE_VARIANT", "")
    if f"Using tiered IQ MoE exact-shape variant {variant}" in logs:
        reporter.passed("decode-kernel", f"startup selected exact variant {variant}")
    else:
        reporter.fail(
            "decode-kernel",
            f"no startup proof that variant {variant} was selected",
            "Verify the container uses the current R9V image and kernel SO, "
            "then rebuild/relaunch with "
            "R9V_TIERED_IQ_MOE_VARIANT=reuse3v2.",
        )
    if "Using tiered IQ MoE grouped-16 prefill" in logs:
        reporter.passed("prefill-kernel", "grouped-16 prefill was observed")
    else:
        reporter.warn(
            "prefill-kernel",
            "grouped-16 prefill has not been observed in recent logs",
            "Send one prompt longer than 64 tokens and rerun the runtime "
            "doctor; absence before such a request is expected.",
        )
    if os.environ.get("R9V_ENABLE_FUSED_GDN_MTP") == "1":
        marker = "Qwen3.8 TP2 fused speculative GDN HIP kernel enabled"
        if marker in logs:
            reporter.passed("gdn-mtp", "fused speculative GDN kernel was enabled")
        else:
            reporter.warn(
                "gdn-mtp",
                "fused speculative GDN enable marker is absent",
                "Verify the fused GDN SO was built into the current image and "
                "relaunch with R9V_ENABLE_FUSED_GDN_MTP=1.",
            )

    timing_lines = [line for line in logs.splitlines() if "PLE worker timing " in line]
    if os.environ.get("R9V_PLE_WORKER_TIMING") == "1":
        if timing_lines:
            reporter.passed("ple-timing", timing_lines[-1].strip())
        else:
            reporter.warn(
                "ple-timing",
                "timing is enabled but no PLE timing sample is in recent logs",
                "Send one generation request, then rerun the runtime doctor.",
            )
    else:
        reporter.note(
            "ple-timing",
            "set R9V_PLE_WORKER_TIMING=1 for one diagnostic relaunch to split SSD/PLE latency",
        )

    metrics_text = _fetch_metrics(os.environ.get("R9V_HOST_PORT", "8004"))
    if metrics_text is None:
        reporter.warn(
            "runtime-metrics",
            "metrics endpoint is not reachable",
            "Check R9V_HOST_PORT, server readiness, and the published port "
            "before rerunning runtime checks.",
        )
        return
    metrics = parse_prometheus_metrics(metrics_text)
    drafts = metrics.get("vllm:spec_decode_num_drafts_total", 0.0)
    draft_tokens = metrics.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    if drafts <= 0:
        reporter.warn(
            "mtp-metrics",
            "no speculative draft cycles have been recorded",
            "Send a generation request with more than one output token; if "
            "drafts remain zero, inspect the MTP startup configuration.",
        )
        return
    mean_length = 1.0 + accepted / drafts
    acceptance = accepted / draft_tokens if draft_tokens else 0.0
    reporter.passed(
        "mtp-metrics",
        f"drafts={drafts:.0f}, accepted={accepted:.0f}, "
        f"mean emitted length={mean_length:.3f}, acceptance={acceptance:.1%}",
        drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted,
        mean_emitted_length=mean_length,
        acceptance=acceptance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also inspect the running container, startup logs, and metrics",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero for warnings as well as failures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reporter = Reporter()
    repo_root = Path(
        os.environ.get("R9V_REPO_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    sys_root = Path(os.environ.get("R9V_SYS_ROOT", "/sys"))
    proc_root = Path(os.environ.get("R9V_PROC_ROOT", "/proc"))
    profile_id = os.environ.get(
        "R9V_PROFILE_ID", "qwen38-flash-next/ud-iq4-xs/dual-r9700-128k"
    )
    expected_count = 2 if profile_id.startswith("qwen38-flash-next/") else 1

    if shutil.which("docker") is None:
        reporter.fail(
            "docker",
            "docker is not installed",
            "Install Docker Engine and the Buildx plugin, then rerun the doctor.",
        )
    else:
        docker_info = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
        if docker_info.returncode == 0:
            reporter.passed(
                "docker", f"daemon is available (server {docker_info.stdout.strip()})"
            )
        else:
            reporter.fail(
                "docker",
                f"daemon is unavailable: {docker_info.stderr.strip()}",
                "Start Docker and grant this user daemon access until `docker info` succeeds.",
            )
    for path in (Path("/dev/kfd"), Path("/dev/dri")):
        if path.exists():
            reporter.passed("device-node", str(path))
        else:
            reporter.fail(
                "device-node",
                f"missing {path}",
                "Load/install the ROCm amdgpu/KFD driver and verify device permissions.",
            )

    if profile_id.startswith("qwen38-flash-next/"):
        for relative in (
            "vendor/vllm/docker/Dockerfile.r9v_rocm714",
            "vendor/vllm-gguf-plugin/setup.py",
            "kernels/r9v-gfx1201/README.md",
        ):
            path = repo_root / relative
            if path.is_file():
                reporter.passed("runtime-source", relative)
            else:
                reporter.fail(
                    "runtime-source",
                    f"missing {relative}",
                    "Run `git submodule update --init --recursive` from the R9V checkout.",
                )

    selected = _selected_gpus(reporter, expected_count, sys_root)
    _check_host_memory(reporter, proc_root)
    _check_profile_policy(reporter, expected_count, selected)
    _check_manifest_budget(reporter, expected_count)
    _check_ple_storage(reporter)
    _check_model_package(reporter, repo_root, profile_id)
    if args.runtime:
        _check_runtime(reporter, expected_count)

    counts = reporter.counts()
    strict = args.strict
    try:
        strict = strict or _parse_bool(os.environ.get("R9V_DOCTOR_STRICT"), False)
    except ValueError as error:
        reporter.fail(
            "doctor-policy",
            f"R9V_DOCTOR_STRICT: {error}",
            "Set R9V_DOCTOR_STRICT to 0 or 1.",
        )
        counts = reporter.counts()
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "r9v.doctor.v1",
                    "profile": profile_id,
                    "checks": [asdict(check) for check in reporter.checks],
                    "summary": counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for check in reporter.checks:
            print(f"{check.status:4} {check.name}: {check.message}")
            if check.remediation:
                print(f"     FIX: {check.remediation}")
        print(
            "SUMMARY "
            + " ".join(f"{name}={counts[name]}" for name in ("PASS", "WARN", "FAIL", "NOTE"))
        )
    if counts["FAIL"]:
        return 1
    if strict and counts["WARN"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
