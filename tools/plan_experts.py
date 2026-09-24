#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Choose the largest ranked expert prefixes within a calibrated memory budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

try:
    from tools.expert_budget import (
        cost_catalog,
        expert_memory,
        headroom_bytes,
        validate_cost_contract,
        validate_manifest,
    )
    from tools.trim_experts import trim
except ModuleNotFoundError:
    from expert_budget import (
        cost_catalog,
        expert_memory,
        headroom_bytes,
        validate_cost_contract,
        validate_manifest,
    )
    from trim_experts import trim

GIB = 2**30
# Storage and observability do not alter the workload. All remaining effective
# settings participate, including unknown future runtime switches.
IGNORED = {
    "R9V_OBSERVABILITY_RUN_ID",
    "R9V_OBSERVABILITY_TARGET",
    "R9V_CONTAINER_NAME",
    "R9V_STATE_DIR",
    "R9V_HEADROOM_SELECTION",
    "R9V_HOST_BIND",
    "R9V_HOST_PORT",
    "R9V_CONFIG_FILE",
    "R9V_PROFILE",
    "R9V_PROFILE_ID",
    "R9V_PROFILE_ROOT",
    "R9V_REPO_ROOT",
    "R9V_CACHE_DIR",
    "R9V_LOG_DIR",
    "R9V_PREFLIGHT",
    "R9V_RUNTIME_PREBUILT",
    "R9V_EXPERT_MANIFEST_PATH",
    "R9V_PLACEMENT_PLAN",
    "R9V_CALIBRATION_PATH",
    "R9V_MIN_FREE_VRAM_GIB_BY_RANK",
    "R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK",
    "R9V_PROFILER_DIR",
    "R9V_CACHE_NAMESPACE",
    "R9V_CAPTURE_AUTO",
    "R9V_EXPERT_CATALOG_PATH",
    "R9V_MEMORY_SEED_PATH",
    "R9V_EXPECTED_PCIE_LINKS",
    "R9V_MIN_PCIE_BANDWIDTH_GBPS",
    "R9V_REFERENCE_PCIE_BANDWIDTH_GBPS",
    "R9V_MIN_HOST_RAM_BYTES",
    "R9V_MIN_HOST_AVAILABLE_BYTES",
    "R9V_REFERENCE_HOST_RAM_BYTES",
    "R9V_DOCTOR_STRICT",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_contract(config, source_sha256, devices, driver):
    settings = {
        k: v for k, v in config.items() if k.startswith("R9V_") and k not in IGNORED
    }
    for key in list(settings):
        if (key.startswith("R9V_DEV_") or key == "R9V_RUNTIME_DESCRIPTOR") and settings[key]:
            settings[key] = hashlib.sha256(Path(settings[key]).read_bytes()).hexdigest()
    return {
        "settings": settings,
        "source_sha256": source_sha256,
        "devices": devices,
        "driver": driver,
    }


def read_runtime(config):
    location = config.get('R9V_RUNTIME_DESCRIPTOR')
    return json.loads(Path(location).read_text()) if location else None


def live_contract(config, source_sha256):
    config = dict(config)
    config["R9V_IMAGE"] = subprocess.check_output(
        ["docker", "image", "inspect", config["R9V_IMAGE"], "--format", "{{.Id}}"],
        text=True,
        timeout=15,
    ).strip()
    devices = []
    for bdf in config["R9V_EXPECTED_GPU_BDFS"].split(","):
        base = Path("/sys/bus/pci/devices") / bdf
        devices.append(
            {"bdf": bdf, "total_bytes": int((base / "mem_info_vram_total").read_text())}
        )
    driver = {"kernel": platform.release()}
    for key in ("version", "srcversion"):
        path = Path("/sys/module/amdgpu") / key
        driver[key] = path.read_text().strip() if path.exists() else None
    return runtime_contract(config, source_sha256, devices, driver)


def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer byte count")
    return value


def plan(
    source, calibration, contract, targets, host_available, *, allow_reference=False,
    allow_capacity_baseline=False, runtime=None
):
    capacity_baseline = calibration.get("schema") == "r9v.memory-capacity-baseline.v1"
    if capacity_baseline:
        if not allow_capacity_baseline:
            raise ValueError("Capacity baseline requires explicit planning opt-in")
        if (
            calibration.get("capacity_baseline_passed") is not True
            or calibration.get("workload_passed") is not False
            or calibration.get("qualification_status") != "failed_headroom_only"
        ):
            raise ValueError("Invalid capacity baseline qualification status")
        measured_hot_counts = calibration.get("measured_hot_counts")
        if (
            not isinstance(measured_hot_counts, list)
            or len(measured_hot_counts) != 2
            or any(type(n) is not int or not 1 <= n <= 512 for n in measured_hot_counts)
        ):
            raise ValueError("Capacity baseline needs measured hot counts for both ranks")
    elif calibration.get("schema") != "r9v.memory-calibration.v1":
        raise ValueError("unsupported memory calibration schema")
    measured_contract = calibration.get("contract")
    ranking = source.get("ranking", {})
    if not isinstance(ranking, dict):
        raise ValueError("catalog ranking must be an object")  # noqa: TRY004 - public validation contract
    inputs = ranking.get("inputs", [])
    catalog_rebase = (
        isinstance(measured_contract, dict)
        and {k: v for k, v in measured_contract.items() if k != "source_sha256"}
        == {k: v for k, v in contract.items() if k != "source_sha256"}
        and ranking.get("heldout_non_regression") is True
        and ranking.get("packed_cost_catalog") == cost_catalog(source)
        and isinstance(inputs, list)
        and len(inputs) == 3
        and isinstance(inputs[0], dict)
        and inputs[0].get("sha256") == measured_contract.get("source_sha256")
    )
    if measured_contract != contract and not catalog_rebase:
        raise ValueError(
            "Calibration is stale for this model/runtime/driver/device/workload; recalibrate"
        )
    if capacity_baseline and catalog_rebase:
        settings = contract["settings"]
        binding = ranking.get("capture_binding", {})
        runtime_hash = hashlib.sha256(
            (settings.get("R9V_RUNTIME_DESCRIPTOR", "") + "\0"
             + settings.get("R9V_IMAGE", "")).encode()
        ).hexdigest()
        if (
            ranking.get("binding_verified") is not True
            or binding.get("runtime_hash") != runtime_hash
            or binding.get("model_hash") != settings.get("R9V_MODEL_PACKAGE_SHA256")
            or binding.get("model_package") != settings.get("R9V_MODEL_PACKAGE")
        ):
            raise ValueError("Capacity catalog lacks matching model/runtime capture binding")
    reference = (
        allow_reference
        and calibration.get("reference_workload_passed") is True
        and calibration.get("scope")
        == "reference estimate awaiting local qualification"
    )
    if not calibration.get("evidence") or not (
        calibration.get("workload_passed") or reference or capacity_baseline
    ):
        raise ValueError("Calibration needs passing workload evidence")
    if len(targets) != 2:
        raise ValueError("two per-rank headroom targets are required")
    costs = cost_catalog(source)
    validate_cost_contract(costs, contract['settings'])
    old = validate_manifest(source)
    settings = contract["settings"]
    slots = int(settings.get("R9V_TIERED_EXPERT_CACHE_SLOTS", "16"))
    ranks = {
        int(v)
        for v in settings.get("R9V_TIERED_EXPERT_CACHE_RANKS", "1").split(",")
        if v
    }
    asynchronous = settings.get("R9V_TIERED_EXPERT_CACHE_ASYNC", "0") == "1"
    minimum, _ = trim(source, [1, 1], contract["source_sha256"])
    minimum_memory = expert_memory(minimum, slots, ranks, asynchronous, runtime=runtime)
    counts, budgets, problems = [], [], []
    measurements = calibration.get("ranks")
    if not isinstance(measurements, list) or len(measurements) != 2:
        raise ValueError("calibration must contain two rank envelopes")
    for rank in range(2):
        measured = measurements[rank]
        capacity = nonnegative(contract["devices"][rank]["total_bytes"], "capacity")
        mandatory = nonnegative(measured["non_expert_peak_bytes"], "non-expert peak")
        external = nonnegative(
            measured["external_allowance_bytes"], "external allowance"
        )
        margin = nonnegative(measured["transient_margin_bytes"], "transient margin")
        target = nonnegative(targets[rank], "headroom")
        per_expert = minimum_memory[rank]["static_packed_bytes"]
        cache = minimum_memory[rank]["cache_packed_bytes"]
        largest = capacity - mandatory - external - margin - cache - per_expert
        budget = capacity - target - mandatory - external - margin - cache
        limit = min(old[rank])
        if capacity_baseline:
            limit = min(limit, measured_hot_counts[rank])
        count = min(limit, budget // per_expert)
        if count < 1:
            problems.append(
                f"Rank {rank}: requested {target / GIB:.3f} GiB headroom does not fit; maximum {max(0, largest) / GIB:.3f} GiB; shortfall {max(0, target - largest) / GIB:.3f} GiB. Explicitly reduce workload or external GPU use."
            )
        counts.append(count)
        budgets.append(
            {
                "rank": rank,
                "target_free_bytes": target,
                "estimated_free_bytes": capacity
                - mandatory
                - external
                - margin
                - cache
                - count * per_expert,
                "non_expert_peak_bytes": mandatory,
                "external_allowance_bytes": external,
                "transient_margin_bytes": margin,
                "cache_bytes": cache,
            }
        )
    if problems:
        raise ValueError("; ".join(problems))
    manifest, provenance = trim(source, counts, contract["source_sha256"])
    memory = expert_memory(manifest, slots, ranks, asynchronous, runtime=runtime)
    reference_cold = nonnegative(
        calibration["reference_cold_bytes"], "reference cold bytes"
    )
    extra_cold = sum(row["cold_pinned_packed_bytes"] for row in memory) - reference_cold
    startup = nonnegative(
        calibration["startup_required_available_bytes"], "startup RAM"
    ) + max(0, extra_cold)
    reserve = nonnegative(calibration["host_reserve_bytes"], "host reserve")
    if nonnegative(host_available, "available host RAM") < startup + reserve:
        raise ValueError(
            f"Host RAM insufficient: need {(startup + reserve) / GIB:.2f} GiB available for loading plus reserve; have {host_available / GIB:.2f} GiB. Fewer hot experts increase pinned RAM."
        )
    result = {
        "schema": "r9v.expert-plan.v1",
        "contract": contract,
        "calibration_sha256": digest(calibration),
        "hot_counts": counts,
        "ranks": budgets,
        "expert_memory": memory,
        "startup_required_available_bytes": startup + reserve,
        "manifest_sha256": digest(manifest),
        "provenance": provenance,
        "qualification": "requires admission and workload validation after loading",
    }
    result["reference_estimate"] = reference
    result["capacity_baseline_estimate"] = capacity_baseline
    if capacity_baseline:
        result["source_qualification_status"] = "failed_headroom_only"
        result["qualification"] = "capacity estimate only; requires fresh full workload and headroom qualification"
    if measured_contract != contract:
        result["catalog_calibration_source_sha256"] = measured_contract["source_sha256"]
    return manifest, result


def check_plan(value, manifest, current_contract, host_available, physical_free):
    if (
        value.get("schema") != "r9v.expert-plan.v1"
        or value.get("contract") != current_contract
    ):
        previous = value.get("contract", {})
        changed = sorted(k for k in set(previous) | set(current_contract)
                         if previous.get(k) != current_contract.get(k))
        before, after = previous.get("settings", {}), current_contract.get("settings", {})
        setting_keys = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        detail = ", ".join(changed + setting_keys)
        raise ValueError(
            "Placement plan is stale; regenerate for the current runtime/workload"
            + ("; changed fields: " + detail if detail else "")
        )
    if value.get("manifest_sha256") != digest(manifest):
        raise ValueError("Placement contents changed after planning")
    if host_available < value["startup_required_available_bytes"]:
        raise ValueError(
            "Available host RAM fell below the planned startup requirement"
        )
    for rank, row in enumerate(value["ranks"]):
        external = (
            current_contract["devices"][rank]["total_bytes"] - physical_free[rank]
        )
        if external > row["external_allowance_bytes"]:
            raise ValueError(
                f"Rank {rank}: external VRAM use exceeds the calibrated allowance"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new directory for immutable plan and manifest",
    )
    parser.add_argument(
        "--headroom", default=os.environ.get("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "3,3")
    )
    parser.add_argument(
        "--capacity-baseline", action="store_true",
        help="explicitly plan a smaller candidate from a failed-headroom capacity measurement; does not qualify it",
    )
    args = parser.parse_args(argv)
    try:
        raw = args.source.read_bytes()
        contract = live_contract(os.environ, hashlib.sha256(raw).hexdigest())
        available = (
            int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        manifest, result = plan(
            json.loads(raw),
            json.loads(args.calibration.read_text()),
            contract,
            headroom_bytes(args.headroom, 2),
            available,
            runtime=read_runtime(os.environ),
            allow_capacity_baseline=args.capacity_baseline,
        )
        result["source_path"] = str(args.source.resolve())
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, value in [("manifest.json", manifest), ("plan.json", result)]:
            with (args.output / name).open("x") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        print(json.dumps(result, indent=2))
        print(f"R9V_EXPERT_MANIFEST_PATH={args.output.resolve() / 'manifest.json'}")
        print(f"R9V_PLACEMENT_PLAN={args.output.resolve() / 'plan.json'}")
        print(
            "R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK="
            + ",".join(
                str(n + row["cache_physical_slots"])
                for n, row in zip(result["hot_counts"], result["expert_memory"])
            )
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Placement not generated: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
