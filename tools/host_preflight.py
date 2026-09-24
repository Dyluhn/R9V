# SPDX-License-Identifier: Apache-2.0
"""Resource and evidence checks shared by the Qwen Flash Next doctor."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import shutil
from pathlib import Path

try:
    from tools import disk_space
    from tools.expert_budget import headroom_bytes
    from tools.image_bundle import ImageBundleError, read_manifest
    from tools.runtime_overlays import CED_HEADROOM_FIX
except ModuleNotFoundError:
    import disk_space
    from expert_budget import headroom_bytes
    from image_bundle import ImageBundleError, read_manifest
    from runtime_overlays import CED_HEADROOM_FIX

GIB = 1024**3
MIB = 1024**2


def check_resources(
    reporter, selected, repo_root: Path, sys_root: Path, proc_root: Path, runtime: bool
) -> None:
    reporter.note("host-kernel", platform.release())
    driver = sys_root / "module/amdgpu/version"
    if driver.exists():
        reporter.note("amdgpu-version", driver.read_text().strip())
    try:
        margins = headroom_bytes(
            os.environ.get("R9V_MIN_FREE_VRAM_GIB_BY_RANK", "3,3"), 2
        )
    except ValueError as error:
        reporter.fail(
            "vram-headroom-policy",
            str(error),
            "Set one non-negative GiB target per TP rank.",
        )
        margins = None
    # Sysfs values follow physical BDFs, not amd-smi display indices.
    for rank, gpu, *_ in selected:
        pci = sys_root / "bus/pci/devices" / gpu.bdf
        try:
            total = int((pci / "mem_info_vram_total").read_text())
            used = int((pci / "mem_info_vram_used").read_text())
            if total < 31 * 1024**3:
                reporter.fail(
                    "gpu-vram",
                    f"rank {rank} {gpu.bdf}: {total / 1024**3:.2f} GiB; dual-R9700 profile needs 32 GiB cards",
                    "Select the two R9700 devices and verify the BDF order.",
                )
            elif margins is not None and (
                margins[rank] > total or total - used < margins[rank]
            ):
                reporter.fail(
                    "gpu-headroom",
                    f"rank {rank} {gpu.bdf}: {(total - used) / 1024**3:.2f} GiB free, requested {margins[rank] / 1024**3:g} GiB",
                    "Reduce hot-expert residency or other GPU use. This check does not resize the manifest or promise future peak headroom.",
                )
            else:
                reporter.passed(
                    "gpu-vram",
                    f"rank {rank}: {total / 1024**3:.2f} GiB total, {(total - used) / 1024**3:.2f} GiB free now; snapshot only",
                )
        except (OSError, ValueError):
            reporter.warn(
                "gpu-vram",
                f"rank {rank}: VRAM capacity/use unavailable",
                "Inspect amd-smi memory telemetry; device presence alone cannot certify capacity.",
            )
        for hop in (pci.resolve(), *pci.resolve().parents):
            for kind in ("aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"):
                try:
                    text = (hop / kind).read_text()
                except OSError:
                    continue
                if any(
                    parts[-1].isdigit() and int(parts[-1]) > 0
                    for line in text.splitlines()
                    if (parts := line.split())
                ):
                    reporter.warn(
                        "pcie-error-history",
                        f"{hop.name}: nonzero {kind} counters",
                        "Capture a soak and compare counter deltas with failure times. These cumulative counts alone do not prove a current fault.",
                    )
    cache = Path(
        os.environ.get("R9V_CACHE_DIR", str(repo_root / ".cache"))
    ).expanduser()
    while not cache.exists() and cache != cache.parent:
        cache = cache.parent
    try:
        free = shutil.disk_usage(cache).free
        if not os.access(cache, os.W_OK | os.X_OK):
            reporter.fail(
                "cache-storage",
                f"cache ancestor {cache} is not writable",
                "Set R9V_CACHE_DIR to writable storage.",
            )
        elif free < 1024**3:
            reporter.warn(
                "cache-storage",
                f"only {free / 1024**3:.2f} GiB free at {cache}",
                "Free cache space before runtime compilation; keep evidence on storage with headroom.",
            )
        else:
            reporter.passed(
                "cache-storage", f"{free / 1024**3:.1f} GiB free at {cache}"
            )
    except OSError as error:
        reporter.fail("cache-storage", str(error), "Check the cache filesystem.")
    if not (Path("/var/log/journal").is_dir()):
        reporter.warn(
            "persistent-kernel-logs",
            "persistent journal directory is absent",
            "Enable persistent journald storage on the host to recover GPU-reset/OOM evidence after reboot; run support before removing the container.",
        )
    for name in ("R9V_MIN_HOST_RAM_BYTES", "R9V_MIN_HOST_AVAILABLE_BYTES"):
        if os.environ.get(name, "0") == "0":
            reporter.warn(
                "memory-qualification",
                f"{name} is unset; no measured host RAM minimum is enforced",
                "Record peak host/cgroup memory during qualification and set an explicit minimum with startup headroom. The logical offload GB value is not a RAM requirement.",
            )


def check_container_limits(reporter, run, container: str) -> None:
    probe = run(["docker", "inspect", "--format", "{{json .HostConfig}}", container])
    try:
        config = json.loads(probe.stdout) if probe.returncode == 0 else None
        if not isinstance(config, dict):
            raise ValueError("container settings unavailable")
    except (ValueError, TypeError):
        reporter.warn(
            "runtime-limits",
            "cannot inspect container resource limits",
            "Check Docker access.",
        )
        return
    logs = config.get("LogConfig", {})
    if logs.get("Type") not in {"json-file", "local", "journald"}:
        reporter.fail(
            "runtime-log-retention",
            f"logging driver {logs.get('Type')!r} may not retain locally retrievable logs",
            "Recreate with the current launcher after saving existing evidence.",
        )
    elif logs.get("Type") in {"json-file", "local"} and not all(
        logs.get("Config", {}).get(key) for key in ("max-size", "max-file")
    ):
        reporter.warn(
            "runtime-log-retention",
            "log rotation bounds are not explicit",
            "Use the current launcher for bounded retained logs.",
        )
    else:
        reporter.passed("runtime-log-retention", f"local logging: {logs.get('Type')}")
    if config.get("AutoRemove"):
        reporter.fail(
            "runtime-auto-remove",
            "container is removed on exit, losing evidence",
            "Use the launcher without --rm.",
        )
    for setting in ("Memory", "MemorySwap", "PidsLimit"):
        value = config.get(setting)
        if value and value > 0:
            reporter.warn(
                "runtime-resource-cap",
                f"{setting}={value}",
                "Compare this cap with the captured cgroup peaks/events; sufficient host RAM does not override a container cap.",
            )
    memlock = next(
        (
            limit
            for limit in config.get("Ulimits") or []
            if limit.get("Name") == "memlock"
        ),
        {},
    )
    if memlock.get("Soft") != -1 or memlock.get("Hard") != -1:
        reporter.warn(
            "runtime-memlock",
            "memlock policy is inherited or limited",
            "Inspect the daemon memlock limits and serving-worker pinned-UVA probes. "
            "Rootless Docker inherits its daemon hard limit; forcing "
            "--ulimit memlock=-1:-1 can prevent startup.",
        )


def check_runtime_arguments(reporter, run, container: str) -> None:
    result = run(["docker", "inspect", "--format", "{{json .Config.Cmd}}", container])
    try:
        command = json.loads(result.stdout) if result.returncode == 0 else None
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise ValueError("command unavailable")
    except (ValueError, TypeError):
        reporter.warn(
            "runtime-arguments",
            "cannot inspect the server launch arguments",
            "Verify the container was created by the selected profile.",
        )
        return
    expected = {
        "--max-model-len": "R9V_MAX_MODEL_LEN",
        "--max-num-seqs": "R9V_MAX_NUM_SEQS",
        "--max-num-batched-tokens": "R9V_MAX_NUM_BATCHED_TOKENS",
        "--kv-cache-memory-bytes": "R9V_KV_CACHE_MEMORY_BYTES",
        "--tensor-parallel-size": "R9V_TENSOR_PARALLEL_SIZE",
    }
    mismatches = {}
    for flag, key in expected.items():
        value = os.environ.get(key)
        if value is None:
            continue
        actual = [
            command[index + 1]
            for index, token in enumerate(command[:-1])
            if token == flag
        ]
        actual += [
            token.split("=", 1)[1] for token in command if token.startswith(flag + "=")
        ]
        if actual != [value]:
            mismatches[flag] = {"expected": value, "actual": actual}
    if mismatches:
        reporter.fail(
            "runtime-arguments",
            "live context, concurrency, prefill, KV, or TP arguments differ from this config",
            "Rerun doctor with the config used to launch, or recreate the container with the intended settings after saving evidence.",
            mismatches=mismatches,
        )
    else:
        reporter.note(
            "runtime-arguments",
            "configured context/concurrency/prefill/KV/TP flags match container launch arguments; this does not inspect worker-internal state",
        )


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _fdinfo_bytes(value: str) -> int:
    """A DRM fdinfo memory value such as '33828 KiB' in bytes."""
    number, _, unit = value.strip().partition(" ")
    scale = {"": 1, "KiB": 1024, "MiB": MIB, "GiB": GIB}.get(unit.strip())
    return int(number) * scale if number.isdigit() and scale else 0


def gpu_process_vram(sys_root: Path, proc_root: Path, gpu_ids: dict[str, int | None]) -> dict[str, dict[int, int]]:
    """VRAM bytes each process holds on each GPU BDF: compute memory from the KFD
    per-process sysfs files, graphics memory (desktop, browser) from DRM fdinfo.
    A compute process appears in both; the larger reading counts. Processes this
    user cannot inspect are missing."""
    usage: dict[str, dict[int, int]] = {bdf: {} for bdf in gpu_ids}
    for process in (sys_root / "class/kfd/kfd/proc").glob("*"):
        for bdf, gpu_id in gpu_ids.items():
            value = _read_int(process / f"vram_{gpu_id}") if process.name.isdigit() and gpu_id else None
            if value:
                usage[bdf][int(process.name)] = value
    for process in proc_root.glob("*"):
        if not process.name.isdigit():
            continue
        clients: dict[tuple[str, str], int] = {}
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                if not os.readlink(descriptor).startswith("/dev/dri/"):
                    continue
                lines = (process / "fdinfo" / descriptor.name).read_text().splitlines()
            except OSError:
                continue
            fields = {key.strip(): value.strip() for key, _, value in (line.partition(":") for line in lines)}
            bdf = fields.get("drm-pdev")
            if bdf in usage:
                size = _fdinfo_bytes(fields.get("drm-total-vram") or fields.get("drm-memory-vram") or "")
                clients[(bdf, fields.get("drm-client-id", descriptor.name))] = size
        pid = int(process.name)
        for bdf in usage:
            size = sum(value for (client_bdf, _), value in clients.items() if client_bdf == bdf)
            if size > usage[bdf].get(pid, 0):
                usage[bdf][pid] = size
    return usage


def _process_name(proc_root: Path, pid: int) -> str:
    try:
        return (proc_root / str(pid) / "comm").read_text().strip() or "?"
    except OSError:
        return "?"


def check_vram_other_processes(reporter, selected, gpu_ids, sys_root: Path, proc_root: Path, need) -> None:
    """Before start: name the processes holding VRAM on the selected GPUs. A warning,
    unless the VRAM they leave free is below what R9V needs before launch (need, bytes
    per rank, or None when unknown without a model directory), which fails."""
    usage = gpu_process_vram(sys_root, proc_root, {gpu.bdf: gpu_ids.get(gpu.bdf) for _, gpu, *_ in selected})
    busy = False
    for rank, gpu, *_ in selected:
        holders = sorted(((size, pid) for pid, size in usage[gpu.bdf].items()
                          if size >= MIB and pid != os.getpid()), reverse=True)
        if not holders:
            continue
        busy = True
        pci = sys_root / "bus/pci/devices" / gpu.bdf
        total, used = _read_int(pci / "mem_info_vram_total"), _read_int(pci / "mem_info_vram_used")
        free = total - used if total is not None and used is not None else None
        named = ", ".join(f"{_process_name(proc_root, pid)} (pid {pid}) {size / GIB:.2f} GiB"
                          for size, pid in holders[:5])
        if len(holders) > 5:
            named += f" and {len(holders) - 5} more"
        message = (f"rank {rank} {gpu.bdf}: other processes hold {sum(size for size, _ in holders) / GIB:.2f} "
                   f"GiB: {named}")
        if free is not None:
            message += f"; {free / GIB:.2f} GiB free now"
        if free is not None and need and free < need[rank]:
            reporter.fail(
                "vram-other-processes",
                f"{message}, below the {need[rank] / GIB:.2f} GiB R9V needs free before launch",
                "Close the apps holding VRAM on this GPU before start, including any R9V or other "
                "model server that is still running."
                + (f" CED is on: {CED_HEADROOM_FIX}." if os.environ.get("R9V_CED") == "on" else ""),
            )
        else:
            if need is None:
                message += "; pass --model-dir to compare that with what R9V needs"
            reporter.warn(
                "vram-other-processes",
                message,
                "Close apps using this GPU (browsers, games, another model server) before start if you "
                "can; R9V's free-VRAM target has to hold while they keep running.",
            )
    if selected and not busy:
        reporter.passed("vram-other-processes", "no other process holds VRAM on the selected GPUs")


def check_api_exposure(reporter, uncensored: bool) -> None:
    """The API has no authentication: say loudly when it is published beyond this machine."""
    bind = os.environ.get("R9V_HOST_BIND", "127.0.0.1")
    port = os.environ.get("R9V_HOST_PORT", "8004")
    try:
        address = ipaddress.ip_address(bind)
    except ValueError:
        reporter.fail(
            "api-exposure",
            f"R9V_HOST_BIND={bind!r} is not an IPv4 or IPv6 address; the launcher refuses it",
            "Set R9V_HOST_BIND to 127.0.0.1 (this machine only) or to one interface's address, then rerun setup.",
        )
        return
    if address.is_loopback:
        reporter.passed("api-exposure", f"the API is published on {bind}:{port}, this machine only")
        return
    where = "every interface" if address.is_unspecified else f"{bind}"
    message = (f"THE API IS OPEN TO THE NETWORK: published on {where}, port {port}, with no "
               "authentication; anyone who can reach this machine can use it")
    fix = "Put authentication in front of the port, or set R9V_HOST_BIND=127.0.0.1 and rerun setup."
    if uncensored:
        message += (". This model's refusals were removed: it will help anyone who reaches the port "
                    "with harmful requests")
        fix = ("Do not expose this model without authentication and moderation in front of it; "
               "otherwise set R9V_HOST_BIND=127.0.0.1 and rerun setup.")
    reporter.warn("api-exposure", message, fix)


def check_disk_space(reporter, repo_root: Path, profile, run) -> None:
    """Before fetch and setup: the free space the package, PLE table, image bundle
    and compile cache still need, per filesystem."""
    model = os.environ.get("R9V_MODEL_DIR")
    if not model or not profile:
        reporter.note("disk-space", "no model directory selected; pass --model-dir to check the space fetch and setup need")
        return
    model_dir = Path(model).expanduser().resolve()
    data_dir = Path(os.environ.get("R9V_DATA_DIR") or model_dir / "r9v-data").expanduser()
    ple = Path(os.environ.get("R9V_PLE_PATH") or data_dir / disk_space.PLE_NAME).expanduser()
    cache = Path(os.environ.get("R9V_CACHE_DIR") or data_dir / "cache").expanduser()
    reuse = os.environ.get("R9V_REUSE_FROM")
    fix = ("Free space on the named filesystem or choose directories with room: --model-dir, "
           "--data-dir and --ple-path for setup, R9V_CACHE_DIR for the compile cache.")
    try:
        package = json.loads((repo_root / profile["descriptors"]["model_package"]).read_text())
        distribution = profile.get("distribution", {})
        bundle = docker_root = None
        if distribution.get("image_bundle"):
            loaded = run(["docker", "image", "inspect", "--format", "{{.Id}}", distribution["image_id"]])
            if loaded.returncode != 0:
                bundle = read_manifest(repo_root / distribution["image_bundle"])
                info = run(["docker", "info", "--format", "{{.DockerRootDir}}"])
                docker_root = Path(info.stdout.strip()) if info.returncode == 0 and info.stdout.strip() else None
        groups = disk_space.by_filesystem(disk_space.install_needs(
            package, model_dir, data_dir, ple, cache, bundle=bundle, docker_root=docker_root,
            reuse_dir=Path(reuse).expanduser().resolve() if reuse else None))
    except (OSError, KeyError, TypeError, ValueError, ImageBundleError) as error:
        reporter.fail("disk-space", f"cannot estimate the space the installation needs: {error}", fix)
        return
    short = [group for group in groups if not group["fits"]]
    if short:
        reporter.fail("disk-space", "not enough free space: " + "; ".join(map(disk_space.describe, short)), fix)
    elif groups:
        reporter.passed("disk-space", "; ".join(map(disk_space.describe, groups)))
    else:
        reporter.passed("disk-space", "package, PLE table, image and compile cache are already in place")
