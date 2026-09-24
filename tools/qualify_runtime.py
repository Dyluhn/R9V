#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a frozen sequence of serving comparisons with a hard cleanup reserve.

Protocol: {config: {R9V_...: value}, prompt: string, arms: [{name, env}],
warmups: 2, trials: 3, max_tokens: 256, seconds: 2700}.
Every arm starts a fresh container. Never reuse an output directory or resume
half an arm: a failed comparison is evidence, not a benchmark result.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:
    from tools.capture_runtime import (
        cgroup_snapshot,
        scheduler_snapshot,
        worker_snapshot,
    )
except ModuleNotFoundError:
    from capture_runtime import (
        cgroup_snapshot,
        scheduler_snapshot,
        worker_snapshot,
    )

try:
    from tools.early_capture import capture as early_capture
    from tools.host_pressure import exhausted as host_exhausted
    from tools.host_pressure import snapshot as host_pressure_snapshot
    from tools.observability import DriverSampler, Reporter
except ModuleNotFoundError:
    from early_capture import capture as early_capture
    from host_pressure import exhausted as host_exhausted
    from host_pressure import snapshot as host_pressure_snapshot
    from observability import DriverSampler, Reporter

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


def validate(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    if not isinstance(protocol.get("prompt"), str) or not protocol["prompt"]:
        raise ValueError("a nonempty reference prompt is required")
    for key, default, low, high in [
        ("seconds", 2700, 180, 2700),
        ("warmups", 2, 1, 20),
        ("trials", 3, 3, 20),
        ("max_tokens", 256, 2, 4096),
        ("startup_seconds", 900, 30, 1800),
        ("workload_request_timeout", 1200, 5, 1200),
    ]:
        value = protocol.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be {low}..{high}")
        protocol[key] = value
    arms = protocol.get("arms")
    context = protocol.get("workload_context")
    if context is not None and (
        type(context) is not int or not 512 <= context <= 131072
    ):
        raise ValueError(
            "workload_context must be 512..131072; partial probes do not qualify the full envelope"
        )
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty list")
    names = set()
    for arm in arms:
        name = arm.get("name", "") if isinstance(arm, dict) else ""
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,35}", name) or name in names:
            raise ValueError("arm names must be unique lowercase container-safe names")
        names.add(name)
        if "workload" in arm and type(arm["workload"]) is not bool:
            raise ValueError("arm workload override must be boolean")
        if arm.get("route_split") not in (None, "train", "holdout"):
            raise ValueError("route_split must be train or holdout")
        if "route_collect" in arm and type(arm["route_collect"]) is not bool:
            raise ValueError("route_collect must be boolean")
        if "route_limit" in arm and (
            type(arm["route_limit"]) is not int or not 1 <= arm["route_limit"] <= 11
        ):
            raise ValueError("route_limit must be 1..11")
        if not arm.get("route_split") and any(
            key in arm for key in ("route_collect", "route_limit")
        ):
            raise ValueError("route_collect and route_limit require route_split")
    for env in [protocol.get("config"), *(a.get("env", {}) for a in arms)]:
        if not isinstance(env, dict) or any(
            not k.startswith("R9V_") or not isinstance(v, str) for k, v in env.items()
        ):
            raise ValueError("config and arm env must be R9V string mappings")
        if env.get("R9V_CONFIG_FILE"):
            raise ValueError(
                "freeze effective R9V settings in the protocol, not a mutable shell config"
            )
    return protocol


def save(path, value):
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def memory_snapshot(bdfs):
    result = {"time": time.time(), "gpus": []}
    result["host_available_bytes"] = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    result["pressure"] = {
        key: Path("/proc/pressure", key).read_text() for key in ("cpu", "memory", "io")
    }
    result["host_pressure"] = host_pressure_snapshot()
    return result


def container_created_after_launch(name, expected_image, launch_started_at, launch_attempted):
    """Claim cleanup only for a newly-created container matching this launch."""
    if not launch_attempted or launch_started_at is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "inspect", name], capture_output=True, text=True, timeout=15, check=False
        )
        if result.returncode != 0:
            return False
        records = json.loads(result.stdout)
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            return False
        record = records[0]
        if record.get("Name", "").lstrip("/") != name or record.get("Image") != expected_image:
            return False
        created = datetime.datetime.fromisoformat(str(record["Created"]).replace("Z", "+00:00")).timestamp()
        # Docker and the supervisor use this host clock. Never claim a container
        # created before this launch attempt.
        return launch_started_at <= created <= time.time()
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def gpu_free_snapshot(bdfs):
    """Read free VRAM for the selected physical GPUs from sysfs."""
    values = []
    for bdf in bdfs:
        base = Path("/sys/bus/pci/devices") / bdf
        total = int((base / "mem_info_vram_total").read_text())
        used = int((base / "mem_info_vram_used").read_text())
        values.append({"bdf": bdf, "free_bytes": total - used})
    return values


class Session:
    def __init__(self, protocol, output):
        self.protocol, self.output = validate(protocol), output
        self.deadline = time.monotonic() + protocol["seconds"]
        self.stop = threading.Event()
        self.failure = None
        self.phase = "admission"
        self.container_pid = 0
        self.root = ROOT
        self.reporter = Reporter(
            target=protocol["config"].get("R9V_OBSERVABILITY_TARGET", "")
        )
        self.container_name = None
        self.early_captured = False
        self.driver = None
        self.bdfs = protocol["config"]["R9V_EXPECTED_GPU_BDFS"].split(",")

    def route_identity(self, config):
        runtime = Path(config.get("R9V_RUNTIME_DESCRIPTOR", ""))
        if not runtime.is_file():
            return None
        model_package = config.get("R9V_MODEL_PACKAGE")
        model_hash = config.get("R9V_MODEL_PACKAGE_SHA256")
        if not isinstance(model_package, str) or not model_package or not isinstance(model_hash, str) or len(model_hash) != 64:
            return None
        profile = Path(config.get("R9V_PROFILE", "")).resolve()
        profile_json = profile.parent / "profile.json"
        if not profile_json.is_file():
            return None
        try:
            profile_data = json.loads(profile_json.read_text())
            descriptor = (self.root / profile_data["descriptors"]["model_package"]).resolve()
            if not descriptor.is_relative_to(self.root.resolve()):
                raise ValueError("model package descriptor escapes frozen source")
            if profile_data.get("model_package") != model_package or hashlib.sha256(descriptor.read_bytes()).hexdigest() != model_hash:
                raise ValueError("model package identity does not match the selected profile descriptor")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid verified model package identity: {error}") from error
        ignored = {
            "R9V_CONFIG_FILE", "R9V_CONTAINER_NAME", "R9V_HOST_BIND", "R9V_HOST_PORT",
            "R9V_PROFILE",
            "R9V_PROFILE_ROOT", "R9V_MODEL_DIR", "R9V_ROUTE_PROFILE_DIR",
            "R9V_RUNTIME_DESCRIPTOR", "R9V_EXPERT_MANIFEST_PATH",
            "R9V_EXPERT_CATALOG_PATH", "R9V_CALIBRATION_PATH", "R9V_MEMORY_SEED_PATH",
            "R9V_CACHE_DIR", "R9V_STATE_DIR", "R9V_REPO_ROOT", "R9V_LOG_DIR",
            "R9V_PLE_PATH", "R9V_OBSERVABILITY_TARGET",
        }
        effective = {k: v for k, v in sorted(config.items()) if k.startswith("R9V_") and k not in ignored}
        manifest = Path(config.get("R9V_EXPERT_MANIFEST_PATH", ""))
        if not manifest.is_file():
            return None
        effective["expert_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        image_id = config.get("R9V_IMAGE", "")
        if not isinstance(image_id, str) or not image_id:
            return None
        return {"model_package": model_package, "model_hash": model_hash,
                "runtime_hash": hashlib.sha256((hashlib.sha256(runtime.read_bytes()).hexdigest() + "\0" + image_id).encode()).hexdigest(),
                "config_hash": hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

    def monitor(self):
        low = 0
        zone_low = 0
        with (self.output / "memory.jsonl").open("x") as stream:
            while not self.stop.is_set():
                try:
                    sample = memory_snapshot(self.bdfs)
                    zone_low = (
                        zone_low + 1
                        if host_exhausted(sample.get("host_pressure", {}))
                        else 0
                    )
                    if zone_low >= 2:
                        self.failure = (
                            "Normal memory zones below minimum watermark for 2 samples"
                        )
                    # RAM guard runs before driver/proc/container or disk evidence operations.
                    low = low + 1 if sample["host_available_bytes"] < 4 * GIB else 0
                    if low >= 3:
                        self.failure = "Host available RAM below 4 GiB for 3 samples"
                    if self.driver:
                        sample["driver"] = self.driver.poll()
                        driver_sample = sample["driver"].get("sample") or {}
                        sample["gpus"] = driver_sample.get("gpus", [])
                        if sample["driver"]["status"] in ("failed", "stalled"):
                            self.failure = (
                                "Driver telemetry " + sample["driver"]["status"]
                            )
                    sample["phase"] = self.phase
                    if self.container_pid:
                        sample["cgroup"] = cgroup_snapshot(self.container_pid)
                        sample["workers"] = worker_snapshot(self.container_pid)
                        sample["scheduler"] = scheduler_snapshot(self.container_pid)
                    self.reporter.send(
                        "r9v.progress",
                        phase=self.phase,
                        container=self.container_name,
                        host_available_bytes=sample["host_available_bytes"],
                        host_pressure=sample.get("host_pressure"),
                        driver_status=sample.get("driver", {}).get("status"),
                        driver_last_read=sample.get("driver", {}).get("last_read"),
                        driver_sample_age_seconds=sample.get("driver", {}).get(
                            "sample_age_seconds"
                        ),
                        gpus=[
                            {
                                key: gpu.get(key)
                                for key in (
                                    "bdf",
                                    "free",
                                    "total",
                                    "gpu_busy_percent",
                                    "mem_busy_percent",
                                    "sensors",
                                )
                            }
                            for gpu in sample.get("gpus", [])
                        ],
                        workers=[
                            {
                                k: w.get(k)
                                for k in (
                                    "rank",
                                    "steps",
                                    "phase",
                                    "last_batch_tokens",
                                    "scheduled_tokens",
                                    "stage_marker",
                                )
                            }
                            for w in sample.get("workers", [])
                        ],
                    )
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                except (OSError, ValueError, StopIteration) as error:
                    self.failure = f"Memory telemetry unavailable: {error}"
                self.stop.wait(1)

    def collect_early(self, reason):
        if self.early_captured or not self.container_name:
            return
        self.early_captured = True
        self.reporter.send("r9v.suspect", container=self.container_name, reason=reason)
        try:
            early_capture(
                self.container_name,
                self.output / (self.container_name + "-early"),
                reason,
            )
        except (OSError, ValueError) as error:
            self.reporter.send("r9v.capture_error", error=str(error))

    def reclaim_gpu_memory(self, path, baseline):
        """Wait for owned-arm VRAM to return near its prelaunch baseline."""
        target = {item["bdf"]: max(0, item["free_bytes"] - 256 * 2**20) for item in baseline}
        evidence = {"baseline": baseline, "target_free_bytes": target,
                    "margin_bytes": 256 * 2**20, "max_wait_seconds": 90, "samples": []}
        started = time.monotonic()
        recovered = False
        error = None
        while time.monotonic() - started <= 90:
            try:
                sample = {item["bdf"]: item["free_bytes"] for item in gpu_free_snapshot(self.bdfs)}
                evidence["samples"].append({"time": time.time(), "free_bytes": sample})
                if all(sample.get(bdf, -1) >= required for bdf, required in target.items()):
                    recovered = True
                    break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                break
            remaining = self.deadline - time.monotonic() - 120
            if remaining <= 0:
                error = "session deadline/cleanup reserve reached"
                break
            time.sleep(min(1, remaining))
        evidence.update(recovered=recovered, seconds=time.monotonic() - started, error=error)
        save(path / "cleanup-reclamation.json", evidence)
        return recovered, evidence

    def command(self, args, path, *, timeout=60, env=None, cleanup=False):
        remaining = self.deadline - time.monotonic() - (0 if cleanup else 120)
        if remaining <= 0:
            raise TimeoutError("Session deadline/cleanup reserve reached")
        until = time.monotonic() + min(timeout, remaining)
        env = {
            **(env if env is not None else os.environ),
            "R9V_OBSERVABILITY_RUN_ID": self.reporter.run_id,
            "R9V_OBSERVABILITY_TARGET": self.protocol["config"].get(
                "R9V_OBSERVABILITY_TARGET", ""
            ),
        }
        self.reporter.send(
            "r9v.event", event="command_start", phase=self.phase, evidence=path.name
        )
        with path.open("x") as log:
            proc = subprocess.Popen(
                list(map(str, args)),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                while proc.poll() is None:
                    if not cleanup and self.failure:
                        raise RuntimeError(self.failure)
                    if time.monotonic() >= until:
                        raise TimeoutError(f"Command timed out: {args[0]}")
                    time.sleep(0.1)
                if proc.returncode:
                    raise RuntimeError(f"Command exited {proc.returncode}: see {path}")
            except (RuntimeError, TimeoutError) as error:
                if not cleanup:
                    self.collect_early(str(error))
                raise
            finally:
                self.reporter.send(
                    "r9v.event",
                    event="command_end",
                    phase=self.phase,
                    evidence=path.name,
                )
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)

    def run(self):
        self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        # Later edits in the working checkout must not change another arm's
        # launcher, defaults, preflight, or workload implementation mid-session.
        self.root = self.output / "source"
        self.root.mkdir()
        input_hashes = {}
        shutil.copy2(ROOT / "r9v", self.root / "r9v")
        input_hashes["r9v"] = hashlib.sha256(
            (self.root / "r9v").read_bytes()
        ).hexdigest()
        for directory in (
            "tools",
            "scripts",
            "profiles",
            "packages",
            "hardware",
            "runtimes",
            "release",
        ):
            for source in (ROOT / directory).rglob("*"):
                packaged_evidence = source.is_relative_to(ROOT / "packages" / "placements") and "evidence" in source.parts
                if not source.is_file() or (source.suffix not in {
                    ".py", ".sh", ".json", ".env",
                } and not packaged_evidence):
                    continue
                if source.stat().st_size > (64 if packaged_evidence else 4) * 2**20:
                    raise ValueError(
                        f"Unexpected large qualification source input: {source}"
                    )
                relative = source.relative_to(ROOT)
                target = self.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                input_hashes[str(relative)] = hashlib.sha256(
                    target.read_bytes()
                ).hexdigest()
        save(self.output / "source-inputs.json", input_hashes)
        # Resolve all image tags before starting any comparison.
        for arm in self.protocol["arms"]:
            config = {**self.protocol["config"], **arm.get("env", {})}
            image = subprocess.check_output(
                [
                    "docker",
                    "image",
                    "inspect",
                    config["R9V_IMAGE"],
                    "--format",
                    "{{.Id}}",
                ],
                text=True,
                timeout=15,
            ).strip()
            arm.setdefault("env", {})["R9V_IMAGE"] = image
            profile_path = Path(
                config.get("R9V_PROFILE")
                or ROOT / "profiles/qwen38-flash-next/dual-r9700/profile.env"
            )
            profile_path = profile_path.resolve()
            if not profile_path.is_relative_to(ROOT.resolve()):
                raise ValueError(f"profile must be inside the qualification checkout: {profile_path}")
            relative_profile = profile_path.relative_to(ROOT)
            frozen_profile = self.root / relative_profile
            if not frozen_profile.is_file():
                frozen_profile.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(profile_path, frozen_profile)
                input_hashes[str(relative_profile)] = hashlib.sha256(
                    frozen_profile.read_bytes()
                ).hexdigest()
            arm["env"]["R9V_PROFILE"] = str(frozen_profile.resolve())
            arm["env"]["R9V_PROFILE_ROOT"] = str(frozen_profile.parent.resolve())
            manifest_path = Path(
                config.get("R9V_EXPERT_MANIFEST_PATH")
                or str(
                    Path(config["R9V_MODEL_DIR"])
                    / config.get(
                        "R9V_MANIFEST_REL",
                        "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
                    )
                )
            )
            raw = manifest_path.read_bytes()
            frozen_manifest = self.output / (
                hashlib.sha256(raw).hexdigest() + ".manifest.json"
            )
            if not frozen_manifest.exists():
                frozen_manifest.write_bytes(raw)
            arm["env"]["R9V_EXPERT_MANIFEST_PATH"] = str(frozen_manifest.resolve())
            descriptor = config.get("R9V_RUNTIME_DESCRIPTOR")
            if descriptor:
                descriptor_path = Path(descriptor).resolve()
                if not descriptor_path.is_relative_to(ROOT.resolve()):
                    raise ValueError("runtime descriptor must be inside the qualification checkout")
                frozen_descriptor = self.root / descriptor_path.relative_to(ROOT)
                if not frozen_descriptor.is_file():
                    raise ValueError("frozen runtime descriptor was not copied")
                arm["env"]["R9V_RUNTIME_DESCRIPTOR"] = str(frozen_descriptor.resolve())
            for key, value in list(config.items()):
                if key.startswith("R9V_DEV_") and value:
                    raw = Path(value).read_bytes()
                    target = self.output / (
                        hashlib.sha256(raw).hexdigest() + Path(value).suffix
                    )
                    if not target.exists():
                        target.write_bytes(raw)
                    arm["env"][key] = str(target.resolve())
        save(self.output / "protocol.json", self.protocol)
        (self.output / "prompt.txt").write_text(self.protocol["prompt"])
        self.driver = DriverSampler(
            self.bdfs,
            command=[
                sys.executable,
                str(self.root / "tools/observability.py"),
                "--driver-worker",
            ],
        )
        self.reporter.send("r9v.session_start", seconds=self.protocol["seconds"])
        thread = threading.Thread(target=self.monitor, daemon=True)
        thread.start()
        results = []
        try:
            for arm in self.protocol["arms"]:
                name = "r9v-q-" + self.output.name[-16:] + "-" + arm["name"]
                path = self.output / arm["name"]
                path.mkdir()
                result = {"arm": arm["name"], "passed": False, "container": name}
                config = {
                    **self.protocol["config"],
                    **arm.get("env", {}),
                    "R9V_CONTAINER_NAME": name,
                }
                env = {**os.environ, **config, "R9V_CONFIG_FILE": ""}
                port = config.get("R9V_HOST_PORT", "8004")
                url = f"http://127.0.0.1:{port}"
                launched = False
                self.container_name = name
                self.early_captured = False
                baseline = []
                launch_attempted = False
                launch_started_at = None
                try:
                    self.phase = arm["name"] + "-startup"
                    self.container_pid = 0
                    if memory_snapshot(self.bdfs)["host_available_bytes"] < 16 * GIB:
                        raise RuntimeError(
                            "Need at least 16 GiB available host RAM before startup"
                        )
                    baseline = gpu_free_snapshot(self.bdfs)
                    save(path / "gpu-baseline.json", baseline)
                    exists = subprocess.run(
                        ["docker", "inspect", name], capture_output=True, timeout=15, check=False
                    )
                    if exists.returncode == 0:
                        raise ValueError(f"Container already exists: {name}")
                    save(path / "config.json", config)
                    launch_attempted = True
                    launch_started_at = time.time()
                    self.command(
                        [self.root / "scripts/launch.sh"], path / "launch.log", env=env
                    )
                    launched = True
                    self.container_pid = int(
                        subprocess.check_output(
                            ["docker", "inspect", "--format", "{{.State.Pid}}", name],
                            text=True,
                            timeout=5,
                        ).strip()
                    )
                    until = min(
                        time.monotonic() + self.protocol["startup_seconds"],
                        self.deadline - 120,
                    )
                    while True:
                        if self.failure:
                            raise RuntimeError(self.failure)
                        if time.monotonic() >= until:
                            raise TimeoutError("Startup exceeded its deadline")
                        try:
                            with urllib.request.urlopen(
                                url + "/health", timeout=2
                            ) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            state = subprocess.check_output(
                                [
                                    "docker",
                                    "inspect",
                                    "--format",
                                    "{{.State.Running}}",
                                    name,
                                ],
                                text=True,
                                timeout=5,
                            ).strip()
                            if state != "true":
                                raise RuntimeError(
                                    "Container exited before readiness; see container.log"
                                )
                            time.sleep(2)
                    samples = []
                    try:
                        with urllib.request.urlopen(
                            url + "/metrics", timeout=5
                        ) as response:
                            (path / "metrics-before.txt").write_bytes(response.read())
                    except OSError:
                        pass
                    for trial in range(
                        self.protocol["warmups"] + self.protocol["trials"]
                    ):
                        warmup = trial < self.protocol["warmups"]
                        self.phase = arm["name"] + ("-warmup" if warmup else "-measure")
                        dest = path / f"request-{trial}.json"
                        self.command(
                            [
                                sys.executable,
                                self.root / "tools/benchmark_openai.py",
                                "--url",
                                url + "/v1",
                                "--model",
                                config["R9V_SERVED_MODEL_NAME"],
                                "--prompt-file",
                                self.output / "prompt.txt",
                                "--max-tokens",
                                str(self.protocol["max_tokens"]),
                                "--disable-thinking",
                                "--timeout",
                                "120",
                            ],
                            dest,
                            timeout=130,
                        )
                        data = json.loads(dest.read_text())
                        if data.get("finish_reason") not in (
                            "stop",
                            "length",
                        ) or not data.get("tg_tokens_per_second"):
                            raise ValueError("Incomplete streaming result")
                        if not warmup:
                            samples.append(data["tg_tokens_per_second"])
                        try:
                            with urllib.request.urlopen(
                                url + "/metrics", timeout=5
                            ) as response:
                                (path / f"metrics-request-{trial}.txt").write_bytes(
                                    response.read()
                                )
                        except OSError:
                            pass
                    if self.protocol.get("pressure_probe"):
                        self.phase = arm["name"] + "-pressure"
                        self.command(
                            [
                                sys.executable,
                                self.root / "tools/pressure_runtime.py",
                                "--container",
                                name,
                                "--url",
                                url,
                                "--model",
                                config["R9V_SERVED_MODEL_NAME"],
                                "--prompt",
                                self.output / "prompt.txt",
                                "--output",
                                path / "pressure",
                            ],
                            path / "pressure.log",
                            timeout=600,
                            env=env,
                        )
                    if arm.get("route_split"):
                        self.phase = arm["name"] + "-routes"
                        directory = config.get("R9V_ROUTE_PROFILE_DIR")
                        if not directory:
                            raise ValueError(
                                "route collection requires R9V_ROUTE_PROFILE_DIR"
                            )
                        identity = self.route_identity(config)
                        identity_args = []
                        if identity is not None:
                            identity_path = self.output / (arm["name"] + "-route-identity.json")
                            save(identity_path, identity)
                            identity_args = ["--identity", identity_path]
                        self.command(
                            [
                                sys.executable,
                                self.root / "tools/capture_routes.py",
                                "--url",
                                url,
                                "--model",
                                config["R9V_SERVED_MODEL_NAME"],
                                "--directory",
                                directory,
                                "--split",
                                arm["route_split"],
                                *identity_args,
                                *(
                                    ["--no-collect"]
                                    if not arm.get("route_collect", True)
                                    else []
                                ),
                                *(
                                    ["--limit", str(arm["route_limit"])]
                                    if "route_limit" in arm
                                    else []
                                ),
                            ],
                            path / "routes.log",
                            timeout=900,
                            env=env,
                        )
                    if arm.get("workload", self.protocol.get("workload", False)):
                        self.phase = arm["name"] + "-workload"
                        self.command(
                            [
                                sys.executable,
                                self.root / "tools/runtime_workload.py",
                                "--url",
                                url,
                                "--model",
                                config["R9V_SERVED_MODEL_NAME"],
                                "--context",
                                str(
                                    self.protocol.get("workload_context")
                                    or config.get("R9V_MAX_MODEL_LEN", "131072")
                                ),
                                "--output",
                                path / "workload",
                                "--request-timeout",
                                str(self.protocol["workload_request_timeout"]),
                            ],
                            path / "workload.log",
                            timeout=1500,
                            env={
                                **env,
                                "R9V_QUALIFY_HEADROOM": "1"
                                if self.protocol.get("check_headroom", True)
                                else "0",
                            },
                        )
                    if self.protocol.get("trace", False):
                        self.phase = arm["name"] + "-profile"
                        for endpoint in ("start_profile",):
                            with urllib.request.urlopen(
                                urllib.request.Request(url + "/" + endpoint, data=b""),
                                timeout=30,
                            ):
                                pass
                        try:
                            self.command(
                                [
                                    sys.executable,
                                    self.root / "tools/benchmark_openai.py",
                                    "--url",
                                    url + "/v1",
                                    "--model",
                                    config["R9V_SERVED_MODEL_NAME"],
                                    "--prompt-file",
                                    self.output / "prompt.txt",
                                    "--max-tokens",
                                    "64",
                                    "--disable-thinking",
                                ],
                                path / "profile-request.json",
                                timeout=120,
                            )
                        finally:
                            with urllib.request.urlopen(
                                urllib.request.Request(url + "/stop_profile", data=b""),
                                timeout=45,
                            ):
                                pass
                    try:
                        with urllib.request.urlopen(
                            url + "/metrics", timeout=5
                        ) as response:
                            (path / "metrics-after.txt").write_bytes(response.read())
                    except OSError:
                        pass
                    self.command(
                        [
                            "docker",
                            "exec",
                            name,
                            "python3",
                            "-c",
                            "import pathlib,json;print(json.dumps([json.loads(p.read_text()) for p in pathlib.Path('/tmp').glob('r9v-worker-*.json')]))",
                        ],
                        path / "workers.json",
                        timeout=15,
                    )
                    result.update(
                        passed=True,
                        tg_samples=samples,
                        tg_median=statistics.median(samples),
                    )
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    TimeoutError,
                    subprocess.SubprocessError,
                ) as error:
                    launched = launched or container_created_after_launch(
                        name, config["R9V_IMAGE"], launch_started_at, launch_attempted
                    )
                    self.collect_early(str(error))
                    result["error"] = str(error)
                finally:
                    if launched:
                        if not (path / "workers.json").exists():
                            records = []
                            for rank in range(len(self.bdfs)):
                                target = path / f"worker-{rank}.json"
                                try:
                                    self.command(
                                        [
                                            "docker",
                                            "cp",
                                            f"{name}:/tmp/r9v-worker-{rank}.json",
                                            target,
                                        ],
                                        path / f"worker-{rank}-copy.log",
                                        timeout=10,
                                        cleanup=True,
                                    )
                                    records.append(json.loads(target.read_text()))
                                except Exception as error:
                                    result.setdefault(
                                        "worker_evidence_errors", []
                                    ).append(str(error))
                            save(path / "workers.json", records)
                        for command, filename, timeout in [
                            (["docker", "stop", "--time", "20", name], "stop.log", 35),
                            (
                                ["docker", "logs", "--timestamps", name],
                                "server.log",
                                20,
                            ),
                            (["docker", "inspect", name], "inspect.json", 15),
                        ]:
                            try:
                                self.command(
                                    command,
                                    path / filename,
                                    timeout=timeout,
                                    cleanup=True,
                                )
                            except Exception as error:
                                result.setdefault("cleanup_errors", []).append(
                                    str(error)
                                )
                                result["passed"] = False
                        if baseline:
                            recovered, reclamation = self.reclaim_gpu_memory(path, baseline)
                            result["reclamation"] = reclamation
                            if not recovered:
                                result.setdefault("cleanup_errors", []).append(
                                    "Owned GPUs did not reclaim VRAM to the prelaunch baseline"
                                )
                                result.setdefault("error", "Owned GPUs did not reclaim VRAM to the prelaunch baseline")
                                result["passed"] = False
                    save(path / "result.json", result)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                continue_workload_failure = (
                    self.protocol.get("continue_on_workload_failure") is True
                    and self.phase == arm["name"] + "-workload"
                    and not self.failure
                    and not result.get("cleanup_errors")
                )
                if not result["passed"] and not continue_workload_failure:
                    break
        finally:
            self.stop.set()
            thread.join(timeout=3)
            self.driver.close()
            self.reporter.send("r9v.session_end", failure=self.failure)
            save(self.output / "results.json", results)
        return (
            0
            if len(results) == len(self.protocol["arms"])
            and all(r["passed"] for r in results)
            else 1
        )


def main():
    def interrupted(signum, frame):
        raise RuntimeError(f"Qualification interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        return Session(json.loads(args.protocol.read_text()), args.output).run()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Qualification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
