# SPDX-License-Identifier: Apache-2.0
"""Run scripts/launch.sh against a fake Docker and check the container it would create."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILES = [
    "target/a",
    "target/b",
    "target/c",
    "metadata/config.json",
    "mtp/config.json",
    "mtp/model.safetensors",
    "vision/mmproj",
    "manifests/hot",
]
FAKE_DOCKER = """#!/usr/bin/env python3
import json, os, sys
if sys.argv[1] == "container":
    sys.exit(1)
if sys.argv[1] == "info":
    print("name=rootless")
    sys.exit(0)
open(os.environ["TEST_ARGS"], "w").write(json.dumps(sys.argv[1:]))
"""


FAKE_LAYOUT = {
    "R9V_TARGET_REL": "target/a",
    "R9V_TARGET_SHARD2_REL": "target/b",
    "R9V_TARGET_SHARD3_REL": "target/c",
    "R9V_MMPROJ_REL": "vision/mmproj",
    "R9V_MANIFEST_REL": "manifests/hot",
}


def run_launcher(tmp_path: Path, env: dict[str, str], files=LAUNCH_FILES, layout=FAKE_LAYOUT):
    """Return (completed process, docker run arguments or None)."""
    model = tmp_path / "models"
    for relative in [*files, "ple"]:
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    docker = tmp_path / "bin/docker"
    docker.parent.mkdir(exist_ok=True)
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    captured = tmp_path / "args.json"
    base = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("R9V_") and not key.startswith("TEST_")
    }
    full_env = {
        **base,
        "PATH": f"{docker.parent}{os.pathsep}{os.environ['PATH']}",
        "TEST_ARGS": str(captured),
        "R9V_MODEL_DIR": str(model),
        "R9V_PLE_PATH": str(model / "ple"),
        "R9V_CACHE_DIR": str(tmp_path / "cache"),
        "R9V_CACHE_NAMESPACE": "test",
        "R9V_PREFLIGHT": "0",
        **layout,
        **env,
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/launch.sh")],
        env=full_env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    args = json.loads(captured.read_text()) if captured.exists() else None
    return result, args


def values(args: list[str], option: str) -> list[str]:
    return [args[index + 1] for index, value in enumerate(args) if value == option]


def test_launcher_passes_prefix_cache_retention_interval(tmp_path):
    result, args = run_launcher(
        tmp_path, {"R9V_PREFIX_CACHE_RETENTION_INTERVAL": "1616"}
    )

    assert result.returncode == 0, result.stderr
    assert values(args, "--prefix-cache-retention-interval") == ["1616"]


def test_launcher_omits_retention_interval_when_prefix_caching_is_off(tmp_path):
    result, args = run_launcher(
        tmp_path,
        {
            "R9V_PREFIX_CACHE_RETENTION_INTERVAL": "1616",
            "R9V_ENABLE_PREFIX_CACHING": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "--prefix-cache-retention-interval" not in args


def test_launcher_refuses_non_integer_retention_interval(tmp_path):
    result, args = run_launcher(
        tmp_path, {"R9V_PREFIX_CACHE_RETENTION_INTERVAL": "16k"}
    )

    assert result.returncode == 2
    assert "R9V_PREFIX_CACHE_RETENTION_INTERVAL must be a non-negative integer" in result.stderr
    assert args is None


def test_launcher_publishes_the_api_on_localhost_by_default(tmp_path):
    result, args = run_launcher(tmp_path, {})

    assert result.returncode == 0, result.stderr
    assert values(args, "--publish") == ["127.0.0.1:8004:8000"]
    assert "WARN the API is published" not in result.stderr


def test_launcher_publishes_on_all_interfaces_only_when_asked(tmp_path):
    result, args = run_launcher(tmp_path, {"R9V_HOST_BIND": "0.0.0.0"})

    assert result.returncode == 0, result.stderr
    assert values(args, "--publish") == ["0.0.0.0:8004:8000"]
    assert "WARN the API is published on 0.0.0.0:8004" in result.stderr


def test_launcher_keeps_localhost_published_beside_a_specific_address(tmp_path):
    # Health checks, doctor and qualification connect to 127.0.0.1.
    result, args = run_launcher(tmp_path, {"R9V_HOST_BIND": "fd00::5", "R9V_HOST_PORT": "8014"})

    assert result.returncode == 0, result.stderr
    assert values(args, "--publish") == ["[fd00::5]:8014:8000", "127.0.0.1:8014:8000"]


def test_launcher_refuses_a_bind_that_is_not_an_ip_address(tmp_path):
    result, args = run_launcher(tmp_path, {"R9V_HOST_BIND": "lan"})

    assert result.returncode == 2
    assert "R9V_HOST_BIND must be an IPv4 or IPv6 address" in result.stderr
    assert args is None


def test_launcher_refuses_an_address_in_the_port_setting(tmp_path):
    result, args = run_launcher(tmp_path, {"R9V_HOST_PORT": "0.0.0.0:8004"})

    assert result.returncode == 2
    assert "R9V_HOST_PORT must be a port number" in result.stderr
    assert "R9V_HOST_BIND" in result.stderr
    assert args is None


RUNTIME_V3 = ROOT / "runtimes/qwen38-flash-next-gfx1201-mtp4-v3/runtime.json"
CED_ON = {
    "R9V_CED": "on",
    "R9V_CED_PROJECTOR_REL": "ced/projector.safetensors",
    "R9V_CED_PRECISION": "bf16",
    "R9V_CED_MIN_PROMPT": "8192",
    "R9V_CED_TAIL": "2048",
    "R9V_CED_DEFAULT": "on",
}


def test_launcher_mounts_pinned_runtime_overlays_without_ced(tmp_path):
    result, args = run_launcher(tmp_path, {"R9V_RUNTIME_DESCRIPTOR": str(RUNTIME_V3)})

    assert result.returncode == 0, result.stderr
    volumes = values(args, "--volume")
    overlays = RUNTIME_V3.parent / "overlays"
    assert f"{overlays}:/r9v-full-mutable:ro" in volumes
    assert sum(volume.startswith(f"{overlays}/") for volume in volumes) == 9
    assert "R9V_FULL_MUTABLE_CACHE=1" in values(args, "--env")
    assert not any(value.startswith("R9V_CED_") for value in values(args, "--env"))


def test_launcher_adds_ced_overlay_and_package_projector(tmp_path):
    files = [*LAUNCH_FILES, "ced/projector.safetensors"]
    result, args = run_launcher(
        tmp_path, {"R9V_RUNTIME_DESCRIPTOR": str(RUNTIME_V3), **CED_ON}, files
    )

    assert result.returncode == 0, result.stderr
    overlays = RUNTIME_V3.parent / "overlays"
    assert any(volume.startswith(f"{overlays}/model.py:") for volume in values(args, "--volume"))
    assert "R9V_CED_PROJECTOR=/models/ced/projector.safetensors" in values(args, "--env")


def test_launcher_refuses_a_modified_overlay_before_docker_run(tmp_path):
    runtime = tmp_path / "runtime/runtime.json"
    shutil.copytree(RUNTIME_V3.parent / "overlays", runtime.parent / "overlays")
    shutil.copy(RUNTIME_V3, runtime)
    (runtime.parent / "overlays/scheduler.py").write_text("# edited\n")

    result, args = run_launcher(tmp_path, {"R9V_RUNTIME_DESCRIPTOR": str(runtime)})

    assert result.returncode == 2
    assert "SHA-256 mismatch" in result.stderr
    assert args is None


# Launch parity: the public profile must create the same container as the
# deployed consolidated 1.3.0 service. The fixture is that service's Docker create
# payload, normalized by tests/golden/launch/make_deployed_fixture.py.
DEPLOYED = json.loads(
    (ROOT / "tests/golden/launch/uncensored-1.3.0-service.json").read_text(encoding="utf-8")
)
UNCENSORED = ROOT / "profiles/qwen38-flash-next/dual-r9700-mtp4-uncensored"
UNCENSORED_FILES = [
    *[f"target/Qwen3.8-Flash-Next-Uncensored-UD-IQ4_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)],
    "metadata/config.json",
    "mtp/config.json",
    "mtp/model.safetensors",
    "vision/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf",
    "ced/ced-projector-split16.safetensors",
]
LOG_CONFIG = {"Type": "json-file", "Config": {"max-file": "5", "max-size": "20m"}}


def command_options(command: list[str]) -> dict[str, str | None]:
    """Model path plus each vLLM option and its value; option order does not matter."""
    options: dict[str, str | None] = {"model": command[0]}
    index = 1
    while index < len(command):
        has_value = index + 1 < len(command) and not command[index + 1].startswith("--")
        options[command[index]] = command[index + 1] if has_value else None
        index += 2 if has_value else 1
    return options


def deployed(env_changes=None, drop_binds=()):
    """The deployed service in canonical form, with the stated changes applied."""
    env = {**DEPLOYED["env"], **(env_changes or {})}
    return {
        "image": DEPLOYED["image"],
        "command": command_options(DEPLOYED["command"]),
        "env": {key: value for key, value in env.items() if value is not None},
        "binds": sorted(b for b in DEPLOYED["binds"] if b.split(":")[0] not in drop_binds),
        "host": DEPLOYED["host"],
    }


def launched(tmp_path: Path, env: dict[str, str]):
    """Run the public uncensored profile and return its container in canonical form."""
    placeholders = DEPLOYED["placeholders"]
    result, args = run_launcher(
        tmp_path,
        {
            "R9V_PROFILE": str(UNCENSORED / "profile.env"),
            "R9V_CACHE_NAMESPACE": placeholders["namespace"],
            "R9V_CONTAINER_NAME": placeholders["container"],
            "R9V_EXPECTED_GPU_BDFS": placeholders["bdfs"],
            "R9V_HOST_PORT": DEPLOYED["host"]["PortBindings"]["8000/tcp"][0]["HostPort"],
            **env,
        },
        UNCENSORED_FILES,
        layout={},
    )
    assert result.returncode == 0, result.stderr
    image_index = 1  # args[0] is "run"; --detach is the only option without a value
    while args[image_index].startswith("--"):
        image_index += 1 if args[image_index] == "--detach" else 2
    options, image, command = args[1:image_index], args[image_index], args[image_index + 1:]
    tokens = {
        str(tmp_path / "models"): "<model_dir>",
        str(tmp_path / "models/ple"): "<ple>",
        str(tmp_path / "cache"): "<cache>",
        str(UNCENSORED.parents[2] / "packages/placements/qwen38-flash-next/uncensored-iq4-xs/"
            "dual-r9700/mtp4-warmstart-r1/manifest.json"): "<manifest>",
        str(RUNTIME_V3.parent / "overlays"): "<overlays>",
    }
    binds = []
    for volume in values(options, "--volume"):
        source, target, *mode = volume.split(":")
        parent, _, name = source.rpartition("/")
        source = tokens.get(source) or f"{tokens[parent]}/{name}"
        binds.append(f"{source}:{target}:{mode[0] if mode else 'rw'}")
    image_env = set(DEPLOYED["image_env"])
    log = dict(value.split("=", 1) for value in values(options, "--log-opt"))
    ports = {}
    for published in values(options, "--publish"):
        host_ip, host_port, container_port = published.rsplit(":", 2)
        ports.setdefault(f"{container_port}/tcp", []).append(
            {"HostIp": host_ip, "HostPort": host_port}
        )
    return {
        "image": image,
        "command": command_options(command),
        "env": dict(v.split("=", 1) for v in values(options, "--env") if v not in image_env),
        "binds": sorted(binds),
        "host": {
            "IpcMode": values(options, "--ipc")[0],
            "SecurityOpt": sorted(values(options, "--security-opt")),
            "Devices": sorted(values(options, "--device")),
            "LogConfig": {"Type": values(options, "--log-driver")[0], "Config": log},
            "PortBindings": ports,
        },
    }


def test_launch_with_the_deployed_ced_default_matches_the_deployed_service(tmp_path):
    # The deployed service ran CED loaded but default off; everything else is the release.
    assert launched(tmp_path, {"R9V_CED_DEFAULT": "off"}) == deployed()


def test_release_defaults_differ_from_the_deployed_service_only_in_ced_default_on(tmp_path):
    assert launched(tmp_path, {}) == deployed({"R9V_CED_DEFAULT": "on"})


def test_ced_off_launch_is_the_deployed_service_without_ced(tmp_path):
    ced_env = {key: None for key in DEPLOYED["env"] if key.startswith("R9V_CED_")}

    assert launched(tmp_path, {"R9V_CED": "off"}) == deployed(
        ced_env, drop_binds={"<overlays>/model.py"}
    )


def test_release_pins_equal_the_deployed_overlays_placement_and_projector():
    runtime = json.loads(RUNTIME_V3.read_text(encoding="utf-8"))
    placement = json.loads(
        (ROOT / "packages/placements/qwen38-flash-next/uncensored-iq4-xs/dual-r9700/"
         "mtp4-full-mutable.json").read_text(encoding="utf-8")
    )
    package = json.loads(
        (ROOT / "packages/models/qwen38-flash-next/uncensored-iq4-xs--mtp-blockfp8--mmproj-f16/"
         "package.json").read_text(encoding="utf-8")
    )
    manifest = ROOT / placement["manifest"]["path"]
    projector = next(a for a in package["artifacts"] if a["role"] == "ced-projector")

    assert runtime["overlays"]["sha256"] == DEPLOYED["overlay_sha256"]
    assert runtime["image_id"] == DEPLOYED["image"]
    assert placement["manifest"]["sha256"] == DEPLOYED["placement_sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == DEPLOYED["placement_sha256"]
    assert projector["path"] == "ced/ced-projector-split16.safetensors"
    assert projector["sha256"] == DEPLOYED["ced_projector_sha256"]
