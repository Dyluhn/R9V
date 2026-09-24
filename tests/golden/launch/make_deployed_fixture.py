#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Normalize a deployed service's Docker create payload into the launch-parity fixture.

Usage: make_deployed_fixture.py CREATE_JSON PACKAGE_MANIFEST_JSON IMAGE_ENV_JSON > fixture.json

CREATE_JSON is the create payload of the deployed consolidated 1.3.0 service,
PACKAGE_MANIFEST_JSON that package's MANIFEST.json, and IMAGE_ENV_JSON the output
of `docker image inspect IMAGE --format '{{json .Config.Env}}'`. Host-specific
values are replaced by bind target and environment key, so the fixture holds no
host paths: bind sources become tokens, and the cache namespace, container name
and GPU BDFs become fixed placeholders the test passes to the launcher.
"""

import hashlib
import json
import sys
from pathlib import Path

SOURCE_TOKENS = {
    "/models": "<model_dir>",
    "/ple/per_layer_token_embd.iq4_nl.bin": "<ple>",
    "/cache": "<cache>",
    "/placement/experts.json": "<manifest>",
    "/r9v-full-mutable": "<overlays>",
}
PLACEHOLDERS = {"namespace": "NAMESPACE", "container": "CONTAINER", "bdfs": "GPU_BDFS"}
PACKAGE_PROJECTOR = "/models/ced/ced-projector-split16.safetensors"


def binds(raw):
    by_target = {bind.split(":")[1]: bind.split(":")[0] for bind in raw}
    overlays = by_target["/r9v-full-mutable"]
    result = []
    for bind in raw:
        source, target, mode = bind.split(":")
        if target.startswith("/ced/"):
            continue  # R9V reads the same projector from the model package under /models.
        if target in SOURCE_TOKENS:
            source = SOURCE_TOKENS[target]
        elif source.startswith(overlays + "/"):
            source = "<overlays>/" + source[len(overlays) + 1:]
        else:
            raise SystemExit(f"unexpected bind target {target}")
        result.append(f"{source}:{target}:{mode}")
    return sorted(result)


def environment(raw, image_env):
    env = dict(entry.split("=", 1) for entry in raw if entry not in image_env)
    namespace = env["VLLM_CACHE_ROOT"].rsplit("/", 1)[1]
    for key, value in env.items():
        env[key] = value.replace(f"/cache/vllm/{namespace}", f"/cache/vllm/{PLACEHOLDERS['namespace']}")
    env["R9V_CONTAINER_NAME"] = PLACEHOLDERS["container"]
    env["R9V_EXPECTED_GPU_BDFS"] = PLACEHOLDERS["bdfs"]
    env["R9V_CED_PROJECTOR"] = PACKAGE_PROJECTOR
    return dict(sorted(env.items()))


def main():
    create_path, manifest_path, image_env_path = map(Path, sys.argv[1:4])
    create = json.loads(create_path.read_text())
    manifest = json.loads(manifest_path.read_text())["files"]
    image_env = json.loads(image_env_path.read_text())
    host = create["HostConfig"]
    fixture = {
        "provenance": {
            "source": "Docker create payload of the deployed consolidated 1.3.0 service "
                      "(evidence/1.3.0/v1.3.0-create.json in the 1.3.0 package)",
            "source_sha256": hashlib.sha256(create_path.read_bytes()).hexdigest(),
            "generated_by": "tests/golden/launch/make_deployed_fixture.py",
            "normalized": [
                "bind sources replaced by tokens chosen from the bind target",
                "the /ced/<projector> bind dropped and R9V_CED_PROJECTOR set to the package path; "
                "both are ced-projector-split16 (SHA-256 below)",
                "cache namespace, container name and GPU BDFs replaced by placeholders",
                "environment entries identical to the image's own environment removed",
            ],
            "deployed_ced_default": "off",
        },
        "image": create["Image"],
        "command": create["Cmd"],
        "env": environment(create["Env"], set(image_env)),
        "binds": binds(host["Binds"]),
        "host": {
            "IpcMode": host["IpcMode"],
            "SecurityOpt": sorted(host["SecurityOpt"]),
            "Devices": sorted(device["PathOnHost"] for device in host["Devices"]),
            "LogConfig": host["LogConfig"],
            "PortBindings": host["PortBindings"],
        },
        "image_env": image_env,
        "overlay_sha256": {name[len("runtime/"):]: digest for name, digest in sorted(manifest.items())
                           if name.startswith("runtime/")},
        "placement_sha256": manifest["config/placement.json"],
        "ced_projector_sha256": "2d14d57e353491d034f0b75caab69af5afb806c3d6ab46cda054825a6a9c57fc",
        "placeholders": PLACEHOLDERS,
    }
    print(json.dumps(fixture, indent=2))


if __name__ == "__main__":
    main()
