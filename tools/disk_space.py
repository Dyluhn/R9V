# SPDX-License-Identifier: Apache-2.0
"""Estimate the free disk space an R9V installation still needs, per filesystem.

The doctor runs this before fetch and setup, so a full disk shows up in seconds
with numbers instead of an hour into a download.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

PLE_NAME = "per_layer_token_embd.iq4_nl.bin"
PLE_BYTES = 28_800_138_240
# One compiled configuration took 0.9-1.2 GiB in the reference host's vLLM
# cache; CED on and CED off compile separately, so allow for both.
COMPILE_CACHE_BYTES = 4 * 1024**3
# The README's measured ~50 GiB for the public bundle plus its containerd image
# store, less the ~10.6 GiB of bundle parts that setup keeps in its data directory.
IMAGE_STORE_BYTES = 40 * 1024**3
RESERVE_BYTES = 1024**3
GIB = 1024**3


@dataclass(frozen=True)
class Need:
    label: str
    path: Path
    bytes: int


def _complete(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def missing_package_bytes(artifacts: list[dict], model_dir: Path, reuse_dir: Path | None = None) -> int:
    """Bytes of required artifacts not yet present at full size. A same-size file
    in reuse_dir counts as present: setup hard-links it instead of downloading."""
    return sum(
        artifact["bytes"]
        for artifact in artifacts
        if artifact.get("required", True)
        and not _complete(model_dir / artifact["path"], artifact["bytes"])
        and not (reuse_dir and _complete(reuse_dir / artifact["path"], artifact["bytes"]))
    )


def install_needs(
    package: dict,
    model_dir: Path,
    data_dir: Path,
    ple_path: Path,
    cache_dir: Path,
    *,
    bundle: dict | None = None,
    docker_root: Path | None = None,
    reuse_dir: Path | None = None,
) -> list[Need]:
    """What is still to be written: package files, the PLE table, the image bundle
    parts and Docker's copy of the image (only when the image is not loaded yet;
    pass bundle=None otherwise), and a first compile cache."""
    needs = [Need("model package", model_dir, missing_package_bytes(package["artifacts"], model_dir, reuse_dir))]
    if not _complete(ple_path, PLE_BYTES):
        needs.append(Need("PLE table", ple_path.parent, PLE_BYTES))
    if bundle is not None:
        parts = data_dir / "image-bundle"
        needs.append(Need("image bundle parts", parts, sum(
            part["bytes"] for part in bundle["parts"] if not _complete(parts / part["name"], part["bytes"]))))
        if docker_root is not None:
            needs.append(Need("Docker image store", docker_root, IMAGE_STORE_BYTES))
    if not (cache_dir.is_dir() and any(cache_dir.iterdir())):
        needs.append(Need("compile cache", cache_dir, COMPILE_CACHE_BYTES))
    return [need for need in needs if need.bytes]


def _existing(path: Path) -> Path:
    """The path itself or its nearest existing parent: where the bytes will land."""
    path = path.expanduser().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def by_filesystem(needs: list[Need], free=lambda path: shutil.disk_usage(path).free) -> list[dict]:
    """Group needs by filesystem: its free bytes, the bytes needed plus a 1 GiB
    reserve, and whether they fit."""
    groups: dict[int, dict] = {}
    for need in needs:
        anchor = _existing(need.path)
        group = groups.setdefault(anchor.stat().st_dev, {"path": anchor, "needs": [], "free": free(anchor)})
        group["needs"].append(need)
    for group in groups.values():
        group["required"] = sum(need.bytes for need in group["needs"]) + RESERVE_BYTES
        group["fits"] = group["free"] >= group["required"]
    return list(groups.values())


def describe(group: dict) -> str:
    parts = " + ".join(f"{need.label} {need.bytes / GIB:.2f}" for need in group["needs"])
    return (f"{group['path']}: needs {group['required'] / GIB:.2f} GiB ({parts} + 1.00 reserve), "
            f"{group['free'] / GIB:.2f} GiB free")
