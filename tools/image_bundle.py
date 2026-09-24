#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verified, streaming loader for Docker save archives split into parts.

Setup calls load_bundle(). As a command, it downloads and verifies a bundle and
loads it into Docker, or with --verify-only stops before Docker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse


SCHEMA = "r9v.image-bundle.v1"
FORMAT = "docker-save-gzip-parts"
_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RELEASE = re.compile(r"^/[^/]+/[^/]+/releases/download/[^/]+/[^/]+$")


class ImageBundleError(ValueError):
    """Manifest, download, verification, or Docker loading failure."""


def _sha256(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _image_id(value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ImageBundleError("image IDs must be sha256:<64 lowercase hex> values")
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _url(value: object) -> str:
    if not isinstance(value, str):
        raise ImageBundleError("part URL must be a string")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ImageBundleError(
            "part URL must be unauthenticated HTTPS without query or fragment"
        )
    if parsed.hostname != "github.com" or not _RELEASE.fullmatch(parsed.path):
        raise ImageBundleError("part URL must be a pinned GitHub release download URL")
    return value


def validate_manifest(manifest: object) -> dict:
    """Validate and return a manifest, rejecting ambiguous or unsafe fields."""
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ImageBundleError(f"schema must be {SCHEMA}")
    if manifest.get("format") != FORMAT:
        raise ImageBundleError(f"format must be {FORMAT}")
    ids = manifest.get("image_ids")
    if not isinstance(ids, list) or not ids:
        raise ImageBundleError("image_ids must be a non-empty list")
    ids = [_image_id(x) for x in ids]
    if len(ids) > 2:
        raise ImageBundleError("at most two image IDs are allowed")
    if len(set(ids)) != len(ids):
        raise ImageBundleError("image_ids must be unique")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ImageBundleError("parts must be a non-empty list")
    if len(parts) > 64:
        raise ImageBundleError("at most 64 parts are allowed")
    checked = []
    names = set()
    for part in parts:
        if not isinstance(part, dict):
            raise ImageBundleError("each part must be an object")
        name = part.get("name")
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in {".", ".."}
        ):
            raise ImageBundleError("part names must be safe filenames")
        if name in names:
            raise ImageBundleError("part names must be unique")
        names.add(name)
        size = part.get("bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size >= 2**31
        ):
            raise ImageBundleError("part bytes must be a positive integer below 2 GiB")
        digest = part.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ImageBundleError("part sha256 must be 64 lowercase hex characters")
        checked.append(
            {
                "name": name,
                "bytes": size,
                "sha256": digest,
                "url": _url(part.get("url")),
            }
        )
    total = manifest.get("bytes")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ImageBundleError("bytes must be a positive integer")
    if total != sum(x["bytes"] for x in checked):
        raise ImageBundleError("bytes does not equal the part sizes")
    if total > 32 * 1024**3:
        raise ImageBundleError("bundle is larger than 32 GiB")
    concatenated = manifest.get("sha256")
    if not isinstance(concatenated, str) or not re.fullmatch(
        r"[0-9a-f]{64}", concatenated
    ):
        raise ImageBundleError("sha256 must be 64 lowercase hex characters")
    return {
        "schema": SCHEMA,
        "format": FORMAT,
        "image_ids": ids,
        "parts": checked,
        "bytes": total,
        "sha256": concatenated,
    }


def read_manifest(path: Path | str) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ImageBundleError(f"cannot read manifest: {exc}") from exc
    return validate_manifest(value)


def _download(url: str, target: Path, size: int) -> None:
    existing = target.stat().st_size if target.exists() else 0
    if existing > size:
        target.unlink()
        existing = 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=60
        )
        status = getattr(response, "status", 200)
        content_range = getattr(response, "headers", {}).get("Content-Range")
        append = bool(existing) and status == 206
        if append and (
            not content_range or not content_range.startswith(f"bytes {existing}-")
        ):
            response.close()
            raise ImageBundleError(
                "server returned an incorrect Content-Range for resume"
            )
        if existing and not append:
            response.close()
            target.unlink()
            existing = 0
            response = urllib.request.urlopen(urllib.request.Request(url), timeout=60)
        mode = "ab" if append else "wb"
        with response, target.open(mode) as out:
            remaining = size - existing
            for block in iter(
                lambda: response.read(min(1024 * 1024, remaining + 1)), b""
            ):
                if len(block) > remaining:
                    raise ImageBundleError("download exceeded the manifest part size")
                out.write(block)
                remaining -= len(block)
            if remaining:
                raise ImageBundleError("download ended before the manifest part size")
    except (OSError, urllib.error.URLError) as exc:
        raise ImageBundleError(f"download failed: {url}: {exc}") from exc


def verify_parts(manifest: dict, cache_dir: Path | str) -> list[Path]:
    """Verify every cached part and the hash of their ordered concatenation."""
    manifest = validate_manifest(manifest)
    root = Path(cache_dir)
    paths = []
    combined = hashlib.sha256()
    total = 0
    for part in manifest["parts"]:
        path = root / part["name"]
        if not path.is_file():
            raise ImageBundleError(f"missing cached part: {part['name']}")
        with path.open("rb") as stream:
            digest, size = _sha256(stream)
        if size != part["bytes"] or digest != part["sha256"]:
            raise ImageBundleError(f"cached part failed verification: {part['name']}")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                combined.update(block)
        total += size
        paths.append(path)
    if total != manifest["bytes"] or combined.hexdigest() != manifest["sha256"]:
        raise ImageBundleError("concatenated image bundle failed verification")
    return paths


def ensure_parts(
    manifest: dict, cache_dir: Path | str, *, force_download: bool = False
) -> list[Path]:
    manifest = validate_manifest(manifest)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ImageBundleError("cache directory must not be a symlink")
    root = root.resolve()
    count = len(manifest["parts"])
    for number, part in enumerate(manifest["parts"], start=1):
        path = root / part["name"]
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ImageBundleError(f"cache path is not a regular file: {part['name']}")
        if force_download or not path.is_file():
            if force_download and path.exists():
                path.unlink()
            partial = path.with_name(path.name + ".part")
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise ImageBundleError(
                    f"partial cache path is not a regular file: {partial.name}"
                )
            if shutil.disk_usage(root).free < part["bytes"]:
                raise ImageBundleError("insufficient free space for image part")
            if partial.exists() and partial.stat().st_size == part["bytes"]:
                with partial.open("rb") as stream:
                    digest, actual_size = _sha256(stream)
                if actual_size == part["bytes"] and digest == part["sha256"]:
                    os.replace(partial, path)
                    continue
                partial.unlink()
            print(
                f"Downloading image bundle part {number}/{count}: {part['name']} "
                f"({part['bytes'] / 1024**3:.2f} GiB)",
                flush=True,
            )
            _download(part["url"], partial, part["bytes"])
            with partial.open("rb") as stream:
                digest, size = _sha256(stream)
            if size != part["bytes"] or digest != part["sha256"]:
                partial.unlink(missing_ok=True)
                raise ImageBundleError(
                    f"downloaded part failed verification: {part['name']}"
                )
            os.replace(partial, path)
    return verify_parts(manifest, root)


def _inspect(docker: tuple[str, ...], image_id: str, timeout: float) -> str | None:
    try:
        result = subprocess.run(
            (*docker, "image", "inspect", "--format", "{{.Id}}", image_id),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    found = result.stdout.strip()
    return found if _SHA256.fullmatch(found) else None


def load_bundle(
    manifest: dict,
    cache_dir: Path | str,
    *,
    docker: tuple[str, ...] = ("docker",),
    expected_image_id: str | None = None,
    allow_existing: bool = False,
    clean_verification: bool = False,
    timeout: float = 300.0,
) -> str:
    """Download, verify, stream-load, and exactly match a Docker image ID."""
    manifest = validate_manifest(manifest)
    expected = _image_id(expected_image_id) if expected_image_id else None
    if expected and expected not in manifest["image_ids"]:
        raise ImageBundleError("expected image ID is absent from manifest")
    if expected is None:
        if len(manifest["image_ids"]) != 1:
            raise ImageBundleError(
                "expected_image_id is required for multi-image bundles"
            )
        expected = manifest["image_ids"][0]
    if (
        allow_existing
        and not clean_verification
        and _inspect(docker, expected, timeout) == expected
    ):
        return expected
    paths = ensure_parts(manifest, cache_dir, force_download=clean_verification)
    try:
        process = subprocess.Popen(
            (*docker, "image", "load"),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdin is not None
        os.set_blocking(process.stdin.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdin, selectors.EVENT_WRITE)
        deadline = time.monotonic() + timeout
        try:
            for path in paths:
                with path.open("rb") as stream:
                    pending = stream.read(1024 * 1024)
                    while pending:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(process.args, timeout)
                        if not selector.select(remaining):
                            raise subprocess.TimeoutExpired(process.args, timeout)
                        try:
                            written = os.write(process.stdin.fileno(), pending)
                        except BlockingIOError:
                            continue
                        pending = pending[written:]
                        if not pending:
                            pending = stream.read(1024 * 1024)
            process.stdin.close()
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise ImageBundleError("docker image load timed out") from exc
        except (BrokenPipeError, OSError):
            process.kill()
            process.wait()
            raise ImageBundleError("docker image load failed while streaming parts")
        finally:
            selector.close()
        try:
            returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise ImageBundleError("docker image load timed out") from exc
        if returncode:
            raise ImageBundleError(
                f"docker image load failed with exit code {returncode}"
            )
    except OSError as exc:
        raise ImageBundleError(f"cannot execute docker image load: {exc}") from exc
    found = _inspect(docker, expected, timeout)
    if found != expected:
        raise ImageBundleError(
            f"loaded image ID mismatch: expected {expected}, got {found}"
        )
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="bundle descriptor, e.g. release/image-bundle-*.json")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="directory for the downloaded parts; needs the bundle's size free")
    parser.add_argument("--verify-only", action="store_true",
                        help="download missing parts, verify every part and the reassembled "
                             "archive, and stop before docker image load")
    parser.add_argument("--image-id", help="image to load from a bundle that holds two")
    args = parser.parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
        if args.verify_only:
            ensure_parts(manifest, args.cache_dir)
            print(f"PASS {len(manifest['parts'])} parts and the reassembled archive "
                  f"({manifest['bytes']} bytes, sha256 {manifest['sha256']}) match "
                  f"{args.manifest}; not loaded into Docker")
        else:
            image = load_bundle(manifest, args.cache_dir, expected_image_id=args.image_id,
                                allow_existing=True, timeout=900)
            print(f"PASS Docker has {image}")
    except ImageBundleError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
