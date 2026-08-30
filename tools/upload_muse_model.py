#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and upload Muse V1 directly from its canonical local payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = RELEASE_ROOT / "packages/models/muse-glimmer-30b/v1-v12"
PUBLICATION_PATH = PACKAGE_DIR / "publication.json"


@dataclass(frozen=True)
class Upload:
    source: Path
    destination: str
    expected_bytes: int | None = None
    expected_sha256: str | None = None
    payload: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_uploads(
    target: Path,
    sidecar_root: Path | None = None,
    include_optional: bool = False,
    package_dir: Path = PACKAGE_DIR,
) -> tuple[str, list[Upload]]:
    publication = _load_json(package_dir / "publication.json")
    package = _load_json(package_dir / publication["package_descriptor"])
    if publication.get("schema") != "r9v.huggingface-publication.v1":
        raise ValueError("unexpected publication manifest schema")
    if package.get("schema") != "r9v.model-package.v1":
        raise ValueError("unexpected model package schema")
    repository = publication["repository"]
    if package["distribution"]["repository"] != repository:
        raise ValueError("publication and package repository IDs disagree")

    artifact_by_path = {
        artifact["path"]: artifact for artifact in package["artifacts"]
    }
    uploads = []
    for item in publication["metadata"]:
        source = Path(item["source"])
        destination = Path(item["destination"])
        if (
            source.is_absolute()
            or destination.is_absolute()
            or ".." in source.parts
            or ".." in destination.parts
        ):
            raise ValueError("publication metadata paths must be portable")
        destination_text = destination.as_posix()
        artifact = artifact_by_path.get(destination_text)
        uploads.append(
            Upload(
                package_dir / source,
                destination_text,
                (
                    int(artifact["bytes"])
                    if artifact is not None
                    else None
                ),
                artifact["sha256"] if artifact is not None else None,
            )
        )
    metadata_destinations = {item.destination for item in uploads}
    source_map = publication["payload_sources"]
    for artifact in package["artifacts"]:
        required = bool(artifact.get("required", False))
        if not required and not include_optional:
            continue
        destination = artifact["path"]
        if destination in metadata_destinations:
            continue
        mapping = source_map.get(destination)
        if mapping is None:
            raise ValueError(f"no publication source mapping for {destination}")
        argument = mapping["argument"]
        if argument == "target":
            source = target
        elif argument == "sidecar_root":
            if sidecar_root is None:
                raise ValueError(
                    f"--sidecar-root is required to include {destination}"
                )
            source = sidecar_root / mapping["relative_path"]
        else:
            raise ValueError(f"unknown publication source argument {argument!r}")
        uploads.append(
            Upload(
                source=source,
                destination=destination,
                expected_bytes=int(artifact["bytes"]),
                expected_sha256=artifact["sha256"],
                payload=True,
            )
        )
    return repository, uploads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="canonical V12 GGUF to publish under the V1 destination name",
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        help="directory containing optional projector and draft files",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="include the descriptor's optional vision and draft sidecars",
    )
    parser.add_argument(
        "--repo-id",
        help="override the repository encoded in publication.json",
    )
    parser.add_argument(
        "--hash-large",
        action="store_true",
        help="stream and verify large payload hashes before upload",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="create/update the repository; default is validation only",
    )
    return parser.parse_args()


def validate_uploads(uploads: list[Upload], hash_large: bool) -> None:
    destinations: set[str] = set()
    for item in uploads:
        if item.destination in destinations:
            raise SystemExit(f"duplicate destination: {item.destination}")
        destinations.add(item.destination)
        if not item.source.is_file():
            raise SystemExit(f"missing source for {item.destination}: {item.source}")
        actual_bytes = item.source.stat().st_size
        if item.expected_bytes is not None and actual_bytes != item.expected_bytes:
            raise SystemExit(
                f"size mismatch for {item.destination}: {actual_bytes} != "
                f"{item.expected_bytes}"
            )
        should_hash = not item.payload or hash_large
        actual_sha256 = sha256(item.source) if should_hash else None
        if (
            item.expected_sha256 is not None
            and actual_sha256 is not None
            and actual_sha256 != item.expected_sha256
        ):
            raise SystemExit(f"sha256 mismatch for {item.destination}")
        print(
            json.dumps(
                {
                    "bytes": actual_bytes,
                    "destination": item.destination,
                    "hash_checked": should_hash,
                    "sha256": actual_sha256,
                },
                sort_keys=True,
            )
        )


def main() -> int:
    args = parse_args()
    if args.execute and not args.hash_large:
        raise SystemExit("refusing --execute without --hash-large")
    repository, uploads = build_uploads(
        target=args.target.resolve(),
        sidecar_root=(
            args.sidecar_root.resolve() if args.sidecar_root is not None else None
        ),
        include_optional=args.include_optional,
    )
    repo_id = args.repo_id or repository
    validate_uploads(uploads, args.hash_large)
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "file_count": len(uploads),
                    "repository": repo_id,
                },
                sort_keys=True,
            )
        )
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit("install huggingface_hub before using --execute") from error

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    for item in uploads:
        api.upload_file(
            path_or_fileobj=item.source,
            path_in_repo=item.destination,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {item.destination}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
