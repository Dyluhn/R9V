#!/usr/bin/env python3
"""Validate and upload the R9V model bundle without making a local 90 GiB copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


SMALL_HASH_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True)
class Upload:
    source: Path
    destination: str
    expected_bytes: int | None = None
    expected_sha256: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument(
        "--package",
        type=Path,
        help="model-package descriptor; defaults to the Qwen release package",
    )
    parser.add_argument(
        "--repo-id", default="Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS"
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--hash-large",
        action="store_true",
        help="stream and verify all large payload hashes before upload",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="create/update the Hugging Face repo; default is validation only",
    )
    return parser.parse_args()


def _load_package(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "r9v.model-package.v1":
        raise ValueError(f"{path}: expected r9v.model-package.v1")
    return payload


def _artifact_source(
    root: Path, release_root: Path, manifest: Path, destination: str
) -> Path:
    target = root / "unsloth-iq4-xs-gguf" / "UD-IQ4_XS"
    metadata = root / "official-metadata"
    mtp = root / "mtp-fp8-block-minimal"
    if destination == "LICENSE":
        return release_root / "model" / "QWEN_LICENSE.txt"
    if destination == "THIRD_PARTY_NOTICES.md":
        return release_root / "model" / "THIRD_PARTY_NOTICES.md"
    if destination.startswith("target/"):
        return target / Path(destination).name
    if destination.startswith("metadata/"):
        return metadata / Path(destination).name
    if destination.startswith("mtp/"):
        return mtp / Path(destination).name
    if destination.startswith("vision/"):
        return root / "vision-q8" / Path(destination).name
    if destination.startswith("manifests/"):
        return manifest
    raise ValueError(f"no local source mapping for package artifact {destination}")


def build_uploads(
    root: Path,
    manifest: Path,
    release_root: Path,
    package_path: Path | None = None,
) -> list[Upload]:
    if package_path is None:
        package_path = (
            release_root
            / "packages/models/qwen38-flash-next/"
            "ud-iq4-xs--mtp-blockfp8--mmproj-q8/package.json"
        )
    package = _load_package(package_path)
    uploads = [
        Upload(release_root / "model" / "README.md", "README.md"),
        Upload(package_path, "package.json"),
        Upload(release_root / "release" / "sources.lock.json", "sources.lock.json"),
    ]
    for artifact in package["artifacts"]:
        destination = artifact["path"]
        uploads.append(
            Upload(
                _artifact_source(root, release_root, manifest, destination),
                destination,
                int(artifact["bytes"]),
                artifact["sha256"],
            )
        )
    return uploads


def main() -> int:
    args = parse_args()
    if args.execute and not args.hash_large:
        raise SystemExit("refusing --execute without --hash-large")
    release_root = Path(__file__).resolve().parents[1]
    package_path = args.package
    if package_path is not None:
        package_path = package_path.resolve()
    uploads = build_uploads(
        args.model_root.resolve(),
        args.expert_manifest.resolve(),
        release_root,
        package_path,
    )
    for item in uploads:
        if not item.source.is_file():
            raise SystemExit(f"missing source: {item.source}")
        actual_bytes = item.source.stat().st_size
        if item.expected_bytes is not None and actual_bytes != item.expected_bytes:
            raise SystemExit(
                f"size mismatch for {item.source}: {actual_bytes} != "
                f"{item.expected_bytes}"
            )
        should_hash = item.expected_sha256 is not None and (
            args.hash_large or actual_bytes <= SMALL_HASH_LIMIT
        )
        if should_hash:
            actual_hash = sha256(item.source)
            if actual_hash != item.expected_sha256:
                raise SystemExit(
                    f"sha256 mismatch for {item.source}: {actual_hash} != "
                    f"{item.expected_sha256}"
                )
        print(
            json.dumps(
                {
                    "source": str(item.source),
                    "destination": item.destination,
                    "bytes": actual_bytes,
                    "hash_checked": should_hash,
                },
                sort_keys=True,
            )
        )

    if not args.execute:
        print(
            "Validation complete; pass --hash-large --execute only after "
            "review and authentication."
        )
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit("install huggingface_hub before using --execute") from error

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    for item in uploads:
        print(f"Uploading {item.destination} ({item.source.stat().st_size} bytes)")
        api.upload_file(
            path_or_fileobj=item.source,
            path_in_repo=item.destination,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add {item.destination}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
