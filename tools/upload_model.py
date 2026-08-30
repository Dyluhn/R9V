#!/usr/bin/env python3
"""Validate and upload the R9V model bundle without making a local 90 GiB copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


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


def build_uploads(root: Path, manifest: Path, release_root: Path) -> list[Upload]:
    target = root / "unsloth-iq4-xs-gguf" / "UD-IQ4_XS"
    metadata = root / "official-metadata"
    mtp = root / "mtp-fp8-block-minimal"
    vision = root / "vision-q8" / "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"
    target_specs = [
        (
            "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
            10_946_624,
            "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4",
        ),
        (
            "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
            49_835_229_856,
            "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7",
        ),
        (
            "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf",
            43_836_407_744,
            "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833",
        ),
    ]
    uploads = [
        Upload(release_root / "model" / "README.md", "README.md"),
        Upload(release_root / "model" / "QWEN_LICENSE.txt", "LICENSE"),
        Upload(release_root / "release" / "sources.lock.json", "sources.lock.json"),
    ]
    uploads.extend(
        Upload(target / name, f"target/{name}", size, digest)
        for name, size, digest in target_specs
    )
    for name in (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ):
        uploads.append(Upload(metadata / name, f"metadata/{name}"))
    uploads.extend(
        [
            Upload(mtp / "config.json", "mtp/config.json"),
            Upload(
                mtp / "model.safetensors",
                "mtp/model.safetensors",
                2_698_415_880,
                "33c1160579174630f4222882da479b67f1554d84963a991b4f4f0b69237110c1",
            ),
            Upload(
                mtp / "mtp-fp8-block-manifest.json",
                "mtp/mtp-fp8-block-manifest.json",
                expected_sha256=(
                    "5a9405a8054262803b50ef78b97ebac14e548a003ab2196762c0d710b0090865"
                ),
            ),
            Upload(
                vision,
                "vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
                616_703_104,
                "b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad",
            ),
            Upload(
                manifest,
                "manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json",
                expected_sha256=(
                    "2f6f0e59f2555673430857d764b461d271941810a7e1a1d07090b599caf81c88"
                ),
            ),
        ]
    )
    return uploads


def main() -> int:
    args = parse_args()
    release_root = Path(__file__).resolve().parents[1]
    uploads = build_uploads(
        args.model_root.resolve(), args.expert_manifest.resolve(), release_root
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
            args.hash_large or actual_bytes < 1_000_000_000
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
        print("Validation complete; pass --execute after `hf auth login` to upload.")
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
