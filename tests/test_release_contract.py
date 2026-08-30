from __future__ import annotations

import json
import re
from pathlib import Path

from tools.upload_model import build_uploads

ROOT = Path(__file__).resolve().parents[1]
QWEN_PACKAGE = (
    ROOT
    / "packages/models/qwen38-flash-next/"
    "ud-iq4-xs--mtp-blockfp8--mmproj-q8/package.json"
)
QWEN_PROFILE = ROOT / "profiles/qwen38-flash-next/dual-r9700/profile.json"
QWEN_PLACEMENT = (
    ROOT
    / "packages/placements/qwen38-flash-next/ud-iq4-xs/dual-r9700/"
    "r1-lru16-vision-128k.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_qwen_public_status_matches_unpublished_distribution() -> None:
    profile = _load(QWEN_PROFILE)
    package = _load(QWEN_PACKAGE)

    assert profile["status"] == "release-candidate"
    assert package["distribution"]["status"] == "prepared"
    assert package["distribution"]["revision"] is None


def test_qwen_package_contains_every_launch_contract_file() -> None:
    package = _load(QWEN_PACKAGE)
    placement = _load(QWEN_PLACEMENT)
    paths = {artifact["path"] for artifact in package["artifacts"]}

    required = {
        "LICENSE",
        "metadata/config.json",
        "metadata/tokenizer.json",
        "metadata/tokenizer_config.json",
        "metadata/chat_template.jinja",
        "metadata/preprocessor_config.json",
        "metadata/video_preprocessor_config.json",
        "metadata/generation_config.json",
        "metadata/merges.txt",
        "metadata/vocab.json",
        "mtp/config.json",
        "mtp/model.safetensors",
        "mtp/mtp-fp8-block-manifest.json",
        "vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf",
        placement["artifact_path"],
    }
    assert required <= paths
    assert len([path for path in paths if path.endswith(".gguf")]) == 4


def test_qwen_uploader_covers_required_package_artifacts() -> None:
    package = _load(QWEN_PACKAGE)
    required = {
        artifact["path"]: artifact
        for artifact in package["artifacts"]
        if artifact.get("required", True)
    }
    uploads = build_uploads(Path("/model-root"), Path("/manifest.json"), ROOT)
    by_destination = {upload.destination: upload for upload in uploads}

    assert required.keys() <= by_destination.keys()
    for destination, artifact in required.items():
        upload = by_destination[destination]
        assert upload.expected_bytes == artifact["bytes"]
        assert upload.expected_sha256 == artifact["sha256"]


def test_derived_ggml_sources_ship_the_historical_mit_notice() -> None:
    notice_paths = (
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "vendor/vllm-gguf-plugin/THIRD_PARTY_NOTICES.md",
        ROOT / "kernels/r9v-gfx1201/THIRD_PARTY_NOTICES.md",
    )
    for path in notice_paths:
        text = path.read_text(encoding="utf-8")
        assert "MIT License" in text
        assert "Copyright (c) 2023-2024 The ggml authors" in text

    manifest = (ROOT / "vendor/vllm-gguf-plugin/MANIFEST.in").read_text()
    project = (ROOT / "vendor/vllm-gguf-plugin/pyproject.toml").read_text()
    assert "include THIRD_PARTY_NOTICES.md" in manifest
    assert 'license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]' in project


def test_historical_radiance_patch_text_is_not_distributed() -> None:
    runner = (
        ROOT / "vendor/vllm/vllm/v1/worker/gpu_model_runner.py"
    ).read_text(encoding="utf-8")

    assert "# RADIANCE: align the placeholder mask" not in runner
    assert "def align_draft_multimodal_mask(" in runner


def test_root_readme_does_not_overstate_release_readiness() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "fully custom native R9V engines" in readme
    assert "adapted vLLM/Radiance" in readme
    assert "package upload is not public yet" in readme
    assert "clean-checkout release benchmark" in readme
    assert "./r9v list --by-topology" in readme


def test_root_readme_local_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)

    local_targets = [target for target in targets if "://" not in target]
    assert local_targets
    for target in local_targets:
        path = target.split("#", maxsplit=1)[0]
        assert (ROOT / path).exists(), target


def test_image_build_requires_buildx_and_loads_local_images() -> None:
    script = (ROOT / "scripts/build-image.sh").read_text(encoding="utf-8")

    assert "docker buildx version" in script
    assert script.count("docker buildx build --load") == 2
    assert 'R9V_VLLM_VERSION="$vllm_version"' in script
