# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from tools.upload_model import build_uploads

ROOT = Path(__file__).resolve().parents[1]
QWEN_PACKAGE = (
    ROOT
    / "packages/models/qwen38-flash-next/"
    "ud-iq4-xs--mtp-blockfp8--mmproj-q8/package.json"
)
QWEN_PROFILE = ROOT / "profiles/qwen38-flash-next/dual-r9700/profile.json"
RADIANCE_RESULT = (
    ROOT
    / "docs/qualification/results/qwen38-public-radiance-dual-r9700.json"
)
QWEN_PLACEMENT = (
    ROOT
    / "packages/placements/qwen38-flash-next/ud-iq4-xs/dual-r9700/"
    "r1-lru16-vision-128k.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_qwen_public_status_matches_published_distribution() -> None:
    profile = _load(QWEN_PROFILE)
    package = _load(QWEN_PACKAGE)

    assert profile["status"] == "release-candidate"
    assert package["distribution"]["status"] == "published"
    assert package["distribution"]["revision"] == (
        "bf836f0c20b6c92fcad4226ad3115eb8a19f7582"
    )


def test_qwen_package_contains_every_launch_contract_file() -> None:
    package = _load(QWEN_PACKAGE)
    placement = _load(QWEN_PLACEMENT)
    paths = {artifact["path"] for artifact in package["artifacts"]}

    required = {
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
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
    assert "package.json" in by_destination


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
    assert "adapted vLLM deployment architecture informed by" in readme
    assert "immutable 90.36 GiB model package is public" in readme
    assert "clean-host package installation test" in readme
    assert "./r9v list --by-topology" in readme
    assert (
        "https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS"
        in readme
    )
    assert "https://huggingface.co/Dyluhn/Muse-Glimmer-30B-R9V-V1" in readme


def test_root_readme_local_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)

    local_targets = [target for target in targets if "://" not in target]
    assert local_targets
    for target in local_targets:
        path = target.split("#", maxsplit=1)[0]
        assert (ROOT / path).exists(), target


def test_clean_clone_runbooks_are_ordered_and_fail_closed() -> None:
    qwen = (ROOT / "docs/installation.md").read_text(encoding="utf-8")
    muse = (ROOT / "docs/muse-v1.md").read_text(encoding="utf-8")
    model_card = (ROOT / "model/README.md").read_text(encoding="utf-8")

    assert qwen.index("git clone --recursive") < qwen.index("./r9v list")
    assert qwen.index("export MODEL_DIR=") < qwen.index('"$MODEL_DIR"')
    assert "./r9v build qwen38" in qwen
    assert "./r9v verify qwen38" in qwen and "-- --hash" in qwen
    assert "28800138240" in qwen
    assert "for attempt in {1..180}" in qwen
    assert "amd-smi list" in qwen
    assert 'host_port="${R9V_HOST_PORT:-8004}"' in qwen
    assert 'container="${R9V_CONTAINER_NAME:-r9v-qwen38-flash-next}"' in qwen
    assert '(\nhost_port="${R9V_HOST_PORT:-8004}"' in qwen
    assert "ready=0" in qwen
    assert "if (( ! ready )); then" in qwen
    assert 'docker logs --tail 200 "$container"' in qwen
    assert 'docker stop "$container"' in qwen
    assert 'docker rm "$container"' in qwen
    assert "THIRD_PARTY_NOTICES.md" in qwen
    assert "--model-dir ..." not in qwen

    assert muse.index("git clone --recursive") < muse.index("./r9v show muse")
    for action in ("fetch", "build", "run"):
        assert f"./r9v {action} muse" in muse
    assert "fails until" in muse

    assert "python tools/prepare_ple.py" not in model_card
    assert "docs/installation.md#2-fetch-or-arrange-the-model-bundle" in model_card


def test_completed_radiance_comparison_matches_result_record() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    result = _load(RADIANCE_RESULT)

    assert "Stock/public Radiance" in readme
    assert "**Pending clean benchmark**" not in readme
    assert f'{result["pp8192"]["mean_tok_s"]:.2f}' in readme
    assert f'{result["tg256"]["mean_tok_s"]:.2f}' in readme
    assert "**1,512.01** (+3,239.98%)" in readme
    assert "**78.11** (+197.90%)" in readme


def test_readme_benchmark_tables_and_inspiration_are_consistent() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[antirez's DS4](https://github.com/antirez/ds4)" in readme
    assert "[Neroued's ninfer](https://github.com/Neroued/ninfer)" in readme
    assert "not shared code provenance" in readme
    for value in (
        "**1,500.68** (+1.54%)",
        "**2,175.17** (+47.21%)",
        "**2,078.20** (+46.36%)",
        "**26.84** (+7.70%)",
    ):
        assert value in readme
    assert readme.count("| Runtime | PP") == 2


def test_public_radiance_result_record_is_reproducible() -> None:
    result = _load(RADIANCE_RESULT)
    runtime = result["runtime"]
    tg = result["tg256"]

    assert result["schema"] == "r9v.comparator-benchmark.v1"
    assert result["topology"] == "dual-r9700-tp2"
    assert runtime["image"].startswith("sha256:")
    for key in ("radiance_revision", "vllm_revision", "gguf_plugin_revision"):
        assert len(runtime[key]) == 40
    assert runtime["r9v_performance_code"] is False
    assert runtime["r9v_custom_kernels"] is False

    assert tg["prompt_tokens"] == 278
    assert tg["completion_tokens"] == 256
    assert tg["warmups"] == 1
    assert len(tg["samples_tok_s"]) == 3
    assert abs(tg["mean_tok_s"] - sum(tg["samples_tok_s"]) / 3) < 1e-12

    pp = result["pp8192"]
    if pp is not None:
        assert pp["prompt_tokens"] in (8136, 8192)
        assert pp["completion_tokens"] == 1
        assert pp["warmups"] >= 0
        assert len(pp["samples_tok_s"]) == 10
        assert pp["prefix_cache_hits_tokens"] == 0
        assert abs(
            pp["mean_tok_s"]
            - sum(pp["samples_tok_s"]) / len(pp["samples_tok_s"])
        ) < 1e-12


def test_image_build_requires_buildx_and_loads_local_images() -> None:
    script = (ROOT / "scripts/build-image.sh").read_text(encoding="utf-8")
    dockerfile = (
        ROOT / "vendor/vllm/docker/Dockerfile.r9v_rocm714"
    ).read_text(encoding="utf-8")

    assert "docker buildx version" in script
    assert script.count("docker buildx build --load") == 2
    assert 'R9V_VLLM_VERSION="$vllm_version"' in script
    assert "Dockerfile.r9v_rocm714" in script
    assert "rocm/dev-ubuntu-24.04:7.14.0-full@sha256:" in dockerfile
    assert "TORCH_VERSION=2.11.0" in dockerfile
    assert "TRITON_VERSION=3.6.0" in dockerfile
    assert "FLYDSL_VERSION=0.2.4" in dockerfile


def test_ci_is_read_only_pinned_and_covers_release_checks() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts/ci-static.sh").read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "pull_request:" in workflow
    assert "submodules: recursive" in workflow
    assert (
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803"
        in workflow
    )
    assert (
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
        in workflow
    )
    assert "self-hosted" not in workflow
    for check in (
        "ruff check",
        "shellcheck",
        "./r9v validate",
        "git submodule status",
    ):
        assert check in script


def test_qwen_launch_pins_measured_rocm_dispatch_policy() -> None:
    launcher = (ROOT / "scripts/launch.sh").read_text(encoding="utf-8")

    assert "--env NCCL_ALGO=Ring" in launcher
    assert "--env NCCL_PROTO=Simple" in launcher
    assert "--env VLLM_ROCM_USE_AITER=1" in launcher
    assert "--env VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1" in launcher
    for subsystem in (
        "LINEAR",
        "MHA",
        "MLA",
        "MOE",
        "RMSNORM",
        "FP8BMM",
        "FP4BMM",
    ):
        assert f"--env VLLM_ROCM_USE_AITER_{subsystem}=0" in launcher


def test_vllm_wheel_retains_r9v_provenance_notice() -> None:
    pyproject = (ROOT / "vendor/vllm/pyproject.toml").read_text(encoding="utf-8")
    manifest = (ROOT / "vendor/vllm/MANIFEST.in").read_text(encoding="utf-8")

    assert 'license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]' in pyproject
    assert "include THIRD_PARTY_NOTICES.md" in manifest


def test_runtime_source_pins_match_checked_out_submodules() -> None:
    runtime = json.loads(
        (ROOT / "runtimes/qwen38-flash-next-gfx1201-v1/runtime.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads(
        (ROOT / "release/sources.lock.json").read_text(encoding="utf-8")
    )
    expected = {
        "vllm_revision": ("vendor/vllm", "vllm"),
        "gguf_plugin_revision": (
            "vendor/vllm-gguf-plugin",
            "vllm_gguf_plugin",
        ),
        "kernel_revision": ("kernels/r9v-gfx1201", "r9v_gfx1201_kernels"),
    }

    for runtime_key, (relative_path, lock_key) in expected.items():
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT / relative_path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        assert runtime["source"][runtime_key] == revision
        assert lock["code"][lock_key]["release_revision"] == revision
