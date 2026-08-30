#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
base_image=${R9V_BASE_IMAGE:-r9v-vllm-qwen38-base:latest}
runtime_image=${R9V_IMAGE:-r9v-qwen38-flash-next:latest}
max_jobs=${R9V_MAX_JOBS:-8}
vllm_tag=$(git -C "$repo_root/vendor/vllm" describe --tags --abbrev=0)
vllm_revision=$(git -C "$repo_root/vendor/vllm" rev-parse --short=12 HEAD)
vllm_version=${R9V_VLLM_VERSION:-"${vllm_tag#v}+r9v.g${vllm_revision}"}

if ! docker buildx version >/dev/null 2>&1; then
    printf '%s\n' \
        'Docker Buildx is required by the vLLM Dockerfile.' \
        'Install the official docker/buildx plugin, then rerun this command.' >&2
    exit 1
fi

for required in \
    "$repo_root/vendor/vllm/docker/Dockerfile.r9v_rocm714" \
    "$repo_root/vendor/vllm-gguf-plugin/setup.py" \
    "$repo_root/kernels/r9v-gfx1201/README.md"; do
    [[ -f "$required" ]] || {
        printf 'Missing submodule content: %s\nRun: git submodule update --init --recursive\n' \
            "$required" >&2
        exit 1
    }
done

docker buildx build --load \
    --file "$repo_root/vendor/vllm/docker/Dockerfile.r9v_rocm714" \
    --target r9v-vllm-base \
    --build-arg R9V_VLLM_VERSION="$vllm_version" \
    --build-arg GFX_ARCH=gfx1201 \
    --build-arg MAX_JOBS="$max_jobs" \
    --tag "$base_image" \
    "$repo_root/vendor/vllm"

docker buildx build --load \
    --file "$repo_root/docker/Dockerfile.runtime" \
    --build-arg BASE_IMAGE="$base_image" \
    --build-arg GFX_ARCH=gfx1201 \
    --tag "$runtime_image" \
    "$repo_root"

printf 'Built %s\n' "$runtime_image"
