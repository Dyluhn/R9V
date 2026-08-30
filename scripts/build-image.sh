#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
base_image=${R9V_BASE_IMAGE:-r9v-vllm-qwen38-base:latest}
runtime_image=${R9V_IMAGE:-r9v-qwen38-flash-next:latest}
max_jobs=${R9V_MAX_JOBS:-8}

for required in \
    "$repo_root/vendor/vllm/docker/Dockerfile.rocm" \
    "$repo_root/vendor/vllm-gguf-plugin/setup.py" \
    "$repo_root/kernels/r9v-gfx1201/README.md"; do
    [[ -f "$required" ]] || {
        printf 'Missing submodule content: %s\nRun: git submodule update --init --recursive\n' \
            "$required" >&2
        exit 1
    }
done

docker build \
    --file "$repo_root/vendor/vllm/docker/Dockerfile.rocm" \
    --target vllm-openai \
    --build-arg REMOTE_VLLM=0 \
    --build-arg ARG_PYTORCH_ROCM_ARCH=gfx1201 \
    --build-arg max_jobs="$max_jobs" \
    --tag "$base_image" \
    "$repo_root/vendor/vllm"

docker build \
    --file "$repo_root/docker/Dockerfile.runtime" \
    --build-arg BASE_IMAGE="$base_image" \
    --build-arg GFX_ARCH=gfx1201 \
    --tag "$runtime_image" \
    "$repo_root"

printf 'Built %s\n' "$runtime_image"
