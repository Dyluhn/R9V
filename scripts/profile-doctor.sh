#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
profile_id=${R9V_PROFILE_ID:-qwen38-flash-next/ud-iq4-xs/dual-r9700-128k}
expected_gpus=1
[[ $profile_id != qwen38-flash-next/* ]] || expected_gpus=2

failures=0
check_file() {
    if [[ -e $1 ]]; then
        printf 'PASS %s\n' "$1"
    else
        printf 'FAIL missing %s\n' "$1" >&2
        failures=$((failures + 1))
    fi
}

command -v docker >/dev/null || { printf 'FAIL docker is not installed\n' >&2; failures=$((failures + 1)); }
check_file /dev/kfd
check_file /dev/dri

gfx1201_count=0
for properties in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [[ -r $properties ]] || continue
    gfx_target=$(awk '$1 == "gfx_target_version" {print $2}' "$properties")
    [[ ${gfx_target:-0} == 120001 ]] || continue
    gfx1201_count=$((gfx1201_count + 1))
done
if ((gfx1201_count >= expected_gpus)); then
    printf 'PASS gfx1201 GPUs: %s (need %s)\n' "$gfx1201_count" "$expected_gpus"
else
    printf 'FAIL gfx1201 GPUs: %s (need %s)\n' "$gfx1201_count" "$expected_gpus" >&2
    failures=$((failures + 1))
fi

case "$profile_id" in
    qwen38-flash-next/*)
        check_file "$repo_root/vendor/vllm/docker/Dockerfile.rocm"
        check_file "$repo_root/vendor/vllm-gguf-plugin/setup.py"
        check_file "$repo_root/kernels/r9v-gfx1201/README.md"
        printf 'NOTE rank order is semantic; confirm R9V_VISIBLE_DEVICES before launch.\n'
        ;;
esac

if [[ -n ${R9V_MODEL_DIR:-} ]]; then
    "$repo_root/r9v" verify "$profile_id" --model-dir "$R9V_MODEL_DIR"
fi

((failures == 0)) || exit 1
printf 'PASS doctor %s (read-only; no GPU workload launched)\n' "$profile_id"
