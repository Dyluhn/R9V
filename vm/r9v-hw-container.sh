#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# r9v-hw-container.sh — the ONLY command that exposes real hardware.
#
# Bare-metal container for hardware-qualified runs: passes /dev/kfd, all
# required KFD GPU topology render nodes (both explicit targets and necessary
# support nodes for HSA initialization), and read-only PCI sysfs into the
# pinned ci/Dockerfile image. ROCR_VISIBLE_DEVICES is set to restrict HIP
# visibility strictly to the selected targets.
#
# Usage: r9v-hw-container.sh [--image-tag NAME] [--dry-run] [-- <gate args>]
# Env: R9V_RENDER_NODES (required, space-separated render nodes, e.g.
#      R9V_RENDER_NODES="/dev/dri/renderD128 /dev/dri/renderD129").
#      Each entry must match /dev/dri/renderD<number>.
#
# With no gate args, runs the full gates (./scripts/ci-gates.sh all) and then
# the r9v-hip GPU smoke test. Never runs on this machine by accident: without
# R9V_RENDER_NODES and /dev/kfd it refuses to start. This script performs no
# hardware run itself; it only defines the audited container invocation.
set -euo pipefail

HW_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HW_DIR/.." && pwd)

IMAGE_TAG=r9v-ci:hw
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-tag)
            IMAGE_TAG=${2:?--image-tag needs a value}
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

die() { printf 'r9v-hw-container: %s\n' "$*" >&2; exit 1; }

# The render nodes are explicit: no default is assumed, since passing the
# wrong GPU node would silently qualify the wrong hardware.
[[ -n ${R9V_RENDER_NODES:-} ]] \
    || die 'R9V_RENDER_NODES is required (e.g. R9V_RENDER_NODES="/dev/dri/renderD128")'

# Split without word splitting on unquoted expansion: read into an array.
RENDER_NODES=()
read -r -a RENDER_NODES <<<"${R9V_RENDER_NODES//,/ }"
(( ${#RENDER_NODES[@]} > 0 )) \
    || die 'R9V_RENDER_NODES must name at least one render node'

# Validate syntax and check for duplicate target render nodes.
declare -A seen_target_nodes=()
for node in "${RENDER_NODES[@]}"; do
    [[ $node =~ ^/dev/dri/renderD[0-9]+$ ]] \
        || die "invalid render node: $node (want /dev/dri/renderD<number>)"
    if [[ -n "${seen_target_nodes[$node]:-}" ]]; then
        die "duplicate render node in R9V_RENDER_NODES: $node"
    fi
    seen_target_nodes["$node"]=1
    if [[ ${R9V_TEST_SKIP_DEV_CHECK:-0} -ne 1 ]]; then
        [[ -e $node ]] || die "render node missing: $node (check R9V_RENDER_NODES)"
    fi
done

if [[ $DRY_RUN -eq 0 ]]; then
    command -v docker >/dev/null 2>&1 \
        || die 'docker is required for bare-metal hardware runs'
    command -v rsync >/dev/null 2>&1 \
        || die 'rsync is required to materialize a standalone source snapshot'
fi

KFD_DEV=${R9V_TEST_KFD_DEV:-/dev/kfd}
if [[ ${R9V_TEST_SKIP_DEV_CHECK:-0} -ne 1 ]]; then
    [[ -e $KFD_DEV ]] || die "$KFD_DEV missing: no AMD GPU node on this host"
fi

# Discover KFD topology nodes deterministically.
if [[ -z "${KFD_TOPOLOGY_ROOT:-}" ]]; then
    if [[ -d /sys/devices/virtual/kfd/kfd/topology/nodes ]]; then
        KFD_TOPOLOGY_ROOT=/sys/devices/virtual/kfd/kfd/topology/nodes
    elif [[ -d /sys/class/kfd/kfd/topology/nodes ]]; then
        KFD_TOPOLOGY_ROOT=/sys/class/kfd/kfd/topology/nodes
    else
        KFD_TOPOLOGY_ROOT=/sys/devices/virtual/kfd/kfd/topology/nodes
    fi
fi
[[ -d "$KFD_TOPOLOGY_ROOT" ]] || die "KFD topology directory not found: $KFD_TOPOLOGY_ROOT"

# Sort node directory IDs numerically (e.g. 0, 1, 2, ... 10).
sorted_node_ids=()
while IFS= read -r node_id; do
    [[ -n "$node_id" ]] && sorted_node_ids+=("$node_id")
done < <(
    for p in "$KFD_TOPOLOGY_ROOT"/*; do
        [[ -d "$p" ]] || continue
        base=$(basename -- "$p")
        [[ "$base" =~ ^[0-9]+$ ]] || continue
        printf '%d\n' "$base"
    done | sort -n
)

(( ${#sorted_node_ids[@]} > 0 )) || die "no topology node directories found in $KFD_TOPOLOGY_ROOT"

declare -A render_to_ordinal=()
declare -A seen_minors=()
topology_gpu_render_nodes=()
gpu_ordinal=0

for node_id in "${sorted_node_ids[@]}"; do
    props_file="$KFD_TOPOLOGY_ROOT/$node_id/properties"
    [[ -r "$props_file" ]] || die "topology node $node_id properties missing or unreadable: $props_file"

    simd_count=""
    drm_render_minor=""
    while read -r key value rest; do
        case "$key" in
            simd_count) simd_count="$value" ;;
            drm_render_minor) drm_render_minor="$value" ;;
        esac
    done < "$props_file"

    if [[ ! "$simd_count" =~ ^[0-9]+$ ]]; then
        die "topology node $node_id missing or invalid simd_count"
    fi

    if (( simd_count > 0 )); then
        if [[ ! "$drm_render_minor" =~ ^[0-9]+$ ]] || (( drm_render_minor <= 0 )); then
            die "GPU topology node $node_id has invalid drm_render_minor: $drm_render_minor"
        fi

        if [[ -n "${seen_minors[$drm_render_minor]:-}" ]]; then
            die "duplicate drm_render_minor $drm_render_minor in topology node $node_id"
        fi
        seen_minors["$drm_render_minor"]=1

        node_path="/dev/dri/renderD$drm_render_minor"
        if [[ ${R9V_TEST_SKIP_DEV_CHECK:-0} -ne 1 ]]; then
            [[ -e "$node_path" ]] || die "KFD GPU topology references render node $node_path, but it does not exist on host"
        fi

        topology_gpu_render_nodes+=("$node_path")
        render_to_ordinal["$node_path"]=$gpu_ordinal
        gpu_ordinal=$((gpu_ordinal + 1))
    fi
done

(( ${#topology_gpu_render_nodes[@]} > 0 )) \
    || die "no GPU nodes found in KFD topology ($KFD_TOPOLOGY_ROOT)"

# Derive HSA GPU ordinals for requested target nodes in explicit user order.
target_ordinals=()
for node in "${RENDER_NODES[@]}"; do
    if [[ ! -v render_to_ordinal["$node"] ]]; then
        die "requested target render node $node does not resolve to any KFD GPU topology node with simd_count>0"
    fi
    target_ordinals+=("${render_to_ordinal[$node]}")
done

rocr_visible_devices=$(IFS=,; echo "${target_ordinals[*]}")

# Identify support nodes: existing KFD GPU render nodes needed for HSA init,
# but not part of the explicit target allowlist.
support_nodes=()
for node in "${topology_gpu_render_nodes[@]}"; do
    if [[ -z "${seen_target_nodes[$node]:-}" ]]; then
        support_nodes+=("$node")
    fi
done

device_args=(--device "$KFD_DEV")
for node in "${RENDER_NODES[@]}"; do
    device_args+=(--device "$node")
done
for node in "${support_nodes[@]}"; do
    device_args+=(--device "$node")
done

if (( ${#support_nodes[@]} > 0 )); then
    support_display="${support_nodes[*]}"
else
    support_display="none"
fi

printf 'bare-metal hardware run: kfd + targets [%s] + support [%s]; visible ordinals: %s; PCI sysfs read-only\n' \
    "${RENDER_NODES[*]}" \
    "$support_display" \
    "$rocr_visible_devices"
for node in "${RENDER_NODES[@]}"; do
    printf '  target: %s -> HSA ordinal %d\n' "$node" "${render_to_ordinal[$node]}"
done
for node in "${support_nodes[@]}"; do
    printf '  support: %s -> HSA ordinal %d\n' "$node" "${render_to_ordinal[$node]}"
done

if [[ $DRY_RUN -eq 1 ]]; then
    gate_args=("$@")
    if (( ${#gate_args[@]} == 0 )); then
        gate_args=(./scripts/ci-gates.sh hardware)
    else
        gate_args=(./scripts/ci-gates.sh "${gate_args[@]}")
    fi
    echo "DRY_RUN: docker run --rm" \
        "${device_args[@]}" \
        "--volume /sys/bus/pci:/sys/bus/pci:ro" \
        "--volume <SOURCE_SNAPSHOT>:/source:ro" \
        "-e ROCR_VISIBLE_DEVICES=$rocr_visible_devices" \
        "-e R9V_REQUIRE_GPU=1" \
        "-e R9V_GPU_LANE=1" \
        "-e CARGO_TARGET_DIR=/tmp/r9v-hw-target" \
        "-e CARGO_INCREMENTAL=0" \
        "-e XDG_CACHE_HOME=/tmp/r9v-hw-cache" \
        "$IMAGE_TAG" bash -lc \
        "'cp -a --no-preserve=ownership /source/. /workspace/ && exec \"\$@\"' bash" \
        "${gate_args[@]}"
    exit 0
fi

docker build -f "$REPO_ROOT/ci/Dockerfile" -t "$IMAGE_TAG" "$REPO_ROOT/ci"

SOURCE_SNAPSHOT=$("$REPO_ROOT/scripts/make-source-snapshot.sh" "$REPO_ROOT")
cleanup() {
    if [[ $SOURCE_SNAPSHOT == /tmp/r9v-vm-source.* && -d $SOURCE_SNAPSHOT ]]; then
        rm -rf -- "$SOURCE_SNAPSHOT"
    fi
}
trap cleanup EXIT

# The source mounts read-only at /source and is copied into the container's
# writable layer. Builds and generators therefore cannot mutate the checkout.
run_hw() {
    docker run --rm \
        "${device_args[@]}" \
        --volume /sys/bus/pci:/sys/bus/pci:ro \
        --volume "$SOURCE_SNAPSHOT:/source:ro" \
        -e ROCR_VISIBLE_DEVICES="$rocr_visible_devices" \
        -e R9V_REQUIRE_GPU=1 \
        -e R9V_GPU_LANE=1 \
        -e CARGO_TARGET_DIR=/tmp/r9v-hw-target \
        -e CARGO_INCREMENTAL=0 \
        -e XDG_CACHE_HOME=/tmp/r9v-hw-cache \
        "$IMAGE_TAG" bash -lc \
        'cp -a --no-preserve=ownership /source/. /workspace/ && exec "$@"' bash "$@"
}

if (($# == 0)); then
    run_hw ./scripts/ci-gates.sh hardware
else
    run_hw ./scripts/ci-gates.sh "$@"
fi
