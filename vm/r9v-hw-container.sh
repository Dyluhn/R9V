#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# r9v-hw-container.sh — the ONLY command that exposes real hardware.
#
# Bare-metal container for hardware-qualified runs: passes /dev/kfd, the
# explicitly selected renderD nodes, and read-only PCI sysfs into the pinned
# ci/Dockerfile image, then runs the consolidated gates plus the r9v-hip GPU
# smoke test. Nothing in vm/ other than this file names a host device node.
# The dev VM (r9v-vm.sh) can never do this; its topology and results stay
# non-authoritative.
#
# Usage: r9v-hw-container.sh [--image-tag NAME] [-- <gate args>]
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
if [[ ${1:-} == '--image-tag' ]]; then
    IMAGE_TAG=${2:?--image-tag needs a value}
    shift 2
fi
if [[ ${1:-} == '--' ]]; then
    shift
fi

die() { printf 'r9v-hw-container: %s\n' "$*" >&2; exit 1; }

# The render nodes are explicit: no default is assumed, since passing the
# wrong GPU node would silently qualify the wrong hardware.
[[ -n ${R9V_RENDER_NODES:-} ]] \
    || die 'R9V_RENDER_NODES is required (e.g. R9V_RENDER_NODES="/dev/dri/renderD128")'

command -v docker >/dev/null 2>&1 \
    || die 'docker is required for bare-metal hardware runs'
command -v rsync >/dev/null 2>&1 \
    || die 'rsync is required to materialize a standalone source snapshot'

# Split without word splitting on unquoted expansion: read into an array.
RENDER_NODES=()
read -r -a RENDER_NODES <<<"$R9V_RENDER_NODES"
(( ${#RENDER_NODES[@]} > 0 )) \
    || die 'R9V_RENDER_NODES must name at least one render node'

[[ -e /dev/kfd ]] || die '/dev/kfd missing: no AMD GPU node on this host'
device_args=(--device /dev/kfd)
node=""
for node in "${RENDER_NODES[@]}"; do
    [[ $node =~ ^/dev/dri/renderD[0-9]+$ ]] \
        || die "invalid render node: $node (want /dev/dri/renderD<number>)"
    [[ -e $node ]] || die "render node missing: $node (check R9V_RENDER_NODES)"
    device_args+=(--device "$node")
done

printf 'bare-metal hardware run: kfd + %s; PCI sysfs read-only\n' \
    "${RENDER_NODES[*]}"
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
