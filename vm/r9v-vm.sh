#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# r9v-vm.sh — audited control interface for the hardware-blind R9V dev VM.
#
# Usage: r9v-vm.sh {check-deps|fetch|up|status|ssh|sync|test|destroy [--with-state --yes]}
#
# Design (see vm/README.md):
# - QEMU/KVM with a generic named CPU, fixed topology, user-mode networking.
# - KVM only when /dev/kvm is usable; TCG fallback otherwise.
# - No host path shares, no sysfs shares, no GPU or other host devices.
# - Sources reach the guest only as a deliberate copy to guest disk over SSH.
# - Keys, images, overlays, and pidfiles live in XDG state, outside the repo.
# - `test` runs the pinned ci/Dockerfile gates inside the guest; guest
#   topology and results are non-authoritative and unable to qualify
#   performance. Only vm/r9v-hw-container.sh touches real hardware.
set -euo pipefail

VM_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$VM_DIR/.." && pwd)
# shellcheck disable=SC1091
. "$VM_DIR/vm-config.sh"
# shellcheck disable=SC1091
. "$VM_DIR/image.pin"

STATE_DIR=${R9V_VM_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/$R9V_VM_STATE_DEFAULT_SUBDIR}
BASE_IMG=$STATE_DIR/base-amd64.img
OVERLAY_IMG=$STATE_DIR/disk.qcow2
SEED_ISO=$STATE_DIR/seed.iso
SSH_KEY=$STATE_DIR/id_ed25519
PIDFILE=$STATE_DIR/qemu.pid
KNOWN_HOSTS=$STATE_DIR/known_hosts
SERIAL_LOG=$STATE_DIR/serial.log
STATE_MARKER=.r9v-vm-state

# Bounds for post-boot readiness probing. Fixed, not flags.
SSH_WAIT_SECS=300
CLOUD_INIT_WAIT_SECS=600

SSH_OPTS=(-i "$SSH_KEY" -p "$R9V_VM_SSH_PORT"
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile="$KNOWN_HOSTS")

disclaimer() {
    printf '%s\n' \
        'NOTE: this VM is hardware-blind by design. Its topology and any' \
        'numbers produced inside it are non-authoritative and unable to' \
        'qualify performance. Qualification happens on bare metal only.'
}

die() { printf 'r9v-vm: %s\n' "$*" >&2; exit 1; }

accel_for_host() {
    # KVM only when the host actually offers a usable node; TCG otherwise.
    if [[ -r /dev/kvm && -w /dev/kvm ]]; then
        printf 'kvm\n'
    else
        printf 'tcg\n'
    fi
}

check_deps() {
    local missing=0
    local tool
    for tool in qemu-system-x86_64 ssh rsync sha256sum ssh-keygen qemu-img curl realpath git; do
        if command -v "$tool" >/dev/null 2>&1; then
            printf 'ok   %s\n' "$tool"
        else
            printf 'miss %s\n' "$tool"
            missing=1
        fi
    done
    if command -v genisoimage >/dev/null 2>&1; then
        printf 'ok   genisoimage (seed ISO)\n'
    elif command -v cloud-localds >/dev/null 2>&1; then
        printf 'ok   cloud-localds (seed ISO)\n'
    else
        printf 'miss genisoimage-or-cloud-localds (seed ISO)\n'
        missing=1
    fi
    printf 'accel %s\n' "$(accel_for_host)"
    return "$missing"
}

# Print the validated QEMU pid, or fail. A pidfile pointing at a dead,
# non-numeric, or non-VM process (stale or recycled pid) is not running:
# /proc/<pid>/cmdline must belong to qemu for this VM.
vm_pid() {
    local pid cmdline
    [[ -f $PIDFILE ]] || return 1
    pid=$(cat -- "$PIDFILE")
    [[ $pid =~ ^[0-9]+$ ]] || return 1
    [[ -r /proc/"$pid"/cmdline ]] || return 1
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    [[ $cmdline == *qemu* ]] || return 1
    [[ $cmdline == *r9v-dev-vm* ]] || return 1
    printf '%s' "$pid"
}

vm_running() {
    vm_pid >/dev/null 2>&1
}

ensure_state_dir() {
    local state_canon repo_canon
    state_canon=$(realpath -m -- "$STATE_DIR")
    repo_canon=$(realpath -m -- "$REPO_ROOT")
    case "$state_canon/" in
        "$repo_canon/"*) die "state directory must stay outside the repo checkout: $STATE_DIR" ;;
    esac
    if [[ -e $STATE_DIR ]]; then
        [[ -d $STATE_DIR ]] || die "state path exists but is not a directory: $STATE_DIR"
        if [[ -f $STATE_DIR/$STATE_MARKER ]]; then
            [[ $(cat -- "$STATE_DIR/$STATE_MARKER") == r9v-dev-vm ]] \
                || die "state marker mismatch: $STATE_DIR/$STATE_MARKER"
            return 0
        fi
        # Never claim an existing directory containing unrelated files. That
        # would make a later --with-state deletion unsafe even with a marker.
        if [[ -n $(find "$STATE_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
            die "refusing to claim non-empty unmarked state directory: $STATE_DIR"
        fi
    else
        mkdir -p -- "$STATE_DIR"
    fi
    printf 'r9v-dev-vm\n' >"$STATE_DIR/$STATE_MARKER"
}

verify_base() {
    printf '%s  %s\n' "$R9V_VM_IMAGE_SHA256" "$1" | sha256sum -c - --status
}

# Fetch the pinned base image into $BASE_IMG (a ~596MB download).
# A verified existing base is never overwritten or re-downloaded. A fresh
# download goes to a temporary file in the state directory, is hash-verified,
# and is then atomically moved into place.
cmd_fetch() {
    ensure_state_dir
    if [[ -f $BASE_IMG ]]; then
        if verify_base "$BASE_IMG"; then
            printf 'base image already verified, keeping %s\n' "$BASE_IMG"
            return 0
        fi
        local invalid_base
        invalid_base=$BASE_IMG.invalid.$(date -u +%Y%m%dT%H%M%SZ)
        printf 'existing base image failed verification; preserving it as %s\n' \
            "$invalid_base" >&2
        mv -- "$BASE_IMG" "$invalid_base"
    fi
    local tmp
    tmp=$(mktemp "$STATE_DIR/.base-img.XXXXXX.tmp")
    printf 'downloading %s (~596MB) to %s\n' "$R9V_VM_IMAGE_URL" "$BASE_IMG"
    if ! curl -fSL --retry 3 -o "$tmp" "$R9V_VM_IMAGE_URL"; then
        rm -f -- "$tmp"
        die 'base image download failed'
    fi
    if ! verify_base "$tmp"; then
        rm -f -- "$tmp"
        die 'downloaded base image hash mismatch; refusing to install'
    fi
    mv -f -- "$tmp" "$BASE_IMG"
    printf 'installed verified base image at %s\n' "$BASE_IMG"
}

ensure_overlay() {
    # The overlay is a qcow2 delta backed by the verified base image, sized
    # to the fixed disk size. A blank overlay with no backing file is
    # unbootable, so an existing overlay without the expected backing file
    # is refused rather than booted.
    if [[ -f $OVERLAY_IMG ]]; then
        qemu-img info "$OVERLAY_IMG" | grep -q -F "backing file: $BASE_IMG" \
            || die "overlay $OVERLAY_IMG has no backing file $BASE_IMG; remove it and re-run up"
        return 0
    fi
    qemu-img create -f qcow2 -F qcow2 -b "$BASE_IMG" "$OVERLAY_IMG"
    qemu-img resize "$OVERLAY_IMG" "${R9V_VM_DISK_GB}G"
}

cmd_status() {
    if vm_running; then
        printf 'up accel=%s cpus=%s ram_mb=%s cpu=%s ssh_port=%s\n' \
            "$(accel_for_host)" "$R9V_VM_CPUS" "$R9V_VM_RAM_MB" \
            "$R9V_VM_CPU_MODEL" "$R9V_VM_SSH_PORT"
    else
        printf 'down\n'
    fi
    disclaimer
}

write_cloud_config() {
    local pubkey seed_dir
    pubkey=$(cat -- "${SSH_KEY}.pub")
    seed_dir=$STATE_DIR/seed
    mkdir -p -- "$seed_dir"
    cat >"$seed_dir/user-data" <<EOF
#cloud-config
users:
  - name: $R9V_VM_GUEST_USER
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $pubkey
package_update: true
packages:
  - docker.io
  - rsync
runcmd:
  - [ systemctl, enable, --now, docker ]
  - [ usermod, -aG, docker, $R9V_VM_GUEST_USER ]
EOF
    printf 'instance-id: r9v-dev-vm\nlocal-hostname: r9v-dev-vm\n' \
        >"$seed_dir/meta-data"
    if command -v genisoimage >/dev/null 2>&1; then
        genisoimage -output "$SEED_ISO" -volid cidata -joliet -rock \
            "$seed_dir/user-data" "$seed_dir/meta-data" >/dev/null
    else
        cloud-localds "$SEED_ISO" "$seed_dir/user-data" "$seed_dir/meta-data" \
            >/dev/null
    fi
}

wait_for_guest_ssh() {
    local deadline=$((SECONDS + SSH_WAIT_SECS))
    while ((SECONDS < deadline)); do
        if ssh "${SSH_OPTS[@]}" "${R9V_VM_GUEST_USER}@127.0.0.1" true \
            >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

wait_for_cloud_init() {
    local deadline=$((SECONDS + CLOUD_INIT_WAIT_SECS))
    local status
    while ((SECONDS < deadline)); do
        status=$(ssh "${SSH_OPTS[@]}" "${R9V_VM_GUEST_USER}@127.0.0.1" \
            "cloud-init status 2>/dev/null" 2>/dev/null || true)
        case "$status" in
            *"status: done"*) return 0 ;;
        esac
        sleep 10
    done
    return 1
}

cmd_up() {
    ensure_state_dir
    [[ -r $BASE_IMG ]] \
        || die "base image missing: run '$0 fetch' first ($BASE_IMG)"
    verify_base "$BASE_IMG" \
        || die 'base image hash mismatch; refusing to boot'
    ensure_overlay
    if [[ ! -f $SSH_KEY ]]; then
        ssh-keygen -t ed25519 -N '' -f "$SSH_KEY" -C r9v-dev-vm >/dev/null
        chmod 600 -- "$SSH_KEY"
    fi
    write_cloud_config
    if vm_running; then
        printf 'already up (pid %s)\n' "$(vm_pid)"
    else
        # A pidfile that fails validation is stale; never boot twice.
        rm -f -- "$PIDFILE"
        local accel
        accel=$(accel_for_host)
        printf 'starting with accel=%s cpu=%s cpus=%s ram=%sMB disk=%sG\n' \
            "$accel" "$R9V_VM_CPU_MODEL" "$R9V_VM_CPUS" "$R9V_VM_RAM_MB" \
            "$R9V_VM_DISK_GB"
        # Deliberately absent: host path shares, sysfs shares, and host
        # devices of any kind. The guest sees only virtual disk, virtual NIC,
        # and the cloud-init seed volume.
        qemu-system-x86_64 \
            -name r9v-dev-vm \
            -machine "$R9V_VM_MACHINE" \
            -accel "$accel" \
            -cpu "$R9V_VM_CPU_MODEL" \
            -smp "$R9V_VM_CPUS" \
            -m "$R9V_VM_RAM_MB" \
            -drive "file=$OVERLAY_IMG,format=qcow2,if=virtio" \
            -drive "file=$SEED_ISO,format=raw,if=virtio,readonly=on" \
            -netdev "user,id=net0,hostfwd=tcp::${R9V_VM_SSH_PORT}-:22" \
            -device virtio-net-pci,netdev=net0 \
            -serial "file:$SERIAL_LOG" \
            -nographic \
            -daemonize \
            -pidfile "$PIDFILE"
    fi
    wait_for_guest_ssh \
        || die "guest SSH not ready; see serial log $SERIAL_LOG"
    wait_for_cloud_init \
        || die "cloud-init not done; inspect with '$0 ssh cloud-init status' (serial log $SERIAL_LOG)"
    printf 'up; ssh with: %s ssh\n' "$0"
}

cmd_ssh() {
    vm_running || die 'vm is not up; run up first'
    exec ssh "${SSH_OPTS[@]}" "${R9V_VM_GUEST_USER}@127.0.0.1" "$@"
}

cmd_sync() {
    vm_running || die 'vm is not up; run up first'
    # Deliberate copy to guest disk over SSH. Never a shared mount. The source
    # is first materialized as a standalone shallow Git snapshot because this
    # checkout may itself be a linked worktree: copying its `.git` pointer
    # would leave an unusable absolute host path in the guest, while omitting
    # Git metadata would break the diff, policy, and submodule gates.
    local snapshot
    snapshot=$("$REPO_ROOT/scripts/make-source-snapshot.sh" "$REPO_ROOT")
    ssh "${SSH_OPTS[@]}" "${R9V_VM_GUEST_USER}@127.0.0.1" \
        "mkdir -p '$R9V_VM_GUEST_SRC'"
    rsync -az --delete \
        -e "ssh -i $SSH_KEY -p $R9V_VM_SSH_PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS" \
        --exclude /target \
        --exclude /.venv \
        --exclude /__pycache__ \
        --exclude /.pytest_cache \
        --exclude /.ruff_cache \
        --exclude /*.qcow2 \
        "$snapshot/" "${R9V_VM_GUEST_USER}@127.0.0.1:${R9V_VM_GUEST_SRC}/"
    rm -rf -- "$snapshot"
    printf 'synced to guest disk %s\n' "$R9V_VM_GUEST_SRC"
}

guest_sh() {
    ssh "${SSH_OPTS[@]}" "${R9V_VM_GUEST_USER}@127.0.0.1" \
        "bash -s" <<<"$1"
}

cmd_test() {
    vm_running || die 'vm is not up; run up first'
    disclaimer
    cmd_sync
    # ci/Dockerfile carries no source (no COPY). The guest-local snapshot is
    # mounted read-only at /source, copied to the container's writable layer,
    # and built with an isolated target directory. This keeps the synchronized
    # snapshot clean while allowing generator gates to prove they make no diff.
    guest_sh "set -euo pipefail
mkdir -p /tmp/r9v-guest-target
cd '$R9V_VM_GUEST_SRC'
if ! docker image inspect r9v-ci:guest >/dev/null 2>&1; then
  printf '%s\\n' 'first build may download/use approximately 30 GiB inside the guest disk at $OVERLAY_IMG'
fi
docker build -f ci/Dockerfile -t r9v-ci:guest ci
docker run --rm \
  -v '$R9V_VM_GUEST_SRC:/source:ro' \
  -e CARGO_TARGET_DIR=/tmp/r9v-guest-target \
  -e CARGO_INCREMENTAL=0 \
  r9v-ci:guest bash -lc 'cp -a /source/. /workspace/ && ./scripts/ci-gates.sh all'"
}

# Canonicalize the state directory and refuse every dangerous value before
# any deletion: empty, relative, root, the home directory, the repo checkout,
# or any ancestor of the repo (removing it would wipe the project). The
# marker file proves this VM created the directory.
safe_state_canon() {
    local canon home_canon repo_canon marker
    [[ -n ${STATE_DIR:-} ]] || die 'refusing to remove state: empty state dir'
    canon=$(realpath -m -- "$STATE_DIR") \
        || die 'refusing to remove state: cannot canonicalize state dir'
    [[ $canon == /* ]] \
        || die 'refusing to remove state: state dir is not absolute'
    [[ $canon != / ]] \
        || die 'refusing to remove state: state dir is /'
    home_canon=$(realpath -m -- "$HOME")
    repo_canon=$(realpath -m -- "$REPO_ROOT")
    [[ $canon != "$home_canon" ]] \
        || die 'refusing to remove state: state dir is the home directory'
    [[ $canon != "$repo_canon" ]] \
        || die 'refusing to remove state: state dir is the repo checkout'
    case "$canon/" in
        "$repo_canon/"*)
            die 'refusing to remove state: state dir is inside the repo checkout'
            ;;
    esac
    case "$repo_canon/" in
        "$canon/"*)
            die 'refusing to remove state: state dir contains the repo checkout'
            ;;
    esac
    marker=$STATE_DIR/$STATE_MARKER
    [[ -f $marker ]] \
        || die 'refusing to remove state: missing VM marker (not created by r9v-vm)'
    [[ $(cat -- "$marker") == r9v-dev-vm ]] \
        || die 'refusing to remove state: VM marker mismatch (not created by r9v-vm)'
    printf '%s' "$canon"
}

cmd_destroy() {
    local with_state=0 yes=0
    local arg
    for arg in "$@"; do
        case $arg in
            --with-state) with_state=1 ;;
            --yes) yes=1 ;;
            *) die "unknown destroy flag: $arg" ;;
        esac
    done
    ((yes == 0 || with_state == 1)) \
        || die '--yes is valid only with --with-state'
    if [[ ! -e $STATE_DIR ]]; then
        printf 'already down; no state directory (%s)\n' "$STATE_DIR"
        return 0
    fi
    local canon
    canon=$(safe_state_canon)
    if vm_running; then
        local pid deadline
        pid=$(vm_pid)
        kill -- "$pid"
        deadline=$((SECONDS + 30))
        while kill -0 "$pid" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 1
        done
        kill -0 "$pid" 2>/dev/null \
            && die "VM process $pid did not stop; state was not removed"
        rm -f -- "$PIDFILE"
        printf 'stopped\n'
    else
        rm -f -- "$PIDFILE"
        printf 'already down\n'
    fi
    if ((with_state == 1)); then
        ((yes == 1)) \
            || die 'refusing to remove state without --yes (destroy --with-state --yes)'
        rm -rf -- "$canon"
        printf 'state removed (%s)\n' "$canon"
    else
        printf 'state kept (%s); re-run with --with-state --yes to remove it\n' \
            "$STATE_DIR"
    fi
}

usage() {
    cat <<EOF
usage: $(basename -- "$0") {check-deps|fetch|up|status|ssh [args]|sync|test|destroy [--with-state --yes]}
EOF
}

main() {
    local cmd=${1:-}
    case $cmd in
        check-deps) check_deps ;;
        fetch) cmd_fetch ;;
        up) cmd_up ;;
        status) cmd_status ;;
        ssh) shift; cmd_ssh "$@" ;;
        sync) cmd_sync ;;
        test) cmd_test ;;
        destroy) shift; cmd_destroy "$@" ;;
        *) usage >&2; exit 2 ;;
    esac
}

main "$@"
