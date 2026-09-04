# SPDX-License-Identifier: Apache-2.0
"""Static guards for the hardware-blind R9V development VM definition.

The VM must stay hardware-blind by construction: a generic named CPU (never
the host model), fixed topology, KVM only with a TCG fallback, no host path
or sysfs shares, no GPU nodes, sources synced to guest disk over SSH, and
results explicitly marked non-authoritative. Only vm/r9v-hw-container.sh may
name host devices. These tests read the checked-in files; they download,
install, boot, and start nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VM_DIR = ROOT / "vm"
PIN = VM_DIR / "image.pin"
CONFIG = VM_DIR / "vm-config.sh"
CONTROL = VM_DIR / "r9v-vm.sh"
HW = VM_DIR / "r9v-hw-container.sh"
README = VM_DIR / "README.md"
GATES = ROOT / "scripts" / "ci-gates.sh"
SNAPSHOT = ROOT / "scripts" / "make-source-snapshot.sh"
DOCKERFILE = ROOT / "ci" / "Dockerfile"
CPU_ONLY = ROOT / ".github" / "workflows" / "cpu-only.yml"

# Files that define and launch the guest. README documents the hw script too,
# so device names there are descriptive, not configuration; the hw script is
# the deliberate sole exception and is asserted separately.
VM_LAUNCH_FILES = (PIN, CONFIG, CONTROL)

PINNED_URL = (
    "https://cloud-images.ubuntu.com/releases/noble/"
    "release-20260826/ubuntu-24.04-server-cloudimg-amd64.img"
)
PINNED_SHA256 = (
    "d0fe84bb5f80853425fa6be28e2c106f30104c3cfe8611933f2e65c9b63f0e30"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_vm_files_exist() -> None:
    for path in (*VM_LAUNCH_FILES, HW, README, GATES, SNAPSHOT, DOCKERFILE):
        assert path.is_file(), path


def test_image_pin_is_dated_immutable() -> None:
    pin = read(PIN)
    url = re.search(r"^R9V_VM_IMAGE_URL=(\S+)$", pin, re.MULTILINE).group(1)
    sha = re.search(r"^R9V_VM_IMAGE_SHA256=([0-9a-f]+)$", pin, re.MULTILINE).group(1)
    assert url == PINNED_URL, url
    assert sha == PINNED_SHA256, "base image SHA256 must match the pinned digest"
    assert "/current/" not in url, "must pin a dated directory, not `current`"
    assert "latest" not in url
    assert re.fullmatch(
        r"https://cloud-images\.ubuntu\.com/releases/noble/"
        r"release-\d{8}/ubuntu-24\.04-server-cloudimg-amd64\.img",
        url,
    ), url
    date = re.search(r"^R9V_VM_IMAGE_DATE=(\d{8})$", pin, re.MULTILINE).group(1)
    assert date in url, "pin date must match the release directory"
    assert "noble-server-cloudimg" not in url, "must use the official 24.04 release name"
    assert "24.04" in read(README) or "noble" in url


def test_fetch_verifies_before_installing() -> None:
    control = read(CONTROL)
    assert "fetch" in control
    assert "mktemp" in control, "fetch must download to a temporary file"
    assert "curl" in control
    assert "sha256sum" in control
    # Atomic install: a verified temp file moved into place.
    assert re.search(r"^ *mv -f -- ", control, re.MULTILINE), "fetch must atomically install"
    # A verified existing base is never overwritten.
    assert "already verified" in control
    assert "596" in control or "596" in read(README), "fetch must document the ~596MB download"


def test_overlay_is_backed_then_resized() -> None:
    control = read(CONTROL)
    assert re.search(r"qemu-img create .* -b ", control), "overlay must be backed by the base image"
    assert "backing file:" in control, "up must refuse an overlay with no backing file"
    assert "resize" in control, "overlay must be resized to the fixed disk size"
    assert "R9V_VM_DISK_GB" in control


def test_destroy_needs_explicit_confirmation() -> None:
    control = read(CONTROL)
    assert "--with-state" in control
    assert "--yes" in control
    assert "without --yes" in control


def test_destroy_rejects_dangerous_state_dirs() -> None:
    control = read(CONTROL)
    assert "realpath" in control, "destroy must canonicalize the state dir"
    assert ".r9v-vm-state" in control, "destroy requires the VM-owned marker file"
    assert "r9v-dev-vm" in control
    for guard in (
        "is /",
        "home directory",
        "repo checkout",
        "inside the repo checkout",
        "contains the repo",
    ):
        assert guard in control, guard
    assert "non-empty unmarked state directory" in control


def test_destroy_waits_for_qemu_before_removing_state() -> None:
    control = read(CONTROL)
    kill_at = control.index('kill -- "$pid"')
    wait_at = control.index('while kill -0 "$pid"')
    remove_at = control.index('rm -rf -- "$canon"')
    assert kill_at < wait_at < remove_at
    assert "state was not removed" in control


def test_pid_validation_rejects_stale_pids() -> None:
    control = read(CONTROL)
    assert re.search(r"\^?\[0-9\]\+\$", control), "pid must be validated numeric"
    assert "/proc/" in control and "cmdline" in control
    assert "*qemu*" in control and "*r9v-dev-vm*" in control


def test_up_captures_serial_and_waits_for_readiness() -> None:
    control = read(CONTROL)
    assert "-serial" in control and "serial.log" in control
    assert "cloud-init" in control
    assert re.search(r"wait_for_guest_ssh|SSH_WAIT", control), "up must wait for SSH"
    assert "mkdir -p" in control, "sync must pre-create the guest destination"


def test_guest_test_mounts_source_with_isolated_target() -> None:
    control = read(CONTROL)
    assert ":/source:ro" in control, "guest test must bind the source read-only"
    assert "cp -a /source/. /workspace/" in control
    assert "CARGO_TARGET_DIR" in control
    assert "approximately 30 GiB" in control


def test_dockerfile_carries_no_source() -> None:
    dockerfile = read(DOCKERFILE)
    assert not re.search(r"^COPY ", dockerfile, re.MULTILINE), "ci/Dockerfile must not COPY source"
    assert "WORKDIR /workspace" in dockerfile


def test_vm_uses_generic_cpu_never_host_model() -> None:
    launch = "\n".join(read(path) for path in VM_LAUNCH_FILES)
    assert "qemu64" in launch
    for forbidden in ("-cpu host", "host-passthrough", "host-model"):
        assert forbidden not in launch, forbidden


def test_vm_topology_is_fixed() -> None:
    config = read(CONFIG)
    assert re.search(r"^R9V_VM_CPUS=8$", config, re.MULTILINE)
    assert re.search(r"^R9V_VM_RAM_MB=16384$", config, re.MULTILINE)
    assert re.search(r"^R9V_VM_DISK_GB=80$", config, re.MULTILINE)


def test_vm_accelerator_is_kvm_with_tcg_fallback() -> None:
    control = read(CONTROL)
    assert "/dev/kvm" in control
    assert "tcg" in control
    assert "-accel" in control


def test_vm_has_no_host_shares_or_mounts() -> None:
    launch = "\n".join(read(path) for path in VM_LAUNCH_FILES)
    for forbidden in (
        "-virtfs",
        "virtio-9p",
        "mount_tag",
        "--volume",
        "9p",
        "/sys",
    ):
        assert forbidden not in launch, forbidden
    # The only `/home` path allowed is the documented guest-disk sync
    # destination; it is a copy target over SSH, never a shared mount.
    for i, line in enumerate(launch.splitlines(), start=1):
        if "/home" in line:
            assert line.startswith("R9V_VM_GUEST_SRC=/home/"), f"line {i}: {line}"


def test_vm_has_no_gpu_devices() -> None:
    launch = "\n".join(read(path) for path in VM_LAUNCH_FILES)
    for forbidden in ("/dev/kfd", "/dev/dri", "vfio", "hostdev", "renderD"):
        assert forbidden not in launch, forbidden


def test_gpu_passthrough_lives_only_in_hw_container() -> None:
    hw = read(HW)
    assert "/dev/kfd" in hw
    assert "renderD" in hw
    assert "/sys/bus/pci" in hw and ":ro" in hw
    # The VM control path must not duplicate any of it.
    assert "kfd" not in read(CONTROL)


def test_hw_container_requires_explicit_nodes() -> None:
    hw = read(HW)
    assert "R9V_RENDER_NODES is required" in hw
    # No silent default: the variable must never expand to a fallback node.
    assert "R9V_RENDER_NODES:-/" not in hw
    assert "R9V_RENDER_NODES-/" not in hw
    assert re.search(r"/dev/dri/renderD\[0-9\]", hw), "each node must match renderD<number>"
    assert "SC2206" not in hw, "hw container must not use unsafe word splitting"
    assert "read -r -a" in hw
    assert ":/source:ro" in hw, "hardware runs mount the source snapshot read-only"
    assert "cp -a /source/. /workspace/" in hw
    assert "CARGO_TARGET_DIR" in hw and "XDG_CACHE_HOME" in hw
    assert "hardware" in hw, "hardware default must include all gates plus GPU smoke"


def test_vm_results_marked_non_authoritative() -> None:
    for path in (README, CONTROL):
        text = read(path)
        assert "non-authoritative" in text, path
        assert "unable to qualify" in text, path


def test_vm_syncs_sources_over_ssh_to_guest_disk() -> None:
    control = read(CONTROL)
    assert "rsync" in control
    assert "ssh" in control
    assert "R9V_VM_GUEST_SRC" in control
    assert "make-source-snapshot.sh" in control
    snapshot = read(SNAPSHOT)
    assert "fetch -q --depth=1" in snapshot
    assert "submodule status --recursive" in snapshot
    assert "submodule does not match pinned gitlink" in snapshot
    assert "--exclude='.git'" in snapshot


def test_vm_state_and_keys_stay_outside_repo() -> None:
    control = read(CONTROL)
    assert "XDG_STATE_HOME" in control
    gitignore = read(ROOT / ".gitignore")
    assert ".qcow2" in gitignore
    assert "id_ed25519" in gitignore or "vm/.state/" in gitignore


def test_guest_runs_pinned_ci_image_and_shared_gates() -> None:
    control = read(CONTROL)
    assert "ci/Dockerfile" in control
    assert "ci-gates.sh" in control
    workflow = read(ROOT / ".github" / "workflows" / "ci.yml")
    assert "ci-gates.sh tests" in workflow
    assert "ci-gates.sh static" in workflow
    gates = read(GATES)
    assert "ci-static.sh" in gates
    assert "pytest" in gates


def test_gates_cover_full_cpu_gates() -> None:
    gates = read(GATES)
    cpu_only = read(CPU_ONLY)
    for gate in (
        "cargo fmt --check",
        "cargo clippy --workspace --all-targets --locked",
        "cargo deny check",
        "cargo test --workspace --locked",
        "cargo xtask docs",
        "cargo xtask gen",
    ):
        assert gate in gates, gate
    assert "git diff --exit-code kernels/gen/" in gates
    assert "asm!" in gates and "unsafe" in gates
    assert "cmd_hardware" in gates and "cmd_gpu_smoke" in gates
    for shared_entrypoint in (
        "./scripts/ci-gates.sh rust",
        "./scripts/ci-gates.sh docs",
        "./scripts/ci-gates.sh gen",
        "./scripts/ci-gates.sh policy",
    ):
        assert shared_entrypoint in cpu_only, shared_entrypoint


def test_ci_image_contains_every_gate_tool() -> None:
    dockerfile = read(DOCKERFILE)
    for tool in ("shellcheck", "pytest==9.1.1", "ruff==0.16.5", "cargo-deny"):
        assert tool in dockerfile, tool
    assert "R9V_CI_REQUIRE_SHELLCHECK=1" in dockerfile


def test_no_ci_vm_shadow() -> None:
    assert not (ROOT / "ci" / "vm").exists(), "vm/ is canonical; ci/vm must not exist"
    for path in (CONTROL, HW, GATES, CPU_ONLY):
        assert "ci/vm" not in read(path), path
