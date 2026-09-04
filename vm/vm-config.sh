# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034
# Fixed topology for the hardware-blind R9V development VM.
# Sourced by r9v-vm.sh; never executed directly.
#
# Hardware blindness is structural: a generic named x86-64 CPU model, fixed
# vCPU/RAM/disk sizes, KVM only when the host offers /dev/kvm with a TCG
# fallback otherwise. There is deliberately no way to select the host CPU
# model, resize the topology, share host paths, or attach host devices here.
# Anything hardware-specific lives only in vm/r9v-hw-container.sh.

# Generic named CPU model. The host model is never selected, by any flag.
R9V_VM_CPU_MODEL=qemu64
R9V_VM_MACHINE=q35

# Fixed topology. Not flags, not environment overrides.
R9V_VM_CPUS=8
R9V_VM_RAM_MB=16384
R9V_VM_DISK_GB=80

# Guest SSH endpoint (host side of user-mode networking forward).
R9V_VM_SSH_PORT=2222
R9V_VM_GUEST_USER=r9v
R9V_VM_GUEST_SRC=/home/r9v/r9v-src

# State (disk overlay, seed ISO, SSH keys, pidfile) lives outside the repo
# under XDG state, never beside the definition. Overridable only to move the
# whole state directory elsewhere off-repo.
R9V_VM_STATE_DEFAULT_SUBDIR=r9v-vm
