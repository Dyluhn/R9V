<!-- SPDX-License-Identifier: Apache-2.0 -->
# Hardware-blind R9V development VM

A checked-in QEMU/KVM definition for R9V CPU-gate development without exposing
the host's AMD hardware or CPU feature set to the development environment.

> **Non-authoritative.** The VM uses a generic named x86-64 CPU, fixed
> vCPU/RAM/disk, no host shares, and no GPU nodes. Its topology and any numbers
> produced inside it are non-authoritative and **unable to qualify
> performance**. Qualification happens on bare metal through
> `vm/r9v-hw-container.sh`.

## Layout

- `image.pin` pins an official dated Ubuntu 24.04 amd64 cloud image and SHA256.
- `vm-config.sh` fixes the generic CPU model, vCPUs, RAM, disk, and SSH port.
- `r9v-vm.sh` provides `check-deps`, `fetch`, `up`, `status`, `ssh`, `sync`,
  `test`, and guarded `destroy` commands.
- `r9v-hw-container.sh` is the separate bare-metal path and the only file in
  `vm/` that passes `/dev/kfd`, explicit render nodes, and read-only PCI sysfs.
- `scripts/make-source-snapshot.sh` converts the live linked worktree and its
  pinned submodules into a standalone shallow Git snapshot.

## Why it is blind

- The CPU is `qemu64`; the host model is never an option. KVM is used only when
  `/dev/kvm` is usable, with a TCG fallback.
- The VM launch definition has no host home/sysfs shares and no GPU nodes.
- `sync` materializes a standalone source snapshot, overlays intentional
  working-tree edits, and copies it to guest disk over SSH. It never shares a
  host path or copies a linked-worktree `.git` pointer that would be invalid in
  the guest.
- Images, overlays, keys, logs, and pidfiles live under
  `${XDG_STATE_HOME:-$HOME/.local/state}/r9v-vm` by default, outside the repo.

## Commands

| Command | Effect |
|---|---|
| `r9v-vm.sh check-deps` | Report host dependency diagnostics |
| `r9v-vm.sh fetch` | Download and verify the pinned base image once (~596 MB) |
| `r9v-vm.sh up` | Verify the image, create the backed overlay, and boot |
| `r9v-vm.sh status` | Show VM state and the non-authoritative disclaimer |
| `r9v-vm.sh ssh [args]` | Open SSH or run a guest command |
| `r9v-vm.sh sync` | Copy a standalone source snapshot to guest disk |
| `r9v-vm.sh test` | Sync, then run all gates in the pinned CI container |
| `r9v-vm.sh destroy` | Stop the VM and retain state |
| `r9v-vm.sh destroy --with-state --yes` | Stop the VM and remove guarded state |

`fetch` downloads to a temporary file in the state directory, verifies the
SHA256, then atomically installs `$STATE_DIR/base-amd64.img`. A verified base is
never downloaded again or overwritten. `up` creates an 80 GiB qcow2 overlay
backed by that image, captures the console in `$STATE_DIR/serial.log`, and waits
for SSH and cloud-init. State removal requires the VM marker, canonical path
guards, explicit `--with-state --yes`, and a fully stopped QEMU process.

`test` bind-mounts the guest-local snapshot at `/source` read-only and copies it
to the container's writable `/workspace`. The Git-diff, generator, policy, and
submodule gates therefore work without mutating the synchronized snapshot. The
first container build may download/use approximately 30 GiB inside the guest
overlay at `$STATE_DIR/disk.qcow2`; the command reports that destination and
size before building.

## Host prerequisites

`qemu-system-x86_64`, `qemu-img`, `ssh`, `rsync`, `git`, `curl`, `sha256sum`,
`ssh-keygen`, and either `genisoimage` or `cloud-localds`. `check-deps` reports
each one and installs nothing. The guest needs outbound network for its initial
cloud-init package install and first container build.
