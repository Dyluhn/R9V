#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Materialize a linked worktree as a standalone shallow Git checkout.
#
# The returned directory contains usable root and submodule Git metadata plus
# the live working-tree overlay. Callers own it and must remove it when done.
set -euo pipefail

SOURCE_ROOT=${1:?usage: make-source-snapshot.sh <source-root>}
SOURCE_ROOT=$(realpath -e -- "$SOURCE_ROOT")
SNAPSHOT=$(mktemp -d /tmp/r9v-vm-source.XXXXXX)
KEEP=0

cleanup() {
    if ((KEEP == 0)) && [[ $SNAPSHOT == /tmp/r9v-vm-source.* ]]; then
        rm -rf -- "$SNAPSHOT"
    fi
}
trap cleanup EXIT

die() { printf 'make-source-snapshot: %s\n' "$*" >&2; exit 1; }

git -C "$SNAPSHOT" init -q
git -C "$SNAPSHOT" fetch -q --depth=1 "$SOURCE_ROOT" HEAD
git -C "$SNAPSHOT" checkout -q --detach FETCH_HEAD

while read -r key path; do
    name=${key#submodule.}
    name=${name%.path}
    expected=$(git -C "$SOURCE_ROOT" rev-parse "HEAD:$path")
    source_top=$(git -C "$SOURCE_ROOT/$path" rev-parse --show-toplevel 2>/dev/null || true)
    [[ -n $source_top ]] || die "submodule is not initialized: $path"
    [[ $(realpath -e -- "$source_top") == $(realpath -e -- "$SOURCE_ROOT/$path") ]] \
        || die "submodule is not initialized: $path"
    [[ $(git -C "$SOURCE_ROOT/$path" rev-parse HEAD) == "$expected" ]] \
        || die "submodule does not match pinned gitlink: $path"

    mkdir -p -- "$SNAPSHOT/$path"
    git -C "$SNAPSHOT/$path" init -q
    git -C "$SNAPSHOT/$path" fetch -q --depth=1 "$SOURCE_ROOT/$path" "$expected"
    git -C "$SNAPSHOT/$path" checkout -q --detach FETCH_HEAD
    git -C "$SNAPSHOT" config "submodule.$name.url" "$SOURCE_ROOT/$path"
done < <(git -C "$SOURCE_ROOT" config --file .gitmodules \
    --get-regexp '^submodule\..*\.path$')

# Preserve the standalone Git directories while overlaying all intentional
# tracked and untracked changes from the source worktree.
rsync -a --delete \
    --filter='protect .git/' \
    --filter='protect .git' \
    --exclude='.git' \
    --exclude=/target \
    --exclude=/.venv \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='*.qcow2' \
    "$SOURCE_ROOT/" "$SNAPSHOT/"

if git -C "$SNAPSHOT" submodule status --recursive | grep -Eq '^[-+U]'; then
    die 'standalone snapshot has invalid submodule state'
fi

KEEP=1
printf '%s\n' "$SNAPSHOT"
