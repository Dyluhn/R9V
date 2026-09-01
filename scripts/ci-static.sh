#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${PYTHON:-python3}

cd -- "$repo_root"

"$python_bin" -m ruff check --select E4,E7,E9,F tools tests
"$python_bin" - <<'PY'
import ast
import json
from pathlib import Path

root = Path.cwd()
excluded = {".git", "kernels", "vendor"}
for path in sorted(root.rglob("*")):
    if excluded.intersection(path.relative_to(root).parts):
        continue
    if path.suffix == ".py" or path == root / "r9v":
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
PY

mapfile -d '' shell_files < <(
    find . \
        -path './.git' -prune -o \
        -path './kernels' -prune -o \
        -path './vendor' -prune -o \
        -type f -name '*.sh' -print0
)
for shell_file in "${shell_files[@]}"; do
    bash -n "$shell_file"
done

if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "${shell_files[@]}"
elif [[ ${R9V_CI_REQUIRE_SHELLCHECK:-0} == 1 ]]; then
    printf 'shellcheck is required but is not installed\n' >&2
    exit 1
else
    printf 'NOTE: shellcheck not installed; bash syntax checks still passed\n'
fi

./r9v validate

submodule_status=$(git submodule status --recursive)
if grep -Eq '^[-+U]' <<<"$submodule_status"; then
    printf 'Submodules are missing, conflicted, or do not match the pinned gitlinks:\n%s\n' \
        "$submodule_status" >&2
    exit 1
fi
