#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile_root=$(cd -- "$script_dir/.." && pwd)
# shellcheck disable=SC1091
source "$profile_root/profile.env"

model_dir=${R9V_MODEL_DIR:?Set R9V_MODEL_DIR to the Muse V1 package root}
runtime_bin=${R9V_MUSE_RUNTIME_BIN:-}
code_object=${R9V_MUSE_CODE_OBJECT:-}
if [[ -z $runtime_bin || ! -x $runtime_bin || -z $code_object || ! -f $code_object ]]; then
    cat >&2 <<'EOF'
Muse V1 currently exposes only a frozen raw-token proof engine. Set both
R9V_MUSE_RUNTIME_BIN and R9V_MUSE_CODE_OBJECT to hash-verified experimental
artifacts, or wait for the curated user runtime. This profile does not claim a
chat or OpenAI-compatible API yet.
EOF
    exit 2
fi

model="$model_dir/$R9V_MUSE_MODEL_REL"
[[ -f $model ]] || { printf 'Missing Muse V1 model: %s\n' "$model" >&2; exit 1; }

exec "$runtime_bin" "$code_object" "$model" 8 "$R9V_MUSE_MAX_NEW_TOKENS" 1 1 "$@"
