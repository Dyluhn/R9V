#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
profile=${R9V_PROFILE:-$repo_root/profiles/qwen38-flash-next/dual-r9700/profile.env}
config_file=${R9V_CONFIG_FILE:-}
if [[ -n $config_file ]]; then
    [[ -r $config_file ]] || {
        printf 'R9V_CONFIG_FILE is not readable: %s\n' "$config_file" >&2
        exit 1
    }
    set -a
    # shellcheck disable=SC1090
    source "$config_file"
    set +a
fi
if [[ -r $profile ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$profile"
    set +a
fi
export R9V_REPO_ROOT=$repo_root
exec python3 "$repo_root/tools/profile_doctor.py" "$@"
