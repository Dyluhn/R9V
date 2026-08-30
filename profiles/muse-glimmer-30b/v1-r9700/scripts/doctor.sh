#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile_root=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$profile_root/../../.." && pwd)

"$repo_root/scripts/profile-doctor.sh"
if [[ -n ${R9V_MODEL_DIR:-} ]]; then
    "$repo_root/tools/verify_package.py" \
        "$repo_root/packages/models/muse-glimmer-30b/v1-v12/package.json"
fi
printf '%s\n' 'WARNING Muse V1 is an experimental rough draft and is not quality-competitive with Unsloth Q5/Q6.'
