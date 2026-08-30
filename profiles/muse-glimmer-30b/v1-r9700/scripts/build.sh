#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
The frozen V1 proof artifacts are benchmarked, but the curated and
license-audited R9V native runtime source has not been imported into this
repository. R9V refuses to build from the legacy research workspace. See the
profile qualification report for the release gate.
EOF
exit 2
