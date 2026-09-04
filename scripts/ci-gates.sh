#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ci-gates.sh — the single consolidated CPU gate entrypoint.
#
# Usage: ci-gates.sh {static|tests|python|rust|docs|gen|policy|gpu-smoke|hardware|all}
#
# Run by the hosted CPU CI jobs (.github/workflows/ci.yml and
# .github/workflows/cpu-only.yml), by the hardware-blind dev VM guest
# (vm/r9v-vm.sh test, inside the pinned ci/Dockerfile container, with the
# guest-local source bind-mounted read-only and a writable CARGO_TARGET_DIR),
# and by the bare-metal hardware path (vm/r9v-hw-container.sh).
#
# `static` and `tests` stay native-only (Python + shell + descriptors), so the
# hosted workflow never needs to build the large ROCm OCI image: the
# Dockerfile is a build environment, not a per-run CI artifact. `all`
# additionally runs the Rust toolchain gates (fmt, clippy, deny, test), the
# xtask docs/gen checks, and the source policy greps, mirroring
# .github/workflows/cpu-only.yml. `gpu-smoke` runs only where real hardware
# is present (the hw container defaults to `all` plus `gpu-smoke`); on hosts
# without HIP it reports an explicit skip instead of failing.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python3}

cd -- "$REPO_ROOT"

cmd_static() {
    R9V_CI_REQUIRE_SHELLCHECK=${R9V_CI_REQUIRE_SHELLCHECK:-0} \
        ./scripts/ci-static.sh
}

cmd_python() {
    "$PYTHON_BIN" -m pytest -q tests
}

cmd_rust_fmt() {
    cargo fmt --check
}

cmd_rust_clippy() {
    cargo clippy --workspace --all-targets --locked -- -D warnings
}

cmd_rust_deny() {
    cargo deny check
}

cmd_rust_test() {
    cargo test --workspace --locked
}

cmd_rust() {
    cmd_rust_fmt
    cmd_rust_clippy
    cmd_rust_deny
    cmd_rust_test
}

cmd_docs() {
    cargo xtask docs
}

cmd_gen() {
    cargo xtask gen
    if [[ -d kernels/gen ]]; then
        git diff --exit-code kernels/gen/
    fi
}

cmd_policy_asm() {
    echo "Checking that inline asm appears only in crates/r9v-kgen/src/leaf/..."
    local illegal_asm
    illegal_asm=$(git grep -En "asm!\s*\(|__asm__|\basm\s+volatile" -- \
        '*.rs' '*.hip' '*.cpp' '*.h' '*.cu' '*.py' \
        ':!crates/r9v-kgen/src/leaf/*' \
        ':!scripts/check_card.py' \
        ':!skills/r9v-card-work/scripts/check_card.py' \
        ':!.agents/skills/r9v-card-work/scripts/check_card.py' \
        ':!tests/test_check_card.py' 2>/dev/null || true)
    if [[ -n $illegal_asm ]]; then
        echo "Found illegal inline assembly outside crates/r9v-kgen/src/leaf/:"
        echo "$illegal_asm"
        exit 1
    fi
    echo "Inline assembly check passed."
}

cmd_policy_unsafe() {
    echo "Checking that unsafe appears only in crates/r9v-hip and crates/r9v-t0 SIMD modules..."
    local illegal_unsafe
    illegal_unsafe=$(git grep -En "\bunsafe\b" -- \
        '*.rs' '*.hip' '*.cpp' '*.h' '*.cu' '*.py' \
        ':!crates/r9v-hip/*' \
        ':!crates/r9v-t0/src/simd/*' \
        ':!tests/*' \
        ':!benches/*' \
        ':!scripts/check_card.py' \
        ':!skills/r9v-card-work/scripts/check_card.py' \
        ':!.agents/skills/r9v-card-work/scripts/check_card.py' 2>/dev/null || true)
    if [[ -n $illegal_unsafe ]]; then
        echo "Found illegal unsafe code:"
        echo "$illegal_unsafe"
        exit 1
    fi
    echo "Unsafe code check passed."
}

cmd_policy() {
    cmd_policy_asm
    cmd_policy_unsafe
}

cmd_gpu_smoke() {
    cargo test --locked -p r9v-hip --test gpu_smoke
}

cmd_all() {
    cmd_static
    cmd_rust_fmt
    cmd_rust_clippy
    cmd_rust_deny
    cmd_gen
    cmd_docs
    cmd_policy
    cmd_rust_test
    cmd_python
}

cmd_hardware() {
    cmd_all
    cmd_gpu_smoke
}

case ${1:-all} in
    static) cmd_static ;;
    tests | python) cmd_python ;;
    rust) cmd_rust ;;
    docs) cmd_docs ;;
    gen) cmd_gen ;;
    policy) cmd_policy ;;
    gpu-smoke) cmd_gpu_smoke ;;
    hardware) cmd_hardware ;;
    all) cmd_all ;;
    *) printf 'usage: %s {static|tests|python|rust|docs|gen|policy|gpu-smoke|hardware|all}\n' "$(basename -- "$0")" >&2; exit 2 ;;
esac
