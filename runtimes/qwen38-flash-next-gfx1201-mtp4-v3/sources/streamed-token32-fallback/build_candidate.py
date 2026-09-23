#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build script for standalone prefill_q8_wmma_streamed_extension.

Intended ONLY to be run inside an already-created CPU-only immutable container environment
(image 2dac17a215fb5b0e3461e4c3e36a2981eec8ac3d6021e73183d247e819740c03) by the parent.
Contains NO Docker lifecycle calls, NO GPU device requests, NO network downloads.
Enforces source and header hash manifest before compilation.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
build = root / "build"
build.mkdir(parents=True, exist_ok=True)

# 1. Enforce source/header hash manifest before compile
spec_file = root / "BUILD_SPEC.json"
assert spec_file.exists(), f"BUILD_SPEC.json missing at {spec_file}"
spec = json.loads(spec_file.read_text())
manifest = spec["source_manifest"]
required = {"q8_wmma_streamed_kernel.cu", "q8_wmma_streamed_kernel.h", "build_candidate.py", "native_cpu_shim.cpp"}
required.update(str(f.relative_to(root)) for f in (root / "csrc").rglob("*") if f.is_file())
assert set(manifest) == required, f"Incomplete or unexpected source manifest: diff={set(manifest) ^ required}"

for filename, expected_sha in manifest.items():
    target = root / filename
    assert target.exists(), f"Manifest file missing: {target}"
    actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    assert actual_sha == expected_sha, (
        f"Source hash manifest failure for {filename}: expected {expected_sha}, got {actual_sha}"
    )
print(f"Pre-compile manifest verification passed ({len(manifest)} files verified).")

plugin_headers = root / "csrc"
print(f"Resolved plugin headers: {plugin_headers}")

assert not Path("/dev/kfd").exists(), "CPU-only build must not receive KFD"
assert not torch.cuda.is_initialized(), "CUDA must not be initialized during CPU compile"

module_name = "prefill_q8_wmma_streamed_extension"
source_file = root / "q8_wmma_streamed_kernel.cu"
header_file = root / "q8_wmma_streamed_kernel.h"

print(f"Building extension {module_name} in {build}...")
ext = load(
    name=module_name,
    sources=[str(source_file)],
    extra_include_paths=[
        str(root),
        str(plugin_headers),
        str(plugin_headers / "gguf"),
    ],
    extra_cflags=["-O3", "-std=c++17", "-fPIC"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-DUSE_ROCM",
        "--offload-arch=gfx1201",
    ],
    build_directory=str(build),
    verbose=True,
)

print(f"Build complete. Verifying module attributes for {module_name}...")
required_attrs = [
    "q8_matmul",
    "q8_matmul_control",
    "q8_matmul_candidate",
]

for attr in required_attrs:
    assert hasattr(ext, attr), f"Candidate module missing expected symbol '{attr}'"

print(f"Candidate extension {module_name} successfully verified: {ext}")

so_path = Path(ext.__file__)
so_sha = hashlib.sha256(so_path.read_bytes()).hexdigest()

result = {
    "status": "COMPILED_CPU_ONLY",
    "module": module_name,
    "so_path": str(so_path),
    "so_sha256": so_sha,
    "manifest_verified": True,
    "gpu_execution": "NOT_RUN",
    "cuda_initialized": torch.cuda.is_initialized(),
    "arch": "gfx1201",
    "exported_symbols": required_attrs,
    "variants": {
        "control": {
            "name": "unspecialized_mmq64",
            "mmq_x": 64,
            "mmq_y": 128,
            "nwarps": 8,
            "launch_bounds": [256, 2],
            "estimated_lds_bytes_per_wg": 28224,
            "overread_protection": False,
        },
        "candidate": {
            "name": "streamed_dense_wmma",
            "supported_k": [2560, 3072],
            "chunk_blocks": 8,
            "chunk_k_elements": 256,
            "chunk_row_bytes": 272,
            "tile_m": 32,
            "tile_n": 64,
            "threads_per_wg": 128,
            "waves_per_wg": 4,
            "launch_bounds": [128, 2],
            "lds_bytes_per_wg": 17408,
            "scale_product_order": "(d_weight * d_activation) * float(intsum)",
            "wmma_intrinsics": ["__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12"],
            "outer_unroll": 1,
            "overread_protection": True,
        }
    }
}

(root / "BUILD_RESULT.json").write_text(json.dumps(result, indent=2))
print(f"BUILD_RESULT.json written: {so_path} ({so_sha})")
