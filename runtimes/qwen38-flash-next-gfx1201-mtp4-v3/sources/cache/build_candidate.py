#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build script for standalone candidate qwen38_full_mutable_device extension.

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
EXPECTED_SOURCES = {'full_mutable_device.h': 'd26391659acc69e912e265089b83ee02cc1e32942b47bf90ac088a08a9eee998', 'full_mutable_device.cu': '2c646cf4c63f480e43d832066a14f241bba83c5134593f55a3f41adfc2234c6a', 'native_cpu_shim.cpp': '1c39efd85f3d50e022f8e02efb7918c75b68ef40f9053add3d8d20c013f9d665', 'coop_choice.h': 'c1c3a828a389b3a0a07ca56d6061e02b66626844edd3dbd249b3ad38db1604d1'}
manifest = spec.get("source_manifest")
assert manifest == EXPECTED_SOURCES, "Exact source manifest missing or altered"
for filename, expected_sha in manifest.items():
    target = root / filename
    assert target.exists(), f"Manifest file missing: {target}"
    actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    assert actual_sha == expected_sha, (
        f"Source hash manifest failure for {filename}: expected {expected_sha}, got {actual_sha}"
    )
print(f"Pre-compile manifest verification passed ({len(manifest)} files verified).")


def resolve_plugin_csrc() -> Path:
    configured = os.environ.get("VLLM_GGUF_PLUGIN_CSRC")
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        spec_import = importlib.util.find_spec("vllm_gguf_plugin")
        locations = () if spec_import is None else spec_import.submodule_search_locations or ()
        if not locations:
            # Fallback to local image path if running inside container
            fallback = Path("/opt/r9v/lib/python3.12/site-packages/vllm_gguf_plugin/csrc")
            if fallback.exists():
                return fallback
            raise RuntimeError(
                "vllm_gguf_plugin is not importable; set VLLM_GGUF_PLUGIN_CSRC "
                "to its csrc directory"
            )
        candidate = Path(next(iter(locations))) / "csrc"
    return candidate


plugin_headers = resolve_plugin_csrc()
print(f"Resolved plugin headers: {plugin_headers}")

assert not Path("/dev/kfd").exists(), "CPU-only build must not receive KFD"
assert not torch.cuda.is_initialized(), "CUDA must not be initialized during CPU compile"

module_name = "qwen38_full_mutable_device"
source_file = root / "full_mutable_device.cu"
header_file = root / "full_mutable_device.h"

print(f"Building standalone candidate {module_name} in {build}...")
ext = load(
    name=module_name,
    sources=[str(source_file)],
    extra_include_paths=[str(root), str(plugin_headers), str(plugin_headers / "gguf")],
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
    "full_mutable_device_init",
    "full_mutable_device_step",
]

for attr in required_attrs:
    assert hasattr(ext, attr), f"Candidate module missing expected symbol '{attr}'"

print(f"Candidate extension {module_name} successfully verified: {ext}")

so_path = Path(ext.__file__)

# Extract exported symbols from the actual compiled .so
extracted_symbols = []
try:
    res = subprocess.run(["nm", "-D", "--defined-only", str(so_path)], capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and "full_mutable_device" in parts[2]:
            extracted_symbols.append(parts[2])
except Exception as e:
    extracted_symbols = [f"Extraction note: {e}"]

metadata = {
    "module": module_name,
    "so_path": str(so_path),
    "so_sha256": hashlib.sha256(so_path.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
    "header_sha256": hashlib.sha256(header_file.read_bytes()).hexdigest(),
    "manifest_verified": True,
    "extracted_symbols": extracted_symbols,
    "torch_version": torch.__version__,
    "torch_cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    "gpu_execution": "NOT_RUN",
    "cuda_initialized": torch.cuda.is_initialized(),
}
assert not metadata["cuda_initialized"]
(root / "BUILD_RESULT.json").write_text(json.dumps(metadata, indent=2))
print("Build metadata written to BUILD_RESULT.json")
