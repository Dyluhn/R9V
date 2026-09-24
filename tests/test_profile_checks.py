# SPDX-License-Identifier: Apache-2.0
"""Doctor checks of the selected profile's pinned files and of the host around it."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tools import disk_space, host_preflight, profile_checks
from tools.profile_doctor import Reporter

ROOT = Path(__file__).resolve().parents[1]
UNCENSORED = json.loads(
    (ROOT / "profiles/qwen38-flash-next/dual-r9700-mtp4-uncensored/profile.json").read_text()
)
PROJECTOR = "ced/ced-projector-split16.safetensors"
GIB = 1024**3
# The uncensored profile's shipped expert limits (profile.env).
EXPERT_ENV = {
    "R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK": "222,428",
    "R9V_TIERED_EXPERT_CACHE_SLOTS": "160",
    "R9V_TIERED_EXPERT_CACHE_RANKS": "0",
}
CED_ON = {
    "R9V_CED": "on",
    "R9V_CED_PROJECTOR_REL": PROJECTOR,
    "R9V_CED_PRECISION": "bf16",
    "R9V_CED_MIN_PROMPT": "8192",
    "R9V_CED_TAIL": "2048",
    "R9V_CED_DEFAULT": "on",
}


def only(reporter: Reporter, name: str):
    checks = [check for check in reporter.checks if check.name == name]
    assert len(checks) == 1, reporter.checks
    return checks[0]


def statuses(reporter: Reporter, name: str) -> list[str]:
    return [check.status for check in reporter.checks if check.name == name]


def safetensors(keys: list[str], metadata: dict | None = None) -> bytes:
    header = {key: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0\0"


SPLIT16 = [f"layer.{index}" for index in range(16, 48)] + ["final"]


def projector_setup(tmp_path: Path, monkeypatch, payload: bytes, pinned: bytes | None = None,
                    relative: str = PROJECTOR):
    """A repo with the uncensored profile's package pinning `pinned` (default: payload),
    and a model directory holding `payload` as the projector at `relative`."""
    pinned = payload if pinned is None else pinned
    repo = tmp_path / "repo"
    package = repo / UNCENSORED["descriptors"]["model_package"]
    package.parent.mkdir(parents=True)
    package.write_text(json.dumps({"artifacts": [{
        "role": "ced-projector", "path": relative, "bytes": len(pinned),
        "sha256": hashlib.sha256(pinned).hexdigest()}]}))
    model = tmp_path / "models"
    (model / "ced").mkdir(parents=True)
    (model / relative).write_bytes(payload)
    for key, value in {**CED_ON, "R9V_MODEL_DIR": str(model)}.items():
        monkeypatch.setenv(key, value)
    return repo


def test_ced_projector_that_matches_its_pin_passes_with_its_vram(tmp_path, monkeypatch):
    payload = safetensors(SPLIT16, {"split": "16"})
    repo = projector_setup(tmp_path, monkeypatch, payload)
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    check = only(reporter, "ced-projector")
    assert check.status == "PASS"
    assert check.details["split"] == 16
    assert check.details["vram_bytes_per_gpu"] == len(payload)


def test_ced_projector_with_another_hash_fails_and_never_suggests_a_substitute(tmp_path, monkeypatch):
    payload = safetensors(SPLIT16)
    repo = projector_setup(tmp_path, monkeypatch, payload, pinned=payload[:-1] + b"\1")
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    check = only(reporter, "ced-projector")
    assert check.status == "FAIL"
    assert "has sha256" in check.message
    assert "Never substitute another projector" in check.remediation


def test_ced_projector_of_another_split_fails(tmp_path, monkeypatch):
    payload = safetensors([f"layer.{index}" for index in range(12, 48)] + ["final"])
    repo = projector_setup(tmp_path, monkeypatch, payload)
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    assert "split-12 projector; the profile runs split 16" in only(reporter, "ced-projector").message


def test_ced_projector_with_an_implausible_header_fails_without_reading_it(tmp_path, monkeypatch):
    repo = projector_setup(tmp_path, monkeypatch, struct.pack("<Q", 10**12) + b"{}")
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    assert "not plausible" in only(reporter, "ced-projector").message


def test_ced_off_skips_the_projector_check_and_its_vram(tmp_path, monkeypatch):
    repo = projector_setup(tmp_path, monkeypatch, safetensors(SPLIT16))
    monkeypatch.setenv("R9V_CED", "off")
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    assert reporter.checks == []
    assert profile_checks.ced_projector_vram(repo, UNCENSORED) == 0


def test_ced_projector_check_waits_for_a_model_directory(tmp_path, monkeypatch):
    repo = projector_setup(tmp_path, monkeypatch, safetensors(SPLIT16))
    monkeypatch.delenv("R9V_MODEL_DIR")
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    assert reporter.checks == []


def test_int8_projector_takes_half_the_file_in_vram(tmp_path, monkeypatch):
    payload = safetensors(SPLIT16)
    repo = projector_setup(tmp_path, monkeypatch, payload)
    monkeypatch.setenv("R9V_CED_PRECISION", "int8")

    assert profile_checks.ced_projector_vram(repo, UNCENSORED) == len(payload) // 2


QUALITY_PROJECTOR = "ced/ced-projector-split16-msfa-int8.safetensors"
MSFA_SOURCES = '["boundary_16", "block_input_3", "block_input_7", "block_input_11", "block_input_15"]'


def quality_setup(tmp_path: Path, monkeypatch, payload: bytes, pinned: bytes | None = None):
    repo = projector_setup(tmp_path, monkeypatch, payload, pinned, relative=QUALITY_PROJECTOR)
    monkeypatch.setenv("R9V_CED", "quality")
    monkeypatch.setenv("R9V_CED_QUALITY_PROJECTOR_REL", QUALITY_PROJECTOR)
    return repo


def test_ced_quality_checks_its_own_pinned_projector_and_counts_the_stored_int8_file(tmp_path, monkeypatch):
    payload = safetensors(SPLIT16, {"split": "16", "sources": MSFA_SOURCES})
    repo = quality_setup(tmp_path, monkeypatch, payload)
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    check = only(reporter, "ced-projector")
    assert check.status == "PASS", check.message
    assert check.message.startswith(f"CED quality: {QUALITY_PROJECTOR} matches")
    assert check.details["precision"] == "int8"
    assert check.details["vram_bytes_per_gpu"] == len(payload)


def test_ced_quality_projector_with_another_hash_fails_and_names_the_quality_setup(tmp_path, monkeypatch):
    payload = safetensors(SPLIT16)
    repo = quality_setup(tmp_path, monkeypatch, payload, pinned=payload[:-1] + b"\1")
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    check = only(reporter, "ced-projector")
    assert check.status == "FAIL"
    assert "has sha256" in check.message
    assert "--ced quality" in check.remediation


def test_ced_quality_without_its_projector_fails_before_hashing(tmp_path, monkeypatch):
    repo = quality_setup(tmp_path, monkeypatch, safetensors(SPLIT16))
    (Path(os.environ["R9V_MODEL_DIR"]) / QUALITY_PROJECTOR).unlink()
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    assert "CED projector missing" in only(reporter, "ced-projector").message


def test_ced_quality_vram_is_the_stored_file_even_with_int8_precision_set(tmp_path, monkeypatch):
    payload = safetensors(SPLIT16)
    repo = quality_setup(tmp_path, monkeypatch, payload)
    monkeypatch.setenv("R9V_CED_PRECISION", "int8")

    assert profile_checks.ced_projector_vram(repo, UNCENSORED) == len(payload)


def test_ced_quality_mounts_its_own_model_file(monkeypatch):
    monkeypatch.setenv("R9V_RUNTIME_DESCRIPTOR", str(ROOT / UNCENSORED["descriptors"]["runtime"]))
    monkeypatch.setenv("R9V_CED", "quality")
    reporter = Reporter()

    profile_checks.check_runtime_overlays(reporter)

    mounted = only(reporter, "runtime-overlays").details["mounted"]
    assert "model_ced_quality.py" in mounted
    assert "model.py" not in mounted


def copied_runtime(tmp_path: Path) -> Path:
    source = ROOT / UNCENSORED["descriptors"]["runtime"]
    target = tmp_path / "runtime"
    shutil.copytree(source.parent / "overlays", target / "overlays")
    shutil.copy(source, target / "runtime.json")
    return target / "runtime.json"


def test_shipped_runtime_overlays_pass(monkeypatch):
    monkeypatch.setenv("R9V_RUNTIME_DESCRIPTOR", str(ROOT / UNCENSORED["descriptors"]["runtime"]))
    monkeypatch.setenv("R9V_CED", "on")
    reporter = Reporter()

    profile_checks.check_runtime_overlays(reporter)

    check = only(reporter, "runtime-overlays")
    assert check.status == "PASS"
    assert "model.py" in check.details["mounted"]


def test_edited_runtime_overlay_fails_and_names_the_file(tmp_path, monkeypatch):
    runtime = copied_runtime(tmp_path)
    (runtime.parent / "overlays/scheduler.py").write_text("# edited\n")
    monkeypatch.setenv("R9V_RUNTIME_DESCRIPTOR", str(runtime))
    reporter = Reporter()

    profile_checks.check_runtime_overlays(reporter)

    check = only(reporter, "runtime-overlays")
    assert check.status == "FAIL"
    assert "scheduler.py" in check.message


def test_shipped_expert_limits_are_consistent(monkeypatch):
    for key, value in EXPERT_ENV.items():
        monkeypatch.setenv(key, value)
    reporter = Reporter()

    profile_checks.check_expert_limits(reporter, ROOT, UNCENSORED)

    check = only(reporter, "expert-limit-consistency")
    assert check.status == "PASS", check.message
    assert check.details["ceilings"] == [222, 428]
    assert check.details["pinned"] == [0, 400]


def test_runtime_pin_list_of_another_size_than_the_placement_expects_fails(monkeypatch):
    for key, value in EXPERT_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(profile_checks, "_runtime_pins", lambda path, runtime: [0, 300])
    reporter = Reporter()

    profile_checks.check_expert_limits(reporter, ROOT, UNCENSORED)

    assert "pins [0, 300] experts per layer; the placement expects [0, 400]" in only(
        reporter, "expert-limit-consistency").message


def test_changed_expert_ceiling_fails_with_the_expected_values(monkeypatch):
    for key, value in {**EXPERT_ENV, "R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK": "206,434"}.items():
        monkeypatch.setenv(key, value)
    reporter = Reporter()

    profile_checks.check_expert_limits(reporter, ROOT, UNCENSORED)

    check = only(reporter, "expert-limit-consistency")
    assert check.status == "FAIL"
    assert "206,434" in check.message and "[222, 428]" in check.message


def test_moved_cache_rank_fails(monkeypatch):
    for key, value in {**EXPERT_ENV, "R9V_TIERED_EXPERT_CACHE_RANKS": "1"}.items():
        monkeypatch.setenv(key, value)
    reporter = Reporter()

    profile_checks.check_expert_limits(reporter, ROOT, UNCENSORED)

    assert "on ranks 1 differ" in only(reporter, "expert-limit-consistency").message


def test_profile_without_a_fixed_placement_skips_the_expert_limit_check():
    reporter = Reporter()

    profile_checks.check_expert_limits(reporter, ROOT, {"descriptors": {}})

    assert reporter.checks == []


def fake_gpu_host(tmp_path: Path, holders: dict[int, tuple[str, int]], free_bytes: int):
    """sysfs and procfs for one GPU (BDF 0000:03:00.0, KFD id 1234): each holder pid has
    compute VRAM in the KFD per-process file."""
    sys_root, proc_root = tmp_path / "sys", tmp_path / "proc"
    pci = sys_root / "bus/pci/devices/0000:03:00.0"
    pci.mkdir(parents=True)
    (pci / "mem_info_vram_total").write_text(str(32 * GIB))
    (pci / "mem_info_vram_used").write_text(str(32 * GIB - free_bytes))
    for pid, (name, size) in holders.items():
        kfd = sys_root / f"class/kfd/kfd/proc/{pid}"
        kfd.mkdir(parents=True)
        (kfd / "vram_1234").write_text(str(size))
        (proc_root / str(pid)).mkdir(parents=True)
        (proc_root / str(pid) / "comm").write_text(name + "\n")
    return sys_root, proc_root


class Gpu:
    bdf = "0000:03:00.0"


SELECTED = [(0, Gpu())]
IDS = {"0000:03:00.0": 1234}


def test_graphics_memory_is_read_from_drm_fdinfo(tmp_path):
    proc = tmp_path / "proc/4242"
    (proc / "fd").mkdir(parents=True)
    (proc / "fdinfo").mkdir()
    os.symlink("/dev/dri/renderD128", proc / "fd/7")
    (proc / "fdinfo/7").write_text(
        "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t9\ndrm-memory-vram:\t2048 MiB\n")

    usage = host_preflight.gpu_process_vram(tmp_path / "sys", tmp_path / "proc", IDS)

    assert usage == {"0000:03:00.0": {4242: 2 * GIB}}


def test_other_process_leaving_enough_vram_is_a_warning_that_names_it(tmp_path):
    sys_root, proc_root = fake_gpu_host(tmp_path, {4242: ("firefox", GIB)}, free_bytes=28 * GIB)
    reporter = Reporter()

    host_preflight.check_vram_other_processes(reporter, SELECTED, IDS, sys_root, proc_root, [20 * GIB])

    check = only(reporter, "vram-other-processes")
    assert check.status == "WARN"
    assert "firefox (pid 4242) 1.00 GiB" in check.message


def test_other_process_leaving_too_little_vram_fails(tmp_path):
    sys_root, proc_root = fake_gpu_host(tmp_path, {4242: ("VLLM::Worker_TP", 28 * GIB)}, free_bytes=3 * GIB)
    reporter = Reporter()

    host_preflight.check_vram_other_processes(reporter, SELECTED, IDS, sys_root, proc_root, [20 * GIB])

    check = only(reporter, "vram-other-processes")
    assert check.status == "FAIL"
    assert "below the 20.00 GiB R9V needs" in check.message


def test_other_process_without_a_known_need_is_a_warning_that_asks_for_the_model_dir(tmp_path):
    sys_root, proc_root = fake_gpu_host(tmp_path, {4242: ("VLLM::Worker_TP", 28 * GIB)}, free_bytes=3 * GIB)
    reporter = Reporter()

    host_preflight.check_vram_other_processes(reporter, SELECTED, IDS, sys_root, proc_root, None)

    check = only(reporter, "vram-other-processes")
    assert check.status == "WARN"
    assert "pass --model-dir" in check.message


def test_no_other_process_on_the_gpu_passes(tmp_path):
    sys_root, proc_root = fake_gpu_host(tmp_path, {}, free_bytes=31 * GIB)
    reporter = Reporter()

    host_preflight.check_vram_other_processes(reporter, SELECTED, IDS, sys_root, proc_root, [20 * GIB])

    assert statuses(reporter, "vram-other-processes") == ["PASS"]


@pytest.mark.parametrize(("bind", "status"), [("127.0.0.1", "PASS"), ("::1", "PASS"),
                                              ("0.0.0.0", "WARN"), ("192.168.1.5", "WARN"),
                                              ("localhost", "FAIL")])
def test_api_exposure_by_bind_address(monkeypatch, bind, status):
    monkeypatch.setenv("R9V_HOST_BIND", bind)
    reporter = Reporter()

    host_preflight.check_api_exposure(reporter, uncensored=False)

    assert statuses(reporter, "api-exposure") == [status]


def test_unset_bind_is_this_machine_only(monkeypatch):
    monkeypatch.delenv("R9V_HOST_BIND", raising=False)
    reporter = Reporter()

    host_preflight.check_api_exposure(reporter, uncensored=True)

    assert statuses(reporter, "api-exposure") == ["PASS"]


def test_exposed_uncensored_model_warns_about_harmful_requests(monkeypatch):
    monkeypatch.setenv("R9V_HOST_BIND", "0.0.0.0")
    reporter = Reporter()

    host_preflight.check_api_exposure(reporter, uncensored=True)

    check = only(reporter, "api-exposure")
    assert "every interface" in check.message and "refusals were removed" in check.message


def no_new_messages_tell_users_to_change_the_host(reporter: Reporter) -> None:
    for check in reporter.checks:
        text = f"{check.message} {check.remediation or ''}".lower()
        assert not any(word in text for word in ("sudo", "sysctl", "ulimit", "memlock", "systemctl"))


def test_new_check_messages_never_ask_for_root_or_host_changes(tmp_path, monkeypatch):
    sys_root, proc_root = fake_gpu_host(tmp_path, {4242: ("x", 28 * GIB)}, free_bytes=GIB)
    monkeypatch.setenv("R9V_HOST_BIND", "0.0.0.0")
    monkeypatch.setenv("R9V_CED", "on")
    reporter = Reporter()

    host_preflight.check_vram_other_processes(reporter, SELECTED, IDS, sys_root, proc_root, [20 * GIB])
    host_preflight.check_api_exposure(reporter, uncensored=True)

    no_new_messages_tell_users_to_change_the_host(reporter)


PACKAGE = {"artifacts": [
    {"path": "target/a.gguf", "bytes": 100},
    {"path": "target/b.gguf", "bytes": 50},
    {"path": "optional.bin", "bytes": 999, "required": False},
]}


def test_package_bytes_count_only_missing_or_short_required_files(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "target/a.gguf").write_bytes(b"x" * 100)
    (tmp_path / "target/b.gguf").write_bytes(b"x" * 10)

    assert disk_space.missing_package_bytes(PACKAGE["artifacts"], tmp_path) == 50


def test_same_size_file_in_the_reuse_directory_counts_as_present(tmp_path):
    reuse = tmp_path / "reuse"
    (reuse / "target").mkdir(parents=True)
    (reuse / "target/b.gguf").write_bytes(b"x" * 50)

    assert disk_space.missing_package_bytes(PACKAGE["artifacts"], tmp_path / "model", reuse) == 100


def test_needs_on_one_filesystem_are_summed_with_a_reserve(tmp_path):
    needs = [disk_space.Need("a", tmp_path / "x/y", GIB), disk_space.Need("b", tmp_path, 2 * GIB)]

    (group,) = disk_space.by_filesystem(needs, free=lambda path: 3 * GIB)

    assert group["required"] == 4 * GIB
    assert group["fits"] is False


def test_fresh_install_needs_package_ple_image_and_compile_cache(tmp_path):
    bundle = {"parts": [{"name": "part-000", "bytes": 7}]}

    needs = disk_space.install_needs(PACKAGE, tmp_path / "m", tmp_path / "d", tmp_path / "d/ple.bin",
                                     tmp_path / "d/cache", bundle=bundle, docker_root=tmp_path / "docker")

    assert [need.label for need in needs] == ["model package", "PLE table", "image bundle parts",
                                              "Docker image store", "compile cache"]


def completed(stdout: str = "", returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout, "")


def test_disk_space_check_fails_with_the_short_filesystem_named(tmp_path, monkeypatch):
    monkeypatch.setenv("R9V_MODEL_DIR", str(tmp_path / "models"))
    for key in ("R9V_DATA_DIR", "R9V_PLE_PATH", "R9V_CACHE_DIR", "R9V_REUSE_FROM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(disk_space.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(0, 0, GIB))
    reporter = Reporter()

    host_preflight.check_disk_space(reporter, ROOT, UNCENSORED, lambda command: completed("sha256:x"))

    check = only(reporter, "disk-space")
    assert check.status == "FAIL"
    assert str(tmp_path) in check.message and "GiB free" in check.message


def test_disk_space_without_a_model_directory_is_a_note(monkeypatch):
    monkeypatch.delenv("R9V_MODEL_DIR", raising=False)
    reporter = Reporter()

    host_preflight.check_disk_space(reporter, ROOT, UNCENSORED, lambda command: completed())

    assert statuses(reporter, "disk-space") == ["NOTE"]


def test_ced_quality_says_its_projector_shares_vram_with_the_vision_encoder(tmp_path, monkeypatch):
    repo = quality_setup(tmp_path, monkeypatch, safetensors(SPLIT16, {"split": "16", "sources": MSFA_SOURCES}))
    reporter = Reporter()

    profile_checks.check_ced_projector(reporter, repo, UNCENSORED)

    assert "shares with the vision encoder's weights" in only(reporter, "ced-projector").message


def test_ced_quality_host_copies_are_the_projector_and_the_mmproj_share_per_rank_in_whole_slabs(tmp_path, monkeypatch):
    payload = safetensors(SPLIT16)
    quality_setup(tmp_path, monkeypatch, payload)
    mmproj = Path(os.environ["R9V_MODEL_DIR"]) / "vision/mmproj.gguf"
    mmproj.parent.mkdir()
    mmproj.write_bytes(b"\0" * 1000)
    monkeypatch.setenv("R9V_MMPROJ_REL", "vision/mmproj.gguf")
    monkeypatch.setenv("R9V_TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setattr(profile_checks, "PINNED_SLAB", 256)

    projector_slabs = -(-len(payload) // 256) * 256
    assert profile_checks.ced_quality_host_bytes() == 2 * (projector_slabs + 512)  # 500 mmproj bytes per rank: 2 slabs


def test_ced_on_holds_no_host_copies(tmp_path, monkeypatch):
    projector_setup(tmp_path, monkeypatch, safetensors(SPLIT16))
    monkeypatch.setenv("R9V_MMPROJ_REL", "vision/mmproj.gguf")

    assert profile_checks.ced_quality_host_bytes() == 0
