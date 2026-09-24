# SPDX-License-Identifier: Apache-2.0
"""CED quality's shared VRAM region on the CPU: the vision encoder's weights and the CED projector
take turns in one region, each at fixed addresses, refilled from their host copies on demand.

The overlay imports vLLM, so the pieces under test are taken from its source (ast) and run alone,
as tests/test_ced_quality_overlay.py does.
"""
from __future__ import annotations

import ast
import logging
import re
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

RUNTIME = Path(__file__).resolve().parents[1] / "runtimes/qwen38-flash-next-gfx1201-mtp4-v3"
OVERLAY = RUNTIME / "overlays/model_ced_quality.py"
SCHEDULER = RUNTIME / "overlays/scheduler.py"
NAMES = {"_SharedVram", "_share_vision"}
CPU = torch.device("cpu")


def overlay() -> dict:
    tree = ast.parse(OVERLAY.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES]
    assert {n.name for n in nodes} == NAMES
    namespace = {"torch": torch, "nn": nn, "time": time, "logger": logging.getLogger("test")}
    exec(compile(ast.Module(nodes, []), str(OVERLAY), "exec"), namespace)
    return namespace


def sets() -> dict:
    """A vision set (bf16 weights, odd sizes) and a projector set (int8 map, bf16 scale and bias)."""
    torch.manual_seed(0)
    return {
        "vision": [torch.randn(3, 5).to(torch.bfloat16), torch.randn(7).to(torch.bfloat16)],
        "ced:16": [torch.randint(-127, 128, (4, 256), dtype=torch.int8),
                   torch.randn(4, 2).to(torch.bfloat16), torch.randn(4).to(torch.bfloat16)],
    }


def test_the_region_is_as_large_as_the_largest_set_not_their_sum():
    shared = overlay()["_SharedVram"](sets(), CPU)

    assert shared.size == 1024 + 256 + 256  # the projector's int8 map, scale and bias, each 256-byte aligned


def test_each_set_reads_back_exactly_after_the_other_set_used_the_region():
    host = sets()
    shared = overlay()["_SharedVram"](host, CPU)

    projector = shared.acquire("ced:16")
    assert all(torch.equal(v, h) for v, h in zip(projector, host["ced:16"]))
    vision = shared.acquire("vision")
    assert all(torch.equal(v, h) for v, h in zip(vision, host["vision"]))
    projector = shared.acquire("ced:16")
    assert all(torch.equal(v, h) for v, h in zip(projector, host["ced:16"]))


def test_a_set_keeps_its_addresses_across_swaps():
    shared = overlay()["_SharedVram"](sets(), CPU)
    first = [v.data_ptr() for v in shared.acquire("vision")]

    shared.acquire("ced:16")
    again = [v.data_ptr() for v in shared.acquire("vision")]

    assert again == first


def test_asking_for_the_resident_set_does_not_copy():
    shared = overlay()["_SharedVram"](sets(), CPU)
    shared.acquire("vision")
    shared.acquire("vision")
    shared.acquire("ced:16")
    shared.acquire("ced:16")

    assert shared.swaps == 2


def test_the_two_sets_overlap_in_the_region():
    shared = overlay()["_SharedVram"](sets(), CPU)
    base = shared.region.data_ptr()

    assert shared.views["vision"][0].data_ptr() == base
    assert shared.views["ced:16"][0].data_ptr() == base


def test_vision_weights_move_into_the_region_and_drop_the_loaders_extra_reference():
    functions = overlay()
    tower = nn.Linear(5, 3, bias=True).to(torch.bfloat16)
    params = list(tower.parameters())
    tower.weight.data_container = [tower.weight.data]  # a single-shard GGUF weight, as the loader leaves it
    expected = [p.detach().clone() for p in params]
    shared = functions["_SharedVram"]({"vision": [p.data for p in params], "ced:16": sets()["ced:16"]}, CPU)

    functions["_share_vision"](shared, params)
    shared.acquire("vision")

    assert [p.data_ptr() for p in params] == [v.data_ptr() for v in shared.views["vision"]]
    assert tower.weight.data_container[0].data_ptr() == shared.views["vision"][0].data_ptr()
    assert all(torch.equal(p, e) for p, e in zip(params, expected))
    x = torch.randn(2, 5).to(torch.bfloat16)
    assert torch.equal(tower(x), nn.functional.linear(x, expected[0], expected[1]))


def test_host_copies_split_across_slabs_read_back_exactly():
    shared_vram = overlay()["_SharedVram"]
    shared_vram.SLAB = 512  # the 1024-byte int8 map spans two slabs
    host = sets()
    shared = shared_vram(host, CPU)

    assert [slab.numel() for slab in shared.host["ced:16"]] == [512, 512, 512]
    shared.acquire("vision")
    projector = shared.acquire("ced:16")
    assert all(torch.equal(v, h) for v, h in zip(projector, host["ced:16"]))


def test_host_copies_are_taken_when_the_region_is_made_not_when_a_set_is_acquired():
    host = sets()
    shared = overlay()["_SharedVram"](host, CPU)
    original = host["vision"][0].clone()

    host["vision"][0].zero_()  # the vision weights' old storage is freed once they move into the region

    assert torch.equal(shared.acquire("vision")[0], original)


def test_without_a_vision_tower_the_region_holds_only_the_projector():
    shared = overlay()["_SharedVram"]({"ced:16": sets()["ced:16"], "vision": []}, CPU)

    assert shared.acquire("vision") == []
    assert shared.size == shared._offsets(sets()["ced:16"])[-1]


def test_an_approximate_step_holds_a_single_request_so_it_never_runs_the_vision_encoder():
    """The scheduler invariant the shared region relies on to swap at most twice per image
    request: a step is approximate only when it schedules exactly one request, and a request
    with images or video never gets a CED plan."""
    source = SCHEDULER.read_text(encoding="utf-8")

    assert re.search(r"if len\(num_scheduled_tokens\) == 1:\n.*\n.*plan = self\.requests\[req_id\]\.r9v_ced\n"
                     r".*\n\s+scheduler_output\.r9v_ced_approx = True", source)
    assert "if self.ced_policy is None or request.resumable or request.mm_features:" in source


def test_the_swap_hooks_sit_on_the_encoder_entry_and_the_approximate_chunk():
    source = OVERLAY.read_text(encoding="utf-8")

    assert re.search(r"def embed_multimodal\(self.*\n.*if _CED_FILES:.*\n\s+_shared_vram\(\)\.acquire\(\"vision\"\)",
                     source)
    assert "proj = _ced_projector(pid)" in source
