# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the full mutable cache's permanent expert pins and smaller host copy.

Rank 1 pins 400 experts per layer; the host copy holds only the other 112 and indexes them
through cold_map. Covers the shipped pin file, warmstart order, host row index, compaction
bytes, and a replay through the real planner core (the CPU build of the shipped sources)
showing pinned experts are never evicted or fetched.
"""
from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

RUNTIME = Path(__file__).resolve().parents[1] / "runtimes/qwen38-flash-next-gfx1201-mtp4-v3"
OVERLAYS = RUNTIME / "overlays"
CACHE_SOURCES = RUNTIME / "sources/cache"
PINS = OVERLAYS / "full_mutable_pins.json"
PLACEMENT_RANK1 = list(range(511, 83, -1))  # 428 ids in a priority order unrelated to the pins


def load(name: str, path: Path):
    """Import an overlay by path without writing __pycache__ into the pinned overlay
    directory, which the launcher would refuse as an unpinned file."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture(autouse=True)
def mutable_cache_on(monkeypatch):
    monkeypatch.setenv("R9V_FULL_MUTABLE_CACHE", "1")


cache = load("r9v_full_mutable_cache", OVERLAYS / "full_mutable_cache.py")
compaction = load("r9v_overlay_tiered_compaction", OVERLAYS / "tiered_compaction.py")
TABLE = cache.pins(PINS)


def cpu_empty(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def compact_rank1(layer: int):
    master = torch.arange(512, dtype=torch.uint8).repeat_interleave(8).reshape(512, 2, 4)
    master[:, 1, :] = torch.arange(512).div(256, rounding_mode="floor").to(torch.uint8)[:, None]
    hot_ids = cache.hot_ids_for_rank(PLACEMENT_RANK1, 1, layer, TABLE)
    result = compaction.compact_expert_master(
        master, hot_ids, 512, torch.device("cpu"), cold_empty=cpu_empty,
        stage_empty=cpu_empty, host_ids=cache.host_ids(1, layer, TABLE))
    return master, hot_ids, result


def hot_map_for(hot_ids: list[int]) -> list[int]:
    hot_map = [-1] * 512
    for slot, expert in enumerate(hot_ids):
        hot_map[expert] = slot
    return hot_map


def test_pin_file_has_400_unique_rank1_pins_per_layer_and_none_on_rank0():
    assert len(TABLE[1]) == 48
    assert {len(set(layer)) for layer in TABLE[1]} == {400}
    assert TABLE[0] == ((),) * 48


def test_edited_pin_file_is_refused(tmp_path):
    edited = tmp_path / "pins.json"
    edited.write_bytes(PINS.read_bytes().replace(b'"split":"train"', b'"split":"all"'))

    with pytest.raises(RuntimeError, match="Unreviewed"):
        cache.pins.__wrapped__(edited)


def test_rank1_warmstart_puts_pins_in_the_first_400_slots():
    hot = cache.hot_ids_for_rank(PLACEMENT_RANK1, 1, 7, TABLE)

    assert len(hot) == 427 and len(set(hot)) == 427
    assert hot[:400] == list(TABLE[1][7])


def test_rank0_warmstart_is_unchanged():
    placement = list(range(100, 162))

    assert cache.hot_ids_for_rank(placement, 0, 3, TABLE) == placement


def test_host_ids_exclude_exactly_the_pins():
    assert len(cache.host_ids(1, 5, TABLE)) == 112
    assert set(cache.host_ids(1, 5, TABLE)).isdisjoint(TABLE[1][5])
    assert cache.host_ids(0, 5, TABLE) == list(range(512))


def test_host_copy_holds_112_rows_each_equal_to_its_expert():
    master, _, result = compact_rank1(11)

    assert result.cold_owner.shape[0] == 112
    for expert in cache.host_ids(1, 11, TABLE):
        assert torch.equal(result.cold_owner[result.cold_map[expert]], master[expert])


def test_pinned_experts_have_no_host_row():
    _, _, result = compact_rank1(11)

    assert set(result.cold_map[list(TABLE[1][11])].tolist()) == {-1}


def test_map_check_accepts_the_real_layout():
    _, hot_ids, result = compact_rank1(2)

    cache.check_maps(hot_map_for(hot_ids), result.cold_map.tolist(), hot_ids, 1)


def test_map_check_refuses_a_pinned_expert_with_a_host_row():
    _, hot_ids, _ = compact_rank1(2)

    with pytest.raises(RuntimeError, match="Pinned expert"):
        cache.check_maps(hot_map_for(hot_ids), list(range(512)), hot_ids, 1)


def test_compaction_refuses_an_expert_neither_hot_nor_on_host():
    master = torch.zeros((512, 1, 1), dtype=torch.uint8)

    with pytest.raises(ValueError, match="Every expert"):
        compaction.compact_expert_master(
            master, [0], 512, torch.device("cpu"), cold_empty=cpu_empty, stage_empty=cpu_empty,
            host_ids=list(range(2, 512)))


@pytest.fixture(scope="module")
def planner_core(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("no g++: cannot build the planner core for the replay")
    out = tmp_path_factory.mktemp("planner") / "core.so"
    subprocess.run(["g++", "-O1", "-std=c++17", "-shared", "-fPIC", "-I", str(CACHE_SOURCES),
                    str(CACHE_SOURCES / "native_cpu_shim.cpp"), "-o", str(out)],
                   check=True, timeout=120)
    return ctypes.CDLL(str(out))


def test_planner_never_evicts_or_admits_a_pinned_expert(planner_core):
    layer, capacity = 9, 427
    hot_ids = cache.hot_ids_for_rank(PLACEMENT_RANK1, 1, layer, TABLE)
    offsets = (ctypes.c_size_t * 11)()
    planner_core.c_get_arena_offsets(1, capacity, offsets)
    arena = np.zeros(32768, dtype=np.uint8)
    pointer = arena.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    warm = np.array(hot_ids, dtype=np.int32)
    assert planner_core.c_init_arena(pointer, warm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     427, 1, capacity, None, None, None) == 0
    pool_map = arena[offsets[3]:offsets[3] + 2048].view(np.int32)
    miss_ids = arena[offsets[5]:offsets[5] + 256].view(np.int32)
    pinned = np.array(TABLE[1][layer])
    host_rows = set(cache.host_ids(1, layer, TABLE))
    routes = np.random.default_rng(0).integers(0, 512, size=(3000, 50), dtype=np.int32)
    admitted_total = 0
    for event in routes:
        admitted = planner_core.c_device_planner(
            pointer, event.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 50, 1, capacity, 64, 1)
        assert admitted >= 0
        assert set(miss_ids[:admitted].tolist()) <= host_rows
        assert (pool_map[pinned] == np.arange(400)).all()
        admitted_total += admitted
    assert admitted_total > 10000  # the 27 rotating slots really rotated


def test_pin_selection_takes_the_most_routed_experts_and_breaks_ties_to_the_lower_id():
    make_pins = load("r9v_make_pins", Path(__file__).resolve().parents[1] / "tools/pin_sim/make_pins.py")
    routes = np.array([[7, 7, 3, 3, 9, 1]])

    assert make_pins.top_experts(routes, 3) == [3, 7, 1]
