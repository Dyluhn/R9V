"""CPU-only tests for the tiered GGUF expert compaction and host-owner policy.

The module under test lives in the vllm-gguf-plugin submodule and deliberately
imports nothing but torch, so it can be loaded by path here without a vLLM
install. These tests never touch an accelerator: every tensor is a CPU tensor
and every pinned allocation is made through an injected recorder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE_PATH = (
    Path(__file__).parents[1]
    / "vendor"
    / "vllm-gguf-plugin"
    / "vllm_gguf_plugin"
    / "quantization"
    / "tiered_compaction.py"
)
SPEC = importlib.util.spec_from_file_location("r9v_tiered_compaction", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compaction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compaction
SPEC.loader.exec_module(compaction)

CPU = torch.device("cpu")


class HostRecorder:
    """Stand-in for a pinned allocator that records every request."""

    def __init__(self) -> None:
        self.requests: list[tuple[tuple[int, ...], torch.dtype]] = []

    def __call__(self, shape, dtype):
        self.requests.append((tuple(shape), dtype))
        return torch.empty(tuple(shape), dtype=dtype, device="cpu")

    @property
    def total_bytes(self) -> int:
        total = 0
        for shape, dtype in self.requests:
            elements = 1
            for extent in shape:
                elements *= extent
            total += elements * torch.empty((), dtype=dtype).element_size()
        return total


def _master(num_experts: int, rows: int = 6, packed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260901)
    return torch.randint(
        0,
        256,
        (num_experts, rows, packed),
        dtype=torch.uint8,
        generator=generator,
    )


def _reference_compaction(master: torch.Tensor, hot_ids: list[int], num_experts: int):
    """The pre-change compaction, transcribed onto CPU tensors.

    Mirrors the original ``_compact_expert_parameter``: hot rows gathered from
    the parameter view in manifest order, cold rows gathered from the host
    master in ascending order, and both maps built by scattering aranges.
    """
    hot_index = torch.tensor(hot_ids, dtype=torch.long)
    hot = master.index_select(0, hot_index).contiguous()

    hot_set = set(hot_ids)
    cold_ids = [expert for expert in range(num_experts) if expert not in hot_set]
    cold_index = torch.tensor(cold_ids, dtype=torch.long)
    cold = torch.empty((len(cold_ids), *master.shape[1:]), dtype=master.dtype)
    torch.index_select(master, 0, cold_index, out=cold)

    hot_map = torch.full((num_experts,), -1, dtype=torch.int32)
    cold_map = torch.full((num_experts,), -1, dtype=torch.int32)
    hot_map[hot_index] = torch.arange(len(hot_ids), dtype=torch.int32)
    cold_map[cold_index] = torch.arange(len(cold_ids), dtype=torch.int32)
    return hot, cold, hot_map, cold_map


@pytest.mark.parametrize(
    ("num_experts", "hot_ids"),
    [
        (4, [1, 3]),
        (8, [0]),
        (8, [7, 0, 4]),
        (16, [11, 2, 15, 3, 8]),
        (5, [4, 3, 2, 1, 0]),
    ],
)
def test_compaction_is_byte_identical_to_the_previous_path(num_experts, hot_ids) -> None:
    master = _master(num_experts)
    reference = _reference_compaction(master, hot_ids, num_experts)

    result = compaction.compact_expert_master(
        master,
        hot_ids,
        num_experts,
        CPU,
        cold_empty=HostRecorder(),
        stage_empty=HostRecorder(),
    )

    assert torch.equal(result.hot, reference[0])
    assert torch.equal(result.cold_owner, reference[1])
    assert torch.equal(result.hot_map, reference[2])
    assert torch.equal(result.cold_map, reference[3])


def test_hot_rows_keep_manifest_order() -> None:
    num_experts = 8
    hot_ids = [5, 1, 6]
    master = _master(num_experts)

    result = compaction.compact_expert_master(
        master,
        hot_ids,
        num_experts,
        CPU,
        cold_empty=HostRecorder(),
        stage_empty=HostRecorder(),
    )

    for row, expert in enumerate(hot_ids):
        assert torch.equal(result.hot[row], master[expert])
        assert int(result.hot_map[expert]) == row
        assert int(result.cold_map[expert]) == -1
    for row, expert in enumerate(compaction.cold_expert_ids(hot_ids, num_experts)):
        assert torch.equal(result.cold_owner[row], master[expert])
        assert int(result.cold_map[expert]) == row
        assert int(result.hot_map[expert]) == -1


def test_compaction_pins_only_the_cold_owner_and_one_hot_slice() -> None:
    num_experts, rows, packed = 16, 6, 5
    hot_ids = [0, 3, 9]
    master = _master(num_experts, rows, packed)
    cold = HostRecorder()
    stage = HostRecorder()

    compaction.compact_expert_master(
        master, hot_ids, num_experts, CPU, cold_empty=cold, stage_empty=stage
    )

    # Exactly two host allocations: the cold owner the runtime keeps, and a
    # staging slice the size of one layer's hot set. The master is never one of
    # them, so it never reaches a pinned allocator.
    assert cold.requests == [((num_experts - len(hot_ids), rows, packed), torch.uint8)]
    assert stage.requests == [((len(hot_ids), rows, packed), torch.uint8)]
    assert stage.total_bytes == len(hot_ids) * rows * packed
    assert cold.total_bytes == (num_experts - len(hot_ids)) * rows * packed


def test_every_layer_requests_the_same_staging_shape() -> None:
    """Identical requests let the caching host allocator reuse one block."""
    num_experts = 12
    stage = HostRecorder()
    for layer in range(4):
        hot_ids = [(layer + offset) % num_experts for offset in (0, 5, 7)]
        compaction.compact_expert_master(
            _master(num_experts),
            hot_ids,
            num_experts,
            CPU,
            cold_empty=HostRecorder(),
            stage_empty=stage,
        )
    assert len(set(stage.requests)) == 1


def test_master_layout_is_validated() -> None:
    with pytest.raises(TypeError):
        compaction.validate_expert_master(torch.zeros((4, 2, 2), dtype=torch.int8), 4)
    with pytest.raises(TypeError):
        compaction.validate_expert_master(torch.zeros((4, 2), dtype=torch.uint8), 4)
    with pytest.raises(ValueError):
        compaction.validate_expert_master(torch.zeros((3, 2, 2), dtype=torch.uint8), 4)
    with pytest.raises(ValueError):
        compaction.validate_expert_master(
            torch.zeros((4, 2, 4), dtype=torch.uint8)[:, :, ::2], 4
        )


def test_a_tiered_master_never_gets_a_pinned_buffer_or_a_view() -> None:
    param = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
    setattr(param, compaction.MASTER_ATTR, True)
    assert compaction.is_tiered_expert_master(param)

    plan = compaction.plan_uva_host_owner(
        pin_memory=True, tiered_master=compaction.is_tiered_expert_master(param)
    )
    assert plan == compaction.HostOwnerPlan(pin_memory=False, accelerator_view=False)

    pinned = HostRecorder()
    owner = compaction.allocate_uva_host_owner(
        (8, 4), torch.uint8, plan, pinned_empty=pinned
    )
    assert pinned.requests == []
    assert owner.device.type == "cpu"
    assert tuple(owner.shape) == (8, 4)


@pytest.mark.parametrize("pin_memory", [True, False])
def test_without_the_manifest_marker_the_owner_policy_is_unchanged(pin_memory) -> None:
    param = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
    assert not compaction.is_tiered_expert_master(param)

    plan = compaction.plan_uva_host_owner(
        pin_memory=pin_memory, tiered_master=compaction.is_tiered_expert_master(param)
    )
    assert plan == compaction.HostOwnerPlan(
        pin_memory=pin_memory, accelerator_view=True
    )

    pinned = HostRecorder()
    compaction.allocate_uva_host_owner((8, 4), torch.uint8, plan, pinned_empty=pinned)
    assert pinned.requests == ([((8, 4), torch.uint8)] if pin_memory else [])
