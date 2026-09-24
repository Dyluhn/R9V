# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side policy and copy logic for the tiered GGUF expert loader.

This module deliberately imports nothing but ``torch`` so the selection, map
construction and host-owner allocation policy can be exercised on CPU tensors
without a GPU or a vLLM installation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch

HostEmpty = Callable[[tuple[int, ...], torch.dtype], torch.Tensor]

# Marks a GGUF expert parameter whose host storage is only a copy source for
# the hot/cold compaction pass.
MASTER_ATTR = "_vllm_tiered_expert_master"


def is_tiered_expert_master(param: Any) -> bool:
    """Whether this parameter's host storage is a compaction source."""
    return bool(getattr(param, MASTER_ATTR, False))


class HostOwnerPlan(NamedTuple):
    """How a UVA-offloaded GGUF parameter's host storage is created."""

    pin_memory: bool
    accelerator_view: bool


def plan_uva_host_owner(*, pin_memory: bool, tiered_master: bool) -> HostOwnerPlan:
    """Decide the host storage for a UVA-offloaded GGUF parameter.

    A tiered expert master exists only so ``compact_expert_master`` can copy
    out of it, and it is dropped as soon as that copy completes. Pinning it
    would put the whole expert set into the pinned startup peak, and taking an
    accelerator view would pin it anyway: ``get_cuda_view_from_cpu_tensor``
    falls back to ``cudaHostAlloc`` plus a copy for unpinned input. Every other
    UVA owner keeps the pinned allocation the runtime's host-UVA reads need.
    """
    if tiered_master:
        return HostOwnerPlan(pin_memory=False, accelerator_view=False)
    return HostOwnerPlan(pin_memory=pin_memory, accelerator_view=True)


def default_pinned_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Pinned host storage for a UVA owner the runtime keeps for the whole run."""
    return torch.empty(shape, dtype=dtype, device="cpu").pin_memory()


def default_stage_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Transient pinned staging storage, allocated pinned without a copy."""
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def allocate_uva_host_owner(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    plan: HostOwnerPlan,
    *,
    pinned_empty: HostEmpty = default_pinned_empty,
) -> torch.Tensor:
    """Allocate the host storage described by ``plan``."""
    if plan.pin_memory:
        return pinned_empty(shape, dtype)
    return torch.empty(shape, dtype=dtype, device="cpu")


def validate_expert_master(master: torch.Tensor, num_experts: int) -> None:
    if master.dtype != torch.uint8 or master.dim() != 3:
        raise TypeError("Tiered GGUF expert weights must be packed uint8 tensors")
    if master.shape[0] != num_experts or not master.is_contiguous():
        raise ValueError("Tiered GGUF expert master has an unexpected layout")


def cold_expert_ids(hot_ids: list[int], num_experts: int) -> list[int]:
    """Experts left in host memory, in ascending global order."""
    hot_set = set(hot_ids)
    return [expert for expert in range(num_experts) if expert not in hot_set]


def build_expert_maps(
    hot_ids: list[int],
    cold_ids: list[int],
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global expert id -> hot row and cold row, with -1 for the other tier."""
    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    cold_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    hot_index = torch.tensor(hot_ids, dtype=torch.long, device=device)
    cold_index = torch.tensor(cold_ids, dtype=torch.long, device=device)
    hot_map[hot_index] = torch.arange(len(hot_ids), dtype=torch.int32, device=device)
    cold_map[cold_index] = torch.arange(len(cold_ids), dtype=torch.int32, device=device)
    return hot_map, cold_map


class CompactedExperts(NamedTuple):
    hot: torch.Tensor
    cold_owner: torch.Tensor
    hot_map: torch.Tensor
    cold_map: torch.Tensor


def compact_expert_master(
    cpu_master: torch.Tensor,
    hot_ids: list[int],
    num_experts: int,
    device: torch.device,
    *,
    cold_empty: HostEmpty = default_pinned_empty,
    stage_empty: HostEmpty = default_stage_empty,
    host_ids: list[int] | None = None,
) -> CompactedExperts:
    """Split one layer's expert master into a hot device set and a cold owner.

    ``cpu_master`` may be pinned (legacy path) or pageable (tiered path); the
    selected bytes are identical either way. The hot rows are staged through a
    pinned buffer the size of one layer's hot slice so the host-to-device copy
    keeps full PCIe bandwidth even when the master itself is pageable. The
    stage is released before returning, and every layer requests the same two
    shapes, so PyTorch's caching host allocator reuses a single block per
    projection for the whole model.
    """
    validate_expert_master(cpu_master, num_experts)
    row_shape = tuple(cpu_master.shape[1:])
    # ``host_ids`` (mutable cache): the host keeps these rows, which may overlap the
    # hot set; by default it keeps exactly the experts that are not hot.
    cold_ids = cold_expert_ids(hot_ids, num_experts) if host_ids is None else list(host_ids)
    if cold_ids != sorted(set(cold_ids)) or set(hot_ids) | set(cold_ids) != set(range(num_experts)):
        raise ValueError("Every expert must be hot or on host, and host ids must be ascending")

    hot_index = torch.tensor(hot_ids, dtype=torch.long, device="cpu")
    stage = stage_empty((len(hot_ids), *row_shape), cpu_master.dtype)
    torch.index_select(cpu_master, 0, hot_index, out=stage)
    hot = torch.empty((len(hot_ids), *row_shape), dtype=cpu_master.dtype, device=device)
    # A blocking copy: the stage is freed below and an explicitly allocated
    # host buffer is not tracked by the caching host allocator's stream events.
    hot.copy_(stage)
    del stage

    cold_index = torch.tensor(cold_ids, dtype=torch.long, device="cpu")
    cold_owner = cold_empty((len(cold_ids), *row_shape), cpu_master.dtype)
    torch.index_select(cpu_master, 0, cold_index, out=cold_owner)
    hot_map, cold_map = build_expert_maps(hot_ids, cold_ids, num_experts, device)
    return CompactedExperts(hot, cold_owner, hot_map, cold_map)
