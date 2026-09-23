# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import weakref
from pathlib import Path

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .params import allocate_tiered_cold_host_empty, allocate_uva_host_empty
from .tiered_compaction import (
    MASTER_ATTR,
    compact_expert_master,
    is_tiered_expert_master,
    validate_expert_master,
)

from . import full_mutable_cache

logger = logging.getLogger(__name__)

_MANIFEST_ENV = "RADIANCE_TIERED_EXPERT_MANIFEST"
_CACHE_SLOTS_ENV = "QWEN38_TIERED_EXPERT_CACHE_SLOTS"
_CACHE_RANKS_ENV = "QWEN38_TIERED_EXPERT_CACHE_RANKS"
_CACHE_ASYNC_ENV = "QWEN38_TIERED_EXPERT_CACHE_ASYNC"
_CACHE_POLICY_ENV = "QWEN38_TIERED_EXPERT_CACHE_POLICY"
_CACHE_FILL_BATCH_ENV = "R9V_CACHE_FILL_BATCH"
_MAX_CACHE_SLOTS = 128
_NUM_LAYERS = 48
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
_STREAM_COMPACTION_ENV = "QWEN38_TIERED_STREAM_COMPACTION"
_STREAM_HOOK_ATTR = "_r9v_tiered_stream_copy_hook"


def _dynamic_cache_slots(rank: int) -> int:
    if full_mutable_cache.enabled():
        return full_mutable_cache.cache_slots_for_rank(rank)
    raw_slots = os.environ.get(_CACHE_SLOTS_ENV, "0")
    try:
        slots = int(raw_slots)
    except ValueError as error:
        raise ValueError(f"{_CACHE_SLOTS_ENV} must be an integer") from error
    capacity = 192 if os.environ.get("R9V_CACHE192", "0") == "1" else _MAX_CACHE_SLOTS
    if slots > 128 and (
        slots not in {160, 192}
        or os.environ.get(_CACHE_POLICY_ENV, "second_touch_rr") != "lru"
        or os.environ.get(_CACHE_ASYNC_ENV, "0") != "0"
        or os.environ.get(_CACHE_FILL_BATCH_ENV, "1") != "1"
    ):
        raise ValueError("Extended cache capacity requires admitted synchronous single-fill LRU mode")
    if not 0 <= slots <= capacity:
        raise ValueError(
            f"{_CACHE_SLOTS_ENV} must be between 0 and {capacity}"
        )
    try:
        ranks = {
            int(value.strip())
            for value in os.environ.get(_CACHE_RANKS_ENV, "1").split(",")
            if value.strip()
        }
    except ValueError as error:
        raise ValueError(
            f"{_CACHE_RANKS_ENV} must be a comma-separated list of TP ranks"
        ) from error
    return slots if rank in ranks else 0


def _async_cache_enabled() -> bool:
    value = os.environ.get(_CACHE_ASYNC_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_CACHE_ASYNC_ENV} must be 0 or 1")
    return value == "1"


def _cache_policy() -> str:
    value = os.environ.get(_CACHE_POLICY_ENV, "second_touch_rr")
    if value not in {"second_touch_rr", "lru"}:
        raise ValueError(f"{_CACHE_POLICY_ENV} must be second_touch_rr or lru")
    return value



def _cache_fill_batch() -> int:
    value = os.environ.get(_CACHE_FILL_BATCH_ENV, "1")
    if value not in {"1", "2", "4"}:
        raise ValueError(f"{_CACHE_FILL_BATCH_ENV} must be 1, 2, or 4")
    return int(value)

def _validate_hot_lists(manifest: dict, rank: int) -> tuple[int, list[list[int]]]:
    if manifest.get("version") != 1:
        raise ValueError("Tiered expert manifest must use schema version 1")
    num_layers = manifest.get("num_layers")
    num_experts = manifest.get("num_experts")
    if num_layers != _NUM_LAYERS or num_experts != 512:
        raise ValueError(
            "Tiered expert manifest must describe 48 layers and 512 experts"
        )
    try:
        rank_config = manifest["ranks"][str(rank)]
        hot_lists = rank_config["hot_experts_by_layer"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Tiered expert manifest has no TP rank {rank}") from error
    if not isinstance(hot_lists, list) or len(hot_lists) != num_layers:
        raise ValueError("Tiered expert manifest must contain one list per layer")
    for layer_id, hot_ids in enumerate(hot_lists):
        if not isinstance(hot_ids, list) or not hot_ids:
            raise ValueError(f"Layer {layer_id} has no hot expert list")
        if len(hot_ids) != len(set(hot_ids)):
            raise ValueError(f"Layer {layer_id} hot expert list contains duplicates")
        if any(not isinstance(expert, int) for expert in hot_ids):
            raise ValueError(f"Layer {layer_id} hot expert IDs must be integers")
        if min(hot_ids) < 0 or max(hot_ids) >= num_experts:
            raise ValueError(f"Layer {layer_id} hot expert ID is out of range")
    return num_experts, hot_lists


def tiered_expert_manifest_path() -> str | None:
    return os.environ.get(_MANIFEST_ENV) or None


def tiered_expert_layer_id(layer_name: str | None) -> int | None:
    """Backbone layer index this expert module compacts, or None."""
    if not layer_name:
        return None
    match = _LAYER_PATTERN.search(layer_name)
    if match is None:
        return None
    layer_id = extract_layer_index(layer_name)
    if layer_id != int(match.group(1)) or not 0 <= layer_id < _NUM_LAYERS:
        return None
    return layer_id


def _tiered_expert_modules(model: torch.nn.Module):
    from .fused_moe import GGUFMoEMethod

    for module in model.modules():
        if not isinstance(module, RoutedExperts) or not isinstance(
            getattr(module, "quant_method", None), GGUFMoEMethod
        ):
            continue
        layer_id = tiered_expert_layer_id(module.layer_name)
        if layer_id is None:
            continue
        yield layer_id, module


def _stream_compaction_enabled() -> bool:
    value = os.environ.get(_STREAM_COMPACTION_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_STREAM_COMPACTION_ENV} must be 0 or 1")
    return value == "1"


class _StreamCompactionHook:
    def __init__(self, model, module, layer_id, hot_ids, num_experts, manifest_hash):
        self.model_ref = weakref.ref(model)
        self.module_ref = weakref.ref(module)
        self.layer_id = layer_id
        self.hot_ids = tuple(hot_ids)
        self.num_experts = num_experts
        self.manifest_hash = manifest_hash
        self.seen: set[str] = set()
        self.loaded_experts = {shard: set() for shard in ("w1", "w2", "w3")}
        self.finalized = False

    def _validate(self, layer, param, shard_id, loaded_weight, expert_id=None):
        module = self.module_ref()
        if module is None or layer is not module:
            raise RuntimeError("Streaming expert copy callback received the wrong module")
        if shard_id not in {"w1", "w2", "w3"}:
            raise RuntimeError(f"Unsupported streaming expert shard: {shard_id}")
        expected = module.w2_qweight if shard_id == "w2" else module.w13_qweight
        if param is not expected or getattr(module, "_gguf_expert_partition", None) is None:
            raise RuntimeError("Streaming expert copy has the wrong parameter or partition")
        if self.finalized:
            raise RuntimeError(f"Streaming expert write after compaction at layer {self.layer_id}")
        if not is_tiered_expert_master(param):
            raise RuntimeError(f"Streaming expert write reached a non-master at layer {self.layer_id}")
        if loaded_weight.ndim == 3 and loaded_weight.shape[0] == self.num_experts:
            expert_ids = range(self.num_experts)
        elif loaded_weight.ndim == 2 and type(expert_id) is int and 0 <= expert_id < self.num_experts:
            expert_ids = (expert_id,)
        else:
            raise RuntimeError(
                "Streaming compaction requires a full 3-D expert tensor or one "
                f"2-D expert with a valid ID (0..{self.num_experts - 1}); "
                f"layer={self.layer_id}, shape={tuple(loaded_weight.shape)}, expert_id={expert_id}"
            )
        if any(index in self.loaded_experts[shard_id] for index in expert_ids):
            raise RuntimeError(f"Duplicate streaming {shard_id} expert copy at layer {self.layer_id}")
        return expert_ids

    def before_copy(self, layer, param, shard_id, loaded_weight, expert_id=None):
        self._validate(layer, param, shard_id, loaded_weight, expert_id)

    def after_copy(self, layer, param, shard_id, loaded_weight, expert_id=None):
        expert_ids = self._validate(layer, param, shard_id, loaded_weight, expert_id)
        self.loaded_experts[shard_id].update(expert_ids)
        if len(self.loaded_experts[shard_id]) == self.num_experts:
            self.seen.add(shard_id)
        if self.seen == {"w1", "w2", "w3"}:
            model = self.model_ref()
            module = self.module_ref()
            if model is None or module is None:
                raise RuntimeError("Streaming compaction owner was released before completion")
            materialize_hot_expert_cache(
                model,
                only_layer=self.layer_id,
                only_module=module,
                manifest_data=model._r9v_stream_manifest,
                manifest_hash=self.manifest_hash,
            )
            self.finalized = True

    def require_complete(self):
        if self.seen != {"w1", "w2", "w3"} or not self.finalized:
            raise RuntimeError(
                f"Streaming tiered compaction incomplete at layer {self.layer_id}: "
                f"copies={sorted(self.seen)}, counts={ {key: len(ids) for key, ids in self.loaded_experts.items()} }, finalized={self.finalized}"
            )


def _install_stream_compaction(model: torch.nn.Module, manifest: dict, rank: int) -> None:
    if not _stream_compaction_enabled():
        return
    if _cache_policy() != "lru" or _async_cache_enabled() or _cache_fill_batch() != 1:
        raise RuntimeError(
            "Streaming tiered compaction requires synchronous single-fill LRU cache mode"
        )
    modules = list(_tiered_expert_modules(model))
    if len(modules) != _NUM_LAYERS or {layer_id for layer_id, _ in modules} != set(range(_NUM_LAYERS)):
        raise RuntimeError("Streaming tiered compaction requires exactly target layers 0..47")
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _validate_hot_lists(manifest, rank)
    _dynamic_cache_slots(rank)
    for layer_id, module in modules:
        if getattr(module, "_gguf_expert_partition", None) is None:
            raise RuntimeError(f"Streaming tiered compaction requires unequal partition at layer {layer_id}")
    model._r9v_stream_manifest = manifest
    model._r9v_stream_manifest_hash = manifest_hash
    hooks = []
    for layer_id, module in modules:
        hook = _StreamCompactionHook(model, module, layer_id, manifest["ranks"][str(rank)]["hot_experts_by_layer"][layer_id], manifest["num_experts"], manifest_hash)
        module.w13_qweight.__dict__[_STREAM_HOOK_ATTR] = hook
        module.w2_qweight.__dict__[_STREAM_HOOK_ATTR] = hook
        hooks.append(hook)
    model._r9v_stream_hooks = hooks


def prepare_tiered_expert_masters(model: torch.nn.Module) -> int:
    """Load the tiered expert masters into pageable host memory.

    Must run before ``load_weights``. The masters exist only so
    ``materialize_hot_expert_cache`` can copy the hot rows to the accelerator
    and the cold rows into their pinned UVA owner, and they are dropped layer
    by layer as that copy proceeds. Pinning them adds the entire expert set
    (~27.7 GiB per TP rank) to the pinned startup peak for storage no kernel
    ever reads.
    """
    if tiered_expert_manifest_path() is None:
        return 0
    layers = 0
    for _, module in _tiered_expert_modules(model):
        for name in ("w13_qweight", "w2_qweight"):
            setattr(getattr(module, name), MASTER_ATTR, True)
        layers += 1
    if tiered_expert_manifest_path() is not None and _stream_compaction_enabled():
        manifest = json.loads(Path(tiered_expert_manifest_path()).resolve(strict=True).read_text())
        _install_stream_compaction(model, manifest, get_tensor_model_parallel_rank())
    if layers != _NUM_LAYERS:
        raise RuntimeError(
            f"Tiered GGUF expert preparation found {layers} target layers, "
            f"expected {_NUM_LAYERS}"
        )
    return layers


def _compaction_device(parameter: torch.nn.Parameter) -> torch.device:
    if parameter.device.type != "cpu":
        return parameter.device
    # A pageable master keeps the parameter on the host until compaction. Its
    # halves belong on the accelerator this rank loads onto, which is also what
    # the pinned path's UVA view resolved to.
    return torch.device("cuda", torch.cuda.current_device())


def _compact_expert_parameter(
    parameter: torch.nn.Parameter,
    hot_ids: list[int],
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    validate_expert_master(parameter, num_experts)
    if not getattr(parameter, "_vllm_is_uva_offloaded", False):
        raise RuntimeError("Tiered GGUF expert master was not UVA offloaded")
    cpu_master = getattr(parameter, "_vllm_uva_cpu_data", None)
    if cpu_master is None:
        raise RuntimeError("Tiered GGUF expert master has no CPU owner")
    if not is_tiered_expert_master(parameter) and not cpu_master.is_pinned():
        raise RuntimeError("Tiered GGUF expert master has no pinned CPU owner")

    device = _compaction_device(parameter)
    compacted = compact_expert_master(
        cpu_master,
        hot_ids,
        num_experts,
        device,
        retain_full_host_owner=full_mutable_cache.enabled(),
        cold_empty=(
            allocate_tiered_cold_host_empty
            if _stream_compaction_enabled()
            else allocate_uva_host_empty
        ),
    )
    del cpu_master

    torch.cuda.synchronize(device)
    cold = get_accelerator_view_from_cpu_tensor(compacted.cold_owner)
    if full_mutable_cache.enabled():
        cold = full_mutable_cache.alias_on_device(cold, compacted.cold_owner, device)
    # Replacing both references releases this layer's master before the next
    # layer is compacted.
    parameter.data = cold
    parameter._vllm_uva_cpu_data = compacted.cold_owner
    parameter._vllm_is_uva_offloaded = True
    setattr(parameter, MASTER_ATTR, False)
    return (
        compacted.hot,
        compacted.hot_map,
        compacted.cold_map,
        compacted.hot.numel(),
        compacted.cold_owner.numel(),
    )


def materialize_hot_expert_cache(
    model: torch.nn.Module,
    only_layer: int | None = None,
    only_module: torch.nn.Module | None = None,
    manifest_data: dict | None = None,
    manifest_hash: str | None = None,
) -> None:
    manifest_path = tiered_expert_manifest_path()
    if not manifest_path:
        return
    path = Path(manifest_path).resolve(strict=True)
    disk_manifest = json.loads(path.read_text())
    manifest = manifest_data if manifest_data is not None else disk_manifest
    expected_hash = manifest_hash or getattr(model, "_r9v_stream_manifest_hash", None)
    if expected_hash is not None:
        for candidate in (disk_manifest, manifest):
            actual_hash = hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError("Tiered expert manifest changed during streaming compaction")
    if (only_layer is None) != (only_module is None):
        raise RuntimeError("Selected stream compaction requires both layer and module")
    if only_module is not None and tiered_expert_layer_id(only_module.layer_name) != only_layer:
        raise RuntimeError("Selected stream compaction layer/module identity mismatch")
    rank = get_tensor_model_parallel_rank()
    num_experts, hot_lists = _validate_hot_lists(manifest, rank)
    cache_slots = _dynamic_cache_slots(rank)
    cache_policy = _cache_policy()
    cache_fill_batch = _cache_fill_batch() if cache_policy == "lru" else 1
    async_cache = _async_cache_enabled() and cache_slots > 0
    if async_cache and cache_policy != "second_touch_rr":
        raise ValueError(
            f"{_CACHE_ASYNC_ENV}=1 is incompatible with "
            f"{_CACHE_POLICY_ENV}={cache_policy}"
        )
    physical_cache_slots = cache_slots + int(async_cache)

    from .fused_moe import _tiered_iq_moe_hip

    layers = 0
    hot_bytes = 0
    cold_bytes = 0
    dynamic_cache_bytes = 0
    async_initialized = False
    hooks = getattr(model, "_r9v_stream_hooks", None)
    if only_layer is None and hooks is not None:
        for hook in hooks:
            hook.require_complete()
    selected_modules = ([(only_layer, only_module)] if only_module is not None
                        else list(_tiered_expert_modules(model)))
    if only_layer is None and hooks is not None:
        if len(selected_modules) != _NUM_LAYERS or len(hooks) != _NUM_LAYERS:
            raise RuntimeError("Streaming compaction target layer set changed")
        for (layer_id, module), hook in zip(selected_modules, hooks):
            if hook.layer_id != layer_id or hook.module_ref() is not module:
                raise RuntimeError("Streaming compaction target module identity changed")
    for layer_id, module in selected_modules:
        if only_layer is None and hooks is not None:
            layers += 1
            hot_bytes += module._gguf_hot_w13.numel() + module._gguf_hot_w2.numel()
            cold_bytes += getattr(module, "_vllm_stream_cold_bytes", 0)
            if hasattr(module, "_gguf_cache_w13"):
                dynamic_cache_bytes += module._gguf_cache_w13.numel() + module._gguf_cache_w2.numel()
            continue
        hot_ids = full_mutable_cache.hot_ids_for_rank(hot_lists[layer_id], rank)
        hot_w13, hot_map, cold_map, w13_hot, w13_cold = _compact_expert_parameter(
            module.w13_qweight, hot_ids, num_experts
        )
        hot_w2, w2_hot_map, w2_cold_map, w2_hot, w2_cold = _compact_expert_parameter(
            module.w2_qweight, hot_ids, num_experts
        )
        if not torch.equal(hot_map, w2_hot_map) or not torch.equal(
            cold_map, w2_cold_map
        ):
            raise RuntimeError("Tiered GGUF expert maps differ between projections")
        module.register_buffer("_gguf_hot_w13", hot_w13, persistent=False)
        module.register_buffer("_gguf_hot_w2", hot_w2, persistent=False)
        module.register_buffer("_gguf_global_to_hot", hot_map, persistent=False)
        module.register_buffer("_gguf_global_to_cold", cold_map, persistent=False)
        module._vllm_stream_cold_bytes = w13_cold + w2_cold
        if cache_slots:
            device = hot_w13.device
            cache_w13 = torch.empty(
                (physical_cache_slots, *hot_w13.shape[1:]),
                dtype=torch.uint8,
                device=device,
            )
            cache_w2 = torch.empty(
                (physical_cache_slots, *hot_w2.shape[1:]),
                dtype=torch.uint8,
                device=device,
            )
            module.register_buffer("_gguf_cache_w13", cache_w13, persistent=False)
            module.register_buffer("_gguf_cache_w2", cache_w2, persistent=False)
            module.register_buffer(
                "_gguf_global_to_cache",
                torch.full((num_experts,), -1, dtype=torch.int32, device=device),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_tags",
                torch.full(
                    (physical_cache_slots,), -1, dtype=torch.int32, device=device
                ),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_clock",
                torch.zeros(
                    (physical_cache_slots + 1,),
                    dtype=torch.int32,
                    device=device,
                ),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_admission",
                torch.zeros((num_experts,), dtype=torch.int32, device=device),
                persistent=False,
            )
            # prepare calls, fills, routed hits, first-touch bypasses, evictions,
            # async schedules/fallbacks, and sync duplicate fills/served routes.
            module.register_buffer(
                "_gguf_cache_stats",
                torch.zeros((9,), dtype=torch.int32, device=device),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_pending",
                torch.zeros((7 * cache_fill_batch,), dtype=torch.int32, device=device),
                persistent=False,
            )
            module._gguf_cache_layer_id = layer_id
            module._gguf_cache_policy = cache_policy
            module._gguf_cache_fill_batch = cache_fill_batch
            if cache_policy == "lru" and not hasattr(
                _tiered_iq_moe_hip(), "tiered_iq_moe_cache_lru_prepare"
            ):
                raise RuntimeError(
                    "The loaded tiered HIP extension does not support "
                    "the synchronous LRU expert cache"
                )
            if cache_policy == "lru" and cache_fill_batch > 1 and not hasattr(
                _tiered_iq_moe_hip(), "tiered_iq_moe_cache_lru_prepare_many"
            ):
                raise RuntimeError(
                    "The loaded tiered HIP extension does not support "
                    "batched synchronous LRU expert-cache fills"
                )
            if async_cache and not async_initialized:
                extension = _tiered_iq_moe_hip()
                if not all(
                    hasattr(extension, name)
                    for name in (
                        "tiered_iq_moe_cache_async_init",
                        "tiered_iq_moe_cache_async_prepare",
                        "tiered_iq_moe_cache_async_commit",
                    )
                ):
                    raise RuntimeError(
                        "The loaded tiered HIP extension does not support "
                        "asynchronous expert-cache fills"
                    )
                with torch.cuda.device(device):
                    extension.tiered_iq_moe_cache_async_init()
                async_initialized = True
            dynamic_cache_bytes += cache_w13.numel() + cache_w2.numel()
        if full_mutable_cache.enabled():
            full_mutable_cache.register(module, rank, hot_ids)
        layers += 1
        hot_bytes += w13_hot + w2_hot
        cold_bytes += w13_cold + w2_cold
    if only_layer is not None and layers != 1:
        raise RuntimeError("Streaming compaction did not finalize its selected layer")
    if only_layer is None and layers != _NUM_LAYERS:
        raise RuntimeError(
            f"Tiered GGUF cache found {layers} target layers, expected {_NUM_LAYERS}"
        )
    parameters = only_module.named_parameters() if only_module is not None else model.named_parameters()
    for name, parameter in parameters:
        if is_tiered_expert_master(parameter):
            raise RuntimeError(
                f"Tiered GGUF expert master {name} was never compacted and is "
                "still pageable host memory"
            )
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()
    if only_layer is not None:
        return
    hooks = getattr(model, "_r9v_stream_hooks", None)
    if hooks is not None:
        for hook in hooks:
            hook.require_complete()
        for _, module in _tiered_expert_modules(model):
            for parameter in (module.w13_qweight, module.w2_qweight):
                parameter.__dict__.pop(_STREAM_HOOK_ATTR, None)
        delattr(model, "_r9v_stream_hooks")
        delattr(model, "_r9v_stream_manifest")
        delattr(model, "_r9v_stream_manifest_hash")
    logger.warning(
        "Tiered GGUF experts ready on TP rank %d: layers=%d hot=%.3f GiB "
        "cold_UVA=%.3f GiB dynamic_cache=%.3f GiB cache_slots=%d "
        "physical_cache_slots=%d cache_policy=%s async_cache=%s manifest=%s",
        rank,
        layers,
        hot_bytes / 1024**3,
        cold_bytes / 1024**3,
        dynamic_cache_bytes / 1024**3,
        cache_slots,
        physical_cache_slots,
        cache_policy,
        async_cache,
        path,
    )
