# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
from functools import partial
from pathlib import Path

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .expert_partition import configure_expert_partition
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, logger

_Q8_BF16_MOE_HIP = None
_IQ_MOE_HIP = None
_TIERED_IQ_MOE_HIP = None
_MOE_WMMA_HIP = None
_MOE_WMMA_SO_ENV = "R9V_MOE_WMMA_SO"
# Group 32 selects the gfx12 int8 WMMA prefill kernel; 4/8/16 select the
# original bit-exact grouped GEMV kernel.
_WMMA_PREFILL_GROUP_SIZE = 32
_TIERED_IQ_MOE_VARIANTS = {
    "generic": 0,
    "auto": 1,
    "u2": 2,
    "u5": 5,
    "u10": 10,
    "reuse3": 30,
    "reuse3v2": 31,
}
_FUSED_SHARED_EPILOGUE_ENV = "VLLM_GGUF_FUSED_MOE_SHARED_EPILOGUE"
_TIERED_CACHE_ASYNC_ENV = "QWEN38_TIERED_EXPERT_CACHE_ASYNC"
_TIERED_CACHE_POLICY_ENV = "QWEN38_TIERED_EXPERT_CACHE_POLICY"
_TIERED_CACHE_FILL_BATCH_ENV = "R9V_CACHE_FILL_BATCH"
_TIERED_PREFILL_GROUP_ENV = "QWEN38_TIERED_PREFILL_GROUP_SIZE"


def _fused_shared_epilogue_enabled() -> bool:
    value = os.environ.get(_FUSED_SHARED_EPILOGUE_ENV, "0")
    if value not in {"0", "1"}:
        raise RuntimeError(f"{_FUSED_SHARED_EPILOGUE_ENV} must be 0 or 1")
    return value == "1"


def _tiered_cache_async_enabled() -> bool:
    value = os.environ.get(_TIERED_CACHE_ASYNC_ENV, "0")
    if value not in {"0", "1"}:
        raise RuntimeError(f"{_TIERED_CACHE_ASYNC_ENV} must be 0 or 1")
    if value == "0":
        return False
    # A breakable capture can end before layer 47, so it cannot safely leave
    # work on the auxiliary stream for the single end-of-model join.  Preserve
    # the existing same-stream prepare path in that mode.
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    except ImportError:
        return True
    return not BreakableCUDAGraphCapture.is_active()


def _tiered_cache_policy() -> str:
    value = os.environ.get(_TIERED_CACHE_POLICY_ENV, "second_touch_rr")
    if value not in {"second_touch_rr", "lru"}:
        raise RuntimeError(f"{_TIERED_CACHE_POLICY_ENV} must be second_touch_rr or lru")
    return value



def _tiered_cache_fill_batch() -> int:
    value = os.environ.get(_TIERED_CACHE_FILL_BATCH_ENV, "1")
    if value not in {"1", "2", "4"}:
        raise RuntimeError(
            f"{_TIERED_CACHE_FILL_BATCH_ENV} must be 1, 2, or 4"
        )
    return int(value)

def _tiered_prefill_group_size() -> int:
    value = os.environ.get(_TIERED_PREFILL_GROUP_ENV, "0")
    if value not in {"0", "4", "8", "16", "32"}:
        raise RuntimeError(f"{_TIERED_PREFILL_GROUP_ENV} must be 0, 4, 8, 16, or 32")
    return int(value)


def _moe_wmma_hip():
    """Load the gfx12 int8 WMMA grouped prefill extension once."""
    global _MOE_WMMA_HIP
    if _MOE_WMMA_HIP is not None:
        return _MOE_WMMA_HIP
    location = os.environ.get(_MOE_WMMA_SO_ENV)
    if not location:
        raise RuntimeError(
            f"{_TIERED_PREFILL_GROUP_ENV}={_WMMA_PREFILL_GROUP_SIZE} requires "
            f"{_MOE_WMMA_SO_ENV} to name the WMMA prefill extension"
        )
    path = Path(location)
    spec = importlib.util.spec_from_file_location("r9v_moe_wmma", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load WMMA MoE prefill extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MOE_WMMA_HIP = module
    logger.info_once("Using gfx12 int8 WMMA grouped MoE prefill kernels")
    return module


def _finalize_moe_output(
    routed: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_output: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    routed = routed.reshape(
        topk_weights.shape[0], topk_weights.shape[1], output.shape[1]
    )
    if shared_output is None:
        routed.mul_(topk_weights.unsqueeze(-1))
        ops.moe_sum(routed, output)
        return

    from ..triton.fused_moe.weighted_sum import fused_weighted_moe_sum_shared

    fused_weighted_moe_sum_shared(routed, topk_weights, shared_output, output)


def _q8_bf16_moe_hip():
    """Load the opt-in RDNA4 Q8_0/BF16 MoE GEMV extension once per worker."""
    global _Q8_BF16_MOE_HIP
    if _Q8_BF16_MOE_HIP is not None:
        return _Q8_BF16_MOE_HIP
    path = Path(os.environ["QWEN38_Q8_BF16_MOE_HIP_SO"])
    spec = importlib.util.spec_from_file_location("qwen38_q8_bf16_moe_hip", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Q8/BF16 MoE extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _Q8_BF16_MOE_HIP = module
    logger.info_once("Using the R9V-derived Q8_0/BF16 MoE GEMV kernel")
    return module


def _iq_moe_hip():
    """Load the opt-in RDNA4 IQ2_XS/IQ4_NL MoE extension once."""
    global _IQ_MOE_HIP
    if _IQ_MOE_HIP is not None:
        return _IQ_MOE_HIP
    path = Path(os.environ["QWEN38_IQ_MOE_HIP_SO"])
    spec = importlib.util.spec_from_file_location("qwen38_iq_moe_hip", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load IQ MoE extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _IQ_MOE_HIP = module
    logger.info_once("Using the R9V-derived IQ2_XS/IQ4_NL MoE GEMV kernel")
    return module


_R9V_CACHE80_MODULE = None

def _r9v_cache80_module():
    global _R9V_CACHE80_MODULE
    if _R9V_CACHE80_MODULE is None:
        import importlib.util
        use128 = os.environ.get("R9V_CACHE128", "0") == "1"
        use192 = os.environ.get("R9V_CACHE192", "0") == "1"
        use_planpublish = os.environ.get("R9V_CACHE_PLANPUBLISH", "0") == "1"
        spec = importlib.util.spec_from_file_location("r9v_cache_planpublish" if use_planpublish else "r9v_cache192" if use192 else "r9v_cache128" if use128 else "r9v_cache80", os.environ["R9V_CACHE_PLANPUBLISH_SO" if use_planpublish else "R9V_CACHE192_SO" if use192 else "R9V_CACHE128_SO" if use128 else "R9V_CACHE80_SO"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R9V_CACHE80_MODULE = module
        if use_planpublish:
            print("[r9v] synchronous cache plan/publication fusion enabled", flush=True)
    return _R9V_CACHE80_MODULE


_R9V_ROUTEPLAN_MODULE = None
_R9V_EXPERT_SILU_QUANT_MODULE = None
_R9V_EXPERT_SILU_QUANT_LOGGED = False

def _r9v_routeplan_module():
    global _R9V_ROUTEPLAN_MODULE
    if _R9V_ROUTEPLAN_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("r9v_expert_routeplan", os.environ["R9V_EXPERT_ROUTEPLAN_SO"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R9V_ROUTEPLAN_MODULE = module
    return _R9V_ROUTEPLAN_MODULE


def _r9v_expert_silu_quant_module():
    global _R9V_EXPERT_SILU_QUANT_MODULE
    if _R9V_EXPERT_SILU_QUANT_MODULE is None:
        path = Path(os.environ["R9V_EXPERT_SILU_QUANT_SO"])
        spec = importlib.util.spec_from_file_location("r9v_expert_silu_quant", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load expert SiLU/Q8 extension from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R9V_EXPERT_SILU_QUANT_MODULE = module
    return _R9V_EXPERT_SILU_QUANT_MODULE


def _tiered_iq_moe_hip():
    """Load the Q4 mixed VRAM/UVA expert GEMV extension once."""
    global _TIERED_IQ_MOE_HIP
    if _TIERED_IQ_MOE_HIP is not None:
        return _TIERED_IQ_MOE_HIP
    path = Path(os.environ["QWEN38_TIERED_IQ_MOE_HIP_SO"])
    spec = importlib.util.spec_from_file_location("qwen38_tiered_iq_moe_hip", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tiered IQ MoE extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TIERED_IQ_MOE_HIP = module
    logger.info_once("Using mixed VRAM/UVA Q4 expert GEMV kernels")
    return module


def _tiered_iq_moe_variant() -> tuple[str, int]:
    name = os.environ.get("QWEN38_TIERED_IQ_MOE_VARIANT", "generic").lower()
    if name not in _TIERED_IQ_MOE_VARIANTS:
        choices = ", ".join(_TIERED_IQ_MOE_VARIANTS)
        raise RuntimeError(
            f"Unknown QWEN38_TIERED_IQ_MOE_VARIANT={name!r}; choose from {choices}"
        )
    return name, _TIERED_IQ_MOE_VARIANTS[name]


def _sanitize_moe_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map padded/invalid expert ids to a safe row with zero contribution.

    vLLM uses ``-1`` expert ids during dummy/profile forwards and can retain
    sentinel slots for padded tokens.  Some GGUF kernels do pointer arithmetic
    before their in-kernel invalid-expert guard takes effect on ROCm.  Routing
    those slots through expert zero while forcing their weights to zero keeps
    every address in bounds without changing the mathematical result.
    """
    valid = (topk_ids >= 0) & (topk_ids < num_experts)
    safe_ids = topk_ids.masked_fill(~valid, 0)
    safe_weights = topk_weights.masked_fill(~valid, 0)
    return safe_ids, safe_weights


def _fused_moe_gguf_impl(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    hot_w1: torch.Tensor | None = None,
    hot_w2: torch.Tensor | None = None,
    hot_map: torch.Tensor | None = None,
    cold_map: torch.Tensor | None = None,
    cache_w1: torch.Tensor | None = None,
    cache_w2: torch.Tensor | None = None,
    cache_map: torch.Tensor | None = None,
    cache_tags: torch.Tensor | None = None,
    cache_clock: torch.Tensor | None = None,
    cache_admission: torch.Tensor | None = None,
    cache_stats: torch.Tensor | None = None,
    cache_pending: torch.Tensor | None = None,
    cache_layer_id: int = -1,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)
    tiered = all(tensor is not None for tensor in (hot_w1, hot_w2, hot_map, cold_map))
    cache_tensors = (
        cache_w1,
        cache_w2,
        cache_map,
        cache_tags,
        cache_clock,
        cache_admission,
        cache_stats,
        cache_pending,
    )
    dynamic_cache = all(tensor is not None for tensor in cache_tensors)
    if any(tensor is not None for tensor in cache_tensors) and not dynamic_cache:
        raise RuntimeError("Tiered expert cache state is incomplete")
    if dynamic_cache and not tiered:
        raise RuntimeError("Dynamic expert caching requires tiered weights")
    if dynamic_cache and not 0 <= cache_layer_id < 48:
        raise RuntimeError("Dynamic expert cache requires a valid layer ID")
    if tiered and os.environ.get("QWEN38_USE_TIERED_IQ_MOE_HIP") != "1":
        raise RuntimeError("Tiered GGUF weights require the tiered HIP kernel")
    # The rebuilt HIP MoE extension rejects vLLM's padded -1 expert sentinel
    # before doing expert-pointer arithmetic.  Older extensions do not, so
    # preserve the tensor-level sanitizer unless that guarded extension was
    # explicitly mounted by the launcher.
    if os.environ.get("VLLM_GGUF_NATIVE_SAFE_MOE_IDS") != "1":
        num_experts = hot_map.numel() if hot_map is not None else w1.shape[0]
        topk_ids, topk_weights = _sanitize_moe_routing(
            topk_ids,
            topk_weights,
            num_experts,
        )

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(activation_enum, out, inp)
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    async_cache_active = False
    use_expert_silu_quant = False  # generic-variant fallback also reaches final activation
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
        and not tiered
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        _finalize_moe_output(out, topk_weights, shared_output, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        use_q8_bf16_hip = (
            os.environ.get("QWEN38_USE_Q8_BF16_MOE_HIP") == "1"
            and qweight_type == 8
            and qweight_type2 == 8
        )
        use_iq_hip = (
            os.environ.get("QWEN38_USE_IQ_MOE_HIP") == "1"
            and qweight_type == 17
            and qweight_type2 == 20
        )
        if tiered:
            assert hot_w1 is not None
            assert hot_w2 is not None
            assert hot_map is not None
            assert cold_map is not None
            tiered_hip = _tiered_iq_moe_hip()
            variant_name, variant = _tiered_iq_moe_variant()
            prefill_group_size = _tiered_prefill_group_size()
            grouped_prefill = prefill_group_size > 0 and num_tokens > 64
            from . import full_mutable_cache
            if full_mutable_cache.enabled():
                full_mutable_cache.prepare(hot_map, topk_ids, num_tokens)
            elif dynamic_cache:
                assert cache_w1 is not None
                assert cache_w2 is not None
                assert cache_map is not None
                assert cache_tags is not None
                assert cache_clock is not None
                assert cache_admission is not None
                assert cache_stats is not None
                assert cache_pending is not None
                async_cache_active = _tiered_cache_async_enabled()
                cache_policy = _tiered_cache_policy()
                if cache_policy == "lru":
                    if async_cache_active:
                        raise RuntimeError(
                            "Synchronous LRU cache does not support async fills"
                        )
                    fill_batch = _tiered_cache_fill_batch()
                    required_prepare = (
                        "tiered_iq_moe_cache_lru_prepare_many"
                        if fill_batch > 1
                        else "tiered_iq_moe_cache_lru_prepare"
                    )
                elif async_cache_active:
                    required_prepare = "tiered_iq_moe_cache_async_prepare"
                else:
                    required_prepare = "tiered_iq_moe_cache_prepare"
                if not hasattr(tiered_hip, required_prepare):
                    raise RuntimeError(
                        "The loaded tiered HIP extension does not support "
                        "the dynamic expert cache"
                    )
                prepare = getattr(tiered_hip, required_prepare)
                if os.environ.get("R9V_CACHE80", "0") == "1":
                    if (cache_policy != "lru" or async_cache_active
                        or required_prepare != "tiered_iq_moe_cache_lru_prepare"
                        or cache_w1.shape[0] not in ((128,160,192) if os.environ.get("R9V_CACHE192", "0") == "1" else (80,96,128) if os.environ.get("R9V_CACHE128", "0") == "1" else (48,64,80))):
                        raise RuntimeError("Cache prepare requires admitted synchronous single-fill LRU capacity")
                    if os.environ.get("R9V_CACHE_PLANPUBLISH", "0") == "1" and os.environ.get("R9V_CACHE192", "0") != "1":
                        raise RuntimeError("Plan/publication fusion requires admitted synchronous cache192 mode")
                    prepare = _r9v_cache80_module().prepare
                prepare_args = (
                    w1,
                    w2,
                    hot_map,
                    cold_map,
                    topk_ids,
                    cache_w1,
                    cache_w2,
                    cache_map,
                    cache_tags,
                    cache_clock,
                    cache_admission,
                    cache_stats,
                )
                if async_cache_active:
                    prepare(*prepare_args, cache_pending)
                elif cache_policy == "lru":
                    if fill_batch > 1:
                        prepare(*prepare_args, cache_pending, fill_batch)
                    else:
                        prepare(*prepare_args, cache_pending)
                else:
                    prepare(*prepare_args)
            if grouped_prefill:
                wmma_prefill = prefill_group_size == _WMMA_PREFILL_GROUP_SIZE
                prefill_hip = _moe_wmma_hip() if wmma_prefill else tiered_hip
                expected_api = (
                    ("tiered_iq_moe_cached_prefill_wmma" if wmma_prefill
                     else "tiered_iq_moe_cached_prefill_grouped")
                    if dynamic_cache
                    else ("tiered_iq_moe_prefill_wmma" if wmma_prefill
                          else "tiered_iq_moe_prefill_grouped")
                )
                if not hasattr(prefill_hip, expected_api):
                    raise RuntimeError(
                        "The loaded tiered HIP extension does not support "
                        f"grouped-{prefill_group_size} prefill"
                    )
                sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(topk_ids, prefill_group_size, hot_map.numel())
                )
                grouped_api = getattr(prefill_hip, expected_api)

                def tiered_grouped_prefill(
                    inp, cold, hot, cached, route_k, qtype, out_rows, token_count
                ):
                    args = (inp, cold, hot)
                    if dynamic_cache:
                        args += (cached,)
                    return grouped_api(
                        *args,
                        hot_map,
                        cold_map,
                        *((cache_map,) if dynamic_cache else ()),
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        route_k,
                        qtype,
                        out_rows,
                        token_count,
                        prefill_group_size,
                    )

                logger.info_once(
                    "Using tiered IQ MoE grouped-%d prefill",
                    prefill_group_size,
                )
                out = tiered_grouped_prefill(
                    x,
                    w1,
                    hot_w1,
                    cache_w1,
                    top_k,
                    qweight_type,
                    N,
                    num_tokens,
                )
                out = act(out)
                out = tiered_grouped_prefill(
                    out,
                    w2,
                    hot_w2,
                    cache_w2,
                    1,
                    qweight_type2,
                    w2.shape[1],
                    num_tokens * top_k,
                )
            elif variant:
                expected_api = (
                    "tiered_iq_moe_cached_gemv_variant"
                    if dynamic_cache
                    else "tiered_iq_moe_gemv_variant"
                )
                if not hasattr(tiered_hip, expected_api):
                    raise RuntimeError(
                        "The loaded tiered HIP extension does not support "
                        f"QWEN38_TIERED_IQ_MOE_VARIANT={variant_name}"
                    )

                compute_hip = tiered_hip
                use_expert_silu_quant = (
                    os.environ.get("R9V_EXPERT_SILU_QUANT", "0") == "1"
                    and os.environ.get("R9V_EXPERT_SILU_QUANT_SO")
                    and variant == 31 and num_tokens in (4, 5) and top_k == 10
                    and activation_enum == MoEActivation.SILU
                    and x.shape[1] == 2560 and N in (832, 448)
                    and int(qweight_type) in (21, 23)
                    and int(qweight_type2) in (20, 8)
                    and w2.shape[1] == 2560
                )
                if (os.environ.get("R9V_EXPERT_ROUTEPLAN", "0") == "1"
                    and variant == 31 and num_tokens == 4 and top_k == 10
                    and x.shape[1] == 2560 and N in (832, 448)
                    and int(qweight_type) in (21, 23) and int(qweight_type2) in (20, 8)
                    and w2.shape[1] == 2560
                    and not use_expert_silu_quant
                    and (os.environ.get("R9V_ROUTEPLAN_RANK0_ONLY", "0") != "1" or N == 832)):
                    compute_hip = _r9v_routeplan_module()
                    logger.info_once("[r9v] exact M4 routeplan dispatch width=%d", N // 2)

                def tiered_gemv(
                    inp, cold, hot, cached, ids, route_k, qtype, out_rows, token_count
                ):
                    if dynamic_cache:
                        return compute_hip.tiered_iq_moe_cached_gemv_variant(
                            inp,
                            cold,
                            hot,
                            cached,
                            hot_map,
                            cold_map,
                            cache_map,
                            ids,
                            route_k,
                            qtype,
                            out_rows,
                            token_count,
                            variant,
                        )
                    return compute_hip.tiered_iq_moe_gemv_variant(
                        inp,
                        cold,
                        hot,
                        hot_map,
                        cold_map,
                        ids,
                        route_k,
                        qtype,
                        out_rows,
                        token_count,
                        variant,
                    )

                logger.info_once(
                    "Using tiered IQ MoE exact-shape variant %s", variant_name
                )
            else:

                def tiered_gemv(
                    inp, cold, hot, cached, ids, route_k, qtype, out_rows, token_count
                ):
                    if dynamic_cache:
                        return tiered_hip.tiered_iq_moe_cached_gemv(
                            inp,
                            cold,
                            hot,
                            cached,
                            hot_map,
                            cold_map,
                            cache_map,
                            ids,
                            route_k,
                            qtype,
                            out_rows,
                            token_count,
                        )
                    return tiered_hip.tiered_iq_moe_gemv(
                        inp,
                        cold,
                        hot,
                        hot_map,
                        cold_map,
                        ids,
                        route_k,
                        qtype,
                        out_rows,
                        token_count,
                    )

            if not grouped_prefill:
                out = tiered_gemv(
                    x,
                    w1,
                    hot_w1,
                    cache_w1,
                    topk_ids,
                    top_k,
                    qweight_type,
                    N,
                    num_tokens,
                )
                if use_expert_silu_quant:
                    global _R9V_EXPERT_SILU_QUANT_LOGGED
                    if not _R9V_EXPERT_SILU_QUANT_LOGGED:
                        print("[r9v] exact M4/M5 expert SiLU/Q8 fusion enabled", flush=True)
                        _R9V_EXPERT_SILU_QUANT_LOGGED = True
                    ids_flat = topk_ids.reshape(-1).contiguous()
                    fusion = _r9v_expert_silu_quant_module()
                    if dynamic_cache:
                        out = fusion.expert_silu_quant_cached(
                            out, w2, hot_w2, cache_w2, hot_map, cold_map,
                            cache_map, ids_flat, qweight_type2, w2.shape[1],
                            num_tokens * top_k)[0]
                    else:
                        out = fusion.expert_silu_quant(
                            out, w2, hot_w2, hot_map, cold_map, ids_flat,
                            qweight_type2, w2.shape[1], num_tokens * top_k)[0]
                else:
                    out = act(out)
                    out = tiered_gemv(
                        out, w2, hot_w2, cache_w2, topk_ids, 1,
                        qweight_type2, w2.shape[1], num_tokens * top_k)
            if async_cache_active:
                assert cache_map is not None
                assert cache_tags is not None
                assert cache_admission is not None
                assert cache_stats is not None
                assert cache_pending is not None
                tiered_hip.tiered_iq_moe_cache_async_commit(
                    cache_map,
                    cache_tags,
                    cache_admission,
                    cache_stats,
                    cache_pending,
                    cache_layer_id == 47,
                )
        elif use_iq_hip:
            iq_hip = _iq_moe_hip()
            out = iq_hip.iq_moe_gemv(
                x, w1, topk_ids, top_k, qweight_type, N, num_tokens, 1
            )
            out = act(out)
            out = iq_hip.iq_moe_gemv(
                out,
                w2,
                topk_ids,
                1,
                qweight_type2,
                w2.shape[1],
                num_tokens * top_k,
                1,
            )
        elif use_q8_bf16_hip:
            q8_hip = _q8_bf16_moe_hip()
            out = q8_hip.q8_bf16_moe_gemv_variant(
                x, w1, topk_ids, top_k, N, num_tokens, 1
            )
            out = act(out)
            out = q8_hip.q8_bf16_moe_gemv_variant(
                out,
                w2,
                topk_ids,
                1,
                w2.shape[1],
                num_tokens * top_k,
                1,
            )
        else:
            out = ops.ggml_moe_a8_vec(
                x, w1, topk_ids, top_k, qweight_type, N, num_tokens
            )
            out = act(out)
            out = ops.ggml_moe_a8_vec(
                out,
                w2,
                topk_ids,
                1,
                qweight_type2,
                w2.shape[1],
                num_tokens * top_k,
            )
        _finalize_moe_output(out, topk_weights, shared_output, out_hidden_states)
    else:
        if shared_output is not None:
            raise RuntimeError(
                "The fused shared-expert epilogue requires a fast GGUF MoE kernel"
            )
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = fused_mul_mat_gguf_op(inp, w1[ii], qweight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], qweight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    return _fused_moe_gguf_impl(
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        shared_output=shared_output,
    )


def _fused_moe_gguf_tiered(
    x: torch.Tensor,
    cold_w1: torch.Tensor,
    cold_w2: torch.Tensor,
    hot_w1: torch.Tensor,
    hot_w2: torch.Tensor,
    hot_map: torch.Tensor,
    cold_map: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    return _fused_moe_gguf_impl(
        x,
        cold_w1,
        cold_w2,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        hot_w1,
        hot_w2,
        hot_map,
        cold_map,
        shared_output=shared_output,
    )


def _fused_moe_gguf_tiered_cached(
    x: torch.Tensor,
    cold_w1: torch.Tensor,
    cold_w2: torch.Tensor,
    hot_w1: torch.Tensor,
    hot_w2: torch.Tensor,
    hot_map: torch.Tensor,
    cold_map: torch.Tensor,
    cache_w1: torch.Tensor,
    cache_w2: torch.Tensor,
    cache_map: torch.Tensor,
    cache_tags: torch.Tensor,
    cache_clock: torch.Tensor,
    cache_admission: torch.Tensor,
    cache_stats: torch.Tensor,
    cache_pending: torch.Tensor,
    cache_layer_id: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    return _fused_moe_gguf_impl(
        x,
        cold_w1,
        cold_w2,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        hot_w1,
        hot_w2,
        hot_map,
        cold_map,
        cache_w1,
        cache_w2,
        cache_map,
        cache_tags,
        cache_clock,
        cache_admission,
        cache_stats,
        cache_pending,
        cache_layer_id,
        shared_output,
    )


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    del (
        w1,
        w2,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        shared_output,
    )
    return torch.empty_like(x)


def _fused_moe_gguf_tiered_fake(
    x: torch.Tensor,
    cold_w1: torch.Tensor,
    cold_w2: torch.Tensor,
    hot_w1: torch.Tensor,
    hot_w2: torch.Tensor,
    hot_map: torch.Tensor,
    cold_map: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    del (
        cold_w1,
        cold_w2,
        hot_w1,
        hot_w2,
        hot_map,
        cold_map,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        shared_output,
    )
    return torch.empty_like(x)


def _fused_moe_gguf_tiered_cached_fake(
    x: torch.Tensor,
    cold_w1: torch.Tensor,
    cold_w2: torch.Tensor,
    hot_w1: torch.Tensor,
    hot_w2: torch.Tensor,
    hot_map: torch.Tensor,
    cold_map: torch.Tensor,
    cache_w1: torch.Tensor,
    cache_w2: torch.Tensor,
    cache_map: torch.Tensor,
    cache_tags: torch.Tensor,
    cache_clock: torch.Tensor,
    cache_admission: torch.Tensor,
    cache_stats: torch.Tensor,
    cache_pending: torch.Tensor,
    cache_layer_id: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    shared_output: torch.Tensor | None = None,
) -> torch.Tensor:
    del (
        cold_w1,
        cold_w2,
        hot_w1,
        hot_w2,
        hot_map,
        cold_map,
        cache_w1,
        cache_w2,
        cache_map,
        cache_tags,
        cache_clock,
        cache_admission,
        cache_stats,
        cache_pending,
        cache_layer_id,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        shared_output,
    )
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
    direct_register_custom_op(
        op_name="_fused_moe_gguf_tiered",
        op_func=_fused_moe_gguf_tiered,
        fake_impl=_fused_moe_gguf_tiered_fake,
    )
    fused_moe_gguf_tiered = torch.ops.vllm._fused_moe_gguf_tiered
    direct_register_custom_op(
        op_name="_fused_moe_gguf_tiered_cached",
        op_func=_fused_moe_gguf_tiered_cached,
        mutates_args=[
            "cache_w1",
            "cache_w2",
            "cache_map",
            "cache_tags",
            "cache_clock",
            "cache_admission",
            "cache_stats",
            "cache_pending",
        ],
        fake_impl=_fused_moe_gguf_tiered_cached_fake,
    )
    fused_moe_gguf_tiered_cached = torch.ops.vllm._fused_moe_gguf_tiered_cached
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        self._fuses_shared_expert_output = _fused_shared_epilogue_enabled()

    @property
    def fuses_shared_expert_output(self) -> bool:
        return self._fuses_shared_expert_output

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        partition = configure_expert_partition(layer, hidden_size, num_experts)
        if partition is not None:
            intermediate_size_per_partition = partition[1]
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        shared_output = None
        if self.fuses_shared_expert_output and shared_experts is not None:
            if shared_experts_input is None:
                raise RuntimeError(
                    "Fused MoE shared epilogue requires shared_experts_input"
                )
            shared_output = shared_experts.pending_output
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op
        from .route_profile import record_route_profile

        record_route_profile(layer.layer_name, topk_ids)

        if os.environ.get("VLLM_QWEN4_EXP_MOE_DEBUG_SYNC") == "1":
            for name, weight in (
                ("w13", layer.w13_qweight),
                ("w2", layer.w2_qweight),
            ):
                cpu_owner = getattr(weight, "_vllm_uva_cpu_data", None)
                owner_ptr = cpu_owner.data_ptr() if cpu_owner is not None else None
                owner_bytes = (
                    cpu_owner.numel() * cpu_owner.element_size()
                    if cpu_owner is not None
                    else None
                )
                owner_pinned = cpu_owner.is_pinned() if cpu_owner is not None else None
                print(
                    f"GGUF MoE debug: device={torch.cuda.current_device()} "
                    f"layer={layer.layer_name} weight={name} "
                    f"shape={tuple(weight.shape)} stride={weight.stride()} "
                    f"device_ptr={weight.data_ptr():#x} "
                    f"owner_ptr={owner_ptr if owner_ptr is None else hex(owner_ptr)} "
                    f"owner_bytes={owner_bytes} "
                    f"owner_pinned={owner_pinned}",
                    flush=True,
                )
                # Touch both ends of every expert allocation before the custom
                # kernel so a bad UVA mapping is distinguished from kernel
                # indexing.
                probe = weight[:, (0, -1), (0, -1)].sum()
                del probe
                torch.cuda.synchronize()
            print(
                f"GGUF MoE debug: layer={layer.layer_name} "
                f"qtypes=({layer.w13_qweight_type.weight_type},"
                f"{layer.w2_qweight_type.weight_type}) "
                f"topk_range=({int(topk_ids.min())},{int(topk_ids.max())}) "
                f"topk_shape={tuple(topk_ids.shape)} x_shape={tuple(x.shape)}",
                flush=True,
            )

        hot_w13 = getattr(layer, "_gguf_hot_w13", None)
        if hot_w13 is not None:
            cache_w13 = getattr(layer, "_gguf_cache_w13", None)
            if cache_w13 is not None:
                return fused_moe_gguf_tiered_cached(
                    x,
                    layer.w13_qweight,
                    layer.w2_qweight,
                    hot_w13,
                    layer._gguf_hot_w2,
                    layer._gguf_global_to_hot,
                    layer._gguf_global_to_cold,
                    cache_w13,
                    layer._gguf_cache_w2,
                    layer._gguf_global_to_cache,
                    layer._gguf_cache_tags,
                    layer._gguf_cache_clock,
                    layer._gguf_cache_admission,
                    layer._gguf_cache_stats,
                    layer._gguf_cache_pending,
                    layer._gguf_cache_layer_id,
                    topk_weights,
                    topk_ids,
                    layer.w13_qweight_type.weight_type,
                    layer.w2_qweight_type.weight_type,
                    layer.activation.value,
                    shared_output,
                )
            return fused_moe_gguf_tiered(
                x,
                layer.w13_qweight,
                layer.w2_qweight,
                hot_w13,
                layer._gguf_hot_w2,
                layer._gguf_global_to_hot,
                layer._gguf_global_to_cold,
                topk_weights,
                topk_ids,
                layer.w13_qweight_type.weight_type,
                layer.w2_qweight_type.weight_type,
                layer.activation.value,
                shared_output,
            )

        return fused_moe_gguf_op(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            layer.activation.value,
            shared_output,
        )
