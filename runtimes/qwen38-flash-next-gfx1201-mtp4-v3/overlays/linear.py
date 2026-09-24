# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
from pathlib import Path

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .layout import GGUFLinearLayout
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


_Q8_PREFILL_WMMA = {}
_Q8_PREFILL_WMMA_ENABLED = os.environ.get("R9V_Q8_PREFILL_COMBINED", "0") == "1"
_Q8_PREFILL_LOGGED = set()
_Q8_TOKEN64_ENABLED = os.environ.get("R9V_Q8_PREFILL_TOKEN64", "0") == "1"
_Q8_TOKEN64_LOGGED = set()
def _q8_prefill_wmma_supported(x, qweight, qweight_type):
    return (_Q8_PREFILL_WMMA_ENABLED and int(qweight_type) == 8
            and x.dtype == torch.bfloat16 and qweight.dtype == torch.uint8
            and x.ndim == 2 and qweight.ndim == 2
            and 1024 <= x.shape[0] <= 65535
            and (qweight.shape[0], x.shape[1]) in {(10240, 320), (2560, 3072), (8192, 2560), (6656, 2560)}
            and qweight.shape[1] == x.shape[1] // 32 * 34
            and x.is_cuda and qweight.device == x.device
            and x.is_contiguous() and qweight.is_contiguous())
def _q8_prefill_wmma(k):
    kind = "small" if k == 320 else "streamed"
    if kind not in _Q8_PREFILL_WMMA:
        import hashlib
        if kind == "small":
            path = Path("/r9v-full-mutable/q8_wmma.so")
            expected = "0b1243a30fd2770c6e5d66800e9b6674f7227f2e88accb27bbb1eb802251b5e9"
            module_name = "prefill_q8_small_k_extension"
        elif _Q8_TOKEN64_ENABLED:
            path = Path("/r9v-full-mutable/q8_token64.so")
            expected = "f87a06a42744f05833c84f91072427a3f04155c23fe7819150917e89c6c6a092"
            module_name = "prefill_q8_streamed_token64_extension"
        else:
            path = Path("/r9v-full-mutable/q8_streamed.so")
            expected = "2cf029ba7ae1f067b1eb99c4738eab676c28a2286ad5b31d2881b81af3ae3714"
            module_name = "prefill_q8_wmma_streamed_extension"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Q8 prefill extension hash mismatch")
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _Q8_PREFILL_WMMA[kind] = module
    return _Q8_PREFILL_WMMA[kind]

_DENSE_MMVQ_HIP = None
_LOGGED_REUSE_SHAPES: set[tuple[str, int, int]] = set()
_GFX1201_DEVICE_CACHE: dict[int, bool] = {}
_PROFILE_DENSE_SHAPES = os.environ.get("QWEN38_PROFILE_DENSE_SHAPES") == "1"

# Runtime (N, K) after TP sharding.  This first arm deliberately covers only
# the five largest dense tensor families in the Qwen3.8 Q4 target: Q/K/V or
# QKV, attention/GDN output, attention gate, full-attention Q, and the output
# head.  TP1 forms are retained for the offline real-tensor harness; the live
# TP2 graph uses the corresponding half-sharded form.
_QWEN38_Q4_REUSE_SHAPES: dict[int, frozenset[tuple[int, int]]] = {
    int(WeightType.Q8_0): frozenset(
        {
            (10240, 2560),
            (5120, 2560),
            (2560, 6144),
            (2560, 3072),
            (6144, 2560),
            (3072, 2560),
            (12288, 2560),
        }
    ),
    int(WeightType.Q4_K): frozenset({(248320, 2560), (124160, 2560)}),
    int(WeightType.Q6_K): frozenset({(248320, 2560), (124160, 2560)}),
}

_QWEN38_Q8_ATTENTION_M3_SHAPES = frozenset({(8192, 2560), (6656, 2560)})
_QWEN38_Q8_ATTENTION_M3_VARIANTS = {
    "reuse3": 0,
    "exact4": 1,
    "exact4-w8": 2,
    "group4": 3,
    "group4-w8": 4,
    "group4-w10": 5,
}


def _dense_mmvq_hip():
    """Load the opt-in RDNA4 dense MMVQ extension."""
    global _DENSE_MMVQ_HIP
    if _DENSE_MMVQ_HIP is not None:
        return _DENSE_MMVQ_HIP
    path = Path(os.environ["QWEN38_DENSE_MMVQ_HIP_SO"])
    spec = importlib.util.spec_from_file_location("qwen38_dense_mmvq_hip", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load dense MMVQ extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _DENSE_MMVQ_HIP = module
    return module


def _is_gfx1201(device: torch.device) -> bool:
    """Return whether ``device`` is the RDNA4 target this code was built for."""
    if torch.version.hip is None or device.type != "cuda":
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cached = _GFX1201_DEVICE_CACHE.get(device_index)
    if cached is not None:
        return cached
    properties = torch.cuda.get_device_properties(device_index)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    result = arch == "gfx1201"
    _GFX1201_DEVICE_CACHE[device_index] = result
    return result


def _use_qwen38_q4_reuse3(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> bool:
    """Strict opt-in gate for the MTP2 target's exact three-row geometries."""
    qtype = int(qweight_type)
    return (
        os.environ.get("QWEN38_USE_DENSE_MMVQ_REUSE3") == "1"
        and x.ndim == 2
        and x.shape[0] == 3
        and qweight.ndim == 2
        and (qweight.shape[0], x.shape[1])
        in _QWEN38_Q4_REUSE_SHAPES.get(qtype, ())
        and x.dtype == torch.bfloat16
        and qweight.dtype == torch.uint8
        and x.device == qweight.device
        and x.is_contiguous()
        and qweight.is_contiguous()
        and _is_gfx1201(x.device)
    )


def _qwen38_q8_attention_m3_variant() -> int:
    name = os.environ.get(
        "QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT", "exact4-w8"
    )
    try:
        return _QWEN38_Q8_ATTENTION_M3_VARIANTS[name]
    except KeyError as error:
        choices = ", ".join(_QWEN38_Q8_ATTENTION_M3_VARIANTS)
        raise ValueError(
            "QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT must be one of "
            f"{choices}; got {name!r}"
        ) from error


def _use_qwen38_q8_attention_m3(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> bool:
    """Gate the exact Q8 attention-input M=3 geometry behind an opt-in."""
    return (
        os.environ.get("QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3") == "1"
        and int(qweight_type) == int(WeightType.Q8_0)
        and x.ndim == 2
        and x.shape[0] == 3
        and qweight.ndim == 2
        and (qweight.shape[0], x.shape[1])
        in _QWEN38_Q8_ATTENTION_M3_SHAPES
        and x.dtype == torch.bfloat16
        and qweight.dtype == torch.uint8
        and x.device == qweight.device
        and x.is_contiguous()
        and qweight.is_contiguous()
        and _is_gfx1201(x.device)
    )


def _use_qwen38_q4_reuse4(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> bool:
    """Strict opt-in gate for the MTP3 target's exact four-row geometries."""
    qtype = int(qweight_type)
    return (
        os.environ.get("QWEN38_USE_DENSE_MMVQ_REUSE4") == "1"
        and x.ndim == 2
        and x.shape[0] == 4
        and qweight.ndim == 2
        and (qweight.shape[0], x.shape[1])
        in _QWEN38_Q4_REUSE_SHAPES.get(qtype, ())
        and x.dtype == torch.bfloat16
        and qweight.dtype == torch.uint8
        and x.device == qweight.device
        and x.is_contiguous()
        and qweight.is_contiguous()
        and _is_gfx1201(x.device)
    )


def _use_qwen38_fused_hc_up_mix(
    raw_lora: torch.Tensor,
    xn: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hc_count: int,
) -> bool:
    """Gate the exact Qwen3.8 HC-up geometry behind an explicit opt-in."""
    return (
        os.environ.get("QWEN38_FUSED_HC_UP_MIX") == "1"
        and int(qweight_type) == int(WeightType.Q8_0)
        and hc_count == 4
        and raw_lora.ndim == 2
        and 1 <= raw_lora.shape[0] <= 4
        and raw_lora.shape[1] == 320
        and xn.shape == (raw_lora.shape[0], 10240)
        and qweight.ndim == 2
        and qweight.shape[0] == 10240
        and raw_lora.dtype == torch.bfloat16
        and xn.dtype == torch.bfloat16
        and qweight.dtype == torch.uint8
        and raw_lora.device == xn.device == qweight.device
        and raw_lora.stride(1) == 1
        and xn.stride(1) == 1
        and qweight.is_contiguous()
        and _is_gfx1201(raw_lora.device)
    )


_R9V_NEXT_MODULES = {}
def _r9v_next_module(kind):
    if kind not in _R9V_NEXT_MODULES:
        name, key = {"pairs": ("r9v_q8_loadpairs", "R9V_Q8_LOADPAIRS_SO"),
                     "hc45": ("r9v_hc_mixmmq45", "R9V_HC_MIX_MMQ45_SO"),
                     "hctiles": ("r9v_hc_tiles", "R9V_HC_TILES_SO"),
                     "mmq5": ("r9v_q8_mmq5", "R9V_Q8_MMQ5_SO"),
                     "mmvq5": ("r9v_q8_mmvq5", "R9V_Q8_MMVQ5_SO")}[kind]
        spec = importlib.util.spec_from_file_location(name, os.environ[key])
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        _R9V_NEXT_MODULES[kind] = module
    return _R9V_NEXT_MODULES[kind]


def _r9v_next_config(key):
    import json
    return json.loads(os.environ.get(key, "{}"))


_R9V_HC_FUSION_MODULES = {}

def _r9v_hc_fusion_module(kind):
    if kind not in _R9V_HC_FUSION_MODULES:
        name = "r9v_hc_guard" if kind == "guard" else "r9v_hc_mixmmq4"
        key = "R9V_HC_GUARD_SO" if kind == "guard" else "R9V_HC_MIX_MMQ4_SO"
        spec = importlib.util.spec_from_file_location(name, os.environ[key])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R9V_HC_FUSION_MODULES[kind] = module
    return _R9V_HC_FUSION_MODULES[kind]


def _fused_qwen38_hc_up_mix(
    raw_lora: torch.Tensor,
    xn: torch.Tensor,
    qweight: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    # hc_count is the HC STREAM count (always4), not the number of tokens.
    tile = os.environ.get("R9V_HC_TILE", "")
    if (tile in {"4", "8", "16"} and hc_count == 4
        and raw_lora.ndim == 2 and raw_lora.shape[0] in (4,5)
        and raw_lora.shape[1] == 320 and tuple(xn.shape) == (raw_lora.shape[0],10240)
        and tuple(qweight.shape) == (10240,340)
        and ((raw_lora.shape[0] == 4 and os.environ.get("R9V_HC_MIX_MMQ4") == "1")
             or (raw_lora.shape[0] == 5 and os.environ.get("R9V_HC_MIX_MMQ45") == "1"))
        and raw_lora.stride(1) == 1 and raw_lora.stride(0) in (320,336)
        and xn.is_contiguous() and qweight.is_contiguous()
        and raw_lora.dtype == torch.bfloat16 and xn.dtype == torch.bfloat16
        and qweight.dtype == torch.uint8 and raw_lora.device == xn.device == qweight.device
        and _is_gfx1201(raw_lora.device)):
        key=("hc-tile",raw_lora.shape[0],int(tile))
        if key not in _LOGGED_REUSE_SHAPES:
            print(f"[r9v] exact HC tile dispatch {key}",flush=True)
            _LOGGED_REUSE_SHAPES.add(key)
        return _r9v_next_module("hctiles").hc_tiles(qweight,raw_lora,xn,int(tile))
    if (os.environ.get("R9V_HC_MIX_MMQ45", "0") == "1"
        and tuple(raw_lora.shape) == (5,320) and tuple(xn.shape) == (5,10240)
        and tuple(qweight.shape) == (10240,340) and hc_count == 4
        and raw_lora.stride(1) == 1 and raw_lora.stride(0) in (320,336)
        and xn.is_contiguous() and qweight.is_contiguous()
        and raw_lora.dtype == torch.bfloat16 and xn.dtype == torch.bfloat16
        and qweight.dtype == torch.uint8 and raw_lora.device == xn.device == qweight.device
        and _is_gfx1201(raw_lora.device)):
        key = ("hc-mix-mmq5", 10240, 320)
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print("[r9v] exact fused HC MMQ5 mix dispatch", flush=True)
        return _r9v_next_module("hc45").hc_mixmmq45(qweight, raw_lora, xn)
    if os.environ.get("R9V_HC_MIX_MMQ4", "0") == "1":
        if (tuple(raw_lora.shape) == (4,320) and tuple(xn.shape) == (4,10240)
            and tuple(qweight.shape) == (10240,340) and hc_count == 4
            and raw_lora.stride(1) == 1 and xn.stride(1) == 1
            and raw_lora.dtype == torch.bfloat16 and xn.dtype == torch.bfloat16
            and qweight.dtype == torch.uint8 and qweight.is_contiguous()
            and raw_lora.device == xn.device == qweight.device
            and _is_gfx1201(raw_lora.device)):
            key = ("hc-mix-mmq4", 10240, 320)
            if key not in _LOGGED_REUSE_SHAPES:
                _LOGGED_REUSE_SHAPES.add(key)
                print("[r9v] exact fused HC MMQ4 mix dispatch", flush=True)
            return _r9v_hc_fusion_module("mix").hc_mmq4(qweight, raw_lora, xn)
        from vllm.models.qwen4_exp.amd.ops.hc import hc_silu, hc_gate_mix
        gate = _fused_mul_mat_gguf_impl(hc_silu(raw_lora, hc_count), qweight, 8)
        return hc_gate_mix(xn, gate, hc_count)
    return _dense_mmvq_hip().dense_gemv_q8_hc_mix(qweight, raw_lora, xn, hc_count)


def _fused_qwen38_hc_up_mix_fake(
    raw_lora: torch.Tensor,
    xn: torch.Tensor,
    qweight: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del qweight
    return raw_lora.new_empty((raw_lora.shape[0], xn.shape[1] // hc_count))


_R9V_HC_MMQ4_MODULE = None

def _r9v_hc_mmq4_module():
    global _R9V_HC_MMQ4_MODULE
    if _R9V_HC_MMQ4_MODULE is None:
        import importlib.util
        path = os.environ["R9V_HC_MMQ4_SO"]
        spec = importlib.util.spec_from_file_location("r9v_hc_mmq4", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R9V_HC_MMQ4_MODULE = module
    return _R9V_HC_MMQ4_MODULE


_R9V_HC_TUNE_MODULE = None

def _r9v_hc_tune_module():
    global _R9V_HC_TUNE_MODULE
    if _R9V_HC_TUNE_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("r9v_hc_mmq4_tune", os.environ["R9V_HC_TUNE_SO"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R9V_HC_TUNE_MODULE = module
    return _R9V_HC_TUNE_MODULE


def _r9v_pad_hc_q8_tail(layer):
    """Give the legacy four-block MMQ loader a zero tail for K=320 (10 blocks)."""
    w = layer.qweight
    eligible = ((os.environ.get("R9V_HC_SAFE_PADDING", "0") == "1" and tuple(w.shape) == (10240,340))
        or (os.environ.get("R9V_SHARED_SAFE_PADDING", "0") == "1" and tuple(w.shape) == (2560,340)))
    if (not eligible or int(layer.qweight_type.weight_type) != 8
        or w.dtype != torch.uint8 or not w.is_contiguous()
        or not w.is_cuda or not _is_gfx1201(w.device)):
        return False
    if getattr(w, "_r9v_hc_tail_padded", False):
        return True
    # The logical matrix remains identical. MMQ loads four Q8_0 blocks
    # per K tile; the final tile at K=320 has only two real blocks.
    backing = torch.empty(w.numel() + 2 * 34, dtype=w.dtype, device=w.device)
    backing[:w.numel()].copy_(w.reshape(-1))
    backing[w.numel():].zero_()
    w.data = backing[:w.numel()].view_as(w)
    w._r9v_hc_tail_padded = True
    key = ("q8-safe-tail", w.shape[0], 320)
    if key not in _LOGGED_REUSE_SHAPES:
        _LOGGED_REUSE_SHAPES.add(key)
        print(f"[r9v] Q8 zero tail enabled N={w.shape[0]} K=320: 68 extra bytes per weight", flush=True)
    return True


def _fused_mul_mat_gguf_impl(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        # A two-token speculative draft is verified as three rows (the target
        # token plus two candidates).  Keeping the old wide-matrix cutoff at
        # two sends Q4_K vocabulary heads through GGML's generic MMQ kernel,
        # which is dramatically slower than MMVQ for this tiny row count on
        # ROCm.  The Qwen3.8 output tensor was checked directly at M=3 for
        # finiteness and numerical agreement before raising this boundary.
        mmvq_safe = 3 if qweight.shape[0] > 5120 else 6
    # Exact existing Q6_K MMVQ kernel: avoid the wide-head MMQ crossover
    # for MTP3/4. The bounded real-weight row fixture must pass before enable.
    if (
        os.environ.get("R9V_Q6K_HEAD_ROWS5", "0") == "1"
        and int(qweight_type) == 14
        and qweight.shape == (124160, 2100)
        and x.shape[1] == 2560
    ):
        mmvq_safe = 5
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    use_q4_reuse2 = (
        os.environ.get("QWEN38_USE_DENSE_MMVQ_REUSE2") == "1"
        and x.shape[0] == 2
        and int(qweight_type) == 12
        and qweight.shape[0] > 100_000
    )
    use_dense_hip = (
        os.environ.get("QWEN38_USE_DENSE_MMVQ_HIP") == "1"
        # This kernel is tuned and parity-tested for the two-row speculative
        # verifier.  Routing M=1 through it regresses the complete decode graph
        # badly even though the representative M=2 projections are faster.
        and x.shape[0] == 2
        and int(qweight_type) in (12, 13, 14)
    )
    use_q8_reuse2 = (
        os.environ.get("QWEN38_USE_DENSE_MMVQ_Q8_REUSE2") == "1"
        and x.shape[0] == 2
        and int(qweight_type) == 8
        # Keep the full-model trial to the recurrent hyperconnection and
        # shared-expert projections.  Every Q8 shape wins in isolation, but
        # capturing the PLE 10240x2560/2560x2560 pair in the complete TP graph
        # caused request-to-request throughput to collapse without a kernel
        # error.  This narrower set lets us establish whether the problem is
        # graph breadth/lifetime rather than the arithmetic itself.
        and (qweight.shape[0], x.shape[1])
        in {
            (10240, 320),
            (2560, 320),
            (320, 10240),
        }
    )
    use_q8_attention_m3 = _use_qwen38_q8_attention_m3(
        x, qweight, qweight_type
    )
    use_qwen38_reuse4 = _use_qwen38_q4_reuse4(x, qweight, qweight_type)
    use_qwen38_reuse3 = _use_qwen38_q4_reuse3(x, qweight, qweight_type)
    use_rows45 = (
        os.environ.get("R9V_DENSE_ROWS45", "0") == "1"
        and x.shape[0] in (4, 5)
        and x.dtype == torch.bfloat16
        and x.is_contiguous() and qweight.is_contiguous()
        and (
            (int(qweight_type) == 8 and (qweight.shape[0], x.shape[1])
             in {(8192, 2560), (6656, 2560)})
            or (int(qweight_type) == 14 and (qweight.shape[0], x.shape[1])
                == (124160, 2560))
        )
    )
    use_rows2 = (
        os.environ.get("R9V_DENSE_ROWS2", "0") == "1"
        and x.shape[0] == 2 and x.dtype == torch.bfloat16
        and x.is_contiguous() and qweight.is_contiguous()
        and (
            (int(qweight_type) == 8 and (qweight.shape[0], x.shape[1])
             in {(8192, 2560), (6656, 2560)})
            or (int(qweight_type) == 14 and (qweight.shape[0], x.shape[1])
                == (124160, 2560))
        )
    )
    use_hc_mmq4 = (
        os.environ.get("R9V_HC_MMQ4", "0") == "1"
        and int(qweight_type) == 8
        and tuple(x.shape) == (4, 320)
        and tuple(qweight.shape) == (10240, 340)
        and x.dtype == torch.bfloat16 and qweight.dtype == torch.uint8
        and x.device == qweight.device
        and x.is_contiguous() and qweight.is_contiguous()
        and _is_gfx1201(x.device)
    )
    use_attention_output4 = (
        os.environ.get("R9V_ATTENTION_OUTPUT4", "0") == "1"
        and int(qweight_type) == 8 and tuple(x.shape) == (4, 3072)
        and tuple(qweight.shape) == (2560, 3264)
        and x.dtype == torch.bfloat16 and qweight.dtype == torch.uint8
        and x.device == qweight.device and x.is_contiguous() and qweight.is_contiguous()
        and _is_gfx1201(x.device)
    )
    use_shared_output4 = (
        os.environ.get("R9V_SHARED_OUTPUT4", "0") == "1"
        and int(qweight_type) == 8 and x.shape[0] == 4
        and (qweight.shape[0], x.shape[1]) in {(640, 2560), (2560, 320)}
        and qweight.shape[1] == x.shape[1] // 32 * 34
        and x.dtype == torch.bfloat16 and qweight.dtype == torch.uint8
        and x.device == qweight.device and x.is_contiguous() and qweight.is_contiguous()
        and _is_gfx1201(x.device)
    )
    next_shape = f"{qweight.shape[0]},{x.shape[1]}" if qweight.ndim == 2 and x.ndim == 2 else ""
    use_q8_pairs = (next_shape in _r9v_next_config("R9V_Q8_PAIR_SHAPES")
        and int(qweight_type) == 8 and x.shape[0] == 4
        and (use_rows45 or use_attention_output4 or use_shared_output4)
        and x.dtype == torch.bfloat16 and qweight.dtype == torch.uint8
        and x.is_contiguous() and qweight.is_contiguous() and x.device == qweight.device
        and _is_gfx1201(x.device))
    mmq5_tile = _r9v_next_config("R9V_Q8_MMQ5_TILES").get(next_shape, 0)
    use_q8_mmq5 = (mmq5_tile in (8,16,64) and int(qweight_type) == 8 and x.shape[0] == 5
        and x.dtype == torch.bfloat16 and qweight.dtype == torch.uint8
        and x.is_contiguous() and qweight.is_contiguous() and x.device == qweight.device
        and _is_gfx1201(x.device))
    use_q8_mmvq5 = (next_shape in _r9v_next_config("R9V_Q8_MMVQ5_SHAPES")
        and int(qweight_type) == 8 and x.shape[0] == 5
        and x.dtype == torch.bfloat16 and qweight.dtype == torch.uint8
        and x.is_contiguous() and qweight.is_contiguous() and x.device == qweight.device
        and _is_gfx1201(x.device))
    if use_q8_mmvq5:
        key = ("mmvq5",next_shape,5)
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(f"[r9v] exact Q8 next dispatch {key}", flush=True)
        y = _r9v_next_module("mmvq5").q8_mmvq5(qweight,x)
    elif use_q8_pairs or use_q8_mmq5:
        kind = "pairs" if use_q8_pairs else "mmq5"
        key = (kind, next_shape, x.shape[0])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(f"[r9v] exact Q8 next dispatch {key}", flush=True)
        y = (_r9v_next_module("pairs").q8_mmvq4(qweight,x) if use_q8_pairs
             else _r9v_next_module("mmq5").q8_mmq5(qweight,x,mmq5_tile))
    elif use_hc_mmq4:
        key = ("q8-hc-mmq4", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(f"[r9v] exact HC MMQ4 dispatch {key}", flush=True)
        groups = int(os.environ.get("R9V_HC_GROUPS", "0"))
        if os.environ.get("R9V_HC_FUSED_GUARD", "0") == "1" and not groups:
            y = _r9v_hc_fusion_module("guard").hc_mmq4(qweight, x)
        else:
            y = (_r9v_hc_tune_module().hc_mmq4(qweight, x, groups)
                 if groups else _r9v_hc_mmq4_module().hc_mmq4(qweight, x))
    elif use_attention_output4 or use_shared_output4:
        family = "attention-output4" if use_attention_output4 else "shared-output4"
        key = (family, qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(f"[r9v] exact Q8 projection dispatch {key}", flush=True)
        y = (_dense_mmvq_hip().dense_gemv_reuse4(qweight, x, 8, qweight.shape[0])
             if use_attention_output4 else _r9v_hc_tune_module().shared_mmvq4(qweight, x))
    elif use_rows2:
        key = (f"q{int(qweight_type)}-rows2", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(f"[r9v] dense rows2 dispatch {key}", flush=True)
        y = _dense_mmvq_hip().dense_gemv_rows2(
            qweight, x, int(qweight_type), qweight.shape[0]
        )
    elif use_rows45:
        key = (f"q{int(qweight_type)}-rows45-m{x.shape[0]}", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(f"[r9v] dense rows45 dispatch {key}", flush=True)
        y = _dense_mmvq_hip().dense_gemv_rows45(
            qweight, x, int(qweight_type), qweight.shape[0]
        )
    elif use_q8_attention_m3:
        variant = _qwen38_q8_attention_m3_variant()
        variant_name = next(
            name
            for name, value in _QWEN38_Q8_ATTENTION_M3_VARIANTS.items()
            if value == variant
        )
        key = (f"q8-attention-m3-{variant_name}", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(
                f"[qwen38] Q8 attention M=3 dispatch "
                f"shape={key[1]}x{key[2]} variant={variant_name}"
            )
        y = _dense_mmvq_hip().dense_gemv_q8_attention_m3(
            qweight, x, qweight.shape[0], variant
        )
    elif use_qwen38_reuse4:
        qtype_name = WeightType(int(qweight_type)).name.lower()
        key = (f"{qtype_name}-reuse4", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(
                f"[qwen38] {qtype_name} reuse4 dispatch "
                f"shape={key[1]}x{key[2]}"
            )
        y = _dense_mmvq_hip().dense_gemv_reuse4(
            qweight, x, int(qweight_type), qweight.shape[0]
        )
    elif use_qwen38_reuse3:
        qtype_name = WeightType(int(qweight_type)).name.lower()
        key = (f"{qtype_name}-reuse3", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(
                f"[qwen38] {qtype_name} reuse3 dispatch "
                f"shape={key[1]}x{key[2]}"
            )
        y = _dense_mmvq_hip().dense_gemv_reuse3(
            qweight, x, int(qweight_type), qweight.shape[0]
        )
    elif use_q4_reuse2:
        key = ("q4", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(f"[qwen38] Q4 reuse2 dispatch shape={key[1]}x{key[2]}")
        y = _dense_mmvq_hip().dense_gemv_q4_reuse2(
            qweight, x, qweight.shape[0], 1
        )
    elif use_q8_reuse2:
        # HC-down has few output rows and benefits from four cooperating
        # waves.  Although two waves win for the other shapes in isolation,
        # the captured TP graph is only stable and fast with one.
        waves = 4 if qweight.shape[0] <= 512 else 1
        key = ("q8", qweight.shape[0], x.shape[1])
        if key not in _LOGGED_REUSE_SHAPES:
            _LOGGED_REUSE_SHAPES.add(key)
            print(
                f"[qwen38] Q8 reuse2 dispatch shape={key[1]}x{key[2]} "
                f"waves={waves}"
            )
        y = _dense_mmvq_hip().dense_gemv_q8_reuse2(
            qweight, x, qweight.shape[0], waves
        )
    elif use_dense_hip:
        y = _dense_mmvq_hip().dense_gemv(
            qweight, x, int(qweight_type), qweight.shape[0], 1
        )
    elif _needs_q8_vision_dequant_fallback(x, qweight, qweight_type):
        weight = ops.ggml_dequantize(
            qweight,
            qweight_type,
            qweight.shape[0],
            x.shape[1],
            x.dtype,
        )
        y = x @ weight.T
    elif x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in MMQ_QUANT_TYPES:
        if _q8_prefill_wmma_supported(x, qweight, qweight_type):
            key = (x.shape[0], qweight.shape[0], x.shape[1])
            if key not in _Q8_PREFILL_LOGGED:
                _Q8_PREFILL_LOGGED.add(key)
                print(f"[qwen38] prefill Q8 combined WMMA enabled M={x.shape[0]} N={key[1]} K={key[2]}", flush=True)
            _token64 = _Q8_TOKEN64_ENABLED and x.shape[1] in (2560, 3072)
            y = _q8_prefill_wmma(x.shape[1]).q8_matmul(qweight, x, 2 if _token64 else 1)
            if _token64:
                _token64_key = (x.shape[0], qweight.shape[0], x.shape[1])
                if _token64_key not in _Q8_TOKEN64_LOGGED:
                    _Q8_TOKEN64_LOGGED.add(_token64_key)
                    print(f"[r9v] Q8 token64 dispatched M={x.shape[0]} N={qweight.shape[0]} K={x.shape[1]}", flush=True)
        else:
            y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")

    # Keep Qwen4Exp's three Q8_0 projection families finite without a
    # device-to-host synchronization.  The native ROCm kernel is fast, but at
    # least one of these shapes can emit a bad lane on gfx1201; allowing that
    # value into the recurrent/residual stream corrupts subsequent tokens.
    guard_already_fused = (use_hc_mmq4 and os.environ.get("R9V_HC_FUSED_GUARD", "0") == "1"
        and int(os.environ.get("R9V_HC_GROUPS", "0")) == 0)
    if _needs_q8_finite_guard(x, qweight, qweight_type) and not guard_already_fused:
        y.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if not _PROFILE_DENSE_SHAPES:
        return _fused_mul_mat_gguf_impl(x, qweight, qweight_type)
    label = (
        f"qwen38_dense_q{int(qweight_type)}_m{x.shape[0]}_"
        f"n{qweight.shape[0]}_k{x.shape[1]}"
    )
    with torch.profiler.record_function(label):
        return _fused_mul_mat_gguf_impl(x, qweight, qweight_type)


def _needs_q8_vision_dequant_fallback(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> bool:
    """Avoid the native ROCm Q8 MMQ path for Qwen vision projections."""
    projection_shape = (x.shape[1], qweight.shape[0])
    return (
        torch.version.hip is not None
        and int(qweight_type) == int(WeightType.Q8_0)
        and x.ndim == 2
        and x.shape[0] > 0
        and qweight.ndim == 2
        and projection_shape
        in {
            # TP1 / data-parallel vision tower.
            (1152, 1152),
            (1152, 3456),
            (1152, 4304),
            (4304, 1152),
            (4608, 2560),
            (4608, 4608),
            # TP2 weight-parallel vision tower.
            (576, 1152),
            (1152, 1728),
            (1152, 2152),
            (2152, 1152),
            (2304, 2560),
            (4608, 2304),
        }
    )


def _needs_q8_finite_guard(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> bool:
    """Limit the asynchronous ROCm Q8_0 guard to Qwen4Exp projections."""
    projection_shape = (x.shape[1], qweight.shape[0])
    return (
        torch.version.hip is not None
        and int(qweight_type) == int(WeightType.Q8_0)
        and x.ndim == 2
        and x.shape[0] > 0
        and qweight.ndim == 2
        and projection_shape
        in {
            (320, 10240),
            (2560, 640),
            (320, 2560),
        }
    )


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
    direct_register_custom_op(
        op_name="_fused_qwen38_hc_up_mix",
        op_func=_fused_qwen38_hc_up_mix,
        fake_impl=_fused_qwen38_hc_up_mix_fake,
    )
    fused_qwen38_hc_up_mix = torch.ops.vllm._fused_qwen38_hc_up_mix
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    def __init__(
        self,
        quant_config,
        layout: GGUFLinearLayout | None = None,
    ) -> None:
        self.quant_config = quant_config
        self.layout = layout

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

        if self.layout is not None:
            set_weight_attrs(
                qweight,
                {
                    "gguf_layout": self.layout,
                    "gguf_logical_input_size": input_size,
                    "gguf_weight_type_parameter": qweight_type,
                },
            )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)
        _r9v_pad_hc_q8_tail(layer)

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_qweight(layer)
        self._materialize_qweight_type(layer)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "qweight")

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "qweight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=qweight.weight_loader,
                input_dim=qweight.input_dim,
                output_dim=qweight.output_dim,
                tensor_shape=qweight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(qweight.shard_id_map)
            if hasattr(qweight, "ignore_warning"):
                padded_param.ignore_warning = qweight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            qweight.data_container.clear()
            qweight.shard_id.clear()
            qweight.shard_id_map.clear()
            if qweight.data.numel() > 0:
                qweight.data = torch.empty(
                    0, dtype=qweight.dtype, device=qweight.device
                )
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        if self.layout is not None:
            x = self.layout.input_to_gguf(x)
        output_leading_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        shard_id = layer.qweight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            fallback_wtype = layer.qweight_type.weight_type
            shard_weight_types = [
                layer.qweight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf_op(x, qweight, shard_weight_types[0])
            else:
                result = []
                for idx in shard_id:
                    start, end, offset = layer.qweight.shard_offset_map[idx]
                    qweight_type = layer.qweight_type.shard_weight_type.get(
                        idx, fallback_wtype
                    )
                    result.append(
                        fused_mul_mat_gguf_op(
                            x,
                            qweight[start:end, :offset].contiguous(),
                            qweight_type,
                        )
                    )
                out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf_op(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out.reshape(*output_leading_shape, out.shape[-1])

    def apply_hc_up_mix(
        self,
        layer: torch.nn.Module,
        raw_lora: torch.Tensor,
        xn: torch.Tensor,
        hc_count: int,
    ) -> torch.Tensor | None:
        """Fuse the exact Qwen3.8 HC-up chain when its strict gate matches."""
        if self.layout is not None or getattr(layer.qweight, "shard_id", None):
            return None
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        output_leading_shape = raw_lora.shape[:-1]
        raw_lora_2d = raw_lora.reshape(-1, raw_lora.shape[-1])
        xn_2d = xn.reshape(-1, xn.shape[-1])
        # Keep token-count dispatch INSIDE the opaque op. A Python M<=4
        # gate is traced away when vLLM initially compiles a larger batch.
        if ((os.environ.get("R9V_HC_MIX_MMQ4", "0") == "1" or os.environ.get("R9V_HC_MIX_MMQ45", "0") == "1")
            and int(qweight_type) == 8 and tuple(qweight.shape) == (10240,340)
            and hc_count == 4 and raw_lora_2d.shape[1] == 320 and xn_2d.shape[1] == 10240
            and raw_lora_2d.dtype == torch.bfloat16 and xn_2d.dtype == torch.bfloat16):
            out = fused_qwen38_hc_up_mix(raw_lora_2d, xn_2d, qweight, hc_count)
            return out.reshape(*output_leading_shape, out.shape[-1])
        if not _use_qwen38_fused_hc_up_mix(
            raw_lora_2d,
            xn_2d,
            qweight,
            qweight_type,
            hc_count,
        ):
            return None
        out = fused_qwen38_hc_up_mix(
            raw_lora_2d,
            xn_2d,
            qweight,
            hc_count,
        )
        return out.reshape(*output_leading_shape, out.shape[-1])
