# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp model."""

from collections.abc import Callable, Iterable
from itertools import islice
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateCopyFuncsByType,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from ..config import Qwen4ExpConfig
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .ple_layer import Qwen4ExpPLELayer
from .qsa import Qwen4ExpQSAAttention


logger = init_logger(__name__)


def _diagnostic_layer_sync(layer_idx: int, stage: str) -> None:
    if os.environ.get("VLLM_QWEN_GDN_DIAGNOSTIC_SYNC", "0") != "1":
        return
    logger.warning("Qwen4 diagnostic: layer=%d %s begin", layer_idx, stage)
    torch.accelerator.synchronize()
    logger.warning("Qwen4 diagnostic: layer=%d %s end", layer_idx, stage)


def without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return ``None`` for weights excluded from Qwen4Exp ModelOpt-FP4."""

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


_QWEN4_EXP_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# The checkpoint keeps down and injection projections separate; runtime packs
# them into adjacent logical shards of one MergedColumnParallelLinear.
_HC_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "hyper_connection.input_mix_weight_down.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        "hyper_connection.block_inject_weight.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
    }
)


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen4Exp HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)


# KVA capture (research only): with R9V_KVA_CAPTURE_DIR set (+ an ENABLE file there) and
# --enforce-eager, TP rank 0 saves, per prefill forward of >= 256 tokens, the materialized
# 4-stream state entering each split layer (boundary_S), every block input from the first
# split on (block_input_L), and the final pre-mixer state (final_multi_hidden, the MTP
# drafter's input), keeping every stride-th token. ENABLE may hold JSON
# {"splits": [..], "stride": N, "subdir": "name"} for the next forwards; empty = the env
# defaults below. Projection targets are computed offline from the GGUF weights, so TP
# sharding never enters the capture.
_KVA_CAPTURE_DIR = os.environ.get("R9V_KVA_CAPTURE_DIR")
_KVA_SPLITS = [int(s) for s in os.environ.get("R9V_KVA_CAPTURE_SPLITS", "24").split(",")]
_KVA_MIN_TOKENS = 256
_KVA_STRIDE = int(os.environ.get("R9V_KVA_CAPTURE_STRIDE", "1"))  # keep every Nth token
_KVA: dict = {"on": False, "count": 0}
_kva_pending: dict[str, torch.Tensor] = {}


def _kva_begin(num_tokens: int) -> None:
    """Decide once per forward whether (and what) rank 0 captures."""
    _KVA["on"] = False
    if not _KVA_CAPTURE_DIR or num_tokens < _KVA_MIN_TOKENS:
        return
    try:  # the ENABLE file keeps startup warmup out of the data
        text = (Path(_KVA_CAPTURE_DIR) / "ENABLE").read_text().strip()
    except FileNotFoundError:
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return
    config = json.loads(text) if text else {}
    splits = sorted(config.get("splits", _KVA_SPLITS))
    _KVA.update(on=True, splits=set(splits), first=splits[0],
                stride=int(config.get("stride", _KVA_STRIDE)), subdir=config.get("subdir", ""))


def _kva_keep(name: str, x: torch.Tensor) -> None:
    _kva_pending[name] = x[:: _KVA["stride"]].detach().clone()


def _kva_flush(positions: torch.Tensor, input_ids: torch.Tensor | None) -> None:
    step = _KVA["stride"]
    record = {k: v.to("cpu") for k, v in _kva_pending.items()}
    record["positions"] = (positions[..., ::step] if positions.dim() == 2 else positions[::step]).to("cpu")
    if input_ids is not None:
        record["input_ids"] = input_ids[::step].to("cpu")
    record["stride"] = step
    out = Path(_KVA_CAPTURE_DIR) / _KVA["subdir"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(record, out / f"capture_{_KVA['count']:05d}.pt")
    _KVA["count"] += 1
    _kva_pending.clear()


# Determinism capture (research only): with R9V_DET_CAPTURE_DIR set (+ an ENABLE file there)
# and --enforce-eager, TP rank 0 records per prefill forward, for every token and every layer,
# a sketch of the attention input (a), attention output (t), MoE input (m) and MoE output (o),
# plus the n-gram (PLE) contribution (p) and the raw n-gram embedding rows in full.
# Sketch = 8-dim fixed random projection + L2 norm (magnitude of a difference) and an exact
# integer hash of the bf16 bits (bitwise equality; order-independent integer sum, so the hash
# itself is deterministic). ~9 KB/token for 48 layers.
_DET_DIR = os.environ.get("R9V_DET_CAPTURE_DIR")
_DET_DIM = 8
# R9V_DET_FULL: comma list of point names (e.g. "m30,o30") also saved in full (bf16).
_DET_FULL = set(filter(None, os.environ.get("R9V_DET_FULL", "").split(",")))
_det_pending: dict[str, torch.Tensor] = {}
_det_state: dict = {"on": False, "count": 0, "proj": {}}


def _det_install_route_hook() -> None:
    """Record each MoE layer's routed experts (top-10 ids, sorted) while capturing: the GGUF
    MoE apply() reports them to route_profile.record_route_profile on every call."""
    try:
        from vllm_gguf_plugin.quantization import route_profile
    except ImportError:
        return
    if getattr(route_profile, "_r9v_det_hooked", False):
        return
    original = route_profile.record_route_profile

    def record(layer_name, topk_ids):
        if _det_state["on"] and "mtp" not in layer_name:
            import re

            match = re.search(r"layers\.(\d+)\.", layer_name)
            if match:
                _det_pending[f"r{match.group(1)}"] = topk_ids.detach().to(torch.int16).sort(1).values
        return original(layer_name, topk_ids)

    route_profile.record_route_profile = record
    route_profile._r9v_det_hooked = True


def _det_begin(num_tokens: int) -> None:
    if _DET_DIR:
        _det_install_route_hook()
    on = False
    if _DET_DIR and num_tokens >= 32 and (Path(_DET_DIR) / "ENABLE").exists():
        from vllm.distributed import get_tensor_model_parallel_rank
        on = get_tensor_model_parallel_rank() == 0
    _det_state["on"] = on


def _det_sketch(name: str, x: torch.Tensor) -> None:
    if not _det_state["on"]:
        return
    x = x.detach().reshape(x.shape[0], -1)
    key = (x.device, x.shape[1])
    if key not in _det_state["proj"]:
        gen = torch.Generator(device="cpu").manual_seed(1234 + x.shape[1])
        proj = torch.randn(x.shape[1], _DET_DIM, generator=gen)
        weights = torch.randint(1, 1 << 20, (x.shape[1],), generator=gen, dtype=torch.int64)
        _det_state["proj"][key] = (proj.to(x.device), weights.to(x.device))
    proj, weights = _det_state["proj"][key]
    xf = x.float()
    bits = x.view(torch.int16) if x.element_size() == 2 else x.view(torch.int32)
    _det_pending[name + ".s"] = torch.cat([xf @ proj, xf.norm(dim=1, keepdim=True)], 1)
    _det_pending[name + ".h"] = (bits.long() * weights).sum(1)
    if name in _DET_FULL:
        _det_pending[name] = x.clone()


def _det_full(name: str, x: torch.Tensor) -> None:
    if _det_state["on"]:
        _det_pending[name] = x.detach().clone()


def _det_flush(positions: torch.Tensor, input_ids: torch.Tensor | None) -> None:
    if not _det_state["on"]:
        return
    record = {k: v.to("cpu") for k, v in _det_pending.items()}
    record["positions"] = (positions[0] if positions.dim() == 2 else positions).to("cpu")
    if input_ids is not None:
        record["input_ids"] = input_ids.to("cpu")
    run_file = Path(_DET_DIR) / "RUN"
    record["run"] = run_file.read_text().strip() if run_file.exists() else ""
    torch.save(record, Path(_DET_DIR) / f"det_{_det_state['count']:05d}.pt")
    _det_state["count"] += 1
    _det_pending.clear()


# CED (approximate late-layer prefill). R9V_CED_PROJECTOR names the projector file, which
# declares its split S by its keys: "layer.S".."layer.47" plus "final" (research: a comma
# list of projectors, one per split, needs R9V_CED_CONTROL). Prefill chunks the scheduler
# marks as approximate (scheduler_output.r9v_ced_approx, with r9v_ced_split naming S) run
# layers 0..S-1 normally; the projector then predicts each later layer's block input from the
# layer-S boundary state, and each later layer only writes its caches from that prediction
# (GDN: conv + recurrent state; attention: K/V and indexer keys). MoE, attention outputs and
# residual updates are skipped. The exact tail and every decode step run the full model.
# r9v_ced_fill == "full" instead runs each late layer's whole attention block and discards
# the output (the original fill; same cache contents, for timing A/Bs). Approximate chunks
# run this model's uncompiled forward (_ced_entry); exact chunks and decode stay compiled.
_CED_PROJECTOR_PATHS = [p for p in os.environ.get("R9V_CED_PROJECTOR", "").split(",") if p]
_CED = {"approx": False, "split": None, "fill": "lean", "proj": {}, "tokens": 0}
# Research only (R9V_CED_CONTROL set): TP rank 0 keeps the running count of tokens that took
# the approximate branch in this file (next to the scheduler's control file), so a driver can
# attribute approximate tokens to each request, and times every approximate chunk
# (_ced_record). Production logs one line per CED request from the scheduler instead.
_CED_COUNT_PATH = (os.environ["R9V_CED_CONTROL"] + ".approx_tokens") if os.environ.get("R9V_CED_CONTROL") else None


def _ced_split_of(path: str) -> int:
    """The split a projector file declares: its first late layer, with every later layer and
    "final" present."""
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        keys = set(f.keys())
    layers = sorted(int(k.split(".")[1]) for k in keys if k.startswith("layer."))
    if not layers or "final" not in keys or layers != list(range(layers[0], layers[-1] + 1)):
        raise ValueError(f"CED projector {path}: expected keys layer.S..layer.N contiguous plus final, got {sorted(keys)[:5]}...")
    return layers[0]


_CED_FILES = {_ced_split_of(p): p for p in _CED_PROJECTOR_PATHS}
if len(_CED_FILES) != len(_CED_PROJECTOR_PATHS):
    raise ValueError(f"R9V_CED_PROJECTOR lists two projectors with the same split: {_CED_PROJECTOR_PATHS}")
if len(_CED_FILES) > 1 and _CED_COUNT_PATH is None:
    raise ValueError(f"R9V_CED_PROJECTOR lists several projectors, which is research-only (R9V_CED_CONTROL): "
                     f"{_CED_PROJECTOR_PATHS}")
# Projector precision in VRAM: bf16 as stored, or int8 quantized at load (about half the VRAM;
# group-128 scales, kva/quantize_projector.py scores it offline against bf16).
_CED_PRECISION = os.environ.get("R9V_CED_PRECISION", "bf16")
if _CED_PRECISION not in ("bf16", "int8"):
    raise ValueError(f"R9V_CED_PRECISION must be bf16 or int8, got {_CED_PRECISION!r}")
_CED_GROUP = 128


def _ced_install_runner_hook() -> None:
    """Workers learn per step whether the scheduled prefill chunk is approximate, at which
    split, and with which fill.

    Hooks the worker, not a model runner class: this image runs the V2 model runner, and
    Worker.execute_model receives the SchedulerOutput under both runners."""
    from vllm.v1.worker import gpu_worker

    worker = gpu_worker.Worker
    if getattr(worker, "_r9v_ced_hooked", False):
        return
    original = worker.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        _CED["approx"] = bool(getattr(scheduler_output, "r9v_ced_approx", False))
        _CED["split"] = getattr(scheduler_output, "r9v_ced_split", None)
        _CED["fill"] = getattr(scheduler_output, "r9v_ced_fill", "lean")
        _CED["step_start"] = time.perf_counter()  # chunk split: host time at step start
        _CED["request"] = next(iter(getattr(scheduler_output, "num_scheduled_tokens", None) or {}), None)
        return original(self, scheduler_output, *args, **kwargs)

    worker.execute_model = execute_model
    worker._r9v_ced_hooked = True


def _ced_active_split() -> int | None:
    """The split this step approximates from, or None for an exact step."""
    if not _CED["approx"]:
        return None
    split = _CED["split"]
    if split is None:
        if len(_CED_FILES) != 1:
            raise RuntimeError(f"CED step names no split but projectors for {sorted(_CED_FILES)} are loaded")
        return next(iter(_CED_FILES))
    if split not in _CED_FILES:
        raise RuntimeError(f"CED step asks for split {split}; R9V_CED_PROJECTOR has {sorted(_CED_FILES)}")
    return split


def _ced_record(num_tokens: int, started: float, spans: list, split: int) -> None:
    """Log one approximate chunk. head_ms = step start to the layer-split boundary (input prep,
    n-gram wait, layers 0..split-1; host time after a device sync); proj_ms / fill_ms = GPU time
    of the projector matmuls / the late layers' cache fills (CUDA events, spans = (start,
    after projector, after fill, layer type) per layer; fill split into GDN and attention);
    rest_ms = their wall time; gap_ms = time since the previous approximate chunk of the same
    request ended (NaN on its first chunk). Rank 0 logs each chunk and appends it as a JSON
    line to <R9V_CED_CONTROL>.chunks. Synchronizes the device twice per chunk: research only."""
    from vllm.distributed import get_tensor_model_parallel_rank

    torch.cuda.synchronize()
    ended = time.perf_counter()
    same_request = _CED.get("last_request") == _CED.get("request")
    gap_ms = (started - _CED["last_end"]) * 1000 if _CED.get("last_end") and same_request else float("nan")
    _CED["last_end"], _CED["last_request"] = ended, _CED.get("request")
    if get_tensor_model_parallel_rank() != 0:
        return
    _CED["tokens"] += num_tokens
    fill = lambda kind: sum(b.elapsed_time(c) for _, b, c, t in spans if t == kind)  # noqa: E731
    chunk = {"t": time.time(), "request": _CED.get("request"), "tokens": num_tokens,
             "split": split, "fill": _CED["fill"],
             "head_ms": (started - _CED.get("step_start", started)) * 1000,
             "proj_ms": sum(a.elapsed_time(b) for a, b, _, _ in spans),
             "fill_gdn_ms": fill("linear_attention"), "fill_fa_ms": fill("full_attention"),
             "rest_ms": (ended - started) * 1000, "gap_ms": gap_ms}
    chunk["fill_ms"] = chunk["fill_gdn_ms"] + chunk["fill_fa_ms"]
    logger.info("CED approx chunk tokens=%d total=%d split=%d fill=%s head_ms=%.1f proj_ms=%.1f fill_ms=%.1f "
                "(gdn %.1f, attn %.1f) rest_ms=%.1f gap_ms=%.1f", num_tokens, _CED["tokens"], split, _CED["fill"],
                chunk["head_ms"], chunk["proj_ms"], chunk["fill_ms"], chunk["fill_gdn_ms"], chunk["fill_fa_ms"],
                chunk["rest_ms"], gap_ms)
    with open(_CED_COUNT_PATH, "w") as f:
        f.write(str(_CED["tokens"]))
    with open(_CED_COUNT_PATH.replace(".approx_tokens", ".chunks"), "a") as f:
        f.write(json.dumps(chunk) + "\n")


def _ced_mark(timed: bool) -> torch.cuda.Event | None:
    """A recorded CUDA event for the research time split; None in production."""
    if not timed:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


if _CED_FILES:
    _ced_install_runner_hook()


def _ced_quantize(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[out, in + 1] bf16 map -> int8 [out, in], bf16 scale per _CED_GROUP input columns of a row,
    bf16 bias [out]. Same as kva/quantize_projector.py's quantize() (tests/ced_int8_apply.py)."""
    rows = weights.shape[0]
    groups = weights[:, :-1].float().view(rows, -1, _CED_GROUP)
    scale = (groups.abs().amax(dim=2).clamp_min(1e-12) / 127).to(torch.bfloat16)
    q = torch.round(groups / scale.float()[:, :, None]).clamp(-127, 127).to(torch.int8)
    return q.view(rows, -1), scale, weights[:, -1].clone()


def _ced_load(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """A projector file on the device at _CED_PRECISION; int8 is quantized one map at a time on
    the CPU, so the bf16 copy never occupies VRAM."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if _CED_PRECISION == "bf16":
        return load_file(path, device=str(device))
    proj = {}
    with safe_open(path, "pt", device="cpu") as f:
        for name in f.keys():
            q, scale, bias = _ced_quantize(f.get_tensor(name))
            proj.update({name: q.to(device), f"scale.{name}": scale.to(device), f"bias.{name}": bias.to(device)})
    return proj


def _ced_projector(device: torch.device, split: int) -> dict[str, torch.Tensor]:
    """One split's projector maps on the device (_ced_load). One projector is resident per
    device: asking for another split (research) frees the current one and loads the new file
    (VRAM headroom does not allow all of them next to the expert cache)."""
    current = _CED["proj"].get(device)
    if current is None or current[0] != split:
        if current is not None:  # free the resident projector (and its cached blocks) before loading the next
            _CED["proj"][device] = current = None
            torch.cuda.empty_cache()
        started = time.perf_counter()
        _CED["proj"][device] = (split, _ced_load(_CED_FILES[split], device))
        logger.info("CED projector split=%d %s loaded on %s in %.2fs (device free %.2f GiB, torch peak reserved %.2f GiB): %s",
                    split, _CED_PRECISION, device, time.perf_counter() - started, torch.cuda.mem_get_info(device)[0] / 2**30,
                    torch.cuda.max_memory_reserved(device) / 2**30, _CED_FILES[split])
    return _CED["proj"][device][1]


def _ced_apply(proj: dict[str, torch.Tensor], name: str, boundary: torch.Tensor) -> torch.Tensor:
    """One projector map applied to the boundary state. bf16 maps are [out, in + 1] (last column
    = bias); int8 maps (_ced_quantize) are dequantized one map at a time, as
    kva/quantize_projector.py does offline."""
    weights = proj[name]
    if weights.dtype != torch.int8:
        return torch.addmm(weights[:, -1], boundary, weights[:, :-1].t())
    scale = proj[f"scale.{name}"]
    matrix = (weights.view(weights.shape[0], scale.shape[1], -1).to(torch.bfloat16) * scale[:, :, None]).view(weights.shape)
    return torch.addmm(proj[f"bias.{name}"], boundary, matrix.t())


def _ced_rows(linear: nn.Module, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Output columns start:end of linear(x) for a GGUF projection without bias, computing
    only those weight rows, with the kernel linear(x) picks for this x: the combined Q8 WMMA
    prefill kernel or GGML MMQ. Each output column is its own weight row's dot product, so
    the columns are bitwise those of the full call (tests/ced_lean_fill_equivalence.py checks
    it on real weights). Any other dispatch (layouts, guards, mixed shard types, < 16 rows)
    computes every row and slices, which is identical by construction."""
    from vllm_gguf_plugin.quantization import linear as gguf_linear

    weight = linear.qweight
    types = linear.qweight_type
    kinds = {types.shard_weight_type.get(i, types.weight_type) for i in weight.shard_id} if weight.shard_id else {types.weight_type}
    kind = int(next(iter(kinds)))
    if (len(kinds) == 1 and getattr(linear, "bias", None) is None and linear.quant_method.layout is None
            and x.dim() == 2 and x.shape[0] >= 16 and x.is_contiguous() and kind in gguf_linear.MMQ_QUANT_TYPES
            and not gguf_linear._needs_q8_finite_guard(x, weight, kind)
            and not gguf_linear._needs_q8_vision_dequant_fallback(x, weight, kind)):
        part = weight[start:end]
        if gguf_linear._q8_prefill_wmma_supported(x, weight, kind):
            token64 = gguf_linear._Q8_TOKEN64_ENABLED and x.shape[1] in (2560, 3072)
            return gguf_linear._q8_prefill_wmma(x.shape[1]).q8_matmul(part, x, 2 if token64 else 1)
        return gguf_linear.ops.ggml_mul_mat_a8(part, x, kind, part.shape[0])
    return linear(x)[0][:, start:end]


def _ced_fill_gdn(attn: nn.Module, block_input: torch.Tensor, lean: bool) -> None:
    """GDN state update (conv + recurrent state) from a predicted block input; the layer output
    is not needed. Lean: in_proj_qkvz computes only the q/k/v rows (z only gates the output),
    written into a qkvz-shaped buffer so the core op sees the same layout as before."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    if not gdn.GDN_AITER_TRITON_AVAILABLE:
        attn(hidden_states=block_input)  # generic path: full forward, output discarded
        return
    num_tokens = block_input.size(0)
    proj = attn.in_proj_qkvz
    if lean:
        qkv_rows = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size  # [q | k | v | z], as the core op splits it
        qkvz = torch.zeros((num_tokens, proj.qweight.shape[0]), dtype=block_input.dtype, device=block_input.device)
        qkvz[:, :qkv_rows] = _ced_rows(proj, block_input, 0, qkv_rows)
    else:
        qkvz, _ = proj(block_input)
    ba, _ = attn.in_proj_ba(block_input)
    core = torch.empty((num_tokens, attn.num_v_heads // attn.tp_size, attn.head_v_dim),
                       dtype=block_input.dtype, device=block_input.device)
    z = torch.empty_like(core)
    torch.ops.vllm.qwen_gdn_attention_core(
        qkvz.view(num_tokens, -1), ba.view(num_tokens, -1), z, core,
        layer_name=gdn._encode_layer_name(attn.prefix), use_aiter=True)


def _ced_fill_attention(attn: nn.Module, block_input: torch.Tensor, positions: torch.Tensor, lean: bool) -> None:
    """QSA attention cache writes from a predicted block input: the indexer's raw and pooled
    keys, then K/V, exactly as Qwen4ExpQSAAttention._run_qsa writes them. Lean skips what only
    the attention output needs: the Q/gate rows of qkv_proj (zeros in their place), the
    indexer's query and block selection, the attention itself, the gate and o_proj (and its
    all-reduce)."""
    if not lean:
        attn(hidden_states=block_input, positions=positions)
        return
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0]
    if not isinstance(metadata, dict):
        return  # profile / dummy run: the full forward writes nothing either
    main_metadata = metadata[attn.layer_name]
    proj = attn.qkv_proj
    start, end = attn.q_size * 2, attn.q_size * 2 + attn.kv_size * 2  # [q+gate | k | v], as _project_qkv_gate splits it
    qkv = torch.zeros((block_input.size(0), proj.qweight.shape[0]), dtype=block_input.dtype, device=block_input.device)
    qkv[:, start:end] = _ced_rows(proj, block_input, start, end)
    _, k, v, _ = attn._project_qkv_gate(qkv, positions)

    indexer = attn.indexer
    raw_metadata, compressed_metadata = indexer._metadata()
    count = raw_metadata.num_actual_tokens
    qk, _ = indexer.index_qk_proj(block_input[:count])
    token_k = qk.split((indexer.index_n_heads * indexer.index_head_dim,
                        indexer.index_kv_heads * indexer.index_head_dim), dim=-1)[1]
    indexer._update_and_compress(token_k.reshape(-1, 1, indexer.index_head_dim),
                                 positions[..., :count], raw_metadata, compressed_metadata)
    num_tokens = block_input.shape[0]
    attn.impl.do_kv_cache_update(
        attn, k.view(num_tokens, attn.num_kv_heads, attn.head_dim),
        v.view(num_tokens, attn.num_kv_heads, attn.head_dim), attn.kv_cache, main_metadata.slot_mapping)


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen4ExpSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
            quant_config=quant_config,
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            _diagnostic_layer_sync(self.layer_idx, "ple-entry")
            # PLE adds directly to the multi-stream state, so pending HC state
            # must be materialized before the addition.
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None
                _diagnostic_layer_sync(self.layer_idx, "ple-pending-combine")

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            ple_out = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )
            if _det_state["on"]:
                _det_sketch(f"p{self.layer_idx}", ple_out)
                raw = getattr(self.ple.ple_embedding, "_gpu_output_buffer", None)
                if raw is not None:
                    _det_full("ple_raw", raw[: hidden_states.shape[0]])
            hidden_states = hidden_states + ple_out
            _diagnostic_layer_sync(self.layer_idx, "ple-lookup")

        if _KVA["on"] and self.layer_idx in _KVA["splits"]:
            boundary = hidden_states
            if prev_block_output is not None and prev_injection is not None:
                boundary = attn_hc.combine(hidden_states, prev_block_output, prev_injection)
            _kva_keep(f"boundary_{self.layer_idx}", boundary)

        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)
        if _KVA["on"] and self.layer_idx >= _KVA["first"]:
            _kva_keep(f"block_input_{self.layer_idx}", block_input)
        _det_sketch(f"a{self.layer_idx}", block_input)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type == "full_attention":
            attn_out = self.self_attn(
                hidden_states=block_input,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        _diagnostic_layer_sync(self.layer_idx, "attention")
        _det_sketch(f"t{self.layer_idx}", attn_out)

        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        _diagnostic_layer_sync(self.layer_idx, "mlp-hyperconnection")
        _det_sketch(f"m{self.layer_idx}", block_input)
        mlp_out = self.mlp(block_input)
        _diagnostic_layer_sync(self.layer_idx, "mlp")
        _det_sketch(f"o{self.layer_idx}", mlp_out)
        return hidden_states, mlp_out, injection


class Qwen4ExpMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen4Exp routed experts through vLLM's EPLB protocol."""

    def set_moe_parameters(self, layers: Iterable[nn.Module]) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen4ExpSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidual | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
                quant_config=vllm_config.quant_config,
            )
        else:
            self.hyper_connection_mixer = None

        spec_config = vllm_config.speculative_config
        # MTP HC multi-stream outputs: when speculative method=="mtp" and the
        # model uses HC with hc_count>1, retain the pre-final-mixer multi-stream
        # hidden state [T, hc_count*H] so the MTP drafter can feed a real
        # multi-stream backbone hidden on its first step (scheme A). Derived
        # purely from config (NOT node identity) so P/D nodes stay consistent.
        needs_mtp_hidden = (
            spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        _det_begin(positions.shape[-1])
        _kva_begin(positions.shape[-1])
        block_output = None
        injection = None
        last_layer = None
        ced_split = _ced_active_split()
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            if layer_idx == ced_split:
                return self._ced_approximate_rest(hidden_states, block_output, injection, positions, ced_split)
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                # Deepstack is an external addition to the materialized
                # multi-stream state and therefore terminates delayed combine.
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple. Materialize
            # with the HC module that produced the pending injection.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        # The final mixer consumes the last pending combine and returns both
        # the sampled single stream and the materialized multi-stream state.
        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        if _KVA["on"]:
            _kva_keep("final_multi_hidden", multi_hidden)  # MTP drafter input
            _kva_flush(positions, input_ids)
        if _det_state["on"]:
            _det_sketch("final", sample_hidden_states)
            _det_flush(positions, input_ids)
        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream hidden state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states

    def _ced_approximate_rest(self, hidden_states, block_output, injection, positions, split):
        """Fill layers split..end caches from projector predictions (see module note)."""
        timed = _CED_COUNT_PATH is not None
        if timed:
            torch.cuda.synchronize()
        started = time.perf_counter()
        layers = self.layers
        boundary = hidden_states
        if block_output is not None and injection is not None:
            boundary = layers[split].attn_hyper_connection.combine(hidden_states, block_output, injection)
        boundary = boundary.to(torch.bfloat16)
        proj = _ced_projector(boundary.device, split)
        lean = _CED["fill"] != "full"
        spans = []  # research: (start, after projector, after fill, layer type) per late layer
        for idx in range(split, len(layers)):
            layer = layers[idx]
            start = _ced_mark(timed)
            block_input = _ced_apply(proj, f"layer.{idx}", boundary)
            projected = _ced_mark(timed)
            if layer.layer_type == "linear_attention":
                _ced_fill_gdn(layer.linear_attn, block_input, lean)
            else:
                _ced_fill_attention(layer.self_attn, block_input, positions, lean)
            spans.append((start, projected, _ced_mark(timed), layer.layer_type))
        start = _ced_mark(timed)
        multi_hidden = _ced_apply(proj, "final", boundary)
        spans.append((start, _ced_mark(timed), None, "final"))
        if self._mtp_hidden_buffer is not None:  # the MTP drafter's input for these positions
            self._mtp_hidden_buffer[: multi_hidden.shape[0]].copy_(multi_hidden)
        if timed:
            _ced_record(multi_hidden.shape[0], started, spans, split)
        # An approximate chunk never ends the prompt, so these logits are never sampled.
        return multi_hidden.view(multi_hidden.shape[0], self.config.hc_count, -1).mean(1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (
                _remap_qsa_cache_scale_name(name, self._qsa_layer_ids),
                weight,
            )
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        skip_prefixes = []
        if self.hyper_connection_mixer is None:
            # The final HC mixer exists only on the last pipeline stage.  Its
            # checkpoint tensors are global, so earlier stages must discard
            # them just like their PPMissingLayer decoder weights.
            skip_prefixes.append("hyper_connection_mixer")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
            skip_substrs=skip_substrs,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
        if _CED_FILES:
            from vllm.model_executor.layers.ple_offload_layer import is_offload_process

            if not is_offload_process():
                # Load the first listed projector now, not on the first long prompt.
                _ced_projector(torch.device("cuda", torch.cuda.current_device()), next(iter(_CED_FILES)))
        return loaded


def _ced_entry(language_model: Qwen4ExpModel) -> Callable[..., torch.Tensor | IntermediateTensors]:
    """What to call for this step's language-model forward. An approximate CED chunk calls the
    uncompiled forward (the path vLLM itself takes for skip_compiled): the compiled graph and
    the decode CUDA graphs are traced with CED off and never see an approximate chunk."""
    return language_model.forward if _CED["approx"] else language_model


class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4Exp currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen4ExpModelState

        return Qwen4ExpModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Forward kwargs unchanged so the runner's _maybe_add_ngram_kwargs
        # path (query_start_loc / ngram_context) reaches Qwen4ExpModel.
        return _ced_entry(self.model)(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_kernel_size = hf_config.ple_conv_kernel_size
        short_conv_dilation = hf_config.ngram_size
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        hc_count = hf_config.hc_count
        hc_hidden_size = hf_config.hidden_size * hc_count
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=hc_hidden_size,
            conv_kernel=conv_state_len + 1,
            num_spec=num_spec,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {
            mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types
        }

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        """Return all MambaSpecs for this model (GDN layers + PLE layer).

        The PLE layer uses a separate short_conv MambaSpec whose page_size_bytes
        may exceed the GDN spec; callers should take the maximum.
        """
        return (
            MambaSpec(
                shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
                tp_replicated=True,
            ),
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen4ExpProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen4ExpConfig:
        return self.ctx.get_hf_config(Qwen4ExpConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen4ExpForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen4ExpMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen4Exp language model."""

    requires_raw_input_tokens = True

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping | {
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ]
    }

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen4ExpModelState

        return Qwen4ExpModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen4ExpForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self.set_moe_parameters(self.language_model.model.layers)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen4Exp language_model_only does not accept multimodal embeddings"
            )

        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = _ced_entry(self.language_model.model)(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["visual."] if self.language_model_only else None,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen4ExpForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return Qwen4ExpForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        return Qwen4ExpForCausalLM.get_mamba_specs_from_config(vllm_config)


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMixtureOfExperts",
    "Qwen4ExpModel",
    "Qwen4ExpSparseMoeBlock",
]
