#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attribute chunked-prefill wall time and GPU work from paired Kineto traces.

The input traces must come from the V2 GPU runner with
``VLLM_CUSTOM_SCOPES_FOR_PROFILING=1``.  Kernel association uses Kineto
external IDs rather than timestamp containment because HIP launches are
asynchronous.  Results deliberately report both correlated kernel-duration
sums and interval unions; neither is mislabeled as end-to-end request time.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Event = dict[str, Any]

_EXECUTE_RE = re.compile(
    r"^execute_context_(?P<context_requests>\d+)\((?P<context_tokens>\d+)\)"
    r"_generation_(?P<generation_requests>\d+)\((?P<generation_tokens>\d+)\)$"
)
_DETAILED_EXECUTE_RE = re.compile(
    r"^execute_(?P<scheduled_tokens>\d+)_context_(?P<context_requests>\d+)"
    r"\(sq(?P<context_tokens>\d+).*\)_generation_"
    r"(?P<generation_requests>\d+)\(sq(?P<generation_tokens>\d+).*\)$"
)
_PHASES = {
    "gpu_model_runner: preprocess": "preprocess",
    "launch_ple_offload": "ple_submit",
    "gpu_model_runner: forward": "forward",
    "gpu_model_runner: postprocess": "postprocess",
    "gpu_model_runner: sample": "sample",
    "gpu_model_runner: draft": "draft",
    "gpu_model_runner: bookkeep": "bookkeep",
}
_DENSE_SHAPE_RE = re.compile(
    r"^qwen38_dense_q(?P<qtype>\d+)_m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)$"
)


def load_events(path: Path) -> list[Event]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise TypeError(f"{path}: missing traceEvents list")
    return events


def _start(event: Event) -> float:
    return float(event["ts"])


def _end(event: Event) -> float:
    return _start(event) + float(event.get("dur", 0.0))


def _inside(inner: Event, outer: Event) -> bool:
    return _start(outer) <= _start(inner) and _end(inner) <= _end(outer)


def _external_id(event: Event) -> int | None:
    value = event.get("args", {}).get("External id")
    return value if isinstance(value, int) else None


def _interval_union_us(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    materialized = list(values)
    if not materialized:
        return {"count": 0}
    return {
        "count": len(materialized),
        "mean_us": statistics.fmean(materialized),
        "median_us": statistics.median(materialized),
        "p95_us": _percentile(materialized, 0.95),
        "min_us": min(materialized),
        "max_us": max(materialized),
    }


def _parse_execute(event: Event) -> dict[str, int] | None:
    name = str(event.get("name", ""))
    match = _EXECUTE_RE.match(name) or _DETAILED_EXECUTE_RE.match(name)
    if match is None:
        return None
    return {key: int(value) for key, value in match.groupdict().items() if value}


def _execute_scopes(events: list[Event]) -> list[tuple[Event, dict[str, int]]]:
    result = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "user_annotation":
            continue
        parsed = _parse_execute(event)
        if parsed is not None:
            result.append((event, parsed))
    return sorted(result, key=lambda item: _start(item[0]))


def _launchers_by_external_id(events: list[Event]) -> dict[int, list[Event]]:
    result: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        external_id = _external_id(event)
        if (
            external_id is not None
            and event.get("ph") == "X"
            and event.get("cat")
            not in {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime"}
        ):
            result[external_id].append(event)
    return result


def _dense_shape_scopes(events: list[Event]) -> list[Event]:
    return [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and _DENSE_SHAPE_RE.match(str(event.get("name", "")))
    ]


def _dense_scope_index(
    scopes: list[Event],
) -> dict[tuple[int, int], tuple[list[float], list[tuple[float, float, str]]]]:
    grouped: dict[tuple[int, int], list[tuple[float, float, str]]] = defaultdict(list)
    for scope in scopes:
        key = (int(scope.get("pid", -1)), int(scope.get("tid", -1)))
        grouped[key].append((_start(scope), _end(scope), str(scope["name"])))
    result = {}
    for key, intervals in grouped.items():
        intervals.sort()
        result[key] = ([start for start, _, _ in intervals], intervals)
    return result


def _dense_label_for_launcher(
    launcher: Event,
    index: dict[
        tuple[int, int], tuple[list[float], list[tuple[float, float, str]]]
    ],
) -> str | None:
    key = (int(launcher.get("pid", -1)), int(launcher.get("tid", -1)))
    indexed = index.get(key)
    if indexed is None:
        return None
    starts, intervals = indexed
    candidate = bisect.bisect_right(starts, _start(launcher)) - 1
    if candidate < 0:
        return None
    start, end, label = intervals[candidate]
    if start <= _start(launcher) and _end(launcher) <= end:
        return label
    return None


def _device_events(events: list[Event]) -> list[Event]:
    return [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]


def _associated_device_events(
    scope: Event,
    device_events: list[Event],
    launchers: dict[int, list[Event]],
) -> list[tuple[Event, list[Event]]]:
    result = []
    for device_event in device_events:
        external_id = _external_id(device_event)
        if external_id is None:
            continue
        inside = [
            launcher
            for launcher in launchers.get(external_id, ())
            if _inside(launcher, scope)
        ]
        if inside:
            result.append((device_event, inside))
    return result


def classify_device_event(event: Event, launchers: list[Event]) -> str:
    text = " ".join(
        [str(event.get("name", ""))]
        + [str(launcher.get("name", "")) for launcher in launchers]
    ).lower()
    category = str(event.get("cat", ""))
    if category == "gpu_memcpy":
        return "memory_copy"
    if category == "gpu_memset":
        return "memory_clear"
    if "ple_offload" in text or "streamwait" in text:
        return "ple_wait_or_transfer"
    if any(
        token in text
        for token in (
            "nccl",
            "rccl",
            "all_reduce",
            "allreduce",
            "all_gather",
            "allgather",
            "reduce_scatter",
        )
    ):
        return "collectives"
    if any(
        token in text
        for token in (
            "gated_delta",
            "causal_conv1d",
            "chunk_scaled_dot",
            "chunk_local_cumsum",
            "chunk_fwd_kernel_o",
            "qwen_gdn",
            "gdn_",
        )
    ):
        return "gdn"
    if "qsa" in text or "topkperrow" in text or "sparse_attn_indexer" in text:
        return "qsa_indexer"
    if any(
        token in text
        for token in (
            "tiered_moe",
            "moe_vec",
            "moe_sum",
            "topkgating",
            "fused_moe",
            "q8_bf16_moe",
            "reuse3",
        )
    ):
        return "routed_experts"
    if any(
        token in text
        for token in (
            "dense_mmvq",
            "ggml_mul_mat",
            "mul_mat_vec_q",
            "mul_mat_q",
            "wvsplitk",
            "gemv",
            "gemm",
            "cijk_",
        )
    ):
        return "dense_linear"
    if any(token in text for token in ("flash_attn", "attention", "paged_gqa")):
        return "attention"
    if any(
        token in text
        for token in (
            "rms_norm",
            "layer_norm",
            "elementwise",
            "reduce_kernel",
            "catarraybatchedcopy",
        )
    ):
        return "norm_and_glue"
    return "other"


def _phase_scopes(events: list[Event], execute: Event) -> dict[str, list[Event]]:
    result: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        phase = _PHASES.get(str(event.get("name", "")))
        if (
            phase is not None
            and event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
            and _inside(event, execute)
        ):
            result[phase].append(event)
    return result


def summarize_rank(events: list[Event]) -> dict[str, Any]:
    execute_scopes = _execute_scopes(events)
    prefills = [
        (scope, details)
        for scope, details in execute_scopes
        if details["context_requests"] > 0 and details["context_tokens"] > 0
    ]
    device_events = _device_events(events)
    launchers = _launchers_by_external_id(events)
    dense_scopes = _dense_shape_scopes(events)
    dense_scope_index = _dense_scope_index(dense_scopes)
    chunks = []
    all_component_names: set[str] = set()
    for index, (scope, details) in enumerate(prefills):
        associated = _associated_device_events(scope, device_events, launchers)
        component_us: Counter[str] = Counter()
        kernel_us: Counter[str] = Counter()
        kernel_calls: Counter[str] = Counter()
        dense_shape_us: Counter[str] = Counter()
        dense_shape_kernel_events: Counter[str] = Counter()
        dense_shape_kernel_us: dict[str, Counter[str]] = defaultdict(Counter)
        dense_shape_kernel_calls: dict[str, Counter[str]] = defaultdict(Counter)
        dense_shape_calls = Counter(
            str(dense_scope["name"])
            for dense_scope in dense_scopes
            if _inside(dense_scope, scope)
        )
        for device_event, event_launchers in associated:
            duration = float(device_event.get("dur", 0.0))
            component_us[classify_device_event(device_event, event_launchers)] += duration
            kernel_name = str(device_event.get("name", ""))
            kernel_us[kernel_name] += duration
            kernel_calls[kernel_name] += 1
            dense_labels = {
                label
                for launcher in event_launchers
                if (label := _dense_label_for_launcher(launcher, dense_scope_index))
                is not None
            }
            if len(dense_labels) > 1:
                raise RuntimeError(
                    f"device event belongs to multiple dense shape scopes: "
                    f"{sorted(dense_labels)}"
                )
            if dense_labels:
                dense_label = next(iter(dense_labels))
                dense_shape_us[dense_label] += duration
                dense_shape_kernel_events[dense_label] += 1
                dense_shape_kernel_us[dense_label][kernel_name] += duration
                dense_shape_kernel_calls[dense_label][kernel_name] += 1
        phases = _phase_scopes(events, scope)
        phase_cpu_us = {
            name: sum(float(event.get("dur", 0.0)) for event in scopes)
            for name, scopes in phases.items()
        }
        phase_device_us = {}
        attributed_ids: set[int] = set()
        for name, scopes in phases.items():
            phase_events: dict[int, Event] = {}
            for phase_scope in scopes:
                for device_event, _ in _associated_device_events(
                    phase_scope, device_events, launchers
                ):
                    phase_events[id(device_event)] = device_event
            attributed_ids.update(phase_events)
            phase_device_us[name] = sum(
                float(event.get("dur", 0.0)) for event in phase_events.values()
            )
        device_sum = sum(float(event.get("dur", 0.0)) for event, _ in associated)
        attributed_sum = sum(
            float(event.get("dur", 0.0))
            for event, _ in associated
            if id(event) in attributed_ids
        )
        all_component_names.update(component_us)
        next_start = (
            _start(prefills[index + 1][0]) if index + 1 < len(prefills) else None
        )
        chunks.append(
            {
                "chunk": index,
                "context_tokens": details["context_tokens"],
                "execute_cpu_us": float(scope.get("dur", 0.0)),
                "start_to_next_prefill_us": (
                    next_start - _start(scope) if next_start is not None else None
                ),
                "device_event_count": len(associated),
                "device_sum_us": device_sum,
                "device_union_us": _interval_union_us(
                    (_start(event), _end(event)) for event, _ in associated
                ),
                "device_attribution_ratio": attributed_sum / device_sum
                if device_sum
                else 1.0,
                "phase_cpu_us": dict(sorted(phase_cpu_us.items())),
                "phase_device_sum_us": dict(sorted(phase_device_us.items())),
                "components_us": dict(sorted(component_us.items())),
                "dense_shapes": {
                    name: {
                        "calls": dense_shape_calls[name],
                        "kernel_events": dense_shape_kernel_events[name],
                        "sum_us": duration,
                        "mean_us": duration / dense_shape_calls[name],
                        "device_events": [
                            {
                                "name": kernel_name,
                                "calls": dense_shape_kernel_calls[name][kernel_name],
                                "sum_us": kernel_duration,
                                "mean_us": kernel_duration
                                / dense_shape_kernel_calls[name][kernel_name],
                            }
                            for kernel_name, kernel_duration in dense_shape_kernel_us[
                                name
                            ].most_common()
                        ],
                    }
                    for name, duration in sorted(dense_shape_us.items())
                },
                "top_device_events_us": [
                    {
                        "name": name,
                        "sum_us": duration,
                        "calls": kernel_calls[name],
                        "mean_us": duration / kernel_calls[name],
                    }
                    for name, duration in kernel_us.most_common(20)
                ],
            }
        )

    component_samples: dict[str, list[float]] = defaultdict(list)
    for chunk in chunks:
        for component in all_component_names:
            component_samples[component].append(
                float(chunk["components_us"].get(component, 0.0))
            )
    return {
        "execute_scope_count": len(execute_scopes),
        "prefill_chunk_count": len(chunks),
        "context_tokens_total": sum(chunk["context_tokens"] for chunk in chunks),
        "execute_cpu_per_chunk": _stats(chunk["execute_cpu_us"] for chunk in chunks),
        "device_sum_per_chunk": _stats(chunk["device_sum_us"] for chunk in chunks),
        "device_union_per_chunk": _stats(chunk["device_union_us"] for chunk in chunks),
        "device_attribution_per_chunk": _stats(
            chunk["device_attribution_ratio"] for chunk in chunks
        ),
        "components_per_chunk": {
            component: _stats(values)
            for component, values in sorted(component_samples.items())
        },
        "chunks": chunks,
    }


def summarize_pair(rank0: dict[str, Any], rank1: dict[str, Any]) -> dict[str, Any]:
    left = rank0["chunks"]
    right = rank1["chunks"]
    count = min(len(left), len(right))
    if not count:
        return {"paired_chunks": 0}
    return {
        "paired_chunks": count,
        "critical_execute_cpu_per_chunk": _stats(
            max(left[index]["execute_cpu_us"], right[index]["execute_cpu_us"])
            for index in range(count)
        ),
        "execute_cpu_rank_skew_per_chunk": _stats(
            abs(left[index]["execute_cpu_us"] - right[index]["execute_cpu_us"])
            for index in range(count)
        ),
        "critical_device_sum_per_chunk": _stats(
            max(left[index]["device_sum_us"], right[index]["device_sum_us"])
            for index in range(count)
        ),
    }


def analyze(rank0_path: Path, rank1_path: Path) -> dict[str, Any]:
    rank0 = summarize_rank(load_events(rank0_path))
    rank1 = summarize_rank(load_events(rank1_path))
    return {
        "rank0_path": str(rank0_path),
        "rank1_path": str(rank1_path),
        "rank0": rank0,
        "rank1": rank1,
        "paired": summarize_pair(rank0, rank1),
        "interpretation": (
            "execute_cpu_us is the profiled worker scope; device_sum_us is a "
            "correlated duration sum and can exceed wall time when streams overlap; "
            "device_union_us removes overlap. The OpenAI request wall and usage token "
            "counts remain the throughput authority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank0", type=Path)
    parser.add_argument("rank1", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.rank0.resolve(strict=True),
        args.rank1.resolve(strict=True),
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
