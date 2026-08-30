# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/analyze-pp-profile.py"
SPEC = importlib.util.spec_from_file_location("analyze_pp_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def event(name, ts, dur, category="user_annotation", external_id=None):
    args = {} if external_id is None else {"External id": external_id}
    return {"name": name, "ts": ts, "dur": dur, "ph": "X", "cat": category, "args": args}


def test_prefill_external_id_attribution_and_rank_pairing():
    rank0_events = [
        event("execute_context_1(1024)_generation_0(0)", 0, 100),
        event("gpu_model_runner: forward", 10, 70, external_id=9),
        event("vllm::ggml_mul_mat_a8", 12, 5, external_id=9),
        event("ggml_mul_mat_vec_a8", 90, 40, category="kernel", external_id=9),
        event("execute_context_1(1024)_generation_0(0)", 200, 110),
        event("gpu_model_runner: forward", 210, 80, external_id=10),
        event("tiered_moe_vec_subgroup", 300, 60, category="kernel", external_id=10),
    ]
    rank1_events = [
        event("execute_context_1(1024)_generation_0(0)", 0, 120),
        event("gpu_model_runner: forward", 10, 90, external_id=19),
        event("ggml_mul_mat_vec_a8", 100, 50, category="kernel", external_id=19),
    ]

    rank0 = MODULE.summarize_rank(rank0_events)
    rank1 = MODULE.summarize_rank(rank1_events)
    paired = MODULE.summarize_pair(rank0, rank1)

    assert rank0["prefill_chunk_count"] == 2
    assert rank0["context_tokens_total"] == 2048
    assert rank0["chunks"][0]["components_us"] == {"dense_linear": 40.0}
    assert rank0["chunks"][1]["components_us"] == {"routed_experts": 60.0}
    assert rank0["chunks"][0]["phase_device_sum_us"] == {"forward": 40.0}
    assert paired["paired_chunks"] == 1
    assert paired["critical_execute_cpu_per_chunk"]["mean_us"] == 120.0


def test_detailed_execute_scope_is_parsed():
    parsed = MODULE._parse_execute(
        event(
            "execute_1024_context_1(sq1024sk2048sqsq1048576sqsk2097152)"
            "_generation_0(sq0sk0sqsq0sqsk0)",
            0,
            10,
        )
    )
    assert parsed is not None
    assert parsed["scheduled_tokens"] == 1024
    assert parsed["context_tokens"] == 1024
    assert parsed["generation_tokens"] == 0


def test_dense_shape_scope_attributes_nested_custom_op_kernel():
    events = [
        event("execute_context_1(1024)_generation_0(0)", 0, 100),
        {
            **event(
                "qwen38_dense_q8_m1024_n8192_k2560",
                10,
                40,
                external_id=100,
            ),
            "pid": 7,
            "tid": 9,
        },
        {
            **event(
                "_C_gguf::ggml_mul_mat_a8",
                12,
                20,
                category="cpu_op",
                external_id=101,
            ),
            "pid": 7,
            "tid": 9,
        },
        event("mul_mat_q8_0", 60, 25, category="kernel", external_id=101),
    ]

    result = MODULE.summarize_rank(events)
    shape = result["chunks"][0]["dense_shapes"][
        "qwen38_dense_q8_m1024_n8192_k2560"
    ]
    assert shape == {
        "calls": 1,
        "kernel_events": 1,
        "sum_us": 25.0,
        "mean_us": 25.0,
        "device_events": [
            {
                "name": "mul_mat_q8_0",
                "calls": 1,
                "sum_us": 25.0,
                "mean_us": 25.0,
            }
        ],
    }
