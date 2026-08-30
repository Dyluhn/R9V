# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_bench_pp():
    path = ROOT / "scripts/bench-pp.py"
    spec = importlib.util.spec_from_file_location("r9v_bench_pp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *args) -> None:
        return None


def test_empty_decoded_token_still_ends_prompt_processing(monkeypatch) -> None:
    bench = _load_bench_pp()
    first = {"choices": [{"text": "", "finish_reason": "length"}]}
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": 100, "completion_tokens": 1},
    }
    response = _Response(
        [
            f"data: {json.dumps(first)}\n".encode(),
            f"data: {json.dumps(usage)}\n".encode(),
            b"data: [DONE]\n",
        ]
    )
    monkeypatch.setattr(bench, "post_json", lambda *args, **kwargs: response)
    times = iter((10.0, 12.0, 12.05, 12.1))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))

    result = bench.timed_completion("http://example", "model", "p", 1, 5)

    assert result["ttft_s"] == 2.0
    assert result["pp_tok_s"] == 50.0
    assert result["completion_tokens"] == 1
