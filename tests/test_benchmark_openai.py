# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_openai", ROOT / "tools/benchmark_openai.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_payload_requests_stream_usage() -> None:
    args = SimpleNamespace(model="model", max_tokens=17)
    payload = json.loads(MODULE._payload(args, "hello"))

    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 17
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
