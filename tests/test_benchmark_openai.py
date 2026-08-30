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
    args = SimpleNamespace(model="model", max_tokens=17, disable_thinking=False)
    payload = json.loads(MODULE._payload(args, "hello"))

    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 17
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_payload_can_disable_thinking() -> None:
    args = SimpleNamespace(model="model", max_tokens=17, disable_thinking=True)
    payload = json.loads(MODULE._payload(args, "hello"))

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_delta_text_accepts_reasoning_variants() -> None:
    assert MODULE._delta_text({"content": "answer"}) == ("answer", None)
    assert MODULE._delta_text({"reasoning": "thought"}) == (None, "thought")
    assert MODULE._delta_text({"reasoning_content": "legacy"}) == (
        None,
        "legacy",
    )


def test_read_prompt_builds_long_context_corpus_in_memory() -> None:
    args = SimpleNamespace(
        prompt=None,
        prompt_file=None,
        repeat_count=3,
        repeat_prefix="prefix:",
        repeat_text="block ",
        repeat_suffix="suffix",
    )

    assert MODULE._read_prompt(args) == "prefix:block block block suffix"
