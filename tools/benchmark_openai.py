# SPDX-License-Identifier: Apache-2.0
"""Measure one OpenAI-compatible streaming request without SDK dependencies."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8004/v1")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    prompt.add_argument(
        "--repeat-count",
        type=int,
        help="Build a deterministic long-context corpus in memory.",
    )
    parser.add_argument(
        "--repeat-prefix",
        default="R9V long-context qualification corpus: ",
    )
    parser.add_argument(
        "--repeat-text",
        default=(
            "Photosynthesis converts light into chemical energy while plants "
            "exchange carbon with the atmosphere. "
        ),
    )
    parser.add_argument(
        "--repeat-suffix",
        default="Continue with a technical numbered analysis.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=false through the model chat template.",
    )
    return parser.parse_args()


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if getattr(args, "repeat_count", None) is not None:
        if args.repeat_count < 1:
            raise ValueError("--repeat-count must be positive")
        return (
            args.repeat_prefix
            + args.repeat_text * args.repeat_count
            + args.repeat_suffix
        )
    return args.prompt_file.read_text(encoding="utf-8")


def _payload(args: argparse.Namespace, prompt: str) -> bytes:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if getattr(args, "disable_thinking", False):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return json.dumps(payload).encode("utf-8")


def _delta_text(delta: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return visible and reasoning text across OpenAI-compatible variants."""
    return delta.get("content"), (
        delta.get("reasoning") or delta.get("reasoning_content")
    )


def _stream(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/chat/completions",
        data=_payload(args, prompt),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content: float | None = None
    last_content: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta", {})
                content, reasoning = _delta_text(delta)
                if content or reasoning:
                    now = time.perf_counter()
                    first_content = first_content or now
                    last_content = now
                    if content:
                        text_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)

    ended = time.perf_counter()
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    ttft = (first_content or ended) - started
    decode_seconds = max(0.0, (last_content or ended) - (first_content or ended))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_seconds": ttft,
        "total_seconds": ended - started,
        "pp_tokens_per_second": prompt_tokens / ttft if ttft else None,
        "tg_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds
            if completion_tokens > 1 and decode_seconds
            else None
        ),
        "text": "".join(text_parts),
        "reasoning_text": "".join(reasoning_parts),
    }


def main() -> int:
    args = _parse_args()
    result = _stream(args, _read_prompt(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
