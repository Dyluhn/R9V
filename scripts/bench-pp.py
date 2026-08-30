#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure OpenAI-compatible prompt processing with prefix-cache misses.

The benchmark slices distinct regions from a caller-supplied text corpus and
uses the server's ``/tokenize`` endpoint to target a requested prompt length.
Every request starts with a unique nonce, so repeated runs cannot silently turn
into prefix-cache benchmarks.  PP is reported from request start to the first
streamed output token (TTFT), matching vLLM's serving benchmark convention.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def token_count(base_url: str, model: str, prompt: str, timeout: float) -> int:
    with post_json(
        f"{base_url}/tokenize", {"model": model, "prompt": prompt}, timeout
    ) as response:
        result = json.load(response)
    return int(result["count"])


def fit_prompt(
    base_url: str,
    model: str,
    prefix: str,
    corpus: str,
    target_tokens: int,
    timeout: float,
) -> tuple[str, int]:
    if not corpus:
        raise ValueError("corpus slice is empty")
    low, high = 1, len(corpus)
    best = prefix + corpus[:1]
    best_count = token_count(base_url, model, best, timeout)
    while low <= high:
        middle = (low + high) // 2
        candidate = prefix + corpus[:middle]
        count = token_count(base_url, model, candidate, timeout)
        if abs(count - target_tokens) < abs(best_count - target_tokens):
            best, best_count = candidate, count
        if count < target_tokens:
            low = middle + 1
        elif count > target_tokens:
            high = middle - 1
        else:
            return candidate, count
    return best, best_count


def timed_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 7,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_content = None
    usage = None
    with post_json(f"{base_url}/v1/completions", payload, timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            now = time.perf_counter()
            item = json.loads(line[6:])
            if item.get("usage"):
                usage = item["usage"]
            choices = item.get("choices") or []
            # A generated token can legitimately decode to an empty string
            # (for example a special token).  The SSE choice still marks the
            # first output-token event and therefore the end of prompt
            # processing; requiring truthy text silently drops valid PP runs.
            if choices and "text" in choices[0] and first_content is None:
                first_content = now
    ended = time.perf_counter()
    if first_content is None or usage is None:
        raise RuntimeError("stream did not return content and usage")
    prompt_tokens = int(usage["prompt_tokens"])
    ttft = first_content - started
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(usage["completion_tokens"]),
        "ttft_s": ttft,
        "e2e_s": ended - started,
        "pp_tok_s": prompt_tokens / ttft,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--start-run",
        type=int,
        default=1,
        help="one-based run number, for resuming an interrupted matrix",
    )
    parser.add_argument(
        "--total-runs",
        type=int,
        help="total matrix size used to preserve corpus offsets when resuming",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="idle time between requests for sustained-service checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the full run list and summary as JSON",
    )
    args = parser.parse_args()
    if args.target_tokens < 128:
        parser.error("--target-tokens must be at least 128")
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.start_run < 1:
        parser.error("--start-run must be positive")
    total_runs = args.total_runs or args.runs
    if total_runs < args.start_run + args.runs - 1:
        parser.error("--total-runs does not cover the requested run range")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be non-negative")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    corpus = args.corpus.resolve(strict=True).read_text(errors="replace")
    minimum_chars = args.target_tokens * 4
    if len(corpus) < minimum_chars:
        raise ValueError(
            f"corpus is too small ({len(corpus)} chars); need at least "
            f"{minimum_chars} chars for a diverse target prompt"
        )
    run_results = []
    max_offset = max(0, len(corpus) - minimum_chars)
    total_runs = args.total_runs or args.runs
    for request_index in range(args.runs):
        run = args.start_run + request_index
        offset = (
            0 if total_runs == 1 else max_offset * (run - 1) // (total_runs - 1)
        )
        nonce = time.time_ns()
        prefix = f"R9V PP validation run={run} nonce={nonce}.\n"
        prompt, fitted_tokens = fit_prompt(
            args.url.rstrip("/"),
            args.model,
            prefix,
            corpus[offset:],
            args.target_tokens,
            args.timeout,
        )
        result = timed_completion(
            args.url.rstrip("/"),
            args.model,
            prompt,
            args.max_tokens,
            args.timeout,
        )
        result.update({"run": run, "offset": offset, "fitted_tokens": fitted_tokens})
        run_results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if request_index + 1 < args.runs and args.pause_seconds:
            time.sleep(args.pause_seconds)

    rates = [float(result["pp_tok_s"]) for result in run_results]
    summary = {
        "runs": len(rates),
        "pp_tok_s_mean": statistics.fmean(rates),
        "pp_tok_s_median": statistics.median(rates),
        "pp_tok_s_p05": percentile(rates, 0.05),
        "pp_tok_s_p95": percentile(rates, 0.95),
        "pp_tok_s_min": min(rates),
        "pp_tok_s_max": max(rates),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    if args.output:
        payload = {
            "corpus": str(args.corpus.resolve()),
            "target_tokens": args.target_tokens,
            "start_run": args.start_run,
            "total_runs": total_runs,
            "max_tokens": args.max_tokens,
            "pause_seconds": args.pause_seconds,
            "results": run_results,
            "summary": summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
