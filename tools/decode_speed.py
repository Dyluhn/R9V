#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure decode speed on a running server, right after start and once warm, and save it.

The first ~1,200 decode tokens after a fresh start run slower while caches warm up.
This sends the same fixed greedy requests in rounds, one at a time: round 1 is the
"after start" number, later rounds are the warm number. Decode speed is milliseconds
per decode step: the time between the first and last streamed chunk divided by the
MTP draft steps the server counted for the request (vLLM's spec-decode metrics), so
it does not depend on how many tokens each step accepted. Without those metrics
(no MTP) a step is one token.

Run it through `./r9v soak PROFILE --state-dir DIR -- --decode-speed FILE`, which
takes the port and model name from the saved setup.
"""

from __future__ import annotations

import http.client
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "r9v.decode-speed.v1"
DRAFTS = "vllm:spec_decode_num_drafts_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
# Fixed prompts, so rounds, sessions and machines compare the same work.
PROMPTS = {
    "prose": "Write a detailed essay on how river deltas form, change over centuries, and "
             "affect the people who live on them.",
    "code": "Write a complete Python module that parses a CSV file of bank transactions, "
            "groups them by month and category, and prints a summary table. Include tests.",
    "explain": "Explain step by step how a CPU executes a function call, from the call "
               "instruction to the return, including the stack and registers.",
}
ROUNDS = 3
MAX_TOKENS = 400
TIMEOUT_SECONDS = 600


def spec_counters(metrics_text: str) -> dict[str, float]:
    """Sum the spec-decode draft and accepted-token counters over all label sets."""
    totals = {DRAFTS: 0.0, ACCEPTED: 0.0}
    for line in metrics_text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in totals:
            totals[name] += float(line.rsplit(" ", 1)[1])
    return totals


def decode_numbers(seconds: float, completion_tokens: int, drafts: float, accepted: float) -> dict:
    """ms per step and tokens per step for one request. `seconds` spans the first to
    the last streamed chunk, so it covers every decode step but not prefill."""
    steps = drafts if drafts > 0 else completion_tokens - 1
    return {
        "completion_tokens": completion_tokens,
        "decode_seconds": round(seconds, 4),
        "steps": int(steps),
        "tokens_per_step": round(1 + accepted / drafts, 3) if drafts > 0 else 1.0,
        "ms_per_step": round(1000 * seconds / steps, 2),
    }


def summary(rounds: list[list[dict]]) -> dict:
    """Round 1 is after start; the rest are warm."""
    after_start = rounds[0]
    warm = [run for round_runs in rounds[1:] for run in round_runs]
    return {
        "after_start_ms_per_step": [run["ms_per_step"] for run in after_start],
        "warm_ms_per_step": [run["ms_per_step"] for run in warm],
        "warm_median_ms_per_step": statistics.median(run["ms_per_step"] for run in warm),
        "warm_tokens_per_step": [run["tokens_per_step"] for run in warm],
    }


def _get(port: int, path: str) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f"GET {path} returned HTTP {response.status}")
        return body
    finally:
        connection.close()


def _stream(port: int, model: str, prompt: str) -> tuple[float, int]:
    """One greedy streamed request of exactly MAX_TOKENS tokens: (first-to-last chunk
    seconds, completion tokens)."""
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    })
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT_SECONDS)
    try:
        connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"chat completion returned HTTP {response.status}: {response.read(300)!r}")
        first = last = None
        usage = None
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            if event.get("choices"):
                last = time.monotonic()
                first = first or last
    finally:
        connection.close()
    if first is None or usage is None:
        raise RuntimeError("the server streamed no tokens or no usage")
    return last - first, int(usage["completion_tokens"])


def _source_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10, check=False)
    return result.stdout.strip() or None


def measure(port: int, model: str, output: Path, profile: str | None, ced: str | None) -> dict:
    """Run the rounds against the server on `port` and write the result to `output`
    (which must not exist yet). Returns the result."""
    rounds = []
    for number in range(1, ROUNDS + 1):
        runs = []
        for name, prompt in PROMPTS.items():
            before = spec_counters(_get(port, "/metrics"))
            seconds, tokens = _stream(port, model, prompt)
            after = spec_counters(_get(port, "/metrics"))
            run = decode_numbers(seconds, tokens, after[DRAFTS] - before[DRAFTS],
                                 after[ACCEPTED] - before[ACCEPTED])
            runs.append({"round": number, "prompt": name, **run})
            print(f"round {number}/{ROUNDS} {name}: {run['ms_per_step']} ms/step, "
                  f"{run['tokens_per_step']} tokens/step", flush=True)
        rounds.append(runs)
    result = {
        "schema": SCHEMA,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": _source_commit(),
        "profile": profile,
        "ced": ced,
        "model": model,
        "method": (f"{ROUNDS} rounds of the same {len(PROMPTS)} greedy {MAX_TOKENS}-token chat "
                   "requests, one at a time, thinking off; ms/step = first-to-last streamed "
                   "chunk time / MTP draft steps from the server's spec-decode metrics. "
                   "Round 1 is after start, later rounds are warm."),
        **summary(rounds),
        "runs": [run for round_runs in rounds for run in round_runs],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result
