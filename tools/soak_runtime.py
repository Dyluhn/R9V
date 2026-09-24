#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repeat synthetic requests while independently capturing runtime evidence."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools import capture_runtime as capture
    from tools import decode_speed
    from tools.support_bundle import collect
except ModuleNotFoundError:
    import capture_runtime as capture
    import decode_speed
    from support_bundle import collect


def request_once(
    port: int, model: str, repeats: int, max_tokens: int, sequence: int
) -> dict:
    # Vary the prefix on every request to avoid benchmarking only prefix-cache hits.
    prompt = (
        f"Trial {sequence}. "
        + ("Compare methods for checking computer hardware reliability. " * repeats)
        + "Write a detailed numbered analysis."
    )
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.monotonic()
    try:
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        data = response.read(1024 * 1024 + 1)
        if response.status != 200:
            return {"error": f"HTTP {response.status}"}
        if len(data) > 1024 * 1024:
            return {"error": "response exceeded 1 MiB"}
        parsed = json.loads(data)
        choices = parsed.get("choices", [])
        usage = parsed.get("usage", {})
        if not choices or choices[0].get("finish_reason") not in {"stop", "length"}:
            return {"error": "missing successful completion finish reason"}
        if (
            not isinstance(usage.get("completion_tokens"), int)
            or usage["completion_tokens"] <= 0
        ):
            return {"error": "missing positive completion token count"}
        if not choices[0].get("message", {}).get("content"):
            return {"error": "empty completion"}
        return {
            "seconds": round(time.monotonic() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage["completion_tokens"],
            "finish_reason": choices[0]["finish_reason"],
        }
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        http.client.HTTPException,
    ) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new evidence directory")
    parser.add_argument(
        "--container",
        default=os.environ.get("R9V_CONTAINER_NAME", "r9v-qwen38-flash-next"),
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("R9V_HOST_PORT", "8004"))
    )
    parser.add_argument(
        "--model", default=os.environ.get("R9V_SERVED_MODEL_NAME", "qwen3.8-flash-next")
    )
    parser.add_argument(
        "--duration", type=int, default=7200, help="seconds, maximum 24 hours"
    )
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=5,
        help="pause after each short/medium/long cycle",
    )
    parser.add_argument(
        "--prompt-repeats",
        default="8,128,2048",
        help="comma-separated corpus sizes, not token counts",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--decode-speed",
        type=Path,
        metavar="FILE",
        help="instead of a soak: measure decode ms/step after start and warm, save it to FILE (new)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sequence", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        repeats = [int(value) for value in args.prompt_repeats.split(",")]
        assert repeats and all(1 <= value <= 8192 for value in repeats)
        assert 1 <= args.duration <= 86400 and 1 <= args.port <= 65535
        assert 1 <= args.max_tokens <= 4096
        assert math.isfinite(args.request_timeout) and 1 <= args.request_timeout <= 3600
        assert math.isfinite(args.idle_seconds) and 0 <= args.idle_seconds <= 3600
        assert args.container and not args.container.startswith("-")
    except (ValueError, AssertionError):
        parser.error(
            "invalid duration, port, token count, corpus sizes, container, or timeout"
        )
    if args.worker:
        print(
            json.dumps(
                request_once(
                    args.port, args.model, repeats[0], args.max_tokens, args.sequence
                )
            )
        )
        return 0
    if args.decode_speed is not None:
        try:
            result = decode_speed.measure(args.port, args.model, args.decode_speed,
                                          os.environ.get("R9V_PROFILE_ID"), os.environ.get("R9V_CED"))
        except (OSError, RuntimeError, ValueError, KeyError, http.client.HTTPException) as error:
            print(f"Decode measurement failed: {error}")
            return 1
        print(f"warm decode median {result['warm_median_ms_per_step']} ms/step; saved to {args.decode_speed}")
        return 0
    if args.output is None:
        parser.error("--output is required")
    try:
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as error:
        print(f"Cannot create evidence directory: {error}")
        return 1
    monitor = None
    monitor_log = None
    status, count, error = "failed", 0, None
    started = time.monotonic()
    # Checkpoints are durable even if the machine loses power before the summary.
    fd = os.open(
        args.output / "requests.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    remaining = 32 * 1024 * 1024
    try:
        with os.fdopen(fd, "wb") as output:

            def record(value):
                nonlocal remaining
                value["timestamp"] = datetime.now(timezone.utc).isoformat()
                remaining = capture.write_record(output, value, remaining)

            record(
                {
                    "event": "started",
                    "duration": args.duration,
                    "prompt_repeats": repeats,
                    "max_tokens": args.max_tokens,
                    "request_timeout": args.request_timeout,
                }
            )
            monitor_log = (args.output / "collector.log").open("xb")
            monitor = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(capture.__file__).resolve()),
                    "--container",
                    args.container,
                    "--port",
                    str(args.port),
                    "--duration",
                    str(
                        min(90000, args.duration + math.ceil(args.request_timeout) + 10)
                    ),
                    "--stall-seconds",
                    str(math.ceil(args.request_timeout)),
                    "--interval",
                    "10",
                    "--output",
                    str(args.output / "timeline.jsonl"),
                ],
                stdout=monitor_log,
                stderr=monitor_log,
            )
            while time.monotonic() - started < args.duration:
                if monitor.poll() is not None:
                    raise RuntimeError(
                        "telemetry collector stopped before the soak completed"
                    )
                size = repeats[count % len(repeats)]
                record(
                    {
                        "event": "request_started",
                        "sequence": count,
                        "repeats": size,
                        "elapsed": round(time.monotonic() - started, 3),
                    }
                )
                result = capture.command_json(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--port",
                        str(args.port),
                        "--model",
                        args.model,
                        "--prompt-repeats",
                        str(size),
                        "--max-tokens",
                        str(args.max_tokens),
                        "--sequence",
                        str(count),
                    ],
                    timeout=args.request_timeout,
                )
                record({"event": "request_finished", "sequence": count, **result})
                if "error" in result:
                    raise RuntimeError(result["error"])
                count += 1
                if count % len(repeats) == 0:
                    time.sleep(
                        min(
                            args.idle_seconds,
                            max(0, args.duration - (time.monotonic() - started)),
                        )
                    )
            status = "passed" if count >= len(repeats) else "incomplete"
    except KeyboardInterrupt:
        status, error = "interrupted", "user interrupted the run"
    except (OSError, RuntimeError) as failure:
        error = str(failure)
    finally:
        if monitor is not None:
            import signal

            if monitor.poll() is not None:
                status, error = (
                    "failed",
                    error
                    or "telemetry collector exited before finalization; inspect collector.log",
                )
            monitor.send_signal(signal.SIGINT)
            try:
                monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if monitor_log is not None:
            monitor_log.close()
        timeline = args.output / "timeline.jsonl"
        try:
            records = [json.loads(line) for line in timeline.read_text().splitlines()]
            events = sorted(
                {event for record in records for event in record.get("events", [])}
            )
            if not records or events:
                status, error = (
                    "failed",
                    error or f"telemetry events: {events or ['no samples']}",
                )
        except (OSError, ValueError) as failure:
            status, error = "failed", error or f"telemetry unavailable: {failure}"
        try:
            bundle = collect(args.output / "support", args.container, args.port)
            if bundle and bundle.get("runtime_state", {}).get("status") != "running":
                status, error = (
                    "failed",
                    error or "container was not running at final evidence collection",
                )
        except (OSError, RuntimeError) as failure:
            status, error = "failed", f"support collection failed: {failure}"
        summary = {
            "schema": "r9v.soak.v1",
            "status": status,
            "completed_requests": count,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": error,
            "scope": "Synthetic sequential text completion transport/liveness test; not a correctness or full-context qualification.",
        }
        with (args.output / "summary.json").open("xb") as output:
            capture.write_record(output, summary, 16384)
    print(json.dumps(summary, indent=2))
    print(f"Evidence: {args.output}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
