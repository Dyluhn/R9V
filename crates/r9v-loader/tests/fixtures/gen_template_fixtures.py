#!/usr/bin/env python3
"""Build A2.9 chat-template fixtures (card A2.9).

Copies three reference chat templates (provenance in `templates.json`),
then renders each with Jinja2 (version recorded) over a fixed context
matrix to produce `golden-templates.json`.

Reference templates are rendered verbatim; golden outputs are the Jinja2
renderings. The Rust parity test asserts byte-identical rendering.

Deterministic: fixed contexts, sorted keys, no timestamps.
"""

import hashlib
import json
from datetime import date, timezone
from pathlib import Path

from jinja2 import Environment

OUT = Path(__file__).resolve().parent

SOURCES = {
    "llama3": "https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/resolve/main/tokenizer_config.json",
    "qwen25": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json",
    "mistral": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer_config.json",
}

CONTEXTS = {
    "plain_user": {
        "messages": [{"role": "user", "content": "Hello world"}],
        "add_generation_prompt": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
    "no_generation_prompt": {
        "messages": [{"role": "user", "content": "Hello world"}],
        "add_generation_prompt": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
    "system_multi": {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks!"},
        ],
        "add_generation_prompt": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
    "with_tools": {
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "add_generation_prompt": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    },
    "tool_roundtrip": {
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        ],
        "add_generation_prompt": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    },
}


def main() -> None:
    import jinja2

    templates = {}
    for name, url in SOURCES.items():
        path = OUT / f"template-{name}.jinja"
        text = path.read_text()
        templates[name] = {
            "file": path.name,
            "source": url,
            "fetched": date.today().isoformat(),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "jinja2": jinja2.__version__,
        }
    (OUT / "templates.json").write_text(json.dumps(templates, indent=2) + "\n")

    # Match minja's default chat configuration (trim_blocks and
    # lstrip_blocks on; one trailing newline stripped).
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    goldens = {"jinja2": jinja2.__version__, "cases": []}
    for name in SOURCES:
        text = (OUT / f"template-{name}.jinja").read_text()
        for ctx_name, ctx in CONTEXTS.items():
            try:
                rendered = env.from_string(text).render(**ctx)
                status = "ok"
            except Exception as e:  # noqa: BLE001 - golden records the failure
                rendered = f"{type(e).__name__}: {e}"
                status = "error"
            goldens["cases"].append(
                {
                    "template": name,
                    "context": ctx_name,
                    "status": status,
                    "output": rendered,
                }
            )
    (OUT / "golden-templates.json").write_text(
        json.dumps(goldens, indent=1, ensure_ascii=False) + "\n"
    )
    ok = sum(1 for c in goldens["cases"] if c["status"] == "ok")
    print(f"templates: {len(templates)}, cases: {len(goldens['cases'])}, ok: {ok}")


if __name__ == "__main__":
    main()
