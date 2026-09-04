#!/usr/bin/env bash
# Generate golden-*.json from the pinned llama.cpp oracle (card A2.9).
#
# Usage: gen_goldens.sh [LLAMA_CPP_DIR]
# Default LLAMA_CPP_DIR is ~/projects/inference/llama.cpp. Uses the
# build-vulkan binaries and records the exact commit in every golden file.
# Deterministic: inputs come from corpus.json; outputs are parsed from
# `llama-tokenize --ids` ([1, 2, 3] format).
set -euo pipefail

LLAMA_CPP="${1:-$HOME/projects/inference/llama.cpp}"
BIN="$LLAMA_CPP/build-vulkan/bin/llama-tokenize"
FIX="$(dirname "$0")"
COMMIT="$(git -C "$LLAMA_CPP" rev-parse HEAD)"
GGUF_PY="$(python3 -c 'from importlib.metadata import version; print(version("gguf"))')"

export LD_LIBRARY_PATH="$LLAMA_CPP/build-vulkan/bin:${LD_LIBRARY_PATH:-}"

if [ ! -x "$BIN" ]; then
  echo "oracle binary not found: $BIN" >&2
  exit 1
fi

for SPEC in "bpe:fixture-bpe.gguf" "spm:fixture-spm.gguf" "bert:fixture-bert.gguf"; do
  NAME="${SPEC%%:*}"
  FILE="${SPEC##*:}"
  echo "== $NAME ($FILE, oracle $COMMIT)"
  python3 - "$FIX/$FILE" "$BIN" "$COMMIT" "$GGUF_PY" "$FIX/corpus.json" "$FIX/golden-$NAME.json" <<'EOF'
import json
import subprocess
import sys

gguf_path, binary, commit, gguf_py, corpus_path, out_path = sys.argv[1:7]
corpus = json.load(open(corpus_path))["inputs"]
ld_path = ":".join([p for p in __import__("os").environ.get("LD_LIBRARY_PATH", "").split(":") if p])
cases = []
for i, text in enumerate(corpus):
    prompt_path = f"/tmp/a29_prompt_{i}.txt"
    open(prompt_path, "w").write(text)
    for add_special, parse_special, flags in [
        (True, True, []),
        (False, True, ["--no-bos"]),
        (True, False, ["--no-parse-special"]),
        (False, False, ["--no-bos", "--no-parse-special"]),
    ]:
        proc = subprocess.run(
            [binary, "-m", gguf_path, "-f", prompt_path, "--ids", "--log-disable"] + flags,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.exit(f"oracle failed on case {i} {flags}: {proc.stderr[-500:]}")
        line = proc.stdout.strip().splitlines()[-1]
        ids = json.loads(line)
        cases.append(
            {
                "input": text,
                "add_special": add_special,
                "parse_special": parse_special,
                "ids": ids,
            }
        )
golden = {
    "oracle": {
        "repo": "llama.cpp",
        "commit": commit,
        "binary": "llama-tokenize",
        "gguf_py": gguf_py,
    },
    "cases": cases,
}
json.dump(golden, open(out_path, "w"), ensure_ascii=False, indent=1, sort_keys=False)
open(out_path, "a").write("\n")
print(f"wrote {out_path}: {len(cases)} cases")
EOF
done
