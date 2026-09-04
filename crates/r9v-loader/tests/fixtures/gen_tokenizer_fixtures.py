#!/usr/bin/env python3
"""Build deterministic A2.9 tokenizer fixtures (card A2.9).

Creates three small GGUF files with path-covering synthetic vocabularies
(built with pinned gguf-py, recorded in `meta.json`) plus a shared corpus:

  fixture-bpe.gguf    model "gpt2", pre "gpt-2"
  fixture-spm.gguf    model "llama" (SentencePiece merge scores)
  fixture-bert.gguf   model "bert" (WordPiece)

Also writes `corpus.json` (the parity inputs) and placeholder
`golden-*.json` files. Golden expected ids are produced from the pinned
llama.cpp oracle by `gen_goldens.sh` (same directory) and committed.

Deterministic: fixed token order, fixed scores, sorted JSON, no timestamps.
"""

import json
import sys
from pathlib import Path

try:
    import gguf
except ImportError:
    sys.exit("gguf-py is required: pip install gguf==0.19.0")

OUT = Path(__file__).resolve().parent
try:
    from importlib.metadata import version as pkg_version

    GGUF_VERSION = pkg_version("gguf")
except Exception:
    GGUF_VERSION = "unknown"

# GPT-2 byte map (mirrors unicode_byte_to_utf8): printable ASCII plus
# U+00A1..U+00AC/U+00AE..U+00FF map to themselves; the other 68 bytes map
# to U+0100... in byte order.
def gpt2_byte_char(byte: int) -> str:
    b = byte
    if 0x21 <= b <= 0x7E or 0xA1 <= b <= 0xAC or 0xAE <= b <= 0xFF:
        return chr(b)
    n = 0
    for candidate in range(256):
        printable = (
            0x21 <= candidate <= 0x7E
            or 0xA1 <= candidate <= 0xAC
            or 0xAE <= candidate <= 0xFF
        )
        if not printable:
            if candidate == b:
                break
            n += 1
    return chr(256 + n)


def byte_encode_text(text: bytes) -> str:
    return "".join(gpt2_byte_char(b) for b in text)


def write_gguf(path: Path, kv: list) -> None:
    w = gguf.GGUFWriter(str(path), "llama")
    for key, kind, value in kv:
        if kind == "str":
            w.add_string(key, value)
        elif kind == "u32":
            w.add_uint32(key, value)
        elif kind == "bool":
            w.add_bool(key, value)
        elif kind == "str_array":
            w.add_array(key, value)
        elif kind == "i32_array":
            w.add_array(key, value)
        elif kind == "f32_array":
            w.add_array(key, value)
        else:
            raise AssertionError(f"bad kind {kind}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()


def build_bpe() -> Path:
    # 256 byte tokens + merges + specials.
    tokens = [byte_encode_text(bytes([b])) for b in range(256)]
    types = [1] * 256
    merges: list[str] = []

    def add_merge(left: str, right: str) -> str:
        merges.append(f"{left} {right}")
        piece = left + right
        if piece not in tokens:
            tokens.append(piece)
            types.append(1)
        return piece

    # Word ladders exercising multi-level merges.
    h = byte_encode_text(b"H")
    e = byte_encode_text(b"e")
    l = byte_encode_text(b"l")
    o = byte_encode_text(b"o")
    he = add_merge(h, e)
    hel = add_merge(he, l)
    hell = add_merge(hel, l)
    hello = add_merge(hell, o)
    sp = byte_encode_text(b" ")
    w = byte_encode_text(b"w")
    r = byte_encode_text(b"r")
    d = byte_encode_text(b"d")
    spw = add_merge(sp, w)
    spwo = add_merge(spw, o)
    wor = add_merge(w, o)
    worl = add_merge(wor, r)
    world = add_merge(worl, d)
    spwor = add_merge(sp, wor)
    spworl = add_merge(spwor, l)
    spworld = add_merge(spworl, d)
    t = byte_encode_text(b"t")
    s = byte_encode_text(b"s")
    i = byte_encode_text(b"i")
    n = byte_encode_text(b"n")
    te = add_merge(t, e)
    tes = add_merge(te, s)
    test = add_merge(tes, t)
    ing = add_merge(add_merge(i, n), byte_encode_text(b"g"))
    # Multi-byte merges: é = C3 A9, 中 = E4 B8 AD.
    e_acute = byte_encode_text("é".encode())
    add_merge(byte_encode_text(bytes([0xC3])), byte_encode_text(bytes([0xA9])))
    zhong = byte_encode_text("中".encode())
    add_merge(
        byte_encode_text(bytes([0xE4])), byte_encode_text(bytes([0xB8]))
    )
    add_merge(byte_encode_text(bytes([0xE4, 0xB8])), byte_encode_text(bytes([0xAD])))

    bos = len(tokens)
    tokens.append("<|endoftext|>")
    types.append(3)
    userdef = len(tokens)
    tokens.append("<|userdef|>")
    types.append(4)

    path = OUT / "fixture-bpe.gguf"
    write_gguf(
        path,
        [
            ("tokenizer.ggml.model", "str", "gpt2"),
            ("tokenizer.ggml.pre", "str", "gpt-2"),
            ("tokenizer.ggml.tokens", "str_array", tokens),
            ("tokenizer.ggml.scores", "f32_array", [0.0] * len(tokens)),
            ("tokenizer.ggml.token_type", "i32_array", types),
            ("tokenizer.ggml.merges", "str_array", merges),
            ("tokenizer.ggml.bos_token_id", "u32", bos),
            ("tokenizer.ggml.eos_token_id", "u32", bos),
            ("tokenizer.ggml.add_bos_token", "bool", False),
            ("tokenizer.ggml.add_eos_token", "bool", False),
            ("tokenizer.chat_template", "str", "bpe says {{ messages[0]['content'] }}"),
        ],
    )
    return path


def build_spm() -> Path:
    tokens = ["<unk>", "<s>", "</s>"]
    types = [2, 3, 3]
    scores = [0.0, 0.0, 0.0]
    for b in range(256):
        tokens.append(f"<0x{b:02X}>")
        types.append(6)
        scores.append(0.0)

    def add(text: str, score: float, typ: int = 1) -> int:
        tokens.append(text)
        types.append(typ)
        scores.append(score)
        return len(tokens) - 1

    # Word pieces with scores chosen so greedy score-order (not longest
    # match) decides: "helloworld" prefers high-score "hello"+"world".
    add("▁", -1.0)
    add("▁Hello", 10.0)
    add("▁hello", 9.0)
    add("Hello", 8.0)
    add("Hell", 7.0)
    add("hell", 6.5)
    add("o", 5.0)
    add("▁world", 9.5)
    add("world", 8.5)
    add("wor", 7.5)
    add("ld", 6.0)
    add("▁test", 9.0)
    add("▁ing", 8.0)
    add("ing", 7.0)
    add("▁é", 4.0)
    add("▁中", 4.0)
    add("中", 3.0)
    add("▁a", 5.5)
    add("a", 5.0)
    add("<zhi>", 0.0, 4)

    path = OUT / "fixture-spm.gguf"
    write_gguf(
        path,
        [
            ("tokenizer.ggml.model", "str", "llama"),
            ("tokenizer.ggml.tokens", "str_array", tokens),
            ("tokenizer.ggml.scores", "f32_array", scores),
            ("tokenizer.ggml.token_type", "i32_array", types),
            ("tokenizer.ggml.bos_token_id", "u32", 1),
            ("tokenizer.ggml.eos_token_id", "u32", 2),
            ("tokenizer.ggml.unknown_token_id", "u32", 0),
            ("tokenizer.ggml.add_bos_token", "bool", True),
            ("tokenizer.ggml.add_eos_token", "bool", False),
        ],
    )
    return path


def build_bert() -> Path:
    # ids: PAD 0 UNK 1 CLS 2 SEP 3 MASK 4, then pieces.
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    types = [3, 2, 3, 3, 3]
    pieces = [
        "hello", "world", "test", "ing", "caf", "##e", "##s", "##ing",
        "a", "i", ",", ".", "?", "!", "中", "文", "e", "##f", "co",
        "##ffee", "##e", "z",
    ]
    for piece in pieces:
        if piece not in tokens:
            tokens.append(piece)
            types.append(1)
    # Remove accidental duplicate "##e".
    seen = set()
    uniq_tokens, uniq_types = [], []
    for text, typ in zip(tokens, types):
        if text in seen:
            continue
        seen.add(text)
        uniq_tokens.append(text)
        uniq_types.append(typ)
    tokens, types = uniq_tokens, uniq_types

    path = OUT / "fixture-bert.gguf"
    write_gguf(
        path,
        [
            ("tokenizer.ggml.model", "str", "bert"),
            ("tokenizer.ggml.tokens", "str_array", tokens),
            ("tokenizer.ggml.scores", "f32_array", [0.0] * len(tokens)),
            ("tokenizer.ggml.token_type", "i32_array", types),
            ("tokenizer.ggml.bos_token_id", "u32", 2),
            ("tokenizer.ggml.eos_token_id", "u32", 3),
            ("tokenizer.ggml.unknown_token_id", "u32", 1),
            ("tokenizer.ggml.seperator_token_id", "u32", 3),
            ("tokenizer.ggml.padding_token_id", "u32", 0),
            ("tokenizer.ggml.mask_token_id", "u32", 4),
            ("tokenizer.ggml.add_bos_token", "bool", True),
            ("tokenizer.ggml.add_eos_token", "bool", False),
            ("tokenizer.ggml.add_sep_token", "bool", True),
        ],
    )
    return path


CORPUS = [
    "",
    " ",
    "  ",
    "Hello world",
    " Hello world",
    "Hello, world!",
    "don't stop",
    "'Twas the night",
    "abc 123 !?!",
    "trailing space ",
    "  leading spaces",
    "line1\nline2\ttabbed",
    "caf\u00e9 \u00c4pfel na\u00efve",
    "\u4e2d\u6587\u6d4b\u8bd5",
    "Hello\u4e2d\u6587",
    "\U0001f680 rocket \U0001f601",
    "3 33 333 3333 33333",
    "3.3 3..3 test-ing",
    "<|endoftext|> split <|userdef|> here",
    "a" * 64,
    "He He He",
]


def main() -> None:
    bpe = build_bpe()
    spm = build_spm()
    bert = build_bert()
    (OUT / "corpus.json").write_text(
        json.dumps({"inputs": CORPUS}, indent=2, ensure_ascii=False) + "\n"
    )
    meta = {
        "gguf_py": GGUF_VERSION,
        "files": [p.name for p in (bpe, spm, bert)],
        "note": "golden-*.json produced by gen_goldens.sh from the pinned llama.cpp oracle",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {bpe.name}, {spm.name}, {bert.name}, corpus.json, meta.json")


if __name__ == "__main__":
    main()
