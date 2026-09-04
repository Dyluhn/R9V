#!/usr/bin/env python3
"""Generate Rust Unicode tables for r9v-loader from llama.cpp's unicode-data.cpp.

The tables replicate llama.cpp's `unicode_cpt_flags_from_cpt`, lowercase /
uppercase maps, and lossy single-codepoint NFD map bit-for-bit so
pre-tokenization, BERT normalization, and byte-collapse behave identically.

Usage:
    python3 gen_unicode_tables.py --unicode-data /path/to/unicode-data.cpp \
        --commit <llama.cpp commit> --out unicode_tables.rs

The `--commit` value is recorded in the generated file header. Deterministic:
parses C++ initializer lists in file order and emits sorted Rust tables.
"""

import argparse
import re
from pathlib import Path


def parse_pairs(text, start_marker):
    """Parse `{0xAAAA, 0xBBBB},` pairs in the block starting at start_marker."""
    idx = text.index(start_marker)
    block = text[idx:]
    # Cut at the closing "};" of the initializer list.
    end = block.index("};")
    block = block[:end]
    return re.findall(r"\{\s*(0x[0-9A-Fa-f]+)\s*,\s*(0x[0-9A-Fa-f]+)\s*\}", block)


def parse_triples(text, start_marker):
    idx = text.index(start_marker)
    block = text[idx:]
    end = block.index("};")
    block = block[:end]
    return re.findall(
        r"\{\s*(0x[0-9A-Fa-f]+)\s*,\s*(0x[0-9A-Fa-f]+)\s*,\s*(0x[0-9A-Fa-f]+)\s*\}",
        block,
    )


def parse_set(text, start_marker):
    idx = text.index(start_marker)
    block = text[idx:]
    end = block.index("};")
    block = block[:end]
    return re.findall(r"(0x[0-9A-Fa-f]+)", block)


def fmt_table(pairs):
    return ",\n".join(f"    (0x{int(a, 16):06X}, 0x{int(b, 16):04X})" for a, b in pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unicode-data", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = Path(args.unicode_data).read_text()

    flags = parse_pairs(text, "unicode_ranges_flags = {")
    whitespace = parse_set(text, "unicode_set_whitespace = {")
    lowercase = parse_pairs(text, "unicode_map_lowercase = {")
    uppercase = parse_pairs(text, "unicode_map_uppercase = {")
    nfd = parse_triples(text, "unicode_ranges_nfd = {")

    assert flags, "no flag ranges parsed"
    assert whitespace, "no whitespace set parsed"
    assert lowercase, "no lowercase map parsed"
    assert uppercase, "no uppercase map parsed"
    assert nfd, "no NFD ranges parsed"

    ws_vals = sorted(int(v, 16) for v in whitespace)
    ws_body = ",\n".join(f"    0x{v:06X}" for v in ws_vals)
    nfd_body = ",\n".join(
        f"    (0x{int(a, 16):06X}, 0x{int(b, 16):06X}, 0x{int(c, 16):06X})"
        for a, b, c in nfd
    )

    out = f"""// SPDX-License-Identifier: Apache-2.0
//! Unicode property tables derived from llama.cpp `src/unicode-data.cpp`
//! at commit {args.commit} (card A2.9).
//!
//! Regenerate with `tests/fixtures/gen_unicode_tables.py --unicode-data
//! <path> --commit <hash> --out src/unicode_tables.rs`. The tables replicate
//! `unicode_cpt_flags_from_cpt`, `unicode_tolower`, `unicode_toupper`, and
//! the lossy single-codepoint NFD map bit-for-bit. Flag bit values match
//! llama.cpp `unicode.h`: UNDEFINED=0x1, NUMBER=0x2, LETTER=0x4,
//! SEPARATOR=0x8, MARK=0x10, PUNCTUATION=0x20, SYMBOL=0x40, CONTROL=0x80.
//! DO NOT HAND-EDIT: this file is generated.

// DECISION(A2.9): tables are mechanically derived from the pinned llama.cpp
// commit rather than re-derived from UCD text so flag behavior matches the
// reference bit-for-bit; rejected re-running gen-unicode-data.py (tracks UCD
// latest, so output drifts). Spec 9 §7 is silent on Unicode versions.

/// (range start, flags); range end is the next start minus one.
pub(crate) const FLAG_RANGES: &[(u32, u16)] = &[
{fmt_table(flags)}
];

/// Codepoints with the whitespace flag.
pub(crate) const WHITESPACE_SET: &[u32] = &[
{ws_body}
];

/// (codepoint, lowercase) map, sorted by codepoint.
pub(crate) const LOWERCASE_MAP: &[(u32, u32)] = &[
{fmt_table(lowercase)}
];

/// (codepoint, uppercase) map, sorted by codepoint. No current caller;
/// kept as reference data for future pre-tokenizers.
#[allow(dead_code)]
pub(crate) const UPPERCASE_MAP: &[(u32, u32)] = &[
{fmt_table(uppercase)}
];

/// (range start, range last, lossy NFD base) map.
pub(crate) const NFD_RANGES: &[(u32, u32, u32)] = &[
{nfd_body}
];
"""
    Path(args.out).write_text(out)
    print(
        f"wrote {args.out}: {len(flags)} flag ranges, {len(ws_vals)} ws, "
        f"{len(lowercase)} lower, {len(uppercase)} upper, {len(nfd)} nfd"
    )


if __name__ == "__main__":
    main()
