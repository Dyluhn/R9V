#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate and verify NumPy .npy v1.0 golden test fixtures for r9v (Card A1.12, Spec 14 §10).

Provenance and External Oracle:
- External oracle: NumPy 2.4.6 (`numpy.save`).
- Emitted format: NumPy .npy format version 1.0 (magic b"\x93NUMPY\x01\x00").
- Descr: '<f4' (32-bit little-endian IEEE-754 float).
- Order: row-major C order ('fortran_order': False).
- Alignment: 64-byte total header alignment (magic + version + 2-byte header len + header dict),
  padded with ASCII spaces (0x20) BEFORE the terminating newline (0x0A).
- Fixtures committed as hexadecimal text files (`.hex`) under `crates/r9v/tests/fixtures/r9v/`:
  1. `golden_f32_3x8.hex`: 2D array of shape (3, 8), values: [i * 0.125 - 1.5 for i in range(24)]
  2. `golden_f32_1d.hex`: 1D array of shape (7,), values: [i * 1.5 - 3.0 for i in range(7)]
     (validates trailing comma in 1-tuple shape representation "(7,)")

Usage:
  python3 crates/r9v/tests/fixtures/r9v/gen_golden_npy.py          # regenerate fixtures
  python3 crates/r9v/tests/fixtures/r9v/gen_golden_npy.py --check  # assert bit-identical regeneration
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np

EXPECTED_NUMPY_VERSION = "2.4.6"
FIXTURE_DIR = Path(__file__).resolve().parent


def generate_case_2d() -> bytes:
    shape = (3, 8)
    data = np.array([i * 0.125 - 1.5 for i in range(24)], dtype=np.float32).reshape(shape)
    buf = io.BytesIO()
    np.save(buf, data)
    return buf.getvalue()


def generate_case_1d() -> bytes:
    shape = (7,)
    data = np.array([i * 1.5 - 3.0 for i in range(7)], dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, data)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed fixtures match NumPy oracle bit-identically",
    )
    args = parser.parse_args()

    cases = [
        ("golden_f32_3x8.hex", generate_case_2d()),
        ("golden_f32_1d.hex", generate_case_1d()),
    ]

    for name, raw_bytes in cases:
        fixture_path = FIXTURE_DIR / name
        hex_text = raw_bytes.hex() + "\n"

        if args.check:
            if not fixture_path.exists():
                print(f"FAIL: fixture {fixture_path} does not exist", file=sys.stderr)
                return 1
            committed_text = fixture_path.read_text(encoding="ascii")
            if committed_text != hex_text:
                print(f"FAIL: fixture {name} differs from NumPy oracle", file=sys.stderr)
                return 1
            print(f"PASS: {name} matches NumPy oracle bit-identically ({len(raw_bytes)} bytes)")
        else:
            fixture_path.write_text(hex_text, encoding="ascii")
            print(f"Wrote {fixture_path} ({len(raw_bytes)} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
