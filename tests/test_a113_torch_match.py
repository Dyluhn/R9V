# SPDX-License-Identifier: Apache-2.0
"""`r9v eval` vs torch on the A1.13 ~30M Llama-family fixture (card A1.13).

Covers spec 8 §8 (family reference match) and spec 13 §12 (`verify-arch`):
a deterministic F16 GGUF plus its torch forward (both under
`tools/r9v-quant/tests/`) is compared against `r9v eval` (A1.12, CPU T0 tier)
on the 64 fixed token sequences. Gates are the spec 1 §6.1 *logits*
criteria — top-1 agreement ≥ 99.9% and mean per-token KL ≤ 1e-3 — loaded
from the data table below, never as literals at the assert site.

Requires torch, numpy, gguf and xxhash (see requirements-ci.txt): a lane
missing any of them fails at collection (no dependency skips anywhere in
this file). Needs a Rust toolchain to build the release `r9v` binary and
to run the `r9v-format` proof; without cargo those tests fail (the
comparison must run for real, never skip). The fixture GGUF carries no
`r9v.*` keys, so it is a standard GGUF per spec 2 §6: proven at the
gguf-py level here and at the repo-code level by
`crates/r9v-t0/tests/a113_standard_fixture.rs` (also ties all 75 tensors
byte-for-byte to the Rust synthetic generation). That Rust test is
`#[ignore]`d by default so plain `cargo test` reports it ignored rather
than a false pass; the last test below drives it with `--ignored --exact`
and the fixture environment, so it must execute for real here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

# Collection-time dependency gate: each import must succeed, so a lane
# missing numpy/torch/gguf/xxhash fails collection instead of skipping
# (no `pytest.importorskip` anywhere in this file). The pinned API names
# below keep the `torch`/`xxhash` imports load-bearing under lint; they
# arrive functionally through the `torch_forward`/`gen_fixture` helpers.
import gguf
import numpy as np
import pytest
import torch
import xxhash

_REQUIRED_DEPS = (gguf.GGUFReader, np.ndarray, torch.Tensor, xxhash.xxh3_64)

ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name: str):
    """Loads a helper module from tools/r9v-quant/tests by path (no sys.path edit)."""
    location = ROOT / "tools" / "r9v-quant" / "tests" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, location)
    assert spec is not None and spec.loader is not None, f"cannot load {location}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen_fixture = _load_helper("gen_fixture")
torch_forward = _load_helper("torch_forward")

# Spec 1 §6.1 logits criteria, loaded as data (CONVENTIONS.md §4.3).
SPEC_1_6_1_LOGITS = {
    "top1_min": 0.999,
    "kl_max": 1e-3,
}
# Strict per-element envelope used for diagnostics only: spec 1 §6.1 states
# per-element tolerances for op paths and top-1/KL for logits, so the
# envelope below is reported, never gated.
DIAG_PER_ELEMENT = {"abs": 2e-3, "rel": 1e-2}

FIXTURE_DIR = ROOT / "tools" / "r9v-quant" / "tests" / "fixtures" / "a113"
R9V_BIN = ROOT / "target" / "release" / "r9v"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_r9v_binary() -> Path:
    """Builds the current release r9v binary (Spec 14 §10, Card A1.12).

    A missing cargo must fail, never skip: the core comparison has no
    fallback lane, and a skip would report green without testing anything.
    """
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.fail(
            "cargo not on PATH so target/release/r9v cannot be built; "
            "the A1.13 comparison must run for real, never skip"
        )
    build = subprocess.run(
        [cargo, "build", "--release", "--locked", "-p", "r9v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert build.returncode == 0, f"cargo build -p r9v failed:\n{build.stderr[-4000:]}"
    assert R9V_BIN.is_file(), "cargo build succeeded but target/release/r9v is missing"
    return R9V_BIN


def _prompts_aggregate(prompts_dir: Path, count: int) -> str:
    """Recomputes the manifest prompt hash from the prompt files on disk."""
    found = sorted(prompts_dir.glob("seq_*.txt"))
    assert len(found) == count, f"expected {count} prompt files in {prompts_dir}, found {len(found)}"
    seqs = []
    for seq_idx in range(count):
        text = (prompts_dir / f"seq_{seq_idx:02d}.txt").read_text(encoding="ascii")
        seqs.append(text.split())
    return hashlib.sha256("".join(" ".join(s) for s in seqs).encode("ascii")).hexdigest()


def _ensure_fixture() -> dict:
    """Generates (or validates the cache of) the deterministic fixture.

    A cached fixture is trusted only after its model spec, generator
    identity, GGUF/model.json bytes, and prompt set all check out; every
    run additionally regenerates to a temp dir and requires identical
    output, so determinism is proven, not assumed (Spec 1 App. B).
    """
    cached_manifest = FIXTURE_DIR / "manifest.json"
    manifest = None
    if cached_manifest.is_file():
        try:
            candidate = json.loads(cached_manifest.read_text(encoding="ascii"))
            cache_valid = (
                candidate["params"] == gen_fixture.FIXTURE_PARAMS
                and candidate.get("sources") == gen_fixture.source_digest()
                and candidate["sequences"] == gen_fixture.N_SEQUENCES
                and all(
                    (FIXTURE_DIR / name).is_file()
                    and _sha256(FIXTURE_DIR / name) == digest
                    for name, digest in candidate["files"].items()
                )
                and _prompts_aggregate(
                    FIXTURE_DIR / "prompts", candidate["sequences"]
                )
                == candidate["prompts_sha256"]
            )
            if cache_valid:
                manifest = candidate
        except (AssertionError, KeyError, OSError, TypeError, ValueError):
            manifest = None
    # A stale or corrupt generated cache is never a human cleanup gate. The
    # deterministic generator replaces it, and the temp-dir proof below then
    # verifies that the replacement is complete and reproducible.
    if manifest is None:
        manifest = gen_fixture.generate(FIXTURE_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        regen = gen_fixture.generate(Path(tmp))
        assert regen["total_params"] == manifest["total_params"]
        assert _sha256(Path(tmp) / "model.gguf") == manifest["files"]["model.gguf"], (
            "GGUF regeneration is not byte-identical"
        )
        assert (Path(tmp) / "model.json").read_bytes() == (FIXTURE_DIR / "model.json").read_bytes(), (
            "model.json regeneration is not byte-identical"
        )
        assert regen["prompts_sha256"] == manifest["prompts_sha256"], (
            "prompt regeneration is not identical"
        )
    return manifest


def _read_logits_npy(path: Path, rows: int, vocab: int) -> np.ndarray:
    """Reads the `.npy` logits written by `r9v eval` (Spec 14 §10, Card A1.12)."""
    raw = path.read_bytes()
    assert raw[:6] == b"\x93NUMPY" and raw[6:8] == b"\x01\x00", f"{path} not npy v1.0"
    (header_len,) = struct.unpack("<H", raw[8:10])
    header = raw[10 : 10 + header_len].decode("ascii")
    assert "'<f4'" in header and "False" in header, f"{path} not <f4 C-order"
    assert f"({rows},{vocab})" in header.replace(" ", ""), f"{path} wrong shape"
    data = np.frombuffer(raw[10 + header_len :], dtype=np.float32)
    assert data.size == rows * vocab, (data.size, rows, vocab)
    return data.reshape(rows, vocab).astype(np.float64)


def _log_softmax_stable(logits: np.ndarray) -> np.ndarray:
    shift = logits.max(axis=-1, keepdims=True)
    return (logits - shift) - np.log(np.exp(logits - shift).sum(axis=-1, keepdims=True))


def _kl_per_token(ref_logits: np.ndarray, actual_logits: np.ndarray) -> np.ndarray:
    """KL(P_ref || P_actual) per row, all in float64 (spec 1 §6.1)."""
    log_ref = _log_softmax_stable(ref_logits)
    log_actual = _log_softmax_stable(actual_logits)
    return (np.exp(log_ref) * (log_ref - log_actual)).sum(axis=-1)


def test_a113_rng_mirror_matches_r9v_common_vectors() -> None:
    """The weight-RNG mirror reproduces r9v-common's Xoshiro256++ stream.

    Vectors from crates/r9v-common/src/rng.rs `reference_algorithm_raw_state`;
    if the Rust stream ever changes, the fixture weights (and the failure
    message here) say so instead of failing downstream opaquely.
    """
    rng = gen_fixture.SeededRng.from_state([1, 2, 3, 4])
    assert [rng.next_u64() for _ in range(5)] == [
        0x2800001,
        0x3800067,
        0xCC00003800067,
        0xCC201994400B2,
        0x8012A2019AC433CD,
    ]


def test_a113_torch_match_logits_within_spec_1_6_1() -> None:
    manifest = _ensure_fixture()
    binary = _ensure_r9v_binary()
    params = manifest["params"]
    vocab = params["vocab"]

    assert gen_fixture.PARAM_LO <= manifest["total_params"] <= gen_fixture.PARAM_HI, (
        f"fixture is {manifest['total_params']} params, not ~30M"
    )
    assert manifest["sequences"] == 64, "spec 13 §12 fixes 64 prompts"

    weights = torch_forward.load_weights(FIXTURE_DIR / "model.gguf")
    out_dir = FIXTURE_DIR / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    top1_hit = 0
    top1_total = 0
    kl_sum = 0.0
    kl_positions = 0
    worst_abs = 0.0
    diag_violations = 0
    diag_total = 0

    for seq_idx in range(64):
        tokens_path = FIXTURE_DIR / "prompts" / f"seq_{seq_idx:02d}.txt"
        ids = [int(piece) for piece in tokens_path.read_text(encoding="ascii").split()]
        assert len(ids) == gen_fixture.SEQ_LEN, (seq_idx, len(ids))

        torch_logits = torch_forward.forward(weights, ids, params).numpy().astype(np.float64)

        # DECISION(A1.13): this milestone drives `r9v eval` through the SI-63
        # model.json synthetic vehicle instead of the fixture GGUF itself.
        # A2.6 is not a card dependency, and metadata parsing alone,
        # whenever integrated, is insufficient: payload materialization plus
        # the loader-to-T0 executor bridge are A2.7/A2.8 seams, so `eval`
        # cannot yet consume the GGUF directly and SI-63 model.json remains
        # the milestone vehicle. The GGUF weights and synthetic
        # regeneration are bit-identical (proven by the Rust fingerprint
        # test below), so
        # the comparison still covers the checkpoint bytes via the torch
        # side; direct GGUF engine execution supersedes this vehicle when
        # those stages land. Rejected hand-rolling those later pipeline
        # stages into A1.13. Applies to both eval call sites in this file.
        npy_path = out_dir / f"seq_{seq_idx:02d}.logits.npy"
        run = subprocess.run(
            [
                str(binary),
                "eval",
                "--logits",
                "--model",
                str(FIXTURE_DIR / "model.json"),
                "--tokens",
                str(tokens_path),
                "--out",
                str(npy_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert run.returncode == 0, f"r9v eval seq {seq_idx} failed: {run.stderr[-2000:]}"
        r9v_logits = _read_logits_npy(npy_path, len(ids), vocab)

        top1_hit += int((torch_logits.argmax(axis=-1) == r9v_logits.argmax(axis=-1)).sum())
        top1_total += len(ids)
        kl = _kl_per_token(torch_logits, r9v_logits)
        kl_sum += float(kl.sum())
        kl_positions += len(ids)

        abs_diff = np.abs(torch_logits - r9v_logits)
        worst_abs = max(worst_abs, float(abs_diff.max()))
        envelope = DIAG_PER_ELEMENT["abs"] + DIAG_PER_ELEMENT["rel"] * np.abs(r9v_logits)
        diag_violations += int((abs_diff > envelope).sum())
        diag_total += abs_diff.size

    top1 = top1_hit / top1_total
    mean_kl = kl_sum / kl_positions
    print(
        f"\nA1.13: top1={top1:.6f} (min {SPEC_1_6_1_LOGITS['top1_min']}) "
        f"mean_kl={mean_kl:.3e} (max {SPEC_1_6_1_LOGITS['kl_max']:.0e}) "
        f"diag worst_abs={worst_abs:.4f} envelope_violations={diag_violations}/{diag_total}"
    )
    assert top1 >= SPEC_1_6_1_LOGITS["top1_min"], f"top-1 {top1} < 99.9% (spec 1 §6.1)"
    assert mean_kl <= SPEC_1_6_1_LOGITS["kl_max"], f"mean KL {mean_kl} > 1e-3 (spec 1 §6.1)"


def test_a113_eval_is_deterministic_across_runs() -> None:
    _ensure_fixture()
    binary = _ensure_r9v_binary()
    out_dir = FIXTURE_DIR / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq_idx in (0, 17, 63):
        tokens_path = FIXTURE_DIR / "prompts" / f"seq_{seq_idx:02d}.txt"
        first = out_dir / f"det_{seq_idx:02d}a.npy"
        second = out_dir / f"det_{seq_idx:02d}b.npy"
        for target in (first, second):
            run = subprocess.run(
                [
                    str(binary),
                    "eval",
                    "--logits",
                    "--model",
                    str(FIXTURE_DIR / "model.json"),
                    "--tokens",
                    str(tokens_path),
                    "--out",
                    str(target),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            assert run.returncode == 0, f"r9v eval failed: {run.stderr[-2000:]}"
        assert first.read_bytes() == second.read_bytes(), f"seq {seq_idx} not bit-identical"


def test_a113_gguf_fixture_binds_llama_keys() -> None:
    manifest = _ensure_fixture()
    params = manifest["params"]
    reader = gguf.GGUFReader(str(FIXTURE_DIR / "model.gguf"))

    fields = {}
    for field in reader.fields.values():
        fields[field.name] = field.contents()

    def field_value(name: str):
        assert name in fields, f"missing metadata {name}"
        return fields[name]

    assert str(field_value("general.architecture")) == "llama"
    assert int(field_value("llama.block_count")) == params["layers"]
    assert int(field_value("llama.embedding_length")) == params["dim"]
    assert int(field_value("llama.feed_forward_length")) == params["ff"]
    assert int(field_value("llama.attention.head_count")) == params["heads"]
    assert int(field_value("llama.attention.head_count_kv")) == params["kv_heads"]
    # Standard GGUF per spec 2 §6: no reserved `r9v.*` keys (provenance lives
    # in manifest.json, never in the checkpoint). The prompt count is a
    # manifest fact, asserted there.
    assert not [name for name in fields if name.startswith("r9v.")], (
        "standard GGUF must carry no r9v.* metadata keys"
    )
    assert manifest["sequences"] == gen_fixture.N_SEQUENCES == 64

    names = [reader.get_tensor(i).name for i in range(len(reader.tensors))]
    assert len(names) == 3 + 9 * params["layers"], names
    assert names[0] == "token_embd.weight"
    assert names[-2:] == ["output_norm.weight", "output.weight"]
    assert "blk.0.attn_q.weight" in names and f"blk.{params['layers'] - 1}.ffn_down.weight" in names

    embed = next(
        reader.get_tensor(i).data
        for i in range(len(reader.tensors))
        if reader.get_tensor(i).name == "token_embd.weight"
    )
    assert tuple(int(d) for d in embed.shape) == (params["vocab"], params["dim"])
    assert str(embed.dtype) == "float16"


def test_a113_fixture_is_standard_gguf_per_r9v_format() -> None:
    """Repo-code proof: `r9v-format` parses the fixture as a standard GGUF.

    Runs `crates/r9v-t0/tests/a113_standard_fixture.rs` against the
    generated checkpoint: `GgufFile::parse` must succeed with no
    native-format validation errors, `is_standard_gguf()` must hold, and
    all 75 tensors must match the Rust synthetic generation by production
    edge/name binding, then canonical llama name, shape, dtype, and exact
    value bits. The Rust test is `#[ignore]`d by default so plain cargo
    reports it ignored rather than a false pass; it is driven here with
    `--ignored --exact` and the fixture environment, so it must execute
    for real. Like the comparison, this needs cargo and fails (never
    skips) without it.
    """
    _ensure_fixture()
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.fail(
            "cargo not on PATH; cannot run the r9v-format standard-GGUF proof"
        )
    assert cargo is not None
    env = dict(
        os.environ,
        R9V_A113_GGUF=str(FIXTURE_DIR / "model.gguf"),
        R9V_A113_MODEL_JSON=str(FIXTURE_DIR / "model.json"),
    )
    run = subprocess.run(
        [
            cargo,
            "test",
            "--locked",
            "-p",
            "r9v-t0",
            "--test",
            "a113_standard_fixture",
            "--",
            "--ignored",
            "--exact",
            "a113_fixture_parses_as_standard_gguf_with_all_75_synthetic_tensors_byte_identical",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert run.returncode == 0, (
        f"r9v-format standard-GGUF proof failed:\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
    )
    assert "test result: ok" in run.stdout, (
        f"fingerprint test did not report ok:\n{run.stdout[-3000:]}"
    )
