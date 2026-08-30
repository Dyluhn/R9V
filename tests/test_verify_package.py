from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "tools" / "verify_package.py"


def descriptor(path: Path, data: bytes) -> Path:
    payload = {
        "schema": "r9v.model-package.v1",
        "id": "test-package",
        "model": "test",
        "quant": "test",
        "license": "Apache-2.0",
        "artifacts": [
            {
                "role": "target",
                "path": "target.bin",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "required": True,
            },
            {
                "role": "optional",
                "path": "optional.bin",
                "bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
                "required": False,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_verify(
    package: Path, model_dir: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(VERIFY), str(package), "--model-dir", str(model_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_size_and_hash_verification(tmp_path: Path) -> None:
    data = b"R9V-PAYLOAD"
    package = descriptor(tmp_path / "package.json", data)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "target.bin").write_bytes(data)
    result = run_verify(package, model_dir, "--hash")
    assert result.returncode == 0, result.stdout
    assert "1 artifacts verified" in result.stdout
    assert "1 optional absent" in result.stdout


def test_missing_required_artifact_fails(tmp_path: Path) -> None:
    package = descriptor(tmp_path / "package.json", b"x")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    result = run_verify(package, model_dir)
    assert result.returncode == 1
    assert "missing:" in result.stdout


def test_hash_mismatch_fails(tmp_path: Path) -> None:
    package = descriptor(tmp_path / "package.json", b"expected")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "target.bin").write_bytes(b"EXPECTEd")
    result = run_verify(package, model_dir, "--hash")
    assert result.returncode == 1
    assert "SHA256 mismatch" in result.stdout
