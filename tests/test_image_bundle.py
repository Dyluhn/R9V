import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import image_bundle


def _manifest(tmp_path, data=b"abcdef", **overrides):
    digest = hashlib.sha256(data).hexdigest()
    value = {
        "schema": image_bundle.SCHEMA,
        "format": image_bundle.FORMAT,
        "image_ids": ["sha256:" + "a" * 64],
        "parts": [
            {
                "name": "part-000.gz",
                "bytes": len(data),
                "sha256": digest,
                "url": "https://github.com/acme/r9v/releases/download/r1/part-000.gz",
            }
        ],
        "bytes": len(data),
        "sha256": digest,
    }
    value.update(overrides)
    return value, data


def test_manifest_rejects_path_escape_duplicate_and_bool_size(tmp_path):
    value, _ = _manifest(tmp_path)
    for bad in ("../x", "part/a", "part-000.gz"):
        candidate = json.loads(json.dumps(value))
        candidate["parts"][0]["name"] = bad
        if bad == "part-000.gz":
            candidate["parts"].append(dict(candidate["parts"][0]))
        with pytest.raises(image_bundle.ImageBundleError):
            image_bundle.validate_manifest(candidate)
    candidate = json.loads(json.dumps(value))
    candidate["parts"][0]["bytes"] = True
    with pytest.raises(image_bundle.ImageBundleError):
        image_bundle.validate_manifest(candidate)


def test_manifest_rejects_unpinned_or_authenticated_url(tmp_path):
    value, _ = _manifest(tmp_path)
    for url in (
        "http://github.com/a/b/releases/download/r/p",
        "https://u:p@github.com/a/b/releases/download/r/p",
        "https://github.com/a/b/archive/r/p",
    ):
        value["parts"][0]["url"] = url
        with pytest.raises(image_bundle.ImageBundleError):
            image_bundle.validate_manifest(value)


def test_download_resume_and_hash_before_rename(tmp_path, monkeypatch):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz.part").write_bytes(data[:2])

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 2-5/6"}

        def read(self, n):
            chunk, self.body = self.body, b""
            return chunk

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    response = Response()
    response.body = data[2:]
    monkeypatch.setattr(
        image_bundle.urllib.request, "urlopen", lambda request, timeout=60: response
    )
    image_bundle.ensure_parts(value, cache)
    assert (cache / "part-000.gz").read_bytes() == data
    assert not (cache / "part-000.gz.part").exists()

    (cache / "part-000.gz").write_bytes(b"bad")
    with pytest.raises(image_bundle.ImageBundleError):
        image_bundle.verify_parts(value, cache)


def test_complete_partial_is_finalized_or_redownloaded(tmp_path, monkeypatch):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / "part-000.gz.part"
    partial.write_bytes(data)

    def unexpected(*args, **kwargs):
        raise AssertionError("complete valid partial must not request HTTP")

    monkeypatch.setattr(image_bundle.urllib.request, "urlopen", unexpected)
    image_bundle.ensure_parts(value, cache)
    assert (cache / "part-000.gz").read_bytes() == data
    partial.write_bytes(b"x" * len(data))

    class Response:
        status = 200
        headers = {}

        def read(self, n):
            chunk, self.body = self.body, b""
            return chunk

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    response = Response()
    response.body = data
    monkeypatch.setattr(
        image_bundle.urllib.request, "urlopen", lambda *a, **k: response
    )
    image_bundle.ensure_parts(value, cache, force_download=True)
    assert (cache / "part-000.gz").read_bytes() == data


def test_download_rejects_truncated_or_oversize_response(tmp_path, monkeypatch):
    value, _ = _manifest(tmp_path, data=b"abcdef")
    cache = tmp_path / "cache"

    class Response:
        status = 200
        headers = {}

        def __init__(self, body):
            self.body = body

        def read(self, n):
            chunk, self.body = self.body, b""
            return chunk

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        image_bundle.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b"short"),
    )
    with pytest.raises(image_bundle.ImageBundleError, match="before"):
        image_bundle.ensure_parts(value, cache)
    assert not (cache / "part-000.gz").exists()
    monkeypatch.setattr(
        image_bundle.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b"toolong!"),
    )
    with pytest.raises(image_bundle.ImageBundleError, match="exceeded"):
        image_bundle.ensure_parts(value, cache)
    assert (cache / "part-000.gz.part").stat().st_size <= value["parts"][0]["bytes"]


def test_download_rejects_wrong_resume_range_and_cache_symlink(tmp_path, monkeypatch):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz.part").write_bytes(data[:2])

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 0-3/6"}

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        image_bundle.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    with pytest.raises(image_bundle.ImageBundleError, match="Content-Range"):
        image_bundle.ensure_parts(value, cache)
    (cache / "part-000.gz.part").unlink()
    (cache / "part-000.gz").symlink_to(tmp_path / "outside")
    with pytest.raises(image_bundle.ImageBundleError, match="regular file"):
        image_bundle.ensure_parts(value, cache)


def test_load_streams_verified_parts_and_returns_exact_id(tmp_path):
    value, data = _manifest(tmp_path, data=b"docker-save-payload")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text("""#!/bin/sh
if [ \"$1 $2\" = \"image load\" ]; then cat > \"$FAKE_CAPTURE\"; exit 0; fi
if [ \"$1 $2\" = \"image inspect\" ]; then echo \"$FAKE_ID\"; exit 0; fi
exit 1
""")
    script.chmod(0o755)
    expected = value["image_ids"][0]
    import os

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FAKE_CAPTURE": str(tmp_path / "capture"),
        "FAKE_ID": expected,
    }
    result = subprocess.run(
        [str(script), "image", "load"], env=env, input=data, capture_output=True
    )
    assert result.returncode == 0
    # Exercise the real loader with the fake executable and inspect response.
    old = os.environ.copy()
    os.environ.update(env)
    try:
        assert (
            image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=5)
            == expected
        )
    finally:
        os.environ.clear()
        os.environ.update(old)
    assert (tmp_path / "capture").read_bytes() == data


def test_load_rejects_wrong_image_and_timeout(tmp_path):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text(
        '#!/bin/sh\nif [ "$2" = load ]; then sleep 2; else echo sha256-'
        + "b" * 64
        + "; fi\n"
    )
    script.chmod(0o755)
    with pytest.raises(image_bundle.ImageBundleError, match="timed out"):
        image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=0.01)


def test_load_rejects_wrong_image_and_docker_failure(tmp_path):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text(
        '#!/bin/sh\nif [ "$2" = load ]; then cat >/dev/null; exit ${FAIL_LOAD:-0}; fi\necho sha256-'
        + "b" * 64
        + "\n"
    )
    script.chmod(0o755)
    with pytest.raises(image_bundle.ImageBundleError, match="mismatch"):
        image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=5)
    import os

    old = os.environ.get("FAIL_LOAD")
    os.environ["FAIL_LOAD"] = "7"
    try:
        with pytest.raises(image_bundle.ImageBundleError, match="exit code 7"):
            image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=5)
    finally:
        if old is None:
            os.environ.pop("FAIL_LOAD", None)
        else:
            os.environ["FAIL_LOAD"] = old


def test_load_timeout_bounds_child_that_never_reads(tmp_path):
    value, data = _manifest(tmp_path, data=b"x" * 2_000_000)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text(
        '#!/bin/sh\nif [ "$2" = load ]; then sleep 60; else echo sha256:'
        + "a" * 64
        + "; fi\n"
    )
    script.chmod(0o755)
    with pytest.raises(image_bundle.ImageBundleError, match="timed out"):
        image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=0.05)


class _Body:
    """A urlopen response that returns one body, then EOF."""

    status = 200
    headers: dict = {}

    def __init__(self, body):
        self.body = body

    def read(self, n):
        chunk, self.body = self.body, b""
        return chunk

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _two_part_manifest(tmp_path, first=b"docker-save-", second=b"payload"):
    value, _ = _manifest(tmp_path)
    part = dict(value["parts"][0])
    value["parts"] = [
        {**part, "name": "part-000.gz", "bytes": len(first),
         "sha256": hashlib.sha256(first).hexdigest()},
        {**part, "name": "part-001.gz", "bytes": len(second),
         "sha256": hashlib.sha256(second).hexdigest(),
         "url": part["url"].replace("part-000", "part-001")},
    ]
    value["bytes"] = len(first + second)
    value["sha256"] = hashlib.sha256(first + second).hexdigest()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(value))
    return path, {"part-000.gz": first, "part-001.gz": second}


def _serve(monkeypatch, bodies):
    monkeypatch.setattr(
        image_bundle.urllib.request, "urlopen",
        lambda request, timeout=60: _Body(bodies[request.full_url.rsplit("/", 1)[1]]),
    )


def _forbid_docker(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("--verify-only must not run docker")

    monkeypatch.setattr(image_bundle.subprocess, "Popen", refuse)
    monkeypatch.setattr(image_bundle.subprocess, "run", refuse)


def test_verify_only_downloads_and_verifies_every_part_without_docker(tmp_path, monkeypatch, capsys):
    manifest, bodies = _two_part_manifest(tmp_path)
    _serve(monkeypatch, bodies)
    _forbid_docker(monkeypatch)
    cache = tmp_path / "cache"

    code = image_bundle.main([str(manifest), "--cache-dir", str(cache), "--verify-only"])

    assert code == 0
    assert (cache / "part-000.gz").read_bytes() == b"docker-save-"
    assert (cache / "part-001.gz").read_bytes() == b"payload"
    out = capsys.readouterr().out
    assert "Downloading image bundle part 2/2: part-001.gz" in out
    assert "PASS 2 parts and the reassembled archive" in out
    assert "not loaded into Docker" in out


def test_verify_only_rejects_a_corrupt_part_and_keeps_nothing(tmp_path, monkeypatch, capsys):
    manifest, bodies = _two_part_manifest(tmp_path)
    _serve(monkeypatch, {**bodies, "part-001.gz": b"PAYLOAD"})
    _forbid_docker(monkeypatch)
    cache = tmp_path / "cache"

    code = image_bundle.main([str(manifest), "--cache-dir", str(cache), "--verify-only"])

    assert code == 1
    assert "downloaded part failed verification: part-001.gz" in capsys.readouterr().err
    assert not (cache / "part-001.gz").exists()
    assert not (cache / "part-001.gz.part").exists()


def test_verify_only_rejects_parts_whose_reassembled_archive_differs(tmp_path, monkeypatch, capsys):
    manifest, bodies = _two_part_manifest(tmp_path)
    value = json.loads(manifest.read_text())
    value["sha256"] = hashlib.sha256(b"payloaddocker-save-").hexdigest()  # parts swapped
    manifest.write_text(json.dumps(value))
    _serve(monkeypatch, bodies)
    _forbid_docker(monkeypatch)

    code = image_bundle.main([str(manifest), "--cache-dir", str(tmp_path / "cache"), "--verify-only"])

    assert code == 1
    assert "concatenated image bundle failed verification" in capsys.readouterr().err


def test_command_without_verify_only_loads_the_verified_parts(tmp_path, monkeypatch, capsys):
    manifest, bodies = _two_part_manifest(tmp_path)
    image = json.loads(manifest.read_text())["image_ids"][0]
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, body in bodies.items():
        (cache / name).write_bytes(body)
    capture = tmp_path / "loaded"
    docker = tmp_path / "bin/docker"
    docker.parent.mkdir()
    docker.write_text(
        "#!/bin/sh\n"
        f'if [ "$2" = load ]; then cat > "{capture}"; exit 0; fi\n'
        f'[ -f "{capture}" ] && echo {image} && exit 0\n'
        "exit 1\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{docker.parent}:{os.environ['PATH']}")

    code = image_bundle.main([str(manifest), "--cache-dir", str(cache)])

    assert code == 0
    assert capture.read_bytes() == b"docker-save-payload"
    assert f"PASS Docker has {image}" in capsys.readouterr().out
