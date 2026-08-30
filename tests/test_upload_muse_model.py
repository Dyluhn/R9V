# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from tools.upload_muse_model import build_uploads


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "packages/models/muse-glimmer-30b/v1-v12"


def test_publication_mapping_is_portable_and_complete() -> None:
    publication_text = (PACKAGE_DIR / "publication.json").read_text(encoding="utf-8")
    assert "".join(("/", "var", "/")) not in publication_text
    assert "".join(("/", "home", "/")) not in publication_text
    publication = json.loads(publication_text)
    package = json.loads((PACKAGE_DIR / "package.json").read_text(encoding="utf-8"))
    assert publication["repository"] == package["distribution"]["repository"]
    assert package["distribution"] == {
        "status": "published",
        "repository": "Dyluhn/Muse-Glimmer-30B-R9V-V1",
        "revision": "093f8ced7a8e2308b0f597084ebdbfa5f6614f75",
        "note": (
            "All 10 files, including the optional projector and DFlash "
            "sidecar, were remotely size- and SHA256-verified at this "
            "immutable revision. V1 is the public release name; V12 is "
            "the internal research lineage."
        ),
    }
    metadata_destinations = {
        item["destination"] for item in publication["metadata"]
    }
    assert set(publication["payload_sources"]) == {
        artifact["path"]
        for artifact in package["artifacts"]
        if artifact["path"] not in metadata_destinations
    }
    for item in publication["metadata"]:
        text = (PACKAGE_DIR / item["source"]).read_text(encoding="utf-8")
        assert "".join(("/", "var", "/")) not in text
        assert "".join(("/", "home", "/")) not in text
        assert "".join(("select", "24")) not in text.lower()


def test_card_frontmatter_and_portable_links() -> None:
    card = (PACKAGE_DIR / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: apache-2.0\n")
    for target in re.findall(r"\]\(([^)]+)\)", card):
        if "://" not in target:
            assert (PACKAGE_DIR / target).is_file(), target


def test_license_matches_pinned_meta_license_bytes() -> None:
    license_bytes = (PACKAGE_DIR / "LICENSE").read_bytes()
    assert len(license_bytes) == 11_358
    assert hashlib.sha256(license_bytes).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def test_required_upload_uses_canonical_public_destination() -> None:
    repository, uploads = build_uploads(Path("/payload/canonical-v12.gguf"))
    assert repository == "Dyluhn/Muse-Glimmer-30B-R9V-V1"
    by_destination = {item.destination: item for item in uploads}
    target = by_destination["target/Muse-Glimmer-30B-R9V-V1.gguf"]
    assert target.source == Path("/payload/canonical-v12.gguf")
    assert target.expected_bytes == 24_554_611_392
    assert "vision/mmproj-kquant.gguf" not in by_destination
    assert "draft/dflash-kquant.gguf" not in by_destination


def test_optional_upload_requires_sidecar_root() -> None:
    with pytest.raises(ValueError, match="--sidecar-root"):
        build_uploads(Path("canonical.gguf"), include_optional=True)


def test_optional_upload_maps_both_sidecars_without_copying() -> None:
    _, uploads = build_uploads(
        Path("canonical.gguf"),
        sidecar_root=Path("sidecars"),
        include_optional=True,
    )
    by_destination = {item.destination: item for item in uploads}
    assert by_destination["vision/mmproj-kquant.gguf"].source == Path(
        "sidecars/mmproj-kquant.gguf"
    )
    assert by_destination["draft/dflash-kquant.gguf"].source == Path(
        "sidecars/dflash-kquant.gguf"
    )
