# SPDX-License-Identifier: Apache-2.0
"""Doctor checks of the selected profile's pinned files: the CED projector, the
runtime overlays, and the expert ceilings of a full mutable expert cache."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

try:
    from tools import runtime_overlays
except ModuleNotFoundError:
    import runtime_overlays

GIB = 1024**3
# The safetensors format caps its JSON header at 100 MB.
SAFETENSORS_HEADER_LIMIT = 100_000_000


def load_profile() -> dict | None:
    """The selected profile.json (R9V_PROFILE_ROOT), or None when none is selected."""
    location = os.environ.get("R9V_PROFILE_ROOT")
    if not location:
        return None
    return json.loads((Path(location) / "profile.json").read_text(encoding="utf-8"))


def descriptor(repo_root: Path, profile: dict, key: str) -> dict:
    return json.loads((repo_root / profile["descriptors"][key]).read_text(encoding="utf-8"))


def _ced_switch() -> str:
    return os.environ.get("R9V_CED", "off")


def _pinned_projector(package: dict) -> dict | None:
    relative = os.environ.get(runtime_overlays.projector_setting(_ced_switch()))
    return next((artifact for artifact in package.get("artifacts", [])
                 if artifact.get("role") == "ced-projector" and artifact.get("path") == relative), None)


def ced_projector_vram(repo_root: Path, profile: dict | None) -> int:
    """Bytes the CED projector takes on each GPU once loaded: the pinned bf16 file,
    about half of it at int8; the quality projector is stored as int8 and loads as
    stored. 0 when CED is off or no projector is pinned."""
    if _ced_switch() == "off" or profile is None:
        return 0
    try:
        pinned = _pinned_projector(descriptor(repo_root, profile, "model_package"))
    except (OSError, KeyError, ValueError):
        return 0
    if pinned is None:
        return 0
    if _ced_switch() == "on" and os.environ.get("R9V_CED_PRECISION") == "int8":
        return pinned["bytes"] // 2
    return pinned["bytes"]


def projector_split(path: Path) -> tuple[int | None, str | None]:
    """The split a projector declares by its keys, read as the runtime reads it
    (layer.S..layer.N contiguous plus final), and by its metadata, if recorded."""
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        if size > SAFETENSORS_HEADER_LIMIT:
            raise ValueError(f"{path}: a {size}-byte safetensors header is not plausible")
        header = json.loads(stream.read(size))
    layers = sorted(int(key.split(".", 1)[1]) for key in header if key.startswith("layer."))
    contiguous = bool(layers) and layers == list(range(layers[0], layers[-1] + 1))
    keyed = layers[0] if contiguous and "final" in header else None
    return keyed, (header.get("__metadata__") or {}).get("split")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_ced_projector(reporter, repo_root: Path, profile: dict | None) -> None:
    """CED on or quality: the projector the runtime will load is the pinned file, of the
    profile's split."""
    # Without a model directory the model-package check already reports the missing install.
    switch = _ced_switch()
    if switch == "off" or profile is None or not os.environ.get("R9V_MODEL_DIR"):
        return
    fix = (f"Rerun setup with --ced {switch} to download the pinned projector and keep the "
           "profile's R9V_CED_* values, or start with --ced off. Never substitute another projector.")
    environment, problems = runtime_overlays.ced_environment(dict(os.environ))
    if problems:
        reporter.fail("ced-projector", "; ".join(problems), fix)
        return
    relative = os.environ[runtime_overlays.projector_setting(switch)]
    path = Path(os.environ["R9V_MODEL_DIR"]) / relative
    try:
        pinned = _pinned_projector(descriptor(repo_root, profile, "model_package"))
        if pinned is None:
            raise ValueError(f"the model package pins no ced-projector at {relative}")
        size = path.stat().st_size
        digest = _sha256(path) if size == pinned["bytes"] else None
        keyed, recorded = projector_split(path)
    except (OSError, KeyError, ValueError, struct.error) as error:
        reporter.fail("ced-projector", f"cannot check the CED projector: {error}", fix)
        return
    expected = profile.get("features", {}).get("ced", {}).get("split")
    if size != pinned["bytes"]:
        problems.append(f"{path} is {size} bytes; the package pins {pinned['bytes']}")
    elif digest != pinned["sha256"]:
        problems.append(f"{path} has sha256 {digest}; the package pins {pinned['sha256']}")
    if keyed is None:
        problems.append(f"{path} lacks contiguous layer.S..layer.N keys plus final")
    elif recorded is not None and recorded != str(keyed):
        problems.append(f"its keys start at layer {keyed} but its metadata records split {recorded}")
    if keyed is not None and keyed != expected:
        problems.append(f"it is a split-{keyed} projector; the profile runs split {expected}")
    if problems:
        reporter.fail("ced-projector", "; ".join(problems), fix)
        return
    precision = environment["R9V_CED_PRECISION"]
    vram = ced_projector_vram(repo_root, profile)
    reporter.passed(
        "ced-projector",
        f"CED {switch}: {relative} matches its pinned sha256 {digest[:12]}..., split {keyed}, "
        f"precision {precision}; it takes {vram / GIB:.2f} GiB on each GPU once loaded",
        sha256=digest, split=keyed, precision=precision, vram_bytes_per_gpu=vram,
    )


def check_runtime_overlays(reporter) -> None:
    """Every file the runtime descriptor mounts over its image matches its pinned SHA-256."""
    location = os.environ.get("R9V_RUNTIME_DESCRIPTOR")
    if not location:
        return
    path = Path(location)
    try:
        overlays = runtime_overlays.load(path)
        if overlays is None:
            return
        problems = runtime_overlays.verify(path, overlays)
    except (OSError, KeyError, TypeError, ValueError) as error:
        overlays, problems = None, [f"cannot read overlays: {error}"]
    if problems:
        reporter.fail(
            "runtime-overlays",
            f"{len(problems)} problem(s) in {path}: " + "; ".join(problems),
            "Restore the checkout (git status, then git checkout -- runtimes/) instead of "
            "editing an overlay; the launcher refuses to start until every file matches.",
        )
        return
    ced = _ced_switch()
    groups = runtime_overlays.CED_GROUPS.get(ced, ["always"])
    mounted = sorted(name for group in groups for name in overlays["mounts"].get(group, {}))
    reporter.passed(
        "runtime-overlays",
        f"all {len(overlays['sha256'])} overlay files match their pinned SHA-256; with CED "
        f"{ced} {len(mounted)} of them replace image files and the directory is mounted at "
        f"{overlays['directory_target']}",
        mounted=mounted,
    )


def _runtime_pins(runtime_path: Path, runtime: dict) -> list[int]:
    """Experts per layer each rank pins, from the pin list the runtime mounts."""
    overlays = runtime["overlays"]
    pins = json.loads((runtime_path.parent / overlays["directory"] / "full_mutable_pins.json").read_text())
    counts = pins["pinned_slots"]
    return [int(counts[str(rank)]) for rank in range(len(counts))]


def check_expert_limits(reporter, repo_root: Path, profile: dict | None) -> None:
    """A full mutable expert cache: R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK must equal the fixed
    placement's hot experts plus cache slots, which must hold what the runtime keeps resident."""
    relative = (profile or {}).get("descriptors", {}).get("placement")
    if not relative:
        return
    fix = ("Restore the profile's values (setup saves them from profile.env). A full mutable "
           "cache runs only its pinned placement, so these numbers cannot be tuned.")
    try:
        placement = json.loads((repo_root / relative).read_text(encoding="utf-8"))
        cache = placement.get("full_mutable_cache")
        if cache is None:
            return
        capacities = [int(value) for value in cache["capacities"]]
        hot = [int(value) for value in placement["hot_counts"]]
        slots = int(placement["cache_slots"])
        ranks = sorted(int(value) for value in placement["cache_ranks"])
        runtime = descriptor(repo_root, profile, "runtime")
        pinned = [int(value) for value in cache.get("pinned_slots", [0] * len(capacities))]
        pins = _runtime_pins(repo_root / profile["descriptors"]["runtime"], runtime) if any(pinned) else None
        configured = [int(value) for value in
                      os.environ.get("R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK", "").split(",") if value.strip()]
    except (OSError, KeyError, TypeError, ValueError) as error:
        reporter.fail("expert-limit-consistency", f"cannot read the full mutable cache limits: {error}", fix)
        return
    allocated = [count + (slots if rank in ranks else 0) for rank, count in enumerate(hot)]
    problems = []
    if (cache.get("runtime") != profile["runtime"]
            or runtime.get("capabilities", {}).get("full_mutable_expert_cache") is not True):
        problems.append(f"the placement's mutable cache is built for runtime {cache.get('runtime')}, "
                        f"not {profile['runtime']}")
    if len(capacities) != len(allocated) or any(c > a for c, a in zip(capacities, allocated)):
        problems.append(f"the runtime keeps {capacities} experts per layer resident, more than the "
                        f"{allocated} the placement allocates")
    if configured != allocated:
        problems.append(f"R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK={','.join(map(str, configured))} differs "
                        f"from the placement's hot experts plus cache slots, {allocated}")
    if pins is not None and pins != pinned:
        problems.append(f"the runtime's pin list pins {pins} experts per layer; the placement expects {pinned}")
    cache_env = (os.environ.get("R9V_TIERED_EXPERT_CACHE_SLOTS"), os.environ.get("R9V_TIERED_EXPERT_CACHE_RANKS"))
    if cache_env != (str(slots), ",".join(map(str, ranks))):
        problems.append(f"cache slots {cache_env[0]} on ranks {cache_env[1]} differ from the placement's "
                        f"{slots} on ranks {','.join(map(str, ranks))}")
    if problems:
        reporter.fail("expert-limit-consistency", "; ".join(problems), fix)
    else:
        reporter.passed(
            "expert-limit-consistency",
            f"expert ceilings {allocated} = hot experts {hot} + {slots} cache slots on rank(s) "
            f"{ranks}; the runtime's mutable cache keeps {capacities} of them resident per layer, "
            f"{pinned} of them pinned for good",
            ceilings=allocated, capacities=capacities, pinned=pinned,
        )
