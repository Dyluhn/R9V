#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resumable installation and bounded startup for the Qwen Flash Next profile."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    from tools.expert_budget import headroom_bytes
    from tools.package_sources import artifact_source, selected_artifacts
    from tools.profile_doctor import SCHEDULER_PROBE, discover_kfd_gpus, parse_preemptions
    from tools.profile_state import default_state_dir, validate_state_profile
    from tools.verify_package import _sha256
except ModuleNotFoundError:
    from expert_budget import headroom_bytes
    from package_sources import artifact_source, selected_artifacts
    from profile_doctor import SCHEDULER_PROBE, discover_kfd_gpus, parse_preemptions
    from profile_state import default_state_dir, validate_state_profile
    from verify_package import _sha256

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/qwen38-flash-next/dual-r9700/profile.json"
PLE_BYTES = 28_800_138_240


def selected_profile():
    location = os.environ.get('R9V_PROFILE_ROOT')
    path = Path(location) / 'profile.json' if location else PROFILE
    path = path.resolve()
    if location and not path.is_relative_to((ROOT / 'profiles').resolve()):
        raise ValueError('Selected profile must be inside the repository profiles directory')
    data = json.loads(path.read_text())
    expected = os.environ.get('R9V_PROFILE_ID')
    if expected and data.get('id') != expected:
        raise ValueError('Selected profile descriptor does not match R9V_PROFILE_ID')
    return path, data


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity(path):
    info = path.stat()
    return [str(path.resolve()), info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns]


def ensure_artifact(artifact, model_dir, receipt, persist, download, full_hash=False):
    path = (model_dir / artifact["path"]).resolve()
    if not path.is_relative_to(model_dir.resolve()):
        raise ValueError("artifact escapes model directory")
    key = artifact["path"]
    old = receipt.get(key, {})
    if not isinstance(old, dict):
        old = {}
    if (not full_hash and path.is_file() and old.get("identity") == identity(path)
            and old.get("sha256") == artifact["sha256"]):
        print(f"Reusing previously hash-verified file: {key}", flush=True)
        return
    if not path.is_file() or path.stat().st_size != artifact["bytes"]:
        download(key)
    if not path.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("downloaded artifact escapes model directory")
    before = identity(path)
    print(f"Verifying {key}", flush=True)
    if before[3] != artifact["bytes"] or _sha256(path) != artifact["sha256"]:
        raise ValueError(f"Integrity check failed: {path}; repair this file and retry")
    if identity(path) != before:
        raise ValueError(f"File changed during verification: {path}")
    receipt[key] = {"identity": before, "sha256": artifact["sha256"]}
    persist()


def reuse_artifact(artifact, model_dir, reuse_dir, verified_source=None):
    """Hard-link an already verified auxiliary payload without duplicating weights."""
    source = verified_source or reusable_source(artifact, model_dir, reuse_dir)
    if source is None:
        return False
    destination = (model_dir / artifact['path']).resolve()
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        raise ValueError(f'Cannot reuse {artifact["path"]} by hard link; choose model storage on the same filesystem or omit --reuse-from: {error}') from error
    print(f'Reused verified model asset by hard link: {artifact["path"]}', flush=True)
    return True


def reusable_source(artifact, model_dir, reuse_dir):
    """Return a verified source path for storage accounting and hard-linking."""
    if reuse_dir is None:
        return None
    reuse_root = Path(reuse_dir).expanduser().resolve()
    source = (reuse_root / artifact['path']).resolve()
    destination = (model_dir / artifact['path']).resolve()
    if not source.is_relative_to(reuse_root) or not destination.is_relative_to(model_dir.resolve()):
        raise ValueError('Reuse artifact escapes its model directory')
    if destination.exists() or not source.is_file() or source.stat().st_size != artifact['bytes']:
        return None
    before = identity(source)
    if _sha256(source) != artifact['sha256']:
        return None
    if identity(source) != before:
        raise ValueError('Reuse source changed during verification')
    return source


def run(command, *, env=None, capture=False, timeout=None):
    print("Running: " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), env=env, check=True,
                          text=True, stdout=subprocess.PIPE if capture else None,
                          timeout=timeout)


def select_devices(bdfs=None):
    gpus = discover_kfd_gpus()
    eligible = []
    for index, gpu in enumerate(gpus):
        if gpu.gfx_target != 120001:
            continue
        try:
            vram = int((Path('/sys/bus/pci/devices') / gpu.bdf /
                        'mem_info_vram_total').read_text())
        except (OSError, ValueError):
            continue
        if vram >= 31 * 2**30:
            eligible.append((index, gpu))
    if bdfs:
        wanted = [b.strip().lower() for b in bdfs.split(',')]
        by_bdf = {g.bdf: (i, g) for i, g in eligible}
        if len(wanted) != 2 or len(set(wanted)) != 2 or any(b not in by_bdf for b in wanted):
            raise ValueError("--gpu-bdfs must select two distinct 32 GiB gfx1201 GPUs")
        eligible = [by_bdf[b] for b in wanted]
    if len(eligible) != 2:
        raise ValueError("Need exactly two 32 GiB gfx1201 GPUs; use --gpu-bdfs to select a pair")
    nodes = [Path('/dev/kfd')]
    for _, gpu in eligible:
        if gpu.render_minor is None:
            raise ValueError(f"Cannot resolve render device for {gpu.bdf}")
        nodes.append(Path('/dev/dri') / f'renderD{gpu.render_minor}')
    for node in nodes:
        if not os.access(node, os.R_OK | os.W_OK):
            raise ValueError(f"Device access required: {node}")
    return {"R9V_VISIBLE_DEVICES": ','.join(str(i) for i, _ in eligible),
            "R9V_EXPECTED_GPU_BDFS": ','.join(g.bdf for _, g in eligible)}


def profile_settings():
    """Freeze the effective profile defaults and explicit R9V overrides together."""
    profile_path, profile = selected_profile()
    profile_env = ROOT / profile['legacy_env'] if profile.get('legacy_env') else profile_path.parent / 'profile.env'
    result = run(['bash', '-c', 'set -a; source "$1"; env -0', 'r9v-setup', profile_env],
                 capture=True, timeout=10)
    excluded = {'R9V_CONFIG_FILE', 'R9V_PROFILE', 'R9V_PROFILE_ID', 'R9V_PROFILE_ROOT',
                'R9V_REPO_ROOT', 'R9V_SYS_ROOT', 'R9V_PROC_ROOT', 'R9V_BASE_IMAGE',
                'R9V_RUNTIME_ONLY', 'R9V_MAX_JOBS', 'R9V_VLLM_VERSION'}
    return {key: value for entry in result.stdout.split('\0') if '=' in entry
            for key, value in [entry.split('=', 1)]
            if key.startswith('R9V_') and key not in excluded}


def container_user_args():
    """Root in rootless Docker maps to the daemon owner, not host root."""
    options = json.loads(run(['docker', 'info', '--format', '{{json .SecurityOptions}}'],
                             capture=True, timeout=30).stdout)
    if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
        raise ValueError('Docker did not report valid security options')
    if 'name=rootless' in options:
        return ['--user', '0:0']
    result = ['--user', f'{os.getuid()}:{os.getgid()}']
    if 'name=userns' in options:
        # Rootful daemon remapping would otherwise turn the host UID into a
        # subordinate UID with no write access to the bind-mounted data directory.
        result += ['--userns', 'host']
    return result


def fixed_placement(profile):
    """True when the profile pins one expert placement that must never be re-planned."""
    relative = profile.get('descriptors', {}).get('placement')
    return bool(relative) and json.loads((ROOT / relative).read_text()).get('fixed') is True


def default_timeout(profile):
    """Seconds start waits for readiness. A full mutable expert cache compiles cold on first start."""
    relative = profile.get('descriptors', {}).get('placement')
    placement = json.loads((ROOT / relative).read_text()) if relative else {}
    return 2400 if 'full_mutable_cache' in placement else 900


def check_profile_options(args, profile):
    """Refuse options this profile cannot honor, before any side effect."""
    replanning = [option for option, name in (('--headroom', 'headroom'), ('--calibration', 'calibration'),
                                              ('--expert-catalog', 'expert_catalog'))
                  if getattr(args, name, None)]
    if replanning and fixed_placement(profile):
        raise ValueError(f"{', '.join(replanning)} would re-plan the expert placement, but "
                         f"{profile['id']} uses a fixed placement that its mutable expert cache "
                         "requires; its free-VRAM target is part of the profile")
    if getattr(args, 'ced', None):
        runtime = json.loads((ROOT / profile['descriptors']['runtime']).read_text())
        group = 'ced-quality' if args.ced == 'quality' else 'ced'
        if group not in runtime.get('overlays', {}).get('mounts', {}):
            mode = 'CED quality' if args.ced == 'quality' else 'CED'
            raise ValueError(f"--ced {args.ced}: runtime {profile['runtime']} of {profile['id']} has no {mode}")


def check_ced_quality_installed(state):
    """CED quality loads a projector that setup downloads only when quality is chosen."""
    relative = state['config'].get('R9V_CED_QUALITY_PROJECTOR_REL')
    if not relative or relative not in state.get('artifacts', {}):
        raise ValueError("--ced quality needs its own projector (about 1.8 GiB), which setup has not "
                         "downloaded yet. Run setup again with --ced quality; it fetches only that file.")


def qualification_identity(env, fixed):
    """What a first-start qualification receipt vouches for, or None if none is needed."""
    if fixed:
        identity = {
            'placement_sha256': hashlib.sha256(Path(env['R9V_EXPERT_MANIFEST_PATH']).read_bytes()).hexdigest(),
            'runtime_sha256': hashlib.sha256(Path(env['R9V_RUNTIME_DESCRIPTOR']).read_bytes()).hexdigest(),
            'image': env['R9V_IMAGE'],
        }
    elif env.get('R9V_PLACEMENT_PLAN'):
        try:
            from tools.plan_experts import digest
        except ModuleNotFoundError:
            from plan_experts import digest
        identity = {'placement_sha256': digest(json.loads(Path(env['R9V_PLACEMENT_PLAN']).read_text()))}
    else:
        return None
    if env.get('R9V_CED'):
        identity['ced'] = env['R9V_CED']
    return identity


def setup(args, state, state_path):
    _, profile = selected_profile()
    check_profile_options(args, profile)
    descriptor = ROOT / profile['descriptors']['model_package']
    package = json.loads(descriptor.read_text())
    runtime_path = ROOT / profile['descriptors']['runtime']
    runtime = json.loads(runtime_path.read_text())
    image = args.image or runtime.get('distribution', {}).get('image')
    distribution = profile.get('distribution', {})
    bundle = None
    if not image and not args.build and distribution.get('image_bundle'):
        bundle = (ROOT / distribution['image_bundle']).resolve()
        if not bundle.is_relative_to(ROOT.resolve()):
            raise ValueError('Image bundle descriptor escapes repository')
        image = distribution['image_id']
    if not image and not args.build:
        raise ValueError("No published image is configured yet. Supply --image REGISTRY/IMAGE@sha256:DIGEST, "
                         "--image LOCAL_IMAGE --local-image, or explicitly opt into --build.")
    if image and image.startswith('-'):
        raise ValueError('Image must not start with a dash')
    if image and bundle is None and not args.local_image and not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', image):
        raise ValueError("Remote images must be pinned with @sha256:<64 lowercase hex digits>")
    if not args.accept_model_license:
        raise ValueError("Read the model license and supply --accept-model-license")
    if os.environ.get('R9V_CONFIG_FILE'):
        raise ValueError("Unset R9V_CONFIG_FILE for setup; this flow saves its own machine configuration")
    run(['docker', 'info', '--format', '{{.DockerRootDir}}'], capture=True, timeout=30)
    config = profile_settings()
    config['R9V_RUNTIME_DESCRIPTOR'] = str(runtime_path.resolve())
    if getattr(args, 'ced', None):
        config['R9V_CED'] = args.ced
    if profile.get('id'):
        config['R9V_PROFILE_ID'] = profile['id']
        state['profile_id'] = profile['id']
    if package.get('id'):
        config['R9V_MODEL_PACKAGE'] = package['id']
        config['R9V_MODEL_PACKAGE_SHA256'] = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    if getattr(args, 'headroom', None):
        headroom_bytes(args.headroom, 2)
        config['R9V_MIN_FREE_VRAM_GIB_BY_RANK'] = args.headroom
        config['R9V_HEADROOM_SELECTION'] = '1'
    if getattr(args, 'calibration', None):
        config['R9V_CALIBRATION_PATH'] = str(args.calibration.resolve())
    if getattr(args, 'expert_catalog', None):
        config['R9V_EXPERT_CATALOG_PATH'] = str(args.expert_catalog.resolve())
    config.update(select_devices(args.gpu_bdfs))
    run([ROOT / 'scripts/profile-doctor.sh', '--host-only'],
        env={**os.environ, **config, 'R9V_RUNTIME_PREBUILT': '1'})
    model = Path(args.model_dir or os.environ.get('R9V_MODEL_DIR', '')).expanduser().resolve()
    if not args.model_dir and not os.environ.get('R9V_MODEL_DIR'):
        raise ValueError("Supply --model-dir to choose the model storage destination")
    data = Path(args.data_dir).expanduser().resolve() if args.data_dir else model / 'r9v-data'
    model.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    artifacts = selected_artifacts(package, config)
    for artifact in artifacts:  # refuse before any download when a missing file has no published source
        if not (model / artifact['path']).is_file():
            try:
                artifact_source(package, artifact)
            except ValueError as error:
                note = artifact.get('distribution', {}).get('note', str(error))
                raise ValueError(f"Cannot download {artifact['path']}: {note}") from error
    missing = sum(
        a['bytes'] for a in artifacts
        if ((model / a['path']).is_file() and (model / a['path']).stat().st_size == a['bytes'])
        or reusable_source(a, model, getattr(args, 'reuse_from', None)) is not None
    )
    missing = sum(a['bytes'] for a in artifacts) - missing
    ple = (Path(args.ple_path).expanduser().resolve() if args.ple_path
           else data / 'per_layer_token_embd.iq4_nl.bin')
    ple.parent.mkdir(parents=True, exist_ok=True)
    if ple.exists() and (not ple.is_file() or ple.stat().st_size != PLE_BYTES):
        raise ValueError(f'Unexpected PLE payload: {ple}; choose a new path or repair the file')
    derived = 0 if ple.is_file() else PLE_BYTES
    print(f"Model destination: {model}; missing/unfinished files up to {missing / 2**30:.2f} GiB")
    print(f"PLE destination: {ple}; new payload {derived / 2**30:.2f} GiB")
    print("Runtime image/build storage is additional in Docker's data root; allow tens of GiB.")
    required = {model.stat().st_dev: [model, missing]}
    entry = required.setdefault(ple.parent.stat().st_dev, [ple.parent, 0])
    entry[1] += derived
    if bundle is not None:
        try:
            from tools.image_bundle import read_manifest
        except ModuleNotFoundError:
            from image_bundle import read_manifest
        bundle_manifest = read_manifest(bundle)
        bundle_cache = data / 'image-bundle'
        bundle_missing = sum(part['bytes'] for part in bundle_manifest['parts']
                             if not (bundle_cache / part['name']).is_file()
                             or (bundle_cache / part['name']).stat().st_size != part['bytes'])
        required.setdefault(data.stat().st_dev, [data, 0])[1] += bundle_missing
        print(f"Public image bundle cache: {bundle_cache}; up to {bundle_missing / 2**30:.2f} GiB additional")
    for location, amount in required.values():
        if shutil.disk_usage(location).free < amount + 2**30:
            raise ValueError(f"Insufficient space at {location}: need {amount / 2**30:.2f} GiB plus 1 GiB reserve")
    if args.build:
        build_command = profile.get('commands', {}).get('build', ['scripts/build-image.sh'])
        image = config.get('R9V_IMAGE') or os.environ.get('R9V_IMAGE', 'r9v-qwen38-flash-next:latest')
        # A qualified image ID is an input identity, not a build output tag.
        if image.startswith('sha256:') or '@sha256:' in image:
            image = 'r9v-qwen38-flash-next-mtp4:local'
        run([ROOT / build_command[0], *build_command[1:]],
            env={**os.environ, **config, 'R9V_IMAGE': image})
    elif bundle is not None:
        try:
            from tools.image_bundle import load_bundle, read_manifest
        except ModuleNotFoundError:
            from image_bundle import load_bundle, read_manifest
        bundle_cache = data / 'image-bundle'
        print(f'Public runtime bundle cache: {bundle_cache}; all parts are hash-verified before Docker loads them.', flush=True)
        image = load_bundle(read_manifest(bundle), bundle_cache,
                            expected_image_id=image, allow_existing=True, timeout=900)
    elif not args.local_image:
        run(['docker', 'pull', image])
    user_args = container_user_args()
    image_id = run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                   capture=True, timeout=30).stdout.strip()
    # Save the immutable local image ID, including when the input was a mutable local tag.
    state['ready'] = False
    save(state_path, state)
    artifacts_by_path = {a['path']: a for a in artifacts}
    receipt = state.setdefault('artifacts', {})
    def download(relative):
        if not shutil.which('hf'):
            raise ValueError("Install the Hugging Face hf CLI to download missing model files")
        repository, revision, remote_path = artifact_source(package, artifacts_by_path[relative])
        run(['hf', 'download', repository, remote_path, '--revision',
             revision, '--local-dir', model])
    for artifact in artifacts:
        reuse_artifact(artifact, model, getattr(args, 'reuse_from', None))
        ensure_artifact(artifact, model, receipt, lambda: save(state_path, state),
                        download, args.hash)
    shards = ['/models/' + a['path'] for a in artifacts
              if (a.get('role') == 'target' or a['path'].startswith('target/')) and a['path'].endswith('.gguf')]
    run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python3',
         *user_args, '--security-opt', 'label=disable',
         '--volume', f'{ROOT}:/r9v:ro', '--volume', f'{model}:/models:ro',
         '--volume', f'{ple.parent}:/r9v-data', image_id, '/r9v/tools/prepare_ple.py',
         *shards, '--output', '/r9v-data/' + ple.name])
    config.update(R9V_IMAGE=image_id, R9V_MODEL_DIR=str(model), R9V_PLE_PATH=str(ple),
                  R9V_CACHE_DIR=str(data / 'cache'), R9V_RUNTIME_PREBUILT='1')
    state['config'] = config
    state['ready'] = False
    save(state_path, state)
    env = {**os.environ, **config}
    # Setup has recorded the user's requested headroom, but start is the phase
    # that can create a placement from a matching seed/calibration. This marker
    # is process-local and is deliberately never persisted in state['config'].
    run([ROOT / 'scripts/profile-doctor.sh'], env={**env, 'R9V_SETUP_PHASE': '1'})
    state['ready'] = True
    save(state_path, state)
    print(f"Setup complete. Configuration: {state_path}. Run ./r9v start {profile.get('id', 'qwen38')}")


def start(args, state, state_path):
    if not state.get('ready'):
        raise ValueError("Run setup successfully before start")
    _, profile = selected_profile()
    check_profile_options(args, profile)
    fixed = fixed_placement(profile)
    if (getattr(args, 'ced', None) or state['config'].get('R9V_CED')) == 'quality':
        check_ced_quality_installed(state)
    if getattr(args, 'ced', None):
        state['config']['R9V_CED'] = args.ced
        save(state_path, state)
    env = {**os.environ, **state['config']}
    # Saved paths/image/GPU identity must not be silently replaced by a shell config.
    env.pop('R9V_CONFIG_FILE', None)
    # A fixed placement never re-plans, even from a calibration left in the shell.
    calibration = None if fixed else (getattr(args, 'calibration', None) or env.get('R9V_CALIBRATION_PATH'))
    if getattr(args, 'expert_catalog', None):
        env['R9V_EXPERT_CATALOG_PATH'] = str(args.expert_catalog.resolve())
    if getattr(args, 'headroom', None):
        headroom_bytes(args.headroom, 2)
        env['R9V_MIN_FREE_VRAM_GIB_BY_RANK'] = args.headroom
        env['R9V_HEADROOM_SELECTION'] = '1'
    if not calibration and not fixed:
        try:
            from tools.prepare_placement import apply
        except ModuleNotFoundError:
            from prepare_placement import apply
        runtime = json.loads((ROOT / profile['descriptors']['runtime']).read_text())
        prepared = apply(env, runtime, args.state_dir) if getattr(args, 'state_dir', None) else False
        if not prepared and (env.get('R9V_HEADROOM_SELECTION') == '1' or env.get('R9V_EXPERT_CATALOG_PATH')):
            raise ValueError('--headroom requires a qualified release memory seed or matching --calibration')
        if prepared:
            state['config'].update({k: env[k] for k in ('R9V_EXPERT_MANIFEST_PATH', 'R9V_PLACEMENT_PLAN', 'R9V_MIN_FREE_VRAM_GIB_BY_RANK', 'R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK')})
            for key in ('R9V_EXPERT_CATALOG_PATH', 'R9V_MEMORY_SEED_PATH', 'R9V_HEADROOM_SELECTION'):
                if env.get(key):
                    state['config'][key] = env[key]
            save(state_path, state)
    if calibration:
        output = args.state_dir / (f'placement-{time.time_ns()}')
        source = Path(env.get('R9V_EXPERT_CATALOG_PATH') or str(Path(env['R9V_MODEL_DIR']) / env.get('R9V_MANIFEST_REL', 'manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json')))
        run([sys.executable, ROOT / 'tools/plan_experts.py', '--source', source,
             '--calibration', calibration, '--headroom', env['R9V_MIN_FREE_VRAM_GIB_BY_RANK'],
             '--output', output], env=env)
        env['R9V_EXPERT_MANIFEST_PATH'] = str(output / 'manifest.json')
        env['R9V_PLACEMENT_PLAN'] = str(output / 'plan.json')
        placement = json.loads((output / 'plan.json').read_text())
        env['R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK'] = ','.join(str(n + row['cache_physical_slots']) for n, row in zip(placement['hot_counts'], placement['expert_memory']))
        state['config'].update({k: env[k] for k in ('R9V_EXPERT_MANIFEST_PATH', 'R9V_PLACEMENT_PLAN', 'R9V_MIN_FREE_VRAM_GIB_BY_RANK', 'R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK')})
        state['config']['R9V_CALIBRATION_PATH'] = str(calibration)
        if env.get('R9V_HEADROOM_SELECTION'):
            state['config']['R9V_HEADROOM_SELECTION'] = env['R9V_HEADROOM_SELECTION']
        if env.get('R9V_EXPERT_CATALOG_PATH'):
            state['config']['R9V_EXPERT_CATALOG_PATH'] = env['R9V_EXPERT_CATALOG_PATH']
        save(state_path, state)
    container = env.get('R9V_CONTAINER_NAME', 'r9v-qwen38-flash-next')
    port = env.get('R9V_HOST_PORT', '8004')
    run([ROOT / 'scripts/launch.sh'], env=env)
    if env.get('R9V_CAPTURE_AUTO', '1') == '1':
        capture_path = args.state_dir / ('capture-' + time.strftime('%Y%m%d-%H%M%S'))
        command = [sys.executable, ROOT / 'tools/watch_runtime.py', '--container', container,
                   '--port', port, '--output', capture_path]
        try:
            if not shutil.which('systemd-run'):
                raise ValueError('systemd-run is unavailable')
            unit = 'r9v-capture-' + hashlib.sha256((container + str(capture_path)).encode()).hexdigest()[:12]
            run(watcher_systemd_command(unit, command, env), timeout=15)
            state['latest_capture'] = str(capture_path)
            save(state_path, state)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            print(f'Automatic capture unavailable: {error}. Docker logs remain retained. Run the watcher in a supervised session:', file=sys.stderr)
            print(' '.join(map(str, command)), file=sys.stderr)
    deadline = time.monotonic() + args.timeout
    print("Waiting for model loading, compilation and graph capture; logs remain in Docker.", flush=True)
    while time.monotonic() < deadline:
        status = run(['docker', 'inspect', container, '--format', '{{.State.Status}}'],
                     capture=True, timeout=10).stdout.strip()
        if status != 'running':
            raise ValueError(f"Container stopped during startup: {status}")
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
                healthy = response.status == 200
        except OSError:
            healthy = False
        if healthy:
            run([ROOT / 'scripts/profile-doctor.sh', '--runtime'], env=env)
            identity = qualification_identity(env, fixed)
            if identity:
                receipt = state.get('qualification', {})
                valid = False
                if isinstance(receipt, dict) and all(receipt.get(k) == v for k, v in identity.items()):
                    try:
                        evidence = Path(receipt['result']).read_bytes()
                        valid = (hashlib.sha256(evidence).hexdigest() == receipt['sha256']
                                 and json.loads(evidence).get('passed') is True)
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                if not valid:
                    output = args.state_dir / f'qualification-{time.time_ns()}'
                    run([ROOT / 'scripts/profile-doctor.sh', '--qualify', '--output', output],
                        env=env, timeout=1560)
                    result_path = output / 'result.json'
                    evidence = result_path.read_bytes()
                    if json.loads(evidence).get('passed') is not True:
                        raise ValueError('Placement workload did not qualify')
                    state['qualification'] = {**identity, 'result': str(result_path),
                                              'sha256': hashlib.sha256(evidence).hexdigest()}
                    preemptions = qualification_preemptions(container)
                    if preemptions:
                        state['qualification']['preemptions'] = preemptions
                    save(state_path, state)
            print(f"Ready: http://127.0.0.1:{port}/v1")
            return
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise ValueError(f"Readiness timed out after {args.timeout}s; container retained")


def qualification_preemptions(container):
    """The container's scheduler rewinds right after qualification, so the runtime doctor
    can discount them: the fixed KV budget preempts the ~131K-token qualification prompt
    several times before it completes. None when they cannot be read."""
    try:
        container_id = run(['docker', 'inspect', container, '--format', '{{.Id}}'],
                           capture=True, timeout=10).stdout.strip()
        count = parse_preemptions(run(['docker', 'exec', container, 'python3', '-c', SCHEDULER_PROBE],
                                      capture=True, timeout=20).stdout)
    except (ValueError, subprocess.SubprocessError) as error:
        print(f'Could not record the preemptions of qualification: {error}', file=sys.stderr)
        return None
    return {'container_id': container_id, 'count': count}


def watcher_systemd_command(unit, command, env):
    """Build the watcher command with only explicit Docker selectors."""
    result = ['systemd-run', '--user', '--collect', '--unit', unit,
              '--property=TimeoutStopSec=15']
    for key in ('DOCKER_HOST', 'DOCKER_CONTEXT'):
        value = env.get(key)
        if value:
            result.extend(['--setenv', f'{key}={value}'])
    result.extend(command)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'start'])
    parser.add_argument('--model-dir')
    parser.add_argument('--data-dir')
    parser.add_argument('--reuse-from', type=Path, help='reuse verified matching assets from another quant installation on the same filesystem')
    parser.add_argument('--ple-path', help='reuse or prepare a PLE file at this exact path')
    _, selected = selected_profile()
    profile_id = selected['id']
    default_state = default_state_dir(profile_id)
    parser.add_argument('--state-dir', type=Path, default=default_state)
    parser.add_argument('--image')
    parser.add_argument('--local-image', action='store_true')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--gpu-bdfs')
    parser.add_argument('--headroom', help='per-card free GiB targets, e.g. 5,5; requires a release seed or local calibration')
    parser.add_argument('--calibration', type=Path, help='matching local memory envelope from a qualified workload')
    parser.add_argument('--expert-catalog', type=Path, help='held-out validated full ranking descended from the calibrated source map')
    parser.add_argument('--ced', choices=['on', 'off', 'quality'],
                        help='CED long-prompt prefill for profiles that ship it: on, off, or quality (the '
                             'multi-source projector: less quality loss, less speedup); saved for later starts')
    parser.add_argument('--accept-model-license', action='store_true')
    parser.add_argument('--hash', action='store_true', help='rehash all artifacts, ignoring verification receipts')
    parser.add_argument('--timeout', type=int, default=default_timeout(selected),
                        help='seconds start waits for readiness: 900, or 2400 for a profile with a '
                             'full mutable expert cache, whose first start compiles cold')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 86400:
        parser.error('--timeout must be 1..86400 seconds')
    if args.build and (args.image or args.local_image):
        parser.error('--build cannot be combined with --image/--local-image')
    args.state_dir = args.state_dir.expanduser().resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = args.state_dir / 'setup.json'
    with (args.state_dir / 'setup.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another setup/start is already using this state directory')
        state = {}
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not isinstance(state, dict):
                state = {}
                raise ValueError('Invalid setup state: expected a JSON object')
            if ('ready' in state and type(state['ready']) is not bool
                    or 'artifacts' in state and not isinstance(state['artifacts'], dict)
                    or 'config' in state and (not isinstance(state['config'], dict)
                        or any(not isinstance(k, str) or not isinstance(v, str)
                               for k, v in state['config'].items()))):
                raise ValueError('Invalid setup state: ready/config/artifacts have unexpected types; preserve this file and rerun setup with a new state directory')
            validate_state_profile(state, profile_id)
            if args.action == 'setup':
                setup(args, state, state_path)
            else:
                start(args, state, state_path)
        except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            print(f'Failed: {error}', file=sys.stderr)
            if args.action == 'start' and isinstance(state.get('config'), dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in state['config'].items()):
                output = args.state_dir / ('failure-' + time.strftime('%Y%m%d-%H%M%S'))
                try:
                    run([ROOT / 'scripts/profile-diagnostics.sh', 'support', '--output', output],
                        env={**os.environ, **state['config'], 'R9V_CONFIG_FILE': ''}, timeout=180)
                    print(f'Diagnostics: {output}; review logs before sharing')
                except (OSError, subprocess.SubprocessError) as capture_error:
                    print(f'Diagnostic collection failed: {capture_error}', file=sys.stderr)
            return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
