# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import setup_profile as setup


def artifact(data=b'weights'):
    return {'path': 'target/model.gguf', 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def test_resume_then_mutation_forces_verification(tmp_path, monkeypatch):
    item = artifact()
    path = tmp_path / item['path']
    path.parent.mkdir()
    path.write_bytes(b'weights')
    receipt = {}
    saved = []
    setup.ensure_artifact(item, tmp_path, receipt, lambda: saved.append(True), None)
    assert saved == [True]
    real = setup._sha256
    monkeypatch.setattr(setup, '_sha256', lambda p: pytest.fail('unnecessary rehash'))
    setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None)
    monkeypatch.setattr(setup, '_sha256', real)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Integrity'):
        setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None)


def test_interrupted_download_resumes_and_checkpoints(tmp_path):
    item = artifact()
    receipt = {}
    def interrupted(name):
        path = tmp_path / name
        path.parent.mkdir()
        path.write_bytes(b'w')
        raise OSError('interrupted')
    with pytest.raises(OSError):
        setup.ensure_artifact(item, tmp_path, receipt, lambda: None, interrupted)
    assert receipt == {}
    def finish(name):
        (tmp_path / name).write_bytes(b'weights')
    state = tmp_path / 'state.json'
    setup.ensure_artifact(item, tmp_path, receipt, lambda: setup.save(state, receipt), finish)
    assert json.loads(state.read_text()) == receipt
    assert state.stat().st_mode & 0o777 == 0o600


def test_changed_during_hash_is_not_receipted(tmp_path, monkeypatch):
    item = artifact()
    path = tmp_path / item['path']
    path.parent.mkdir()
    path.write_bytes(b'weights')
    def changing(p):
        p.write_bytes(b'changed')
        return item['sha256']
    monkeypatch.setattr(setup, '_sha256', changing)
    receipt = {}
    with pytest.raises(ValueError, match='changed during'):
        setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None)
    assert not receipt


def test_rejects_path_escape(tmp_path):
    item = artifact()
    item['path'] = '../outside'
    with pytest.raises(ValueError, match='escapes'):
        setup.ensure_artifact(item, tmp_path, {}, lambda: None, None)


def test_explicit_hash_bypasses_receipt(tmp_path, monkeypatch):
    item = artifact()
    path = tmp_path / item['path']
    path.parent.mkdir()
    path.write_bytes(b'weights')
    receipt = {item['path']: {'identity': setup.identity(path), 'sha256': item['sha256']}}
    calls = []
    monkeypatch.setattr(setup, '_sha256', lambda p: calls.append(p) or item['sha256'])
    setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None, True)
    assert calls == [path]


def test_start_stopped_container_fails_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setenv('R9V_CAPTURE_AUTO', '0')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='exited')
    monkeypatch.setattr(setup, 'run', run)
    with pytest.raises(ValueError, match='stopped'):
        setup.start(SimpleNamespace(timeout=10), {'ready': True, 'config': {}}, tmp_path)
    assert len(calls) == 2


def test_watcher_systemd_propagates_only_explicit_docker_selectors():
    command = ['python3', 'watch_runtime.py']
    base = setup.watcher_systemd_command('r9v-capture-unit', command, {})
    assert base[-2:] == command
    assert '--setenv' not in base
    propagated = setup.watcher_systemd_command(
        'r9v-capture-unit', command,
        {'DOCKER_HOST': 'unix:///run/user/1000/public.sock',
         'DOCKER_CONTEXT': 'public-context', 'DOCKER_CONFIG': '/secret'},
    )
    assert propagated[-2:] == command
    assert propagated[6:10] == [
        '--setenv', 'DOCKER_HOST=unix:///run/user/1000/public.sock',
        '--setenv', 'DOCKER_CONTEXT=public-context',
    ]
    assert 'DOCKER_CONFIG' not in ' '.join(propagated)


def test_setup_requires_explicit_image_before_side_effects(tmp_path):
    args = SimpleNamespace(image=None, build=False)
    with pytest.raises(ValueError, match='No published image'):
        setup.setup(args, {}, tmp_path / 'setup.json')


def test_cli_dispatch():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    for action in ('setup', 'start'):
        result = subprocess.run([str(root / 'r9v'), action, 'qwen38', '--dry-run'],
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)['command'][-1] == action


def test_setup_reuses_assets_pins_image_and_requires_final_doctor(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    model = tmp_path / 'models'
    model.mkdir()
    item = artifact()
    (model / 'target').mkdir()
    (model / item['path']).write_bytes(b'weights')
    (root / 'package.json').write_text(json.dumps({
        'artifacts': [item], 'distribution': {'repository': 'test/repo', 'revision': 'abc'}}))
    (root / 'runtime.json').write_text('{}')
    profile = root / 'profile.json'
    profile.write_text(json.dumps({'id': 'qwen38-flash-next/ud-iq4-xs/dual-r9700-128k', 'descriptors': {
        'model_package': 'package.json', 'runtime': 'runtime.json'}}))
    monkeypatch.setattr(setup, 'ROOT', root)
    monkeypatch.setattr(setup, 'PROFILE', profile)
    monkeypatch.setattr(setup, 'PLE_BYTES', 1)
    monkeypatch.setattr(setup.os, 'access', lambda *a: True)
    monkeypatch.delenv('R9V_CONFIG_FILE', raising=False)
    monkeypatch.setattr(setup, 'profile_settings', lambda: {})
    monkeypatch.setattr(setup, 'container_user_args', lambda: ['--user', '0:0'])
    monkeypatch.setattr(setup, 'select_devices', lambda b: {
        'R9V_VISIBLE_DEVICES': '1,2', 'R9V_EXPECTED_GPU_BDFS': 'a,b'})
    calls = []
    doctor_envs = []
    fail_doctor = [True]
    def run(command, **kwargs):
        calls.append([str(c) for c in command])
        if str(command[0]).endswith('profile-doctor.sh') and len(command) == 1:
            doctor_envs.append(kwargs.get('env', {}))
        if str(command[0]).endswith('profile-doctor.sh') and len(command) == 1 and fail_doctor[0]:
            raise ValueError('doctor failed')
        return SimpleNamespace(stdout='sha256:resolved-image\n')
    monkeypatch.setattr(setup, 'run', run)
    args = SimpleNamespace(image='local:test', local_image=True, build=False,
                           accept_model_license=True, gpu_bdfs=None,
                           model_dir=str(model), data_dir=None, ple_path=None, hash=False)
    state = {}
    state_path = tmp_path / 'setup.json'
    with pytest.raises(ValueError, match='doctor failed'):
        setup.setup(args, state, state_path)
    assert json.loads(state_path.read_text())['ready'] is False
    assert state['config']['R9V_IMAGE'] == 'sha256:resolved-image'
    monkeypatch.setattr(setup, '_sha256', lambda p: pytest.fail('rehash on retry'))
    fail_doctor[0] = False
    setup.setup(args, state, state_path)
    assert json.loads(state_path.read_text())['ready'] is True
    assert doctor_envs[-1].get('R9V_SETUP_PHASE') == '1'
    assert 'R9V_SETUP_PHASE' not in json.loads(state_path.read_text())['config']
    assert not any(c[:2] == ['docker', 'pull'] or c[0] == 'hf' for c in calls)
    extraction = next(c for c in calls if c[:2] == ['docker', 'run'])
    assert 'sha256:resolved-image' in extraction
    assert '/models/target/model.gguf' in extraction


def test_cli_interrupted_download_resumes_with_persistent_receipts(tmp_path):
    """Exercise separate setup processes with tiny assets and fake external services."""
    import os
    import subprocess
    import sys
    import textwrap

    root = tmp_path / 'fixture'
    (root / 'scripts').mkdir(parents=True)
    (root / 'tools').mkdir()
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    artifacts = [dict(artifact(payload), path=name) for name, payload in
                 [('first.txt', b'first'), ('target/model.gguf', b'weights')]]
    (root / 'package.json').write_text(json.dumps({'artifacts': artifacts,
        'distribution': {'repository': 'fixture/model', 'revision': 'a' * 40}}))
    (root / 'runtime.json').write_text('{}')
    (root / 'profile.json').write_text(json.dumps({'id': 'qwen38-flash-next/ud-iq4-xs/dual-r9700-128k', 'descriptors': {
        'model_package': 'package.json', 'runtime': 'runtime.json'}}))
    doctor = root / 'scripts/profile-doctor.sh'
    doctor.write_text('#!/bin/sh\nexit 0\n')
    doctor.chmod(0o755)
    hf = binaries / 'hf'
    hf.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''
        import os, sys
        from pathlib import Path
        name = sys.argv[3]
        root = Path(sys.argv[sys.argv.index('--local-dir') + 1])
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(os.environ['TEST_CALLS'], 'a') as stream: stream.write(name + '\\n')
        if name == 'first.txt': path.write_bytes(b'first')
        elif not (root / 'interrupted').exists():
            path.write_bytes(b'w')
            (root / 'interrupted').touch()
            raise SystemExit(17)
        else: path.write_bytes(b'weights')
    '''))
    hf.chmod(0o755)
    docker = binaries / 'docker'
    docker.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''
        import sys
        from pathlib import Path
        if sys.argv[1] == 'info':
            print('[]' if 'SecurityOptions' in sys.argv[-1] else '/fixture/docker')
        elif sys.argv[1:3] == ['image', 'inspect']: print('sha256:fixture')
        elif sys.argv[1] == 'run':
            volume = next(x for x in sys.argv if x.endswith(':/r9v-data'))
            (Path(volume[:-10]) / 'per_layer_token_embd.iq4_nl.bin').write_bytes(b'p')
        else: raise SystemExit('unexpected Docker call')
    '''))
    docker.chmod(0o755)
    bootstrap = tmp_path / 'bootstrap.py'
    bootstrap.write_text(textwrap.dedent('''
        import os, sys
        from pathlib import Path
        sys.path.insert(0, os.environ['TEST_SOURCE'])
        from tools import setup_profile as s
        s.ROOT = Path(os.environ['TEST_ROOT'])
        s.PROFILE = s.ROOT / 'profile.json'
        s.PLE_BYTES = 1
        s.profile_settings = lambda: {'R9V_MAX_MODEL_LEN': '4096'}
        s.select_devices = lambda b: {'R9V_VISIBLE_DEVICES': '0,1'}
        raise SystemExit(s.main())
    '''))
    state = tmp_path / 'state'
    model = tmp_path / 'model'
    env = {**os.environ, 'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
           'TEST_ROOT': str(root), 'TEST_SOURCE': str(Path(__file__).resolve().parents[1]),
           'TEST_CALLS': str(tmp_path / 'calls')}
    env.pop('R9V_CONFIG_FILE', None)
    command = [sys.executable, str(bootstrap), 'setup', '--model-dir', str(model),
               '--state-dir', str(state), '--image', 'fixture:local', '--local-image',
               '--accept-model-license']
    failed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    receipt = json.loads((state / 'setup.json').read_text())
    assert not receipt['ready']
    assert set(receipt['artifacts']) == {'first.txt'}
    success = subprocess.run(command, env=env, capture_output=True, text=True)
    assert success.returncode == 0, success.stdout + success.stderr
    assert 'Reusing previously hash-verified file: first.txt' in success.stdout
    receipt = json.loads((state / 'setup.json').read_text())
    assert receipt['ready']
    assert receipt['config']['R9V_MAX_MODEL_LEN'] == '4096'
    assert (tmp_path / 'calls').read_text().splitlines() == [
        'first.txt', 'target/model.gguf', 'target/model.gguf']
    # Invalid retry arguments must preserve the successful installation.
    invalid = subprocess.run(command + ['--image=-bad'],
                             env=env, capture_output=True, text=True)
    assert invalid.returncode == 1
    assert json.loads((state / 'setup.json').read_text())['ready']


@pytest.mark.parametrize('options, expected', [
    (['name=rootless', 'name=seccomp,profile=builtin'], ['--user', '0:0']),
    ([], None),
    (['name=userns'], 'remapped'),
])
def test_extractor_user_matches_docker_namespace(monkeypatch, options, expected):
    import os
    monkeypatch.setattr(setup, 'run', lambda *a, **k: SimpleNamespace(stdout=json.dumps(options)))
    if expected is None:
        expected = ['--user', f'{os.getuid()}:{os.getgid()}']
    elif expected == 'remapped':
        expected = ['--user', f'{os.getuid()}:{os.getgid()}', '--userns', 'host']
    assert setup.container_user_args() == expected


def test_start_attaches_supervised_passive_capture_before_readiness(tmp_path, monkeypatch):
    monkeypatch.setenv('R9V_CAPTURE_AUTO', '1')
    monkeypatch.setattr(setup.shutil, 'which', lambda _: '/usr/bin/systemd-run')
    commands = []
    def run(command, **kwargs):
        commands.append(list(map(str, command)))
        return SimpleNamespace(stdout='exited')
    monkeypatch.setattr(setup, 'run', run)
    state = {'ready': True, 'config': {}}
    with pytest.raises(ValueError, match='stopped'):
        setup.start(SimpleNamespace(timeout=10, state_dir=tmp_path), state, tmp_path / 'setup.json')
    assert commands[1][:3] == ['systemd-run', '--user', '--collect']
    assert 'watch_runtime.py' in ' '.join(commands[1])
    assert json.loads((tmp_path / 'setup.json').read_text())['latest_capture']


def test_start_qualifies_new_placement_once_and_rechecks_changed_evidence(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    monkeypatch.setenv('R9V_CAPTURE_AUTO', '0')
    placement = tmp_path / 'plan.json'
    placement.write_text('{"schema":"fixture","counts":[300,310]}')
    state = {'ready': True, 'config': {'R9V_PLACEMENT_PLAN': str(placement)}}
    calls = []
    def run(command, **kwargs):
        calls.append(list(map(str, command)))
        if '--qualify' in command:
            output = Path(command[command.index('--output') + 1])
            output.mkdir()
            (output / 'result.json').write_text('{"passed":true}')
        return SimpleNamespace(stdout='running')
    monkeypatch.setattr(setup, 'run', run)
    monkeypatch.setattr(setup.urllib.request, 'urlopen', lambda *a, **k: nullcontext(SimpleNamespace(status=200)))
    args = SimpleNamespace(timeout=10, state_dir=tmp_path)
    setup.start(args, state, tmp_path / 'setup.json')
    setup.start(args, state, tmp_path / 'setup.json')
    assert sum('--qualify' in command for command in calls) == 1
    Path(state['qualification']['result']).write_text('{"passed":false}')
    setup.start(args, state, tmp_path / 'setup.json')
    assert sum('--qualify' in command for command in calls) == 2


def test_start_records_the_rewinds_qualification_caused_in_this_container(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setenv('R9V_CAPTURE_AUTO', '0')
    placement = tmp_path / 'plan.json'
    placement.write_text('{"schema":"fixture","counts":[300,310]}')
    state = {'ready': True, 'config': {'R9V_PLACEMENT_PLAN': str(placement)}}
    def run(command, **kwargs):
        command = list(map(str, command))
        if '--qualify' in command:
            output = Path(command[command.index('--output') + 1])
            output.mkdir()
            (output / 'result.json').write_text('{"passed":true}')
        if command[:2] == ['docker', 'exec']:
            return SimpleNamespace(stdout='{"total_preemptions": 7}')
        if '{{.Id}}' in command:
            return SimpleNamespace(stdout='abc123\n')
        return SimpleNamespace(stdout='running')
    monkeypatch.setattr(setup, 'run', run)
    monkeypatch.setattr(setup.urllib.request, 'urlopen', lambda *a, **k: nullcontext(SimpleNamespace(status=200)))

    setup.start(SimpleNamespace(timeout=10, state_dir=tmp_path), state, tmp_path / 'setup.json')

    saved = json.loads((tmp_path / 'setup.json').read_text())
    assert saved['qualification']['preemptions'] == {'container_id': 'abc123', 'count': 7}


def test_reuse_across_quants_links_only_hash_matching_assets(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir()
    new.mkdir()
    item = artifact()
    (old / 'target').mkdir()
    source = old / item['path']
    source.write_bytes(b'weights')
    assert setup.reuse_artifact(item, new, old)
    assert source.stat().st_ino == (new / item['path']).stat().st_ino
    another = dict(item, path='target/different.gguf')
    (old / another['path']).write_bytes(b'changed')
    assert not setup.reuse_artifact(another, new, old)
    assert not (new / another['path']).exists()


def test_reusable_source_is_counted_without_allocating_duplicate_storage(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir()
    new.mkdir()
    item = artifact()
    source = old / item['path']
    source.parent.mkdir()
    source.write_bytes(b'weights')
    assert setup.reusable_source(item, new, old) == source.resolve()
    assert not (new / item['path']).exists()


UNCENSORED = Path(__file__).resolve().parents[1] / 'profiles/qwen38-flash-next/dual-r9700-mtp4-uncensored'
UNCENSORED_ID = 'qwen38-flash-next/uncensored-iq4-xs/dual-r9700-mtp4-128k'


def select_uncensored(monkeypatch):
    monkeypatch.setenv('R9V_PROFILE_ROOT', str(UNCENSORED))
    monkeypatch.setenv('R9V_PROFILE_ID', UNCENSORED_ID)
    for key in ('R9V_CED', 'R9V_PLACEMENT_PLAN', 'R9V_CALIBRATION_PATH', 'R9V_CONFIG_FILE'):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize('option', ['headroom', 'calibration', 'expert_catalog'])
@pytest.mark.parametrize('action', ['setup', 'start'])
def test_fixed_placement_refuses_replanning_options_before_side_effects(
        tmp_path, monkeypatch, option, action):
    select_uncensored(monkeypatch)
    monkeypatch.setattr(setup, 'run', lambda *a, **k: pytest.fail('side effect before refusal'))
    args = SimpleNamespace(**{option: '1.5,1.5' if option == 'headroom' else tmp_path / 'x.json'},
                           image=None, build=False, timeout=10, state_dir=tmp_path)
    state = {'ready': True, 'config': {}}

    with pytest.raises(ValueError, match='fixed placement'):
        getattr(setup, action)(args, state, tmp_path / 'setup.json')


def test_ced_option_is_refused_for_a_runtime_without_ced(tmp_path, monkeypatch):
    monkeypatch.setenv('R9V_PROFILE_ROOT', str(UNCENSORED.parent / 'dual-r9700-mtp4'))
    monkeypatch.setenv('R9V_PROFILE_ID', 'qwen38-flash-next/ud-iq4-xs/dual-r9700-mtp4-128k')
    monkeypatch.setattr(setup, 'run', lambda *a, **k: pytest.fail('side effect before refusal'))

    with pytest.raises(ValueError, match='has no CED'):
        setup.start(SimpleNamespace(ced='on', timeout=10, state_dir=tmp_path),
                    {'ready': True, 'config': {}}, tmp_path / 'setup.json')


def test_fixed_placement_qualifies_first_start_and_requalifies_when_ced_changes(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from tools import prepare_placement
    select_uncensored(monkeypatch)
    monkeypatch.setenv('R9V_CAPTURE_AUTO', '0')
    monkeypatch.setattr(prepare_placement, 'apply', lambda *a: pytest.fail('fixed placement re-planned'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"fixture": true}')
    runtime = tmp_path / 'runtime.json'
    runtime.write_text('{}')
    state = {'ready': True, 'config': {'R9V_EXPERT_MANIFEST_PATH': str(manifest), 'R9V_CED': 'on',
                                       'R9V_RUNTIME_DESCRIPTOR': str(runtime), 'R9V_IMAGE': 'sha256:fixture'}}
    launches = []
    def run(command, **kwargs):
        if str(command[0]).endswith('launch.sh'):
            launches.append(kwargs['env']['R9V_CED'])
        if '--qualify' in command:
            output = Path(command[command.index('--output') + 1])
            output.mkdir()
            (output / 'result.json').write_text('{"passed":true}')
        return SimpleNamespace(stdout='running')
    monkeypatch.setattr(setup, 'run', run)
    monkeypatch.setattr(setup.urllib.request, 'urlopen', lambda *a, **k: nullcontext(SimpleNamespace(status=200)))
    qualified = []
    def start(**options):
        before = state.get('qualification')
        setup.start(SimpleNamespace(timeout=10, state_dir=tmp_path, **options), state, tmp_path / 'setup.json')
        qualified.append(state['qualification'] is not before)

    start()
    start()
    start(ced='off')
    start()

    assert qualified == [True, False, True, False]
    assert launches == ['on', 'on', 'off', 'off']
    saved = json.loads((tmp_path / 'setup.json').read_text())
    assert saved['config']['R9V_CED'] == 'off'
    assert saved['qualification']['ced'] == 'off'
    assert saved['qualification']['placement_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize(('profile', 'default'), [('dual-r9700-mtp4-uncensored', 2400),
                                                  ('dual-r9700-mtp4', 900)])
def test_start_timeout_defaults_to_2400_s_only_for_the_full_mutable_cache_and_keeps_the_override(
        tmp_path, monkeypatch, profile, default):
    import sys
    root = UNCENSORED.parent / profile
    monkeypatch.setenv('R9V_PROFILE_ROOT', str(root))
    monkeypatch.setenv('R9V_PROFILE_ID', json.loads((root / 'profile.json').read_text())['id'])
    monkeypatch.delenv('R9V_CONFIG_FILE', raising=False)
    waits = []
    monkeypatch.setattr(setup, 'start', lambda args, state, path: waits.append(args.timeout))
    for override in ([], ['--timeout', '600']):
        monkeypatch.setattr(sys, 'argv', ['setup_profile.py', 'start', '--state-dir', str(tmp_path), *override])
        assert setup.main() == 0

    assert waits == [default, 600]


QUALITY = 'ced/quality-int8.safetensors'


def quality_repo(tmp_path, monkeypatch, published):
    """A repo whose package holds one present required file and the optional CED quality
    projector (published at a pinned revision, or only prepared), with setup's host checks faked."""
    root, model = tmp_path / 'repo', tmp_path / 'models'
    root.mkdir()
    (model / 'target').mkdir(parents=True)
    item = artifact()
    (model / item['path']).write_bytes(b'weights')
    quality = dict(artifact(b'quality'), path=QUALITY, role='ced-projector', required=False,
                   distribution={'repository': 'test/repo', 'revision': 'b' * 40 if published else None,
                                 'note': 'not published yet'})
    (root / 'package.json').write_text(json.dumps({
        'artifacts': [item, quality], 'distribution': {'repository': 'test/repo', 'revision': 'a' * 40}}))
    (root / 'runtime.json').write_text('{"overlays": {"mounts": {"ced": {}, "ced-quality": {}}}}')
    profile = root / 'profile.json'
    profile.write_text(json.dumps({'id': 'qwen38-flash-next/ud-iq4-xs/dual-r9700-128k', 'descriptors': {
        'model_package': 'package.json', 'runtime': 'runtime.json'}}))
    monkeypatch.setattr(setup, 'ROOT', root)
    monkeypatch.setattr(setup, 'PROFILE', profile)
    monkeypatch.setattr(setup, 'PLE_BYTES', 1)
    monkeypatch.delenv('R9V_CONFIG_FILE', raising=False)
    monkeypatch.delenv('R9V_PROFILE_ROOT', raising=False)
    monkeypatch.delenv('R9V_PROFILE_ID', raising=False)
    monkeypatch.setattr(setup, 'profile_settings', lambda: {
        'R9V_CED': 'on', 'R9V_CED_QUALITY_PROJECTOR_REL': QUALITY})
    monkeypatch.setattr(setup, 'container_user_args', lambda: ['--user', '0:0'])
    monkeypatch.setattr(setup, 'select_devices', lambda b: {})
    monkeypatch.setattr(setup.shutil, 'which', lambda name: '/usr/bin/' + name)
    calls = []
    def run(command, **kwargs):
        command = [str(c) for c in command]
        calls.append(command)
        if command[:2] == ['hf', 'download']:
            (model / command[3]).parent.mkdir(parents=True, exist_ok=True)
            (model / command[3]).write_bytes(b'quality')
        return SimpleNamespace(stdout='sha256:image\n')
    monkeypatch.setattr(setup, 'run', run)
    args = SimpleNamespace(image='local:test', local_image=True, build=False, accept_model_license=True,
                           gpu_bdfs=None, model_dir=str(model), data_dir=None, ple_path=None, hash=False)
    return args, model, calls


def downloads(calls):
    return [command[3] for command in calls if command[:2] == ['hf', 'download']]


def test_setup_with_ced_quality_downloads_only_then_its_projector(tmp_path, monkeypatch):
    args, model, calls = quality_repo(tmp_path, monkeypatch, published=True)
    state = {}

    setup.setup(SimpleNamespace(**vars(args), ced='quality'), state, tmp_path / 'setup.json')

    assert downloads(calls) == [QUALITY]
    assert state['config']['R9V_CED'] == 'quality'
    assert QUALITY in state['artifacts']


def test_setup_without_ced_quality_never_downloads_its_projector(tmp_path, monkeypatch):
    args, model, calls = quality_repo(tmp_path, monkeypatch, published=True)
    state = {}

    setup.setup(SimpleNamespace(**vars(args), ced=None), state, tmp_path / 'setup.json')

    assert downloads(calls) == []
    assert state['config']['R9V_CED'] == 'on'
    assert QUALITY not in state['artifacts']


def test_setup_with_an_unpublished_quality_projector_refuses_before_any_download(tmp_path, monkeypatch):
    args, model, calls = quality_repo(tmp_path, monkeypatch, published=False)

    with pytest.raises(ValueError, match=f'Cannot download {QUALITY}: not published yet'):
        setup.setup(SimpleNamespace(**vars(args), ced='quality'), {}, tmp_path / 'setup.json')

    assert downloads(calls) == []
    assert not any(command[:2] == ['docker', 'run'] for command in calls)


def test_start_with_ced_quality_before_setup_fetched_it_refuses_and_keeps_the_saved_mode(tmp_path, monkeypatch):
    select_uncensored(monkeypatch)
    monkeypatch.setattr(setup, 'run', lambda *a, **k: pytest.fail('launched without the quality projector'))
    state = {'ready': True, 'artifacts': {}, 'config': {
        'R9V_CED': 'on', 'R9V_CED_QUALITY_PROJECTOR_REL': QUALITY}}

    with pytest.raises(ValueError, match='Run setup again with --ced quality'):
        setup.start(SimpleNamespace(ced='quality', timeout=10, state_dir=tmp_path), state, tmp_path / 'setup.json')

    assert state['config']['R9V_CED'] == 'on'


def test_ced_quality_is_refused_for_a_runtime_without_it(tmp_path, monkeypatch):
    monkeypatch.setenv('R9V_PROFILE_ROOT', str(UNCENSORED.parent / 'dual-r9700-mtp4'))
    monkeypatch.setenv('R9V_PROFILE_ID', 'qwen38-flash-next/ud-iq4-xs/dual-r9700-mtp4-128k')
    monkeypatch.setattr(setup, 'run', lambda *a, **k: pytest.fail('side effect before refusal'))

    with pytest.raises(ValueError, match='--ced quality: .* has no CED quality'):
        setup.start(SimpleNamespace(ced='quality', timeout=10, state_dir=tmp_path),
                    {'ready': True, 'config': {}}, tmp_path / 'setup.json')


@pytest.mark.parametrize('value', ['quality', 'on', 'off'])
def test_cli_accepts_each_ced_mode(tmp_path, monkeypatch, value):
    import sys
    select_uncensored(monkeypatch)
    seen = []
    monkeypatch.setattr(setup, 'start', lambda args, state, path: seen.append(args.ced))
    monkeypatch.setattr(sys, 'argv', ['setup_profile.py', 'start', '--state-dir', str(tmp_path), '--ced', value])

    assert setup.main() == 0
    assert seen == [value]


def test_cli_refuses_an_unknown_ced_mode_with_the_choices(tmp_path, monkeypatch, capsys):
    import sys
    select_uncensored(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['setup_profile.py', 'start', '--state-dir', str(tmp_path), '--ced', 'high'])

    with pytest.raises(SystemExit):
        setup.main()
    error = capsys.readouterr().err
    assert "argument --ced: invalid choice: 'high'" in error
    assert "quality" in error
