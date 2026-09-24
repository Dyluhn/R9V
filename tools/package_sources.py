# SPDX-License-Identifier: Apache-2.0
"""Resolve immutable artifact origins for packages assembled from upstreams."""

import re
from pathlib import PurePosixPath


def artifact_source(package, artifact):
    relative = artifact['path']
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Artifact path must stay inside the model directory')
    distribution = artifact.get('distribution', package.get('distribution', {}))
    repository, revision = distribution.get('repository'), distribution.get('revision')
    if not isinstance(repository, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', repository):
        raise ValueError(f'No valid upstream repository for {relative}')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError(f'Artifact {relative} requires an immutable 40-character revision')
    return repository, revision, relative


def selected_artifacts(package, env):
    """The artifacts an install needs: every required one, plus the CED quality
    projector (optional in the package) when env selects R9V_CED=quality."""
    wanted = env.get('R9V_CED_QUALITY_PROJECTOR_REL') if env.get('R9V_CED') == 'quality' else None
    selected = [a for a in package['artifacts'] if a.get('required', True) or a['path'] == wanted]
    if wanted and not any(a['path'] == wanted for a in selected):
        raise ValueError(f"CED quality needs {wanted}, which model package {package.get('id')} does not list")
    return selected
