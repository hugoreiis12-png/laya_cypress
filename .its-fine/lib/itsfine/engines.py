"""Closed registry of training engines installed under .its-fine/engines/."""
from pathlib import Path
import re
from .storage import read_json

REGISTRY = {
    'hf-peft': {'train': 'train.py', 'checkpoint': 'checkpoint.py', 'requires_checkpoint': False,
                'snapshot_data': ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'check.json')},
    'hf-encoder': {'train': 'train.py', 'checkpoint': 'checkpoint.py', 'requires_checkpoint': True,
                   # calibration and test partitions never enter the training snapshot
                   'snapshot_data': ('train.jsonl', 'validation.jsonl', 'task.json', 'check.json')},
    # audio clips travel with the rows; test stays out of the training snapshot
    'hf-tts': {'train': 'train.py', 'checkpoint': 'checkpoint.py', 'requires_checkpoint': False,
               'snapshot_data': ('train.jsonl', 'validation.jsonl', 'check.json'), 'snapshot_dirs': ('audio',)},
}
DEFAULT = 'hf-peft'


def engine_name(recipe):
    name = recipe.get('engine', DEFAULT) if isinstance(recipe, dict) else DEFAULT
    if name not in REGISTRY: raise ValueError('UNKNOWN_ENGINE:' + str(name))
    return name


def resolve(store, name):
    if name not in REGISTRY: raise ValueError('UNKNOWN_ENGINE:' + str(name))
    folder = store.home / 'engines' / name
    if not (folder / 'engine.json').is_file(): raise ValueError('ENGINE_NOT_INSTALLED:' + name)
    return folder


def capabilities(folder):
    info = read_json(Path(folder) / 'engine.json')
    return {'engine': info['engine'], 'version': info.get('version'), 'colab': info.get('colab', True),
            'perfil': info.get('perfil'), 'tipos': info.get('tipos')}


def own_environment(folder):
    """Engine whose pinned packages conflict with the shared venv runs in .its-fine/venvs/<engine>."""
    return read_json(Path(folder) / 'engine.json').get('ambiente') == 'proprio'


VENV = 'venv:'


def resolve_python(home, value):
    """`venv:<engine>` in a spec means that engine's own venv under the executing store."""
    if not value.startswith(VENV): return value
    name = value[len(VENV):]
    if name not in REGISTRY or not re.fullmatch(r'[a-z0-9-]+', name): raise ValueError('UNKNOWN_ENGINE_VENV:' + name)
    venv = Path(home) / 'venvs' / name
    python = venv / 'Scripts/python.exe' if (venv / 'Scripts').is_dir() else venv / 'bin/python'
    if not python.is_file(): raise ValueError('ENGINE_VENV_MISSING:' + name)
    return str(python)


def remote_python(root, spec):
    """For a remote session: `venv:<engine>` when the spec's engine needs its own venv, else None."""
    from .storage import confined
    manifest = next((e for e in spec['inputs'] if e['name'] == 'engine/engine.json'), None)
    if manifest is None: return None
    info = read_json(confined(root, manifest['path']))
    return VENV + info['engine'] if info.get('ambiente') == 'proprio' else None


def venv_commands(python, info, venv):
    """Commands that build an engine's own venv on top of the session's torch (remote Linux sessions)."""
    req = info['requisitos']
    pip = [str(venv / 'bin/python'), '-m', 'pip', 'install', '-q']
    # {torchaudio}: a sessao troca pela versao exata do torch dela; sem isso o pip pode trazer outro torch
    pacotes = ['{torchaudio}' if p == 'torchaudio' else p for p in req['pacotes']]
    return ([[python, '-m', 'venv', '--system-site-packages', str(venv)], pip + pacotes] +
            ([pip + ['--no-deps'] + list(req['sem_dependencias'])] if req.get('sem_dependencias') else []))


def installed(store):
    return sorted(name for name in REGISTRY if (store.home / 'engines' / name / 'engine.json').is_file())
