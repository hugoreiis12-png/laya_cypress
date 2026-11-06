"""Recibo de checkpoint completo, conferido antes de retomar: estado do treino, LoRA e melhor epoca."""
from pathlib import Path
from common import gravar_json, ler_json, sha256_arquivo


def receipt(folder, identity):
    folder = Path(folder)
    required = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
    if any(not (folder / name).is_file() for name in required): raise ValueError('CHECKPOINT_STATE_INCOMPLETE')
    if not (folder / 'adapter_model.safetensors').is_file() or not (folder / 'melhor/adapter_model.safetensors').is_file():
        raise ValueError('CHECKPOINT_WEIGHTS_MISSING')
    files = {}
    for p in sorted(folder.rglob('*')):
        if p.is_symlink(): raise ValueError('CHECKPOINT_LINK')
        if p.is_file() and p.name != 'itsfine-checkpoint.json': files[p.relative_to(folder).as_posix()] = sha256_arquivo(p)
    return {'schema_version': 1, 'identity': identity, 'step': ler_json(folder / 'trainer_state.json')['global_step'],
            'files': files, 'complete': True}


def seal(folder, identity):
    value = receipt(folder, identity)
    gravar_json(Path(folder) / 'itsfine-checkpoint.json', value)
    return value


def verify(folder, identity):
    folder = Path(folder)
    old = ler_json(folder / 'itsfine-checkpoint.json')
    if old != receipt(folder, identity): raise ValueError('CHECKPOINT_IDENTITY_OR_HASH_MISMATCH')
    return old
