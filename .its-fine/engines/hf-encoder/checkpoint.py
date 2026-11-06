"""Recibo de checkpoint completo do hf-encoder: adapter, cabecas, otimizador, scheduler e RNG."""
from pathlib import Path
from common import gravar_json, ler_json, sha256_arquivo

EXIGIDOS = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors',
            'adapter_config.json', 'heads.safetensors')


def receipt(folder, identity):
    folder = Path(folder)
    faltam = [nome for nome in EXIGIDOS if not (folder / nome).is_file()]
    if faltam: raise ValueError('CHECKPOINT_INCOMPLETE:' + ','.join(faltam))
    if (folder / 'checkpoint-is-incomplete.txt').exists(): raise ValueError('CHECKPOINT_INCOMPLETE:marker')
    files = {}
    for p in sorted(folder.rglob('*')):
        if p.is_symlink(): raise ValueError('CHECKPOINT_LINK')
        if p.is_file() and p.name != 'itsfine-checkpoint.json': files[p.relative_to(folder).as_posix()] = sha256_arquivo(p)
    return {'schema_version': 1, 'engine': 'hf-encoder', 'identity': identity, 'exigidos': list(EXIGIDOS),
            'step': ler_json(folder / 'trainer_state.json')['global_step'], 'files': files, 'complete': True}


def seal(folder, identity):
    value = receipt(folder, identity)
    gravar_json(Path(folder) / 'itsfine-checkpoint.json', value)
    return value


def verify(folder, identity):
    folder = Path(folder)
    old = ler_json(folder / 'itsfine-checkpoint.json')
    if old != receipt(folder, identity): raise ValueError('CHECKPOINT_IDENTITY_OR_HASH_MISMATCH')
    return old
