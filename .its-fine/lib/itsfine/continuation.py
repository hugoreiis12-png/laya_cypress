"""Continue training from the approved adapter of an earlier run: a new start, never a resume."""
from pathlib import Path
from .storage import confined, digest, file_hash, read_json
from . import engines


def adapter_hash(folder):
    """Same value the engines record in treino.json: adapter weights and, in hf-encoder, the heads; in hf-tts, the extras."""
    folder = Path(folder)
    parts = {'adapter': file_hash(folder / 'adapter_model.safetensors')}
    if (folder / 'heads.safetensors').is_file(): parts['heads'] = file_hash(folder / 'heads.safetensors')
    if (folder / 'extras.safetensors').is_file(): parts['extras'] = file_hash(folder / 'extras.safetensors')
    return digest(parts)


def check_lora(folder, lora, targets):
    """The loaded adapter fixes rank, alpha and target modules; a recipe that says otherwise is refused."""
    config = read_json(Path(folder) / 'adapter_config.json')
    found = config.get('target_modules')
    found = sorted(found) if isinstance(found, list) else found
    if (config.get('r'), config.get('lora_alpha'), found) != (lora['r'], lora['alpha'], sorted(targets)):
        raise ValueError('START_FROM_LORA_MISMATCH')


def verify(store, recipe, data):
    """Adapter folder the recipe starts from, after checking it is the one approved in the parent run."""
    from .contracts.validate import DEPENDENCIES, evaluation_decision, evaluation_decision_v2
    engine = engines.engine_name(recipe)
    start = recipe['start_from']
    if not isinstance(start, dict) or set(start) != {'run_id', 'adapter_hash', 'path'}: raise ValueError('START_FROM_INVALID')
    parent = store.run_dir(start['run_id'])
    artifacts = store.artifacts(start['run_id'])
    ev = artifacts.get('evaluation')
    if not ev or any(dep not in artifacts or ev['input_hashes'].get(dep) != digest(artifacts[dep]) for dep in DEPENDENCIES['evaluation']):
        raise ValueError('START_FROM_NOT_APPROVED')
    goal, data_ev = artifacts['objective']['data'], ev['data']
    decision = (evaluation_decision_v2(goal, data_ev) if ev['schema_version'] == 2 else
                evaluation_decision(goal, data_ev['baseline'], data_ev['candidate'], data_ev['sample_count'],
                                    data_ev.get('start_from', {}).get('value')))
    if data_ev['decision'] != 'aprovada' or decision != 'aprovada': raise ValueError('START_FROM_NOT_APPROVED')

    adapter = confined(store.root, start['path'])
    if not adapter.is_relative_to(parent) or not (adapter / 'adapter_model.safetensors').is_file():
        raise ValueError('START_FROM_ADAPTER_NOT_IN_PARENT')
    found = adapter_hash(adapter)
    report = adapter.parent / 'treino.json'
    if found != start['adapter_hash'] or not report.is_file() or read_json(report).get('adapter_hash', found) != found:
        raise ValueError('START_FROM_HASH_MISMATCH')

    # the recipe the parent job really ran with, bound to its proposal by hash
    inputs = {i['name']: i for i in read_json(parent / 'execution-proposed.json')['inputs']}
    source = inputs.get('recipe.json')
    copies = [parent / 'inputs/recipe.json'] + ([confined(store.root, source['path'])] if source else [])
    used = next((p for p in copies if source and p.is_file() and file_hash(p) == source['sha256']), None)
    if used is None: raise ValueError('START_FROM_PARENT_RECIPE_UNAVAILABLE')
    before = read_json(used)
    if engines.engine_name(before) != engine: raise ValueError('START_FROM_BASE_MISMATCH')
    if engine == 'hf-encoder':
        same = all(before['checkpoint'].get(k) == recipe['checkpoint'].get(k) for k in ('id', 'revision'))
        task = inputs.get('data/task.json')
        if not task or task['sha256'] != file_hash(Path(data) / 'task.json'): raise ValueError('START_FROM_TASK_MISMATCH')
    else:
        same = all(before.get(k) == recipe.get(k) for k in ('model_id', 'revision'))
    if not same: raise ValueError('START_FROM_BASE_MISMATCH')
    return adapter
