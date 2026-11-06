"""Build a supervised engine execution with a snapshot of code and data."""
from pathlib import Path
import sys
from .storage import Store, file_hash, atomic_json, confined, read_json
from . import engines


def referenced(data,files,folder):
    """Relative paths under `folder` named by the `audio` field of the rows in `files`."""
    import json
    found=set()
    for name in files:
        p=data/name
        if not p.is_file() or p.suffix!='.jsonl': continue
        with p.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip(): continue
                rel=json.loads(line).get('audio')
                if rel is None: continue
                if not isinstance(rel,str) or not rel.startswith(folder+'/') or '..' in rel.split('/'): raise ValueError('INVALID_AUDIO_REFERENCE')
                found.add(rel)
    return sorted(found)


def build(root, run_id, recipe, data, executor='process', image=None, timeout=3600, probe=False, checkpoint=None):
    store=Store(root); folder=store.run_dir(run_id)
    recipe_value=read_json(recipe)
    name=engines.engine_name(recipe_value)
    engine_id=name  # `name` is reused by the snapshot loops below
    engine=engines.resolve(store,name)
    spec_engine=engines.REGISTRY[name]
    files=[]
    def add(source,name):
        source=Path(source).resolve()
        if not source.is_relative_to(store.root) or source.is_symlink(): raise ValueError('INPUT_OUTSIDE_PROJECT')
        files.append({'name':name,'path':source.relative_to(store.root).as_posix(),'sha256':file_hash(source)})
    for p in sorted(engine.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json','.jsonl','.txt','.md') and '__pycache__' not in p.parts:
            add(p,'engine/'+p.relative_to(engine).as_posix())
    runtime=Path(__file__).resolve().parent
    for p in sorted(runtime.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts:
            # installed runtime is inside root; source runtime can be too.
            add(p,'lib/itsfine/'+p.relative_to(runtime).as_posix())
    add(recipe,'recipe.json')
    for name in spec_engine['snapshot_data']:
        p=Path(data)/name
        if p.exists(): add(p,'data/'+name)
    for folder_name in spec_engine.get('snapshot_dirs',()):
        # only the files the snapshotted rows reference: held-out audio stays out of training
        for rel in referenced(Path(data),spec_engine['snapshot_data'],folder_name):
            add(Path(data)/rel,'data/'+rel)
    argv=['{python}','{input:engine/train.py}','--receita','{input:recipe.json}',
          '--dados','{inputs}/data','--saida','{output}']
    if spec_engine['requires_checkpoint']:
        # Docker runs without network: the base checkpoint must already be local, and is hashed here.
        base=recipe_value.get('checkpoint',{}).get('path') if isinstance(recipe_value.get('checkpoint'),dict) else None
        base_dir=confined(store.root,base) if base else None
        if base_dir is None or not any(p.suffix in ('.safetensors','.bin') for p in base_dir.rglob('*') if p.is_file()): raise ValueError('CHECKPOINT_NOT_MATERIALIZED')
        for p in sorted(base_dir.rglob('*')):
            if p.is_file(): add(p,'base/'+p.relative_to(base_dir).as_posix())
        argv+=['--checkpoint','{inputs}/base']
    if recipe_value.get('start_from') is not None and not checkpoint:
        # new start from the approved adapter of an earlier run; a resume already carries these weights
        from .continuation import verify
        adapter=verify(store,recipe_value,data)
        for p in sorted(adapter.rglob('*')):
            if p.is_file(): add(p,'start/'+p.relative_to(adapter).as_posix())
        argv+=['--start-from','{inputs}/start']
    if probe: argv.append('--probe')
    if checkpoint:
        for p in sorted(Path(checkpoint).rglob('*')):
            if p.is_file(): add(p,'checkpoint/'+p.relative_to(checkpoint).as_posix())
        argv+=['--resume-from-checkpoint','{inputs}/checkpoint']
    spec={'schema_version':1,'executor':executor,'argv':argv,'inputs':files,'timeout_seconds':timeout,
          'metered':False,'engine_job':True,'gpus':True}
    if executor=='docker': spec['image']=image
    else:
        setup=store.home/'engine-setup.json'
        if engines.own_environment(engine):
            # pinned packages that conflict with the shared venv: python of the engine's own venv
            prepared=read_json(setup).get(engine_id,{}) if setup.exists() else {}
            if not prepared.get('pronto') or not prepared.get('python'): raise ValueError('ENGINE_NOT_PREPARED:'+engine_id)
            spec['python']=prepared['python']
        elif setup.exists():
            spec['python']=read_json(setup)['python']
    target=folder/'execution-proposed.json'
    if target.exists(): raise FileExistsError('EXECUTION_PROPOSAL_EXISTS')
    atomic_json(target,spec)
    # tamanho total, para avisar o usuario antes de um envio longo ao Colab
    mb=round(sum((store.root/f['path']).stat().st_size for f in files)/1024**2,1)
    return {'path':str(target),'engine':engine_id,'inputs':len(files),'mb':mb,'probe':probe}
