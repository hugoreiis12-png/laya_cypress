"""Durable JSON state, kernel locks and confined artifact paths."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import time

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): sha.update(chunk)
    return sha.hexdigest()

def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', value):
        raise ValueError('INVALID_IDENTIFIER')
    return value

def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root: raise ValueError('PATH_ESCAPE')
    return path

def read_json(path):
    # on Windows a read can hit the instant atomic_json replaces the file: retry briefly, like the writer
    deadline=time.monotonic()+2
    while True:
        try: return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if time.monotonic()>deadline: raise
            time.sleep(0.02)

def atomic_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers/antivirus can briefly deny rename without corruption.
        deadline=time.monotonic()+2
        while True:
            try:
                os.replace(temp,path)
                break
            except PermissionError:
                if os.name!='nt' or time.monotonic()>=deadline: raise
                time.sleep(.01)
    finally:
        if os.path.exists(temp): os.unlink(temp)

@contextmanager
def lock(path, timeout=10):
    """Kernel releases locks on crash; the empty lock file is intentionally retained."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline: raise TimeoutError('STATE_LOCKED')
                time.sleep(0.05)
        try: yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.home = confined(self.root, '.its-fine')
        self.runs = confined(self.root, '_itsfine_runs')

    def state(self):
        return read_json(self.home / 'state.json')

    def initialize(self, user='user', language='pt-br'):
        with lock(self.home / 'state.lock'):
            if (self.home / 'state.json').exists(): return self.state()
            state = dict(schema_version=1, revision=1, user_name=user,
                         chat_language=language, doc_language='Português', active_run_id=None)
            atomic_json(self.home / 'state.json', state)
            return state

    def update_state(self, expected_revision, **changes):
        if set(changes) - {'active_run_id', 'user_name', 'chat_language', 'doc_language'}:
            raise ValueError('UNKNOWN_STATE_FIELD')
        if changes.get('active_run_id') is not None: identifier(changes['active_run_id'])
        with lock(self.home / 'state.lock'):
            old = self.state()
            if old['revision'] != expected_revision: raise ValueError('REVISION_CONFLICT')
            new = {**old, **changes, 'revision': old['revision'] + 1}
            history = confined(self.home, f'history/state-{old["revision"]}.json')
            if not history.exists(): atomic_json(history, old)
            atomic_json(self.home / 'state.json', new)
            return new

    def run_dir(self, run_id): return confined(self.runs, identifier(run_id))

    def put(self, run_id, artifact):
        from .contracts.validate import validate_artifact, DEPENDENCIES
        validate_artifact(artifact)
        kind = artifact['kind']
        if kind not in DEPENDENCIES: raise ValueError('NOT_A_RUN_ARTIFACT')
        folder = self.run_dir(run_id)
        with lock(folder / 'artifacts.lock'):
            current = self.artifacts(run_id)
            expected = {dep: digest(current[dep]) for dep in DEPENDENCIES[kind] if dep in current}
            if set(expected) != set(DEPENDENCIES[kind]) or artifact['input_hashes'] != expected:
                raise ValueError('STALE_INPUT:' + kind)
            previous = current.get(kind)
            revision = 1 if previous is None else previous['revision'] + 1
            if artifact['revision'] != revision: raise ValueError('REVISION_CONFLICT')
            job = folder / 'job.json'
            if (job.exists() or (folder/'remote-job.json').exists()) and kind in ('objective', 'environment', 'model', 'dataset', 'recipe'):
                raise ValueError('RUN_INPUTS_FROZEN: create a new run')
            for other in current.values():
                if other['id'] == artifact['id'] and other['kind'] != kind: raise ValueError('DUPLICATE_ID')
            path = confined(folder, f'artifacts/{kind}/{revision}.json')
            if path.exists(): raise ValueError('REVISION_EXISTS')
            # Revision files are the source of truth; there is no fallible second pointer update.
            atomic_json(path, artifact)
        return {'path': str(path), 'hash': digest(artifact)}

    def artifacts(self, run_id):
        folder = self.run_dir(run_id) / 'artifacts'
        result = {}
        if folder.exists():
            for kind in folder.iterdir():
                if kind.is_dir():
                    files = [p for p in kind.glob('*.json') if p.stem.isdigit()]
                    if files: result[kind.name] = read_json(max(files, key=lambda p: int(p.stem)))
        return result

    def route(self, run_id):
        from .contracts.validate import DEPENDENCIES
        artifacts = self.artifacts(run_id)
        folder=self.run_dir(run_id)
        if (folder/'remote-job.json').exists() and 'run' not in artifacts:
            remote = 'fine-kaggle' if read_json(folder/'remote-job.json').get('backend') == 'kaggle' else 'fine-colab'
            return {'next':remote,'reason':'remote-job-reconcile-and-import; never resubmit blindly'}
        if (folder/'job.json').exists() and 'run' not in artifacts:
            return {'next':'fine-resume','reason':'job-reconcile-before-new-training'}
        # Drafts break the metric/data/model cycle; published artifacts retain
        # their existing immutable dependency contract.
        if 'objective' not in artifacts and (folder/'intent.json').exists():
            stages=[('environment-draft.json','fine-environment'),('dataset/resumo.json','fine-dataset'),
                    ('model-draft.json','fine-model'),('prepared-v1/check.json','fine-dataset'),
                    ('metric-calibration.json','fine-metric'),('baseline/resumo.json','fine-metric')]
            for path,command in stages:
                if not (folder/path).exists(): return {'next':command,'reason':'draft-missing:'+path}
            if read_json(folder/'metric-calibration.json').get('valid') is not True:
                return {'next':'fine-metric','reason':'metric-calibration-invalid'}
            return {'next':'fine-objective','reason':'freeze-calibrated-objective-and-publish-drafts'}
        commands = dict(objective='fine-objective', environment='fine-environment', model='fine-model',
                        dataset='fine-dataset', recipe='fine-config', run='fine-train',
                        evaluation='fine-evaluate', export='fine-export', delivery='fine-deliver')
        for kind, deps in DEPENDENCIES.items():
            if kind not in artifacts:
                environment = artifacts.get('environment',{}).get('data',{})
                if kind=='run' and environment.get('mode')=='notebook':
                    return {'next':'fine-kaggle' if environment.get('destination')=='kaggle' else 'fine-colab','reason':'missing:run'}
                return {'next': commands[kind], 'reason': 'missing:' + kind}
            expected = {dep: digest(artifacts[dep]) for dep in deps}
            if artifacts[kind]['input_hashes'] != expected:
                return {'next': commands[kind], 'reason': 'stale:' + kind}
            if kind == 'run' and artifacts[kind]['data']['status'] != 'concluida':
                return {'next': 'fine-resume', 'reason': 'training-not-complete'}
            if kind == 'evaluation' and artifacts[kind]['data']['decision'] != 'aprovada':
                return {'next': 'fine-evaluate', 'reason': artifacts[kind]['data']['decision']}
        from .contracts.validate import validate_bundle
        validate_bundle(artifacts)
        return {'next': None, 'reason': 'complete'}
