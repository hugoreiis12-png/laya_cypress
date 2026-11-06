"""Idempotent local supervisor. Docker isolates writes; process mode is opt-in."""
from pathlib import Path
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from .storage import Store, atomic_json, confined, digest, file_hash, lock, read_json
from .privacy import load_policy, require_local_executor

TERMINAL = {'concluida', 'falhou', 'interrompida'}

def validate_spec(root, spec):
    if not isinstance(spec, dict) or spec.get('schema_version') != 1: raise ValueError('INVALID_EXECUTION_SPEC')
    if spec.get('executor') not in ('docker', 'process'): raise ValueError('UNSUPPORTED_EXECUTOR')
    argv = spec.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a or '\x00' in a for a in argv):
        raise ValueError('INVALID_ARGV')
    duration = spec.get('timeout_seconds')
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('INVALID_TIMEOUT')
    inputs = spec.get('inputs')
    if not isinstance(inputs, list) or not inputs: raise ValueError('INPUTS_REQUIRED')
    names = set()
    for entry in inputs:
        if not isinstance(entry, dict) or not isinstance(entry.get('name'),str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_./-]{0,240}', entry['name']) or any(p in ('','..','.') for p in entry['name'].split('/')):
            raise ValueError('INVALID_INPUT_NAME')
        if entry['name'] in names: raise ValueError('DUPLICATE_INPUT_NAME')
        names.add(entry['name'])
        source = confined(root, entry['path'])
        if not source.is_file() or file_hash(source) != entry.get('sha256'): raise ValueError('INPUT_HASH_MISMATCH')
    if spec['executor'] == 'docker':
        if not re.fullmatch(r'[^\s]+@sha256:[a-f0-9]{64}', spec.get('image', '')): raise ValueError('PINNED_IMAGE_REQUIRED')
        if not isinstance(spec.get('gpus', False), bool): raise ValueError('INVALID_GPU_FLAG')
    if spec.get('metered', False):
        # A provider must enforce a monetary cap. This local adapter cannot attest that.
        raise ValueError('METERED_EXECUTION_REQUIRES_PROVIDER_BUDGET_ADAPTER')
    return digest(spec)

def authorize(root, spec, actor, acknowledge_process=False):
    store = Store(root)
    policy = load_policy(store.root)
    require_local_executor(policy)
    spec_hash = validate_spec(store.root, spec)
    if not isinstance(actor, str) or not actor.strip(): raise ValueError('ACTOR_REQUIRED')
    if spec['executor'] == 'process':
        if policy.get('allow_process') is not True or not acknowledge_process:
            raise ValueError('PROCESS_MODE_NOT_AUTHORIZED: no filesystem sandbox')
    record = dict(schema_version=1, action='execute', actor=actor,
                  spec_hash=spec_hash, acknowledged_process=acknowledge_process, at=time.time())
    atomic_json(confined(store.home, f'approvals/{spec_hash}.json'), record)
    return record

def approved(store, spec):
    policy = load_policy(store.root)
    require_local_executor(policy)
    spec_hash = validate_spec(store.root, spec)
    record = read_json(confined(store.home, f'approvals/{spec_hash}.json'))
    if record.get('spec_hash') != spec_hash or record.get('action') != 'execute': raise ValueError('APPROVAL_REQUIRED')
    if spec['executor'] == 'process':
        if policy.get('allow_process') is not True or record.get('acknowledged_process') is not True:
            raise ValueError('PROCESS_MODE_NOT_AUTHORIZED')
    return spec_hash

def submit(root, run_id, spec):
    store = Store(root)
    spec_hash = approved(store, spec)
    folder = store.run_dir(run_id)
    with lock(folder / 'job.lock'):
        jobfile = folder / 'job.json'
        if jobfile.exists():
            job = read_json(jobfile)
            if job['spec_hash'] != spec_hash: raise ValueError('RUN_INPUTS_FROZEN')
            return job  # also for ambiguous submission or terminal jobs: never replay
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'execution.json', spec)
        job = dict(schema_version=1, run_id=run_id, spec_hash=spec_hash,
                   submission_key=digest({'root':str(store.root), 'run_id':run_id, 'spec_hash':spec_hash}),
                   status='submetendo', submitted_at=time.time(), heartbeat=None)
        atomic_json(jobfile, job)  # durable intent before spawning any process
        module = Path(__file__).resolve().parent.parent
        code = 'import sys; sys.path.insert(0,sys.argv[1]); from itsfine.execution import work; work(sys.argv[2],sys.argv[3])'
        args = [sys.executable, '-c', code, str(module), str(store.root), run_id]
        options = {'start_new_session': True} if os.name != 'nt' else {
            'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
        try:
            with (folder / 'supervisor.log').open('ab') as log:
                supervisor = subprocess.Popen(args, cwd=store.root, stdin=subprocess.DEVNULL,
                                              stdout=log, stderr=log, close_fds=True, **options)
                # Reap while the submitting process lives; do not tie job lifetime to it.
                threading.Thread(target=supervisor.wait, daemon=True).start()
        except OSError as exc:
            job.update(status='falhou', error=type(exc).__name__, finished_at=time.time())
            atomic_json(jobfile, job)
            raise
    return job

def command_for(store, folder, spec, key):
    from .engines import resolve_python
    replacements = {'{python}': resolve_python(store.home, spec.get('python',sys.executable)) if spec['executor'] == 'process' else 'python',
                    '{inputs}':str(folder/'inputs') if spec['executor']=='process' else '/inputs',
                    '{output}': str(confined(folder, 'output')) if spec['executor'] == 'process' else '/outputs'}
    for entry in spec['inputs']:
        replacements['{input:'+entry['name']+'}'] = str(confined(folder, 'inputs/'+entry['name'])) if spec['executor'] == 'process' else '/inputs/'+entry['name']
    argv = []
    for arg in spec['argv']:
        for marker, value in replacements.items(): arg = arg.replace(marker, value)
        if '{input:' in arg: raise ValueError('UNKNOWN_INPUT_PLACEHOLDER')
        argv.append(arg)
    if spec['executor'] == 'process': return argv
    output = confined(folder, 'output')
    output.mkdir(exist_ok=True)
    args = ['docker', 'run', '--rm', '--pull', 'never', '--name', 'itsfine-' + key[:24],
            '--read-only', '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--pids-limit', '256', '--tmpfs', '/tmp:rw,nosuid,size=1g',
            '--mount', f'type=bind,source={output},target=/outputs',
            '--workdir', '/outputs', '--env', 'ITS_FINE_OUTPUT=/outputs']
    if spec.get('gpus'): args += ['--gpus', 'all']
    if spec.get('engine_job'): args += ['--env','ITS_FINE_SUPERVISED_CONTAINER=1']
    for entry in spec['inputs']:
        source = confined(folder, 'inputs/'+entry['name'])
        if ',' in str(source): raise ValueError('DOCKER_MOUNT_PATH_CONTAINS_COMMA')
        args += ['--mount', f'type=bind,source={source},target=/inputs/{entry["name"]},readonly']
    if ',' in str(output): raise ValueError('DOCKER_MOUNT_PATH_CONTAINS_COMMA')
    return args + [spec['image']] + argv

def terminate_tree(process, spec, key):
    if spec['executor'] == 'docker':
        result = subprocess.run(['docker','rm','-f','itsfine-'+key[:24]], capture_output=True, timeout=30)
        if result.returncode and process.poll() is None: raise RuntimeError('STOP_NOT_CONFIRMED')
    if process.poll() is not None: return
    if os.name == 'nt':
        result = subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'], capture_output=True, timeout=15)
        if result.returncode and process.poll() is None: raise RuntimeError('STOP_NOT_CONFIRMED')
    else: os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)

def work(root, run_id):
    store = Store(root)
    folder = store.run_dir(run_id)
    with lock(folder / 'worker.lock', timeout=0):
        with lock(folder / 'job.lock'):
            job = read_json(folder / 'job.json')
            if job['status'] != 'submetendo': return
            spec = read_json(folder / 'execution.json')
            try:
                if approved(store, spec) != job['spec_hash']: raise ValueError('SPEC_CHANGED')
                for entry in spec['inputs']:
                    snapshot = confined(folder, 'inputs/'+entry['name'])
                    snapshot.parent.mkdir(parents=True,exist_ok=True)
                    with snapshot.open('xb') as destination, confined(store.root,entry['path']).open('rb') as source:
                        shutil.copyfileobj(source,destination)
                    if file_hash(snapshot) != entry['sha256']: raise ValueError('INPUT_CHANGED_DURING_SNAPSHOT')
                    # copying large inputs takes time: keep the heartbeat so status does not read stale
                    job['heartbeat'] = time.time(); atomic_json(folder / 'job.json', job)
                command = command_for(store, folder, spec, job['submission_key'])
            except Exception as exc:
                job.update(status='falhou', error=type(exc).__name__, finished_at=time.time())
                atomic_json(folder / 'job.json', job); return
            job.update(status='executando', supervisor_pid=os.getpid(), heartbeat=time.time())
            atomic_json(folder / 'job.json', job)
        process = None
        started = time.monotonic()
        try:
            env = os.environ.copy()
            output = confined(folder, 'output'); output.mkdir(exist_ok=True)
            env['ITS_FINE_OUTPUT'] = str(output)
            env['ITS_FINE_JOB_ROOT']=str(store.root)
            env['ITS_FINE_RUN_ID']=run_id
            options = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
            with (folder / 'execution.log').open('ab') as log:
                process = subprocess.Popen(command, cwd=output, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=log, **options)
                while process.poll() is None:
                    cancelled = (folder / 'cancel.json').exists()
                    expired = time.monotonic() - started >= spec['timeout_seconds']
                    if cancelled or expired:
                        terminate_tree(process, spec, job['submission_key'])
                        job.update(status='interrompida', reason='cancelled' if cancelled else 'timeout')
                        break
                    job.update(heartbeat=time.time(), pid=process.pid)
                    atomic_json(folder / 'job.json', job)
                    time.sleep(0.1)
                if job['status'] != 'interrompida':
                    job['status'] = 'concluida' if process.returncode == 0 else 'falhou'
                job['exit_code'] = process.returncode
        except Exception as exc:
            job.update(error=type(exc).__name__, status='falhou')
            if process is not None and process.poll() is None:
                try: terminate_tree(process, spec, job['submission_key'])
                except Exception: job['status'] = 'desconhecida'
        job.update(finished_at=time.time(), heartbeat=time.time())
        atomic_json(folder / 'job.json', job)

def error_summary(folder, job, full=False):
    """Where a failed job broke: exception type and file:line of the last traceback frames.
    Messages can quote data, so they come only with full=True (allow_data_to_harness)."""
    summary = {'tipo': job.get('error'), 'exit_code': job.get('exit_code'), 'local': []}
    log = folder / 'execution.log'
    lines = log.read_text(encoding='utf-8', errors='replace').splitlines() if log.exists() else []
    # a progress bar can leave the header glued to the end of its line
    starts = [i for i, line in enumerate(lines) if 'Traceback (most recent call last)' in line]
    if starts:
        frames = []; after = starts[-1] + 1
        for i in range(starts[-1] + 1, len(lines)):
            frame = re.match(r'\s+File "(.+)", line (\d+), in (\S+)', lines[i])
            if frame:
                frames.append('%s:%s em %s' % (re.split(r'[\\/]', frame.group(1))[-1], frame.group(2), frame.group(3))); after = i + 1
        summary['local'] = frames[-4:]
        # only the first unindented line after the last frame, and only if it names an exception:
        # anything else could be data
        final = next((line for line in lines[after:] if line and not line[:1].isspace()), '')
        name = re.match(r'([A-Za-z_][\w.]*)(?::\s?(.*))?$', final)
        if name and re.search(r'(Error|Exception|Interrupt|Exit|Iteration|Warning)$', name.group(1)):
            summary['tipo'] = name.group(1)
            if full: summary['mensagem'] = name.group(2)
    if full and lines: summary['fim_do_log'] = lines[-20:]
    return summary

def status(root, run_id, full=False):
    folder = Store(root).run_dir(run_id)
    if not (folder/'job.json').exists() and (folder/'remote-job.json').exists():
        remote=read_json(folder/'remote-job.json')
        return {'run_id':run_id,'spec_hash':remote['spec_hash'],'status':'desconhecida',
                'reason':'remote observation required; use %s status' % ('fine-kaggle' if remote.get('backend')=='kaggle' else 'fine-colab')}
    job = read_json(folder / 'job.json')
    heartbeat = job.get('heartbeat') or job['submitted_at']
    if job['status'] not in TERMINAL and time.time() - heartbeat > 30:
        return {**job, 'status':'desconhecida', 'reason':'stale-heartbeat; reconcile before any new submission'}
    from .progress import read as read_progress
    progress = read_progress(folder)
    if progress: job = {**job, 'progresso': progress}
    if job['status'] == 'falhou': return {**job, 'erro': error_summary(folder, job, full)}
    return job

def cancel(root, run_id):
    folder = Store(root).run_dir(run_id)
    job = status(root, run_id)
    if job['status'] not in TERMINAL: atomic_json(folder / 'cancel.json', {'requested_at':time.time()})
    return {'requested': job['status'] not in TERMINAL, 'status':job['status']}


def parent_pid(pid):
    """Windows only: parent of any process. A venv python.exe is a launcher that runs the
    real interpreter as its child, so the engine's parent is the launcher, not the supervisor."""
    if os.name != 'nt': return None
    import ctypes
    import ctypes.wintypes as w
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD), ('th32ProcessID', w.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                    ('th32ModuleID', w.DWORD), ('cntThreads', w.DWORD), ('th32ParentProcessID', w.DWORD), ('pcPriClassBase', ctypes.c_long),
                    ('dwFlags', w.DWORD), ('szExeFile', ctypes.c_char * 260)]
    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel.Process32First.argtypes = kernel.Process32Next.argtypes = (ctypes.c_void_p, ctypes.POINTER(Entry))
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value: return None
    try:
        entry = Entry(); entry.dwSize = ctypes.sizeof(Entry)
        found = kernel.Process32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32ProcessID == pid: return int(entry.th32ParentProcessID)
            found = kernel.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return None


def require_supervisor():
    if os.environ.get('ITS_FINE_SUPERVISED_CONTAINER')=='1' and Path('/.dockerenv').exists(): return
    root=os.environ.get('ITS_FINE_JOB_ROOT'); run_id=os.environ.get('ITS_FINE_RUN_ID')
    if not root or not run_id: raise PermissionError('SUPERVISED_EXECUTION_REQUIRED')
    store=Store(root); folder=store.run_dir(run_id)
    job=read_json(folder/'job.json'); spec=read_json(folder/'execution.json')
    parent=os.getppid()
    if job.get('status')!='executando' or job.get('supervisor_pid') not in (parent, parent_pid(parent)) or approved(store,spec)!=job['spec_hash']:
        raise PermissionError('SUPERVISOR_IDENTITY_MISMATCH')
