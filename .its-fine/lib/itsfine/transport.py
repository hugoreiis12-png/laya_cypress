"""Verified result archives and portable notebook/SSH handoffs."""
from pathlib import Path
import base64
import io
import json
import os
import re
import shlex
import subprocess
import tempfile
import zipfile
from .storage import Store, atomic_json, confined, digest, file_hash, identifier, read_json
from .execution import validate_spec

def pack_result(root, run_id, target, checkpoints=True):
    """checkpoints=False leaves output/checkpoint-*/ behind (they only serve to resume where the job ran);
    the manifest still records their hashes as remote_only."""
    store = Store(root); folder = store.run_dir(run_id)
    job = read_json(folder / 'job.json')
    if job['status'] not in ('concluida','falhou','interrompida'): raise ValueError('JOB_NOT_TERMINAL')
    target = Path(target).resolve()
    if target.is_relative_to(folder): raise ValueError('ARCHIVE_INSIDE_RUN')
    files = []; remote_only = []
    for path in sorted(folder.rglob('*')):
        if path.is_symlink(): raise ValueError('LINK_IN_RESULT')
        parts = path.relative_to(folder).parts
        # inputs/ is the snapshot the sender already holds, bound by spec_hash; do not ship it back
        if parts[0] == 'inputs': continue
        if path.is_file() and path.suffix != '.lock':
            confined(folder, path.relative_to(folder))
            entry = {'path':path.relative_to(folder).as_posix(),'sha256':file_hash(path),'size':path.stat().st_size}
            left = not checkpoints and len(parts) > 2 and parts[0] == 'output' and parts[1].startswith('checkpoint-')
            (remote_only if left else files).append(entry)
    manifest = dict(schema_version=1, run_id=run_id, spec_hash=job['spec_hash'], status=job['status'], files=files)
    if remote_only: manifest['remote_only'] = remote_only
    if target.exists(): raise FileExistsError('ARCHIVE_EXISTS')
    # temp name + rename: an interrupted pack never leaves a truncated zip under the final name
    partial = target.with_name(target.name + '.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('result-manifest.json', json.dumps(manifest))
        for entry in files:
            # weights barely compress; storing them keeps packing fast enough for a tunnel request
            binary = Path(entry['path']).suffix in ('.safetensors', '.pt', '.pth', '.bin', '.gguf')
            archive.write(folder / entry['path'], 'files/'+entry['path'], compress_type=zipfile.ZIP_STORED if binary else None)
    os.replace(partial, target)
    return {'path':str(target), 'sha256':file_hash(target), 'spec_hash':job['spec_hash']}

def import_result(root, archive_path, run_id, expected_spec_hash, max_bytes=10 * 1024**3):
    store = Store(root); identifier(run_id)
    target = confined(store.run_dir(run_id), 'imports/' + file_hash(archive_path))
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names): raise ValueError('DUPLICATE_ZIP_MEMBER')
        if sum(info.file_size for info in archive.infolist()) > max_bytes: raise ValueError('RESULT_TOO_LARGE')
        manifest = json.loads(archive.read('result-manifest.json'))
        if manifest.get('schema_version') != 1 or manifest.get('run_id') != run_id or manifest.get('spec_hash') != expected_spec_hash:
            raise ValueError('RESULT_IDENTITY_MISMATCH')
        if manifest.get('status') not in ('concluida','falhou','interrompida'): raise ValueError('RESULT_NOT_TERMINAL')
        expected = {'result-manifest.json'}
        verified = []
        import hashlib
        for entry in manifest['files']:
            relative = entry['path']
            if '\\' in relative or relative.startswith('/') or any(p in ('','..','.') for p in relative.split('/')):
                raise ValueError('PATH_ESCAPE')
            destination = confined(target, relative)
            name = 'files/' + relative
            if name in expected: raise ValueError('DUPLICATE_RESULT_PATH')
            expected.add(name)
            sha=hashlib.sha256(); size=0
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    size += len(chunk); sha.update(chunk)
            if size != entry['size'] or sha.hexdigest() != entry['sha256']:
                raise ValueError('RESULT_HASH_MISMATCH')
            verified.append((relative, name))
        if set(names) != expected: raise ValueError('UNDECLARED_RESULT_FILE')
        # Validate the whole archive before writing, with a per-import lock.
        from .storage import lock
        with lock(target.parent / (target.name+'.lock')):
            if target.exists():
                if read_json(target/'import-manifest.json') != manifest: raise ValueError('EXISTING_IMPORT_CONFLICT')
                for entry in manifest['files']:
                    if file_hash(confined(target,entry['path'])) != entry['sha256']: raise ValueError('EXISTING_IMPORT_MODIFIED')
                return {'path':str(target), 'already_imported':True}
            staging=Path(tempfile.mkdtemp(prefix='.import-',dir=target.parent)).resolve()
            if not staging.is_relative_to(target.parent.resolve()): raise ValueError('PATH_ESCAPE')
            import shutil
            for relative, name in verified:
                destination=confined(staging,relative); destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source, destination.open('xb') as out: shutil.copyfileobj(source,out)
            atomic_json(staging / 'import-manifest.json', manifest)
            os.rename(staging,target)
    return {'path':str(target), 'status':manifest['status'], 'files':len(verified)}

def portable_payload(root, spec):
    validate_spec(root, spec)
    if sum(confined(root,entry['path']).stat().st_size for entry in spec['inputs']) > 32 * 1024**2:
        raise ValueError('NOTEBOOK_INPUTS_TOO_LARGE: use an authorized transfer instead of embedding more than 32 MiB')
    memory = io.BytesIO()
    rewritten = {**spec, 'inputs':[]}
    with zipfile.ZipFile(memory, 'w', zipfile.ZIP_DEFLATED) as archive:
        package = Path(__file__).resolve().parent
        for source in sorted(package.rglob('*')):
            if source.is_file() and source.suffix in ('.py','.json') and '__pycache__' not in source.parts:
                archive.write(source, 'itsfine/'+source.relative_to(package).as_posix())
        for entry in spec['inputs']:
            relative = 'inputs/'+entry['name']
            archive.write(confined(root, entry['path']), relative)
            rewritten['inputs'].append({**entry,'path':relative})
        # Process argv must refer to portable input paths; original absolute host paths are not rewritten silently.
        archive.writestr('execution.json', json.dumps(rewritten))
    return memory.getvalue(), rewritten

def notebook(root, run_id, spec, destination):
    identifier(run_id)
    if spec['executor'] != 'process': raise ValueError('NOTEBOOK_REQUIRES_PROCESS_SPEC')
    payload, rewritten = portable_payload(root, spec)
    encoded = base64.b64encode(payload).decode()
    cells = [
        {'cell_type':'markdown','metadata':{},'source':[
            '# ITS-FINE: execução em notebook\n',
            'Este arquivo contém os inputs selecionados. Envie-o somente ao destino autorizado.\n',
            'Instale a engine e jsonschema 4.x no ambiente escolhido. Revise execution.json antes de executar.\n',
            'Execução no host do notebook não é sandbox de filesystem. O limite de duração depende do supervisor permanecer ativo.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':[
            'import base64, io, json, pathlib, sys, zipfile\n',
            "workspace = pathlib.Path('its-fine-workspace').resolve()\n",
            "if workspace.exists(): raise RuntimeError('Use uma pasta nova ou retome explicitamente a anterior')\n",
            'workspace.mkdir()\n',
            f"zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))).extractall(workspace)\n",
            'sys.path.insert(0, str(workspace))\n',
            'from itsfine.storage import Store, atomic_json, read_json\n',
            'from itsfine.execution import authorize, submit, status\n',
            "Store(workspace).initialize()\n",
            "spec = read_json(workspace / 'execution.json')\n", 'print(json.dumps(spec, indent=2))\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':[
            'ALLOW_HOST_EXECUTION = False  # Altere conscientemente após revisar o script e os dados.\n',
            "if not ALLOW_HOST_EXECUTION: raise RuntimeError('Aprovação de execução pendente')\n",
            "atomic_json(workspace / '.its-fine/policy.json', {'schema_version':1, 'allow_process':True})\n",
            "authorize(workspace, spec, actor='notebook-user', acknowledge_process=True)\n",
            f"job = submit(workspace, {run_id!r}, spec)\n", 'print(job)\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':[
            f"print(status(workspace, {run_id!r}))\n",
            '# Reexecute esta célula para observar; não crie outro job para a mesma rodada.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':[
            'from itsfine.transport import pack_result\n',
            f"print(pack_result(workspace, {run_id!r}, workspace.parent / ({run_id!r} + '-result.zip')))\n"]},
    ]
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('NOTEBOOK_EXISTS')
    atomic_json(destination, {'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':cells})
    return {'path':str(destination),'expected_spec_hash':digest(rewritten), 'embedded_input_bytes':len(payload)}

def remote_request(host, remote_root, action, run_id, spec_path='execution.json', timeout=30):
    """No upload, provider allocation, credentials or shell interpolation of user input."""
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.@-]*', host): raise ValueError('INVALID_SSH_HOST')
    identifier(run_id)
    if not remote_root.startswith('/') or '\x00' in remote_root: raise ValueError('ABSOLUTE_REMOTE_ROOT_REQUIRED')
    if action not in ('status','cancel','submit'): raise ValueError('INVALID_REMOTE_ACTION')
    argv = ['python3','-m','itsfine','--root',remote_root,action,run_id]
    if action == 'submit': argv.append(spec_path)
    command = 'cd ' + shlex.quote(remote_root) + ' && ' + shlex.join(argv)
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','--',host,command],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode: raise RuntimeError('REMOTE_UNAVAILABLE_OR_FAILED: ' + result.stderr.strip())
    return json.loads(result.stdout)
