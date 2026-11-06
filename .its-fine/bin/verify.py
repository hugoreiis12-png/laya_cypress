"""Verificador instalado: hashes gerenciados, skills, hook, CLI, schemas, presets e motores.

Roda dentro do projeto do usuario, sem depender do repositorio de desenvolvimento,
sem baixar pesos e sem treinar. Distingue arquivos integros, dependencias ausentes,
motor nao sondado e checkpoint certificado.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

ENGINE_DIRS = {'claude-code': ('.claude', '.agents'), 'kiro': ('.kiro', '.agents')}
PASTA_PADRAO = ('.agents',)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ler(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verificar_motor(root, nome, preparados):
    pasta = root / '.its-fine/engines' / nome
    info = ler(pasta / 'engine.json')
    estado = {'versao': info.get('version'), 'colab': info.get('colab', True), 'arquivos_integros': True,
              'preparado': nome in preparados, 'estado': 'arquivos_integros'}
    if nome == 'hf-encoder':
        sys.path.insert(0, str(root / '.its-fine/lib'))
        from itsfine.contracts.validate import validate_encoder
        # o checkpoint da receita vem do fine-model; o preset so fixa o resto
        exemplo = {'id': 'modelo/escolhido-no-fine-model', 'revision': 'commit-do-lock'}
        for preset in sorted((pasta / 'presets').glob('*.json')):
            validate_encoder('recipe', {'checkpoint': exemplo, **ler(preset)['recipe']}); validate_encoder('task', ler(preset)['task_exemplo'])
        validate_encoder('task', ler(pasta / 'examples/task.json'))
        certificados = ler(pasta / 'certification.json')['checkpoints'] if (pasta / 'certification.json').is_file() else []
        estado['sugestoes'] = [s['id'] for s in ler(pasta / info['checkpoints'])['sugestoes']]
        estado['certificados'] = [{'id': c['id'], 'revision': c['revision'][:10], 'layout': c.get('layout'),
                                   'paridade': (c.get('paridade') or {}).get('ok'), 'gpu': c.get('gpu'), 'em': c.get('data')}
                                  for c in certificados if c.get('download', {}).get('conferido_contra_hub')]
    if nome not in preparados:
        estado['estado'] = 'dependencias_ausentes'
    else:
        estado['estado'] = 'motor_nao_sondado'
        registro = ler(root / '.its-fine/engine-setup.json')
        bloco = registro.get(nome) or registro
        estado['versoes'] = bloco.get('versoes')
    return estado


def verify(root):
    root = Path(root).resolve()
    manifest = ler(root / '.its-fine/install-manifest.json')
    for name, expected in manifest['files'].items():
        target = root / name
        if not target.is_file(): raise AssertionError('Arquivo gerenciado ausente: ' + name)
        if file_hash(target) != expected: raise AssertionError('Arquivo gerenciado alterado: ' + name)
    engines = manifest.get('engines', ['claude-code', 'codex'])
    folders = manifest.get('folders') or sorted({p for name in engines for p in ENGINE_DIRS.get(name, PASTA_PADRAO)})
    conjuntos = {}
    for folder in folders:
        installed = sorted((root / folder / 'skills').glob('fine-*/SKILL.md'))
        if not installed: raise AssertionError('Nenhuma skill instalada em ' + folder)
        conjuntos[folder] = {skill.parent.name: file_hash(skill) for skill in installed}
        for skill in installed:
            for reference in re.findall(r'\]\(([^)]+)\)', skill.read_text(encoding='utf-8')):
                if not (skill.parent / reference).is_file(): raise AssertionError('Referencia quebrada: ' + reference)
    if len({json.dumps(v, sort_keys=True) for v in conjuntos.values()}) != 1:
        raise AssertionError('Conjuntos de skills divergentes entre harness (nome ou conteudo)')
    report = {'skills': len(next(iter(conjuntos.values()))), 'engines': engines, 'folders': folders,
              'managed_files': len(manifest['files'])}
    if 'claude-code' in engines:
        settings = ler(root / '.claude/settings.json')
        configured = [hook['command'] for entry in settings['hooks']['PreToolUse'] for hook in entry['hooks']]
        if configured.count(manifest['hook_command']) != 1: raise AssertionError('Hook nao configurado uma unica vez')
        probe = subprocess.run(['node', str(root / '.its-fine/hooks/file-policy.mjs')],
                               input=json.dumps(dict(cwd=str(root), tool_name='Write',
                                                     tool_input={'file_path': '.its-fine/policy.json'})),
                               capture_output=True, text=True)
        if probe.returncode:
            report['hook'] = 'configurado; nao foi possivel invocar o node para a prova de recusa'
        else:
            decision = json.loads(probe.stdout)['hookSpecificOutput']['permissionDecision']
            if decision != 'deny': raise AssertionError('Hook nao recusou escrita na politica')
            report['hook'] = 'configurado e recusa verificada; intercepcao ao vivo nao certificada'
    else:
        report['hook'] = 'nao aplicavel: Claude Code fora dos harness instalados'
    cli = subprocess.run([sys.executable, str(root / '.its-fine/bin/fine.py'), '--root', str(root), '--help'],
                         capture_output=True, text=True)
    if cli.returncode: raise AssertionError('CLI falhou: ' + cli.stderr.strip())
    report['cli'] = 'ok'
    contratos = root / '.its-fine/lib/itsfine/contracts'
    for schema in ('artifact.schema.json', 'encoder.schema.json'):
        if not (contratos / schema).is_file(): raise AssertionError('Schema ausente: ' + schema)
        ler(contratos / schema)
    report['schemas'] = ['artifact.schema.json (v1 e v2)', 'encoder.schema.json']
    setup = root / '.its-fine/engine-setup.json'
    registro = ler(setup) if setup.is_file() else {}
    preparados = set(registro.get('engines_prontos') or (['hf-peft'] if registro.get('pronto') else []))
    report['motores'] = {nome: verificar_motor(root, nome, preparados)
                         for nome in manifest.get('training_engines', ['hf-peft']) if (root / '.its-fine/engines' / nome / 'engine.json').is_file()}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    print(json.dumps(verify(parser.parse_args().root), ensure_ascii=False, indent=2))
