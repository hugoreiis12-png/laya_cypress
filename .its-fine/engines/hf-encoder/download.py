"""Baixa um modelo do Hugging Face para o projeto, conferindo cada arquivo contra o proprio Hub.

Serve para qualquer repositorio de encoder. A revisao pedida (branch ou commit) vira
commit. Entram configuracoes, tokenizer e pesos: safetensors, ou pytorch_model*.bin so
quando nao ha safetensors. README, flax, tf e onnx ficam de fora. Cada arquivo e
conferido durante o download contra o que o Hub declara (sha256 do LFS ou sha1 do blob
git); divergencia apaga tudo. No fim grava itsfine-download.json (id, commit, sha256 por
arquivo), que o loader confere antes de treinar, avaliar ou exportar.

Sem --autorizado so mostra o plano e sai com codigo 2: baixar exige autorizacao do
usuario, como setup.py --install-torch. --conferir registra uma pasta ja baixada por
outro meio, sem baixar de novo: confere cada arquivo contra o Hub e grava o lock.
Repositorio fechado usa HF_TOKEN do ambiente.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json  # noqa: E402
import loader  # noqa: E402

HUB = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
EXTENSOES = ('.json', '.txt', '.model', '.safetensors')
PASTAS_FORA = {'onnx', 'openvino', 'coreml', 'tflite', 'flax', 'tf'}


def abrir(url, timeout=60):
    token = os.environ.get('HF_TOKEN')
    return urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token} if token else {}), timeout=timeout)


def commit(ident, revision):
    with abrir('%s/api/models/%s/revision/%s' % (HUB, ident, urllib.parse.quote(revision, safe=''))) as resposta:
        return json.load(resposta)['sha']


def arvore(ident, sha):
    """Arquivos do repositorio no commit, com oid git e, quando LFS, sha256."""
    url, arquivos = '%s/api/models/%s/tree/%s?recursive=true' % (HUB, ident, sha), []
    while url:
        with abrir(url) as resposta:
            arquivos += [f for f in json.load(resposta) if f.get('type') == 'file']
            proximo = [p for p in (resposta.headers.get('Link') or '').split(',') if 'rel="next"' in p]
        url = proximo[0].split(';')[0].strip(' <>') if proximo else None
    return arquivos


def selecionar(arquivos):
    candidatos = [f for f in arquivos if not any(p.startswith('.') or p in PASTAS_FORA for p in PurePosixPath(f['path']).parts)]
    tem_safetensors = any(f['path'].endswith('.safetensors') for f in candidatos)
    escolhidos = [f for f in candidatos if PurePosixPath(f['path']).suffix in EXTENSOES or
                  (not tem_safetensors and PurePosixPath(f['path']).name.startswith('pytorch_model') and f['path'].endswith('.bin'))]
    if not any(f['path'].endswith(('.safetensors', '.bin')) for f in escolhidos): raise ValueError('HUB_WEIGHTS_MISSING')
    return sorted(escolhidos, key=lambda f: f['path'])


def confere_com_hub(info, sha256, sha1_git):
    lfs = info.get('lfs')
    return sha256 == lfs['oid'] if lfs else sha1_git == info['oid']


def hashes(fluxo, tamanho, destino=None):
    """sha256 e sha1 de blob git do conteudo; grava em destino quando dado."""
    sha256, sha1 = hashlib.sha256(), hashlib.sha1(b'blob %d\0' % tamanho)
    saida = open(destino, 'wb') if destino else None
    try:
        for bloco in iter(lambda: fluxo.read(1024 * 1024), b''):
            sha256.update(bloco); sha1.update(bloco)
            if saida: saida.write(bloco)
    finally:
        if saida: saida.close()
    return sha256.hexdigest(), sha1.hexdigest()


def gravar_lock(pasta, ident, sha, pedida, conferidos, modo):
    lock = {'schema_version': 1, 'id': ident, 'revision': sha, 'revisao_pedida': pedida, 'modo': modo,
            'conferido_contra_hub': True, 'files': {f['path']: h for f, h in conferidos},
            'hub': {f['path']: {'oid': f['oid'], 'lfs': bool(f.get('lfs'))} for f, _ in conferidos}}
    gravar_json(Path(pasta) / loader.LOCK, lock)
    lock['layout'] = loader.layout(pasta)
    return lock


def baixar(ident, sha, pedida, saida, arquivos):
    saida = Path(saida)
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('DOWNLOAD_OUTPUT_NOT_EMPTY')
    temp = saida.with_name('.' + saida.name + '.download')
    shutil.rmtree(temp, ignore_errors=True)
    conferidos = []
    try:
        for info in arquivos:
            destino = temp / info['path']; destino.parent.mkdir(parents=True, exist_ok=True)
            url = '%s/%s/resolve/%s/%s' % (HUB, ident, sha, urllib.parse.quote(info['path']))
            with abrir(url, timeout=600) as resposta:
                sha256, sha1 = hashes(resposta, int(info['size']), destino)
            if not confere_com_hub(info, sha256, sha1): raise ValueError('HUB_FILE_HASH_MISMATCH:' + info['path'])
            conferidos.append((info, sha256))
        if saida.exists(): saida.rmdir()
        saida.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, saida)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return gravar_lock(saida, ident, sha, pedida, conferidos, 'download')


def conferir(ident, sha, pedida, pasta, arvore_hub, arquivos):
    """Pasta ja baixada: os arquivos escolhidos precisam existir; qualquer outro precisa ser do Hub. Tudo com o hash do Hub."""
    pasta = Path(pasta)
    por_nome = {f['path']: f for f in arvore_hub}
    locais = loader.arquivos_do_modelo(pasta)
    faltam = sorted({f['path'] for f in arquivos} - set(locais))
    if faltam: raise ValueError('CHECKPOINT_FILE_MISSING:%s' % faltam[:5])
    estranhos = sorted(n for n in locais if n not in por_nome)
    if estranhos: raise ValueError('CHECKPOINT_FILE_UNEXPECTED:%s' % estranhos[:5])
    conferidos = []
    for nome in locais:
        info = por_nome[nome]
        with (pasta / nome).open('rb') as fluxo:
            sha256, sha1 = hashes(fluxo, (pasta / nome).stat().st_size)
        if not confere_com_hub(info, sha256, sha1): raise ValueError('CHECKPOINT_FILE_HASH_MISMATCH:' + nome)
        conferidos.append((info, sha256))
    return gravar_lock(pasta, ident, sha, pedida, conferidos, 'conferencia')


def main():
    parser = argparse.ArgumentParser(description='baixa ou confere um modelo do Hugging Face contra o Hub e grava itsfine-download.json')
    parser.add_argument('--id', required=True, help='repositorio no Hugging Face, ex.: jhu-clsp/mmBERT-base')
    parser.add_argument('--revision', default='main', help='branch ou commit; o lock grava sempre o commit')
    parser.add_argument('--saida', required=True, help='pasta do modelo, ex.: .its-fine/models/<nome>')
    parser.add_argument('--autorizado', action='store_true', help='executa o download previamente autorizado pelo usuario')
    parser.add_argument('--conferir', action='store_true', help='nao baixa: confere a pasta existente contra o Hub e grava o lock')
    args = parser.parse_args()
    sha = commit(args.id, args.revision)
    arvore_hub = arvore(args.id, sha)
    arquivos = selecionar(arvore_hub)
    plano = {'id': args.id, 'revision': sha, 'revisao_pedida': args.revision, 'saida': str(Path(args.saida).resolve()),
             'arquivos': {f['path']: int(f['size']) for f in arquivos}, 'bytes': sum(int(f['size']) for f in arquivos)}
    if args.conferir:
        lock = conferir(args.id, sha, args.revision, args.saida, arvore_hub, arquivos)
    elif not args.autorizado:
        print(json.dumps({**plano, 'status': 'DOWNLOAD_AUTHORIZATION_REQUIRED',
                          'proximo_passo': 'repita com --autorizado apos o usuario aprovar o download'}, ensure_ascii=False, indent=2))
        return 2
    else:
        lock = baixar(args.id, sha, args.revision, args.saida, arquivos)
    print(json.dumps({**plano, 'status': 'ok', 'modo': lock['modo'], 'layout': lock['layout'], 'arquivos_no_lock': len(lock['files'])},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
