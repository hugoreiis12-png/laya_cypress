"""Prepara o hf-encoder na mesma venv do hf-peft, sem tocar no Python do usuario.

As faixas de dependencia compartilhadas precisam ser identicas as do hf-peft;
divergencia e DEPENDENCY_CONFLICT e nada e instalado. Registra os motores
preparados em .its-fine/engine-setup.json, preservando o que o hf-peft gravou.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

AQUI = Path(__file__).resolve().parent


def executar(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[-3000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, type(exc).__name__


def primeiro_json(texto):
    for linha in texto.splitlines():
        linha = linha.strip()
        if linha.startswith('{'):
            try: return json.loads(linha)
            except ValueError: continue
    return None


def nome_do_pacote(spec):
    for sep in ('>=', '==', '<', '>', '~=', '!='):
        spec = spec.split(sep)[0]
    return spec.strip().lower()


def conflitos(pacotes, outros):
    meus = {nome_do_pacote(p): p for p in pacotes}
    deles = {nome_do_pacote(p): p for p in outros}
    return sorted(n for n in meus if n in deles and meus[n] != deles[n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raiz', default=str(AQUI.parents[2]))
    parser.add_argument('--venv', default=None)
    parser.add_argument('--install-torch', action='store_true', help='instala torch pelo indice oficial (CUDA; PyPI com Metal no Mac), apos autorizacao')
    parser.add_argument('--torch-index', default='https://download.pytorch.org/whl/cu128')
    args = parser.parse_args()
    raiz = Path(args.raiz).resolve()
    venv = Path(args.venv).resolve() if args.venv else raiz / '.its-fine' / 'venv'
    engine = json.loads((AQUI / 'engine.json').read_text(encoding='utf-8'))
    pacotes = list(engine['requisitos']['pacotes'])
    relatorio = {'schema_version': 1, 'engine': 'hf-encoder', 'raiz': str(raiz), 'python_sistema': sys.executable}
    irmao = AQUI.parent / 'hf-peft' / 'engine.json'
    if irmao.exists():
        divergentes = conflitos(pacotes, json.loads(irmao.read_text(encoding='utf-8'))['requisitos']['pacotes'])
        if divergentes:
            relatorio.update({'pronto': False, 'erro': 'DEPENDENCY_CONFLICT', 'pacotes': divergentes})
            print(json.dumps(relatorio, ensure_ascii=False, indent=2)); return 2
    registro = raiz / '.its-fine' / 'engine-setup.json'
    existente = json.loads(registro.read_text(encoding='utf-8')) if registro.exists() else {}
    if not venv.exists():
        codigo, saida = executar([sys.executable, '-c', 'import torch,json;m=getattr(torch.backends,"mps",None);'
                                  'print(json.dumps({"cuda":torch.cuda.is_available(),"mps":bool(m and m.is_available() and m.is_built())}))'], 120)
        info = primeiro_json(saida) or {}
        herdar = codigo == 0 and bool(info.get('cuda') or info.get('mps'))
        codigo, log = executar([sys.executable, '-m', 'venv'] + (['--system-site-packages'] if herdar else []) + [str(venv)])
        if codigo != 0:
            relatorio.update({'pronto': False, 'erro': 'falha ao criar venv', 'log': log}); print(json.dumps(relatorio, ensure_ascii=False, indent=2)); return 1
        existente.update({'modo': 'venv', 'venv': str(venv), 'herda_torch_do_sistema': herdar})
    python = venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if args.install_torch:
        if sys.platform == 'darwin': comando = [str(python), '-m', 'pip', 'install', 'torch']
        else:
            from urllib.parse import urlsplit
            parsed = urlsplit(args.torch_index)
            if parsed.scheme != 'https' or parsed.netloc != 'download.pytorch.org' or not parsed.path.startswith('/whl/'):
                raise ValueError('OFFICIAL_TORCH_INDEX_REQUIRED')
            comando = [str(python), '-m', 'pip', 'install', 'torch', '--index-url', args.torch_index]
        code, log = executar(comando, 3600)
        if code: print(json.dumps({'pronto': False, 'erro': 'TORCH_INSTALL_FAILED'})); return 2
    codigo, log = executar([str(python), '-m', 'pip', 'install', '--quiet', '--upgrade'] + pacotes, 3600)
    relatorio['instalacao_ok'] = codigo == 0; relatorio['pacotes'] = pacotes
    if codigo != 0: relatorio['log'] = log[-1200:]
    codigo2, versoes = executar([str(python), '-c', 'import torch,transformers,peft,safetensors,numpy,json;m=getattr(torch.backends,"mps",None);'
                                 'print(json.dumps({"torch":torch.__version__,"cuda":torch.cuda.is_available(),'
                                 '"mps":bool(m and m.is_available() and m.is_built()),"transformers":transformers.__version__,'
                                 '"peft":peft.__version__,"safetensors":safetensors.__version__,"numpy":numpy.__version__}))'], 120)
    versoes = primeiro_json(versoes) if codigo2 == 0 else None
    relatorio['versoes'] = versoes
    acelerado = bool(versoes and (versoes.get('cuda') or versoes.get('mps')))
    relatorio['pronto'] = bool(relatorio['instalacao_ok'] and acelerado)
    if versoes and not acelerado:
        relatorio['acao_do_usuario'] = 'torch sem CUDA (ou Metal no Mac) na venv: rode setup.py --install-torch apos aprovar o download.'
    prontos = set(existente.get('engines_prontos') or (['hf-peft'] if existente.get('pronto') else []))
    if relatorio['pronto']: prontos.add('hf-encoder')
    else: prontos.discard('hf-encoder')
    existente.update({'schema_version': 1, 'python': str(python), 'engines_prontos': sorted(prontos), 'hf-encoder': relatorio})
    existente.setdefault('engine', 'hf-peft'); existente.setdefault('modo', 'venv'); existente.setdefault('venv', str(venv))
    registro.parent.mkdir(parents=True, exist_ok=True)
    registro.write_text(json.dumps(existente, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    relatorio['registro'] = str(registro); relatorio['engines_prontos'] = sorted(prontos)
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return 0 if relatorio['pronto'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
