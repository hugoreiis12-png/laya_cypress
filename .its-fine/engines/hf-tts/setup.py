"""Prepara a venv propria do hf-tts em .its-fine/venvs/hf-tts, sem tocar no Python do usuario.

Venv separada porque as familias de voz fixam versoes (transformers) que conflitam com a
venv compartilhada dos outros motores. Se o Python que roda este script ja enxerga a GPU
pelo torch (CUDA ou Metal no Mac), a venv herda o torch dele; senao `--install-torch`
instala torch e torchaudio pelo indice oficial apos autorizacao. Registra o motor em
.its-fine/engine-setup.json sem mexer no que os outros motores gravaram.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

AQUI = Path(__file__).resolve().parent
VERIFICA = ('import json,torch,torchaudio,transformers,peft,librosa,soundfile,jiwer;'
            'm=getattr(torch.backends,"mps",None);'
            'print(json.dumps({"torch":torch.__version__,"torchaudio":torchaudio.__version__,"cuda":torch.cuda.is_available(),'
            '"mps":bool(m and m.is_available() and m.is_built()),"transformers":transformers.__version__,"peft":peft.__version__}))')


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


def torch_de(python):
    codigo, saida = executar([str(python), '-c', 'import torch,json;m=getattr(torch.backends,"mps",None);'
                              'print(json.dumps({"v":torch.__version__,"cuda":torch.cuda.is_available(),'
                              '"mps":bool(m and m.is_available() and m.is_built())}))'], 120)
    return primeiro_json(saida) if codigo == 0 else None


def comando_torch(python, indice):
    """torch e torchaudio do mesmo indice: PyPI (com Metal) no macOS, indice CUDA oficial fora dele."""
    if sys.platform == 'darwin': return [str(python), '-m', 'pip', 'install', 'torch', 'torchaudio']
    from urllib.parse import urlsplit
    parsed = urlsplit(indice)
    if parsed.scheme != 'https' or parsed.netloc != 'download.pytorch.org' or not parsed.path.startswith('/whl/'):
        raise ValueError('OFFICIAL_TORCH_INDEX_REQUIRED')
    return [str(python), '-m', 'pip', 'install', 'torch', 'torchaudio', '--index-url', indice]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raiz', default=str(AQUI.parents[2]), help='raiz do projeto; padrao: dois niveis acima de .its-fine/engines')
    parser.add_argument('--install-torch', action='store_true', help='instala torch e torchaudio pelo indice oficial, apos autorizacao')
    parser.add_argument('--torch-index', default='https://download.pytorch.org/whl/cu128')
    args = parser.parse_args()
    raiz = Path(args.raiz).resolve()
    venv = raiz / '.its-fine' / 'venvs' / 'hf-tts'
    requisitos = json.loads((AQUI / 'engine.json').read_text(encoding='utf-8'))['requisitos']
    relatorio = {'schema_version': 1, 'engine': 'hf-tts', 'raiz': str(raiz), 'python_sistema': sys.executable, 'venv': str(venv)}

    if not venv.exists():
        sistema = torch_de(sys.executable)
        herdar = bool(sistema and (sistema.get('cuda') or sistema.get('mps')))
        codigo, log = executar([sys.executable, '-m', 'venv'] + (['--system-site-packages'] if herdar else []) + [str(venv)])
        if codigo != 0:
            relatorio.update({'pronto': False, 'erro': 'falha ao criar venv', 'log': log})
            print(json.dumps(relatorio, ensure_ascii=False, indent=2)); return 1
        relatorio['herda_torch_do_sistema'] = herdar
    python = venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if args.install_torch:
        codigo, log = executar(comando_torch(python, args.torch_index), 3600)
        if codigo: print(json.dumps({'pronto': False, 'erro': 'TORCH_INSTALL_FAILED', 'log': log[-800:]})); return 2
    torch = torch_de(python)
    if torch is None:
        relatorio.update({'pronto': False, 'acao_do_usuario': 'venv sem torch: rode setup.py --install-torch apos aprovar o download.',
                          'comando_torch': ' '.join(comando_torch(python, args.torch_index))})
        registrar(raiz, relatorio); print(json.dumps(relatorio, ensure_ascii=False, indent=2)); return 2
    # torchaudio na versao exata do torch: sem isso o pip troca o torch CUDA por um de CPU
    base = torch['v'].split('+')[0]
    pacotes = [('torchaudio==' + base if p == 'torchaudio' else p) for p in requisitos['pacotes']]
    codigo, log = executar([str(python), '-m', 'pip', 'install', '--quiet'] + pacotes, 3600)
    if codigo == 0 and requisitos.get('sem_dependencias'):
        codigo, log = executar([str(python), '-m', 'pip', 'install', '--quiet', '--no-deps'] + requisitos['sem_dependencias'], 1800)
    relatorio['instalacao_ok'] = codigo == 0
    relatorio['pacotes'] = pacotes + requisitos.get('sem_dependencias', [])
    if codigo != 0: relatorio['log'] = log[-1200:]
    codigo, saida = executar([str(python), '-c', VERIFICA], 300)
    versoes = primeiro_json(saida) if codigo == 0 else None
    relatorio['versoes'] = versoes
    relatorio['python'] = str(python)
    acelerado = bool(versoes and (versoes.get('cuda') or versoes.get('mps')))
    relatorio['pronto'] = bool(relatorio['instalacao_ok'] and acelerado)
    if versoes and not acelerado:
        relatorio['acao_do_usuario'] = 'torch sem CUDA (ou Metal no Mac) na venv: rode setup.py --install-torch apos aprovar o download.'
    registrar(raiz, relatorio)
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return 0 if relatorio['pronto'] else 2


def registrar(raiz, relatorio):
    registro = raiz / '.its-fine' / 'engine-setup.json'
    existente = json.loads(registro.read_text(encoding='utf-8')) if registro.exists() else {'schema_version': 1}
    prontos = set(existente.get('engines_prontos') or (['hf-peft'] if existente.get('pronto') else []))
    if relatorio.get('pronto'): prontos.add('hf-tts')
    else: prontos.discard('hf-tts')
    existente.update({'engines_prontos': sorted(prontos), 'hf-tts': relatorio})
    registro.parent.mkdir(parents=True, exist_ok=True)
    registro.write_text(json.dumps(existente, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
