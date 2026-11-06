"""Prepara o ambiente do motor sem tocar no Python do usuario.

Cria `.its-fine/venv/`. Se o Python que roda este script ja enxerga a GPU
pelo torch (CUDA ou, no Mac com chip M, Metal), a venv herda os pacotes do
sistema e instala so o que falta. No macOS o torch vem do PyPI padrao, que ja
traz o backend Metal (mps).
Se nao, a venv nasce vazia e o script imprime o comando exato de instalacao
do torch para o CUDA detectado, sem executa-lo: instalar torch e decisao do
usuario, porque muda de acordo com driver e disco.

No Colab, nao cria venv: usa o interpretador do notebook.
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
    except (OSError,subprocess.TimeoutExpired) as exc:
        return 1,type(exc).__name__


def primeiro_json(texto):
    """Avisos de biblioteca vao para stderr e ficam depois do stdout; pega a linha que e JSON."""
    for linha in texto.splitlines():
        linha = linha.strip()
        if linha.startswith('{'):
            try:
                return json.loads(linha)
            except ValueError:
                continue
    return None


def torch_com_cuda(python):
    codigo, saida = executar([python, '-c', 'import torch,json;m=getattr(torch.backends,"mps",None);'
                              'print(json.dumps({"v":torch.__version__,"cuda":torch.cuda.is_available(),'
                              '"mps":bool(m and m.is_available() and m.is_built())}))'], 120)
    return primeiro_json(saida) if codigo == 0 else None


def acelerado(info):
    return bool(info and (info.get('cuda') or info.get('mps')))


def comando_torch(python, indice):
    """No macOS o indice oficial e o PyPI padrao (torch com Metal); fora dele, o indice CUDA do PyTorch."""
    if sys.platform == 'darwin': return [str(python), '-m', 'pip', 'install', 'torch']
    from urllib.parse import urlsplit
    parsed = urlsplit(indice)
    if parsed.scheme != 'https' or parsed.netloc != 'download.pytorch.org' or not parsed.path.startswith('/whl/'):
        raise ValueError('OFFICIAL_TORCH_INDEX_REQUIRED')
    return [str(python), '-m', 'pip', 'install', 'torch', '--index-url', indice]


def cuda_do_driver():
    codigo, saida = executar(['nvidia-smi'], 20)
    if codigo != 0:
        return None
    for token in saida.split():
        if token.replace('.', '').isdigit() and saida.find('CUDA Version') >= 0 and saida.find(token) > saida.find('CUDA Version'):
            return token
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raiz', default=str(AQUI.parents[2]), help='raiz do projeto; padrao: dois niveis acima de .its-fine/engines')
    parser.add_argument('--venv', default=None)
    parser.add_argument('--qlora', action='store_true', help='instala tambem bitsandbytes')
    parser.add_argument('--install-torch', action='store_true',help='executa a instalacao do torch (CUDA, ou Metal no Mac) previamente autorizada')
    parser.add_argument('--torch-index',default='https://download.pytorch.org/whl/cu128')
    parser.add_argument('--gguf', action='store_true',
                        help='clona o llama.cpp em .its-fine/tools/ para exportar GGUF e importar no Ollama')
    args = parser.parse_args()
    raiz = Path(args.raiz).resolve()
    venv = Path(args.venv).resolve() if args.venv else raiz / '.its-fine' / 'venv'
    engine = json.loads((AQUI / 'engine.json').read_text(encoding='utf-8'))
    pacotes = list(engine['requisitos']['pacotes'])
    if args.qlora:
        pacotes += engine['requisitos']['opcionais']['qlora']
    relatorio = {'schema_version': 1, 'engine': 'hf-peft', 'raiz': str(raiz), 'python_sistema': sys.executable}

    try:
        import google.colab  # noqa: F401
        colab = True
    except Exception:
        colab = False
    if colab:
        codigo, log = executar([sys.executable, '-m', 'pip', 'install', '--quiet'] + pacotes)
        relatorio.update({'modo': 'colab', 'python': sys.executable, 'instalacao_ok': codigo == 0, 'log': log[-800:],
                          'pronto': codigo == 0})
        registro = raiz / '.its-fine' / 'engine-setup.json'
        registro.parent.mkdir(parents=True, exist_ok=True)
        registro.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(relatorio, ensure_ascii=False, indent=2))
        return 0 if codigo == 0 else 1

    sistema = torch_com_cuda(sys.executable)
    herdar = acelerado(sistema)
    if not venv.exists():
        cmd = [sys.executable, '-m', 'venv'] + (['--system-site-packages'] if herdar else []) + [str(venv)]
        codigo, log = executar(cmd)
        if codigo != 0:
            relatorio.update({'erro': 'falha ao criar venv', 'log': log})
            print(json.dumps(relatorio, ensure_ascii=False, indent=2))
            return 1
    python = venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if args.install_torch:
        code,log=executar(comando_torch(python,args.torch_index),3600)
        if code: print(json.dumps({'pronto':False,'erro':'TORCH_INSTALL_FAILED'})); return 2
    relatorio.update({'modo': 'venv', 'venv': str(venv), 'python': str(python), 'herda_torch_do_sistema': herdar,
                      'torch_sistema': sistema})

    codigo, log = executar([str(python), '-m', 'pip', 'install', '--quiet', '--upgrade'] + pacotes, 3600)
    relatorio['instalacao_ok'] = codigo == 0
    relatorio['pacotes'] = pacotes
    if codigo != 0:
        relatorio['log'] = log[-1200:]

    verificado = torch_com_cuda(str(python))
    relatorio['torch_na_venv'] = verificado
    if not acelerado(verificado):
        relatorio['acao_do_usuario'] = (
            'O framework pode instalar torch na venv com --install-torch apos aprovar o download e verificar o driver.')
        relatorio['comando_torch'] = ' '.join('"%s"' % c if ' ' in c else c for c in comando_torch(python, args.torch_index))
        relatorio['driver_cuda'] = cuda_do_driver()
    codigo2, versoes = executar([str(python), '-c',
                                 'import transformers,peft,datasets,accelerate,pymupdf,json;'
                                 'print(json.dumps({"transformers":transformers.__version__,"peft":peft.__version__,'
                                 '"datasets":datasets.__version__,"accelerate":accelerate.__version__,"pymupdf":pymupdf.__version__}))'], 120)
    relatorio['versoes'] = primeiro_json(versoes) if codigo2 == 0 else None
    if args.gguf:
        ferramentas = raiz / '.its-fine' / 'tools'
        destino = ferramentas / 'llama.cpp'
        if (destino / 'convert_hf_to_gguf.py').exists():
            relatorio['llama_cpp'] = {'pronto': True, 'caminho': str(destino)}
        else:
            ferramentas.mkdir(parents=True, exist_ok=True)
            codigo3, log3 = executar(['git', 'clone', '--depth', '1', '--quiet', 'https://github.com/ggml-org/llama.cpp', str(destino)], 900)
            relatorio['llama_cpp'] = {'pronto': codigo3 == 0 and (destino / 'convert_hf_to_gguf.py').exists(),
                                      'caminho': str(destino), 'log': log3[-300:] if codigo3 != 0 else '',
                                      'nota': 'so o conversor Python; a quantizacao e feita pelo ollama create --quantize'}
    relatorio['pronto'] = bool(relatorio['instalacao_ok'] and relatorio['versoes'] and acelerado(verificado))
    # Fora da pasta do motor: ela e gerenciada pelo instalador e nao pode receber arquivo novo.
    registro = raiz / '.its-fine' / 'engine-setup.json'
    registro.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    relatorio['registro'] = str(registro)
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return 0 if relatorio['pronto'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
