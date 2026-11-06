"""Exporta o modelo de voz: base + adapter fundidos num pacote que fala sozinho, reavaliado como sera usado.

O pacote leva modelo/ (pesos ja fundidos, voz propria), familia.py (codigo de inferencia da
familia), falar.py (linha de comando sem o ITS-FINE), requirements.txt, LEIAME.md e
itsfine-voz.json. Depois o proprio pacote e avaliado no teste pelo evaluate.py --pacote, com o
mesmo protocolo: a nota que vai para o export e a do pacote, nao a do adapter.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AQUI, data_entrypoint, engine_info, familia_do_modelo, gravar_json, ler_json, sha256_arquivo  # noqa: E402

FALAR = r'''"""Fala com a voz treinada pelo ITS-FINE. Uso: python falar.py "texto" --saida fala.wav"""
from pathlib import Path
import argparse, json, sys, types
AQUI = Path(__file__).resolve().parent

def carregar_audio(caminho, taxa):
    import librosa, numpy as np
    onda, _ = librosa.load(str(caminho), sr=taxa, mono=True)
    return np.ascontiguousarray(onda, dtype=np.float32)

def main():
    p = argparse.ArgumentParser(); p.add_argument('texto'); p.add_argument('--saida', default='fala.wav')
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto'); a = p.parse_args()
    import soundfile as sf, torch
    sys.modules['common'] = types.SimpleNamespace(carregar_audio=carregar_audio)
    sys.path.insert(0, str(AQUI))
    from familia import Familia
    mps = torch.backends.mps.is_available()
    d = a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if mps else 'cpu')
    bf16 = (d == 'cuda' and torch.cuda.is_bf16_supported()) or (d == 'mps' and torch.backends.mps.is_macos_or_newer(14, 0))
    m = json.loads((AQUI / 'itsfine-voz.json').read_text(encoding='utf-8'))
    voz = Familia.do_pacote(AQUI / 'modelo', d, torch.bfloat16 if bf16 else torch.float32, idioma=m.get('idioma'))
    onda, taxa = voz.falar(a.texto, voz=m['voz'])
    sf.write(a.saida, onda, taxa); print(a.saida)

if __name__ == '__main__':
    main()
'''

LEIAME = '''# Voz treinada pelo ITS-FINE

Modelo de voz propria ({familia}, a partir de `{modelo}`), idioma `{idioma}`, voz `{voz}`.

Instale as dependencias numa venv (`pip install -r requirements.txt` e depois
`pip install --no-deps {sem_dependencias}`; o torch vem antes, do indice da sua GPU) e rode:

    python falar.py "Ola, esta e a minha voz." --saida fala.wav

Funciona em GPU NVIDIA, Mac com chip M (Metal) ou CPU (lento). Nota medida no teste do projeto:
similaridade_inteligivel = {nota}.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True, help='adapter/ do treino aprovado')
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--familia')
    parser.add_argument('--dados', required=True, help='pasta preparada (voz-alvo e referencia vem do treino)')
    parser.add_argument('--teste', required=True)
    parser.add_argument('--calibracao', required=True)
    parser.add_argument('--calibracao-hash', required=True)
    parser.add_argument('--saida', required=True, help='pasta nova do pacote')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()

    from itsfine import device as acelerador
    from itsfine.continuation import adapter_hash
    from evaluate import materializar
    saida = Path(args.saida).resolve()
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('OUTPUT_EXISTS')
    familia = familia_do_modelo(args.modelo, args.familia)
    info = engine_info()
    dispositivo = acelerador.escolher(args.device)
    meta = materializar(args.modelo, args.revision, familia, Path(args.adapter), saida / 'modelo', dispositivo, acelerador.dtype(dispositivo))
    shutil.copyfile(AQUI / info['familias'][familia]['adaptador'], saida / 'familia.py')
    (saida / 'falar.py').write_text(FALAR, encoding='utf-8')
    req = info['requisitos']
    (saida / 'requirements.txt').write_text('\n'.join(req['pacotes']) + '\n', encoding='utf-8')
    gravar_json(saida / 'itsfine-voz.json', {'schema_version': 1, 'engine': 'hf-tts', 'familia': familia, 'voz': meta['voz'],
                                            'idioma': meta.get('idioma'), 'modelo_base': args.modelo, 'revision': args.revision,
                                            'adapter_hash': adapter_hash(args.adapter), 'sem_dependencias': req.get('sem_dependencias', [])})

    # o pacote avaliado como sera usado: processo novo, so com os arquivos do pacote
    avaliacao = saida.parent / (saida.name + '-avaliacao')
    cmd = [sys.executable, str(AQUI / 'evaluate.py'), '--teste', args.teste, '--dados', args.dados, '--modelo', args.modelo,
           '--revision', args.revision, '--familia', familia, '--idioma', meta.get('idioma') or '', '--pacote', str(saida),
           '--saida', str(avaliacao), '--calibracao', args.calibracao, '--calibracao-hash', args.calibracao_hash, '--device', args.device]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    avaliado = (avaliacao / 'resumo.json').is_file() and resultado.returncode == 0
    nota = ler_json(avaliacao / 'resumo.json') if avaliado else {}
    (saida / 'LEIAME.md').write_text(LEIAME.format(familia=familia, modelo=args.modelo, idioma=meta.get('idioma'), voz=meta['voz'],
                                                   sem_dependencias=' '.join(req.get('sem_dependencias', [])),
                                                   nota=nota.get('similaridade_inteligivel')), encoding='utf-8')
    arquivos = [{'path': p.relative_to(saida).as_posix(), 'hash': sha256_arquivo(p)} for p in sorted(saida.rglob('*')) if p.is_file()]
    pacote_hash = hashlib.sha256(json.dumps(arquivos, sort_keys=True).encode()).hexdigest()
    import itsfine
    import transformers
    gravar_json(saida.parent / (saida.name + '-export.json'), {
        'schema_version': 1, 'engine': 'hf-tts', 'format': 'hf-tts-voz', 'consumer': 'falar.py', 'pacote': saida.name,
        'consumer_version': f'itsfine {itsfine.__version__}; transformers {transformers.__version__}',
        'package_hash': pacote_hash, 'files': arquivos, 'tokenizer_revision': args.revision,
        'template_hash': sha256_arquivo(saida / 'familia.py'), 'carregou': avaliado, 'loaded': avaliado,
        'sintetizou': bool(nota.get('amostra')), 'inference_passed': bool(nota.get('inteligiveis')),
        'metric_value': nota.get('similaridade_inteligivel'), 'avaliacao': avaliacao.name,
        'test_hash': nota.get('test_hash'), 'calibracao_hash': args.calibracao_hash, 'ok': avaliado and bool(nota.get('inteligiveis'))})
    return 0 if avaliado else 1


if __name__ == '__main__':
    # o registro fica ao lado do pacote (dentro mudaria o hash do pacote): <saida>-export.json
    nome = Path(sys.argv[sys.argv.index('--saida') + 1]).name if '--saida' in sys.argv else ''
    raise SystemExit(data_entrypoint(main, '../' + nome + '-export.json'))
