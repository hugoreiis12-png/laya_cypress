"""Funcoes compartilhadas do motor hf-tts: JSON, hashes, audio, politica e fronteira CLI.

Nada aqui conhece uma familia de modelo: o que e de cada arquitetura fica em familias/.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# Mac com chip M: operacao sem kernel Metal cai para a CPU; precisa valer antes de importar torch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# Runtime canonico no repositorio, na copia instalada ou no snapshot do job.
_project = Path(__file__).resolve().parents[2]
_snapshot_lib = Path(__file__).resolve().parent.parent / 'lib'
sys.path.insert(0, str(_snapshot_lib if (_snapshot_lib / 'itsfine').is_dir() else
                      (_project / 'lib' if (_project / 'lib/itsfine').is_dir() else _project.parent)))
from itsfine.privacy import load_policy as politica_privacidade  # noqa: E402

AQUI = Path(__file__).resolve().parent
ENGINE = 'hf-tts'


def ler_json(caminho):
    return json.loads(Path(caminho).read_text(encoding='utf-8'))


def gravar_json(caminho, dados):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + caminho.name, dir=caminho.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(dados, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, caminho)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def ler_jsonl(caminho):
    with Path(caminho).open(encoding='utf-8') as stream:
        return [json.loads(l) for l in stream if l.strip()]


def gravar_jsonl(caminho, linhas):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open('w', encoding='utf-8', newline='\n') as saida:
        for linha in linhas: saida.write(json.dumps(linha, ensure_ascii=False) + '\n')


def sha256_arquivo(caminho):
    sha = hashlib.sha256()
    with Path(caminho).open('rb') as stream:
        for bloco in iter(lambda: stream.read(1024 * 1024), b''): sha.update(bloco)
    return sha.hexdigest()


def engine_info():
    return ler_json(AQUI / 'engine.json')


def familia_do_modelo(model_id, familia=None):
    """Familia declarada na receita ou, para modelo do catalogo, a do catalogo."""
    if familia: return familia
    for item in engine_info()['modelos']:
        if item['id'] == model_id: return item['familia']
    raise ValueError('FAMILY_REQUIRED: modelo fora do catalogo precisa de "familia" na receita')


def adaptador(familia):
    """Classe do adaptador da familia, carregada pelo caminho declarado no engine.json."""
    info = engine_info()['familias'].get(familia)
    if info is None: raise ValueError('UNKNOWN_FAMILY:' + str(familia))
    caminho = AQUI / info['adaptador']
    spec = importlib.util.spec_from_file_location('familia_' + familia.replace('-', '_'), caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.Familia


def hash_adaptador(familia):
    """O 'template' de um modelo de voz e o layout de sequencia da familia: o codigo do adaptador."""
    return sha256_arquivo(AQUI / engine_info()['familias'][familia]['adaptador'])


# Lidos pelo FFmpeg embutido no PyAV (vem com o faster-whisper): video e audio compactado
VIA_FFMPEG = ('.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.m4a', '.aac', '.opus')


def extrair_audio(caminho, taxa):
    """Primeira faixa de audio de um video ou audio compactado, mono float32 na taxa pedida."""
    import av
    import numpy as np
    with av.open(str(caminho)) as container:
        faixa = next((s for s in container.streams if s.type == 'audio'), None)
        if faixa is None: raise ValueError('NO_AUDIO_TRACK')
        conversor = av.AudioResampler(format='flt', layout='mono', rate=taxa)
        partes = [q.to_ndarray().reshape(-1) for quadro in container.decode(faixa) for q in conversor.resample(quadro)]
        partes += [q.to_ndarray().reshape(-1) for q in conversor.resample(None)]
    if not partes: raise ValueError('EMPTY_AUDIO_TRACK')
    return np.ascontiguousarray(np.concatenate(partes), dtype=np.float32)


def carregar_audio(caminho, taxa):
    """Mono float32 na taxa pedida; de video (mp4, mov, mkv, webm...) extrai a faixa de audio."""
    import numpy as np
    import soundfile as sf
    if Path(caminho).suffix.lower() in VIA_FFMPEG: return extrair_audio(caminho, taxa)
    onda, origem = sf.read(str(caminho), dtype='float32', always_2d=True)
    onda = onda.mean(axis=1)
    if origem != taxa:
        import librosa
        onda = librosa.resample(onda, orig_sr=origem, target_sr=taxa)
    return np.ascontiguousarray(onda, dtype=np.float32)


def duracao(caminho):
    import soundfile as sf
    info = sf.info(str(caminho))
    return info.frames / info.samplerate


def referencia(linhas, raiz):
    """Clipe de voz de referencia, deterministico: o do treino com duracao mais perto de 8 s (entre 3 e 15 s).

    Serve ao modelo base para clonar (baseline) e as familias que condicionam a voz num clipe.
    """
    candidatos = []
    for linha in linhas:
        d = linha.get('duracao') or duracao(Path(raiz) / linha['audio'])
        if 3 <= d <= 15: candidatos.append((abs(d - 8), linha['audio'], linha))
    if not candidatos: raise ValueError('NO_REFERENCE_CLIP: nenhum clipe de treino entre 3 e 15 s')
    return sorted(candidatos, key=lambda c: (c[0], c[1]))[0][2]


def acaso(vocabulario):
    """Perda de quem chuta ao acaso entre `vocabulario` opcoes."""
    return math.log(vocabulario)


def data_entrypoint(main, summary=None):
    """Fronteira CLI: saida de bibliotecas fica no log local; devolve so agregados permitidos."""
    import contextlib
    from itsfine.privacy import require_local_executor
    if '--help' in sys.argv or '-h' in sys.argv:
        return main()
    try:
        output = Path(sys.argv[sys.argv.index('--saida') + 1]).resolve()
        policy = politica_privacidade(output)
        require_local_executor(policy)
    except Exception:
        print(json.dumps({'status': 'refused', 'error': 'POLICY_OR_OUTPUT_INVALID'}))
        return 3
    logdir = Path(policy['_policy_path']).parent / 'private-logs'
    logdir.mkdir(parents=True, exist_ok=True)
    fd, _ = tempfile.mkstemp(prefix='engine-', suffix='.log', dir=logdir)
    saved_out, saved_err = os.dup(1), os.dup(2)
    result = 1
    try:
        sys.stdout.flush(); sys.stderr.flush()
        with os.fdopen(fd, 'w', encoding='utf-8') as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    result = main()
                except Exception:
                    import traceback
                    traceback.print_exc()
                    result = 1
                finally: log.flush()
    finally:
        os.dup2(saved_out, 1); os.dup2(saved_err, 2)
        os.close(saved_out); os.close(saved_err)
    response = {'status': 'completed' if result == 0 else 'failed', 'exit_code': result}
    if summary and result == 0:
        try:
            value = ler_json(output if summary == '.' else output / summary)
            # So chaves numericas conhecidas: nunca texto, transcricao ou nome de arquivo.
            allowed = {'arquivos', 'videos', 'sem_audio', 'clipes', 'grupos', 'minutos', 'descartes', 'curtos', 'longos', 'sem_fala',
                       'transcricao_ruim', 'falhas', 'valid', 'mesma_voz', 'outra_voz', 'texto_trocado',
                       'similaridade_media', 'similaridade_min', 'similaridade_max', 'wer', 'wer_medio', 'margem',
                       'amostra', 'base', 'candidato', 'inteligiveis', 'similaridade_inteligivel', 'ok', 'clipes_validos',
                       'duracao_media_s', 'duracao_max_s', 'carregou', 'sintetizou', 'metric_value',
                       'splits', 'train', 'validation', 'test', 'separacao', 'limiar_sugerido', 'checagens', 'separa_vozes',
                       'margem_minima', 'fala_real_inteligivel', 'texto_trocado_detectado', 'loaded', 'inference_passed'}
            def aggregates(obj):
                if not isinstance(obj, dict): return {}
                return {k: (aggregates(v) if isinstance(v, dict) else v) for k, v in obj.items()
                        if k in allowed and (v is None or type(v) in (bool, int, float, dict))}
            response['summary'] = aggregates(value)
        except Exception:
            response = {'status': 'failed', 'error': 'SUMMARY_INVALID', 'exit_code': 1}
            result = 1
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return result
