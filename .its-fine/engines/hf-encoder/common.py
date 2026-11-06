"""Funcoes compartilhadas do motor hf-encoder: JSON, hashes, politica e fronteira CLI."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# Mac com chip M: operacao sem kernel Metal cai para a CPU; precisa valer antes de importar torch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# Runtime canonico no repositorio ou na copia instalada.
_project = Path(__file__).resolve().parents[2]
_snapshot_lib = Path(__file__).resolve().parent.parent / 'lib'
sys.path.insert(0, str(_snapshot_lib if (_snapshot_lib / 'itsfine').is_dir() else
                      (_project / 'lib' if (_project / 'lib/itsfine').is_dir() else _project.parent)))
from itsfine.privacy import load_policy as politica_privacidade  # noqa: E402

AQUI = Path(__file__).resolve().parent
ENGINE = 'hf-encoder'


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


def iterar_jsonl(caminho):
    with Path(caminho).open(encoding='utf-8') as stream:
        for bruta in stream:
            if bruta.strip(): yield json.loads(bruta)


def ler_jsonl(caminho):
    return list(iterar_jsonl(caminho))


def gravar_jsonl(caminho, linhas):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open('w', encoding='utf-8', newline='\n') as saida:
        for linha in linhas:
            saida.write(json.dumps(linha, ensure_ascii=False) + '\n')


def sha256_arquivo(caminho):
    sha = hashlib.sha256()
    with Path(caminho).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): sha.update(block)
    return sha.hexdigest()


def sha256_texto(texto):
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()


def sha256_json(valor):
    return hashlib.sha256(json.dumps(valor, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def engine_info():
    return ler_json(AQUI / 'engine.json')


def dispositivo(device='auto'):
    import torch
    cuda = torch.cuda.is_available()
    escolhido = ('cuda' if cuda else 'cpu') if device == 'auto' else device
    if escolhido == 'cuda' and not cuda: raise ValueError('CUDA_UNAVAILABLE')
    bf16 = escolhido == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    return escolhido, (torch.bfloat16 if bf16 else (torch.float16 if escolhido == 'cuda' else torch.float32))


def modulos_lineares(modulo):
    """Nomes-folha dos Linear do encoder, alvo de LoRA em qualquer familia."""
    import torch
    nomes = set()
    for nome, m in modulo.named_modules():
        if isinstance(m, torch.nn.Linear):
            nomes.add(nome.split('.')[-1])
    return sorted(nomes)


# Chaves numericas/booleanas que a fronteira CLI pode devolver ao harness.
CHAVES_AGREGADAS = {
    'linhas', 'estados', 'sequencias', 'grupos', 'splits', 'train', 'validation', 'calibration', 'test',
    'aprovado', 'problemas', 'avisos', 'max_seq_length_sugerido', 'amostra', 'candidato', 'baseline_calibrado',
    'original_sem_calibracao', 'metricas', 'saidas', 'log_loss', 'brier', 'acuracia', 'ece', 'mae', 'n',
    'rotulados', 'cobertura_automatica', 'encaminhados', 'tp', 'fp', 'tn', 'fn', 'fn_liberados',
    'taxa_fn_liberados', 'recall', 'precision', 'f1', 'limites', 'ok', 'valor', 'limite', 'valid',
    'suporte_insuficiente', 'coube', 'pico_alocado_gb', 'pico_reservado_gb', 'total_gb', 'passos',
    'loss_treino', 'duracao_s', 'segundos_por_passo', 'arquivos', 'bytes', 'consumidores', 'predict',
    'servidor', 'carregou', 'decisoes_iguais', 'temperature', 'support', 'buckets', 'metricas_validacao',
    'exemplos_treino', 'calibracao', 'operacional', 'revisar_como_negativo', 'automatica', 'latencia_ms_p50',
    'paridade', 'delta_logits_max', 'tokens_iguais', 'marcadores_iguais', 'casos', 'estados_por_split',
    'opcoes', 'suporte', 'itens', 'diferenca_maxima', 'passos_confirmados', 'parametros', 'treinaveis',
    'total', 'fracao', 'inconclusiva_por_suporte', 'baseline_frequencia', 'cabecas_novas',
}
CHAVES_TEXTO = {'nome', 'decisao', 'modo', 'engine', 'status', 'acao', 'veredito', 'bucket', 'saida', 'tipo'}


def agregados(obj):
    if isinstance(obj, dict):
        saida = {}
        for k, v in obj.items():
            if k in CHAVES_AGREGADAS and (v is None or type(v) in (bool, int, float, dict, list)):
                saida[k] = agregados(v)
            elif k in CHAVES_TEXTO and isinstance(v, str) and len(v) <= 80:
                saida[k] = v
            elif k in CHAVES_AGREGADAS and isinstance(v, str) and len(v) <= 80:
                saida[k] = v
        return saida
    if isinstance(obj, list):
        return [agregados(x) for x in obj if type(x) in (bool, int, float, dict)]
    return obj


def data_entrypoint(main, summary=None):
    """Fronteira CLI: stdout/stderr das bibliotecas ficam em log privado; devolve so agregados."""
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
    fd, logpath = tempfile.mkstemp(prefix='encoder-', suffix='.log', dir=logdir)
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
            value = ler_json(output / summary if summary != '.' else output)
            response['summary'] = agregados(value)
        except Exception:
            response = {'status': 'failed', 'error': 'SUMMARY_INVALID', 'exit_code': 1}
            result = 1
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return result
