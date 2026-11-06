"""Avaliacao congelada: baseline calibrado contra candidato no mesmo teste.

Baseline e o checkpoint de origem com temperaturas reajustadas na particao de
calibracao; o original sem calibracao aparece so como diagnostico. Com cabecas
novas esse baseline e o acaso; a referencia honesta passa a ser baseline_frequencia,
que responde cada opcao pela frequencia na calibracao sem ler o texto. Candidato e
adapter, pacote exportado ou servidor local (reavaliacao no consumidor entregue).
Metricas por saida (log-loss, Brier, ECE, curva de confiabilidade, acuracia, MAE)
e, na saida que decide, a matriz das decisoes automaticas no ponto de operacao.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, gravar_jsonl, ler_json, ler_jsonl, sha256_arquivo, sha256_json, politica_privacidade  # noqa: E402
import infer  # noqa: E402
import calibration  # noqa: E402
from decisions import metricas_saida, matriz_operacional, verificar_limites, validar_politica  # noqa: E402
from sequence import QTYPES, indice_do_rotulo, opcoes, perguntas_da_tarefa  # noqa: E402
from itsfine.privacy import loopback, require_destination, request_json, require_local_executor  # noqa: E402


def blocos(ctx, regs_por_linha, calib):
    """Metricas por saida e matriz operacional de um conjunto de registros."""
    por_saida = {}
    for regs in regs_por_linha:
        for r in regs: por_saida.setdefault(r['nome'], []).append(r)
    saidas = [{'nome': nome, **metricas_saida(regs, calib)} for nome, regs in por_saida.items()]
    bloco = {'saidas': saidas}
    if ctx['decide'] and ctx['politica'] and ctx['decide'] in por_saida:
        bloco['metricas_decisao'] = next(s for s in saidas if s['nome'] == ctx['decide'])
        bloco['operacional'] = matriz_operacional(por_saida[ctx['decide']], ctx['politica'], calib)
    return bloco


def registros_frequencia(ctx, calibracao, linhas):
    """Baseline que ignora o texto: cada opcao recebe sua frequencia na particao de calibracao (+1 por opcao)."""
    prior = {}
    for nome, q in ctx['internas'].items():
        contagem = [1.0] * len(opcoes(q))
        for linha in calibracao:
            i = indice_do_rotulo(q, linha.get('answers', {}).get(nome))
            if i is not None: contagem[i] += 1
        prior[nome] = [c / sum(contagem) for c in contagem]
    saida = []
    for linha in linhas:
        regs = []
        for nome, q in ctx['internas'].items():
            i = indice_do_rotulo(q, linha.get('answers', {}).get(nome))
            regs.append({'nome': nome, 'qtype': QTYPES[q['t']], 'k': len(prior[nome]), 'probs': prior[nome], 'label': -1 if i is None else i})
        saida.append(regs)
    return saida


def calibrar(ctx, linhas, min_support, batch):
    regs = [r for linha in infer.registros(ctx, linhas, batch) for r in linha]
    calib = calibration.ajustar(regs, min_support)
    return calib, regs


def registros_do_servidor(ctx, url, linhas, batch, policy):
    """Consumidor HTTP: probabilidades ja calibradas; identidade conferida antes e depois."""
    if not loopback(url): require_destination(policy, 'transfer', url)
    identidade = request_json(url + '/v1/package')
    saida = []
    for i in range(0, len(linhas), batch):
        parte = linhas[i:i + batch]
        resposta = request_json(url + '/v1/decide', {'states': [l['state'] for l in parte]})
        if resposta.get('package_id') != identidade.get('package_id'): raise ValueError('PACKAGE_CHANGED_DURING_EVALUATION')
        for linha, resultado in zip(parte, resposta['results']):
            regs = []
            for nome, q in ctx['internas'].items():
                r = resultado['answers'][nome]
                rotulo = indice_do_rotulo(q, linha.get('answers', {}).get(nome))
                regs.append({'nome': nome, 'qtype': QTYPES[q['t']], 'k': len(r['probs']), 'probs': r['probs'],
                             'label': -1 if rotulo is None else rotulo})
            saida.append(regs)
    if request_json(url + '/v1/package').get('package_id') != identidade.get('package_id'): raise ValueError('PACKAGE_CHANGED_DURING_EVALUATION')
    return saida, identidade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teste', required=True, help='test.jsonl congelado')
    parser.add_argument('--tarefa', help='task.json; padrao: ao lado do teste ou dentro do pacote')
    parser.add_argument('--checkpoint', help='pasta do checkpoint de origem (baseline)')
    parser.add_argument('--checkpoint-id'); parser.add_argument('--checkpoint-revision')
    parser.add_argument('--adapter', help='candidato: adapter LoRA + heads')
    parser.add_argument('--pacote', help='candidato: pacote exportado; avalia so o candidato')
    parser.add_argument('--servidor', help='candidato: URL do serve.py; avalia so o consumidor')
    parser.add_argument('--calibracao-dados', help='calibration.jsonl para ajustar temperaturas de baseline e candidato')
    parser.add_argument('--calibracao', help='calibration.json ja ajustado para o candidato')
    parser.add_argument('--min-support', type=int, default=None)
    parser.add_argument('--limites', help='JSON (arquivo ou texto) com limites obrigatorios do protocolo')
    parser.add_argument('--limite', type=int, help='avalia so os N primeiros itens')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--gravar-logits', action='store_true', help='grava logits por item para calibration.py')
    parser.add_argument('--saida', required=True)
    args = parser.parse_args()
    policy = politica_privacidade(args.saida)
    require_local_executor(policy)
    if sum(bool(x) for x in (args.adapter, args.pacote, args.servidor)) > 1: raise ValueError('EXCLUSIVE_CANDIDATE_REQUIRED')
    if args.limite is not None and args.limite < 1: raise ValueError('INVALID_EVALUATION_LIMIT')
    saida = Path(args.saida).resolve(); saida.mkdir(parents=True, exist_ok=True)
    linhas = ler_jsonl(args.teste)[: args.limite or None]
    if not linhas: raise ValueError('EMPTY_EVALUATION')
    tarefa = args.tarefa or (str(Path(args.pacote) / 'task.json') if args.pacote else str(Path(args.teste).with_name('task.json')))
    limites = None
    if args.limites:
        limites = ler_json(args.limites) if Path(args.limites).is_file() else json.loads(args.limites)
    engine = ler_json(Path(__file__).with_name('engine.json'))
    min_support = args.min_support or engine['defaults']['calibration']['min_support']
    relatorio = {'schema_version': 1, 'engine': 'hf-encoder', 'teste': str(args.teste), 'teste_hash': sha256_arquivo(args.teste),
                 'tarefa_hash': sha256_arquivo(tarefa), 'amostra': len(linhas), 'min_support': min_support,
                 'metric_roles': {'log_loss': 'proper-scoring', 'brier': 'proper-scoring', 'ece': 'calibration-15-bins',
                                  'acuracia': 'argmax', 'operacional': 'decisoes-no-ponto-de-operacao',
                                  'baseline_frequencia': 'frequencia-das-opcoes-na-calibracao-sem-ler-o-texto'}}
    verificar = {'id': args.checkpoint_id, 'revision': args.checkpoint_revision} if args.checkpoint_id else None
    itens = []

    if args.servidor:
        task = ler_json(tarefa)
        ctx = {'internas': perguntas_da_tarefa(task), 'decide': task.get('decide'), 'politica': None, 'calib': None}
        if ctx['decide']:
            ctx['politica'] = validar_politica(task.get('policy'), ctx['internas'][ctx['decide']])
        regs, identidade = registros_do_servidor(ctx, args.servidor, linhas, args.batch, policy)
        relatorio['consumer'] = 'serve.py'; relatorio['consumer_identity'] = identidade
        relatorio['candidato'] = blocos(ctx, regs, None)
        itens = regs
    elif args.pacote:
        ctx = infer.preparar(args.pacote, args.device)
        regs = infer.registros(ctx, linhas, args.batch)
        relatorio['consumer'] = 'pacote'; relatorio['consumer_identity'] = ctx['carga']['identidade']
        relatorio['candidato'] = blocos(ctx, regs, ctx['calib'])
        relatorio['latencia_ms_p50'] = infer.latencia_p50(ctx)
        itens = regs
    else:
        if not args.checkpoint: raise ValueError('CHECKPOINT_REQUIRED')
        ctx = infer.preparar(args.checkpoint, args.device, verificar=verificar, tarefa=tarefa, calibracao=None)
        relatorio['cabecas_novas'] = ctx['carga']['desc']['cabecas_novas']
        base = infer.registros(ctx, linhas, args.batch)
        relatorio['original_sem_calibracao'] = blocos(ctx, base, None)
        calib_base = None
        if args.calibracao_dados:
            linhas_calibracao = ler_jsonl(args.calibracao_dados)
            relatorio['baseline_frequencia'] = blocos(ctx, registros_frequencia(ctx, linhas_calibracao, linhas), None)
            calib_base, _ = calibrar(ctx, linhas_calibracao, min_support, args.batch)
            calib_base['partition_hash'] = sha256_arquivo(args.calibracao_dados); calib_base['weights_hash'] = ctx['carga']['identidade']['package_id']
            calib_base['hash'] = sha256_json({k: v for k, v in calib_base.items() if k != 'hash'})
            gravar_json(saida / 'calibracao-baseline.json', calib_base)
            relatorio['baseline_calibrado'] = blocos(ctx, base, calib_base)
            relatorio['calibracao_baseline_hash'] = calib_base['hash']
        relatorio['latencia_ms_p50'] = infer.latencia_p50(ctx)
        itens = base
        if args.adapter:
            adapter = Path(args.adapter)
            adapter_hash = sha256_json({'adapter': sha256_arquivo(adapter / 'adapter_model.safetensors'), 'heads': sha256_arquivo(adapter / 'heads.safetensors')})
            ctx = infer.preparar(args.checkpoint, args.device, adapter=str(adapter), verificar=verificar, tarefa=tarefa, calibracao=None)
            cand = infer.registros(ctx, linhas, args.batch)
            if args.calibracao:
                calib = ler_json(args.calibracao)
                if calib.get('weights_hash') not in (None, adapter_hash): raise ValueError('CALIBRATION_WEIGHTS_MISMATCH')
            elif args.calibracao_dados:
                calib, regs_calib = calibrar(ctx, ler_jsonl(args.calibracao_dados), min_support, args.batch)
                calib['partition_hash'] = sha256_arquivo(args.calibracao_dados); calib['weights_hash'] = adapter_hash
                calib['hash'] = sha256_json({k: v for k, v in calib.items() if k != 'hash'})
                if args.gravar_logits: gravar_jsonl(saida / 'logits-calibracao.jsonl', regs_calib)
            else:
                calib = None
            if calib: gravar_json(saida / 'calibracao-candidato.json', calib); relatorio['calibracao_candidato_hash'] = calib['hash']
            relatorio['adapter_hash'] = adapter_hash
            relatorio['candidato'] = blocos(ctx, cand, calib)
            itens = cand
        else:
            relatorio['modo'] = 'baseline-only'
    if 'candidato' in relatorio:
        relatorio['limites'] = verificar_limites(relatorio['candidato'], limites)
        relatorio['limites_ok'] = all(x['ok'] for x in relatorio['limites'])
    if args.gravar_logits: gravar_jsonl(saida / 'logits-teste.jsonl', [r for regs in itens for r in regs])
    relatorio['itens'] = [{'id': l.get('id'), 'registros': regs} for l, regs in zip(linhas, itens)]
    gravar_json(saida / 'avaliacao.json', relatorio)
    resumo = {k: v for k, v in relatorio.items() if k != 'itens'}
    for bloco in ('original_sem_calibracao', 'baseline_frequencia', 'baseline_calibrado', 'candidato'):
        if bloco in resumo:
            resumo[bloco] = {**resumo[bloco], 'saidas': [{k: v for k, v in s.items() if k != 'curva'} for s in resumo[bloco]['saidas']]}
            if 'metricas_decisao' in resumo[bloco]: resumo[bloco]['metricas_decisao'] = {k: v for k, v in resumo[bloco]['metricas_decisao'].items() if k != 'curva'}
    gravar_json(saida / 'resumo.json', resumo)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main, 'resumo.json'))
