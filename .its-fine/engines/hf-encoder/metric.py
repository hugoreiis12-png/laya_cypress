"""Valida o avaliador de decisoes tipadas com casos de resultado conhecido.

O rotulo e a verdade, mas o instrumento ainda pode errar: classe positiva trocada,
opcoes fora de ordem, limiares invertidos. Cada caso tem resultado esperado; o
relatorio so e valido quando todos batem. Sondas rotuladas do projeto (opcional)
passam pelo mesmo avaliador e entram no relatorio.
"""
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_json, ler_jsonl, sha256_arquivo, politica_privacidade  # noqa: E402
from decisions import metricas_saida, matriz_operacional, validar_politica, verificar_limites  # noqa: E402
from sequence import QTYPES, perguntas_da_tarefa  # noqa: E402
from itsfine.privacy import require_local_executor  # noqa: E402


def registros_sinteticos(qtype, k, n, certos=True, confianca=0.95, label_fixo=None):
    regs = []
    for i in range(n):
        label = i % k if label_fixo is None else label_fixo
        previsto = label if certos else (label + 1) % k
        base = math.log((1 - confianca) / (k - 1)) if k > 1 else 0.0
        logits = [base] * k; logits[previsto] = math.log(confianca)
        regs.append({'nome': 'sintetica', 'qtype': qtype, 'k': k, 'logits': logits, 'label': label})
    return regs


def casos():
    q_noul = {'t': 'noul', 'ins': 'x', 'crit': None}
    politica = validar_politica({'high': 0.6, 'low': 0.4, 'middle_action': 'revisar', 'positive': 'true'}, q_noul)
    invertida = validar_politica({'high': 0.6, 'low': 0.4, 'middle_action': 'revisar', 'positive': 'false'}, q_noul)
    perfeito = registros_sinteticos(QTYPES['noul'], 2, 40)
    errado = registros_sinteticos(QTYPES['noul'], 2, 40, certos=False)
    resultados = []
    m = metricas_saida(perfeito)
    resultados.append({'nome': 'perfeito_noul', 'esperado': 'acuracia 1, log_loss < 0.1, ece < 0.1',
                       'ok': m['acuracia'] == 1.0 and m['log_loss'] < 0.1 and m['ece'] < 0.1, 'acuracia': m['acuracia'], 'log_loss': m['log_loss'], 'ece': m['ece']})
    m = metricas_saida(errado)
    resultados.append({'nome': 'invertido_noul', 'esperado': 'acuracia 0', 'ok': m['acuracia'] == 0.0, 'acuracia': m['acuracia']})
    op = matriz_operacional(perfeito, politica)
    resultados.append({'nome': 'operacional_perfeito', 'esperado': 'fn 0, fp 0, cobertura 1',
                       'ok': op['automatica']['fn'] == 0 and op['automatica']['fp'] == 0 and op['automatica']['cobertura_automatica'] == 1.0,
                       'fn': op['automatica']['fn'], 'fp': op['automatica']['fp']})
    so_positivos = registros_sinteticos(QTYPES['noul'], 2, 40, label_fixo=1)
    certa, op2 = matriz_operacional(so_positivos, politica), matriz_operacional(so_positivos, invertida)
    resultados.append({'nome': 'classe_positiva_trocada', 'esperado': 'com a politica certa 40 positivos e 40 tp; com a classe trocada 0 positivos',
                       'ok': certa['positivos'] == 40 and certa['automatica']['tp'] == 40 and op2['positivos'] == 0, 'positivos': op2['positivos']})
    incerto = registros_sinteticos(QTYPES['noul'], 2, 40, confianca=0.55)
    op3 = matriz_operacional(incerto, politica)
    resultados.append({'nome': 'faixa_revisar', 'esperado': 'tudo encaminhado, cobertura 0',
                       'ok': op3['automatica']['encaminhados'] == 40 and op3['automatica']['cobertura_automatica'] == 0.0, 'encaminhados': op3['automatica']['encaminhados']})
    score = registros_sinteticos(QTYPES['score'], 3, 30)
    m = metricas_saida(score)
    resultados.append({'nome': 'score_mae', 'esperado': 'mae < 0.2 com probabilidade 0,95 no nivel certo', 'ok': m['mae'] < 0.2, 'mae': m['mae']})
    permutado = [{**r, 'label': (r['label'] + 1) % 3} for r in score]
    m = metricas_saida(permutado)
    resultados.append({'nome': 'opcoes_fora_de_ordem', 'esperado': 'acuracia 0 quando os rotulos usam outra ordem', 'ok': m['acuracia'] == 0.0, 'acuracia': m['acuracia']})
    limites = verificar_limites({'operacional': op, 'metricas_decisao': metricas_saida(perfeito)}, {'max_taxa_fn_liberados': 0.0, 'min_cobertura_automatica': 0.9})
    resultados.append({'nome': 'limites', 'esperado': 'ambos ok', 'ok': all(x['ok'] for x in limites)})
    return resultados


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarefa', help='task.json do projeto: valida perguntas, decide e politica')
    parser.add_argument('--sondas', help='JSONL opcional de itens rotulados com logits (de evaluate.py --gravar-logits)')
    parser.add_argument('--saida', required=True, help='metric-calibration.json')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    resultados = casos()
    relatorio = {'schema_version': 1, 'metric': 'decisao_tipada', 'engine': 'hf-encoder', 'casos': resultados}
    if args.tarefa:
        task = ler_json(args.tarefa)
        internas = perguntas_da_tarefa(task)
        decide = task.get('decide')
        if decide:
            if decide not in internas: raise ValueError('TASK_DECIDE_UNKNOWN')
            validar_politica(task.get('policy'), internas[decide])
        relatorio['tarefa'] = {'hash': sha256_arquivo(args.tarefa), 'perguntas': len(internas), 'decide': decide, 'politica_valida': True}
    if args.sondas:
        regs = ler_jsonl(args.sondas)
        por_saida = {}
        for r in regs: por_saida.setdefault(r['nome'], []).append(r)
        relatorio['sondas'] = {'hash': sha256_arquivo(args.sondas), 'saidas': [{'nome': n, **{k: v for k, v in metricas_saida(v).items() if k != 'curva'}} for n, v in por_saida.items()]}
    relatorio['valid'] = all(c['ok'] for c in resultados)
    relatorio['limitation'] = 'Valida o instrumento, nao a qualidade dos rotulos; rotulos errados continuam errados.'
    gravar_json(args.saida, relatorio)
    print(json.dumps({'valid': relatorio['valid'], 'casos': [{'nome': c['nome'], 'ok': c['ok']} for c in resultados]}, ensure_ascii=False, indent=2))
    return 0 if relatorio['valid'] else 2


if __name__ == '__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main, '.'))
