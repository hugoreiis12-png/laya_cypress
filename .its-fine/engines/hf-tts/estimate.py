"""Estima se um modelo de voz cabe na GPU (ou na memoria unificada do Mac) antes de baixar pesos.

Formula: pesos bf16 + overhead + ativacao proporcional a duracao maxima do clipe (lote 1), com o
fator por segundo de cada familia medido nesta engine. E estimativa; a sondagem e que mede.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import engine_info  # noqa: E402

FRACAO_METAL = 0.66  # parte da memoria unificada que o Metal deixa a GPU usar


def estimar(params, segundos, gb_por_segundo, encaixe):
    pesos = params * encaixe['bytes_por_parametro']['lora'] / 1e9
    ativacao = segundos * gb_por_segundo
    return {'pesos_gb': round(pesos, 2), 'ativacoes_gb': round(ativacao, 2), 'overhead_gb': encaixe['overhead_gb'],
            'total_gb': round(pesos + ativacao + encaixe['overhead_gb'], 2)}


def veredito(necessario, disponivel, limites):
    razao = necessario / disponivel
    for nome in ('folga', 'cabe', 'aperta'):
        if razao <= limites[nome]: return nome
    return 'nao_cabe'


def main():
    parser = argparse.ArgumentParser()
    memoria = parser.add_mutually_exclusive_group(required=True)
    memoria.add_argument('--vram-gb', type=float, help='VRAM total da GPU NVIDIA em GB')
    memoria.add_argument('--memoria-unificada-gb', type=float, help='Mac com chip M: memoria unificada total em GB')
    parser.add_argument('--modelo', help='id do catalogo; sem ele avalia o catalogo inteiro')
    parser.add_argument('--segundos', type=float, help='duracao maxima do clipe; padrao o da receita')
    args = parser.parse_args()
    info = engine_info()
    disponivel = args.vram_gb or round(args.memoria_unificada_gb * FRACAO_METAL, 1)
    segundos = args.segundos or float(info['defaults']['train']['max_audio_seconds'])
    resultados = []
    for item in info['modelos']:
        if args.modelo and item['id'] != args.modelo: continue
        familia = info['familias'][item['familia']]
        est = estimar(item['params'], segundos, familia['gb_por_segundo'], info['encaixe'])
        resultados.append({'id': item['id'], 'familia': item['familia'], 'idiomas': familia['idiomas'], **est,
                           'veredito': veredito(est['total_gb'], disponivel, info['encaixe']['limites']), 'nota': item.get('nota')})
    if args.modelo and not resultados: raise SystemExit('MODELO_FORA_DO_CATALOGO: informe a familia na receita e use a sondagem')
    print(json.dumps({'disponivel_gb': disponivel, 'mac': args.memoria_unificada_gb is not None, 'segundos_por_clipe': segundos,
                      'cabem': [r['id'] for r in resultados if r['veredito'] != 'nao_cabe'], 'resultados': resultados,
                      'limitacao': 'estimativa por formula calibrada na bancada desta engine; a sondagem de treino e que mede'},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
