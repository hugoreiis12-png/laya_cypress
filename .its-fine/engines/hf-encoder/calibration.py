"""Temperatura por (tipo, numero de opcoes) ajustada na particao de calibracao.

Minimiza a log-loss em cada bucket dentro de [TEMP_MIN, TEMP_MAX]. Bucket com suporte
abaixo de min_support fica em 1,0 e e listado. O resultado leva o hash dos pesos
calibrados e o hash da particao usada.
"""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_json, sha256_arquivo, sha256_json, politica_privacidade  # noqa: E402
from decisions import TEMP_MIN, TEMP_MAX, probabilidades  # noqa: E402
from sequence import QTYPES, temp_bucket  # noqa: E402
from itsfine.privacy import require_local_executor  # noqa: E402


def log_loss(registros, T):
    total = 0.0
    for r in registros:
        p = probabilidades(r['logits'], r['qtype'], r['k'], {'temperature': [T, T, T]})
        total -= math.log(max(float(p[r['label']]), 1e-12))
    return total / len(registros)


def ajustar_temperatura(registros, lo=TEMP_MIN, hi=TEMP_MAX):
    """Busca em log(T): grade grossa e depois secao aurea no melhor intervalo."""
    grade = np.exp(np.linspace(math.log(lo), math.log(hi), 25))
    perdas = [log_loss(registros, float(t)) for t in grade]
    i = int(np.argmin(perdas))
    a, b = math.log(grade[max(0, i - 1)]), math.log(grade[min(len(grade) - 1, i + 1)])
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = log_loss(registros, math.exp(c)), log_loss(registros, math.exp(d))
    for _ in range(40):
        if fc < fd: b, d, fd = d, c, fc; c = b - phi * (b - a); fc = log_loss(registros, math.exp(c))
        else: a, c, fc = c, d, fd; d = a + phi * (b - a); fd = log_loss(registros, math.exp(d))
    T = math.exp((a + b) / 2)
    return float(min(hi, max(lo, T))), log_loss(registros, 1.0), log_loss(registros, T)


def ajustar(registros, min_support=30):
    """registros: itens rotulados {'qtype','k','logits','label'}. Devolve o calibrador."""
    rotulados = [r for r in registros if r.get('label', -1) >= 0]
    buckets, por_tipo = {}, {}
    for r in rotulados:
        buckets.setdefault(temp_bucket(r['qtype'], r['k']), []).append(r)
        por_tipo.setdefault(int(r['qtype']), []).append(r)
    resultado = {'schema_version': 1, 'method': 'temperature', 'min_support': int(min_support),
                 'bucket': ['type', 'n_options'], 'temperature': [1.0, 1.0, 1.0], 'temperature_by_options': {},
                 'buckets': {}, 'sem_suporte': [], 'itens': len(rotulados)}
    for nome, itens in sorted(buckets.items()):
        if len(itens) < min_support:
            resultado['buckets'][nome] = {'temperature': 1.0, 'support': len(itens), 'ajustado': False}
            resultado['sem_suporte'].append(nome)
            continue
        T, antes, depois = ajustar_temperatura(itens)
        resultado['buckets'][nome] = {'temperature': round(T, 4), 'support': len(itens), 'ajustado': True,
                                      'log_loss_antes': round(antes, 4), 'log_loss_depois': round(depois, 4)}
        resultado['temperature_by_options'][nome] = round(T, 4)
    for tipo, itens in por_tipo.items():
        if len(itens) >= min_support:
            resultado['temperature'][tipo] = round(ajustar_temperatura(itens)[0], 4)
    return resultado


def main():
    parser = argparse.ArgumentParser(description='ajusta temperaturas a partir de logits gravados por evaluate.py --gravar-logits')
    parser.add_argument('--logits', required=True, help='JSONL de itens rotulados com logits')
    parser.add_argument('--saida', required=True, help='calibration.json')
    parser.add_argument('--min-support', type=int, default=30)
    parser.add_argument('--pesos-hash', default=None)
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    registros = [json.loads(l) for l in Path(args.logits).read_text(encoding='utf-8').splitlines() if l.strip()]
    calib = ajustar(registros, args.min_support)
    calib['partition_hash'] = sha256_arquivo(args.logits)
    calib['weights_hash'] = args.pesos_hash
    calib['hash'] = sha256_json({k: v for k, v in calib.items() if k != 'hash'})
    gravar_json(args.saida, calib)
    print(json.dumps({'itens': calib['itens'], 'buckets': len(calib['buckets']), 'sem_suporte': len(calib['sem_suporte'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
