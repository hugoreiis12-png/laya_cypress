"""Estima se o treino do encoder cabe na GPU. Custo = estados x perguntas x comprimento.

Estimativa por formula; a sondagem (train.py --probe) e que mede.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AQUI, ler_json  # noqa: E402


def parametros_do_checkpoint(pasta, pesos):
    """Conta parametros pelo cabecalho do safetensors, sem carregar pesos; .bin estima por tamanho em fp32."""
    from safetensors import safe_open
    total = 0
    for nome in pesos:
        if not nome.endswith('.safetensors'):
            total += (Path(pasta) / nome).stat().st_size // 4
            continue
        with safe_open(str(Path(pasta) / nome), 'pt') as f:
            for k in f.keys():
                forma = f.get_slice(k).get_shape()
                n = 1
                for d in forma: n *= d
                total += n
    return total


def estimar(params, batch, perguntas, seq, arq, encaixe):
    pesos = params * encaixe['bytes_por_parametro'] / 1e9
    lora = params * 0.01 * 12 / 1e9
    cabecas = 0.06 * params * 12 / 1e9  # blocos extras, embedding de tipo e pontuador com Adam
    sequencias = batch * perguntas
    ativacao = sequencias * seq * arq['hidden'] * (arq['camadas'] + arq['camadas_cabeca']) * 2 * encaixe['fator_ativacao'] / 1e9
    total = pesos + lora + cabecas + ativacao + encaixe['overhead_gb']
    return {'pesos_gb': round(pesos, 2), 'lora_e_otimizador_gb': round(lora + cabecas, 2), 'ativacoes_gb': round(ativacao, 2),
            'overhead_gb': encaixe['overhead_gb'], 'total_gb': round(total, 2), 'sequencias_por_passo': sequencias}


def veredito(necessario, disponivel, limites):
    if not disponivel: return 'sem_gpu', 'Nenhuma GPU informada. O treino local exige GPU NVIDIA.'
    razao = necessario / disponivel
    if razao <= limites['folga']: return 'folga', 'Cabe com folga.'
    if razao <= limites['cabe']: return 'cabe', 'Cabe.'
    if razao <= limites['aperta']: return 'aperta', 'Aperta. Reduza o batch; a sondagem decide.'
    return 'nao_cabe', 'Nao cabe nessa GPU com esse batch.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vram-gb', type=float, required=True)
    parser.add_argument('--checkpoint', help='pasta do modelo local; sem ela usa as sugestoes do catalogo')
    parser.add_argument('--batch', type=int, default=None, help='estados por passo')
    parser.add_argument('--perguntas-por-estado', type=int, default=4)
    parser.add_argument('--seq', type=int, default=None, help='comprimento; padrao: max_len do checkpoint')
    args = parser.parse_args()
    engine = ler_json(AQUI / 'engine.json')
    encaixe, batch = engine['encaixe'], args.batch or engine['defaults']['train']['batch']
    resultados = []
    if args.checkpoint:
        import loader
        pasta = Path(args.checkpoint)
        desc = loader.descrever(pasta)
        cfg, ecfg = desc['cfg'], ler_json(Path(desc['encoder_dir']) / 'config.json')
        alvos = [{'id': str(pasta), 'params': parametros_do_checkpoint(pasta, desc['pesos']), 'hidden': ecfg['hidden_size'], 'camadas': ecfg['num_hidden_layers'],
                  'camadas_cabeca': cfg['head_layers'], 'max_len': cfg['max_len']}]
    else:
        alvos = [{k: c[k] for k in ('id', 'params', 'hidden', 'camadas', 'camadas_cabeca', 'max_len')}
                 for c in ler_json(AQUI / engine['checkpoints'])['sugestoes']]
    for alvo in alvos:
        seq = args.seq or alvo['max_len']
        est = estimar(alvo['params'], batch, args.perguntas_por_estado, seq, alvo, encaixe)
        classe, texto = veredito(est['total_gb'], args.vram_gb, encaixe['limites'])
        resultados.append({'id': alvo['id'], 'params': alvo['params'], 'batch': batch, 'perguntas_por_estado': args.perguntas_por_estado,
                           'seq': seq, **est, 'veredito': classe, 'explicacao': texto})
    print(json.dumps({'vram_gb': args.vram_gb, 'estimativa': True, 'cabem': [r['id'] for r in resultados if r['veredito'] in ('folga', 'cabe', 'aperta')],
                      'resultados': resultados, 'limitacao': encaixe['nota']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
