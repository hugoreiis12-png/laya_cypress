"""Estima se um modelo cabe na GPU antes de baixar pesos ou treinar.

Principio: falhar antes de tentar, nao no meio do treino.
E estimativa por formula; a sondagem (train.py --probe) e que mede.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ler_json  # noqa: E402

AQUI = Path(__file__).resolve().parent
# Mac com chip M: o Metal deixa a GPU usar cerca de 2/3 da memoria unificada (recommendedMaxWorkingSetSize)
FRACAO_METAL = 0.66


def arquitetura(model_id, revision=None):
    """hidden e camadas pelo config do modelo; baixa so o config.json."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, revision=revision)
        texto = getattr(cfg, 'text_config', None) or cfg
        return {'hidden': int(getattr(texto, 'hidden_size', 0) or 0),
                'camadas': int(getattr(texto, 'num_hidden_layers', 0) or 0)}
    except Exception:
        return {'hidden': 0, 'camadas': 0}


def estimar(params, metodo, batch, seq, arq, encaixe):
    bpp = encaixe['bytes_por_parametro'][metodo]
    pesos = params * bpp / 1e9
    lora = 0.0 if metodo == 'full' else params * 0.01 * 12 / 1e9
    if arq['hidden'] and arq['camadas']:
        ativacao = batch * seq * arq['hidden'] * arq['camadas'] * 2 * encaixe['fator_ativacao_checkpointing'] / 1e9
    else:
        ativacao = batch * seq * 2048 * 24 * 2 * encaixe['fator_ativacao_checkpointing'] / 1e9
    total = pesos + lora + ativacao + encaixe['overhead_gb']
    return {'pesos_gb': round(pesos, 2), 'lora_e_otimizador_gb': round(lora, 2),
            'ativacoes_gb': round(ativacao, 2), 'overhead_gb': encaixe['overhead_gb'],
            'total_gb': round(total, 2)}


def veredito(necessario, disponivel, limites):
    if not disponivel:
        return 'sem_gpu', 'Nenhuma GPU informada. Treino em CPU nao e viavel para esses tamanhos.'
    razao = necessario / disponivel
    if razao <= limites['folga']:
        return 'folga', 'Cabe com folga.'
    if razao <= limites['cabe']:
        return 'cabe', 'Cabe.'
    if razao <= limites['aperta']:
        return 'aperta', 'Aperta. Reduza o batch para 2 ou o comprimento maximo; a sondagem decide.'
    return 'nao_cabe', 'Nao cabe nessa GPU com esse metodo.'


def main():
    parser = argparse.ArgumentParser()
    memoria = parser.add_mutually_exclusive_group(required=True)
    memoria.add_argument('--vram-gb', type=float, help='VRAM total da GPU NVIDIA em GB')
    memoria.add_argument('--memoria-unificada-gb', type=float, help='Mac com chip M: memoria unificada total em GB')
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--seq', type=int, default=512)
    parser.add_argument('--modelo', help='um id do catalogo ou do Hugging Face; sem esse argumento, avalia o catalogo inteiro')
    parser.add_argument('--params', type=int, help='parametros, para modelo fora do catalogo')
    parser.add_argument('--metodo', choices=['lora', 'qlora', 'full'], default=None)
    parser.add_argument('--sem-rede', action='store_true', help='nao baixa config.json; usa arquitetura generica')
    args = parser.parse_args()

    mac = args.memoria_unificada_gb is not None
    if mac:
        args.vram_gb = round(args.memoria_unificada_gb * FRACAO_METAL, 1)
        if args.metodo == 'qlora': raise SystemExit('QUANTIZATION_REQUIRES_CUDA: QLoRA nao existe no Mac')
    engine = ler_json(AQUI / 'engine.json')
    encaixe, defaults = engine['encaixe'], engine['defaults']
    batch = args.batch or defaults['train']['batch']
    catalogo = {m['id']: m for m in engine['modelos']}

    alvos = [args.modelo] if args.modelo else list(catalogo)
    resultados = []
    for ident in alvos:
        item = catalogo.get(ident, {'id': ident, 'params': args.params, 'qlora': 'desconhecido', 'nota': 'fora do catalogo'})
        if not item.get('params'):
            resultados.append({'id': ident, 'erro': 'informe --params para modelo fora do catalogo'})
            continue
        arq = {'hidden': 0, 'camadas': 0} if args.sem_rede else arquitetura(ident, None)
        metodos = [args.metodo] if args.metodo else (['lora'] if mac else ['lora', 'qlora'])
        por_metodo = {}
        for metodo in metodos:
            est = estimar(item['params'], metodo, batch, args.seq, arq, encaixe)
            classe, texto = veredito(est['total_gb'], args.vram_gb, encaixe['limites'])
            if metodo == 'qlora' and item.get('qlora') == 'desaconselhado':
                texto += ' Atencao: QLoRA e desaconselhado pelo fornecedor para esta familia.'
                classe = 'desaconselhado' if classe != 'nao_cabe' else classe
            por_metodo[metodo] = {**est, 'veredito': classe, 'explicacao': texto}
        melhor = None
        for metodo in ('lora', 'qlora'):
            if metodo in por_metodo and por_metodo[metodo]['veredito'] in ('folga', 'cabe', 'aperta'):
                melhor = metodo
                break
        resultados.append({'id': ident, 'params': item['params'], 'familia': item.get('familia'),
                           'licenca': item.get('licenca'), 'gated': item.get('gated'),
                           'arquitetura': arq, 'batch': batch, 'seq': args.seq,
                           'metodos': por_metodo, 'recomendado': melhor, 'nota': item.get('nota')})

    cabem = [r for r in resultados if r.get('recomendado')]
    print(json.dumps({'vram_gb': args.vram_gb, **({'mac': {'memoria_unificada_gb': args.memoria_unificada_gb, 'fracao_metal': FRACAO_METAL}} if mac else {}),
                      'cabem': [r['id'] for r in cabem],
                      'resultados': resultados,
                      'limitacao': 'estimativa por formula, calibrada em medicoes publicas e desta engine; a sondagem de treino e que mede'},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
