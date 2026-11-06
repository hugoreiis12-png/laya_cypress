"""Checagem de tokenizacao antes de treinar.

Confere, exemplo a exemplo, que a mascara de loss cobre exatamente o turno do
assistente, que o alvo termina em token de parada, e mede comprimentos para
sugerir max_seq_length. Deriva os tokens de parada do template, nao do config.
Nao carrega pesos.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_jsonl, prefixo_e_alvo, sha256_texto, tokens_de_parada, politica_privacidade  # noqa: E402
from itsfine.privacy import require_local_executor

LIMITES = [512, 768, 1024, 2048, 4096]


def percentil(valores, fracao):
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(len(ordenados) * fracao))] if ordenados else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', required=True, help='pasta com train/validation/test .jsonl')
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--saida', required=True, help='arquivo check.json')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.modelo, revision=args.revision)
    paradas = tokens_de_parada(tok)
    relatorio = {'schema_version': 1, 'modelo': args.modelo, 'revision': args.revision,
                 'tokenizer': type(tok).__name__, 'vocab': len(tok),
                 'eos_config': tok.eos_token, 'eos_config_id': tok.eos_token_id,
                 'tokens_de_parada': [{'id': i, 'token': tok.convert_ids_to_tokens(i)} for i in paradas],
                 'template_hash': sha256_texto(tok.chat_template or ''), 'splits': {}, 'problemas': []}
    if paradas and paradas[0] != tok.eos_token_id:
        relatorio['aviso_eos'] = ('O turno fecha com %s mas o config declara EOS %s. A geracao precisa dos dois; '
                                  'a engine usa a lista tokens_de_parada.' % (tok.convert_ids_to_tokens(paradas[0]), tok.eos_token))

    modo = None
    for nome in ('train', 'validation', 'test'):
        caminho = Path(args.dados) / (nome + '.jsonl')
        if not caminho.exists():
            continue
        linhas = ler_jsonl(caminho)
        comprimentos, alvos = [], []
        for linha in linhas:
            if 'messages' in linha:
                modo = 'messages'
                try:
                    completo, corte = prefixo_e_alvo(tok, linha['messages'])
                except ValueError as erro:
                    relatorio['problemas'].append({'id': linha.get('id'), 'erro': str(erro)})
                    continue
                alvo = completo[corte:]
                if not any(i in paradas for i in alvo):
                    relatorio['problemas'].append({'id': linha.get('id'), 'erro': 'alvo sem token de parada'})
                texto = tok.decode(alvo, skip_special_tokens=True).strip()
                if texto[:40] != linha['messages'][-1]['content'].strip()[:40]:
                    relatorio['problemas'].append({'id': linha.get('id'), 'erro': 'alvo nao corresponde ao turno assistant'})
                comprimentos.append(len(completo))
                alvos.append(len(alvo))
            else:
                modo = 'text'
                ids = tok(linha['text'])['input_ids']
                comprimentos.append(len(ids))
                alvos.append(len(ids))
        relatorio['splits'][nome] = {
            'linhas': len(linhas), 'tokens_total': sum(comprimentos),
            'comprimento': {'min': min(comprimentos) if comprimentos else 0, 'mediana': percentil(comprimentos, 0.5),
                            'p95': percentil(comprimentos, 0.95), 'max': max(comprimentos) if comprimentos else 0},
            'tokens_alvo': {'mediana': percentil(alvos, 0.5), 'max': max(alvos) if alvos else 0},
            'excedem_limite': {str(l): sum(1 for c in comprimentos if c > l) for l in LIMITES},
        }
    relatorio['modo'] = modo
    sugerido = None
    for limite in LIMITES:
        if all(s['excedem_limite'][str(limite)] == 0 for s in relatorio['splits'].values()):
            sugerido = limite
            break
    relatorio['max_seq_length_sugerido'] = sugerido
    relatorio['aprovado'] = not relatorio['problemas'] and sugerido is not None
    gravar_json(args.saida, relatorio)
    resumo = {k: relatorio[k] for k in ('aprovado', 'modo', 'max_seq_length_sugerido', 'tokens_de_parada')}
    resumo['problemas'] = len(relatorio['problemas'])
    resumo['splits'] = {k: {'linhas': v['linhas'], 'max': v['comprimento']['max']} for k, v in relatorio['splits'].items()}
    if 'aviso_eos' in relatorio:
        resumo['aviso_eos'] = relatorio['aviso_eos']
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0 if relatorio['aprovado'] else 1


if __name__=='__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main,"."))
