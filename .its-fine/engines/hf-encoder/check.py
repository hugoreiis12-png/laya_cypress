"""Checagem antes de treinar: rotulos, marcadores, truncamento, comprimentos e suporte.

Constroi a sequencia de cada (estado, pergunta) com o tokenizer do checkpoint e
recusa a linha cujo truncamento removeria um marcador de opcao. Nao carrega pesos.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_json, ler_jsonl, sha256_arquivo, sha256_texto, politica_privacidade  # noqa: E402
import loader  # noqa: E402
from sequence import SEQUENCE_VERSION, perguntas_da_tarefa, expandir, opcoes  # noqa: E402
from itsfine.privacy import require_local_executor  # noqa: E402

LIMITES = [256, 512, 768, 1024, 2048]
SPLITS = ('train', 'validation', 'calibration', 'test')


def percentil(valores, fracao):
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(len(ordenados) * fracao))] if ordenados else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', required=True, help='pasta com os splits e task.json')
    parser.add_argument('--checkpoint', required=True, help='pasta do checkpoint (tokenizer e limites)')
    parser.add_argument('--tarefa', help='task.json; padrao: dados/task.json')
    parser.add_argument('--saida', required=True, help='arquivo check.json')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    dados = Path(args.dados)
    tarefa = Path(args.tarefa) if args.tarefa else dados / 'task.json'
    task = ler_json(tarefa)
    internas = perguntas_da_tarefa(task)
    cfg = loader.descrever(args.checkpoint)['cfg']
    tok = loader.carregar_tokenizer(args.checkpoint)
    max_len, head_max_len = cfg['max_len'], cfg['head_max_len']
    relatorio = {'schema_version': 1, 'engine': 'hf-encoder', 'checkpoint': str(args.checkpoint), 'tokenizer': type(tok).__name__,
                 'vocab': len(tok), 'max_len': max_len, 'head_max_len': head_max_len, 'sequence_version': SEQUENCE_VERSION,
                 'template_hash': sha256_texto('hf-encoder:sequence:%d:%d:%d' % (SEQUENCE_VERSION, max_len, head_max_len)),
                 'task_hash': sha256_arquivo(tarefa), 'perguntas': {n: {'tipo': q['t'], 'opcoes': len(opcoes(q))} for n, q in internas.items()},
                 'splits': {}, 'problemas': [], 'avisos': []}
    rotulados_total = {nome: 0 for nome in internas}
    for nome_split in SPLITS:
        caminho = dados / (nome_split + '.jsonl')
        if not caminho.exists(): continue
        linhas = ler_jsonl(caminho)
        comprimentos, sequencias = [], 0
        suporte = {nome: {op: 0 for op in opcoes(q)} for nome, q in internas.items()}
        for linha in linhas:
            try:
                itens = expandir(tok, linha['state'], internas, max_len, head_max_len, linha.get('answers', {}))
            except (ValueError, KeyError, TypeError) as erro:
                relatorio['problemas'].append({'split': nome_split, 'id': linha.get('id'), 'erro': str(erro)[:80]}); continue
            for item in itens:
                comprimentos.append(len(item['ids'])); sequencias += 1
                if item['label'] >= 0:
                    suporte[item['nome']][opcoes(internas[item['nome']])[item['label']]] += 1
                    rotulados_total[item['nome']] += 1
        relatorio['splits'][nome_split] = {
            'estados': len(linhas), 'sequencias': sequencias, 'tokens_total': sum(comprimentos),
            'comprimento': {'min': min(comprimentos) if comprimentos else 0, 'mediana': percentil(comprimentos, 0.5),
                            'p95': percentil(comprimentos, 0.95), 'max': max(comprimentos) if comprimentos else 0},
            'no_limite': sum(1 for c in comprimentos if c >= max_len),
            'suporte': suporte,
            'suporte_insuficiente': [{'saida': nome, 'opcao': op, 'n': n} for nome, ops in suporte.items() for op, n in ops.items()
                                     if n < 5 and nome_split in ('validation', 'calibration')]}
    for nome, n in rotulados_total.items():
        if n == 0: relatorio['avisos'].append({'nome': nome, 'erro': 'OUTPUT_NEVER_LABELED'})
    maior = max((s['comprimento']['max'] for s in relatorio['splits'].values()), default=0)
    sugerido = next((l for l in LIMITES if l >= maior and l <= max_len), max_len)
    relatorio['max_seq_length_sugerido'] = sugerido
    relatorio['estados_no_limite'] = sum(s['no_limite'] for s in relatorio['splits'].values())
    relatorio['aprovado'] = not relatorio['problemas'] and bool(relatorio['splits'])
    gravar_json(args.saida, relatorio)
    resumo = {k: relatorio[k] for k in ('aprovado', 'max_seq_length_sugerido', 'max_len', 'head_max_len')}
    resumo['problemas'] = len(relatorio['problemas']); resumo['avisos'] = relatorio['avisos']
    resumo['splits'] = {k: {'estados': v['estados'], 'sequencias': v['sequencias'], 'max': v['comprimento']['max'],
                            'suporte_insuficiente': len(v['suporte_insuficiente'])} for k, v in relatorio['splits'].items()}
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0 if relatorio['aprovado'] else 1


if __name__ == '__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main, '.'))
