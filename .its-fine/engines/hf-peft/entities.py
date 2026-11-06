"""Extrai entidades das respostas de referencia para a metrica entity_recall.

Entidade aqui e termo que a resposta certa precisa conter: comandos com barra,
identificadores com ponto ou sublinhado, nomes proprios com maiuscula no meio
da frase e termos que o usuario listar. Imprime so contagens; a lista vai
para o arquivo, que o usuario revisa.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_jsonl, politica_privacidade  # noqa: E402
from itsfine.privacy import require_local_executor

COMANDO = re.compile(r'(?<![\w/])/[a-z][\w-]{2,}')
IDENT = re.compile(r'\b[a-zA-Z_][\w]*(?:[._-][\w]+){1,}\b')
NOME = re.compile(r'(?<![.!?]\s)(?<!^)\b([A-ZÀ-Ý][\wÀ-ÿ]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ]+){0,3})\b')
PARAR = {'O', 'A', 'Os', 'As', 'Um', 'Uma', 'Ele', 'Ela', 'Isso', 'Esse', 'Essa', 'Este', 'Esta', 'Não', 'Sim', 'Para', 'Com', 'Sem', 'Por', 'The', 'This', 'That', 'It'}


def extrair(texto):
    achadas = set(COMANDO.findall(texto))
    achadas |= {m for m in IDENT.findall(texto) if not m.replace('.', '').replace('_', '').replace('-', '').isdigit()}
    for m in NOME.findall(texto):
        if m.split()[0] not in PARAR and len(m) > 3:
            achadas.add(m)
    return achadas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--referencias', required=True, help='JSONL com messages; usa o ultimo turno assistant')
    parser.add_argument('--extras', nargs='*', default=[], help='termos que o usuario quer exigir')
    parser.add_argument('--minimo', type=int, default=2, help='ocorrencias minimas para entrar')
    parser.add_argument('--saida', required=True)
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    contagem = {}
    linhas = ler_jsonl(args.referencias)
    for linha in linhas:
        for e in extrair(linha['messages'][-1]['content']):
            contagem[e] = contagem.get(e, 0) + 1
    escolhidas = sorted([e for e, n in contagem.items() if n >= args.minimo] + list(args.extras), key=str.lower)
    cobertura = sum(1 for l in linhas if any(unicodedata.normalize('NFKD', e).encode('ascii', 'ignore').decode().lower() in
                                             unicodedata.normalize('NFKD', l['messages'][-1]['content']).encode('ascii', 'ignore').decode().lower()
                                             for e in escolhidas))
    gravar_json(args.saida, {'schema_version': 1, 'entidades': escolhidas, 'ocorrencias': {e: contagem.get(e, 0) for e in escolhidas},
                             'referencias': len(linhas), 'referencias_com_entidade': cobertura, 'minimo': args.minimo,
                             'revisao_humana': 'pendente: remova termos genericos e acrescente os que faltam'})
    print(json.dumps({'entidades': len(escolhidas), 'referencias': len(linhas), 'referencias_com_entidade': cobertura,
                      'arquivo': str(args.saida)}, ensure_ascii=False, indent=2))
    return 0


if __name__=='__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main,"."))
