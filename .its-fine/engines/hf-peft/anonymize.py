"""Anonimiza dados pessoais de um JSONL de treino; a ingestao usa as mesmas funcoes.

CPF, CNPJ, e-mail, telefone e CEP viram marcadores ([CPF], [CNPJ], [EMAIL],
[TELEFONE], [CEP]) por expressao regular. CPF e CNPJ formatados sempre saem; so
com digitos, apenas quando o digito verificador confere, para nao trocar qualquer
numero. Telefone e CEP so formatados. Com --nomes, um modelo local de
reconhecimento de entidades (token classification, baixado pelo
hf-encoder/download.py) troca nomes de pessoas por [PESSOA]. O modelo nao acerta
tudo: cada troca vai para revisao-anonimizacao.jsonl, que o usuario revisa localmente.

Percorre todos os textos da linha (messages, text, evidencia, state...) menos ids,
grupos, papeis, respostas rotuladas e caminhos de audio. Imprime so contagens; a
entrada fica intacta.
"""
from pathlib import Path
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, iterar_jsonl, ler_json, sha256_arquivo, politica_privacidade  # noqa: E402
from itsfine.privacy import require_local_executor  # noqa: E402

EMAIL = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
CNPJ = re.compile(r'(?<!\d)(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})(?!\d)')
CPF = re.compile(r'(?<!\d)(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})(?!\d)')
# Com DDD, celular de 9 digitos com hifen ou +55 seguido dos digitos. Fixo sem DDD
# fica de fora: "1939-1945" e intervalo de anos, nao telefone. DDD sem parenteses
# pede hifen ou formato de celular: "12 1500 2300" e linha de tabela.
TELEFONE = re.compile(r'(?<![\w+])(?:\+55\s?)?\(\d{2}\)\s?(?:9\s?)?\d{4}[-\s]?\d{4}(?!\d)'
                      r'|(?<![\w+])(?:\+55\s?)?\d{2}\s(?:9\d{4}[-\s]?\d{4}|\d{4}-\d{4})(?!\d)'
                      r'|(?<![\w-])9\d{4}-\d{4}(?!\d)|\+55\s?\d{10,11}(?!\d)')
CEP = re.compile(r'(?<![\w.-])(?:\d{5}|\d{2}\.\d{3})-\d{3}(?![\w-])')
FORA = {'id', 'group_id', 'source_hash', 'role', 'answers', 'audio'}
ROTULOS_PESSOA = {'PER', 'PESSOA', 'PERSON'}
TIPOS = ('cpf', 'cnpj', 'email', 'telefone', 'cep', 'pessoa')
REGRAS = 'br-v1'


def cpf_valido(d):
    if len(set(d)) == 1: return False
    for n in (9, 10):
        if sum(int(d[i]) * (n + 1 - i) for i in range(n)) * 10 % 11 % 10 != int(d[n]): return False
    return True


def cnpj_valido(d):
    if len(set(d)) == 1: return False
    for n in (12, 13):
        pesos = list(range(n - 7, 1, -1)) + list(range(9, 1, -1))
        resto = sum(int(x) * p for x, p in zip(d, pesos)) % 11
        if (0 if resto < 2 else 11 - resto) != int(d[n]): return False
    return True


PADROES = (('email', EMAIL, None), ('cnpj', CNPJ, cnpj_valido), ('cpf', CPF, cpf_valido),
           ('telefone', TELEFONE, None), ('cep', CEP, None))


class Nomes:
    """Nomes de pessoas por um modelo local de token classification, conferido contra o lock do download."""

    def __init__(self, pasta, limiar=0.6):
        pasta = Path(pasta)
        lock = ler_json(pasta / 'itsfine-download.json')
        for nome, esperado in lock['files'].items():
            if sha256_arquivo(pasta / nome) != esperado: raise ValueError('NER_MODEL_HASH_MISMATCH')
        self.id, self.revision, self.limiar = lock['id'], lock['revision'], limiar
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        tok = AutoTokenizer.from_pretrained(pasta, local_files_only=True)
        modelo = AutoModelForTokenClassification.from_pretrained(pasta, local_files_only=True)
        # Tokenizer sem limite declarado deixaria passar sequencia maior que as posicoes do modelo.
        tok.model_max_length = min(tok.model_max_length, modelo.config.max_position_embeddings, 512)
        self.pipe = pipeline('token-classification', model=modelo, tokenizer=tok, aggregation_strategy='first',
                             stride=64, device=0 if torch.cuda.is_available() else -1)

    def __call__(self, texto):
        spans = sorted((e['start'], e['end']) for e in self.pipe(texto)
                       if e['entity_group'].split('-')[-1] in ROTULOS_PESSOA and e['score'] >= self.limiar)
        juntos = []
        for ini, fim in spans:
            # "Maria" e "Silva" vindos separados viram um nome so
            if juntos and not texto[juntos[-1][1]:ini].strip(): juntos[-1][1] = max(fim, juntos[-1][1])
            else: juntos.append([ini, fim])
        return juntos


def anonimizar(texto, nomes=None):
    """Texto com os dados pessoais trocados por marcadores e a lista (tipo, original) do que saiu."""
    trocas = []

    def trocar(tipo, valido):
        def f(m):
            if valido and m.group(0).isdigit() and not valido(m.group(0)): return m.group(0)
            trocas.append((tipo, m.group(0)))
            return '[%s]' % tipo.upper()
        return f
    for tipo, padrao, valido in PADROES:
        texto = padrao.sub(trocar(tipo, valido), texto)
    if nomes and texto.strip():
        for ini, fim in reversed(nomes(texto)):
            trocas.append(('pessoa', texto[ini:fim]))
            texto = texto[:ini] + '[PESSOA]' + texto[fim:]
    return texto, trocas


def anonimizar_valor(valor, nomes=None):
    """Anonimiza cada texto de uma linha, menos ids, grupos, papeis, rotulos e caminhos de audio."""
    if isinstance(valor, str): return anonimizar(valor, nomes)
    trocas = []
    if isinstance(valor, dict):
        saida = {}
        for k, v in valor.items():
            saida[k], t = (v, []) if k in FORA else anonimizar_valor(v, nomes)
            trocas += t
        return saida, trocas
    if isinstance(valor, list):
        saida = []
        for v in valor:
            novo, t = anonimizar_valor(v, nomes)
            saida.append(novo); trocas += t
        return saida, trocas
    return valor, trocas


def contagem(tipos):
    c = Counter(tipos)
    return {t: c[t] for t in TIPOS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entrada', required=True, help='JSONL de treino: pares.jsonl, trechos.jsonl ou o do usuario')
    parser.add_argument('--saida', required=True, help='pasta de saida, dentro da rodada')
    parser.add_argument('--nomes', help='pasta do modelo de nomes baixado pelo hf-encoder/download.py')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    entrada, saida = Path(args.entrada).resolve(), Path(args.saida).resolve()
    # Nome proprio da revisao: a ingestao grava trocas.jsonl na mesma pasta.
    destino, revisao = saida / 'anonimizado.jsonl', saida / 'revisao-anonimizacao.jsonl'
    if entrada in (destino, revisao): raise ValueError('INPUT_IS_OUTPUT')
    nomes = Nomes(args.nomes) if args.nomes else None
    saida.mkdir(parents=True, exist_ok=True)
    linhas, alteradas, tipos = 0, 0, []
    # Temporario e troca no fim: rodada que falha nao deixa anonimizado.jsonl pela metade.
    temp_destino, temp_revisao = destino.with_suffix('.jsonl.tmp'), revisao.with_suffix('.jsonl.tmp')
    with temp_destino.open('w', encoding='utf-8', newline='\n') as out, temp_revisao.open('w', encoding='utf-8', newline='\n') as rev:
        for linha in iterar_jsonl(entrada):
            nova, trocas = anonimizar_valor(linha, nomes)
            out.write(json.dumps(nova, ensure_ascii=False) + '\n')
            for tipo, original in trocas:
                rev.write(json.dumps({'linha': linha.get('id', linhas), 'tipo': tipo, 'original': original}, ensure_ascii=False) + '\n')
            linhas += 1; alteradas += bool(trocas); tipos += [t for t, _ in trocas]
    os.replace(temp_revisao, revisao); os.replace(temp_destino, destino)
    gravar_json(saida / 'anonimizacao.json', {
        'schema_version': 1, 'regras': REGRAS, 'linhas': linhas, 'linhas_alteradas': alteradas, 'trocas': contagem(tipos),
        'modelo_nomes': {'id': nomes.id, 'revision': nomes.revision} if nomes else None,
        'saida_hash': sha256_arquivo(destino), 'revisao_humana': 'pendente: o usuario confere revisao-anonimizacao.jsonl localmente'})
    return 0


if __name__ == '__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main, 'anonimizacao.json'))
