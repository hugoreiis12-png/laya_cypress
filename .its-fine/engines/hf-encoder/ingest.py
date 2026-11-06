"""Mapeamento declarativo: dados do projeto -> linhas tipadas {id, group_id, state, answers} e task.json.

O mapeamento diz de onde vem o estado, as perguntas e cada resposta. Duas fontes:
`campos` (JSONL com colunas) e `messages` (linhas de chat do proprio ITS-FINE, com
o estado no ultimo turno user e as respostas em JSON no turno assistant). Campos
fora do mapeamento vao para `meta` e nunca entram na sequencia. Sem Python escrito
na hora: o agente preenche o mapeamento e chama este script.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, gravar_jsonl, iterar_jsonl, ler_json, sha256_arquivo, politica_privacidade  # noqa: E402
from sequence import perguntas_da_tarefa, indice_do_rotulo  # noqa: E402
from itsfine.privacy import require_local_executor  # noqa: E402


def valor_mapeado(bruto, mapa):
    if mapa is None: return bruto
    chave = json.dumps(bruto, ensure_ascii=False) if not isinstance(bruto, str) else bruto
    if chave in mapa: return mapa[chave]
    if str(bruto).lower() in mapa: return mapa[str(bruto).lower()]
    raise ValueError('ANSWER_VALUE_NOT_MAPPED:%r' % (bruto,))


def normalizar_resposta(q, valor):
    """Aceita as formas comuns e devolve o valor canonico do tipo; None e rotulo ausente."""
    if valor is None or valor == '': return None
    if q['t'] == 'noul':
        if isinstance(valor, str) and valor.strip().lower() in ('true', 'false', 'sim', 'nao', 'não', 'yes', 'no', '1', '0'):
            return valor.strip().lower() in ('true', 'sim', 'yes', '1')
        if isinstance(valor, bool): return valor
        if valor in (0, 1): return bool(valor)
        raise ValueError('LABEL_NOT_BOOLEAN:%r' % (valor,))
    if q['t'] == 'score':
        if isinstance(valor, str) and valor.strip().lstrip('-').isdigit(): valor = int(valor)
        if isinstance(valor, float) and valor.is_integer(): valor = int(valor)
        indice_do_rotulo(q, valor)
        return valor
    valor = str(valor)
    indice_do_rotulo(q, valor)
    return valor


def extrair(linha, mapeamento, internas):
    fonte = mapeamento.get('fonte', 'campos')
    if fonte == 'messages':
        turnos = linha.get('messages', [])
        user = next((m['content'] for m in reversed(turnos) if m.get('role') == 'user'), None)
        assistant = next((m['content'] for m in reversed(turnos) if m.get('role') == 'assistant'), None)
        if user is None: raise ValueError('MESSAGES_WITHOUT_USER')
        try: state = json.loads(user)
        except ValueError: state = user
        try: brutas = json.loads(assistant) if assistant is not None else {}
        except ValueError: raise ValueError('ASSISTANT_NOT_JSON')
        if not isinstance(brutas, dict): raise ValueError('ASSISTANT_NOT_OBJECT')
    else:
        estado = mapeamento['state']
        campos = estado.get('campos') or ([estado['campo']] if estado.get('campo') else None)
        if not campos: raise ValueError('MAPPING_STATE_REQUIRED')
        # lista copia os nomes das colunas; dicionario renomeia: {"mensagem": "texto"} le a coluna texto como mensagem
        if isinstance(campos, dict):
            state = {novo: linha.get(origem) for novo, origem in campos.items()}
        else:
            state = linha[campos[0]] if len(campos) == 1 else {c: linha.get(c) for c in campos}
        brutas = linha
    answers = {}
    for nome, regra in mapeamento.get('answers', {}).items():
        if nome not in internas: raise ValueError('MAPPING_ANSWER_UNKNOWN:' + nome)
        regra = {'campo': regra} if isinstance(regra, str) else regra
        bruto = brutas.get(regra['campo'])
        if bruto is None: continue
        valor = normalizar_resposta(internas[nome], valor_mapeado(bruto, regra.get('map')))
        if valor is not None: answers[nome] = valor
    return state, answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fonte', required=True, help='JSONL do projeto')
    parser.add_argument('--mapeamento', required=True, help='JSON com fonte, state, answers, questions, decide, policy')
    parser.add_argument('--saida', required=True, help='pasta: rows.jsonl, task.json, resumo.json')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    mapeamento = ler_json(args.mapeamento)
    questions = mapeamento.get('questions') or ler_json(mapeamento['questions_file'])
    task = {'schema_version': 1, 'questions': questions}
    if mapeamento.get('decide'): task['decide'] = mapeamento['decide']
    if mapeamento.get('policy'): task['policy'] = mapeamento['policy']
    internas = perguntas_da_tarefa(task)
    if task.get('decide') and task['decide'] not in internas: raise ValueError('TASK_DECIDE_UNKNOWN')
    saida = Path(args.saida).resolve(); saida.mkdir(parents=True, exist_ok=True)
    if (saida / 'rows.jsonl').exists(): raise FileExistsError('INGEST_OUTPUT_EXISTS')
    campo_id, campo_grupo = mapeamento.get('id', 'id'), mapeamento.get('group', 'group_id')
    meta_campos = mapeamento.get('meta', [])
    linhas, problemas, contagem = [], [], {nome: {} for nome in internas}
    for numero, linha in enumerate(iterar_jsonl(args.fonte), 1):
        try:
            state, answers = extrair(linha, mapeamento, internas)
        except (ValueError, KeyError) as erro:
            problemas.append({'linha': numero, 'erro': str(erro)[:80]}); continue
        chave = hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        ident = str(linha.get(campo_id, 'row-%d-%s' % (numero, chave)))
        grupo = str(linha.get(campo_grupo, ident))
        registro = {'id': ident, 'group_id': grupo, 'state': state, 'answers': answers}
        meta = {c: linha[c] for c in meta_campos if c in linha}
        if meta: registro['meta'] = meta
        linhas.append(registro)
        for nome, valor in answers.items():
            chave_valor = str(valor).lower() if isinstance(valor, bool) else str(valor)
            contagem[nome][chave_valor] = contagem[nome].get(chave_valor, 0) + 1
    gravar_jsonl(saida / 'rows.jsonl', linhas)
    gravar_json(saida / 'task.json', task)
    resumo = {'schema_version': 1, 'engine': 'hf-encoder', 'linhas': len(linhas), 'problemas': len(problemas),
              'grupos': len({l['group_id'] for l in linhas}), 'fonte_hash': sha256_arquivo(args.fonte),
              'task_hash': sha256_arquivo(saida / 'task.json'),
              'saidas': [{'nome': nome, 'rotulados': sum(v.values()), 'opcoes': len(v)} for nome, v in contagem.items()],
              'avisos': [{'nome': nome, 'erro': 'OUTPUT_NEVER_LABELED'} for nome, v in contagem.items() if not v]}
    gravar_json(saida / 'resumo.json', resumo)
    gravar_json(saida / 'problemas.json', problemas)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0 if linhas else 1


if __name__ == '__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main, 'resumo.json'))
