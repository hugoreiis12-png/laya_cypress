"""Sequencia por pergunta, marcadores e lote, iguais a referencia do Laya.

Formato: [CLS] <tipo> instrucao [SEP] [MASK] opcao0 [MASK] opcao1 ... [SEP] estado [SEP].
O modelo pontua o estado oculto de cada [MASK]. Uma sequencia por pergunta.
Este modulo e o unico preparador: treino, avaliacao, predict e serve usam ele.
Reimplementado a partir de laya/common.py e laya/agent.py no commit citado em
third_party/laya/NOTICE.md (Apache 2.0). Vale para qualquer tokenizer com [MASK], [CLS] e [SEP].
"""
import json

QTYPES = {'choice': 0, 'score': 1, 'noul': 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
SEQUENCE_VERSION = 1
_NOUL_PADRAO = {'false': 'false', 'true': 'true'}


def serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(', ', ': '), default=str)


def _resolve_noul_labels(labels=None):
    labels = _NOUL_PADRAO if labels is None else labels
    if not isinstance(labels, dict) or set(labels) != {'false', 'true'}:
        raise ValueError('NOUL_LABELS_INVALID')
    f, t = labels['false'], labels['true']
    if not isinstance(f, str) or not isinstance(t, str):
        raise ValueError('NOUL_LABELS_INVALID')
    f, t = f.strip(), t.strip()
    if not f or not t or f == t:
        raise ValueError('NOUL_LABELS_INVALID')
    return f, t


def render_options(q):
    """Textos das opcoes na ordem dos rotulos. Noul e sempre [false, true]."""
    t, crit = q['t'], q.get('crit')
    if t != 'noul' and 'labels' in q:
        raise ValueError('LABELS_ONLY_FOR_NOUL')
    if t == 'choice':
        return [k if v is None or v == '' else '%s: %s' % (k, render_criterion(v)) for k, v in crit.items()]
    if t == 'score':
        return ['level %d: %s' % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get('labels'))
    false_crit, true_crit = crit.get('false'), crit.get('true')
    return [false_label + ': ' + (render_criterion(false_crit) if false_crit not in (None, '') else 'no, the statement does not hold'),
            true_label + ': ' + (render_criterion(true_crit) if true_crit not in (None, '') else 'yes, the statement holds')]


def build_sequence(tok, state, q, max_len=512, head_max_len=192, option_order=None, truncate_left=False):
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q['ins']).replace(mask_tok, ' ')
    head_ids = tok('%s question: %s' % (q['t'], ins), add_special_tokens=False)['input_ids']
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id] + tok(' ' + opts[i].replace(mask_tok, ' '), add_special_tokens=False)['input_ids'][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, ' '), add_special_tokens=False)['input_ids']
    # listas (conversas) truncam pela esquerda para preservar o turno mais recente
    st = st[max(0, len(st) - room):] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def validar_pergunta(nome, qdef):
    """Recusa pergunta mal definida nomeando o problema (espelha Agent._check_question)."""
    if not isinstance(qdef, dict): raise ValueError('QUESTION_NOT_OBJECT:' + str(nome))
    t = qdef.get('type')
    if t not in QTYPES: raise ValueError('QUESTION_TYPE_UNKNOWN:%s:%s' % (nome, t))
    if 'instructions' not in qdef: raise ValueError('QUESTION_WITHOUT_INSTRUCTIONS:' + str(nome))
    crit = qdef.get('criteria')
    if t == 'choice':
        if not isinstance(crit, (dict, list)) or not crit: raise ValueError('CHOICE_CRITERIA_REQUIRED:' + str(nome))
    elif t == 'score':
        if not isinstance(crit, list) or not crit: raise ValueError('SCORE_CRITERIA_REQUIRED:' + str(nome))
    elif crit is not None and not isinstance(crit, dict):
        raise ValueError('NOUL_CRITERIA_INVALID:' + str(nome))
    elif isinstance(crit, dict):
        if not {str(k).lower() for k in crit} <= {'true', 'false'}: raise ValueError('NOUL_CRITERIA_KEYS:' + str(nome))
    if 'labels' in qdef:
        if t != 'noul': raise ValueError('LABELS_ONLY_FOR_NOUL:' + str(nome))
        _resolve_noul_labels(qdef['labels'])


def internalizar(qdef):
    """Formato externo {type, instructions, criteria} -> interno {t, ins, crit}."""
    t = qdef['type']
    crit = qdef.get('criteria')
    if t == 'choice' and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == 'noul' and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef['instructions']
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    q = {'t': t, 'ins': ins, 'crit': crit}
    if 'labels' in qdef: q['labels'] = qdef['labels']
    return q


def opcoes(q):
    """Nomes das opcoes na ordem em que o modelo as pontua."""
    if q['t'] == 'choice': return list(q['crit'].keys())
    if q['t'] == 'score': return [str(i) for i in range(len(q['crit']))]
    return ['false', 'true']


def indice_do_rotulo(q, valor):
    """Valor do dataset -> indice da opcao; None quando nao ha rotulo."""
    if valor is None: return None
    if q['t'] == 'choice':
        chaves = list(q['crit'].keys())
        if valor not in chaves: raise ValueError('LABEL_NOT_AN_OPTION:%r' % (valor,))
        return chaves.index(valor)
    if q['t'] == 'score':
        if isinstance(valor, bool) or not isinstance(valor, int) or not 0 <= valor < len(q['crit']):
            raise ValueError('LABEL_NOT_A_LEVEL:%r' % (valor,))
        return valor
    if isinstance(valor, bool): return int(valor)
    if valor in (0, 1): return int(valor)
    if isinstance(valor, str) and valor.lower() in ('true', 'false'): return int(valor.lower() == 'true')
    raise ValueError('LABEL_NOT_BOOLEAN:%r' % (valor,))


def temp_bucket(qtype, k):
    size = '2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'
    return '%s:%s' % (QTYPE_NAMES[int(qtype)], size)


def perguntas_da_tarefa(task):
    """Valida e internaliza as perguntas de task.json, preservando a ordem declarada."""
    questions = task.get('questions')
    if not isinstance(questions, dict) or not questions: raise ValueError('TASK_WITHOUT_QUESTIONS')
    internas = {}
    for nome, qdef in questions.items():
        validar_pergunta(nome, qdef)
        internas[nome] = internalizar(qdef)
    return internas


def expandir(tok, state, internas, max_len, head_max_len, answers=None):
    """Um item por pergunta do estado. Rotulo ausente vira label None (mascarado)."""
    truncate_left = isinstance(state, list)
    itens = []
    for nome, q in internas.items():
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len, truncate_left=truncate_left)
        n_opcoes = len(render_options(q))
        if len(markers) != n_opcoes:
            raise ValueError('TRUNCATION_REMOVED_MARKER:%s' % nome)
        rotulo = indice_do_rotulo(q, (answers or {}).get(nome)) if answers is not None else None
        itens.append({'ids': seq, 'markers': markers, 'qtype': QTYPES[q['t']], 'k': n_opcoes,
                      'nome': nome, 'label': -1 if rotulo is None else rotulo})
    return itens


def collate_items(batch, pad_id):
    """Lote de grupos (um grupo por estado) -> tensores. Ordem dos itens preservada."""
    import torch
    items = [it for group in batch for it in group]
    if not items: return None
    n, L = len(items), max(len(it['ids']) for it in items)
    kmax = max(len(it['markers']) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    for i, it in enumerate(items):
        ids[i, :len(it['ids'])] = torch.tensor(it['ids'])
        att[i, :len(it['ids'])] = 1
        k = len(it['markers'])
        mpos[i, :k] = torch.tensor(it['markers'])
        mmask[i, :k] = True
    return {'input_ids': ids, 'attention_mask': att, 'marker_pos': mpos, 'marker_mask': mmask,
            'qtype': torch.tensor([it['qtype'] for it in items]),
            'label': torch.tensor([it.get('label', -1) for it in items]),
            'meta': [{k: it[k] for k in it if k not in ('ids', 'markers')} for it in items]}
