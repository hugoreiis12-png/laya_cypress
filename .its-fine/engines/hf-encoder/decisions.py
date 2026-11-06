"""Probabilidades, respostas tipadas, politica de decisao e metricas de classificacao.

Registro de avaliacao por item: {'nome','qtype','k','logits','label'} com label -1
quando a saida nao tem rotulo. As metricas usam so itens rotulados.
"""
import math

import numpy as np

from sequence import QTYPES, QTYPE_NAMES, temp_bucket, opcoes

TEMP_MIN, TEMP_MAX = 0.5, 5.0
INTERVALOS_ECE = 15


def clamp_temperature(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float('inf'), float('-inf')): return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def temperatura(calib, qtype, k):
    """Temperatura do bucket (tipo, numero de opcoes); cai para a do tipo; sem calibracao e 1."""
    if not calib: return 1.0
    por_opcoes = calib.get('temperature_by_options', {})
    chave = temp_bucket(qtype, k)
    if chave in por_opcoes: return clamp_temperature(por_opcoes[chave])
    return clamp_temperature(calib.get('temperature', [1.0, 1.0, 1.0])[int(qtype)])


def probabilidades(logits, qtype, k, calib=None):
    z = np.asarray(logits, dtype=np.float64)[:k] / temperatura(calib, qtype, k)
    p = np.exp(z - z.max())
    return p / p.sum()


def confidence_from_probs(p, k):
    if k < 2: return 1.0
    p = np.asarray(p)[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def decodificar(p, q):
    """Resposta no formato do Laya/Jev para uma pergunta interna q."""
    k = len(p)
    conf = round(confidence_from_probs(p, k), 4)
    if q['t'] == 'choice':
        keys = list(q['crit'].keys())
        return {'type': 'choice', 'choice': keys[int(p.argmax())],
                'probabilities': {kk: round(float(v), 4) for kk, v in zip(keys, p)}, 'confidence': conf}
    if q['t'] == 'score':
        return {'type': 'score', 'score': round(float((np.arange(k) * p).sum()), 4),
                'legend': {str(i): c for i, c in enumerate(q['crit'])},
                'probabilities': {str(i): round(float(v), 4) for i, v in enumerate(p)}, 'confidence': conf}
    return {'type': 'noul', 'noul': round(float(p[1]), 4), 'confidence': round(max(float(p[1]), 1.0 - float(p[1])), 4)}


def indice_positivo(q, positive):
    """Indice da opcao positiva da politica: 'true' em noul, o nome da opcao em choice."""
    nomes = opcoes(q)
    if positive is None:
        positive = 'true' if q['t'] == 'noul' else nomes[-1]
    positive = str(positive).lower() if q['t'] == 'noul' else str(positive)
    if positive not in nomes: raise ValueError('POLICY_POSITIVE_NOT_AN_OPTION:%s' % positive)
    return nomes.index(positive)


def validar_politica(policy, q):
    if policy is None: return None
    for campo in ('high', 'low'):
        if not isinstance(policy.get(campo), (int, float)) or isinstance(policy.get(campo), bool):
            raise ValueError('POLICY_THRESHOLD_INVALID:' + campo)
    if not 0 <= policy['low'] <= policy['high'] <= 1: raise ValueError('POLICY_THRESHOLDS_ORDER')
    if q['t'] == 'score': raise ValueError('POLICY_ON_SCORE_UNSUPPORTED')
    return {'high': float(policy['high']), 'low': float(policy['low']),
            'middle_action': str(policy.get('middle_action', 'revisar')), 'positive_index': indice_positivo(q, policy.get('positive'))}


def acao(p, politica):
    """Aplica os limiares a probabilidade da opcao positiva, antes de qualquer arredondamento."""
    valor = float(p[politica['positive_index']])
    if valor >= politica['high']: return 'positiva', valor
    if valor <= politica['low']: return 'negativa', valor
    return politica['middle_action'], valor


def ece_e_curva(conf, correto, intervalos=INTERVALOS_ECE):
    conf = np.asarray(conf, dtype=np.float64); correto = np.asarray(correto, dtype=np.float64)
    if len(conf) == 0: return None, []
    bordas = np.linspace(0, 1, intervalos + 1)
    total, curva = 0.0, []
    for i, (lo, hi) in enumerate(zip(bordas[:-1], bordas[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            gap = abs(conf[sel].mean() - correto[sel].mean())
            total += sel.mean() * gap
            curva.append({'de': round(float(lo), 4), 'ate': round(float(hi), 4), 'n': int(sel.sum()),
                          'confianca_media': round(float(conf[sel].mean()), 4), 'acuracia': round(float(correto[sel].mean()), 4)})
    return round(float(total), 4), curva


def probs_do_registro(r, calib=None):
    """Registro com logits recebe o calibrador; registro vindo do servidor ja traz probs."""
    if 'probs' in r: return np.asarray(r['probs'], dtype=np.float64)
    return probabilidades(r['logits'], r['qtype'], r['k'], calib)


def metricas_saida(registros, calib=None):
    """Metricas de uma saida sobre itens rotulados. Devolve None sem rotulo."""
    rotulados = [r for r in registros if r.get('label', -1) is not None and r['label'] >= 0]
    n = len(rotulados)
    if not n: return {'n': len(registros), 'rotulados': 0}
    ps = [probs_do_registro(r, calib) for r in rotulados]
    y = np.array([r['label'] for r in rotulados])
    logp = np.array([math.log(max(p[t], 1e-12)) for p, t in zip(ps, y)])
    brier = float(np.mean([np.sum((p - np.eye(len(p))[t]) ** 2) for p, t in zip(ps, y)]))
    previsto = np.array([int(p.argmax()) for p in ps])
    conf = np.array([float(p.max()) for p in ps])
    ece, curva = ece_e_curva(conf, previsto == y)
    resultado = {'n': len(registros), 'rotulados': n, 'log_loss': round(float(-logp.mean()), 4), 'brier': round(brier, 4),
                 'acuracia': round(float((previsto == y).mean()), 4), 'ece': ece, 'intervalos_ece': INTERVALOS_ECE, 'curva': curva,
                 'tipo': QTYPE_NAMES[rotulados[0]['qtype']]}
    if rotulados[0]['qtype'] == QTYPES['score']:
        esperado = np.array([float((np.arange(len(p)) * p).sum()) for p in ps])
        resultado['mae'] = round(float(np.abs(esperado - y).mean()), 4)
    return resultado


def matriz_operacional(registros, politica, calib=None):
    """Decisoes automaticas no ponto de operacao congelado, mais a leitura 'revisar como negativo'."""
    rotulados = [r for r in registros if r.get('label', -1) is not None and r['label'] >= 0]
    tp = fp = tn = fn = 0; encaminhados = 0; encaminhados_positivos = 0
    for r in rotulados:
        p = probs_do_registro(r, calib)
        decisao, _ = acao(p, politica)
        positivo = r['label'] == politica['positive_index']
        if decisao == 'positiva':
            tp += positivo; fp += (not positivo)
        elif decisao == 'negativa':
            fn += positivo; tn += (not positivo)
        else:
            encaminhados += 1; encaminhados_positivos += positivo
    n = len(rotulados); positivos = tp + fn + encaminhados_positivos
    auto = tp + fp + tn + fn
    fn2 = fn + encaminhados_positivos; tn2 = tn + (encaminhados - encaminhados_positivos)
    def taxa(a, b): return round(a / b, 4) if b else None
    return {'n': n, 'positivos': int(positivos),
            'automatica': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn), 'cobertura_automatica': taxa(auto, n),
                           'encaminhados': int(encaminhados), 'fn_liberados': int(fn), 'taxa_fn_liberados': taxa(fn, positivos),
                           'precision': taxa(tp, tp + fp), 'recall': taxa(tp, tp + fn)},
            'revisar_como_negativo': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn2), 'fn': int(fn2),
                                      'acuracia': taxa(tp + tn2, n), 'recall': taxa(tp, tp + fn2), 'precision': taxa(tp, tp + fp),
                                      'f1': taxa(2 * tp, 2 * tp + fp + fn2)}}


def verificar_limites(resumo, limites):
    """Limites obrigatorios do protocolo -> lista com valor, limite e ok. Nome desconhecido e erro."""
    op = resumo.get('operacional') or {}
    auto = op.get('automatica', {}); rev = op.get('revisar_como_negativo', {}); m = resumo.get('metricas_decisao') or {}
    valores = {'max_taxa_fn_liberados': (auto.get('taxa_fn_liberados'), '<='), 'min_cobertura_automatica': (auto.get('cobertura_automatica'), '>='),
               'max_ece': (m.get('ece'), '<='), 'min_acuracia': (m.get('acuracia'), '>='), 'max_log_loss': (m.get('log_loss'), '<='),
               'min_recall_revisar_negativo': (rev.get('recall'), '>='), 'max_fp': (auto.get('fp'), '<='), 'max_fn_liberados': (auto.get('fn_liberados'), '<=')}
    saida = []
    for nome, limite in (limites or {}).items():
        if nome not in valores: raise ValueError('LIMIT_UNKNOWN:' + nome)
        valor, comp = valores[nome]
        ok = valor is not None and ((valor <= limite) if comp == '<=' else (valor >= limite))
        saida.append({'nome': nome, 'valor': valor, 'limite': limite, 'ok': bool(ok)})
    return saida
