"""Inferencia compartilhada por evaluate, predict e serve: um preparador, um decodificador."""
from pathlib import Path
import contextlib
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loader  # noqa: E402
from common import ler_json  # noqa: E402
from decisions import probabilidades, decodificar, validar_politica, acao  # noqa: E402
from sequence import perguntas_da_tarefa, expandir, collate_items  # noqa: E402


def preparar(pasta, device='cpu', adapter=None, verificar=None, tarefa=None, calibracao=None):
    """Contexto de inferencia: modelo, tokenizer, perguntas, calibrador e politica.

    Pacote exportado traz task.json e calibration.json; checkpoint de origem
    recebe a tarefa por --tarefa e calibracao opcional.
    """
    import torch
    pasta = Path(pasta)
    from itsfine.device import escolher
    escolhido = escolher(device)
    carga = loader.carregar(pasta, device=escolhido, adapter=adapter, verificar=verificar)
    task = ler_json(tarefa) if tarefa else ler_json(pasta / 'task.json')
    calib = calibracao if calibracao is not None else (ler_json(pasta / 'calibration.json') if (pasta / 'calibration.json').is_file() else None)
    internas = perguntas_da_tarefa(task)
    decide = task.get('decide')
    politica = None
    if decide:
        if decide not in internas: raise ValueError('TASK_DECIDE_UNKNOWN:' + str(decide))
        politica = validar_politica(task.get('policy'), internas[decide])
    cfg = carga['cfg']
    amp = escolhido == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    return {'carga': carga, 'modelo': carga['modelo'], 'tok': carga['tok'], 'cfg': cfg, 'task': task, 'internas': internas,
            'calib': calib, 'politica': politica, 'decide': decide, 'device': escolhido, 'amp': amp,
            'max_len': int(cfg.get('max_len', 512)), 'head_max_len': int(cfg.get('head_max_len', 192))}


def registros(ctx, linhas, batch=16, com_rotulo=True):
    """Logits por (estado, pergunta). Cada linha vira uma lista de registros na ordem das perguntas."""
    import torch
    tok, modelo = ctx['tok'], ctx['modelo']
    saida, latencias = [], []
    for inicio in range(0, len(linhas), batch):
        parte = linhas[inicio:inicio + batch]
        grupos = [expandir(tok, l['state'], ctx['internas'], ctx['max_len'], ctx['head_max_len'],
                           l.get('answers', {}) if com_rotulo else None) for l in parte]
        lote = collate_items(grupos, tok.pad_token_id)
        t0 = time.perf_counter()
        with torch.no_grad(), (torch.autocast('cuda', dtype=torch.bfloat16) if ctx['amp'] else contextlib.nullcontext()):
            logits, _ = modelo(lote['input_ids'].to(ctx['device']), lote['attention_mask'].to(ctx['device']),
                               lote['marker_pos'].to(ctx['device']), lote['marker_mask'].to(ctx['device']), lote['qtype'].to(ctx['device']))
        latencias.append((time.perf_counter() - t0) * 1000 / max(1, len(parte)))
        logits = logits.float().cpu().numpy()
        linha_idx = 0
        for grupo in grupos:
            regs = []
            for item in grupo:
                regs.append({'nome': item['nome'], 'qtype': item['qtype'], 'k': item['k'],
                             'logits': [float(x) for x in logits[linha_idx, :item['k']]], 'label': item['label']})
                linha_idx += 1
            saida.append(regs)
    ctx.setdefault('latencias_ms', []).extend(latencias)
    return saida


def respostas(ctx, regs):
    """Registros de um estado -> respostas tipadas, probabilidades completas e acao da politica."""
    answers, action = {}, None
    for r in regs:
        q = ctx['internas'][r['nome']]
        p = probabilidades(r['logits'], r['qtype'], r['k'], ctx['calib'])
        resposta = decodificar(p, q)
        resposta['probs'] = [float(x) for x in p]
        answers[r['nome']] = resposta
        if ctx['politica'] and r['nome'] == ctx['decide']:
            nome_acao, valor = acao(p, ctx['politica'])
            action = {'saida': ctx['decide'], 'acao': nome_acao, 'p': valor}
    return {'answers': answers, 'action': action}


def latencia_p50(ctx):
    valores = sorted(ctx.get('latencias_ms', []))
    return round(float(valores[len(valores) // 2]), 1) if valores else None
