"""Treino LoRA de modelo de voz a partir de uma receita. Generico: a arquitetura entra pelo adaptador da familia.

Le a receita e os splits (com os clipes em audio/) e escreve so no diretorio de saida.
Antes de treinar mede a perda do modelo base na validacao: se algum componente nao fica
abaixo de uma fracao do acaso (ln do vocabulario), o layout de sequencia esta errado e o
treino e recusado (SANITY_GATE_FAILED), com os numeros em sanidade.json. Cada epoca termina
com a perda de validacao e um checkpoint completo; o adapter final e o da melhor epoca, nao o
ultimo. Com --probe roda poucos passos para medir memoria real. Com --start-from e treino
novo a partir do adapter aprovado de outra rodada: otimizador e passos do zero.
"""
from pathlib import Path
import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import adaptador, engine_info, familia_do_modelo, gravar_json, ler_json, ler_jsonl, referencia, sha256_arquivo
from checkpoint import verify as verify_checkpoint, seal as seal_checkpoint
import lora as L
from itsfine.execution import require_supervisor
from itsfine.storage import digest
from itsfine.progress import Progress
from itsfine.continuation import adapter_hash, check_lora
from itsfine import device as acelerador

PASSOS_SONDAGEM = 8


def receita_efetiva(caminho):
    """Defaults do motor sobrescritos pela receita da rodada."""
    engine = engine_info()
    receita = ler_json(caminho)
    return {'model_id': receita['model_id'], 'revision': receita.get('revision'),
            'familia': familia_do_modelo(receita['model_id'], receita.get('familia')),
            'idioma': receita.get('idioma'), 'voz': receita.get('voz', 'minha_voz').lower(), 'method': 'lora',
            'lora': {**engine['defaults']['lora'], **receita.get('lora', {})},
            'train': {**engine['defaults']['train'], **receita.get('train', {})},
            'sanidade': {**engine['defaults']['sanidade'], **receita.get('sanidade', {})},
            **({'start_from': receita['start_from']} if 'start_from' in receita else {})}


def linhas(dados, nome, maximo):
    """Linhas do split com clipe ate `maximo` segundos; as mais longas ficam de fora e sao contadas."""
    todas = ler_jsonl(dados / nome)
    from common import duracao
    ficam = [l for l in todas if (l.get('duracao') or duracao(dados / l['audio'])) <= maximo]
    return ficam, len(todas) - len(ficam)


def precisao(dispositivo):
    """dtype de carga e contexto de autocast: bf16 direto; GPU so com fp16 (T4) carrega fp32 e usa autocast + scaler."""
    import torch
    dtype = acelerador.dtype(dispositivo)
    if dtype == torch.float16:
        return torch.float32, lambda: torch.autocast('cuda', dtype=torch.float16), torch.amp.GradScaler('cuda')
    return dtype, contextlib.nullcontext, None


def media(familia, exemplos, contexto, lote):
    """Perda media (total e por componente) sem gradiente."""
    import torch
    soma, partes = 0.0, {}
    with torch.no_grad():
        for i in range(0, len(exemplos), lote):
            with contexto():
                total, comp = familia.perda(exemplos[i:i + lote])
            n = len(exemplos[i:i + lote])
            soma += float(total) * n
            for k, v in comp.items(): partes[k] = partes.get(k, 0.0) + v * n
    n = max(1, len(exemplos))
    return {'total': round(soma / n, 4), **{k: round(v / n, 4) for k, v in partes.items()}}


def sanidade(base, acaso, fracao):
    componentes = {k: {'base': base[k], 'acaso': round(acaso[k], 4), 'limite': round(fracao * acaso[k], 4),
                       'ok': base[k] < fracao * acaso[k]} for k in acaso}
    return {'fracao_do_acaso': fracao, 'componentes': componentes, 'ok': all(c['ok'] for c in componentes.values())}


def salvar_checkpoint(pasta, modulo, cfg, familia, meta, opt, sched, scaler, passo, epoca, historico, melhor, staging, identity):
    import torch
    L.salvar(pasta, modulo, cfg, familia.extras(), meta)
    torch.save(opt.state_dict(), pasta / 'optimizer.pt')
    torch.save({'scheduler': sched.state_dict(), 'scaler': scaler.state_dict() if scaler else None}, pasta / 'scheduler.pt')
    torch.save({'python': random.getstate(), 'numpy': __import__('numpy').random.get_state(), 'torch': torch.get_rng_state(),
                'acelerador': acelerador.estado_rng(familia.dispositivo)}, pasta / 'rng_state.pth')
    gravar_json(pasta / 'trainer_state.json', {'global_step': passo, 'epoch': epoca, 'log_history': historico, 'melhor': melhor})
    shutil.copytree(staging, pasta / 'melhor')
    seal_checkpoint(pasta, identity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receita', required=True, help='recipe da rodada (model_id, revision, familia, idioma, voz, lora, train)')
    parser.add_argument('--dados', required=True, help='pasta preparada: train/validation .jsonl e audio/')
    parser.add_argument('--saida', required=True)
    parser.add_argument('--probe', action='store_true', help='poucos passos so para medir memoria e a porta de sanidade')
    parser.add_argument('--batch', type=int, help='sobrescreve a receita')
    parser.add_argument('--accum', type=int, help='sobrescreve a receita')
    parser.add_argument('--resume-from-checkpoint', help='checkpoint completo com recibo e hashes')
    parser.add_argument('--start-from', help='adapter aprovado de rodada anterior: pesos iniciais de um treino novo, nao retomada')
    args = parser.parse_args()
    require_supervisor()

    import numpy as np
    import torch
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    rec = receita_efetiva(args.receita)
    tr, lo = rec['train'], rec['lora']
    if args.batch: tr['batch'] = args.batch
    if args.accum: tr['grad_accum'] = args.accum
    dados, saida = Path(args.dados).resolve(), Path(args.saida).resolve()
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('TRAIN_OUTPUT_NOT_EMPTY')
    saida.mkdir(parents=True, exist_ok=True)
    identity = {'recipe': digest(rec), 'train': sha256_arquivo(dados / 'train.jsonl'), 'validation': sha256_arquivo(dados / 'validation.jsonl')}
    if args.resume_from_checkpoint:
        if args.probe: raise ValueError('PROBE_CANNOT_RESUME')
        verify_checkpoint(args.resume_from_checkpoint, identity)
    if bool(args.start_from) != bool(rec.get('start_from') and not args.resume_from_checkpoint): raise ValueError('START_FROM_RECIPE_MISMATCH')
    if args.start_from and adapter_hash(args.start_from) != rec['start_from']['adapter_hash']: raise ValueError('START_FROM_HASH_MISMATCH')
    random.seed(tr['seed']); np.random.seed(tr['seed']); torch.manual_seed(tr['seed'])

    dispositivo = acelerador.escolher('auto', treino=True)
    dtype, contexto, scaler = precisao(dispositivo)
    Familia = adaptador(rec['familia'])
    familia = Familia(rec['model_id'], dispositivo, dtype, revision=rec['revision'], idioma=rec['idioma'])
    familia.modelo.eval()  # sem dropout: LoRA sem dropout e codec congelado; o gradiente passa igual

    treino_linhas, longos_treino = linhas(dados, 'train.jsonl', float(tr['max_audio_seconds']))
    valid_linhas, longos_valid = linhas(dados, 'validation.jsonl', float(tr['max_audio_seconds']))
    if not treino_linhas or not valid_linhas: raise ValueError('EMPTY_SPLIT_AFTER_DURATION_FILTER')
    ref = referencia(treino_linhas, dados)
    familia.definir_voz(dados / ref['audio'])
    treino = [familia.preparar(l['text'], dados / l['audio']) for l in treino_linhas]
    validacao = [familia.preparar(l['text'], dados / l['audio']) for l in valid_linhas]
    batch, accum = int(tr['batch']), int(tr['grad_accum'])

    base = media(familia, validacao, contexto, batch)
    gate = sanidade(base, familia.acaso(), float(rec['sanidade']['fracao_do_acaso']))
    gravar_json(saida / 'sanidade.json', gate)
    if not gate['ok']:
        raise ValueError('SANITY_GATE_FAILED: perda do modelo base nao fica abaixo do acaso; layout de sequencia da familia errado')

    modulo, alvos, excluir = familia.lora()
    cfg = L.configuracao(lo, alvos, excluir)
    params = L.injetar(modulo, cfg)
    if args.start_from:
        check_lora(args.start_from, lo, alvos)
        familia.carregar_extras(L.carregar(args.start_from, modulo))
    treinaveis, total = sum(p.numel() for p in params), sum(p.numel() for p in familia.modelo.parameters())

    passos_por_epoca = max(1, math.ceil(len(treino) / batch / accum))
    epocas = 1 if args.probe else int(tr['epochs'])
    total_passos = PASSOS_SONDAGEM if args.probe else epocas * passos_por_epoca
    opt = torch.optim.AdamW(params, lr=float(tr['lr']), weight_decay=0.01)
    agendador = get_cosine_schedule_with_warmup if tr['scheduler'] == 'cosine' else get_linear_schedule_with_warmup
    sched = agendador(opt, int(tr['warmup_steps']), total_passos)
    meta = {'schema_version': 1, 'engine': 'hf-tts', 'familia': rec['familia'], 'model_id': rec['model_id'], 'revision': rec['revision'],
            'idioma': rec['idioma'], 'voz': rec['voz'], 'referencia_audio': ref['audio']}
    passo, epoca_inicial, historico = 0, 0, []
    melhor = {'perda': math.inf, 'epoca': None, 'passo': None}
    staging = saida / 'melhor'
    if args.resume_from_checkpoint:
        origem = Path(args.resume_from_checkpoint)
        familia.carregar_extras(L.carregar(origem, modulo))
        opt.load_state_dict(torch.load(origem / 'optimizer.pt', weights_only=False))
        agenda = torch.load(origem / 'scheduler.pt', weights_only=False)
        sched.load_state_dict(agenda['scheduler'])
        if scaler and agenda.get('scaler'): scaler.load_state_dict(agenda['scaler'])
        rng = torch.load(origem / 'rng_state.pth', weights_only=False)
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
        acelerador.restaurar_rng(dispositivo, rng.get('acelerador'))
        estado = ler_json(origem / 'trainer_state.json')
        passo, epoca_inicial, historico, melhor = estado['global_step'], int(estado['epoch']), estado['log_history'], estado['melhor']
        shutil.copytree(origem / 'melhor', staging)
    progresso = Progress(saida, epocas, total_passos, passo)
    memoria = acelerador.Memoria(dispositivo)

    inicio = time.time(); perdas = []; checkpoints = []
    for epoca in range(epoca_inicial, epocas):
        ordem = np.random.RandomState(int(tr['seed']) + epoca).permutation(len(treino))
        micro, acumulada = 0, 0.0
        for i in range(0, len(ordem), batch):
            with contexto():
                loss, _ = familia.perda([treino[j] for j in ordem[i:i + batch]])
            (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            acumulada += float(loss.detach()); micro += 1
            if micro % accum == 0 or i + batch >= len(ordem):
                if scaler: scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
                sched.step(); opt.zero_grad(set_to_none=True)
                passo += 1; memoria.amostrar()
                perdas.append(acumulada / (micro % accum or accum)); acumulada = 0.0
                if passo % 5 == 0 or args.probe: historico.append({'step': passo, 'loss': round(perdas[-1], 4)})
                progresso.update(passo, epoca + min(1.0, (i + batch) / len(ordem)), perdas[-1])
                if args.probe and passo >= PASSOS_SONDAGEM: break
        if args.probe: break
        valor = media(familia, validacao, contexto, batch)
        historico.append({'step': passo, 'epoch': epoca + 1, 'validacao': valor})
        if not math.isfinite(valor['total']): raise ValueError('VALIDATION_LOSS_NOT_FINITE')
        if valor['total'] < melhor['perda']:
            melhor = {'perda': valor['total'], 'epoca': epoca + 1, 'passo': passo, 'validacao': valor}
            if staging.exists(): shutil.rmtree(staging)
            L.salvar(staging, modulo, cfg, familia.extras(), {**meta, 'epoca': epoca + 1})
        pasta = saida / ('checkpoint-' + str(passo))
        salvar_checkpoint(pasta, modulo, cfg, familia, meta, opt, sched, scaler, passo, epoca + 1, historico, melhor, staging, identity)
        checkpoints.append(pasta)
        while len(checkpoints) > int(tr['save_total_limit']):
            shutil.rmtree(checkpoints.pop(0), ignore_errors=True)
    duracao = time.time() - inicio
    progresso.update(passo, epocas, perdas[-1] if perdas else None, force=True)

    relatorio = {
        'schema_version': 1, 'modo': 'probe' if args.probe else 'treino', 'engine': 'hf-tts', 'familia': rec['familia'],
        'modelo': rec['model_id'], 'revision': rec['revision'], 'metodo': 'lora',
        'gpu': acelerador.nome_do_hardware(dispositivo), 'dispositivo': dispositivo, 'vram': memoria.relatorio(),
        'sanidade': gate,
        'parametros': {'treinaveis': treinaveis, 'total': total, 'fracao': round(treinaveis / total, 5), 'alvos_lora': sorted(alvos)},
        'receita_efetiva': {k: rec[k] for k in ('lora', 'train', 'idioma', 'voz')},
        'exemplos_treino': len(treino), 'exemplos_validacao': len(validacao),
        'clipes_longos_fora': {'treino': longos_treino, 'validacao': longos_valid},
        'passos': passo, 'loss_treino': round(sum(perdas) / max(1, len(perdas)), 4) if perdas else None,
        'duracao_s': round(duracao, 1), 'segundos_por_passo': round(duracao / max(1, len(perdas)), 2),
    }
    if not args.probe:
        if not staging.is_dir(): raise ValueError('NO_EPOCH_COMPLETED')
        shutil.copytree(staging, saida / 'adapter')
        relatorio.update({'adapter': 'adapter', 'adapter_hash': adapter_hash(saida / 'adapter'), 'melhor_epoca': melhor['epoca'],
                          'metricas_validacao': {'base': base, 'melhor': melhor['validacao']}, 'historico': historico})
    else:
        relatorio['historico'] = historico
    gravar_json(saida / ('probe.json' if args.probe else 'treino.json'), relatorio)
    print(json.dumps({k: relatorio.get(k) for k in ('modo', 'vram', 'passos', 'loss_treino', 'duracao_s', 'segundos_por_passo', 'melhor_epoca')}
                     | {'sanidade_ok': gate['ok']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    raise SystemExit(main())
