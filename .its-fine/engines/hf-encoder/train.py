"""Fine-tuning supervisionado de decisao tipada sobre qualquer encoder ou checkpoint de decisao.

LoRA nos Linear do encoder; type_emb, blocos extras e pontuador treinaveis por
inteiro (novos sobre um encoder comum, continuados num checkpoint de decisao);
act_head congelada. Loss: entropia cruzada sobre as opcoes de cada
pergunta, so nos itens rotulados, com peso por pergunta. Um estado com N perguntas
rotuladas vira N sequencias no lote. Com --probe roda 8 passos para medir VRAM.
Checkpoints por epoca com recibo; retomada so de checkpoint integro. Com --start-from
e treino novo a partir do adapter aprovado de outra rodada: otimizador e passos do zero.
"""
from pathlib import Path
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AQUI, gravar_json, ler_json, ler_jsonl, sha256_arquivo, sha256_json, modulos_lineares  # noqa: E402
from checkpoint import verify as verify_checkpoint, seal as seal_checkpoint  # noqa: E402
import decision_model  # noqa: E402
import loader  # noqa: E402
from decisions import metricas_saida  # noqa: E402
from sequence import perguntas_da_tarefa, expandir, collate_items  # noqa: E402
from itsfine.execution import require_supervisor  # noqa: E402
from itsfine.storage import digest  # noqa: E402
from itsfine.progress import Progress  # noqa: E402
from itsfine.continuation import adapter_hash, check_lora  # noqa: E402


def receita_efetiva(caminho):
    engine = ler_json(AQUI / 'engine.json')
    receita = ler_json(caminho)
    if receita.get('engine', 'hf-encoder') != 'hf-encoder': raise ValueError('UNKNOWN_ENGINE:' + str(receita.get('engine')))
    if receita.get('profile', engine['perfil']) != engine['perfil']: raise ValueError('UNKNOWN_PROFILE:' + str(receita.get('profile')))
    if not isinstance(receita.get('checkpoint'), dict) or not receita['checkpoint'].get('id'): raise ValueError('RECIPE_WITHOUT_CHECKPOINT')
    d = engine['defaults']
    return {'engine': 'hf-encoder', 'profile': engine['perfil'], 'checkpoint': receita['checkpoint'],
            'lora': {**d['lora'], **receita.get('lora', {})}, 'train': {**d['train'], **receita.get('train', {})},
            'trainable': receita.get('trainable', d['trainable']), 'loss_weights': receita.get('loss_weights', {}),
            'calibration': {**d['calibration'], **receita.get('calibration', {})},
            **({'start_from': receita['start_from']} if 'start_from' in receita else {})}


def grupos_rotulados(tok, linhas, internas, max_len, head_max_len):
    grupos = []
    for linha in linhas:
        itens = [it for it in expandir(tok, linha['state'], internas, max_len, head_max_len, linha.get('answers', {})) if it['label'] >= 0]
        if itens: grupos.append(itens)
    return grupos


def estado_rng():
    import torch
    from itsfine.device import mps_disponivel
    return {'python': random.getstate(), 'numpy': __import__('numpy').random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'mps': torch.mps.get_rng_state() if mps_disponivel(torch) else None}


def restaurar_rng(estado):
    import torch
    random.setstate(estado['python']); __import__('numpy').random.set_state(estado['numpy']); torch.set_rng_state(estado['torch'])
    if estado.get('cuda') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(estado['cuda'])
    if estado.get('mps') is not None: torch.mps.set_rng_state(estado['mps'])


def avaliar(modelo, grupos, device, amp, batch, pad_id):
    import torch
    modelo.eval()
    por_saida = {}
    with torch.no_grad():
        for i in range(0, len(grupos), batch):
            lote = collate_items(grupos[i:i + batch], pad_id)
            with (torch.autocast('cuda', dtype=torch.bfloat16) if amp else __import__('contextlib').nullcontext()):
                logits, _ = modelo(*(lote[k].to(device) for k in ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')))
            logits = logits.float().cpu().numpy()
            for j, meta in enumerate(lote['meta']):
                por_saida.setdefault(meta['nome'], []).append({'nome': meta['nome'], 'qtype': meta['qtype'], 'k': meta['k'],
                                                               'logits': logits[j, :meta['k']].tolist(), 'label': meta['label']})
    modelo.train()
    resultado = {}
    for nome, regs in por_saida.items():
        m = metricas_saida(regs)
        resultado[nome] = {k: m[k] for k in ('rotulados', 'log_loss', 'acuracia') if k in m}
    return resultado


def salvar_estado(pasta, modelo, opt, sched, passo, epoca, historico, identity):
    from safetensors.torch import save_file
    import torch
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / 'checkpoint-is-incomplete.txt').write_text('gravando', encoding='utf-8')
    modelo.encoder.save_pretrained(str(pasta))
    cabecas = {k: v.detach().cpu().contiguous() for k, v in modelo.state_dict().items()
               if k.split('.')[0] in ('type_emb', 'head', 'scorer')}
    save_file(cabecas, str(pasta / 'heads.safetensors'))
    torch.save(opt.state_dict(), pasta / 'optimizer.pt'); torch.save(sched.state_dict(), pasta / 'scheduler.pt')
    torch.save(estado_rng(), pasta / 'rng_state.pth')
    gravar_json(pasta / 'trainer_state.json', {'global_step': passo, 'epoch': epoca, 'log_history': historico})
    (pasta / 'checkpoint-is-incomplete.txt').unlink()
    seal_checkpoint(pasta, identity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receita', required=True, help='recipe do motor (engine, profile, checkpoint, lora, train, trainable)')
    parser.add_argument('--dados', required=True, help='pasta com train/validation .jsonl e task.json')
    parser.add_argument('--checkpoint', required=True, help='pasta do checkpoint de origem (conferido contra a lista suportada)')
    parser.add_argument('--saida', required=True)
    parser.add_argument('--probe', action='store_true', help='8 passos so para medir VRAM')
    parser.add_argument('--batch', type=int); parser.add_argument('--accum', type=int)
    parser.add_argument('--resume-from-checkpoint', help='checkpoint completo com recibo e hashes')
    parser.add_argument('--start-from', help='adapter aprovado de rodada anterior: pesos iniciais de um treino novo, nao retomada')
    args = parser.parse_args()
    require_supervisor()

    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    rec = receita_efetiva(args.receita)
    tr, lo = rec['train'], rec['lora']
    if args.batch: tr['batch'] = args.batch
    if args.accum: tr['grad_accum'] = args.accum
    dados, saida = Path(args.dados).resolve(), Path(args.saida).resolve()
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('TRAIN_OUTPUT_NOT_EMPTY')
    saida.mkdir(parents=True, exist_ok=True)
    task = ler_json(dados / 'task.json')
    internas = perguntas_da_tarefa(task)
    pesos = {nome: float(rec['loss_weights'].get(nome, 1.0)) for nome in internas}
    identity = {'recipe': digest(rec), 'train': sha256_arquivo(dados / 'train.jsonl'), 'validation': sha256_arquivo(dados / 'validation.jsonl'),
                'task': sha256_arquivo(dados / 'task.json'), 'checkpoint': loader.identidade(args.checkpoint)['package_id']}
    if args.resume_from_checkpoint:
        if args.probe: raise ValueError('PROBE_CANNOT_RESUME')
        verify_checkpoint(args.resume_from_checkpoint, identity)
    # receita com start_from: treino novo parte do adapter; a retomada ja traz esses pesos no checkpoint
    if bool(args.start_from) != bool(rec.get('start_from') and not args.resume_from_checkpoint): raise ValueError('START_FROM_RECIPE_MISMATCH')
    if args.start_from and adapter_hash(args.start_from) != rec['start_from']['adapter_hash']: raise ValueError('START_FROM_HASH_MISMATCH')
    random.seed(tr['seed']); np.random.seed(tr['seed']); torch.manual_seed(tr['seed'])

    carga = loader.carregar(args.checkpoint, device='cpu', verificar=rec['checkpoint'])
    modelo, tok, cfg = carga['modelo'], carga['tok'], carga['cfg']
    max_seq = tr['max_seq_length']
    if max_seq == 'auto':
        check = dados / 'check.json'
        max_seq = (ler_json(check).get('max_seq_length_sugerido') if check.exists() else None) or int(cfg.get('max_len', 512))
    max_seq = min(int(max_seq), int(cfg.get('max_len', 512)))
    head_max_len = int(cfg.get('head_max_len', 192))
    treino = grupos_rotulados(tok, ler_jsonl(dados / 'train.jsonl'), internas, max_seq, head_max_len)
    validacao = grupos_rotulados(tok, ler_jsonl(dados / 'validation.jsonl'), internas, max_seq, head_max_len)
    if not treino: raise ValueError('NO_LABELED_TRAINING_ITEMS')

    alvos = modulos_lineares(modelo.encoder) if lo['targets'] == 'auto' else lo['targets']
    if args.start_from: check_lora(args.start_from, lo, alvos)
    origem = args.resume_from_checkpoint or args.start_from
    if origem:
        modelo.encoder = PeftModel.from_pretrained(modelo.encoder, origem, is_trainable=True)
        from safetensors.torch import load_file
        modelo.load_state_dict(load_file(str(Path(origem) / 'heads.safetensors')), strict=False)
    else:
        modelo.encoder = get_peft_model(modelo.encoder, LoraConfig(r=lo['r'], lora_alpha=lo['alpha'], lora_dropout=lo['dropout'],
                                                                   bias='none', target_modules=alvos))
    treinaveis_cabecas = decision_model.configurar_treinaveis(modelo, rec['trainable'])
    from itsfine import device as acelerador
    device = acelerador.escolher('auto', treino=True)
    # autocast bf16 so na GPU NVIDIA; no mps o encoder treina em fp32, que cabe na memoria unificada
    amp = device == 'cuda' and acelerador.bf16(device)
    modelo = modelo.to(device=device, dtype=torch.float32).train()
    if tr.get('gradient_checkpointing'):
        base = modelo.encoder.get_base_model() if hasattr(modelo.encoder, 'get_base_model') else modelo.encoder
        if hasattr(base, 'gradient_checkpointing_enable'): base.gradient_checkpointing_enable()
    params = [p for p in modelo.parameters() if p.requires_grad]
    treinaveis = sum(p.numel() for p in params); total = sum(p.numel() for p in modelo.parameters())
    if any(n.split('.')[0] == 'act_head' and p.requires_grad for n, p in modelo.named_parameters()):
        raise ValueError('ACT_HEAD_MUST_STAY_FROZEN')

    batch, accum = int(tr['batch']), int(tr['grad_accum'])
    passos_por_epoca = max(1, math.ceil(len(treino) / batch / accum))
    epocas = 1 if args.probe else int(tr['epochs'])
    total_passos = 8 if args.probe else epocas * passos_por_epoca
    opt = torch.optim.AdamW(params, lr=float(tr['lr']), weight_decay=float(tr.get('weight_decay', 0.0)))
    agendador = get_cosine_schedule_with_warmup if tr['scheduler'] == 'cosine' else get_linear_schedule_with_warmup
    sched = agendador(opt, int(tr['warmup_steps']), total_passos)
    passo, epoca_inicial, historico = 0, 0, []
    if args.resume_from_checkpoint:
        estado = ler_json(Path(args.resume_from_checkpoint) / 'trainer_state.json')
        opt.load_state_dict(torch.load(Path(args.resume_from_checkpoint) / 'optimizer.pt', weights_only=False))
        sched.load_state_dict(torch.load(Path(args.resume_from_checkpoint) / 'scheduler.pt', weights_only=False))
        restaurar_rng(torch.load(Path(args.resume_from_checkpoint) / 'rng_state.pth', weights_only=False))
        passo, epoca_inicial, historico = estado['global_step'], int(estado['epoch']), estado.get('log_history', [])
    peso_por_nome = {nome: torch.tensor(w) for nome, w in pesos.items()}
    progresso = Progress(saida, epocas, total_passos, passo)

    memoria = acelerador.Memoria(device)
    inicio = time.time(); perdas = []; checkpoints = []
    for epoca in range(epoca_inicial, epocas):
        ordem = np.random.RandomState(int(tr['seed']) + epoca).permutation(len(treino))
        micro = 0; acumulada = 0.0
        for i in range(0, len(ordem), batch):
            grupos = [treino[j] for j in ordem[i:i + batch]]
            lote = collate_items(grupos, tok.pad_token_id)
            with (torch.autocast('cuda', dtype=torch.bfloat16) if amp else __import__('contextlib').nullcontext()):
                logits, _ = modelo(*(lote[k].to(device) for k in ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')))
            ce = torch.nn.functional.cross_entropy(logits.float(), lote['label'].to(device), reduction='none')
            w = torch.stack([peso_por_nome[m['nome']] for m in lote['meta']]).to(device)
            loss = (ce * w).sum() / w.sum()
            (loss / accum).backward()
            acumulada += float(loss.detach()); micro += 1
            if micro % accum == 0 or i + batch >= len(ordem):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                passo += 1; memoria.amostrar(); perdas.append(acumulada / (micro if micro % accum else accum)); acumulada = 0.0
                if passo % 5 == 0 or args.probe: historico.append({'step': passo, 'loss': round(perdas[-1], 4)})
                progresso.update(passo, epoca + min(1.0, (i + batch) / len(ordem)), perdas[-1])
                if args.probe and passo >= 8: break
        if args.probe: break
        metricas = avaliar(modelo, validacao, device, amp, batch, tok.pad_token_id) if validacao else {}
        historico.append({'step': passo, 'epoch': epoca + 1, 'eval': metricas})
        pasta = saida / ('checkpoint-' + str(passo))
        salvar_estado(pasta, modelo, opt, sched, passo, epoca + 1, historico, identity)
        checkpoints.append(pasta)
        while len(checkpoints) > int(tr['save_total_limit']):
            shutil.rmtree(checkpoints.pop(0), ignore_errors=True)
    duracao = time.time() - inicio
    progresso.update(passo, epocas, perdas[-1] if perdas else None, force=True)
    relatorio = {'schema_version': 1, 'modo': 'probe' if args.probe else 'treino', 'engine': 'hf-encoder',
                 'checkpoint': {'id': rec['checkpoint']['id'], 'revision': rec['checkpoint'].get('revision'), 'hash': identity['checkpoint']},
                 'gpu': acelerador.nome_do_hardware(device) if device != 'cpu' else 'cpu', 'dispositivo': device,
                 'parametros': {'treinaveis': treinaveis, 'total': total, 'fracao': round(treinaveis / total, 5), 'alvos_lora': alvos,
                                'cabecas': treinaveis_cabecas, 'cabecas_novas': carga['desc']['cabecas_novas'], 'act_head_congelada': True},
                 'receita_efetiva': {**rec, 'train': {**tr, 'max_seq_length': max_seq}},
                 'exemplos_treino': {'estados': len(treino), 'sequencias': sum(len(g) for g in treino)},
                 'passos': passo, 'loss_treino': round(sum(perdas) / max(1, len(perdas)), 4), 'duracao_s': round(duracao, 1),
                 'segundos_por_passo': round(duracao / max(1, len(perdas)), 1), 'historico': historico}
    relatorio['vram'] = memoria.relatorio() if device != 'cpu' else {'pico_alocado_gb': 0.0, 'pico_reservado_gb': 0.0, 'total_gb': 0.0, 'coube': True, 'nota': 'cpu'}
    if not args.probe:
        adapter = saida / 'adapter'
        modelo.encoder.save_pretrained(str(adapter))
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in modelo.state_dict().items() if k.split('.')[0] in ('type_emb', 'head', 'scorer')},
                  str(adapter / 'heads.safetensors'))
        shutil.copy2(dados / 'task.json', adapter / 'task.json')
        gravar_json(adapter / 'itsfine-adapter.json', {'schema_version': 1, 'engine': 'hf-encoder', 'identity': identity,
                                                       'checkpoint': rec['checkpoint'], 'trainable': rec['trainable'], 'passos': passo})
        relatorio['metricas_validacao'] = avaliar(modelo, validacao, device, amp, batch, tok.pad_token_id) if validacao else {}
        relatorio['adapter'] = 'adapter'
        relatorio['adapter_hash'] = sha256_json({'adapter': sha256_arquivo(adapter / 'adapter_model.safetensors'), 'heads': sha256_arquivo(adapter / 'heads.safetensors')})
    gravar_json(saida / ('probe.json' if args.probe else 'treino.json'), relatorio)
    print(json.dumps({k: relatorio[k] for k in ('modo', 'vram', 'passos', 'loss_treino', 'duracao_s', 'segundos_por_passo', 'exemplos_treino')}
                     | ({'metricas_validacao': relatorio.get('metricas_validacao')} if not args.probe else {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    raise SystemExit(main())
