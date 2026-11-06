"""Treino LoRA (bf16) ou QLoRA (4 bits) a partir de uma receita.

Le a receita, os splits e escreve so no diretorio de saida. Com --probe roda
poucos passos para medir a VRAM real: e a sondagem que decide se cabe, nao a
estimativa. Usa Trainer direto porque muitos chat templates nao trazem o
marcador de geracao que a trl exige para mascarar; a mascara e construida
pela diferenca de prefixo, conferida antes em check.py. Com --start-from e treino
novo a partir do adapter aprovado de outra rodada: otimizador e passos do zero.
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_json, modulos_lineares, prefixo_e_alvo, sha256_arquivo, carga_modelo
from checkpoint import verify as verify_checkpoint, seal as seal_checkpoint
from itsfine.execution import require_supervisor
from itsfine.storage import digest
from itsfine.progress import Progress
from itsfine.continuation import adapter_hash, check_lora

IGNORAR = -100
AQUI = Path(__file__).resolve().parent


def receita_efetiva(caminho):
    """Defaults da engine sobrescritos pela receita da rodada."""
    engine = ler_json(AQUI / 'engine.json')
    receita = ler_json(caminho)
    lora = {**engine['defaults']['lora'], **receita.get('lora', {})}
    train = {**engine['defaults']['train'], **receita.get('train', {})}
    return {'model_id': receita['model_id'], 'revision': receita.get('revision'),
            'method': receita.get('method', engine['defaults']['method']), 'lora': lora, 'train': train,
            **({'start_from': receita['start_from']} if 'start_from' in receita else {})}


def pretokenizar(conjunto, tok, max_seq, modo):
    def processar(exemplo):
        if modo == 'messages':
            completo, corte = prefixo_e_alvo(tok, exemplo['messages'])
            completo = completo[:max_seq]
            rotulos = [IGNORAR] * min(corte, len(completo)) + completo[corte:]
            if not any(x!=IGNORAR for x in rotulos): raise ValueError('TRUNCATION_REMOVED_TARGET')
        else:
            completo = tok(exemplo['text'])['input_ids'][:max_seq]
            if not completo: raise ValueError('EMPTY_TRAINING_EXAMPLE')
            if tok.eos_token_id is not None and completo[-1] != tok.eos_token_id and len(completo) < max_seq:
                completo = completo + [tok.eos_token_id]
            rotulos = list(completo)
        return {'input_ids': completo, 'labels': rotulos}
    return conjunto.map(processar, remove_columns=conjunto.column_names)


class Colator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, lote):
        import torch
        largura = max(len(x['input_ids']) for x in lote)
        ids, rotulos, atencao = [], [], []
        for ex in lote:
            falta = largura - len(ex['input_ids'])
            ids.append(ex['input_ids'] + [self.pad_id] * falta)
            rotulos.append(ex['labels'] + [IGNORAR] * falta)
            atencao.append([1] * len(ex['input_ids']) + [0] * falta)
        return {'input_ids': torch.tensor(ids), 'labels': torch.tensor(rotulos), 'attention_mask': torch.tensor(atencao)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receita', required=True, help='recipe da rodada (model_id, revision, method, lora, train)')
    parser.add_argument('--dados', required=True, help='pasta com train/validation .jsonl')
    parser.add_argument('--saida', required=True)
    parser.add_argument('--probe', action='store_true', help='8 passos so para medir VRAM')
    parser.add_argument('--batch', type=int, help='sobrescreve a receita')
    parser.add_argument('--accum', type=int, help='sobrescreve a receita')
    parser.add_argument('--resume-from-checkpoint', help='checkpoint completo com recibo e hashes')
    parser.add_argument('--start-from', help='adapter aprovado de rodada anterior: pesos iniciais de um treino novo, nao retomada')
    args = parser.parse_args()
    require_supervisor()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

    rec = receita_efetiva(args.receita)
    tr, lo = rec['train'], rec['lora']
    if args.batch:
        tr['batch'] = args.batch
    if args.accum:
        tr['grad_accum'] = args.accum
    dados, saida = Path(args.dados).resolve(), Path(args.saida).resolve()
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('TRAIN_OUTPUT_NOT_EMPTY')
    saida.mkdir(parents=True, exist_ok=True)
    identity={'recipe':digest(rec),'train':sha256_arquivo(dados/'train.jsonl'),
              'validation':sha256_arquivo(dados/'validation.jsonl')}
    if args.resume_from_checkpoint:
        if args.probe: raise ValueError('PROBE_CANNOT_RESUME')
        verify_checkpoint(args.resume_from_checkpoint,identity)
    # receita com start_from: treino novo parte do adapter; a retomada ja traz esses pesos no checkpoint
    if bool(args.start_from) != bool(rec.get('start_from') and not args.resume_from_checkpoint): raise ValueError('START_FROM_RECIPE_MISMATCH')
    if args.start_from and adapter_hash(args.start_from) != rec['start_from']['adapter_hash']: raise ValueError('START_FROM_HASH_MISMATCH')
    torch.manual_seed(tr['seed'])

    tok = AutoTokenizer.from_pretrained(rec['model_id'], revision=rec['revision'])
    bruto = load_dataset('json', data_files={'train': str(dados / 'train.jsonl'), 'validation': str(dados / 'validation.jsonl')})
    modo = 'messages' if 'messages' in bruto['train'].column_names else 'text'
    max_seq = tr['max_seq_length']
    if max_seq == 'auto':
        max_seq = None
        for check in (dados / 'check.json', dados.parent / 'check.json'):
            if check.exists():
                max_seq = ler_json(check).get('max_seq_length_sugerido')
                break
        max_seq = max_seq or 1024
    treino = pretokenizar(bruto['train'], tok, max_seq, modo)
    validacao = pretokenizar(bruto['validation'], tok, max_seq, modo)

    from itsfine import device as acelerador
    dispositivo = acelerador.escolher('auto', treino=True)
    carga = {'revision':rec['revision'],**carga_modelo(dispositivo,rec['method']=='qlora',treino=True)}
    bf16, fp16 = carga['dtype']==torch.bfloat16, carga['dtype']==torch.float16
    modelo = AutoModelForCausalLM.from_pretrained(rec['model_id'], **carga)
    modelo.config.use_cache = False
    if rec['method'] == 'qlora':
        from peft import prepare_model_for_kbit_training
        modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=tr['gradient_checkpointing'])
    alvos = modulos_lineares(modelo) if lo['targets'] == 'auto' else lo['targets']
    if args.start_from:
        from peft import PeftModel
        check_lora(args.start_from, lo, alvos)
        modelo = PeftModel.from_pretrained(modelo, args.start_from, is_trainable=True)
    else:
        modelo = get_peft_model(modelo, LoraConfig(r=lo['r'], lora_alpha=lo['alpha'], lora_dropout=lo['dropout'],
                                                   bias='none', task_type='CAUSAL_LM', target_modules=alvos))
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())

    config = TrainingArguments(
        output_dir=str(saida), seed=tr['seed'], bf16=bf16, fp16=fp16, dataloader_pin_memory=dispositivo=='cuda',
        per_device_train_batch_size=tr['batch'], gradient_accumulation_steps=tr['grad_accum'],
        per_device_eval_batch_size=tr['batch'],
        num_train_epochs=(1.0 if args.probe else float(tr['epochs'])), max_steps=(8 if args.probe else -1),
        learning_rate=float(tr['lr']), lr_scheduler_type=tr['scheduler'], warmup_steps=int(tr['warmup_steps']),
        logging_steps=5, eval_strategy=('no' if args.probe else tr['eval_strategy']),
        save_strategy=('no' if args.probe else tr['save_strategy']), save_total_limit=int(tr['save_total_limit']),
        gradient_checkpointing=bool(tr['gradient_checkpointing']), report_to=[], remove_unused_columns=False)
    class ReceiptCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            seal_checkpoint(Path(args.output_dir)/('checkpoint-'+str(state.global_step)),identity)
        # progresso so com numeros, lido pelo status e pela barra no chat
        def on_train_begin(self, args, state, control, **kwargs):
            self.progresso = Progress(args.output_dir, int(math.ceil(args.num_train_epochs)), state.max_steps, state.global_step)
        def on_step_end(self, args, state, control, **kwargs):
            memoria.amostrar()
            perdas = [h['loss'] for h in state.log_history if 'loss' in h]
            self.progresso.update(state.global_step, state.epoch or 0, perdas[-1] if perdas else None)
        def on_train_end(self, args, state, control, **kwargs):
            perdas = [h['loss'] for h in state.log_history if 'loss' in h]
            self.progresso.update(state.global_step, state.epoch or 0, perdas[-1] if perdas else None, force=True)
    treinador = Trainer(model=modelo, args=config, train_dataset=treino, callbacks=[ReceiptCallback()],
                        eval_dataset=None if args.probe else validacao,
                        data_collator=Colator(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id))

    memoria = acelerador.Memoria(dispositivo)
    inicio = time.time()
    resultado = treinador.train(resume_from_checkpoint=args.resume_from_checkpoint)
    duracao = time.time() - inicio
    relatorio = {
        'schema_version': 1, 'modo': 'probe' if args.probe else 'treino', 'engine': 'hf-peft',
        'modelo': rec['model_id'], 'revision': rec['revision'], 'metodo': rec['method'],
        'gpu': acelerador.nome_do_hardware(dispositivo), 'dispositivo': dispositivo,
        'vram': memoria.relatorio(),
        'parametros': {'treinaveis': treinaveis, 'total': total, 'fracao': round(treinaveis / total, 5), 'alvos_lora': alvos},
        'receita_efetiva': {'lora': lo, 'train': {**tr, 'max_seq_length': max_seq}, 'modo_dados': modo},
        'exemplos_treino': len(treino), 'passos': resultado.global_step,
        'loss_treino': resultado.training_loss, 'duracao_s': round(duracao, 1),
        'segundos_por_passo': round(duracao / max(1, resultado.global_step), 1),
    }
    if not args.probe:
        modelo.save_pretrained(str(saida / 'adapter'))
        tok.save_pretrained(str(saida / 'adapter'))
        relatorio['metricas_validacao'] = {k: float(v) for k, v in treinador.evaluate().items() if isinstance(v, (int, float))}
        relatorio['adapter'] = 'adapter'
        relatorio['adapter_hash'] = adapter_hash(saida / 'adapter')
        relatorio['historico'] = [h for h in treinador.state.log_history if 'loss' in h or 'eval_loss' in h]
    gravar_json(saida / ('probe.json' if args.probe else 'treino.json'), relatorio)
    print(json.dumps({k: relatorio[k] for k in ('modo', 'vram', 'passos', 'loss_treino', 'duracao_s', 'segundos_por_passo')}
                     | ({'metricas_validacao': relatorio.get('metricas_validacao')} if not args.probe else {}),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    raise SystemExit(main())
