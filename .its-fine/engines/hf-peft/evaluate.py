"""Avaliacao congelada: base contra candidato no mesmo teste, mesmos parametros.

Metricas por item: token_f1 (sobreposicao lexical), entity_recall (cobertura
de entidades, nunca correcao factual), razao de comprimento. O relatorio detalhado
guarda as respostas; o resumo impresso e o resumo.json trazem so agregados,
para o harness nunca precisar ler conteudo quando a politica proibe.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import entity_recall, gravar_json, ler_json, ler_jsonl, sha256_arquivo, token_f1, tokens_de_parada, carga_modelo, politica_privacidade
from metric import accepted_match
from itsfine.privacy import loopback, require_destination, request_json, require_local_executor


def responder(modelo, tok, prompts, paradas, max_novos, rotulo):
    import torch
    respostas, inicio = [], time.time()
    for n, mensagens in enumerate(prompts, 1):
        entrada = (tok(mensagens,return_tensors='pt')['input_ids'] if isinstance(mensagens,str) else
                   tok.apply_chat_template(mensagens, tokenize=True, return_dict=False, add_generation_prompt=True,
                                          return_tensors='pt')).to(modelo.device)
        with torch.no_grad():
            saida = modelo.generate(entrada, max_new_tokens=max_novos, do_sample=False, temperature=None, top_p=None,
                                    top_k=None, eos_token_id=paradas, pad_token_id=paradas[0])
        respostas.append(tok.decode(saida[0][entrada.shape[1]:], skip_special_tokens=True).strip())
        if n % 20 == 0:
            print('  %s %d/%d  %.0fs' % (rotulo, n, len(prompts), time.time() - inicio), flush=True)
    return respostas


def agregados(f1, ent, comp_ref, comp_ger):
    if not f1: raise ValueError('EMPTY_EVALUATION')
    validos = [e for e in ent if e is not None]
    return {'token_f1': round(sum(f1) / len(f1), 4),
            'entity_recall': round(sum(validos) / len(validos), 4) if validos else None,
            'itens_com_entidade': len(validos),
            'palavras_medianas': sorted(comp_ger)[len(comp_ger) // 2],
            'razao_comprimento': round(sorted(g / max(1, r) for g, r in zip(comp_ger, comp_ref))[len(comp_ger) // 2], 2)}


def ollama_identity(url,name):
    entries=request_json(url+'/api/tags')['models']
    normalized=name if ':' in name.rsplit('/',1)[-1] else name+':latest'
    matches=[m for m in entries if m.get('name') in (name,normalized)]
    if len(matches)!=1 or not re.fullmatch(r'(sha256:)?[a-f0-9]{64}',matches[0].get('digest','')):
        raise ValueError('OLLAMA_MODEL_IDENTITY_UNAVAILABLE')
    item=matches[0]
    return {'digest':item['digest'],'quantization':item.get('details',{}).get('quantization_level')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teste', required=True)
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--adapter', help='pasta do adapter LoRA; omita com --pacote')
    parser.add_argument('--pacote', help='pasta de modelo mesclado; avalia so o candidato')
    parser.add_argument('--saida', required=True)
    parser.add_argument('--entidades', help='JSON com lista de entidades para entity_recall')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--limite', type=int, help='avalia so os N primeiros itens')
    parser.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--ollama', help='nome exato do modelo final registrado; avalia somente esse consumidor')
    parser.add_argument('--api-base', default='http://127.0.0.1:11434')
    parser.add_argument('--calibracao', help='relatorio validado do metric.py')
    parser.add_argument('--calibracao-hash', help='SHA-256 congelado no protocolo')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    if sum(bool(x) for x in (args.adapter,args.pacote,args.ollama))>1: raise ValueError('EXCLUSIVE_CANDIDATE_REQUIRED')
    if args.max_new_tokens<1 or (args.limite is not None and args.limite<1): raise ValueError('INVALID_EVALUATION_LIMIT')

    saida = Path(args.saida).resolve()
    saida.mkdir(parents=True, exist_ok=True)
    linhas = ler_jsonl(args.teste)[: args.limite or None]
    prompts, referencias, alternativas = evaluation_inputs(linhas)
    calibration = None
    if args.calibracao:
        if sha256_arquivo(args.calibracao) != args.calibracao_hash: raise ValueError('CALIBRATION_HASH_MISMATCH')
        calibration = ler_json(args.calibracao)
        if calibration.get('valid') is not True or calibration.get('metric')!='accepted_match': raise ValueError('INVALID_CALIBRATION')
    entidades = ler_json(args.entidades) if args.entidades else []
    if isinstance(entidades, dict):
        entidades = entidades.get('entidades', [])

    if not args.ollama:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.pacote or args.modelo, revision=None if args.pacote else args.revision)
        paradas = tokens_de_parada(tok)
        carga = carga_modelo(args.device,args.load_in_4bit)
    else:
        paradas=[]
    relatorio = {'schema_version': 1, 'engine': 'hf-peft', 'modelo': args.modelo, 'revision': args.revision,
                 'teste': str(args.teste), 'teste_hash': sha256_arquivo(args.teste), 'amostra': len(linhas),
                 'geracao': {'do_sample': False, 'max_new_tokens': args.max_new_tokens, 'tokens_de_parada': paradas},
                 'entidades': len(entidades)}
    comp_ref = [len(r.split()) for r in referencias]
    relatorio['metric_roles']={'token_f1':'vocabulary-overlap','entity_recall':'entity-coverage-only',
                               'accepted_match':'reviewed-answer-matching'}
    relatorio['calibration_hash']=args.calibracao_hash
    relatorio['factual_decision_eligible']=calibration is not None

    if args.pacote or args.ollama:
        if args.ollama:
            if not loopback(args.api_base): require_destination(politica_privacidade(args.saida),'generate',args.api_base)
            consumer_identity=ollama_identity(args.api_base,args.ollama)
            cand=[]
            for prompt in prompts:
                if not loopback(args.api_base): require_destination(politica_privacidade(args.saida),'generate',args.api_base)
                body={'model':args.ollama,'stream':False,'options':{'temperature':0,'num_predict':args.max_new_tokens}}
                if isinstance(prompt,str): body['prompt']=prompt; api='/api/generate'; field='response'
                else: body['messages']=prompt; api='/api/chat'; field='message'
                response=request_json(args.api_base+api,body)
                cand.append(response[field]['content'] if field=='message' else response[field])
            relatorio['consumer']='ollama'; relatorio['consumer_model']=args.ollama
            if ollama_identity(args.api_base,args.ollama)!=consumer_identity: raise ValueError('OLLAMA_MODEL_CHANGED_DURING_EVALUATION')
            relatorio['consumer_identity']=consumer_identity
        else:
            modelo = AutoModelForCausalLM.from_pretrained(args.pacote, **carga).eval()
            cand = responder(modelo, tok, prompts, paradas, args.max_new_tokens, 'pacote')
        f1 = [token_f1(g, r) for g, r in zip(cand, referencias)]
        ent = [entity_recall(g, r, entidades) for g, r in zip(cand, referencias)] if entidades else [None] * len(cand)
        relatorio['candidato'] = agregados(f1, ent, comp_ref, [len(c.split()) for c in cand])
        relatorio['candidato']['accepted_match']=sum(accepted_match(c,a) for c,a in zip(cand,alternativas))/len(cand)
        relatorio['respostas'] = [{'id': l.get('id'), 'candidato': c, 'f1': round(a, 4)} for l, c, a in zip(linhas, cand, f1)]
    else:
        modelo = AutoModelForCausalLM.from_pretrained(args.modelo, revision=args.revision, **carga).eval()
        base = responder(modelo, tok, prompts, paradas, args.max_new_tokens, 'base')
        relatorio['base']=agregados([token_f1(g,r) for g,r in zip(base,referencias)],
            [entity_recall(g,r,entidades) for g,r in zip(base,referencias)] if entidades else [None]*len(base),
            comp_ref,[len(x.split()) for x in base])
        relatorio['base']['accepted_match']=sum(accepted_match(b,a) for b,a in zip(base,alternativas))/len(base)
        if not args.adapter:
            relatorio['modo']='baseline-only'
            return salvar(saida,relatorio)
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, args.adapter).eval()
        cand = responder(modelo, tok, prompts, paradas, args.max_new_tokens, 'candidato')
        f1b = [token_f1(g, r) for g, r in zip(base, referencias)]
        f1c = [token_f1(g, r) for g, r in zip(cand, referencias)]
        entb = [entity_recall(g, r, entidades) for g, r in zip(base, referencias)] if entidades else [None] * len(base)
        entc = [entity_recall(g, r, entidades) for g, r in zip(cand, referencias)] if entidades else [None] * len(cand)
        relatorio['base'] = agregados(f1b, entb, comp_ref, [len(b.split()) for b in base])
        relatorio['candidato'] = agregados(f1c, entc, comp_ref, [len(c.split()) for c in cand])
        relatorio['base']['accepted_match']=sum(accepted_match(b,a) for b,a in zip(base,alternativas))/len(base)
        relatorio['candidato']['accepted_match']=sum(accepted_match(c,a) for c,a in zip(cand,alternativas))/len(cand)
        relatorio['melhoras'] = sum(1 for a, b in zip(f1b, f1c) if b > a)
        relatorio['pioras'] = sum(1 for a, b in zip(f1b, f1c) if b < a)
        relatorio['diagnostico'] = diagnostico(relatorio)
        relatorio['respostas'] = [{'id': l.get('id'), 'prompt': p, 'referencia': r,
                                   'base': b, 'candidato': c, 'f1_base': round(x, 4), 'f1_candidato': round(y, 4),
                                   'ent_base': eb, 'ent_candidato': ec}
                                  for l, p, r, b, c, x, y, eb, ec in zip(linhas, prompts, referencias, base, cand, f1b, f1c, entb, entc)]
    return salvar(saida,relatorio)


def salvar(saida,relatorio):
    gravar_json(saida / 'avaliacao.json', relatorio)
    resumo = {k: v for k, v in relatorio.items() if k != 'respostas'}
    gravar_json(saida / 'resumo.json', resumo)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0


def evaluation_inputs(rows):
    if not rows: raise ValueError('EMPTY_EVALUATION')
    prompts, references, alternatives = [], [], []
    for row in rows:
        if 'messages' in row:
            if len(row['messages'])<2 or row['messages'][-1]['role']!='assistant': raise ValueError('INVALID_EVALUATION_MESSAGES')
            prompts.append(row['messages'][:-1]); references.append(row['messages'][-1]['content'])
        elif isinstance(row.get('text'),str):
            # Frozen deterministic continuation protocol for raw text, not chat.
            words=row['text'].split(); cut=max(1,len(words)//2)
            if cut>=len(words): raise ValueError('TEXT_TOO_SHORT_FOR_CONTINUATION')
            prompts.append(' '.join(words[:cut])); references.append(' '.join(words[cut:]))
        else: raise ValueError('UNSUPPORTED_EVALUATION_ROW')
        alternatives.append(row.get('accepted_answers',[references[-1]]))
        accepted_match('',alternatives[-1])  # validate before loading weights
    return prompts,references,alternatives


def diagnostico(r):
    """Separa o que o treino ensinou. Leitura para o usuario leigo."""
    b, c = r['base'], r['candidato']
    linhas = []
    if abs(c['razao_comprimento'] - 1) < abs(b['razao_comprimento'] - 1):
        linhas.append('Comprimento mais proximo da referencia (razao %.2f contra %.2f do base); isso nao prova formato nem correcao.' % (c['razao_comprimento'], b['razao_comprimento']))
    else:
        linhas.append('Formato: nao mudou de forma clara.')
    linhas.append('Correcao factual: sobreposicao de palavras e entidades nao prova aprendizado de fatos.')
    if r.get('factual_decision_eligible'):
        linhas.append('Respostas aceitas pelo protocolo: %.2f no candidato e %.2f na base. Formulacoes nao cadastradas exigem revisao.' % (c['accepted_match'],b['accepted_match']))
    else:
        linhas.append('Aprovacao factual pendente: valide o instrumento com sondas rotuladas e congele o protocolo.')
    return linhas


if __name__=='__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main,"resumo.json"))
