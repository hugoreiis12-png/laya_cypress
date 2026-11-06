"""Exporta o adapter mesclado e entrega o modelo pronto para o usuario usar.

Saidas em `<saida>/`: `merged/` em safetensors bf16 para transformers,
`modelo-bf16.gguf` quando o llama.cpp esta disponivel, `Modelfile`, e, com
--ollama-nome, o modelo registrado no Ollama, quantizado e testado com uma
pergunta. A mesclagem roda na CPU para nao disputar VRAM; o pacote e
recarregado do disco para a sanidade.

Caminho para o Ollama: primeiro GGUF, porque o Ollama nao importa safetensors
de toda familia (Qwen3.5 falha com "unsupported architecture"); sem GGUF,
tenta o import direto; se ambos falharem, diz isso e entrega o merged.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_json, sha256_arquivo, sha256_texto, tokens_de_parada, carga_modelo, politica_privacidade
from itsfine.privacy import require_local_executor

AQUI = Path(__file__).resolve().parent


def rodar(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()[-2500:]
    except (OSError, subprocess.TimeoutExpired) as erro:
        return None, str(erro)


def localizar_llama_cpp(informado):
    candidatos = [Path(informado)] if informado else []
    candidatos += [AQUI.parents[1] / 'tools' / 'llama.cpp']
    for c in candidatos:
        if (c / 'convert_hf_to_gguf.py').exists():
            return c
    return None


def flags_conversao(pacote):
    """Flags extras do conversor conforme o config do modelo mesclado.

    O checkpoint original do Qwen3.5 traz uma camada de previsao multi-token
    (`mtp_num_hidden_layers`) cujos pesos nao entram no modelo causal salvo
    pelo peft. Sem aviso, o conversor conta essa camada como bloco extra e o
    Ollama recusa o GGUF por tensor ausente; zerar o campo faz o conversor
    abortar por assert. `--no-mtp` e o caminho previsto pelo llama.cpp: a
    camada so serve para decodificacao especulativa e o Ollama nao a usa.
    """
    cfg = ler_json(pacote / 'config.json')
    if cfg.get('mtp_num_hidden_layers'):
        return ['--no-mtp'], {'mtp_num_hidden_layers': cfg['mtp_num_hidden_layers'], 'acao': '--no-mtp na conversao'}
    return [], {}


def modelfile(origem, system, paradas_txt):
    linhas = ['FROM %s' % Path(origem).as_posix()]
    if system:
        linhas.append('SYSTEM """%s"""' % system.replace('"""', '"'))
    linhas.append('PARAMETER temperature 0')
    for p in paradas_txt:
        linhas.append('PARAMETER stop "%s"' % p)
    return '\n'.join(linhas) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--saida', required=True)
    parser.add_argument('--system', default='')
    parser.add_argument('--ollama-nome', help='registra no Ollama com esse nome e testa')
    parser.add_argument('--quantize', default='q4_K_M', help='quantizacao aplicada pelo ollama create; vazio para nenhuma')
    parser.add_argument('--pergunta-sanidade', default='Explique em uma frase o que voce faz.')
    parser.add_argument('--llama-cpp', help='pasta do llama.cpp; sem isso procura .its-fine/tools/llama.cpp')
    parser.add_argument('--sem-gguf', action='store_true')
    parser.add_argument('--device', choices=['cpu','cuda','mps','auto'],default='cpu',help='destino da sanidade; CPU por padrão')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    saida = Path(args.saida).resolve()
    pacote = saida / 'merged'
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('EXPORT_OUTPUT_NOT_EMPTY')
    saida.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.modelo, revision=args.revision)
    paradas = tokens_de_parada(tok)
    paradas_txt = [tok.convert_ids_to_tokens(i) for i in paradas]
    print('mesclando na CPU...', flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.modelo, revision=args.revision, dtype=torch.bfloat16, device_map={'': 'cpu'})
    mesclado = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    mesclado.save_pretrained(str(pacote), safe_serialization=True)
    tok.save_pretrained(str(pacote))
    del mesclado, base
    torch.cuda.empty_cache()
    flags_extra, ajustes = flags_conversao(pacote)
    arquivos = sorted(p for p in pacote.rglob('*') if p.is_file())
    relatorio = {'schema_version': 1, 'engine': 'hf-peft', 'formato': 'safetensors bf16 mesclado',
                 'modelo_base': args.modelo, 'revision_base': args.revision, 'adapter': str(args.adapter),
                 'pacote': 'merged', 'arquivos': len(arquivos), 'bytes': sum(p.stat().st_size for p in arquivos),
                 'checksums': {p.relative_to(pacote).as_posix(): sha256_arquivo(p) for p in arquivos},
                 'config_ajustado': ajustes, 'template_hash': sha256_texto(tok.chat_template or ''),
                 'tokens_de_parada': paradas_txt, 'consumidores': {}}

    print('sanidade: recarregando o pacote do disco...', flush=True)
    modelo = AutoModelForCausalLM.from_pretrained(str(pacote), **carga_modelo(args.device)).eval()
    entrada = tok.apply_chat_template([{'role': 'user', 'content': args.pergunta_sanidade}], tokenize=True,
                                      return_dict=False, add_generation_prompt=True, return_tensors='pt').to(modelo.device)
    with torch.no_grad():
        gerado = modelo.generate(entrada, max_new_tokens=80, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 eos_token_id=paradas, pad_token_id=paradas[0])
    texto = tok.decode(gerado[0][entrada.shape[1]:], skip_special_tokens=True).strip()
    relatorio['consumidores']['transformers'] = {'ok': bool(texto), 'palavras': len(texto.split()),
                                                 'uso': 'AutoModelForCausalLM.from_pretrained("%s")' % pacote.as_posix()}
    del modelo
    torch.cuda.empty_cache()

    origem_ollama = pacote
    gguf = None
    llama = None if args.sem_gguf else localizar_llama_cpp(args.llama_cpp)
    if llama:
        gguf = saida / 'modelo-bf16.gguf'
        print('convertendo para GGUF com llama.cpp...', flush=True)
        codigo, log = rodar([sys.executable, str(llama / 'convert_hf_to_gguf.py'), str(pacote), '--outfile', str(gguf),
                             '--outtype', 'bf16'] + flags_extra, 3600)
        relatorio['consumidores']['gguf'] = {'ok': codigo == 0 and gguf.exists(), 'arquivo': gguf.name, 'llama_cpp': str(llama),
                                             'tipo': 'bf16 sem quantizar; o ollama create quantiza', 'log': log[-300:]}
        if codigo == 0 and gguf.exists():
            origem_ollama = gguf
            relatorio['consumidores']['gguf']['sha256']=sha256_arquivo(gguf)
        else:
            gguf = None
    else:
        relatorio['consumidores']['gguf'] = {'ok': False, 'motivo': 'llama.cpp nao encontrado; clone em .its-fine/tools/llama.cpp ou passe --llama-cpp'}

    (saida / 'Modelfile').write_text(modelfile(origem_ollama, args.system, paradas_txt), encoding='utf-8')
    relatorio['modelfile'] = {'arquivo': 'Modelfile', 'from': origem_ollama.name}

    if args.ollama_nome:
        cmd = ['ollama', 'create', args.ollama_nome, '-f', str(saida / 'Modelfile')]
        if gguf is not None and args.quantize:
            cmd += ['--quantize', args.quantize]
        codigo, log = rodar(cmd, 3600)
        info = {'nome': args.ollama_nome, 'origem': origem_ollama.name, 'quantize': args.quantize if gguf is not None else None,
                'create_ok': codigo == 0, 'log': log[-300:]}
        if codigo != 0 and gguf is None:
            info['motivo'] = 'Ollama nao importa safetensors desta familia; gere GGUF com llama.cpp'
        if codigo == 0:
            codigo2, resposta = rodar(['ollama', 'run', args.ollama_nome, args.pergunta_sanidade], 900)
            limpo = ' '.join(resposta.split())
            info['run_ok'] = codigo2 == 0 and 'error' not in limpo.lower()[:200] and bool(limpo)
            info['palavras'] = len(limpo.split())
            info['uso'] = 'ollama run %s' % args.ollama_nome
            if not info['run_ok']:
                info['log_run'] = limpo[-300:]
        relatorio['consumidores']['ollama'] = info

    relatorio['qualidade_final']='pendente: reavaliar no consumidor e quantizacao entregues'
    gravar_json(saida / 'export.json', relatorio)
    resumo = {k: v for k, v in relatorio.items() if k != 'checksums'}
    resumo['consumidores']={name:{k:v for k,v in info.items() if k not in ('log','log_run')} for name,info in relatorio['consumidores'].items()}
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0 if (not args.ollama_nome or relatorio['consumidores'].get('ollama',{}).get('run_ok')) else 2


if __name__=='__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main,"export.json"))
