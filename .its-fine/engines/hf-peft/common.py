"""Funcoes compartilhadas do motor hf-peft.

Tudo aqui foi escrito contra problemas reais encontrados em rodadas:
- o transformers 5 devolve BatchEncoding em apply_chat_template por padrao;
- o config de alguns modelos declara um EOS diferente do token que fecha o
  turno no chat template, e a geracao nao para;
- o chat template pode nao trazer o marcador de geracao que a trl exige,
  entao a mascara de loss e construida pela diferenca de prefixo.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import unicodedata
import sys
import tempfile
from collections import Counter

# Avisos do Hub e do transformers vao para stdout e quebram quem le o JSON impresso.
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# Mac com chip M: operacao sem kernel Metal cai para a CPU; precisa valer antes de importar torch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


# Canonical runtime in the repository or its installed copy.
_project = Path(__file__).resolve().parents[2]
_snapshot_lib = Path(__file__).resolve().parent.parent / 'lib'
sys.path.insert(0, str(_snapshot_lib if (_snapshot_lib/'itsfine').is_dir() else
                      (_project / 'lib' if (_project / 'lib/itsfine').is_dir() else _project.parent)))
from itsfine.privacy import load_policy as politica_privacidade


def ler_json(caminho):
    return json.loads(Path(caminho).read_text(encoding='utf-8'))


def gravar_json(caminho, dados):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + caminho.name, dir=caminho.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(dados, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, caminho)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def ler_jsonl(caminho):
    return list(iterar_jsonl(caminho))


def iterar_jsonl(caminho):
    with Path(caminho).open(encoding='utf-8') as stream:
        for bruta in stream:
            if bruta.strip(): yield json.loads(bruta)


def gravar_jsonl(caminho, linhas):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open('w', encoding='utf-8', newline='\n') as saida:
        for linha in linhas:
            saida.write(json.dumps(linha, ensure_ascii=False) + '\n')


def sha256_arquivo(caminho):
    sha = hashlib.sha256()
    with Path(caminho).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): sha.update(block)
    return sha.hexdigest()


def sha256_texto(texto):
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()


def ids_do_template(tok, mensagens, gerar=False):
    """apply_chat_template que sempre devolve lista de ids."""
    return tok.apply_chat_template(mensagens, tokenize=True, return_dict=False,
                                   add_generation_prompt=gerar)


def prefixo_e_alvo(tok, mensagens):
    """Separa prompt e alvo pela diferenca de prefixo.

    Devolve (completo, tamanho_do_prefixo). Levanta ValueError se o prompt
    renderizado nao for prefixo da sequencia completa, o que indica template
    incompativel com esse metodo.
    """
    completo = ids_do_template(tok, mensagens)
    prefixo = ids_do_template(tok, mensagens[:-1], gerar=True)
    if completo[:len(prefixo)] != prefixo:
        raise ValueError('PREFIXO_DIVERGENTE')
    if len(completo) <= len(prefixo):
        raise ValueError('ALVO_VAZIO')
    return completo, len(prefixo)


def tokens_de_parada(tok):
    """Deriva os tokens que encerram um turno a partir do template, nao do config.

    Renderiza uma conversa minima e olha o que vem depois do conteudo do
    assistente. Junta com o EOS do tokenizer e o do config quando existirem.
    """
    paradas = []
    try:
        mensagens = [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}]
        completo, corte = prefixo_e_alvo(tok, mensagens)
        alvo = completo[corte:]
        for ident in alvo:
            if ident in tok.all_special_ids:
                paradas.append(ident)
    except Exception:
        pass
    for ident in (tok.eos_token_id, getattr(tok, 'pad_token_id', None)):
        if isinstance(ident, int) and ident >= 0 and ident not in paradas:
            paradas.append(ident)
    for nome in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>', '<end_of_turn>', '</s>'):
        ident = tok.convert_tokens_to_ids(nome)
        if isinstance(ident, int) and ident >= 0 and ident != getattr(tok, 'unk_token_id', -1) \
                and ident not in paradas:
            paradas.append(ident)
    return paradas


def normalizar_texto(texto):
    t = unicodedata.normalize('NFKC', texto).lower()
    t = re.sub(r'[^\w\s]', ' ', t, flags=re.UNICODE)
    return t.split()


def token_f1(gerado, referencia):
    """F1 em nivel de token. Mede sobreposicao de vocabulario, nao correcao."""
    a, b = normalizar_texto(gerado), normalizar_texto(referencia)
    if not a or not b:
        return 0.0
    comuns = sum((Counter(a) & Counter(b)).values())
    if comuns == 0:
        return 0.0
    precisao, revocacao = comuns / len(a), comuns / len(b)
    return 2 * precisao * revocacao / (precisao + revocacao)


def entidades_presentes(texto, entidades):
    """Quais entidades da lista aparecem no texto, ignorando caixa e acento."""
    base = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode().lower()
    achadas = set()
    for entidade in entidades:
        alvo = unicodedata.normalize('NFKD', entidade).encode('ascii', 'ignore').decode().lower()
        if alvo and alvo in base:
            achadas.add(entidade)
    return achadas


def entity_recall(gerado, referencia, entidades):
    """Fracao das entidades presentes na referencia que a resposta reproduziu.

    Devolve None quando a referencia nao contem nenhuma entidade da lista,
    para o chamador excluir o item da media em vez de contar zero.
    """
    esperadas = entidades_presentes(referencia, entidades)
    if not esperadas:
        return None
    obtidas = entidades_presentes(gerado, entidades)
    return len(esperadas & obtidas) / len(esperadas)


def modulos_lineares(modelo, excluir=('lm_head',)):
    """Nomes de modulos Linear do modelo, para alvo de LoRA em qualquer familia."""
    import torch
    nomes = set()
    for nome, modulo in modelo.named_modules():
        if isinstance(modulo, torch.nn.Linear):
            folha = nome.split('.')[-1]
            if folha not in excluir:
                nomes.add(folha)
    return sorted(nomes)


def carga_modelo(device='auto', quantized=False, treino=False):
    """Argumentos de from_pretrained para cuda, mps (chip Apple) ou cpu."""
    from itsfine import device as acelerador
    chosen = acelerador.escolher(device, treino)
    acelerador.exigir_cuda_para_quantizar(chosen, quantized)
    dtype = acelerador.dtype(chosen)
    result = {'device_map':acelerador.device_map(chosen),'dtype':dtype}
    if quantized:
        from transformers import BitsAndBytesConfig
        result['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
    return result


def data_entrypoint(main, summary=None):
    """CLI boundary: library stdout/stderr stay local; return whitelisted aggregates.

    This controls these executables, not arbitrary programs launched by a harness.
    """
    import contextlib
    from itsfine.privacy import require_local_executor
    if '--help' in sys.argv or '-h' in sys.argv:
        return main()
    try:
        output=Path(sys.argv[sys.argv.index('--saida')+1]).resolve()
        policy=politica_privacidade(output)
        require_local_executor(policy)
    except Exception:
        print(json.dumps({'status':'refused','error':'POLICY_OR_OUTPUT_INVALID'}))
        return 3
    logdir=Path(policy['_policy_path']).parent/'private-logs'
    logdir.mkdir(parents=True,exist_ok=True)
    fd,logpath=tempfile.mkstemp(prefix='engine-',suffix='.log',dir=logdir)
    saved_out,saved_err=os.dup(1),os.dup(2)
    result=1
    try:
        sys.stdout.flush(); sys.stderr.flush()
        with os.fdopen(fd,'w',encoding='utf-8') as log:
            os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:
                    result=main()
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    result=1
                finally: log.flush()
    finally:
        os.dup2(saved_out,1); os.dup2(saved_err,2)
        os.close(saved_out); os.close(saved_err)
    response={'status':'completed' if result==0 else 'failed','exit_code':result}
    if summary and result==0:
        try:
            value=ler_json(output/summary if summary!='.' else output)
            # Keys are explicit; unknown strings/identifiers are never copied.
            allowed={'linhas','arquivos','grupos','trechos','documentos_sem_texto','falhas_pendentes',
                'amostra','base','candidato','token_f1','entity_recall','accepted_match','razao_comprimento',
                'palavras_medianas','itens_com_entidade','factual_decision_eligible','valid','correct_count',
                'incorrect_count','correct_min','incorrect_max','splits','train','validation','test',
                'exemplos','max_seq_length_sugerido','bytes','consumidores','transformers','ollama',
                'gguf','ok','create_ok','run_ok','palavras','anonimizacao','trocas','linhas_alteradas',
                'cpf','cnpj','email','telefone','cep','pessoa'}
            def aggregates(obj):
                if not isinstance(obj,dict): return {}
                return {k:(aggregates(v) if isinstance(v,dict) else v) for k,v in obj.items()
                        if k in allowed and (v is None or type(v) in (bool,int,float,dict))}
            response['summary']=aggregates(value)
        except Exception:
            response={'status':'failed','error':'SUMMARY_INVALID','exit_code':1}
            result=1
    print(json.dumps(response,ensure_ascii=False,allow_nan=False))
    return result
