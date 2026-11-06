"""Carregador generico: qualquer encoder do Hugging Face, checkpoint de decisao ou pacote exportado.

O layout e detectado pelos arquivos da pasta:
- pacote: itsfine-package.json e itsfine-model.json, saida do export.py;
- decisao: itsfine-model.json ou rl_agent_config.json (formato do Laya), com encoder/,
  tokenizer/ e model.safetensors trazendo encoder e cabecas ja treinadas;
- encoder: config.json na raiz (BERT, RoBERTa, ModernBERT, mmBERT, DeBERTa...); o
  encoder vem do checkpoint e as cabecas nascem novas, com semente fixa.
Pasta baixada pelo download.py traz itsfine-download.json com o sha256 de cada arquivo,
conferido contra o Hub no download. O motor nunca grava na pasta do modelo.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AQUI, ler_json, sha256_arquivo, sha256_json  # noqa: E402
import decision_model  # noqa: E402

MANIFESTO = 'itsfine-package.json'
LOCK = 'itsfine-download.json'
CONFIGS_DECISAO = ('itsfine-model.json', 'rl_agent_config.json')
PESOS = ('.safetensors', '.bin')


def layout(pasta):
    pasta = Path(pasta)
    if (pasta / MANIFESTO).is_file(): return 'pacote'
    if any((pasta / nome).is_file() for nome in CONFIGS_DECISAO): return 'decisao'
    if (pasta / 'config.json').is_file(): return 'encoder'
    raise ValueError('CHECKPOINT_LAYOUT_UNKNOWN:' + str(pasta))


def arquivos_do_modelo(pasta):
    """Arquivos que compoem o modelo: tudo fora de pastas e arquivos ocultos, menos o lock."""
    pasta = Path(pasta)
    return sorted(p.relative_to(pasta).as_posix() for p in pasta.rglob('*')
                  if p.is_file() and p.name != LOCK and not any(parte.startswith('.') for parte in p.relative_to(pasta).parts))


def _max_len_encoder(pasta, ecfg, limite):
    """Menor entre o limite do motor, as posicoes do encoder e o limite declarado pelo tokenizer."""
    valores = [int(limite)]
    posicoes = ecfg.get('max_position_embeddings')
    if isinstance(posicoes, int):
        # familia RoBERTa reserva pad_token_id + 1 posicoes
        valores.append(posicoes - 2 if ecfg.get('model_type') in ('roberta', 'xlm-roberta', 'camembert') else posicoes)
    tcfg = pasta / 'tokenizer_config.json'
    declarado = ler_json(tcfg).get('model_max_length') if tcfg.is_file() else None
    if isinstance(declarado, int) and 0 < declarado < 1_000_000: valores.append(declarado)
    return min(valores)


def descrever(pasta):
    """Layout, configuracao normalizada, pastas do encoder e do tokenizer e arquivos de pesos."""
    pasta = Path(pasta)
    tipo = layout(pasta)
    if tipo == 'encoder':
        novas = ler_json(AQUI / 'engine.json')['cabecas_novas']
        ecfg = ler_json(pasta / 'config.json')
        pesos = sorted(p.name for p in pasta.glob('*.safetensors')) or sorted(p.name for p in pasta.glob('pytorch_model*.bin'))
        if not pesos: raise ValueError('CHECKPOINT_WEIGHTS_MISSING')
        cfg = {'head_layers': int(novas['head_layers']), 'n_act': int(novas['n_act']), 'head_max_len': int(novas['head_max_len']),
               'max_len': _max_len_encoder(pasta, ecfg, novas['max_len'])}
        return {'layout': tipo, 'cabecas_novas': True, 'cfg': cfg, 'encoder_dir': pasta, 'tokenizer_dir': pasta, 'pesos': pesos}
    bruta = ler_json(next(pasta / nome for nome in CONFIGS_DECISAO if (pasta / nome).is_file()))
    n_act = bruta['n_act'] if 'n_act' in bruta else len(bruta.get('act_costs', {})) + 1
    cfg = {**bruta, 'head_layers': int(bruta.get('head_layers', 2)), 'n_act': int(n_act),
           'max_len': int(bruta.get('max_len', 512)), 'head_max_len': int(bruta.get('head_max_len', 192))}
    if not (pasta / 'model.safetensors').is_file(): raise ValueError('CHECKPOINT_WEIGHTS_MISSING')
    return {'layout': tipo, 'cabecas_novas': False, 'cfg': cfg, 'encoder_dir': pasta / 'encoder',
            'tokenizer_dir': pasta / 'tokenizer', 'pesos': ['model.safetensors']}


def arquivos_tokenizer(desc):
    """Arquivos do tokenizer a copiar para o pacote."""
    pasta = Path(desc['tokenizer_dir'])
    if desc['layout'] != 'encoder':
        return [p for p in sorted(pasta.rglob('*')) if p.is_file()]
    fora = {'config.json', LOCK, MANIFESTO, 'generation_config.json'}
    return [p for p in sorted(pasta.iterdir()) if p.is_file() and p.name not in fora and not p.name.startswith('.')
            and p.suffix not in PESOS + ('.md', '.msgpack', '.h5', '.onnx', '.ot')]


def verificar_checkpoint(pasta, ident, revision=None):
    """Confere a pasta contra o lock do download.py: mesmo repositorio, mesma revisao, mesmos arquivos e hashes."""
    pasta = Path(pasta)
    if not (pasta / LOCK).is_file(): raise ValueError('CHECKPOINT_LOCK_MISSING: baixe ou confira a pasta com download.py')
    lock = ler_json(pasta / LOCK)
    if lock.get('id') != ident: raise ValueError('CHECKPOINT_ID_MISMATCH:%s' % lock.get('id'))
    if revision and lock.get('revision') != revision: raise ValueError('CHECKPOINT_REVISION_MISMATCH:%s' % lock.get('revision'))
    presentes = set(arquivos_do_modelo(pasta))
    for nome, esperado in lock['files'].items():
        if nome not in presentes: raise ValueError('CHECKPOINT_FILE_MISSING:' + nome)
        if sha256_arquivo(pasta / nome) != esperado: raise ValueError('CHECKPOINT_FILE_HASH_MISMATCH:' + nome)
    extras = sorted(presentes - set(lock['files']))
    if extras: raise ValueError('CHECKPOINT_FILE_UNEXPECTED:%s' % extras[:5])
    return lock


def verificar_pacote(pasta):
    """Pacote exportado: todos os arquivos do manifesto presentes e com o hash declarado."""
    pasta = Path(pasta)
    manifesto = ler_json(pasta / MANIFESTO)
    for nome, esperado in manifesto['files'].items():
        alvo = pasta / nome
        if not alvo.is_file(): raise ValueError('PACKAGE_FILE_MISSING:' + nome)
        if sha256_arquivo(alvo) != esperado: raise ValueError('PACKAGE_FILE_HASH_MISMATCH:' + nome)
    if manifesto['package_id'] != sha256_json(manifesto['files']): raise ValueError('PACKAGE_ID_MISMATCH')
    return manifesto


def identidade(pasta):
    """Identidade do que esta na pasta: package_id do pacote ou hash de todos os arquivos do modelo."""
    pasta = Path(pasta)
    if (pasta / MANIFESTO).is_file():
        return {'tipo': 'pacote', 'package_id': ler_json(pasta / MANIFESTO)['package_id']}
    return {'tipo': 'checkpoint', 'package_id': sha256_json({nome: sha256_arquivo(pasta / nome) for nome in arquivos_do_modelo(pasta)})}


def carregar_tokenizer(pasta):
    from transformers import AutoConfig, AutoTokenizer
    desc = descrever(pasta)
    # config do encoder resolve a classe quando o tokenizer_config nao a declara (ex.: BERT so com vocab.txt)
    tok = AutoTokenizer.from_pretrained(str(desc['tokenizer_dir']), config=AutoConfig.from_pretrained(str(desc['encoder_dir'])))
    for atributo in ('cls_token_id', 'sep_token_id', 'mask_token_id', 'pad_token_id'):
        if getattr(tok, atributo, None) is None: raise ValueError('TOKENIZER_WITHOUT_' + atributo.upper())
    return tok


def carregar(pasta, device='cpu', dtype=None, verificar=None, adapter=None):
    """Modelo, tokenizer e configuracao de uma pasta de encoder, checkpoint de decisao ou pacote.

    verificar: {'id':..., 'revision':...} confere a pasta contra o lock do download;
    None aceita pacote com manifesto ou pasta ja conferida pelo chamador.
    adapter: pasta com adapter LoRA (peft) e heads.safetensors treinados.
    """
    import torch
    pasta = Path(pasta)
    if (pasta / MANIFESTO).is_file():
        manifesto = verificar_pacote(pasta)
    else:
        manifesto = None
        if verificar: verificar_checkpoint(pasta, verificar['id'], verificar.get('revision'))
    desc = descrever(pasta)
    tok = carregar_tokenizer(pasta)
    modelo = decision_model.construir(desc)
    tensores = None if desc['cabecas_novas'] else decision_model.carregar_pesos(modelo, pasta / 'model.safetensors')['tensores']
    if adapter:
        aplicar_adapter(modelo, adapter)
    modelo = modelo.to(device=device, dtype=dtype or torch.float32).eval()
    return {'modelo': modelo, 'tok': tok, 'cfg': desc['cfg'], 'desc': desc, 'pasta': pasta, 'manifesto': manifesto,
            'tensores': tensores, 'identidade': identidade(pasta)}


def aplicar_adapter(modelo, adapter, treinavel=False):
    """Encoder recebe o adapter LoRA; cabecas recebem heads.safetensors."""
    from peft import PeftModel
    from safetensors.torch import load_file
    adapter = Path(adapter)
    if not (adapter / 'adapter_model.safetensors').is_file() or not (adapter / 'heads.safetensors').is_file():
        raise ValueError('ADAPTER_INCOMPLETE')
    modelo.encoder = PeftModel.from_pretrained(modelo.encoder, str(adapter), is_trainable=treinavel)
    cabecas = load_file(str(adapter / 'heads.safetensors'))
    esperado = {k: v for k, v in modelo.state_dict().items() if k.split('.')[0] in decision_model.COMPONENTES}
    faltam = sorted(set(cabecas) - set(esperado))
    if faltam: raise ValueError('HEADS_TENSORS_UNKNOWN:%s' % faltam[:5])
    modelo.load_state_dict(cabecas, strict=False)
    return sorted({k.split('.')[0] for k in cabecas})
