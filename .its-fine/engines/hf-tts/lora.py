"""LoRA generico do hf-tts: injeta, salva, carrega e funde, em qualquer modulo que a familia indicar.

Formato do adapter igual ao do peft (adapter_model.safetensors + adapter_config.json), para a
continuacao de rodada aprovada valer como nos outros motores. Tensores que a familia precisa
alem do LoRA (ex.: embedding da voz) vao em extras.safetensors, coberto pelo adapter_hash.
"""
from pathlib import Path
import json
from common import gravar_json, ler_json

PESOS, CONFIG, EXTRAS, META = 'adapter_model.safetensors', 'adapter_config.json', 'extras.safetensors', 'itsfine-adapter.json'


def configuracao(lo, alvos, excluir=None):
    from peft import LoraConfig
    return LoraConfig(r=int(lo['r']), lora_alpha=int(lo['alpha']), lora_dropout=float(lo['dropout']), bias='none',
                      target_modules=sorted(alvos), exclude_modules=excluir)


def configuracao_salva(pasta):
    """A configuracao com que um adapter salvo foi treinado."""
    from peft import LoraConfig
    return LoraConfig.from_pretrained(str(pasta))


def injetar(modulo, cfg):
    """LoRA nos Linear alvo; so os pesos LoRA treinam, em fp32 mesmo com o modelo em bf16."""
    from peft import inject_adapter_in_model
    for p in modulo.parameters(): p.requires_grad = False
    inject_adapter_in_model(cfg, modulo)
    treinaveis = []
    for nome, p in modulo.named_parameters():
        if 'lora_' in nome:
            p.data = p.data.float(); p.requires_grad = True; treinaveis.append(p)
    if not treinaveis: raise ValueError('LORA_TARGETS_NOT_FOUND')
    return treinaveis


def estado(modulo):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(modulo).items()}


def salvar(pasta, modulo, cfg, extras, meta):
    from safetensors.torch import save_file
    pasta = Path(pasta); pasta.mkdir(parents=True, exist_ok=True)
    save_file(estado(modulo), str(pasta / PESOS))
    valor = cfg.to_dict()
    valor['target_modules'] = sorted(valor['target_modules'])
    valor = {k: (sorted(v) if isinstance(v, set) else v) for k, v in valor.items()}
    (pasta / CONFIG).write_text(json.dumps(valor, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    if extras: save_file({k: v.detach().cpu().contiguous() for k, v in extras.items()}, str(pasta / EXTRAS))
    gravar_json(pasta / META, meta)


def carregar(pasta, modulo):
    """Pesos LoRA de um adapter salvo sobre um modulo ja injetado com a mesma configuracao; devolve os extras."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    pasta = Path(pasta)
    resultado = set_peft_model_state_dict(modulo, load_file(str(pasta / PESOS)))
    if getattr(resultado, 'unexpected_keys', None): raise ValueError('ADAPTER_KEYS_MISMATCH')
    return load_file(str(pasta / EXTRAS)) if (pasta / EXTRAS).is_file() else {}


def meta(pasta):
    return ler_json(Path(pasta) / META)


def fundir(modulo):
    """Soma o LoRA nos pesos base e devolve o modulo sem camadas LoRA, pronto para salvar como modelo comum."""
    from peft.tuners.lora import LoraLayer
    for nome, filho in list(modulo.named_modules()):
        if isinstance(filho, LoraLayer):
            filho.merge()
            pai = modulo.get_submodule(nome.rsplit('.', 1)[0]) if '.' in nome else modulo
            setattr(pai, nome.rsplit('.', 1)[-1], filho.get_base_layer())
    return modulo
