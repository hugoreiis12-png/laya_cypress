"""Modelo de decisao tipada: qualquer encoder + embedding de tipo + blocos extras + pontuador.

Os nomes de tensores seguem o DecisionModel de laya/common.py (commit em
third_party/laya/NOTICE.md), para um checkpoint nesse formato carregar com
strict=True. Sobre um encoder comum as cabecas nascem novas. act_head e
preservada, mas fica congelada e fora da politica nesta versao.
"""
from pathlib import Path
import contextlib

import torch
import torch.nn as nn

COMPONENTES = ('type_emb', 'head', 'scorer', 'act_head')
SEMENTE_CABECAS = 0


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        while d % nhead: nhead -= 1
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer('temperature', torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def _apply_rope_config(ecfg):
    """transformers 5 guarda RoPE em rope_parameters; o 4.x le global/local_rope_theta."""
    rope = getattr(ecfg, 'rope_parameters', None)
    if not isinstance(rope, dict): return
    flat = rope.get('rope_theta')
    for layer_type, attr in (('full_attention', 'global_rope_theta'), ('sliding_attention', 'local_rope_theta')):
        params = rope.get(layer_type)
        theta = params.get('rope_theta') if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def _sem_inicializacao():
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            return contextlib.nullcontext()
    return no_init_weights()


def construir(desc):
    """Modelo a partir de loader.descrever: checkpoint de decisao sem pesos (carregados depois,
    estritos) ou encoder comum com seus pesos e cabecas novas de semente fixa."""
    from transformers import AutoConfig, AutoModel
    cfg = desc['cfg']
    ecfg = AutoConfig.from_pretrained(str(desc['encoder_dir']))
    _apply_rope_config(ecfg)
    try: ecfg.reference_compile = False  # ModernBERT compilaria o encoder por padrao
    except Exception: pass
    if not desc['cabecas_novas']:
        with _sem_inicializacao():
            enc = AutoModel.from_config(ecfg, attn_implementation='sdpa')
            return DecisionModel(enc, cfg['head_layers'], cfg['n_act'])
    enc, info = AutoModel.from_pretrained(str(desc['encoder_dir']), config=ecfg, output_loading_info=True)
    # pooler fica fora do forward; qualquer outro peso ausente seria encoder aleatorio
    faltam = [k for k in info.get('missing_keys', []) if 'pooler' not in k]
    if faltam: raise ValueError('ENCODER_WEIGHTS_MISSING:%s' % sorted(faltam)[:5])
    # semente isolada: baseline, candidato e exportacao montam as mesmas cabecas em qualquer processo
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEMENTE_CABECAS)
        return DecisionModel(enc, cfg['head_layers'], cfg['n_act'])


def carregar_pesos(modelo, caminho):
    """Carga estrita: nenhum tensor faltando ou sobrando, formas iguais."""
    from safetensors.torch import load_file
    pesos = load_file(str(caminho))
    esperado = modelo.state_dict()
    faltam = sorted(set(esperado) - set(pesos))
    sobram = sorted(set(pesos) - set(esperado))
    formas = [k for k in pesos if k in esperado and tuple(pesos[k].shape) != tuple(esperado[k].shape)]
    if faltam or sobram or formas:
        raise ValueError('CHECKPOINT_TENSORS_MISMATCH:faltam=%s;sobram=%s;formas=%s' % (faltam[:5], sobram[:5], formas[:5]))
    modelo.load_state_dict(pesos, strict=True)
    return {'tensores': len(pesos)}


def salvar_pesos(modelo, caminho, dtype=torch.float16):
    """Pesos completos em fp16 (como o checkpoint de origem); temperature fica em fp32."""
    from safetensors.torch import save_file
    estado = {}
    for k, v in modelo.state_dict().items():
        v = v.detach().cpu().contiguous()
        estado[k] = v if k == 'temperature' else v.to(dtype)
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    save_file(estado, str(caminho))


def parametros_por_componente(modelo):
    contagem = {'encoder': 0}
    for nome, p in modelo.named_parameters():
        raiz = nome.split('.')[0]
        contagem[raiz] = contagem.get(raiz, 0) + p.numel()
    return contagem


def configurar_treinaveis(modelo, componentes):
    """Congela tudo fora dos componentes listados; act_head nunca treina; encoder via LoRA."""
    desconhecidos = sorted(set(componentes) - set(COMPONENTES))
    if desconhecidos or 'act_head' in componentes: raise ValueError('TRAINABLE_COMPONENT_INVALID:%s' % (desconhecidos or ['act_head']))
    for nome, p in modelo.named_parameters():
        raiz = nome.split('.')[0]
        if raiz == 'encoder': continue  # decidido pelo adapter LoRA
        p.requires_grad = raiz in componentes
    return {nome: sum(p.numel() for n, p in modelo.named_parameters() if n.split('.')[0] == nome and p.requires_grad)
            for nome in COMPONENTES}
