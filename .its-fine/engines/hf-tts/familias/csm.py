"""Adaptador da familia CSM (Sesame): backbone Llama + decodificador de profundidade sobre o codec Mimi.

Nativo do transformers: o processador monta a conversa (papel = id do locutor) e devolve os
rotulos de treino, e a perda do modelo ja chega com o alinhamento certo (o backbone desloca uma
vez; o decodificador de profundidade recebe rotulos ja deslocados). O codec Mimi fica fora do
LoRA. A voz propria e o locutor "0" sem contexto; o modelo base clona com um clipe de contexto.
"""
from pathlib import Path

LOCUTOR = '0'
QUADROS_POR_SEGUNDO = 12.5


class Familia:
    nome = 'csm'
    taxa = 24000

    def __init__(self, origem, dispositivo, dtype, revision=None, idioma=None):
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration
        if not Path(origem).is_dir():
            from huggingface_hub import snapshot_download
            origem = snapshot_download(origem, revision=revision)
        self.origem, self.torch, self.dispositivo, self.dtype = Path(origem), torch, dispositivo, dtype
        self.processor = AutoProcessor.from_pretrained(str(origem))
        self.modelo = CsmForConditionalGeneration.from_pretrained(str(origem), dtype=dtype).to(dispositivo)
        self.modelo.codec_model.eval()

    @classmethod
    def do_pacote(cls, pasta, dispositivo, dtype, idioma=None):
        return cls(pasta, dispositivo, dtype)

    # --- treino ---
    def lora(self):
        return self.modelo, ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], r'.*codec_model.*'

    def acaso(self):
        import math
        cfg = self.modelo.config
        return {'backbone': math.log(cfg.vocab_size), 'profundidade': math.log(cfg.depth_decoder_config.vocab_size)}

    def definir_voz(self, referencia_audio):
        pass  # a voz e aprendida pelo locutor "0"; nao ha embedding separado

    def extras(self):
        return {}

    def carregar_extras(self, tensores):
        pass

    def preparar(self, texto, audio):
        from common import carregar_audio
        return {'texto': texto, 'audio': carregar_audio(audio, self.taxa)}

    def _entradas(self, conversas, rotulos):
        entradas = self.processor.apply_chat_template(conversas, tokenize=True, return_dict=True, output_labels=rotulos)
        return {k: (v.to(self.dispositivo, self.dtype) if k == 'input_values' else v.to(self.dispositivo))
                for k, v in entradas.items() if hasattr(v, 'to')}

    def perda(self, exemplos):
        conversas = [[{'role': LOCUTOR, 'content': [{'type': 'text', 'text': e['texto']}, {'type': 'audio', 'path': e['audio']}]}]
                     for e in exemplos]
        saida = self.modelo(**self._entradas(conversas, True))
        return saida.loss, {'backbone': float(saida.backbone_loss.detach()), 'profundidade': float(saida.depth_decoder_loss.detach())}

    def salvar(self, pasta, voz):
        self.modelo.save_pretrained(str(pasta))
        self.processor.save_pretrained(str(pasta))

    # --- inferencia ---
    def falar(self, texto, voz=None, referencia=None):
        import numpy as np
        from common import carregar_audio
        conversa = []
        if referencia:
            audio, transcricao = referencia
            conversa.append({'role': LOCUTOR, 'content': [{'type': 'text', 'text': transcricao},
                                                          {'type': 'audio', 'path': carregar_audio(audio, self.taxa)}]})
        conversa.append({'role': LOCUTOR, 'content': [{'type': 'text', 'text': texto}]})
        # teto de quadros pelo tamanho do texto (~12 caracteres por segundo de fala), com folga
        teto = int(QUADROS_POR_SEGUNDO * (len(texto) / 12 + 3))
        with self.torch.no_grad():
            onda = self.modelo.generate(**self._entradas(conversa, False), output_audio=True, max_new_tokens=teto)[0]
        return np.asarray(onda.float().cpu(), dtype=np.float32), self.taxa
