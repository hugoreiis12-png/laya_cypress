"""Adaptador da familia Qwen3-TTS (12 Hz): LM sobre codec de 16 camadas + sub-preditor.

Layout de sequencia derivado do codigo de inferencia do proprio modelo (modo nao-streaming de
generate_custom_voice), nao do script oficial de fine-tuning, que tem dois erros medidos:
rotulos deslocados duas vezes e sub-preditor alimentado pelo estado do passo errado. Aqui a
perda e calculada a mao, com o alinhamento conferido contra a inferencia:
- a posicao p preve o codigo 0 do quadro em p+1;
- o sub-preditor recebe o estado da posicao que previu o codigo 0 do quadro (a anterior a ele);
- o texto passa pela text_projection, como na inferencia (sem isso o 0.6B nem roda).
O modelo ajustado vira voz propria (custom voice): o embedding da voz de referencia ocupa a
linha VOZ_ID do codec_embedding e o config ganha o nome da voz.
"""
from pathlib import Path
import json
import shutil

IDIOMAS = {'pt': 'portuguese', 'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian',
           'ja': 'japanese', 'ko': 'korean', 'zh': 'chinese', 'ru': 'russian'}
VOZ_ID = 3000  # linha livre do codec_embedding usada pelo fine-tuning de voz propria
PESO_SUB = 0.3  # peso da perda do sub-preditor, o mesmo da receita do fornecedor


def _registrar():
    from qwen_tts import Qwen3TTSModel  # noqa: F401  registra o tipo qwen3_tts no transformers
    return Qwen3TTSModel


class Familia:
    nome = 'qwen3-tts'
    taxa = 24000

    def __init__(self, origem, dispositivo, dtype, revision=None, idioma=None, _tts=None):
        import torch
        Qwen3TTSModel = _registrar()
        if _tts is None:
            if not Path(origem).is_dir():
                from huggingface_hub import snapshot_download
                origem = snapshot_download(origem, revision=revision)
            # device_map leva junto o tokenizer de fala, que nao e submodulo do modelo; sem itsfine: o pacote fala sozinho
            mapa = {'': 0 if dispositivo == 'cuda' else dispositivo}
            _tts = Qwen3TTSModel.from_pretrained(str(origem), dtype=dtype, attn_implementation='sdpa', device_map=mapa)
        self.origem, self.tts, self.torch = Path(origem), _tts, torch
        self.modelo = _tts.model
        self.dispositivo, self.dtype = dispositivo, dtype
        self.cfg = self.modelo.config
        self.idioma = IDIOMAS.get(idioma) if idioma else None
        self.voz_embedding = None

    @classmethod
    def do_pacote(cls, pasta, dispositivo, dtype, idioma=None):
        return cls(pasta, dispositivo, dtype, idioma=idioma)

    # --- treino ---
    def lora(self):
        """Modulo onde o LoRA entra, nomes-folha alvo e o que excluir."""
        return self.modelo.talker, ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], None

    def acaso(self):
        import math
        return {'codec0': math.log(self.modelo.talker.codec_head.out_features),
                'camadas_finas': math.log(self.modelo.talker.code_predictor.lm_head[0].out_features)}

    def definir_voz(self, referencia_audio):
        from common import carregar_audio
        with self.torch.no_grad():
            emb = self.modelo.extract_speaker_embedding(carregar_audio(referencia_audio, self.taxa), self.taxa)
        self.voz_embedding = emb.detach().float().clone()  # fora do inference_mode, para entrar no grafo de treino

    def extras(self):
        return {'voz_embedding': self.voz_embedding}

    def carregar_extras(self, tensores):
        self.voz_embedding = tensores['voz_embedding'].float()

    def preparar(self, texto, audio):
        """Exemplo de treino: ids do texto e codigos do codec do clipe (CPU)."""
        from common import carregar_audio
        torch = self.torch
        ids = self.tts.processor(text=f'<|im_start|>assistant\n{texto}<|im_end|>\n<|im_start|>assistant\n', return_tensors='pt')['input_ids']
        ids = ids[0, :-5]  # tira im_end, \n, im_start, assistant, \n: fica papel (3 tokens) + texto
        with torch.no_grad():
            codigos = self.modelo.speech_tokenizer.encode(carregar_audio(audio, self.taxa), sr=self.taxa).audio_codes[0]
        return {'texto': ids.long().cpu(), 'codigos': torch.as_tensor(codigos).long().cpu()}

    def _prefixo(self):
        t = self.cfg.talker_config
        if self.idioma:
            return [t.codec_think_id, t.codec_think_bos_id, t.codec_language_id[self.idioma], t.codec_think_eos_id, None, t.codec_pad_id]
        return [t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id, None, t.codec_pad_id]

    def _lote(self, exemplos):
        torch, cfg, t = self.torch, self.cfg, self.cfg.talker_config
        prefixo = self._prefixo(); k = len(prefixo); voz = 3 + prefixo.index(None)
        tamanhos = [3 + k + (len(e['texto']) - 3) + 2 + len(e['codigos']) + 1 for e in exemplos]
        b, n = len(exemplos), max(tamanhos)
        texto = torch.full((b, n), cfg.tts_pad_token_id, dtype=torch.long)
        codec = torch.full((b, n), t.codec_pad_id, dtype=torch.long)
        finas = torch.zeros((b, n, 16), dtype=torch.long)
        mascara_texto = torch.zeros((b, n), dtype=torch.bool); mascara_codec = torch.zeros((b, n), dtype=torch.bool)
        quadros = torch.zeros((b, n), dtype=torch.bool); atencao = torch.zeros((b, n), dtype=torch.long)
        rotulos = torch.full((b, n), -100, dtype=torch.long)
        for i, e in enumerate(exemplos):
            ids, cod = e['texto'], e['codigos']; T, N = len(ids) - 3, len(cod)
            texto[i, :3] = ids[:3]
            texto[i, 3 + k - 1] = cfg.tts_bos_token_id
            texto[i, 3 + k:3 + k + T] = ids[3:]
            texto[i, 3 + k + T] = cfg.tts_eos_token_id
            codec[i, 3:3 + k] = torch.tensor([0 if c is None else c for c in prefixo])
            inicio = 3 + k + T + 2  # depois do texto, do eos de texto e do codec_bos
            codec[i, inicio - 1] = t.codec_bos_id
            codec[i, inicio:inicio + N] = cod[:, 0]
            codec[i, inicio + N] = t.codec_eos_token_id
            finas[i, inicio:inicio + N] = cod
            rotulos[i, inicio:inicio + N] = cod[:, 0]; rotulos[i, inicio + N] = t.codec_eos_token_id
            fim = inicio + N + 1
            mascara_texto[i, :fim] = True; mascara_codec[i, 3:fim] = True; mascara_codec[i, voz] = False
            quadros[i, inicio:inicio + N] = True; atencao[i, :fim] = 1
        return {'texto': texto, 'codec': codec, 'finas': finas, 'mascara_texto': mascara_texto, 'mascara_codec': mascara_codec,
                'quadros': quadros, 'atencao': atencao, 'rotulos': rotulos, 'voz': voz}

    def perda(self, exemplos):
        torch = self.torch; F = torch.nn.functional
        lote = self._lote(exemplos); voz = lote.pop('voz')
        lote = {c: v.to(self.dispositivo) for c, v in lote.items()}
        talker = self.modelo.talker
        x = talker.text_projection(talker.model.text_embedding(lote['texto'])) * lote['mascara_texto'].unsqueeze(-1)
        c = talker.model.codec_embedding(lote['codec']) * lote['mascara_codec'].unsqueeze(-1)
        c[:, voz, :] = self.voz_embedding.to(c.dtype).to(c.device)
        x = x + c
        for i in range(1, 16):
            x = x + talker.code_predictor.get_input_embeddings()[i - 1](lote['finas'][:, :, i]) * lote['quadros'].unsqueeze(-1)
        saida = talker(inputs_embeds=x[:, :-1], attention_mask=lote['atencao'][:, :-1], output_hidden_states=True)
        # posicao p preve o rotulo de p+1: rotulos ja vem alinhados, sem novo deslocamento
        l0 = F.cross_entropy(saida.logits.float().flatten(0, 1), lote['rotulos'][:, 1:].flatten(), ignore_index=-100)
        # estado da posicao anterior a cada quadro: e o que a inferencia entrega ao sub-preditor
        estado = saida.hidden_states[0][-1][lote['quadros'][:, 1:]]
        codigos = lote['finas'][lote['quadros']]
        logits, _ = talker.forward_sub_talker_finetune(codigos, estado)
        sub = F.cross_entropy(logits.float().flatten(0, 1), codigos[:, 1:].flatten())
        return l0 + PESO_SUB * sub, {'codec0': float(l0.detach()), 'camadas_finas': float(sub.detach())}

    def salvar(self, pasta, voz):
        """Modelo completo de voz propria (pesos ja fundidos), carregavel por do_pacote."""
        from safetensors.torch import save_file
        pasta = Path(pasta)
        shutil.copytree(self.origem, pasta, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.cache', '*.lock'))
        estado = {k: v.detach().to('cpu') for k, v in self.modelo.state_dict().items() if not k.startswith('speaker_encoder')}
        estado = {k: v for k, v in estado.items() if not k.startswith('speech_tokenizer')}
        tabela = estado['talker.model.codec_embedding.weight'].clone()
        tabela[VOZ_ID] = self.voz_embedding.to(tabela.dtype).cpu()
        estado['talker.model.codec_embedding.weight'] = tabela
        save_file({k: v.contiguous() for k, v in estado.items()}, str(pasta / 'model.safetensors'))
        config = json.loads((self.origem / 'config.json').read_text(encoding='utf-8'))
        config['tts_model_type'] = 'custom_voice'
        config['talker_config']['spk_id'] = {voz: VOZ_ID}
        config['talker_config']['spk_is_dialect'] = {voz: False}
        (pasta / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    # --- inferencia ---
    def falar(self, texto, voz=None, referencia=None):
        """Voz propria (modelo ajustado) por nome, ou clonagem a partir de (audio, transcricao) no modelo base."""
        import numpy as np
        idioma = self.idioma or 'auto'
        with self.torch.no_grad():
            if voz:
                ondas, taxa = self.tts.generate_custom_voice(text=texto, speaker=voz, language=idioma)
            else:
                audio, transcricao = referencia
                ondas, taxa = self.tts.generate_voice_clone(text=texto, language=idioma, ref_audio=str(audio), ref_text=transcricao)
        return np.asarray(ondas[0], dtype=np.float32), taxa
