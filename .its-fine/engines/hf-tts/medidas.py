"""Instrumento de medida do hf-tts: inteligibilidade (WER pelo faster-whisper) e semelhanca de voz (verificador de locutor).

Nao sabe qual modelo gerou o audio: recebe onda, texto esperado e a voz-alvo. E a mesma regua
para o modelo base, o ajustado e o pacote exportado, e para a ingestao (transcrever gravacoes).
"""
import re
import unicodedata

TAXA_ASR = 16000


def normalizar(texto):
    """Minusculas, sem acento nem pontuacao: o WER mede palavras, nao grafia."""
    t = unicodedata.normalize('NFKD', texto.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return ' '.join(re.sub(r"[^\w' ]", ' ', t).split())


def wer(referencia, hipotese):
    import jiwer
    ref, hip = normalizar(referencia), normalizar(hipotese)
    if not ref: return 0.0 if not hip else 1.0
    return float(jiwer.wer(ref, hip or '<vazio>'))


def _reamostrar(onda, taxa, destino):
    import librosa
    import numpy as np
    return np.ascontiguousarray(librosa.resample(onda, orig_sr=taxa, target_sr=destino) if taxa != destino else onda, dtype=np.float32)


class Transcritor:
    """faster-whisper (CTranslate2). Sem Metal: no Mac e sem GPU NVIDIA roda na CPU em int8."""

    def __init__(self, modelo_id, dispositivo):
        from faster_whisper import WhisperModel
        cuda = dispositivo == 'cuda'
        self.modelo = WhisperModel(modelo_id, device='cuda' if cuda else 'cpu', compute_type='float16' if cuda else 'int8')

    def _rodar(self, onda, taxa, idioma, palavras):
        trechos, _ = self.modelo.transcribe(_reamostrar(onda, taxa, TAXA_ASR), language=idioma or None, beam_size=5,
                                            word_timestamps=palavras, vad_filter=palavras, condition_on_previous_text=False)
        return list(trechos)

    def transcrever(self, onda, taxa, idioma=None):
        return ' '.join(t.text.strip() for t in self._rodar(onda, taxa, idioma, False)).strip()

    def palavras(self, onda, taxa, idioma=None):
        """Cada palavra com inicio e fim em segundos: e o que corta uma gravacao longa sem desalinhar fala e texto."""
        return [{'inicio': p.start, 'fim': p.end, 'palavra': p.word} for t in self._rodar(onda, taxa, idioma, True) for p in (t.words or [])]


class Verificador:
    """Embedding de locutor (x-vector) e semelhanca por cosseno contra uma voz-alvo."""

    def __init__(self, modelo_id, dispositivo):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch, self.dispositivo = torch, dispositivo
        self.extrator = AutoFeatureExtractor.from_pretrained(modelo_id)
        self.modelo = WavLMForXVector.from_pretrained(modelo_id).to(dispositivo).eval()

    def embedding(self, onda, taxa):
        torch = self.torch
        entrada = self.extrator(_reamostrar(onda, taxa, TAXA_ASR), sampling_rate=TAXA_ASR, return_tensors='pt')
        with torch.no_grad():
            e = self.modelo(**{k: v.to(self.dispositivo) for k, v in entrada.items()}).embeddings[0].float()
        return torch.nn.functional.normalize(e, dim=-1).cpu()

    def voz(self, ondas):
        """Voz-alvo: media normalizada dos embeddings de varios clipes reais."""
        torch = self.torch
        media = torch.stack([self.embedding(o, t) for o, t in ondas]).mean(0)
        return media / media.norm()

    def semelhanca(self, onda, taxa, alvo):
        return float(self.embedding(onda, taxa) @ alvo)


def pontuar(itens, wer_maximo):
    """Metrica do protocolo: media da semelhanca de voz, contando zero na frase ininteligivel (WER acima do maximo).

    Uma frase que nao se entende nao vale pela voz parecida: assim a inteligibilidade trava a nota.
    """
    notas = [i['semelhanca'] if i['wer'] <= wer_maximo else 0.0 for i in itens]
    return {'similaridade_inteligivel': round(sum(notas) / max(1, len(notas)), 4),
            'similaridade_media': round(sum(i['semelhanca'] for i in itens) / max(1, len(itens)), 4),
            'wer_medio': round(sum(i['wer'] for i in itens) / max(1, len(itens)), 4),
            'inteligiveis': sum(1 for i in itens if i['wer'] <= wer_maximo), 'amostra': len(itens)}
