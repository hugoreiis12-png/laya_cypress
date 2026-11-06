"""Calibracao do instrumento de voz antes do baseline, so com audio real do usuario (nenhum modelo de voz).

Casos de resultado conhecido, na validacao:
- mesma voz: clipes reais contra a voz-alvo (media dos embeddings de clipes do treino): semelhanca alta, WER baixo;
- outra voz: os mesmos clipes com o tom deslocado 4 semitons (soa como outra pessoa): semelhanca tem de cair;
- texto trocado: cada clipe medido contra o texto de outro: WER tem de subir.
Valido quando a regua separa os casos: mesma voz acima da outra voz em pelo menos 95% dos pares,
margem media de pelo menos 0,05, WER real dentro do maximo por frase e texto trocado acima de 0,7.
Sem isso, nem baseline nem aprovacao valem: o numero nao distingue a voz do usuario de outra.
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import carregar_audio, data_entrypoint, engine_info, gravar_json, ler_jsonl  # noqa: E402

TAXA = 24000
SEMITONS = 4
CLIPES_DA_VOZ = 20
MAXIMO_DE_CASOS = 40


def voz_alvo(verificador, dados, linhas):
    """Media dos embeddings de ate 20 clipes do treino, ordem deterministica por id."""
    escolhidos = sorted(linhas, key=lambda l: str(l['id']))[:CLIPES_DA_VOZ]
    return verificador.voz([(carregar_audio(dados / l['audio'], TAXA), TAXA) for l in escolhidos])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', required=True, help='pasta preparada (train, validation e audio/)')
    parser.add_argument('--idioma', required=True)
    parser.add_argument('--saida', required=True, help='<run>/metric-calibration.json')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()

    import librosa
    from itsfine import device as acelerador
    from medidas import Transcritor, Verificador, wer
    info = engine_info()['defaults']['avaliacao']
    dados = Path(args.dados).resolve()
    dispositivo = acelerador.escolher(args.device)
    verificador = Verificador(info['verificador'], dispositivo)
    asr = Transcritor(info['asr'], dispositivo)
    alvo = voz_alvo(verificador, dados, ler_jsonl(dados / 'train.jsonl'))
    casos = sorted(ler_jsonl(dados / 'validation.jsonl'), key=lambda l: str(l['id']))[:MAXIMO_DE_CASOS]
    if len(casos) < 3: raise ValueError('CALIBRATION_NEEDS_3_VALIDATION_CLIPS')

    mesma, outra, wer_real, wer_trocado = [], [], [], []
    for i, l in enumerate(casos):
        onda = carregar_audio(dados / l['audio'], TAXA)
        mesma.append(verificador.semelhanca(onda, TAXA, alvo))
        deslocada = librosa.effects.pitch_shift(onda, sr=TAXA, n_steps=SEMITONS if i % 2 == 0 else -SEMITONS)
        outra.append(verificador.semelhanca(deslocada, TAXA, alvo))
        ouvido = asr.transcrever(onda, TAXA, args.idioma)
        wer_real.append(wer(l['text'], ouvido))
        wer_trocado.append(wer(casos[(i + 1) % len(casos)]['text'], ouvido))

    separacao = sum(1 for a in mesma for b in outra if a > b) / (len(mesma) * len(outra))
    margem = sum(mesma) / len(mesma) - sum(outra) / len(outra)
    media = lambda v: round(sum(v) / len(v), 4)
    checagens = {'separa_vozes': separacao >= 0.95, 'margem_minima': margem >= 0.05,
                 'fala_real_inteligivel': media(wer_real) <= info['wer_maximo_por_frase'],
                 'texto_trocado_detectado': media(wer_trocado) >= 0.7}
    relatorio = {
        'schema_version': 1, 'engine': 'hf-tts', 'metric': 'similaridade_inteligivel', 'direction': 'maximize',
        'definicao': 'media da semelhanca de voz por frase, contando zero quando o WER da frase passa do maximo',
        'avaliadores': {'asr': info['asr'], 'verificador': info['verificador']}, 'wer_maximo_por_frase': info['wer_maximo_por_frase'],
        'amostra': len(casos),
        'mesma_voz': {'similaridade_media': media(mesma), 'similaridade_min': round(min(mesma), 4), 'wer_medio': media(wer_real)},
        'outra_voz': {'similaridade_media': media(outra), 'similaridade_max': round(max(outra), 4)},
        'texto_trocado': {'wer_medio': media(wer_trocado)},
        'separacao': round(separacao, 4), 'margem': round(margem, 4),
        'limiar_sugerido': round((sum(mesma) / len(mesma) + sum(outra) / len(outra)) / 2, 4),
        'checagens': checagens, 'valid': all(checagens.values()),
        'limite': 'mede semelhanca de timbre e inteligibilidade; naturalidade e prosodia pedem a escuta cega (escuta.py)'}
    gravar_json(args.saida, relatorio)
    return 0


if __name__ == '__main__':
    raise SystemExit(data_entrypoint(main, '.'))
