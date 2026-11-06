"""Ingestao de voz: gravacoes do usuario viram clipes curtos com transcricao, em linhas {id, group_id, audio, text}.

Duas formas de entrada, misturaveis na mesma pasta, em audio ou video (mp4, mov, mkv, webm, avi: so a
faixa de audio e usada, extraida pelo FFmpeg embutido no PyAV, sem ffmpeg instalado no sistema):
- clipe pronto: audio com um .txt de mesmo nome ao lado (a transcricao). O faster-whisper confere: se o
  que ele ouve diverge demais do .txt, o clipe sai como transcricao_ruim;
- gravacao longa sem .txt: o faster-whisper transcreve com marca de tempo por palavra e o audio
  e cortado entre palavras, de preferencia numa pausa, em clipes de 2 a 15 s; o texto do clipe e
  exatamente o das palavras dentro dele. Marca por frase nao serve: medida numa gravacao de teste,
  errava o fim das frases e desalinhava fala e texto.
Cada gravacao e um grupo (os clipes de uma gravacao ficam no mesmo split); gravacao longa vira
grupos de blocos de 5 min, para que validacao e teste venham de trechos diferentes. Saida em
<saida>/linhas.jsonl, <saida>/clipes/ (wav mono 24 kHz) e resumo.json so com contagens.
"""
from pathlib import Path
import argparse
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import VIA_FFMPEG, carregar_audio, data_entrypoint, engine_info, gravar_json, gravar_jsonl  # noqa: E402

EXTENSOES = ('.wav', '.flac', '.ogg', '.mp3') + VIA_FFMPEG
TAXA = 24000
MINIMO, MAXIMO, ALVO = 2.0, 15.0, 10.0
BLOCO_S = 300
LIMITE_TRANSCRICAO = 0.35  # WER entre o .txt e o que o faster-whisper ouve acima disso: transcricao nao confere


def fontes(caminhos):
    arquivos = []
    for c in caminhos:
        c = Path(c)
        arquivos += sorted(p for p in (c.rglob('*') if c.is_dir() else [c]) if p.suffix.lower() in EXTENSOES and p.is_file())
    return arquivos


def ler(arquivo):
    """Mono float32 em TAXA; de video, a primeira faixa de audio (carregar_audio cuida do formato)."""
    return carregar_audio(arquivo, TAXA)


PAUSA = 0.3       # silencio entre palavras que serve de corte
PAUSA_LONGA = 1.0  # sempre corta: troca de frase ou de assunto
FOLGA = 0.15       # respiro antes da primeira e depois da ultima palavra, sem invadir a vizinha


def cortar_por_palavras(palavras, total):
    """Clipes {inicio, fim, texto} cortados entre palavras: numa pausa depois de ALVO/2 s de fala, sempre
    numa pausa longa, e entre quaisquer duas palavras se passaria de MAXIMO s sem pausa."""
    grupos, atual = [], []
    for p in palavras:
        if atual:
            pausa, dur = p['inicio'] - atual[-1]['fim'], atual[-1]['fim'] - atual[0]['inicio']
            if pausa >= PAUSA_LONGA or (pausa >= PAUSA and dur >= ALVO / 2) or p['fim'] - atual[0]['inicio'] > MAXIMO:
                grupos.append(atual); atual = []
        atual.append(p)
    if atual: grupos.append(atual)
    clipes = []
    for i, g in enumerate(grupos):
        antes = grupos[i - 1][-1]['fim'] if i else 0.0
        depois = grupos[i + 1][0]['inicio'] if i + 1 < len(grupos) else total
        inicio = max(antes + (g[0]['inicio'] - antes) / 2, g[0]['inicio'] - FOLGA)
        fim = min(g[-1]['fim'] + (depois - g[-1]['fim']) / 2, g[-1]['fim'] + FOLGA)
        clipes.append({'inicio': inicio, 'fim': fim, 'texto': ''.join(p['palavra'] for p in g).strip()})
    return clipes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fonte', action='append', required=True, help='pasta ou arquivo de audio ou video; repita para varias')
    parser.add_argument('--saida', required=True, help='pasta nova do lote, ex.: _itsfine_runs/<run>/dataset')
    parser.add_argument('--idioma', required=True, help='codigo do idioma falado, ex.: pt')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()

    import soundfile as sf
    from itsfine import device as acelerador
    from medidas import Transcritor, wer
    saida = Path(args.saida).resolve()
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('OUTPUT_EXISTS')
    (saida / 'clipes').mkdir(parents=True, exist_ok=True)
    dispositivo = acelerador.escolher(args.device)
    asr = Transcritor(engine_info()['defaults']['avaliacao']['asr'], dispositivo)

    linhas, grupos = [], set()
    descartes = {'curtos': 0, 'longos': 0, 'sem_fala': 0, 'transcricao_ruim': 0}
    arquivos = fontes(args.fonte); falhas = sem_audio = 0
    for arquivo in arquivos:
        try:
            onda = ler(arquivo)
        except Exception as erro:
            # video sem faixa de audio e motivo proprio, para o usuario saber qual arquivo trocar
            if str(erro) in ('NO_AUDIO_TRACK', 'EMPTY_AUDIO_TRACK'): sem_audio += 1
            else: falhas += 1
            continue
        origem = hashlib.sha256(arquivo.read_bytes()).hexdigest()[:16]
        transcricao = arquivo.with_suffix('.txt')
        if transcricao.is_file():
            texto = transcricao.read_text(encoding='utf-8').strip()
            pedacos = [{'inicio': 0.0, 'fim': len(onda) / TAXA, 'texto': texto, 'conferir': True}]
        else:
            pedacos = cortar_por_palavras(asr.palavras(onda, TAXA, args.idioma), len(onda) / TAXA)
        for n, p in enumerate(pedacos):
            trecho = onda[int(p['inicio'] * TAXA):int(p['fim'] * TAXA)]
            d = len(trecho) / TAXA
            if d < MINIMO: descartes['curtos'] += 1; continue
            if d > (20.0 if p.get('conferir') else MAXIMO): descartes['longos'] += 1; continue
            if p.get('conferir') and wer(p['texto'], asr.transcrever(trecho, TAXA, args.idioma)) > LIMITE_TRANSCRICAO:
                descartes['transcricao_ruim'] += 1; continue
            if len(p['texto'].split()) < 2: descartes['sem_fala'] += 1; continue
            ident = f'{origem}-{n:04d}'
            grupo = origem if p.get('conferir') else f'{origem}#{int(p["inicio"] // BLOCO_S)}'
            rel = f'clipes/{ident}.wav'
            sf.write(str(saida / rel), trecho, TAXA, subtype='PCM_16')
            linhas.append({'id': ident, 'group_id': grupo, 'audio': rel, 'text': p['texto'], 'duracao': round(d, 3), 'idioma': args.idioma})
            grupos.add(grupo)
    gravar_jsonl(saida / 'linhas.jsonl', linhas)
    minutos = round(sum(l['duracao'] for l in linhas) / 60, 1)
    resumo = {'schema_version': 1, 'arquivos': len(arquivos), 'videos': sum(a.suffix.lower() in VIA_FFMPEG for a in arquivos),
              'sem_audio': sem_audio, 'falhas': falhas, 'clipes': len(linhas), 'grupos': len(grupos),
              'minutos': minutos, 'descartes': descartes, 'ok': bool(linhas) and len(grupos) >= 3 and falhas == 0}
    gravar_json(saida / 'resumo.json', resumo)
    return 0


if __name__ == '__main__':
    raise SystemExit(data_entrypoint(main, 'resumo.json'))
