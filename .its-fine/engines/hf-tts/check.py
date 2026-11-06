"""Checagem dos dados preparados de voz contra o modelo escolhido, antes da metrica e da receita.

Bloqueia: clipe ilegivel, split vazio depois do limite de duracao, nenhum clipe de referencia
(3 a 15 s) no treino, idioma que a familia nao fala. Avisa: menos de 15 min de voz no treino,
clipes acima do limite (ficam fora do treino). Grava check.json com contagens, o hash do
adaptador da familia (o "template" do modelo de voz) e a revisao fixada.
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_entrypoint, duracao, engine_info, familia_do_modelo, gravar_json, hash_adaptador, ler_jsonl, referencia  # noqa: E402

MINUTOS_MINIMOS = 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', required=True, help='pasta preparada (train/validation/test .jsonl e audio/)')
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--familia', help='obrigatoria para modelo fora do catalogo')
    parser.add_argument('--idioma', required=True)
    parser.add_argument('--saida', required=True, help='<prepared>/check.json')
    args = parser.parse_args()

    info = engine_info()
    maximo = float(info['defaults']['train']['max_audio_seconds'])
    familia = familia_do_modelo(args.modelo, args.familia)
    dados = Path(args.dados).resolve()
    bloqueios, avisos, splits = [], [], {}
    for nome in ('train', 'validation', 'test'):
        linhas = ler_jsonl(dados / (nome + '.jsonl'))
        duracoes, ilegiveis = [], 0
        for l in linhas:
            try: duracoes.append(duracao(dados / l['audio']))
            except Exception: ilegiveis += 1
        dentro = [d for d in duracoes if d <= maximo]
        splits[nome] = {'clipes': len(linhas), 'clipes_validos': len(dentro), 'longos': len(duracoes) - len(dentro),
                        'minutos': round(sum(dentro) / 60, 1), 'duracao_media_s': round(sum(duracoes) / max(1, len(duracoes)), 1),
                        'duracao_max_s': round(max(duracoes or [0]), 1)}
        if ilegiveis: bloqueios.append(f'{nome}: {ilegiveis} clipes ilegiveis')
        if not dentro: bloqueios.append(f'{nome}: nenhum clipe ate {maximo:g} s')
        if len(duracoes) - len(dentro): avisos.append(f'{nome}: {len(duracoes) - len(dentro)} clipes acima de {maximo:g} s ficam fora')
    try: referencia(ler_jsonl(dados / 'train.jsonl'), dados)
    except ValueError: bloqueios.append('treino sem clipe de referencia entre 3 e 15 s')
    if args.idioma not in info['familias'][familia]['idiomas']: bloqueios.append(f'a familia {familia} nao fala o idioma {args.idioma}')
    if splits['train']['minutos'] < MINUTOS_MINIMOS:
        avisos.append(f"so {splits['train']['minutos']} min de voz no treino; o recomendado e de 15 a 60 min")
    gravar_json(args.saida, {'schema_version': 1, 'engine': 'hf-tts', 'familia': familia, 'modelo': args.modelo,
                             'tokenizer_revision': args.revision, 'template_hash': hash_adaptador(familia), 'idioma': args.idioma,
                             'max_audio_seconds': maximo, 'splits': splits, 'bloqueios': bloqueios, 'avisos': avisos,
                             'ok': not bloqueios})
    return 0


if __name__ == '__main__':
    raise SystemExit(data_entrypoint(main, '.'))
