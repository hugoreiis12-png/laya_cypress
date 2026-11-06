"""Avaliacao de voz sob o protocolo congelado: o mesmo instrumento para base, candidato e pacote exportado.

Para cada frase de --teste o modelo gera a fala e o instrumento mede WER (faster-whisper) e semelhanca
com a voz-alvo (clipes reais do treino). Tres modos:
- base: o modelo original clona a voz a partir do clipe de referencia do treino (zero-shot);
- --adapter: o adapter treinado e fundido no modelo base, salvo como modelo de voz propria e
  recarregado do disco (confere o caminho de salvar que a exportacao usa);
- --pacote: o modelo de um pacote exportado, como o usuario vai usar.
Os audios gerados ficam em <saida>/audios/ para a escuta cega (escuta.py). resumo.json so tem numeros.
"""
from pathlib import Path
import argparse
import gc
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (adaptador, carregar_audio, data_entrypoint, engine_info, familia_do_modelo, gravar_json, ler_json,  # noqa: E402
                    ler_jsonl, referencia, sha256_arquivo)

TAXA_REAL = 24000


def materializar(model_id, revision, familia, adapter, destino, dispositivo, dtype):
    """Modelo base + adapter fundidos e salvos como modelo de voz propria em `destino`."""
    import lora as L
    meta = L.meta(adapter)
    Familia = adaptador(familia)
    f = Familia(model_id, dispositivo, dtype, revision=revision, idioma=meta.get('idioma'))
    modulo, _, _ = f.lora()
    L.injetar(modulo, L.configuracao_salva(adapter))
    f.carregar_extras(L.carregar(adapter, modulo))
    L.fundir(modulo)
    f.salvar(destino, meta['voz'])
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teste', required=True, help='validation.jsonl (baseline) ou test.jsonl (avaliacao final)')
    parser.add_argument('--dados', required=True, help='pasta preparada: treino (voz-alvo e referencia) e audio/')
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--familia')
    parser.add_argument('--idioma', required=True)
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument('--adapter', help='pasta adapter/ do treino: avalia o candidato')
    modo.add_argument('--pacote', help='pasta de um pacote exportado')
    parser.add_argument('--saida', required=True)
    parser.add_argument('--calibracao', required=True, help='metric-calibration.json valido')
    parser.add_argument('--calibracao-hash', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from itsfine import device as acelerador
    from itsfine.continuation import adapter_hash
    from medidas import Transcritor, Verificador, pontuar, wer
    from metric import voz_alvo
    if sha256_arquivo(args.calibracao) != args.calibracao_hash: raise ValueError('CALIBRATION_HASH_MISMATCH')
    calibracao = ler_json(args.calibracao)
    if calibracao.get('valid') is not True: raise ValueError('CALIBRATION_INVALID')
    info = engine_info()['defaults']['avaliacao']
    dados, saida = Path(args.dados).resolve(), Path(args.saida).resolve()
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('OUTPUT_EXISTS')
    (saida / 'audios').mkdir(parents=True)
    familia_nome = familia_do_modelo(args.modelo, args.familia)
    Familia = adaptador(familia_nome)
    dispositivo = acelerador.escolher(args.device)
    dtype = acelerador.dtype(dispositivo)
    frases = ler_jsonl(args.teste)
    treino = ler_jsonl(dados / 'train.jsonl')

    if args.adapter:
        meta = materializar(args.modelo, args.revision, familia_nome, Path(args.adapter), saida / 'modelo', dispositivo, dtype)
        gc.collect()
        if dispositivo == 'cuda': torch.cuda.empty_cache()
        gerador, voz, modo_nome = Familia.do_pacote(saida / 'modelo', dispositivo, dtype, idioma=meta.get('idioma')), meta['voz'], 'candidato'
    elif args.pacote:
        manifesto = ler_json(Path(args.pacote) / 'itsfine-voz.json')
        gerador = Familia.do_pacote(Path(args.pacote) / 'modelo', dispositivo, dtype, idioma=manifesto.get('idioma'))
        voz, modo_nome = manifesto['voz'], 'pacote'
    else:
        gerador, voz, modo_nome = Familia(args.modelo, dispositivo, dtype, revision=args.revision, idioma=args.idioma), None, 'base'
    ref = referencia(treino, dados)
    gerados = []
    for i, frase in enumerate(frases):
        torch.manual_seed(i)  # a geracao amostra: semente por frase deixa a nota reproduzivel
        onda, taxa = gerador.falar(frase['text'], voz=voz, referencia=None if voz else (dados / ref['audio'], ref['text']))
        caminho = saida / 'audios' / f'{i:04d}.wav'
        sf.write(str(caminho), onda, taxa)
        gerados.append((caminho, frase['text']))
    del gerador; gc.collect()
    if dispositivo == 'cuda': torch.cuda.empty_cache()

    verificador = Verificador(info['verificador'], dispositivo)
    asr = Transcritor(info['asr'], dispositivo)
    alvo = voz_alvo(verificador, dados, treino)
    itens = []
    for caminho, texto in gerados:
        onda = carregar_audio(caminho, TAXA_REAL)
        itens.append({'wer': round(wer(texto, asr.transcrever(onda, TAXA_REAL, args.idioma)), 4),
                      'semelhanca': round(verificador.semelhanca(onda, TAXA_REAL, alvo), 4),
                      'duracao_s': round(len(onda) / TAXA_REAL, 2)})
    gravar_json(saida / 'itens.json', itens)
    resumo = {'schema_version': 1, 'engine': 'hf-tts', 'modo': modo_nome, 'familia': familia_nome, 'modelo': args.modelo,
              'revision': args.revision, 'metric': calibracao['metric'], 'test_hash': sha256_arquivo(args.teste),
              'calibracao_hash': args.calibracao_hash, 'wer_maximo_por_frase': calibracao['wer_maximo_por_frase'],
              **pontuar(itens, calibracao['wer_maximo_por_frase'])}
    if args.adapter: resumo['adapter_hash'] = adapter_hash(args.adapter)
    gravar_json(saida / 'resumo.json', resumo)
    return 0


if __name__ == '__main__':
    raise SystemExit(data_entrypoint(main, 'resumo.json'))
