"""Consumidor local: um estado ou um JSONL de estados -> decisoes tipadas do pacote."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, gravar_jsonl, ler_jsonl  # noqa: E402
import infer  # noqa: E402


def interpretar_estado(texto):
    texto = texto.strip()
    if texto[:1] in ('{', '['):
        try: return json.loads(texto)
        except ValueError: pass
    return texto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacote', required=True)
    parser.add_argument('--estado', help='texto ou JSON de um estado')
    parser.add_argument('--entrada', help='JSONL com linhas {"id", "state"}')
    parser.add_argument('--saida', help='arquivo .json (um estado) ou .jsonl (varios); sem --saida imprime')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--sem-laya', action='store_true', help='falha se o pacote laya estiver importado neste processo')
    args = parser.parse_args()
    if bool(args.estado) == bool(args.entrada): raise ValueError('STATE_OR_INPUT_REQUIRED')
    ctx = infer.preparar(args.pacote, args.device)
    package_id = ctx['carga']['identidade']['package_id']
    sem_laya = 'laya' not in sys.modules and not any(m == 'laya' or m.startswith('laya.') for m in sys.modules)
    if args.sem_laya and not sem_laya: raise RuntimeError('LAYA_PACKAGE_IMPORTED')
    if args.estado:
        regs = infer.registros(ctx, [{'state': interpretar_estado(args.estado)}], 1, com_rotulo=False)[0]
        resultado = {**infer.respostas(ctx, regs), 'package_id': package_id, 'sem_laya': sem_laya}
        if args.saida: gravar_json(args.saida, resultado)
        else: print(json.dumps(resultado, ensure_ascii=False, indent=2))
        return 0
    linhas = ler_jsonl(args.entrada)
    regs = infer.registros(ctx, linhas, args.batch, com_rotulo=False)
    resultados = [{'id': l.get('id'), **infer.respostas(ctx, r), 'package_id': package_id} for l, r in zip(linhas, regs)]
    if args.saida: gravar_jsonl(args.saida, resultados)
    else:
        for r in resultados: print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
