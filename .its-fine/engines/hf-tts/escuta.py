"""Pagina local de escuta cega A/B: para cada frase, duas falas em ordem sorteada, sem dizer qual e qual.

Diagnostico humano de naturalidade e prosodia, que o instrumento automatico nao mede. Nao decide
aprovacao. A pagina fica na pasta da rodada e aponta para os audios pelo caminho relativo; abra no
navegador, escolha a melhor de cada par e so no fim clique em "revelar". Os audios sao copiados para
<pagina>-audios/ com nomes neutros (par e posicao), para o caminho nao entregar qual e qual.
"""
from pathlib import Path
import argparse
import html
import json
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_entrypoint, ler_jsonl  # noqa: E402

PAGINA = '''<!doctype html><meta charset="utf-8"><title>Escuta cega</title>
<style>body{{font:15px system-ui;max-width:760px;margin:2em auto;padding:0 1em}}.par{{border:1px solid #ccc;border-radius:8px;padding:1em;margin:1em 0}}
audio{{width:100%}}label{{display:block;margin:.3em 0}}#placar{{font-weight:bold}}</style>
<h1>Escuta cega</h1><p>Em cada frase, ouça as duas falas e marque a que soa mais natural e mais parecida com a sua voz. No fim, clique em revelar.</p>
{pares}<button onclick="revelar()">Revelar</button><p id="placar"></p>
<script>const chave={chave}, A={nome_a};
function revelar(){{let a=0,b=0;chave.forEach((k,i)=>{{const m=document.querySelector('input[name=p'+i+']:checked');
if(m){{if(k[m.value]===A)a++;else b++;}}document.getElementById('r'+i).textContent='1 = '+k['1']+', 2 = '+k['2'];}});
document.getElementById('placar').textContent={rotulo_a}+': '+a+'  |  '+{rotulo_b}+': '+b;}}</script>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True, help='pasta audios/ da primeira avaliacao (ex.: baseline)')
    parser.add_argument('--b', required=True, help='pasta audios/ da segunda (ex.: candidato)')
    parser.add_argument('--nome-a', default='modelo original')
    parser.add_argument('--nome-b', default='modelo treinado')
    parser.add_argument('--textos', required=True, help='o jsonl das frases avaliadas, na mesma ordem')
    parser.add_argument('--saida', required=True, help='arquivo .html dentro da pasta da rodada')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    saida = Path(args.saida).resolve()
    copias = saida.parent / (saida.stem + '-audios')
    if copias.exists(): raise FileExistsError('OUTPUT_EXISTS')
    copias.mkdir()
    sorteio = random.Random(args.seed)
    nome = {'a': args.nome_a, 'b': args.nome_b}
    pares, chave = [], []
    for i, linha in enumerate(ler_jsonl(args.textos)):
        lados = [('a', Path(args.a) / f'{i:04d}.wav'), ('b', Path(args.b) / f'{i:04d}.wav')]
        if not all(p.is_file() for _, p in lados): continue
        sorteio.shuffle(lados)
        n = len(chave)
        chave.append({'1': nome[lados[0][0]], '2': nome[lados[1][0]]})
        rel = []
        for posicao, (_, origem) in enumerate(lados, 1):
            shutil.copyfile(origem, copias / f'{n:04d}-{posicao}.wav')
            rel.append(html.escape(f'{copias.name}/{n:04d}-{posicao}.wav'))
        pares.append(f'<div class="par"><p>{html.escape(linha["text"])}</p>'
                     f'<label><input type="radio" name="p{n}" value="1"> 1 <audio controls src="{rel[0]}"></audio></label>'
                     f'<label><input type="radio" name="p{n}" value="2"> 2 <audio controls src="{rel[1]}"></audio></label>'
                     f'<small id="r{n}"></small></div>')
    saida.write_text(PAGINA.format(pares='\n'.join(pares), chave=json.dumps(chave, ensure_ascii=False), nome_a=json.dumps(args.nome_a),
                                   rotulo_a=json.dumps(args.nome_a), rotulo_b=json.dumps(args.nome_b)), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(data_entrypoint(main, None))
