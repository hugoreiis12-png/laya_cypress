"""Ingestao: arquivos do usuario viram dataset de treino.

PDF, DOCX, Markdown, TXT e HTML entram; sai JSONL com `messages` (modo qa)
ou `text` (modo text), com `id` e `group_id` preenchidos. No modo qa, um
gerador local (Ollama) ou por API compativel com OpenAI escreve pares
pergunta/resposta com citacao de evidencia obrigatoria, e um filtro
deterministico descarta o que nao serve para treino. Com --anonimizar, dados
pessoais saem de cada trecho antes do gerador (anonymize.py) e o que saiu vai
para trocas.jsonl, que o usuario revisa localmente.

Escreve de forma incremental: interromper e religar continua de onde parou.
"""
from pathlib import Path
import argparse
import html.parser
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
import zipfile
import sqlite3
import hashlib
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gravar_json, ler_jsonl, sha256_arquivo, politica_privacidade  # noqa: E402
from anonymize import Nomes, REGRAS, anonimizar, contagem  # noqa: E402
from itsfine.privacy import require_policy, require_local_executor, require_destination, loopback, request_json
from itsfine.storage import digest, lock

MIN_CHARS = 600
MAX_CHARS = 4500
SIMILARIDADE_MAXIMA = 0.70

PERGUNTA_DEPENDENTE = re.compile(
    r'\b(no|do|deste|desse|neste|nesse)\s+(trecho|texto|excerto|documento|capitulo|capítulo)\b'
    r'|\b(texto|trecho)\s+acima\b|\b(conforme|segundo)\s+o\s+(texto|trecho)\b'
    r'|\bde\s+acordo\s+com\s+o\s+(texto|trecho)\b|\bmencionad[oa]s?\s+(no|na|pelo|pela)\b'
    r'|\bcitad[oa]s?\s+(no|na)\s+(texto|trecho)\b|\bFigura\s+\d+|\b(figura|tabela)\s+(acima|citada|mencionada)\b'
    r'|\bin\s+the\s+(text|passage|excerpt)\b|\baccording\s+to\s+the\s+(text|passage)\b', re.IGNORECASE)
RESPOSTA_ANAFORICA = re.compile(
    r'^\s*(isso|isto|aquilo|ela|ele|elas|eles|esse|essa|este|esta|esses|essas|nesse|nessa|neste|nesta|'
    r'nele|nela|dele|dela|tal|tais|assim|it|this|that|they|he|she)\b', re.IGNORECASE)


# ---------------------------------------------------------------- extracao

class _Html(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.partes, self._pular = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._pular += 1
        if tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'h4', 'tr'):
            self.partes.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self._pular:
            self._pular -= 1

    def handle_data(self, dado):
        if not self._pular:
            self.partes.append(dado)


def texto_docx(caminho):
    with zipfile.ZipFile(caminho) as z:
        if z.getinfo('word/document.xml').file_size>64*1024*1024: raise ValueError('DOCUMENT_TOO_LARGE')
        xml = z.read('word/document.xml').decode('utf-8', 'ignore')
    paragrafos = re.findall(r'<w:p[ >].*?</w:p>', xml, flags=re.S)
    linhas = []
    for p in paragrafos:
        linhas.append(''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, flags=re.S)))
    return html.unescape('\n'.join(linhas))


def _remover_legendas(texto, maximo=3):
    """Remove legenda de figura ou tabela.

    A legenda comeca em "Figura N:" e segue nas linhas seguintes enquanto a
    linha anterior nao termina em ponto, ate `maximo` linhas. Um limite fixo
    engoliria o paragrafo seguinte, porque o PDF separa paragrafo com uma
    quebra so.
    """
    linhas = texto.split('\n')
    saida, i = [], 0
    while i < len(linhas):
        if re.match(r'^\s*(Figura|Figure|Tabela|Table)\s+\d+(\.\d+)?[:.]', linhas[i]):
            consumidas = 1
            while (consumidas < maximo and i + consumidas < len(linhas)
                   and not linhas[i + consumidas - 1].rstrip().endswith(('.', '!', '?'))
                   and linhas[i + consumidas].strip()):
                consumidas += 1
            i += consumidas
            continue
        saida.append(linhas[i])
        i += 1
    return '\n'.join(saida)


def limpar(texto, cabecalhos=()):
    t = unicodedata.normalize('NFKC', texto)
    for c in cabecalhos:
        t = t.replace(c, '')
    t = re.sub(r'(?im)^\s*CAP[ÍI]TULO\s+\d+\.?.*$', '', t)
    t = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', t)
    t = _remover_legendas(t)
    t = re.sub(r'(\w)-\n(\w)', r'\1\2', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def cabecalho_recorrente(paginas):
    """Linha que abre a maioria das paginas e cabecalho corrente."""
    primeiras = {}
    for p in paginas:
        linha = p.strip().split('\n', 1)[0].strip()
        if 8 < len(linha) < 120:
            primeiras[linha] = primeiras.get(linha, 0) + 1
    return [l for l, n in primeiras.items() if n >= max(3, len(paginas) // 3)]


def fatiar(texto, alvo=1800):
    """Bound chunks even for a single long paragraph; preserve short documents."""
    if alvo < 1: raise ValueError('INVALID_CHUNK_SIZE')
    texto = texto.strip()
    pedacos, start = [], 0
    while len(texto)-start > alvo:
        end=start+alvo
        cut=texto.rfind('\n',start,end+1)
        if cut<start+alvo//2: cut=texto.rfind(' ',start,end+1)
        if cut<start+alvo//2: cut=end
        pedacos.append(texto[start:cut].strip())
        start=cut
        while start<len(texto) and texto[start].isspace(): start+=1
    if start<len(texto): pedacos.append(texto[start:])
    return pedacos


def trechos_pdf(caminho, agrupamento):
    """Stream one PDF page at a time; never materialize the book in RAM."""
    import fitz
    with fitz.open(str(caminho)) as doc:
        # Bounded sample for header detection; headings only, no full page list.
        cabecalhos = cabecalho_recorrente([doc[i].get_text() for i in range(min(12,doc.page_count))])
        caps=sorted((p-1,t) for level,t,p in doc.get_toc() if level==1 and 1<=p<=doc.page_count)
        cap=-1
        for page in range(doc.page_count):
            while cap+1<len(caps) and caps[cap+1][0]<=page: cap+=1
            chapter=agrupamento in ('auto','capitulo') and cap>=0
            group='cap%04d'%cap if chapter else 'arquivo'
            title=caps[cap][1] if chapter else 'documento'
            for piece in fatiar(limpar(doc[page].get_text(),cabecalhos),MAX_CHARS):
                yield {'grupo':group,'secao':title,'texto':piece}


def trechos_texto(caminho, texto):
    secoes = re.split(r'(?m)^(#{1,3}\s+.*)$', texto)
    if len(secoes) > 1:
        blocos = [('abertura', secoes[0])] + [(secoes[i].lstrip('# ').strip(), secoes[i + 1]) for i in range(1, len(secoes) - 1, 2)]
    else:
        blocos = [('corpo', texto)]
    saida = []
    for titulo, corpo in blocos:
        for pedaco in fatiar(limpar(corpo)):
            saida.append({'grupo': caminho.stem[:40], 'secao': titulo[:80], 'texto': pedaco})
    return saida


def extrair(caminho, agrupamento):
    sufixo = caminho.suffix.lower()
    if sufixo!='.pdf' and caminho.stat().st_size>64*1024*1024:
        raise ValueError('DOCUMENT_TOO_LARGE: split into files below 64 MiB')
    if sufixo == '.pdf':
        return trechos_pdf(caminho, agrupamento)
    if sufixo == '.docx':
        return trechos_texto(caminho, texto_docx(caminho))
    if sufixo in ('.html', '.htm'):
        p = _Html()
        p.feed(caminho.read_text(encoding='utf-8', errors='ignore'))
        return trechos_texto(caminho, ''.join(p.partes))
    if sufixo in ('.md', '.txt', '.markdown', '.rst'):
        return trechos_texto(caminho, caminho.read_text(encoding='utf-8', errors='ignore'))
    return []


# ---------------------------------------------------------------- geracao

def montar_prompt(trecho, n, idioma):
    return (
        'A partir do trecho abaixo, escreva %d pares pergunta/resposta em %s para treinar um assistente.\n\n'
        'REGRAS OBRIGATORIAS:\n'
        '1. A resposta tem que estar contida no trecho. Nao use conhecimento externo.\n'
        '2. A pergunta tem que ser autossuficiente: quem for responder nao vera o trecho. '
        'Nomeie o conceito, a entidade ou o comando. Nunca escreva "segundo o trecho", "no texto acima" nem pronome solto.\n'
        '3. A resposta deve copiar literalmente uma afirmacao completa do trecho, sem alterar seu sentido.\n'
        '4. Cada par traz "evidencia": uma citacao literal e curta do trecho que sustenta a resposta.\n'
        '5. Varie a forma das perguntas e cubra partes diferentes do trecho.\n\n'
        'Responda SO com JSON valido: {"pares":[{"pergunta":"...","resposta":"...","evidencia":"..."}]}\n\n'
        'TRECHO (secao "%s"):\n%s' % (n, idioma, trecho['secao'], trecho['texto']))


def chamar_ollama(base, modelo, prompt, seed):
    return request_json(base.rstrip('/') + '/api/generate', {'model': modelo, 'prompt': prompt,
        'stream': False, 'think': False, 'options': {'temperature': 0.4, 'num_ctx': 8192, 'seed': seed}})['response']


def chamar_openai(base, modelo, prompt, chave):
    return request_json(base.rstrip('/') + '/chat/completions', {'model': modelo, 'temperature': 0.4,
        'messages': [{'role': 'user', 'content': prompt}]}, {'Authorization': 'Bearer ' + chave})['choices'][0]['message']['content']


def extrair_json(bruto):
    limpo = re.sub(r'^```(?:json)?|```$', '', bruto.strip(), flags=re.M).strip()
    ini, fim = limpo.find('{'), limpo.rfind('}')
    if ini < 0 or fim < 0:
        raise ValueError('SEM_JSON')
    return json.loads(limpo[ini:fim + 1])


# ---------------------------------------------------------------- filtro

def _norm(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s)).strip().lower()


def _tokens(s):
    return {w for w in re.sub(r'[^\w\s]', ' ', _norm(s)).split() if len(w) > 3}


def motivo_descarte(par, trecho, vistas):
    if not isinstance(par, dict) or any(not isinstance(par.get(k), str) for k in ('pergunta','resposta','evidencia')):
        return 'formato_invalido'
    pergunta, resposta, evidencia = par.get('pergunta', ''), par.get('resposta', ''), par.get('evidencia', '')
    if len(pergunta) < 12 or len(resposta) < 20:
        return 'curto_demais'
    if PERGUNTA_DEPENDENTE.search(pergunta):
        return 'pergunta_dependente'
    if RESPOSTA_ANAFORICA.match(resposta):
        return 'resposta_anaforica'
    if not evidencia or len(_norm(evidencia)) < 15 or _norm(evidencia) not in _norm(trecho):
        return 'evidencia_ausente'
    if _norm(resposta) not in _norm(trecho):
        return 'resposta_sem_suporte_literal'
    atual = _tokens(pergunta)
    for anterior in vistas:
        uniao = atual | anterior
        if uniao and len(atual & anterior) / len(uniao) > SIMILARIDADE_MAXIMA:
            return 'pergunta_repetida'
    vistas.append(atual)
    return None


# ---------------------------------------------------------------- principal

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fontes', nargs='+', required=True, help='arquivos ou pastas')
    parser.add_argument('--saida', required=True, help='pasta de saida, dentro da rodada')
    parser.add_argument('--modo', choices=['qa', 'text'], default='qa')
    parser.add_argument('--agrupamento', choices=['auto', 'capitulo', 'arquivo'], default='auto')
    parser.add_argument('--gerador', choices=['ollama', 'openai', 'none'], default='ollama')
    parser.add_argument('--modelo', default='qwen3.5:9b')
    parser.add_argument('--api-base', default='http://localhost:11434')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY', help='nome da variavel de ambiente com a chave; nunca a chave')
    parser.add_argument('--pares', type=int, default=12)
    parser.add_argument('--idioma', default='português')
    parser.add_argument('--system', default='', help='system prompt gravado em cada exemplo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mostrar-conteudo', action='store_true',
                        help='imprime titulos de secao e grava prompt de exemplo; sem isso a saida so tem ids e contagens')
    parser.add_argument('--anonimizar', action='store_true',
                        help='troca CPF, CNPJ, e-mail, telefone e CEP por marcadores antes do gerador')
    parser.add_argument('--nomes', help='pasta do modelo de nomes (hf-encoder/download.py); troca nomes de pessoas e liga --anonimizar')
    parser.add_argument('--policy', help='caminho exato da política manual do executor')
    args = parser.parse_args()

    try:
        policy = politica_privacidade(args.policy or args.saida)
        require_local_executor(policy)
        if args.mostrar_conteudo and not policy['allow_data_to_harness']:
            raise PermissionError('DATA_TO_HARNESS_NOT_AUTHORIZED')
        if args.modo == 'qa':
            if args.gerador == 'none': raise ValueError('QA_REQUIRES_GENERATOR')
            if args.gerador == 'openai' or not loopback(args.api_base):
                require_destination(policy, 'generate', args.api_base)
            if args.gerador == 'openai' and not os.environ.get(args.api_key_env):
                raise ValueError('API_KEY_ENV_MISSING')
        return executar_lote(args, policy)
    except Exception as exc:
        # Fixed error codes only; never print document contents or library exceptions.
        code=str(exc).split(':')[0]
        if not re.fullmatch(r'[A-Z_]+',code): code='INGEST_FAILED_'+type(exc).__name__
        print(json.dumps({'erro': code}), file=sys.stderr)
        return 3


def fontes(paths):
    supported = {'.pdf', '.docx', '.md', '.txt', '.html', '.htm', '.markdown', '.rst'}
    for source in paths:
        p = Path(source).resolve()
        if not p.exists(): raise ValueError('SOURCE_NOT_FOUND')
        if p.is_dir():
            for folder, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not (Path(folder)/d).is_symlink())
                for name in sorted(files):
                    item = Path(folder) / name
                    if item.suffix.lower() in supported and not item.is_symlink(): yield item
        else:
            if p.suffix.lower() not in supported: raise ValueError('UNSUPPORTED_SOURCE')
            yield p


def executar_lote(args, policy):
    saida = Path(args.saida).resolve()
    for source in args.fontes:
        if saida == Path(source).resolve() or saida.is_relative_to(Path(source).resolve()):
            raise ValueError('OUTPUT_INSIDE_SOURCE')
    saida.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ('modo','agrupamento','gerador','modelo','api_base','pares','idioma','system','seed')}
    config.update(version=2, filter='literal-support-v1')
    nomes = Nomes(args.nomes) if args.nomes else None
    anonimiza = args.anonimizar or nomes is not None
    if anonimiza:
        # So entra na configuracao quando ligada: lote antigo sem anonimizacao retoma igual.
        config['anonimizacao'] = {'regras': REGRAS, 'nomes': {'id': nomes.id, 'revision': nomes.revision} if nomes else None}
    config_hash = digest(config)
    with lock(saida / 'ingest.lock'):
        # Legacy append-only output has no trustworthy journal; preserve it.
        if not (saida/'journal.sqlite').exists() and any((saida/n).exists() for n in ('pares.jsonl','trechos.jsonl')):
            raise ValueError('LEGACY_OUTPUT_REQUIRES_NEW_BATCH')
        db = sqlite3.connect(saida/'journal.sqlite')
        try:
            db.executescript('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);'
                'CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, hash TEXT);'
                'CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, done INTEGER, discarded TEXT);'
                'CREATE TABLE IF NOT EXISTS rows (id TEXT PRIMARY KEY, group_id TEXT, payload TEXT);'
                'CREATE TABLE IF NOT EXISTS trocas (chunk TEXT, tipo TEXT, original TEXT);')
            old = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
            if old and old[0] != config_hash: raise ValueError('BATCH_CONFIG_CHANGED: use a new output directory')
            db.execute('CREATE TEMP TABLE inventory (path TEXT PRIMARY KEY, hash TEXT)')
            for source in fontes(args.fontes):
                db.execute('INSERT OR IGNORE INTO inventory VALUES (?,?)', (str(source), sha256_arquivo(source)))
            if old:
                changed = db.execute('SELECT * FROM inventory EXCEPT SELECT * FROM sources').fetchone()
                removed = db.execute('SELECT * FROM sources EXCEPT SELECT * FROM inventory').fetchone()
                if changed or removed: raise ValueError('BATCH_SOURCES_CHANGED: use a new output directory')
            else:
                db.execute('INSERT INTO sources SELECT * FROM inventory')
                db.execute('INSERT INTO meta VALUES (?,?)', ('config',config_hash))
                db.commit()
            failed, empty, documents = 0, 0, 0
            for path, source_hash in db.execute('SELECT path,hash FROM sources ORDER BY path'):
                documents += 1
                source = Path(path)
                chunks = extrair(source, args.agrupamento)
                chunk_count=0
                for ordinal, chunk in enumerate(chunks):
                    chunk_count+=1
                    group = source_hash + '-' + hashlib.sha256(chunk['grupo'].encode()).hexdigest()[:12]
                    ident = digest([source_hash, ordinal, chunk['texto'], config_hash])
                    if db.execute('SELECT 1 FROM chunks WHERE id=? AND done=1',(ident,)).fetchone(): continue
                    rows, discarded, trocas = [], Counter(), []
                    if anonimiza:
                        # O id vem do texto original: rodar de novo o modelo de nomes nao duplica linhas.
                        texto, trocas = anonimizar(chunk['texto'], nomes)
                        secao, t = anonimizar(chunk['secao'], nomes)
                        chunk, trocas = {**chunk, 'texto': texto, 'secao': secao}, trocas + t
                    if args.modo == 'text': rows = [{'id':ident, 'group_id':group, 'text':chunk['texto']}]
                    else:
                        prompt = montar_prompt(chunk, args.pares, args.idioma)
                        try:
                            # Recheck the user policy before each network operation.
                            latest = politica_privacidade(args.policy or args.saida)
                            require_local_executor(latest)
                            if args.gerador == 'openai' or not loopback(args.api_base):
                                require_destination(latest, 'generate', args.api_base)
                            raw = (chamar_ollama(args.api_base,args.modelo,prompt,args.seed) if args.gerador=='ollama'
                                   else chamar_openai(args.api_base,args.modelo,prompt,os.environ[args.api_key_env]))
                            pairs = extrair_json(raw).get('pares')
                            if not isinstance(pairs,list): raise ValueError('INVALID_PAIRS')
                        except PermissionError: raise
                        except Exception:
                            failed += 1
                            continue  # no completed marker; retry only this chunk
                        seen = []
                        for k, pair in enumerate(pairs):
                            reason = motivo_descarte(pair,chunk['texto'],seen)
                            if reason: discarded[reason] += 1; continue
                            messages = ([{'role':'system','content':args.system}] if args.system else []) + [
                                {'role':'user','content':pair['pergunta'].strip()},
                                {'role':'assistant','content':pair['resposta'].strip()}]
                            rows.append({'id':ident+'-q'+str(k), 'group_id':group, 'messages':messages,
                                         'evidencia':pair['evidencia'], 'source_hash':source_hash})
                    with db:
                        for row in rows:
                            db.execute('INSERT INTO rows VALUES (?,?,?)',(row['id'],group,json.dumps(row,ensure_ascii=False)))
                        db.executemany('INSERT INTO trocas VALUES (?,?,?)',[(ident,tipo,original) for tipo,original in trocas])
                        db.execute('INSERT OR REPLACE INTO chunks VALUES (?,?,?)',(ident,1,json.dumps(discarded)))
                if not chunk_count: empty+=1
                print(json.dumps({'documentos_processados':documents}), flush=True)
            destination = saida / ('pares.jsonl' if args.modo=='qa' else 'trechos.jsonl')
            temp = destination.with_suffix('.jsonl.tmp')
            with temp.open('w',encoding='utf-8',newline='\n') as output:
                for payload, in db.execute('SELECT payload FROM rows ORDER BY id'): output.write(payload+'\n')
            os.replace(temp,destination)
            reasons = Counter()
            for value, in db.execute('SELECT discarded FROM chunks'): reasons.update(json.loads(value))
            summary = {'schema_version':1,'modo':args.modo,'arquivos':documents,'documentos_sem_texto':empty,
                'linhas':db.execute('SELECT COUNT(*) FROM rows').fetchone()[0],
                'grupos':db.execute('SELECT COUNT(DISTINCT group_id) FROM rows').fetchone()[0],
                'trechos':db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0],
                'falhas_pendentes':failed, 'descartes':dict(reasons),'config_hash':config_hash,
                'saida_hash':sha256_arquivo(destination),'revisao_humana':'pendente'}
            if anonimiza:
                review = saida / 'trocas.jsonl'
                with review.open('w',encoding='utf-8',newline='\n') as output:
                    for chunk_id, tipo, original in db.execute('SELECT chunk,tipo,original FROM trocas ORDER BY chunk,rowid'):
                        output.write(json.dumps({'trecho':chunk_id,'tipo':tipo,'original':original},ensure_ascii=False)+'\n')
                summary['anonimizacao'] = contagem(t for t, in db.execute('SELECT tipo FROM trocas'))
            gravar_json(saida/'curation.json', {'schema_version':2,'config':config,'resumo':summary,
                'proveniencia':'journal.sqlite','garantia':'suporte literal; nao valida a verdade da fonte'})
            gravar_json(saida/'resumo.json',summary)
            print(json.dumps(summary,ensure_ascii=False))
            return 0 if summary['linhas'] and not failed and not empty else 2
        finally: db.close()


if __name__=='__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main,"resumo.json"))
