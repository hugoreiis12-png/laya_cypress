"""Tela local para o usuario ligar e desligar as permissoes, no lugar de editar o policy.json na mao."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import secrets
import threading
import webbrowser
from .privacy import FLAGS, endpoint, policy_path
from .storage import atomic_json

# nome, o que libera, campo de lista que a permissao usa (se houver)
ITENS = [
    ('allow_data_to_harness', 'O agente ler seus documentos', 'Sem isso o agente trabalha só com números e resumos, nunca vê o conteúdo.', None),
    ('allow_external_generator', 'Gerar exemplos em um serviço externo', 'Só nos endereços que você listar abaixo.', 'external_generator_endpoints'),
    ('allow_data_transfer', 'Enviar dados a destinos exatos', 'Só para os endereços que você listar abaixo.', 'transfer_destinations'),
    ('allow_colab', 'Treinar no Google Colab', 'Envia dados ao Colab aberto por um notebook deste projeto.', None),
    ('allow_kaggle', 'Treinar no Kaggle', 'Envia dados a um dataset privado da sua conta Kaggle, que ficam lá até você mandar apagar.', None),
    ('allow_process', 'Treinar neste computador', 'Roda o treino como processo local, sem isolamento de arquivos.', None),
    ('require_isolated_executor', 'Exigir executor isolado', 'Recusa qualquer processamento local neste ambiente.', None),
]
LISTAS = ('external_generator_endpoints', 'transfer_destinations', 'protected_data_paths')

PAGINA = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Permissões do ITS-FINE</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.4rem}.p{display:flex;gap:1rem;padding:.8rem 0;border-bottom:1px solid #ddd;align-items:flex-start}
.p input[type=checkbox]{width:1.4rem;height:1.4rem;margin-top:.2rem}.p b{display:block}.p small{color:#555}
textarea{width:100%;min-height:4rem;margin-top:.4rem;font:14px monospace}h2{font-size:1.1rem;margin-top:1.6rem}
button{font-size:1rem;padding:.6rem 1.2rem;margin-right:.6rem;cursor:pointer}#msg{margin-top:1rem;white-space:pre-wrap}
.ok{color:#0a6b2d}.erro{color:#b00020}</style>
<h1>Permissões do ITS-FINE</h1>
<p>Tudo começa desligado. Marque só o que você quer liberar. As mudanças só valem depois de salvar.</p>
<div id="lista"></div><h2>Pastas e arquivos que o agente nunca lê (um padrão por linha)</h2>
<textarea id="protected_data_paths" placeholder="ex.: clientes/**"></textarea>
<div style="margin-top:1.2rem"><button id="salvar">Salvar</button><button id="sair">Fechar sem salvar</button></div>
<div id="msg"></div>
<script>
const T=location.pathname.split('/')[1], ITENS=__ITENS__; let inicial;
const $=s=>document.querySelector(s), msg=(t,c)=>{$('#msg').className=c||'';$('#msg').textContent=t};
const linhas=id=>$('#'+id).value.split('\\n').map(s=>s.trim()).filter(Boolean);
function ler(){const v={};for(const [k] of ITENS)v[k]=$('#'+k).checked;
 for(const l of __LISTAS__)v[l]=linhas(l);return v}
function montar(p){const d=$('#lista');d.innerHTML='';
 for(const [k,nome,ajuda,lista] of ITENS){const w=document.createElement('div');w.className='p';
  w.innerHTML=`<input type=checkbox id=${k}><div><b></b><small></small>${lista?`<textarea id=${lista} placeholder="um endereço por linha, ex.: https://exemplo.com/v1"></textarea>`:''}</div>`;
  w.querySelector('b').textContent=nome;w.querySelector('small').textContent=ajuda;d.append(w);$('#'+k).checked=p[k]}
 for(const l of __LISTAS__)$('#'+l).value=(p[l]||[]).join('\\n');inicial=ler()}
fetch(`/${T}/estado`).then(r=>r.json()).then(montar).catch(()=>msg('Não consegui carregar. Rode o comando de novo.','erro'));
$('#salvar').onclick=async()=>{const v=ler(),mud=Object.keys(v).filter(k=>JSON.stringify(v[k])!==JSON.stringify(inicial[k]));
 if(!mud.length)return msg('Nada mudou.');
 const r=await fetch(`/${T}/salvar`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(v)});
 const j=await r.json();if(!r.ok)return msg(j.erro,'erro');inicial=v;msg('Salvo: '+mud.join(', ')+'.\\nPode fechar esta aba.','ok')};
$('#sair').onclick=async()=>{await fetch(`/${T}/sair`,{method:'POST'});msg('Fechado. Nada foi salvo.')};
</script></html>'''


def carregar(caminho):
    try: valor = json.loads(Path(caminho).read_text(encoding='utf-8'))
    except (OSError, ValueError): valor = {}
    return valor if isinstance(valor, dict) else {}


def validar(entrada):
    """Devolve so os campos permitidos, ja conferidos; erro com texto para o usuario."""
    saida = {}
    for nome in FLAGS:
        if type(entrada.get(nome)) is not bool: raise ValueError('Valor inválido em '+nome)
        saida[nome] = entrada[nome]
    for nome in LISTAS:
        valores = entrada.get(nome, [])
        if not isinstance(valores, list) or any(not isinstance(v, str) or not v.strip() for v in valores): raise ValueError('Lista inválida em '+nome)
        valores = [v.strip() for v in valores]
        if nome != 'protected_data_paths':
            try: valores = [endpoint(v) for v in valores]
            except ValueError: raise ValueError('Endereço inválido em '+nome+': use http:// ou https:// sem senha nem parâmetros') from None
        elif any('\\' in v or ':' in v or v.startswith('/') or '..' in v.split('/') for v in valores):
            raise ValueError('Padrão inválido em pastas protegidas: use caminhos relativos com /')
        saida[nome] = list(dict.fromkeys(valores))
    return saida


def salvar(caminho, entrada):
    novo = validar(entrada)
    atual = carregar(caminho)
    atomic_json(Path(caminho), {**atual, 'schema_version': 1, **novo})
    return novo


def abrir(root='.', navegar=True, tempo=900):
    caminho = policy_path(Path(root))
    caminho.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(16); encerrar = threading.Event()

    class Tela(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def responder(self, codigo, corpo, tipo='application/json'):
            dados = corpo.encode('utf-8') if isinstance(corpo, str) else corpo
            self.send_response(codigo); self.send_header('Content-Type', tipo+'; charset=utf-8')
            self.send_header('Content-Length', str(len(dados))); self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(dados)
        def permitido(self):
            porta = self.server.server_address[1]
            origem = self.headers.get('Origin')
            return (self.path.split('/')[1:2] == [token] and self.headers.get('Host') in ('127.0.0.1:%d' % porta, 'localhost:%d' % porta)
                    and origem in (None, 'http://127.0.0.1:%d' % porta, 'http://localhost:%d' % porta))
        def do_GET(self):
            if not self.permitido(): return self.responder(403, '{"erro":"acesso negado"}')
            if self.path == '/'+token+'/estado':
                atual = carregar(caminho)
                return self.responder(200, json.dumps({**{k: atual.get(k) is True for k in FLAGS}, **{k: atual.get(k, []) for k in LISTAS}}))
            pagina = PAGINA.replace('__ITENS__', json.dumps(ITENS)).replace('__LISTAS__', json.dumps(LISTAS))
            self.responder(200, pagina, 'text/html')
        def do_POST(self):
            if not self.permitido(): return self.responder(403, '{"erro":"acesso negado"}')
            if self.path.endswith('/sair'): encerrar.set(); return self.responder(200, '{}')
            try:
                tamanho = int(self.headers.get('Content-Length', 0))
                if tamanho > 100_000: raise ValueError('Pedido grande demais')
                salvar(caminho, json.loads(self.rfile.read(tamanho)))
            except (ValueError, TypeError) as exc: return self.responder(400, json.dumps({'erro': str(exc)}, ensure_ascii=False))
            self.responder(200, '{}'); encerrar.set()

    servidor = HTTPServer(('127.0.0.1', 0), Tela)
    endereco = 'http://127.0.0.1:%d/%s/' % (servidor.server_address[1], token)
    print('Tela de permissões: '+endereco, flush=True)
    if navegar: webbrowser.open(endereco)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    encerrar.wait(tempo)
    threading.Timer(1.5, servidor.shutdown).start()  # deixa a resposta do ultimo pedido chegar
    return 0
