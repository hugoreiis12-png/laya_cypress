"""Colab transport backed by the durable local job supervisor."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import os
import re
import secrets
import shutil
import subprocess
from itsfine.storage import Store, confined, file_hash, identifier, lock
from itsfine.execution import authorize, submit, status, cancel
from itsfine.transport import pack_result

RAIZ=Path(os.environ.get('ITSFINE_RAIZ','/content/itsfine')).resolve()
TOKEN=os.environ.get('ITSFINE_TOKEN') or secrets.token_urlsafe(32)
PORTA=int(os.environ.get('ITSFINE_PORTA','8765'))
SESSAO=os.environ.get('ITSFINE_SESSAO')  # identificador do notebook que abriu esta sessao
MAX_UPLOAD=10*1024**3

def confinado(rel): return confined(RAIZ,rel)

def gpu():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=10)
        name,total,used=r.stdout.strip().splitlines()[0].split(',')
        return {'nome':name.strip(),'vram_total_gb':round(int(total)/1024,2),'vram_usada_gb':round(int(used)/1024,2)}
    except Exception: return None

class Handler(BaseHTTPRequestHandler):
    timeout=120  # conexao meio aberta pelo tunel nao prende a thread para sempre
    def log_message(self,*args): pass

    def _json(self,code,value):
        data=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)

    def _autorizado(self):
        if not secrets.compare_digest(self.headers.get('Authorization',''), 'Bearer '+TOKEN):
            self._json(401,{'erro':'TOKEN_INVALIDO'}); return False
        return True

    def _corpo(self,limite=1024*1024):
        # pelo tunel em http2 o corpo pode chegar em pedacos (chunked), sem Content-Length
        dados=bytearray()
        if 'Content-Length' in self.headers:
            size=int(self.headers['Content-Length'])
            if size<0 or size>limite: raise ValueError('REQUEST_TOO_LARGE')
            while len(dados)<size:
                chunk=self.rfile.read(min(size-len(dados),1024*1024))
                if not chunk: raise ValueError('INCOMPLETE_UPLOAD')
                dados+=chunk
        elif self.headers.get('Transfer-Encoding','').lower()=='chunked':
            while True:
                n=int(self.rfile.readline(1024).split(b';')[0].strip(),16)
                if n==0:
                    while self.rfile.readline(1024) not in (b'\r\n',b'\n',b''): pass  # trailers
                    break
                if len(dados)+n>limite: raise ValueError('REQUEST_TOO_LARGE')
                chunk=self.rfile.read(n)
                if len(chunk)!=n: raise ValueError('INCOMPLETE_UPLOAD')
                dados+=chunk; self.rfile.read(2)
        return bytes(dados)

    def _job(self,ident,full=False):
        ident=identifier(ident)
        # VM nova (sessao perdida) nao conhece jobs da anterior: estado definido, nunca "reconcilie" eterno
        if not (RAIZ/'_itsfine_runs'/ident/'job.json').exists(): return {'run_id':ident,'status':'inexistente'}
        j=status(RAIZ,ident,full)
        return {k:j.get(k) for k in ('run_id','status','spec_hash','submission_key','heartbeat','exit_code','reason','erro','progresso')}

    def do_GET(self):
        if not self._autorizado(): return
        url=urlparse(self.path); query=parse_qs(url.query)
        try:
            if url.path=='/status':
                jobs={p.parent.name:self._job(p.parent.name) for p in (RAIZ/'_itsfine_runs').glob('*/job.json')}
                self._json(200,{'ok':True,'gpu':gpu(),'jobs':jobs,'protocol':3,'sessao':SESSAO})
            elif url.path.startswith('/jobs/'):
                self._json(200,self._job(url.path.split('/')[2],query.get('completo')==['1']))
            elif url.path=='/files':
                rel=query['path'][0]
                if not rel.startswith('results/') or not rel.endswith('.zip'): raise ValueError('RESULT_ARCHIVE_REQUIRED')
                target=confinado(rel); size=target.stat().st_size
                # faixa pedida pelo cliente, que baixa partes em paralelo e confere o hash do arquivo inteiro
                inicio=int(query.get('inicio',['0'])[0]); tamanho=int(query.get('tamanho',[str(size)])[0])
                if inicio<0 or tamanho<0 or inicio+tamanho>size: raise ValueError('RANGE_INVALID')
                self.send_response(200); self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(tamanho)); self.end_headers()
                try:
                    with target.open('rb') as stream:
                        stream.seek(inicio); restante=tamanho
                        while restante:
                            bloco=stream.read(min(restante,1024*1024)); self.wfile.write(bloco); restante-=len(bloco)
                except OSError: self.close_connection=True  # cabecalho ja foi: nada de segunda resposta no fluxo
            else: self._json(404,{'erro':'ROTA'})
        except Exception as exc: self._json(400,{'erro':type(exc).__name__})

    def do_PUT(self):
        # Upload em partes: o tunel limita o corpo de cada requisicao. Sonda vazia (offset 0, corpo 0)
        # informa quanto ja chegou, para o cliente retomar; a ultima parte confere o sha256 do arquivo.
        if not self._autorizado(): return
        try:
            url=urlparse(self.path)
            if url.path!='/files': raise ValueError('INVALID_ROUTE')
            query=parse_qs(url.query); rel=query['path'][0]
            if not rel.startswith('incoming/'): raise ValueError('UPLOAD_NAMESPACE_REQUIRED')
            target=confinado(rel)
            # corpo lido antes da trava: conexao lenta ou meio aberta nao segura a trava
            dados=self._corpo(64*1024**2); size=len(dados)
            total=int(self.headers.get('X-Total-Size',size)); offset=int(query.get('offset',['0'])[0])
            expected=self.headers.get('X-Content-SHA256','')
            if offset<0 or total<0 or total>MAX_UPLOAD or offset+size>total: raise ValueError('UPLOAD_SIZE')
            if not re.fullmatch(r'[0-9a-f]{64}',expected): raise ValueError('UPLOAD_HASH_REQUIRED')
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name('.'+target.name+'.'+expected[:16]+'.partial')
            with lock(target.with_name(target.name+'.lock')):
                if target.exists():
                    if file_hash(target)!=expected: raise ValueError('IMMUTABLE_UPLOAD')
                    self._json(200,{'ok':True,'sha256':expected,'bytes':total}); return
                recebido=partial.stat().st_size if partial.exists() else 0
                if size==0 and total>0:
                    self._json(200,{'ok':True,'recebido':recebido}); return
                if offset!=recebido and offset!=0: raise ValueError('UPLOAD_OFFSET_MISMATCH')
                with partial.open('ab' if offset else 'wb') as out: out.write(dados)
                if offset+size<total:
                    self._json(200,{'ok':True,'recebido':offset+size}); return
                if file_hash(partial)!=expected:
                    partial.unlink(); raise ValueError('UPLOAD_HASH_MISMATCH')
                os.replace(partial,target)
            self._json(200,{'ok':True,'sha256':expected,'bytes':total})
        except Exception as exc: self._json(400,{'erro':type(exc).__name__})

    def do_POST(self):
        if not self._autorizado(): return
        try:
            request=json.loads(self._corpo()); route=urlparse(self.path).path
            if route=='/approve':
                value=authorize(RAIZ,request['spec'],request['actor'],request.get('acknowledge_process') is True)
                self._json(200,{'spec_hash':value['spec_hash']})
            elif route=='/run':
                ident=identifier(request['run_id']); submit(RAIZ,ident,request['spec'])
                self._json(200,self._job(ident))
            elif route=='/cancel': self._json(200,cancel(RAIZ,identifier(request['run_id'])))
            elif route=='/zip':
                ident=identifier(request['run_id']); folder=RAIZ/'results'; folder.mkdir(exist_ok=True)
                checkpoints=request.get('checkpoints') is True
                target=folder/(ident+('-com-checkpoints' if checkpoints else '')+'.zip')
                # empacotar pode passar do limite do tunel; o pedido repetido espera a trava
                with lock(folder/(ident+'.lock'),timeout=90):
                    if not target.exists(): pack_result(RAIZ,ident,target,checkpoints)
                job=status(RAIZ,ident)
                self._json(200,{'zip':target.relative_to(RAIZ).as_posix(),'sha256':file_hash(target),'bytes':target.stat().st_size,'spec_hash':job['spec_hash']})
            else: self._json(404,{'erro':'ROTA'})
        except Exception as exc: self._json(400,{'erro':type(exc).__name__})

def main():
    RAIZ.mkdir(parents=True,exist_ok=True); Store(RAIZ).initialize('colab-user')
    server=ThreadingHTTPServer(('127.0.0.1',PORTA),Handler)
    try: server.serve_forever()
    finally: server.server_close()

if __name__=='__main__': main()
