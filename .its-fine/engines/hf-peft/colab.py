"""Colab connection, verified transfers and durable supervised jobs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import argparse
import base64
import http.client
import io
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
import zipfile
from common import gravar_json, ler_json, politica_privacidade, sha256_arquivo
from itsfine.privacy import require_destination, require_colab, NoRedirect, endpoint, loopback
from itsfine.storage import digest, Store, atomic_json, confined
from itsfine.execution import validate_spec
from itsfine.engines import remote_python, venv_commands
from itsfine.transport import import_result

AQUI=Path(__file__).resolve().parent
# o tunel corta requisicao acima de ~100 s: partes pequenas cabem mesmo em conexao lenta
PARTE=16*1024**2           # corpo de cada PUT (16 MB passa em 100 s acima de ~1,5 Mbps de upload)
PARTE_DOWNLOAD=4*1024**2   # faixa de cada GET do resultado
PARALELO=8                 # faixas baixadas ao mesmo tempo
NOTEBOOKS='.its-fine/colab/notebooks'
SESSAO_LOCAL='.its-fine/colab/sessao.json'
CELULA=r'''# ITS-FINE __VERSAO__: rode esta celula e volte ao chat. Nao cole tokens no chat.
import base64, io, json, os, pathlib, re, secrets, subprocess, sys, time, urllib.request, zipfile
SESSAO=__SESSAO__
raiz=pathlib.Path('/content/itsfine')
connection=raiz/('conexao-itsfine-'+SESSAO+'.json')
def preparar():
    if (raiz/'bootstrap.json').exists():
        print('Preparo anterior incompleto. Use Ambiente de execucao > Desconectar e excluir ambiente de execucao e rode esta celula de novo.')
        return False
    resposta=input('Autoriza o ITS-FINE a preparar este Colab e executar os treinos que voce aprovar? Digite SIM (ou YES): ').strip()
    if resposta.lower() not in ('sim','yes'):
        print('Nada foi feito: a resposta foi "%s". Por seguranca, o ITS-FINE so prepara o Colab depois da confirmacao SIM (ou YES). '
              'Rode esta celula de novo e digite SIM.' % resposta)
        return False
    raiz.mkdir(exist_ok=True)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(__RUNTIME__))).extractall(raiz)
    print('Instalando os pacotes do ITS-FINE...')
    subprocess.run([sys.executable,'-m','pip','install','-q']+__REQUIREMENTS__,check=True)
    # o Colab traz torchao antigo, que o peft recusa mesmo sem usar; nenhum motor usa torchao
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','-q','torchao'],check=False)
    # motores com dependencias fixadas que conflitam com as acima: venv propria sobre o torch da sessao
    audio='torchaudio=='+subprocess.run([sys.executable,'-c','import torch;print(torch.__version__.split("+")[0])'],capture_output=True,text=True).stdout.strip()
    for nome,comandos in __VENVS__:
        print('Preparando o ambiente do motor',nome,'...')
        for comando in comandos: subprocess.run([sys.executable if c=='{python}' else audio if c=='{torchaudio}' else c for c in comando],check=True)
    # ambiente conferido aqui, antes do tunel: problema aparece nesta celula, nao depois do envio
    checagem=subprocess.run([sys.executable,'-c','import torch, transformers, peft\nfrom peft.import_utils import is_torchao_available\nis_torchao_available()\n'
                             'assert torch.cuda.is_available(), "Sem GPU: Ambiente de execucao > Alterar o tipo de ambiente de execucao > T4 GPU"'],
                            capture_output=True,text=True)
    if checagem.returncode:
        print('O ambiente do Colab nao ficou pronto:',(checagem.stderr.strip().splitlines() or ['erro sem mensagem'])[-1])
        print('Corrija, use Ambiente de execucao > Desconectar e excluir ambiente de execucao e rode esta celula de novo.')
        return False
    sys.path.insert(0,str(raiz))
    from itsfine.storage import atomic_json
    atomic_json(raiz/'.its-fine/policy.json',{'schema_version':1,'allow_process':True,'allow_data_to_harness':False})
    token=secrets.token_urlsafe(32)
    env={**os.environ,'ITSFINE_RAIZ':str(raiz),'ITSFINE_TOKEN':token,'ITSFINE_SESSAO':SESSAO,'PYTHONPATH':str(raiz)}
    # sessao propria: parar a celula (botao de parar) nao derruba o servidor nem o tunel
    server=subprocess.Popen([sys.executable,str(raiz/'colab_server.py')],env=env,stdout=open(raiz/'server.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(raiz/'bootstrap.json',{'pid':server.pid})
    binary=raiz/'cloudflared'
    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',binary)
    os.chmod(binary,0o755)
    url=None
    for tentativa in range(3):
        log=open(raiz/'tunnel.log','w')
        # http2 sobre TCP: pelo QUIC (UDP) padrao a saida do Colab ficou em ~110 KB/s
        tunnel=subprocess.Popen([str(binary),'tunnel','--url','http://127.0.0.1:8765','--no-autoupdate','--protocol','http2'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(60):
            # api.trycloudflare.com aparece no log quando a Cloudflare recusa o tunel: nao e o endereco
            match=re.search(r'https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com',(raiz/'tunnel.log').read_text())
            if match: url=match.group(0); break
            if tunnel.poll() is not None: break
            time.sleep(1)
        if url: break
        tunnel.kill(); print('A Cloudflare nao abriu o tunel; tentando de novo...',flush=True); time.sleep(20)
    if not url:
        print('A Cloudflare nao abriu o tunel agora. Use Ambiente de execucao > Desconectar e excluir ambiente de execucao e rode esta celula de novo em alguns minutos.')
        return False
    atomic_json(connection,{'url':url,'token':token,'protocol':3,'sessao':SESSAO})
    return True
if connection.exists() or preparar():
    # download feito pelo proprio navegador: files.download depende do kernel livre, e esta celula segue rodando
    from IPython.display import HTML, Javascript, display
    conteudo=base64.b64encode(connection.read_bytes()).decode()
    display(Javascript("const b=new Blob([atob('%s')],{type:'application/json'});const a=document.createElement('a');"
                       "a.href=URL.createObjectURL(b);a.download='%s';document.body.appendChild(a);a.click();a.remove();" % (conteudo,connection.name)))
    display(HTML('<a download="%s" href="data:application/json;base64,%s">Se o download nao comecou, clique aqui para baixar a conexao</a>' % (connection.name,conteudo)))
    print('Pronto. Volte ao chat: o agente encontra a conexao na pasta Downloads. Deixe esta aba aberta ate o fim do treino.')
    # celula ocupada de proposito: com ela rodando o Colab nao desliga a maquina por inatividade
    sys.path.insert(0,str(raiz))
    from itsfine.execution import status
    from itsfine.progress import bar_line
    vistos={}
    while True:
        for job in sorted((raiz/'_itsfine_runs').glob('*/job.json')):
            try: estado=status(raiz,job.parent.name)
            except Exception: continue
            p=estado.get('progresso') or {}
            chave=(estado['status'],int(100*p.get('passo',0)/max(1,p.get('total_passos') or 1))//5)
            if vistos.get(job.parent.name)!=chave:
                vistos[job.parent.name]=chave; print(time.strftime('%H:%M'),job.parent.name+':',bar_line(estado),flush=True)
        time.sleep(30)
'''

def gerar_notebook(destino,root='.'):
    import itsfine
    package=Path(itsfine.__file__).parent
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(AQUI/'colab_server.py','colab_server.py')
        for p in sorted(package.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts:
                z.write(p,'itsfine/'+p.relative_to(package).as_posix())
    # o identificador liga a conexao baixada a este projeto: allow_colab so vale para ele
    sessao=secrets.token_hex(8)
    source=(CELULA.replace('__RUNTIME__',repr(base64.b64encode(memory.getvalue()).decode())).replace('__SESSAO__',repr(sessao))
            .replace('__VERSAO__',itsfine.__version__))
    # a sessao atende qualquer motor instalado: pacotes dos motores compartilhados e uma venv para cada motor proprio
    pacotes=['jsonschema>=4,<5']; venvs=[]
    for engine in sorted(AQUI.parent.glob('*/engine.json')):
        info=ler_json(engine); requirements=info['requisitos']
        if info.get('ambiente')=='proprio':
            venvs.append((info['engine'],venv_commands('{python}',info,PurePosixPath('/content/itsfine/.its-fine/venvs')/info['engine'])))
            continue
        for pacote in requirements['pacotes']+requirements.get('opcionais',{}).get('qlora',[]):
            if pacote not in pacotes: pacotes.append(pacote)
    source=source.replace('__REQUIREMENTS__',repr(pacotes)).replace('__VENVS__',repr(venvs))
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},
        'cells':[{'cell_type':'markdown','id':'intro','metadata':{},'source':['# ITS-FINE %s no Colab\n' % itsfine.__version__,'Escolha a GPU (Ambiente de execucao > Alterar o tipo), rode a celula abaixo e digite SIM. Depois volte ao chat.\n']},
                 {'cell_type':'code','id':'bootstrap','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(keepends=True)}]}
    if Path(destino).exists(): raise FileExistsError('NOTEBOOK_EXISTS')
    gravar_json(destino,nb)
    atomic_json(Path(root).resolve()/NOTEBOOKS/(sessao+'.json'),{'sessao':sessao,'notebook':str(Path(destino).resolve()),'criado':time.time()})
    return {'notebook':str(destino),'sessao':sessao}

class Cliente:
    def __init__(self,url,token,sessao=None,do_projeto=False):
        self.url=endpoint(url); self.token=token; self.sessao=sessao; self.do_projeto=do_projeto
        if not self.url.startswith('https://') and not loopback(self.url): raise ValueError('TLS_REQUIRED')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())

    def autorizar(self,policy):
        # sessao aberta por notebook deste projeto: vale allow_colab; qualquer outra, destino listado
        if self.do_projeto and policy.get('allow_colab') and not loopback(self.url): require_colab(policy,self.url)
        else: require_destination(policy,'transfer',self.url)

    def _req(self,method,route,body=None,timeout=120):
        req=urllib.request.Request(self.url+route,data=body,method=method,
              headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        try:
            with self.opener.open(req,timeout=timeout) as response: return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError('HTTP %d: remote operation refused' % exc.code) from None

    def status(self): return self._req('GET','/status')

    def _put(self,remoto,offset,parte,total,expected):
        req=urllib.request.Request(self.url+'/files?path='+quote(remoto,safe='')+'&offset='+str(offset),data=parte,method='PUT',
            headers={'Authorization':'Bearer '+self.token,'Content-Length':str(len(parte)),'X-Total-Size':str(total),'X-Content-SHA256':expected})
        with self.opener.open(req,timeout=300) as response: return json.load(response)

    def enviar(self,local,remoto,policy):
        self.autorizar(policy)
        if not remoto.startswith('incoming/') or any(p in ('','..','.') for p in remoto.split('/')) or '\\' in remoto or ':' in remoto:
            raise ValueError('INVALID_UPLOAD_PATH')
        source=Path(local); expected=sha256_arquivo(source); total=source.stat().st_size
        falhas=0
        with source.open('rb') as stream:
            while True:
                try:
                    # sonda vazia: o servidor diz se ja tem o arquivo inteiro ou quanto ja chegou
                    result=self._put(remoto,0,b'',total,expected)
                    if 'sha256' not in result:
                        offset=result['recebido']; stream.seek(offset)  # retoma de onde parou
                        while 'sha256' not in result:
                            parte=stream.read(PARTE)
                            if not parte: raise ValueError('UPLOAD_INCOMPLETE')
                            result=self._put(remoto,offset,parte,total,expected); offset+=len(parte)
                    break
                except (OSError,http.client.HTTPException):
                    # falha de rede no tunel: tenta de novo a partir do que o servidor ja tem
                    falhas+=1
                    if falhas>=5: raise
                    time.sleep(3*falhas)
        if result.get('sha256')!=expected: raise ValueError('UPLOAD_HASH_MISMATCH')
        return result

    def approve(self,spec,actor):
        return self._req('POST','/approve',json.dumps({'spec':spec,'actor':actor,'acknowledge_process':True}).encode())

    def run(self,spec,run_id):
        if not isinstance(spec,dict): raise ValueError('EXECUTION_SPEC_REQUIRED')
        return self._req('POST','/run',json.dumps({'spec':spec,'run_id':run_id}).encode())

    def job(self,ident,completo=False): return self._req('GET','/jobs/'+ident+('?completo=1' if completo else ''))
    def cancel(self,ident): return self._req('POST','/cancel',json.dumps({'run_id':ident}).encode())

    def esperar(self,ident,intervalo=2,limite=3600):
        start=time.monotonic(); desconhecida=None
        while time.monotonic()-start<limite:
            value=self.job(ident)
            if value['status'] in ('concluida','falhou','interrompida'): return value
            if value['status']=='inexistente': raise RuntimeError('JOB_LOST: a VM do Colab nao tem este job')
            # desconhecida pode ser passageira; so vira erro se persistir
            desconhecida=(desconhecida or time.monotonic()) if value['status']=='desconhecida' else None
            if desconhecida and time.monotonic()-desconhecida>600: raise RuntimeError('JOB_UNKNOWN: reconcile without resubmitting')
            time.sleep(intervalo)
        raise TimeoutError('OBSERVATION_TIMEOUT: job remains under its server-side deadline')

    def zip(self,ident,checkpoints=False):
        # empacotar resultado grande pode passar do limite do tunel: o servidor continua e o pedido repete
        for tentativa in range(20):
            try: return self._req('POST','/zip',json.dumps({'run_id':ident,'checkpoints':checkpoints}).encode(),timeout=120)
            except (OSError,RuntimeError,http.client.HTTPException):
                if tentativa==19: raise
                time.sleep(5)

    def baixar(self,remote,local,expected,size):
        target=Path(local); target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha256_arquivo(target)==expected: return
            raise FileExistsError('RESULT_DESTINATION_EXISTS')
        temp=target.with_name(target.name+'.download')
        with temp.open('wb') as out: out.truncate(size)
        def faixa(inicio):
            tamanho=min(PARTE_DOWNLOAD,size-inicio)
            req=urllib.request.Request(self.url+'/files?path='+quote(remote,safe='')+'&inicio=%d&tamanho=%d'%(inicio,tamanho),
                                       headers={'Authorization':'Bearer '+self.token})
            for tentativa in range(5):
                try:
                    with self.opener.open(req,timeout=120) as response: dados=response.read()
                    if len(dados)!=tamanho: raise ValueError('SHORT_READ')
                    with temp.open('r+b') as out: out.seek(inicio); out.write(dados)
                    return
                except (OSError,ValueError,http.client.HTTPException):  # resposta cortada pelo tunel: IncompleteRead
                    if tentativa==4: raise
                    time.sleep(2*(tentativa+1))
        # faixas em paralelo: uma conexao so pelo tunel e lenta
        with ThreadPoolExecutor(PARALELO) as pool: list(pool.map(faixa,range(0,size,PARTE_DOWNLOAD)))
        if sha256_arquivo(temp)!=expected: raise ValueError('DOWNLOAD_HASH_MISMATCH')
        os.replace(temp,target)

def registrados(root):
    return {p.stem for p in (Path(root).resolve()/NOTEBOOKS).glob('*.json')}

def sessao(root,path=None):
    path=Path(path) if path else Path(root).resolve()/SESSAO_LOCAL
    if not path.is_file(): raise ValueError('COLAB_NOT_CONNECTED: rode colab.py conectar')
    value=ler_json(path)
    return Cliente(value['url'],value['token'],value.get('sessao'),value.get('sessao') in registrados(root))

def pastas_de_download():
    pastas=[]
    if os.name=='nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders') as key:
                pastas.append(Path(winreg.QueryValueEx(key,'{374DE290-123F-4565-9164-39C4925E467B}')[0]))
        except OSError: pass
    pastas.append(Path.home()/'Downloads')
    return [p for i,p in enumerate(pastas) if p.is_dir() and p not in pastas[:i]]

def conectar(root,esperar=0,pastas=None,arquivo=None):
    """Acha nos Downloads a conexao baixada por notebook deste projeto, confere a sessao no servidor
    e a guarda no projeto. O usuario so roda a celula; o token nunca passa pelo chat.
    Quem salvou o arquivo em outro lugar informa o caminho (arquivo)."""
    conhecidos=registrados(root)
    if not conhecidos: raise ValueError('NOTEBOOK_NOT_GENERATED: gere o notebook com colab.py notebook')
    pastas=[Path(p) for p in pastas] if pastas else pastas_de_download()
    limite=time.monotonic()+esperar; visto=None
    while True:
        caminhos=[Path(arquivo)] if arquivo else [p for pasta in pastas for p in pasta.glob('conexao-itsfine*.json')]
        candidatos=[]
        for caminho in caminhos:
            try: valor=ler_json(caminho); instante=caminho.stat().st_mtime
            except (OSError,ValueError): continue
            if isinstance(valor,dict) and valor.get('sessao') in conhecidos: candidatos.append((instante,caminho,valor))
        if arquivo and Path(arquivo).is_file() and not candidatos:
            raise ValueError('CONNECTION_FILE_NOT_FROM_THIS_PROJECT: gere o notebook neste projeto')
        if candidatos:
            _,encontrado,valor=max(candidatos,key=lambda c:c[0]); visto=encontrado.name
            try:
                estado=Cliente(valor['url'],valor['token'],valor['sessao'],True).status()
                if estado.get('sessao')==valor['sessao']:
                    destino=Path(root).resolve()/SESSAO_LOCAL
                    atomic_json(destino,{**valor,'origem':encontrado.name,'conectado':time.time()})
                    (destino.parent/'.gitignore').write_text('*\n',encoding='utf-8')  # a sessao tem token: nunca vai para o git
                    try: encontrado.unlink()  # o token nao fica largado na pasta do usuario
                    except OSError: pass
                    return {'conectado':True,'arquivo':str(encontrado),'gpu':estado.get('gpu'),'sessao':valor['sessao']}
            except (OSError,RuntimeError,ValueError): pass  # tunel ainda subindo ou sessao antiga
        if time.monotonic()>=limite:
            if visto: raise ValueError('COLAB_SESSION_UNREACHABLE:'+visto)
            onde=[str(arquivo)] if arquivo else [str(p) for p in pastas]
            raise ValueError('CONNECTION_FILE_NOT_FOUND: procurei em %s; se salvou em outro lugar, informe o caminho com --arquivo' % onde)
        time.sleep(3)

def push(root,client,spec,target):
    validate_spec(root,spec)
    if spec['executor']!='process': raise ValueError('COLAB_PROCESS_SPEC_REQUIRED')
    policy=politica_privacidade(root); client.autorizar(policy)
    rewritten={**spec,'inputs':[]}; rewritten.pop('python',None)
    # motor com venv propria: a sessao criou essa venv no preparo; o caminho local nao vale na VM
    if (venv:=remote_python(root,spec)): rewritten['python']=venv
    for entry in spec['inputs']:
        # endereco pelo conteudo: arquivo que ja esta na VM nao sobe de novo em outra tentativa
        remote='incoming/blobs/'+entry['sha256']
        client.enviar(confined(root,entry['path']),remote,policy)
        rewritten['inputs'].append({**entry,'path':remote})
    gravar_json(target,rewritten)
    gravar_json(str(target)+'.transfer.json',{'schema_version':1,'destination':client.url,
        'local_spec_hash':digest(spec),'remote_spec_hash':digest(rewritten),
        'inputs':[{k:e[k] for k in ('name','sha256')} for e in spec['inputs']]})
    return {'spec_hash':digest(rewritten),'inputs':len(rewritten['inputs'])}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.')
    sub=p.add_subparsers(dest='cmd',required=True)
    nb=sub.add_parser('notebook'); nb.add_argument('--saida',required=True)
    con=sub.add_parser('conectar',help='acha a conexao baixada pelo notebook deste projeto')
    con.add_argument('--esperar',type=int,default=900,help='segundos esperando o usuario rodar a celula')
    con.add_argument('--pasta',action='append',help='pasta de downloads; padrao: a do sistema')
    con.add_argument('--arquivo',help='caminho do arquivo de conexao, quando o usuario salvou fora da pasta Downloads')
    for name in ('status','push','approve','run','cancel','pull','acompanhar'):
        child=sub.add_parser(name); child.add_argument('--sessao',help='padrao: a conexao guardada por conectar')
        if name=='status': child.add_argument('--run-id')
        if name in ('push','approve','run'): child.add_argument('--spec',required=True)
        if name in ('run','cancel','pull','acompanhar'): child.add_argument('--run-id',required=True)
        if name in ('push','pull'): child.add_argument('--saida',required=True)
        if name=='approve': child.add_argument('--actor',required=True)
        if name=='pull':
            child.add_argument('--spec-hash',required=True)
            child.add_argument('--com-checkpoints',action='store_true',help='traz tambem os checkpoints por epoca; padrao: so o resultado')
    args=p.parse_args()
    if args.cmd=='notebook': result=gerar_notebook(args.saida,args.root)
    elif args.cmd=='conectar': result=conectar(args.root,args.esperar,args.pasta,args.arquivo)
    else:
        c=sessao(args.root,args.sessao)
        if args.cmd=='acompanhar':
            # uma linha de barra por mudanca, ate o job terminar; feito para o monitor do chat
            from itsfine.progress import follow
            follow(lambda: c.job(args.run_id))
            return 0
        if args.cmd=='status':
            if not args.run_id: result=c.status()
            else:
                # mensagem completa do erro so com leitura de dados liberada pelo usuario
                result=c.job(args.run_id,politica_privacidade(args.root)['allow_data_to_harness'] is True)
                path=Store(args.root).run_dir(args.run_id)/'remote-job.json'
                if result['status']=='inexistente' and path.exists():
                    atomic_json(path,{**ler_json(path),'status':'perdida','reason':'VM do Colab perdida: submeta de novo com outro run id'})
        elif args.cmd=='push': result=push(args.root,c,ler_json(args.spec),args.saida)
        elif args.cmd=='approve': result=c.approve(ler_json(args.spec),args.actor)
        elif args.cmd=='cancel': result=c.cancel(args.run_id)
        elif args.cmd=='run':
            spec=ler_json(args.spec)
            folder=Store(args.root).run_dir(args.run_id)
            record={'schema_version':1,'run_id':args.run_id,'spec_hash':digest(spec),'session':c.url,'status':'submetendo'}
            path=folder/'remote-job.json'
            if path.exists():
                previous=ler_json(path)
                if previous['spec_hash']!=record['spec_hash']: raise ValueError('RUN_INPUTS_FROZEN')
            else: atomic_json(path,record) # persist intent even if the response is lost
            result=c.run(spec,args.run_id)
            atomic_json(path,{**record,**result})
        elif args.cmd=='pull':
            receipt=c.zip(args.run_id,args.com_checkpoints)
            if receipt['spec_hash']!=args.spec_hash: raise ValueError('RESULT_IDENTITY_MISMATCH')
            inicio=time.monotonic()
            c.baixar(receipt['zip'],args.saida,receipt['sha256'],receipt['bytes'])
            result={**import_result(args.root,args.saida,args.run_id,args.spec_hash),
                    'mb':round(receipt['bytes']/1024**2,1),'segundos_download':round(time.monotonic()-inicio,1)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
