"""Kaggle: inputs em dataset privado, treino em kernel supervisionado e retorno verificado, sem tunel."""
from pathlib import Path, PurePosixPath
import argparse
import base64
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from common import gravar_json, ler_json, politica_privacidade, sha256_arquivo
from itsfine.privacy import require_kaggle
from itsfine.storage import digest, identifier, Store, atomic_json, confined
from itsfine.execution import validate_spec
from itsfine.engines import remote_python, venv_commands
from itsfine.transport import import_result

AQUI=Path(__file__).resolve().parent
PASTA='.its-fine/kaggle'
PACOTE='kaggle>=2.2.4,<3'
GPU='NvidiaTeslaT4'
# a sessao GPU do Kaggle termina em ~9 h; o job precisa acabar antes para o resultado ser empacotado
LIMITE_JOB=8*3600+1800
FOLGA=30*60                # instalacao dos pacotes e empacotamento, alem do timeout do job
PREFIXO='ITSFINE|'         # unica parte do log que o agente le: numeros de progresso, nunca dados
RECIBO='itsfine-recibo.json'
RESULTADO='itsfine-resultado.zip'
SCRIPT=r'''# ITS-FINE __VERSAO__ no Kaggle. Gerado pelo agente a partir da proposta aprovada; nao edite.
import base64, io, json, os, pathlib, shutil, subprocess, sys, time, traceback, zipfile
RUN_ID=__RUN_ID__; SPEC=json.loads(__SPEC__); SPEC_HASH=__SPEC_HASH__; ACTOR=__ACTOR__; CHECKPOINTS=__CHECKPOINTS__
ENTRADA=pathlib.Path(os.environ.get('ITSFINE_KAGGLE_ENTRADA','/kaggle/input'))
SAIDA=pathlib.Path(os.environ.get('ITSFINE_KAGGLE_SAIDA','/kaggle/working'))
# fora de /kaggle/working: so o zip do resultado e o recibo ficam salvos no Kaggle
raiz=pathlib.Path(os.environ.get('ITSFINE_KAGGLE_RAIZ','/tmp/itsfine'))
def linha(**v): print('ITSFINE|'+json.dumps(v,ensure_ascii=False),flush=True)
def recibo(**v): (SAIDA/'itsfine-recibo.json').write_text(json.dumps({'schema_version':1,'run_id':RUN_ID,'spec_hash':SPEC_HASH,**v}))
etapa='preparo'
try:
    raiz.mkdir(parents=True,exist_ok=True); SAIDA.mkdir(parents=True,exist_ok=True)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(__RUNTIME__))).extractall(raiz)
    sys.path.insert(0,str(raiz))
    if not os.environ.get('ITSFINE_KAGGLE_SEM_PIP'):
        linha(etapa='instalando')
        subprocess.run([sys.executable,'-m','pip','install','-q']+__REQUIREMENTS__,check=True)
        # como no Colab: torchao antigo da imagem faz o peft recusar, e nenhum motor usa torchao
        subprocess.run([sys.executable,'-m','pip','uninstall','-y','-q','torchao'],check=False)
        subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available()'],check=True)
        # motor com dependencias fixadas: venv propria sobre o torch do kernel
        audio='torchaudio=='+subprocess.run([sys.executable,'-c','import torch;print(torch.__version__.split("+")[0])'],capture_output=True,text=True).stdout.strip()
        for comando in __VENV__: subprocess.run([sys.executable if c=='{python}' else audio if c=='{torchaudio}' else c.replace('{raiz}',str(raiz)) for c in comando],check=True)
    from itsfine.storage import Store, atomic_json, digest, file_hash
    from itsfine.execution import authorize, submit, status
    from itsfine.transport import pack_result
    from itsfine.progress import bar_line
    if digest(SPEC)!=SPEC_HASH: raise ValueError('SPEC_CHANGED')
    Store(raiz).initialize('kaggle-user')
    atomic_json(raiz/'.its-fine/policy.json',{'schema_version':1,'allow_process':True,'allow_data_to_harness':False})
    etapa='entradas'
    # o caminho de montagem do dataset nao e fixo: acha cada arquivo pelo nome, que e o sha256
    blobs={p.name[:-5]:p for p in ENTRADA.rglob('*.blob')}
    for entry in SPEC['inputs']:
        origem=blobs.get(entry['sha256']); destino=raiz/entry['path']
        if origem is None: raise FileNotFoundError('INPUT_NOT_IN_DATASET')
        destino.parent.mkdir(parents=True,exist_ok=True)
        if not destino.exists(): shutil.copyfile(origem,destino)
        if file_hash(destino)!=entry['sha256']: raise ValueError('INPUT_HASH_MISMATCH')
    authorize(raiz,SPEC,ACTOR,True)
    etapa='treino'
    submit(raiz,RUN_ID,SPEC)
    visto=None; desconhecida=None
    while True:
        estado=status(raiz,RUN_ID); p=estado.get('progresso') or {}
        chave=(estado['status'],int(100*p.get('passo',0)/max(1,p.get('total_passos') or 1))//5)
        if chave!=visto:
            visto=chave; linha(etapa='treino',status=estado['status'],progresso=p,barra=bar_line(estado))
        if estado['status'] in ('concluida','falhou','interrompida'): break
        desconhecida=(desconhecida or time.monotonic()) if estado['status']=='desconhecida' else None
        if desconhecida and time.monotonic()-desconhecida>600: raise RuntimeError('JOB_UNKNOWN')
        time.sleep(2)  # so arquivos locais; o log so ganha linha quando o estado muda
    etapa='empacotando'
    pacote=pack_result(raiz,RUN_ID,SAIDA/'itsfine-resultado.zip',CHECKPOINTS)
    recibo(status=estado['status'],erro=estado.get('erro'),zip='itsfine-resultado.zip',sha256=pacote['sha256'],
           bytes=(SAIDA/'itsfine-resultado.zip').stat().st_size)
    linha(etapa='fim',status=estado['status'])
except Exception as exc:
    local=['%s:%d' % (pathlib.Path(f.filename).name,f.lineno) for f in traceback.extract_tb(exc.__traceback__)[-3:]]
    recibo(status='erro',etapa=etapa,erro=type(exc).__name__,local=local)
    linha(etapa='erro',erro=type(exc).__name__,local=local)
    traceback.print_exc()  # para o usuario na pagina do Kaggle; o agente so le as linhas ITSFINE|
'''

class Kaggle:
    """A parte da API oficial que o ITS-FINE usa. Os testes trocam esta classe por uma falsa."""
    def __init__(self):
        # o pacote imprime a ajuda de login no stdout, que aqui e JSON para o agente
        with contextlib.redirect_stdout(io.StringIO()):
            try: from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError: raise RuntimeError('KAGGLE_PACKAGE_MISSING: rode kaggle_job.py preparar') from None
            self.api=KaggleApi()
            try: self.api.authenticate()
            except (Exception,SystemExit): raise RuntimeError('KAGGLE_CREDENTIALS_MISSING') from None

    def usuario(self): return self.api.get_config_value(self.api.CONFIG_NAME_USER)
    def dataset_status(self,ref):
        # None: o dataset nao existe
        try: return self.api.dataset_status(ref)
        except Exception as exc:
            if getattr(getattr(exc,'response',None),'status_code',None) in (403,404): return None
            raise
    def criar_dataset(self,pasta):
        resposta=self.api.dataset_create_new(str(pasta),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
        if resposta.error: raise RuntimeError('KAGGLE_DATASET_REFUSED')
    def enviar_kernel(self,pasta,timeout):
        resposta=self.api.kernels_push(str(pasta),timeout=str(timeout),acc=GPU)
        if resposta.error or resposta.invalid_dataset_sources: raise RuntimeError('KAGGLE_KERNEL_REFUSED')
        return {'url':resposta.url,'versao':resposta.version_number}
    def kernel_status(self,ref):
        # None: o kernel nao existe (envio anterior nao chegou ao Kaggle)
        try: resposta=self.api.kernels_status(ref)
        except ValueError: return None  # a API troca o 403 de kernel inexistente por ValueError
        except Exception as exc:
            if getattr(getattr(exc,'response',None),'status_code',None)==404: return None
            raise
        return {'status':resposta.status.name.lower(),'falha':resposta.failure_message}
    def _saida(self,ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        dono,slug=ref.split('/')
        with self.api.build_kaggle_client() as cliente:
            pedido=ApiListKernelSessionOutputRequest(); pedido.user_name=dono; pedido.kernel_slug=slug; pedido.page_size=50
            return cliente.kernels.kernels_api_client.list_kernel_session_output(pedido)
    def log(self,ref): return self._saida(ref).log or ''
    def arquivos(self,ref): return {f.file_name:f.url for f in self._saida(ref).files or []}
    def baixar(self,url,destino):
        # em partes: o resultado pode ter GB e nao cabe na memoria de uma vez
        import requests
        with requests.get(url,stream=True,timeout=120) as resposta, open(destino,'wb') as out:
            resposta.raise_for_status()
            for bloco in resposta.iter_content(4*1024**2): out.write(bloco)
    def apagar_dataset(self,ref): self.api.dataset_delete(*ref.split('/'),no_confirm=True)
    def apagar_kernel(self,ref): self.api.kernels_delete(ref,no_confirm=True)

def home(root):
    folder=Path(root).resolve()/PASTA; folder.mkdir(parents=True,exist_ok=True)
    (folder/'.gitignore').write_text('*\n',encoding='utf-8')  # copias de dados e recibos: nunca vao para o git
    return folder

def slug(texto):
    return re.sub(r'[^a-z0-9]+','-',texto.lower()).strip('-')

def preparar(autorizado=False,cliente=Kaggle):
    try:
        with contextlib.redirect_stdout(io.StringIO()): import kaggle.api.kaggle_api_extended  # noqa: F401
    except ImportError:
        # nunca no Python global do usuario: so na venv do framework
        if sys.prefix==sys.base_prefix: return {'pronto':False,'erro':'VENV_REQUIRED: rode com o Python de .its-fine/engine-setup.json'}
        if not autorizado:
            return {'pronto':False,'falta':PACOTE,'acao':'autorize o download e rode de novo com --autorizado'}
        codigo=subprocess.run([sys.executable,'-m','pip','install','-q',PACOTE],capture_output=True).returncode
        if codigo: return {'pronto':False,'erro':'KAGGLE_INSTALL_FAILED'}
    try: return {'pronto':True,'usuario':cliente().usuario()}
    except RuntimeError as exc:
        from itsfine.kaggle_token import criar_pasta
        criar_pasta()
        return {'pronto':False,'erro':str(exc),'acao':'o usuario cria o token em kaggle.com/settings, secao API, e roda no terminal dele, '
                'fora do chat: python .its-fine/bin/fine.py kaggle-token (o mesmo comando troca o token depois; --remover apaga); '
                'a conta precisa de telefone verificado para GPU e internet'}

def transferencia(spec_path):
    path=Path(str(spec_path)+'.transfer.json')
    if not path.is_file(): raise ValueError('KAGGLE_PUSH_REQUIRED: rode push antes')
    registro=ler_json(path)
    if registro.get('backend')!='kaggle' or registro['remote_spec_hash']!=digest(ler_json(spec_path)): raise ValueError('REMOTE_SPEC_CHANGED')
    return registro

def push(root,spec,target,api,esperar=1800):
    validate_spec(root,spec)
    if spec['executor']!='process': raise ValueError('KAGGLE_PROCESS_SPEC_REQUIRED')
    require_kaggle(politica_privacidade(root))
    rewritten={**spec,'inputs':[{**e,'path':'incoming/blobs/'+e['sha256']} for e in spec['inputs']]}; rewritten.pop('python',None)
    # motor com venv propria: o kernel cria essa venv antes do treino; o caminho local nao vale la
    if (venv:=remote_python(root,spec)): rewritten['python']=venv
    # nome pelos inputs, nao pela proposta: sondagem, treino e nova tentativa reaproveitam o que ja subiu
    nome='itsfine-'+digest(sorted({e['sha256'] for e in spec['inputs']}))[:24]; ref=api.usuario()+'/'+nome
    if api.dataset_status(ref) is None:
        pasta=home(root)/'envio'/nome
        if pasta.exists(): shutil.rmtree(pasta)
        pasta.mkdir(parents=True)
        for entry in {e['sha256']:e for e in spec['inputs']}.values():
            origem=confined(root,entry['path']); destino=pasta/(entry['sha256']+'.blob')
            # extensao neutra: o Kaggle descompacta .zip; link evita duplicar checkpoint grande no disco
            try: os.link(origem,destino)
            except OSError: shutil.copyfile(origem,destino)
        gravar_json(pasta/'dataset-metadata.json',{'title':nome,'id':ref,'licenses':[{'name':'CC0-1.0'}]})
        try: api.criar_dataset(pasta)
        finally: shutil.rmtree(pasta,ignore_errors=True)
    limite=time.monotonic()+esperar
    while (estado:=api.dataset_status(ref))!='ready':
        if estado=='failed': raise RuntimeError('KAGGLE_DATASET_FAILED')
        if time.monotonic()>limite: raise TimeoutError('KAGGLE_DATASET_NOT_READY: rode push de novo, o envio ja feito e reaproveitado')
        time.sleep(10)
    gravar_json(target,rewritten)
    gravar_json(str(target)+'.transfer.json',{'schema_version':1,'backend':'kaggle','destination':'kaggle:'+ref,'dataset':ref,
        'local_spec_hash':digest(spec),'remote_spec_hash':digest(rewritten),
        'inputs':[{k:e[k] for k in ('name','sha256')} for e in spec['inputs']]})
    mb=sum(confined(root,e['path']).stat().st_size for e in spec['inputs'])/1024**2
    return {'spec_hash':digest(rewritten),'dataset':ref,'inputs':len(rewritten['inputs']),'mb':round(mb,1)}

def approve(root,spec_path,actor):
    transferencia(spec_path)
    if not isinstance(actor,str) or not actor.strip(): raise ValueError('ACTOR_REQUIRED')
    spec_hash=digest(ler_json(spec_path))
    record={'schema_version':1,'action':'execute','backend':'kaggle','actor':actor,'spec_hash':spec_hash,'at':time.time()}
    atomic_json(home(root)/'aprovacoes'/(spec_hash+'.json'),record)
    return {'spec_hash':spec_hash}

def script(run_id,spec,actor,checkpoints):
    import itsfine
    package=Path(itsfine.__file__).parent; memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts:
                z.write(p,'itsfine/'+p.relative_to(package).as_posix())
    # pacotes dos motores compartilhados, como no Colab; motor com venv propria so quando este job o usa
    pacotes=['jsonschema>=4,<5']; venv=[]
    for engine in sorted(AQUI.parent.glob('*/engine.json')):
        info=ler_json(engine); requirements=info['requisitos']
        if info.get('ambiente')=='proprio':
            if spec.get('python')=='venv:'+info['engine']:
                venv=venv_commands('{python}',info,PurePosixPath('{raiz}/.its-fine/venvs')/info['engine'])
            continue
        for pacote in requirements['pacotes']+requirements.get('opcionais',{}).get('qlora',[]):
            if pacote not in pacotes: pacotes.append(pacote)
    valores={'__VERSAO__':itsfine.__version__,'__RUN_ID__':repr(run_id),'__SPEC__':repr(json.dumps(spec)),
             '__SPEC_HASH__':repr(digest(spec)),'__ACTOR__':repr(actor),'__CHECKPOINTS__':repr(bool(checkpoints)),
             '__RUNTIME__':repr(base64.b64encode(memory.getvalue()).decode()),'__REQUIREMENTS__':repr(pacotes),
             '__VENV__':repr(venv)}
    return re.sub('|'.join(valores),lambda m:valores[m.group(0)],SCRIPT)

def run(root,spec_path,run_id,api,checkpoints=True):
    identifier(run_id); spec=ler_json(spec_path); spec_hash=digest(spec)
    registro=transferencia(spec_path)
    require_kaggle(politica_privacidade(root))
    aprovacao=home(root)/'aprovacoes'/(spec_hash+'.json')
    if not aprovacao.is_file() or ler_json(aprovacao).get('spec_hash')!=spec_hash: raise ValueError('APPROVAL_REQUIRED')
    if spec['timeout_seconds']>LIMITE_JOB:
        raise ValueError('KAGGLE_TIMEOUT_TOO_LONG: a sessao GPU do Kaggle dura ~9 h; use --timeout ate %d no engine-spec' % LIMITE_JOB)
    folder=Store(root).run_dir(run_id); path=folder/'remote-job.json'
    # um kernel por run id: enviar de novo ao mesmo kernel roda o treino outra vez
    # titulo igual ao slug, sem hifen duplo: senao o Kaggle cria o kernel com outro endereco
    kernel=api.usuario()+'/'+('itsfine-'+slug(run_id)[:30].strip('-')+'-'+digest({'run_id':run_id,'spec_hash':spec_hash})[:8])
    record={'schema_version':1,'backend':'kaggle','run_id':run_id,'spec_hash':spec_hash,'kernel':kernel,
            'dataset':registro['dataset'],'checkpoints':bool(checkpoints),'status':'submetendo'}
    if path.exists():
        previous=ler_json(path)
        if previous['spec_hash']!=spec_hash: raise ValueError('RUN_INPUTS_FROZEN')
        if previous.get('status')!='submetendo': return previous
        # resposta perdida no envio anterior: so envia se o kernel nao chegou ao Kaggle
        if api.kernel_status(kernel) is not None:
            record={**previous,'status':'enviado','url':'https://www.kaggle.com/code/'+kernel}; atomic_json(path,record); return record
    # dataset apagado por limpar: o kernel gastaria cota de GPU so para falhar nas entradas
    if api.dataset_status(registro['dataset'])!='ready': raise ValueError('KAGGLE_DATASET_MISSING: rode push de novo')
    if not path.exists(): atomic_json(path,record)  # intencao gravada antes do envio
    pasta=home(root)/'kernels'/kernel.split('/')[1]; pasta.mkdir(parents=True,exist_ok=True)
    (pasta/'itsfine_kaggle.py').write_text(script(run_id,spec,ler_json(aprovacao)['actor'],checkpoints),encoding='utf-8')
    gravar_json(pasta/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],'code_file':'itsfine_kaggle.py',
        'language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,
        'machine_shape':GPU,'dataset_sources':[registro['dataset']],'competition_sources':[],'kernel_sources':[],'model_sources':[]})
    resposta=api.enviar_kernel(pasta,int(spec['timeout_seconds']+FOLGA))
    record={**record,**resposta,'status':'enviado'}; atomic_json(path,record)
    return record

def job(root,run_id):
    path=Store(root).run_dir(run_id)/'remote-job.json'
    if not path.is_file(): raise ValueError('KAGGLE_JOB_NOT_SUBMITTED')
    record=ler_json(path)
    if record.get('backend')!='kaggle': raise ValueError('NOT_A_KAGGLE_JOB')
    return path,record

def linhas(log):
    """So as linhas ITSFINE| do log, que trazem numeros de progresso; o resto pode citar dados."""
    try: texto=''.join(e.get('data','') for e in json.loads(log) if isinstance(e,dict))
    except (ValueError,TypeError): texto=log
    result=[]
    for linha in texto.splitlines():
        if linha.startswith(PREFIXO):
            try: result.append(json.loads(linha[len(PREFIXO):]))
            except ValueError: pass
    return result

def status(root,run_id,api,completo=False):
    path,record=job(root,run_id)
    estado=api.kernel_status(record['kernel'])
    if estado is None: return {'run_id':run_id,'status':'inexistente','kernel':record['kernel']}
    result={'run_id':run_id,'kaggle':estado['status'],'url':record.get('url')}
    if completo and estado['falha']: result['falha']=estado['falha']
    try: ultimas=linhas(api.log(record['kernel']))
    except Exception: ultimas=[]  # log ao vivo nem sempre disponivel: o estado do kernel basta
    if ultimas: result['ultima']=ultimas[-1]
    atomic_json(path,{**record,'kaggle':estado['status']})
    return result

def acompanhar(root,run_id,api,intervalo=60):
    """Uma linha por mudanca, ate o kernel terminar; feito para o monitor do chat."""
    ultima=None; falhas=0
    while True:
        try: estado=status(root,run_id,api); falhas=0
        except Exception:
            falhas+=1
            if falhas>=20: raise
            time.sleep(intervalo); continue
        etapa=estado.get('ultima') or {}; kaggle=estado['kaggle']
        if kaggle in ('queued','new_script'): texto='Na fila do Kaggle...'
        elif etapa.get('etapa')=='erro': texto='Erro no Kaggle: %s em %s' % (etapa.get('erro'),(etapa.get('local') or ['?'])[-1])
        elif kaggle=='running':
            texto=etapa.get('barra') or ('Instalando os pacotes no Kaggle...' if etapa.get('etapa')=='instalando' else 'Rodando no Kaggle...')
        elif kaggle=='complete': texto='Kaggle terminou: traga o resultado com pull.'
        else: texto='Kaggle: '+kaggle
        if texto!=ultima: print(time.strftime('%H:%M'),texto,flush=True); ultima=texto
        if estado['kaggle'] not in ('queued','running','new_script'): return estado
        time.sleep(intervalo)

def pull(root,run_id,spec_hash,target,api):
    path,record=job(root,run_id)
    if record['spec_hash']!=spec_hash: raise ValueError('RESULT_IDENTITY_MISMATCH')
    estado=api.kernel_status(record['kernel'])
    if estado is None or estado['status'] in ('queued','running','new_script'): raise ValueError('KAGGLE_JOB_NOT_FINISHED')
    arquivos=api.arquivos(record['kernel'])
    if RECIBO not in arquivos: raise ValueError('KAGGLE_RESULT_MISSING: a sessao terminou antes de empacotar')
    pasta=home(root)/'retorno'/run_id; pasta.mkdir(parents=True,exist_ok=True)
    api.baixar(arquivos[RECIBO],pasta/RECIBO); recibo=ler_json(pasta/RECIBO)
    if recibo.get('run_id')!=run_id or recibo.get('spec_hash')!=spec_hash: raise ValueError('RESULT_IDENTITY_MISMATCH')
    if recibo['status']=='erro':
        atomic_json(path,{**record,'status':'erro_no_kaggle'})
        return {k:recibo.get(k) for k in ('status','etapa','erro','local')}
    alvo=Path(target)
    if not (alvo.exists() and sha256_arquivo(alvo)==recibo['sha256']):
        if alvo.exists(): raise FileExistsError('RESULT_DESTINATION_EXISTS')
        inicio=time.monotonic(); temp=alvo.with_name(alvo.name+'.download')
        api.baixar(arquivos[RESULTADO],temp)
        if sha256_arquivo(temp)!=recibo['sha256']: temp.unlink(); raise ValueError('DOWNLOAD_HASH_MISMATCH')
        os.replace(temp,alvo); segundos=round(time.monotonic()-inicio,1)
    else: segundos=0
    result=import_result(root,alvo,run_id,spec_hash)
    atomic_json(path,{**record,'status':'importado'})
    return {**result,'mb':round(recibo['bytes']/1024**2,1),'segundos_download':segundos}

def limpar(root,run_id,api):
    """Apaga no Kaggle o dataset com os inputs e o kernel com o resultado, depois do resultado importado."""
    path,record=job(root,run_id)
    if record.get('status') not in ('importado','erro_no_kaggle'): raise ValueError('KAGGLE_RESULT_NOT_IMPORTED')
    # outro job ainda pendente pode usar o mesmo dataset (mesma proposta remota)
    for outro in Store(root).runs.glob('*/remote-job.json'):
        valor=ler_json(outro)
        if outro!=path and valor.get('dataset')==record['dataset'] and valor.get('status') not in ('importado','erro_no_kaggle','limpo'):
            raise ValueError('KAGGLE_DATASET_IN_USE:'+outro.parent.name)
    if api.dataset_status(record['dataset']) is not None: api.apagar_dataset(record['dataset'])
    if api.kernel_status(record['kernel']) is not None: api.apagar_kernel(record['kernel'])
    atomic_json(path,{**record,'status':'limpo'})
    return {'apagados':[record['dataset'],record['kernel']]}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.')
    sub=p.add_subparsers(dest='cmd',required=True)
    prep=sub.add_parser('preparar',help='instala o pacote kaggle na venv e confere a conta')
    prep.add_argument('--autorizado',action='store_true',help='o usuario autorizou o download do pacote')
    for name in ('push','approve','run','status','acompanhar','pull','cancel','limpar'):
        child=sub.add_parser(name)
        if name in ('push','approve','run'): child.add_argument('--spec',required=True)
        if name in ('run','status','acompanhar','pull','cancel','limpar'): child.add_argument('--run-id',required=True)
        if name in ('push','pull'): child.add_argument('--saida',required=True)
        if name=='approve': child.add_argument('--actor',required=True)
        if name=='pull': child.add_argument('--spec-hash',required=True)
        # a VM some no fim da sessao: sem os checkpoints no resultado, treino interrompido nao retoma
        if name=='run': child.add_argument('--sem-checkpoints',action='store_true',help='deixa de fora os checkpoints por epoca (sondagem)')
    args=p.parse_args()
    if args.cmd=='preparar':
        # allow_kaggle ja cobre instalar o pacote da API do Kaggle: nao ha o que perguntar de novo
        result=preparar(args.autorizado or politica_privacidade(args.root).get('allow_kaggle') is True)
    elif args.cmd=='approve': result=approve(args.root,args.spec,args.actor)
    elif args.cmd=='cancel':
        _,record=job(args.root,args.run_id)
        result={'cancelavel_pela_api':False,'url':record.get('url') or 'https://www.kaggle.com/code/'+record['kernel'],
                'acao':'o usuario abre o link e para a execucao na pagina do notebook; o resultado dessa execucao se perde'}
    else:
        api=Kaggle()
        if args.cmd=='push': result=push(args.root,ler_json(args.spec),args.saida,api)
        elif args.cmd=='run': result=run(args.root,args.spec,args.run_id,api,not args.sem_checkpoints)
        elif args.cmd=='status':
            result=status(args.root,args.run_id,api,politica_privacidade(args.root)['allow_data_to_harness'] is True)
        elif args.cmd=='acompanhar': acompanhar(args.root,args.run_id,api); return 0
        elif args.cmd=='pull': result=pull(args.root,args.run_id,args.spec_hash,args.saida,api)
        elif args.cmd=='limpar': result=limpar(args.root,args.run_id,api)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
