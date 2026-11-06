"""CLI shared by installed skills and remote/notebook workspaces."""
from pathlib import Path
import argparse
import json
import sys
from .storage import Store, atomic_json, digest, read_json

def parser():
    p = argparse.ArgumentParser(prog='fine', description='ITS-FINE runtime')
    p.add_argument('--root', default='.')
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init'); init.add_argument('--user', default='user')
    new = sub.add_parser('new'); new.add_argument('run_id')
    for command in ('route','artifacts','status','cancel','acompanhar'):
        child = sub.add_parser(command); child.add_argument('run_id')
    put = sub.add_parser('put'); put.add_argument('run_id'); put.add_argument('artifact')
    validate = sub.add_parser('validate'); validate.add_argument('bundle')
    hash_cmd = sub.add_parser('hash'); hash_cmd.add_argument('artifact')
    ds = sub.add_parser('dataset'); ds.add_argument('source'); ds.add_argument('destination', help='pasta nova dentro de _itsfine_runs/, ex.: <run_id>/prepared-v1'); ds.add_argument('--seed', type=int, default=42)
    ds.add_argument('--tarefa', help='task.json das linhas tipadas; padrao: ao lado da fonte')
    ds.add_argument('--teste', help='test.jsonl declarado pelo projeto (linhas tipadas)')
    ds.add_argument('--fracoes', help='fracoes por particao, ex.: 0.7,0.1,0.1,0.1')
    ds.add_argument('--estratificar', help='saida de task.json cujas opcoes devem aparecer em todas as particoes (linhas tipadas)')
    ds.add_argument('--base', help='continuacao: pasta preparada da rodada mae; validacao, calibracao e teste ficam identicos e os dados novos entram so no treino')
    ah = sub.add_parser('adapter-hash'); ah.add_argument('adapter', help='pasta do adapter; o valor vai para start_from.adapter_hash da receita')
    approve = sub.add_parser('approve'); approve.add_argument('spec'); approve.add_argument('--actor', required=True); approve.add_argument('--acknowledge-process', action='store_true')
    submit = sub.add_parser('submit'); submit.add_argument('run_id'); submit.add_argument('spec')
    nb = sub.add_parser('notebook'); nb.add_argument('run_id'); nb.add_argument('spec'); nb.add_argument('destination')
    pack = sub.add_parser('pack'); pack.add_argument('run_id'); pack.add_argument('destination')
    imp = sub.add_parser('import'); imp.add_argument('run_id'); imp.add_argument('archive'); imp.add_argument('--spec-hash', required=True)
    remote = sub.add_parser('remote'); remote.add_argument('action', choices=['status','submit','cancel']); remote.add_argument('run_id'); remote.add_argument('--host', required=True); remote.add_argument('--remote-root', required=True); remote.add_argument('--spec-path', default='execution.json')
    eng=sub.add_parser('engine-spec'); eng.add_argument('run_id'); eng.add_argument('--recipe',required=True); eng.add_argument('--data',required=True)
    eng.add_argument('--executor',choices=['process','docker'],default='docker'); eng.add_argument('--image')
    eng.add_argument('--timeout',type=int,default=3600); eng.add_argument('--probe',action='store_true'); eng.add_argument('--checkpoint')
    sub.add_parser('engines')
    kt = sub.add_parser('kaggle-token', help='salva ou troca o token do Kaggle, digitado oculto no seu terminal')
    kt.add_argument('--remover', action='store_true')
    sub.add_parser('permissoes', help='abre uma tela no navegador para ligar e desligar as permissoes')
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'kaggle-token':
        from .kaggle_token import executar
        return executar(args.remover)
    if args.command == 'permissoes':
        from .permissoes import abrir
        return abrir(args.root)
    store = Store(args.root)
    try:
        if args.command == 'init': result = store.initialize(args.user)
        elif args.command == 'new':
            state = store.state()
            store.run_dir(args.run_id).mkdir(parents=True, exist_ok=False)
            result = store.update_state(state['revision'], active_run_id=args.run_id)
        elif args.command == 'route': result = store.route(args.run_id)
        elif args.command == 'artifacts': result = store.artifacts(args.run_id)
        elif args.command == 'put': result = store.put(args.run_id, read_json(args.artifact))
        elif args.command == 'hash': result = {'hash':digest(read_json(args.artifact))}
        elif args.command == 'validate':
            from .contracts.validate import validate_bundle
            result = validate_bundle(read_json(args.bundle))
        elif args.command == 'dataset':
            from .dataset import prepare
            from .storage import confined
            from .privacy import require_local_executor, load_policy
            require_local_executor(load_policy(store.root))
            options={}
            if args.tarefa: options['task']=args.tarefa
            if args.teste: options['declared_test']=args.teste
            if args.fracoes: options['fractions']=tuple(float(x) for x in args.fracoes.split(','))
            if args.estratificar: options['stratify']=args.estratificar
            # folders are relative to _itsfine_runs/; a path already starting there is accepted as is
            def in_runs(value):
                value=value.replace('\\','/')
                if not Path(value).is_absolute() and value.startswith('_itsfine_runs/'): value=value[len('_itsfine_runs/'):]
                return confined(store.runs, value)
            if args.base:
                if options.keys()-{'task'}: raise ValueError('BASE_KEEPS_PARTITIONS: --teste, --fracoes and --estratificar do not apply')
                from .dataset import extend
                result = extend(in_runs(args.base), Path(args.source), in_runs(args.destination), options.get('task'))
            else:
                result = prepare(Path(args.source), in_runs(args.destination), args.seed, **options)
            result={'schema_version':result['schema_version'],'duplicate_rows_removed':result['duplicate_rows_removed'],
                    'splits':{k:{f:v[f] for f in ('hash','rows','group_count')} for k,v in result['splits'].items()},
                    **({'suporte_insuficiente':len(result['suporte_insuficiente']),'task_hash':result['task_hash']} if 'task_hash' in result else {}),
                    **({'estratificacao_faltando':len(result['estratificacao']['faltando'])} if 'estratificacao' in result else {}),
                    **({'continuacao':{k:result[k] for k in ('rows_added','rows_removed_present_in_held_out')}} if 'base' in result else {})}
        elif args.command == 'adapter-hash':
            from .continuation import adapter_hash
            result={'adapter_hash':adapter_hash(args.adapter)}
        elif args.command == 'engine-spec':
            from .engine_job import build
            result=build(store.root,args.run_id,args.recipe,args.data,args.executor,args.image,args.timeout,args.probe,args.checkpoint)
        elif args.command == 'engines':
            from . import engines
            setup=store.home/'engine-setup.json'
            ready=read_json(setup).get('engines_prontos') if setup.exists() else []
            result={name:{**engines.capabilities(store.home/'engines'/name),'preparado':name in (ready or [])} for name in engines.installed(store)}
        elif args.command == 'acompanhar':
            # uma linha de barra por mudanca, ate o job terminar; feito para o monitor do chat
            from .execution import status
            from .progress import follow
            follow(lambda: status(store.root, args.run_id))
            return 0
        elif args.command in ('approve','submit','status','cancel'):
            from .execution import authorize, submit, status, cancel
            if args.command == 'approve': result = authorize(store.root, read_json(args.spec), args.actor, args.acknowledge_process)
            elif args.command == 'submit': result = submit(store.root, args.run_id, read_json(args.spec))
            elif args.command == 'status':
                from .privacy import load_policy
                result = status(store.root, args.run_id, full=load_policy(store.root)['allow_data_to_harness'] is True)
            else: result = cancel(store.root, args.run_id)
        else:
            from .transport import notebook, pack_result, import_result, remote_request
            if args.command == 'notebook': result = notebook(store.root, args.run_id, read_json(args.spec), args.destination)
            elif args.command == 'pack': result = pack_result(store.root, args.run_id, args.destination)
            elif args.command == 'import': result = import_result(store.root, args.archive, args.run_id, args.spec_hash)
            else: result = remote_request(args.host, args.remote_root, args.action, args.run_id, args.spec_path)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'error':str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

if __name__ == '__main__': raise SystemExit(main())
