"""Exporta o pacote de decisao: pesos mesclados, tokenizer, tarefa, calibrador, politica e manifesto.

Qualquer que seja a origem (encoder comum ou checkpoint de decisao), o pacote tem o
layout proprio do motor: itsfine-model.json, model.safetensors, encoder/config.json e
tokenizer/, mais task.json, calibration.json, policy.json e itsfine-package.json. A
sanidade recarrega o pacote em processo novo pelo predict.py, sem codigo externo.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AQUI, gravar_json, ler_json, sha256_arquivo, sha256_json, politica_privacidade  # noqa: E402
import decision_model  # noqa: E402
import loader  # noqa: E402
from sequence import SEQUENCE_VERSION  # noqa: E402
from itsfine.privacy import require_local_executor  # noqa: E402


def versoes():
    import torch, transformers, peft, safetensors, numpy
    return {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
            'safetensors': safetensors.__version__, 'numpy': numpy.__version__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='pasta do checkpoint de origem')
    parser.add_argument('--checkpoint-id'); parser.add_argument('--checkpoint-revision')
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--calibracao', required=True, help='calibration.json do candidato')
    parser.add_argument('--tarefa', help='task.json; padrao: o do adapter')
    parser.add_argument('--saida', required=True)
    parser.add_argument('--exemplo', default='Exemplo de estado para a sanidade do pacote.', help='estado (texto ou JSON) para a sanidade')
    args = parser.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    import torch
    saida = Path(args.saida).resolve()
    if saida.exists() and any(saida.iterdir()): raise FileExistsError('EXPORT_OUTPUT_NOT_EMPTY')
    saida.mkdir(parents=True, exist_ok=True)
    origem = Path(args.checkpoint)
    adapter = Path(args.adapter)
    tarefa = Path(args.tarefa) if args.tarefa else adapter / 'task.json'
    calib = ler_json(args.calibracao)
    adapter_hash = sha256_json({'adapter': sha256_arquivo(adapter / 'adapter_model.safetensors'), 'heads': sha256_arquivo(adapter / 'heads.safetensors')})
    if calib.get('weights_hash') not in (None, adapter_hash): raise ValueError('CALIBRATION_WEIGHTS_MISMATCH')
    verificar = {'id': args.checkpoint_id, 'revision': args.checkpoint_revision} if args.checkpoint_id else None

    print('mesclando na CPU...', flush=True)
    carga = loader.carregar(origem, device='cpu', verificar=verificar, adapter=str(adapter))
    modelo, cfg, desc = carga['modelo'], carga['cfg'], carga['desc']
    modelo.encoder = modelo.encoder.merge_and_unload()
    modelo.temperature.copy_(torch.tensor([float(x) for x in calib['temperature']]))
    decision_model.salvar_pesos(modelo, saida / 'model.safetensors')
    base_tokenizer = Path(desc['tokenizer_dir'])
    for arquivo in loader.arquivos_tokenizer(desc):
        destino = saida / 'tokenizer' / arquivo.relative_to(base_tokenizer)
        destino.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(arquivo, destino)
    (saida / 'encoder').mkdir(); shutil.copy2(Path(desc['encoder_dir']) / 'config.json', saida / 'encoder/config.json')
    cfg_pacote = {**cfg, 'temperature': [float(x) for x in calib['temperature']], 'temperature_by_options': calib.get('temperature_by_options', {}),
                  'training': {**cfg.get('training', {}), 'fine_tuned_from_checkpoint': True, 'itsfine_engine': 'hf-encoder',
                               'origem_layout': desc['layout'], 'cabecas_novas': desc['cabecas_novas']}}
    gravar_json(saida / 'itsfine-model.json', cfg_pacote)
    task = ler_json(tarefa)
    gravar_json(saida / 'task.json', task)
    gravar_json(saida / 'calibration.json', calib)
    gravar_json(saida / 'policy.json', {'decide': task.get('decide'), 'policy': task.get('policy')})
    del modelo
    arquivos = sorted(p for p in saida.rglob('*') if p.is_file())
    files = {p.relative_to(saida).as_posix(): sha256_arquivo(p) for p in arquivos}
    manifesto = {'schema_version': 1, 'engine': 'hf-encoder', 'engine_version': ler_json(AQUI / 'engine.json')['version'],
                 'sequence_version': SEQUENCE_VERSION, 'origem': {'id': args.checkpoint_id, 'revision': args.checkpoint_revision,
                                                                  'hash': carga['identidade']['package_id'] if carga['manifesto'] is None else None},
                 'adapter_hash': adapter_hash, 'calibration_hash': calib.get('hash'), 'task_hash': sha256_arquivo(tarefa),
                 'dependencies': versoes(), 'files': files, 'package_id': sha256_json(files)}
    gravar_json(saida / loader.MANIFESTO, manifesto)
    relatorio = {'schema_version': 1, 'engine': 'hf-encoder', 'formato': 'pacote-decisao-tipada (safetensors fp16 mesclado)',
                 'pacote': str(saida), 'package_id': manifesto['package_id'], 'arquivos': len(files),
                 'bytes': sum(p.stat().st_size for p in arquivos), 'checksums': files, 'consumidores': {},
                 'adapter_hash': adapter_hash, 'calibration_hash': calib.get('hash')}

    print('sanidade: recarregando o pacote em processo novo...', flush=True)
    sanidade = saida.parent / (saida.name + '-sanidade.json')
    cmd = [sys.executable, str(AQUI / 'predict.py'), '--pacote', str(saida), '--estado', args.exemplo, '--saida', str(sanidade), '--device', 'cpu', '--sem-laya']
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    ok = run.returncode == 0 and sanidade.is_file()
    info = {'ok': ok, 'uso': 'python %s --pacote %s --estado <texto ou JSON>' % ((AQUI / 'predict.py').as_posix(), saida.as_posix())}
    if ok:
        resultado = ler_json(sanidade)
        info['carregou'] = resultado.get('package_id') == manifesto['package_id']
        info['respostas'] = len(resultado.get('answers', {}))
        info['sem_laya'] = resultado.get('sem_laya')
        sanidade.unlink()
    else:
        info['log'] = (run.stdout + run.stderr)[-300:]
    relatorio['consumidores']['predict'] = info
    relatorio['consumidores']['servidor'] = {'ok': None, 'uso': 'python %s --pacote %s' % ((AQUI / 'serve.py').as_posix(), saida.as_posix()),
                                             'nota': 'reavalie com evaluate.py --servidor http://127.0.0.1:<porta>'}
    relatorio['qualidade_final'] = 'pendente: reavaliar pelo consumidor entregue (evaluate.py --pacote ou --servidor)'
    # Fora do manifesto: o pacote e os arquivos listados; o relatorio acompanha a pasta.
    gravar_json(saida / 'export.json', relatorio)
    resumo = {k: v for k, v in relatorio.items() if k != 'checksums'}
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0 if info['ok'] and info.get('carregou') else 2


if __name__ == '__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main, 'export.json'))
