"""Validate metadata only: this tool never executes training or verifies remote I/O."""
from pathlib import Path
import hashlib
import json
import math
import sys
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / 'artifact.schema.json').read_text(encoding='utf-8'))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)
ENCODER_SCHEMA = json.loads((HERE / 'encoder.schema.json').read_text(encoding='utf-8'))
Draft202012Validator.check_schema(ENCODER_SCHEMA)
DEPENDENCIES = {
    'objective': (), 'environment': ('objective',), 'model': ('objective', 'environment'),
    'dataset': ('objective', 'environment', 'model'),
    'recipe': ('objective', 'environment', 'model', 'dataset'), 'run': ('recipe',),
    'evaluation': ('objective', 'dataset', 'model', 'run'),
    'export': ('evaluation', 'recipe'), 'delivery': ('export', 'evaluation'),
}
PARTITIONS = {1: ('train', 'validation', 'test'), 2: ('train', 'validation', 'calibration', 'test')}

def artifact_hash(artifact):
    return hashlib.sha256(json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def require(condition, code):
    if not condition: raise ValueError(code)

def validate_artifact(artifact):
    VALIDATOR.validate(artifact)
    artifact_hash(artifact)  # reject non-finite JSON numbers, including nested values

def validate_encoder(kind, value):
    """Validate a typed-decision file (task, row, mapping, recipe, calibration, prediction, package)."""
    require(kind in ENCODER_SCHEMA['$defs'], 'UNKNOWN_ENCODER_CONTRACT:' + str(kind))
    Draft202012Validator({**ENCODER_SCHEMA, **ENCODER_SCHEMA['$defs'][kind]}).validate(value)
    return {'valid': True, 'contract': kind}

def evaluation_decision(objective, baseline, candidate, sample_count, parent=None):
    """parent: score of the approved model a continuation started from; the candidate may not fall below it."""
    require(math.isfinite(baseline) and math.isfinite(candidate) and (parent is None or math.isfinite(parent)), 'NONFINITE_METRIC')
    if sample_count < objective['minimum_samples']: return 'inconclusiva'
    sign = 1 if objective['direction'] == 'maximize' else -1
    if parent is not None and sign * (candidate - parent) < 0: return 'reprovada'
    return 'aprovada' if (sign * candidate >= sign * objective['threshold'] and
                          sign * (candidate - baseline) >= objective['minimum_gain']) else 'reprovada'

def evaluation_decision_v2(objective, evaluation):
    """Typed decisions: insufficient support is inconclusive; a violated limit rejects before the metric."""
    checked = {item['nome']: item for item in evaluation.get('limits_checked', [])}
    require(set(checked) == set(objective.get('limits', {})), 'LIMITS_NOT_CHECKED')
    for name, item in checked.items():
        require(item['limite'] == objective['limits'][name], 'LIMIT_VALUE_MISMATCH:' + name)
    if not evaluation.get('support_sufficient', False): return 'inconclusiva'
    if any(not item['ok'] for item in checked.values()): return 'reprovada'
    return evaluation_decision(objective, evaluation['baseline'], evaluation['candidate'], evaluation['sample_count'],
                               evaluation.get('start_from', {}).get('value'))

def resume_action(run, observation):
    """Pure decision; an executor must supply a trustworthy observation."""
    validate_artifact(run)
    require(run['kind'] == 'run', 'INVALID_RUN')
    require(observation in ('active', 'completed', 'stopped', 'unreachable', 'absent'), 'INVALID_OBSERVATION')
    data = run['data']
    if observation == 'unreachable': return 'aguardar-reconciliacao'
    if observation == 'active':
        require(bool(data.get('job_id')), 'JOB_UNKNOWN')
        require(data['status'] not in ('concluida', 'falhou', 'interrompida'), 'JOB_STATE_CONFLICT')
        return 'observar-job'
    if observation == 'completed': return 'avaliar'
    if data['status'] == 'planejada' and observation == 'absent':
        require(data['approved_recipe_hash'] == run['input_hashes'].get('recipe'), 'APPROVAL_REQUIRED')
        return 'persistir-intencao-e-submeter'
    if data['status'] in ('submetendo', 'desconhecida', 'executando'):
        return 'aguardar-reconciliacao'
    if data['status'] == 'concluida': return 'avaliar'
    cp = data.get('checkpoint', {})
    require(cp.get('complete') and cp.get('verified') and cp.get('engine_state_present') and
            cp.get('recipe_hash') == run['input_hashes'].get('recipe'), 'CHECKPOINT_INVALID')
    return 'nova-rodada-do-checkpoint'

def validate_bundle(bundle):
    require(isinstance(bundle, dict) and set(bundle) == set(DEPENDENCIES), 'INVALID_BUNDLE')
    seen = set()
    for kind, artifact in bundle.items():
        validate_artifact(artifact)
        require(artifact['kind'] == kind and artifact['id'] not in seen, 'INVALID_IDENTITY')
        seen.add(artifact['id'])
    versions = {artifact['schema_version'] for artifact in bundle.values()}
    require(len(versions) == 1, 'MIXED_SCHEMA_VERSIONS')
    version = versions.pop()
    for kind, deps in DEPENDENCIES.items():
        expected = {dep: artifact_hash(bundle[dep]) for dep in deps}
        require(bundle[kind]['input_hashes'] == expected, 'STALE_INPUT:' + kind)

    goal, env, model, ds, recipe, run, ev, export, delivery = [bundle[k]['data'] for k in DEPENDENCIES]
    require(env['protection_verified'], 'PROTECTION_NOT_VERIFIED')
    require(not env['metered'] or ('max_cost' in env and 'currency' in env), 'BUDGET_REQUIRED')
    require(ds['verified_at_destination'] and ds['destination'] == env['destination'], 'DESTINATION_UNVERIFIED')
    if env['transfer_mode'] == 'none':
        require(ds['source'] == ds['destination'], 'TRANSFER_NOT_AUTHORIZED')
    elif env['transfer_mode'] == 'authorized':
        a = env.get('transfer_approval', {})
        require(a.get('action') == 'transfer' and a.get('subject_hash') == ds['content_hash'] and
                a.get('destination') == ds['destination'] and a.get('source') == ds['source'] and
                a.get('byte_count') == ds['byte_count'], 'TRANSFER_NOT_AUTHORIZED')

    require(ds['frozen'], 'TEST_NOT_FROZEN')
    require(ds['tokenizer_revision'] == model['tokenizer_revision'] and
            ds['template_hash'] == model['template_hash'], 'TOKENIZER_MISMATCH')
    names = PARTITIONS[version]
    for field in ('ids', 'groups'):
        splits = [set(ds['splits'][name][field]) for name in names]
        require(all(not splits[i] & splits[j] for i in range(len(names)) for j in range(i + 1, len(names))), 'SPLIT_LEAKAGE')
    profile = model['profile']
    require(model['viability'] == 'aprovada' and profile['verified'], 'UNSUPPORTED_PROFILE')
    require(profile['hardware'] == env['hardware'] and profile['engine'] == recipe['engine'] and
            profile['version'] == recipe['engine_version'], 'PROFILE_MISMATCH')
    require(set(recipe['parameters']) <= set(profile['training_parameters']), 'UNSUPPORTED_PARAMETER')
    require(run['status'] == 'concluida' and run.get('job_id') and run.get('model_hash'), 'TRAINING_INCOMPLETE')
    require(run['approved_recipe_hash'] == artifact_hash(bundle['recipe']), 'APPROVAL_REQUIRED')
    require(ev['test_hash'] == ds['splits']['test']['hash'], 'TEST_CHANGED')
    require(ev['protocol_hash'] == ev['baseline_protocol_hash'] == ev['candidate_protocol_hash'] ==
            goal['protocol_hash'], 'PROTOCOL_MISMATCH')
    # a continuation is compared with the approved model it started from, on the same test and protocol
    if 'start_from' in recipe:
        parent = ev.get('start_from', {})
        require(parent.get('run_id') == recipe['start_from']['run_id'] and
                parent.get('adapter_hash') == recipe['start_from']['adapter_hash'] and
                parent.get('test_hash') == ev['test_hash'] and parent.get('protocol_hash') == goal['protocol_hash'],
                'START_FROM_NOT_EVALUATED')
    else:
        require('start_from' not in ev, 'START_FROM_NOT_IN_RECIPE')
    indexed=any('index' in split for split in ds['splits'].values())
    if indexed:
        require('split_manifest_hash' in ds and all('index' in split and not split['ids'] and not split['groups']
                for split in ds['splits'].values()), 'INCOMPLETE_SPLIT_INDEX')
    sample_count=ds['splits']['test']['index']['row_count'] if indexed else len(ds['splits']['test']['ids'])
    require(ev['sample_count'] == sample_count, 'SAMPLE_COUNT_MISMATCH')
    if version == 2:
        require(goal['task_hash'] == ds['task_hash'] == recipe['task_hash'], 'TASK_MISMATCH')
        require(recipe['engine'] == 'hf-encoder' and recipe['profile'] == 'escolha_por_marcador', 'PROFILE_MISMATCH')
        require(ev['calibration_hash'] == export['calibration_hash'], 'CALIBRATION_MISMATCH')
        require(model['parity'] == 'certificada', 'PARITY_NOT_CERTIFIED')
        decision = evaluation_decision_v2(goal, ev)
    else:
        decision = evaluation_decision(goal, ev['baseline'], ev['candidate'], ev['sample_count'],
                                       ev.get('start_from', {}).get('value'))
    require(ev['decision'] == decision, 'INVALID_EVALUATION_DECISION')
    require(decision == 'aprovada', 'EVALUATION_REJECTED')
    require(export['loaded'] and export['inference_passed'], 'EXPORT_INVALID')
    require(export['protocol_hash'] == goal['protocol_hash'] and
            export['template_hash'] == model['template_hash'] and
            export['tokenizer_revision'] == model['tokenizer_revision'], 'EXPORT_PROTOCOL_MISMATCH')
    sign = 1 if goal['direction'] == 'maximize' else -1
    require(sign * (ev['candidate'] - export['metric_value']) <= goal['export_tolerance'], 'EXPORT_REGRESSION')
    require(sign * export['metric_value'] >= sign * goal['threshold'], 'EXPORT_BELOW_THRESHOLD')
    require(delivery['package_hash'] == export['package_hash'], 'PACKAGE_MISMATCH')
    if delivery['mode'] != 'local':
        a = delivery.get('approval', {})
        require(a.get('action') == delivery['mode'] and a.get('subject_hash') == export['package_hash'] and
                a.get('destination') == delivery['destination'] and a.get('visibility') == delivery['visibility'],
                'PUBLICATION_NOT_AUTHORIZED')
    return {'valid': True, 'scope': 'declared-metadata-only', 'artifacts': len(bundle), 'schema_version': version}

if __name__ == '__main__':
    try:
        require(len(sys.argv) == 2, 'Usage: validate.py <bundle.json>')
        bundle = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        print(json.dumps(validate_bundle(bundle)))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
