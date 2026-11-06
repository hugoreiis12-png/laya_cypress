"""Typed decision rows: dedupe, conflicts, group reconciliation and four group-disjoint partitions.

Row: {"id", "group_id", "state", "answers": {output: value}, "meta"?}. The task
(questions, decide, policy) lives in task.json and is copied next to the splits.
A declared test file becomes the test partition; its groups leave the others.
"""
from pathlib import Path
import hashlib
import json
import unicodedata
from .storage import atomic_json, digest, file_hash, lock, read_json

PARTITIONS = ('train', 'validation', 'calibration', 'test')
MIN_SUPPORT = 5


def _state_key(state):
    return hashlib.sha256(unicodedata.normalize('NFC', json.dumps(state, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def _options(question):
    t = question.get('type')
    if t == 'choice':
        crit = question.get('criteria')
        return list(crit.keys()) if isinstance(crit, dict) else [str(c) for c in crit]
    if t == 'score': return [str(i) for i in range(len(question.get('criteria', [])))]
    return ['false', 'true']


def _option_of(question, value):
    """Canonical option name of an answer value, validated against the question."""
    t = question.get('type')
    if t == 'choice':
        if not isinstance(value, str) or value not in _options(question): raise ValueError('LABEL_NOT_AN_OPTION')
        return value
    if t == 'score':
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(question.get('criteria', [])): raise ValueError('LABEL_NOT_A_LEVEL')
        return str(value)
    if isinstance(value, bool): return 'true' if value else 'false'
    if value in (0, 1): return 'true' if value else 'false'
    if isinstance(value, str) and value.lower() in ('true', 'false'): return value.lower()
    raise ValueError('LABEL_NOT_BOOLEAN')


def _read_rows(source, questions, line_offset=0):
    rows = []
    with Path(source).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row, dict) or 'state' not in row: raise ValueError(f'ROW_NOT_TYPED:{number}')
            if not isinstance(row['state'], (str, dict, list)) or row['state'] in ('', {}, []): raise ValueError(f'STATE_INVALID:{number}')
            answers = row.get('answers', {})
            if not isinstance(answers, dict): raise ValueError(f'ANSWERS_NOT_OBJECT:{number}')
            for name, value in answers.items():
                if name not in questions: raise ValueError(f'ANSWER_UNKNOWN_OUTPUT:{number}:{name}')
                if value is not None: _option_of(questions[name], value)
            if 'questions' in row and row['questions'] != questions: raise ValueError(f'QUESTIONS_MISMATCH:{number}')
            rows.append({'row': row, 'line': number})
    return rows


class _Union:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[max(ra, rb)] = min(ra, rb)


def _stratified(ranked, names, counts, kept, question):
    """Group assignment that puts every option of one output in each partition when groups allow.

    Rarest option first: one group for train, then one for each other partition, always
    group-disjoint. The remaining groups fill the other partitions to their quota in hash
    order and the rest go to train. Options without enough groups are reported, not forced.
    """
    options_by_group = {}
    for record in kept:
        option = record['answers'].get(question['name'])
        if option is not None: options_by_group.setdefault(record['group'], set()).add(option)
    with_option = {opt: [g for g in ranked if opt in options_by_group.get(g, ())] for opt in _options(question['value'])}
    assignment, missing = {}, []
    for opt in sorted(with_option, key=lambda o: (len(with_option[o]), o)):
        for name in names:
            if any(assignment.get(g) == name for g in with_option[opt]): continue
            free = [g for g in with_option[opt] if g not in assignment]
            if free: assignment[free[0]] = name
            else: missing.append({'split': name, 'opcao': opt})
    rest = [g for g in ranked if g not in assignment]
    for name, count in zip(names[1:], counts[1:]):
        need = max(0, count - sum(1 for v in assignment.values() if v == name))
        for g in rest[:need]: assignment[g] = name
        rest = rest[need:]
    for g in rest: assignment[g] = 'train'
    return assignment, missing


def prepare(source, destination, task, seed=42, fractions=(0.7, 0.1, 0.1, 0.1), declared_test=None, stratify=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if len(fractions) != 4 or any(v <= 0 for v in fractions) or abs(sum(fractions) - 1) > 1e-9: raise ValueError('INVALID_SPLIT')
    if destination == source or source.is_relative_to(destination): raise ValueError('SOURCE_IN_OUTPUT')
    task_path = Path(task).resolve()
    task_value = read_json(task_path)
    questions = task_value.get('questions')
    if not isinstance(questions, dict) or not questions: raise ValueError('TASK_WITHOUT_QUESTIONS')
    if task_value.get('decide') and task_value['decide'] not in questions: raise ValueError('TASK_DECIDE_UNKNOWN')
    if stratify is not None and stratify not in questions: raise ValueError('STRATIFY_UNKNOWN_OUTPUT:' + str(stratify))
    destination.parent.mkdir(parents=True, exist_ok=True)

    kept, by_key, ids, duplicates, conflicts = [], {}, set(), 0, []
    groups = _Union(); reconciled = 0
    for entry in _read_rows(source, questions):
        row = entry['row']
        key = _state_key(row['state'])
        answers = {k: _option_of(questions[k], v) for k, v in row.get('answers', {}).items() if v is not None}
        identity = str(row.get('id', key)); group = str(row.get('group_id', identity))
        if key in by_key:
            previous = by_key[key]
            if previous['answers'] != answers:
                conflicts.append({'ids': [previous['id'], identity], 'line': entry['line']}); continue
            duplicates += 1
            if previous['group'] != group: groups.union(previous['group'], group); reconciled += 1
            continue
        if identity in ids: raise ValueError('DUPLICATE_ID:' + identity)
        ids.add(identity); groups.find(group)
        record = {'id': identity, 'group': group, 'answers': answers, 'key': key,
                  'payload': {k: v for k, v in row.items() if k != 'questions'} | {'id': identity, 'group_id': group}}
        by_key[key] = record; kept.append(record)
    if conflicts: raise ValueError('CONFLICTING_ANSWERS:' + json.dumps(conflicts[:5], ensure_ascii=False))
    for record in kept: record['group'] = groups.find(record['group'])

    test_rows, declared_groups = [], set()
    if declared_test:
        declared_path = Path(declared_test).resolve()
        for entry in _read_rows(declared_path, questions):
            row = entry['row']
            identity = str(row.get('id', _state_key(row['state']))); group = str(row.get('group_id', identity))
            declared_groups.add(groups.find(group))
            test_rows.append({'id': identity, 'group': groups.find(group), 'payload': {**row, 'id': identity, 'group_id': group}})
        declared_keys = {_state_key(r['payload']['state']) for r in test_rows}
        leaked = [r['id'] for r in kept if r['key'] in declared_keys]
        kept = [r for r in kept if r['group'] not in declared_groups and r['key'] not in declared_keys]
    else:
        leaked = []
    group_ids = sorted({r['group'] for r in kept})
    minimum = 3 if declared_test else 4
    if len(group_ids) < minimum: raise ValueError('INSUFFICIENT_GROUPS: at least %d independent groups required' % minimum)
    ranked = sorted(group_ids, key=lambda g: (digest([seed, g]), g))
    names = ('train', 'validation', 'calibration') if declared_test else PARTITIONS
    shares = fractions[:3] if declared_test else fractions
    total_share = sum(shares)
    counts = [max(1, int(len(ranked) * (s / total_share))) for s in shares]
    counts[0] = max(1, len(ranked) - sum(counts[1:]))
    while sum(counts) > len(ranked):
        i = max(range(len(counts)), key=lambda j: counts[j]); counts[i] -= 1
    missing = []
    if stratify:
        assignment, missing = _stratified(ranked, names, counts, kept, {'name': stratify, 'value': questions[stratify]})
    else:
        assignment, cursor = {}, 0
        for name, count in zip(names, counts):
            for g in ranked[cursor:cursor + count]: assignment[g] = name
            cursor += count
        for g in ranked[cursor:]: assignment[g] = 'train'
    algorithm = 'group-hash-typed-strat-v1' if stratify else 'group-hash-typed-v1'
    partitions = {name: [] for name in PARTITIONS}
    for record in sorted(kept, key=lambda r: (digest([seed, r['group']]), r['group'], r['id'])):
        partitions[assignment[record['group']]].append(record)
    if declared_test: partitions['test'] = sorted(test_rows, key=lambda r: r['id'])

    with lock(destination.parent / (destination.name + '.lock')):
        if destination.exists(): raise FileExistsError('OUTPUT_EXISTS')
        destination.mkdir(parents=True)
        atomic_json(destination / 'task.json', task_value)
        summary = {'schema_version': 2, 'algorithm': algorithm, 'source_hash': file_hash(source), 'seed': seed,
                   'task_hash': file_hash(destination / 'task.json'), 'declared_test': bool(declared_test),
                   'declared_test_hash': file_hash(declared_test) if declared_test else None,
                   'duplicate_rows_removed': duplicates, 'groups_reconciled': reconciled, 'rows_removed_present_in_declared_test': len(leaked),
                   'fractions': list(fractions), 'splits': {}, 'support': {}, 'suporte_insuficiente': [],
                   **({'estratificacao': {'saida': stratify, 'faltando': missing}} if stratify else {}),
                   'limitations': ['exact-state dedupe only', 'group proportions, not row proportions', 'no tokenization']}
        for name in PARTITIONS:
            records = partitions[name]
            target = destination / (name + '.jsonl'); index_path = destination / (name + '.index.jsonl')
            with target.open('x', encoding='utf-8', newline='\n') as out, index_path.open('x', encoding='utf-8', newline='\n') as index:
                for record in records:
                    out.write(json.dumps(record['payload'], ensure_ascii=False) + '\n')
                    index.write(json.dumps({'id': record['id'], 'group_id': record['group']}, ensure_ascii=False) + '\n')
            group_list = sorted({r['group'] for r in records})
            support = {q: {opt: 0 for opt in _options(questions[q])} for q in questions}
            sequences = 0
            for record in records:
                answers = record.get('answers') or {k: _option_of(questions[k], v) for k, v in record['payload'].get('answers', {}).items() if v is not None}
                for q, opt in answers.items(): support[q][opt] += 1; sequences += 1
            summary['splits'][name] = {'path': target.name, 'hash': file_hash(target), 'rows': len(records), 'group_count': len(group_list),
                                       'ids': [r['id'] for r in records][:10000], 'groups': group_list[:10000], 'labeled_items': sequences,
                                       'index': {'hash': file_hash(index_path), 'row_count': len(records), 'group_count': len(group_list)},
                                       'index_truncated': len(records) > 10000 or len(group_list) > 10000}
            summary['support'][name] = support
            if name in ('validation', 'calibration'):
                for q, opts in support.items():
                    for opt, n in opts.items():
                        if n < MIN_SUPPORT: summary['suporte_insuficiente'].append({'split': name, 'saida': q, 'opcao': opt, 'n': n})
        atomic_json(destination / 'split-manifest.json', {'schema_version': 2, 'algorithm': algorithm, 'source_hash': summary['source_hash'],
                    'ids_disjoint': True, 'groups_disjoint': True,
                    'splits': {n: {'hash': v['hash'], 'index': v['index']} for n, v in summary['splits'].items()}})
        summary['split_manifest_hash'] = file_hash(destination / 'split-manifest.json')
        atomic_json(destination / 'summary.json', summary)
    return summary
