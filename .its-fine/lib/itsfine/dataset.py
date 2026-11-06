"""Deterministic JSONL cleaning and group-disjoint splits, before tokenization."""
from pathlib import Path
import hashlib
import json
import random
import shutil
import unicodedata
import sqlite3
import tempfile
from contextlib import closing
from .storage import atomic_json, digest, file_hash, lock, read_json

AUDIO_SUFFIXES = ('.wav', '.flac', '.ogg', '.mp3')
AUDIO_RETAINED = ('duracao', 'idioma')  # clip duration and language, kept next to the transcript


def is_typed(source):
    """Typed decision rows carry `state`; generative rows carry `messages` or `text`."""
    with Path(source).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                return isinstance(row, dict) and 'state' in row and 'messages' not in row and 'text' not in row
    return False


def prepare(source, destination, seed=42, fractions=(0.8, 0.1, 0.1), task=None, declared_test=None, stratify=None):
    if is_typed(source):
        # The generative normalisation below never sees typed rows; hashes of messages/text rows are unchanged.
        from .encoder_dataset import prepare as prepare_typed
        task = task or Path(source).with_name('task.json')
        typed_fractions = fractions if len(fractions) == 4 else (0.7, 0.1, 0.1, 0.1)
        return prepare_typed(source, destination, task, seed, typed_fractions, declared_test, stratify)
    if stratify: raise ValueError('STRATIFY_TYPED_ROWS_ONLY')
    with tempfile.TemporaryDirectory(prefix='itsfine-split-') as scratch:
        with closing(sqlite3.connect(Path(scratch)/'split.sqlite')) as db:
            return _prepare(source, destination, seed, fractions, db)


RETAINED = ('evidencia', 'source_hash', 'accepted_answers')  # evidence and accepted alternatives used by factual evaluation


def _audio_source(row, line_number, base):
    """Audio file of a speech row: relative path inside the folder of the rows, known suffix."""
    rel = row['audio']
    if base is None or not isinstance(rel, str) or not rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
        raise ValueError(f'INVALID_AUDIO_PATH:{line_number}')
    path = (Path(base) / rel).resolve()
    if not path.is_relative_to(Path(base).resolve()) or path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file() or path.is_symlink():
        raise ValueError(f'INVALID_AUDIO_FILE:{line_number}')
    return path


def _content(row, line_number, base=None):
    """Validated training content of one row and its dedupe key.

    Speech rows carry `audio` (a file next to the rows) and `text` (its transcript); the clip is
    named by its sha256, so the dedupe key and the prepared folder depend on the audio bytes.
    """
    if not isinstance(row, dict): raise ValueError(f'ROW_NOT_OBJECT:{line_number}')
    if 'audio' in row:
        if not isinstance(row.get('text'), str) or not row['text'].strip(): raise ValueError(f'AUDIO_WITHOUT_TEXT:{line_number}')
        path = _audio_source(row, line_number, base)
        content = {'audio': 'audio/' + file_hash(path) + path.suffix.lower(), 'text': row['text']}
    elif 'messages' in row:
        messages = row['messages']
        if not isinstance(messages, list) or not messages or not any(m.get('role') == 'assistant' for m in messages if isinstance(m, dict)):
            raise ValueError(f'INVALID_MESSAGES:{line_number}')
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in ('system','user','assistant','tool') or not isinstance(message.get('content'), str):
                raise ValueError(f'INVALID_MESSAGE:{line_number}')
        content = {'messages': messages}
    elif isinstance(row.get('text'), str) and row['text'].strip(): content = {'text': row['text']}
    else: raise ValueError(f'UNSUPPORTED_ROW:{line_number}')
    return content, hashlib.sha256(unicodedata.normalize('NFC', json.dumps(content, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def _kind(content):
    return 'audio' if 'audio' in content else 'generative'


def _copy_audio(clips, destination):
    """Speech clips into <destination>/audio/, named by content hash, verified after the copy."""
    if not clips: return
    (destination / 'audio').mkdir()
    for rel, origin in sorted(clips.items()):
        target = destination / rel
        shutil.copyfile(origin, target)
        if 'audio/' + file_hash(target) + target.suffix != rel: raise ValueError('AUDIO_CHANGED_DURING_COPY')


def _prepare(source, destination, seed, fractions, db):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if len(fractions) != 3 or any(v <= 0 for v in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError('INVALID_SPLIT')
    if destination == source or source.is_relative_to(destination): raise ValueError('SOURCE_IN_OUTPUT')
    destination.parent.mkdir(parents=True, exist_ok=True)
    db.executescript('CREATE TABLE rows (key TEXT UNIQUE, id TEXT UNIQUE, grp TEXT, payload TEXT);'
                    'CREATE TABLE groups (id TEXT PRIMARY KEY, rank TEXT, split TEXT);')
    duplicates = 0; kinds = set(); clips = {}
    with source.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            row = json.loads(line)
            content, key = _content(row, line_number, source.parent)
            kinds.add(_kind(content))
            if len(kinds) > 1: raise ValueError('MIXED_ROW_TYPES: speech rows and text rows in the same source')
            if db.execute('SELECT 1 FROM rows WHERE key=?',(key,)).fetchone():
                duplicates += 1; continue
            identity = str(row.get('id', key))
            if db.execute('SELECT 1 FROM rows WHERE id=?',(identity,)).fetchone(): raise ValueError('DUPLICATE_ID')
            group = str(row.get('group_id', identity))
            if 'audio' in content: clips[content['audio']] = _audio_source(row, line_number, source.parent)
            for field in RETAINED + (AUDIO_RETAINED if 'audio' in content else ()):
                if field in row: content[field] = row[field]
            db.execute('INSERT INTO rows VALUES (?,?,?,?)',(key,identity,group,json.dumps({**content,'id':identity,'group_id':group},ensure_ascii=False)))
            db.execute('INSERT OR IGNORE INTO groups VALUES (?,?,NULL)',(group,digest([seed,group])))
    count = db.execute('SELECT COUNT(*) FROM groups').fetchone()[0]
    indexed = db.execute('SELECT COUNT(*) FROM rows').fetchone()[0]>10000
    if count < 3:
        raise ValueError('INSUFFICIENT_GROUPS: at least three independent groups required')
    train_n = max(1, min(count-2, int(count*fractions[0])))
    valid_n = max(1, min(count-train_n-1, int(count*fractions[1])))
    for i,(group,) in enumerate(db.execute('SELECT id FROM groups ORDER BY rank,id')):
        split = 'train' if i < train_n else ('validation' if i < train_n+valid_n else 'test')
        db.execute('UPDATE groups SET split=? WHERE id=?',(split,group))
    db.execute('CREATE INDEX by_group ON rows(grp)')
    db.commit()
    with lock(destination.parent / (destination.name + '.lock')):
        if destination.exists(): raise FileExistsError('OUTPUT_EXISTS')
        destination.mkdir(parents=True)
        _copy_audio(clips, destination)
        summary = {'schema_version': 1, 'source_hash': file_hash(source), 'seed': seed,
                   'duplicate_rows_removed': duplicates, 'splits': {},
                   'algorithm':'group-hash-v2',
                   'limitations': ['exact-content dedupe only', 'group proportions, not row proportions', 'no tokenization']}
        if clips: summary['audio'] = {'clips': len(clips), 'folder': 'audio'}
        for name in ('train','validation','test'):
            selected = [g for g, in db.execute('SELECT id FROM groups WHERE split=? ORDER BY rank,id LIMIT 10000',(name,))]
            ids, total = [], 0
            target = destination / (name + '.jsonl')
            index_path=destination/(name+'.index.jsonl')
            with target.open('x', encoding='utf-8', newline='\n') as out, index_path.open('x',encoding='utf-8',newline='\n') as index:
                for ident,group,payload in db.execute('SELECT rows.id,rows.grp,rows.payload FROM rows JOIN groups ON grp=groups.id WHERE split=? ORDER BY rank,groups.id,rows.rowid',(name,)):
                    out.write(payload+'\n'); total += 1
                    index.write(json.dumps({'id':ident,'group_id':group},ensure_ascii=False)+'\n')
                    if len(ids) < 10000: ids.append(ident)
            group_count = db.execute('SELECT COUNT(*) FROM groups WHERE split=?',(name,)).fetchone()[0]
            summary['splits'][name] = {'path': target.name, 'hash': file_hash(target),
                                       'ids': ids, 'groups': selected, 'rows':total,'group_count':group_count,
                                       'index_truncated':total>10000 or group_count>10000}
            if indexed:
                summary['splits'][name].update(ids=[],groups=[],index_truncated=False,
                    index={'hash':file_hash(index_path),'row_count':total,'group_count':group_count})
        if indexed:
            report={'schema_version':1,'algorithm':'group-hash-v2','source_hash':summary['source_hash'],
                    'ids_disjoint':True,'groups_disjoint':True,
                    'splits':{name:{'hash':value['hash'],'index':value['index']} for name,value in summary['splits'].items()}}
            atomic_json(destination/'split-manifest.json',report)
            summary['split_manifest_hash']=file_hash(destination/'split-manifest.json')
        atomic_json(destination / 'summary.json', summary)
    return summary


def extend(base, source, destination, task=None):
    """Continuation data. The held-out partitions of an earlier prepared folder are copied byte for
    byte, so the continued model is compared with the approved one on the same validation and test.
    The earlier train rows plus the new rows form the new train, minus any row whose content, id or
    group a held-out partition already holds."""
    base, source, destination = Path(base).resolve(), Path(source).resolve(), Path(destination).resolve()
    if destination in (base, source) or source.is_relative_to(destination) or base.is_relative_to(destination): raise ValueError('SOURCE_IN_OUTPUT')
    previous = read_json(base / 'summary.json')
    held = [name for name in previous['splits'] if name != 'train']
    typed = (base / 'task.json').is_file()
    if typed:
        from .encoder_dataset import _read_rows, _state_key, _option_of, _options
        questions = read_json(base / 'task.json')['questions']
        if task is not None and file_hash(task) != file_hash(base / 'task.json'): raise ValueError('TASK_MISMATCH')
        def read(path): return [(e['line'], e['row']) for e in _read_rows(path, questions)]
        def norm(number, row):
            answers = {k: _option_of(questions[k], v) for k, v in row.get('answers', {}).items() if v is not None}
            return _state_key(row['state']), {k: v for k, v in row.items() if k != 'questions'}, answers
    else:
        if task is not None: raise ValueError('TASK_TYPED_ROWS_ONLY')
        def read(path):
            with Path(path).open(encoding='utf-8') as stream:
                return [(n, {**json.loads(line), '_base': Path(path).parent}) for n, line in enumerate(stream, 1) if line.strip()]
        def norm(number, row):
            content, key = _content(row, number, row['_base'])
            retained = RETAINED + (AUDIO_RETAINED if 'audio' in content else ())
            return key, {**content, **{f: row[f] for f in retained if f in row}}, None

    held_keys, held_ids, held_groups = set(), set(), set()
    for name in held:
        for number, row in read(base / (name + '.jsonl')):
            held_keys.add(norm(number, row)[0]); held_ids.add(str(row['id'])); held_groups.add(str(row['group_id']))
    kept, seen, ids, clips = [], {}, set(), {}
    duplicates = removed = added = 0
    for origin in (base / 'train.jsonl', source):
        for number, row in read(origin):
            key, content, answers = norm(number, row)
            if 'audio' in content: clips[content['audio']] = _audio_source(row, number, row['_base'])
            identity = str(row.get('id', key)); group = str(row.get('group_id', identity))
            if key in held_keys or identity in held_ids or group in held_groups: removed += 1; continue
            if key in seen:
                if seen[key] != answers: raise ValueError('CONFLICTING_ANSWERS:' + identity)
                duplicates += 1; continue
            if identity in ids: raise ValueError('DUPLICATE_ID:' + identity)
            seen[key] = answers; ids.add(identity)
            kept.append({'id': identity, 'group': group, 'answers': answers, 'payload': {**content, 'id': identity, 'group_id': group}})
            added += origin == source
    if not kept: raise ValueError('EMPTY_TRAIN')

    with lock(destination.parent / (destination.name + '.lock')):
        if destination.exists(): raise FileExistsError('OUTPUT_EXISTS')
        destination.mkdir(parents=True)
        for name in held:
            for suffix in ('.jsonl', '.index.jsonl'):
                if (base / (name + suffix)).is_file(): shutil.copyfile(base / (name + suffix), destination / (name + suffix))
        if typed: shutil.copyfile(base / 'task.json', destination / 'task.json')
        if (base / 'audio').is_dir():
            # held-out clips of the base, then the train clips (base and new)
            held_clips = {}
            for name in held:
                for number, row in read(base / (name + '.jsonl')):
                    if 'audio' in row: held_clips[row['audio']] = _audio_source(row, number, base)
            _copy_audio({**held_clips, **clips}, destination)
        target, index_path = destination / 'train.jsonl', destination / 'train.index.jsonl'
        with target.open('x', encoding='utf-8', newline='\n') as out, index_path.open('x', encoding='utf-8', newline='\n') as index:
            for record in kept:
                out.write(json.dumps(record['payload'], ensure_ascii=False) + '\n')
                index.write(json.dumps({'id': record['id'], 'group_id': record['group']}, ensure_ascii=False) + '\n')
        groups = sorted({r['group'] for r in kept})
        entry = dict(previous['splits']['train'])
        listed = bool(entry.get('ids')) or 'index' not in entry  # follow the listing convention of the base folder
        entry.update(hash=file_hash(target), rows=len(kept), group_count=len(groups),
                     ids=[r['id'] for r in kept][:10000] if listed else [], groups=groups[:10000] if listed else [],
                     index_truncated=listed and (len(kept) > 10000 or len(groups) > 10000))
        if 'index' in entry: entry['index'] = {'hash': file_hash(index_path), 'row_count': len(kept), 'group_count': len(groups)}
        summary = {**previous, 'algorithm': 'extend-train-v1', 'source_hash': file_hash(source), 'rows_added': added,
                   'duplicate_rows_removed': duplicates, 'rows_removed_present_in_held_out': removed,
                   'base': {'summary_hash': file_hash(base / 'summary.json'), 'algorithm': previous['algorithm'],
                            'split_manifest_hash': previous.get('split_manifest_hash')},
                   'splits': {**previous['splits'], 'train': entry}}
        if typed:
            entry['labeled_items'] = sum(len(r['answers']) for r in kept)
            support = {q: {opt: 0 for opt in _options(questions[q])} for q in questions}
            for r in kept:
                for q, opt in r['answers'].items(): support[q][opt] += 1
            summary['support'] = {**previous['support'], 'train': support}
        if (base / 'split-manifest.json').is_file():
            manifest = read_json(base / 'split-manifest.json')
            manifest['splits']['train'] = {'hash': entry['hash'], 'index': {'hash': file_hash(index_path), 'row_count': len(kept), 'group_count': len(groups)}}
            manifest.update(algorithm='extend-train-v1', source_hash=summary['source_hash'], base_manifest_hash=previous.get('split_manifest_hash'))
            atomic_json(destination / 'split-manifest.json', manifest)
            summary['split_manifest_hash'] = file_hash(destination / 'split-manifest.json')
        atomic_json(destination / 'summary.json', summary)
    return summary
