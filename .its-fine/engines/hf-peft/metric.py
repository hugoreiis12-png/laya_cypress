"""Calibrate accepted-answer matching on human-labelled adversarial probes."""
import argparse
import json
import unicodedata
from common import ler_jsonl, gravar_json, sha256_arquivo, politica_privacidade
from itsfine.privacy import require_local_executor


def normalize(text):
    # Preserve negation, numbers, punctuation and relations. No entity overlap.
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def accepted_match(answer, alternatives):
    if not isinstance(answer,str) or not isinstance(alternatives,list) or not alternatives or any(not isinstance(x,str) or not x.strip() for x in alternatives):
        raise ValueError('INVALID_ACCEPTED_ANSWERS')
    return float(normalize(answer) in {normalize(x) for x in alternatives})


def calibrate(items):
    good, bad = [], []
    for item in items:
        if type(item.get('correct')) is not bool: raise ValueError('HUMAN_LABEL_REQUIRED')
        score = accepted_match(item['response'], item['accepted_answers'])
        (good if item['correct'] else bad).append(score)
    if not good or not bad: raise ValueError('BOTH_LABEL_CLASSES_REQUIRED')
    return {'schema_version':1, 'metric':'accepted_match', 'valid':max(bad)<min(good),
            'correct_count':len(good),'incorrect_count':len(bad),
            'correct_min':min(good),'incorrect_max':max(bad),
            'limitation':'Only reviewed accepted formulations count; unmatched answers need review.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sondas',required=True); p.add_argument('--saida',required=True)
    args=p.parse_args()
    require_local_executor(politica_privacidade(args.saida))
    result=calibrate(ler_jsonl(args.sondas)); result['probes_hash']=sha256_arquivo(args.sondas)
    gravar_json(args.saida,result); print(json.dumps(result))
    return 0 if result['valid'] else 2


if __name__=='__main__':
    from common import data_entrypoint
    raise SystemExit(data_entrypoint(main,"."))
