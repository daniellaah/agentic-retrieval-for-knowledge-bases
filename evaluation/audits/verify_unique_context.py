"""Post-run redundancy diagnostic from frozen observed evidence, without gold."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

from tokenizers import Tokenizer
from arkb.evaluation.v2 import load_dataset,unique_evidence_tokens
from arkb.knowledge.embeddings import count_tokens,tokenizer_fingerprint
import arkb.evaluation.v2 as evaluator


def main():
    bundle=Path(sys.argv[1]);tokenizer=Tokenizer.from_file(str(bundle/'reference-tokenizer.json'))
    dataset=load_dataset(bundle/'dataset',notes_dir=bundle/'dataset/corpus',allow_provisional=True)
    counter=lambda text:count_tokens(text,tokenizer=tokenizer)
    rows=[];groups=defaultdict(list)
    for r in [json.loads(line) for line in (bundle/'agent-results.jsonl').read_text().splitlines()]:
        assert r['observation']['counter_identity']=='reference-text:'+tokenizer_fingerprint(tokenizer)
        evidence=r['packet']['evidence']
        row={'case_id':r['case_id'],'arm':r['arm'],
             'submitted_excerpt_tokens':sum(counter(e['content']) for e in evidence),
             **unique_evidence_tokens(evidence,dataset,counter=counter)}
        rows.append(row);groups[r['arm']].append(row)
    result={'diagnostic':'Offline interval-union redundancy, not a rerun under a different context policy',
        'by_arm':{arm:{'trials':len(group),'mean_submitted_excerpt_tokens':mean(r['submitted_excerpt_tokens'] for r in group),
                       'mean_union_reference_tokens':mean(r['union_reference_tokens'] for r in group)} for arm,group in groups.items()},
        'rows':rows,'raw_sha256':hashlib.sha256((bundle/'agent-results.jsonl').read_bytes()).hexdigest(),
        'evaluator_sha256':hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
