"""Export blind all-corpus judgment queues and explicit draft annotation review.

Different reviewer seeds randomize order without exposing model rankings or
provisional grades. Exports never overwrite files or mark reviews completed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from arkb.evaluation.v2 import load_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('evaluation/data/v2/pilot'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    dataset=load_dataset(args.dataset,notes_dir=args.dataset/'corpus',allow_provisional=True)
    rows=[]
    for case in dataset.cases:
        if case['task_type']=='no_retrieval':continue
        for source in dataset.bodies:
            rows.append({'query_id':case['id'],'query':case['query'],'source':source,
                         'grade':None,'reviewer':None,'rationale':None})
    random.Random(args.seed).shuffle(rows)
    args.output.mkdir(parents=True,exist_ok=False)
    files={'document-judgments.jsonl':rows,
           'evidence-review.jsonl':[{'evidence':e,'decision':None,'reviewer':None,'rationale':None}
                                    for e in dataset.evidence.values()],
           'case-review.jsonl':[{'case':c,'decision':None,'reviewer':None,'rationale':None} for c in dataset.cases]}
    for name,items in files.items():
        (args.output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in items))
    metadata={'dataset_manifest_sha256':hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
              'seed':args.seed,'document_judgments':len(rows),'evidence_reviews':len(dataset.evidence),
              'case_reviews':len(dataset.cases),'status':'pending',
              'instructions':'Grade documents 0..3; blank is unjudged. Evidence/case decisions: accept, revise, reject. Identify reviewer; keep revisions/reasons. No ranks or proposed grades in document queue.'}
    (args.output/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
