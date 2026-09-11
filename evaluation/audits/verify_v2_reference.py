"""Cross-check v2 against pinned ir-measures / trec_eval without runtime deps.

Install ir-measures==0.4.3 into an isolated evaluation environment. The explicit
pytrec_eval provider does not support RR cutoff; truncate its input to K and
request uncut RR. Transform qrels into binary positives and exponential gains
separately, so no provider-specific relevance/gain defaults are assumed.
"""
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import random

import ir_measures as ir
from arkb.evaluation.v2 import ranking_scores


def main():
    if version('ir-measures') != '0.4.3' or version('pytrec-eval-terrier') != '0.5.10':
        raise ValueError('Reference versions differ from the recorded protocol.')
    rng=random.Random(20260910)
    provider=ir.providers.registry['pytrec_eval']
    cases=[({'best':3,'background':1,'also':2},['unknown','background','best'],2),
           ({'best':3},['best'],10),({'a':0},[],10),({'a':1},['a'],1)]
    for _ in range(500):
        qrels={f'd{i}':rng.randrange(4) for i in range(rng.randrange(1,31))}
        ranked=rng.sample([f'd{i}' for i in range(40)],rng.randrange(41))
        cases.append((qrels,ranked,rng.choice([1,3,5,10,20])))
    differences=[];nulls=0
    for qrels,ranked,k in cases:
        actual=ranking_scores(qrels,ranked,k=k)
        # No ties: trec_eval consumes scores, whereas our API consumes ranks.
        run={'q':{d:float(len(ranked)-i) for i,d in enumerate(ranked[:k])}}
        binary={'q':{d:int(g>=2) for d,g in qrels.items()}}
        exponential={'q':{d:2**g-1 for d,g in qrels.items()}}
        definitions=[(ir.R@k,f'recall@{k}'),(ir.RR,f'mrr@{k}'),(ir.P@k,f'precision@{k}')]
        reference=provider.calc_aggregate([m for m,_ in definitions],binary,run)
        ndcg=ir.nDCG@k
        reference_ndcg=provider.calc_aggregate([ndcg],exponential,run)[ndcg]
        for key,expected in [(key,reference[m]) for m,key in definitions]+[(f'ndcg_exp@{k}',reference_ndcg)]:
            if actual[key] is None:
                # ARKB reports undefined denominators; trec_eval emits 0.
                assert expected==0,(key,expected)
                nulls+=1
            else:
                difference=abs(actual[key]-expected)
                assert difference<1e-12,(qrels,ranked,k,key,actual[key],expected)
                differences.append(difference)
    root=Path(__file__).resolve().parents[2]
    report={'valid':True,'seed':20260910,'queries':len(cases),'numeric_comparisons':len(differences),
            'undefined_denominators_checked':nulls,'max_absolute_difference':max(differences),
            'reference_packages':{n:version(n) for n in ('ir-measures','pytrec-eval-terrier')},
            'provider':'pytrec_eval','tie_rule':'unique descending scores preserve input order',
            'adapter':'truncate run at k; binary qrels for R/RR/P; exponential qrels for linear nDCG',
            'undefined_policy':'ARKB null vs reference zero; checked separately',
            'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),root/'src/arkb/evaluation/v2.py',root/'src/arkb/evaluation/metrics.py']}}
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
