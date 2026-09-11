"""Create provisional diagnostics; no task success or release decision."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

from arkb.evaluation.comparison import paired_family_bootstrap
from arkb.evaluation.v2 import load_dataset


def main():
    bundle,output=map(Path,sys.argv[1:])
    metadata=json.loads((bundle/'run_metadata.json').read_text())
    if metadata['status']!='completed':raise ValueError('Incomplete run.')
    dataset=load_dataset(bundle/'dataset',notes_dir=bundle/'dataset/corpus',allow_provisional=True)
    cases={c['id']:c for c in dataset.cases}
    rows=[json.loads(line) for line in (bundle/'results.jsonl').read_text().splitlines()]
    by_method=defaultdict(list)
    for row in rows:by_method[row['method']].append(row)
    metrics={};failures=[]
    for method,group in by_method.items():
        valid=[r for r in group if r['status']=='ok'];ranking=[r['metrics']['source_metrics_from_top_k_chunks'] for r in valid if r['metrics']['source_metrics_from_top_k_chunks']]
        defined=[r for r in valid if r['metrics']['evidence']['evidence_coverage'] is not None]
        values=defaultdict(list)
        for row in defined:values[cases[row['case_id']]['intent_family_id']].append(row['metrics']['evidence']['evidence_coverage'])
        metrics[method]={'rows':len(group),'statuses':dict(Counter(r['status'] for r in group)),
            'coverage_defined_cases':len(defined),'coverage_defined_families':len(values),
            'mean_case_evidence_coverage':mean(r['metrics']['evidence']['evidence_coverage'] for r in defined) if defined else None,
            'mean_family_evidence_coverage':mean(mean(v) for v in values.values()) if values else None,
            'mean_assessed_at_10':mean(r['assessed@10'] for r in ranking) if ranking else None,
            'ranking_cases':len(ranking),
            'rejected_observations':sum(r['metrics']['evidence']['rejected_observations'] for r in valid),
            'exact_set_pass_count':sum(r['metrics'].get('exact_source_set_equal') is True for r in valid)}
        for row in defined:
            if row['metrics']['evidence']['evidence_coverage']<1:
                failures.append({'case_id':row['case_id'],'method':method,
                    'missing_facets':[k for k,v in row['metrics']['evidence']['facet_satisfied'].items() if not v],
                    'coverage':row['metrics']['evidence']['evidence_coverage'],
                    'ranked_sources':row['metrics']['ranked_sources'],
                    'span_coverage':row['metrics']['evidence']['span_coverage']})
    comparable=[c for c in dataset.cases if c['evidence_requirements'] and c['task_type'] not in ('exact_lookup','direct_read','no_retrieval')]
    families={c['id']:c['intent_family_id'] for c in comparable};comparisons={}
    for left,right in [('semantic','hybrid'),('hybrid','hybrid_rerank')]:
        def arm(method):
            return {r['case_id']:[r['metrics']['evidence']['evidence_coverage'] if r['status']=='ok' else None]
                    for r in by_method[method] if r['case_id'] in families}
        try:
            comparisons[f'{left}_vs_{right}']=paired_family_bootstrap(arm(left),arm(right),families,seed=20260910)
        except ValueError as error:
            comparisons[f'{left}_vs_{right}']={'available':False,'reason':str(error)}
    result={'bundle':bundle.name,'release_eligible':False,'provisional':dataset.provisional,
            'rows':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
            'by_method':metrics,'exploratory_family_comparisons':comparisons,'coverage_failures':failures,
            'raw_sha256':hashlib.sha256((bundle/'results.jsonl').read_bytes()).hexdigest(),
            'limits':['All labels provisional; sampled families are not a production sample',
                      'Nonjudged documents conservatively score zero but are not verified negatives',
                      'Evidence coverage is not answer quality; explicit tools are not Agent routing',
                      'Bootstrap uncertainty does not include annotation error or repeated model trials']}
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'rows':len(rows),'statuses':result['statuses'],'by_method':metrics},ensure_ascii=False))


if __name__=='__main__':main()
