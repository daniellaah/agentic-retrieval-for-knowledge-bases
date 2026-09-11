"""Export compact, tracked P0 measurements from the complete retained bundle."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


def read(path):return json.loads(path.read_text())


def main():
    bundle=Path(sys.argv[1]);output=Path(sys.argv[2])
    metadata=read(bundle/'experiment.json')
    if metadata['status']!='completed':raise ValueError('Incomplete experiment.')
    agent=read(bundle/'agent-4b/summary.json');baseline=read(bundle/'baselines/summary.json')
    rows=[json.loads(line) for line in (bundle/'agent-4b/results.jsonl').read_text().splitlines()]
    by_case=defaultdict(list)
    for row in rows:by_case[row['case']['id']].append(row)
    failures=[]
    for case,trials in by_case.items():
        failed=[r for r in trials if not r['metrics']['success']]
        if failed:
            failures.append({'case_id':case,'failed_trials':len(failed),'total_trials':len(trials),
                'failure_reasons':dict(Counter(reason for r in failed for reason in r['metrics']['failure_reasons'])),
                'source_recall':[r['metrics']['source_recall'] for r in trials],
                'stop_reasons':[r['metrics']['stop_reason'] for r in trials]})
    result={'bundle':bundle.name,'status':metadata['status'],'validation':read(bundle/'validation.json'),
            'source_commit':metadata['source_commit'],'models':metadata['models_before'],
            'snapshot':read(bundle/'snapshot-before.json')['manifest'],
            'agent':{k:v for k,v in agent.items() if k not in ('failed_trials','by_task_type')},
            'agent_by_task_type':agent['by_task_type'],
            'agent_average_latency_ms':mean(r['runtime_metadata']['elapsed_ms'] for r in rows),
            'runtime_errors':sum(r['error'] is not None for r in rows),
            'cases_with_mixed_success':sum(0<sum(r['metrics']['success'] for r in trials)<len(trials) for trials in by_case.values()),
            'failed_cases':failures,'baseline_overall':baseline['overall'],
            'raw_file_sha256':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in
                ['agent-4b/results.jsonl','agent-4b/summary.json','baselines/results.jsonl','baselines/summary.json','checksums.json']},
            'interpretation_limits':['v1 source/tool behavior, not answer correctness',
                'Agent and baseline budgets differ','timing is not a warm/cold performance experiment',
                'known development corpus; repeated trials are not independent information needs']}
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','runtime_errors','cases_with_mixed_success','agent_average_latency_ms']}))


if __name__=='__main__':main()
