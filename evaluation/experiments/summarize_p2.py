"""Export complete P2 diagnostics without inventing unreviewed task quality."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


def main():
    bundle,output=map(Path,sys.argv[1:])
    metadata=json.loads((bundle/'experiment.json').read_text())
    if metadata['status']!='completed':raise ValueError('Incomplete experiment.')
    rows=[json.loads(line) for line in (bundle/'agent-results.jsonl').read_text().splitlines()]
    synthesis=[json.loads(line) for line in (bundle/'synthesis-results.jsonl').read_text().splitlines()]
    judges=json.loads((bundle/'judge-results.json').read_text())
    groups=defaultdict(list)
    for row in rows:groups[row['arm']].append(row)
    arms={}
    for arm,group in groups.items():
        evidence=[r for r in group if r['metrics']['evidence']['evidence_coverage'] is not None]
        final=[r for r in evidence if r['stop_reason']=='final' and not r['constraint_failures']]
        def native(key):
            known=[r['observation']['usage'][key]['total'] for r in group]
            values=[v for v in known if v is not None]
            return {'mean':mean(values) if values else None,'defined_trials':len(values),'total_trials':len(group)}
        arms[arm]={'trials':len(group),'families':len({r['packet']['case']['intent_family_id'] for r in group}),
            'stop_reasons':dict(Counter(r['stop_reason'] for r in group)),
            'budget_stop_reasons':dict(Counter(r['observation']['budget_stop_reason'] for r in group if r['observation']['budget_stop_reason'])),
            'runtime_errors':sum(r['error'] is not None for r in group),
            'mean_elapsed_ms':mean(r['elapsed_ms'] for r in group),
            'mean_requested_tools':mean(len(r['observation']['tools']) for r in group),
            'mean_executed_tools':mean(sum(t['executed'] for t in r['observation']['tools']) for r in group),
            'mean_delivered_evidence_tokens':mean(r['observation']['evidence']['delivered_tokens'] for r in group),
            'mean_returned_evidence_tokens':mean(r['observation']['evidence']['returned_tokens'] for r in group),
            'agent_native_prompt_tokens':native('prompt_eval_count'),'agent_native_output_tokens':native('eval_count'),
            'evidence_applicable_cases':len(evidence),'final_evidence_cases':len(final),
            'conditional_final_evidence_coverage':mean(r['metrics']['evidence']['evidence_coverage'] for r in final) if final else None,
            'evidence_delivery_utility':mean(r['metrics']['evidence']['evidence_coverage'] if r in final else 0 for r in evidence) if evidence else None,
            'grounded_task_success':None,'output_review_coverage':0.0}
    result={'bundle':bundle.name,'gate':json.loads((bundle/'gate.json').read_text()),'arms':arms,
        'per_case':[{'case_id':r['case_id'],'arm':r['arm'],'stop_reason':r['stop_reason'],
                    'budget_stop_reason':r['observation']['budget_stop_reason'],'error':r['error'],
                    'evidence_coverage':r['metrics']['evidence']['evidence_coverage'],
                    'delivered_evidence_tokens':r['observation']['evidence']['delivered_tokens'],
                    'constraints':r['constraint_failures']} for r in rows],
        'synthesis':[{'case_id':r['case_id'],'condition':r['condition'],'stop_reason':r['packet']['stop_reason'],
                      'evidence_coverage':r['metrics']['evidence']['evidence_coverage'],
                      'grounded_task_success':r['metrics']['grounded_task_success']} for r in synthesis],
        'judge_states':[{'case_id':j['case_id'],'condition':j['condition'],'modes':[{k:r[k] for k in ('mode','status','error')} for r in j['records']]} for j in judges],
        'raw_sha256':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in ('agent-results.jsonl','synthesis-results.jsonl','judge-results.json','reference-tokenizer.json')},
        'limits':['Known provisional dev cases and one trial per arm; no superiority claim',
                  'Deadline is cooperative and cannot cancel in-flight work',
                  'Native token counts cover Agent chat requests; embedding usage is not separately measured',
                  'Model judge results are uncalibrated; normal final stop is not task correctness']}
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'arms':arms,'judge_states':result['judge_states']},ensure_ascii=False))


if __name__=='__main__':main()
