"""Summarize observed Agent operation costs without imputing unavailable usage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from arkb.evaluation.external import digest, read_jsonl, verify_checksums, write_json


def distribution(values):
    return {'observed':len(values),'total':sum(values),'mean':float(np.mean(values)) if values else None,
        'p50':float(np.quantile(values,.5)) if values else None,
        'p95':float(np.quantile(values,.95)) if values else None,
        'max':max(values) if values else None}


def summarize(run):
    verify_checksums(run);metadata=json.loads((run/'experiment.json').read_text())
    if metadata['status']!='completed':raise ValueError('Require the complete registered run.')
    rows=read_jsonl(run/'rows.jsonl');reports=[(r['result'] or {}).get('observation') for r in rows]
    models=[m for r in reports if r for m in r['models']]
    tools=[t for r in reports if r for t in r['tools']]
    usage={}
    for name in ('prompt_eval_count','eval_count'):
        values=[m['usage'].get(name) if m['usage'] else None for m in models]
        known=[x for x in values if type(x) is int and x>=0]
        usage[name]={'known_total':sum(known),'observed_requests':len(known),'requests':len(models),
            'total':sum(known) if len(known)==len(values) and all(reports) else None}
    return {'run':run.name,'rows_sha256':digest(run/'rows.jsonl'),'attempts':len(rows),
        'observed_attempts':sum(r is not None for r in reports),'preparation_ms':metadata.get('preparation_ms'),
        'agent_elapsed_ms':distribution([r['elapsed_ms'] for r in rows]),
        'context_indexing_elapsed_ms':distribution([r['context_indexing']['elapsed_ms'] for r in rows if 'context_indexing' in r]),
        'model_requests':len(models),'requested_tools':dict(Counter(t['name'] for t in tools)),
        'executed_tools':dict(Counter(t['name'] for t in tools if t['executed'])),
        'requested_search_modes':dict(Counter(t['arguments'].get('mode','semantic (default)') for t in tools if t['name']=='search')),
        'tool_elapsed_ms':{name:distribution([t['elapsed_ms'] for t in tools if t['name']==name and t['elapsed_ms'] is not None]) for name in ('search','match','read')},
        'provider_native_usage':usage,
        'scope':'Observed attempts include errors and budget stops. Agent elapsed excludes per-context indexing and shared preparation. Native provider counters are not a billing estimate; missing usage remains undefined. Descriptive local resource costs, not isolated production latency.',
        'release_eligible':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new report path.')
    write_json(args.output,{'summarizer_sha256':digest(Path(__file__)),'runs':[summarize(p) for p in args.run]})


if __name__=='__main__':main()
