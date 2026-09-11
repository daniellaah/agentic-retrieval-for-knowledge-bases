"""Validate an extracted experiment and replay scores with its frozen sources.

No model, Qdrant or live corpus access. Install the retained uv.lock environment
first when using a fresh machine. Verifies file integrity, not annotation truth.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay(bundle):
    import arkb
    if not Path(arkb.__file__).resolve().is_relative_to(bundle/'measured-source/src'):
        raise ValueError('Expected frozen sources.')
    if (bundle/'agent-results.jsonl').exists():
        from tokenizers import Tokenizer
        from arkb.evaluation.v2 import load_dataset
        from arkb.evaluation.outputs import score_agent_output,score_output
        from arkb.evaluation.gates import evaluation_gate
        from arkb.knowledge.embeddings import tokenizer_fingerprint,count_tokens
        assert read(bundle/'experiment.json')['status']=='completed'
        protocol=read(bundle/'protocol.json')
        dataset=load_dataset(bundle/'dataset',notes_dir=bundle/'dataset/corpus',allow_provisional=True)
        cases={c['id']:c for c in dataset.cases}
        agent=rows(bundle/'agent-results.jsonl')
        is_p3=protocol.get('protocol_version')=='p3-controlled-dev-v1'
        synthesis=[] if is_p3 else rows(bundle/'synthesis-results.jsonl')
        if is_p3:
            from arkb.evaluation.controlled import component_rows,stopping_diagnostics,stopping_order
            from arkb.agent.loop import SYSTEM_INSTRUCTION,_search_stalled,_SEARCH_STALLED_INSTRUCTION
            raw=rows(bundle/'candidate-inputs.jsonl');components=rows(bundle/'component-results.jsonl')
            selected={c['id'] for c in dataset.cases if c['task_type'] not in protocol['component']['excluded_tasks']}
            assert len(raw)==len(selected)==protocol['component']['planned_queries']
            assert {r['case_id'] for r in raw}==selected
            assert components==[r for item in raw for r in component_rows(item,cases[item['case_id']],dataset,protocol)]
            assert len(components)==protocol['component']['planned_rows']
            assert [(r['case_id'],r['arm'],r['trial']) for r in agent]==stopping_order(protocol['stopping'])
            for item in raw:
                assert item['rerank']['identity']==read(bundle/'experiment.json')['reranker_identity']
                assert all(len(leg)<=protocol['component']['max_leg_depth'] for leg in item['legs'].values())
        tokenizer=Tokenizer.from_file(str(bundle/'reference-tokenizer.json'))
        count=lambda text:count_tokens(text,tokenizer=tokenizer)
        identity='reference-text:'+tokenizer_fingerprint(tokenizer)
        for row in agent:
            report=row['observation']
            assert report['counter_identity']==identity
            assert report['budget']==(protocol['stopping']['budget'] if is_p3 else protocol['arms'][row['arm']])
            if is_p3:
                assert row['stopping_diagnostics']==stopping_diagnostics(cases[row['case_id']],report,dataset)
                expected_messages=[{'role':'system','content':SYSTEM_INSTRUCTION},{'role':'user','content':cases[row['case_id']]['query']}]
                for position,model in enumerate(report['models']):
                    request=model['request']
                    assert request['messages']==expected_messages
                    assert request['model']==protocol['agent_model'] and request['think']==protocol['stopping']['think']
                    assert request['options']=={'temperature':0} and request['stream'] is False
                    assert request['tools']==report['models'][0]['request']['tools']
                    if position+1<len(report['models']):
                        turn_start=len(expected_messages)
                        expected_messages.append(model['response']['message'])
                        for event in report['tools']:
                            if event['turn']==model['turn'] and event['delivered_to_conversation']:
                                expected_messages.append({'role':'tool','tool_name':event['name'],
                                    'content':json.dumps(event['raw_result'],ensure_ascii=False,allow_nan=False)})
                        if protocol['stopping']['arms'][row['arm']] and _search_stalled(expected_messages,turn_start):
                            expected_messages.append({'role':'system','content':_SEARCH_STALLED_INSTRUCTION})
            assert report['stop_reason']==row['stop_reason']==row['trace']['stop_reason']
            score=score_agent_output(cases[row['case_id']],row['trace']['final_response'],row['stop_reason'],report,dataset)
            assert all(row[k]==v for k,v in score.items())
            if row['stop_reason']=='final':
                assert report['models'][-1]['response']['message']['content']==row['trace']['final_response']
            for model in report['models']:
                assert model['request_json_reference_tokens']==count(json.dumps(model['request'],ensure_ascii=False,sort_keys=True))
                if model['usage'] is not None:
                    assert all(value==model['response'].get(key) for key,value in model['usage'].items())
            for key in ('prompt_eval_count','eval_count'):
                values=[m['usage'].get(key) if m['usage'] else None for m in report['models']]
                known=[v for v in values if type(v) is int and v>=0]
                assert report['usage'][key]=={'total':sum(known) if len(known)==len(values) else None,
                    'known_total':sum(known),'defined_requests':len(known),'requests':len(values)}
            delivered=returned=0;unique={}
            assert len(report['tools'])==len(row['trace']['tool_calls'])
            for event,trace in zip(report['tools'],row['trace']['tool_calls']):
                assert (event['name'],event['turn'],event['arguments'])==(trace['name'],trace['turn'],trace['arguments'])
                assert trace['result']==(event['raw_result'] if event['delivered_to_conversation'] else None)
                assert event['submitted_to_model']==(event['delivered_to_conversation'] and any(m['turn']>event['turn'] for m in report['models']))
                if event['returned_evidence_tokens'] is not None:
                    hits=[event['raw_result']['result']] if event['name']=='read' else event['raw_result']['results']
                    value=sum(count(h['content']) for h in hits)
                    assert value==event['returned_evidence_tokens'];returned+=value
                    if event['delivered_to_conversation']:
                        assert value==event['delivered_evidence_tokens'];delivered+=value
                        for hit in hits:
                            key=json.dumps({k:hit.get(k) for k in ('source','document_revision','start_char','end_char','content')},sort_keys=True,ensure_ascii=False)
                            unique[key]=count(hit['content'])
            assert returned==report['evidence']['returned_tokens'] and delivered==report['evidence']['delivered_tokens']
            assert sum(unique.values())==report['evidence']['unique_exact_excerpt_tokens']
            assert delivered<=report['budget']['max_evidence_tokens']
            executed=[e for e in report['tools'] if e['executed']]
            assert len(executed)<=report['budget']['max_tool_calls']
            assert sum(e['name'] in ('search','match') for e in executed)<=report['budget']['max_query_calls']
            assert sum(e['name']=='read' for e in executed)<=report['budget']['max_read_calls']
        for row in synthesis:assert score_output(row['packet'],dataset)==row['metrics']
        if not is_p3:
            assert len(synthesis)==protocol['planned_synthesis_rows']
            assert {(r['case_id'],r['condition']) for r in synthesis}=={(c,condition) for c in protocol['synthesis_cases'] for condition in protocol['synthesis_conditions']}
        expected=(set(stopping_order(protocol['stopping'])) if is_p3 else {(cid,arm,0) for cid in protocol['cases'] for arm in protocol['arms']})
        gate=evaluation_gate(agent,expected,
                             inputs_stable=True,replay_valid=True,dataset_provisional=dataset.provisional)
        assert gate==read(bundle/'gate.json') and gate['experiment_valid']
        return {'agent_rows':len(agent),'synthesis_rows':len(synthesis),
                'component_rows':len(components) if is_p3 else None,'token_counts_replayed':True,'release_eligible':False}
    if (bundle/'experiment.json').exists():
        from arkb.agent.state import AgentTrace,AgentToolTrace
        from arkb.evaluation.agent_metrics import evaluate_case,summarize_agent_results
        from arkb.evaluation.datasets import load_agent_eval_dataset
        from arkb.evaluation.baselines import source_metrics
        assert read(bundle/'experiment.json')['status']=='completed'
        protocol=read(bundle/'protocol.json')
        cases={c.id:c for c in load_agent_eval_dataset(bundle/'cases.jsonl',notes_dir=bundle/'corpus')}
        trials=rows(bundle/'agent-4b/results.jsonl')
        assert len(trials)==len(cases)*protocol['agent_trials']
        assert {(r['case']['id'],r['trial']) for r in trials}=={(c,t) for c in cases for t in range(protocol['agent_trials'])}
        scores=[]
        for row in trials:
            payload=row['trace']
            trace=AgentTrace(**{**payload,'tool_calls':[AgentToolTrace(**c) for c in payload['tool_calls']]}) if payload else None
            result=evaluate_case(cases[row['case']['id']],trace)
            assert json.loads(json.dumps(asdict(result)))==row['metrics']
            scores.append(result)
        summary=read(bundle/'agent-4b/summary.json')
        assert all(summary[k]==v for k,v in summarize_agent_results(scores).items())
        baseline=rows(bundle/'baselines/results.jsonl')
        assert len(baseline)==len(cases)*len(protocol['baseline_methods'])
        assert {(r['case_id'],r['baseline']) for r in baseline}=={(c,m) for c in cases for m in protocol['baseline_methods']}
        for row in baseline:
            if row['status']=='ok':
                ranked=tuple(dict.fromkeys(h['source'] for h in row['response']['results']))
                assert source_metrics(cases[row['case_id']].expected_sources,ranked,ks=(1,3,5,10))==row['metrics']
        return {'agent_rows':len(trials),'baseline_rows':len(baseline)}
    from arkb.evaluation.v2 import load_dataset
    from arkb.evaluation.pilot import score_case
    metadata=read(bundle/'run_metadata.json')
    assert metadata['status']=='completed'
    dataset=load_dataset(bundle/'dataset',notes_dir=bundle/'dataset/corpus',allow_provisional=True)
    cases={c['id']:c for c in dataset.cases}
    observed=rows(bundle/'results.jsonl')
    expected=set()
    for case in dataset.cases:
        special={'exact_lookup':'explicit_match','direct_read':'explicit_read','no_retrieval':'not_applicable'}
        methods=[special[case['task_type']]] if case['task_type'] in special else ['bm25','semantic','hybrid','hybrid_rerank']
        expected.update((case['id'],method) for method in methods)
    assert len(observed)==len(expected)==metadata['planned_rows']
    assert {(r['case_id'],r['method']) for r in observed}==expected
    for row in observed:
        if row['status']=='ok':
            assert score_case(cases[row['case_id']],row['observations'],dataset,method=row['method'])==row['metrics']
    return {'pilot_rows':len(observed),'provisional':dataset.provisional}


def main():
    if not __debug__:
        raise RuntimeError('Run verification without Python -O; checks must remain enabled.')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--require-release',action='store_true',help='Exit 2 if the run is not eligible for a release quality claim.')
    args=p.parse_args();bundle=args.bundle.resolve()
    if args.child:
        print(json.dumps({'valid':True,**replay(bundle)}));return
    hashes=read(bundle/'checksums.json')
    for name,expected in hashes.items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():
            raise ValueError(f'Invalid/missing artifact: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Checksum mismatch: {name}')
    command=[sys.executable,str(Path(__file__).resolve()),str(bundle),'--child']
    env={**os.environ,'PYTHONPATH':str(bundle/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1'}
    subprocess.run(command,env=env,check=True)
    if args.require_release:
        gate=read(bundle/'gate.json') if (bundle/'gate.json').exists() else {'release_eligible':False,'release_blockers':['No release gate for this historical/development experiment.']}
        if not gate['release_eligible']:
            print(json.dumps({'release_eligible':False,'blockers':gate['release_blockers']}))
            raise SystemExit(2)


if __name__=='__main__':main()
