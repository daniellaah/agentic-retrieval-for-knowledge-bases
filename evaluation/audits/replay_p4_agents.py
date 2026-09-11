"""Replay public Agent outputs, actual evidence accounting and request identity offline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def replay(run,dataset):
    import arkb
    from tokenizers import Tokenizer
    from arkb.agent.loop import SYSTEM_INSTRUCTION
    from arkb.agent.state import AgentResult,AgentState
    from arkb.agent.tools import tool_definitions
    from arkb.knowledge.embeddings import count_tokens,tokenizer_fingerprint
    from arkb.knowledge.models import _document_id
    from arkb.evaluation.external import ExternalDataset,load_external,read_jsonl,digest
    from arkb.evaluation.external_agent import evidence_packet
    from arkb.evaluation.multihop import parse_prediction,musique_metrics,OUTPUT_INSTRUCTION
    if not Path(arkb.__file__).resolve().is_relative_to(run/'measured-source/src'):raise ValueError('Expected measured sources.')
    protocol=json.loads((run/'protocol.json').read_text());rows=read_jsonl(run/'rows.jsonl')
    tokenizer=Tokenizer.from_file(str(run/'reference-tokenizer.json'))
    count=lambda text:count_tokens(text,tokenizer=tokenizer)
    identity='reference-text:'+tokenizer_fingerprint(tokenizer)
    is_mu=protocol['track']=='musique';predictions=[];model_checks=tool_checks=span_checks=0;missing=0
    selected=run/('selected.json' if is_mu else 'agentic_sample_ids.json')
    if digest(selected)!=protocol['selection_sha256']:raise ValueError('Selection changed.')
    if is_mu:cases=json.loads(selected.read_text())['rows']
    else:
        data=load_external(dataset)
        if digest(dataset/'manifest.json')!=protocol['dataset_manifest_sha256']:raise ValueError('Dataset changed.')
        ids=json.loads(selected.read_text())['tasks'][data.name.removeprefix('bright-')]
        cases=[{'id':q,'question':data.queries[q]} for q in ids]
        notes={n.source:n for n in data.notes()};source_map=data.source_map();vault=data.name
    if len(rows)!=len(cases):raise ValueError('Incomplete Agent matrix.')
    schemas=[{'type':'function','function':d} for d in tool_definitions(('bm25','semantic','hybrid'),default_mode='semantic')]
    budget=protocol['budget'];expected_budget=dict(max_tool_calls=budget['tools'],max_query_calls=budget['queries'],
        max_read_calls=budget['reads'],max_evidence_tokens=budget['evidence_tokens'],max_elapsed_ms=budget['cooperative_deadline_ms'])
    for i,(row,case) in enumerate(zip(rows,cases)):
        if row['id']!=case['id'] or row['case_index']!=i:raise ValueError('Case order changed.')
        if is_mu:
            d=ExternalDataset(f'musique-{i}',[{'id':str(p['idx']),'title':p['title'],'text':p['paragraph_text']} for p in case['paragraphs']],
                              {'q':case['question']},{'q':{}},{},{})
            notes={n.source:n for n in d.notes()};source_map={k:int(v) for k,v in d.source_map().items()};vault='p4-musique'
        value=row['result'];text=value['response'] if value else None
        if is_mu:
            prediction,error=parse_prediction(text or '',source_map,stopped=row['stop_reason'])
            prediction={'id':case['id'],**prediction};predictions.append(prediction)
            if prediction!=row['prediction'] or error!=row['parse_error']:raise ValueError('Prediction parse differs.')
        if value is None:
            if row['error'] is None:raise ValueError('Missing result without retained failure.')
            missing+=1;continue
        report=value['observation'];state=AgentState(**value['state'])
        trace=AgentResult(value['response'],value['stop_reason'],state).trace
        query=case['question']+(OUTPUT_INSTRUCTION if is_mu else '')
        if state.messages[:2]!=[{'role':'system','content':SYSTEM_INSTRUCTION},{'role':'user','content':query}]:raise ValueError('Initial prompt mismatch.')
        if trace.stop_reason!=row['stop_reason'] or report['stop_reason']!=row['stop_reason']:raise ValueError('Stop reason mismatch.')
        if report['counter_identity']!=identity or report['budget']!=expected_budget:raise ValueError('Budget or tokenizer mismatch.')
        if len(report['models'])!=state.turn or state.turn>protocol['max_turns']:raise ValueError('Turn accounting mismatch.')
        for model in report['models']:
            request=model['request']
            if request['messages']!=state.messages[:len(request['messages'])]:raise ValueError('Request differs from conversation history.')
            if (request['model']!=protocol['model'] or request['think']!=protocol['think'] or request['options']!={'temperature':0}
                    or request['stream'] is not False or request['tools']!=schemas):raise ValueError('Request configuration mismatch.')
            if model['request_json_reference_tokens']!=count(json.dumps(request,ensure_ascii=False,sort_keys=True)):raise ValueError('Request token count mismatch.')
            if model['usage'] is not None and any(v!=model['response'].get(k) for k,v in model['usage'].items()):raise ValueError('Provider usage mismatch.')
            model_checks+=1
        if row['stop_reason']=='final' and report['models'][-1]['response']['message']['content']!=text:raise ValueError('Final text differs from model output.')
        if len(trace.tool_calls)!=len(report['tools']):raise ValueError('Tool request counts differ.')
        delivered=returned=0;unique={}
        for event,tool in zip(report['tools'],trace.tool_calls):
            if (event['turn'],event['name'],event['arguments'])!=(tool.turn,tool.name,tool.arguments):raise ValueError('Tool request mismatch.')
            if tool.result!=(event['raw_result'] if event['delivered_to_conversation'] else None):raise ValueError('Delivered observation mismatch.')
            if event['returned_evidence_tokens'] is not None:
                hits=[event['raw_result']['result']] if event['name']=='read' else event['raw_result']['results']
                value_tokens=sum(count(h['content']) for h in hits)
                if value_tokens!=event['returned_evidence_tokens']:raise ValueError('Returned token count differs.')
                returned+=value_tokens
                for hit in hits:
                    note=notes[hit['source']]
                    if (hit['document_id']!=_document_id(vault,hit['source']) or hit['document_revision']!=note.document_revision
                            or hit['content']!=note.content[hit['start_char']:hit['end_char']]):raise ValueError('Tool text/source provenance differs.')
                    span_checks+=1
                if event['delivered_to_conversation']:
                    if value_tokens!=event['delivered_evidence_tokens']:raise ValueError('Delivered token count differs.')
                    delivered+=value_tokens
                    for h in hits:
                        key=json.dumps({k:h.get(k) for k in ('source','document_revision','start_char','end_char','content')},sort_keys=True,ensure_ascii=False)
                        unique[key]=count(h['content'])
            tool_checks+=1
        if report['evidence']!={'returned_tokens':returned,'delivered_tokens':delivered,'unique_exact_excerpt_tokens':sum(unique.values())}:
            raise ValueError('Evidence totals differ.')
        executed=[e for e in report['tools'] if e['executed']]
        if (len(executed)>budget['tools'] or sum(e['name'] in ('search','match') for e in executed)>budget['queries']
                or sum(e['name']=='read' for e in executed)>budget['reads'] or delivered>budget['evidence_tokens']):raise ValueError('Budget exceeded.')
        for k in ('prompt_eval_count','eval_count'):
            values=[m['usage'].get(k) if m['usage'] else None for m in report['models']]
            known=[v for v in values if type(v) is int and v>=0]
            expected={'total':sum(known) if len(values)==len(known) else None,'known_total':sum(known),'defined_requests':len(known),'requests':len(values)}
            if report['usage'][k]!=expected:raise ValueError('Usage total differs.')
        evidence_packet(row['result'],source_map)
    if is_mu and musique_metrics(cases,predictions)!=json.loads((run/'summary.json').read_text())['metrics']:raise ValueError('MuSiQue score differs.')
    return {'rows_replayed':len(rows),'model_requests_verified':model_checks,'tool_events_verified':tool_checks,
            'source_spans_verified':span_checks,'missing_error_results':missing,'release_eligible':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--child',action='store_true');p.add_argument('--archive-root',type=Path)
    a=p.parse_args();a.run=a.run.resolve();a.dataset=a.dataset.resolve() if a.dataset else None
    if a.child:print(json.dumps(replay(a.run,a.dataset)));return
    from arkb.evaluation.external import verify_checksums,digest
    if a.archive_root:
        root=a.archive_root.resolve();manifest=json.loads((root/'archive-manifest.json').read_text())
        if not a.run.is_relative_to(root) or (a.dataset and not a.dataset.is_relative_to(root)):raise ValueError('Inputs outside archive.')
        for n,h in manifest['files'].items():
            path=root/n
            if not path.resolve().is_relative_to(root) or path.is_symlink() or digest(path)!=h:raise ValueError('Archive hash mismatch.')
    else:verify_checksums(a.run)
    if json.loads((a.run/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete run.')
    command=[sys.executable,str(Path(__file__).resolve()),str(a.run),'--child']
    if a.dataset:command+=['--dataset',str(a.dataset)]
    subprocess.run(command,env={**os.environ,'PYTHONPATH':str(a.run/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1'},check=True)


if __name__=='__main__':main()
