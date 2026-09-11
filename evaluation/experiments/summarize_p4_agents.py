"""Export public Agent diagnostics and pending human review from retained evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from arkb.evaluation.external import digest,load_external,read_jsonl,verify_checksums,write_json,ExternalDataset
from arkb.evaluation.external_agent import evidence_packet,coverage,operation_cost
from arkb.evaluation.multihop import parse_prediction,musique_metrics,answer_f1


def musique_breakdown(cases, predictions, rows):
    """Describe the preselected hop strata and keep undefined predictions distinct."""
    if len(cases) != len(predictions) or len(cases) != len(rows):
        raise ValueError('Incomplete slice inputs.')
    by_hops = {}
    for hops in (2, 3, 4):
        indices = [i for i,c in enumerate(cases) if len(c['question_decomposition']) == hops]
        if not indices:
            continue
        metrics = musique_metrics([cases[i] for i in indices], [predictions[i] for i in indices])
        by_hops[str(hops)] = {'metrics': metrics,
            'stop_reasons': dict(Counter(rows[i]['stop_reason'] for i in indices)),
            'valid_structured_outputs': sum(rows[i]['parse_error'] is None for i in indices)}
    confusion = {str(gold).lower(): dict(Counter(
        'undefined' if p['predicted_answerable'] is None else str(p['predicted_answerable']).lower()
        for c,p in zip(cases,predictions) if c['answerable'] is gold)) for gold in (True, False)}
    return {'by_hops': by_hops, 'answerability_confusion': confusion,
        'slice_scope': 'Fixed 2/3/4-hop selection strata; all attempts retained. Confusion keys are gold answerability then predicted answerability. Undefined includes failed or invalid structured output; it is not a correct abstention.'}


def failure_tags(row, evidence, qrels, *, case=None, prediction=None, parse_error=None):
    """Overlapping observations, not causal attribution or semantic judgments."""
    tags=[];report=(row.get('result') or {}).get('observation',{})
    error=report.get('error') or row.get('error')
    if error:
        tags.append('execution_error:'+error.get('stage','unobserved'))
        if error['type']=='EvaluationDeadlineExceeded':tags.append('harness_hard_deadline')
    if report.get('budget_stop_reason'):tags.append('budget:'+report['budget_stop_reason'])
    if row['stop_reason']!='final':tags.append('no_final_response')
    elif not evidence['submitted']:tags.append('final_without_submitted_body_evidence')
    positives={d for d,r in qrels.items() if r>0}
    known={phase:positives.intersection(evidence[phase]) for phase in ('returned','delivered','submitted')}
    if positives-known['returned']:tags.append('known_positive_documents_not_returned')
    if known['returned']-known['delivered']:tags.append('known_positive_delivery_loss')
    if known['delivered']-known['submitted']:tags.append('known_positive_submission_loss')
    if case is not None:
        if parse_error:tags.append('invalid_or_unfinished_structured_output')
        if prediction['predicted_answerable'] is not case['answerable']:
            tags.append('answerability_wrong_or_undefined')
        if case['answerable'] and positives-set(prediction['predicted_support_idxs']):
            tags.append('known_support_not_selected_in_answer')
        if case['answerable']:
            value=answer_f1(prediction['predicted_answer'],[case['answer'],*case['answer_aliases']])
            if value<1:
                tags.append('answerable_answer_string_f1_below_one')
                if positives and positives<=known['submitted']:
                    tags.append('all_known_support_sources_submitted_but_answer_string_f1_below_one')
            if prediction['predicted_answerable'] is False:tags.append('abstained_on_answerable_context')
        if not case['answerable'] and prediction['predicted_answerable'] is True:
            tags.append('answered_insufficient_context')
    return tags


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify_checksums(a.run)
    metadata=json.loads((a.run/'experiment.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    if metadata['status']!='completed':raise ValueError('Require all registered attempts before summarizing.')
    raw=read_jsonl(a.run/'rows.jsonl');track=protocol['track'];rows=[];packets=[];predictions=[]
    if track=='bright':
        data=load_external(a.dataset)
        if digest(a.dataset/'manifest.json')!=metadata['dataset_manifest_sha256']:raise ValueError('Dataset mismatch.')
        selection=json.loads((a.run/'agentic_sample_ids.json').read_text())
        ids=selection['tasks'][data.name.removeprefix('bright-')]
        if [r['id'] for r in raw]!=ids:raise ValueError('Agent matrix differs from registered selection.')
        source_map=data.source_map();docs={d['id']:d for d in data.corpus}
    else:
        cases=json.loads((a.run/'selected.json').read_text())['rows']
        if len(raw)!=len(cases):raise ValueError('Incomplete MuSiQue attempts.')
    for i,row in enumerate(raw):
        if row['case_index']!=i:raise ValueError('Attempt order changed.')
        observed=row['result'];response=observed['response'] if observed else None
        if track=='bright':
            qid=row['id'];query=data.queries[qid];qrels=data.qrels[qid];aspects=data.aspects[qid]
            gold={'aspects':aspects,'positive_documents':[docs[d] for d,r in qrels.items() if r>0]}
        else:
            case=cases[i]
            if row['id']!=case['id'] or row['answerable_gold']!=case['answerable']:raise ValueError('MuSiQue variant mismatch.')
            corpus=[{'id':str(p['idx']),'title':p['title'],'text':p['paragraph_text']} for p in case['paragraphs']]
            d=ExternalDataset(f'musique-{i}',corpus,{'q':case['question']},{'q':{}},{},{})
            source_map={k:int(v) for k,v in d.source_map().items()}
            prediction,error=parse_prediction(response or '',source_map,stopped=row['stop_reason'])
            prediction={'id':case['id'],**prediction}
            if prediction!=row['prediction'] or error!=row['parse_error']:raise ValueError('MuSiQue prediction replay mismatch.')
            predictions.append(prediction);query=case['question']
            qrels={p['idx']:1 for p in case['paragraphs'] if p['is_supporting']};aspects=[]
            gold={'answer':case['answer'],'aliases':case['answer_aliases'],'answerable':case['answerable'],
                  'context':case['paragraphs'],'warning':'Unanswerable variants retain known support labels; recall alone cannot establish sufficient evidence.'}
        evidence=evidence_packet(observed,source_map)
        rows.append({'case_index':i,'id':row['id'],'stop_reason':row['stop_reason'],'error':row['error'],
            'cost':operation_cost(row),'document_coverage_proxy':coverage(qrels,aspects,evidence)})
        rows[-1]['failure_tags']=failure_tags(row,evidence,qrels,**({'case':case,'prediction':prediction,'parse_error':error} if track!='bright' else {}))
        packet={'query':query,'response':response,'stop_reason':row['stop_reason'],
                'submitted_evidence':evidence['submitted_excerpts'],'gold_for_reviewer_only':gold}
        sha=hashlib.sha256(json.dumps(packet,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        rows[-1]['packet_id']=sha
        packets.append({'packet_id':sha,'packet':packet,'review':{'status':'pending','reviewer_id':None,
            'answer_correctness':None,'claim_support':None,'aspect_completeness':None,'abstention_appropriateness':None,
            'notes':None}})
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'diagnostics.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    random.Random(20260910).shuffle(packets);write_json(a.output/'review-packets.json',packets)
    (a.output/'README.md').write_text('''# P4 independent output review

Status: pending. No human reviews are complete; these packets are not release gold.

`review-packets.json` uses a fixed shuffle seed and binds the query, original answer,
submitted evidence and reviewer-only references to a packet_id. Review the original
packet before opening its diagnostic labels in `diagnostics.jsonl`.

Keep each packet unchanged and fill only `review`. Use a project-approved reviewer_id
and status pending or reviewed. Grade these fields pass / partial / fail /
not_applicable; notes must identify the specific claims, evidence ranges and reasons:

- answer_correctness: Does the answer correctly resolve the question? Retrieval hits
  alone do not establish correctness.
- claim_support: Are factual claims supported by submitted_evidence? A submitted
  request does not establish that the provider processed it successfully.
- aspect_completeness: Are the relevant Bright aspects or MuSiQue support chains
  covered? Mark not_applicable where appropriate.
- abstention_appropriateness: Is insufficient evidence handled correctly? Timeouts,
  missing outputs and format errors are not correct abstentions.

Record suspected Bright annotation aliases or omissions separately; do not rewrite
original gold. Unanswerable MuSiQue variants may retain partial support labels, so
partial recall does not establish sufficient context. Have a second independent
reviewer adjudicate disagreements and retain both judgments. Public-data reviews
cannot replace unseen ARKB release evaluation.

The manifest binds the exported files. Save reviewed results as a new version,
retaining the original manifest and immutable packet contents.
''')
    summary={'track':track,'attempts':len(rows),'stop_reasons':dict(Counter(r['stop_reason'] for r in rows)),
        'budget_stop_reasons':dict(Counter(r['cost']['budget_stop_reason'] for r in rows)),
        'human_reviewed':0,'release_eligible':False,'quality_status':'pending independent output review',
        'failure_tags':dict(Counter(tag for r in rows for tag in r['failure_tags'])),
        'failure_tag_scope':'Overlapping descriptive flags; missing a labeled document does not prove every relevant passage is absent. Do not sum these as mutually exclusive failure rates.',
        'coverage_scope':'All unique documents with nonempty body excerpts, in tool order. These are document-level proxies, not claim support or official Agent answer quality.'}
    if track=='bright':
        summary['document_coverage_proxy']={phase:{m:sum(r['document_coverage_proxy'][phase][m] for r in rows)/len(rows)
            for m in rows[0]['document_coverage_proxy'][phase]} for phase in ('returned','delivered','submitted')}
    else:
        summary['metrics']=musique_metrics(cases,predictions)
        expected=json.loads((a.run/'summary.json').read_text())['metrics']
        if summary['metrics']!=expected:raise ValueError('MuSiQue summary replay mismatch.')
        summary.update(musique_breakdown(cases, predictions, raw))
        # Resample complete original pairs, never independent answerable variants.
        import numpy as np
        ids=list(dict.fromkeys(c['id'] for c in cases))
        pair_metrics=[musique_metrics([c for c in cases if c['id']==qid],
            [p for p in predictions if p['id']==qid]) for qid in ids]
        names=[k for k in summary['metrics'] if k not in ('groups','rows')]
        values=np.array([[m[k] for k in names] for m in pair_metrics])
        choices=np.random.default_rng(20260910).integers(0,len(ids),size=(2000,len(ids)))
        intervals=np.quantile(values[choices].mean(axis=1),[.025,.975],axis=0)
        summary['paired_bootstrap']={'unit':'complete original pair','seed':20260910,'resamples':2000,
            'ci95':{k:intervals[:,i].tolist() for i,k in enumerate(names)},
            'limits':'Exploratory sampling intervals on this selected mixture; no multiple-comparison correction, no model retrials, and no correction for public-data training exposure.'}
    write_json(a.output/'summary.json',summary)
    write_json(a.output/'manifest.json',{'schema':'arkb-p4-agent-review-v1','input_rows_sha256':digest(a.run/'rows.jsonl'),
        'run_checksums_sha256':digest(a.run/'checksums.json'),'summarizer_sha256':digest(Path(__file__)),
        'files':{p.name:digest(p) for p in a.output.iterdir() if p.is_file()}})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
