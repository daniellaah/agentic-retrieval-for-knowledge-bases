"""Bind output judgments to exact answers, rubrics and submitted evidence."""
from dataclasses import asdict
import hashlib
import json

from arkb.evaluation.v2 import evidence_scores


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def submitted_evidence(observation):
    """Only observations sent in a subsequent model request can support output."""
    evidence=[]
    for event in observation['tools']:
        if not event['submitted_to_model'] or not event['delivered_to_conversation']:continue
        raw=event['raw_result']
        hits=[raw['result']] if event['name']=='read' else raw['results']
        for index,hit in enumerate(hits):
            evidence.append({**hit,'id':f'E{event["index"]}_{index}','tool':event['name']})
    return evidence


def output_packet(case,answer,evidence,*,stop_reason,execution_error=None):
    """Model identity is kept outside the blind judgment material."""
    if answer is not None and not isinstance(answer,str):raise ValueError('Answer must be text or null.')
    packet={'schema_version':'arkb-output-review-v1','case':case,'answer':answer,'evidence':evidence,
            'stop_reason':stop_reason,'execution_error':execution_error}
    return {**packet,'packet_sha256':fingerprint(packet)}


def verify_packet(packet):
    required={'schema_version','case','answer','evidence','stop_reason','execution_error','packet_sha256'}
    if set(packet)!=required or packet['schema_version']!='arkb-output-review-v1':raise ValueError('Invalid output packet.')
    if fingerprint({k:v for k,v in packet.items() if k!='packet_sha256'})!=packet['packet_sha256']:
        raise ValueError('Output packet hash mismatch.')
    ids=[e['id'] for e in packet['evidence']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate evidence identities.')


def score_output(packet,dataset,*,review=None):
    """No semantic pass is inferred from evidence, formatting or model guesses.

    Human labels are supplied explicitly and bound to packet_sha256. A software
    check cannot authenticate a reviewer or establish independent calibration.
    """
    verify_packet(packet)
    case=packet['case']
    canonical=next((c for c in dataset.cases if c['id']==case['id']),None)
    if case!=canonical:raise ValueError('Packet rubric differs from the dataset.')
    evidence=evidence_scores(case,packet['evidence'],dataset)
    executed_ok=packet['stop_reason']=='final' and packet['execution_error'] is None
    semantic=None
    if review is not None:
        required={'packet_sha256','status','reviewer','criteria','rationale'}
        if set(review)!=required or review['packet_sha256']!=packet['packet_sha256']:
            raise ValueError('Review is invalid or belongs to a different output.')
        criteria={'fact_correct','fact_complete','claims_supported','citations_complete','task_fulfilled'}
        if not isinstance(review['criteria'],dict) or set(review['criteria'])!=criteria:
            raise ValueError('Review must address every output criterion.')
        if review['status'] not in ('pending','human_reviewed','model_provisional'):
            raise ValueError('Unknown review status.')
        if review['status']=='human_reviewed':
            if (not isinstance(review['reviewer'],str) or not review['reviewer'].strip()
                    or not isinstance(review['rationale'],str) or not review['rationale'].strip()
                    or any(type(v) is not bool for v in review['criteria'].values())):
                raise ValueError('Human review requires identity, rationale and boolean criteria.')
            semantic=all(review['criteria'].values())
    required_coverage=evidence['evidence_coverage']
    evidence_ok=required_coverage==1 if required_coverage is not None else True
    grounded=(executed_ok and evidence_ok and semantic) if semantic is not None else None
    return {'execution_ok':executed_ok,'evidence':evidence,'output_semantics_pass':semantic,
            'grounded_task_success':grounded,
            'delivery_utility':0.0 if not executed_ok else (float(grounded) if grounded is not None else None),
            'release_eligible':False,'review_status':review['status'] if review else 'pending',
            'limits':'Labels remain independently reviewable; this score is not a release gate.'}


def score_agent_output(case,answer,stop_reason,report,dataset,*,review=None):
    packet=output_packet(case,answer,submitted_evidence(report),stop_reason=stop_reason,
                         execution_error=report['error'])
    scored=score_output(packet,dataset,review=review)
    requested=report['tools']
    constraints=[]
    if case['task_type']=='no_retrieval' and requested:constraints.append('unnecessary_retrieval_requested')
    if case['task_type']=='direct_read' and not any(e['tool']=='read' and e['source']==case['read_source'] for e in packet['evidence']):
        constraints.append('target_not_submitted_from_read')
    if report['budget_stop_reason']:constraints.append('budget_exhausted')
    if constraints:
        scored['delivery_utility']=0.0
        if scored['grounded_task_success'] is not None:scored['grounded_task_success']=False
    return {'packet':packet,'metrics':scored,'constraint_failures':constraints}


def score_agent(case,result,dataset,*,review=None):
    return {**score_agent_output(case,result.response,result.stop_reason,result.observation,dataset,review=review),
            'trace':asdict(result.trace)}


def pending_review(packet):
    verify_packet(packet)
    return {'packet_sha256':packet['packet_sha256'],'status':'pending','reviewer':None,
            'criteria':{key:None for key in ('fact_correct','fact_complete','claims_supported','citations_complete','task_fulfilled')},
            'rationale':None}


def gold_context(case,dataset):
    """Diagnostic oracle only: select the first annotated alternative per facet.

    It is not a deployable retriever and never feeds the natural Agent path.
    """
    ids=list(dict.fromkeys(eid for facet in case['evidence_requirements'] for eid in facet['alternatives'][0]))
    return [{'id':f'G{i}','tool':'oracle','source':span['source'],'document_revision':span['document_revision'],
             'start_char':span['start_char'],'end_char':span['end_char'],'content':span['quote']}
            for i,eid in enumerate(ids) for span in [dataset.evidence[eid]]]
