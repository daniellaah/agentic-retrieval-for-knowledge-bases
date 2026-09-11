import json

import pytest

from arkb.evaluation.judges import judge_material,parse_judgment,judge_output
from arkb.evaluation.outputs import output_packet,submitted_evidence
from arkb.evaluation.gates import evaluation_gate
from tests.agent.helpers import ScriptedModel,reply


def packet():
    case={'query':'Q','required_facts':['GOLD_ONLY'],'expected_behavior':'Complete the requested task.',
          'answerability':'answerable'}
    return output_packet(case,'answer',[{'id':'E0','content':'OBSERVED_ONLY'}],stop_reason='final')


def test_support_cannot_see_gold_and_correctness_has_separate_material():
    p=packet()
    assert 'GOLD_ONLY' not in json.dumps(judge_material(p,'support'))
    assert 'GOLD_ONLY' in json.dumps(judge_material(p,'correctness'))
    assert 'OBSERVED_ONLY' not in json.dumps(judge_material(p,'correctness'))


def test_support_rejects_unreturned_reference_and_invented_quote():
    claim={'claim':'a claim','supported':True,'evidence_ids':['G0'],'quote':'GOLD_ONLY','rationale':'reason'}
    result={'claims':[claim],'claims_complete':True,'citations_complete':True,'rationale':'reason'}
    with pytest.raises(ValueError,match='reference'):parse_judgment(json.dumps(result),packet(),'support')
    claim['evidence_ids']=['E0']
    with pytest.raises(ValueError,match='quote'):parse_judgment(json.dumps(result),packet(),'support')
    claim['quote']='OBSERVED_ONLY'
    assert parse_judgment(json.dumps(result),packet(),'support')['claims'][0]['supported']


def test_invalid_judge_output_is_retained_without_retry_or_semantic_promotion():
    model=ScriptedModel(reply('not JSON'),reply('partial',done_reason='length'))
    result=judge_output(packet(),client=model,model='judge')
    assert len(model.requests)==2 and [r['status'] for r in result['records']]==['error','error']
    assert result['records'][0]['response']['message']['content']=='not JSON'
    assert not result['human_calibrated'] and not result['release_eligible']


def test_incomplete_fact_judgments_and_duplicate_keys_are_rejected():
    with pytest.raises(ValueError,match='fact'):
        parse_judgment('{"facts":[],"complete":true,"task_fulfilled":true,"rationale":"ok"}',packet(),'correctness')
    with pytest.raises(ValueError):parse_judgment('{"claims":[],"claims":[]}',packet(),'support')


def test_budget_withheld_and_never_submitted_evidence_do_not_enter_review_packet():
    def event(index,delivered,submitted):
        return {'index':index,'name':'read','delivered_to_conversation':delivered,'submitted_to_model':submitted,
                'raw_result':{'result':{'source':'a.md','content':str(index)}}}
    evidence=submitted_evidence({'tools':[event(0,True,True),event(1,True,False),event(2,False,False)]})
    assert [e['content'] for e in evidence]==['0']


def test_gate_separates_missing_runs_from_failed_tasks_and_unknown_quality():
    rows=[{'case_id':'q','arm':'a','trial':0,'stop_reason':'final',
           'metrics':{'review_status':'pending','delivery_utility':None}}]
    gate=evaluation_gate(rows,{('q','a',0)},inputs_stable=True,replay_valid=True,dataset_provisional=True)
    assert gate['experiment_valid'] and gate['mean_delivery_utility'] is None and not gate['release_eligible']
    assert not evaluation_gate([], {('q','a',0)},inputs_stable=True,replay_valid=True,dataset_provisional=True)['experiment_valid']
