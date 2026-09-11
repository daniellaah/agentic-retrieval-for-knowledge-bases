from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest

from arkb.agent import AgentBudget,AgentObserver,AgentTools,TOOL_DEFINITIONS,run_agent
from tests.agent.helpers import ScriptedModel,reply,tool_call


def observer(**limits):
    return AgentObserver(budget=AgentBudget(**limits),counter=len,counter_identity='test-character-counter')


def tools():
    t=Mock(spec=AgentTools,tool_definitions=Mock(return_value=TOOL_DEFINITIONS))
    t.read.return_value={'result':{'source':'a.md','content':'abc'}}
    t.search.return_value={'results':[{'source':'a.md','content':'abc'}]}
    t.match.return_value={'results':[]}
    return t


def test_observe_without_limits_preserves_messages_and_legacy_trace():
    steps=[reply(calls=[tool_call('read',source='a.md')]),reply('answer')]
    plain=ScriptedModel(*steps);measured=ScriptedModel(*steps)
    a=run_agent('q',client=plain,tools=tools(),model='fake')
    b=run_agent('q',client=measured,tools=tools(),model='fake',observer=observer())
    assert plain.requests==measured.requests and asdict(a.trace)==asdict(b.trace)
    assert b.observation['tools'][0]['submitted_to_model']
    assert b.observation['usage']['eval_count']['total'] is None
    json.dumps(asdict(b),allow_nan=False)


def test_partial_batch_call_limit_skips_unexecuted_requests_and_does_not_synthesize_answer():
    t=tools();model=ScriptedModel(reply(calls=[tool_call('read',source='a.md')]*3))
    result=run_agent('q',client=model,tools=t,model='fake',observer=observer(max_tool_calls=1))
    assert result.stop_reason=='budget' and result.response is None
    assert t.read.call_count==1 and len(model.requests)==1
    assert [e['status'] for e in result.observation['tools']]==['success','skipped','skipped']
    assert result.trace.tool_calls[0].result is not None
    assert all(c.result is None for c in result.trace.tool_calls[1:])
    assert not result.observation['tools'][0]['submitted_to_model']


def test_query_limit_does_not_spend_read_allowance():
    t=tools();model=ScriptedModel(reply(calls=[tool_call('read',source='a.md'),tool_call('search',query='q')]))
    result=run_agent('q',client=model,tools=t,model='fake',observer=observer(max_query_calls=0,max_read_calls=1))
    t.read.assert_called_once();t.search.assert_not_called()
    assert result.observation['budget_stop_reason']=='max_query_calls'


def test_evidence_limit_counts_repeats_and_retains_but_withholds_oversized_result():
    t=tools();model=ScriptedModel(reply(calls=[tool_call('read',source='a.md')]*2))
    result=run_agent('q',client=model,tools=t,model='fake',observer=observer(max_evidence_tokens=5))
    assert result.observation['evidence']=={'returned_tokens':6,'delivered_tokens':3,'unique_exact_excerpt_tokens':3}
    last=result.observation['tools'][-1]
    assert last['executed'] and last['raw_result']==t.read.return_value
    assert not last['delivered_to_conversation'] and result.trace.tool_calls[-1].result is None


def test_tool_failure_and_later_skipped_call_are_distinct():
    t=tools();t.read.side_effect=LookupError('missing')
    model=ScriptedModel(reply(calls=[tool_call('read',source='missing'),tool_call('search',query='q')]))
    with pytest.raises(LookupError) as error:
        run_agent('q',client=model,tools=t,model='fake',observer=observer())
    events=error.value.agent_result.observation['tools']
    assert events[0]['status']=='error' and events[0]['error']['stage']=='tool_execution'
    assert events[1]['status']=='skipped' and not events[1]['executed']


def test_native_usage_keeps_missing_separate_from_zero_and_records_protocol_failure():
    first=reply(calls=[tool_call('read',source='a.md')]);first.prompt_eval_count=10;first.eval_count=2
    last=reply('partial',done_reason='length');last.prompt_eval_count=0;last.eval_count=3
    with pytest.raises(ValueError) as error:
        run_agent('q',client=ScriptedModel(first,last),tools=tools(),model='fake',observer=observer())
    report=error.value.agent_result.observation
    assert report['usage']['prompt_eval_count']['total']==10 and report['usage']['eval_count']['total']==5
    assert report['error']['stage']=='model_protocol'
    assert report['models'][-1]['response']['message']['content']=='partial'


@pytest.mark.parametrize('during',['model','tool'])
def test_deadline_retains_late_raw_result_but_never_delivers_it(during):
    clock=[0.];t=tools()
    def late_model(messages):
        clock[0]=1.
        return reply('late answer')
    def late_tool(**args):
        clock[0]=1.
        return {'result':{'source':'a.md','content':'late body'}}
    if during=='tool':t.read.side_effect=late_tool
    model=ScriptedModel(late_model if during=='model' else reply(calls=[tool_call('read',source='a.md')]))
    o=AgentObserver(budget=AgentBudget(max_elapsed_ms=100),counter=len,counter_identity='chars',clock=lambda:clock[0])
    result=run_agent('q',client=model,tools=t,model='fake',observer=o)
    assert result.stop_reason=='budget' and result.response is None
    assert result.observation['elapsed_ms']==1000
    assert not any(e['delivered_to_conversation'] for e in result.observation['tools'])


def test_observer_reuse_and_missing_counter_are_rejected():
    with pytest.raises(ValueError,match='counter'):AgentObserver(budget=AgentBudget(max_evidence_tokens=1))
    with pytest.raises(ValueError):AgentBudget(max_query_calls=True)
    o=observer();run_agent('q',client=ScriptedModel(reply('ok')),tools=tools(),model='fake',observer=o)
    with pytest.raises(ValueError,match='fresh'):
        run_agent('q',client=ScriptedModel(reply('ok')),tools=tools(),model='fake',observer=o)
