"""Deterministic traces over scripted models and tool observations."""

from dataclasses import asdict
import json
from unittest.mock import Mock, call

from ollama import ResponseError
import pytest

from arkb.agent import AgentTools, AgentTrace, TOOL_DEFINITIONS, run_agent
from arkb.runtime import Runtime
from tests.agent.helpers import reply, tool_call


@pytest.fixture
def trace_tools():
    tools = Mock(spec=AgentTools, tool_definitions=Mock(return_value=TOOL_DEFINITIONS))
    tools.read.side_effect = lambda **args: {'result': {'source': args['source'], 'content': 'éø café'}}
    tools.match.return_value = {'query': 'needle', 'results': []}
    return tools


def test_final_trace_records_query_turns_order_arguments_results_and_exact_response(trace_tools):
    calls = [tool_call('read', source='a.md'), tool_call('read', source='b.md')]
    client = Mock()
    client.chat.side_effect = [reply('Reading', calls=calls), reply(calls=[calls[0]]), reply('  Answer\n')]
    with Runtime() as runtime:
        result = runtime.run_agent('  Find notes\n', client=client, tools=trace_tools, model='fake')

    trace = result.trace
    assert isinstance(trace, AgentTrace)
    assert asdict(trace) == {
        'query': '  Find notes\n', 'turns': 3, 'final_response': '  Answer\n', 'stop_reason': 'final',
        'tool_calls': [
            {'turn': turn, 'name': 'read', 'arguments': {'source': source},
             'result': {'result': {'source': source, 'content': 'éø café'}}}
            for turn, source in [(1, 'a.md'), (1, 'b.md'), (2, 'a.md')]
        ],
    }
    assert json.loads(json.dumps(asdict(trace), ensure_ascii=False)) == asdict(trace)
    assert client.chat.call_count == 3
    assert trace_tools.read.call_args_list == [call(source='a.md'), call(source='b.md'), call(source='a.md')]
    # Derivation adds no stored fields and cannot mutate the conversation.
    assert set(asdict(result)) == {'response', 'stop_reason', 'state'}
    assert set(asdict(result.state)) == {'messages', 'turn'}
    trace.tool_calls[0].arguments['source'] = 'changed.md'
    trace.tool_calls[0].result['result']['content'] = 'changed'
    assert result.trace.tool_calls[0].arguments == {'source': 'a.md'}
    assert result.trace.tool_calls[0].result['result']['content'] == 'éø café'


def test_direct_final_has_no_tool_trajectory(trace_tools):
    client = Mock(chat=Mock(return_value=reply('Hello')))
    result = run_agent('Hello', client=client, tools=trace_tools, model='fake', max_turns=1)
    assert asdict(result.trace) == {'query': 'Hello', 'turns': 1, 'tool_calls': [],
                                  'final_response': 'Hello', 'stop_reason': 'final'}
    assert trace_tools.mock_calls == [call.tool_definitions()]


@pytest.mark.parametrize('max_turns', [1, 3])
def test_max_turns_trace_includes_every_observation_in_the_last_batch(trace_tools, max_turns):
    calls = [tool_call('read', source='a.md'), tool_call('read', source='b.md')]
    client = Mock(chat=Mock(return_value=reply('Still reading', calls=calls)))
    result = run_agent('Read notes', client=client, tools=trace_tools, model='fake', max_turns=max_turns)
    trace = result.trace
    assert (trace.query, trace.turns, trace.final_response, trace.stop_reason) == (
        'Read notes', max_turns, None, 'max_turns')
    assert [(c.turn, c.arguments['source'], c.result['result']['source']) for c in trace.tool_calls] == [
        (turn, source, source) for turn in range(1, max_turns + 1) for source in ('a.md', 'b.md')]
    assert client.chat.call_count == max_turns
    assert trace_tools.read.call_count == 2 * max_turns


@pytest.mark.parametrize('completed_turns', [0, 1])
@pytest.mark.parametrize('error_type', [ConnectionError, ResponseError])
def test_model_error_preserves_completed_trajectory_and_counts_failed_request(
    trace_tools, completed_turns, error_type,
):
    error = error_type('model offline')
    client = Mock()
    client.chat.side_effect = [reply(calls=[tool_call('read', source='a.md')])] * completed_turns + [error]
    with Runtime() as runtime, pytest.raises(error_type) as raised:
        runtime.run_agent('Read notes', client=client, tools=trace_tools, model='fake')
    assert raised.value is error
    result = error.agent_result
    trace = result.trace
    assert (trace.query, trace.turns, trace.final_response, trace.stop_reason) == (
        'Read notes', completed_turns + 1, None, 'error')
    assert [m['role'] for m in result.state.messages] == ['system', 'user'] + ['assistant', 'tool'] * completed_turns
    assert len(trace.tool_calls) == completed_turns
    if completed_turns:
        assert trace.tool_calls[0].result == {'result': {'source': 'a.md', 'content': 'éø café'}}
    assert client.chat.call_count == completed_turns + 1
    assert trace_tools.read.call_count == completed_turns


def test_tool_error_preserves_prior_turn_and_partial_batch_without_inventing_results(trace_tools):
    error = LookupError('missing document')
    observation = {'result': {'source': 'a.md', 'content': 'already read'}}
    trace_tools.read.side_effect = [observation, error]
    first = tool_call('read', source='a.md')
    batch = [tool_call('match', query='needle'), tool_call('read', source='missing.md'),
             tool_call('read', source='never.md')]
    client = Mock(chat=Mock(side_effect=[reply(calls=[first]), reply('Inspecting', calls=batch)]))
    with pytest.raises(LookupError) as raised:
        run_agent('Read notes', client=client, tools=trace_tools, model='fake')
    assert raised.value is error
    result = error.agent_result
    trace = result.trace
    assert (trace.query, trace.turns, trace.final_response, trace.stop_reason) == ('Read notes', 2, None, 'error')
    assert [(c.turn, c.name, c.arguments) for c in trace.tool_calls] == [
        (turn, c['function']['name'], c['function']['arguments'])
        for turn, c in [(1, first), *((2, c) for c in batch)]]
    assert [c.result for c in trace.tool_calls] == [observation, {'query': 'needle', 'results': []}, None, None]
    assert [m['role'] for m in result.state.messages] == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool']
    assert trace_tools.read.call_args_list == [call(source='a.md'), call(source='missing.md')]
    trace_tools.match.assert_called_once_with(query='needle')
    assert client.chat.call_count == 2


@pytest.mark.parametrize('response', [reply(), reply('partial', done_reason='length')])
def test_protocol_errors_keep_prior_observations_without_claiming_a_final(trace_tools, response):
    client = Mock(chat=Mock(side_effect=[reply(calls=[tool_call('read', source='a.md')]), response]))
    with pytest.raises(ValueError) as raised:
        run_agent('Read notes', client=client, tools=trace_tools, model='fake')
    trace = raised.value.agent_result.trace
    assert trace.turns == 2 and trace.stop_reason == 'error' and trace.final_response is None
    assert trace.tool_calls[0].result == {'result': {'source': 'a.md', 'content': 'éø café'}}
    assert client.chat.call_count == 2


def test_trace_attachment_cannot_replace_an_exception_that_rejects_attributes(trace_tools):
    class ReadOnlyError(RuntimeError):
        def __setattr__(self, name, value):
            raise TypeError('attributes are read-only')

    error = ReadOnlyError('model failed')
    client = Mock(chat=Mock(side_effect=error))
    with pytest.raises(ReadOnlyError) as raised:
        run_agent('Read notes', client=client, tools=trace_tools, model='fake')
    assert raised.value is error
    assert client.chat.call_count == 1
