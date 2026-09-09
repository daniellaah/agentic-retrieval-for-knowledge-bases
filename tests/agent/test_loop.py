import json
from unittest.mock import Mock

from ollama import ResponseError
import pytest

from arkb.agent import AgentState, AgentTools, run_agent
from arkb.agent.loop import SYSTEM_INSTRUCTION
from tests.agent.helpers import ScriptedModel, reply, tool_call


@pytest.mark.parametrize('query,call,observation', [
    ('哪些笔记提到了 RAG', tool_call('match', query='RAG'),
     {'query': 'RAG', 'results': [{'source': 'rag.md', 'content': 'RAG'}]}),
    ('有哪些笔记和 RAG 相关', tool_call('search', query='RAG'),
     {'query': 'RAG', 'results': [{'source': 'retrieval.md', 'content': '相关知识'}]}),
    ('读取 rag.md', tool_call('read', source='rag.md'),
     {'result': {'source': 'rag.md', 'content': '完整文档内容'}}),
])
def test_model_selected_tool_receives_arguments_and_returns_observation(query, call, observation):
    tools = Mock(spec=AgentTools)
    name = call['function']['name']
    getattr(tools, name).return_value = observation
    final = reply('  model response\n')

    def after_tool(messages):
        assert messages[1] == {'role': 'user', 'content': query}
        assert messages[-2]['tool_calls'] == [call]
        assert messages[-1]['role'] == 'tool'
        assert messages[-1]['tool_name'] == name
        assert json.loads(messages[-1]['content']) == observation
        return final

    model = ScriptedModel(reply(calls=[call]), after_tool)
    result = run_agent(query, client=model, tools=tools, model='fake')
    assert result.stop_reason == 'final'
    assert result.response == final.message.content
    assert result.state.turn == 2
    assert result.state.tool_calls == [call]
    getattr(tools, name).assert_called_once_with(**call['function']['arguments'])
    for other in {'match', 'search', 'read'} - {name}:
        getattr(tools, other).assert_not_called()
    assert len(model.requests[0]['messages']) == 2
    for request in model.requests:
        assert request['model'] == 'fake'
        assert request['stream'] is False
        assert request['messages'][0] == {'role': 'system', 'content': SYSTEM_INSTRUCTION}
        assert [d['function']['name'] for d in request['tools']] == ['match', 'search', 'read']
        assert all(d['type'] == 'function' for d in request['tools'])
        assert set(request['tools'][1]['function']['parameters']['properties']) == {'query', 'source', 'limit'}


def test_ordinary_input_can_finish_without_tools(tools, engine):
    final = reply('你好！')
    model = ScriptedModel(final)
    result = run_agent('你好', client=model, tools=tools, model='fake', max_turns=1)
    assert result.response == final.message.content
    assert result.stop_reason == 'final'
    assert result.state.turn == 1
    assert result.state.tool_calls == []
    assert [m['role'] for m in result.state.messages] == ['system', 'user', 'assistant']
    engine.search.assert_not_called()


def test_multiple_calls_in_one_turn_preserve_order_and_are_all_observed(tools):
    calls = [tool_call('match', query='foo()'), tool_call('read', source='a.md'),
             tool_call('read', source='b.md')]

    def after_batch(messages):
        assert messages[2]['content']  # Text accompanying calls is not a final answer.
        assert messages[2]['tool_calls'] == calls
        assert [m['tool_name'] for m in messages[3:]] == ['match', 'read', 'read']
        assert json.loads(messages[4]['content'])['result']['source'] == 'a.md'
        assert json.loads(messages[5]['content'])['result']['source'] == 'b.md'
        return reply('complete')

    model = ScriptedModel(reply('Inspecting documents', calls=calls), after_batch)
    result = run_agent('Read matching notes', client=model, tools=tools, model='fake')
    assert result.stop_reason == 'final'
    assert result.state.turn == 2
    assert result.state.tool_calls == calls


@pytest.mark.parametrize('max_turns', [1, 3])
def test_repeated_calls_stop_at_exact_turn_limit_without_an_extra_model_request(tools, max_turns):
    call = tool_call('search', query='question')
    model = Mock()
    model.chat.return_value = reply('Still searching', calls=[call])
    result = run_agent('question', client=model, tools=tools, model='fake', max_turns=max_turns)
    assert result.stop_reason == 'max_turns'
    assert result.response is None
    assert result.state.turn == model.chat.call_count == max_turns
    assert result.state.tool_calls == [call] * max_turns
    assert len([m for m in result.state.messages if m['role'] == 'tool']) == max_turns
    assert result.state.messages[-1]['role'] == 'tool'


def test_empty_results_remain_observations_and_allow_a_followup(tools):
    def broaden(messages):
        assert json.loads(messages[-1]['content'])['results'] == []
        return reply(calls=[tool_call('match', query='foo()')])

    model = ScriptedModel(reply(calls=[tool_call('search', query='missing')]), broaden, reply('done'))
    result = run_agent('Find notes', client=model, tools=tools, model='fake')
    assert result.stop_reason == 'final'
    assert [c['function']['name'] for c in result.state.tool_calls] == ['search', 'match']


def test_runs_have_independent_conversations(tools):
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md')]), reply('first'), reply('second'))
    first = run_agent('Read a.md', client=model, tools=tools, model='fake')
    second = run_agent('Hello', client=model, tools=tools, model='fake')
    assert first.state is not second.state
    assert first.state.turn == 2 and second.state.turn == 1
    assert second.state.tool_calls == []
    assert [m['role'] for m in model.requests[2]['messages']] == ['system', 'user']
    left, right = AgentState(), AgentState()
    left.messages.append({'role': 'user', 'content': 'left'})
    assert right.messages == []


@pytest.mark.parametrize('options', [
    {'query': ''}, {'query': ' '}, {'query': None}, {'model': ''}, {'model': None},
    {'max_turns': 0}, {'max_turns': -1}, {'max_turns': True}, {'max_turns': 1.5},
])
def test_invalid_run_options_fail_before_model_or_tools(options):
    client, tools = Mock(), Mock(spec=AgentTools)
    with pytest.raises(ValueError):
        run_agent(client=client, tools=tools, **{'query': 'x', 'model': 'fake', **options})
    client.chat.assert_not_called()
    assert tools.mock_calls == []


@pytest.mark.parametrize('name', ['bm25', 'semantic', 'hybrid', 'rrf', 'rerank', '_documents', '__init__'])
def test_only_public_tools_can_be_dispatched(name):
    tools = Mock(spec=AgentTools)
    model = ScriptedModel(reply(calls=[tool_call(name, query='x')]))
    with pytest.raises(ValueError, match='Unknown agent tool'):
        run_agent('x', client=model, tools=tools, model='fake')
    assert tools.mock_calls == []


@pytest.mark.parametrize('call,error_type', [
    (tool_call('search'), TypeError),
    (tool_call('search', query='x', mode='bm25'), TypeError),
    (tool_call('search', query='x', limit=True), ValueError),
    (tool_call('match', query='[', regex=True), ValueError),
    (tool_call('read', source='absent.md'), LookupError),
    (tool_call('read', source='a.md', document_id='0' * 64), LookupError),
])
def test_invalid_calls_propagate_without_followup_or_fake_success(tools, call, error_type):
    model = ScriptedModel(reply(calls=[call]))
    with pytest.raises(error_type):
        run_agent('x', client=model, tools=tools, model='fake')
    assert len(model.requests) == 1


@pytest.mark.parametrize('error', [ConnectionError('model offline'), ResponseError('failed', 503)])
def test_model_errors_propagate_unchanged(tools, error):
    client = Mock()
    client.chat.side_effect = error
    with pytest.raises(type(error)) as raised:
        run_agent('x', client=client, tools=tools, model='fake')
    assert raised.value is error
    assert client.chat.call_count == 1


def test_backend_errors_propagate_unchanged(tools, engine):
    error = RuntimeError('backend unavailable')
    engine.search.side_effect = error
    model = ScriptedModel(reply(calls=[tool_call('search', query='x')]))
    with pytest.raises(RuntimeError) as raised:
        run_agent('x', client=model, tools=tools, model='fake')
    assert raised.value is error
    assert len(model.requests) == 1


@pytest.mark.parametrize('response,pattern', [
    (reply(), 'neither tool calls nor a final response'),
    (reply('  '), 'neither tool calls nor a final response'),
    (reply('partial', done_reason='length'), 'truncated'),
    (reply(calls=[tool_call('read', source='a.md')], done_reason='length'), 'truncated'),
])
def test_empty_or_truncated_responses_are_not_successful_finals(response, pattern):
    tools = Mock(spec=AgentTools)
    with pytest.raises(ValueError, match=pattern):
        run_agent('x', client=ScriptedModel(response), tools=tools, model='fake')
    assert tools.mock_calls == []


def test_non_assistant_response_is_rejected(tools):
    response = reply('invalid role')
    response.message.role = 'user'
    with pytest.raises(ValueError, match='assistant message'):
        run_agent('x', client=ScriptedModel(response), tools=tools, model='fake')
