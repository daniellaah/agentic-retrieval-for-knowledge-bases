"""Offline wire-contract checks using the real SDK and an in-process transport."""

import json

import httpx
from ollama import Client
from pydantic import ValidationError
import pytest

from arkb.agent import run_agent
from tests.agent.helpers import tool_call


def test_ollama_serializes_tool_calls_and_all_observations(tools):
    requests = []
    calls = [tool_call('read', source='a.md'), tool_call('read', source='b.md')]

    def handle(request):
        assert request.url.path == '/api/chat'
        body = json.loads(request.content)
        requests.append(body)
        definitions = {t['function']['name']: t['function'] for t in body['tools']}
        assert set(definitions) == {'match', 'search', 'read'}
        assert set(definitions['search']['parameters']['properties']) == {'query', 'source', 'limit'}
        assert {'document_id', 'source'} <= definitions['read']['parameters']['properties'].keys()
        if len(requests) == 1:
            message = {'role': 'assistant', 'tool_calls': calls}
        else:
            assert body['messages'][2]['tool_calls'] == calls
            observations = body['messages'][3:]
            assert [m['role'] for m in observations] == ['tool', 'tool']
            assert [m['tool_name'] for m in observations] == ['read', 'read']
            assert [json.loads(m['content'])['result']['source'] for m in observations] == ['a.md', 'b.md']
            message = {'role': 'assistant', 'content': 'finished'}
        return httpx.Response(200, json={'message': message, 'done': True, 'done_reason': 'stop'})

    with Client(host='http://ollama.test', transport=httpx.MockTransport(handle), trust_env=False) as client:
        result = run_agent('Read a.md and b.md', client=client, tools=tools, model='fake')
    assert result.stop_reason == 'final'
    assert result.state.turn == len(requests) == 2
    assert result.state.tool_calls == calls


@pytest.mark.parametrize('arguments', ['{"source":', 'not json', [], None])
def test_malformed_tool_arguments_fail_at_sdk_boundary(tools, arguments):
    def handle(request):
        return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [
            {'function': {'name': 'read', 'arguments': arguments}},
        ]}, 'done': True})

    with Client(host='http://ollama.test', transport=httpx.MockTransport(handle), trust_env=False) as client:
        with pytest.raises(ValidationError):
            run_agent('Read a.md', client=client, tools=tools, model='fake')
