"""Public resource ownership and reuse, independent of retrieval algorithms."""

from unittest.mock import MagicMock, Mock

import pytest


def test_runtime_opens_clients_on_demand_reuses_them_and_closes_on_failure(monkeypatch):
    from arkb.config import RuntimeConfig
    from arkb.runtime import Runtime

    client = MagicMock()
    client.__enter__.return_value = client
    model_factory = Mock(return_value=client)
    vector_client = Mock()
    vector_factory = Mock(return_value=vector_client)
    monkeypatch.setattr('ollama.Client', model_factory)
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', vector_factory)

    with pytest.raises(ValueError, match='query failed'):
        with Runtime(RuntimeConfig()) as runtime:
            model_factory.assert_not_called()
            vector_factory.assert_not_called()
            assert runtime.model_client() is runtime.model_client() is client
            assert runtime.qdrant_client('http://localhost:6333') is vector_client
            assert runtime.qdrant_client('http://localhost:6333') is vector_client
            raise ValueError('query failed')

    model_factory.assert_called_once()
    vector_factory.assert_called_once()
    client.__exit__.assert_called_once()
    vector_client.close.assert_called_once()
    with pytest.raises(RuntimeError, match='closed'):
        runtime.model_client()


def test_runtime_injects_tools_without_opening_or_owning_extra_resources(tmp_path, monkeypatch):
    from arkb.retrieval import RetrievalEngine, SearchResponse
    from arkb.runtime import Runtime

    (tmp_path / 'a.md').write_text('# A\nneedle', encoding='utf-8')
    engine = Mock(spec=RetrievalEngine)
    engine.search.return_value = SearchResponse(query='question', method='hybrid')
    model = Mock()
    qdrant = Mock()
    monkeypatch.setattr('ollama.Client', model)
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', qdrant)
    with Runtime() as runtime:
        tools = runtime.agent_tools(engine=engine, directory=tmp_path, vault_id='v',
                                    mode='hybrid', rerank=True)
        hit, = tools.match('needle')['results']
        assert tools.read(hit['document_id'])['result']['content'] == 'needle'
        tools.search('question', source='a.md', limit=2)
        engine.search.assert_called_once_with('question', mode='hybrid', rerank=True,
                                             filters={'source': 'a.md'}, top_k=2)
        model.assert_not_called()
        qdrant.assert_not_called()
    with pytest.raises(RuntimeError, match='closed'):
        runtime.agent_tools(engine=engine, directory=tmp_path, vault_id='v')


@pytest.mark.parametrize('options', [{'vault_id': ''}, {'mode': 'unknown'}, {'rerank': 1}])
def test_runtime_rejects_invalid_tool_composition(tmp_path, options):
    from arkb.retrieval import RetrievalEngine
    from arkb.runtime import Runtime

    settings = {'directory': tmp_path, 'vault_id': 'v', **options}
    with Runtime() as runtime, pytest.raises(ValueError):
        runtime.agent_tools(engine=RetrievalEngine(), **settings)


def test_runtime_runs_agent_with_reused_owned_model_client(monkeypatch):
    from arkb.agent import AgentTools
    from arkb.config import DEFAULT_GENERATION_MODEL, RuntimeConfig
    from arkb.runtime import Runtime
    from tests.agent.helpers import reply

    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.return_value = reply('hello')
    factory = Mock(return_value=client)
    monkeypatch.setattr('ollama.Client', factory)
    with Runtime(RuntimeConfig(host='http://model', timeout=12)) as runtime:
        tools = Mock(spec=AgentTools)
        first = runtime.run_agent('Hello', tools=tools)
        second = runtime.run_agent('Again', tools=tools, model='custom', max_turns=1)
        assert first.stop_reason == second.stop_reason == 'final'
        assert first.state is not second.state
        factory.assert_called_once_with(host='http://model', timeout=12, trust_env=False)
        assert client.chat.call_args_list[0].kwargs['model'] == DEFAULT_GENERATION_MODEL
        assert client.chat.call_args_list[1].kwargs['model'] == 'custom'
        assert tools.mock_calls == []
    client.__exit__.assert_called_once()
    with pytest.raises(RuntimeError, match='closed'):
        runtime.run_agent('Hello', tools=tools)


def test_runtime_leaves_injected_model_client_caller_owned(monkeypatch):
    from arkb.agent import AgentTools
    from arkb.runtime import Runtime
    from tests.agent.helpers import reply

    factory, vector_factory = Mock(), Mock()
    monkeypatch.setattr('ollama.Client', factory)
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', vector_factory)
    client = MagicMock()
    client.chat.return_value = reply('hello')
    with Runtime() as runtime:
        result = runtime.run_agent('Hello', tools=Mock(spec=AgentTools), client=client)
        assert result.stop_reason == 'final'
    factory.assert_not_called()
    vector_factory.assert_not_called()
    client.__enter__.assert_not_called()
    client.__exit__.assert_not_called()
    client.close.assert_not_called()


def test_agent_vertical_slice_uses_real_tools_and_retrieval_over_multiple_turns(tmp_path):
    import json

    from arkb.knowledge.documents import DocumentAccess
    from arkb.retrieval import BM25Retriever, RetrievalEngine
    from arkb.runtime import Runtime
    from tests.agent.helpers import ScriptedModel, reply, tool_call

    (tmp_path / 'memory.md').write_text(
        '# Memory\nAgent Memory keeps useful experience.\nFollow-up: episodic retention', encoding='utf-8')
    (tmp_path / 'episodic.md').write_text(
        '# Episodic\nEpisodic retention stores past events and their context.', encoding='utf-8')
    documents = DocumentAccess(tmp_path, vault_id='v')
    engine = Mock(wraps=RetrievalEngine(bm25=BM25Retriever(list(documents.records()), index_id='test')))

    def read_found_document(messages):
        observation = json.loads(messages[-1]['content'])
        hit = observation['results'][0]
        assert hit['source'] == 'memory.md'
        return reply(calls=[tool_call('read', document_id=hit['document_id'])])

    def follow_reference(messages):
        observation = json.loads(messages[-1]['content'])
        assert observation['result']['source'] == 'memory.md'
        query = observation['result']['content'].split('Follow-up: ')[1]
        return reply(calls=[tool_call('search', query=query)])

    def finish_with_evidence(messages):
        observation = json.loads(messages[-1]['content'])
        assert any(hit['source'] == 'episodic.md' for hit in observation['results'])
        return reply('Enough material collected.')

    client = ScriptedModel(reply(calls=[tool_call('search', query='Agent Memory')]),
                           read_found_document, follow_reference, finish_with_evidence)
    with Runtime() as runtime:
        tools = runtime.agent_tools(engine=engine, directory=tmp_path, vault_id='v', mode='bm25')
        result = runtime.run_agent('帮我找一些写 Agent Memory 的素材', tools=tools, client=client, max_turns=4)
    assert result.stop_reason == 'final'
    assert result.state.turn == 4
    assert [c['function']['name'] for c in result.state.tool_calls] == ['search', 'read', 'search']
    assert [c.args[0] for c in engine.search.call_args_list] == ['Agent Memory', 'episodic retention']
    assert [m['role'] for m in result.state.messages] == [
        'system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant', 'tool', 'assistant']
