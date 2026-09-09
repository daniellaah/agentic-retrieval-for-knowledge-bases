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
