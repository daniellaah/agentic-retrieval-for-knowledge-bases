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
