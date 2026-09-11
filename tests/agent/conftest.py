from unittest.mock import Mock

import pytest

from arkb.agent import AgentTools
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval import ExactRetriever, RetrievalEngine, SearchResponse


@pytest.fixture
def documents(tmp_path):
    (tmp_path / 'a.md').write_text('# Alpha\n\néø café foo() fooX\n\n## Detail\nrare idea', encoding='utf-8')
    (tmp_path / 'b.md').write_text('# Beta\n\nFOO() foo() other', encoding='utf-8')
    (tmp_path / 'empty.md').write_text('# Empty\n', encoding='utf-8')
    return DocumentAccess(tmp_path, vault_id='v')


@pytest.fixture
def engine():
    engine = Mock(spec=RetrievalEngine)
    engine.search.return_value = SearchResponse(query='question', method='semantic')
    return engine


@pytest.fixture
def tools(documents, engine):
    return AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine)
