import json
from unittest.mock import Mock

import pytest

from arkb.agent import AgentTools, TOOL_DEFINITIONS
from arkb.knowledge.chunking import chunk_notes
from arkb.knowledge.documents import load_notes
from arkb.knowledge.models import ChunkRecord
from arkb.retrieval import BM25Retriever, ExactRetriever, RetrievalEngine, SearchResponse
from arkb.retrieval.models import chunk_result


def test_search_maps_arguments_once_and_projects_only_evidence(tools, engine, documents):
    record = next(documents.records())
    hit = chunk_result(record, method='hybrid', score=0.2, score_type='rrf')
    hit.metadata['fusion'] = {'private': 'details'}
    engine.search.return_value = SearchResponse(query=' original ', method='hybrid', results=(hit,))
    output = tools.search(' original ', source='a.md', limit=7)
    engine.search.assert_called_once_with(' original ', mode='semantic', rerank=False,
                                         filters={'source': 'a.md'}, top_k=7)
    result, = output['results']
    assert set(result) == {'document_id', 'source', 'title', 'content', 'document_revision',
                           'chunk_id', 'section_id', 'start_char', 'end_char'}
    assert result['document_id'] == record.document_id
    assert result['chunk_id'] == record.chunk_id
    assert output['query'] == ' original '
    assert json.loads(json.dumps(output)) == output


def test_search_empty_defaults_and_underlying_error(tools, engine):
    assert tools.search('question') == {'query': 'question', 'results': []}
    engine.search.assert_called_once_with('question', mode='semantic', rerank=False, filters={}, top_k=5)
    error = RuntimeError('backend unavailable')
    engine.search.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        tools.search('question')
    assert raised.value is error


@pytest.mark.parametrize('query, options', [
    ('', {}), ('\n', {}), (None, {}), ('x', {'limit': 0}), ('x', {'limit': -1}),
    ('x', {'limit': True}), ('x', {'limit': 1.5}), ('x', {'source': ' '}), ('x', {'source': 1}),
    ('x', {'mode': 'unknown'}), ('x', {'mode': True}),
])
def test_search_rejects_invalid_input_before_engine(tools, engine, query, options):
    with pytest.raises(ValueError):
        tools.search(query, **options)
    engine.search.assert_not_called()


def test_search_default_is_host_configured_and_agent_can_override_it(documents, engine):
    tools = AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine,
                       mode='hybrid', rerank=True)
    tools.search('question')
    engine.search.assert_called_once_with('question', mode='hybrid', rerank=True, filters={}, top_k=5)
    for mode in ('bm25', 'semantic', 'hybrid', None):
        tools.search('question', mode=mode)
        engine.search.assert_called_with('question', mode=mode or 'hybrid', rerank=True, filters={}, top_k=5)
    with pytest.raises(TypeError):
        tools.search('question', rerank=True)
    definitions = json.loads(json.dumps(TOOL_DEFINITIONS))
    assert [d['name'] for d in definitions] == ['match', 'search', 'read']
    assert set(definitions[1]['parameters']['properties']) == {'query', 'source', 'limit', 'mode'}
    assert definitions[1]['parameters']['properties']['mode']['enum'] == ['bm25', 'semantic', 'hybrid', None]


def test_real_engine_search_to_read_sections_ranges_and_eventual_consistency(documents):
    notes = load_notes(documents.directory)
    by_source = {n.source: n for n in notes}
    records = [ChunkRecord.from_note(c, note=by_source[c.source], vault_id='v')
               for c in chunk_notes(notes, count_tokens=len, chunk_size=100, chunk_overlap=0)]
    engine = RetrievalEngine(bm25=BM25Retriever(records, index_id='built'))
    exact = Mock(spec=ExactRetriever)
    tools = AgentTools(documents=documents, exact=exact, engine=engine, mode='bm25')
    hit, = tools.search('rare', source='a.md', limit=1)['results']
    assert hit['source'] == 'a.md'
    assert tools.search('rare', source='b.md')['results'] == []
    assert tools.read(hit['document_id'], section_id=hit['section_id'])['result']['content'] == hit['content']
    assert tools.read(hit['document_id'], start_char=hit['start_char'],
                      end_char=hit['end_char'])['result']['content'] == hit['content']
    (documents.directory / 'a.md').write_text('# New\n\ncurrent body', encoding='utf-8')
    assert tools.search('rare')['results'][0] == hit  # Existing engine retains its built index.
    current = tools.read(hit['document_id'])['result']
    assert current['content'] == 'current body'
    assert current['document_revision'] != hit['document_revision']
    with pytest.raises(LookupError):
        tools.read(hit['document_id'], section_id=hit['section_id'])
    exact.search.assert_not_called()


@pytest.mark.parametrize('modes,expected', [
    (['bm25'], ['bm25', None]), (['semantic'], ['semantic', None]),
    (['bm25', 'semantic'], ['bm25', 'semantic', 'hybrid', None]),
])
def test_tool_schema_advertises_only_prepared_modes_without_mutating_catalog(documents, modes, expected):
    engine = RetrievalEngine(**{mode: Mock() for mode in modes})
    tools = AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine, mode=modes[0])
    definitions = tools.tool_definitions()
    assert definitions[1]['parameters']['properties']['mode']['enum'] == expected
    assert TOOL_DEFINITIONS[1]['parameters']['properties']['mode']['enum'] == ['bm25', 'semantic', 'hybrid', None]
    definitions[1]['parameters']['properties']['mode']['enum'].clear()
    assert tools.tool_definitions()[1]['parameters']['properties']['mode']['enum'] == expected
