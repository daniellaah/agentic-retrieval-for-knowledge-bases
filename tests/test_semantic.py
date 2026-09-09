"""The primitive contract runs without a model service or vector database."""

from dataclasses import asdict, replace
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arkb.retrieval import SearchResponse, SearchResult, SemanticRetriever
from arkb.schema import EmbeddingSpec


@pytest.fixture
def capabilities():
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1')
    embedder = SimpleNamespace(spec=spec, embed_query=Mock(return_value=[1.0, 0.0]))
    index = SimpleNamespace(spec=spec, index_id='snapshot', search=Mock(return_value=()))
    return embedder, index


def evidence(**changes):
    return SearchResult(**{'source_id': 'stable-document', 'source': 'notes/a.md',
                           'content': 'Evidence.', 'method': 'semantic', 'score': -.25,
                           'score_type': 'cosine_similarity', **changes})


def test_semantic_embeds_original_query_once_then_passes_explicit_search_parameters(capabilities):
    embedder, index = capabilities
    calls = []
    embedder.embed_query.side_effect = lambda query: calls.append(('embed', query)) or [1.0, 0.0]
    hit = evidence(metadata={'index_version': 'snapshot', 'document_revision': 'revision'})
    index.search.side_effect = lambda vector, **options: calls.append(('search', vector, options)) or [hit]
    filters = {'source': 'notes/a.md'}
    response = SemanticRetriever(embedder, index).search('  原始 query?  ', top_k=7, filters=filters)
    assert calls == [('embed', '  原始 query?  '),
                     ('search', [1.0, 0.0], {'top_k': 7, 'filters': filters})]
    assert response == SearchResponse(query='  原始 query?  ', method='semantic', results=(hit,), index_id='snapshot')
    assert response.results[0].source_id == 'stable-document'
    assert response.results[0].metadata['document_revision'] == 'revision'
    assert filters == {'source': 'notes/a.md'}
    assert json.loads(json.dumps(asdict(response)))['results'][0]['score'] == -.25


def test_public_contract_needs_neither_score_nor_chunk_and_keeps_source_identity():
    result = SearchResult(source_id='document', source='a.md', content='Snippet', method='grep')
    assert result.score is result.score_type is result.chunk_id is result.start_char is result.end_char is None
    assert replace(result, method='metadata', content='').source_id == result.source_id
    metadata = {'tags': ['original']}
    tagged = replace(result, metadata=metadata)
    metadata['tags'].append('changed')
    assert tagged.metadata == {'tags': ['original']}


@pytest.mark.parametrize('changes', [
    {'source_id': ''}, {'content': None}, {'chunk_id': ''},
    {'score': float('nan')}, {'score': True}, {'score_type': None},
    {'score': None}, {'start_char': 0}, {'start_char': 0, 'end_char': 99},
])
def test_contract_rejects_ambiguous_scores_and_locations(changes):
    with pytest.raises(ValueError):
        evidence(**changes)


def test_semantic_preserves_backend_ranking_and_score_semantics_without_normalizing(capabilities):
    embedder, index = capabilities
    # A replacement backend may report distances instead of cosine similarities.
    hits = (evidence(score=3.5, score_type='squared_l2_distance'),
            evidence(source_id='other', score=8.0, score_type='squared_l2_distance'))
    index.search.return_value = hits
    semantic = SemanticRetriever(embedder, index)
    assert semantic.search('Q?').results == hits
    assert semantic.search('Q?') == semantic.search('Q?')


def test_empty_matches_are_successful_responses(capabilities):
    embedder, index = capabilities
    response = SemanticRetriever(embedder, index).search('Q?')
    assert response.results == () and response.index_id == 'snapshot'
    embedder.embed_query.assert_called_once_with('Q?')
    index.search.assert_called_once_with([1.0, 0.0], top_k=2, filters={})


@pytest.mark.parametrize('query,options', [
    ('', {}), (' \n', {}), (None, {}), ('Q?', {'top_k': 0}),
    ('Q?', {'top_k': True}), ('Q?', {'top_k': 1.5}),
    ('Q?', {'filters': {'unknown': 'value'}}), ('Q?', {'filters': {'source': ''}}),
    ('Q?', {'filters': {'source': None}}), ('Q?', {'filters': []}),
])
def test_invalid_requests_fail_before_calling_either_capability(capabilities, query, options):
    embedder, index = capabilities
    with pytest.raises(ValueError):
        SemanticRetriever(embedder, index).search(query, **options)
    embedder.embed_query.assert_not_called()
    index.search.assert_not_called()


def test_incompatible_embedding_spaces_fail_before_queries(capabilities):
    embedder, index = capabilities
    index.spec = replace(index.spec, model_revision='another-model')
    with pytest.raises(ValueError, match='incompatible'):
        SemanticRetriever(embedder, index)
    embedder.embed_query.assert_not_called()
    index.search.assert_not_called()


@pytest.mark.parametrize('failing_capability', ['embedder', 'index'])
def test_backend_failures_propagate_without_fallback_or_partial_success(capabilities, failing_capability):
    embedder, index = capabilities
    failure = ConnectionError('backend unavailable')
    operation = embedder.embed_query if failing_capability == 'embedder' else index.search
    operation.side_effect = failure
    with pytest.raises(ConnectionError) as raised:
        SemanticRetriever(embedder, index).search('Q?')
    assert raised.value is failure
    if failing_capability == 'embedder':
        index.search.assert_not_called()


@pytest.mark.parametrize('hits', [
    [object()], [evidence(method='grep')], [evidence(score=None, score_type=None)],
    [evidence(source='wrong.md')], [evidence(), evidence()],
])
def test_invalid_backend_evidence_does_not_leak_through_public_api(capabilities, hits):
    embedder, index = capabilities
    index.search.return_value = hits
    with pytest.raises(ValueError, match='Vector index'):
        SemanticRetriever(embedder, index).search('Q?', top_k=1, filters={'source': 'notes/a.md'})


def test_primitive_import_and_search_are_independent_of_providers_context_and_indexing():
    script = '''
import sys
class BlockDependencies:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('ollama', 'qdrant_client', 'numpy', 'tokenizers') or fullname in (
            'arkb.context', 'arkb.generation', 'arkb.indexing', 'arkb.storage', 'arkb.embeddings'
        ):
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockDependencies())
from types import SimpleNamespace
from arkb.retrieval import SemanticRetriever
from arkb.schema import EmbeddingSpec
spec = EmbeddingSpec(model='stub', model_revision='v1', dimensions=2, document_template='plain')
embedder = SimpleNamespace(spec=spec, embed_query=lambda query: [1.0, 0.0])
index = SimpleNamespace(spec=spec, index_id='v1', search=lambda vector, **kw: ())
assert SemanticRetriever(embedder, index).search('Q?').results == ()
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True, capture_output=True, text=True)
