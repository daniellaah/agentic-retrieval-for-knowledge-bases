from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arkb.retrieval import BM25Retriever, Reranker, SearchResponse
from arkb.chunking import whole_note_chunks
from arkb.schema import ChunkRecord, Note


def test_engine_explicit_modes_preserve_primitive_behavior_and_optional_reranking():
    from arkb.retrieval.engine import RetrievalEngine
    note = Note('', 'rare identifier', 'a.md')
    record = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id='v')
    bm25 = BM25Retriever([record], index_id='snapshot')
    semantic = SimpleNamespace(search=Mock(return_value=SearchResponse(query='rare', method='semantic', index_id='snapshot')))
    scorer = SimpleNamespace(identity='fixed', score_type='logit', score=Mock(return_value=[7.]))
    engine = RetrievalEngine(semantic=semantic, bm25=bm25, reranker=Reranker(scorer))
    assert engine.search('rare') == semantic.search('rare')
    assert engine.search('rare', mode='lexical') == bm25.search('rare')
    assert engine.search('rare', mode='bm25') == bm25.search('rare')
    assert engine.search('rare', mode='hybrid').results[0].identity == bm25.search('rare').results[0].identity
    scorer.score.assert_not_called()
    result = engine.search('rare', mode='bm25', rerank=True, top_k=1)
    assert result.method == 'bm25+rerank' and result.results[0].score == 7.
    assert result.results[0].metadata['rerank']['input_method'] == 'bm25'


def test_engine_rejects_unconfigured_modes_and_rerankers_without_fallback():
    from arkb.retrieval.engine import RetrievalEngine
    engine = RetrievalEngine()
    for options in ({}, {'mode': 'hybrid'}, {'mode': 'unknown'}, {'rerank': True}):
        with pytest.raises(ValueError):
            engine.search('query', **options)
