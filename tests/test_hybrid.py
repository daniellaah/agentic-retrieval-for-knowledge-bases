from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arkb.chunking import whole_note_chunks
from arkb.retrieval import BM25Retriever, SemanticRetriever, rrf
from arkb.retrieval.snapshot import chunk_result
from arkb.schema import ChunkRecord, EmbeddingSpec, Note


def components():
    notes = [Note('', text, name) for text, name in [('exact rare', 'a.md'), ('related meaning', 'b.md')]]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='v') for n in notes]
    spec = EmbeddingSpec(model='fixed', model_revision='v1', dimensions=2, document_template='plain')
    index = SimpleNamespace(spec=spec, index_id='snapshot', search=Mock(return_value=tuple(
        chunk_result(r, method='semantic', index_id='snapshot', score=s, score_type='cosine_similarity')
        for r, s in zip(reversed(records), (.9, .5)))))
    hits = index.search.return_value
    index.search.side_effect = lambda vector, top_k, filters: tuple(
        hit for hit in hits if 'source' not in filters or hit.source == filters['source'])[:top_k]
    embedder = SimpleNamespace(spec=spec, embed_query=Mock(return_value=[1., 0.]))
    return BM25Retriever(records, index_id='snapshot'), SemanticRetriever(embedder, index)


def test_hybrid_combines_unchanged_lexical_and_semantic_rankings():
    from arkb.retrieval.hybrid import HybridRetriever
    lexical, semantic = components()
    before = [lexical.search('rare', top_k=2), semantic.search('rare', top_k=2)]
    response = HybridRetriever(lexical, semantic, candidate_k=2, rrf_k=0).search('rare', top_k=2)
    assert response.method == 'hybrid' and response.index_id == 'snapshot'
    assert [h.source for h in response.results] == ['a.md', 'b.md']
    assert [h.score for h in response.results] == [1.5, 1.]
    assert [h.identity for h in response.results] == [h.identity for h in rrf(
        {'bm25': before[0].results, 'semantic': before[1].results}, k=0)]
    assert lexical.search('rare', top_k=2) == before[0]
    assert semantic.search('rare', top_k=2) == before[1]


def test_hybrid_forwards_original_query_filters_and_depth_and_propagates_failures():
    from arkb.retrieval import SearchResponse
    from arkb.retrieval.hybrid import HybridRetriever
    calls = []
    def search(query, **options):
        calls.append((query, options))
        return SearchResponse(query=query, method='frozen', index_id='same')
    retriever = SimpleNamespace(search=search)
    hybrid = HybridRetriever(retriever, retriever, candidate_k=7)
    assert hybrid.search('  rare  ', top_k=1, filters={'source': 'x.md'}).results == ()
    assert calls == [('  rare  ', {'top_k': 7, 'filters': {'source': 'x.md'}})] * 2
    with pytest.raises(ValueError, match='candidate_k'):
        hybrid.search('q', top_k=8)
    assert len(calls) == 2
    failure = SimpleNamespace(search=Mock(side_effect=ConnectionError('offline')))
    with pytest.raises(ConnectionError):
        HybridRetriever(retriever, failure).search('q')


def test_hybrid_rejects_snapshot_mismatch_and_keeps_semantic_only_matches():
    from arkb.retrieval.hybrid import HybridRetriever
    lexical, semantic = components()
    hybrid = HybridRetriever(lexical, semantic)
    assert [h.source for h in hybrid.search('no lexical match').results] == ['b.md', 'a.md']
    lexical.index_id = 'different'
    with pytest.raises(ValueError, match='same pinned snapshot'):
        hybrid.search('rare')


def test_three_baselines_share_one_evaluation_dataset():
    from arkb.retrieval.hybrid import HybridRetriever
    from arkb.retrieval_evaluation import evaluate_retrievers
    lexical, semantic = components()
    report = evaluate_retrievers({'bm25': lexical, 'semantic': semantic,
                                 'hybrid': HybridRetriever(lexical, semantic)},
                                [{'id': 'q', 'question': 'rare', 'relevance': {'a.md': 1}}], top_k=2)
    assert report['summary']['bm25']['mrr'] == report['summary']['hybrid']['mrr'] == 1
    assert report['summary']['semantic']['mrr'] == .5


def test_optional_reranking_scores_full_hybrid_pool_before_final_top_k():
    from arkb.retrieval.hybrid import HybridRetriever
    from arkb.retrieval.reranker import Reranker, RerankedRetriever
    from arkb.retrieval_evaluation import evaluate_retrievers
    lexical, semantic = components()
    hybrid = HybridRetriever(lexical, semantic, candidate_k=2)
    scorer = SimpleNamespace(identity='frozen-relevance', score_type='logit',
                             score=Mock(return_value=[-1., 3.]))
    reranked = RerankedRetriever(hybrid, Reranker(scorer), candidate_k=2)
    report = evaluate_retrievers({'semantic': semantic, 'bm25': lexical, 'hybrid': hybrid,
                                 'hybrid_reranked': reranked},
                                [{'id': 'q', 'question': 'rare', 'relevance': {'b.md': 1}}], top_k=1)
    assert report['summary']['hybrid']['mrr'] == 0
    assert report['summary']['hybrid_reranked']['mrr'] == 1
    pool = scorer.score.call_args.args[1]
    assert [hit.source for hit in pool] == ['a.md', 'b.md']
    hit = reranked.search('rare', top_k=1).results[0]
    assert hit.metadata['rerank']['input_rank'] == 2
    assert hit.metadata['fusion']['contributions'][0]['method'] == 'semantic'
    assert hit.score_type == 'logit'
