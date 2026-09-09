import pytest


def test_relevance_metrics_use_all_judgments_and_graded_discounted_gain():
    from arkb.retrieval_evaluation import ranking_metrics
    result = ranking_metrics({'a': 3, 'b': 1, 'c': 1}, ['x', 'a', 'b'], k=2)
    assert result['recall_at_k'] == pytest.approx(1 / 3)
    assert result['mrr'] == .5
    assert result['ndcg_at_k'] == pytest.approx(0.5787641110093001)
    assert ranking_metrics({}, [], k=2) == {'recall_at_k': None, 'mrr': None, 'ndcg_at_k': None}
    assert ranking_metrics({'a': 1}, ['x'], k=2)['mrr'] == 0
    with pytest.raises(ValueError, match='duplicate'):
        ranking_metrics({'a': 1}, ['a', 'a'], k=2)


def test_comparison_uses_same_questions_preserves_ranks_and_reports_latency():
    from arkb.retrieval import SearchResult, SearchResponse
    from arkb.retrieval_evaluation import evaluate_retrievers
    class FrozenRetriever:
        def search(self, query, *, top_k=2, filters=None):
            assert query == 'original question'
            return SearchResponse(query=query, method='frozen', results=tuple(
                SearchResult(source_id=s, source=s, content=s, method='frozen', chunk_id=str(i))
                for i, s in enumerate(['wrong', 'right', 'right'][:top_k])))
    cases = [{'id': 'q1', 'question': 'original question', 'relevance': {'right': 2}}]
    report = evaluate_retrievers({'one': FrozenRetriever(), 'two': FrozenRetriever()}, cases, top_k=3)
    for name in ('one', 'two'):
        row = report['results'][0]['modes'][name]
        assert row['metrics'] == {'recall_at_k': 1., 'mrr': .5, 'ndcg_at_k': pytest.approx(.6309297535714575)}
        assert row['latency_ms'] >= 0
        assert len(row['response']['results']) == 3
        assert report['summary'][name]['mrr'] == .5
