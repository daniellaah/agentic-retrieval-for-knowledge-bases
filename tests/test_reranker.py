from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arkb.retrieval import SearchResult


def candidates():
    return tuple(SearchResult(source_id=name, source=name + '.md', chunk_id=name,
                              content=text, method='semantic', score=score,
                              score_type='cosine_similarity', metadata={'title': name})
                 for name, text, score in [('a', 'Dogs bark.', .9), ('b', 'Paris is in France.', .4)])


def test_reranker_reorders_frozen_candidates_and_retains_input_rank_score_and_text():
    from arkb.retrieval.reranker import Reranker
    scorer = SimpleNamespace(identity='frozen-model-v1', score_type='relevance_logit',
                             score=Mock(return_value=[-2., 5.]))
    before = candidates()
    reranker = Reranker(scorer)
    result = reranker.rerank('Where is Paris?', before, top_k=1)
    assert result[0].chunk_id == 'b' and result[0].content == before[1].content
    assert result[0].score == 5. and result[0].score_type == 'relevance_logit'
    provenance = result[0].metadata['rerank']
    assert (provenance['input_rank'], provenance['input_score'], provenance['input_score_type']) == (2, .4, 'cosine_similarity')
    assert provenance['scorer'] == 'frozen-model-v1'
    assert 'rerank' not in before[1].metadata
    scorer.score.assert_called_once_with('Where is Paris?', before)
    assert reranker.rerank('Where is Paris?', before, top_k=1) == result


def test_reranker_empty_ties_invalid_scores_and_duplicate_identity():
    from arkb.retrieval.reranker import Reranker
    scorer = SimpleNamespace(identity='fixed', score_type='logit', score=Mock(return_value=[1., 1.]))
    reranker = Reranker(scorer)
    assert reranker.rerank('query', []) == ()
    scorer.score.assert_not_called()
    assert [h.chunk_id for h in reranker.rerank('query', list(reversed(candidates())))] == ['a', 'b']
    for scores in ([1.], [True, 1.], [float('nan'), 1.], [[1.], [2.]]):
        scorer.score.return_value = scores
        with pytest.raises(ValueError, match='one finite score'):
            reranker.rerank('query', candidates())
    with pytest.raises(ValueError, match='duplicate'):
        reranker.rerank('query', [candidates()[0]] * 2)
    with pytest.raises(ValueError):
        reranker.rerank(' ', [])


def test_frozen_candidate_evaluation_isolates_ranking_changes_and_latency():
    from arkb.retrieval.reranker import Reranker
    from arkb.retrieval_evaluation import evaluate_reranker
    scorer = SimpleNamespace(identity='frozen', score_type='logit', score=Mock(return_value=[-2., 5.]))
    row = evaluate_reranker(Reranker(scorer), 'Paris?', candidates(), {'b.md': 1}, top_k=1)
    assert row['before']['recall_at_k'] == 0
    assert row['after']['recall_at_k'] == 1
    assert row['rank_changes'][0] == {'identity': ['b', 'chunk', 'b'], 'before': 2, 'after': 1}
    assert row['latency_ms'] >= 0
    assert len(row['candidates']) == 2 and len(row['results']) == 1
