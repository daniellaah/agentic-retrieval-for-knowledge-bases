from dataclasses import replace

import pytest

from arkb.retrieval import SearchResult


def hit(name, *, method='test', score=None):
    return SearchResult(source_id='doc-' + name, source=name + '.md', chunk_id=name,
                       content=name, method=method, score=score,
                       score_type='test_score' if score is not None else None)


def test_rrf_adds_rank_votes_without_mixing_raw_scores_and_keeps_provenance():
    from arkb.retrieval.hybrid import rrf
    a, b, c = hit('a', score=9000), hit('b', score=-12), hit('c')
    result = rrf({'lexical': [a, b], 'semantic': [replace(b, method='semantic', score=.2), c]}, k=0)
    assert [r.chunk_id for r in result] == ['b', 'a', 'c']
    assert [r.score for r in result] == [1.5, 1., .5]
    assert result[0].score_type == result[0].method == 'rrf'
    votes = result[0].metadata['fusion']['contributions']
    assert [(v['list'], v['rank'], v['score']) for v in votes] == [('lexical', 2, -12), ('semantic', 1, .2)]
    assert b.metadata == {}


def test_duplicates_vote_once_at_first_original_rank_and_parent_identity_is_scoped():
    from arkb.retrieval.hybrid import rrf
    a, b = hit('a'), hit('b')
    other = replace(a, source_id='another-document')
    result = rrf([[a, a, b], [other]], k=1)
    assert len(result) == 3
    by_id = {r.identity: r.score for r in result}
    assert by_id[a.identity] == by_id[other.identity] == .5
    assert by_id[b.identity] == .25
    with pytest.raises(ValueError, match='conflicting'):
        rrf([[a], [replace(a, content='different')]])


def test_disjoint_ties_and_list_order_are_deterministic_with_top_k_and_empty_inputs():
    from arkb.retrieval.hybrid import rrf
    lists = {'z': [hit('b')], 'a': [hit('a')]}
    expected = rrf(lists)
    assert [h.chunk_id for h in expected] == ['a', 'b']
    assert expected[0].score == pytest.approx(1 / 61)
    assert rrf(dict(reversed(list(lists.items())))) == expected
    assert rrf(lists, top_k=1) == expected[:1]
    assert rrf([]) == rrf([[], []]) == ()
    assert rrf([[], [hit('a')]], k=0)[0].score == 1


@pytest.mark.parametrize('options', [{'k': -1}, {'k': float('inf')}, {'k': True}, {'top_k': 0}])
def test_invalid_rrf_parameters_fail(options):
    from arkb.retrieval.hybrid import rrf
    with pytest.raises(ValueError):
        rrf([], **options)


def test_unknown_provenance_does_not_hide_conflicting_known_snapshots():
    from arkb.retrieval.hybrid import rrf
    a = hit('a')
    with pytest.raises(ValueError, match='conflicting'):
        rrf([[a], [replace(a, metadata={'index_version': 'v1'})],
             [replace(a, metadata={'index_version': 'v2'})]])
