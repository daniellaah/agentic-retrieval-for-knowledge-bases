import pytest
from arkb.knowledge.models import Chunk
from arkb.evaluation.metrics import evidence_statistics, recall_at_k
from tests.evaluation.helpers import citation_fixture

def test_neighbor_recall_counts_unique_exact_neighbors_and_handles_no_reference():
    assert recall_at_k(['a', 'b'], ['b', 'c'], 2) == .5
    assert recall_at_k(['a'], ['a'], 10) == 1
    assert recall_at_k([], [], 2) is None
    with pytest.raises(ValueError):
        recall_at_k(['a'], ['a', 'a'], 2)


def test_section_coverage_uses_union_and_not_just_file_hits():
    chunks = [Chunk(content='x' * 6, title='T', source='a.md', chunk_index=i,
                    start_char=start, end_char=start + 6) for i, start in enumerate((0, 4))]
    case = {'required_source_groups': [['a.md', 'equivalent.md'], ['b.md']],
            'evidence_anchors': [{'source': 'a.md', 'body_start_char': 0, 'body_end_char': 12}]}
    metrics = evidence_statistics(chunks, case)
    assert metrics['source_group_recall'] == .5
    assert metrics['section_coverage'] == pytest.approx(10 / 12)
    assert metrics['all_sections_complete'] is False
    assert evidence_statistics([], {})['section_coverage'] is None
    with pytest.raises(ValueError, match='body coordinates'):
        evidence_statistics(chunks, {'evidence_anchors': [{'source': 'a.md', 'start_char': 0, 'end_char': 12}]})


def test_citation_metrics_separate_valid_links_from_wrong_claims_and_missing_facts():
    import json
    from arkb.evaluation.metrics import citation_statistics
    raw = json.dumps({'status': 'answered', 'claims': [
        {'text': 'Every dataset was faster.', 'source_ids': ['S1']},
        {'text': 'It was also cheaper.', 'source_ids': []}], 'missing_information': []})
    sources = citation_fixture().citation_sources
    unreviewed = citation_statistics(raw, sources)
    assert unreviewed['citation_id_validity'] == 1
    assert unreviewed['claim_reference_coverage'] == .5
    assert unreviewed['references_valid'] is False
    assert unreviewed['supported_claim_rate'] is None
    review = {'reviewer': 'fixture', 'claim_support': ['contradicted', 'insufficient'],
              'answer_correct': False, 'answer_complete': False}
    checked = citation_statistics(raw, sources, review=review)
    assert checked['supported_claim_rate'] == 0 and checked['support_review_coverage'] == 1
    assert checked['answer_correct'] is False
    with pytest.raises(ValueError, match='valid source'):
        citation_statistics(raw, sources, review={**review, 'claim_support': ['contradicted', 'supported']})


def test_citation_metrics_keep_undefined_denominators_and_partial_review_visible():
    import json
    from arkb.evaluation.metrics import citation_statistics
    sources = citation_fixture().citation_sources
    abstain = json.dumps({'status': 'insufficient_evidence', 'claims': [], 'missing_information': ['No evidence.']})
    metrics = citation_statistics(abstain, sources)
    assert metrics['citation_id_validity'] is metrics['claim_reference_coverage'] is None
    assert citation_statistics('{', sources)['claim_count'] is None
    raw = json.dumps({'status': 'answered', 'claims': [
        {'text': 'A', 'source_ids': ['S1']}, {'text': 'B', 'source_ids': ['S1']}], 'missing_information': []})
    checked = citation_statistics(raw, sources, review={'reviewer': 'fixture', 'claim_support': ['supported', None]})
    assert checked['supported_claim_rate'] == 1 and checked['support_review_coverage'] == .5
    with pytest.raises(ValueError, match='one support label'):
        citation_statistics(raw, sources, review={'reviewer': 'fixture', 'claim_support': ['supported']})


def test_relevance_metrics_use_all_judgments_and_graded_discounted_gain():
    from arkb.evaluation.metrics import ranking_metrics
    result = ranking_metrics({'a': 3, 'b': 1, 'c': 1}, ['x', 'a', 'b'], k=2)
    assert result['recall_at_k'] == pytest.approx(1 / 3)
    assert result['mrr'] == .5
    assert result['ndcg_at_k'] == pytest.approx(0.5787641110093001)
    assert ranking_metrics({}, [], k=2) == {'recall_at_k': None, 'mrr': None, 'ndcg_at_k': None}
    assert ranking_metrics({'a': 1}, ['x'], k=2)['mrr'] == 0
    with pytest.raises(ValueError, match='duplicate'):
        ranking_metrics({'a': 1}, ['a', 'a'], k=2)
