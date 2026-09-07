from functools import partial

import numpy as np
import pytest

from obsidian_rag.knowledge_base.chunking import Chunk, whole_note_chunks
from obsidian_rag.evaluation import compare_retrieval, evidence_statistics, recall_at_k
from obsidian_rag.knowledge_base.loaders import Note
from obsidian_rag.knowledge_base.vector_index.qdrant import search_qdrant, QdrantIndex
from qdrant_client import QdrantClient
from contextlib import closing
from obsidian_rag.knowledge_base.vector_index.manifest import ChunkRecord, EmbeddingSpec


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


@pytest.mark.filterwarnings('ignore:.*local Qdrant.*:UserWarning')
@pytest.mark.filterwarnings('ignore:Local mode performs exact.*:UserWarning')
def test_comparison_separates_neighbor_recall_from_evidence_coverage():
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2, document_template='title-body-v1')
    notes = [Note(title='T', content='evidence', source=f'{i}.md') for i in range(2)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='v') for n in notes]
    vectors = np.eye(2)
    cases = [{'id': 'q1', 'question': 'Question?', 'required_source_groups': [['1.md']],
              'evidence_anchors': [{'source': '1.md', 'body_start_char': 0, 'body_end_char': 8}]}]
    with closing(QdrantClient(':memory:')) as client:
        QdrantIndex(client, 'reference', spec, vault_id='v', create=True).upsert(records, vectors)
        QdrantIndex(client, 'different', spec, vault_id='v', create=True).upsert(records, vectors[::-1])
        reference = partial(search_qdrant, client, 'reference', spec=spec, vault_id='v')
        alternate = partial(search_qdrant, client, 'different', spec=spec, vault_id='v')
        result = compare_retrieval(records, vectors, [[1, 0]], cases, spec=spec, vault_id='v', top_k=1,
                                   qdrant_search=reference, backends={'different': (alternate, False)})
        assert result['summary']['qdrant_exact']['neighbor_recall_at_k'] == 1
        assert result['summary']['qdrant_exact']['section_coverage'] == 0
        assert result['summary']['different']['neighbor_recall_at_k'] == 0
        assert result['summary']['different']['section_coverage'] == 1
        assert result['settings']['reference'] == 'qdrant_exact'
        assert result['settings']['vector_bytes'] == vectors.nbytes
        with pytest.raises(ValueError, match='unique'):
            compare_retrieval(records, vectors, [[1, 0], [1, 0]], cases * 2, spec=spec, vault_id='v', qdrant_search=reference)


def test_context_comparison_isolates_processing_from_budget_and_measures_span_union():
    from obsidian_rag.context import ContextConfig, GenerationCounter, build_context
    from obsidian_rag.evaluation import compare_contexts
    from tests.result_fixtures import make_result as SearchResult, chunk_of, record_of
    note = Note('Title', 'x' * 300, 'a.md')
    hits = []
    for index, (start, end) in enumerate([(0, 200), (150, 300)]):
        chunk = Chunk(note.content[start:end], note.title, note.source, index, start, end)
        record = ChunkRecord.from_note(chunk, note=note, vault_id='v')
        hits.append(SearchResult(chunk, .9 - index / 10, record, 'snapshot'))
    counter = GenerationCounter('test', 'test-count', lambda messages: 10 + sum(len(m['content']) for m in messages))
    limit = counter(build_context('Q?', hits).messages)
    config = ContextConfig(limit + 20, 20, 0)
    case = {'required_source_groups': [['a.md']],
            'evidence_anchors': [{'source': 'a.md', 'body_start_char': 0, 'body_end_char': 300}]}
    modes = compare_contexts('Q?', hits, case, config=config, counter=counter)
    raw, budgeted, built = [modes[name]['metrics'] for name in ('raw', 'raw_budgeted', 'built')]
    assert raw['body_characters'] == 350
    assert raw['duplicate_span_fraction'] == pytest.approx(50 / 350)
    assert raw['section_coverage'] == 1 and not raw['fits_budget']
    assert budgeted['section_coverage'] == pytest.approx(2 / 3)
    assert built['section_coverage'] == built['section_coverage_retention'] == 1
    assert built['duplicate_span_fraction'] == 0
    assert built['fits_budget'] and built['prompt_tokens'] == limit
    assert built['body_characters'] == 300 and built['block_count'] == 1
    assert modes['built']['context']['citation_map'] == {'a.md': [0]}
    with pytest.raises(ValueError, match='snapshot-identified'):
        compare_contexts('Q?', [SearchResult(chunk_of(hits[0]), .5)], case, config=config, counter=counter)


def citation_fixture():
    from obsidian_rag.context import ContextConfig, GenerationCounter, build_context
    from tests.result_fixtures import make_result as SearchResult, chunk_of, record_of
    note = Note('Title', 'Only small datasets were faster.', 'a.md')
    chunk = whole_note_chunks([note])[0]
    counter = GenerationCounter('test', 'chars', lambda m: 10 + sum(len(x['content']) for x in m))
    return build_context('Which datasets were faster?', [SearchResult(chunk, .9)],
                          config=ContextConfig(), counter=counter, citation_mode='structured')


def test_citation_metrics_separate_valid_links_from_wrong_claims_and_missing_facts():
    import json
    from obsidian_rag.evaluation import citation_statistics
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
    from obsidian_rag.evaluation import citation_statistics
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


def test_citation_evaluation_retains_failed_raw_output_and_counts_failures():
    from unittest.mock import Mock
    from ollama import Client, ChatResponse, Message
    from obsidian_rag.evaluation import evaluate_citation_context, summarize_citations
    client = Mock(spec=Client)
    client.chat.return_value = ChatResponse(message=Message(role='assistant', content='{'), done_reason='length')
    row = evaluate_citation_context(citation_fixture(), client=client)
    assert row['success'] is False and row['raw_response'] == '{'
    assert row['error']['code'] == 'truncated_output'
    assert row['context']['citation_sources'][0]['content'] == 'Only small datasets were faster.'
    summary = summarize_citations([row])
    assert summary['error_count'] == 1 and summary['structure_valid'] == 0
    assert summary['supported_claim_rate'] is None
    assert summary['citation_id_validity_defined_cases'] == 0


def test_citation_case_records_prompt_budget_failure_without_calling_model():
    from unittest.mock import Mock
    from obsidian_rag.context import ContextConfig
    from obsidian_rag.evaluation import evaluate_citation_case
    context = citation_fixture()
    client = Mock()
    row = evaluate_citation_case('Question?', list(context.evidence_blocks[0].origins),
                                 config=ContextConfig(30, 10, 0), counter=context.counter, client=client)
    assert row['error']['code'] == 'context_budget' and row['context'] is None
    client.chat.assert_not_called()
