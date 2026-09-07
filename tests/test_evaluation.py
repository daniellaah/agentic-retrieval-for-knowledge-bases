from functools import partial

import numpy as np
import pytest

from obsidian_rag.chunking import Chunk, whole_note_chunks
from obsidian_rag.evaluation import compare_retrieval, evidence_statistics, recall_at_k
from obsidian_rag.loaders import Note
from obsidian_rag.retrieval import search_numpy
from obsidian_rag.schema import ChunkRecord, EmbeddingSpec


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


def test_comparison_separates_neighbor_recall_from_evidence_coverage():
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2, document_template='title-body-v1')
    notes = [Note(title='T', content='evidence', source=f'{i}.md') for i in range(2)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='v') for n in notes]
    vectors = np.eye(2)
    alternate = partial(search_numpy, records, vectors[::-1], spec=spec, vault_id='v')
    cases = [{'id': 'q1', 'question': 'Question?', 'required_source_groups': [['1.md']],
              'evidence_anchors': [{'source': '1.md', 'body_start_char': 0, 'body_end_char': 8}]}]
    result = compare_retrieval(records, vectors, [[1, 0]], cases, spec=spec, vault_id='v', top_k=1,
                               backends={'different': (alternate, False)})
    assert result['summary']['numpy_exact']['neighbor_recall_at_k'] == 1
    assert result['summary']['numpy_exact']['section_coverage'] == 0
    assert result['summary']['different']['neighbor_recall_at_k'] == 0
    assert result['summary']['different']['section_coverage'] == 1
    assert result['settings']['vector_bytes'] == vectors.nbytes
    with pytest.raises(ValueError, match='unique'):
        compare_retrieval(records, vectors, [[1, 0], [1, 0]], cases * 2, spec=spec, vault_id='v')


def test_context_comparison_isolates_processing_from_budget_and_measures_span_union():
    from obsidian_rag.context import ContextConfig, GenerationCounter, build_context
    from obsidian_rag.evaluation import compare_contexts
    from obsidian_rag.retrieval import SearchResult
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
        compare_contexts('Q?', [SearchResult(hits[0].chunk, .5)], case, config=config, counter=counter)
