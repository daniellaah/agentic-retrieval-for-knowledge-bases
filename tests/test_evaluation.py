import numpy as np
import pytest

from obsidian_rag.chunking import Chunk, whole_note_chunks
from obsidian_rag.evaluation import compare_retrieval, evidence_statistics, recall_at_k
from obsidian_rag.schema import ChunkRecord, EmbeddingSpec
from obsidian_rag.loaders import Note
from obsidian_rag.vector_store import NumpyVectorStore


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
    alternate = NumpyVectorStore(spec, vault_id='v')
    alternate.upsert(records, vectors[::-1])
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
