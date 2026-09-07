from dataclasses import replace

import numpy as np
import pytest

from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.schema import ChunkRecord, EmbeddingSpec
from obsidian_rag.loaders import Note
from obsidian_rag.vector_store import NumpyVectorStore


def records():
    notes = [Note(title='Title', content='Same body', source=f'{n}.md') for n in range(3)]
    return [ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id='vault') for note in notes]


@pytest.fixture
def store():
    return NumpyVectorStore(EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                            document_template='title-body-v1', normalization='none'), vault_id='vault')


def test_exact_ranking_filters_before_limit_and_preserves_duplicate_sources(store):
    items = records()
    store.upsert(items, [[0, 1], [3, 4], [1, 0]])
    hits = store.search([1, 0], top_k=3)
    assert [h.chunk_id for h in hits] == [items[2].chunk_id, items[1].chunk_id, items[0].chunk_id]
    np.testing.assert_allclose([h.score for h in hits], [1, .6, 0])
    assert store.search([1, 0], top_k=1, source='0.md')[0].chunk_id == items[0].chunk_id
    assert store.search([1, 0], source='missing.md') == []


def test_upsert_delete_empty_and_ties(store):
    items = records()
    vectors = np.array([[1., 0], [1, 0], [0, 1]])
    store.upsert(items, vectors)
    vectors[:] = 0
    assert [h.chunk_id for h in store.search([1, 0])] == [r.chunk_id for r in items[:2]]
    store.upsert([items[0]], [[-1, 0]])
    assert store.count() == 3
    assert store.search([1, 0])[0].chunk_id == items[1].chunk_id
    store.delete([r.chunk_id for r in items] + ['absent'])
    assert store.count() == 0
    assert store.search([1, 0]) == []


def test_invalid_batch_is_atomic(store):
    items = records()
    store.upsert([items[0]], [[1, 0]])
    for batch, vectors in ((items[1:], [[0, 1], [0, 0]]),
                           ([items[1], items[1]], [[0, 1], [0, 1]]),
                           ([replace(items[1], vault_id='other')], [[0, 1]])):
        with pytest.raises(ValueError):
            store.upsert(batch, vectors)
        assert store.count() == 1


@pytest.mark.parametrize('options', [{'top_k': 0}, {'top_k': True}, {'source': ''}, {'exact': 1}])
def test_empty_store_still_validates_search_options(store, options):
    with pytest.raises(ValueError):
        store.search([1, 0], **options)


def test_query_dimension_and_norm_are_checked(store):
    for query in ([1, 0, 0], [0, 0], [float('nan'), 0]):
        with pytest.raises(ValueError):
            store.search(query)


def test_distinct_records_can_reference_the_same_chunk_object(store):
    first = records()[0]
    second = replace(first, document_revision='a' * 64)
    store.upsert([first, second], [[1, 0], [0, 1]])
    assert [h.chunk_id for h in store.search([1, 0])] == [first.chunk_id, second.chunk_id]
