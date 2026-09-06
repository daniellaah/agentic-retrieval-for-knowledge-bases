from contextlib import closing
from dataclasses import replace
import warnings

import numpy as np
import pytest
from qdrant_client import QdrantClient

from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.index_schema import ChunkRecord, EmbeddingSpec
from obsidian_rag.notes import Note
from obsidian_rag.vector_store_qdrant import QdrantVectorStore


pytestmark = pytest.mark.filterwarnings('ignore:Local mode performs exact.*:UserWarning')

@pytest.fixture
def data():
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1', normalization='none')
    notes = [Note(title='Title', content='body', source=f'{n}.md') for n in range(3)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='vault') for n in notes]
    return spec, records


def create(client, spec):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        return QdrantVectorStore(client, 'test', spec, vault_id='vault', create=True)


def test_qdrant_local_contract_roundtrip_filter_and_delete(tmp_path, data):
    spec, records = data
    with closing(QdrantClient(path=str(tmp_path / 'qdrant'))) as client:
        store = create(client, spec)
        store.upsert(records, [[0, 1], [3, 4], [1, 0]])
        assert store.count() == 3
        hits = store.search([1, 0], top_k=3, exact=True)
        assert [h.chunk_id for h in hits] == [records[2].chunk_id, records[1].chunk_id, records[0].chunk_id]
        np.testing.assert_allclose([h.score for h in hits], [1, .6, 0], atol=1e-6)
        assert store.search([1, 0], source='0.md')[0].chunk_id == records[0].chunk_id
    with closing(QdrantClient(path=str(tmp_path / 'qdrant'))) as client:
        store = QdrantVectorStore(client, 'test', spec, vault_id='vault')
        assert store.count() == 3
        store.upsert([records[2]], [[-1, 0]])
        assert store.count() == 3
        store.delete([records[0].chunk_id, records[0].chunk_id])
        assert store.count() == 2
        assert store.search([1, 0], source='0.md') == []


def test_incompatible_collection_is_rejected_without_overwriting(data):
    spec, records = data
    with closing(QdrantClient(':memory:')) as client:
        store = create(client, spec)
        store.upsert(records[:1], [[1, 0]])
        for changed in (replace(spec, dimensions=3), replace(spec, model_revision='different')):
            with pytest.raises(ValueError, match='configuration'):
                QdrantVectorStore(client, 'test', changed, vault_id='vault')
        with pytest.raises(ValueError):
            create(client, spec)
        assert store.count() == 1


def test_invalid_vectors_and_foreign_vault_fail_before_upsert(data):
    spec, records = data
    with closing(QdrantClient(':memory:')) as client:
        store = create(client, spec)
        for items, vectors in ((records[:1], [[0, 0]]),
                               ([replace(records[0], vault_id='other')], [[1, 0]])):
            with pytest.raises(ValueError):
                store.upsert(items, vectors)
        assert store.count() == 0


def test_snapshot_verification_detects_payload_and_vector_corruption(data):
    from obsidian_rag.vector_store_qdrant import point_id
    spec, records = data
    with closing(QdrantClient(':memory:')) as client:
        store = create(client, spec)
        vectors = [[1, 0], [.6, .8], [0, 1]]
        store.upsert(records, vectors)
        store.verify_snapshot(records, vectors)
        client.set_payload('test', payload={'source': 'wrong.md'}, points=[point_id(records[0].chunk_id)])
        with pytest.raises(ValueError, match='payload'):
            store.verify_snapshot(records, vectors)


def test_hnsw_readiness_timeout_does_not_claim_a_flat_index_is_built(data):
    spec, records = data
    with closing(QdrantClient(':memory:')) as client:
        store = create(client, spec)
        store.upsert(records[:1], [[1, 0]])
        assert store.wait_ready(expected_count=1)['points'] == 1
        with pytest.raises(ValueError, match='Timed out'):
            store.wait_ready(expected_count=1, timeout=.01, require_hnsw=True)


def test_invalid_full_scan_threshold_fails_before_contacting_server(data):
    spec, _ = data
    for threshold in (0, 9, -1, True):
        with pytest.raises(ValueError, match='full_scan_threshold'):
            QdrantVectorStore(None, 'test', spec, vault_id='vault', create=True,
                             full_scan_threshold=threshold)
