from contextlib import closing
import warnings

import numpy as np
import pytest
from qdrant_client import QdrantClient

from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.models import ChunkRecord, EmbeddingSpec, Note
from arkb.knowledge.qdrant import QdrantIndex, search_qdrant


@pytest.fixture
def vector_data():
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1', normalization='none')
    notes = [Note(title='Title', content='body', source=f'{n}.md') for n in range(3)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='vault') for n in notes]
    return spec, records


def create_qdrant_index(client, spec):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        return QdrantIndex(client, 'test', spec, vault_id='vault', create=True)


@pytest.mark.filterwarnings('ignore:Local mode performs exact.*:UserWarning')
def test_qdrant_local_contract_roundtrip_filter_and_update(tmp_path, vector_data):
    spec, records = vector_data
    with closing(QdrantClient(path=str(tmp_path / 'qdrant'))) as client:
        store = create_qdrant_index(client, spec)
        store.upsert(records, [[0, 1], [3, 4], [1, 0]])
        assert store.count() == 3
        hits = search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', top_k=3, exact=True)
        assert [h.chunk_id for h in hits] == [records[2].chunk_id, records[1].chunk_id, records[0].chunk_id]
        np.testing.assert_allclose([h.score for h in hits], [1, .6, 0], atol=1e-6)
        assert search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', source='0.md')[0].chunk_id == records[0].chunk_id
    with closing(QdrantClient(path=str(tmp_path / 'qdrant'))) as client:
        store = QdrantIndex(client, 'test', spec, vault_id='vault')
        assert store.count() == 3
        store.upsert([records[2]], [[-1, 0]])
        assert store.count() == 3
        assert search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', source='missing.md') == []


@pytest.mark.parametrize('query', [[1, 0, 0], [0, 0], [float('nan'), 0], [[1, 0]]])
def test_qdrant_rejects_invalid_query_vectors_before_search(vector_data, query):
    from unittest.mock import Mock
    spec, _ = vector_data
    client = Mock()
    with pytest.raises(ValueError):
        search_qdrant(client, 'test', query, spec=spec, vault_id='vault')
    client.query_points.assert_not_called()


@pytest.mark.parametrize('options', [{'top_k': 0}, {'top_k': True}, {'source': ''}, {'exact': 1}])
def test_qdrant_rejects_invalid_search_options_before_search(vector_data, options):
    from unittest.mock import Mock
    spec, _ = vector_data
    client = Mock()
    with pytest.raises(ValueError):
        search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', **options)
    client.query_points.assert_not_called()


def test_qdrant_ties_and_float32_score_tolerance_preserve_defined_cosine_semantics(vector_data):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from arkb.knowledge.qdrant import point_id
    spec, records = vector_data
    points = [SimpleNamespace(id=point_id(r.chunk_id), score=1.000001,
                               payload={'chunk_id': r.chunk_id, 'vault_id': 'vault',
                                        'embedding_spec': spec.fingerprint}) for r in records]
    client = Mock()
    client.query_points.return_value = SimpleNamespace(points=points[::-1])
    hits = search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', top_k=3)
    assert [h.chunk_id for h in hits] == sorted(r.chunk_id for r in records)
    assert [h.score for h in hits] == [1.0] * 3
    points[0].score = 1.01
    with pytest.raises(ValueError, match='cosine score'):
        search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', top_k=3)
