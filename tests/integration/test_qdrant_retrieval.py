"""Opt-in real Qdrant Server contract; collections are unique and cleaned up."""

from contextlib import closing
from functools import partial
import os
from uuid import uuid4

import numpy as np
import pytest
from qdrant_client import QdrantClient

from arkb.indexing.chunking import whole_note_chunks
from arkb.indexing import QdrantIndex
from arkb.indexing.loaders import Note
from arkb.storage import check_qdrant_collection
from arkb.retrieval import search_qdrant
from arkb.schema import ChunkRecord, EmbeddingSpec


pytestmark = pytest.mark.skipif(not os.environ.get('OBSIDIAN_RAG_QDRANT_URL'),
                               reason='Set OBSIDIAN_RAG_QDRANT_URL to a test Qdrant Server.')


def test_server_persistence_filters_and_scores():
    name = 'obsidian_rag_test_' + uuid4().hex
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2,
                         document_template='title-body-v1', normalization='none')
    records = []
    for i in range(3):
        note = Note(title='Title', content='body', source=f'{i}.md')
        records.append(ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id='test'))
    url = os.environ['OBSIDIAN_RAG_QDRANT_URL']
    with closing(QdrantClient(url=url, timeout=15, trust_env=False)) as client:
        try:
            store = QdrantIndex(client, name, spec, vault_id='test', create=True)
            store.upsert(records, [[1, 0], [.6, .8], [0, 1]])
            assert set(store.check_configuration().payload_schema) >= {'source', 'vault_id', 'embedding_spec'}
            with closing(QdrantClient(url=url, timeout=15, trust_env=False)) as second:
                reopened = QdrantIndex(second, name, spec, vault_id='test')
                check_qdrant_collection(second, name, spec=spec, vault_id='test')
                search = partial(search_qdrant, second, name, spec=spec, vault_id='test')
                hits = search([1, 0], top_k=3, exact=True)
                assert [h.chunk_id for h in hits] == [r.chunk_id for r in records]
                np.testing.assert_allclose([h.score for h in hits], [1, .6, 0], atol=1e-6)
                assert search([1, 0], top_k=1, source='2.md')[0].chunk_id == records[2].chunk_id
                assert reopened.count() == 3
                assert search([1, 0], top_k=1, source='missing.md') == []
        finally:
            if client.collection_exists(name):
                client.delete_collection(name)
