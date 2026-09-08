from contextlib import closing
from dataclasses import replace
import warnings
import numpy as np
import pytest
from qdrant_client import QdrantClient
from obsidian_rag.knowledge_base.chunking import Chunk, chunk_notes, whole_note_chunks
from obsidian_rag.knowledge_base.vector_index.indexing import QdrantIndex
from obsidian_rag.knowledge_base.loaders import Note
from obsidian_rag.knowledge_base.vector_index.qdrant import search_qdrant
from obsidian_rag.knowledge_base.vector_index.manifest import ChunkRecord, EmbeddingSpec


@pytest.fixture
def published_index(tmp_path):
    from unittest.mock import Mock
    from ollama import Client, EmbedResponse
    from tokenizers import Tokenizer, models, pre_tokenizers
    from obsidian_rag.knowledge_base.vector_index.manifest import EmbeddingSpec
    from obsidian_rag.knowledge_base.vector_index.indexing import build_index
    from obsidian_rag.knowledge_base.loaders import Note
    from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2, document_template='title-body-v1')
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1, 0] for _ in kw['input']])
    with closing(QdrantClient(':memory:')) as qc, SQLiteStorage(tmp_path / 'db') as store:
        build_index(store, [Note(title='Title', content='Original source text', source='a.md')],
                    spec=spec, vault_id='vault', tokenizer=tokenizer, max_input_tokens=100,
                    client=client, chunking='none', index_version='v1', query_instruction='Find evidence.', qdrant_client=qc,
                    backend={'kind': 'qdrant', 'url': 'http://test'})
        client.embed.reset_mock()
        yield store, dict(vault_id='vault', spec=spec, tokenizer=tokenizer, client=client, qdrant_client=qc)


def test_indexed_search_only_embeds_query_and_restores_snapshot_content(published_index):
    from obsidian_rag.retrieval.vector import vector_search
    store, kwargs = published_index
    results = vector_search(store, 'Question?', **kwargs).items
    kwargs['client'].embed.assert_called_once_with(
        model='test', input=['Instruct: Find evidence.\nQuery:Question?'], truncate=False, options={'num_ctx': 100})
    assert results[0].excerpts[0].content == 'Original source text'
    assert results[0].source.path == 'a.md'
    assert results[0].score.value == 1
    assert results[0].source.snapshot_id == 'v1'
    assert results[0].target.chunk_id == store.snapshot_records('v1')[0].chunk_id


def test_indexed_search_rejects_model_and_tokenizer_mismatches_before_embedding(published_index):
    from dataclasses import replace
    from obsidian_rag.retrieval.vector import vector_search
    store, kwargs = published_index
    with pytest.raises(ValueError, match='incompatible'):
        vector_search(store, 'Question?', **{**kwargs, 'spec': replace(kwargs['spec'], model_revision='changed')})
    kwargs['tokenizer'].enable_padding(length=10)
    with pytest.raises(ValueError, match='tokenizer'):
        vector_search(store, 'Question?', **kwargs)
    kwargs['client'].embed.assert_not_called()


def test_indexed_search_rejects_missing_and_unpublished_versions(published_index):
    from dataclasses import replace
    from obsidian_rag.retrieval.vector import vector_search
    store, kwargs = published_index
    with pytest.raises(ValueError, match='No published'):
        vector_search(store, 'Question?', **{**kwargs, 'vault_id': 'missing'})
    building = replace(store.get_manifest('v1'), index_version='v2', status='building')
    store.create_build(building, corpus_fingerprint='pending')
    with pytest.raises(ValueError, match='ready'):
        vector_search(store, 'Question?', **kwargs, index_version='v2')
    kwargs['client'].embed.assert_not_called()


def test_indexed_search_rejects_unknown_backend_hits(published_index, monkeypatch):
    from obsidian_rag.retrieval.vector import vector_search
    from obsidian_rag.knowledge_base.vector_index.manifest import VectorHit
    store, kwargs = published_index
    monkeypatch.setattr('obsidian_rag.retrieval.vector.search_qdrant', lambda *a, **kw: [VectorHit('orphan', 0.5)])
    with pytest.raises(ValueError, match='snapshot'):
        vector_search(store, 'Question?', **kwargs)


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
def test_qdrant_local_contract_roundtrip_filter_and_delete(tmp_path, vector_data):
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
        store.delete([records[0].chunk_id, records[0].chunk_id])
        assert store.count() == 2
        assert search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', source='0.md') == []


def test_search_keeps_requested_snapshot_after_a_new_revision_is_published(published_index):
    from obsidian_rag.knowledge_base.vector_index.indexing import build_index
    from obsidian_rag.retrieval.vector import vector_search
    store, kwargs = published_index
    build_index(store, [Note('Title', 'Updated source text', 'a.md')],
                spec=kwargs['spec'], vault_id='vault', tokenizer=kwargs['tokenizer'],
                max_input_tokens=100, client=kwargs['client'], chunking='none',
                index_version='v2', query_instruction='Find evidence.', qdrant_client=kwargs['qdrant_client'],
                    backend={'kind': 'qdrant', 'url': 'http://test'})
    old = vector_search(store, 'Question?', **kwargs, index_version='v1').items[0]
    new = vector_search(store, 'Question?', **kwargs).items[0]
    assert old.excerpts[0].content == 'Original source text'
    assert old.source.snapshot_id == 'v1'
    assert new.source.snapshot_id == 'v2'
    assert old.source.document_id == new.source.document_id
    assert old.source.document_revision != new.source.document_revision
