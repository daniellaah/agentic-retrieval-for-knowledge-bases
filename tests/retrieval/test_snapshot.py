from dataclasses import replace

import pytest

from arkb.knowledge.models import Note, QdrantConfig


@pytest.fixture
def published_index(tmp_path, qdrant, qdrant_config):
    from unittest.mock import Mock
    from ollama import Client, EmbedResponse
    from tokenizers import Tokenizer, models, pre_tokenizers
    from arkb.knowledge.models import EmbeddingSpec
    from arkb.knowledge.indexing import build_index
    from arkb.knowledge.models import Note
    from arkb.knowledge.sqlite import SQLiteStorage
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2, document_template='title-body-v1')
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1, 0] for _ in kw['input']])
    with SQLiteStorage(tmp_path / 'db') as store:
        build_index(store, [Note(title='Title', content='Original source text', source='a.md')],
                    spec=spec, vault_id='vault', tokenizer=tokenizer, max_input_tokens=100,
                    client=client, qdrant_client=qdrant, qdrant_config=qdrant_config, chunking='none', index_version='v1', query_instruction='Find evidence.')
        client.embed.reset_mock()
        yield store, dict(vault_id='vault', spec=spec, tokenizer=tokenizer, client=client, qdrant_client=qdrant)


def test_indexed_search_only_embeds_query_and_restores_snapshot_content(published_index):
    from arkb.runtime import search_index
    store, kwargs = published_index
    response = search_index(store, 'Question?', **kwargs)
    results = response.results
    assert response.query == 'Question?' and response.index_id == 'v1'
    kwargs['client'].embed.assert_called_once_with(
        model='test', input=['Instruct: Find evidence.\nQuery:Question?'], truncate=False, options={'num_ctx': 100})
    assert results[0].content == 'Original source text'
    assert results[0].source == 'a.md'
    assert results[0].score == 1
    assert results[0].metadata['index_version'] == 'v1'
    record = store.snapshot_records('v1')[0]
    assert results[0].source_id == record.document_id
    assert results[0].chunk_id == record.chunk_id
    assert results[0].metadata['document_revision'] == record.document_revision
    assert results[0].score_type == 'cosine_similarity'


def test_indexed_search_rejects_model_and_tokenizer_mismatches_before_embedding(published_index):
    from dataclasses import replace
    from arkb.runtime import search_index
    store, kwargs = published_index
    with pytest.raises(ValueError, match='incompatible'):
        search_index(store, 'Question?', **{**kwargs, 'spec': replace(kwargs['spec'], model_revision='changed')})
    kwargs['tokenizer'].enable_padding(length=10)
    with pytest.raises(ValueError, match='tokenizer'):
        search_index(store, 'Question?', **kwargs)
    kwargs['client'].embed.assert_not_called()


def test_indexed_search_rejects_missing_and_unpublished_versions(published_index):
    from dataclasses import replace
    from arkb.runtime import search_index
    store, kwargs = published_index
    with pytest.raises(ValueError, match='No published'):
        search_index(store, 'Question?', **{**kwargs, 'vault_id': 'missing'})
    building = replace(store.get_manifest('v1'), index_version='v2', status='building')
    store.create_build(building, corpus_fingerprint='pending', backend={'kind': 'qdrant'})
    with pytest.raises(ValueError, match='ready'):
        search_index(store, 'Question?', **kwargs, index_version='v2')
    kwargs['client'].embed.assert_not_called()


def test_indexed_search_rejects_unknown_backend_hits(published_index, monkeypatch):
    from arkb.runtime import search_index
    from arkb.knowledge.models import VectorHit
    store, kwargs = published_index
    monkeypatch.setattr('arkb.knowledge.qdrant.search_qdrant', lambda *a, **kw: [VectorHit('orphan', 0.5)])
    with pytest.raises(ValueError, match='snapshot'):
        search_index(store, 'Question?', **kwargs)








def test_search_keeps_requested_snapshot_after_a_new_revision_is_published(published_index):
    from arkb.knowledge.indexing import build_index
    from arkb.runtime import search_index
    store, kwargs = published_index
    build_index(store, [Note('Title', 'Updated source text', 'a.md')],
                spec=kwargs['spec'], vault_id='vault', tokenizer=kwargs['tokenizer'],
                max_input_tokens=100, client=kwargs['client'], chunking='none',
                qdrant_client=kwargs['qdrant_client'], qdrant_config=QdrantConfig.from_metadata(store.build_metadata('v1')['backend']),
                index_version='v2', query_instruction='Find evidence.')
    old = search_index(store, 'Question?', **kwargs, index_version='v1').results[0]
    new = search_index(store, 'Question?', **kwargs).results[0]
    assert old.content == 'Original source text'
    assert old.metadata['index_version'] == 'v1'
    assert new.metadata['index_version'] == 'v2'
    assert old.source_id == new.source_id
    assert old.metadata['document_revision'] != new.metadata['document_revision']






def test_empty_retired_snapshot_still_requires_rebuild(published_index):
    import json
    from arkb.runtime import search_index
    store, kwargs = published_index
    manifest = replace(store.get_manifest('v1'), document_count=0, chunk_count=0)
    from dataclasses import asdict
    store.connection.execute('UPDATE builds SET manifest=?, backend=? WHERE version=?',
                             (json.dumps(asdict(manifest)), json.dumps({'kind': 'numpy'}), 'v1'))
    store.connection.execute("DELETE FROM snapshot_chunks WHERE version='v1'")
    store.connection.commit()
    with pytest.raises(ValueError, match='run arkb index'):
        search_index(store, 'Q?', **kwargs)
    kwargs['client'].embed.assert_not_called()


def test_query_adapter_forwards_options_and_reads_only_hit_records(published_index, monkeypatch):
    from unittest.mock import Mock
    from arkb.runtime import search_index
    from arkb.knowledge.sqlite import SQLiteStorage
    store, kwargs = published_index
    client = kwargs['qdrant_client']
    search = Mock(wraps=client.query_points)
    monkeypatch.setattr(client, 'query_points', search)
    for operation in ('upsert', 'create_collection', 'delete_collection'):
        monkeypatch.setattr(client, operation, lambda *a, **kw: pytest.fail('retrieval wrote to Qdrant'))
    with SQLiteStorage(store.path, read_only=True) as reader:
        for operation in ('load_snapshot', 'put_embeddings', 'get_embedding', 'snapshot_records'):
            monkeypatch.setattr(reader, operation, lambda *a, **kw: pytest.fail('retrieval accessed offline vector state'))
        response = search_index(reader, 'Q?', **kwargs, top_k=1, source='a.md', exact=True, ef_search=37)
    options = search.call_args.kwargs
    assert options['limit'] == 1
    assert options['search_params'].exact is True
    assert options['search_params'].hnsw_ef == 37
    assert {condition.key: condition.match.value for condition in options['query_filter'].must} == {
        'source': 'a.md', 'vault_id': 'vault', 'embedding_spec': kwargs['spec'].fingerprint,
    }
    hit = response.results[0]
    assert type(hit).__module__ == 'arkb.retrieval.models'
    assert hit.source_id == store.snapshot_records('v1')[0].document_id
    assert hit.metadata['index_version'] == 'v1'




@pytest.mark.parametrize('corruption', ['duplicate', 'wrong_source', 'bad_score'])
def test_snapshot_adapter_rejects_corrupt_backend_hits(published_index, monkeypatch, corruption):
    from arkb.runtime import search_index
    from arkb.knowledge.models import VectorHit
    store, kwargs = published_index
    record = store.snapshot_records('v1')[0]
    hits = [VectorHit(record.chunk_id, float('nan') if corruption == 'bad_score' else .5)]
    if corruption == 'duplicate':
        hits *= 2
    monkeypatch.setattr('arkb.knowledge.qdrant.search_qdrant', lambda *a, **kw: hits)
    with pytest.raises(ValueError, match='duplicate|snapshot'):
        search_index(store, 'Q?', **kwargs, source='wrong.md' if corruption == 'wrong_source' else None)


def test_empty_snapshot_validates_input_without_model_or_vector_search(published_index):
    from arkb.knowledge.indexing import build_index
    from arkb.runtime import search_index
    store, kwargs = published_index
    build_index(store, [], spec=kwargs['spec'], vault_id='vault', tokenizer=kwargs['tokenizer'],
                max_input_tokens=100, client=kwargs['client'], qdrant_client=kwargs['qdrant_client'],
                qdrant_config=QdrantConfig(), index_version='empty')
    response = search_index(store, 'Q?', **{**kwargs, 'qdrant_client': None})
    assert response.results == () and response.index_id == 'empty'
    with pytest.raises(ValueError, match='nonblank'):
        search_index(store, ' ', **kwargs)
    with pytest.raises(ValueError, match='maximum'):
        search_index(store, 'word ' * 101, **kwargs)
    kwargs['client'].embed.assert_not_called()


def test_opened_snapshot_stays_pinned_and_preserves_markdown_provenance(published_index):
    from arkb.knowledge.indexing import build_index
    from arkb.retrieval.semantic import QdrantSnapshotIndex
    from arkb.runtime import search_index
    store, kwargs = published_index
    pinned = QdrantSnapshotIndex(store, kwargs['qdrant_client'], vault_id='vault', exact=True)
    build_index(store, [Note('Title', '## Section\nBody facts.', 'a.md')],
                spec=kwargs['spec'], vault_id='vault', tokenizer=kwargs['tokenizer'],
                max_input_tokens=100, client=kwargs['client'], qdrant_client=kwargs['qdrant_client'],
                qdrant_config=QdrantConfig(), index_version='markdown', chunk_size=20, chunk_overlap=0)
    old = pinned.search([1, 0], top_k=1, filters={})[0]
    new = search_index(store, 'Q?', **kwargs).results[0]
    assert pinned.index_id == old.metadata['index_version'] == 'v1'
    assert old.content == 'Original source text'
    assert new.metadata['index_version'] == 'markdown'
    assert old.source_id == new.source_id
    record = store.get_record('markdown', new.chunk_id)
    assert new.metadata['heading_path'] == list(record.chunk.heading_path) == ['Section']
    assert new.metadata['section_id'] == record.chunk.section_id
    assert new.metadata['section_start_char'] == record.chunk.section_start_char
    assert new.metadata['section_end_char'] == record.chunk.section_end_char
    assert new.start_char == record.chunk.start_char and new.end_char == record.chunk.end_char
    assert new.metadata['document_revision'] == record.document_revision
