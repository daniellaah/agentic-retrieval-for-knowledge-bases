from contextlib import closing
from dataclasses import replace
from unittest.mock import Mock
import warnings

from ollama import Client, EmbedResponse
import pytest
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.indexing import build_index
from arkb.knowledge.models import QdrantConfig
from arkb.knowledge.qdrant import QdrantIndex
from arkb.knowledge.models import Note
from arkb.knowledge.models import EmbeddingSpec, ChunkRecord
from arkb.knowledge.sqlite import SQLiteStorage


@pytest.fixture
def setup(tmp_path, qdrant, qdrant_config):
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[END]': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(single='$A [END]', special_tokens=[('[END]', 1)])
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2, document_template='title-body-v1')
    with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        yield storage, dict(spec=spec, vault_id='vault', client=client, tokenizer=tokenizer,
                            max_input_tokens=100, chunking='none', qdrant_client=qdrant, qdrant_config=qdrant_config)


def test_complete_build_preserves_duplicate_occurrences_and_reuses_cache(setup):
    storage, options = setup
    notes = [Note(title='Title', content='body', source=f'{n}.md') for n in range(2)]
    report = build_index(storage, notes, **options, index_version='first')
    assert report.manifest.status == 'ready'
    assert report.manifest.schema_version == 2
    assert storage.build_metadata('first')['backend']['collection'].startswith('arkb_')
    assert report.embedded_inputs == 1
    assert len(storage.load_snapshot('first')[1]) == 2
    again = build_index(storage, notes, **options, index_version='second')
    assert again.embedded_inputs == 0 and again.cached_inputs == 1
    assert options['client'].embed.call_count == 1
    assert storage.active_manifest('vault').index_version == 'second'


@pytest.mark.parametrize('corruption,message', [
    ('record', 'snapshot record'), ('cache', 'checksum'), ('qdrant', 'payload'),
])
def test_unchanged_build_still_rejects_corrupt_snapshot(setup, corruption, message):
    from arkb.knowledge.qdrant import point_id

    storage, options = setup
    notes = [Note(title='A', content='first', source='a.md')]
    first = build_index(storage, notes, **options)
    version = first.manifest.index_version
    if corruption == 'record':
        storage.connection.execute('UPDATE snapshot_chunks SET ordinal=1 WHERE version=?', (version,))
    elif corruption == 'cache':
        storage.connection.execute("UPDATE embeddings SET checksum='corrupt'")
    else:
        collection = storage.build_metadata(version)['backend']['collection']
        record = storage.snapshot_records(version)[0]
        options['qdrant_client'].set_payload(collection, payload={'source': 'wrong.md'},
                                              points=[point_id(record.chunk_id)])
    options['client'].embed.reset_mock()
    with pytest.raises(ValueError, match=message):
        build_index(storage, notes, **options)
    options['client'].embed.assert_not_called()
    assert storage.active_manifest('vault') == first.manifest
    assert storage.list_builds('vault') == [first.manifest]


def test_failure_keeps_previous_snapshot_and_successful_embedding_batches(setup):
    storage, options = setup
    notes = [Note(title='A', content='first', source='a.md')]
    build_index(storage, notes, **options, index_version='old')
    options['client'].embed.side_effect = [EmbedResponse(embeddings=[[0, 1]]), ConnectionError('offline')]
    new_notes = [Note(title='B', content='second', source='b.md'), Note(title='C', content='third', source='c.md')]
    with pytest.raises(ConnectionError):
        build_index(storage, new_notes, **options, batch_size=1, index_version='failed')
    assert storage.get_manifest('failed').status == 'failed'
    assert storage.active_manifest('vault').index_version == 'old'
    options['client'].embed.reset_mock()
    options['client'].embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1, 0] for _ in kw['input']])
    report = build_index(storage, new_notes, **options, index_version='retry')
    assert report.embedded_inputs == 1 and report.cached_inputs == 1


def test_preflight_overflow_and_duplicate_sources_make_no_requests_or_builds(setup):
    storage, options = setup
    note = Note(title='A title', content=' '.join(['word'] * 200), source='a.md')
    with pytest.raises(ValueError, match='a.md'):
        build_index(storage, [note], **options)
    with pytest.raises(ValueError, match='unique'):
        build_index(storage, [note, note], **options)
    options['client'].embed.assert_not_called()
    assert storage.list_builds('vault') == []


def test_recursive_build_and_empty_snapshot(setup):
    storage, options = setup
    options['chunking'] = 'recursive'
    note = Note(title='Title', content='a b c d e f g h', source='a.md')
    report = build_index(storage, [note], **options, chunk_size=3, chunk_overlap=0)
    assert report.manifest.chunk_count > 1
    empty = build_index(storage, [], **options)
    assert empty.manifest.chunk_count == 0
    assert storage.load_snapshot(empty.manifest.index_version)[2].shape == (0, 2)


def test_changed_model_cannot_reuse_previous_document_vectors(setup):
    storage, options = setup
    notes = [Note(title='Title', content='body', source='a.md')]
    build_index(storage, notes, **options)
    options['spec'] = replace(options['spec'], model_revision='changed')
    assert build_index(storage, notes, **options).embedded_inputs == 1


def test_incremental_edit_rename_delete_and_noop(setup):
    storage, options = setup
    notes = [Note(title='A', content='first', source='a.md'), Note(title='B', content='second', source='b.md')]
    first = build_index(storage, notes, **options)
    untouched = build_index(storage, notes, **options)
    assert untouched.reused_index and untouched.manifest == first.manifest
    assert len(storage.list_builds('vault')) == 1
    notes[0] = replace(notes[0], content='edited')
    edited = build_index(storage, notes, **options)
    assert (edited.embedded_inputs, edited.modified_documents) == (1, 1)
    notes[0] = replace(notes[0], source='renamed.md')
    renamed = build_index(storage, notes, **options)
    assert (renamed.embedded_inputs, renamed.added_documents, renamed.deleted_documents) == (0, 1, 1)
    assert {r.chunk.source for r in storage.snapshot_records(renamed.manifest.index_version)} == {'renamed.md', 'b.md'}
    empty = build_index(storage, [], **options)
    assert empty.deleted_documents == 2 and empty.manifest.chunk_count == 0
    assert storage.load_snapshot(first.manifest.index_version)[1][0].chunk.content == 'first'


def test_forced_rebuild_uses_cache_and_recovers_interrupted_candidates(setup):
    storage, options = setup
    notes = [Note(title='A', content='first', source='a.md')]
    first = build_index(storage, notes, **options)
    pending = replace(first.manifest, index_version='interrupted', status='building')
    storage.create_build(pending, corpus_fingerprint='pending', backend={'kind': 'qdrant'})
    rebuilt = build_index(storage, notes, **options, force=True)
    assert not rebuilt.reused_index and rebuilt.embedded_inputs == 0
    assert rebuilt.manifest.index_version != first.manifest.index_version
    assert storage.get_manifest('interrupted').status == 'failed'


def test_vault_scope_cannot_silently_switch_directories(setup):
    storage, options = setup
    notes = [Note(title='A', content='first', source='a.md')]
    original = build_index(storage, notes, **options, source_scope='scope-a')
    with pytest.raises(ValueError, match='scope'):
        build_index(storage, [], **options, source_scope='scope-b')
    assert storage.active_manifest('vault') == original.manifest


@pytest.fixture
def qdrant_data():
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1', normalization='none')
    notes = [Note(title='Title', content='body', source=f'{n}.md') for n in range(3)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='vault') for n in notes]
    return spec, records


def create_qdrant_index(client, spec):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        return QdrantIndex(client, 'test', spec, vault_id='vault', create=True)


def test_incompatible_collection_is_rejected_without_overwriting(qdrant_data):
    spec, records = qdrant_data
    with closing(QdrantClient(':memory:')) as client:
        store = create_qdrant_index(client, spec)
        store.upsert(records[:1], [[1, 0]])
        for changed in (replace(spec, dimensions=3), replace(spec, model_revision='different')):
            with pytest.raises(ValueError, match='configuration'):
                QdrantIndex(client, 'test', changed, vault_id='vault')
        with pytest.raises(ValueError):
            create_qdrant_index(client, spec)
        assert store.count() == 1


def test_invalid_vectors_and_foreign_vault_fail_before_upsert(qdrant_data):
    spec, records = qdrant_data
    with closing(QdrantClient(':memory:')) as client:
        store = create_qdrant_index(client, spec)
        for items, vectors in ((records[:1], [[0, 0]]),
                               ([replace(records[0], vault_id='other')], [[1, 0]])):
            with pytest.raises(ValueError):
                store.upsert(items, vectors)
        assert store.count() == 0


def test_snapshot_verification_detects_payload_and_vector_corruption(qdrant_data):
    from arkb.knowledge.qdrant import point_id
    spec, records = qdrant_data
    with closing(QdrantClient(':memory:')) as client:
        store = create_qdrant_index(client, spec)
        vectors = [[1, 0], [.6, .8], [0, 1]]
        store.upsert(records, vectors)
        store.verify_snapshot(records, vectors)
        client.set_payload('test', payload={'source': 'wrong.md'}, points=[point_id(records[0].chunk_id)])
        with pytest.raises(ValueError, match='payload'):
            store.verify_snapshot(records, vectors)


def test_hnsw_readiness_timeout_does_not_claim_a_flat_index_is_built(qdrant_data):
    spec, records = qdrant_data
    with closing(QdrantClient(':memory:')) as client:
        store = create_qdrant_index(client, spec)
        store.upsert(records[:1], [[1, 0]])
        assert store.wait_ready(expected_count=1)['points'] == 1
        strict = QdrantIndex(client, 'test', spec, vault_id='vault',
                              config=QdrantConfig(index_timeout=.01, require_hnsw=True))
        with pytest.raises(ValueError, match='Timed out'):
            strict.wait_ready(expected_count=1)


def test_invalid_full_scan_threshold_fails_before_contacting_server(qdrant_data):
    spec, _ = qdrant_data
    for threshold in (0, 9, -1, True):
        with pytest.raises(ValueError, match='full_scan_threshold'):
            QdrantIndex(None, 'test', spec, vault_id='vault', create=True,
                             config=QdrantConfig(full_scan_threshold=threshold))


def test_retired_snapshot_rebuild_reuses_cache_and_switches_only_after_success(setup, monkeypatch):
    import json
    storage, options = setup
    notes = [Note('Title', 'Cached source text', 'a.md')]
    original = build_index(storage, notes, **options)
    version = original.manifest.index_version
    metadata = storage.build_metadata(version)['backend']
    metadata = {k: v for k, v in metadata.items() if k not in ('collection', 'url', 'index_stats')}
    metadata['kind'] = 'numpy'
    storage.connection.execute('UPDATE builds SET backend=? WHERE version=?', (json.dumps(metadata), version))
    storage.connection.commit()
    options['client'].embed.reset_mock()
    with monkeypatch.context() as patch:
        patch.setattr(QdrantIndex, 'upsert', Mock(side_effect=ConnectionError('Qdrant unavailable')))
        with pytest.raises(ConnectionError):
            build_index(storage, notes, **options)
    assert storage.active_manifest('vault').index_version == version
    assert storage.build_metadata(version)['backend']['kind'] == 'numpy'
    rebuilt = build_index(storage, notes, **options)
    assert rebuilt.embedded_inputs == 0 and rebuilt.cached_inputs == 1
    assert not rebuilt.reused_index
    assert storage.active_manifest('vault') == rebuilt.manifest
    assert storage.build_metadata(rebuilt.manifest.index_version)['backend']['kind'] == 'qdrant'
    assert storage.snapshot_records(version) == storage.snapshot_records(rebuilt.manifest.index_version)
    options['client'].embed.assert_not_called()


def test_old_metadata_with_omitted_defaults_reuses_the_published_snapshot(setup):
    import json
    storage, options = setup
    notes = [Note('Title', 'Existing source', 'a.md')]
    first = build_index(storage, notes, **options)
    version = first.manifest.index_version
    metadata = storage.build_metadata(version)['backend']
    for field in ('hnsw_m', 'ef_construct', 'indexing_threshold', 'full_scan_threshold', 'index_timeout', 'require_hnsw'):
        metadata.pop(field)
    storage.connection.execute('UPDATE builds SET backend=? WHERE version=?', (json.dumps(metadata), version))
    storage.connection.commit()
    options['client'].embed.reset_mock()
    reopened = build_index(storage, notes, **options)
    assert reopened.reused_index and reopened.manifest == first.manifest
    assert storage.build_metadata(version)['backend'] == metadata
    options['client'].embed.assert_not_called()


def test_qdrant_config_roundtrips_saved_settings_and_ignores_snapshot_fields():
    config = QdrantConfig(url='http://localhost:6333', hnsw_m=8, ef_construct=50,
                          indexing_threshold=1, full_scan_threshold=10, index_timeout=2.5, require_hnsw=True)
    assert QdrantConfig.from_metadata({**config.to_metadata(), 'collection': 'saved', 'index_stats': {}}) == config
    assert QdrantConfig.from_metadata({'kind': 'qdrant', 'url': QdrantConfig.url}) == QdrantConfig()
    with pytest.raises(ValueError, match='retired backend'):
        QdrantConfig.from_metadata({'kind': 'numpy'})


@pytest.mark.parametrize('changes', [
    {'hnsw_m': True}, {'hnsw_m': 1}, {'ef_construct': 0}, {'indexing_threshold': -1},
    {'full_scan_threshold': 9}, {'index_timeout': float('nan')}, {'index_timeout': float('inf')},
    {'index_timeout': 0}, {'index_timeout': True}, {'require_hnsw': 1},
    {'require_hnsw': True, 'indexing_threshold': 0}, {'url': ''},
])
def test_invalid_qdrant_config_fails_before_any_collection_operation(qdrant_data, changes):
    spec, _ = qdrant_data
    client = Mock(spec=QdrantClient)
    with pytest.raises(ValueError):
        QdrantIndex(client, 'test', spec, vault_id='vault', create=True, config=QdrantConfig(**changes))
    client.create_collection.assert_not_called()
    client.get_collection.assert_not_called()


def test_markdown_section_edit_reuses_unchanged_chunk_identity_and_embedding(setup):
    storage, options = setup
    options['chunking'] = 'recursive'
    note = Note('Title', '# First\nOriginal\n\n# Keep\nUnchanged\n', 'note.md')
    first = build_index(storage, [note], **options)
    old = storage.snapshot_records(first.manifest.index_version)
    edited = replace(note, content=note.content.replace('Original', 'A longer edited section'))
    second = build_index(storage, [edited], **options)
    new = storage.snapshot_records(second.manifest.index_version)

    assert second.embedded_inputs == 1 and second.cached_inputs == 1
    assert len(old) == len(new) == 2
    assert old[1].chunk_id == new[1].chunk_id
    assert old[1].chunk.start_char != new[1].chunk.start_char
    assert old[1].document_revision != new[1].document_revision
    assert old[1].chunk.heading_path == new[1].chunk.heading_path == ('Keep',)
    assert build_index(storage, [edited], **options).reused_index


@pytest.mark.parametrize('mode,old_algorithm', [('recursive', 'recursive-v1'), ('none', 'none-v1')])
def test_old_chunking_fingerprint_requires_a_new_snapshot_but_reuses_embeddings(setup, mode, old_algorithm):
    from dataclasses import asdict
    import json

    from arkb.knowledge.models import fingerprint_config
    from arkb.knowledge.embeddings import tokenizer_fingerprint

    storage, options = setup
    options['chunking'] = mode
    note = Note('Title', 'Body', 'note.md')
    first = build_index(storage, [note], **options)
    config = {'algorithm': old_algorithm, 'tokenizer': tokenizer_fingerprint(options['tokenizer'])}
    if mode == 'recursive':
        config.update(chunk_size=512, chunk_overlap=64)
    legacy = replace(first.manifest, chunking_fingerprint=fingerprint_config(config))
    storage.connection.execute('UPDATE builds SET manifest=? WHERE version=?',
                               (json.dumps(asdict(legacy)), legacy.index_version))
    second = build_index(storage, [note], **options)
    assert not second.reused_index
    assert second.manifest.index_version != first.manifest.index_version
    assert second.embedded_inputs == 0 and second.cached_inputs == 1
