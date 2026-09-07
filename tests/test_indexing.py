from contextlib import closing
from dataclasses import replace
from unittest.mock import Mock
import warnings

from ollama import Client, EmbedResponse
import pytest
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from obsidian_rag.knowledge_base.chunking import whole_note_chunks
from obsidian_rag.knowledge_base.vector_index.indexing import build_index, QdrantIndex
from obsidian_rag.knowledge_base.loaders import Note
from obsidian_rag.knowledge_base.vector_index.manifest import EmbeddingSpec, ChunkRecord
from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage


@pytest.fixture
def setup(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[END]': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(single='$A [END]', special_tokens=[('[END]', 1)])
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2, document_template='title-body-v1')
    with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        yield storage, dict(spec=spec, vault_id='vault', client=client, tokenizer=tokenizer,
                            max_input_tokens=100, chunking='none')


def test_complete_build_preserves_duplicate_occurrences_and_reuses_cache(setup):
    storage, options = setup
    notes = [Note(title='Title', content='body', source=f'{n}.md') for n in range(2)]
    report = build_index(storage, notes, **options, index_version='first')
    assert report.manifest.status == 'ready'
    assert report.embedded_inputs == 1
    assert len(storage.load_snapshot('first')[1]) == 2
    again = build_index(storage, notes, **options, index_version='second')
    assert again.embedded_inputs == 0 and again.cached_inputs == 1
    assert options['client'].embed.call_count == 1
    assert storage.active_manifest('vault').index_version == 'second'


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
    storage.create_build(pending, corpus_fingerprint='pending')
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


def test_incomplete_snapshot_never_publishes_the_candidate(setup, monkeypatch):
    storage, options = setup
    notes = [Note(title='A', content='first', source='a.md')]
    original = build_index(storage, notes, **options)
    load_snapshot = storage.load_snapshot
    def incomplete(version):
        manifest, records, vectors = load_snapshot(version)
        return manifest, records[:-1], vectors[:-1]
    monkeypatch.setattr(storage, 'load_snapshot', incomplete)
    with pytest.raises(ValueError, match='snapshot count'):
        build_index(storage, notes, **options, index_version='incomplete')
    assert storage.active_manifest('vault') == original.manifest
    assert storage.get_manifest('incomplete').status == 'failed'


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
    from obsidian_rag.knowledge_base.vector_index.manifest import point_id
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
        with pytest.raises(ValueError, match='Timed out'):
            store.wait_ready(expected_count=1, timeout=.01, require_hnsw=True)


def test_invalid_full_scan_threshold_fails_before_contacting_server(qdrant_data):
    spec, _ = qdrant_data
    for threshold in (0, 9, -1, True):
        with pytest.raises(ValueError, match='full_scan_threshold'):
            QdrantIndex(None, 'test', spec, vault_id='vault', create=True,
                             full_scan_threshold=threshold)
