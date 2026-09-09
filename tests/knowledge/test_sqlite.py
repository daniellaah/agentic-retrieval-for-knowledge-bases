from dataclasses import replace
import sqlite3

import numpy as np
import pytest

from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.embeddings import prepare_document
from arkb.knowledge.models import Note
from arkb.knowledge.models import ChunkRecord, EmbeddingSpec, IndexManifest, fingerprint_config
from arkb.knowledge.sqlite import SQLiteStorage


@pytest.fixture
def sample():
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1')
    note = Note(title='标题', content='正文 e\u0301\n', source='notes/a.md')
    record = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id='vault')
    manifest = IndexManifest(index_version='v1', vault_id='vault', embedding_spec=spec,
                             chunking_fingerprint=fingerprint_config({}), document_count=1, chunk_count=1)
    return spec, record, manifest


def populate(store, sample):
    spec, record, manifest = sample
    store.put_embeddings(spec, [prepare_document(record.chunk)], [[0.6, 0.8]])
    store.create_build(manifest, corpus_fingerprint='corpus', backend={'kind': 'qdrant'})
    store.add_chunk(manifest.index_version, record, ordinal=0)


def test_snapshot_survives_reopening_and_restores_exact_sources(tmp_path, sample):
    path = tmp_path / 'index.sqlite'
    with SQLiteStorage(path) as store:
        populate(store, sample)
        assert store.active_manifest('vault') is None
        store.publish('v1')
    with SQLiteStorage(path, read_only=True) as store:
        manifest, records, vectors = store.load_snapshot(store.active_manifest('vault').index_version)
        assert manifest.status == 'ready'
        assert records == [sample[1]]
        np.testing.assert_array_equal(vectors, [[0.6, 0.8]])


def test_failed_publication_rolls_back_and_preserves_active_version(tmp_path, sample):
    with SQLiteStorage(tmp_path / 'db') as store:
        populate(store, sample)
        store.publish('v1')
        store.create_build(replace(sample[2], index_version='v2'), corpus_fingerprint='other', backend={'kind': 'qdrant'})
        with pytest.raises(ValueError, match='counts'):
            store.publish('v2')
        assert store.active_manifest('vault').index_version == 'v1'
        assert store.get_manifest('v2').status == 'building'
        store.mark_failed('v2', 'incomplete')
        assert store.build_metadata('v2')['error'] == 'incomplete'
        with pytest.raises(ValueError):
            store.add_chunk('v1', sample[1], ordinal=1)


def test_cache_batch_conflict_rolls_back_all_new_entries(tmp_path, sample):
    spec, _, _ = sample
    with SQLiteStorage(tmp_path / 'db') as store:
        store.put_embeddings(spec, ['old'], [[1, 0]])
        with pytest.raises(ValueError, match='Conflicting'):
            store.put_embeddings(spec, ['new', 'old'], [[0, 1], [0, 1]])
        assert store.get_embedding(spec, 'new') is None
        np.testing.assert_array_equal(store.get_embedding(spec, 'old'), [1, 0])
        assert store.get_embedding(replace(spec, model_revision='changed'), 'old') is None


def test_corrupt_cache_and_record_are_rejected(tmp_path, sample):
    with SQLiteStorage(tmp_path / 'db') as store:
        populate(store, sample)
        store.connection.execute("UPDATE embeddings SET vector=x'00'")
        with pytest.raises(ValueError, match='checksum'):
            store.publish('v1')
    with SQLiteStorage(tmp_path / 'other') as store:
        populate(store, sample)
        store.connection.execute("UPDATE snapshot_chunks SET chunk_id='wrong'")
        with pytest.raises(ValueError, match='identity'):
            store.load_snapshot('v1')


def test_empty_snapshot_and_readonly_restrictions(tmp_path, sample):
    path = tmp_path / 'db'
    manifest = replace(sample[2], document_count=0, chunk_count=0)
    with SQLiteStorage(path) as store:
        store.create_build(manifest, corpus_fingerprint='empty', backend={'kind': 'qdrant'})
        store.publish('v1')
        assert store.load_snapshot('v1')[2].shape == (0, 2)
    with SQLiteStorage(path, read_only=True) as store:
        with pytest.raises(sqlite3.OperationalError):
            store.create_build(replace(manifest, index_version='v2'), corpus_fingerprint='empty', backend={'kind': 'qdrant'})


def test_storage_rejects_unknown_schema_without_reinitializing(tmp_path):
    path = tmp_path / 'db'
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=99')
    with pytest.raises(ValueError, match='version'):
        SQLiteStorage(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 99


def test_cache_validates_shape_and_normalization_before_writing(tmp_path, sample):
    with SQLiteStorage(tmp_path / 'db') as store:
        for vectors in ([[1, 0, 0]], [[3, 4]], [[float('nan'), 0]]):
            with pytest.raises(ValueError):
                store.put_embeddings(sample[0], ['text'], vectors)
        assert store.connection.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0] == 0


def test_writer_lock_is_shared_across_connections_and_symlink_paths(tmp_path):
    path = tmp_path / 'db'
    with SQLiteStorage(path) as first:
        alias = tmp_path / 'alias'
        alias.symlink_to(path)
        with SQLiteStorage(alias) as second:
            with first.writer_lock():
                with pytest.raises(ValueError, match='Another index build'):
                    with second.writer_lock():
                        pytest.fail('second writer acquired the lock')
            with second.writer_lock():
                pass


def test_process_exit_releases_build_lock(tmp_path):
    import subprocess
    import sys
    script = '''
import sys, time
from pathlib import Path
from arkb.knowledge.sqlite import SQLiteStorage
with SQLiteStorage(Path(sys.argv[1])) as storage:
    with storage.writer_lock():
        print('locked', flush=True)
        time.sleep(30)
'''
    path = tmp_path / 'db'
    process = subprocess.Popen([sys.executable, '-B', '-c', script, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'locked'
        with SQLiteStorage(path) as storage:
            with pytest.raises(ValueError, match='Another index build'):
                with storage.writer_lock():
                    pass
        process.terminate()
        process.wait(timeout=5)
        with SQLiteStorage(path) as storage:
            with storage.writer_lock():
                pass
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        process.stdout.close()


@pytest.mark.parametrize('legacy', [False, True])
def test_section_metadata_and_legacy_records_survive_storage_roundtrip(tmp_path, sample, legacy):
    from dataclasses import asdict
    import json

    from arkb.knowledge.chunking import chunk_notes
    from arkb.knowledge.models import Chunk

    spec, _, manifest = sample
    note = Note('Title', '## Section\nBody', 'folder/note.md')
    chunk = (Chunk(note.content, note.title, note.source, 0, 0, len(note.content)) if legacy
             else chunk_notes([note], count_tokens=len)[0])
    record = ChunkRecord.from_note(chunk, note=note, vault_id='vault')
    path = tmp_path / 'db'
    with SQLiteStorage(path) as store:
        populate(store, (spec, record, manifest))
        if legacy:
            # Match the old on-disk shape, not just the default-valued new dataclass.
            payload = asdict(record)
            payload['chunk'] = {name: payload['chunk'][name] for name in (
                'content', 'title', 'source', 'chunk_index', 'start_char', 'end_char',
            )}
            store.connection.execute('UPDATE snapshot_chunks SET record=?', (json.dumps(payload),))
        store.publish('v1')
    with SQLiteStorage(path, read_only=True) as store:
        restored = store.get_record('v1', record.chunk_id)
        assert restored == record
        assert store.snapshot_records('v1') == [record]
        assert restored.chunk.note_id == note.note_id
        assert restored.chunk.heading_path == (() if legacy else ('Section',))
        assert restored.chunk.chunk_id == chunk.chunk_id
