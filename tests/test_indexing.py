from dataclasses import replace
from unittest.mock import Mock

import pytest
from ollama import Client, EmbedResponse
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from obsidian_rag.schema import EmbeddingSpec
from obsidian_rag.indexing import build_index
from obsidian_rag.loaders import Note
from obsidian_rag.storage import SQLiteStorage


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



def test_projection_failure_never_publishes_the_candidate(setup):
    from obsidian_rag.vector_store import NumpyVectorStore
    storage, options = setup
    notes = [Note(title='A', content='first', source='a.md')]
    original = build_index(storage, notes, **options)
    class BrokenProjection(NumpyVectorStore):
        def upsert(self, records, vectors):
            raise ValueError('projection unavailable')
    broken = BrokenProjection(options['spec'], vault_id='vault')
    with pytest.raises(ValueError, match='projection unavailable'):
        build_index(storage, notes, **options, vector_store=broken, index_version='failed-projection')
    assert storage.active_manifest('vault') == original.manifest
    assert storage.get_manifest('failed-projection').status == 'failed'
