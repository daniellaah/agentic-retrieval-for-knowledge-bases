from obsidian_rag.knowledge_base.loaders import load_notes
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot


def test_loaded_notes_can_be_read_from_an_immutable_snapshot_without_embeddings(tmp_path):
    path = tmp_path / 'memory.md'
    path.write_text('# Memory\n\n短期与长期记忆。', encoding='utf-8')
    snapshot = KnowledgeSnapshot.from_notes(load_notes(tmp_path), vault_id='personal')
    reference = snapshot.note_refs()[0]
    path.write_text('# Changed\nNew content', encoding='utf-8')
    assert reference.path == 'memory.md'
    assert snapshot.read_note(reference).content == '短期与长期记忆。'
    excerpt = snapshot.read_span(reference, 3, 7)
    assert excerpt.content == '长期记忆'
    assert (excerpt.start_char, excerpt.end_char) == (3, 7)


def test_snapshot_rejects_ambiguous_duplicate_source_paths():
    import pytest
    from obsidian_rag.knowledge_base.models import Note
    notes = [Note('A', 'old', 'a.md'), Note('A', 'new', 'a.md')]
    with pytest.raises(ValueError, match='unique'):
        KnowledgeSnapshot.from_notes(notes, vault_id='personal')


def test_chunks_can_be_bound_to_a_snapshot_without_a_vector_index():
    import pytest
    from dataclasses import replace
    from obsidian_rag.knowledge_base.models import Note
    from obsidian_rag.knowledge_base.chunking import chunk_notes
    snapshot = KnowledgeSnapshot.from_notes([Note('A', 'alpha beta gamma', 'a.md')], vault_id='personal')
    chunks = chunk_notes(snapshot.notes, count_tokens=lambda text: len(text.split()), chunk_size=2, chunk_overlap=0)
    records = snapshot.bind_chunks(chunks)
    assert [r.chunk.content for r in records] == ['alpha beta ', 'gamma']
    assert records[0].document_id == snapshot.note_refs()[0].document_id
    with pytest.raises(ValueError, match='source'):
        snapshot.bind_chunks([replace(chunks[0], content='invalid body')])


def test_source_references_reject_partial_or_inconsistent_identity():
    from dataclasses import replace
    import pytest
    from obsidian_rag.knowledge_base.models import Note
    snapshot = KnowledgeSnapshot.from_notes([Note('A', 'body', 'a.md')], vault_id='v')
    source = snapshot.note_refs()[0]
    for fields in ({'document_id': '0' * 64}, {'path': 'b.md'}, {'path': '../a.md'},
                   {'snapshot_id': None}, {'document_revision': None}, {'vault_id': ''}, {'title': 1}):
        with pytest.raises(ValueError):
            replace(source, **fields)
    legacy = replace(source, snapshot_id=None, document_revision=None)
    assert legacy.document_revision is None
    with pytest.raises(ValueError):
        replace(snapshot.read_span(source, 0, 4), source='a.md')


def test_snapshot_restoration_checks_complete_coverage_overlap_and_content_hash():
    from dataclasses import replace
    import pytest
    from obsidian_rag.knowledge_base.models import Note
    from obsidian_rag.knowledge_base.chunking import chunk_notes
    snapshot = KnowledgeSnapshot.from_notes([Note('A', 'alpha beta gamma delta', 'a.md')], vault_id='v')
    chunks = chunk_notes(snapshot.notes, count_tokens=len, chunk_size=12, chunk_overlap=6)
    records = snapshot.bind_chunks(chunks)
    from obsidian_rag.knowledge_base.identity import fingerprint_config
    expected = fingerprint_config({'notes': [{'title': n.title, 'content': n.content, 'source': n.source} for n in snapshot.notes]})
    restored = KnowledgeSnapshot.from_records(records, vault_id='v', snapshot_id='published', corpus_fingerprint=expected)
    assert restored.notes == snapshot.notes
    for broken in (records[1:], records[:-1],
                   [replace(records[0], chunk=replace(records[0].chunk, content='X' * len(records[0].chunk.content))), *records[1:]]):
        with pytest.raises(ValueError):
            KnowledgeSnapshot.from_records(broken, vault_id='v', snapshot_id='published', corpus_fingerprint=expected)
