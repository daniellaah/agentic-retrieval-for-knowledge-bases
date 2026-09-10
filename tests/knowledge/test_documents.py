from pathlib import Path

import pytest

from arkb.knowledge.documents import DocumentAccess, load_notes


def test_load_notes_reads_title_content_and_source(tmp_path: Path) -> None:
    (tmp_path / "reading.md").write_text(
        "# Reading Notes\n\n"
        "Keep the author's meaning.\n\n"
        "## Source\n\n"
        "Sönke Ahrens.\n",
        encoding="utf-8",
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content, note.source) for note in notes] == [
        (
            "Reading Notes",
            "Keep the author's meaning.\n\n## Source\n\nSönke Ahrens.",
            "reading.md",
        )
    ]


def test_load_notes_returns_notes_in_filename_order(tmp_path: Path) -> None:
    for filename in ["zeta.md", "alpha.md", "Beta.md"]:
        (tmp_path / filename).write_text("# A Note\n\nAn idea.\n", encoding="utf-8")

    notes = load_notes(tmp_path)

    assert [note.source for note in notes] == ["Beta.md", "alpha.md", "zeta.md"]


def test_load_notes_uses_filename_when_no_level_one_heading_exists(
    tmp_path: Path,
) -> None:
    (tmp_path / "reading_notes.md").write_text(
        "## Reading\n\nA useful passage.\n", encoding="utf-8"
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content) for note in notes] == [
        ("reading_notes", "## Reading\n\nA useful passage.")
    ]


def test_load_notes_uses_first_level_one_heading_and_preserves_other_content(
    tmp_path: Path,
) -> None:
    (tmp_path / "ideas.md").write_text(
        "Introductory text.\n"
        "# Main Idea\n\n"
        "Develop one idea.\n\n"
        "# Another Heading\n\n"
        "Keep this section.\n",
        encoding="utf-8",
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content) for note in notes] == [
        (
            "Main Idea",
            "Introductory text.\n\nDevelop one idea.\n\n"
            "# Another Heading\n\nKeep this section.",
        )
    ]


def test_load_notes_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_notes(tmp_path / "missing")


def test_load_notes_reads_only_markdown_files_in_the_given_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "current.md").write_text("# Current\n\nAn idea.\n", encoding="utf-8")
    (tmp_path / "draft.txt").write_text("An unfinished draft.\n", encoding="utf-8")
    archive = tmp_path / "archive.md"
    archive.mkdir()
    (archive / "old.md").write_text("# Old\n\nAn earlier idea.\n", encoding="utf-8")

    notes = load_notes(tmp_path)

    assert [note.source for note in notes] == ["current.md"]


def test_load_notes_returns_an_empty_list_for_an_empty_directory(tmp_path: Path) -> None:
    assert load_notes(tmp_path) == []


def test_scan_rejects_changes_during_reading(tmp_path, monkeypatch):
    import arkb.knowledge.documents as loaders
    (tmp_path / 'a.md').write_text('# A\nbody')
    original = loaders.load_notes
    def changing(directory):
        notes = original(directory)
        (directory / 'b.md').write_text('# B\nnew')
        return notes
    monkeypatch.setattr(loaders, 'load_notes', changing)
    with pytest.raises(ValueError, match='changed during scanning'):
        loaders.scan_notes(tmp_path)


def test_access_identity_matches_index_records_and_only_reads_resolved_file(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import arkb.knowledge.documents as documents
    from arkb.knowledge.chunking import whole_note_chunks
    from arkb.knowledge.models import ChunkRecord

    (tmp_path / 'a.md').write_text('# A\n\nbody', encoding='utf-8')
    note, = load_notes(tmp_path)
    indexed = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id='v')
    (tmp_path / 'unrelated.md').write_bytes(b'\xff')
    load = Mock(wraps=documents._load_note)
    monkeypatch.setattr(documents, '_load_note', load)
    access = DocumentAccess(tmp_path, vault_id='v')
    read = access.read(indexed.document_id)
    assert (read.document_id, read.document_revision, read.source, read.title, read.content) == (
        indexed.document_id, indexed.document_revision, note.source, note.title, note.content)
    assert not hasattr(read, 'chunk_id')
    load.assert_called_once_with(tmp_path / 'a.md')
    load.reset_mock()
    assert access.read(source='a.md') == read
    load.assert_called_once_with(tmp_path / 'a.md')
    load.reset_mock()
    with pytest.raises(LookupError):
        access.read(indexed.document_id, source='unrelated.md')
    load.assert_not_called()
    with pytest.raises(LookupError):
        DocumentAccess(tmp_path, vault_id='other').read(indexed.document_id)


def test_access_sections_reuse_chunker_coordinates_and_ids(tmp_path):
    from arkb.knowledge.chunking import chunk_notes

    text = '# Title\n\nintro\n## One\nbody\n### Child\nchild body\n## One\nsecond body'
    (tmp_path / 'a.md').write_text(text, encoding='utf-8')
    access = DocumentAccess(tmp_path, vault_id='v')
    whole = next(access.records())
    chunks = chunk_notes(load_notes(tmp_path), count_tokens=len, chunk_size=100, chunk_overlap=0)
    for chunk in chunks:
        read = access.read(whole.document_id, section_id=chunk.section_id)
        assert read.content == chunk.content
        assert read.heading_path == chunk.heading_path
        assert read.section_id == chunk.section_id
        assert (read.start_char, read.end_char) == (chunk.start_char, chunk.end_char)
    assert access.read(whole.document_id).content == whole.chunk.content


def test_access_tracks_new_deleted_renamed_files_without_retaining_bodies(tmp_path):
    access = DocumentAccess(tmp_path, vault_id='v')
    assert list(access.records()) == []
    path = tmp_path / 'a.md'
    path.write_text('# A\nold', encoding='utf-8')
    original = next(access.records())
    path.write_text('# A\nnew', encoding='utf-8')
    current = access.read(original.document_id)
    assert current.content == 'new'
    assert current.document_revision != original.document_revision
    path.rename(tmp_path / 'renamed.md')
    with pytest.raises(LookupError):
        access.read(original.document_id)
    assert next(access.records()).document_id != original.document_id


def test_access_preserves_flat_scope_and_excludes_external_symlinks(tmp_path):
    root = tmp_path / 'notes'
    root.mkdir()
    (root / 'a.md').write_text('inside', encoding='utf-8')
    (root / 'ignore.txt').write_text('ignore', encoding='utf-8')
    (root / 'nested').mkdir()
    (root / 'nested' / 'nested.md').write_text('nested', encoding='utf-8')
    outside = tmp_path / 'private.md'
    outside.write_text('outside', encoding='utf-8')
    (root / 'link.md').symlink_to(outside)
    access = DocumentAccess(root, vault_id='v')
    assert [r.chunk.source for r in access.records()] == ['a.md']
    assert list(access.records(source='../private.md')) == []
    assert access.read(source='a.md').content == 'inside'
    for source in ('../private.md', str(outside), 'link.md', 'nested/nested.md', 'ignore.txt'):
        with pytest.raises(LookupError):
            access.read(source=source)


def test_access_propagates_filesystem_and_decoding_errors(tmp_path, monkeypatch):
    access = DocumentAccess(tmp_path, vault_id='v')
    (tmp_path / 'a.md').write_text('body', encoding='utf-8')
    record = next(access.records())
    (tmp_path / 'a.md').write_bytes(b'\xff')
    with pytest.raises(UnicodeDecodeError):
        access.read(record.document_id)
    with pytest.raises(FileNotFoundError):
        list(DocumentAccess(tmp_path / 'missing', vault_id='v').records())

    def disappeared(path):
        raise FileNotFoundError('removed during read')
    monkeypatch.setattr('arkb.knowledge.documents._load_note', disappeared)
    with pytest.raises(LookupError, match='no longer exists'):
        access.read(record.document_id)
