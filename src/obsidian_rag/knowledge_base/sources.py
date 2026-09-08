"""Immutable note snapshots and validated reads without embedding or vector indexes."""

from dataclasses import asdict, dataclass
from collections.abc import Sequence

from .models import Note, Chunk, ChunkRecord
from .identity import digest, fingerprint_config, require_digest, require_text, require_source_path


@dataclass(frozen=True)
class SourceRef:
    vault_id: str
    document_id: str
    path: str
    title: str
    document_revision: str | None
    snapshot_id: str | None

    def __post_init__(self):
        require_text(self.vault_id, 'vault_id')
        require_source_path(self.path)
        require_digest(self.document_id, 'document_id')
        if self.document_id != digest('document-id', {'vault_id': self.vault_id, 'source': self.path}):
            raise ValueError('Source document identity differs from its vault and path.')
        if not isinstance(self.title, str):
            raise ValueError('Source title must be a string.')
        if (self.document_revision is None) != (self.snapshot_id is None):
            raise ValueError('Source revision and snapshot identity must be supplied together.')
        if self.document_revision is not None:
            require_digest(self.document_revision, 'document_revision')
            require_text(self.snapshot_id, 'snapshot_id')

    @classmethod
    def from_note(cls, note: Note, *, vault_id: str, snapshot_id: str):
        return cls(vault_id, digest('document-id', {'vault_id': vault_id, 'source': note.source}),
                   note.source, note.title,
                   digest('document-revision', {'title': note.title, 'content': note.content}), snapshot_id)


@dataclass(frozen=True)
class SourceExcerpt:
    source: SourceRef
    start_char: int
    end_char: int
    content: str

    def __post_init__(self):
        if (not isinstance(self.source, SourceRef) or not isinstance(self.content, str)
                or type(self.start_char) is not int or type(self.end_char) is not int
                or not 0 <= self.start_char <= self.end_char
                or self.end_char - self.start_char != len(self.content)):
            raise ValueError('Excerpt must match its source character span.')


@dataclass(frozen=True)
class KnowledgeSnapshot:
    vault_id: str
    snapshot_id: str
    notes: tuple[Note, ...]
    corpus_fingerprint: str | None = None

    def __post_init__(self):
        if not isinstance(self.vault_id, str) or not self.vault_id.strip():
            raise ValueError('vault_id must be nonblank.')
        if not isinstance(self.notes, tuple) or any(not isinstance(n, Note) for n in self.notes):
            raise ValueError('Snapshot requires immutable notes.')
        if len({n.source for n in self.notes}) != len(self.notes):
            raise ValueError('Snapshot requires unique source paths.')
        for note in self.notes:
            ChunkRecord.from_note(Chunk(note.content, note.title, note.source, 0, 0, len(note.content)),
                                  note=note, vault_id=self.vault_id)
        require_text(self.snapshot_id, 'snapshot_id')
        actual = fingerprint_config({'notes': [asdict(n) for n in self.notes]})
        if self.corpus_fingerprint is None:
            if self.snapshot_id != 'memory:' + actual:
                raise ValueError('Snapshot identity differs from its content.')
            object.__setattr__(self, 'corpus_fingerprint', actual)
        elif self.corpus_fingerprint != actual:
            raise ValueError('Snapshot corpus fingerprint differs from its content.')

    @classmethod
    def from_notes(cls, notes: Sequence[Note], *, vault_id: str):
        notes = tuple(notes)
        version = 'memory:' + fingerprint_config({'notes': [asdict(note) for note in notes]})
        return cls(vault_id, version, notes)

    @classmethod
    def from_records(cls, records: Sequence[ChunkRecord], *, vault_id: str, snapshot_id: str,
                     corpus_fingerprint: str):
        """Restore old snapshots from complete source spans, never vector values.

        Gaps, conflicting overlap, changed titles/revisions and incomplete source
        bodies fail validation. Named published snapshots retain their saved ID;
        the caller supplies the saved corpus fingerprint to verify completeness.
        """
        groups = {}
        for record in records:
            if not isinstance(record, ChunkRecord) or record.vault_id != vault_id:
                raise ValueError('Snapshot records must belong to the requested vault.')
            groups.setdefault(record.chunk.source, []).append(record)
        notes = []
        for path, group in groups.items():
            first = group[0]
            content = ''
            for record in sorted(group, key=lambda r: (r.chunk.start_char, r.chunk.end_char)):
                chunk = record.chunk
                if record.document_revision != first.document_revision or chunk.title != first.chunk.title:
                    raise ValueError('Snapshot mixes document revisions or titles.')
                if chunk.start_char > len(content):
                    raise ValueError('Snapshot has a gap in source coverage.')
                overlap = min(chunk.end_char, len(content)) - chunk.start_char
                if content[chunk.start_char:chunk.start_char + overlap] != chunk.content[:overlap]:
                    raise ValueError('Snapshot contains conflicting source overlap.')
                content += chunk.content[overlap:]
            note = Note(first.chunk.title, content, path)
            if digest('document-revision', {'title': note.title, 'content': note.content}) != first.document_revision:
                raise ValueError('Snapshot source body does not match its document revision.')
            notes.append(note)
        return cls(vault_id, snapshot_id, tuple(notes), corpus_fingerprint)

    def note_refs(self) -> tuple[SourceRef, ...]:
        return tuple(SourceRef.from_note(n, vault_id=self.vault_id, snapshot_id=self.snapshot_id)
                     for n in self.notes)

    def read_note(self, reference: SourceRef) -> Note:
        if not isinstance(reference, SourceRef):
            raise ValueError('Expected a source reference.')
        for note in self.notes:
            if note.source == reference.path and reference == SourceRef.from_note(
                    note, vault_id=self.vault_id, snapshot_id=self.snapshot_id):
                return note
        raise ValueError('Source does not belong to this snapshot.')

    def read_span(self, reference: SourceRef, start: int, end: int) -> SourceExcerpt:
        note = self.read_note(reference)
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(note.content):
            raise ValueError('Requested span is outside the source note.')
        return SourceExcerpt(reference, start, end, note.content[start:end])

    def bind_chunks(self, chunks: Sequence[Chunk]) -> tuple[ChunkRecord, ...]:
        notes = {note.source: note for note in self.notes}
        records = []
        for chunk in chunks:
            note = notes.get(chunk.source)
            if note is None:
                raise ValueError('Chunk source is absent from this snapshot.')
            try:
                records.append(ChunkRecord.from_note(chunk, note=note, vault_id=self.vault_id))
            except ValueError as error:
                raise ValueError('Chunk does not match its snapshot source.') from error
        return tuple(records)
