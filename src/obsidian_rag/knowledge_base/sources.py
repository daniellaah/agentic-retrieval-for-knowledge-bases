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
        if self.snapshot_id != 'memory:' + fingerprint_config({'notes': [asdict(n) for n in self.notes]}):
            raise ValueError('Snapshot identity differs from its content.')

    @classmethod
    def from_notes(cls, notes: Sequence[Note], *, vault_id: str):
        notes = tuple(notes)
        version = 'memory:' + fingerprint_config({'notes': [asdict(note) for note in notes]})
        return cls(vault_id, version, notes)

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
