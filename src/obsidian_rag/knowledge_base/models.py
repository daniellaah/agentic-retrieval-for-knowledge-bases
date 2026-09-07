"""Immutable knowledge records, independent of indexing methods."""

from dataclasses import asdict, dataclass
import re
from .identity import digest, require_text, require_integer, require_digest

@dataclass(frozen=True)
class Note:
    """A Markdown note with a title, body, and source filename."""

    title: str
    content: str
    source: str

@dataclass(frozen=True)
class Chunk:
    """A verbatim slice of Note.content; end_char is exclusive."""

    content: str
    title: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int

@dataclass(frozen=True)
class ChunkRecord:
    """One source occurrence of a Chunk in a loaded document revision.

    document_revision hashes the loaded Note's title and complete body, not raw
    file bytes. Offsets remain Python character positions in Note.content, with
    an exclusive end. Use from_note when creating a record from source data;
    direct construction validates local fields but cannot verify a source body
    that is not supplied (for example when loading a stored record).

    IDs do not depend on the embedding model, build version, or machine path.
    Actual chunk fields determine identity; a chunking configuration change that
    produces identical chunks need not change their IDs.
    """

    vault_id: str
    document_revision: str
    chunk: Chunk

    def __post_init__(self) -> None:
        require_text(self.vault_id, "vault_id")
        require_digest(self.document_revision, "document_revision")
        if not isinstance(self.chunk, Chunk):
            raise ValueError("chunk must be a Chunk.")
        source = self.chunk.source
        require_text(source, "source")
        if "\\" in source or any(part in ("", ".", "..") for part in source.split("/")):
            raise ValueError("source must be a canonical vault-relative POSIX path.")
        if re.match(r"^[A-Za-z]:", source):
            raise ValueError("source must be a canonical vault-relative POSIX path.")
        if not isinstance(self.chunk.title, str) or not isinstance(self.chunk.content, str):
            raise ValueError("chunk title and content must be strings.")
        for name in ("chunk_index", "start_char", "end_char"):
            require_integer(getattr(self.chunk, name), name, minimum=0)
        if self.chunk.end_char - self.chunk.start_char != len(self.chunk.content):
            raise ValueError("chunk span must match its content length.")

    @classmethod
    def from_note(cls, chunk: Chunk, *, note: Note, vault_id: str) -> "ChunkRecord":
        """Create a record and verify its title, source, and exact source slice."""
        if not isinstance(note, Note):
            raise ValueError("note must be a Note.")
        if not isinstance(note.title, str) or not isinstance(note.content, str):
            raise ValueError("note title and content must be strings.")
        revision = digest("document-revision", {"title": note.title, "content": note.content})
        record = cls(vault_id=vault_id, document_revision=revision, chunk=chunk)
        if (
            chunk.source != note.source
            or chunk.title != note.title
            or chunk.end_char > len(note.content)
            or chunk.content != note.content[chunk.start_char:chunk.end_char]
        ):
            raise ValueError("chunk must match its source note and character span.")
        return record

    @property
    def document_id(self) -> str:
        return digest("document-id", {"vault_id": self.vault_id, "source": self.chunk.source})

    @property
    def chunk_id(self) -> str:
        return digest("chunk-id", {
            "document_id": self.document_id,
            "document_revision": self.document_revision,
            "chunk": asdict(self.chunk),
        })
