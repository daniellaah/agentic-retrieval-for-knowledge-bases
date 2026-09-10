"""Shared immutable records and deterministic, versioned SHA-256 identities.

This module describes data; it does not call models, store vectors, or publish
indexes. Sources are canonical vault-relative POSIX paths. A rename changes a
document's identity; identical embedding inputs may still reuse cached vectors.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
import math
import hashlib
import json
import re
from typing import Literal


@dataclass(frozen=True)
class Note:
    """A Markdown note with a title, body, and source filename."""

    title: str
    content: str
    source: str

    @property
    def note_id(self) -> str:
        """Stable identity within a source collection; a path rename changes it."""
        return _digest("note-id", {"path": self.source})

    @property
    def path(self) -> str:
        return self.source


@dataclass(frozen=True)
class Chunk:
    """A verbatim slice of Note.content; character range ends are exclusive.

    Section ranges cover a heading and its direct body, up to the next heading.
    heading_path includes ancestor headings. A root section has an empty path.
    occurrence distinguishes identical slices within one section. Optional
    section fields also allow reading records created before Markdown chunking.
    """

    content: str
    title: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    heading_path: tuple[str, ...] = ()
    section_id: str | None = None
    section_start_char: int | None = None
    section_end_char: int | None = None
    occurrence: int = 0

    def __post_init__(self) -> None:
        # JSON stores tuples as arrays; restore immutable metadata on decoding.
        if isinstance(self.heading_path, list):
            object.__setattr__(self, "heading_path", tuple(self.heading_path))

    @property
    def path(self) -> str:
        return self.source

    @property
    def note_id(self) -> str:
        return _digest("note-id", {"path": self.source})

    @property
    def parent_id(self) -> str:
        return self.section_id or self.note_id

    @property
    def chunk_id(self) -> str:
        """Collection-local content identity, independent of document revision.

        Moving an unchanged slice within its section keeps its ID. Inserting an
        identical slice before it can change its occurrence number. Legacy
        chunks without section metadata use their original ordinal instead.
        """
        return _digest("chunk-content", {
            "parent_id": self.parent_id, "content": self.content,
            "occurrence": self.occurrence if self.section_id else self.chunk_index,
        })


# Version 2 establishes ARKB identities; version 1 snapshots must be rebuilt.
SCHEMA_VERSION = 2
type ConfigValue = (
    None | bool | int | float | str | list[ConfigValue] | dict[str, ConfigValue]
)


def fingerprint_config(config: dict[str, ConfigValue]) -> str:
    """Hash JSON configuration independent of dictionary insertion order.

    Preserve string contents and list order; reject non-JSON values, non-string
    keys, and non-finite numbers. Include every setting affecting the operation.
    For chunking this includes algorithm revision, budgets, tokenizer identity
    and revision, normalization, and special-token handling. Do not include
    machine paths, timestamps, or batch sizes in a chunking configuration.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object.")
    _validate_config(config)
    return _digest("config", config)


@dataclass(frozen=True)
class EmbeddingSpec:
    """Configuration required to reuse document vectors without conversion.

    model_revision must identify the actual model artifact (for example its
    digest), not just a mutable tag. provider also identifies the inference
    implementation; change it when encoding behavior changes. document_template
    identifies the input format and must change when that format changes.

    Query instructions, chunking, search parameters, and model host addresses
    are deliberately separate. dtype describes the stored vector representation;
    float64 preserves the current embedding function's representation. These
    declarations do not validate an actual model response or normalize vectors.
    """

    model: str
    model_revision: str
    dimensions: int
    document_template: str
    provider: str = "ollama"
    normalization: Literal["none", "l2"] = "l2"
    dtype: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        for name in ("model", "model_revision", "document_template", "provider"):
            _require_text(getattr(self, name), name)
        _require_integer(self.dimensions, "dimensions", minimum=1)
        if self.normalization not in ("none", "l2"):
            raise ValueError("normalization must be 'none' or 'l2'.")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'.")

    @property
    def fingerprint(self) -> str:
        return _digest("embedding-spec", asdict(self))

    def is_compatible_with(self, other: "EmbeddingSpec") -> bool:
        """Conservatively require identical settings for document-vector reuse."""
        return self == other

    def embedding_key(self, text: str) -> str:
        """Key one complete, nonblank document input; never strip or normalize it.

        Pass the final model input, including its title/template, not just the
        chunk body. This is a document cache key, not a query cache key.
        """
        _require_text(text, "embedding text")
        return _digest("document-embedding", {"spec": self.fingerprint, "text": text})


@dataclass(frozen=True)
class ChunkRecord:
    """One source occurrence of a Chunk in a loaded document revision.

    document_revision hashes the loaded Note's title and complete body, not raw
    file bytes. Offsets remain Python character positions in Note.content, with
    an exclusive end. Use from_note when creating a record from source data;
    direct construction validates local fields but cannot verify a source body
    that is not supplied (for example when loading a stored record).

    IDs do not depend on the embedding model, build version, or machine path.
    Section-aware chunk identities are scoped to vault_id without including the
    document revision or positions. Records without section metadata use the
    revision-based identity formula within the current schema namespace.
    """

    vault_id: str
    document_revision: str
    chunk: Chunk

    def __post_init__(self) -> None:
        _require_text(self.vault_id, "vault_id")
        _require_digest(self.document_revision, "document_revision")
        if not isinstance(self.chunk, Chunk):
            raise ValueError("chunk must be a Chunk.")
        source = self.chunk.source
        _require_text(source, "source")
        if "\\" in source or any(part in ("", ".", "..") for part in source.split("/")):
            raise ValueError("source must be a canonical vault-relative POSIX path.")
        if re.match(r"^[A-Za-z]:", source):
            raise ValueError("source must be a canonical vault-relative POSIX path.")
        if not isinstance(self.chunk.title, str) or not isinstance(self.chunk.content, str):
            raise ValueError("chunk title and content must be strings.")
        for name in ("chunk_index", "start_char", "end_char"):
            _require_integer(getattr(self.chunk, name), name, minimum=0)
        if self.chunk.end_char - self.chunk.start_char != len(self.chunk.content):
            raise ValueError("chunk span must match its content length.")
        if (not isinstance(self.chunk.heading_path, tuple)
                or any(not isinstance(heading, str) for heading in self.chunk.heading_path)):
            raise ValueError("heading_path must contain strings.")
        _require_integer(self.chunk.occurrence, "occurrence", minimum=0)
        if self.chunk.section_id is not None:
            _require_digest(self.chunk.section_id, "section_id")
            for name in ("section_start_char", "section_end_char"):
                _require_integer(getattr(self.chunk, name), name, minimum=0)
            if not (self.chunk.section_start_char <= self.chunk.start_char
                    <= self.chunk.end_char <= self.chunk.section_end_char):
                raise ValueError("chunk must lie within its section span.")
        elif (self.chunk.heading_path or self.chunk.section_start_char is not None
              or self.chunk.section_end_char is not None or self.chunk.occurrence):
            raise ValueError("section metadata requires a section_id.")

    @classmethod
    def from_note(cls, chunk: Chunk, *, note: Note, vault_id: str) -> "ChunkRecord":
        """Create a record and verify its title, source, and exact source slice."""
        if not isinstance(note, Note):
            raise ValueError("note must be a Note.")
        if not isinstance(note.title, str) or not isinstance(note.content, str):
            raise ValueError("note title and content must be strings.")
        revision = _digest("document-revision", {"title": note.title, "content": note.content})
        record = cls(vault_id=vault_id, document_revision=revision, chunk=chunk)
        if (
            chunk.source != note.source
            or chunk.title != note.title
            or chunk.end_char > len(note.content)
            or (chunk.section_end_char is not None and chunk.section_end_char > len(note.content))
            or chunk.content != note.content[chunk.start_char:chunk.end_char]
        ):
            raise ValueError("chunk must match its source note and character span.")
        return record

    @property
    def document_id(self) -> str:
        return _digest("document-id", {"vault_id": self.vault_id, "source": self.chunk.source})

    @property
    def chunk_id(self) -> str:
        if self.chunk.section_id is not None:
            return _digest("scoped-chunk", {
                "document_id": self.document_id, "chunk_id": self.chunk.chunk_id,
            })
        # Hash precisely the original six fields for pre-Markdown snapshots.
        return _digest("chunk-id", {
            "document_id": self.document_id,
            "document_revision": self.document_revision,
            "chunk": {name: getattr(self.chunk, name) for name in (
                "content", "title", "source", "chunk_index", "start_char", "end_char",
            )},
        })


@dataclass(frozen=True)
class IndexManifest:
    """Describe one build, its configuration, and intended snapshot counts.

    index_version is an opaque build ID supplied by the caller. Configuration
    equality is not corpus equality: changes to source documents require a new
    snapshot even if configuration_fingerprint is unchanged. Counts describe
    the intended complete snapshot, not partial progress during a failed build.

    Query instructions affect the configuration fingerprint but not document
    embedding cache keys. status records a declaration; validating stored data,
    enforcing transitions, and atomically publishing READY are indexing duties.
    """

    index_version: str
    vault_id: str
    embedding_spec: EmbeddingSpec
    chunking_fingerprint: str
    document_count: int
    chunk_count: int
    query_instruction: str = ""
    status: Literal["building", "ready", "failed"] = "building"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.index_version, "index_version")
        _require_text(self.vault_id, "vault_id")
        if not isinstance(self.embedding_spec, EmbeddingSpec):
            raise ValueError("embedding_spec must be an EmbeddingSpec.")
        _require_digest(self.chunking_fingerprint, "chunking_fingerprint")
        _require_integer(self.document_count, "document_count", minimum=0)
        _require_integer(self.chunk_count, "chunk_count", minimum=0)
        # Both current chunkers emit at least one chunk per loaded note.
        if self.chunk_count < self.document_count or (
            self.document_count == 0 and self.chunk_count != 0
        ):
            raise ValueError("snapshot counts must include at least one chunk per note.")
        if not isinstance(self.query_instruction, str):
            raise ValueError("query_instruction must be a string.")
        if self.status not in ("building", "ready", "failed"):
            raise ValueError("status must be 'building', 'ready', or 'failed'.")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version!r}. "
                             "Rebuild with arkb index --db <new-database-path>.")

    @property
    def configuration_fingerprint(self) -> str:
        return _digest("index-configuration", {
            "embedding": self.embedding_spec.fingerprint,
            "chunking": self.chunking_fingerprint,
            "query_instruction": self.query_instruction,
        })


def _digest(kind: str, data: dict) -> str:
    # Persisted ARKB identities change only with an explicit schema version bump.
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(
        f"arkb/{kind}/v{SCHEMA_VERSION}\n{payload}".encode("utf-8")
    ).hexdigest()


def _validate_config(value: ConfigValue) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings.")
            _validate_config(item)
    elif isinstance(value, list):
        for item in value:
            _validate_config(item)
    elif value is not None and type(value) not in (bool, int, float, str):
        raise ValueError("Configuration must contain only JSON values.")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string.")


def _require_integer(value: int, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest.")


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float


def validate_records(records: Sequence[ChunkRecord], *, vault_id: str) -> None:
    if any(not isinstance(record, ChunkRecord) or record.vault_id != vault_id for record in records):
        raise ValueError("All vector records must belong to this vault.")
    if len({record.chunk_id for record in records}) != len(records):
        raise ValueError("Duplicate chunk IDs in one vector batch.")

def require_qdrant_backend(metadata: dict) -> None:
    """Retired snapshots remain readable as metadata, but cannot serve queries."""
    if metadata.get('kind') != 'qdrant':
        raise ValueError('This index uses a retired backend; run arkb index to rebuild it in Qdrant. '
                         'Compatible cached embeddings will be reused.')

@dataclass(frozen=True)
class QdrantConfig:
    """Endpoint and build settings; defaults also decode existing snapshots.

    The caller supplies a client connected to this endpoint. Credentials belong
    to that client, never to the persisted configuration.
    """

    url: str = 'http://127.0.0.1:6333'
    hnsw_m: int = 16
    ef_construct: int = 100
    indexing_threshold: int = 10000
    full_scan_threshold: int = 10000
    index_timeout: float = 30
    require_hnsw: bool = False

    def __post_init__(self):
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError('Qdrant url must be nonblank.')
        for name, minimum in (('hnsw_m', 2), ('ef_construct', 1),
                              ('indexing_threshold', 0), ('full_scan_threshold', 10)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if (type(self.index_timeout) not in (int, float)
                or not math.isfinite(self.index_timeout) or self.index_timeout <= 0):
            raise ValueError('Index readiness timeout must be positive and finite.')
        if type(self.require_hnsw) is not bool:
            raise ValueError('require_hnsw must be boolean.')
        if self.require_hnsw and self.indexing_threshold == 0:
            raise ValueError('require_hnsw needs a positive indexing_threshold.')

    def to_metadata(self) -> dict:
        return {'kind': 'qdrant', **asdict(self)}

    @classmethod
    def from_metadata(cls, metadata: dict) -> 'QdrantConfig':
        require_qdrant_backend(metadata)
        return cls(**{field.name: metadata[field.name] for field in fields(cls) if field.name in metadata})
