"""Immutable indexing records and deterministic, versioned SHA-256 identities.

This module describes data; it does not call models, store vectors, or publish
indexes. Sources are canonical vault-relative POSIX paths. A rename changes a
document's identity; identical embedding inputs may still reuse cached vectors.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import re
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from obsidian_rag.knowledge_base.chunking import Chunk
from obsidian_rag.knowledge_base.loaders import Note


from obsidian_rag.knowledge_base.models import ChunkRecord
from obsidian_rag.knowledge_base.identity import (SCHEMA_VERSION, fingerprint_config, digest as _digest,
    require_text as _require_text, require_integer as _require_integer, require_digest as _require_digest)
type ConfigValue = (
    None | bool | int | float | str | list[ConfigValue] | dict[str, ConfigValue]
)


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
            raise ValueError(f"Unsupported schema_version: {self.schema_version!r}.")

    @property
    def configuration_fingerprint(self) -> str:
        return _digest("index-configuration", {
            "embedding": self.embedding_spec.fingerprint,
            "chunking": self.chunking_fingerprint,
            "query_instruction": self.query_instruction,
        })


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float


def validate_records(records: Sequence[ChunkRecord], *, vault_id: str) -> None:
    if any(not isinstance(record, ChunkRecord) or record.vault_id != vault_id for record in records):
        raise ValueError("All vector records must belong to this vault.")
    if len({record.chunk_id for record in records}) != len(records):
        raise ValueError("Duplicate chunk IDs in one vector batch.")


def point_id(chunk_id: str) -> str:
    if not isinstance(chunk_id, str) or re.fullmatch('[0-9a-f]{64}', chunk_id) is None:
        raise ValueError('Expected a SHA-256 chunk ID.')
    return str(uuid5(NAMESPACE_URL, 'obsidian-rag/chunk/' + chunk_id))


def qdrant_identity(spec: EmbeddingSpec, vault_id: str) -> dict:
    """Metadata identifying one owned Qdrant collection's embedding space."""
    return {'owner': 'obsidian-rag', 'schema': 1, 'embedding_spec': spec.fingerprint, 'vault_id': vault_id}
