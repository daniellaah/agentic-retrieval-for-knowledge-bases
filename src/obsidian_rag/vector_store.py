"""Small vector-store contract and the exact NumPy reference implementation."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from obsidian_rag.embeddings import validate_vectors
from obsidian_rag.schema import ChunkRecord, EmbeddingSpec, VectorHit, validate_records
from obsidian_rag.retrieval import search_numpy, validate_search


class VectorStore(Protocol):
    """A store belongs to one embedding space and vault.

    Search scores are cosine similarities (larger is better); hits retain stable
    chunk IDs. Source filtering happens before top-k selection. exact=True asks
    for exhaustive search where available. Implementations never embed texts.
    """

    spec: EmbeddingSpec
    vault_id: str

    def upsert(self, records: Sequence[ChunkRecord], vectors) -> None: ...
    def search(self, query_vector, *, top_k: int = 2, source: str | None = None,
               exact: bool = False) -> list[VectorHit]: ...
    def delete(self, chunk_ids: Sequence[str]) -> None: ...
    def count(self) -> int: ...


class NumpyVectorStore:
    """In-memory exact search; durability is supplied by SQLite snapshots.

    Inputs are copied and fully checked before mutations. Existing IDs are
    replaced in place; equal scores retain insertion order, matching retrieve.
    """

    def __init__(self, spec: EmbeddingSpec, *, vault_id: str):
        if not isinstance(spec, EmbeddingSpec) or not isinstance(vault_id, str) or not vault_id.strip():
            raise ValueError("A valid embedding spec and vault_id are required.")
        self.spec = spec
        self.vault_id = vault_id
        self._entries = {}

    def upsert(self, records: Sequence[ChunkRecord], vectors) -> None:
        records = list(records)
        validate_records(records, vault_id=self.vault_id)
        matrix = validate_vectors(vectors, rows=len(records), dimensions=self.spec.dimensions,
                                  dtype=self.spec.dtype, normalization=self.spec.normalization)
        for record, vector in zip(records, matrix):
            self._entries[record.chunk_id] = (record, vector)

    def search(self, query_vector, *, top_k: int = 2, source: str | None = None,
               exact: bool = False) -> list[VectorHit]:
        records = [record for record, _ in self._entries.values()]
        vectors = np.stack([vector for _, vector in self._entries.values()]) if records else np.empty((0, self.spec.dimensions))
        return search_numpy(records, vectors, query_vector, spec=self.spec, vault_id=self.vault_id,
                            top_k=top_k, source=source, exact=exact)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._entries.pop(chunk_id, None)

    def count(self) -> int:
        return len(self._entries)
