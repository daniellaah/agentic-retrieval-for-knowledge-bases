"""Query immutable snapshots with NumPy or Qdrant and return source evidence."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

from obsidian_rag.knowledge_base.chunking import Chunk
from obsidian_rag.knowledge_base.embeddings import validate_vectors
from obsidian_rag.knowledge_base.vector_index.manifest import (
    ChunkRecord,
    EmbeddingSpec,
    VectorHit,
    validate_records,
    point_id,
    qdrant_identity,
)


if TYPE_CHECKING:
    from ollama import Client
    from tokenizers import Tokenizer
    from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage


@dataclass(frozen=True)
class SearchResult:
    """A cosine hit, optionally tied to a verified immutable snapshot.

    Legacy in-memory hits have neither record nor index_version. They must not
    be treated as evidence of a known document revision when merging content.
    """

    chunk: Chunk
    score: float
    record: ChunkRecord | None = None
    index_version: str | None = None

    def __post_init__(self) -> None:
        if (self.record is None) != (self.index_version is None):
            raise ValueError('record and index_version must be supplied together.')
        if self.record is not None:
            if not isinstance(self.record, ChunkRecord) or self.record.chunk != self.chunk:
                raise ValueError('Search result chunk must match its source record.')
            if not isinstance(self.index_version, str) or not self.index_version.strip():
                raise ValueError('index_version must be nonblank.')


def _rank_vectors(chunk_vectors, query_vector, *, rows: int, top_k: int):
    """Return stable cosine ranks without coupling the math to records or backends."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    matrix = np.asarray(chunk_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != rows:
        raise ValueError("Expected a matrix with one vector per chunk.")
    if not rows:
        return []

    query = np.asarray(query_vector, dtype=np.float64)
    if query.ndim != 1 or query.size == 0 or matrix.shape[1] != query.size:
        raise ValueError("Expected a nonempty query vector matching the chunk dimension.")
    if not np.isfinite(matrix).all() or not np.isfinite(query).all():
        raise ValueError("Vectors must contain only finite values.")

    with np.errstate(over="ignore", under="ignore"):
        chunk_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
    if not np.isfinite(chunk_norms).all() or not np.isfinite(query_norm):
        raise ValueError("Vector norms must be finite.")
    if np.any(chunk_norms == 0) or query_norm == 0:
        raise ValueError("Vectors must have nonzero norms.")

    normalized_chunks = matrix / chunk_norms[:, None]
    normalized_query = query / query_norm
    scores = np.clip(normalized_chunks @ normalized_query, -1.0, 1.0)
    indices = np.argsort(-scores, kind="stable")[:top_k]

    return [(int(index), float(scores[index])) for index in indices]


def retrieve(
    chunks: Sequence[Chunk], chunk_vectors: NDArray[np.float64], query_vector: NDArray[np.float64],
    *, top_k: int = 2,
) -> list[SearchResult]:
    """Rank chunks by cosine similarity, preserving objects, spans and input-order ties.

    This existing entry point shares the NumPy ranking implementation with
    search_numpy. Empty collections and validation retain their original behavior.
    """
    return [SearchResult(chunks[index], score) for index, score in
            _rank_vectors(chunk_vectors, query_vector, rows=len(chunks), top_k=top_k)]


def search_numpy(
    records: Sequence[ChunkRecord], vectors, query_vector, *, spec: EmbeddingSpec,
    vault_id: str, top_k: int = 2, source: str | None = None, exact: bool = False,
) -> list[VectorHit]:
    """Search snapshot vectors exactly; filter before top-k and preserve input-order ties."""
    validate_search(top_k, source, exact)
    records = list(records)
    validate_records(records, vault_id=vault_id)
    matrix = validate_vectors(vectors, rows=len(records), dimensions=spec.dimensions,
                              dtype=spec.dtype, normalization=spec.normalization)
    query = validate_vectors([query_vector], rows=1, dimensions=spec.dimensions,
                             dtype=spec.dtype, normalization=spec.normalization)[0]
    selected = [i for i, record in enumerate(records) if source is None or record.chunk.source == source]
    ranked = _rank_vectors(matrix[selected], query, rows=len(selected), top_k=top_k)
    return [VectorHit(records[selected[index]].chunk_id, score) for index, score in ranked]


def search_index(
    storage: "SQLiteStorage", question: str, *, vault_id: str, spec: "EmbeddingSpec",
    tokenizer: "Tokenizer", client: "Client",
    top_k: int = 2, source: str | None = None, exact: bool = False,
    index_version: str | None = None, qdrant_client=None,
) -> list[SearchResult]:
    """Search one captured READY snapshot, embedding only the query.

    The caller supplies the actual runtime model specification, not an arbitrary
    model tag. Match it and the tokenizer against the stored configuration before
    making a model request. A version may be supplied to pin a query while another
    writer publishes. Document text and positions always come from that snapshot.
    """
    from obsidian_rag.knowledge_base.embeddings import prepare_query, validate_input_tokens
    from obsidian_rag.knowledge_base.embeddings import embed_texts
    from obsidian_rag.knowledge_base.tokenization import tokenizer_fingerprint

    validate_search(top_k, source, exact)
    manifest = storage.get_manifest(index_version) if index_version is not None else storage.active_manifest(vault_id)
    if manifest is None:
        raise ValueError('No published index for this vault; build an index first.')
    if manifest.status != 'ready' or manifest.vault_id != vault_id:
        raise ValueError('Queries require a ready snapshot in the requested vault.')
    if not manifest.embedding_spec.is_compatible_with(spec):
        raise ValueError('Query embedding model/configuration is incompatible with the stored index; rebuild it.')
    metadata = storage.build_metadata(manifest.index_version)['backend']
    inputs = metadata['input']
    if tokenizer_fingerprint(tokenizer) != inputs['tokenizer']:
        raise ValueError('Query tokenizer differs from the indexed tokenizer; rebuild with matching settings.')
    query = prepare_query(question, instruction=manifest.query_instruction)
    validate_input_tokens(query, tokenizer=tokenizer, max_tokens=inputs['max_tokens'], source='query')
    if manifest.chunk_count == 0:
        return []
    records, matrix = None, None
    if metadata['kind'] == 'qdrant':
        if qdrant_client is None:
            raise ValueError('This index requires a Qdrant client.')
        check_qdrant_collection(qdrant_client, metadata['collection'], spec=spec, vault_id=vault_id)
    elif metadata['kind'] == 'numpy':
        _, records, matrix = storage.load_snapshot(manifest.index_version)
    else:
        raise ValueError('Unsupported index backend.')
    query_vector = embed_texts([query], client=client, model=spec.model,
                              dimensions=spec.dimensions, dtype=spec.dtype,
                              normalization=spec.normalization, context_length=inputs['max_tokens'])[0]
    if metadata['kind'] == 'qdrant':
        hits = search_qdrant(qdrant_client, metadata['collection'], query_vector,
                             spec=spec, vault_id=vault_id, top_k=top_k, source=source, exact=exact)
    else:
        hits = search_numpy(records, matrix, query_vector, spec=spec, vault_id=vault_id,
                            top_k=top_k, source=source, exact=exact)
    if len(hits) > top_k or len({hit.chunk_id for hit in hits}) != len(hits):
        raise ValueError('Vector store returned invalid or duplicate hits.')
    results = []
    for hit in hits:
        record = storage.get_record(manifest.index_version, hit.chunk_id)
        if (record is None or not np.isfinite(hit.score) or not -1 <= hit.score <= 1
                or (source is not None and record.chunk.source != source)):
            raise ValueError('Vector hit does not match the snapshot, filter, or cosine score contract.')
        results.append(SearchResult(record.chunk, hit.score, record, manifest.index_version))
    return results


from obsidian_rag.knowledge_base.vector_index.qdrant import (
    validate_search, check_qdrant_collection, search_qdrant, connect_qdrant)
