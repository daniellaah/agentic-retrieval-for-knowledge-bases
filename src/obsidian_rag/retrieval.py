"""Rank document chunks by cosine similarity."""

import os
from urllib.parse import urlsplit
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from obsidian_rag.chunking import Chunk

if TYPE_CHECKING:
    from ollama import Client
    from tokenizers import Tokenizer
    from obsidian_rag.schema import EmbeddingSpec
    from obsidian_rag.storage import SQLiteStorage
    from obsidian_rag.vector_store import VectorStore


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its cosine similarity to a query."""

    chunk: Chunk
    score: float


def retrieve(
    chunks: Sequence[Chunk],
    chunk_vectors: NDArray[np.float64],
    query_vector: NDArray[np.float64],
    *,
    top_k: int = 2,
) -> list[SearchResult]:
    """Return up to top_k chunks ordered by descending cosine similarity.

    Each vector row must correspond to the chunk at the same index. Chunks from
    the same source are ranked independently; no document-level deduplication
    takes place. The original objects and their source spans are preserved.
    The query must be one vector with the same dimension as the chunk vectors.
    Equal scores preserve input order, and input vectors are left unchanged.
    An empty collection with a zero-row matrix returns no results.

    Raise ValueError for an invalid result count, incompatible vector shapes,
    non-finite values, or vector norms that are zero or non-finite.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    matrix = np.asarray(chunk_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("Expected a matrix with one vector per chunk.")
    if not chunks:
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

    return [
        SearchResult(chunk=chunks[index], score=float(scores[index]))
        for index in indices
    ]


def search_index(
    storage: "SQLiteStorage", question: str, *, vault_id: str, spec: "EmbeddingSpec",
    tokenizer: "Tokenizer", client: "Client",
    top_k: int = 2, source: str | None = None, exact: bool = False,
    index_version: str | None = None, vector_store: "VectorStore | None" = None, qdrant_client=None,
) -> list[SearchResult]:
    """Search one captured READY snapshot, embedding only the query.

    The caller supplies the actual runtime model specification, not an arbitrary
    model tag. Match it and the tokenizer against the stored configuration before
    making a model request. A version may be supplied to pin a query while another
    writer publishes. Document text and positions always come from that snapshot.
    """
    from obsidian_rag.embeddings import prepare_query, validate_input_tokens
    from obsidian_rag.embeddings import embed_texts
    from obsidian_rag.tokenization import tokenizer_fingerprint
    from obsidian_rag.vector_store import NumpyVectorStore, validate_search

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
    if vector_store is None:
        if metadata['kind'] == 'qdrant':
            from obsidian_rag.indexing import _qdrant_projection
            if qdrant_client is None:
                raise ValueError('This index requires a Qdrant client.')
            vector_store = _qdrant_projection(qdrant_client, metadata, manifest, create=False)
        elif metadata['kind'] == 'numpy':
            _, records, matrix = storage.load_snapshot(manifest.index_version)
            vector_store = NumpyVectorStore(spec, vault_id=vault_id)
            vector_store.upsert(records, matrix)
        else:
            raise ValueError('Unsupported index backend.')
    if vector_store.spec != spec or vector_store.vault_id != vault_id:
        raise ValueError('Vector store does not match the query embedding spec and vault.')
    query_vector = embed_texts([query], client=client, model=spec.model,
                              dimensions=spec.dimensions, dtype=spec.dtype,
                              normalization=spec.normalization, context_length=inputs['max_tokens'])[0]
    hits = vector_store.search(query_vector, top_k=top_k, source=source, exact=exact)
    if len(hits) > top_k or len({hit.chunk_id for hit in hits}) != len(hits):
        raise ValueError('Vector store returned invalid or duplicate hits.')
    results = []
    for hit in hits:
        record = storage.get_record(manifest.index_version, hit.chunk_id)
        if (record is None or not np.isfinite(hit.score) or not -1 <= hit.score <= 1
                or (source is not None and record.chunk.source != source)):
            raise ValueError('Vector hit does not match the snapshot, filter, or cosine score contract.')
        results.append(SearchResult(record.chunk, hit.score))
    return results


def connect_qdrant(url: str, timeout: float):
    from qdrant_client import QdrantClient
    parts = urlsplit(url)
    if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError('Use an HTTP(S) Qdrant URL without embedded credentials; set QDRANT_API_KEY if needed.')
    return QdrantClient(url=url, api_key=os.environ.get('QDRANT_API_KEY'), timeout=timeout, trust_env=False)
