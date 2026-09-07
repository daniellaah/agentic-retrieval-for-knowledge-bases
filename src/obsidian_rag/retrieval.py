"""Query immutable snapshots with NumPy or Qdrant and return source evidence."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

from obsidian_rag.chunking import Chunk
from obsidian_rag.embeddings import validate_vectors
from obsidian_rag.schema import (
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
    from obsidian_rag.storage import SQLiteStorage


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its cosine similarity to a query."""

    chunk: Chunk
    score: float


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


def validate_search(top_k: int, source: str | None, exact: bool) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("source must be a nonblank filename or None.")
    if type(exact) is not bool:
        raise ValueError("exact must be a boolean.")


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


def check_qdrant_collection(client, collection: str, *, spec: EmbeddingSpec, vault_id: str):
    """Validate identity when opening a snapshot, before embedding or querying it."""
    from qdrant_client import models
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError('collection must be nonblank.')
    if not isinstance(vault_id, str) or not vault_id.strip():
        raise ValueError('vault_id must be nonblank.')
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    if (not isinstance(vectors, models.VectorParams) or vectors.size != spec.dimensions
            or vectors.distance != models.Distance.COSINE or info.config.metadata != qdrant_identity(spec, vault_id)):
        raise ValueError('Qdrant collection configuration does not match the embedding spec and vault.')
    return info


def search_qdrant(
    client, collection: str, query_vector, *, spec: EmbeddingSpec, vault_id: str,
    top_k: int = 2, source: str | None = None, exact: bool = False, ef_search: int | None = None,
) -> list[VectorHit]:
    """Query an opened snapshot; check_qdrant_collection validates it once before use.

    No collection creation or writes occur here. Point IDs, filter payload and
    cosine scores are checked on every result, with the existing float32 tolerance.
    """
    from qdrant_client import models
    validate_search(top_k, source, exact)
    if ef_search is not None and (type(ef_search) is not int or ef_search <= 0):
        raise ValueError('ef_search must be positive.')
    query = validate_vectors([query_vector], rows=1, dimensions=spec.dimensions,
                             dtype=spec.dtype, normalization=spec.normalization)[0]
    values = {'vault_id': vault_id, 'embedding_spec': spec.fingerprint}
    if source is not None:
        values['source'] = source
    query_filter = models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
                                       for k, v in values.items()])
    points = client.query_points(
        collection_name=collection, query=query.tolist(), query_filter=query_filter,
        limit=top_k, with_payload=True, with_vectors=False,
        search_params=models.SearchParams(exact=exact, hnsw_ef=ef_search),
    ).points
    hits = []
    for point in points:
        payload = point.payload or {}
        chunk_id = payload.get('chunk_id')
        if (point_id(chunk_id) != str(point.id) or any(payload.get(k) != v for k, v in values.items())
                or not math.isfinite(point.score) or not -1.00001 <= point.score <= 1.00001):
            raise ValueError('Qdrant returned invalid identity, filter metadata, or cosine score.')
        hits.append(VectorHit(chunk_id, max(-1.0, min(1.0, float(point.score)))))
    return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))


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
    from obsidian_rag.embeddings import prepare_query, validate_input_tokens
    from obsidian_rag.embeddings import embed_texts
    from obsidian_rag.tokenization import tokenizer_fingerprint

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
        results.append(SearchResult(record.chunk, hit.score))
    return results


def connect_qdrant(url: str, timeout: float):
    from qdrant_client import QdrantClient
    parts = urlsplit(url)
    if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError('Use an HTTP(S) Qdrant URL without embedded credentials; set QDRANT_API_KEY if needed.')
    return QdrantClient(url=url, api_key=os.environ.get('QDRANT_API_KEY'), timeout=timeout, trust_env=False)
