"""Query immutable snapshots with Qdrant and return source evidence."""

import math
from typing import TYPE_CHECKING


from arkb.embeddings import validate_vectors
from arkb.schema import (
    EmbeddingSpec,
    SearchResult,
    VectorHit,
    point_id,
)
from arkb.storage import check_qdrant_collection, require_qdrant_backend


if TYPE_CHECKING:
    from ollama import Client
    from tokenizers import Tokenizer
    from arkb.storage import SQLiteStorage


def validate_search(top_k: int, source: str | None, exact: bool) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("source must be a nonblank filename or None.")
    if type(exact) is not bool:
        raise ValueError("exact must be a boolean.")


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
    from arkb.embeddings import prepare_query, validate_input_tokens
    from arkb.embeddings import embed_texts
    from arkb.tokenization import tokenizer_fingerprint

    validate_search(top_k, source, exact)
    manifest = storage.get_manifest(index_version) if index_version is not None else storage.active_manifest(vault_id)
    if manifest is None:
        raise ValueError('No published index for this vault; build an index first.')
    if manifest.status != 'ready' or manifest.vault_id != vault_id:
        raise ValueError('Queries require a ready snapshot in the requested vault.')
    if not manifest.embedding_spec.is_compatible_with(spec):
        raise ValueError('Query embedding model/configuration is incompatible with the stored index; rebuild it.')
    metadata = storage.build_metadata(manifest.index_version)['backend']
    require_qdrant_backend(metadata)
    inputs = metadata['input']
    if tokenizer_fingerprint(tokenizer) != inputs['tokenizer']:
        raise ValueError('Query tokenizer differs from the indexed tokenizer; rebuild with matching settings.')
    query = prepare_query(question, instruction=manifest.query_instruction)
    validate_input_tokens(query, tokenizer=tokenizer, max_tokens=inputs['max_tokens'], source='query')
    if manifest.chunk_count == 0:
        return []
    if qdrant_client is None:
        raise ValueError('This index requires a Qdrant client.')
    check_qdrant_collection(qdrant_client, metadata['collection'], spec=spec, vault_id=vault_id)
    query_vector = embed_texts([query], client=client, model=spec.model,
                              dimensions=spec.dimensions, dtype=spec.dtype,
                              normalization=spec.normalization, context_length=inputs['max_tokens'])[0]
    hits = search_qdrant(qdrant_client, metadata['collection'], query_vector,
                         spec=spec, vault_id=vault_id, top_k=top_k, source=source, exact=exact)
    if len(hits) > top_k or len({hit.chunk_id for hit in hits}) != len(hits):
        raise ValueError('Vector store returned invalid or duplicate hits.')
    results = []
    for hit in hits:
        record = storage.get_record(manifest.index_version, hit.chunk_id)
        if (record is None or not math.isfinite(hit.score) or not -1 <= hit.score <= 1
                or (source is not None and record.chunk.source != source)):
            raise ValueError('Vector hit does not match the snapshot, filter, or cosine score contract.')
        results.append(SearchResult(record.chunk, hit.score, record, manifest.index_version))
    return results
