"""Read-only Qdrant/SQLite snapshot adapter and application wiring.

Index creation, upserts, cache writes and publication stay in arkb.indexing.
"""

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING

from arkb.embeddings import validate_vectors
from arkb.retrieval.contracts import SearchResponse, SearchResult, validate_options, validate_request
from arkb.retrieval.semantic import SemanticRetriever
from arkb.schema import ChunkRecord, EmbeddingSpec, VectorHit, point_id
from arkb.storage import SQLiteStorage, check_qdrant_collection, require_qdrant_backend

if TYPE_CHECKING:
    from ollama import Client
    from tokenizers import Tokenizer


def _validate_qdrant_options(exact: bool, ef_search: int | None) -> None:
    if type(exact) is not bool:
        raise ValueError("exact must be a boolean.")
    if ef_search is not None and (type(ef_search) is not int or ef_search <= 0):
        raise ValueError('ef_search must be positive.')


def search_qdrant(
    client, collection: str, query_vector, *, spec: EmbeddingSpec, vault_id: str,
    top_k: int = 2, source: str | None = None, exact: bool = False, ef_search: int | None = None,
) -> list[VectorHit]:
    """Query an opened snapshot; check_qdrant_collection validates it once before use.

    No collection creation or writes occur here. Point IDs, filter payload and
    cosine scores are checked on every result, with the existing float32 tolerance.
    """
    from qdrant_client import models
    validate_options(top_k, {'source': source} if source is not None else None)
    _validate_qdrant_options(exact, ef_search)
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


def snapshot_result(record: ChunkRecord, score: float, index_version: str) -> SearchResult:
    """Translate stored evidence without losing revision or Markdown provenance.

    cosine_similarity is higher-is-better, in [-1, 1]. search_qdrant preserves
    Qdrant scores except for clipping float32 drift within 1e-5 of the bounds;
    this is the existing numeric tolerance, not relevance normalization.
    """
    if not isinstance(index_version, str) or not index_version.strip():
        raise ValueError('index_version must be nonblank.')
    if not isinstance(record, ChunkRecord):
        raise ValueError('Snapshot evidence requires a ChunkRecord.')
    if type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError('Snapshot evidence requires a finite cosine score.')
    chunk = record.chunk
    return SearchResult(
        source_id=record.document_id, source=chunk.source, content=chunk.content,
        method='semantic', chunk_id=record.chunk_id,
        start_char=chunk.start_char, end_char=chunk.end_char,
        score=score, score_type='cosine_similarity',
        metadata={'title': chunk.title, 'vault_id': record.vault_id,
                  'document_revision': record.document_revision, 'index_version': index_version,
                  'chunk_index': chunk.chunk_index, 'heading_path': list(chunk.heading_path),
                  'section_id': chunk.section_id, 'section_start_char': chunk.section_start_char,
                  'section_end_char': chunk.section_end_char, 'occurrence': chunk.occurrence},
    )


class QdrantSnapshotIndex:
    """Capture a READY snapshot once; never follow a later active publication.

    The caller owns both clients. SQLite supplies original content and source
    coordinates; Qdrant supplies IDs and cosine scores. Exact/ANN and ef_search
    are explicit, fixed query settings for this adapter instance.
    """

    def __init__(self, storage: SQLiteStorage, client, *, vault_id: str,
                 index_version: str | None = None, exact: bool = False,
                 ef_search: int | None = None):
        _validate_qdrant_options(exact, ef_search)
        manifest = storage.get_manifest(index_version) if index_version is not None else storage.active_manifest(vault_id)
        if manifest is None:
            raise ValueError('No published index for this vault; build an index first.')
        if manifest.status != 'ready' or manifest.vault_id != vault_id:
            raise ValueError('Queries require a ready snapshot in the requested vault.')
        metadata = storage.build_metadata(manifest.index_version)['backend']
        require_qdrant_backend(metadata)
        self.manifest = manifest
        self.spec = manifest.embedding_spec
        self.index_id = manifest.index_version
        self.inputs = dict(metadata['input'])
        self.storage, self.client = storage, client
        self.collection = metadata['collection']
        self.exact, self.ef_search = exact, ef_search
        if manifest.chunk_count:
            if client is None:
                raise ValueError('This index requires a Qdrant client.')
            check_qdrant_collection(client, self.collection, spec=self.spec, vault_id=vault_id)

    def search(self, vector: Sequence[float], *, top_k: int,
               filters: Mapping[str, str]) -> tuple[SearchResult, ...]:
        filters = validate_options(top_k, filters)
        if not self.manifest.chunk_count:
            return ()
        hits = search_qdrant(self.client, self.collection, vector, spec=self.spec,
                             vault_id=self.manifest.vault_id, top_k=top_k,
                             source=filters.get('source'), exact=self.exact, ef_search=self.ef_search)
        if len(hits) > top_k or len({hit.chunk_id for hit in hits}) != len(hits):
            raise ValueError('Vector store returned invalid or duplicate hits.')
        results = []
        for hit in hits:
            record = self.storage.get_record(self.index_id, hit.chunk_id)
            if (record is None or not math.isfinite(hit.score) or not -1 <= hit.score <= 1
                    or ('source' in filters and record.chunk.source != filters['source'])):
                raise ValueError('Vector hit does not match the snapshot, filter, or cosine score contract.')
            results.append(snapshot_result(record, hit.score, self.index_id))
        return tuple(results)


def search_index(
    storage: SQLiteStorage, question: str, *, vault_id: str, spec: EmbeddingSpec,
    tokenizer: 'Tokenizer', client: 'Client', top_k: int = 2,
    source: str | None = None, exact: bool = False, ef_search: int | None = None,
    index_version: str | None = None, qdrant_client=None,
) -> SearchResponse:
    """Compose the current adapters for a one-shot application query.

    Use SemanticRetriever directly to reuse an opened snapshot or swap either
    capability. Runtime embedding identity and tokenizer settings must match
    the captured snapshot. Empty snapshots validate the query without embedding.
    """
    from arkb.retrieval.ollama import OllamaQueryEmbedder

    filters = validate_request(question, top_k, {'source': source} if source is not None else None)
    index = QdrantSnapshotIndex(storage, qdrant_client, vault_id=vault_id,
                                index_version=index_version, exact=exact, ef_search=ef_search)
    embedder = OllamaQueryEmbedder(client=client, spec=spec, tokenizer=tokenizer,
                                   tokenizer_identity=index.inputs['tokenizer'],
                                   max_input_tokens=index.inputs['max_tokens'],
                                   query_instruction=index.manifest.query_instruction)
    semantic = SemanticRetriever(embedder, index)
    if not index.manifest.chunk_count:
        embedder.prepare(question)
        return SearchResponse(query=question, method='semantic', index_id=index.index_id)
    return semantic.search(question, top_k=top_k, filters=filters)
