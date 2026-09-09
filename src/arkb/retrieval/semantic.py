"""Semantic query orchestration, independent of providers and index lifecycle."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING
import math

from arkb.retrieval.models import SearchResponse, SearchResult, validate_request, validate_options
from arkb.knowledge.models import EmbeddingSpec, ChunkRecord, require_qdrant_backend


if TYPE_CHECKING:
    from arkb.knowledge.sqlite import SQLiteStorage


class Embedder(Protocol):
    """Embed a query in an explicitly identified document embedding space."""

    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_query(self, query: str) -> Sequence[float]: ...


class VectorIndex(Protocol):
    """Read one pinned index; adapt backend hits to ranked source evidence.

    Implementations validate vectors and returned identities, apply filters
    before top-k, and define score semantics. Resolve ties by stable identity.
    No database response objects cross this boundary. No writes are required.
    """

    @property
    def spec(self) -> EmbeddingSpec: ...

    @property
    def index_id(self) -> str: ...

    def search(self, vector: Sequence[float], *, top_k: int,
               filters: Mapping[str, str]) -> Sequence[SearchResult]: ...


@dataclass(frozen=True)
class SemanticRetriever:
    """Embed once, search once, and return ranked evidence without generation.

    The provider prepares model-specific query inputs. The index adapter handles
    backend conversion and ordering; scores are never normalized here. Exact
    versus approximate search is explicit adapter configuration. Repeatability
    depends on the pinned index, provider and backend (ANN is not guaranteed).
    """

    embedder: Embedder
    index: VectorIndex

    def __post_init__(self) -> None:
        if not self.embedder.spec.is_compatible_with(self.index.spec):
            raise ValueError('Query embedding model/configuration is incompatible with the stored index; rebuild it.')

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        vector = self.embedder.embed_query(query)
        results = tuple(self.index.search(vector, top_k=top_k, filters=filters))
        if len(results) > top_k:
            raise ValueError('Vector index returned more than top_k results.')
        for result in results:
            if (not isinstance(result, SearchResult) or result.method != 'semantic'
                    or result.score is None
                    or ('source' in filters and result.source != filters['source'])):
                raise ValueError('Vector index returned invalid semantic evidence or filter metadata.')
        return SearchResponse(query=query, method='semantic', results=results, index_id=self.index.index_id)


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
    from arkb.retrieval.models import chunk_result
    return chunk_result(record, method='semantic', index_id=index_version,
                        score=score, score_type='cosine_similarity')


class QdrantSnapshotIndex:
    """Capture a READY snapshot once; never follow a later active publication.

    The caller owns both clients. SQLite supplies original content and source
    coordinates; Qdrant supplies IDs and cosine scores. Exact/ANN and ef_search
    are explicit, fixed query settings for this adapter instance.
    """

    def __init__(self, storage: "SQLiteStorage", client, *, vault_id: str,
                 index_version: str | None = None, exact: bool = False,
                 ef_search: int | None = None):
        from arkb.knowledge.qdrant import _validate_qdrant_options, check_qdrant_collection
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
        from arkb.knowledge.qdrant import search_qdrant
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
