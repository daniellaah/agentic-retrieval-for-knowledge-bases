"""Read-only Qdrant/SQLite snapshot adapter and application wiring.

Index creation, upserts, cache writes and publication stay in arkb.knowledge.indexing.
"""

from collections.abc import Mapping, Sequence
import math
from arkb.knowledge.qdrant import search_qdrant, _validate_qdrant_options
from typing import TYPE_CHECKING

from arkb.knowledge.embeddings import validate_vectors
from arkb.retrieval.contracts import SearchResponse, SearchResult, validate_options, validate_request
from arkb.retrieval.semantic import SemanticRetriever
from arkb.knowledge.models import ChunkRecord, EmbeddingSpec, VectorHit
from arkb.knowledge.qdrant import point_id
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.knowledge.qdrant import check_qdrant_collection
from arkb.knowledge.models import require_qdrant_backend

if TYPE_CHECKING:
    from ollama import Client
    from tokenizers import Tokenizer






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
    from arkb.retrieval.snapshot import chunk_result
    return chunk_result(record, method='semantic', index_id=index_version,
                        score=score, score_type='cosine_similarity')


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



class SnapshotSemanticRetriever:
    """Reusable application composition, including empty-snapshot validation.

    Caller-owned clients and storage remain open for the lifetime of queries.
    """
    def __init__(self, storage: SQLiteStorage, *, vault_id: str, spec: EmbeddingSpec,
                 tokenizer: 'Tokenizer', client: 'Client', exact: bool = False,
                 ef_search: int | None = None, index_version: str | None = None, qdrant_client=None):
        from arkb.knowledge.embeddings import OllamaQueryEmbedder
        self.index = QdrantSnapshotIndex(storage, qdrant_client, vault_id=vault_id,
                                        index_version=index_version, exact=exact, ef_search=ef_search)
        self.embedder = OllamaQueryEmbedder(client=client, spec=spec, tokenizer=tokenizer,
            tokenizer_identity=self.index.inputs['tokenizer'], max_input_tokens=self.index.inputs['max_tokens'],
            query_instruction=self.index.manifest.query_instruction)
        self.semantic = SemanticRetriever(self.embedder, self.index)

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        if not self.index.manifest.chunk_count:
            self.embedder.prepare(query)
            return SearchResponse(query=query, method='semantic', index_id=self.index.index_id)
        return self.semantic.search(query, top_k=top_k, filters=filters)


def search_index(
    storage: SQLiteStorage, question: str, *, vault_id: str, spec: EmbeddingSpec,
    tokenizer: 'Tokenizer', client: 'Client', top_k: int = 2,
    source: str | None = None, exact: bool = False, ef_search: int | None = None,
    index_version: str | None = None, qdrant_client=None,
) -> SearchResponse:
    """One-shot compatibility entry point for the current semantic adapters."""
    filters = validate_request(question, top_k, {'source': source} if source is not None else None)
    retriever = SnapshotSemanticRetriever(storage, vault_id=vault_id, spec=spec, tokenizer=tokenizer,
        client=client, exact=exact, ef_search=ef_search, index_version=index_version, qdrant_client=qdrant_client)
    return retriever.search(question, top_k=top_k, filters=filters)
