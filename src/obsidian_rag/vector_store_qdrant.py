"""Qdrant adapter using externally computed vectors and source metadata."""

from collections.abc import Sequence
import math
import re
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from obsidian_rag.embeddings import validate_vectors
from obsidian_rag.index_schema import ChunkRecord, EmbeddingSpec
from obsidian_rag.vector_store import VectorHit, validate_records, validate_search


def point_id(chunk_id: str) -> str:
    if not isinstance(chunk_id, str) or re.fullmatch('[0-9a-f]{64}', chunk_id) is None:
        raise ValueError('Expected a SHA-256 chunk ID.')
    return str(uuid5(NAMESPACE_URL, 'obsidian-rag/chunk/' + chunk_id))


class QdrantVectorStore:
    """One collection per immutable index candidate.

    create=True creates a new collection, never recreates/deletes an existing one.
    Configuration metadata guards against same-dimension incompatible models.
    SQLite remains the source of chunk text and original float64/float32 vectors;
    Qdrant uses float32 cosine vectors, so exact scores can differ by rounding.
    Local Qdrant clients are useful for API tests; ANN requires Qdrant Server.
    """

    def __init__(self, client: QdrantClient, collection: str, spec: EmbeddingSpec, *,
                 vault_id: str, create: bool = False, hnsw_m: int = 16,
                 ef_construct: int = 100, indexing_threshold: int = 10000):
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError('collection must be nonblank.')
        if not isinstance(vault_id, str) or not vault_id.strip():
            raise ValueError('vault_id must be nonblank.')
        for name, value, minimum in (('hnsw_m', hnsw_m, 2), ('ef_construct', ef_construct, 1),
                                     ('indexing_threshold', indexing_threshold, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        self.client, self.collection, self.spec, self.vault_id = client, collection, spec, vault_id
        self.identity = {'owner': 'obsidian-rag', 'schema': 1, 'embedding_spec': spec.fingerprint, 'vault_id': vault_id}
        if create:
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=spec.dimensions, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=ef_construct),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
                metadata=self.identity,
            )
            for field in ('vault_id', 'embedding_spec', 'source'):
                client.create_payload_index(collection_name=collection, field_name=field,
                                            field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
        self.check_configuration()

    def check_configuration(self):
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if (not isinstance(vectors, models.VectorParams) or vectors.size != self.spec.dimensions
                or vectors.distance != models.Distance.COSINE or info.config.metadata != self.identity):
            raise ValueError('Qdrant collection configuration does not match the embedding spec and vault.')
        return info

    def upsert(self, records: Sequence[ChunkRecord], vectors) -> None:
        records = list(records)
        validate_records(records, vault_id=self.vault_id)
        matrix = validate_vectors(vectors, rows=len(records), dimensions=self.spec.dimensions,
                                  dtype=self.spec.dtype, normalization=self.spec.normalization)
        points = [models.PointStruct(id=point_id(record.chunk_id), vector=vector.tolist(), payload={
            'chunk_id': record.chunk_id, 'vault_id': self.vault_id, 'embedding_spec': self.spec.fingerprint,
            'source': record.chunk.source, 'document_id': record.document_id,
            'document_revision': record.document_revision, 'chunk_index': record.chunk.chunk_index,
        }) for record, vector in zip(records, matrix)]
        for start in range(0, len(points), 128):
            self.client.upsert(collection_name=self.collection, points=points[start:start + 128], wait=True)

    def search(self, query_vector, *, top_k: int = 2, source: str | None = None,
               exact: bool = False, ef_search: int | None = None) -> list[VectorHit]:
        validate_search(top_k, source, exact)
        if ef_search is not None and (type(ef_search) is not int or ef_search <= 0):
            raise ValueError('ef_search must be positive.')
        query = validate_vectors([query_vector], rows=1, dimensions=self.spec.dimensions,
                                 dtype=self.spec.dtype, normalization=self.spec.normalization)[0]
        values = {'vault_id': self.vault_id, 'embedding_spec': self.spec.fingerprint}
        if source is not None:
            values['source'] = source
        query_filter = models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
                                           for k, v in values.items()])
        points = self.client.query_points(
            collection_name=self.collection, query=query.tolist(), query_filter=query_filter,
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

    def delete(self, chunk_ids: Sequence[str]) -> None:
        ids = [point_id(chunk_id) for chunk_id in chunk_ids]
        if ids:
            self.client.delete(collection_name=self.collection,
                               points_selector=models.PointIdsList(points=ids), wait=True)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count
