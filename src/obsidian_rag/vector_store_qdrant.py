"""Qdrant adapter using externally computed vectors and source metadata."""

from collections.abc import Sequence
import math
from qdrant_client import QdrantClient, models
from obsidian_rag.embeddings import validate_vectors
from obsidian_rag.schema import (
    ChunkRecord,
    EmbeddingSpec,
    point_id,
    VectorHit,
    validate_records,
    qdrant_identity,
)
from obsidian_rag.retrieval import check_qdrant_collection, search_qdrant


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
                 ef_construct: int = 100, indexing_threshold: int = 10000, full_scan_threshold: int = 10000):
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError('collection must be nonblank.')
        if not isinstance(vault_id, str) or not vault_id.strip():
            raise ValueError('vault_id must be nonblank.')
        for name, value, minimum in (('hnsw_m', hnsw_m, 2), ('ef_construct', ef_construct, 1),
                                     ('indexing_threshold', indexing_threshold, 0), ('full_scan_threshold', full_scan_threshold, 10)):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        self.client, self.collection, self.spec, self.vault_id = client, collection, spec, vault_id
        self.identity = qdrant_identity(spec, vault_id)
        if create:
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=spec.dimensions, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=ef_construct, full_scan_threshold=full_scan_threshold),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
                metadata=self.identity,
            )
            for field in ('vault_id', 'embedding_spec', 'source'):
                client.create_payload_index(collection_name=collection, field_name=field,
                                            field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
        self.check_configuration()

    def check_configuration(self):
        return check_qdrant_collection(self.client, self.collection, spec=self.spec, vault_id=self.vault_id)

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
        return search_qdrant(self.client, self.collection, query_vector, spec=self.spec,
                             vault_id=self.vault_id, top_k=top_k, source=source, exact=exact, ef_search=ef_search)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        ids = [point_id(chunk_id) for chunk_id in chunk_ids]
        if ids:
            self.client.delete(collection_name=self.collection,
                               points_selector=models.PointIdsList(points=ids), wait=True)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def verify_snapshot(self, records: Sequence[ChunkRecord], vectors) -> None:
        """Verify every point and vector before SQLite can publish this collection."""
        import numpy as np
        if self.count() != len(records):
            raise ValueError('Qdrant point count does not match snapshot.')
        for start in range(0, len(records), 128):
            batch = records[start:start + 128]
            points = self.client.retrieve(self.collection, ids=[point_id(r.chunk_id) for r in batch],
                                          with_payload=True, with_vectors=True)
            by_id = {str(point.id): point for point in points}
            for index, record in enumerate(batch, start):
                point = by_id.get(point_id(record.chunk_id))
                if point is None:
                    raise ValueError('Qdrant snapshot is missing a point.')
                expected = {'chunk_id': record.chunk_id, 'vault_id': self.vault_id,
                            'embedding_spec': self.spec.fingerprint, 'source': record.chunk.source,
                            'document_id': record.document_id, 'document_revision': record.document_revision,
                            'chunk_index': record.chunk.chunk_index}
                if point.payload != expected:
                    raise ValueError('Qdrant snapshot payload differs from source records.')
                target = np.asarray(vectors[index], dtype=np.float64)
                target = target / np.linalg.norm(target)
                actual = np.asarray(point.vector, dtype=np.float64)
                if actual.shape != target.shape or not np.allclose(actual, target, atol=1e-6, rtol=1e-5):
                    raise ValueError('Qdrant snapshot vector differs from cached vector.')

    def wait_ready(self, *, expected_count: int, timeout: float = 30,
                   require_hnsw: bool = False) -> dict:
        """Distinguish query-ready small collections from fully built HNSW indexes."""
        import time
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Index readiness timeout must be positive and finite.')
        deadline = time.monotonic() + timeout
        while True:
            info = self.check_configuration()
            if info.optimizer_status != 'ok' or info.status == models.CollectionStatus.RED:
                raise ValueError(f'Qdrant optimizer failed: {info.optimizer_status}.')
            indexed = info.indexed_vectors_count or 0
            if (info.status == models.CollectionStatus.GREEN and self.count() == expected_count
                    and (not require_hnsw or indexed >= expected_count)):
                return {'points': expected_count, 'indexed_vectors': indexed, 'status': 'green'}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('Timed out waiting for the requested Qdrant index readiness.')
            time.sleep(min(.2, remaining))

    def drop(self) -> None:
        """Delete only an explicitly selected collection with matching ownership."""
        self.check_configuration()
        self.client.delete_collection(self.collection)
