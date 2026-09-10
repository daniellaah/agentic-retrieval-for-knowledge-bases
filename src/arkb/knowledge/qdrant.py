"""Qdrant persistence and raw vector reads; no retrieval result contracts."""

from collections.abc import Sequence
import math
import os
import re
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient, models
from arkb.knowledge.embeddings import validate_vectors
from arkb.knowledge.models import (
    SCHEMA_VERSION, ChunkRecord, EmbeddingSpec, QdrantConfig, VectorHit, validate_records,
)

def point_id(chunk_id: str) -> str:
    if not isinstance(chunk_id, str) or re.fullmatch('[0-9a-f]{64}', chunk_id) is None:
        raise ValueError('Expected a SHA-256 chunk ID.')
    # Chunk IDs already encode the current schema; use the ARKB UUID namespace.
    return str(uuid5(NAMESPACE_URL, 'arkb/chunk/' + chunk_id))


def qdrant_identity(spec: EmbeddingSpec, vault_id: str) -> dict:
    """Metadata identifying one owned Qdrant collection's embedding space."""
    return {'owner': 'arkb', 'schema': SCHEMA_VERSION, 'embedding_spec': spec.fingerprint, 'vault_id': vault_id}

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


def connect_qdrant(url: str, timeout: float):
    from qdrant_client import QdrantClient
    parts = urlsplit(url)
    if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError('Use an HTTP(S) Qdrant URL without embedded credentials; set QDRANT_API_KEY if needed.')
    return QdrantClient(url=url, api_key=os.environ.get('QDRANT_API_KEY'), timeout=timeout, trust_env=False)

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
    if type(top_k) is not int or top_k <= 0:
        raise ValueError('top_k must be a positive integer.')
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError('source filter must be nonblank text.')
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

class QdrantIndex:
    """One collection per immutable index candidate.

    create=True creates a new collection, never recreates/deletes an existing one.
    Configuration metadata guards against same-dimension incompatible models.
    SQLite remains the source of chunk text and original float64/float32 vectors;
    Qdrant uses float32 cosine vectors, so exact scores can differ by rounding.
    Local Qdrant clients are useful for API tests; ANN requires Qdrant Server.
    """

    def __init__(self, client: QdrantClient, collection: str, spec: EmbeddingSpec, *,
                 vault_id: str, create: bool = False, config: QdrantConfig = QdrantConfig()):
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError('collection must be nonblank.')
        if not isinstance(vault_id, str) or not vault_id.strip():
            raise ValueError('vault_id must be nonblank.')
        if not isinstance(config, QdrantConfig):
            raise ValueError('QdrantIndex requires a QdrantConfig.')
        self.config = config
        self.client, self.collection, self.spec, self.vault_id = client, collection, spec, vault_id
        self.identity = qdrant_identity(spec, vault_id)
        if create:
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=spec.dimensions, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=config.hnsw_m, ef_construct=config.ef_construct,
                                                    full_scan_threshold=config.full_scan_threshold),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=config.indexing_threshold),
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

    def wait_ready(self, *, expected_count: int) -> dict:
        """Distinguish query-ready small collections from fully built HNSW indexes."""
        import time
        deadline = time.monotonic() + self.config.index_timeout
        while True:
            info = self.check_configuration()
            if info.optimizer_status != 'ok' or info.status == models.CollectionStatus.RED:
                raise ValueError(f'Qdrant optimizer failed: {info.optimizer_status}.')
            indexed = info.indexed_vectors_count or 0
            if (info.status == models.CollectionStatus.GREEN and self.count() == expected_count
                    and (not self.config.require_hnsw or indexed >= expected_count)):
                return {'points': expected_count, 'indexed_vectors': indexed, 'status': 'green'}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('Timed out waiting for the requested Qdrant index readiness.')
            time.sleep(min(.2, remaining))

    def drop(self) -> None:
        """Delete only an explicitly selected collection with matching ownership."""
        self.check_configuration()
        self.client.delete_collection(self.collection)
