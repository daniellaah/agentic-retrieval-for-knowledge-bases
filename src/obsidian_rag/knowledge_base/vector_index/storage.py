"""SQLite snapshots, document-vector cache, and atomic index publication."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from obsidian_rag.knowledge_base.chunking import Chunk
from obsidian_rag.knowledge_base.embeddings import prepare_document, validate_vectors
from obsidian_rag.knowledge_base.vector_index.manifest import ChunkRecord, EmbeddingSpec, IndexManifest


STORAGE_VERSION = 1
_DDL = """
CREATE TABLE embeddings (
    key TEXT PRIMARY KEY, spec TEXT NOT NULL, vector BLOB NOT NULL, checksum TEXT NOT NULL
);
CREATE TABLE builds (
    version TEXT PRIMARY KEY, vault_id TEXT NOT NULL, manifest TEXT NOT NULL,
    corpus_fingerprint TEXT NOT NULL, backend TEXT NOT NULL, error TEXT
);
CREATE TABLE snapshot_chunks (
    version TEXT NOT NULL REFERENCES builds(version) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL, chunk_id TEXT NOT NULL, record TEXT NOT NULL,
    embedding_key TEXT NOT NULL REFERENCES embeddings(key),
    PRIMARY KEY(version, chunk_id), UNIQUE(version, ordinal)
);
CREATE TABLE active_indexes (
    vault_id TEXT PRIMARY KEY, version TEXT NOT NULL REFERENCES builds(version)
);
PRAGMA user_version = 1;
"""


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _manifest(text: str) -> IndexManifest:
    data = json.loads(text)
    data["embedding_spec"] = EmbeddingSpec(**data["embedding_spec"])
    return IndexManifest(**data)


class SQLiteStorage:
    """One connection per owner; queries can open an existing database read-only.

    Cache entries are immutable and checksummed. READY snapshots are immutable;
    publication verifies every record/vector then changes the active pointer in
    one transaction. A failed candidate never replaces an existing active index.
    Text snapshots preserve coordinates into the loaded Note.content, not files.
    """

    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro",
                                             uri=True, isolation_level=None, timeout=5)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0 and not read_only:
                tables = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if tables:
                    raise ValueError("Refusing to initialize an unrelated SQLite database.")
                self.connection.executescript("BEGIN IMMEDIATE;\n" + _DDL + "COMMIT;")
                version = STORAGE_VERSION
            if version != STORAGE_VERSION:
                raise ValueError(f"Unsupported storage version: {version}.")
            if not read_only:
                self.connection.execute("PRAGMA journal_mode=WAL")
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def _transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    @contextmanager
    def writer_lock(self):
        """Serialize complete builds across processes; OS releases locks on exit."""
        if self.read_only:
            raise ValueError('A read-only database cannot acquire a writer lock.')
        resolved = self.path.resolve()
        path = resolved.with_name(resolved.name + '.writer.lock')
        with path.open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Another index build is running for this database.') from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def recover_builds(self, vault_id: str) -> int:
        """Call only while holding writer_lock; abandon interrupted candidates."""
        pending = [m for m in self.list_builds(vault_id) if m.status == 'building']
        for manifest in pending:
            self.mark_failed(manifest.index_version, 'Interrupted build; rerun uses cached batches.')
        return len(pending)

    def put_embeddings(self, spec: EmbeddingSpec, texts: list[str], vectors) -> list[str]:
        keys = [spec.embedding_key(text) for text in texts]
        if not texts:
            return []
        matrix = validate_vectors(vectors, rows=len(texts), dimensions=spec.dimensions,
                                  dtype=spec.dtype, normalization=spec.normalization)
        with self._transaction():
            for key, vector in zip(keys, matrix):
                data = vector.astype('<f4' if spec.dtype == 'float32' else '<f8').tobytes()
                previous = self.connection.execute("SELECT * FROM embeddings WHERE key=?", (key,)).fetchone()
                digest = hashlib.sha256(data).hexdigest()
                if previous is not None:
                    if (previous['spec'], previous['vector'], previous['checksum']) != (spec.fingerprint, data, digest):
                        raise ValueError("Conflicting or corrupt cached embedding.")
                else:
                    self.connection.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?)",
                                            (key, spec.fingerprint, data, digest))
        return keys

    def get_embedding(self, spec: EmbeddingSpec, text: str):
        key = spec.embedding_key(text)
        row = self.connection.execute("SELECT * FROM embeddings WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row['spec'] != spec.fingerprint or hashlib.sha256(row['vector']).hexdigest() != row['checksum']:
            raise ValueError("Corrupt embedding cache metadata or checksum.")
        dtype = '<f4' if spec.dtype == 'float32' else '<f8'
        if len(row['vector']) != spec.dimensions * np.dtype(dtype).itemsize:
            raise ValueError("Corrupt cached vector length.")
        matrix = np.frombuffer(row['vector'], dtype=dtype).reshape(1, -1)
        return validate_vectors(matrix, rows=1, dimensions=spec.dimensions,
                                dtype=spec.dtype, normalization=spec.normalization)[0]

    def create_build(self, manifest: IndexManifest, *, corpus_fingerprint: str,
                     backend: dict | None = None) -> None:
        if manifest.status != 'building':
            raise ValueError("New builds must start in building state.")
        with self._transaction():
            self.connection.execute("INSERT INTO builds VALUES (?, ?, ?, ?, ?, NULL)",
                                    (manifest.index_version, manifest.vault_id, _json(asdict(manifest)),
                                     corpus_fingerprint, _json(backend or {"kind": "qdrant"})))

    def get_manifest(self, version: str) -> IndexManifest:
        row = self.connection.execute("SELECT manifest FROM builds WHERE version=?", (version,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown index version: {version}.")
        return _manifest(row['manifest'])

    def build_metadata(self, version: str) -> dict:
        row = self.connection.execute("SELECT * FROM builds WHERE version=?", (version,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown index version: {version}.")
        return {"corpus_fingerprint": row['corpus_fingerprint'],
                "backend": json.loads(row['backend']), "error": row['error']}

    def set_backend(self, version: str, backend: dict) -> None:
        with self._transaction():
            self._building(version)
            self.connection.execute("UPDATE builds SET backend=? WHERE version=?", (_json(backend), version))

    def _building(self, version: str) -> IndexManifest:
        manifest = self.get_manifest(version)
        if manifest.status != 'building':
            raise ValueError("Only building snapshots may be changed.")
        return manifest

    def add_chunk(self, version: str, record: ChunkRecord, *, ordinal: int) -> None:
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer.")
        with self._transaction():
            manifest = self._building(version)
            if record.vault_id != manifest.vault_id:
                raise ValueError("Chunk belongs to a different vault.")
            spec = manifest.embedding_spec
            text = prepare_document(record.chunk, document_template=spec.document_template)
            if self.get_embedding(spec, text) is None:
                raise ValueError("Chunk embedding is missing from the cache.")
            self.connection.execute("INSERT INTO snapshot_chunks VALUES (?, ?, ?, ?, ?)",
                                    (version, ordinal, record.chunk_id, _json(asdict(record)), spec.embedding_key(text)))

    def snapshot_records(self, version: str) -> list[ChunkRecord]:
        manifest = self.get_manifest(version)
        spec = manifest.embedding_spec
        rows = self.connection.execute("SELECT * FROM snapshot_chunks WHERE version=? ORDER BY ordinal",
                                       (version,)).fetchall()
        records = []
        for ordinal, row in enumerate(rows):
            data = json.loads(row['record'])
            data['chunk'] = Chunk(**data['chunk'])
            record = ChunkRecord(**data)
            text = prepare_document(record.chunk, document_template=spec.document_template)
            if (row['ordinal'] != ordinal or record.chunk_id != row['chunk_id']
                    or record.vault_id != manifest.vault_id or spec.embedding_key(text) != row['embedding_key']):
                raise ValueError("Corrupt snapshot record identity or ordering.")
            records.append(record)
        revisions = {}
        for record in records:
            previous = revisions.setdefault(record.document_id, record.document_revision)
            if previous != record.document_revision:
                raise ValueError("Snapshot mixes document revisions.")
        if len(records) != manifest.chunk_count or len(revisions) != manifest.document_count:
            raise ValueError("Snapshot counts do not match its manifest.")
        return records

    def knowledge_snapshot(self, version: str):
        """Open a published source view for inspection, grep, metadata and BM25.

        Existing v1 databases retain complete chunk coverage. Reconstruct and
        verify full notes using the shared knowledge layer without reading any
        embedding BLOBs or contacting Qdrant/the model. No schema migration.
        """
        from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot
        manifest = self.get_manifest(version)
        if manifest.status != 'ready':
            raise ValueError('Knowledge inspection requires a ready snapshot.')
        return KnowledgeSnapshot.from_records(
            self.snapshot_records(version), vault_id=manifest.vault_id, snapshot_id=version,
            corpus_fingerprint=self.build_metadata(version)['corpus_fingerprint'])

    def load_snapshot(self, version: str):
        manifest = self.get_manifest(version)
        spec = manifest.embedding_spec
        records = self.snapshot_records(version)
        vectors = [self.get_embedding(spec, prepare_document(r.chunk, document_template=spec.document_template))
                   for r in records]
        if any(vector is None for vector in vectors):
            raise ValueError('Snapshot embedding is missing.')
        matrix = np.asarray(vectors, dtype=spec.dtype).reshape(len(records), spec.dimensions)
        return manifest, records, matrix

    def publish(self, version: str) -> IndexManifest:
        with self._transaction():
            manifest = self._building(version)
            self.load_snapshot(version)
            ready = replace(manifest, status='ready')
            self.connection.execute("UPDATE builds SET manifest=? WHERE version=?", (_json(asdict(ready)), version))
            self.connection.execute("INSERT INTO active_indexes VALUES (?, ?) ON CONFLICT(vault_id) "
                                    "DO UPDATE SET version=excluded.version", (ready.vault_id, version))
        return ready

    def mark_failed(self, version: str, error: str) -> None:
        with self._transaction():
            failed = replace(self._building(version), status='failed')
            self.connection.execute("UPDATE builds SET manifest=?, error=? WHERE version=?",
                                    (_json(asdict(failed)), error, version))

    def active_manifest(self, vault_id: str) -> IndexManifest | None:
        row = self.connection.execute("SELECT version FROM active_indexes WHERE vault_id=?", (vault_id,)).fetchone()
        if row is None:
            return None
        manifest = self.get_manifest(row['version'])
        if manifest.status != 'ready' or manifest.vault_id != vault_id:
            raise ValueError("Active index must reference a ready snapshot in its vault.")
        return manifest

    def list_builds(self, vault_id: str) -> list[IndexManifest]:
        return [_manifest(row['manifest']) for row in self.connection.execute(
            "SELECT manifest FROM builds WHERE vault_id=? ORDER BY rowid", (vault_id,))]

    def delete_build(self, version: str) -> None:
        with self._transaction():
            if self.connection.execute("SELECT 1 FROM active_indexes WHERE version=?", (version,)).fetchone():
                raise ValueError("Cannot delete an active snapshot.")
            self.connection.execute("DELETE FROM builds WHERE version=?", (version,))

    def get_record(self, version: str, chunk_id: str) -> ChunkRecord:
        """Fetch one verified source record without loading any document vectors."""
        manifest = self.get_manifest(version)
        row = self.connection.execute('SELECT * FROM snapshot_chunks WHERE version=? AND chunk_id=?',
                                      (version, chunk_id)).fetchone()
        if row is None:
            raise ValueError('Vector hit is absent from the snapshot.')
        data = json.loads(row['record'])
        data['chunk'] = Chunk(**data['chunk'])
        record = ChunkRecord(**data)
        text = prepare_document(record.chunk, document_template=manifest.embedding_spec.document_template)
        if (record.chunk_id != chunk_id or record.vault_id != manifest.vault_id
                or manifest.embedding_spec.embedding_key(text) != row['embedding_key']):
            raise ValueError('Corrupt snapshot hit identity.')
        return record
