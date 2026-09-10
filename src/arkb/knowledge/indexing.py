"""Build complete, validated index candidates and atomically publish them."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial, wraps
from time import perf_counter
from uuid import uuid4

from ollama import Client
from tokenizers import Tokenizer

from arkb.knowledge.models import (
    QdrantConfig, ChunkRecord, EmbeddingSpec, IndexManifest, Note, fingerprint_config,
)
from arkb.knowledge.qdrant import QdrantIndex

from arkb.knowledge.chunking import chunk_notes, whole_note_chunks
from arkb.knowledge.embeddings import (
    DEFAULT_QUERY_INSTRUCTION, prepare_document, prepare_query, validate_input_tokens,
    iter_embedding_batches, count_tokens, tokenizer_fingerprint,
)
from arkb.knowledge.sqlite import SQLiteStorage


@dataclass(frozen=True)
class BuildReport:
    manifest: IndexManifest
    embedded_inputs: int
    cached_inputs: int
    reused_index: bool = False
    added_documents: int = 0
    modified_documents: int = 0
    deleted_documents: int = 0
    build_seconds: float = 0.0


def _exclusive_build(function):
    @wraps(function)
    def run(storage, *args, **kwargs):
        with storage.writer_lock():
            started = perf_counter()
            report = function(storage, *args, **kwargs)
            return replace(report, build_seconds=perf_counter() - started)
    return run


@_exclusive_build
def build_index(
    storage: SQLiteStorage, notes: Sequence[Note], *, spec: EmbeddingSpec,
    vault_id: str, client: Client, tokenizer: Tokenizer, max_input_tokens: int,
    chunking: str = 'recursive', chunk_size: int = 512, chunk_overlap: int = 64,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION, index_version: str | None = None,
    batch_size: int = 32, max_batch_tokens: int | None = None, max_retries: int = 0,
    qdrant_config: QdrantConfig,
    force: bool = False, source_scope: str | None = None, qdrant_client=None,
) -> BuildReport:
    """Preflight inputs, reuse cached vectors, checkpoint batches, then publish.

    The caller resolves the model artifact and supplies its matching tokenizer
    and active input limit. Sources must be unique within a complete vault scan.
    Source errors and input validation happen before creating a candidate. Every
    successful embedding batch is durable even if a later batch fails.
    Qdrant receives a separate candidate
    collection that must be verified before publication.
    """
    notes = list(notes)
    if any(not isinstance(note, Note) for note in notes) or len({n.source for n in notes}) != len(notes):
        raise ValueError('Expected notes with unique source paths.')
    prepare_query('validation', instruction=query_instruction)
    if type(max_input_tokens) is not int or max_input_tokens <= 0:
        raise ValueError('max_input_tokens must be a positive integer.')
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise ValueError('Input tokenizer must have padding and truncation disabled.')
    if chunking == 'recursive':
        chunks = chunk_notes(notes, count_tokens=partial(count_tokens, tokenizer=tokenizer),
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunking == 'none':
        chunks = whole_note_chunks(notes)
    else:
        raise ValueError('chunking must be recursive or none.')
    source_notes = {note.source: note for note in notes}
    records = [ChunkRecord.from_note(chunk, note=source_notes[chunk.source], vault_id=vault_id) for chunk in chunks]
    texts = [prepare_document(record.chunk, document_template=spec.document_template) for record in records]
    counts = [validate_input_tokens(text, tokenizer=tokenizer, max_tokens=max_input_tokens,
                                   source=f'{r.chunk.source}, chunk {r.chunk.chunk_index}')
              for text, r in zip(texts, records)]
    token_identity = tokenizer_fingerprint(tokenizer)
    # Keep the existing CLI mode names, but invalidate pre-Markdown snapshots.
    algorithm = 'markdown-v1' if chunking == 'recursive' else 'whole-note-v2'
    chunk_config = {'algorithm': algorithm, 'tokenizer': token_identity}
    if chunking == 'recursive':
        chunk_config.update(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    manifest = IndexManifest(
        index_version=index_version or uuid4().hex, vault_id=vault_id, embedding_spec=spec,
        chunking_fingerprint=fingerprint_config(chunk_config), document_count=len(notes),
        chunk_count=len(records), query_instruction=query_instruction,
    )
    if not isinstance(qdrant_config, QdrantConfig):
        raise ValueError('Qdrant indexing requires a QdrantConfig.')
    metadata = qdrant_config.to_metadata()
    if qdrant_client is None:
        raise ValueError('Qdrant indexing requires an explicit client.')
    metadata['input'] = {'max_tokens': max_input_tokens, 'tokenizer': token_identity}
    if source_scope is not None:
        metadata['source_scope'] = source_scope
    corpus = fingerprint_config({'notes': [asdict(n) for n in notes]})
    active = storage.active_manifest(vault_id)
    if active is not None:
        previous = storage.build_metadata(active.index_version)
        old_scope = previous['backend'].get('source_scope')
        if old_scope is not None and old_scope != source_scope:
            raise ValueError('Source scope differs from this vault; use a separate vault ID.')
    storage.recover_builds(vault_id)
    _cleanup_failed_qdrant(storage, vault_id=vault_id, client=qdrant_client, url=metadata['url'])
    unique_count = len(set(texts))
    if (active is not None and not force and index_version is None
            and previous['corpus_fingerprint'] == corpus and _backend_settings(previous['backend']) == _backend_settings(metadata)
            and active.configuration_fingerprint == manifest.configuration_fingerprint):
        _, prior_records, prior_vectors = storage.load_snapshot(active.index_version)
        remote = _qdrant_index(qdrant_client, previous['backend'], active, create=False)
        remote.verify_snapshot(prior_records, prior_vectors)
        remote.wait_ready(expected_count=active.chunk_count)
        return BuildReport(active, 0, unique_count, reused_index=True)
    old_records = storage.snapshot_records(active.index_version) if active is not None else []
    old = {r.chunk.source: r.document_revision for r in old_records}
    new = {r.chunk.source: r.document_revision for r in records}
    added = len(new.keys() - old.keys())
    modified = sum(old[source] != new[source] for source in new.keys() & old.keys())
    deleted = len(old.keys() - new.keys())
    metadata['collection'] = 'arkb_' + fingerprint_config({'version': manifest.index_version, 'vault': vault_id})[:32]
    storage.create_build(manifest, corpus_fingerprint=corpus, backend=metadata)
    try:
        remote = _qdrant_index(qdrant_client, metadata, manifest, create=True)
        unique = dict(zip(texts, counts))
        missing = [text for text in unique if storage.get_embedding(spec, text) is None]
        for start, vectors in iter_embedding_batches(
            missing, client=client, model=spec.model, batch_size=batch_size,
            token_counts=[unique[text] for text in missing], max_batch_tokens=max_batch_tokens,
            dimensions=spec.dimensions, dtype=spec.dtype, normalization=spec.normalization,
            max_retries=max_retries, context_length=max_input_tokens,
        ):
            storage.put_embeddings(spec, missing[start:start + len(vectors)], vectors)
        for ordinal, record in enumerate(records):
            storage.add_chunk(manifest.index_version, record, ordinal=ordinal)
        _, loaded, vectors = storage.load_snapshot(manifest.index_version)
        if len(loaded) != len(records):
            raise ValueError('Candidate snapshot count does not match source records.')
        if loaded:
            remote.upsert(loaded, vectors)
        remote.verify_snapshot(loaded, vectors)
        metadata['index_stats'] = remote.wait_ready(expected_count=len(records))
        storage.set_backend(manifest.index_version, metadata)
        ready = storage.publish(manifest.index_version)
        return BuildReport(ready, len(missing), len(unique) - len(missing),
                           added_documents=added, modified_documents=modified, deleted_documents=deleted)
    except BaseException as error:
        storage.mark_failed(manifest.index_version, str(error) or type(error).__name__)
        raise


def _backend_settings(metadata: dict) -> dict:
    settings = {key: value for key, value in metadata.items() if key not in ('collection', 'index_stats')}
    if metadata.get('kind') == 'qdrant':
        # Old Python builds omitted defaults; equal settings must still be a no-op.
        settings.update(QdrantConfig.from_metadata(metadata).to_metadata())
    return settings


def _qdrant_index(client, metadata: dict, manifest: IndexManifest, *, create: bool):
    return QdrantIndex(client, metadata['collection'], manifest.embedding_spec,
                       vault_id=manifest.vault_id, create=create,
                       config=QdrantConfig.from_metadata(metadata))


def _cleanup_failed_qdrant(storage: SQLiteStorage, *, vault_id: str, client, url: str) -> int:
    """Under the writer lock, remove only owned failed candidates on this server.

    Historical READY collections are retained for rollback and in-flight readers.
    Failed build records and their successful embedding caches remain available.
    """
    count = 0
    for manifest in storage.list_builds(vault_id):
        metadata = storage.build_metadata(manifest.index_version)['backend']
        collection = metadata.get('collection')
        if (manifest.status == 'failed' and metadata.get('kind') == 'qdrant'
                and metadata.get('url') == url and collection and client.collection_exists(collection)):
            _qdrant_index(client, metadata, manifest, create=False).drop()
            count += 1
    return count
