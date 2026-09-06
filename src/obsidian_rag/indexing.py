"""Build complete, validated index candidates and atomically publish them."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
from uuid import uuid4

from ollama import Client
from tokenizers import Tokenizer

from obsidian_rag.chunking import chunk_notes, whole_note_chunks
from obsidian_rag.embedding_inputs import DEFAULT_QUERY_INSTRUCTION, prepare_document, prepare_query, validate_input_tokens
from obsidian_rag.embeddings import iter_embedding_batches
from obsidian_rag.index_schema import ChunkRecord, EmbeddingSpec, IndexManifest, fingerprint_config
from obsidian_rag.notes import Note
from obsidian_rag.storage import SQLiteStorage
from obsidian_rag.tokenization import count_tokens
from obsidian_rag.vector_store import NumpyVectorStore, VectorStore


@dataclass(frozen=True)
class BuildReport:
    manifest: IndexManifest
    embedded_inputs: int
    cached_inputs: int
    reused_index: bool = False


def tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    return hashlib.sha256(tokenizer.to_str().encode('utf-8')).hexdigest()


def build_index(
    storage: SQLiteStorage, notes: Sequence[Note], *, spec: EmbeddingSpec,
    vault_id: str, client: Client, tokenizer: Tokenizer, max_input_tokens: int,
    chunking: str = 'recursive', chunk_size: int = 512, chunk_overlap: int = 64,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION, index_version: str | None = None,
    batch_size: int = 32, max_batch_tokens: int | None = None, max_retries: int = 0,
    vector_store: VectorStore | None = None, backend: dict | None = None,
) -> BuildReport:
    """Preflight inputs, reuse cached vectors, checkpoint batches, then publish.

    The caller resolves the model artifact and supplies its matching tokenizer
    and active input limit. Sources must be unique within a complete vault scan.
    Source errors and input validation happen before creating a candidate. Every
    successful embedding batch is durable even if a later batch fails. A custom
    vector store must be empty and belong to this vault/spec; it is a candidate,
    never an existing published store. No network or model setup is hidden here.
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
    chunk_config = {'algorithm': f'{chunking}-v1', 'tokenizer': token_identity}
    if chunking == 'recursive':
        chunk_config.update(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    manifest = IndexManifest(
        index_version=index_version or uuid4().hex, vault_id=vault_id, embedding_spec=spec,
        chunking_fingerprint=fingerprint_config(chunk_config), document_count=len(notes),
        chunk_count=len(records), query_instruction=query_instruction,
    )
    projection = vector_store if vector_store is not None else NumpyVectorStore(spec, vault_id=vault_id)
    if projection.spec != spec or projection.vault_id != vault_id or projection.count() != 0:
        raise ValueError('Build requires an empty vector store matching the vault and embedding spec.')
    metadata = dict(backend or {'kind': 'numpy'})
    metadata['input'] = {'max_tokens': max_input_tokens, 'tokenizer': token_identity}
    storage.create_build(manifest, corpus_fingerprint=fingerprint_config({'notes': [asdict(n) for n in notes]}),
                         backend=metadata)
    try:
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
        if loaded:
            projection.upsert(loaded, vectors)
        if projection.count() != len(records):
            raise ValueError('Vector-store count does not match candidate snapshot.')
        ready = storage.publish(manifest.index_version)
        return BuildReport(ready, len(missing), len(unique) - len(missing))
    except BaseException as error:
        storage.mark_failed(manifest.index_version, str(error) or type(error).__name__)
        raise
