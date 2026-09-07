"""Construct test evidence at the shared public result boundary."""

from obsidian_rag.knowledge_base.models import Chunk, ChunkRecord
from obsidian_rag.knowledge_base.identity import digest
from obsidian_rag.knowledge_base.sources import SourceRef, SourceExcerpt
from obsidian_rag.retrieval.models import SearchResult, SpanTarget, ChunkTarget, SearchScore


def make_result(chunk, score, record=None, index_version=None):
    score = score if isinstance(score, SearchScore) else SearchScore(score, 'cosine')
    if record is not None:
        return SearchResult.from_record(record, score, snapshot_id=index_version, rank=1, method='vector')
    source = SourceRef('legacy', digest('document-id', {'vault_id':'legacy','source':chunk.source}),
                       chunk.source, chunk.title, None, None)
    return SearchResult(source, SpanTarget(chunk.start_char, chunk.end_char), 'vector', 1,
                        (SourceExcerpt(source, chunk.start_char, chunk.end_char, chunk.content),), score)


def chunk_of(hit):
    e=hit.excerpts[0]
    return Chunk(e.content, hit.source.title, hit.source.path,
                 hit.target.chunk_index if isinstance(hit.target, ChunkTarget) else 0,
                 e.start_char, e.end_char)


def record_of(hit):
    return ChunkRecord(hit.source.vault_id, hit.source.document_revision, chunk_of(hit))
