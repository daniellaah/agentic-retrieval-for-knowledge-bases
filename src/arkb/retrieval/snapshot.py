"""Shared translation of immutable stored chunks to retrieval evidence."""

from arkb.retrieval.contracts import SearchResult
from arkb.schema import ChunkRecord


def chunk_result(record: ChunkRecord, *, method: str, index_id: str | None = None,
                 score: float | None = None, score_type: str | None = None) -> SearchResult:
    chunk = record.chunk
    return SearchResult(
        source_id=record.document_id, source=chunk.source, content=chunk.content,
        method=method, chunk_id=record.chunk_id,
        start_char=chunk.start_char, end_char=chunk.end_char,
        score=score, score_type=score_type,
        metadata={'title': chunk.title, 'vault_id': record.vault_id,
                  'document_revision': record.document_revision, 'index_version': index_id,
                  'chunk_index': chunk.chunk_index, 'heading_path': list(chunk.heading_path),
                  'section_id': chunk.section_id, 'section_start_char': chunk.section_start_char,
                  'section_end_char': chunk.section_end_char, 'occurrence': chunk.occurrence},
    )
