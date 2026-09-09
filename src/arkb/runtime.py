"""Compose retrieval capabilities with explicit, caller-owned resources."""
from collections.abc import Mapping
from typing import TYPE_CHECKING
from arkb.knowledge.models import EmbeddingSpec
from arkb.retrieval.models import SearchResponse, validate_request
from arkb.retrieval.semantic import QdrantSnapshotIndex, SemanticRetriever
if TYPE_CHECKING:
    from arkb.knowledge.sqlite import SQLiteStorage
    from ollama import Client
    from tokenizers import Tokenizer


class SnapshotSemanticRetriever:
    """Reusable application composition, including empty-snapshot validation.

    Caller-owned clients and storage remain open for the lifetime of queries.
    """
    def __init__(self, storage: "SQLiteStorage", *, vault_id: str, spec: EmbeddingSpec,
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
    storage: "SQLiteStorage", question: str, *, vault_id: str, spec: EmbeddingSpec,
    tokenizer: 'Tokenizer', client: 'Client', top_k: int = 2,
    source: str | None = None, exact: bool = False, ef_search: int | None = None,
    index_version: str | None = None, qdrant_client=None,
) -> SearchResponse:
    """One-shot compatibility entry point for the current semantic adapters."""
    filters = validate_request(question, top_k, {'source': source} if source is not None else None)
    retriever = SnapshotSemanticRetriever(storage, vault_id=vault_id, spec=spec, tokenizer=tokenizer,
        client=client, exact=exact, ef_search=ef_search, index_version=index_version, qdrant_client=qdrant_client)
    return retriever.search(question, top_k=top_k, filters=filters)
