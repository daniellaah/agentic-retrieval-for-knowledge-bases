"""Deterministic vector retrieval over one published Qdrant snapshot."""

from dataclasses import dataclass

from obsidian_rag.knowledge_base.embeddings import prepare_query, validate_input_tokens, embed_texts
from obsidian_rag.knowledge_base.tokenization import tokenizer_fingerprint
from obsidian_rag.knowledge_base.vector_index.qdrant import check_qdrant_collection, search_qdrant, validate_search
from .models import SearchResult, SearchResponse, SearchScope, SearchScore


@dataclass(frozen=True)
class VectorSearchConfig:
    top_k: int = 2
    source: str | None = None
    exact: bool = False
    ef_search: int | None = None

    def __post_init__(self):
        validate_search(self.top_k, self.source, self.exact)
        if self.ef_search is not None and (type(self.ef_search) is not int or self.ef_search < 1):
            raise ValueError('ef_search must be positive.')


def vector_search(storage, question: str, *, vault_id: str, spec, tokenizer, client,
                  qdrant_client, config: VectorSearchConfig | None = None,
                  index_version: str | None = None) -> SearchResponse:
    """Read source evidence from the captured snapshot, embedding only the query.

    A snapshot version pins every call in a retrieval task. Query configuration
    must match the indexed embedding artifact and tokenizer. NumPy index metadata
    is explicitly unsupported; compatible cached vectors may be rebuilt in Qdrant.
    """
    config = config or VectorSearchConfig()
    if not isinstance(config, VectorSearchConfig):
        raise ValueError('Expected VectorSearchConfig.')
    manifest = storage.get_manifest(index_version) if index_version is not None else storage.active_manifest(vault_id)
    if manifest is None:
        raise ValueError('No published index; run the index command first.')
    if manifest.status != 'ready' or manifest.vault_id != vault_id:
        raise ValueError('Queries require a ready snapshot in the requested vault.')
    metadata = storage.build_metadata(manifest.index_version)['backend']
    if metadata['kind'] != 'qdrant':
        raise ValueError('This index is not Qdrant; rebuild it with the Qdrant backend.')
    if not manifest.embedding_spec.is_compatible_with(spec):
        raise ValueError('Query embedding configuration is incompatible with the stored index.')
    inputs = metadata['input']
    if tokenizer_fingerprint(tokenizer) != inputs['tokenizer']:
        raise ValueError('Query tokenizer differs from the indexed tokenizer.')
    query = prepare_query(question, instruction=manifest.query_instruction)
    validate_input_tokens(query, tokenizer=tokenizer, max_tokens=inputs['max_tokens'], source='query')
    scope = SearchScope(vault_id, manifest.index_version, (config.source,) if config.source else ())
    if manifest.chunk_count == 0:
        return SearchResponse(question, 'vector', scope, (), config.top_k, has_more=False)
    if qdrant_client is None:
        raise ValueError('This index requires a Qdrant client.')
    check_qdrant_collection(qdrant_client, metadata['collection'], spec=spec, vault_id=vault_id)
    query_vector = embed_texts([query], client=client, model=spec.model,
                              dimensions=spec.dimensions, dtype=spec.dtype,
                              normalization=spec.normalization, context_length=inputs['max_tokens'])[0]
    hits = search_qdrant(qdrant_client, metadata['collection'], query_vector, spec=spec,
                        vault_id=vault_id, top_k=config.top_k, source=config.source,
                        exact=config.exact, ef_search=config.ef_search)
    results = tuple(SearchResult.from_record(storage.get_record(manifest.index_version, hit.chunk_id),
                    SearchScore(hit.score, 'cosine'), snapshot_id=manifest.index_version,
                    rank=rank, method='vector') for rank, hit in enumerate(hits, 1))
    return SearchResponse(question, 'vector', scope, results, config.top_k)
