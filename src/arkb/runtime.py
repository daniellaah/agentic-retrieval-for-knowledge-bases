"""Compose capabilities and manage the lifetime of resources opened here."""

from collections.abc import Mapping
from contextlib import ExitStack, closing
from pathlib import Path
from typing import TYPE_CHECKING
from arkb.config import RuntimeConfig, RetrievalConfig
from arkb.knowledge.models import EmbeddingSpec
from arkb.retrieval.models import SearchResponse, validate_request
from arkb.retrieval.semantic import QdrantSnapshotIndex, SemanticRetriever
if TYPE_CHECKING:
    from arkb.agent.tools import AgentTools
    from arkb.retrieval.engine import RetrievalEngine
    from arkb.knowledge.sqlite import SQLiteStorage
    from ollama import Client
    from qdrant_client import QdrantClient
    from tokenizers import Tokenizer


class Runtime:
    """Own only resources opened here; load each service on first use.

    Storage and explicitly composed adapters remain caller-owned. Keep a
    prepared engine for a query session to retain its captured snapshot.
    """

    def __init__(self, config: RuntimeConfig = RuntimeConfig()):
        self.config = config
        self._resources = ExitStack()
        self._client = None
        self._qdrant_clients: dict[str, 'QdrantClient'] = {}
        self._tokenizer = None
        self._closed = False

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self._closed = True
        return self._resources.__exit__(*exc)

    def _require_open(self):
        if self._closed:
            raise RuntimeError('Runtime is closed.')

    def model_client(self):
        self._require_open()
        if self._client is None:
            from ollama import Client
            self._client = self._resources.enter_context(
                Client(host=self.config.host, timeout=self.config.timeout, trust_env=False))
        return self._client

    def qdrant_client(self, url: str):
        self._require_open()
        if url not in self._qdrant_clients:
            from arkb.knowledge.qdrant import connect_qdrant
            client = self._resources.enter_context(closing(connect_qdrant(url, self.config.timeout
                if self.config.qdrant_timeout is None else self.config.qdrant_timeout)))
            self._qdrant_clients[url] = client
        return self._qdrant_clients[url]

    def tokenizer(self):
        self._require_open()
        if self._tokenizer is None:
            from arkb.knowledge.embeddings import load_tokenizer
            self._tokenizer = load_tokenizer(cache_dir=self.config.tokenizer_cache,
                                             local_files_only=self.config.offline)
        return self._tokenizer

    def semantic(self, storage, manifest, *, exact=False, ef_search=None):
        from arkb.knowledge.embeddings import resolve_embedding_spec
        from arkb.knowledge.models import require_qdrant_backend
        metadata = storage.build_metadata(manifest.index_version)['backend']
        require_qdrant_backend(metadata)
        qclient = self.qdrant_client(self.config.qdrant_url or metadata['url'])
        tokenizer = self.tokenizer()
        client = self.model_client()
        spec = resolve_embedding_spec(client, self.config.embedding_model or manifest.embedding_spec.model,
                                      context_length=metadata['input']['max_tokens'])
        return SnapshotSemanticRetriever(storage, vault_id=manifest.vault_id, spec=spec,
            tokenizer=tokenizer, client=client, exact=exact, ef_search=ef_search,
            index_version=manifest.index_version, qdrant_client=qclient)

    def retrieval_engine(self, storage, manifest, *, modes=('semantic',), rerank=False,
                         settings: RetrievalConfig = RetrievalConfig(), exact=False, records=None):
        """Prepare only requested capabilities; algorithms remain independently callable."""
        self._require_open()
        from arkb.retrieval import BM25Retriever, RetrievalEngine, Reranker
        from arkb.retrieval.rerank import CrossEncoderScorer
        modes = set(modes)
        if not modes or modes - {'semantic', 'bm25', 'lexical', 'hybrid', 'hybrid_reranked'}:
            raise ValueError('Unknown or empty retrieval modes.')
        bm25 = semantic = reranker = None
        if modes & {'bm25', 'lexical', 'hybrid', 'hybrid_reranked'}:
            if records is None:
                bm25 = BM25Retriever.from_snapshot(storage, vault_id=manifest.vault_id,
                    index_version=manifest.index_version, k1=settings.bm25_k1, b=settings.bm25_b)
            else:
                bm25 = BM25Retriever(records, index_id=manifest.index_version,
                                     k1=settings.bm25_k1, b=settings.bm25_b)
        if modes & {'semantic', 'hybrid', 'hybrid_reranked'}:
            semantic = self.semantic(storage, manifest, exact=exact)
        if rerank or 'hybrid_reranked' in modes:
            reranker = Reranker(CrossEncoderScorer(model=settings.reranker_model,
                revision=settings.reranker_revision, max_length=settings.reranker_max_length,
                cache_folder=settings.reranker_cache, local_files_only=self.config.offline))
        return RetrievalEngine(semantic=semantic, bm25=bm25, reranker=reranker,
            candidate_k=settings.candidate_k, rrf_k=settings.rrf_k,
            rerank_candidates=settings.rerank_candidates)

    def agent_tools(self, *, engine: 'RetrievalEngine', directory: Path, vault_id: str,
                    mode: str = 'semantic', rerank: bool = False) -> 'AgentTools':
        """Bind live document tools to an already prepared retrieval engine.

        Use the same directory/vault as indexing. The caller selects the search
        policy here; the agent only supplies a query, source filter, and limit.
        This composition opens no clients and creates no new resource owners.
        """
        self._require_open()
        from arkb.agent.tools import AgentTools
        from arkb.knowledge.documents import DocumentAccess
        from arkb.retrieval.exact import ExactRetriever

        documents = DocumentAccess(directory, vault_id=vault_id)
        return AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine,
                          mode=mode, rerank=rerank)


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
