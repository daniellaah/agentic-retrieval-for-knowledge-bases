"""Compose capabilities and manage the lifetime of resources opened here."""

from collections.abc import Mapping
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from arkb.config import (
    DEFAULT_DB, DEFAULT_NOTES_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL,
    DEFAULT_AGENT_THINK, DEFAULT_RETRIEVAL_MODE, RuntimeConfig, RetrievalConfig,
)
from arkb.knowledge.embeddings import DEFAULT_QUERY_INSTRUCTION
from arkb.knowledge.models import EmbeddingSpec, QdrantConfig
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.retrieval.models import SearchResponse, validate_request
from arkb.retrieval.semantic import QdrantSnapshotIndex, SemanticRetriever
if TYPE_CHECKING:
    from arkb.agent.observation import AgentBudget, AgentObserver
    from arkb.agent.state import AgentResult
    from arkb.agent.tools import AgentTools
    from arkb.retrieval.engine import RetrievalEngine
    from arkb.knowledge.indexing import BuildReport
    from ollama import Client
    from qdrant_client import QdrantClient
    from tokenizers import Tokenizer


class Runtime:
    """Own only resources opened here; load each service on first use.

    One-shot entry points own their SQLite connections. Explicitly supplied
    storage and adapters remain caller-owned. Keep a prepared engine for a
    query session to retain its captured snapshot.
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

    @contextmanager
    def _snapshot(self, db: Path, vault_id: str, *, required: bool = True):
        self._require_open()
        if not isinstance(vault_id, str) or not vault_id.strip():
            raise ValueError('vault_id must not be blank.')
        with ExitStack() as resources:
            storage = (resources.enter_context(SQLiteStorage(Path(db), read_only=True))
                       if Path(db).exists() else None)
            manifest = storage.active_manifest(vault_id) if storage is not None else None
            if required and manifest is None:
                raise ValueError('No published index; run the index command first.')
            yield storage, manifest

    @staticmethod
    def _notes_directory(storage, manifest, notes_dir: Path | None) -> Path:
        scope = (storage.build_metadata(manifest.index_version)['backend'].get('source_scope')
                 if manifest is not None else None)
        if notes_dir is not None and scope is not None and Path(notes_dir).resolve() != Path(scope).resolve():
            raise ValueError('--notes-dir differs from the indexed knowledge base; use its directory or another vault/database.')
        return Path(notes_dir if notes_dir is not None else scope or DEFAULT_NOTES_DIR)

    def match(self, pattern: str, *, db: Path = DEFAULT_DB, vault_id: str = 'default',
              notes_dir: Path | None = None, top_k: int = 5,
              source: str | None = None) -> SearchResponse:
        """Find literal occurrences in live files without models or a required index."""
        from arkb.knowledge.documents import DocumentAccess
        from arkb.retrieval.exact import ExactRetriever

        filters = validate_request(pattern, top_k, {'source': source} if source is not None else None)
        with self._snapshot(db, vault_id, required=False) as (storage, manifest):
            documents = DocumentAccess(self._notes_directory(storage, manifest, notes_dir), vault_id=vault_id)
            return ExactRetriever(documents).search(pattern, top_k=top_k, filters=filters)

    def search(self, query: str, *, db: Path = DEFAULT_DB, vault_id: str = 'default',
               mode: str = DEFAULT_RETRIEVAL_MODE, top_k: int = 2, source: str | None = None,
               settings: RetrievalConfig = RetrievalConfig(), rerank: bool = False,
               exact: bool = False) -> SearchResponse:
        """Run one explicit ranked strategy on a published snapshot; never generate."""
        filters = validate_request(query, top_k, {'source': source} if source is not None else None)
        with self._snapshot(db, vault_id) as (storage, manifest):
            engine = self.retrieval_engine(storage, manifest, modes=(mode,), settings=settings,
                                           rerank=rerank, exact=exact)
            return engine.search(query, mode=mode, top_k=top_k, filters=filters, rerank=rerank)

    def ask(self, query: str, *, db: Path = DEFAULT_DB, vault_id: str = 'default',
            notes_dir: Path | None = None, model: str = DEFAULT_GENERATION_MODEL,
            max_turns: int = 8, think: bool = DEFAULT_AGENT_THINK,
            client: 'Client | None' = None, observer: 'AgentObserver | None' = None,
            search_stall_reminder: bool = True) -> 'AgentResult':
        """Let the existing agent choose tools and search modes over one snapshot.

        Match/read remain live. Ranked capabilities load only if chosen; direct
        replies and live tools work without an index or embedding/vector services.
        """
        from arkb.retrieval.engine import RetrievalEngine

        with self._snapshot(db, vault_id, required=False) as (storage, manifest):
            directory = self._notes_directory(storage, manifest, notes_dir)
            engine = RetrievalEngine(
                bm25=_LazySnapshotRetriever(self, storage, manifest, 'bm25'),
                semantic=_LazySnapshotRetriever(self, storage, manifest, 'semantic'))
            tools = self.agent_tools(engine=engine, directory=directory, vault_id=vault_id,
                                     mode=DEFAULT_RETRIEVAL_MODE)
            return self.run_agent(query, tools=tools, model=model, max_turns=max_turns,
                                  think=think, client=client,observer=observer,
                                  search_stall_reminder=search_stall_reminder)

    def index(self, *, db: Path = DEFAULT_DB, vault_id: str = 'default',
              notes_dir: Path = DEFAULT_NOTES_DIR, qdrant_config: QdrantConfig | None = None,
              force: bool = False, chunking: str = 'recursive', chunk_size: int = 512,
              chunk_overlap: int = 64, context_length: int = 8192, batch_size: int = 32,
              max_batch_tokens: int | None = None, max_retries: int = 2,
              query_instruction: str = DEFAULT_QUERY_INSTRUCTION) -> 'BuildReport':
        """Scan before opening index state, then reuse the existing index builder."""
        self._require_open()
        from arkb.knowledge.documents import scan_notes
        from arkb.knowledge.embeddings import resolve_embedding_spec
        from arkb.knowledge.indexing import build_index

        notes_dir = Path(notes_dir)
        notes = scan_notes(notes_dir)
        tokenizer = self.tokenizer()
        client = self.model_client()
        spec = resolve_embedding_spec(client, self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
                                      context_length=context_length)
        qdrant_config = qdrant_config or QdrantConfig(url=self.config.qdrant_url or QdrantConfig.url)
        with SQLiteStorage(Path(db)) as storage:
            qclient = self.qdrant_client(qdrant_config.url)
            return build_index(storage, notes, spec=spec, vault_id=vault_id, client=client,
                tokenizer=tokenizer, max_input_tokens=context_length, chunking=chunking,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, batch_size=batch_size,
                max_batch_tokens=max_batch_tokens, max_retries=max_retries,
                query_instruction=query_instruction, force=force, source_scope=str(notes_dir.resolve()),
                qdrant_config=qdrant_config, qdrant_client=qclient)

    def status(self, *, db: Path = DEFAULT_DB, vault_id: str = 'default') -> dict:
        """Read saved manifests and backend configuration without contacting services."""
        self._require_open()
        with SQLiteStorage(Path(db), read_only=True) as storage:
            active = storage.active_manifest(vault_id)
            backend = storage.build_metadata(active.index_version)['backend'] if active else {}
            return {'vault_id': vault_id, 'active_version': active.index_version if active else None,
                    'builds': [asdict(m) for m in storage.list_builds(vault_id)],
                    'notes_dir': backend.get('source_scope'),
                    'backend': {key: backend[key] for key in ('kind', 'url', 'collection') if key in backend}}

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
        from arkb.retrieval.qwen_rerank import QwenRerankerScorer
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
            reranker = Reranker(QwenRerankerScorer(max_length=settings.reranker_max_length,
                cache_folder=settings.reranker_cache, local_files_only=self.config.offline))
        return RetrievalEngine(semantic=semantic, bm25=bm25, reranker=reranker,
            candidate_k=settings.candidate_k, rrf_k=settings.rrf_k,
            rerank_candidates=settings.rerank_candidates)

    def agent_tools(self, *, engine: 'RetrievalEngine', directory: Path, vault_id: str,
                    mode: str = 'semantic', rerank: bool = False) -> 'AgentTools':
        """Bind live document tools to an already prepared retrieval engine.

        Use the same directory/vault as indexing. The caller selects a default
        search mode; the agent can choose any mode supported by the engine.
        This composition opens no clients and creates no new resource owners.
        """
        self._require_open()
        from arkb.agent.tools import AgentTools
        from arkb.knowledge.documents import DocumentAccess
        from arkb.retrieval.exact import ExactRetriever

        documents = DocumentAccess(directory, vault_id=vault_id)
        return AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine,
                          mode=mode, rerank=rerank)

    def run_agent(self, query: str, *, tools: 'AgentTools',
                  model: str = DEFAULT_GENERATION_MODEL, max_turns: int = 8,
                  think: bool = DEFAULT_AGENT_THINK,
                  client: 'Client | None' = None, observer: 'AgentObserver | None' = None,
                  search_stall_reminder: bool = True) -> 'AgentResult':
        """Run with prepared tools and a reused model client, or a caller-owned one.

        Compose tools with agent_tools and keep their directory/vault aligned
        with the prepared engine. Each query gets an independent conversation.
        """
        self._require_open()
        if type(think) is not bool:
            raise ValueError('think must be a boolean.')
        from arkb.agent.loop import run_agent
        return run_agent(query, client=self.model_client() if client is None else client,
                         tools=tools, model=model, max_turns=max_turns, think=think,observer=observer,
                         search_stall_reminder=search_stall_reminder)

    def agent_observer(self, *, budget: 'AgentBudget | None' = None) -> 'AgentObserver':
        """Use the pinned embedding tokenizer as a common evidence-text ruler.

        Its fingerprint is recorded; it does not estimate the model chat template.
        Creating an observer can load a local tokenizer, but opens no model client.
        """
        from arkb.agent.observation import AgentObserver
        from arkb.knowledge.embeddings import count_tokens,tokenizer_fingerprint
        tokenizer=self.tokenizer()
        return AgentObserver(budget=budget,counter=lambda text:count_tokens(text,tokenizer=tokenizer),
                             counter_identity='reference-text:'+tokenizer_fingerprint(tokenizer))


class _LazySnapshotRetriever:
    """Prepare one existing capability on first use and retain its captured snapshot."""

    def __init__(self, runtime, storage, manifest, mode):
        self.runtime, self.storage, self.manifest, self.mode = runtime, storage, manifest, mode
        self.engine = None

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        self.runtime._require_open()
        if self.manifest is None:
            raise ValueError('No published index; run the index command first.')
        if self.engine is None:
            self.engine = self.runtime.retrieval_engine(self.storage, self.manifest, modes=(self.mode,))
        return self.engine.search(query, mode=self.mode, top_k=top_k, filters=filters)


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
