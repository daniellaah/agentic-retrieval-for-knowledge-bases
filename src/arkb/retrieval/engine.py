"""Explicit retrieval mode selection; no routing, planning or hidden fallbacks."""

from collections.abc import Mapping
from dataclasses import dataclass

from arkb.retrieval.contracts import Retriever, SearchResponse, validate_request
from arkb.retrieval.hybrid import HybridRetriever
from arkb.retrieval.reranker import Reranker, RerankedRetriever


@dataclass(frozen=True, kw_only=True)
class RetrievalEngine:
    semantic: Retriever | None = None
    bm25: Retriever | None = None
    candidate_k: int = 20
    rrf_k: float = 60
    reranker: Reranker | None = None
    rerank_candidates: int = 20

    def search(self, query: str, *, mode: str = 'semantic', top_k: int = 2,
               filters: Mapping[str, str] | None = None, rerank: bool = False) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        if type(rerank) is not bool:
            raise ValueError('rerank must be a boolean.')
        if mode == 'semantic':
            retriever = self.semantic
        elif mode in ('bm25', 'lexical'):
            retriever = self.bm25
        elif mode == 'hybrid':
            if self.bm25 is None or self.semantic is None:
                raise ValueError('Hybrid mode requires both BM25 and semantic retrievers.')
            retriever = HybridRetriever(self.bm25, self.semantic, self.candidate_k, self.rrf_k)
        else:
            raise ValueError(f'Unknown retrieval mode: {mode}')
        if retriever is None:
            raise ValueError(f'Retrieval mode is not configured: {mode}')
        if rerank:
            if self.reranker is None:
                raise ValueError('Reranking was requested but no reranker is configured.')
            retriever = RerankedRetriever(retriever, self.reranker, self.rerank_candidates)
        return retriever.search(query, top_k=top_k, filters=filters)
