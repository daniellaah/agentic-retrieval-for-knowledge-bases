"""Public deterministic retrieval primitives and explicit composition."""

from arkb.retrieval.bm25 import BM25Retriever
from arkb.retrieval.contracts import Retriever, SearchResponse, SearchResult
from arkb.retrieval.engine import RetrievalEngine
from arkb.retrieval.fusion import rrf
from arkb.retrieval.hybrid import HybridRetriever
from arkb.retrieval.reranker import CandidateScorer, RerankedRetriever, Reranker
from arkb.retrieval.semantic import Embedder, SemanticRetriever, VectorIndex

__all__ = [
    'SearchResult', 'SearchResponse', 'Retriever', 'Embedder', 'VectorIndex',
    'SemanticRetriever', 'BM25Retriever', 'rrf', 'HybridRetriever',
    'CandidateScorer', 'Reranker', 'RerankedRetriever', 'RetrievalEngine',
]
