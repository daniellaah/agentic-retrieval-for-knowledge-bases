"""Explicit application settings; importing configuration opens no resources."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
DEFAULT_GENERATION_MODEL = 'qwen3.5:4b'
DEFAULT_AGENT_THINK = True
DEFAULT_RETRIEVAL_MODE = 'semantic'
DEFAULT_DB = Path('.obsidian-rag/index.sqlite')
DEFAULT_NOTES_DIR = Path('example_notes')


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    host: str = 'http://127.0.0.1:11434'
    timeout: float = 180.0
    tokenizer_cache: Path | None = None
    offline: bool = False
    embedding_model: str | None = None
    qdrant_url: str | None = None
    qdrant_timeout: float | None = None


@dataclass(frozen=True, kw_only=True)
class RetrievalConfig:
    candidate_k: int = 20
    rrf_k: float = 60
    rerank_candidates: int = 20
    reranker_max_length: int = 512
    reranker_cache: str | None = None
    bm25_k1: float = 1.2
    bm25_b: float = .75
