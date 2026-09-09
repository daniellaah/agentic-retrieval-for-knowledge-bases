"""Rerank frozen candidates with a replaceable, higher-is-better scorer."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
import re
from typing import Protocol

from arkb.retrieval.models import Retriever, SearchResponse, SearchResult, validate_options, validate_request


class CandidateScorer(Protocol):
    """Return one finite relevance score per candidate, in input order.

    identity must name the model/strategy revision and score-affecting settings.
    Scores must be higher-is-better; distances need an explicit scorer adapter.
    """
    @property
    def identity(self) -> str: ...
    @property
    def score_type(self) -> str: ...
    def score(self, query: str, candidates: Sequence[SearchResult]) -> Sequence[float]: ...


@dataclass(frozen=True)
class Reranker:
    scorer: CandidateScorer

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.scorer.identity, self.scorer.score_type)):
            raise ValueError('Reranker requires scorer identity and score semantics.')

    def rerank(self, query: str, candidates: Sequence[SearchResult], *,
               top_k: int | None = None) -> tuple[SearchResult, ...]:
        validate_request(query, 1 if top_k is None else top_k, None)
        candidates = tuple(candidates)
        if any(not isinstance(hit, SearchResult) for hit in candidates):
            raise ValueError('Reranker requires SearchResult candidates.')
        if len({hit.identity for hit in candidates}) != len(candidates):
            raise ValueError('Reranker candidates contain duplicate identities.')
        if not candidates:
            return ()
        scores = tuple(self.scorer.score(query, candidates))
        if len(scores) != len(candidates) or any(
            type(score) not in (int, float) or not math.isfinite(score) for score in scores
        ):
            raise ValueError('Scorer must return one finite score per candidate.')
        # Stable identity resolves score ties independently of candidate order.
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].identity))
        return tuple(replace(candidates[i], method='reranked', score=scores[i], score_type=self.scorer.score_type,
                             metadata={**candidates[i].metadata, 'rerank': {
                                 'scorer': self.scorer.identity, 'input_rank': i + 1,
                                 'input_method': candidates[i].method, 'input_score': candidates[i].score,
                                 'input_score_type': candidates[i].score_type,
                                 'candidate_count': len(candidates),
                                 'previous': candidates[i].metadata.get('rerank')}})
                     for i in order[:top_k])


@dataclass(frozen=True)
class RerankedRetriever:
    """Optional composition for any retriever; rerank before final truncation."""
    retriever: Retriever
    reranker: Reranker
    candidate_k: int = 20

    def __post_init__(self):
        validate_options(self.candidate_k, None)

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        if top_k > self.candidate_k:
            raise ValueError('top_k cannot exceed reranking candidate_k.')
        response = self.retriever.search(query, top_k=self.candidate_k, filters=dict(filters))
        if (not isinstance(response, SearchResponse) or response.query != query
                or len(response.results) > self.candidate_k
                or any('source' in filters and h.source != filters['source'] for h in response.results)):
            raise ValueError('Retriever returned invalid reranking candidates or filters.')
        results = self.reranker.rerank(query, response.results, top_k=top_k)
        return SearchResponse(query=query, method=response.method + '+rerank',
                              results=results, index_id=response.index_id)


DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'

DEFAULT_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'

class CrossEncoderScorer:
    """Pinned single-logit model, CPU inference, title+body passage pairs.

    The model tokenizer truncates pairs to max_length (including special tokens).
    Returned source evidence stays verbatim and untruncated. CPU inference and
    stable tie handling are reproducible within one software/hardware setup;
    floating point results need not match across library or hardware versions.
    """
    score_type = 'cross_encoder_logit'

    def __init__(self, *, model: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION,
                 max_length: int = 512, batch_size: int = 16,
                 cache_folder: str | None = None, local_files_only: bool = False):
        validate_options(max_length, None)
        validate_options(batch_size, None)
        if not isinstance(model, str) or not model.strip() or not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Cross-encoder requires a model name and pinned 40-character commit revision.')
        try:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity
        except ImportError as error:
            raise ValueError('Install the optional reranker with: uv sync --locked --extra rerank') from error
        self._model = CrossEncoder(model, revision=revision, device='cpu', max_length=max_length,
                                   cache_folder=cache_folder, local_files_only=local_files_only,
                                   trust_remote_code=False)
        if self._model.config.num_labels != 1:
            raise ValueError('Cross-encoder must return one relevance logit per pair.')
        self._activation = Identity()
        self.batch_size = batch_size
        self.identity = f'{model}@{revision}/cpu/max_length={max_length}/batch_size={batch_size}/title-body-v1/raw-logit'

    def score(self, query: str, candidates: Sequence[SearchResult]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, f"{hit.metadata.get('title') or ''}\n\n{hit.content}") for hit in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                     activation_fn=self._activation, apply_softmax=False, convert_to_numpy=True)
        return scores.tolist()
