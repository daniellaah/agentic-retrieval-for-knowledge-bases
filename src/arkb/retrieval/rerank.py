"""Rerank frozen candidates with a replaceable, higher-is-better scorer."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Protocol

from arkb.retrieval.models import (
    Retriever, SearchResponse, SearchResult, validate_options, validate_request,
)


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
        results = self.reranker.rerank(query, response.results, top_k=top_k)
        return SearchResponse(query=query, method=response.method + '+rerank',
                              results=results, index_id=response.index_id)
