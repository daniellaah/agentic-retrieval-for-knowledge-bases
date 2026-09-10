"""Fixed BM25 + semantic candidate retrieval followed by rank fusion."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

from arkb.retrieval.models import Retriever, SearchResponse, SearchResult, validate_options, validate_request


@dataclass(frozen=True)
class HybridRetriever:
    bm25: Retriever
    semantic: Retriever
    candidate_k: int = 20
    rrf_k: float = 60

    def __post_init__(self):
        validate_options(self.candidate_k, None)
        rrf([], k=self.rrf_k)

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        if top_k > self.candidate_k:
            raise ValueError('top_k cannot exceed hybrid candidate_k; configure a larger candidate pool.')
        responses = {}
        for name, retriever in (('bm25', self.bm25), ('semantic', self.semantic)):
            response = retriever.search(query, top_k=self.candidate_k, filters=dict(filters))
            if (not isinstance(response, SearchResponse) or response.query != query
                    or len(response.results) > self.candidate_k
                    or any('source' in filters and h.source != filters['source'] for h in response.results)):
                raise ValueError('Hybrid retriever returned invalid candidates or filters.')
            responses[name] = response
        if responses['bm25'].index_id != responses['semantic'].index_id:
            raise ValueError('Hybrid retrieval requires the same pinned snapshot for both retrievers.')
        fused = rrf({name: response.results for name, response in responses.items()},
                    k=self.rrf_k, top_k=top_k)
        hits = tuple(replace(hit, method='hybrid', metadata={**hit.metadata, 'hybrid': {
            'candidate_k': self.candidate_k, 'rrf_k': self.rrf_k}}) for hit in fused)
        return SearchResponse(query=query, method='hybrid', results=hits,
                              index_id=responses['bm25'].index_id)


def rrf(ranked_lists: Mapping[str, Sequence[SearchResult]] | Sequence[Sequence[SearchResult]],
        *, k: float = 60, top_k: int | None = None) -> tuple[SearchResult, ...]:
    """Fuse ranks with one vote per list/identity and stable identity tie breaking.

    Raw scores do not affect fusion. Preserve the public named/unnamed input
    forms and diagnostic output while keeping the implementation with hybrid.
    """
    if type(k) not in (int, float) or not math.isfinite(k) or k < 0:
        raise ValueError('RRF k must be nonnegative and finite.')
    if top_k is not None:
        validate_options(top_k, None)
    lists = ranked_lists if isinstance(ranked_lists, Mapping) else {
        str(i): values for i, values in enumerate(ranked_lists)}
    if any(not isinstance(name, str) or not name.strip() for name in lists):
        raise ValueError('RRF list names must be nonblank strings.')
    evidence, contributions, provenance = {}, {}, {}
    for name in sorted(lists):
        seen = set()
        for rank, hit in enumerate(lists[name], 1):
            if not isinstance(hit, SearchResult):
                raise ValueError('RRF requires SearchResult evidence.')
            key = hit.identity
            previous = evidence.setdefault(key, hit)
            if ((previous.source, previous.content, previous.start_char, previous.end_char) !=
                    (hit.source, hit.content, hit.start_char, hit.end_char)):
                raise ValueError('RRF received conflicting evidence for one identity.')
            known = provenance.setdefault(key, {})
            for field in ('index_version', 'document_revision', 'vault_id'):
                value = hit.metadata.get(field)
                if value is not None and known.setdefault(field, value) != value:
                    raise ValueError('RRF received conflicting provenance for one identity.')
            if key in seen:
                continue
            seen.add(key)
            contributions.setdefault(key, []).append({
                'list': name, 'rank': rank, 'method': hit.method,
                'score': hit.score, 'score_type': hit.score_type, 'metadata': hit.metadata})
    scores = {key: math.fsum(1 / (k + vote['rank']) for vote in votes)
              for key, votes in contributions.items()}
    keys = sorted(evidence, key=lambda key: (-scores[key], key))
    return tuple(replace(evidence[key], method='rrf', score=scores[key], score_type='rrf',
                         metadata={**evidence[key].metadata, 'fusion': {
                             'algorithm': 'rrf', 'k': k, 'contributions': contributions[key]}})
                 for key in keys[:top_k])
