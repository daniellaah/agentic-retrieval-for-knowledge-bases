"""Fixed BM25 + semantic candidate retrieval followed by rank fusion."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from arkb.retrieval.models import Retriever, SearchResponse, validate_options, validate_request
from arkb.retrieval.fusion import rrf


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
