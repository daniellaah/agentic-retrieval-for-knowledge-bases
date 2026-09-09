"""Semantic query orchestration, independent of providers and index lifecycle."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from arkb.retrieval.contracts import SearchResponse, SearchResult, validate_request, validate_options
from arkb.schema import EmbeddingSpec


class Embedder(Protocol):
    """Embed a query in an explicitly identified document embedding space."""

    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_query(self, query: str) -> Sequence[float]: ...


class VectorIndex(Protocol):
    """Read one pinned index; adapt backend hits to ranked source evidence.

    Implementations validate vectors and returned identities, apply filters
    before top-k, and define score semantics. Resolve ties by stable identity.
    No database response objects cross this boundary. No writes are required.
    """

    @property
    def spec(self) -> EmbeddingSpec: ...

    @property
    def index_id(self) -> str: ...

    def search(self, vector: Sequence[float], *, top_k: int,
               filters: Mapping[str, str]) -> Sequence[SearchResult]: ...


@dataclass(frozen=True)
class SemanticRetriever:
    """Embed once, search once, and return ranked evidence without generation.

    The provider prepares model-specific query inputs. The index adapter handles
    backend conversion and ordering; scores are never normalized here. Exact
    versus approximate search is explicit adapter configuration. Repeatability
    depends on the pinned index, provider and backend (ANN is not guaranteed).
    """

    embedder: Embedder
    index: VectorIndex

    def __post_init__(self) -> None:
        if not self.embedder.spec.is_compatible_with(self.index.spec):
            raise ValueError('Query embedding model/configuration is incompatible with the stored index; rebuild it.')

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        vector = self.embedder.embed_query(query)
        results = tuple(self.index.search(vector, top_k=top_k, filters=filters))
        if len(results) > top_k:
            raise ValueError('Vector index returned more than top_k results.')
        for result in results:
            if (not isinstance(result, SearchResult) or result.method != 'semantic'
                    or result.score is None
                    or ('source' in filters and result.source != filters['source'])):
                raise ValueError('Vector index returned invalid semantic evidence or filter metadata.')
        return SearchResponse(query=query, method='semantic', results=results, index_id=self.index.index_id)

