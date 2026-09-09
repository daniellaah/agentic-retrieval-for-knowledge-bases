"""Backend-independent evidence returned by retrieval methods.

Scores are optional and only comparable within their declared semantics and
retrieval configuration. Result order, not a universal score scale, defines rank.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Protocol

from arkb.schema import ConfigValue, fingerprint_config


@dataclass(frozen=True, kw_only=True)
class SearchResult:
    """One source or verbatim excerpt; no chunk or score is required.

    source_id identifies the document independently of chunks and revisions;
    source is its address (currently a vault-relative path). Optional character
    offsets are end-exclusive coordinates into the loaded source content.
    metadata carries method-specific provenance, such as snapshot and revision.
    score_type must define a supplied score's meaning and ranking direction.
    """

    source_id: str
    source: str
    content: str
    method: str
    metadata: dict[str, ConfigValue] = field(default_factory=dict)
    chunk_id: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    score: float | None = None
    score_type: str | None = None

    @property
    def identity(self) -> tuple[str, ...]:
        """Method-independent deduplication key, scoped by parent document.

        Prefer chunk identity, then a located span, otherwise the whole source.
        Unlocated snippets from one document therefore represent one source hit.
        """
        if self.chunk_id is not None:
            return (self.source_id, 'chunk', self.chunk_id)
        if self.start_char is not None:
            return (self.source_id, 'span', str(self.start_char), str(self.end_char))
        return (self.source_id, 'source')

    def __post_init__(self) -> None:
        for name in ('source_id', 'source', 'method'):
            _text(getattr(self, name), name)
        if not isinstance(self.content, str):
            raise ValueError('content must be a string.')
        if self.chunk_id is not None:
            _text(self.chunk_id, 'chunk_id')
        if self.start_char is not None or self.end_char is not None:
            if (type(self.start_char) is not int or type(self.end_char) is not int
                    or not 0 <= self.start_char <= self.end_char
                    or self.end_char - self.start_char != len(self.content)):
                raise ValueError('Source span must match the verbatim content length.')
        if self.score is None:
            if self.score_type is not None:
                raise ValueError('score_type requires a score.')
        else:
            _text(self.score_type, 'score_type')
            if type(self.score) not in (int, float) or not math.isfinite(self.score):
                raise ValueError('score must be finite.')
        fingerprint_config(self.metadata)
        object.__setattr__(self, 'metadata', deepcopy(self.metadata))


@dataclass(frozen=True, kw_only=True)
class SearchResponse:
    """Ranked results for the original query; an empty tuple means no matches.

    Rank is the one-based position in results, never a stale per-hit field.
    index_id identifies the pinned index when the method uses one. Failures
    raise exceptions rather than masquerading as successful empty responses.
    """

    query: str
    method: str
    results: tuple[SearchResult, ...] = ()
    index_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.query, 'query')
        _text(self.method, 'method')
        if self.index_id is not None:
            _text(self.index_id, 'index_id')
        if not isinstance(self.results, tuple) or any(not isinstance(r, SearchResult) for r in self.results):
            raise ValueError('results must be a ranked tuple of SearchResult objects.')


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonblank text.')


class Retriever(Protocol):
    """One explicit deterministic strategy; filtering happens before top_k."""

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse: ...


def validate_request(query: str, top_k: int, filters: Mapping[str, str] | None) -> dict[str, str]:
    """Validate the original query and explicit retrieval parameters."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be nonblank text.')
    return validate_options(top_k, filters)


def validate_options(top_k: int, filters: Mapping[str, str] | None) -> dict[str, str]:
    """Only source equality is supported today; never silently ignore filters."""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError('top_k must be a positive integer.')
    if filters is None:
        return {}
    if not isinstance(filters, Mapping) or set(filters) - {'source'}:
        raise ValueError('filters supports only source equality.')
    if any(not isinstance(value, str) or not value.strip() for value in filters.values()):
        raise ValueError('source filter must be nonblank text.')
    return dict(filters)
