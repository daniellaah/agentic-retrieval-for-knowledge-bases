"""Reciprocal rank fusion of arbitrary evidence rankings, with no retrievers.

Each list contributes 1/(k + one-based rank). A duplicate within one list votes
only at its first position; later duplicates still occupy their original slots.
Ties sort by stable result identity. Raw scores never affect the fused ranking.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math

from arkb.retrieval.contracts import SearchResult, validate_options


def rrf(ranked_lists: Mapping[str, Sequence[SearchResult]] | Sequence[Sequence[SearchResult]],
        *, k: float = 60, top_k: int | None = None) -> tuple[SearchResult, ...]:
    if type(k) not in (int, float) or not math.isfinite(k) or k < 0:
        raise ValueError('RRF k must be nonnegative and finite.')
    if top_k is not None:
        validate_options(top_k, None)
    lists = ranked_lists if isinstance(ranked_lists, Mapping) else {
        str(i): values for i, values in enumerate(ranked_lists)}
    if any(not isinstance(name, str) or not name.strip() for name in lists):
        raise ValueError('RRF list names must be nonblank strings.')
    evidence, contributions = {}, {}
    for name in sorted(lists):
        seen = set()
        for rank, hit in enumerate(lists[name], 1):
            if not isinstance(hit, SearchResult):
                raise ValueError('RRF requires SearchResult evidence.')
            key = hit.identity
            previous = evidence.setdefault(key, hit)
            if ((previous.source, previous.content, previous.start_char, previous.end_char) !=
                    (hit.source, hit.content, hit.start_char, hit.end_char)
                    or any(previous.metadata.get(field) != hit.metadata.get(field)
                           for field in ('index_version', 'document_revision', 'vault_id')
                           if previous.metadata.get(field) is not None and hit.metadata.get(field) is not None)):
                raise ValueError('RRF received conflicting evidence for one identity.')
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
