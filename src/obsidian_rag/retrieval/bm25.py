"""BM25 lexical retrieval using snapshot statistics and explicit score semantics."""

import math

from obsidian_rag.knowledge_base.lexical_index import LexicalIndex, lexical_terms, ANALYZER_ID
from obsidian_rag.knowledge_base.identity import require_text
from .models import SearchResult, SearchResponse, SearchScore, SearchScope, NoteTarget
from .pagination import paginate


def bm25_search(index: LexicalIndex, query: str, *, paths: tuple[str, ...] = (), limit: int = 20,
                cursor: str | None = None, k1: float = 1.2, b: float = .75) -> SearchResponse:
    """Score title+body terms with positive Robertson IDF and BM25 tf saturation.

    Query terms count once. Only matching documents are returned, with path order
    breaking score ties. Filters precede pagination and retain whole-index IDF.
    Note-level candidates require inspection before use as source evidence.
    """
    require_text(query, 'query')
    if index.analyzer_id != ANALYZER_ID:
        raise ValueError('Query analyzer differs from the lexical index; rebuild it.')
    if (type(k1) not in (int, float) or not math.isfinite(k1) or k1 < 0
            or type(b) not in (int, float) or not math.isfinite(b) or not 0 <= b <= 1):
        raise ValueError('BM25 requires finite k1 >= 0 and b in [0, 1].')
    scope = SearchScope(index.snapshot.vault_id, index.snapshot.snapshot_id, paths)
    return paginate(_ranked(index, query, paths, k1, b), query=query, method='bm25', scope=scope,
                    limit=limit, cursor=cursor, parameters={'index': index.fingerprint, 'k1': k1, 'b': b})


def _ranked(index, query, paths, k1, b):
    count = len(index.units)
    average = sum(unit.length for unit in index.units) / count if count else 0
    terms = sorted(set(lexical_terms(query)))
    scored = []
    for unit in index.units:
        if paths and unit.source.path not in paths:
            continue
        score = 0.
        for term in terms:
            frequency = unit.frequencies.get(term, 0)
            if frequency:
                df = index.document_frequency[term]
                idf = math.log1p((count - df + .5) / (df + .5))
                score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * unit.length / average))
        if score > 0:
            scored.append((score, unit))
    for score, unit in sorted(scored, key=lambda pair: (-pair[0], pair[1].source.path,
                               pair[1].record.chunk.start_char if pair[1].record else 0,
                               pair[1].record.chunk_id if pair[1].record else '')):
        metric = SearchScore(score, 'bm25')
        if unit.record is not None:
            yield SearchResult.from_record(unit.record, metric, snapshot_id=index.snapshot.snapshot_id, rank=1, method='bm25')
        else:
            yield SearchResult(unit.source, NoteTarget(), 'bm25', 1, score=metric)
