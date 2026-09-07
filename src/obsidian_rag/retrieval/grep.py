"""Literal body search returning matching lines with original source coordinates."""

import re

from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot, SourceExcerpt
from .models import SearchResult, SearchResponse, SearchScope, SpanTarget
from .pagination import paginate


def grep_search(snapshot: KnowledgeSnapshot, query: str, *, case_sensitive: bool = True,
                paths: tuple[str, ...] = (), limit: int = 20, cursor: str | None = None) -> SearchResponse:
    """Return lines containing literal matches, in path/position order.

    Multiple occurrences on a matching line yield one result. A multiline query
    returns all intersected lines. Scores are absent; has_more is exhaustive for
    the declared immutable snapshot/scope. Ranks start at one within each page.
    Matching never normalizes or rewrites source text and requires no chunks.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Search query must be nonblank.')
    if type(case_sensitive) is not bool:
        raise ValueError('case_sensitive must be boolean.')
    scope = SearchScope(snapshot.vault_id, snapshot.snapshot_id, paths)
    pattern = re.compile(re.escape(query), flags=0 if case_sensitive else re.IGNORECASE)
    return paginate(_matching_lines(snapshot, pattern, paths), query=query, method='grep', scope=scope,
                    limit=limit, cursor=cursor, parameters={'case_sensitive': case_sensitive})


def _matching_lines(snapshot, pattern, paths):
    for note, source in sorted(zip(snapshot.notes, snapshot.note_refs()), key=lambda pair: pair[1].path):
        if paths and source.path not in paths:
            continue
        text = note.content
        seen = set()
        for match in pattern.finditer(text):
            start = text.rfind('\n', 0, match.start()) + 1
            newline = text.find('\n', match.end() - 1)
            end = len(text) if newline == -1 else newline + 1
            if (start, end) not in seen:
                seen.add((start, end))
                excerpt = SourceExcerpt(source, start, end, text[start:end])
                yield SearchResult(source, SpanTarget(start, end), 'grep', 1, (excerpt,))
