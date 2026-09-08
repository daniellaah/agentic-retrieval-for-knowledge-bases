"""Discover notes by explicit metadata filters without embeddings or fabricated scores."""

from dataclasses import asdict, dataclass
import json

from obsidian_rag.knowledge_base.metadata import read_metadata
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot
from .models import NoteTarget, SearchResult, SearchResponse, SearchScope
from .pagination import paginate


@dataclass(frozen=True)
class MetadataQuery:
    title_contains: str | None = None
    path_prefix: str | None = None
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (self.title_contains, self.path_prefix):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError('Metadata text filters must be nonblank strings.')
        for values in (self.tags, self.aliases):
            if not isinstance(values, tuple) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError('Metadata value filters must be immutable nonblank strings.')


def metadata_search(snapshot: KnowledgeSnapshot, query: MetadataQuery, *, paths: tuple[str, ...] = (),
                    limit: int = 20, cursor: str | None = None) -> SearchResponse:
    """AND all filters; title/tags/aliases ignore case, path prefix is literal.

    Empty filters enumerate the declared scope. Results have no excerpts: use
    read_note/read_span to inspect discovered sources before building context.
    Only YAML frontmatter tags/aliases are properties; inline hashtags are text.
    """
    if not isinstance(query, MetadataQuery):
        raise ValueError('Expected MetadataQuery.')
    scope = SearchScope(snapshot.vault_id, snapshot.snapshot_id, paths)
    return paginate(_matches(snapshot, query, paths),
                    query=json.dumps(asdict(query), sort_keys=True, ensure_ascii=False), method='metadata',
                    scope=scope, limit=limit, cursor=cursor, parameters={})


def _matches(snapshot, query, paths):
    for source in sorted(snapshot.note_refs(), key=lambda ref: ref.path):
        if paths and source.path not in paths:
            continue
        if query.path_prefix is not None and not source.path.startswith(query.path_prefix):
            continue
        if query.title_contains is not None and query.title_contains.casefold() not in source.title.casefold():
            continue
        metadata = read_metadata(snapshot, source)
        if not {tag.casefold() for tag in query.tags} <= {tag.casefold() for tag in metadata.tags}:
            continue
        if not {alias.casefold() for alias in query.aliases} <= {alias.casefold() for alias in metadata.aliases}:
            continue
        yield SearchResult(source, NoteTarget(), 'metadata', 1)
