"""Retrieval-independent search contracts."""

from dataclasses import asdict, dataclass, field
import json
import math

from obsidian_rag.knowledge_base.sources import SourceRef, SourceExcerpt
from obsidian_rag.knowledge_base.models import ChunkRecord
from obsidian_rag.knowledge_base.identity import require_digest


@dataclass(frozen=True)
class NoteTarget:
    kind: str = field(default='note', init=False)


@dataclass(frozen=True)
class SpanTarget:
    start_char: int
    end_char: int
    kind: str = field(default='span', init=False)

    def __post_init__(self):
        if type(self.start_char) is not int or type(self.end_char) is not int or not 0 <= self.start_char <= self.end_char:
            raise ValueError('Target requires a valid character span.')


@dataclass(frozen=True)
class ChunkTarget:
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    kind: str = field(default='chunk', init=False)

    def __post_init__(self):
        require_digest(self.chunk_id, 'chunk_id')
        SpanTarget(self.start_char, self.end_char)
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError('Chunk index must be nonnegative.')


@dataclass(frozen=True)
class SearchScore:
    value: float
    metric: str
    higher_is_better: bool = True

    def __post_init__(self):
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError('Search score must be finite.')
        if not isinstance(self.metric, str) or not self.metric.strip() or type(self.higher_is_better) is not bool:
            raise ValueError('Score requires a metric and sort direction.')
        if self.metric == 'cosine' and (not -1 <= self.value <= 1 or not self.higher_is_better):
            raise ValueError('Cosine similarity must be in [-1, 1] and sorted descending.')


@dataclass(frozen=True)
class SearchResult:
    source: SourceRef
    target: NoteTarget | SpanTarget | ChunkTarget
    method: str
    rank: int
    excerpts: tuple[SourceExcerpt, ...] = ()
    score: SearchScore | None = None

    def __post_init__(self):
        if not isinstance(self.source, SourceRef) or not isinstance(self.target, (NoteTarget, SpanTarget, ChunkTarget)):
            raise ValueError('Result requires a source and a supported target.')
        if not isinstance(self.method, str) or not self.method.strip() or type(self.rank) is not int or self.rank < 1:
            raise ValueError('Result requires a method and positive rank.')
        if self.score is not None and not isinstance(self.score, SearchScore):
            raise ValueError('Result score must declare its metric.')
        if not isinstance(self.excerpts, tuple):
            raise ValueError('Result excerpts must be immutable.')
        for excerpt in self.excerpts:
            if not isinstance(excerpt, SourceExcerpt) or excerpt.source != self.source:
                raise ValueError('Excerpt must belong to the result source.')
            if isinstance(self.target, (SpanTarget, ChunkTarget)) and not self.target.start_char <= excerpt.start_char <= excerpt.end_char <= self.target.end_char:
                raise ValueError('Excerpt lies outside the result target.')


    @classmethod
    def from_record(cls, record: ChunkRecord, score: SearchScore | None, *,
                    snapshot_id: str, rank: int, method: str):
        chunk = record.chunk
        source = SourceRef(record.vault_id, record.document_id, chunk.source, chunk.title,
                           record.document_revision, snapshot_id)
        target = ChunkTarget(record.chunk_id, chunk.chunk_index, chunk.start_char, chunk.end_char)
        excerpt = SourceExcerpt(source, chunk.start_char, chunk.end_char, chunk.content)
        return cls(source, target, method, rank, (excerpt,), score)


@dataclass(frozen=True)
class SearchScope:
    vault_id: str
    snapshot_id: str
    paths: tuple[str, ...] = ()

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.vault_id, self.snapshot_id)):
            raise ValueError('Scope requires a vault and snapshot.')
        if not isinstance(self.paths, tuple) or any(not isinstance(p, str) or not p for p in self.paths):
            raise ValueError('Scope paths must be an immutable tuple.')


@dataclass(frozen=True)
class SearchResponse:
    query: str
    method: str
    scope: SearchScope
    items: tuple[SearchResult, ...]
    limit: int
    has_more: bool | None = None
    next_cursor: str | None = None

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.query, self.method)):
            raise ValueError('Search requires a query and method.')
        if not isinstance(self.scope, SearchScope) or type(self.limit) is not int or self.limit < 1:
            raise ValueError('Search requires a scope and positive limit.')
        if not isinstance(self.items, tuple) or len(self.items) > self.limit:
            raise ValueError('Search items must be immutable and within the limit.')
        if self.has_more is not None and type(self.has_more) is not bool:
            raise ValueError('has_more must be true, false or unknown.')
        if self.next_cursor is not None and (not isinstance(self.next_cursor, str) or not self.next_cursor or self.has_more is not True):
            raise ValueError('A continuation cursor requires more results.')
        seen = set()
        for rank, item in enumerate(self.items, 1):
            if not isinstance(item, SearchResult) or item.rank != rank or item.method != self.method:
                raise ValueError('Search items must preserve method and contiguous ranks.')
            if (item.source.vault_id != self.scope.vault_id or item.source.snapshot_id != self.scope.snapshot_id
                    or (self.scope.paths and item.source.path not in self.scope.paths)):
                raise ValueError('Result source lies outside the declared search scope.')
            key = (item.source, item.target)
            if key in seen:
                raise ValueError('Search returned duplicate targets.')
            seen.add(key)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), ensure_ascii=False, allow_nan=False))
