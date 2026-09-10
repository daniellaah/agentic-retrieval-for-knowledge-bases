"""Three agent-facing primitives; capability objects are supplied by the runtime."""

from copy import deepcopy
from typing import TypedDict

from arkb.knowledge.documents import DocumentAccess
from arkb.knowledge.models import ConfigValue
from arkb.retrieval.engine import RetrievalEngine
from arkb.retrieval.exact import ExactRetriever
from arkb.retrieval.models import SearchResponse, SearchResult, chunk_result, validate_request


class Evidence(TypedDict):
    document_id: str
    source: str
    title: str | None
    content: str
    document_revision: str | None
    chunk_id: str | None
    section_id: str | None
    start_char: int | None
    end_char: int | None


class QueryResult(TypedDict):
    query: str
    results: list[Evidence]


class ReadResult(TypedDict):
    result: Evidence


def _evidence(result: SearchResult) -> Evidence:
    def text(key: str) -> str | None:
        value = result.metadata.get(key)
        return value if isinstance(value, str) else None

    return Evidence(document_id=result.source_id, source=result.source, title=text('title'),
                    content=result.content, document_revision=text('document_revision'),
                    chunk_id=result.chunk_id, section_id=text('section_id'),
                    start_char=result.start_char, end_char=result.end_char)


def _query_result(response: SearchResponse) -> QueryResult:
    return QueryResult(query=response.query, results=[_evidence(hit) for hit in response.results])


def tool_definitions(modes: tuple[str, ...], *, default_mode: str) -> tuple[dict[str, ConfigValue], ...]:
    """Render the effective tool schemas without constructing capability objects."""
    definitions = deepcopy(TOOL_DEFINITIONS)
    search = next(definition for definition in definitions if definition['name'] == 'search')
    search['parameters']['properties']['mode']['enum'] = [*modes, None]
    meanings = {'bm25': 'keyword ranking', 'semantic': 'meaning', 'hybrid': 'keywords and meaning'}
    strategies = ', '.join(f'{mode} ({meanings[mode]})' for mode in modes)
    search['parameters']['properties']['mode']['description'] = (
        f'Available strategies: {strategies or "none"}; omitted/null uses {default_mode}.')
    return definitions


class AgentTools:
    """Adapt injected capabilities without owning resources or making retrieval decisions.

    mode is the default strategy; an agent may override it on each search.
    Reranking remains an application composition setting.
    The existing engine remains responsible for all retrieval strategy execution.
    """

    def __init__(self, *, documents: DocumentAccess, exact: ExactRetriever,
                 engine: RetrievalEngine, mode: str = 'semantic', rerank: bool = False):
        if mode not in ('semantic', 'bm25', 'lexical', 'hybrid'):
            raise ValueError('Unknown configured retrieval mode.')
        if type(rerank) is not bool:
            raise ValueError('rerank must be a boolean.')
        self._documents = documents
        self._exact = exact
        self._engine = engine
        self._mode = mode
        self._rerank = rerank

    def tool_definitions(self) -> tuple[dict[str, ConfigValue], ...]:
        """Describe only search modes supported by the composed engine.

        This advertises capabilities, without choosing a strategy for the agent.
        The shared catalog remains unchanged for other compositions.
        """
        modes = []
        if self._engine.bm25 is not None:
            modes.append('bm25')
        if self._engine.semantic is not None:
            modes.append('semantic')
        if len(modes) == 2:
            modes.append('hybrid')
        return tool_definitions(tuple(modes), default_mode=self._mode)

    def match(self, query: str, *, target: str = 'content', regex: bool = False,
              case_sensitive: bool = True, source: str | None = None, limit: int = 5) -> QueryResult:
        """Use when you know an exact word, phrase, symbol, filename, or text pattern."""
        return _query_result(self._exact.search(
            query, target=target, regex=regex, case_sensitive=case_sensitive,
            filters={'source': source} if source is not None else None, top_k=limit,
        ))

    def search(self, query: str, *, source: str | None = None, limit: int = 5,
               mode: str | None = None) -> QueryResult:
        """Use to discover relevant knowledge about a question, topic, or concept
        when you do not know the document's exact wording.
        """
        filters = validate_request(query, limit, {'source': source} if source is not None else None)
        if mode is not None and mode not in ('bm25', 'semantic', 'hybrid'):
            raise ValueError('mode must be bm25, semantic or hybrid.')
        return _query_result(self._engine.search(query, mode=self._mode if mode is None else mode,
                                                rerank=self._rerank,
                                                filters=filters, top_k=limit))

    def read(self, document_id: str | None = None, *, source: str | None = None,
             section_id: str | None = None,
             start_char: int | None = None, end_char: int | None = None) -> ReadResult:
        """Read by returned document ID or known source filename. Both selectors
        must agree when supplied together. Optionally select a section or range.
        """
        record = self._documents.read(document_id, source=source, section_id=section_id,
                                      start_char=start_char, end_char=end_char)
        evidence = _evidence(chunk_result(record, method='read'))
        # A live source slice is not an indexed chunk. Only claim a selected section.
        evidence['chunk_id'] = None
        evidence['section_id'] = section_id
        return ReadResult(result=evidence)


# Plain JSON schemas, independent of any model provider, framework, or dispatcher.
TOOL_DEFINITIONS: tuple[dict[str, ConfigValue], ...] = (
    {
        'name': 'match',
        'description': 'Use when you know an exact word, phrase, symbol, filename, or text '
                       'pattern. Returns literal occurrences by default; enable regex for '
                       'patterns. Use target=source for filenames instead of document bodies.',
        'parameters': {
            'type': 'object', 'required': ['query'], 'additionalProperties': False,
            'properties': {
                'query': {'type': 'string', 'minLength': 1},
                'target': {'type': 'string', 'enum': ['content', 'source'], 'default': 'content'},
                'regex': {'type': 'boolean', 'default': False},
                'case_sensitive': {'type': 'boolean', 'default': True},
                'source': {'type': ['string', 'null'], 'minLength': 1,
                           'description': 'Restrict to this exact knowledge-relative source path.'},
                'limit': {'type': 'integer', 'minimum': 1, 'default': 5},
            },
        },
    },
    {
        'name': 'search',
        'description': 'Use to discover relevant knowledge about a question, topic, or '
                       'concept when you do not know the exact wording. Results are ordered '
                       'by relevance. Choose an available strategy from the mode parameter, '
                       'or omit it for the default. Use read with a returned document_id to expand context.',
        'parameters': {
            'type': 'object', 'required': ['query'], 'additionalProperties': False,
            'properties': {
                'query': {'type': 'string', 'minLength': 1},
                'mode': {'type': ['string', 'null'], 'enum': ['bm25', 'semantic', 'hybrid', None],
                         'description': 'Retrieval strategy for this call; omitted/null uses the configured default.'},
                'source': {'type': ['string', 'null'], 'minLength': 1,
                           'description': 'Restrict to this exact knowledge-relative source path.'},
                'limit': {'type': 'integer', 'minimum': 1, 'default': 5},
            },
        },
    },
    {
        'name': 'read',
        'description': 'Read current original text or expand context. Supply '
                       'document_id from a result or source for a known filename. '
                       'If both are supplied, they must identify the same document. '
                       'Select a section or character range, not both. '
                       'Ranges address the current body: zero-based, end-exclusive; omitted '
                       'endpoints use document boundaries. Search locations can become stale '
                       'after files are edited.',
        'parameters': {
            'type': 'object', 'anyOf': [{'required': ['document_id']}, {'required': ['source']}],
            'additionalProperties': False,
            'properties': {
                'document_id': {'type': 'string', 'pattern': '^[0-9a-f]{64}$',
                                'description': 'Use a document ID returned by match or search.'},
                'source': {'type': 'string', 'minLength': 1,
                           'description': 'Exact knowledge-relative filename, e.g. rag.md.'},
                'section_id': {'type': ['string', 'null'], 'pattern': '^[0-9a-f]{64}$'},
                'start_char': {'type': ['integer', 'null'], 'minimum': 0},
                'end_char': {'type': ['integer', 'null'], 'minimum': 0},
            },
        },
    },
)
