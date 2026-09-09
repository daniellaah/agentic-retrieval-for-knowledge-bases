"""Shared generation evidence, budget and citation data; no model or backend calls."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import math
import re
from typing import Literal
from arkb.retrieval.models import SearchResult


@dataclass(frozen=True)
class ContextConfig:
    context_window: int = 8192
    max_output_tokens: int = 1024
    safety_margin: int = 128

    def __post_init__(self) -> None:
        for name, minimum in (('context_window', 1), ('max_output_tokens', 1), ('safety_margin', 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if self.input_budget <= 0:
            raise ValueError('Output reserve and safety margin leave no input budget.')

    @property
    def input_budget(self) -> int:
        return self.context_window - self.max_output_tokens - self.safety_margin


@dataclass(frozen=True)
class GenerationCounter:
    """An explicitly identified model/message counter; estimates are labeled.

    count_messages must include the serving chat template and assistant prefix.
    The caller is responsible for an adapter's model/template fidelity.
    The built-in model adapter lives in arkb.generation.
    """

    model: str
    identity: str
    count_messages: Callable[[Sequence[dict[str, str]]], int]
    is_estimate: bool = False
    context_limit: int | None = None

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.identity.strip() or not callable(self.count_messages):
            raise ValueError('Counter requires a model, identity and callable.')
        if type(self.is_estimate) is not bool:
            raise ValueError('is_estimate must be boolean.')
        if self.context_limit is not None and (type(self.context_limit) is not int or self.context_limit <= 0):
            raise ValueError('context_limit must be positive.')

    def __call__(self, messages: Sequence[dict[str, str]]) -> int:
        count = self.count_messages([dict(m) for m in messages])
        if type(count) is not int or count <= 0:
            raise ValueError('Message counter must return a positive integer.')
        return count


@dataclass(frozen=True)
class EvidenceBlock:
    """Verbatim text in Note.content coordinates, with all contributing hits.

    Origins retain chunk IDs, document revisions, snapshot versions and scores.
    """

    content: str
    title: str
    source: str
    start_char: int
    end_char: int
    origins: tuple[SearchResult, ...]

    def __post_init__(self) -> None:
        if (type(self.start_char) is not int or type(self.end_char) is not int
                or self.start_char < 0 or self.end_char < self.start_char
                or self.end_char - self.start_char != len(self.content)):
            raise ValueError('Evidence span must match its content length.')
        if not self.origins:
            raise ValueError('Evidence must retain its source origins.')
        if len({_document_key(hit) for hit in self.origins}) != 1:
            raise ValueError('Merged evidence requires one known document revision and snapshot.')
        for hit in self.origins:
            if (hit.source != self.source or hit.metadata['title'] != self.title
                    or hit.start_char < self.start_char or hit.end_char > self.end_char
                    or self.content[hit.start_char - self.start_char:hit.end_char - self.start_char]
                    != hit.content):
                raise ValueError('Evidence must contain the verbatim source spans.')


def _document_key(hit: SearchResult) -> tuple:
    return (hit.metadata['index_version'], hit.source_id, hit.metadata['document_revision'])

def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonblank text.')


def _source_id(value):
    if not isinstance(value, str) or re.fullmatch(r'S[1-9][0-9]*', value) is None:
        raise ValueError('Source IDs must have the form S1, S2, ...')


def _span(start, end):
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('Source span requires nonnegative start and exclusive end > start.')


@dataclass(frozen=True)
class CitationOrigin:
    start_char: int
    end_char: int
    score: float
    chunk_id: str
    document_id: str
    document_revision: str
    vault_id: str
    index_version: str
    score_type: str = 'cosine_similarity'
    method: str = 'semantic'

    def __post_init__(self):
        _span(self.start_char, self.end_char)
        if type(self.score) not in (int, float) or not math.isfinite(self.score):
            raise ValueError('Origin score must be finite.')
        _text(self.score_type, 'Origin score_type')
        _text(self.method, 'Origin method')
        if self.score_type == 'cosine_similarity' and not -1 <= self.score <= 1:
            raise ValueError('Origin cosine score must be in [-1, 1].')
        for value in (self.chunk_id, self.document_id, self.document_revision, self.vault_id, self.index_version):
            _text(value, 'Origin identity')


@dataclass(frozen=True)
class CitationSource:
    source_id: str
    source: str
    title: str
    content: str
    start_char: int
    end_char: int
    origins: tuple[CitationOrigin, ...]

    def __post_init__(self):
        _source_id(self.source_id)
        _text(self.source, 'source')
        if not isinstance(self.title, str):
            raise ValueError('title must be text.')
        _text(self.content, 'content')
        _span(self.start_char, self.end_char)
        if self.end_char - self.start_char != len(self.content):
            raise ValueError('Source span must match content length.')
        if not isinstance(self.origins, tuple) or not self.origins:
            raise ValueError('Source requires immutable origins.')
        for origin in self.origins:
            if not isinstance(origin, CitationOrigin) or not (
                self.start_char <= origin.start_char < origin.end_char <= self.end_char
            ):
                raise ValueError('Origin must lie within source span.')
        if len(self.origins) > 1:
            keys = {(o.vault_id, o.document_id, o.document_revision, o.index_version) for o in self.origins}
            if len(keys) != 1:
                raise ValueError('Merged sources require one known document revision and snapshot.')


@dataclass(frozen=True)
class CitationQuote:
    source_id: str
    text: str

    def __post_init__(self):
        _source_id(self.source_id)
        _text(self.text, 'Quote text')


@dataclass(frozen=True)
class Claim:
    """Model-proposed fact; empty references remain inspectable validation failures."""

    text: str
    source_ids: tuple[str, ...]
    quotes: tuple[CitationQuote, ...] = ()

    def __post_init__(self):
        _text(self.text, 'Claim text')
        if not isinstance(self.source_ids, tuple):
            raise ValueError('source_ids must be an immutable tuple.')
        for source_id in self.source_ids:
            _source_id(source_id)
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError('Duplicate source IDs in one claim.')
        if not isinstance(self.quotes, tuple) or any(not isinstance(q, CitationQuote) for q in self.quotes):
            raise ValueError('quotes must be an immutable tuple of CitationQuote objects.')
        if len(set(self.quotes)) != len(self.quotes):
            raise ValueError('Duplicate quotes in one claim.')


@dataclass(frozen=True)
class CitedAnswer:
    status: Literal['answered', 'partial', 'insufficient_evidence']
    claims: tuple[Claim, ...]
    missing_information: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in ('answered', 'partial', 'insufficient_evidence'):
            raise ValueError('Unknown answer status.')
        if not isinstance(self.claims, tuple) or any(not isinstance(c, Claim) for c in self.claims):
            raise ValueError('claims must be an immutable tuple of Claim objects.')
        if not isinstance(self.missing_information, tuple):
            raise ValueError('missing_information must be an immutable tuple.')
        for item in self.missing_information:
            _text(item, 'Missing information')
        expected = {'answered': (True, False), 'partial': (True, True),
                    'insufficient_evidence': (False, True)}[self.status]
        if (bool(self.claims), bool(self.missing_information)) != expected:
            raise ValueError('Answer status does not match claims and missing information.')

    def to_dict(self) -> dict:
        return {'status': self.status,
                'claims': [{'text': c.text, 'source_ids': list(c.source_ids),
                            **({'quotes': [asdict(q) for q in c.quotes]} if c.quotes else {})} for c in self.claims],
                'missing_information': list(self.missing_information)}


@dataclass(frozen=True)
class CitationIssue:
    code: str
    claim_index: int
    source_id: str | None = None


@dataclass(frozen=True)
class ResolvedQuote:
    claim_index: int
    source_id: str
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class CitationValidation:
    """Only constructed after parsing; semantic support is explicitly unchecked."""

    issues: tuple[CitationIssue, ...] = ()
    quotes: tuple[ResolvedQuote, ...] = ()

    @property
    def references_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {'structure_valid': True, 'references_valid': self.references_valid,
                'support_status': 'not_checked', 'issues': [asdict(i) for i in self.issues],
                'resolved_quotes': [asdict(q) for q in self.quotes]}
