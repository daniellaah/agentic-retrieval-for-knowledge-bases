"""Citation data and pure validation/rendering; no models or filesystem access.

Source IDs are local to one context. Persistent identities and coordinates refer
to the loaded Note.content snapshot, never raw Markdown bytes or line numbers.
Structural validity must never be presented as semantic evidence support.
"""

from dataclasses import asdict, dataclass
from collections.abc import Sequence
import html
import json
import math
import re
import unicodedata
from typing import Literal


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
    chunk_id: str | None = None
    document_id: str | None = None
    document_revision: str | None = None
    vault_id: str | None = None
    index_version: str | None = None

    def __post_init__(self):
        _span(self.start_char, self.end_char)
        if type(self.score) not in (int, float) or not math.isfinite(self.score) or not -1 <= self.score <= 1:
            raise ValueError('Origin score must be a finite cosine score.')
        identity = (self.chunk_id, self.document_id, self.document_revision, self.vault_id)
        if any(v is not None for v in identity):
            for value in identity:
                _text(value, 'Origin identity')
        if self.index_version is not None:
            _text(self.index_version, 'index_version')


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
            if len(keys) != 1 or self.origins[0].document_revision is None:
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


class CitationParseError(ValueError):
    """Retain the unmodified model response for an evaluator, not user display."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


def citation_json_schema(source_ids: Sequence[str], *, include_quotes: bool = False) -> dict:
    """Fresh response schema; application validation remains authoritative."""
    ids = list(source_ids)
    for source_id in ids:
        _source_id(source_id)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate source IDs in schema.')
    schema = {
        'type': 'object', 'additionalProperties': False,
        'required': ['status', 'claims', 'missing_information'],
        'properties': {
            'status': {'type': 'string', 'enum': ['answered', 'partial', 'insufficient_evidence']},
            'claims': {'type': 'array', 'items': {
                'type': 'object', 'additionalProperties': False,
                'required': ['text', 'source_ids'],
                'properties': {
                    'text': {'type': 'string', 'minLength': 1},
                    'source_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                                   'items': {'type': 'string', 'enum': ids} if ids else {'type': 'string'}},
                },
            }},
            'missing_information': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}},
        },
    }
    if type(include_quotes) is not bool:
        raise ValueError('include_quotes must be boolean.')
    if include_quotes:
        claim = schema['properties']['claims']['items']
        claim['required'].append('quotes')
        claim['properties']['quotes'] = {
            'type': 'array', 'minItems': 1, 'uniqueItems': True,
            'items': {'type': 'object', 'additionalProperties': False, 'required': ['source_id', 'text'],
                      'properties': {'source_id': {'type': 'string', 'enum': ids} if ids else {'type': 'string'},
                                     'text': {'type': 'string', 'minLength': 1}}},
        }
    return schema


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f'Duplicate JSON key: {key}.')
        obj[key] = value
    return obj


def _keys(obj, expected, optional=()):
    if (not isinstance(obj, dict) or not set(expected) <= set(obj)
            or not set(obj) <= set(expected) | set(optional)):
        raise ValueError(f'Expected exactly these object fields: {", ".join(expected)}.')


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}.')


def parse_cited_answer(raw_response: str) -> CitedAnswer:
    """Parse one strict JSON object. Never repair, strip fences or coerce types."""
    try:
        _text(raw_response, 'Model response')
        obj = json.loads(raw_response, object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
        _keys(obj, ('status', 'claims', 'missing_information'))
        if not isinstance(obj['claims'], list) or not isinstance(obj['missing_information'], list):
            raise ValueError('claims and missing_information must be arrays.')
        claims = []
        for entry in obj['claims']:
            _keys(entry, ('text', 'source_ids'), ('quotes',))
            if not isinstance(entry['source_ids'], list):
                raise ValueError('source_ids must be an array.')
            quotes = entry.get('quotes', [])
            if not isinstance(quotes, list):
                raise ValueError('quotes must be an array.')
            for quote in quotes:
                _keys(quote, ('source_id', 'text'))
            claims.append(Claim(entry['text'], tuple(entry['source_ids']),
                                tuple(CitationQuote(q['source_id'], q['text']) for q in quotes)))
        return CitedAnswer(obj['status'], tuple(claims), tuple(obj['missing_information']))
    except (ValueError, TypeError, RecursionError) as error:
        raise CitationParseError(f'Invalid cited answer structure: {error}', raw_response) from error


def _registry(sources: Sequence[CitationSource]) -> dict[str, CitationSource]:
    if any(not isinstance(s, CitationSource) for s in sources):
        raise ValueError('Expected CitationSource objects.')
    registry = {s.source_id: s for s in sources}
    if len(registry) != len(sources):
        raise ValueError('Duplicate source IDs in registry.')
    return registry


# Model prose cannot impersonate the application's citation markers or links.
_INLINE_REFERENCE = re.compile(r'\[(?:S?[0-9]+|[^\]\n]*\.md)\]|\]\(|\b(?:https?|file|obsidian)://', re.I)


def validate_citations(answer: CitedAnswer, sources: Sequence[CitationSource], *,
                       require_quotes: bool = False) -> CitationValidation:
    """Check references against sent evidence; no semantic support inference."""
    registry = _registry(sources)
    if type(require_quotes) is not bool:
        raise ValueError('require_quotes must be boolean.')
    issues, resolved = [], []
    for i, claim in enumerate(answer.claims):
        if not claim.source_ids:
            issues.append(CitationIssue('missing_reference', i))
        for source_id in claim.source_ids:
            if source_id not in registry:
                issues.append(CitationIssue('unknown_source', i, source_id))
        if _INLINE_REFERENCE.search(claim.text):
            issues.append(CitationIssue('inline_reference', i))
        if require_quotes:
            for source_id in claim.source_ids:
                if source_id not in {q.source_id for q in claim.quotes}:
                    issues.append(CitationIssue('missing_quote', i, source_id))
        for quote in claim.quotes:
            if quote.source_id not in claim.source_ids:
                issues.append(CitationIssue('quote_not_referenced', i, quote.source_id))
                continue
            source = registry.get(quote.source_id)
            if source is None:
                continue  # unknown_source already reported for this reference
            offset = source.content.find(quote.text)
            if offset < 0:
                issues.append(CitationIssue('quote_not_found', i, quote.source_id))
            elif source.content.find(quote.text, offset + 1) >= 0:
                issues.append(CitationIssue('ambiguous_quote', i, quote.source_id))
            else:
                start = source.start_char + offset
                resolved.append(ResolvedQuote(i, quote.source_id, quote.text, start, start + len(quote.text)))
    if any(_INLINE_REFERENCE.search(text) for text in answer.missing_information):
        issues.append(CitationIssue('inline_reference_in_missing_information', -1))
    return CitationValidation(tuple(issues), tuple(resolved))


def used_citation_sources(answer: CitedAnswer, sources: Sequence[CitationSource]) -> tuple[CitationSource, ...]:
    """Order by first use, preserving each claim's joint-reference order."""
    validation = validate_citations(answer, sources)
    if not validation.references_valid:
        raise ValueError('Cannot render invalid citations: ' + ', '.join(i.code for i in validation.issues))
    registry = _registry(sources)
    ids = dict.fromkeys(source_id for claim in answer.claims for source_id in claim.source_ids)
    return tuple(registry[source_id] for source_id in ids)


def _display_text(text: str) -> str:
    """Render data as one inert Markdown line; raw text remains in JSON artifacts."""
    visible = ''.join(f'\\u{ord(c):04x}' if unicodedata.category(c) in ('Cc', 'Cf') and c not in '\n\r\t'
                      else c for c in text)
    visible = html.escape(' '.join(visible.split()), quote=False)
    return re.sub(r'([\\`*_\[\]()#!|:])', r'\\\1', visible)


def render_cited_answer(answer: CitedAnswer, sources: Sequence[CitationSource]) -> str:
    """Render validated IDs and snapshot excerpts. Never invent filesystem URLs.

    Display numbers are first-use order, distinct from request-local source IDs.
    Full source identities and verbatim evidence are available in structured data.
    """
    used = used_citation_sources(answer, sources)
    quotes = validate_citations(answer, sources).quotes
    numbers = {s.source_id: i for i, s in enumerate(used, 1)}
    paragraphs = []
    for i, claim in enumerate(answer.claims):
        paragraph = f'{_display_text(claim.text)} ' + ''.join(f'[{numbers[s]}]' for s in claim.source_ids)
        for quote in (q for q in quotes if q.claim_index == i):
            paragraph += f'\n> [{numbers[quote.source_id]}] body chars [{quote.start_char}, {quote.end_char}): {_display_text(quote.text)}'
        paragraphs.append(paragraph)
    if answer.missing_information:
        paragraphs.append('Missing information in the provided notes: ' +
                          ' '.join(_display_text(t) for t in answer.missing_information))
    if used:
        lines = ['Sources (Note.content snapshot characters, end exclusive):']
        for s in used:
            origin = s.origins[0]
            identity = (f'revision {_display_text(origin.document_revision)}, '
                        f'index {_display_text(origin.index_version or "unknown")}') if origin.document_revision else 'unversioned'
            lines.append(f'[{numbers[s.source_id]}] {_display_text(s.source)} — {_display_text(s.title)}; '
                         f'body chars [{s.start_char}, {s.end_char}); {identity}')
            if not any(q.source_id == s.source_id for q in quotes):
                lines.append('> ' + _display_text(s.content))
        paragraphs.append('\n'.join(lines))
    return '\n\n'.join(paragraphs)
