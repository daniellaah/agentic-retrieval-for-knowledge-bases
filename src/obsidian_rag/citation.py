"""Citation data and pure validation/rendering; no models or filesystem access.

Source IDs are local to one context. Persistent identities and coordinates refer
to the loaded Note.content snapshot, never raw Markdown bytes or line numbers.
Structural validity must never be presented as semantic evidence support.
"""

from dataclasses import asdict, dataclass
import math
import re
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
class Claim:
    """Model-proposed fact; empty references remain inspectable validation failures."""

    text: str
    source_ids: tuple[str, ...]

    def __post_init__(self):
        _text(self.text, 'Claim text')
        if not isinstance(self.source_ids, tuple):
            raise ValueError('source_ids must be an immutable tuple.')
        for source_id in self.source_ids:
            _source_id(source_id)
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError('Duplicate source IDs in one claim.')


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
                'claims': [{'text': c.text, 'source_ids': list(c.source_ids)} for c in self.claims],
                'missing_information': list(self.missing_information)}


@dataclass(frozen=True)
class CitationIssue:
    code: str
    claim_index: int
    source_id: str | None = None


@dataclass(frozen=True)
class CitationValidation:
    """Only constructed after parsing; semantic support is explicitly unchecked."""

    issues: tuple[CitationIssue, ...] = ()

    @property
    def references_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        return {'structure_valid': True, 'references_valid': self.references_valid,
                'support_status': 'not_checked', 'issues': [asdict(i) for i in self.issues]}
