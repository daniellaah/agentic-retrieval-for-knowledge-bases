"""Immutable term statistics over a knowledge snapshot, independent of vector indexes."""

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata
from types import MappingProxyType
from collections.abc import Mapping

from .identity import fingerprint_config
from .sources import KnowledgeSnapshot, SourceRef
from .models import ChunkRecord

ANALYZER_ID = 'unicode-nfc-casefold-han-unigram-bigram-v1'
_HAN = r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]+'


def lexical_terms(text: str) -> tuple[str, ...]:
    terms = []
    for part in re.split(f'({_HAN})', unicodedata.normalize('NFC', text).casefold()):
        if re.fullmatch(_HAN, part):
            terms.extend(part)
            terms.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            terms.extend(re.findall(r'[^\W_]+', part))
    return tuple(terms)


@dataclass(frozen=True)
class LexicalUnit:
    source: SourceRef
    frequencies: Mapping[str, int]
    length: int
    record: ChunkRecord | None = None


@dataclass(frozen=True)
class LexicalIndex:
    snapshot: KnowledgeSnapshot
    units: tuple[LexicalUnit, ...]
    document_frequency: Mapping[str, int]
    analyzer_id: str = ANALYZER_ID

    @classmethod
    def build(cls, snapshot: KnowledgeSnapshot, *, chunks=None):
        """Index title+body for notes, or title+chunk for validated shared chunks.

        Statistics are immutable and rebuilt explicitly for a changed snapshot.
        Han unigrams+bigrams support Chinese without a downloaded segmenter;
        normalization applies only to terms, never stored source text.
        """
        sources = {ref.path: ref for ref in snapshot.note_refs()}
        if chunks is None:
            inputs = [(sources[n.source], n.title + '\n' + n.content, None) for n in snapshot.notes]
        else:
            records = snapshot.bind_chunks(chunks)
            if len({r.chunk_id for r in records}) != len(records):
                raise ValueError('Lexical chunks must have unique identities.')
            inputs = [(sources[r.chunk.source], r.chunk.title + '\n' + r.chunk.content, r) for r in records]
        units, counts = [], Counter()
        for source, text, record in inputs:
            terms = Counter(lexical_terms(text))
            counts.update(terms.keys())
            units.append(LexicalUnit(source, MappingProxyType(dict(terms)), sum(terms.values()), record))
        return cls(snapshot, tuple(units), MappingProxyType(dict(counts)))

    @property
    def fingerprint(self):
        return fingerprint_config({'analyzer': self.analyzer_id, 'snapshot': self.snapshot.snapshot_id,
                                   'units': [{'source': u.source.document_id, 'chunk': u.record.chunk_id if u.record else None}
                                             for u in self.units]})
