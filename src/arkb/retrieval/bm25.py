"""In-memory BM25 over pinned chunks; no embedding or vector-store dependency.

Index title + body with NFC/casefold Unicode word tokens (underscores retained).
No stemming, stopwords, or language segmentation. Each distinct query term votes
once. IDF is log(1 + (N-df+.5)/(df+.5)); TF uses (k1+1)*tf divided by
tf + k1*(1-b+b*length/average_length). All chunks count toward corpus statistics.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
import re
import unicodedata

from arkb.retrieval.models import SearchResponse, validate_request, chunk_result
from arkb.knowledge.models import ChunkRecord, validate_records


def _tokens(text: str) -> list[str]:
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class BM25Retriever:
    def __init__(self, records: Sequence[ChunkRecord], *, index_id: str | None = None,
                 k1: float = 1.2, b: float = .75):
        if (type(k1) not in (int, float) or not math.isfinite(k1) or k1 <= 0
                or type(b) not in (int, float) or not math.isfinite(b) or not 0 <= b <= 1):
            raise ValueError('BM25 requires finite k1 > 0 and b in [0, 1].')
        self.records = tuple(records)
        if self.records:
            validate_records(list(self.records), vault_id=self.records[0].vault_id)
        revisions = {}
        for record in self.records:
            if revisions.setdefault(record.document_id, record.document_revision) != record.document_revision:
                raise ValueError('BM25 cannot mix document revisions.')
        self.index_id, self.k1, self.b = index_id, k1, b
        if index_id is not None and (not isinstance(index_id, str) or not index_id.strip()):
            raise ValueError('index_id must be nonblank text.')
        self._postings = defaultdict(dict)
        self._lengths = []
        for ordinal, record in enumerate(self.records):
            terms = _tokens(record.chunk.title + '\n\n' + record.chunk.content)
            self._lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self._postings[term][ordinal] = count
        self._average = sum(self._lengths) / len(self.records) if self.records else 0

    @classmethod
    def from_snapshot(cls, storage, *, vault_id: str, index_version: str | None = None,
                      k1: float = 1.2, b: float = .75):
        """Read text only from a READY SQLite snapshot, without vector cache reads.

        The caller owns storage. A later publication cannot change this instance.
        """
        manifest = storage.get_manifest(index_version) if index_version is not None else storage.active_manifest(vault_id)
        if manifest is None or manifest.status != 'ready' or manifest.vault_id != vault_id:
            raise ValueError('BM25 requires a ready snapshot in the requested vault.')
        return cls(storage.snapshot_records(manifest.index_version),
                   index_id=manifest.index_version, k1=k1, b=b)

    def search(self, query: str, *, top_k: int = 2,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        scores = defaultdict(float)
        for term in sorted(set(_tokens(query))):
            postings = self._postings.get(term, {})
            idf = math.log1p((len(self.records) - len(postings) + .5) / (len(postings) + .5))
            for ordinal, frequency in postings.items():
                record = self.records[ordinal]
                if 'source' in filters and record.chunk.source != filters['source']:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._lengths[ordinal] / self._average)
                scores[ordinal] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = sorted(scores, key=lambda i: (-scores[i], self.records[i].document_id, self.records[i].chunk_id))
        hits = []
        for i in ranked[:top_k]:
            hit = chunk_result(self.records[i], method='bm25', index_id=self.index_id,
                               score=scores[i], score_type='bm25')
            hits.append(replace(hit, metadata={**hit.metadata, 'bm25': {
                'k1': self.k1, 'b': self.b, 'tokenizer': 'nfc-casefold-word-v1',
                'fields': 'title+body', 'query_terms': 'unique'}}))
        return SearchResponse(query=query, method='bm25', results=tuple(hits), index_id=self.index_id)
