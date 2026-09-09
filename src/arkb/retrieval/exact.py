"""Exact lexical matching over live source documents, currently backed by rg.

The subprocess receives normalized source text through stdin, never a command
string or agent-supplied filesystem path. Byte offsets are converted back to
Python character coordinates in Note.content. No model or index is involved.
"""

from collections.abc import Iterator, Mapping
import json
import subprocess

from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval.models import SearchResponse, SearchResult, validate_request


def _matches(text: str, query: str, *, regex: bool, case_sensitive: bool) -> Iterator[tuple[int, int]]:
    # Disable transcoding/BOM removal: offsets must address the supplied UTF-8 bytes.
    command = ['rg', '--no-config', '--json', '--text', '--multiline', '--encoding', 'none',
               '--case-sensitive' if case_sensitive else '--ignore-case']
    if not regex:
        command.append('--fixed-strings')
    # -e makes leading dashes data; shell=False is the subprocess default.
    completed = subprocess.run([*command, '-e', query], input=text, capture_output=True,
                               encoding='utf-8', check=False)
    if completed.returncode not in (0, 1):
        if completed.returncode == 2 and 'regex parse error' in completed.stderr:
            raise ValueError(completed.stderr.strip())
        raise subprocess.CalledProcessError(completed.returncode, command,
                                            output=completed.stdout, stderr=completed.stderr)
    data = text.encode('utf-8')
    for line in completed.stdout.splitlines():
        event = json.loads(line)
        if event['type'] != 'match':
            continue
        for match in event['data']['submatches']:
            start = event['data']['absolute_offset'] + match['start']
            end = event['data']['absolute_offset'] + match['end']
            try:
                span = len(data[:start].decode('utf-8')), len(data[:end].decode('utf-8'))
            except UnicodeDecodeError as error:
                raise ValueError('Patterns must match complete Unicode characters.') from error
            yield span


class ExactRetriever:
    """Find literal strings or explicit patterns, ordered by source then position."""

    def __init__(self, documents: DocumentAccess):
        self.documents = documents

    def search(self, query: str, *, target: str = 'content', regex: bool = False,
               case_sensitive: bool = True, top_k: int = 5,
               filters: Mapping[str, str] | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        if target not in ('content', 'source'):
            raise ValueError('target must be content or source.')
        if type(regex) is not bool or type(case_sensitive) is not bool:
            raise ValueError('regex and case_sensitive must be booleans.')
        if '\x00' in query:
            raise ValueError('query must not contain a NUL character.')
        if regex:
            # Invalid patterns must fail even when there are no source documents.
            list(_matches('', query, regex=True, case_sensitive=case_sensitive))
        results = []
        for record in self.documents.records(source=filters.get('source')):
            chunk = record.chunk
            text = chunk.content if target == 'content' else chunk.source
            for start, end in _matches(text, query, regex=regex, case_sensitive=case_sensitive):
                results.append(SearchResult(
                    source_id=record.document_id, source=chunk.source, method='exact',
                    content=chunk.content[start:end] if target == 'content' else chunk.content,
                    start_char=start if target == 'content' else None,
                    end_char=end if target == 'content' else None,
                    metadata={'title': chunk.title, 'document_revision': record.document_revision},
                ))
                if len(results) == top_k:
                    return SearchResponse(query=query, method='exact', results=tuple(results))
                if target == 'source':
                    break  # Filename matches identify a document only once.
        return SearchResponse(query=query, method='exact', results=tuple(results))
