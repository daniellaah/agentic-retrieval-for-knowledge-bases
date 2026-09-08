from obsidian_rag.knowledge_base.models import Note
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot
from obsidian_rag.retrieval.grep import grep_search


def test_literal_grep_returns_original_line_evidence_and_no_score():
    text = '第一行\nRAG[1] 与 e\u0301 🧠。\n最后一行'
    snapshot = KnowledgeSnapshot.from_notes([Note('A', text, 'a.md')], vault_id='v')
    response = grep_search(snapshot, 'RAG[1]')
    assert response.method == 'grep'
    assert response.has_more is False
    hit = response.items[0]
    assert hit.target.kind == 'span'
    assert hit.score is None
    assert hit.excerpts[0].content == 'RAG[1] 与 e\u0301 🧠。\n'
    assert text[hit.target.start_char:hit.target.end_char] == hit.excerpts[0].content
    assert grep_search(snapshot, 'RAG1').items == ()


def test_grep_filters_before_paging_deduplicates_lines_and_binds_cursor_to_snapshot():
    from dataclasses import replace
    import pytest
    notes = [Note('B', 'RAG RAG\nrag\nRAG', 'b.md'), Note('A', 'RAG', 'a.md')]
    snapshot = KnowledgeSnapshot.from_notes(notes, vault_id='v')
    first = grep_search(snapshot, 'rag', case_sensitive=False, paths=('b.md',), limit=1)
    assert first.items[0].excerpts[0].content == 'RAG RAG\n'
    assert first.scope.paths == ('b.md',)
    assert first.has_more and first.next_cursor
    second = grep_search(snapshot, 'rag', case_sensitive=False, paths=('b.md',), limit=2, cursor=first.next_cursor)
    assert [h.excerpts[0].content for h in second.items] == ['rag\n', 'RAG']
    assert [h.rank for h in second.items] == [1, 2]
    assert second.has_more is False and second.next_cursor is None
    changed = KnowledgeSnapshot.from_notes([replace(notes[0], content='changed')], vault_id='v')
    for options in ({'query': 'RAG'}, {'snapshot': changed}, {'case_sensitive': True}, {'paths': ('a.md',)}):
        with pytest.raises(ValueError, match='cursor'):
            grep_search(**dict(snapshot=snapshot, query='rag', case_sensitive=False, paths=('b.md',), cursor=first.next_cursor) | options)
    assert grep_search(snapshot, 'RAG', paths=('missing.md',)).items == ()


def test_grep_crosses_chunk_boundaries_and_handles_case_without_rewriting_source():
    snapshot = KnowledgeSnapshot.from_notes([Note('A', 'alpha\nBETA\nİstanbul', 'a.md')], vault_id='v')
    assert grep_search(snapshot, 'alpha\nBETA').items[0].excerpts[0].content == 'alpha\nBETA\n'
    hit = grep_search(snapshot, 'istanbul', case_sensitive=False).items[0]
    assert hit.excerpts[0].content == 'İstanbul'
    assert snapshot.read_span(hit.source, hit.target.start_char, hit.target.end_char) == hit.excerpts[0]


def test_grep_rejects_invalid_requests_even_when_scope_has_no_results():
    import pytest
    snapshot = KnowledgeSnapshot.from_notes([], vault_id='v')
    for options in ({'query': ' '}, {'limit': 0}, {'limit': True}, {'paths': []},
                    {'case_sensitive': 1}, {'cursor': 'not a cursor'}, {'cursor': 1}):
        with pytest.raises(ValueError):
            grep_search(**dict(snapshot=snapshot, query='RAG') | options)


def test_grep_and_source_read_do_not_import_vector_dependencies():
    import subprocess
    import sys
    code = '''
import sys
from obsidian_rag.retrieval.grep import grep_search
from obsidian_rag.retrieval.metadata import metadata_search, MetadataQuery
from obsidian_rag.retrieval.bm25 import bm25_search
from obsidian_rag.knowledge_base.lexical_index import LexicalIndex
from obsidian_rag.knowledge_base.models import Note
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot
s = KnowledgeSnapshot.from_notes([Note('A', 'RAG fact', 'a.md')], vault_id='v')
assert grep_search(s, 'RAG').items
assert metadata_search(s, MetadataQuery()).items
assert bm25_search(LexicalIndex.build(s), 'RAG').items
assert not ({'numpy', 'ollama', 'qdrant_client', 'tokenizers'} & sys.modules.keys())
'''
    process = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
