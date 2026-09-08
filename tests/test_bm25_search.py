import pytest

from obsidian_rag.knowledge_base.models import Note
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot
from obsidian_rag.knowledge_base.lexical_index import LexicalIndex
from obsidian_rag.retrieval.bm25 import bm25_search


def test_bm25_ranks_note_candidates_with_declared_scores_and_no_invented_chunks():
    snapshot = KnowledgeSnapshot.from_notes([Note('A', 'apple apple', 'a.md'), Note('B', 'banana', 'b.md')], vault_id='v')
    index = LexicalIndex.build(snapshot)
    response = bm25_search(index, 'apple')
    assert response.method == 'bm25'
    assert response.has_more is False
    assert [hit.source.path for hit in response.items] == ['a.md']
    hit = response.items[0]
    assert hit.target.kind == 'note' and hit.excerpts == ()
    assert hit.score.metric == 'bm25' and hit.score.higher_is_better
    # Hand-calculated: N=2, df=1, tf=2, lengths=(3,2), avgdl=2.5, k1=1.2, b=.75.
    assert hit.score.value == pytest.approx(0.902321773509988)
    assert bm25_search(index, 'missing').items == ()


def test_bm25_chunk_retrieval_reuses_shared_chunks_and_preserves_original_unicode():
    from obsidian_rag.knowledge_base.chunking import chunk_notes
    from obsidian_rag.context import build_context
    body = '长期记忆和 RAG。 cafe\u0301 🧠\n其他内容'
    snapshot = KnowledgeSnapshot.from_notes([Note('Memory', body, 'a.md')], vault_id='v')
    chunks = chunk_notes(snapshot.notes, count_tokens=len, chunk_size=18, chunk_overlap=0)
    index = LexicalIndex.build(snapshot, chunks=chunks)
    response = bm25_search(index, '记忆 CAFÉ')
    assert response.items
    hit = response.items[0]
    assert hit.target.kind == 'chunk'
    assert hit.target.chunk_id in {r.chunk_id for r in snapshot.bind_chunks(chunks)}
    assert body[hit.target.start_char:hit.target.end_char] == hit.excerpts[0].content
    assert '长期记忆' in hit.excerpts[0].content
    built = build_context('Q?', response.items, citation_mode='structured')
    assert built.has_evidence
    assert built.citation_sources[0].origins[0].score_metric == 'bm25'


def test_bm25_finds_chinese_words_inside_unspaced_text():
    snapshot = KnowledgeSnapshot.from_notes([Note('A', '这是长期记忆的笔记', 'a.md'), Note('B', '完全无关', 'b.md')], vault_id='v')
    index = LexicalIndex.build(snapshot)
    assert [h.source.path for h in bm25_search(index, '记忆').items] == ['a.md']


def test_bm25_pagination_filters_ties_and_index_identity_are_explicit():
    from obsidian_rag.knowledge_base.chunking import whole_note_chunks
    snapshot = KnowledgeSnapshot.from_notes([Note('A', 'RAG', 'b.md'), Note('A', 'RAG', 'a.md')], vault_id='v')
    index = LexicalIndex.build(snapshot)
    first = bm25_search(index, 'rag', limit=1)
    assert first.items[0].source.path == 'a.md' and first.has_more
    second = bm25_search(index, 'rag', limit=1, cursor=first.next_cursor)
    assert second.items[0].source.path == 'b.md' and not second.has_more
    assert bm25_search(index, 'rag', paths=('b.md',), limit=1).items == second.items
    chunks = LexicalIndex.build(snapshot, chunks=whole_note_chunks(snapshot.notes))
    with pytest.raises(ValueError, match='cursor'):
        bm25_search(chunks, 'rag', cursor=first.next_cursor)
    with pytest.raises(ValueError, match='cursor'):
        bm25_search(index, 'rag', cursor=first.next_cursor, k1=2)
    with pytest.raises(TypeError):
        index.document_frequency['rag'] = 99


def test_bm25_validates_empty_indexes_and_retains_frozen_content():
    from dataclasses import replace
    from obsidian_rag.knowledge_base.chunking import whole_note_chunks
    empty = LexicalIndex.build(KnowledgeSnapshot.from_notes([], vault_id='v'))
    assert bm25_search(empty, 'word').items == ()
    assert bm25_search(empty, '!!!').has_more is False
    for kwargs in ({'k1': -1}, {'k1': float('nan')}, {'b': 1.1}, {'b': True}, {'limit': 0}, {'query': ' '}):
        with pytest.raises(ValueError):
            bm25_search(**dict(index=empty, query='word') | kwargs)
    old = KnowledgeSnapshot.from_notes([Note('A', 'old evidence', 'a.md')], vault_id='v')
    chunks = whole_note_chunks(old.notes)
    index = LexicalIndex.build(old, chunks=chunks)
    changed = KnowledgeSnapshot.from_notes([Note('A', 'new evidence', 'a.md')], vault_id='v')
    with pytest.raises(ValueError, match='snapshot'):
        LexicalIndex.build(changed, chunks=chunks)
    with pytest.raises(ValueError, match='unique'):
        LexicalIndex.build(old, chunks=chunks * 2)
    assert bm25_search(index, 'old').items[0].excerpts[0].content == 'old evidence'
    with pytest.raises(ValueError, match='analyzer'):
        bm25_search(replace(index, analyzer_id='different'), 'old')
