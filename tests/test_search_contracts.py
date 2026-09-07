from obsidian_rag.knowledge_base.models import Note
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot
from obsidian_rag.retrieval.models import NoteTarget, SearchResult, SearchResponse, SearchScope


def test_note_only_search_results_need_no_chunks_scores_or_embeddings():
    snapshot = KnowledgeSnapshot.from_notes([Note('Memory', 'body', 'memory.md')], vault_id='personal')
    result = SearchResult(source=snapshot.note_refs()[0], target=NoteTarget(), method='metadata', rank=1)
    response = SearchResponse(query='tag:memory', method='metadata',
        scope=SearchScope('personal', snapshot.snapshot_id), items=(result,), limit=10, has_more=False)
    payload = response.to_dict()
    assert payload['items'][0]['score'] is None
    assert payload['items'][0]['target'] == {'kind': 'note'}
    assert payload['items'][0]['excerpts'] == []
    assert payload['has_more'] is False


def test_text_matches_and_noncosine_scores_preserve_their_original_meaning():
    from obsidian_rag.retrieval.models import SpanTarget, SearchScore
    snapshot = KnowledgeSnapshot.from_notes([Note('Memory', 'alpha memory beta', 'memory.md')], vault_id='personal')
    source = snapshot.note_refs()[0]
    excerpt = snapshot.read_span(source, 6, 12)
    target = SpanTarget(6, 12)
    grep = SearchResult(source, target, 'grep', 1, (excerpt,))
    bm25 = SearchResult(source, target, 'bm25', 1, (excerpt,), SearchScore(17.5, 'bm25'))
    assert grep.score is None
    assert bm25.score.value == 17.5
    assert bm25.excerpts[0].content == 'memory'


def test_scores_are_validated_by_their_metric_not_a_universal_cosine_range():
    import pytest
    from obsidian_rag.retrieval.models import SearchScore
    assert SearchScore(17.5, 'bm25').value == 17.5
    for value, metric in [(1.1, 'cosine'), (float('nan'), 'bm25'), (float('inf'), 'cosine')]:
        with pytest.raises(ValueError):
            SearchScore(value, metric)


def test_results_cannot_mix_snapshots_or_claim_excerpts_outside_their_target():
    import pytest
    from obsidian_rag.retrieval.models import SpanTarget
    left = KnowledgeSnapshot.from_notes([Note('A', 'old body', 'a.md')], vault_id='personal')
    right = KnowledgeSnapshot.from_notes([Note('A', 'new body', 'a.md')], vault_id='personal')
    source = left.note_refs()[0]
    with pytest.raises(ValueError, match='source'):
        SearchResult(source, SpanTarget(0, 3), 'grep', 1, (right.read_span(right.note_refs()[0], 0, 3),))
    with pytest.raises(ValueError, match='target'):
        SearchResult(source, SpanTarget(0, 3), 'grep', 1, (left.read_span(source, 0, 8),))
    result = SearchResult(source, NoteTarget(), 'metadata', 1)
    with pytest.raises(ValueError, match='scope'):
        SearchResponse('tag:x', 'metadata', SearchScope('personal', right.snapshot_id), (result,), 10)


def test_vector_records_map_to_the_same_source_ids_and_exact_text():
    from obsidian_rag.knowledge_base.chunking import whole_note_chunks
    from obsidian_rag.retrieval.models import ChunkTarget, SearchScore
    snapshot = KnowledgeSnapshot.from_notes([Note('A', 'source body', 'a.md')], vault_id='personal')
    record = snapshot.bind_chunks(whole_note_chunks(snapshot.notes))[0]
    result = SearchResult.from_record(record, SearchScore(.75, 'cosine'), snapshot_id=snapshot.snapshot_id, rank=1, method='vector')
    assert isinstance(result.target, ChunkTarget)
    assert result.target.chunk_id == record.chunk_id
    assert result.source.document_id == record.document_id
    assert result.excerpts[0].content == 'source body'
    assert result.score == SearchScore(.75, 'cosine')
