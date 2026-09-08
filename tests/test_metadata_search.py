from obsidian_rag.knowledge_base.models import Note
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot
from obsidian_rag.knowledge_base.metadata import read_metadata
from obsidian_rag.retrieval.metadata import metadata_search, MetadataQuery


def test_metadata_filters_note_identity_and_properties_without_creating_evidence():
    body = '---\ntags: [AI, agent/memory]\naliases:\n  - 工作记忆\n---\n原文 🧠'
    snapshot = KnowledgeSnapshot.from_notes([
        Note('Agent Memory', body, 'ai/memory.md'),
        Note('Another', '---\ntags: [AI]\n---\nbody', 'archive/old.md')], vault_id='v')
    query = MetadataQuery(title_contains='memory', path_prefix='ai/', tags=('AI',), aliases=('工作记忆',))
    response = metadata_search(snapshot, query)
    assert response.method == 'metadata'
    assert len(response.items) == 1
    hit = response.items[0]
    assert hit.target.kind == 'note' and hit.excerpts == () and hit.score is None
    assert snapshot.read_note(hit.source).content == body
    metadata = read_metadata(snapshot, hit.source)
    assert metadata.tags == ('AI', 'agent/memory')
    assert metadata.aliases == ('工作记忆',)
    from obsidian_rag.context import build_context
    assert build_context('What is memory?', response.items).decisions == ((0, 'needs_inspection'),)


def test_metadata_rejects_ambiguous_frontmatter_and_invalid_filter_values():
    import pytest
    for frontmatter in ('tags: [one]\ntags: [two]', 'tags: [unfinished', 'tags: [7]',
                        '- tags', 'false', '[]', '0', '!!python/object:builtins.object {}'):
        snapshot = KnowledgeSnapshot.from_notes([Note('A', '---\n' + frontmatter + '\n---\nbody', 'a.md')], vault_id='v')
        with pytest.raises(ValueError):
            metadata_search(snapshot, MetadataQuery())
    for args in ({'title_contains': ' '}, {'path_prefix': 1}, {'tags': ['AI']}, {'aliases': ('',)}):
        with pytest.raises(ValueError):
            MetadataQuery(**args)


def test_metadata_paging_and_filters_keep_original_sources_and_reject_stale_cursors():
    import pytest
    snapshot = KnowledgeSnapshot.from_notes([
        Note('B', '---\ntags: AI\n---\n#text', 'b.md'),
        Note('A', '---\ntags: [ai, memory]\n---\nbody', 'a.md'),
        Note('C', 'Text mentions AI but has no tag.', 'c.md')], vault_id='v')
    query = MetadataQuery(tags=('AI',))
    first = metadata_search(snapshot, query, limit=1)
    assert first.items[0].source.path == 'a.md'
    assert first.has_more and first.next_cursor
    second = metadata_search(snapshot, query, limit=1, cursor=first.next_cursor)
    assert second.items[0].source.path == 'b.md'
    assert not second.has_more
    assert metadata_search(snapshot, MetadataQuery(tags=('ai', 'memory'))).items == first.items
    assert metadata_search(snapshot, query, paths=('c.md',)).items == ()
    with pytest.raises(ValueError, match='cursor'):
        metadata_search(snapshot, MetadataQuery(), cursor=first.next_cursor)
    assert read_metadata(snapshot, snapshot.note_refs()[2]).tags == ()
