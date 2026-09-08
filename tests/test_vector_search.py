from contextlib import closing
from dataclasses import replace
from unittest.mock import Mock
import pytest
from ollama import Client, EmbedResponse
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models, pre_tokenizers

from obsidian_rag.knowledge_base.models import Note
from obsidian_rag.knowledge_base.vector_index.manifest import EmbeddingSpec
from obsidian_rag.knowledge_base.vector_index.indexing import build_index
from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage
from obsidian_rag.retrieval.vector import vector_search, VectorSearchConfig

pytestmark = pytest.mark.filterwarnings('ignore:.*local Qdrant.*:UserWarning')


@pytest.fixture
def indexed(tmp_path):
    spec = EmbeddingSpec('test', 'digest', 2, 'title-body-v1')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[([0., 1.] if t.startswith('B\n') else [1., 0.]) for t in kw['input']])
    with closing(QdrantClient(':memory:')) as qc, SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        build = dict(spec=spec, vault_id='vault', client=client, tokenizer=tokenizer, max_input_tokens=100,
                     chunking='none', backend={'kind': 'qdrant', 'url': 'http://test'}, qdrant_client=qc)
        build_index(storage, [Note('A', 'alpha body', 'a.md'), Note('B', 'beta body', 'b.md')],
                    **build, index_version='first')
        yield storage, dict(spec=spec, vault_id='vault', client=client, tokenizer=tokenizer, qdrant_client=qc), build


def test_vector_search_returns_scoped_common_results_and_filters_before_limit(indexed):
    storage, options, _ = indexed
    response = vector_search(storage, 'Question?', **options, config=VectorSearchConfig(top_k=1, source='b.md', exact=True))
    assert response.method == 'vector'
    assert response.scope.snapshot_id == 'first'
    assert response.scope.paths == ('b.md',)
    assert len(response.items) == 1
    result = response.items[0]
    assert result.source.path == 'b.md'
    assert result.excerpts[0].content == 'beta body'
    assert result.score.metric == 'cosine'
    assert result.score.value == 0
    assert response.has_more is None


def test_search_keeps_its_requested_snapshot_when_a_new_revision_is_published(indexed):
    storage, options, build = indexed
    build_index(storage, [Note('A', 'updated body', 'a.md')], **build, index_version='second')
    old = vector_search(storage, 'Question?', **options, index_version='first').items[0]
    new = vector_search(storage, 'Question?', **options).items[0]
    assert old.excerpts[0].content == 'alpha body'
    assert new.excerpts[0].content == 'updated body'
    assert old.source.document_id == new.source.document_id
    assert old.source.document_revision != new.source.document_revision


def test_incompatible_index_fails_before_contacting_the_embedding_model(indexed):
    storage, options, _ = indexed
    options['client'].embed.reset_mock()
    with pytest.raises(ValueError, match='incompatible'):
        vector_search(storage, 'Question?', **{**options, 'spec': replace(options['spec'], model_revision='changed')})
    options['client'].embed.assert_not_called()


def test_old_numpy_indexes_are_rejected_instead_of_silently_scanned(indexed):
    storage, options, _ = indexed
    legacy = replace(storage.get_manifest('first'), index_version='legacy', status='building', document_count=0, chunk_count=0)
    storage.create_build(legacy, corpus_fingerprint='legacy', backend={'kind': 'numpy'})
    storage.publish('legacy')
    options['client'].embed.reset_mock()
    with pytest.raises(ValueError, match='Qdrant'):
        vector_search(storage, 'Question?', **options)
    options['client'].embed.assert_not_called()


def test_other_methods_and_note_inspection_share_the_pinned_vector_snapshot(indexed):
    from obsidian_rag.retrieval.grep import grep_search
    from obsidian_rag.retrieval.metadata import metadata_search, MetadataQuery
    from obsidian_rag.retrieval.bm25 import bm25_search
    from obsidian_rag.knowledge_base.lexical_index import LexicalIndex
    from obsidian_rag.context import build_context
    storage, options, build = indexed
    build_index(storage, [Note('A', 'new version', 'a.md')], **build, index_version='second')
    vector = vector_search(storage, 'alpha', **options, index_version='first', config=VectorSearchConfig(source='a.md')).items[0]
    knowledge = storage.knowledge_snapshot('first')
    assert knowledge.read_note(vector.source).content == 'alpha body'
    grep = grep_search(knowledge, 'alpha').items[0]
    metadata = metadata_search(knowledge, MetadataQuery(path_prefix='a.md')).items[0]
    records = storage.snapshot_records('first')
    lexical = LexicalIndex.build(knowledge, chunks=[r.chunk for r in records])
    bm25 = bm25_search(lexical, 'alpha').items[0]
    assert vector.source == grep.source == metadata.source == bm25.source
    context = build_context('Q?', [vector, grep, metadata, bm25], citation_mode='structured')
    assert len(context.evidence_blocks) == 1
    assert {origin.retrieval_method for origin in context.citation_sources[0].origins} == {'vector', 'grep', 'bm25'}
    assert context.evidence_blocks[0].content == 'alpha body'
