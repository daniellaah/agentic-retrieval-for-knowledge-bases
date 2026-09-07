from contextlib import closing
from unittest.mock import Mock
import pytest
from ollama import Client, EmbedResponse
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models

from obsidian_rag.knowledge_base.models import Note
from obsidian_rag.knowledge_base.vector_index.manifest import EmbeddingSpec
from obsidian_rag.knowledge_base.vector_index.indexing import build_index
from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage
from obsidian_rag.knowledge_base.vector_index.qdrant import search_qdrant


@pytest.mark.filterwarnings('ignore:Payload indexes have no effect.*:UserWarning')
def test_qdrant_index_reopens_immutable_sources_and_reuses_embeddings(tmp_path):
    spec = EmbeddingSpec('test', 'digest', 2, 'title-body-v1')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    client = Mock(spec=Client)
    client.embed.return_value = EmbedResponse(embeddings=[[1., 0.]])
    with closing(QdrantClient(':memory:')) as qclient:
        options = dict(spec=spec, vault_id='vault', client=client, tokenizer=tokenizer,
                       max_input_tokens=100, chunking='none', qdrant_client=qclient,
                       backend={'kind': 'qdrant', 'url': 'http://test'})
        with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
            first = build_index(storage, [Note('A', 'original body', 'a.md')], **options)
            collection = storage.build_metadata(first.manifest.index_version)['backend']['collection']
        with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
            hits = search_qdrant(qclient, collection, [1., 0.], spec=spec, vault_id='vault', exact=True)
            record = storage.get_record(first.manifest.index_version, hits[0].chunk_id)
            assert record.chunk.content == 'original body'
            again = build_index(storage, [Note('A', 'original body', 'a.md')], **options, force=True)
            assert again.embedded_inputs == 0
            assert storage.get_record(again.manifest.index_version, record.chunk_id) == record
