"""Real-server publication, ANN readiness, failure recovery and pinned queries."""

from contextlib import closing
from dataclasses import replace
import os
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
from ollama import Client, EmbedResponse
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models, pre_tokenizers

from obsidian_rag.index_schema import EmbeddingSpec
from obsidian_rag.indexing import build_index
from obsidian_rag.notes import Note
from obsidian_rag.retrieval import search_index
from obsidian_rag.storage import SQLiteStorage
from obsidian_rag.vector_store_qdrant import QdrantVectorStore

pytestmark = pytest.mark.skipif(not os.environ.get('OBSIDIAN_RAG_QDRANT_URL'),
                               reason='Set OBSIDIAN_RAG_QDRANT_URL to a test Qdrant Server.')


def test_qdrant_publication_hnsw_and_failure_recovery(tmp_path, monkeypatch):
    url = os.environ['OBSIDIAN_RAG_QDRANT_URL']
    vault = 'test-' + uuid4().hex
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    spec = EmbeddingSpec(model='synthetic', model_revision='fixed', dimensions=64, document_template='title-body-v1')
    notes = [Note(title=str(i), content='test body', source=f'{i:03}.md') for i in range(256)]
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(256, 64))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    ollama = Mock(spec=Client)
    ollama.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[vectors[int(text.split('\n')[0])].tolist()
                                                                    for text in kw['input']])
    with SQLiteStorage(tmp_path / 'db') as storage, closing(QdrantClient(url=url, timeout=15, trust_env=False)) as client:
        options = dict(spec=spec, vault_id=vault, client=ollama, tokenizer=tokenizer, max_input_tokens=100,
                       chunking='none', qdrant_client=client,
                       backend={'kind': 'qdrant', 'url': url, 'indexing_threshold': 1, 'require_hnsw': True,
                                'index_timeout': 30})
        try:
            first = build_index(storage, notes, **options)
            metadata = storage.build_metadata(first.manifest.index_version)['backend']
            assert metadata['index_stats']['indexed_vectors'] >= 256
            assert build_index(storage, notes, **options).reused_index
            original_wait = QdrantVectorStore.wait_ready
            def fail(*args, **kwargs):
                raise ValueError('simulated index failure')
            monkeypatch.setattr(QdrantVectorStore, 'wait_ready', fail)
            with pytest.raises(ValueError, match='simulated'):
                build_index(storage, notes, **options, force=True)
            failed = storage.list_builds(vault)[-1]
            failed_collection = storage.build_metadata(failed.index_version)['backend']['collection']
            assert failed.status == 'failed' and client.collection_exists(failed_collection)
            assert storage.active_manifest(vault) == first.manifest
            monkeypatch.setattr(QdrantVectorStore, 'wait_ready', original_wait)
            second = build_index(storage, notes, **options, force=True)
            assert second.embedded_inputs == 0
            assert not client.collection_exists(failed_collection)
            ollama.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[vectors[0].tolist()])
            # Qdrant queries fetch only hit records; they must never load all cached vectors.
            monkeypatch.setattr(storage, 'load_snapshot', lambda *a: pytest.fail('loaded every vector during query'))
            results = search_index(storage, 'find', vault_id=vault, spec=spec, client=ollama, tokenizer=tokenizer,
                                   qdrant_client=client, index_version=first.manifest.index_version, top_k=1)
            assert results[0].chunk.source == '000.md'
            with pytest.raises(ValueError, match='incompatible'):
                search_index(storage, 'find', vault_id=vault, spec=replace(spec, model_revision='changed'),
                             client=ollama, tokenizer=tokenizer, qdrant_client=client)
        finally:
            for manifest in storage.list_builds(vault):
                name = storage.build_metadata(manifest.index_version)['backend'].get('collection')
                if name and client.collection_exists(name):
                    client.delete_collection(name)
