"""Production adapters against SQLite and Qdrant Local, with fixed model outputs."""

from types import SimpleNamespace
from unittest.mock import Mock

from ollama import Client, EmbedResponse
from tokenizers import Tokenizer, models

from arkb.knowledge.indexing import build_index
from arkb.retrieval import BM25Retriever, HybridRetriever, RerankedRetriever, Reranker, SemanticRetriever
from arkb.knowledge.embeddings import OllamaQueryEmbedder
from arkb.retrieval.qdrant import QdrantSnapshotIndex
from arkb.retrieval_evaluation import evaluate_retrievers
from arkb.knowledge.models import EmbeddingSpec, Note
from arkb.knowledge.sqlite import SQLiteStorage


def test_all_four_configurations_share_snapshot_identity_and_apply_filters_before_ranking(tmp_path, qdrant, qdrant_config):
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    spec = EmbeddingSpec(model='fixed', model_revision='v1', dimensions=2, document_template='title-body-v1')
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[
        [1., 0.] if 'France' in text else [0., 1.] for text in kw['input']])
    path = tmp_path / 'index.sqlite'
    with SQLiteStorage(path) as storage:
        build_index(storage, [Note('', 'Paris is in France.', 'paris.md'),
                              Note('', 'Paris is a name used for a dog.', 'dog.md')],
                    spec=spec, vault_id='v', client=client, tokenizer=tokenizer, chunking='none',
                    max_input_tokens=100, qdrant_client=qdrant, qdrant_config=qdrant_config)
    before = path.read_bytes()
    with SQLiteStorage(path, read_only=True) as storage:
        index = QdrantSnapshotIndex(storage, qdrant, vault_id='v', exact=True)
        embedder = OllamaQueryEmbedder(client=client, spec=spec, tokenizer=tokenizer,
            tokenizer_identity=index.inputs['tokenizer'], max_input_tokens=100,
            query_instruction=index.manifest.query_instruction)
        semantic = SemanticRetriever(embedder, index)
        bm25 = BM25Retriever.from_snapshot(storage, vault_id='v', index_version=index.index_id)
        hybrid = HybridRetriever(bm25, semantic, candidate_k=2)
        scorer = SimpleNamespace(identity='frozen-fixture', score_type='fixture-relevance',
            score=lambda query, candidates: [3. if 'France' in h.content else -1. for h in candidates])
        reranked = RerankedRetriever(hybrid, Reranker(scorer), candidate_k=2)
        modes = {'semantic': semantic, 'bm25': bm25, 'hybrid': hybrid, 'hybrid_reranked': reranked}
        report = evaluate_retrievers(modes, [{'id': 'q', 'question': 'Paris France',
                                            'relevance': {'paris.md': 3}}], top_k=1)
        assert all(value['mrr'] == value['recall_at_k'] == value['ndcg_at_k'] == 1
                   for value in report['summary'].values())
        hits = [retriever.search('Paris France', top_k=1).results[0] for retriever in modes.values()]
        assert len({h.identity for h in hits}) == len({h.metadata['index_version'] for h in hits}) == 1
        assert [h.score_type for h in hits] == ['cosine_similarity', 'bm25', 'rrf', 'fixture-relevance']
        for retriever in modes.values():
            assert [h.source for h in retriever.search('Paris France', top_k=1,
                    filters={'source': 'dog.md'}).results] == ['dog.md']
    assert path.read_bytes() == before
