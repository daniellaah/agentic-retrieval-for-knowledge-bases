from functools import partial
import numpy as np
import pytest
from arkb.knowledge.models import Note, ChunkRecord, EmbeddingSpec
from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.qdrant import search_qdrant
from arkb.evaluation.retrieval import compare_retrieval

def test_comparison_separates_neighbor_recall_from_evidence_coverage(qdrant):
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2, document_template='title-body-v1')
    notes = [Note(title='T', content='evidence', source=f'{i}.md') for i in range(2)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='v') for n in notes]
    vectors = np.eye(2)
    from arkb.knowledge.qdrant import QdrantIndex
    index = QdrantIndex(qdrant, 'evaluation', spec, vault_id='v', create=True)
    index.upsert(records, vectors)
    search = partial(search_qdrant, qdrant, 'evaluation', spec=spec, vault_id='v')
    cases = [{'id': 'q1', 'question': 'Question?', 'required_source_groups': [['1.md']],
              'evidence_anchors': [{'source': '1.md', 'body_start_char': 0, 'body_end_char': 8}]}]
    result = compare_retrieval(records, vectors, [[1, 0]], cases, spec=spec, search=search, top_k=1)
    assert result['summary']['qdrant_exact']['neighbor_recall_at_k'] == 1
    assert result['summary']['qdrant_exact']['section_coverage'] == 0
    assert result['summary']['qdrant_ann']['neighbor_recall_at_k'] == 1
    assert result['summary']['qdrant_ann']['section_coverage'] == 0
    assert result['settings']['vector_bytes'] == vectors.nbytes
    with pytest.raises(ValueError, match='unique'):
        compare_retrieval(records, vectors, [[1, 0], [1, 0]], cases * 2, spec=spec, search=search)


def test_runner_hashes_nested_sources_and_preserves_snapshot_and_artifacts(tmp_path, monkeypatch, capsys, qdrant, qdrant_config):
    import hashlib
    import json
    from unittest.mock import MagicMock

    from ollama import Client, ChatResponse, EmbedResponse, Message
    from tokenizers import Tokenizer, models, pre_tokenizers

    import arkb.evaluation.retrieval as evaluation
    from arkb.generation.models import GenerationCounter
    from arkb.knowledge.indexing import build_index
    from arkb.knowledge.sqlite import SQLiteStorage

    # Identical basenames in different packages must retain distinct identities.
    package_dir = tmp_path / 'src' / 'arkb'
    sources = {
        '__init__.py': b'"""Package."""\n',
        'evaluation/retrieval.py': b'"""Evaluation."""\n',
        'knowledge/__init__.py': b'"""Indexing."""\n',
        'retrieval/__init__.py': b'"""Retrieval."""\n',
        'retrieval/semantic.py': b'"""Semantic search."""\n',
        'generation/context.py': b'"""Context construction."""\n',
    }
    for name, data in sources.items():
        path = package_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (package_dir / 'unrelated.txt').write_text('Not Python source.')
    monkeypatch.setattr(evaluation, '__file__', str(package_dir / 'evaluation' / 'retrieval.py'))

    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2,
                         document_template='title-body-v1')
    client = MagicMock(spec=Client)
    client.__enter__.return_value = client
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    client.chat.return_value = ChatResponse(message=Message(role='assistant', content=json.dumps({
        'status': 'answered', 'claims': [{'text': 'A fact.', 'source_ids': ['S1']}],
        'missing_information': [],
    })))
    db = tmp_path / 'index.sqlite'
    with SQLiteStorage(db) as storage:
        build_index(storage, [Note('Title', 'A fact.', 'a.md')], spec=spec, vault_id='default',
                    client=client, tokenizer=tokenizer, max_input_tokens=100, chunking='none',
                    qdrant_client=qdrant, qdrant_config=qdrant_config)
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', lambda *a: qdrant)
    before = db.read_bytes()
    client.embed.reset_mock()
    monkeypatch.setattr('ollama.Client', lambda **kw: client)
    monkeypatch.setattr('arkb.knowledge.embeddings.resolve_embedding_spec', lambda *a, **kw: spec)
    monkeypatch.setattr('arkb.knowledge.embeddings.load_tokenizer', lambda **kw: tokenizer)
    counter = GenerationCounter('test-generation', 'test-count',
                                 lambda messages: 10 + sum(len(m['content']) for m in messages))
    monkeypatch.setattr('arkb.generation.generate.load_generation_counter', lambda **kw: counter)
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'fact', 'question': 'What is stated?'}) + '\n')
    output = tmp_path / 'evaluation'
    args = ['--db', str(db), '--cases', str(cases), '--output', str(output),
            '--offline', '--context', '--citations']

    assert evaluation.ann_main(args) == 0
    capsys.readouterr()
    metadata = json.loads((output / 'run_metadata.json').read_text())
    assert metadata['source_hashes'] == {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    row = json.loads((output / 'citation_results.jsonl').read_text())
    assert row['success'] and row['context']['citation_sources'][0]['content'] == 'A fact.'
    assert json.loads((output / 'metrics.json').read_text())['citations']['summary']['success_count'] == 1
    client.embed.assert_called_once()
    assert db.read_bytes() == before
    artifacts = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit) as error:
        evaluation.ann_main(args)
    assert error.value.code == 2
    assert {p.name: p.read_bytes() for p in output.iterdir()} == artifacts


def test_comparison_uses_same_questions_preserves_ranks_and_reports_latency():
    from arkb.retrieval import SearchResult, SearchResponse
    from arkb.evaluation.retrieval import evaluate_retrievers
    class FrozenRetriever:
        def search(self, query, *, top_k=2, filters=None):
            assert query == 'original question'
            return SearchResponse(query=query, method='frozen', results=tuple(
                SearchResult(source_id=s, source=s, content=s, method='frozen', chunk_id=str(i))
                for i, s in enumerate(['wrong', 'right', 'right'][:top_k])))
    cases = [{'id': 'q1', 'question': 'original question', 'relevance': {'right': 2}}]
    report = evaluate_retrievers({'one': FrozenRetriever(), 'two': FrozenRetriever()}, cases, top_k=3)
    for name in ('one', 'two'):
        row = report['results'][0]['modes'][name]
        assert row['metrics'] == {'recall_at_k': 1., 'mrr': .5, 'ndcg_at_k': pytest.approx(.6309297535714575)}
        assert row['latency_ms'] >= 0
        assert len(row['response']['results']) == 3
        assert report['summary'][name]['mrr'] == .5
