from dataclasses import asdict
import json

import pytest

from obsidian_rag.benchmark import prepare_benchmark, load_prepared


@pytest.fixture
def runtime():
    from unittest.mock import Mock
    from ollama import Client, EmbedResponse
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from obsidian_rag.schema import EmbeddingSpec
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[END]': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(single='$A [END]', special_tokens=[('[END]', 1)])
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[
        [1., 0.] if 'ORCHID' in text or 'Query:' in text else [0., 1.] for text in kw['input']])
    spec = EmbeddingSpec(model='test', model_revision='fixture', dimensions=2, document_template='title-body-v1')
    return dict(client=client, tokenizer=tokenizer, spec=spec, max_input_tokens=100, chunking='none')


@pytest.fixture
def dataset(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    cases = tmp_path / 'cases.jsonl'
    docs = [{'docid': str(i), 'text': text, 'url': f'https://example.org/{i}'} for i, text in
            enumerate(['The project code is ORCHID-42.', 'An unrelated document.', 'Another distraction.'], 1)]
    corpus.write_text(''.join(json.dumps(doc) + '\n' for doc in docs))
    cases.write_text(''.join(json.dumps({'query_id': str(i), 'query': question, 'answer': 'LABEL-ONLY',
                                       'evidence_docids': ['1'], 'gold_docids': ['1']}) + '\n'
                             for i, question in enumerate(['What is the project code?', 'Broken?'], 1)))
    return corpus, cases


def test_preparation_freezes_questions_and_mapping_but_keeps_labels_separate(dataset, tmp_path):
    corpus, cases = dataset
    out = tmp_path / 'prepared'
    manifest = prepare_benchmark(corpus, cases, out, query_ids=['1'], dataset_revision='fixture-v1')
    loaded, questions, sources = load_prepared(out)
    assert loaded == manifest
    assert [asdict(q) for q in questions] == [{'query_id': '1', 'question': 'What is the project code?'}]
    assert {v['docid'] for v in sources.values()} == {'1', '2', '3'}
    assert manifest['corpus_count'] == 3 and manifest['query_ids'] == ['1']
    assert 'LABEL-ONLY' not in (out / 'questions.jsonl').read_text()
    with pytest.raises(FileExistsError):
        prepare_benchmark(corpus, cases, out, dataset_revision='fixture-v1')
    (out / 'questions.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash'):
        load_prepared(out)


def test_benchmark_index_reuses_cache_and_rejects_changed_corpus(dataset, tmp_path, runtime):
    from obsidian_rag.benchmark import build_benchmark_index
    from obsidian_rag.storage import SQLiteStorage
    corpus, cases = dataset
    prepared = tmp_path / 'prepared'
    manifest = prepare_benchmark(corpus, cases, prepared, dataset_revision='fixture-v1')
    with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        first = build_benchmark_index(prepared, storage=storage, **runtime)
        again = build_benchmark_index(prepared, storage=storage, **runtime)
        assert first.manifest.vault_id == manifest['vault_id']
        assert first.manifest.document_count == 3
        assert again.reused_index and again.embedded_inputs == 0
        assert storage.active_manifest('default') is None
        corpus.write_text(corpus.read_text() + '\n')
        with pytest.raises(ValueError, match='corpus.*hash'):
            build_benchmark_index(prepared, storage=storage, **runtime)
        assert storage.active_manifest(manifest['vault_id']) == first.manifest


def test_retrieval_run_records_failures_and_scores_all_questions_without_index_writes(dataset, tmp_path, runtime):
    from obsidian_rag.benchmark import build_benchmark_index, run_benchmark
    from obsidian_rag.storage import SQLiteStorage
    corpus, cases = dataset
    prepared = tmp_path / 'prepared'
    prepare_benchmark(corpus, cases, prepared, dataset_revision='fixture-v1')
    db = tmp_path / 'index.sqlite'
    with SQLiteStorage(db) as storage:
        built = build_benchmark_index(prepared, storage=storage, **runtime)
    embed = runtime['client'].embed.side_effect
    def sometimes_fails(**kw):
        if any('Broken?' in text for text in kw['input']):
            raise OSError('fixture unavailable')
        return embed(**kw)
    runtime['client'].embed.side_effect = sometimes_fails
    before = db.read_bytes()
    output = tmp_path / 'run'
    with SQLiteStorage(db, read_only=True) as storage:
        summary = run_benchmark(prepared, storage=storage, output=output,
                                client=runtime['client'], tokenizer=runtime['tokenizer'], spec=runtime['spec'],
                                chunk_top_k=2, doc_ks=(1, 5))
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    assert [r['query_id'] for r in rows] == ['1', '2']
    assert rows[0]['documents'][0]['docid'] == '1'
    assert rows[0]['index_version'] == built.manifest.index_version
    assert rows[1]['error']['stage'] == 'retrieval'
    assert summary['error_count'] == 1
    assert summary['retrieval']['evidence']['1']['recall'] == .5
    assert summary['retrieval']['evidence']['1']['defined_cases'] == 2
    assert db.read_bytes() == before
    for call in runtime['client'].embed.call_args_list:
        assert all('LABEL-ONLY' not in text for text in call.kwargs['input'])
