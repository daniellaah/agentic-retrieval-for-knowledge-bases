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


@pytest.mark.parametrize('mode', ['valid', 'truncated', 'context_exhausted'])
def test_generation_tracks_retrieved_sent_and_cited_evidence_and_preserves_failures(dataset, tmp_path, runtime, mode):
    from ollama import ChatResponse
    from obsidian_rag.benchmark import build_benchmark_index, run_benchmark
    from obsidian_rag.context import ContextConfig, GenerationCounter
    from obsidian_rag.storage import SQLiteStorage
    corpus, cases = dataset
    prepared = tmp_path / 'prepared'
    prepare_benchmark(corpus, cases, prepared, query_ids=['1'], dataset_revision='fixture-v1')
    raw = json.dumps({'status': 'answered', 'claims': [{'text': 'The code is ORCHID-42.',
                                                     'source_ids': ['S1']}], 'missing_information': []})
    runtime['client'].chat.return_value = ChatResponse(message={'role': 'assistant', 'content': raw},
        done_reason='length' if mode == 'truncated' else 'stop', prompt_eval_count=100, eval_count=20)
    counter = GenerationCounter('test', 'fixture-counter', lambda messages: 100)
    config = ContextConfig(64, 1, 0) if mode == 'context_exhausted' else ContextConfig()
    with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        build_benchmark_index(prepared, storage=storage, **runtime)
        result = run_benchmark(prepared, storage=storage, output=tmp_path / 'run',
            client=runtime['client'], tokenizer=runtime['tokenizer'], spec=runtime['spec'],
            chunk_top_k=3, doc_ks=(1,), generate=True, context_top_k=1,
            generation_counter=counter, context_config=config)
    row = json.loads((tmp_path / 'run' / 'results.jsonl').read_text())
    assert row['retrieval_success'] and len(row['documents']) == 3
    if mode == 'valid':
        assert row['context_docids'] == row['cited_docids'] == ['1']
        assert row['generation']['raw_response'] == raw
        assert row['generation']['metrics']['supported_claim_rate'] is None
        assert result['success_count'] == 1
    elif mode == 'truncated':
        assert row['generation']['raw_response'] == raw
        assert row['error']['code'] == 'truncated_output'
        assert result['error_count'] == 1
    else:
        assert row['error']['stage'] == 'context'
        runtime['client'].chat.assert_not_called()
    for call in runtime['client'].chat.call_args_list:
        assert 'LABEL-ONLY' not in json.dumps(call.kwargs['messages'])


@pytest.fixture
def completed_run(dataset, tmp_path, runtime, request):
    from ollama import ChatResponse
    from obsidian_rag.benchmark import build_benchmark_index, run_benchmark
    from obsidian_rag.context import ContextConfig, GenerationCounter
    from obsidian_rag.storage import SQLiteStorage
    prepared, output = tmp_path / 'prepared', tmp_path / 'run'
    prepare_benchmark(*dataset, prepared, dataset_revision='fixture-v1')
    status = getattr(request, 'param', 'answered')
    claims = [] if status == 'insufficient_evidence' else [{'text': 'The code is ORCHID-42.', 'source_ids': ['S1']}]
    raw = json.dumps({'status': status, 'claims': claims,
                      'missing_information': [] if status == 'answered' else ['An additional detail is not established.']})
    runtime['client'].chat.return_value = ChatResponse(message={'role': 'assistant', 'content': raw},
                                                      done_reason='stop', prompt_eval_count=100, eval_count=20)
    with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        build_benchmark_index(prepared, storage=storage, **runtime)
        run_benchmark(prepared, storage=storage, output=output, client=runtime['client'],
                      tokenizer=runtime['tokenizer'], spec=runtime['spec'], generate=True,
                      generation_counter=GenerationCounter('test', 'fixture', lambda m: 100),
                      context_config=ContextConfig(), doc_ks=(1, 5))
    return prepared, output, runtime['client']


def test_saved_answers_can_be_rejudged_without_regeneration_and_judge_errors_stay_visible(completed_run, tmp_path):
    from unittest.mock import Mock
    from ollama import Client, ChatResponse
    from obsidian_rag.benchmark import judge_run
    from obsidian_rag.judging import JudgeConfig
    prepared, run, generator = completed_run
    before, calls = (run / 'results.jsonl').read_bytes(), generator.chat.call_count
    judge = Mock(spec=Client)
    good = ChatResponse(message={'role': 'assistant', 'content':
        'extracted_final_answer: ORCHID-42\nreasoning: Equivalent.\ncorrect: yes'}, done_reason='stop')
    judge.chat.side_effect = [good, OSError('judge offline')]
    summary = judge_run(run, tmp_path / 'judged', client=judge,
                        config=JudgeConfig('fixture-judge', 'fixed'))
    assert summary['correct_count'] == summary['judge_error_count'] == 1
    assert summary['end_to_end_accuracy'] is None
    assert (summary['accuracy_lower_bound'], summary['accuracy_upper_bound']) == (.5, 1.)
    assert (run / 'results.jsonl').read_bytes() == before
    assert generator.chat.call_count == calls
    assert all('An unrelated document.' not in str(call.kwargs['messages']) for call in judge.chat.call_args_list)
    judge.chat.side_effect = None
    judge.chat.return_value = good
    again = judge_run(run, tmp_path / 'rejudged', client=judge, config=JudgeConfig('fixture-judge', 'fixed'))
    assert again['end_to_end_accuracy'] == 1.


@pytest.mark.parametrize('completed_run', ['answered', 'partial', 'insufficient_evidence'], indirect=True)
def test_offline_report_and_official_export_preserve_execution_status_and_docids(completed_run, tmp_path):
    from obsidian_rag.benchmark import report_run, export_run
    prepared, run, _ = completed_run
    before = (run / 'results.jsonl').read_bytes()
    report = report_run(run, tmp_path / 'report', doc_ks=(1,))
    assert report['query_count'] == 2
    assert report['retrieval']['evidence']['1']['recall'] == 1.
    assert report['semantic_citation_support'] is None
    exported = export_run(run, tmp_path / 'export')
    files = sorted((tmp_path / 'export' / 'runs').glob('*.json'))
    payloads = [json.loads(p.read_text()) for p in files]
    assert {p['query_id'] for p in payloads} == {'1', '2'}
    assert all(p['status'] == 'completed' and p['retrieved_docids'] == ['1', '2'] for p in payloads)
    assert all('Sources (Note.content' not in p['result'][0]['output'] for p in payloads)
    assert exported['query_count'] == 2
    assert (run / 'results.jsonl').read_bytes() == before
    with pytest.raises(FileExistsError):
        report_run(run, tmp_path / 'report')


def test_cli_prepares_and_reports_offline_and_rejects_invalid_arguments(dataset, completed_run, tmp_path, capsys):
    from obsidian_rag.benchmark import main
    corpus, cases = dataset
    prepared, run, _ = completed_run
    target = tmp_path / 'cli-prepared'
    assert main(['prepare', '--corpus', str(corpus), '--cases', str(cases), '--output', str(target),
                 '--dataset-revision', 'fixture-v1', '--query-ids', '1']) == 0
    assert json.loads(capsys.readouterr().out)['query_ids'] == ['1']
    report = tmp_path / 'cli-report'
    args = ['report', '--run', str(run), '--output', str(report)]
    assert main(args) == 0
    capsys.readouterr()
    before = (report / 'metrics.json').read_bytes()
    assert main(args) == 1
    assert (report / 'metrics.json').read_bytes() == before
    with pytest.raises(SystemExit) as error:
        main(['run', '--prepared', str(prepared), '--db', str(tmp_path / 'index.sqlite'),
              '--output', str(tmp_path / 'invalid'), '--chunk-top-k', '0'])
    assert error.value.code == 2


def test_offline_scoring_rejects_incomplete_or_changed_raw_results(completed_run, tmp_path):
    from obsidian_rag.benchmark import report_run
    _, run, _ = completed_run
    with (run / 'results.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='hash'):
        report_run(run, tmp_path / 'report')
