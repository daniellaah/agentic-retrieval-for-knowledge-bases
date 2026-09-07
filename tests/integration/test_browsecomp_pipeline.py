"""Benchmark integration through real storage, provider HTTP and CLI boundaries."""

from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
from ollama import Client
import pytest
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from obsidian_rag.benchmark import build_benchmark_index, prepare_benchmark, run_benchmark
from obsidian_rag.context import GenerationCounter
from obsidian_rag.schema import EmbeddingSpec
from obsidian_rag.storage import SQLiteStorage


@pytest.fixture
def dataset(tmp_path):
    documents = [
        {'docid': '001/a', 'text': '# Project\r\nThe project code is ORCHID-42.\r\n',
         'url': 'https://example.org/project'},
        {'docid': '002', 'text': 'A distraction about unrelated ocean tides.',
         'url': 'https://example.org/ocean'},
    ]
    corpus, cases = tmp_path / 'corpus.jsonl', tmp_path / 'cases.jsonl'
    corpus.write_text(''.join(json.dumps(doc) + '\n' for doc in documents), encoding='utf-8')
    cases.write_text(json.dumps({'query_id': 'code', 'query': 'What is the project code?',
        'answer': 'ORCHID-42', 'evidence_docids': ['001/a'], 'gold_docids': ['001/a']}) + '\n', encoding='utf-8')
    return corpus, cases, documents


@pytest.mark.parametrize('backend', ['numpy', 'qdrant'])
@pytest.mark.filterwarnings('ignore:Payload indexes have no effect in the local Qdrant:UserWarning')
@pytest.mark.filterwarnings('ignore:Local mode performs exact .* search:UserWarning')
def test_http_provider_to_persistent_benchmark_preserves_docids_and_source_spans(dataset, tmp_path, backend):
    corpus, cases, documents = dataset
    prepared, output, db = tmp_path / 'prepared', tmp_path / 'run', tmp_path / 'index.sqlite'
    prepare_benchmark(corpus, cases, prepared, dataset_revision='synthetic-integration-v1')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[END]': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(single='$A [END]', special_tokens=[('[END]', 1)])
    spec = EmbeddingSpec(model='test', model_revision='fixture', dimensions=2, document_template='title-body-v1')
    requests = []

    def serve(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == '/api/embed':
            return httpx.Response(200, json={'embeddings': [
                [1., 0.] if 'ORCHID' in text or 'Query:' in text else [0., 1.] for text in body['input']]})
        assert request.url.path == '/api/chat'
        assert body['format']['type'] == 'object'
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': json.dumps({
            'status': 'answered', 'claims': [{'text': 'The code is ORCHID-42.', 'source_ids': ['S1']}],
            'missing_information': []})}, 'done_reason': 'stop', 'prompt_eval_count': 100, 'eval_count': 20})

    with ExitStack() as resources:
        client = resources.enter_context(Client(host='http://fixture', transport=httpx.MockTransport(serve), trust_env=False))
        qclient = resources.enter_context(closing(QdrantClient(':memory:'))) if backend == 'qdrant' else None
        options = {'client': client, 'tokenizer': tokenizer, 'spec': spec, 'max_input_tokens': 100,
                   'chunking': 'none', 'backend': {'kind': backend, **({'url': 'local-test'} if qclient else {})},
                   'qdrant_client': qclient}
        with SQLiteStorage(db) as storage:
            built = build_benchmark_index(prepared, storage=storage, **options)
        before = db.read_bytes()
        with SQLiteStorage(db, read_only=True) as storage:
            summary = run_benchmark(prepared, storage=storage, output=output, client=client,
                tokenizer=tokenizer, spec=spec, qdrant_client=qclient, generate=True, chunk_top_k=2,
                context_top_k=1, doc_ks=(1, 5),
                generation_counter=GenerationCounter('test', 'fixture', lambda messages: 100))
        assert db.read_bytes() == before
    row = json.loads((output / 'results.jsonl').read_text())
    assert summary['success_count'] == summary['query_count'] == 1
    assert summary['retrieval']['evidence']['1']['recall'] == 1.
    assert [doc['docid'] for doc in row['documents']] == ['001/a', '002']
    assert row['context_docids'] == row['cited_docids'] == ['001/a']
    assert row['hits'][0]['record']['chunk']['content'] == documents[0]['text']
    source = row['generation']['result']['sources'][0]
    assert documents[0]['text'][source['start_char']:source['end_char']] == source['content']
    assert source['origins'][0]['index_version'] == built.manifest.index_version
    assert len([r for r in requests if r[0] == '/api/chat']) == 1


@pytest.mark.skipif(os.environ.get('OBSIDIAN_RAG_RUN_MODEL_TESTS') != '1',
                   reason='Enable real model tests with Qwen cached and local Ollama running.')
def test_real_benchmark_cli_across_processes_with_regrading_and_failed_output(dataset, tmp_path):
    corpus, cases, _ = dataset
    repo = Path(__file__).resolve().parents[2]

    def run(*args, expected=0):
        process = subprocess.run([sys.executable, '-B', '-m', 'obsidian_rag.benchmark', *map(str, args)],
                                 cwd=repo, capture_output=True, text=True, timeout=180)
        assert process.returncode == expected, process.stderr + process.stdout
        return json.loads(process.stdout)

    prepared, db, output = tmp_path / 'prepared', tmp_path / 'index.sqlite', tmp_path / 'run'
    run('prepare', '--corpus', corpus, '--cases', cases, '--output', prepared,
        '--dataset-revision', 'synthetic-cli-v1')
    index_args = ('index', '--prepared', prepared, '--db', db, '--offline')
    built = run(*index_args)
    repeated = run(*index_args)
    assert repeated['reused_index'] and repeated['embedded_inputs'] == 0
    assert repeated['manifest']['index_version'] == built['manifest']['index_version']
    before = db.read_bytes()
    run_args = ('run', '--prepared', prepared, '--db', db, '--offline', '--chunk-top-k', 2,
                '--context-top-k', 1, '--doc-ks', 1, 5, '--generate')
    summary = run(*run_args, '--output', output)
    assert summary['success_count'] == 1
    assert summary['retrieval']['evidence']['1']['recall'] == 1.
    row = json.loads((output / 'results.jsonl').read_text())
    answer = row['generation']['result']
    assert any('ORCHID-42' in claim['text'] for claim in answer['answer']['claims'])
    assert answer['token_usage']['prompt_tokens'] == answer['token_usage']['actual_prompt_tokens']
    assert row['context_docids'] == row['cited_docids'] == ['001/a']
    assert row['index_version'] == built['manifest']['index_version']
    raw = (output / 'results.jsonl').read_bytes()
    judge_args = ('judge', '--run', output, '--judge-model', 'qwen3.5:4b', '--no-think',
                  '--temperature', 0, '--max-output-tokens', 512)
    judged = run(*judge_args, '--output', tmp_path / 'judged')
    assert judged['graded_count'] == 1 and judged['judge_error_count'] == 0
    run(*judge_args, '--output', tmp_path / 'rejudged')
    report = run('report', '--run', output, '--judgments', tmp_path / 'judged', '--output', tmp_path / 'report')
    assert report['answer_judgments'] == judged
    assert report['semantic_citation_support'] is None
    run('export', '--run', output, '--output', tmp_path / 'export')
    exported = json.loads(next((tmp_path / 'export' / 'runs').glob('*.json')).read_text())
    assert exported['status'] == 'completed' and exported['retrieved_docids'][0] == '001/a'
    assert 'ORCHID-42' in exported['result'][0]['output']
    assert (output / 'results.jsonl').read_bytes() == raw

    failed_dir = tmp_path / 'truncated'
    failed = run(*run_args, '--max-output-tokens', 1, '--output', failed_dir, expected=1)
    assert failed['error_count'] == 1
    failed_row = json.loads((failed_dir / 'results.jsonl').read_text())
    assert failed_row['error']['code'] == 'truncated_output'
    assert failed_row['generation']['raw_response']
    failed_judge = run('judge', '--run', failed_dir, '--output', tmp_path / 'failed-judge',
                       '--judge-model', 'qwen3.5:4b', '--no-think')
    assert failed_judge['pipeline_error_count'] == 1 and failed_judge['graded_count'] == 0
    assert failed_judge['end_to_end_accuracy'] == 0.
    assert db.read_bytes() == before
