"""Opt-in schema generation and exact serving-token checks, without indexing."""

import os
import json
from pathlib import Path
import subprocess
import sys

from ollama import Client
import pytest

from obsidian_rag.knowledge_base.chunking import whole_note_chunks
from obsidian_rag.context import ContextConfig, build_context, load_generation_counter
from obsidian_rag.generation import CitedGenerationError, generate_cited_answer
from obsidian_rag.knowledge_base.loaders import Note
from tests.result_fixtures import make_result as SearchResult, chunk_of, record_of


pytestmark = pytest.mark.skipif(os.environ.get('OBSIDIAN_RAG_RUN_MODEL_TESTS') != '1',
                              reason='Set OBSIDIAN_RAG_RUN_MODEL_TESTS=1 with Qwen cached and Ollama running.')


@pytest.mark.parametrize('question,body', [
    ('What is the project code?', 'The project code is ORCHID-42.'),
    ('项目代号是什么？', '项目代号是 ORCHID-42。原样字符：e\u0301 🧠。'),
])
def test_real_citation_schema_and_message_counts(question, body):
    with Client(host='http://127.0.0.1:11434', timeout=120, trust_env=False) as client:
        counter = load_generation_counter(client=client, local_files_only=True)
        hits = [SearchResult(whole_note_chunks([Note('Project', body, 'project.md')])[0], .9)]
        context = build_context(question, hits, config=ContextConfig(), counter=counter, citation_mode='structured')
        result = generate_cited_answer(context, client=client)
    assert result.actual_prompt_tokens == context.prompt_tokens
    assert result.answer.status == 'answered'
    assert any('ORCHID-42' in c.text for c in result.answer.claims)
    assert all(c.source_ids == ('S1',) for c in result.answer.claims)


def test_real_output_truncation_is_reported():
    with Client(host='http://127.0.0.1:11434', timeout=120, trust_env=False) as client:
        counter = load_generation_counter(client=client, local_files_only=True)
        hits = [SearchResult(whole_note_chunks([Note('Project', 'The code is ORCHID-42.', 'project.md')])[0], .9)]
        context = build_context('What is the code?', hits, config=ContextConfig(max_output_tokens=1),
                                counter=counter, citation_mode='structured')
        with pytest.raises(CitedGenerationError) as error:
            generate_cited_answer(context, client=client)
    assert error.value.code == 'truncated_output'


def test_citation_cli_shorthand_and_persistent_subprocess(tmp_path, citation_qdrant):
    notes = tmp_path / 'notes'
    notes.mkdir()
    (notes / 'project.md').write_text('# Project\nThe project code is ORCHID-42.\n')
    db = tmp_path / 'index.sqlite'
    repo = Path(__file__).resolve().parents[2]

    def run(*args):
        process = subprocess.run([sys.executable, '-B', '-m', 'obsidian_rag.cli', *args],
                                 cwd=repo, capture_output=True, text=True, timeout=180)
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)

    indexed = run('index', '--db', str(db), '--notes-dir', str(notes), '--offline', '--qdrant-url', citation_qdrant)
    shorthand = run('What is the project code?', '--db', str(db), '--offline', '--answer-json')
    assert shorthand['sources'][0]['origins'][0]['index_version'] == indexed['manifest']['index_version']
    (notes / 'project.md').write_text('# Project\nThe live file has changed.\n')
    saved = run('query', 'What is the project code?', '--db', str(db), '--offline', '--answer-json')
    assert 'ORCHID-42' in saved['text']
    assert saved['sources'][0]['origins'][0]['index_version'] == indexed['manifest']['index_version']
    retrieved = run('query', 'What is the project code?', '--db', str(db), '--offline', '--json')
    assert 'items' in retrieved and 'answer' not in retrieved


def test_citation_evaluation_cli_records_results_without_modifying_index(tmp_path, citation_qdrant):
    notes = tmp_path / 'notes'
    notes.mkdir()
    (notes / 'project.md').write_text('# Project\nThe project code is ORCHID-42.\n')
    db = tmp_path / 'index.sqlite'
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'code', 'question': 'What is the project code?',
                                 'required_source_groups': [['project.md']]}) + '\n')
    out = tmp_path / 'evaluation'
    repo = Path(__file__).resolve().parents[2]
    built = subprocess.run([sys.executable, '-B', '-m', 'obsidian_rag.cli', 'index',
                            '--db', str(db), '--notes-dir', str(notes), '--offline', '--qdrant-url', citation_qdrant],
                           cwd=repo, capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stderr
    before = db.read_bytes()
    command = [sys.executable, '-B', '-m', 'obsidian_rag.evaluation', '--db', str(db),
               '--cases', str(cases), '--output', str(out), '--offline', '--context', '--citations']
    process = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stderr
    report = json.loads((out / 'metrics.json').read_text())
    assert set(report['summary']) == {'qdrant_exact', 'qdrant_ann'}
    assert report['settings']['reference'] == 'qdrant_exact'
    assert report['citations']['summary']['success_count'] == 1
    assert report['citations']['summary']['supported_claim_rate'] is None
    row = json.loads((out / 'citation_results.jsonl').read_text())
    assert row['raw_response'] and row['context']['citation_sources']
    assert db.read_bytes() == before
    preserved = (out / 'citation_results.jsonl').read_bytes()
    repeated = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 2
    assert (out / 'citation_results.jsonl').read_bytes() == preserved


def test_real_quoted_generation_preserves_unicode_and_computes_offsets():
    body = 'Prefix. The code is ORCHID-42. Unicode: e\u0301 🧠.'
    with Client(host='http://127.0.0.1:11434', timeout=120, trust_env=False) as client:
        counter = load_generation_counter(client=client, local_files_only=True)
        hits = [SearchResult(whole_note_chunks([Note('Project', body, 'project.md')])[0], .9)]
        context = build_context('What is the code?', hits, config=ContextConfig(), counter=counter, citation_mode='quoted')
        result = generate_cited_answer(context, client=client)
    assert result.validation.quotes
    assert result.actual_prompt_tokens == context.prompt_tokens
    for quote in result.validation.quotes:
        assert body[quote.start_char:quote.end_char] == quote.text
    assert 'ORCHID-42' in result.text


@pytest.fixture
def citation_qdrant(tmp_path):
    from contextlib import closing
    from qdrant_client import QdrantClient
    from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage
    url = os.environ.get('OBSIDIAN_RAG_QDRANT_URL')
    if not url:
        pytest.skip('Set OBSIDIAN_RAG_QDRANT_URL for persistent CLI tests.')
    try:
        yield url
    finally:
        db = tmp_path / 'index.sqlite'
        if db.exists():
            with SQLiteStorage(db, read_only=True) as storage, closing(QdrantClient(url=url, trust_env=False)) as client:
                for manifest in storage.list_builds('default'):
                    name = storage.build_metadata(manifest.index_version)['backend'].get('collection')
                    if name and client.collection_exists(name):
                        client.delete_collection(name)
