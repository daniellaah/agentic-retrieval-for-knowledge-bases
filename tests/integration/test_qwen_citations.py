"""Opt-in schema generation and exact serving-token checks, without indexing."""

import os
import json
from pathlib import Path
import subprocess
import sys

from ollama import Client
import pytest

from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.context import ContextConfig, build_context, load_generation_counter
from obsidian_rag.generation import CitedGenerationError, generate_cited_answer
from obsidian_rag.loaders import Note
from obsidian_rag.retrieval import SearchResult


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


def test_citation_cli_memory_and_persistent_subprocess(tmp_path):
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

    memory = run('What is the project code?', '--notes-dir', str(notes), '--offline', '--answer-json')
    assert memory['sources'][0]['origins'][0]['index_version'].startswith('memory:')
    indexed = run('index', '--db', str(db), '--notes-dir', str(notes), '--offline')
    (notes / 'project.md').write_text('# Project\nThe live file has changed.\n')
    saved = run('query', 'What is the project code?', '--db', str(db), '--offline', '--answer-json')
    assert 'ORCHID-42' in saved['text']
    assert saved['sources'][0]['origins'][0]['index_version'] == indexed['manifest']['index_version']
    retrieved = run('query', 'What is the project code?', '--db', str(db), '--offline', '--json')
    assert 'results' in retrieved and 'answer' not in retrieved
