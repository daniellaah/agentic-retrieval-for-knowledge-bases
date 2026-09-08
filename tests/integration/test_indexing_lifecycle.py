"""Opt-in subprocess CLI lifecycle with real model calls and optional Qdrant."""

from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from obsidian_rag.storage import SQLiteStorage


pytestmark = pytest.mark.skipif(os.environ.get('OBSIDIAN_RAG_RUN_MODEL_TESTS') != '1',
                               reason='Enable real model integration tests explicitly.')


@pytest.mark.parametrize('backend', ['numpy', 'qdrant'])
def test_new_process_query_and_incremental_cli_lifecycle(tmp_path, backend):
    url = os.environ.get('OBSIDIAN_RAG_QDRANT_URL')
    if backend == 'qdrant' and not url:
        pytest.skip('Set OBSIDIAN_RAG_QDRANT_URL for the real server lifecycle.')
    notes = tmp_path / 'notes'
    notes.mkdir()
    (notes / 'a.md').write_text('# Cache\nStore completed vectors for reuse.')
    (notes / 'b.md').write_text('# Index\nIndex structures accelerate similarity search.')
    db = tmp_path / 'index.sqlite'
    vault = 'test-' + uuid4().hex
    def run(*args):
        result = subprocess.run([sys.executable, '-B', '-m', 'obsidian_rag.cli', *args,
                                 '--db', str(db), '--vault-id', vault],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    index_args = ['index', '--notes-dir', str(notes), '--offline', '--context-length', '512', '--backend', backend]
    if backend == 'qdrant':
        index_args += ['--qdrant-url', url]
    try:
        first = run(*index_args)
        assert first['embedded_inputs'] == 2
        second = run(*index_args)
        assert second['reused_index'] and second['embedded_inputs'] == 0
        query = run('query', 'Why cache vectors?', '--offline', '--json', '--source', 'a.md')
        assert query['index_version'] == first['manifest']['index_version']
        assert query['results'][0]['chunk']['content'] == 'Store completed vectors for reuse.'
        context = run('query', 'Why cache vectors?', '--offline', '--show-context', '--source', 'a.md')
        assert context['status'] == 'ready'
        assert context['evidence_blocks'][0]['origins'][0]['index_version'] == first['manifest']['index_version']
        assert context['token_usage']['is_estimate'] is False
        assert context['token_usage']['prompt_tokens'] <= context['token_usage']['input_budget']
        generated = subprocess.run([sys.executable, '-B', '-m', 'obsidian_rag.cli',
                                    'query', 'Why cache vectors?', '--offline', '--source', 'a.md',
                                    '--db', str(db), '--vault-id', vault, '--max-output-tokens', '128'],
                                   capture_output=True, text=True, timeout=180)
        assert generated.returncode == 0, generated.stderr
        assert generated.stdout.strip()

        (notes / 'a.md').write_text('# Cache\nOnly changed inputs need new vectors.')
        (notes / 'b.md').unlink()
        changed = run(*index_args)
        assert changed['embedded_inputs'] == 1 and changed['deleted_documents'] == 1
        assert changed['modified_documents'] == 1
        assert run('query', 'Index?', '--offline', '--json', '--source', 'b.md')['results'] == []
    finally:
        if backend == 'qdrant' and db.exists():
            with SQLiteStorage(db, read_only=True) as storage, closing(QdrantClient(url=url, timeout=15, trust_env=False)) as client:
                for manifest in storage.list_builds(vault):
                    name = storage.build_metadata(manifest.index_version)['backend'].get('collection')
                    if name and client.collection_exists(name):
                        client.delete_collection(name)
