"""Opt-in subprocess CLI lifecycle with real model calls and Qdrant."""

from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from arkb.knowledge.sqlite import SQLiteStorage


pytestmark = pytest.mark.skipif(os.environ.get('ARKB_RUN_MODEL_TESTS') != '1',
                               reason='Enable real model integration tests explicitly.')
pytestmark = [pytest.mark.integration, pytestmark]


def test_new_process_search_and_incremental_cli_lifecycle(tmp_path):
    url = os.environ.get('ARKB_QDRANT_URL')
    if not url:
        pytest.skip('Set ARKB_QDRANT_URL for the real server lifecycle.')
    notes = tmp_path / 'notes'
    notes.mkdir()
    (notes / 'a.md').write_text('# Cache\nStore completed vectors for reuse.')
    (notes / 'b.md').write_text('# Index\nIndex structures accelerate similarity search.')
    db = tmp_path / 'index.sqlite'
    vault = 'test-' + uuid4().hex
    def run(*args):
        result = subprocess.run([sys.executable, '-B', '-m', 'arkb.interfaces.cli', *args,
                                 '--db', str(db), '--vault-id', vault, '--json'],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    index_args = ['index', '--notes-dir', str(notes), '--offline', '--context-length', '512', '--qdrant-url', url]
    try:
        first = run(*index_args)
        assert first['embedded_inputs'] == 2
        second = run(*index_args)
        assert second['reused_index'] and second['embedded_inputs'] == 0
        for mode in ('bm25', 'semantic', 'hybrid'):
            query = run('search', 'Why cache vectors?', '--mode', mode, '--offline', '--source', 'a.md')
            assert query['method'] == mode
            assert query['index_id'] == first['manifest']['index_version']
            assert query['results'][0]['content'] == 'Store completed vectors for reuse.'
        matched = run('match', 'vectors', '--source', 'a.md')
        assert matched['results'][0]['source'] == 'a.md'
        assert run('status')['active_version'] == first['manifest']['index_version']
        # Exercise the production turn budget; exhaustion has separate coverage.
        generated = subprocess.run([sys.executable, '-B', '-m', 'arkb.interfaces.cli',
                                    'ask', 'Read a.md and explain why cache vectors.', '--offline',
                                    '--db', str(db), '--vault-id', vault, '--json', '--trace'],
                                   capture_output=True, text=True, timeout=180)
        assert generated.returncode == 0, generated.stderr
        assert json.loads(generated.stdout)['stop_reason'] == 'final'
        assert 'final' in generated.stderr

        (notes / 'a.md').write_text('# Cache\nOnly changed inputs need new vectors.')
        (notes / 'b.md').unlink()
        stale = run('search', 'vectors', '--mode', 'bm25', '--source', 'a.md')
        assert stale['results'][0]['content'] == 'Store completed vectors for reuse.'
        live = run('match', 'Only changed', '--source', 'a.md')
        assert live['results'][0]['content'] == 'Only changed'
        changed = run(*index_args)
        assert changed['embedded_inputs'] == 1 and changed['deleted_documents'] == 1
        assert changed['modified_documents'] == 1
        assert run('search', 'Index?', '--offline', '--source', 'b.md')['results'] == []
    finally:
        if db.exists():
            with SQLiteStorage(db, read_only=True) as storage, closing(QdrantClient(url=url, timeout=15, trust_env=False)) as client:
                for manifest in storage.list_builds(vault):
                    name = storage.build_metadata(manifest.index_version)['backend'].get('collection')
                    if name and client.collection_exists(name):
                        client.delete_collection(name)
