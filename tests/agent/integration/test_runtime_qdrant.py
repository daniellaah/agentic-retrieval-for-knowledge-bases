"""Real indexing, persisted retrieval, and agent conversations; no fake clients."""

from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from ollama import Client

from arkb.config import DEFAULT_AGENT_THINK, DEFAULT_GENERATION_MODEL, RuntimeConfig
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.runtime import Runtime


pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get('ARKB_RUN_MODEL_TESTS') != '1',
    reason='Enable real model integration tests explicitly.')]

MATERIAL_QUERY = (
    'Find material for an article about Agent Memory. Read candidate notes and check their linked policy. '
    'Report the episodic-memory retention period and experiment verification code, with supporting filenames.'
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def trajectory_metrics(messages):
    """Record tool work and source/snippet gains relative to previous searches.

    These are diagnostic counts, not semantic novelty or answer-quality scores.
    """
    calls = [call for message in messages for call in message.get('tool_calls', [])]
    names = [call['function']['name'] for call in calls]
    functions = iter(call['function'] for call in calls)
    sources_seen, snippets_seen, steps = set(), set(), []
    for message in messages:
        if message['role'] != 'tool':
            continue
        function = next(functions)
        if function['name'] != 'search':
            continue
        hits = json.loads(message['content'])['results']
        sources = {hit['source'] for hit in hits}
        snippets = {(hit['document_id'], hit['chunk_id'], hit['start_char'], hit['end_char'], hit['content'])
                    for hit in hits}
        steps.append({'arguments': function['arguments'], 'result_count': len(hits),
                      'sources': sorted(sources), 'new_sources': len(sources - sources_seen),
                      'new_snippets': len(snippets - snippets_seen)})
        sources_seen.update(sources)
        snippets_seen.update(snippets)
    return {'tool_calls': len(calls), 'search_calls': names.count('search'), 'read_calls': names.count('read'),
            'searches_without_new_snippets': sum(step['new_snippets'] == 0 for step in steps),
            'search_steps': steps}


@pytest.fixture(scope='module')
def persisted_knowledge(tmp_path_factory):
    url = os.environ.get('ARKB_QDRANT_URL')
    if not url:
        pytest.skip('Set ARKB_QDRANT_URL to a real Qdrant Server.')
    output = os.environ.get('ARKB_AGENT_REPORT_DIR')
    if output:
        root = Path(output).resolve()
        root.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous run.
    else:
        root = tmp_path_factory.mktemp('agent-runtime')
    notes = root / 'notes'
    notes.mkdir()
    db = root / 'index.sqlite'
    vault = 'agent-runtime-' + uuid4().hex
    code = 'MEM-' + uuid4().hex[:10].upper()
    documents = {
        'rag.md': '# RAG\nRAG grounds answers in retrieved notes. Retrieval and generation run separately.',
        'architecture.md': '# Retrieval architecture\nRAG can combine keyword and vector retrieval. Read the source before answering.',
        'memory.md': '# Agent Memory\n'
                     'Agent Memory includes working, episodic and procedural memory. '
                     'These notes support articles about agent memory design.\n\n'
                     '## Working memory\nWorking memory holds temporary task information and can be released after the task.\n\n'
                     '## Episodic memory\nEpisodic memory records past events; it does not retain every conversation forever. '
                     'The retention period and verification code are in the linked policy, not this note.\n\n'
                     '## Procedural memory\nProcedural memory stores reusable steps; updates require version tracking.\n\n'
                     '## Linked policy\nThis experiment uses [Memory archive policy](memory-policy.md).',
        'memory-policy.md': '# Memory archive policy\n'
                            'This policy applies only to the local Agent Memory integration experiment.\n\n'
                            '## Retention policy\nRetain episodic memory for 30 days, after which raw records are deleted.\n\n'
                            f'## Experiment verification\nThe verification code for this experiment is {code}. Return the verification code exactly.',
        'cache.md': '# Vector cache\nCaching computed embedding reduces repeated computation. The cache does not determine episodic-memory retention.',
    }
    for source, content in documents.items():
        (notes / source).write_text(content, encoding='utf-8')

    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'database': str(db),
              'notes_directory': str(notes), 'vault_id': vault, 'qdrant_url': url,
              'agent_model': os.environ.get('ARKB_AGENT_MODEL', DEFAULT_GENERATION_MODEL),
              'default_think': DEFAULT_AGENT_THINK,
              'retrieval_mode': 'hybrid', 'expected_code': code, 'collections_cleaned': []}
    try:
        command = [sys.executable, '-B', '-m', 'arkb.interfaces.cli', 'index', '--json',
                   '--notes-dir', str(notes), '--db', str(db), '--vault-id', vault,
                   '--qdrant-url', url, '--offline', '--context-length', '1024',
                   '--chunk-size', '128', '--chunk-overlap', '16', '--batch-size', '4']
        started = perf_counter()
        indexed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        report.update(index_command=command, index_seconds=perf_counter() - started,
                      index_stdout=indexed.stdout, index_stderr=indexed.stderr)
        assert indexed.returncode == 0, indexed.stderr
        report['build'] = json.loads(indexed.stdout)
        assert report['build']['manifest']['status'] == 'ready'
        assert report['build']['manifest']['document_count'] == len(documents)
        assert report['build']['embedded_inputs'] > 0

        # Reopen everything after the indexing subprocess has exited. This
        # verifies persistence, not reuse of an in-memory engine or writer.
        with SQLiteStorage(db, read_only=True) as storage, closing(
                QdrantClient(url=url, timeout=15, trust_env=False)) as client:
            manifest = storage.active_manifest(vault)
            backend = storage.build_metadata(manifest.index_version)['backend']
            assert backend['kind'] == 'qdrant'
            report['collection'] = backend['collection']
            report['qdrant_version'] = client.info().version
            report['point_count'] = client.count(backend['collection'], exact=True).count
            assert report['point_count'] == manifest.chunk_count > 0
        report['database_sha256'] = hashlib.sha256(db.read_bytes()).hexdigest()
        write_json(root / 'index-report.json', report)
        yield {'root': root, 'db': db, 'notes': notes, 'vault': vault, 'url': url,
               'code': code, 'model': report['agent_model'], 'report': report}
        assert hashlib.sha256(db.read_bytes()).hexdigest() == report['database_sha256']
    finally:
        # Delete only collections recorded by this run's unique vault/database.
        if db.exists():
            with SQLiteStorage(db, read_only=True) as storage, closing(
                    QdrantClient(url=url, timeout=15, trust_env=False)) as client:
                for manifest in storage.list_builds(vault):
                    collection = storage.build_metadata(manifest.index_version)['backend'].get('collection')
                    if collection and client.collection_exists(collection):
                        client.delete_collection(collection)
                        report['collections_cleaned'].append(collection)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(root / 'index-report.json', report)


@pytest.mark.parametrize('case,query,first_tool', [
    ('A-match', 'Which notes mention RAG', 'match'),
    ('B-search', 'Which notes relate to RAG', 'search'),
    ('C-read', 'Read rag.md', 'read'),
    ('D-materials', MATERIAL_QUERY, None),
    ('E-direct', 'Hello', None),
    ('F-limit', MATERIAL_QUERY, None),
])
def test_real_runtime_with_persisted_hybrid_retrieval(persisted_knowledge, case, query, first_tool):
    data = persisted_knowledge
    max_turns = 1 if case == 'F-limit' else 8
    report = {'case': case, 'query': query, 'max_turns': max_turns,
              'think': DEFAULT_AGENT_THINK, 'checks_passed': False}
    started = perf_counter()
    try:
        with Runtime(RuntimeConfig(offline=True, timeout=120, qdrant_url=data['url'])) as runtime, \
                SQLiteStorage(data['db'], read_only=True) as storage:
            manifest = storage.active_manifest(data['vault'])
            report['index_version'] = manifest.index_version
            engine = runtime.retrieval_engine(storage, manifest, modes=('hybrid',))
            tools = runtime.agent_tools(engine=engine, directory=data['notes'],
                                        vault_id=data['vault'], mode='hybrid')
            result = runtime.run_agent(query, tools=tools, model=data['model'], max_turns=max_turns)
        report['result'] = asdict(result)
        report['metrics'] = trajectory_metrics(result.state.messages)
        names = [call['function']['name'] for call in result.state.tool_calls]
        report['trajectory'] = names
        observations = [m for m in result.state.messages if m['role'] == 'tool']
        assert [m['tool_name'] for m in observations] == names
        assert result.state.turn <= max_turns
        if case == 'F-limit':
            assert result.stop_reason == 'max_turns'
            assert result.response is None and result.state.turn == 1
            assert names and observations
        else:
            assert result.stop_reason == 'final', names
            assert result.response and result.response.strip()
            if first_tool:
                assert names and names[0] == first_tool
            if case == 'D-materials':
                # Allow any ordering/additional useful calls, but require the
                # real model to expand context and use evidence it cannot know.
                assert 'search' in names and 'read' in names
                assert result.state.turn >= 3
                assert data['code'] in result.response
                assert '30' in result.response or 'thirty' in result.response
                assert any(data['code'] in m['content'] for m in observations)
            if case == 'E-direct':
                assert names == [] and result.state.turn == 1
        report['checks_passed'] = True
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['elapsed_seconds'] = perf_counter() - started
        write_json(data['root'] / f'{case}.json', report)
        print(json.dumps({'case': case, 'trajectory': report.get('trajectory'),
                          'checks_passed': report['checks_passed'],
                          'seconds': round(report['elapsed_seconds'], 2)}, ensure_ascii=False))


@pytest.mark.parametrize('case,query,max_turns,expected_exit,flags', [
    ('CLI-materials', MATERIAL_QUERY, 8, 0, []),
    ('CLI-limit', 'Read rag.md and explain RAG using the note.', 1, 1, ['--think']),
    ('CLI-limit-no-think', 'Read rag.md and explain RAG using the note.', 1, 1, ['--no-think']),
])
def test_real_ask_cli_uses_runtime_entry_point(persisted_knowledge, case, query, max_turns, expected_exit, flags):
    data = persisted_knowledge
    command = [sys.executable, '-B', '-m', 'arkb.interfaces.cli', 'ask', query,
               '--db', str(data['db']), '--vault-id', data['vault'], '--offline',
               '--generation-model', data['model'], '--max-turns', str(max_turns), '--json', '--trace', *flags]
    started = perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
    report = {'command': command, 'returncode': completed.returncode, 'stdout': completed.stdout,
              'stderr': completed.stderr, 'elapsed_seconds': perf_counter() - started}
    write_json(data['root'] / f'{case}.json', report)
    assert completed.returncode == expected_exit, completed.stderr
    result = json.loads(completed.stdout)
    report['metrics'] = trajectory_metrics(result['state']['messages'])
    report['think'] = False if '--no-think' in flags else DEFAULT_AGENT_THINK
    write_json(data['root'] / f'{case}.json', report)
    calls = [call for message in result['state']['messages'] if message['role'] == 'assistant'
             for call in message.get('tool_calls', [])]
    names = [call['function']['name'] for call in calls]
    assert names and f'[1] {names[0]}' in completed.stderr
    if max_turns == 1:
        assert result['response'] is None and result['stop_reason'] == 'max_turns'
        assert 'without a final response' in completed.stderr
    else:
        assert result['stop_reason'] == 'final'
        assert 'search' in names and 'read' in names
        assert data['code'] in result['response']
        assert '30' in result['response'] or 'thirty' in result['response']
    assert result['state']['turn'] <= max_turns


@pytest.mark.parametrize('repeat', [1, 2])
@pytest.mark.parametrize('case,query', [('material', MATERIAL_QUERY), ('related', 'Which notes relate to RAG')])
@pytest.mark.parametrize('think', [False, True])
def test_thinking_configuration_quality_and_work(persisted_knowledge, repeat, case, query, think):
    """Compare the public think setting on the same real hybrid index.

    Non-thinking is a measured control, not a requirement to reproduce failure.
    Preserve all observations and work counts so passing protocol checks cannot
    be confused with task completeness or efficient retrieval.
    """
    data = persisted_knowledge
    report = {'case': case, 'repeat': repeat, 'think': think, 'query': query, 'checks_passed': False}
    requests = []

    def capture_request(request):
        body = json.loads(request.content)
        if request.url.path == '/api/chat':
            requests.append({'think': body['think'], 'model': body['model'],
                             'options': body['options'], 'message_count': len(body['messages'])})

    started = perf_counter()
    try:
        with Runtime(RuntimeConfig(offline=True, timeout=120, qdrant_url=data['url'])) as runtime, \
                SQLiteStorage(data['db'], read_only=True) as storage, \
                Client(host='http://127.0.0.1:11434', timeout=120, trust_env=False,
                       event_hooks={'request': [capture_request]}) as client:
            manifest = storage.active_manifest(data['vault'])
            engine = runtime.retrieval_engine(storage, manifest, modes=('hybrid',))
            tools = runtime.agent_tools(engine=engine, directory=data['notes'], vault_id=data['vault'], mode='hybrid')
            result = runtime.run_agent(query, tools=tools, client=client, model=data['model'], think=think)
        report['result'] = asdict(result)
        report['metrics'] = trajectory_metrics(result.state.messages)
        assert result.stop_reason == 'final' and result.response
        assert result.state.turn == len(requests) <= 8
        assert requests and all(request['think'] is think for request in requests)
        if case == 'material':
            report['facts_complete'] = (data['code'] in result.response
                                        and ('30' in result.response or 'thirty' in result.response))
            if think:
                assert report['facts_complete'], 'Thinking run must answer both requested facts.'
                assert any(data['code'] in m.get('content', '') for m in result.state.messages if m['role'] == 'tool')
        report['checks_passed'] = True
    finally:
        report['requests'] = requests
        report['elapsed_seconds'] = perf_counter() - started
        write_json(data['root'] / f'think-{case}-{think}-{repeat}.json', report)
        row = {key: value for key, value in report.items() if key not in {'query', 'result', 'requests', 'metrics'}}
        row['metrics'] = {key: value for key, value in report.get('metrics', {}).items() if key != 'search_steps'}
        print(json.dumps(row, ensure_ascii=False))
