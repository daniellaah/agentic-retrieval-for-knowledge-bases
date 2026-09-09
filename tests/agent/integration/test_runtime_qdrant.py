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
    os.environ.get('OBSIDIAN_RAG_RUN_MODEL_TESTS') != '1',
    reason='Enable real model integration tests explicitly.')]

MATERIAL_QUERY = (
    '帮我找一些写 Agent Memory 的素材。请打开候选笔记阅读全文，核对其中关联的规范，'
    '说明事件记忆的保留期限和本次实验验收口令，并列出依据文件名。'
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
    url = os.environ.get('OBSIDIAN_RAG_QDRANT_URL')
    if not url:
        pytest.skip('Set OBSIDIAN_RAG_QDRANT_URL to a real Qdrant Server.')
    output = os.environ.get('OBSIDIAN_RAG_AGENT_REPORT_DIR')
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
        'rag.md': '# RAG\nRAG 使用检索到的笔记为回答提供依据。检索与生成分别执行。',
        'architecture.md': '# 检索架构\nRAG 可以结合关键词召回与向量召回，读取原文后再回答。',
        'memory.md': '# Agent Memory\n'
                     'Agent Memory 包括工作记忆、事件记忆和程序性记忆。'
                     '这些材料可以用于撰写智能体记忆设计文章。\n\n'
                     '## 工作记忆\n工作记忆保存当前任务需要的短期信息，任务结束后可以释放。\n\n'
                     '## 事件记忆\n事件记忆记录过去发生的事情。它不意味着永久保存全部对话。'
                     '具体保留期限和验收口令不在本笔记中，需核对关联规范。\n\n'
                     '## 程序性记忆\n程序性记忆保存可复用的步骤，更新时需要记录版本。\n\n'
                     '## 关联规范\n本次实验使用 [记忆归档规范](memory-policy.md)。',
        'memory-policy.md': '# 记忆归档规范\n'
                            '本规范只适用于本地 Agent Memory 集成实验，不是通用产品规则。\n\n'
                            '## 保留策略\n事件记忆保留 30 天，到期后清除原始记录。\n\n'
                            f'## 实验验收\n本次实验的验收口令是 {code}。核验时应原样返回口令。',
        'cache.md': '# 向量缓存\n缓存已计算的 embedding 可以减少重复计算。缓存不决定事件记忆保留期。',
    }
    for source, content in documents.items():
        (notes / source).write_text(content, encoding='utf-8')

    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'database': str(db),
              'notes_directory': str(notes), 'vault_id': vault, 'qdrant_url': url,
              'agent_model': os.environ.get('OBSIDIAN_RAG_AGENT_MODEL', DEFAULT_GENERATION_MODEL),
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
    ('A-match', '哪些笔记提到了 RAG', 'match'),
    ('B-search', '有哪些笔记和 RAG 相关', 'search'),
    ('C-read', '读取 rag.md', 'read'),
    ('D-materials', MATERIAL_QUERY, None),
    ('E-direct', '你好', None),
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
                assert '30' in result.response or '三十' in result.response
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
    ('CLI-limit', '读取 rag.md，并根据原文解释 RAG。', 1, 1, ['--think']),
    ('CLI-limit-no-think', '读取 rag.md，并根据原文解释 RAG。', 1, 1, ['--no-think']),
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
        assert '30' in result['response'] or '三十' in result['response']
    assert result['state']['turn'] <= max_turns


@pytest.mark.parametrize('repeat', [1, 2])
@pytest.mark.parametrize('case,query', [('material', MATERIAL_QUERY), ('related', '有哪些笔记和 RAG 相关')])
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
                                        and ('30' in result.response or '三十' in result.response))
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
