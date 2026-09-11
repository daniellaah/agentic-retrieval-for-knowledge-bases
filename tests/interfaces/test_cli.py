"""CLI contracts against a fake Runtime: no storage, model, or retrieval services."""

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess
from unittest.mock import create_autospec

from httpx import ReadTimeout
import pytest

from arkb.agent.state import AgentResult, AgentState
from arkb.config import DEFAULT_DB, DEFAULT_GENERATION_MODEL, DEFAULT_RETRIEVAL_MODE, RetrievalConfig
from arkb.interfaces.cli import _parser, main
from arkb.knowledge.indexing import BuildReport
from arkb.knowledge.models import EmbeddingSpec, IndexManifest
from arkb.retrieval.models import SearchResponse, SearchResult
from arkb.runtime import Runtime
from tests.agent.helpers import tool_call


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    factory = create_autospec(Runtime)
    runtime = factory.return_value
    runtime.__enter__.return_value = runtime
    monkeypatch.setattr('arkb.interfaces.cli.Runtime', factory)
    calls = [tool_call('search', query='agent memory', mode='bm25'),
             tool_call('read', source='34_agent_memory_lifecycle.md'),
             tool_call('search', query='episodic memory agents', mode='hybrid')]
    messages = [{'role': 'system', 'content': 'internal instructions'}]
    for call in calls:
        messages.extend([{'role': 'assistant', 'tool_calls': [call], 'thinking': 'internal model reasoning'},
                         {'role': 'tool', 'tool_name': call['function']['name'], 'content': '{}'}])
    messages.append({'role': 'assistant', 'content': 'Relevant material found.'})
    runtime.ask.return_value = AgentResult('Relevant material found.', 'final', AgentState(messages, 4))
    hit = SearchResult(source_id='document-id', source='rag.md', content='RAG', method='exact',
                       start_char=4, end_char=7, metadata={'title': 'Retrieval'})
    runtime.match.return_value = SearchResponse(query='RAG', method='exact', results=(hit,))
    runtime.search.return_value = SearchResponse(query='Agent Memory', method='semantic', index_id='snapshot')
    manifest = IndexManifest(index_version='snapshot', vault_id='default',
        embedding_spec=EmbeddingSpec(model='fake', model_revision='fake', dimensions=2,
                                     document_template='title-body-v1'),
        chunking_fingerprint='0' * 64, document_count=2, chunk_count=3, status='ready')
    runtime.index.return_value = BuildReport(manifest, embedded_inputs=2, cached_inputs=1)
    runtime.status.return_value = {'vault_id': 'default', 'active_version': 'snapshot',
        'builds': [asdict(manifest)], 'notes_dir': '/notes', 'backend': {'kind': 'qdrant'}}
    return runtime, factory


def test_only_five_top_level_commands():
    commands = next(action for action in _parser()._actions if action.dest == 'command')
    assert set(commands.choices) == {'match', 'search', 'ask', 'index', 'status'}


def test_match_calls_only_runtime_match_and_formats_occurrences(runtime, capsys):
    fake, _ = runtime
    assert main(['match', 'RAG', '--source', 'rag.md', '--top-k', '3', '--notes-dir', 'notes']) == 0
    fake.match.assert_called_once_with('RAG', db=DEFAULT_DB, vault_id='default',
                                       notes_dir=Path('notes'), source='rag.md', top_k=3)
    assert all(not getattr(fake, name).called for name in ('search', 'ask', 'index', 'status'))
    output = capsys.readouterr()
    assert 'rag.md (body chars 4:7)' in output.out and 'RAG' in output.out
    assert output.err == ''


@pytest.mark.parametrize('mode', ['bm25', 'semantic', 'hybrid'])
def test_search_passes_mode_top_k_and_source(runtime, mode):
    fake, _ = runtime
    assert main(['search', 'How to manage long-term agent memory', '--mode', mode, '--top-k', '5',
                 '--source', 'memory.md', '--db', 'kb.sqlite', '--vault-id', 'kb']) == 0
    fake.search.assert_called_once_with('How to manage long-term agent memory', db=Path('kb.sqlite'), vault_id='kb',
        mode=mode, top_k=5, source='memory.md', settings=RetrievalConfig(), rerank=False, exact=False)
    fake.ask.assert_not_called()
    fake.run_agent.assert_not_called()
    fake.model_client.assert_not_called()


def test_search_uses_project_default(runtime):
    fake, _ = runtime
    assert main(['search', 'Agent Memory']) == 0
    assert fake.search.call_args.kwargs['mode'] == DEFAULT_RETRIEVAL_MODE == 'semantic'


@pytest.mark.parametrize('option', ['--reranker-model', '--reranker-revision'])
def test_search_rejects_removed_reranker_overrides(runtime, option):
    _, factory = runtime
    with pytest.raises(SystemExit) as error:
        main(['search', 'query', '--rerank', option, 'unsupported'])
    assert error.value.code == 2
    factory.assert_not_called()


@pytest.mark.parametrize('command,arguments', [
    ('match', ['RAG']), ('search', ['Agent Memory']), ('ask', ['Find relevant material']), ('index', []), ('status', []),
    ('ask', ['Find relevant material', '--think']), ('ask', ['Find relevant material', '--no-think']),
])
def test_json_changes_only_format_for_every_command(runtime, capsys, command, arguments):
    fake, factory = runtime
    assert main([command, *arguments]) == 0
    human = capsys.readouterr()
    first_call = getattr(fake, command).call_args
    config = factory.call_args
    fake.reset_mock()
    assert main([command, *arguments, '--json']) == 0
    output = capsys.readouterr()
    assert getattr(fake, command).call_args_list == [first_call]
    assert factory.call_args == config
    assert all(not getattr(fake, other).called for other in ('match', 'search', 'ask', 'index', 'status')
               if other != command)
    value = getattr(fake, command).return_value
    assert json.loads(output.out) == json.loads(json.dumps(value if command == 'status' else asdict(value)))
    assert human.out and human.out != output.out
    assert human.err == output.err == ''
    fake.model_client.assert_not_called()


def test_ask_calls_agent_entry_point_with_model_and_turn_limit(runtime, capsys):
    fake, _ = runtime
    assert main(['ask', 'Which notes mention RAG?', '--max-turns', '6', '--generation-model', 'fake-agent']) == 0
    fake.ask.assert_called_once_with('Which notes mention RAG?', db=DEFAULT_DB, vault_id='default',
                                     notes_dir=None, model='fake-agent', max_turns=6, think=True)
    assert capsys.readouterr().out == 'Relevant material found.\n'
    fake.match.assert_not_called()
    fake.search.assert_not_called()


@pytest.mark.parametrize('flags,think', [([], True), (['--think'], True), (['--no-think'], False)])
def test_ask_thinking_flags_only_set_the_model_option(runtime, capsys, flags, think):
    fake, _ = runtime
    assert main(['ask', 'Question', *flags]) == 0
    assert fake.ask.call_args.kwargs['think'] is think
    output = capsys.readouterr()
    assert output.out == 'Relevant material found.\n' and output.err == ''


@pytest.mark.parametrize('json_output', [False, True])
@pytest.mark.parametrize('think_flag', ['--think', '--no-think'])
def test_ask_trace_is_stderr_and_does_not_change_execution(runtime, capsys, json_output, think_flag):
    fake, _ = runtime
    args = ['ask', 'Agent Memory', think_flag] + (['--json'] if json_output else [])
    assert main(args) == 0
    original = capsys.readouterr()
    first_call = fake.ask.call_args
    assert main([*args, '--trace']) == 0
    output = capsys.readouterr()
    assert fake.ask.call_args == first_call
    assert original.out == output.out and original.err == ''
    assert output.err == ('[1] search\nquery: "agent memory"\nmode: "bm25"\n\n'
        '[2] read\nsource: "34_agent_memory_lifecycle.md"\n\n'
        '[3] search\nquery: "episodic memory agents"\nmode: "hybrid"\n\n[4] final\n')
    assert 'internal instructions' not in output.err
    assert 'internal model reasoning' not in output.err


@pytest.mark.parametrize('json_output', [False, True])
def test_ask_turn_limit_reports_no_fabricated_answer(runtime, capsys, json_output):
    fake, _ = runtime
    fake.ask.return_value = AgentResult(None, 'max_turns', AgentState(turn=1))
    assert main(['ask', 'Q', '--max-turns', '1', '--trace'] + (['--json'] if json_output else [])) == 1
    output = capsys.readouterr()
    if json_output:
        assert json.loads(output.out) == {'response': None, 'stop_reason': 'max_turns',
                                          'state': {'messages': [], 'turn': 1}}
    else:
        assert output.out == ''
    assert '[1] max_turns' in output.err and 'without a final response' in output.err


def test_index_preserves_build_options_and_status_scope(runtime):
    fake, _ = runtime
    assert main(['index', '--notes-dir', 'notes', '--force', '--chunking', 'none', '--batch-size', '4',
                 '--context-length', '512', '--max-batch-tokens', '1024', '--max-retries', '3',
                 '--query-instruction', '', '--db', 'kb.sqlite', '--vault-id', 'kb']) == 0
    options = fake.index.call_args.kwargs
    assert options['notes_dir'] == Path('notes') and options['force'] is True
    assert options['chunking'] == 'none' and options['query_instruction'] == ''
    assert (options['batch_size'], options['context_length'], options['max_batch_tokens'], options['max_retries']) == (4, 512, 1024, 3)
    assert main(['status', '--db', 'kb.sqlite', '--vault-id', 'kb']) == 0
    fake.status.assert_called_once_with(db=Path('kb.sqlite'), vault_id='kb')


@pytest.mark.parametrize('arguments', [
    [], ['query', 'Q'], ['answer', 'Q'], ['chat'], ['read', 'a.md'], ['match'], ['search'], ['ask'],
    ['match', ' '], ['search', ' '], ['ask', ' '], ['status', '--vault-id', ' '],
    ['match', 'RAG', '--top-k', '0'], ['search', 'Q', '--top-k', '-1'], ['search', 'Q', '--source', ' '],
    ['search', 'Q', '--mode', 'bad'], ['search', 'Q', '--top-k', '1.5'],
    ['search', 'Q', '--timeout', 'nan'], ['ask', 'Q', '--timeout', 'inf'],
    ['ask', 'Q', '--max-turns', '0'], ['ask', 'Q', '--max-turns', 'bad'],
    ['ask', 'Q', '--mode', 'bm25'], ['ask', 'Q', '--mode', 'semantic'], ['ask', 'Q', '--mode', 'hybrid'],
    ['ask', 'Q', '--top-k', '2'], ['ask', 'Q', '--generation-model', ' '],
    ['search', 'Q', '--show-context'], ['search', 'Q', '--answer-json'],
    ['index', '--chunk-size', '0'], ['index', '--batch-size', '0'], ['index', '--max-retries', '9'],
    ['ask', 'Q', '--think', 'false'], ['search', 'Q', '--think'], ['match', 'Q', '--no-think'],
    ['index', '--think'], ['status', '--think'],
])
def test_invalid_arguments_fail_before_constructing_runtime(runtime, capsys, arguments):
    _, factory = runtime
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert 'error:' in capsys.readouterr().err
    factory.assert_not_called()


@pytest.mark.parametrize('command', ['', 'match', 'search', 'ask', 'index', 'status'])
def test_help_without_runtime(runtime, capsys, command):
    _, factory = runtime
    with pytest.raises(SystemExit) as error:
        main(([command] if command else []) + ['--help'])
    assert error.value.code == 0
    assert 'usage:' in capsys.readouterr().out
    factory.assert_not_called()


@pytest.mark.parametrize('command,error', [
    ('match', FileNotFoundError('rg is missing')), ('match', subprocess.CalledProcessError(2, ['rg'])),
    ('search', ReadTimeout('embedding unavailable')), ('ask', LookupError('Document no longer exists')),
    ('ask', ValueError('Agent model response was truncated.')), ('index', OSError('scan failed')),
    ('status', sqlite3.OperationalError('unable to open database')),
])
def test_runtime_errors_go_to_stderr_without_partial_results(runtime, capsys, command, error):
    fake, _ = runtime
    getattr(fake, command).side_effect = error
    assert main([command, *(['Q'] if command in ('match', 'search', 'ask') else []), '--json']) == 1
    output = capsys.readouterr()
    assert output.out == '' and str(error) in output.err and 'Traceback' not in output.err
    fake.__exit__.assert_called_once()
