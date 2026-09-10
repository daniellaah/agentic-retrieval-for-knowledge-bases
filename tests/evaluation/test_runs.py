from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from arkb.agent.state import AgentResult, AgentState, AgentToolTrace, AgentTrace
from arkb.evaluation.agent_metrics import evaluate_case, summarize_agent_results
from arkb.evaluation.runs import main, run_agent_evaluation
from arkb.evaluation.datasets import load_agent_eval_dataset
from arkb.evaluation.models import AgentEvalCase, AgentEvalConfig
from tests.agent.helpers import tool_call
from tests.evaluation.test_agent_dataset import DATASET, NOTES
from tests.evaluation.test_agent_metrics import observation, trace


class FakeRuntime:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    def ask(self, query, **options):
        self.requests.append((query, options))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(trace=replace(outcome, query=query))

    def __enter__(self):
        raise AssertionError('Injected runtime remains caller-owned')

    def __exit__(self, *args):
        raise AssertionError('Injected runtime remains caller-owned')


@pytest.fixture
def config(tmp_path):
    notes = tmp_path / 'notes'
    notes.mkdir()
    for source in ('a.md', 'b.md'):
        (notes / source).write_text('# Note\nEvidence')
    dataset = tmp_path / 'cases.jsonl'
    cases = [
        AgentEvalCase(id='lookup', query='查找 | evidence\nnow', task_type='exact_lookup', expected_sources=('a.md', 'b.md')),
        AgentEvalCase(id='read', query='Read a.md', task_type='direct_read', expected_sources=('a.md',)),
        AgentEvalCase(id='none', query='Hello', task_type='no_retrieval'),
    ]
    dataset.write_text(''.join(json.dumps(asdict(case), ensure_ascii=False) + '\n' for case in cases), encoding='utf-8')
    return AgentEvalConfig(dataset_path=dataset, output_dir=tmp_path / 'run',
                           notes_dir=notes, db=tmp_path / 'absent.sqlite',
                           model='fake-model', max_turns=3, num_trials=2, think=False)


def test_every_versioned_case_and_multiple_trials_execute_independently(config):
    config = replace(config, dataset_path=DATASET, notes_dir=NOTES)
    cases = load_agent_eval_dataset(DATASET)
    outcomes = [trace(*(observation('read', source) for source in case.expected_sources))
                for case in cases for trial in range(2)]
    runtime = FakeRuntime(outcomes)
    sentinel_client = object()
    run = run_agent_evaluation(config, runtime=runtime, client=sentinel_client)
    assert len(run.results) == len(runtime.requests) == 80
    assert [(r.case.id, r.trial) for r in run.results] == [(c.id, t) for c in cases for t in range(2)]
    assert all(row.metrics.success for row in run.results)
    assert run.summary['total_cases'] == 40 and run.summary['total_trials'] == 80
    assert run.summary['task_success_rate'] == 1 and run.summary['average_source_recall'] == 1
    assert run.summary['unnecessary_retrieval_rate'] == 0
    assert run.summary['tool_usage_distribution']['read'] == sum(len(c.expected_sources) for c in cases) * 2
    for (query, options), case in zip(runtime.requests, [c for c in cases for _ in range(2)]):
        assert query == case.query
        assert options == {'db': config.db, 'vault_id': config.vault_id, 'notes_dir': NOTES,
                           'model': 'fake-model', 'max_turns': 3, 'think': False, 'client': sentinel_client}
    assert (run.output_dir / 'cases.jsonl').read_bytes() == DATASET.read_bytes()
    metadata = json.loads((run.output_dir / 'run_metadata.json').read_text())
    assert metadata['dataset_sha256'] == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert metadata['status'] == 'completed' and metadata['completed_trials'] == 80
    assert len(metadata['knowledge_before']['notes_sha256']) == 40
    assert metadata['knowledge_before']['snapshot'] is None
    assert metadata['knowledge_changed'] is False
    assert 'evaluation/runs.py' in metadata['source_hashes']
    assert not config.db.exists()
    assert 'No failed trials.' in (run.output_dir / 'report.md').read_text()


def test_errors_keep_partial_trace_continue_other_trials_and_round_trip_metrics(config):
    partial = AgentResult(None, 'error', AgentState(messages=[
        {'role': 'user', 'content': 'Read a.md'},
        {'role': 'assistant', 'tool_calls': [tool_call('read', source='a.md')]},
        {'role': 'tool', 'content': json.dumps({'result': {'source': 'a.md', 'content': '中文 café'}})},
    ], turn=1))
    error = LookupError('tool failed | details\nnext')
    error.agent_result = partial
    runtime = FakeRuntime([
        trace(observation('search', 'a.md', 'b.md')),
        trace(observation('match', 'a.md'), reason='max_turns', turns=3),
        error, ConnectionError('No partial trace available'),
        trace(turns=1), trace(observation('read', 'a.md')),
    ])
    run = run_agent_evaluation(config, runtime=runtime)
    assert len(run.results) == 6
    summary = run.summary
    assert summary['total_cases'] == 3 and summary['task_success_rate'] == pytest.approx(1 / 3)
    assert summary['average_source_recall'] == pytest.approx(2.5 / 3)
    assert summary['source_recall_defined_trials'] == 3
    assert summary['average_tool_calls'] == .8 and summary['average_turns'] == 1.8
    assert summary['unnecessary_retrieval_rate'] == .5 and summary['max_turn_failure_rate'] == .2
    assert summary['stop_reason_distribution'] == {'error': 1, 'final': 3, 'max_turns': 1, 'other': 0, 'unavailable': 1}
    assert [(f['case_id'], f['trial']) for f in summary['failed_trials']] == [('lookup', 1), ('read', 0), ('read', 1), ('none', 1)]
    assert run.results[2].trace == partial.trace
    assert run.results[3].trace is None and run.results[3].metrics.tool_call_count is None
    rows = [json.loads(line) for line in (run.output_dir / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 6 and rows[2]['error']['type'] == 'LookupError'
    assert rows[2]['trace']['tool_calls'][0]['result']['result']['content'] == '中文 café'
    recalculated = []
    for row in rows:
        saved_trace = row['trace']
        restored = (AgentTrace(**{**saved_trace, 'tool_calls': [AgentToolTrace(**c) for c in saved_trace['tool_calls']]})
                    if saved_trace is not None else None)
        metrics = evaluate_case(AgentEvalCase(**row['case']), restored)
        assert json.loads(json.dumps(asdict(metrics))) == row['metrics']
        assert row['runtime_metadata']['model'] == 'fake-model'
        assert row['runtime_metadata']['elapsed_ms'] >= 0
        recalculated.append(metrics)
    loaded_summary = json.loads((run.output_dir / 'summary.json').read_text())
    for key, value in summarize_agent_results(recalculated).items():
        assert loaded_summary[key] == value
    report = (run.output_dir / 'report.md').read_text()
    for required in ('lookup / trial 1', 'read / trial 0', 'read / trial 1', 'none / trial 1',
                     'query', 'task_type', 'expected_sources', 'retrieved_sources', 'tool_sequence', 'stop_reason'):
        assert required in report
    assert '查找 \\| evidence<br>now' in report
    assert 'tool failed \\| details<br>next' in report


@pytest.mark.parametrize('options', [
    {'num_trials': 0}, {'num_trials': True}, {'max_turns': -1}, {'max_turns': 1.5},
    {'model': ''}, {'vault_id': None}, {'think': 'false'}, {'notes_dir': None},
    {'dataset_path': ''}, {'output_dir': 1}, {'runtime_config': {}},
])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        AgentEvalConfig(**options)


def test_validation_and_existing_output_fail_before_asking_and_preserve_artifacts(config):
    runtime = FakeRuntime([])
    config.output_dir.mkdir()
    sentinel = config.output_dir / 'results.jsonl'
    sentinel.write_text('previous result\n')
    with pytest.raises(FileExistsError, match='new directory'):
        run_agent_evaluation(config, runtime=runtime)
    assert sentinel.read_text() == 'previous result\n'
    config.dataset_path.write_text('{bad json')
    with pytest.raises(ValueError, match='line 1'):
        run_agent_evaluation(replace(config, output_dir=config.output_dir / 'new'), runtime=runtime)
    assert not runtime.requests
    assert not (config.output_dir / 'new').exists()


def test_interrupt_preserves_completed_rows_and_marks_run_incomplete(config):
    runtime = FakeRuntime([trace(observation('match', 'a.md', 'b.md')), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        run_agent_evaluation(config, runtime=runtime)
    rows = (config.output_dir / 'results.jsonl').read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])['trial'] == 0
    metadata = json.loads((config.output_dir / 'run_metadata.json').read_text())
    assert metadata['status'] == 'interrupted' and metadata['completed_trials'] == 1
    assert not (config.output_dir / 'summary.json').exists()


def test_non_json_trace_cannot_silently_use_python_repr(config):
    runtime = FakeRuntime([trace(AgentToolTrace(1, 'search', {}, {'opaque': object()}))])
    with pytest.raises(TypeError, match='Cannot serialize object'):
        run_agent_evaluation(config, runtime=runtime)
    metadata = json.loads((config.output_dir / 'run_metadata.json').read_text())
    assert metadata['status'] == 'failed' and metadata['completed_trials'] == 0
    assert (config.output_dir / 'results.jsonl').read_text() == ''


def test_all_untraced_failures_are_retained_with_undefined_behavior(config):
    run = run_agent_evaluation(config, runtime=FakeRuntime([OSError('setup failed')] * 6))
    assert run.summary['failure_count'] == run.summary['missing_trace_trials'] == 6
    assert run.summary['average_tool_calls'] is run.summary['unnecessary_retrieval_rate'] is None
    assert len(run.summary['failed_trials']) == 6


def test_default_output_creates_distinct_runs_and_records_input_drift(config, monkeypatch):
    monkeypatch.chdir(config.output_dir.parent)
    class ChangingRuntime(FakeRuntime):
        def ask(self, query, **options):
            (config.notes_dir / 'a.md').write_text('# Changed\nNew evidence')
            return super().ask(query, **options)
    config = replace(config, output_dir=None)
    run = run_agent_evaluation(config, runtime=ChangingRuntime([trace()] * 6))
    assert run.output_dir.parent.as_posix() == 'evaluation/results'
    assert run.summary['knowledge_changed'] is True
    assert 'Knowledge inputs changed' in (run.output_dir / 'report.md').read_text()
    again = run_agent_evaluation(config, runtime=FakeRuntime([trace()] * 6))
    assert again.output_dir != run.output_dir and again.summary['knowledge_changed'] is False


def test_thin_cli_uses_owned_real_runtime_without_extending_main_cli(config, monkeypatch, capsys):
    from arkb.runtime import Runtime
    from tests.agent.helpers import ScriptedModel, reply

    # All requests answer directly: retrieval tasks fail evaluation but the run completes.
    client = ScriptedModel(*[reply('Hello') for _ in range(6)])
    monkeypatch.setattr(Runtime, 'model_client', lambda self: client)
    assert main(['--dataset', str(config.dataset_path), '--output', str(config.output_dir),
                 '--notes-dir', str(config.notes_dir), '--db', str(config.db),
                 '--model', 'scripted', '--num-trials', '2', '--max-turns', '3', '--no-think']) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['summary']['total_trials'] == 6
    assert printed['summary']['failure_count'] == 4
    assert all(r['model'] == 'scripted' and r['think'] is False for r in client.requests)
    assert all(len(r['messages']) == 2 for r in client.requests)
    assert not config.db.exists()
    with pytest.raises(SystemExit) as raised:
        main(['--num-trials', '0'])
    assert raised.value.code == 2
